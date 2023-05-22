# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import fire
import torch
from torch.utils.data import Dataset
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import LLaMA, ModelArgs, Tokenizer, Transformer

from math import ceil
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

class InstructionDatasetForClassification(Dataset):

    def __init__(self, data_path, model_path, max_words=30, terms_dict = None):

        self.data_df = pd.read_pickle(data_path)
        self.entry_names = self.data_df['proteins'].tolist()
        self.descriptions = self.data_df["descriptions"].tolist()
        self.go_annots = self.data_df["exp_annotations"].tolist()
        
        self.max_words = max_words
        tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer1 = tokenizer

        self.terms_dict = terms_dict
        self.labels = torch.zeros(len(self.terms_dict), dtype=torch.int64)
        
    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):

        entry_name = self.entry_names[index]
        example = self.descriptions[index]
        example = torch.tensor(self.tokenizer1.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
                    
        example_mask = example.ge(0)
        example[~example_mask] = 0
        annots = self.go_annots[index]
        annots = [self.terms_dict[term] for term in annots if term in self.terms_dict]
        #assert len(annots) > 0
        labels = torch.zeros(len(self.terms_dict), dtype=torch.int64)
        labels[annots] = 1
        
        return entry_name, example




def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
        classifier_model_path: str,
        tokenizer_path: str,
        llama_dir: str,
        data_path: str,
        local_rank: int,
        world_size: int,
        max_seq_len: int,
        max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
     
    print("Loading")
    checkpoint = torch.load(classifier_model_path, map_location="cpu")
    del checkpoint["optimizer"]
    del checkpoint["epoch"]
    del checkpoint["scaler"]
    del checkpoint["args"]
    #adapter_checkpoint = torch.load(adapter_path, map_location="cpu")
    with open(Path(llama_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    go_terms_file = os.path.join(data_path, "terms.pkl")
    terms_df = pd.read_pickle(go_terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

        
    model_args: ModelArgs = ModelArgs(num_gos = len(terms_dict), max_seq_len=max_seq_len, max_batch_size=max_batch_size, adapter_len=5, adapter_layer=15, **params)
    #model_args.adapter_layer = int(adapter_checkpoint["adapter_query.weight"].shape[0] / model_args.adapter_len)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    model.load_state_dict(checkpoint["model"], strict=True)
   # model.load_state_dict(adapter_checkpoint, strict=False)
    return model


def main(
    llama_dir: str,
    tokenizer_path: str,
    classifier_model_path: str,
    data_path: str,
    temperature: float = 0.1,
    top_p: float = 0.99,
        max_seq_len: int = 25,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    go_terms_file = os.path.join(data_path, "terms.pkl")
    cafa_data_diam = os.path.join(data_path, "cafa_data_diam.pkl")
    cafa_df = pd.read_pickle(cafa_data_diam)
    proteins_cafa = cafa_df["proteins"].tolist() # right order of valid data
    
    terms_df = pd.read_pickle(go_terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

        
    model = load(classifier_model_path, tokenizer_path, llama_dir, data_path, local_rank, world_size, max_seq_len, max_batch_size)

    valid_data_path = os.path.join(data_path, "valid_with_desc.pkl")
    valid_data = InstructionDatasetForClassification(valid_data_path, llama_dir, max_words=max_seq_len, terms_dict = terms_dict)

    valid_df = pd.read_pickle(valid_data_path)
        
    results = dict()
    for entry_name, example in tqdm(valid_data):
        preds = model.predict(example.unsqueeze(0))
        results[entry_name] = preds[0].squeeze().cpu().numpy()

    assert len(results) == len(valid_data)
    ordered_results = []
    for protein in proteins_cafa:
        if protein in results:
            ordered_results.append(results[protein])
        else:
            ordered_results.append(np.zeros(len(terms_dict)))
            print(f"Missing {protein}")
            

    assert len(ordered_results) == len(proteins_cafa)
    valid_df["preds"] = ordered_results
    valid_df.to_pickle(os.path.join(data_path, "valid_data_with_predictions.pkl"), index=False)
                                                    
                
if __name__ == "__main__":
    fire.Fire(main)
