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
from tqdm import tqdm

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class InstructionDatasetForClassification(Dataset):

    def __init__(self, data_path, model_path, max_words=30, terms_dict = None):
        self.ann = json.load(open(data_path))
        
        self.max_words = max_words
        tokenizer = Tokenizer(model_path=model_path + "/tokenizer.model")
        self.tokenizer1 = tokenizer

        self.terms_dict = terms_dict
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt# + ann["output"]
        prompt = torch.tensor(self.tokenizer1.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer1.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
                    
        #labels = copy.deepcopy(example)
        #labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        #label_mask = labels.ge(0)
        example[~example_mask] = 0
        #labels[~label_mask] = 0
        example_mask = example_mask.float()
        #label_mask = label_mask.float()

        output = ann["output"]
        gostring = output.split(",")
        gostring = [x.strip() for x in gostring if x != '']
        #labels = torch.zeros(len(self.terms_dict), dtype=torch.int64)
        #positives = [self.terms_dict[x] for x in gostring]
        #labels[positives] = 1
        
        return example#, labels, example_mask


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

        
    model_args: ModelArgs = ModelArgs(num_gos = len(terms_dict), max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
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
        max_seq_len: int = 256,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    model = load(classifier_model_path, tokenizer_path, llama_dir, data_path, local_rank, world_size, max_seq_len, max_batch_size)

    valid_data_path = os.path.join(data_path, "valid_instruction_data_for_classification.json")
    valid_data = InstructionDatasetForClassification(valid_data_path, llama_dir, max_words=max_seq_len) #, terms_dict = model.terms_dict

    valid_instruction_data = json.load(open(valid_data_path, "r"))
    gene_prot_names = [x.get("input") for x in valid_instruction_data]

    results = []
    for example in tqdm(valid_data):
        preds = model.predict(example.unsqueeze(0))
        results.append(preds.squeeze(0).cpu().numpy())
        
    predictions = pd.DataFrame({"gene_prot_name": gene_prot_names, "prediction": results})
    predictions.to_pickle(os.path.join(data_path, "predictions.pkl"))
                                                    
                
if __name__ == "__main__":
    fire.Fire(main)
