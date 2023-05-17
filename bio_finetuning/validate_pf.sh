TARGET_FOLDER="../../LLaMA-7B"
#ADAPTER_PATH="bio_finetuning/checkpoint_adapter/adapter.pth"
MODEL_PATH="checkpoint_classification/checkpoint-10.pth"
DATA_PATH="data/mf"

torchrun --nproc_per_node 1 validate_pf.py --classifier_model_path $MODEL_PATH --tokenizer_path $TARGET_FOLDER/tokenizer.model --llama_dir $TARGET_FOLDER --data_path $DATA_PATH
