TARGET_FOLDER="/data/zhapacfp/LLaMA-7B/"
#ADAPTER_PATH="bio_finetuning/checkpoint_adapter/adapter.pth"
MODEL_PATH="/data/zhapacfp/llama_adapter/checkpoint_classification/checkpoint-1.pth"
DATA_PATH="/data/zhapacfp/llama_adapter/data/mf/"

torchrun --nproc_per_node 1 validate_pf.py --classifier_model_path $MODEL_PATH --tokenizer_path $TARGET_FOLDER/tokenizer.model --llama_dir $TARGET_FOLDER --data_path $DATA_PATH
