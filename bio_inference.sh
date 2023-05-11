TARGET_FOLDER="../LLaMA-7B"
ADAPTER_PATH="bio_finetuning/checkpoint_adapter/adapter.pth"

torchrun --nproc_per_node 1 example_pf.py --ckpt_dir $TARGET_FOLDER/ --tokenizer_path $TARGET_FOLDER/tokenizer.model --adapter_path $ADAPTER_PATH
