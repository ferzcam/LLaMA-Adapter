TARGET_FOLDER="../LLaMA-7B"
ADAPTER_PATH="adapter/llama_adapter_len10_layer30_release.pth"

torchrun --nproc_per_node 1 example.py --ckpt_dir $TARGET_FOLDER/ --tokenizer_path $TARGET_FOLDER/tokenizer.model --adapter_path $ADAPTER_PATH
