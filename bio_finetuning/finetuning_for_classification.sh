TARGET_FOLDER="../../LLaMA-7B"
DATA_PATH="data/mf/"

export TORCHELASTIC_ERROR_FILE="/ibex/user/zhapacfp/elastic.logs.txt"

torchrun --nproc_per_node 8\
    finetuning_for_classification.py \
    --model Llama7B_adapter \
    --llama_model_path $TARGET_FOLDER/ \
    --data_path $DATA_PATH/ \
    --adapter_layer 30 \
    --adapter_len 10 \
    --max_seq_len 256 \
    --batch_size 12 \
    --epochs 11 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir ./checkpoint_classification/ \
    --log_dir logs/ \
    --resume "checkpoint/checkpoint-7.pth" \
    --training_mode "classification"
