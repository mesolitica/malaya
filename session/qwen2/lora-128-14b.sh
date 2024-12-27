WANDB_PROJECT=Qwen-Qwen2.5-14B-Instruct-lora-128-embedding-8k-multipack \
HF_HOME="/workspace/cache" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 4 \
-m run-instruction-lora-embedding-multipack \
--deepspeed ds_config_zero3.json \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 6 \
--output_dir Qwen-Qwen2.5-14B-Instruct-lora-128-embedding-8k-multipack \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 5 \
--train_file 'malaysian-qwen2.5-8k-language-multipack' \
--logging_steps 1 \
--learning_rate 2e-5 \
--warmup_steps 10 \
--weight_decay 0.01 \
--block_size 24576 \
--save_steps 20 \
--save_total_limit 3 \
--gradient_checkpointing true \
--neftune_noise_alpha 5.0 \
--torch_dtype bfloat16 \
--rank 128 \
--dataloader_num_workers 3 \
--dataloader_prefetch_factor 4