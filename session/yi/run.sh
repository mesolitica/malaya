rm -rf /dev/shm/*
WANDB_PROJECT="finetune-yi-6b" \
torchrun --nproc_per_node 8 \
-m train \
--model_name_or_path 01-ai/Yi-6B \
--per_device_train_batch_size 12 \
--gradient_accumulation_steps 1 \
--output_dir finetune-yi-6b \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 2 \
--train_file "mosaic-yi" \
--logging_steps 1 \
--learning_rate 2e-5 \
--block_size 4096 \
--save_steps 200 \
--save_total_limit 3 \
--gradient_checkpointing true \
--deepspeed ds_config_zero3.json \
--log_level "info"