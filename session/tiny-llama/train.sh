rm -rf /dev/shm/*
WANDB_PROJECT="finetune-tinyllama-1.1B" \
~/.local/bin/torchrun --nproc_per_node 4 \
-m train \
--tokenizer_name PY007/TinyLlama-1.1B-intermediate-step-715k-1.5T \
--model_name_or_path PY007/TinyLlama-1.1B-intermediate-step-715k-1.5T \
--per_device_train_batch_size 32 \
--gradient_accumulation_steps 1 \
--output_dir finetune-tinyllama-1.1b-715-1.5T \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 2 \
--train_file "tokenized" \
--logging_steps 1 \
--learning_rate 2e-5 \
--block_size 4096 \
--save_steps 200 \
--save_total_limit 3 \
--gradient_checkpointing true \
--deepspeed ds_config_zero3.json