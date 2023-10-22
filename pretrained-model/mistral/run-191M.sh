rm -rf /dev/shm/*
WANDB_PROJECT=mistral-158M \
~/.local/bin/torchrun --nproc_per_node 4 \
-m run_clm_mosaic \
--tokenizer_name malaysia-ai/bpe-tokenizer \
--model_name_or_path ./mistral-191M \
--per_device_train_batch_size 38 \
--gradient_accumulation_steps 1 \
--output_dir pretrain-mistral-191M \
--bf16 \
--torch_dtype "bfloat16" \
--do_train \
--do_eval false \
--num_train_epochs 10 \
--train_file "tokenized" \
--logging_steps 1 \
--learning_rate 2e-4 \
--block_size 4096 \
--save_steps 200 \
--save_total_limit 2 \
--warmup_steps 50 \
--gradient_checkpointing true \
