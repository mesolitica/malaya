rm -rf /dev/shm/*
WANDB_PROJECT=deberta-large \
~/.local/bin/torchrun --nproc_per_node 4 \
-m run \
--tokenizer_name malaysia-ai/bpe-tokenizer \
--config_name rtd_large.json \
--per_device_train_batch_size 22 \
--gradient_accumulation_steps 1 \
--per_device_eval_batch_size 1 \
--do_train \
--max_seq_len 512 \
--output_dir debertav3-large-v2 \
--mlm_probability 0.15 \
--train_file "tokenized" \
--logging_steps="1" \
--save_steps="1000" \
--bf16 \
--learning_rate 2e-4 \
--warmup_steps 5000 \
--do_train \
--do_eval false \
--num_train_epochs 10 \
--save_total_limit 2