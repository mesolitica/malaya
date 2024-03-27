rm -rf /dev/shm/*
WANDB_PROJECT=deberta-small \
~/.local/bin/torchrun --nproc_per_node 4 \
-m run \
--tokenizer_name malaysia-ai/bpe-tokenizer \
--config_name rtd_small.json \
--per_device_train_batch_size 92 \
--per_device_eval_batch_size 1 \
--do_train \
--max_seq_len 512 \
--output_dir debertav3-small \
--mlm_probability 0.15 \
--train_file "tokenized" \
--logging_steps="1" \
--save_steps="1000" \
--bf16 \
--learning_rate 2e-4 \
--warmup_steps 1000 \
--do_train \
--do_eval false \
--num_train_epochs 10 \
--save_total_limit 2