rm -rf /dev/shm/*
WANDB_PROJECT=deberta-large \
~/.local/bin/torchrun --nproc_per_node 4 \
-m run \
--tokenizer_name malaysia-ai/bpe-tokenizer \
--config_name rtd_large.json \
--per_device_train_batch_size 128 \
--gradient_accumulation_steps 2 \
--do_train \
--max_seq_len 512 \
--output_dir debertav3-large \
--mlm_probability 0.15 \
--train_file "tokenized" \
--logging_steps="1" \
--save_steps="1000"