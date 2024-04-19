WANDB_PROJECT=mistral-191M-mlm \
python3 run_mlm.py \
--model_name_or_path mesolitica/malaysian-mistral-191M-4096 \
--tokenizer_name mesolitica/malaysian-mistral-191M-4096 \
--per_device_train_batch_size 105 \
--do_train \
--max_seq_len 512 \
--output_dir mistral-191M-mlm \
--mlm_probability 0.1 \
--train_file "combine-all" \
--logging_steps="1" \
--save_steps="1000" \
--bf16 \
--learning_rate 2e-4 \
--warmup_steps 10000 \
--weight_decay 1e-2 \
--do_train \
--do_eval false \
--num_train_epochs 3 \
--save_total_limit 3 