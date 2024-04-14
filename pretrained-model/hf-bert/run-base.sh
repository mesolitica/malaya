WANDB_PROJECT=bert-base \
python3 run.py \
--model_name_or_path bert-base \
--per_device_train_batch_size 104 \
--per_device_eval_batch_size 1 \
--do_train \
--max_seq_len 512 \
--output_dir bert-base-output \
--mlm_probability 0.15 \
--train_file "mosaic-combine-512" \
--logging_steps="1" \
--save_steps="1000" \
--bf16 \
--learning_rate 2e-5 \
--warmup_steps 10000 \
--do_train \
--do_eval false \
--num_train_epochs 3 \
--save_total_limit 3