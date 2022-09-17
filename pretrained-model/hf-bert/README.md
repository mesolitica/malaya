```bash
python3 train_mlm.py \
--model_name_or_path malay-cased-bert-base \
--train_file train-v2.txt \
--validation_file sample-wiki.txt \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 8 \
--do_train \
--do_eval \
--output_dir malay-cased-bert-base-mlm \
--max_seq_length 256 \
--line_by_line \
--save_total_limit 5 \
--save_steps 10000 \
--ignore_data_skip \
--fp16
```