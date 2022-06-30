```
python3 train_mlm.py \
--model_name_or_path bert-multilanguage-8layers \
--train_file train.txt \
--validation_file test.txt \
--per_device_train_batch_size 20 \
--per_device_eval_batch_size 8 \
--do_train \
--do_eval \
--output_dir bert-multilanguage-8layers-mlm \
--max_seq_length 256 \
--line_by_line
```