# RoBERTa-Bahasa

Thanks to Huggingface for opensourcing source code to pretrain RoBERTa.

**This directory is very lack of comments, understand PyTorch and Transformers are really helpful**.

## How-to

1. Create tokenizer, [create-tokenizer.ipynb](create-tokenizer.ipynb).

2. Create BASE model [base-model.ipynb](base-model.ipynb) and train,

Or single GPU,

```bash
python3 train_mlm.py \
--model_name_or_path malay-cased-roberta-base \
--train_file train-v3.txt \
--validation_file sample-wiki.txt \
--per_device_train_batch_size 42 \
--per_device_eval_batch_size 8 \
--do_train \
--do_eval \
--output_dir malay-cased-roberta-base-mlm \
--max_seq_length 256 \
--line_by_line \
--save_total_limit 10 \
--save_steps 10000 \
--ignore_data_skip \
--fp16
```

3. Create TINY model [tiny-model.ipynb](tiny-model.ipynb) and train,

```bash
python3 train_mlm.py \
--model_name_or_path malay-cased-roberta-tiny \
--train_file train-v3.txt \
--validation_file sample-wiki.txt \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 8 \
--do_train \
--do_eval \
--output_dir malay-cased-roberta-tiny-mlm \
--max_seq_length 256 \
--line_by_line \
--save_total_limit 10 \
--save_steps 10000 \
--num_train_epochs 20 \
--ignore_data_skip \
--fp16
```