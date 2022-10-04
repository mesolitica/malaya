# BERT-Bahasa

Thanks to Huggingface for opensourcing source code to pretrain BERT.

**This directory is very lack of comments, understand PyTorch and Transformers are really helpful**.

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
  * [Download](#download)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. Provide BERT for Bahasa.

## Acknowledgement

Thanks to [Mesolitica](https://mesolitica.com/) for RTX servers.

## How-to

1. Create tokenizer, [prepare-wordpiece.ipynb](prepare-wordpiece.ipynb).

2. Create BASE model [prepare-base.ipynb](prepare-base.ipynb) and train,


```bash
python3 run_mlm_wwm.py \
--model_name_or_path malay-cased-bert-base \
--train_file train-v3.txt \
--validation_file sample-wiki.txt \
--per_device_train_batch_size 52 \
--per_device_eval_batch_size 8 \
--do_train \
--do_eval \
--output_dir malay-cased-bert-base-mlm \
--max_seq_length 256 \
--save_total_limit 5 \
--save_steps 10000 \
--num_train_epochs 20 \
--ignore_data_skip \
--fp16
```

3. Create TINY model [prepare-tiny.ipynb](prepare-tiny.ipynb) and train,

```bash
python3 run_mlm_wwm.py \
--model_name_or_path malay-cased-bert-tiny \
--train_file train-v3.txt \
--validation_file sample-wiki.txt \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 8 \
--do_train \
--do_eval \
--output_dir malay-cased-bert-tiny-mlm \
--max_seq_length 256 \
--save_total_limit 5 \
--save_steps 10000 \
--num_train_epochs 20 \
--ignore_data_skip \
--fp16
```

