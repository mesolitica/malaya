# RoBERTa-Bahasa

Thanks to Huggingface for opensourcing source code to pretrain RoBERTa.

**This directory is very lack of comments, understand PyTorch and Transformers are really helpful**.

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
  * [Download](#download)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. Provide RoBERTa for Bahasa.

## Acknowledgement

Thanks to [Mesolitica](https://mesolitica.com/) for RTX servers.

## How-to

1. Create tokenizer, [create-tokenizer.ipynb](create-tokenizer.ipynb).

2. Create BASE model [base-model.ipynb](base-model.ipynb) and train,

```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.launch --nproc_per_node 2 \
train_mlm.py \
--model_name_or_path malay-cased-roberta-base \
--train_file train.txt \
--validation_file sample-wiki.txt \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 8 \
--do_train \
--do_eval \
--output_dir malay-cased-roberta-base-mlm \
--max_seq_length 256 \
--line_by_line \
--save_total_limit 5 \
--save_steps 10000 \
--ignore_data_skip
```

Or single GPU,

```bash
python3 train_mlm.py \
--model_name_or_path malay-cased-roberta-base \
--train_file train.txt \
--validation_file sample-wiki.txt \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 8 \
--do_train \
--do_eval \
--output_dir malay-cased-roberta-base-mlm \
--max_seq_length 256 \
--line_by_line \
--save_total_limit 5 \
--save_steps 10000 \
--ignore_data_skip \
--fp16
```

3. Create TINY model [tiny-model.ipynb](tiny-model.ipynb) and train,

```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.launch --nproc_per_node 2 \
train_mlm.py \
--model_name_or_path malay-cased-roberta-base \
--train_file train.txt \
--validation_file sample-wiki.txt \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 8 \
--do_train \
--do_eval \
--output_dir malay-cased-roberta-base-mlm \
--max_seq_length 256 \
--line_by_line \
--save_total_limit 5 \
--save_steps 10000 \
--ignore_data_skip
```

Or single GPU,

```bash
python3 train_mlm.py \
--model_name_or_path malay-cased-roberta-tiny \
--train_file train.txt \
--validation_file sample-wiki.txt \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 8 \
--do_train \
--do_eval \
--output_dir malay-cased-roberta-tiny-mlm \
--max_seq_length 256 \
--line_by_line \
--save_total_limit 5 \
--save_steps 10000 \
--ignore_data_skip \
--fp16
```