# ALBERT-Bahasa

Thanks to brightmart for opensourcing most of the source code to develop ALBERT, https://github.com/brightmart/albert_zh

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
    * [Multigpus](#multigpus)
  * [Download](#download)
  * [Comparison using Subjectivity Dataset](#comparison-using-subjectivity-dataset)
  * [Comparison using Emotion Dataset](#comparison-using-emotion-dataset)
  * [Feedbacks](#feedbacks)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. There is no multilanguage implementation of ALBERT, and obviously no Bahasa Malaysia implemented. So we decided to train ALBERT from scratch and finetune using available dataset we have. [Dataset we use for pretraining](https://github.com/huseinzol05/Malaya-Dataset#dumping).

2. Provide **SMALL**, **BASE** and **LARGE** ALBERT for Bahasa.

## How-to

1. Git clone https://github.com/brightmart/albert_zh,

```bash
git clone https://github.com/brightmart/albert_zh.git
cd albert_zh
```

2. Run [dumping.ipynb](dumping.ipynb) to create text dataset for pretraining.

You need to download [sp10m.cased.v5.model](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/sp10m.cased.v5.model) and [sp10m.cased.v5.vocab](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/sp10m.cased.v5.vocab) first.

#### Multigpus

Original BERT implementation not support multi-gpus, only single gpu. Here I created MirroredStrategy to pretrain using multi-gpus.

1. Run [multigpu_pretraining.py](multigpu_pretraining.py),

Run multigpus using MirroredStrategy,
```bash
python3 multigpu_pretraining.py \
--input_file=tests_output.tfrecord \
--output_dir=pretraining_output \
--do_train=True \
--do_eval=False \
--bert_config_file=albert_config/albert_config_base.json \
--train_batch_size=90 \
--max_seq_length=512 \
--max_predictions_per_seq=76 \
--masked_lm_prob=0.15 \
--num_train_steps=1000000 \
--num_warmup_steps=10 \
--learning_rate=2e-5 \
--save_checkpoints_steps=200000 \
--use_gpu=True \
--num_gpu_cores=3 \
--eval_batch_size=12
```

- `num_gpu_cores`: Number of gpus.
- `train_batch_size`: Make sure `train_batch_size` % `num_gpu_cores` is 0 and the batch will automatically distribute among gpus. If `num_gpu_cores` is 60 and `num_gpu_cores` is 2, so each gpus will get 30 batch size.
