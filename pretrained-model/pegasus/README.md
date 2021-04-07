# Pegasus-Bahasa

Thanks to Google for opensourcing most of the source code to develop Pegasus, https://github.com/google-research/pegasus

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## Table of contents
  * [Objective](#objective)

## Objective

1. Provide **SMALL** and **BASE** Pegasus.
1. Provide **SMALL** and **BASE** Pegasus trained multitasks (T5 style).

## how-to

1. Generate wordpiece, [generate-wordpiece.ipynb](generate-wordpiece.ipynb).

2. Split text file to multiple text files,

```bash
mkdir splitted
cd splitted
split -l 300000 -d --additional-suffix=.txt ../filtered-dumping-wiki.txt splitted-wiki
split -l 300000 -d --additional-suffix=.txt ../dumping-cleaned-news.txt splitted-news
split -l 300000 -d --additional-suffix=.txt ../filtered-dumping-academia.txt splitted-academia
split -l 300000 -d --additional-suffix=.txt ../dumping-parliament.txt splitted-parliament
```

3. Create pretraining dataset,

```bash
python3 create-pretraining-data.py # mask sentence using ROUGE score
```

Make sure you set proper constants in [create_pretraining_data.py](create_pretraining.py),

```python
max_seq_length_encoder = 512
max_seq_length_decoder = 256
masked_lm_prob = 0.0
max_predictions_per_seq = 0
do_whole_word_mask = True
```

And we also add a function to filter tokens candidate to mask,

```python
def reject(token):
    t = token.replace('##', '')
    if is_number_regex(t):
        return True
    if t.startswith('RM'):
        return True
    if token in '!{<>}:;.,"\'':
        return True
    return False
```

We do not want to mask numbers or words started with `RM`.

4. Execute pretraining,

**TPU BASE**,

```bash
python3 pretraining-base.py \
--input_file=gs://mesolitica-tpu-general/pegasus-data-v2/tfrecord/*.tfrecord \
--output_dir=gs://mesolitica-tpu-general/pegasus-base-v3 \
--do_train=True \
--train_batch_size=128 \
--num_train_steps=1500000 \
--iterations_per_loop=100 \
--tpu_name=node-1 \
--tpu_zone=europe-west4-a \
--save_checkpoints_steps=25000 \
--learning_rate=0.01 \
--use_tpu=True
```

**TPU SMALL**,

```bash
python3 pretraining-small.py \
--input_file=gs://mesolitica-tpu-general/pegasus-data-v2/tfrecord/*.tfrecord \
--output_dir=gs://mesolitica-tpu-general/pegasus-small-v3 \
--do_train=True \
--train_batch_size=256 \
--num_train_steps=1500000 \
--iterations_per_loop=100 \
--tpu_name=node-2 \
--tpu_zone=europe-west4-a \
--save_checkpoints_steps=25000 \
--learning_rate=0.001 \
--use_tpu=True
```

## how-to multitasks

1. Follow step 1-4 from [../lm-transformer](../lm-transformer).

2. Execute pretraining,

**TPU BASE**,

```bash
python3 pretraining-base-multitasks.py \
--input_file=gs://mesolitica-tpu-general/t2t/data/seq2* \
--output_dir=gs://mesolitica-tpu-general/pegasus-base-multitasks-v2 \
--do_train=True \
--train_batch_size=32 \
--num_train_steps=700000 \
--iterations_per_loop=100 \
--tpu_name=node-3 \
--tpu_zone=europe-west4-a \
--save_checkpoints_steps=25000 \
--use_tpu=True
```

**TPU SMALL**,

```bash
python3 pretraining-small-multitasks.py \
--input_file=gs://mesolitica-tpu-general/t2t/data/seq2* \
--output_dir=gs://mesolitica-tpu-general/pegasus-small-multitasks-v2 \
--do_train=True \
--train_batch_size=64 \
--num_train_steps=700000 \
--iterations_per_loop=100 \
--tpu_name=node-4 \
--tpu_zone=europe-west4-a \
--save_checkpoints_steps=25000 \
--use_tpu=True
```

## Downloads

1. **Multitasks BASE**, last update 16th February 2021, [pegasus-base-multitasks-2021-02-16.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/pegasus-base-multitasks-2021-02-16.tar.gz)

  - Vocab size 32k.
  - 1M steps, V3-8 TPU.

2. **Multitasks SMALL**, last update 16th February 2021, [pegasus-small-multitasks-2021-02-16.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/pegasus-base-multitasks-2021-02-16.tar.gz)

  - Vocab size 32k.
  - 1M steps, V3-8 TPU.

1. **Sentence gap BASE**, last update 16th February 2021, [pegasus-base-sentencegap-2021-02-16.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/pegasus-base-multitasks-2021-02-16.tar.gz)

  - Vocab size 32k.
  - 1.5M steps, V3-8 TPU.

2. **Sentence gap SMALL**, last update 16th February 2021, [pegasus-small-sentencegap-2021-02-16.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/pegasus-base-multitasks-2021-02-16.tar.gz)

  - Vocab size 32k.
  - 1.5M steps, V3-8 TPU.