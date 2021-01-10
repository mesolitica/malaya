# Pegasus-Bahasa

Thanks to Google for opensourcing most of the source code to develop Pegasus, https://github.com/google-research/pegasus

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## Table of contents
  * [Objective](#objective)

## Objective

1. Provide **SMALL** and **BASE** Pegasus.
2. Provide **SMALL** and **BASE** Pegasus for General tasks (T5 style).


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
python3 create-pretraining-data.py
python3 create-pretraining-data-random.py
```

Make sure you set proper constants in [create_pretraining_data.py](create_pretraining.py),

```python
max_seq_length_encoder = 512
max_seq_length_decoder = 128
masked_lm_prob = 0.2
max_predictions_per_seq = int(masked_lm_prob * max_seq_length_encoder)
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
  --input_file=gs://mesolitica-tpu-general/pegasus-data/*.tfrecord \
  --output_dir=gs://mesolitica-tpu-general/pegasus-base \
  --do_train=True \
  --train_batch_size=256 \
  --num_train_steps=500000 \
  --iterations_per_loop=100 \
  --tpu_name=node-3 \
  --tpu_zone=europe-west4-a \
  --save_checkpoints_steps=25000 \
  --use_tpu=True
```