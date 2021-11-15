# FastFormer-Bahasa

Thanks to Google for opensourcing most of the source code to develop FastFormer, https://github.com/Rishit-dagli/Fast-Transformer. Malaya just create custom pretraining on TPU.

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
  * [Download](#download)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. Provide **SMALL** and **BASE** FastFormer for Bahasa.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/), [KeyReply](https://www.keyreply.com/) and [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc) for sponsoring AWS, Google and GPU clouds to train T5 for Bahasa.

## How-to

1. Use BERT steps to generate data.

2. Execute pretraining,

**TPU TINY**,

```bash
python3 run_pretraining.py \
  --input_file=gs://mesolitica-tpu-general/bert-data/tfrecord/*.tfrecord \
  --output_dir=gs://mesolitica-tpu-general/fastformer-tiny \
  --do_train=True \
  --do_eval=False \
  --depth=4 \
  --heads=12 \
  --dim=336 \
  --num_tokens=32000 \
  --max_seq_len=2048 \
  --train_batch_size=1024 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=500000 \
  --learning_rate=2e-5 \
  --iterations_per_loop=100 \
  --tpu_name=node-2 \
  --tpu_zone=us-central1-f \
  --save_checkpoints_steps=25000 \
  --num_warmup_steps=50000 \
  --use_tpu=True
```

```bash
python3 run_pretraining.py \
  --input_file=gs://mesolitica-tpu-general/bert-data-social-media/tfrecord/*.tfrecord \
  --output_dir=gs://mesolitica-tpu-general/fastformer-tiny-social-media \
  --do_train=True \
  --do_eval=False \
  --depth=4 \
  --heads=12 \
  --dim=336 \
  --num_tokens=32000 \
  --max_seq_len=2048 \
  --train_batch_size=2048 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=500000 \
  --learning_rate=2e-5 \
  --iterations_per_loop=100 \
  --tpu_name=node-2 \
  --tpu_zone=europe-west4-a \
  --save_checkpoints_steps=25000 \
  --num_warmup_steps=50000 \
  --use_tpu=True
```

**TPU BASE**,

```bash
python3 run_pretraining.py \
  --input_file=gs://mesolitica-tpu-general/bert-data/tfrecord/*.tfrecord \
  --output_dir=gs://mesolitica-tpu-general/fastformer-base \
  --do_train=True \
  --do_eval=False \
  --depth=12 \
  --heads=12 \
  --dim=768 \
  --num_tokens=32000 \
  --max_seq_len=2048 \
  --train_batch_size=256 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=1000000 \
  --learning_rate=2e-5 \
  --iterations_per_loop=100 \
  --tpu_name=node-1 \
  --tpu_zone=us-central1-f \
  --save_checkpoints_steps=25000 \
  --num_warmup_steps=50000 \
  --use_tpu=True
```

```bash
python3 run_pretraining.py \
  --input_file=gs://mesolitica-tpu-general/bert-data-social-media/tfrecord/*.tfrecord \
  --output_dir=gs://mesolitica-tpu-general/fastformer-base-social-media \
  --do_train=True \
  --do_eval=False \
  --depth=12 \
  --heads=12 \
  --dim=768 \
  --num_tokens=32000 \
  --max_seq_len=2048 \
  --train_batch_size=512 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=1000000 \
  --learning_rate=2e-5 \
  --iterations_per_loop=100 \
  --tpu_name=node-1 \
  --tpu_zone=europe-west4-a \
  --save_checkpoints_steps=25000 \
  --num_warmup_steps=50000 \
  --use_tpu=True
```

## Download

1. **BASE**, last update 2nd November 2021, [fastformer-base-2021-11-02.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/fastformer-base-2021-11-02.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession and translated the pile.
  - 500k steps, 1 TPU V2-8.
  - [BERT.wordpiece](../bert/tokenizer/BERT.wordpiece)

2. **TINY**, last update 2nd November 2021, [fastformer-tiny-2021-11-02.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/fastformer-tiny-2021-11-02.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession and translated the pile.
  - 500k steps, 1 TPU V2-8.
  - [BERT.wordpiece](../bert/tokenizer/BERT.wordpiece)

3. **BASE**, last update 2nd November 2021, [fastformer-base-social-media-2021-11-02.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/fastformer-base-social-media-2021-11-02.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession, twitter and instagram.
  - 500k steps, 1 TPU V2-8.
  - [BERT.wordpiece](../bert/tokenizer/BERT.wordpiece)

4. **TINY**, last update 2nd November 2021, [fastformer-tiny-social-media-2021-11-02.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/fastformer-tiny-social-media-2021-11-02.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession, twitter and instagram..
  - 500k steps, 1 TPU V2-8.
  - [BERT.wordpiece](../bert/tokenizer/BERT.wordpiece)

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/), [KeyReply](https://www.keyreply.com/) and [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc) for sponsoring AWS, Google and GPU clouds to train FastFormer for Bahasa.

## How-to

1. Generate dataset using BERT repository, follow https://github.com/google-research/bert

