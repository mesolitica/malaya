# BigBird-Bahasa

Thanks to Google for opensourcing most of the source code to develop BigBird, https://github.com/google-research/bigbird.

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
    * [Multigpus](#multigpus)
  * [Download](#download)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. Provide **SMALL** and **BASE** BigBird Encoder for Bahasa (formal and social media).
2. Provide **SMALL** and **BASE** BigBird Encoder-Decoder for Bahasa (formal only).

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/), [KeyReply](https://www.keyreply.com/) and [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc) for sponsoring AWS, Google and GPU clouds to train BERT for Bahasa.

## How-to Encoder

1. Use [../bert/BERT.wordpiece](../bert/BERT.wordpiece) for wordpiece tokenizer.

2. Split text file to multiple text files,

```bash
mkdir splitted
cd splitted
split -l 300000 -d --additional-suffix=.txt ../dumping-instagram.txt splitted-instagram
split -l 300000 -d --additional-suffix=.txt ../dumping-twitter.txt splitted-twitter
```

3. Create pretraining dataset,

```bash
python3 create-pretraining-data.py
```

4. Execute pretraining,

**TPU BASE**,

```bash
python3 run_pretraining.py \
  --input_file=gs://mesolitica-tpu-general/bert-data/*.tfrecord \
  --output_dir=gs://mesolitica-tpu-general/bigbird-base \
  --do_train=True \
  --do_eval=False \
  --train_batch_size=512 \
  --max_seq_length=512 \
  --max_predictions_per_seq=20 \
  --num_train_steps=500000 \
  --learning_rate=1e-4 \
  --iterations_per_loop=100 \
  --tpu_name=node-3 \
  --tpu_zone=europe-west4-a \
  --save_checkpoints_steps=25000 \
  --use_tpu=True
```