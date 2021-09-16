# ALBERT-Bahasa

Thanks to official implementation from Google, https://github.com/google-research/google-research, Malaya just create custom pretraining and optimizer to support multigpus.

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
  * [Download](#download)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. Provide **BASE**, **TINY** and **LARGE** ALBERT for Bahasa.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/), [KeyReply](https://www.keyreply.com/) and [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc) for sponsoring AWS, Google and GPU clouds to train ALBERT for Bahasa.

## How-to

1. Follow README in [tokenizer](tokenizer) for tokenizer and steps to generate it.

2. Create pretraining dataset,

```bash
python3 create-pretraining-data.py tfrecord
python3 create-pretraining-data.py tfrecord2
```

2. Execute pretraining,

**ALBERT required TPU to pretrain. I never had successful pretraining on GPUs even on a small dataset.**

For LARGE,

```bash
python3 -m albert.run_pretraining \
--input_file=gs://mesolitica-tpu-general/albert-data/*/*.tfrecord \
--output_dir=gs://mesolitica-tpu-general/albert-large-v2 \
--do_train=True --do_eval=False \
--albert_config_file=gs://mesolitica-tpu-general/albert-config/LARGE_config.json \
--train_batch_size=256 --max_seq_length=512 --max_predictions_per_seq=20 \
--num_train_steps=500000 --num_warmup_steps=3125 --learning_rate=1e-4 \
--save_checkpoints_steps=25000 --use_tpu=True --tpu_name=node-5 --tpu_zone=europe-west4-a \
--iterations_per_loop=100
```

For BASE,

```bash
python3 -m albert.run_pretraining \
--input_file=gs://mesolitica-tpu-general/albert-data/*/*.tfrecord \
--output_dir=gs://mesolitica-tpu-general/albert-base-v2 \
--do_train=True --do_eval=False \
--albert_config_file=gs://mesolitica-tpu-general/albert-config/BASE_config.json \
--train_batch_size=1040 --max_seq_length=512 --max_predictions_per_seq=20 \
--num_train_steps=500000 --num_warmup_steps=3125 --learning_rate=1e-4 \
--save_checkpoints_steps=25000 --use_tpu=True --tpu_name=node-6 --tpu_zone=europe-west4-a \
--iterations_per_loop=100
```

For TINY,

```bash
python3 -m albert.run_pretraining \
--input_file=gs://mesolitica-tpu-general/albert-data/*/*.tfrecord \
--output_dir=gs://mesolitica-tpu-general/albert-tiny-v2 \
--do_train=True --do_eval=False \
--albert_config_file=gs://mesolitica-tpu-general/albert-config/TINY_config.json \
--train_batch_size=3120 --max_seq_length=512 --max_predictions_per_seq=20 \
--num_train_steps=500000 --num_warmup_steps=3125 --learning_rate=1e-4 \
--save_checkpoints_steps=25000 --use_tpu=True --tpu_name=node-7 --tpu_zone=europe-west4-a \
--iterations_per_loop=100
```

## Download

1. **BASE**, last update 10th April 2020, [albert-base-2020-04-10.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/albert-base-2020-04-10.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw twitter, raw instagram, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession.
  - 400k steps, TPU v3-8.
  - BASE size (43.3MB).

2. **TINY**, last update 17th April 2020, [albert-tiny-2020-04-17.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/albert-tiny-2020-04-17.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw twitter, raw instagram, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession.
  - 1M steps, TPU v3-8.
  - TINY size (21MB).

3. **BASE**, last update 19th October 2020, [albert-base-500k-19-10-2020.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/albert-base-500k-19-10-2020.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw common crawl, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession.
  - 500k steps, TPU v3-8.
  - BASE size (43.3MB).
  - Tensorboard, https://tensorboard.dev/experiment/pOunE2WwSAGroQgNmXDFqQ/

4. **TINY**, last update 19th October 2020, [albert-tiny-700k-19-10-2020.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/albert-tiny-700k-19-10-2020.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw common crawl, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession.
  - 700k steps, TPU v3-8.
  - TINY size (21MB).
  - Tensorboard, https://tensorboard.dev/experiment/Sjx1sEF6TQeDXIJ4J6C5iQ/

5. **LARGE**, last update 19th October 2020, [albert-large-475k-19-10-2020.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/albert-large-475k-19-10-2020.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw common crawl, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession.
  - 475k steps, TPU v3-8.
  - LARGE size.
  - Tensorboard, https://tensorboard.dev/experiment/LT9KbeWHRTOWi6vqqJuqeQ/

## Citation

1. Please citate the repository if use these checkpoints.

```
@misc{Malaya, Natural-Language-Toolkit library for bahasa Malaysia, powered by Deep Learning Tensorflow,
  author = {Husein, Zolkepli},
  title = {Malaya},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huseinzol05/malaya}}
}
```

2. Please at least email us first before distributing these checkpoints. Remember all these hard workings we want to give it for free.
3. What do you see just the checkpoints, but nobody can see how much we spent our cost to make it public.

## Donation

<a href="https://www.patreon.com/bePatron?u=7291337"><img src="https://static1.squarespace.com/static/54a1b506e4b097c5f153486a/t/58a722ec893fc0a0b7745b45/1487348853811/patreon+art.jpeg" width="40%"></a>

Or, One time donation without credit card hustle, **7053174643, CIMB Bank, Husein Zolkepli**
