# GPT2-Bahasa

Thanks to [openai/gpt2](https://github.com/openai/gpt-2) for opensourcing most of the source code to develop GPT2, https://github.com/openai/gpt-2 and [minimaxir/gpt-2-simple](https://github.com/minimaxir/gpt-2-simple). Malaya just edit the scripts to train on huge dataset.

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
  * [Download](#download)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. Provide GPT2-117M and GPT2-345M for Bahasa.

## How-to

**training session required TPU**,

1. Prepare dataset, [prepare-tfrecord.py](prepare-tfrecord.py).

2. Start training, [train_tpu.py](train_tpu.py),

```bash
# using TPU v3-8
python3 train_tpu.py \
--input_file=gs://mesolitica-tpu-general/gpt2-data/*.tfrecord \
--output_dir=gs://mesolitica-tpu-general/gpt2-117m \
--config=gs://mesolitica-tpu-general/117m-hparams.json \
--tpu_name=node-8 --tpu_zone=europe-west4-a --gcp_project=mesolitica-tpu \
--learning_rate=2e-5 \
--batch_size=192 \
--num_train_steps=500000 \
--iterations_per_loop=100 \
--do_train=True
```

```bash
# using TPU v3-8
python3 train_tpu.py \
--input_file=gs://mesolitica-tpu-general/gpt2-data/*.tfrecord \
--output_dir=gs://mesolitica-tpu-general/gpt2-345m \
--config=gs://mesolitica-tpu-general/345m-hparams.json \
--tpu_name=node-1 --tpu_zone=europe-west4-a --gcp_project=mesolitica-tpu \
--learning_rate=2e-5 \
--batch_size=64 \
--num_train_steps=500000 \
--iterations_per_loop=100 \
--do_train=True
```

## Download

1. **117M**, last update 30th April 2020, [117m-bahasa.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/117m-bahasa-v3.tar.gz)

  - Vocab size 57k.
  - Trained on raw wikipedia, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession and raw common-crawl, ~0.9B words.
  - 20k steps, 192 batch size, 1 V3-8 TPU.
  - perplexity, 5.4739473917272.

Use [117m-hparams.json](117m-hparams.json) to load parameter config.

2. **345M**, last update 1st May 2020, [345m-bahasa.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/345m-bahasa.tar.gz)

  - Vocab size 57k.
  - Trained on raw wikipedia, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession and raw common-crawl, ~0.9B words.
  - 55k steps, 64 batch size, 1 V3-8 TPU.
  - perplexity, 2.45960311115695

Use [345m-hparams.json](345m-hparams.json) to load parameter config.

1. **117M**, last update 23rd September 2021, [117m-bahasa.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/117m-bahasa.tar.gz)

  - Vocab size 57k.
  - Trained on raw wikipedia, raw parliament, raw news, raw wattpad, raw academia, translated the pile and raw common-crawl, ~2B words.
  - perplexity, 6.232461.

Use [117m-hparams.json](117m-hparams.json) to load parameter config.

2. **345M**, last update 23rd September 2021, [345m-bahasa.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/345m-bahasa.tar.gz)

  - Vocab size 57k.
  - Trained on raw wikipedia, raw parliament, raw news, raw wattpad, raw academia, translated the pile and raw common-crawl, ~2B words.
  - perplexity, 2.45960311115695

Use [345m-hparams.json](345m-hparams.json) to load parameter config.

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

