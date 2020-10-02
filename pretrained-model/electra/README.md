# ELECTRA-Bahasa

Thanks to Google for opensourcing most of the source code to develop ELECTRA, https://github.com/google-research/electra. Malaya just create custom pretraining and optimizer to support multigpus.

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
  * [Download](#download)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. Provide **SMALL**, **BASE** and **LARGE** ELECTRA for Bahasa.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/), [KeyReply](https://www.keyreply.com/) and [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc) for sponsoring AWS, Google and GPU clouds to train ELECTRA for Bahasa.

## How-to

1. Run [preprocess](../preprocess).

2. git clone https://github.com/google-research/electra, and,

```bash
git clone https://github.com/google-research/electra
cd electra
cp multigpu_pretraining.py custom_optimization.py
```

3. Create pretraining dataset,

For **SMALL**,

```bash
mkdir text-files
cp dumping-*.txt ./text-files
python3 build_pretraining_dataset.py \
--corpus-dir text-files \
--vocab-file bahasa.wordpiece \
--output-dir dataset \
--num-processes 10 \
--no-lower-case
```

For **BASE** / **LARGE**,

```bash
mkdir text-files
cp dumping-*.txt ./text-files
python3 build_pretraining_dataset.py \
--corpus-dir text-files \
--vocab-file bahasa.wordpiece \
--output-dir dataset \
--num-processes 10 \
--max-seq-length 512 \
--no-lower-case
```

3. Create pretraining dataset,

```bash
python3 create-pretraining-data.py
```

4. Execute pretraining,

For **SMALL**,

```bash
python3 run_pretraining.py --data-dir directory \
--model-name electra-small \
--hparams SMALL-config.json
```

For **BASE**,

```bash
python3 run_pretraining.py --data-dir directory \
--model-name electra-base \
--hparams gs://bucket/BASE-config-tpu.json
```

## Download

1. **SMALL**, last update 27th April 2020, [electra-bahasa-small-27-04-2020.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/electra-bahasa-small-27-04-2020.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw twitter, raw instagram, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession.
  - 1.0M steps, 1 GPU TESLA V100.
  - SMALL size (55MB).

1. **BASE**, last update 27th April 2020, [electra-bahasa-base-27-04-2020.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/electra-bahasa-base-27-04-2020.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw twitter, raw instagram, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession.
  - 728.8k steps, V3-8 TPU
  - BASE size (443MB).

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
