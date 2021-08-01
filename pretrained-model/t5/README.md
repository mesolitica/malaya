# T5-Bahasa

Thanks to Google for opensourcing most of the source code to develop T5, https://github.com/google-research/text-to-text-transfer-transformer.

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
  * [Download](#download)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. Provide **SMALL**, **BASE** and **LARGE** T5 for Bahasa.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/), [KeyReply](https://www.keyreply.com/) and [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc) for sponsoring AWS, Google and GPU clouds to train T5 for Bahasa.

## How-to

**training session required TPU**,

1. Follow README in [tokenizer](tokenizer) for tokenizer and steps to generate it.

2. Run all generate notebooks in [prepare](prepare).

3. Upload data to GCS, [upload-gcs.ipynb](prepare-upload-gcs.ipynb).

4. Validate dataset from GCS, [test-dataset-gcs.ipynb](test-dataset-gcs.ipynb).

5. Train using TPU,

**SMALL**,

```bash
python3 train_tpu_small.py
```

**BASE**,

```bash
python3 train_tpu_base.py
```

**LARGE**,

```bash
python3 train_tpu_large.py
```

## Download

1. **BASE**, last update 20th May 2020, [t5-base-05-20-2020.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/t5-base-05-20-2020.tar.gz)

  - Vocab size 32k.
  - Trained on unsupervised, question-answer, pairing, news-title, stemming, synonym and SNLI tasks.
  - 487.6k steps, 128 batch size, 1 V3-8 TPU.
  - BASE size (427MB).
  - Tensorboard included.

2. **SMALL**, last update 1st October 2020, [t5-small-10-01-2020.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/t5-small-10-01-2020.tar.gz)

  - Vocab size 32k.
  - Trained on unsupervised, question-answer, pairing, news-title, stemming, synonym and SNLI tasks.
  - 500k steps, 256 batch size, 1 V3-8 TPU.
  - SMALL size (112MB).
  - Tensorboard, https://tensorboard.dev/experiment/v0ZAAd2cQjOGO6ETmA1IRQ/

3. **BASE**, last update 28th July 2021, [t5-base-2021-07-28.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/t5-base-2021-07-28.tar.gz)

  - Vocab size 32k.
  - Trained on unsupervised, question-answer, pairing, news-title, news summarization, EN-MS translation, MS-EN translation, knowledge gprah triplets, paraphrase and SNLI tasks.
  - 1M steps, 1 V3-8 TPU.
  - BASE size (427MB).

4. **SMALL**, last update 28th July 2021, [t5-small-2021-07-28.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/t5-small-2021-07-28.tar.gz)

  - Vocab size 32k.
  - Trained on unsupervised, question-answer, pairing, news-title, news summarization, EN-MS translation, MS-EN translation, knowledge gprah triplets, paraphrase and SNLI tasks.
  - 1M steps, 1 V3-8 TPU.
  - SMALL size (112MB).

5. **TINY**, last update 28th July 2021, [t5-tiny-2021-07-28.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/t5-tiny-2021-07-28.tar.gz)

  - Vocab size 32k.
  - Trained on unsupervised, question-answer, pairing, news-title, news summarization, EN-MS translation, MS-EN translation, knowledge gprah triplets, paraphrase and SNLI tasks.
  - 1M steps, 1 V3-8 TPU.
  - TINY size (66.8MB).

5. **SUPER TINY**, last update 28th July 2021, [t5-super-tiny-2021-07-28.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/t5-tiny-2021-07-28.tar.gz)

  - Vocab size 32k.
  - Trained on unsupervised, question-answer, pairing, news-title, news summarization, EN-MS translation, MS-EN translation, knowledge gprah triplets, paraphrase and SNLI tasks.
  - 1M steps, 1 V3-8 TPU.
  - SUPER TINY size (25.6MB).

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


