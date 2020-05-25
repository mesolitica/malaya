# T5-Bahasa

Thanks to Google for opensourcing most of the source code to develop T5, https://github.com/google-research/text-to-text-transfer-transformer.

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
  * [Download](#download)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. Provide **SMALL** and **BASE** T5 for Bahasa.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/) and [KeyReply](https://www.keyreply.com/) for sponsoring AWS, Google and GPU clouds to train BERT for Bahasa.

## How-to

**training session required TPU**,

1. Download [sp10m.cased.t5.model](../preprocess/sp10m.cased.t5.model) and [sp10m.cased.t5.vocab](../preprocess/sp10m.cased.t5.vocab).

2. Generate [stemming data](generate-stemming.ipynb) and [synonyms data](generate-synonym.ipynb).

3. Upload data to GCS, [prepare-upload-gcs.ipynb](prepare-upload-gcs.ipynb).

4. Validate dataset from GCS, [test-dataset-gcs.ipynb](test-dataset-gcs.ipynb).

5. Train using TPU, [train_tpu.py](train_tpu.py).

## Download

1. 20th May 2020, [t5-base-05-20-2020.tar.gz](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/t5-base-05-20-2020.tar.gz)

  - Vocab size 32k.
  - Trained on unsupervised, question-answer, pairing, news-title, stemming, synonym and SNLI tasks.
  - 487.6k steps, 128 batch size, 1 V3-8 TPU.
  - BASE size (427MB).
  - Tensorboard included.

2. 25th May 2020, [t5-small-05-25-2020.tar.gz](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/t5-small-05-25-2020.tar.gz)

  - Vocab size 32k.
  - Trained on unsupervised, question-answer, pairing, news-title, stemming, synonym and SNLI tasks.
  - 249.9k steps, 256 batch size, 1 V3-8 TPU, stopped early because of loss plateu.
  - BASE size (112MB).
  - Tensorboard included.


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


