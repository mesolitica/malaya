## Word2Vec Bahasa

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
  * [Download](#download)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. Provide word2vec for bahasa wikipedia, news and social media

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/) and [KeyReply](https://www.keyreply.com/) for sponsoring AWS, Google and GPU clouds to train BERT for Bahasa.

## How-to

1. Run [train.py](train.py),

```bash
python3 train.py -t text.txt
```

For more info,

```bash
python3 train.py -h
```

```text
usage: train.py [-h] -t TEXT -e EMBEDDING [-b BATCH] [-v VOCAB]
                [-lr LEARNING_RATE] [-epoch EPOCH]

optional arguments:
  -h, --help            show this help message and exit
  -t TEXT, --text TEXT  text file to train
  -e EMBEDDING, --embedding EMBEDDING
                        embedding size to train
  -b BATCH, --batch BATCH
                        batch size, default is 128
  -v VOCAB, --vocab VOCAB
                        maximum vocab size, default is 1000000
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate, default is 0.01
  -epoch EPOCH, --epoch EPOCH
                        epoch size, default is 10
```

## Download

1. Bahasa Wikipedia, last update 8th January 2020.

  - Vocab size 763350.
  - Trained on cleaned wikipedia, lower case.
  - 10 epochs.
  - embedding size 256.

[download vector](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/word2vec-wiki-ms-256.npy), [download dictionary](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/word2vec-wiki-ms-256.json)

2. Social media malaysia, last update 8th January 2020.

  - Vocab size 1294638.
  - Trained on cleaned twitter and instagram, lower case.
  - 10 epochs.
  - embedding size 256.

[download vector](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/word2vec-ms-socialmedia-256.npy), [download dictionary](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/word2vec-ms-socialmedia-256.json)

3. News malaysia, last update 8th January 2020.

  - Vocab size 195466.
  - Trained on cleaned news, lower case.
  - 10 epochs.
  - embedding size 256.

[download vector](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/word2vec-news-ms-256.npy), [download dictionary](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/word2vec-news-ms-256.json)

4. Wikipedia + Social media + News, last update 9th January 2020.

  - Vocab size 1903143.
  - Trained on cleaned news + social media + wikipedia, lower case.
  - 10 epochs.
  - embedding size 256.

[download vector](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/word2vec-combined-256.npy), [download dictionary](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/word2vec-combined-256.json)

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