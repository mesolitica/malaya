# ELMO-Bahasa

**_Last update 1-August-2019, release new pretrained checkpoint._**

## Table of contents
  * [Objective](https://github.com/huseinzol05/Malaya/tree/master/elmo#objective)
  * [Acknowledgement](https://github.com/huseinzol05/Malaya/tree/master/elmo#acknowledgement)
  * [How-to](https://github.com/huseinzol05/Malaya/tree/master/elmo#how-to)
  * [Download](https://github.com/huseinzol05/Malaya/tree/master/elmo#download)
  * [Citation](https://github.com/huseinzol05/Malaya/tree/master/elmo#citation)
  * [Donation](https://github.com/huseinzol05/Malaya/tree/master/elmo#donation)

## Objective

1. train ELMO from scratch and finetune using available dataset we have. [Dataset we use for pretraining](https://github.com/huseinzol05/Malaya-Dataset#dumping).

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou) and [Mesolitica](https://mesolitica.com/) for sponsoring AWS and Google cloud to train XLNET for Bahasa.

## How-to

1. Download dumped wikipedia, [this one, 240.2 MB, 1663373 sentences](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/dumping-wiki-6-july-2019.json) and [this one, 255.1 MB, 1303844 sentences]( download link).

2. Run [elmo-128.ipynb](elmo-128.ipynb), make sure you tune parameters as you wanted,

```python
batch_size = 256
n_train_tokens = len(dictionary)
options = {
    'bidirectional': True,
    'char_cnn': {
        'activation': 'relu',
        'embedding': {'dim': 128},
        'filters': [
            [1, 32],
            [2, 32],
            [3, 64],
            [4, 128],
            [5, 256],
            [6, 512],
            [7, 1024],
        ],
        'max_characters_per_token': 50,
        'n_characters': 261,
        'n_highway': 2,
    },
    'dropout': 0.1,
    'lstm': {
        'cell_clip': 3,
        'dim': 256,
        'n_layers': 2,
        'projection_dim': 128,
        'proj_clip': 3,
        'use_skip_connections': True,
    },
    'all_clip_norm_val': 10.0,
    'n_epochs': 500,
    'n_train_tokens': n_train_tokens,
    'batch_size': batch_size,
    'n_tokens_vocab': uni.size,
    'unroll_steps': 20,
    'n_negative_samples_batch': 0.001,
    'sample_softmax': True,
    'share_embedding_softmax': False,
}
```

## Download

1.  1st August 2019, [elmo-128.tar.gz]().

**Vocab size 678833, Case Sensitive, 66k steps batch size 256, 128 embedded size (430MB)**.

1.  1st August 2019, [elmo-256.tar.gz]().

**Vocab size 678833, Case Sensitive, 66k steps batch size 128, 256 embedded size (430MB)**.

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
