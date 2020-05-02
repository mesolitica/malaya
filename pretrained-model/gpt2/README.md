# GPT2-Bahasa

Thanks to [openai/gpt2](https://github.com/openai/gpt-2) for opensourcing most of the source code to develop GPT2, https://github.com/openai/gpt-2 and [minimaxir/gpt-2-simple](https://github.com/minimaxir/gpt-2-simple). Malaya just edit the scripts to train on huge dataset.

**Do not finetuned this model to build fake news generator**.

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
  * [Download](#download)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. Provide GPT2-117M and GPT2-345M for Bahasa.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/) and [KeyReply](https://www.keyreply.com/) for sponsoring AWS, Google and GPU clouds to train Tiny-BERT for Bahasa.

## How-to

**training session required TPU**,

1. Download common-crawl data, [parse-common-crawl.ipynb](parse-common-crawl.ipynb).

2. Copy bahasa BPE tokenizer you need, in the script, I will use [bahasa-merges.txt](../preprocess/bahasa-merges.txt) and [bahasa-vocab.json](../preprocess/bahasa-vocab.json)

3. generate data for training session, [generate-data.ipynb](generate-data.ipynb).

4. Convert dataset into tfrecord, [convert-tfrecord.ipynb](convert-tfrecord.ipynb) and upload to your own GCS.

5. Download pretrained GPT2 from OpenAI repository, [download-pretrained-to-gcs.ipynb](download-pretrained-to-gcs.ipynb)

6. Start training, [train_tpu.py](train_tpu.py),

```bash
python3 train_tpu.py \
--input_file=gs://mesolitica-general/gpt2-data/dataset.tfrecord \
--output_dir=gs://mesolitica-general/gpt2-117m \
--tpu_name=node-1 --tpu_zone=us-central1-a --gcp_project=mesolitica-cloud \
--init_checkpoint=gs://mesolitica-general/gpt2-117M-pretrained/model.ckpt \
--do_train=True
```

## Download

1. **117M**, last update 30th April 2020, [117m-bahasa.tar.gz](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/117m-bahasa-v3.tar.gz)

  - Vocab size 57k.
  - Trained on raw wikipedia, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession and raw common-crawl, ~0.9B words.
  - 20k steps, 192 batch size, 1 V3-8 TPU.
  - perplexity, 5.4739473917272.

Use [small-hparams.json](small-hparams.json) to load parameter config.

1. **345M**, last update 1st May 2020, [345m-bahasa.tar.gz](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/345m-bahasa.tar.gz)

  - Vocab size 57k.
  - Trained on raw wikipedia, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession and raw common-crawl, ~0.9B words.
  - 55k steps, 64 batch size, 1 V3-8 TPU.
  - perplexity, 2.45960311115695

Use [base-hparams.json](base-hparams.json) to load parameter config.


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

