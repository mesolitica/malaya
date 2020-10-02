# MASS-Bahasa

Thanks to [microsoft/MASS](https://github.com/microsoft/MASS/tree/master/MASS-supNMT) for wonderful paper. Even we do not use the source code, but we still follow literature steps inside the paper.

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
  * [Download](#download)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. Provide MASS-Base for Bahasa

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/) and [KeyReply](https://www.keyreply.com/) for sponsoring AWS, Google and GPU clouds to train Tiny-BERT for Bahasa.

## How-to

**training session required TPU**,

1. Download necessary data from https://github.com/huseinzol05/Malay-Dataset#translation.

2. Prepare dataset using [prepare-dataset-ms-en.ipynb](prepare-dataset-ms-en.ipynb) and [prepare-dataset-ms-en.ipynb](prepare-dataset-en-ms.ipynb).

3. Run single pair postprocessing, [preprocess-single-tfrecord.ipynb](preprocess-single-tfrecord.ipynb).

4. Run double pair postprocessing, [preprocessing-pair-tfrecord.ipynb](preprocess-pair-tfrecord.ipynb).

5. Run pretraining,

```bash
python3 run_pretraining.py --bert_config_file=gs://mesolitica-general/mass-data/BASE_config.json --input_file=gs://mesolitica-general/mass-data/*.tfrecord --output_dir=gs://mesolitica-general/mass-base --do_train=True --train_batch_size=320 --tpu_name=node-1 --tpu_zone=us-central1-a --gcp_project=mesolitica-cloud --use_tpu=True
```

## Download

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