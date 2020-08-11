# Seq2Seq-Bahasa

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [Download](#download)
  * [Test](#test)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. Provide alternative Seq2Seq model like T5. We combined BERT as encoder and vanilla-transformer decoder as decoder.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/) and [KeyReply](https://www.keyreply.com/) for sponsoring AWS, Google and GPU clouds to train BERT for Bahasa.

## How-to

```bash
python3 run_pretraining.py --bert_config_file=gs://mesolitica-general/b2b-data/BASE_config.json --input_file=gs://mesolitica-general/b2b-data/*.tfrecord --output_dir=gs://mesolitica-general/b2b-base --do_train=True --train_batch_size=160 --save_checkpoints_steps=5000 --tpu_name=node-1 --tpu_zone=us-central1-a --gcp_project=mesolitica-cloud --use_tpu=True --num_train_steps=500000
```

## Download

1. 5th August 2020, [b2t.tar.gz](https://f000.backblazeb2.com/file/malaya-model/b2t.tar.gz)

## Test

Check [test-base.ipynb](test-base.ipynb).

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