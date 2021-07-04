# FNET-Bahasa

Original paper, https://arxiv.org/pdf/2105.03824.pdf

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
  * [Download](#download)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. Provide **BASE** and **LARGE** FNET for Bahasa.

## Acknowledgement

Thanks to [KeyReply](https://www.keyreply.com/) and [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc) for sponsoring AWS, Google and GPU clouds to train FNET for Bahasa.

**TPU BASE**,

## How-to

1. Execute pretraining,

**BASE**,

```bash
python3 run_pretraining_base_tpu.py \
--input_file=gs://mesolitica-tpu-general/bert-data-v2/*.tfrecord \
--output_dir=fnet-base \
--do_train=True \
--do_eval=False \
--train_batch_size=32 \
--max_seq_length=512 \
--max_predictions_per_seq=20 \
--num_train_steps=500000 \
--learning_rate=5e-5 \
--iterations_per_loop=100 \
--save_checkpoints_steps=25000 \
--use_tpu=False
```

**LARGE**,

```bash
python3 run_pretraining_large_tpu.py \
--input_file=gs://mesolitica-tpu-general/bert-data-v2/*.tfrecord \
--output_dir=fnet-large \
--do_train=True \
--do_eval=False \
--train_batch_size=16 \
--max_seq_length=512 \
--max_predictions_per_seq=20 \
--num_train_steps=500000 \
--learning_rate=5e-5 \
--iterations_per_loop=100 \
--save_checkpoints_steps=25000 \
--use_tpu=False
```

## Downloads

1. **BASE**, last update 28th June 2021, [fnet-base.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/fnet-base.tar.gz)

  - Vocab size 32k.
  - 475k steps, 1 Tesla V100 32GB VRAM.

1. **LARGE**, last update 28th June 2021, [fnet-large.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/fnet-large.tar.gz)

  - Vocab size 32k.
  - 475k steps, 1 Tesla V100 32GB VRAM.

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