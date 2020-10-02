# BERT-Bahasa

Thanks to Google for opensourcing most of the source code to develop BERT, https://github.com/google-research/bert. Malaya just create custom pretraining and optimizer to support multigpus.

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
    * [Multigpus](#multigpus)
  * [Download](#download)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. Provide **SMALL**, **BASE** and **LARGE** BERT for Bahasa.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/), [KeyReply](https://www.keyreply.com/) and [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc) for sponsoring AWS, Google and GPU clouds to train BERT for Bahasa.

## How-to

1. Run [build-wordpiece.ipynb](build-wordpiece.ipynb).

2. Create pretraining dataset,

```bash
python3 create-pretraining-data.py
```

3. Execute pretraining,

```bash
python3 multigpu_pretraining.py \
--input_file=dumping-*.tfrecord \
--output_dir=pretraining_output5 \
--do_train=True \
--do_eval=False \
--bert_config_file=BASE_config.json \
--train_batch_size=150 \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--num_train_steps=1000000 \
--num_warmup_steps=10 \
--learning_rate=2e-5 \
--save_checkpoints_steps=200000 \
--use_gpu=True \
--num_gpu_cores=3 \
--eval_batch_size=12
```

- `num_gpu_cores`: Number of gpus.
- `train_batch_size`: Make sure `train_batch_size` % `num_gpu_cores` is 0 and the batch will automatically distribute among gpus. If `num_gpu_cores` is 60 and `num_gpu_cores` is 2, so each gpus will get 30 batch size.

**TPU BASE**,

```bash
python3 run_pretraining.py \
  --input_file=gs://mesolitica-tpu-general/bert-data/*.tfrecord \
  --output_dir=gs://mesolitica-tpu-general/bert-base \
  --do_train=True \
  --do_eval=False \
  --bert_config_file=gs://mesolitica-tpu-general/bert-config/BASE_config.json \
  --train_batch_size=128 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=500000 \
  --learning_rate=2e-5 \
  --iterations_per_loop=100 \
  --tpu_name=node-3 \
  --tpu_zone=europe-west4-a \
  --save_checkpoints_steps=25000 \
  --use_tpu=True
```

4. Execute validation,

```bash
python3 validation.py --input_file=tests_output.tfrecord --output_dir=pretraining_output --bert_config_file=bert_config.json --train_batch_size=50 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=3000000 --num_warmup_steps=10 --learning_rate=2e-5
```

```text
INFO:tensorflow:***** Eval results *****
I0910 11:20:31.561826 140220436277056 validation.py:595] ***** Eval results *****
INFO:tensorflow:  global_step = 480048
I0910 11:20:31.561924 140220436277056 validation.py:597]   global_step = 480048
INFO:tensorflow:  loss = 3.5268908
I0910 11:20:31.562081 140220436277056 validation.py:597]   loss = 3.5268908
INFO:tensorflow:  masked_lm_accuracy = 0.46958354
I0910 11:20:31.562179 140220436277056 validation.py:597]   masked_lm_accuracy = 0.46958354
INFO:tensorflow:  masked_lm_loss = 3.1709714
I0910 11:20:31.562261 140220436277056 validation.py:597]   masked_lm_loss = 3.1709714
INFO:tensorflow:  next_sentence_accuracy = 0.7625
I0910 11:20:31.562338 140220436277056 validation.py:597]   next_sentence_accuracy = 0.7625
INFO:tensorflow:  next_sentence_loss = 0.3507687
I0910 11:20:31.562414 140220436277056 validation.py:597]   next_sentence_loss = 0.3507687
```

## Download

1. **BASE**, last update 30th July 2019, [bert-bahasa-base.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/bert-bahasa-base.tar.gz)

  - Vocab size 40k.
  - Trained on cleaned wikipedia, raw twitter, raw instagram, raw parliament, raw news.
  - 1.5M steps, single GPU.
  - BASE size (467MB).

2. **SMALL**, last update 2nd August 2019, [bert-bahasa-small.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/bert-bahasa-small.tar.gz)

  - Vocab size 40k.
  - Trained on cleaned wikipedia, raw twitter, raw instagram, raw parliament, raw news.
  - 1.5M steps, single GPU.
  - SMALL size (184MB).

3. **BASE**, last update 13th September 2019, [bert-bahasa-base-13-9-2019.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/bert-base-13-9-2019.tar.gz)

  - Vocab size 40k.
  - Trained on raw wikipedia, raw twitter, raw instagram, raw parliament, raw news.
  - 1.0M steps, 3 GPUs TESLA V100.
  - BASE size (467MB).

4. **SMALL**, last update 19th September 2019, [bert-small-19-9-2019.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/bert-small-19-9-2019.tar.gz)

  - Vocab size 40k.
  - Trained on raw wikipedia, raw twitter, raw instagram, raw parliament, raw news.
  - 1.0M steps, 3 GPUs TESLA V100.
  - SMALL size (184MB).

5. **BASE**, last update 19th March 2020, [bert-base-2020-03-19.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/bert-base-2020-03-19.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw twitter, raw instagram, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession.
  - 2.0M steps, 3 GPUs TESLA V100.
  - BASE size (467MB).

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
