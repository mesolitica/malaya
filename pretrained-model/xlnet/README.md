# XLNET-Bahasa

Thanks to [zihangdai](https://github.com/zihangdai) for opensourcing XLNET, https://github.com/zihangdai/xlnet. Malaya just create custom pretraining and optimizer to support multigpus.

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

1. Provide **BASE**, **LARGE** XLNet for Bahasa.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/), [KeyReply](https://www.keyreply.com/) and [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc) for sponsoring AWS, Google and GPU clouds to train XLNET for Bahasa.

## How-to

1. Follow README in [tokenizer](tokenizer) for tokenizer and steps to generate it.

2. Convert text files to tfrecord,

```bash
mkdir text-files
cp /home/husein/pure-text/splitted/*.txt text-files
cp /home/husein/pure-text/the-pile/00.jsonl-1*.translated.txt text-files
```

```bash
mkdir save-location
python3 data_utils.py \
  --bsz_per_host=32 \
  --seq_len=512 \
  --reuse_len=256 \
  --input_glob=text-files/*.txt \
  --save_dir=save-location \
  --num_passes=20 \
  --bi_data=True \
  --sp_path=sp10m.cased.v9.model \
  --mask_alpha=6 \
  --mask_beta=1 \
  --num_predict=85 \
  --num_core_per_host=8 \
  --uncased=False
```

3. Run pretained,

**BASE**,

```bash
python3 train_gpu.py \
  --corpus_info_path=save-location/corpus_info.json \
  --record_info_dir=save-location/tfrecords \
  --train_batch_size=20 \
  --seq_len=512 \
  --reuse_len=256 \
  --mem_len=384 \
  --perm_size=256 \
  --n_layer=12 \
  --d_model=768 \
  --d_embed=768 \
  --n_head=12 \
  --d_head=64 \
  --d_inner=3072 \
  --untie_r=True \
  --mask_alpha=6 \
  --mask_beta=1 \
  --num_predict=85 \
  --model_dir=output-model \
  --uncased=False \
  --num_core_per_host=1 \
  --train_steps=2000000  --iterations=10 --learning_rate=5e-5 \
  --num_gpu_cores=2
```

**TPU TINY**
```
python3 train.py \
--record_info_dir=gs://mesolitica-tpu-general/xlnet-data/tfrecords \
--train_batch_size=32 \
--seq_len=512 \
--reuse_len=256 \
--mem_len=384 \
--perm_size=256 \
--n_layer=4 \
--d_model=384 \
--d_embed=384 \
--n_head=6 \
--d_head=64 \
--d_inner=1344 \
--untie_r=True \
--mask_alpha=6 \
--mask_beta=1 \
--num_predict=85 \
--model_dir=gs://mesolitica-tpu-general/xlnet-tiny-v3 \
--uncased=False \
--num_core_per_host=8 \
--train_steps=500000 \
--learning_rate=4e-4 \
--ff_activation=gelu \
--adam_epsilon=1e-6 \
--weight_decay=0.01 \
--warmup_steps=40000 \
--tpu=node-4 \
--tpu_zone=us-central1-f \
--dropout=0.0 --dropatt=0.0 \
--iterations=100 \
--save_steps=20000
```

**TPU BASE**
```
python3 train.py \
--record_info_dir=gs://mesolitica-tpu-general/xlnet-data/tfrecords \
--train_batch_size=32 \
--seq_len=512 \
--reuse_len=256 \
--mem_len=384 \
--perm_size=256 \
--n_layer=12 \
--d_model=768 \
--d_embed=768 \
--n_head=12 \
--d_head=64 \
--d_inner=3072 \
--untie_r=True \
--mask_alpha=6 \
--mask_beta=1 \
--num_predict=85 \
--model_dir=gs://mesolitica-tpu-general/xlnet-base-v2 \
--uncased=False \
--num_core_per_host=8 \
--train_steps=500000 \
--learning_rate=4e-4 \
--ff_activation=gelu \
--adam_epsilon=1e-6 \
--weight_decay=0.01 \
--warmup_steps=40000 \
--tpu=node-3 \
--tpu_zone=us-central1-f \
--dropout=0.1 --dropatt=0.1 \
--iterations=100 \
--save_steps=20000
```

**TPU LARGE**
```
python3 train.py \
--record_info_dir=gs://mesolitica-tpu-general/xlnet-data/tfrecords \
--train_batch_size=32 \
--seq_len=512 \
--reuse_len=256 \
--mem_len=384 \
--perm_size=256 \
--n_layer=20 \
--d_model=1024 \
--d_embed=1024 \
--n_head=16 \
--d_head=64 \
--d_inner=4096 \
--untie_r=True \
--mask_alpha=6 \
--mask_beta=1 \
--num_predict=85 \
--model_dir=gs://mesolitica-tpu-general/xlnet-large-v2 \
--uncased=False \
--num_core_per_host=8 \
--train_steps=500000 \
--learning_rate=4e-4 \
--ff_activation=gelu \
--adam_epsilon=1e-6 \
--weight_decay=0.01 \
--warmup_steps=100000 \
--tpu=node-9 \
--tpu_zone=us-central1-f \
--dropout=0.1 --dropatt=0.1 \
--iterations=100 \
--save_steps=20000
```

3. Run validation,

```bash
python3 validation.py \
  --corpus_info_path=save-location/corpus_info.json \
  --record_info_dir=save-location/tfrecords \
  --train_batch_size=20 \
  --seq_len=512 \
  --reuse_len=256 \
  --mem_len=384 \
  --perm_size=256 \
  --n_layer=12 \
  --d_model=768 \
  --d_embed=768 \
  --n_head=12 \
  --d_head=64 \
  --d_inner=3072 \
  --untie_r=True \
  --mask_alpha=6 \
  --mask_beta=1 \
  --num_predict=85 \
  --model_dir=output-model \
  --uncased=False \
  --num_core_per_host=1
```

#### Multigpus

I really not suggest to use multi-gpus from original XLNET implementation, not optimized and can lead to huge memory leak. Here I created MirroredStrategy to pretrain using multi-gpus.

1. Run [multigpu_pretraining.py](multigpu_pretraining.py),

Run multigpus using MirroredStrategy,

```bash
python3 multigpu_pretraining.py \
  --corpus_info_path=save-location/corpus_info.json \
  --record_info_dir=save-location/tfrecords \
  --train_batch_size=60 \
  --seq_len=512 \
  --reuse_len=256 \
  --mem_len=384 \
  --perm_size=256 \
  --n_layer=12 \
  --d_model=768 \
  --d_embed=768 \
  --n_head=12 \
  --d_head=64 \
  --d_inner=3072 \
  --untie_r=True \
  --mask_alpha=6 \
  --mask_beta=1 \
  --num_predict=85 \
  --model_dir=output-model \
  --uncased=False \
  --num_core_per_host=1 \
  --train_steps=600000 \
  --iterations=10 \
  --learning_rate=5e-5 \
  --num_gpu_cores=3 \
  --save_steps=6000 \
  --ff_activation=gelu
```

- `num_gpu_cores`: Number of gpus.
- `train_batch_size`: If `bsz_per_host` during `data_utils.py` is 20, so `train_batch_size` must `bsz_per_host` * `num_gpu_cores`. Make sure `train_batch_size` % `num_gpu_cores` is 0 and the batch will automatically distribute among gpus. If `num_gpu_cores` is 60 and `num_gpu_cores` is 2, so each gpus will get 30 batch size.

## Download

1. **BASE**, last update 29th March 2020, [xlnet-base-29-03-2020.tar.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/xlnet-base-29-03-2020.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw twitter, raw instagram, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession.
  - 300k steps, 3 GPUs TESLA V100.
  - BASE size (467MB).

2. **BASE**, last update 20th October 2020, [xlnet-base-500k-20-10-2020.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/xlnet-base-500k-20-10-2020.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw common crawl, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession.
  - 500k steps, 1 TPU V3-8.
  - BASE size (467MB).
  - Tensorboard, https://tensorboard.dev/experiment/qGNflXX1R22GJ2Pk8ECDmA/

3. **LARGE**, last update 20th October 2020, [xlnet-large-500k-20-10-2020.gz](https://f000.backblazeb2.com/file/malaya-model/bert-bahasa/xlnet-large-500k-20-10-2020.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw common crawl, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession.
  - 500k steps, 1 TPU V3-8.
  - LARGE size.
  - Tensorboard, https://tensorboard.dev/experiment/B2RQd2yATZOjsAayQm8ncA/

4. **TINY**, last update 6th September 2021, [xlnet-tiny-2021-09-06.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/xlnet-tiny-2021-09-06.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession and translated The Pile.
  - 500k steps, 1 TPU V2-8.
  - TINY size.

5. **BASE**, last update 6th September 2021, [xlnet-base-2021-09-06.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/xlnet-base-2021-09-06.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession and translated The Pile.
  - 500k steps, 1 TPU V2-8.
  - BASE size.

5. **LARGE**, last update 6th September 2021, [xlnet-large-2021-09-06.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/xlnet-large-2021-09-06.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw parliament, raw news, raw wattpad, raw academia, raw iium-confession and translated The Pile.
  - 500k steps, 1 TPU V3-8.
  - LARGE size).

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
