# XLNET-Bahasa

Thanks to [zihangdai](https://github.com/zihangdai) for opensourcing XLNET, https://github.com/zihangdai/xlnet. Malaya just create custom pretraining and optimizer to support multigpus.

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
    * [Multigpus](#multigpus)
  * [Download](#download)
  * [Comparison using Emotion Dataset](#comparison-using-emotion-dataset)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. There is no multilanguage implementation of XLNET, and obviously no Bahasa Malaysia implemented. So we decided to train XLNET from scratch and finetune using available dataset we have. [Dataset we use for pretraining](https://github.com/huseinzol05/Malaya-Dataset#dumping).

2. Provide **SMALL** and **BASE** XLNet for Bahasa.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/) and [KeyReply](https://www.keyreply.com/) for sponsoring AWS, Google and GPU clouds to train XLNET for Bahasa.

## How-to

1. Run [dumping.ipynb](dumping.ipynb) to create text dataset for pretraining.

2. Git clone [Sentence-Piece](https://github.com/google/sentencepiece),

```bash
git clone https://github.com/google/sentencepiece.git
```

3. Install [Sentence-Piece](https://github.com/google/sentencepiece),

On 23rd June 2019, we cannot use latest master to compile sentence-piece using bazel, after a few googled, we need to revert to some commit.

```bash
cd sentencepiece
git checkout d4dd947fe71c4fa4ee24ad8297beee32887d8828
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v
```

Make sure you tested to run `spm_train` to make sure everything is fine,

```bash
spm_train
```

```text
ERROR: --input must not be empty

sentencepiece

Usage: sentencepiece [options] files

   --accept_language (comma-separated list of languages this model can accept)  type: string  default:
   --add_dummy_prefix (Add dummy whitespace at the beginning of text)  type: bool  default: true
   --bos_id (Override BOS (<s>) id. Set -1 to disable BOS.)  type: int32  default: 1
   --bos_piece (Override BOS (<s>) piece.)  type: string  default: <s>
   --character_coverage (character coverage to determine the minimum symbols)  type: double  default: 0.9995
   --control_symbols (comma separated list of control symbols)  type: string  default:
   --eos_id (Override EOS (</s>) id. Set -1 to disable EOS.)  type: int32  default: 2
...
```

4. Create tokenizer using Sentence-Piece,

```bash
cd ../
spm_train \
--input=dumping-all.txt \
--model_prefix=sp10m.cased.v5 \
--vocab_size=32000 \
--character_coverage=0.99995 \
--model_type=unigram \
--control_symbols=\<cls\>,\<sep\>,\<pad\>,\<mask\>,\<eod\> \
--user_defined_symbols=\<eop\>,.,\(,\),\",-,–,£,€ \
--shuffle_input_sentence \
--input_sentence_size=10000000
```

5. Convert text files to tfrecord,

```bash
mkdir save-location
python3 data_utils.py \
  --bsz_per_host=20 \
  --seq_len=512 \
  --reuse_len=256 \
  --input_glob=../dumping-all.txt \
  --save_dir=save-location \
  --num_passes=20 \
  --bi_data=True \
  --sp_path=sp10m.cased.v5.model \
  --mask_alpha=6 \
  --mask_beta=1 \
  --num_predict=85 \
  --num_core_per_host=1 \
  --uncased=False
```

6. Run pretained,

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
  --d_model=512 \
  --d_embed=512 \
  --n_head=16 \
  --d_head=64 \
  --d_inner=2048 \
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

**SMALL**,

```bash
python3 train_gpu.py \
  --corpus_info_path=save-location/corpus_info.json \
  --record_info_dir=save-location/tfrecords \
  --train_batch_size=4 \
  --seq_len=512 \
  --reuse_len=256 \
  --mem_len=384 \
  --perm_size=256 \
  --n_layer=6 \
  --d_model=512 \
  --d_embed=512 \
  --n_head=16 \
  --d_head=64 \
  --d_inner=2048 \
  --untie_r=True \
  --mask_alpha=6 \
  --mask_beta=1 \
  --num_predict=85 \
  --model_dir=output-model \
  --uncased=False \
  --num_core_per_host=1 \
  --train_steps=700000  --iterations=10 --learning_rate=5e-5
```

7. Run validation,

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
  --d_model=512 \
  --d_embed=512 \
  --n_head=16 \
  --d_head=64 \
  --d_inner=2048 \
  --untie_r=True \
  --mask_alpha=6 \
  --mask_beta=1 \
  --num_predict=85 \
  --model_dir=output-model \
  --uncased=False \
  --num_core_per_host=1
```

#### Multigpus

I really not suggest to use multi-gpus from original XL-NET implementation, not optimized and can lead to huge memory leak. Here I created MirroredStrategy to pretrain using multi-gpus.

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
  --d_model=512 \
  --d_embed=512 \
  --n_head=16 \
  --d_head=64 \
  --d_inner=2048 \
  --untie_r=True \
  --mask_alpha=6 \
  --mask_beta=1 \
  --num_predict=85 \
  --model_dir=output-model \
  --uncased=False \
  --num_core_per_host=1 \
  --train_steps=300000 \
  --iterations=10 \
  --learning_rate=5e-5 \
  --num_gpu_cores=3 \
  --save_steps=30000 \
```

- `num_gpu_cores`: Number of gpus.
- `train_batch_size`: If `bsz_per_host` during `data_utils.py` is 20, so `train_batch_size` must `bsz_per_host` * `num_gpu_cores`. Make sure `train_batch_size` % `num_gpu_cores` is 0 and the batch will automatically distribute among gpus. If `num_gpu_cores` is 60 and `num_gpu_cores` is 2, so each gpus will get 30 batch size.

**All training session will be recorded in Tensorboard if use multigpu training**, to open tensorboard,

```bash
tensorboard --logdir=tensorboard --host=0.0.0.0
```

## Download

1.  9th July 2019, [xlnet-9-july-2019.tar.gz](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/xlnet-9-july-2019-v2.tar.gz).

  - Vocab size 32k.
  - Trained on cleaned wikipedia, raw twitter, raw instagram, raw parliament, raw news.
  - 700k steps, single GPU.
  - BASE size (878MB).
  - test loss, 4.XX (last remember).

2. 15 July 2019, [xlnet-15-july-2019.tar.gz](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/xlnet-bahasa-small.tar.gz)

  - Vocab size 32k.
  - Trained on cleaned wikipedia, raw twitter, raw instagram, raw parliament, raw news.
  - 700k steps, single GPU.
  - BASE size (231MB).
  - test loss, 4.XX (last remember).

3. 30 September 2019, [xlnet-base-30-september-2019.tar.gz](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/xlnet-base-30-9-2019.tar.gz)

  - Vocab size 32k.
  - Trained on raw wikipedia, raw twitter, raw instagram, raw parliament, raw news.
  - 270k steps, 3 GPUs TESLA V100.
  - BASE size (231MB).

```text
I0930 13:31:34.584934 140152603084608 evaluation.py:275] Finished evaluation at 2019-09-30-13:31:34
INFO:tensorflow:Saving dict for global step 270000: accuracy = 0.53644705, global_step = 270000, loss = 2.6062012
I0930 13:31:34.585190 140152603084608 estimator.py:2039] Saving dict for global step 270000: accuracy = 0.53644705, global_step = 270000, loss = 2.6062012
```

## Comparison using Emotion Dataset

Link to [emotion dataset](https://github.com/huseinzol05/Malaya-Dataset#emotion).

Link to [notebooks](transfer-learning-emotion-base.ipynb).

<img src="barplot/emotion.png" width="70%" align="">

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
