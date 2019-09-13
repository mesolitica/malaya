# BERT-Bahasa

Thanks to Google for opensourcing most of the source code to develop BERT, https://github.com/google-research/bert

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
    * [Multigpus](#multigpus)
  * [Download](#download)
  * [Comparison using Subjectivity Dataset](#comparison-using-subjectivity-dataset)
  * [Comparison using Emotion Dataset](#comparison-using-emotion-dataset)
  * [Comparison using Text Similarity Dataset](#comparison-using-text-similarity-dataset)
  * [Feedbacks](#feedbacks)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. We saw tokenization process from original BERT Multilanguage is not really targeted to Malaysia language landscape, and pretrained provided only trained on Wikipedia dataset, no social media texts (bahasa pasar). So we decided to train BERT from scratch and finetune using available dataset we have. [Dataset we use for pretraining](https://github.com/huseinzol05/Malaya-Dataset#dumping).

2. Provide **SMALL**, **BASE** and **LARGE** BERT for Bahasa.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou) and [Mesolitica](https://mesolitica.com/) for sponsoring AWS and Google cloud to train BERT for Bahasa.

## How-to

1. Run [dumping.ipynb](dumping.ipynb) to create text dataset for pretraining.

You need to download [sp10m.cased.v4.model](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/v27/sp10m.cased.v4.model) and [sp10m.cased.v4.vocab](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/v27/sp10m.cased.v4.vocab) first.

**_We implemented our own tokenizer because Google not open source WordPiece tokenizer, [stated here](https://github.com/google-research/bert#learning-a-new-wordpiece-vocabulary)._**

2. git clone https://github.com/google-research/bert, and,

```bash
git clone https://github.com/google-research/bert.git
cd bert
cp create-pretraining-data.py prepro_utils.py multigpu_pretraining.py custom_optimization.py
```

3. Create pretraining dataset,
```bash
python3 create-pretraining-data.py
```

4. Execute pretraining,
```bash
python3 run_pretraining.py --input_file=tests_output.tfrecord --output_dir=pretraining_output --do_train=True --do_eval=True --bert_config_file=bert_config.json --train_batch_size=50 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=3000000 --num_warmup_steps=10 --learning_rate=2e-5 --save_checkpoints_steps=500000
```

**LARGE** size, [LARGE_config.json](config/LARGE_config.json),
```json
{
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "max_position_embeddings": 512,
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "type_vocab_size": 2,
  "vocab_size": 40000
}

```

**BASE** size, [BASE_config.json](config/BASE_config.json),
```json
{
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "type_vocab_size": 2,
  "vocab_size": 40000
}
```

**SMALL** size, [SMALL_config.json](config/SMALL_config,json),
```json
{
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 512,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 256,
  "num_attention_heads": 8,
  "num_hidden_layers": 6,
  "pooler_fc_size": 512,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "type_vocab_size": 2,
  "vocab_size": 40000
}
```

**All training session will be recorded in Tensorboard**, to open tensorboard,
```bash
tensorboard --logdir=tensorboard --host=0.0.0.0
```

5. Execute validation,
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

#### Multigpus

Original BERT implementation not support multi-gpus, only single gpu. Here I created MirroredStrategy to pretrain using multi-gpus.

1. Run [multigpu_pretraining.py](multigpu_pretraining.py),

Run multigpus using MirroredStrategy,
```bash
python3 multigpu_pretraining.py \
--input_file=tests_output.tfrecord \
--output_dir=pretraining_output \
--do_train=True \
--do_eval=False \
--bert_config_file=bert_config.json \
--train_batch_size=90 \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--num_train_steps=500000 \
--num_warmup_steps=10 \
--learning_rate=2e-5 \
--save_checkpoints_steps=20000 \
--use_gpu=True \
--num_gpu_cores=3 \
--eval_batch_size=12
```

- `num_gpu_cores`: Number of gpus.
- `train_batch_size`: Make sure `train_batch_size` % `num_gpu_cores` is 0 and the batch will automatically distribute among gpus. If `num_gpu_cores` is 60 and `num_gpu_cores` is 2, so each gpus will get 30 batch size.

## Download

1. **BASE**, last update 30th July 2019, [bert-bahasa-base.tar.gz](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/bert-bahasa-base.tar.gz) [Tensorboard data](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/bert-base-30-july-2019-tensorboard.instance-3)

  1. Vocab size 40k.
  2. Trained on cleaned wikipedia, raw twitter, raw instagram, raw parliament, raw news.
  3. 1.5M steps, single GPU.
  4. BASE size (467MB).

2. **SMALL**, last update 2nd August 2019,
[bert-bahasa-small.tar.gz](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/bert-bahasa-small.tar.gz) [Tensorboard data](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/events.out.tfevents.1564477991.instance-3)

  1. Vocab size 40k.
  2. Trained on cleaned wikipedia, raw twitter, raw instagram, raw parliament, raw news.
  3. 1.5M steps, single GPU.
  4. SMALL size (184MB).

3. **BASE**, last update 13th September 2019,
[bert-bahasa-base-13-9-2019.tar.gz](https://huseinhouse-storage.s3-ap-southeast-1.amazonaws.com/bert-bahasa/bert-base-13-9-2019.tar.gz)

  1. Vocab size 40k.
  2. Trained on raw wikipedia, raw twitter, raw instagram, raw parliament, raw news.
  3. 1.0M steps, 3 GPUs.
  4. BASE size (467MB).

```text
INFO:tensorflow:***** Eval results *****
I0913 04:02:36.334070 140621913737024 validation.py:595] ***** Eval results *****
INFO:tensorflow:  global_step = 1000002
I0913 04:02:36.334207 140621913737024 validation.py:597]   global_step = 1000002
INFO:tensorflow:  loss = 3.2660308
I0913 04:02:36.334458 140621913737024 validation.py:597]   loss = 3.2660308
INFO:tensorflow:  masked_lm_accuracy = 0.49669307
I0913 04:02:36.334600 140621913737024 validation.py:597]   masked_lm_accuracy = 0.49669307
INFO:tensorflow:  masked_lm_loss = 2.9214077
I0913 04:02:36.334724 140621913737024 validation.py:597]   masked_lm_loss = 2.9214077
INFO:tensorflow:  next_sentence_accuracy = 0.78
I0913 04:02:36.334844 140621913737024 validation.py:597]   next_sentence_accuracy = 0.78
INFO:tensorflow:  next_sentence_loss = 0.33995274
I0913 04:02:36.334962 140621913737024 validation.py:597]   next_sentence_loss = 0.33995274
```

## Comparison using Subjectivity Dataset

Link to [subjectivity dataset](https://github.com/huseinzol05/Malaya-Dataset#subjectivity).

Link to [notebooks](finetune-subjectivity).

<img src="barplot/subjective.png" width="70%" align="">

## Comparison using Emotion Dataset

Link to [emotion dataset](https://github.com/huseinzol05/Malaya-Dataset#emotion).

Link to [notebooks](finetune-emotion).

<img src="barplot/emotion.png" width="70%" align="">

## Comparison using Text Similarity Dataset

Link to [text similarity dataset](https://github.com/huseinzol05/Malaya-Dataset#text-similarity).

Link to [notebooks](finetune-similarity).

<img src="barplot/similarity.png" width="70%" align="">

## Feedbacks

1. Feel free to suggest me to add more any kind of finetune, like, QA, Neural Machine Translation and etc.

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
