# ERNIE-Bahasa

Thanks to Google for opensourcing most of the source code BERT to develop ERNIE, https://github.com/google-research/ernie

## Table of contents
  * [Objective](https://github.com/huseinzol05/Malaya/tree/master/ernie#objective)
  * [How-to](https://github.com/huseinzol05/Malaya/tree/master/ernie#how-to)
  * [Citation](https://github.com/huseinzol05/Malaya/tree/master/ernie#citation)
  * [Donation](https://github.com/huseinzol05/Malaya/tree/master/ernie#donation)

## Objective

1. There is no multilanguage implementation of ERNIE, and obviously no Bahasa Malaysia implemented. So we decided to train ERNIE from scratch and finetune using available dataset we have. [Dataset we use for pretraining](https://github.com/huseinzol05/Malaya-Dataset#dumping).

2. Provide **SMALL** and **BASE** ERNIE for Bahasa.

## How-to

1. Run [dumping.ipynb](dumping.ipynb) to create text dataset for pretraining.

You need to download `sp10m.cased.v4.model`, you can get this tokenizer from any checkpoints below after extract.

**_We implemented our own tokenizer because Google not open source WordPiece tokenizer, [stated here](https://github.com/google-research/bert#learning-a-new-wordpiece-vocabulary)._**

2. git clone https://github.com/google-research/bert, and copy [create-pretraining-data.py](create-pretraining-data.py), [prepro_utils.py](prepro_utils.py) inside bert folder,

```bash
git clone https://github.com/google-research/bert.git
cd bert
cp create-pretraining-data.py
cp prepro_utils.py
```

3. Create pretraining dataset,
```bash
python3 create-pretraining-data.py
```

4. Execute pretraining,
```bash
mkdir pretraining_output
python3 run_pretraining.py --input_file=tests_output.tfrecord --output_dir=pretraining_output --do_train=True --do_eval=True --bert_config_file=checkpoint/bert_config.json --train_batch_size=50 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=3000000 --num_warmup_steps=10 --learning_rate=2e-5 --save_checkpoints_steps=500000
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
