# Tiny-BERT-Bahasa

Thanks to [huawei-noah](https://github.com/huawei-noah) for opensourcing most of the source code to develop Tiny-BERT, https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT. Malaya just edit the scripts to accept sentencepience tokenizer and save it to Tensorflow.

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. Provide Tiny-BERT for Bahasa.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/) and [KeyReply](https://www.keyreply.com/) for sponsoring AWS, Google and GPU clouds to train BERT for Bahasa.

## How-to

1. Copy any sentencepiece tokenizer you need, in the script, I will use [sp10m.cased.v9.model](../preprocess/sp10m.cased.v9.model) and [sp10m.cased.v9.vocab](../preprocess/sp10m.cased.v9.vocab)

Simply edit [pregenerate_training_data.py](pregenerate_training_data.py),

```python
sp_model = spm.SentencePieceProcessor()
sp_model.Load('sp10m.cased.v9.model')

with open('sp10m.cased.v9.vocab') as fopen:
    v = fopen.read().split('\n')[:-1]
v = [i.split('\t') for i in v]
v = {i[0]: i[1] for i in v}
```

2. Run [pregenerate_training_data.py](pregenerate_training_data.py),

```bash
python pregenerate_training_data.py --train_corpus ${CORPUS_RAW} \
                  --reduce_memory --do_lower_case \
                  --epochs_to_generate 3 \
                  --output_dir ${CORPUS_JSON_DIR}$ 
```

3. Run [general_distill.py](general_distill.py),

```bash
# ${STUDENT_CONFIG_DIR}$ includes the config file of student_model.
python general_distill.py --pregenerated_data ${CORPUS_JSON}$ \ 
                          --teacher_model ${BERT_BASE}$ \
                          --student_model ${STUDENT_CONFIG_DIR}$ \
                          --reduce_memory --do_lower_case \
                          --train_batch_size 256 \
                          --output_dir ${GENERAL_TINYBERT_DIR}$ 
```

4. Run [bert-pytorch-to-tf1.ipynb](bert-pytorch-to-tf1.ipynb) to convert Pytorch to TF 1.X model.

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
