# ALBERT-Bahasa

Thanks to official implementation from Google, https://github.com/google-research/google-research, Malaya just create custom pretraining and optimizer to support multigpus.

**Tested on 4 Tesla V100 mirror strategy, loss is not decreasing, use [albert-brightmart](../albert-brightmart) instead.**

## Table of contents
  * [Objective](#objective)
  * [Acknowledgement](#acknowledgement)
  * [How-to](#how-to)
  * [Citation](#citation)
  * [Donation](#donation)

## Objective

1. There is no multilanguage implementation of ALBERT, and obviously no Bahasa Malaysia implemented. So we decided to train ALBERT from scratch and finetune using available dataset we have. [Dataset we use for pretraining](https://github.com/huseinzol05/Malaya-Dataset#dumping).

2. Provide **BASE** and **LARGE** ALBERT for Bahasa.

## Acknowledgement

Thanks to [Im Big](https://www.facebook.com/imbigofficial/), [LigBlou](https://www.facebook.com/ligblou), [Mesolitica](https://mesolitica.com/) and [KeyReply](https://www.keyreply.com/) for sponsoring AWS, Google and GPU clouds to train ALBERT for Bahasa.

## How-to

1. Create a folder and copy these python scripts,

```bash
mkdir albert
cd albert
cp *.py albert
pip3 install albert-tensorflow
```

2. Run [preprocess](../preprocess).

3. Create pretraining dataset,

```bash
python3 create_pretraining_data.py \
--input_file=../parliament-text.txt,../wiki-text.txt,../dumping-instagram.txt,../dumping-twitter.txt,../news-text.txt \
--output_file=albert1.tfrecord,albert2.tfrecord,albert3.tfrecord --vocab_file=sp10m.cased.v8.vocab \
--spm_model_file=sp10m.cased.v8.model --do_lower_case=False --dupe_factor=5
```

4. Execute pretraining,

Run pretraining on Wikipedia on parliament texts,

```bash
python3 multigpu_pretraining.py --input_file=../bert/bert-0.tfrecord,../bert/bert-1.tfrecord,../bert/bert-3.tfrecord --output_dir=pretraining_output --output_dir=pretraining_output --do_train=True --do_eval=False --albert_config_file=BASE_config.json --train_batch_size=300 --num_train_steps=2000000 --learning_rate=2e-5 --save_checkpoints_steps=100000 --use_gpu=True --num_gpu_cores=3 --max_seq_length=128
```

**ALBERT required multiGPUs or multiTPUs to pretrain. I never had successful pretraining on single GPU even on a small dataset.**

- `num_gpu_cores`: Number of gpus.
- `train_batch_size`: Make sure `train_batch_size` % `num_gpu_cores` is 0 and the batch will automatically distribute among gpus. If `num_gpu_cores` is 60 and `num_gpu_cores` is 2, so each gpus will get 30 batch size.

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
