# how-to

**This directory is very lack of comments, able to understand Tensorflow, Tensorflow estimator and Tensorflow Dataset are really helpful**.

## Table of contents
  * [prepare dataset](#prepare-dataset)
  * [training pretrained models](#training-pretrained-models)
  * [deploy model](#deploy-model)

## prepare dataset

1. Get the dataset at, [news](https://github.com/huseinzol05/Malay-Dataset/tree/master/news).

2. Run [keyword-extraction-headline.ipynb](keyword-extraction-headline.ipynb).

3. Run [prepare-finetune-headline.ipynb](prepare-finetune-headline.ipynb).

## training pretrained models

1. Train pretrained model, example for T5 Base, [finetune_base.py](finetune_base.py).

## deploy model

1. Freeze finetuned model, [save-model-generator-base.ipynb](save-model-generator-base.ipynb).