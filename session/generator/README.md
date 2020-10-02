# how-to

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## Table of contents
  * [prepare dataset](#prepare-dataset)
  * [training pretrained models](#training-pretrained-models)
  * [deploy model](#deploy-model)

## prepare dataset

1. Get the dataset at, [summarization/semisupervised](https://github.com/huseinzol05/Malay-Dataset/tree/master/summarization/semisupervised).

2. Run [prepare-finetune-generator.ipynb](prepare-finetune-generator.ipynb).

## training pretrained models

1. Train pretrained model, example for T5 Base, [finetune_base.py](finetune_base.py).

## deploy model

1. Freeze finetuned model, [save-model-generator-base.ipynb](save-model-generator-base.ipynb).