# SMITH-Bahasa

Thanks to Google for opensourcing most of the source code to develop SMITH, https://github.com/google-research/google-research/tree/master/smith

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator, Tensorflow Dataset really helpful**.

## Table of contents
  * [Objective](#objective)

## Objective

1. Provide **SMALL** and **BASE** SMITH.

## how-to

1. Generate wordpiece, [build-wordpiece.ipynb](build-wordpiece.ipynb), or simply download [BERT.wordpiece](../preprocess/BERT.wordpiece).

2. Split text file to multiple text files,

```bash
mkdir splitted
cd splitted
split -l 300000 -d --additional-suffix=.txt ../filtered-dumping-wiki.txt splitted-wiki
split -l 300000 -d --additional-suffix=.txt ../dumping-cleaned-news.txt splitted-news
split -l 300000 -d --additional-suffix=.txt ../filtered-dumping-academia.txt splitted-academia
split -l 300000 -d --additional-suffix=.txt ../filtered-dumping-cleaned-common-crawl.txt splitted-common-crawl
```