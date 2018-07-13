<p align="center">
    <img src="entities-pos/towns-of-malaya.jpg" alt="malaya logo" />
</p>

# Malaya
NLTK like for bahasa Malaysia, powered by Deep Learning

## Requirements
  * Python >= 3.0
  * Tensorflow >= 1.5
  * NumPy >= 1.0
  * Sklearn >= 0.18
  * Fuzzywuzzy
  * unidecode
  * nltk
  * scipy

## Installation
1. Install dependencies
```bash
# using gpu if you installed CUDA and libcudnn
pip3 install sklearn fuzzywuzzy nltk unidecode numpy scipy python-levenshtein tensorflow-gpu==1.5

# using cpu
pip3 install sklearn fuzzywuzzy nltk unidecode numpy scipy python-levenshtein tensorflow==1.5
```

```python
import nltk
nltk.download('punkt')
```
2. Clone this repository
```bash
git clone https://github.com/DevconX/Malaya
```
3. install using setup.py
```bash
python3 setup.py install
```

## Features
  * Deep learning Named entity recognition
  * Deep learning Part-of-speech tagging
  * Naive Part-of-speech tagging
  * Text Normalizer
  * Naive Stemming
  * Naive Bayes
  * Topic Modelling
  * Pretrained word2vec from local news
  * Google news crawler
  * Pretrained deep learning sentiment analysis with attention

You can read [README](https://github.com/DevconX/Malaya/tree/master/training) for supported Entities and POS, also for comparison accuracies among models.

You can read [EXAMPLE](EXAMPLE.md) for some examples.

## How-to

1. import
```python3
import malaya
```
2. crawler
```bash
cd crawl/crawler
```
```python3
# change into any topic / issue in main.py
issue ='isu sekolah'
```

## To do

- [x] tokenizer
- [x] stemming
- [x] tatabahasa gathering
- [x] Multinomial naive bayes
- [x] Part-Of-Speech using Regex
- [x] Part-Of-Speech using Deep Learning
- [x] number to words (currency, numbers, position, year)
- [x] stopwords
- [x] normalizer
- [x] top-k hot topic (LSA, NMF, LDA)
- [x] semantic and similar words searching
- [ ] deep learning topic modelling
- [ ] deep learning stemming
- [x] pretrained deep learning sentiment analysis
- [ ] bahasa malaysia checker

## Warning

Install this library with fully aware that this library is still in alpha stage.

Documentation will be released soon.

## Disclaimer

Most of the data gathered using crawlers crawled through targeted malaysia websites. I am not aware of any data protection.

## Authors

* **Husein Zolkepli** - *Initial work* - [huseinzol05](https://github.com/huseinzol05)
