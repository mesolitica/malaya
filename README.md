<p align="center">
    <img src="entities-pos/towns-of-malaya.jpg" alt="malaya logo" />
</p>

# Malaya ![alt text](https://travis-ci.org/DevconX/Malaya.svg?branch=master) [![Coverage Status-90](https://coveralls.io/repos/github/DevconX/Malaya/badge.svg?branch=master)](https://coveralls.io/github/DevconX/Malaya?branch=master)
Natural-Language-Toolkit for bahasa Malaysia, powered by Deep Learning.

## Requirements
  * Python < 3.7

## Installation
1. Install dependencies

Using CPU
```bash
pip3 install scikit-learn==0.19.1 requests fuzzywuzzy tqdm nltk unidecode numpy scipy python-levenshtein tensorflow==1.5 pandas
python3 -m nltk.downloader punkt
```

Using GPU
```bash
pip3 install scikit-learn==0.19.1 requests fuzzywuzzy tqdm nltk unidecode numpy scipy python-levenshtein tensorflow-gpu==1.5 pandas
python3 -m nltk.downloader punkt
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
  * Deep learning sentiment analysis with attention
  * Pretrained word2vec from local news
  * Multinomial sentiment analysis
  * Multinomial language detection
  * Naive Part-of-speech tagging
  * Text Normalizer
  * Naive Stemming
  * Topic Modelling (LSA, NMF, LDA)
  * Summarization (LSA, NMF, LDA)
  * Google news crawler

You can check [EXAMPLE DIRECTORY](example) for some APIs and examples.

## How-to

1. import
```python3
import malaya
```

2. crawler
```bash
cd crawl
# change into any topic / issue in main.py
# issue ='isu sekolah'
python3 main.py
```

## To do

- [x] tokenizer
- [x] naive stemmer
- [x] tatabahasa gathering
- [x] Multinomial naive bayes
- [x] naive Part-Of-Speech
- [x] Part-Of-Speech using Deep Learning
- [x] number to words (currency, numbers, position, year)
- [x] stopwords combination of indonesian and malaysian
- [x] naive normalizer
- [x] top-k hot topic (LSA, NMF, LDA)
- [x] summarization agent (LSA, NMF, LDA)
- [x] semantic and similar words searching (Word2Vec)
- [x] pretrained deep learning sentiment analysis
- [x] pretrained naive bayes sentiment analysis
- [x] bahasa malaysia checker
- [ ] deep learning topic modelling
- [ ] deep learning stemming

## Disclaimer

Most of the data gathered using crawlers crawled through targeted malaysia websites. I am not aware of any data protection.

Documentation will be released soon.

## References

1. Banko, M., Cafarella, M.J., Soderland, S., Broadhead, M. and Etzioni, O., 2007, January. Open Information Extraction from the Web. In IJCAI (Vol. 7, pp. 2670-2676).
2. Angeli, G., Premkumar, M.J. and Manning, C.D., 2015, July. Leveraging linguistic structure for open domain information extraction. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics (ACL 2015).
3. Suhartono, D., 2014. Lemmatization Technique in Bahasa: Indonesian. Journal of Software, 9(5), p.1203.

## Authors

* **Husein Zolkepli** - *Initial work* - [huseinzol05](https://github.com/huseinzol05)
