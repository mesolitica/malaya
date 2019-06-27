.. raw:: html

    <p align="center">
        <a href="#readme">
            <img alt="logo" width="70%" src="https://raw.githubusercontent.com/DevconX/Malaya/master/session/towns-of-malaya.jpg">
        </a>
    </p>
    <p align="center">
        <a href="https://pypi.python.org/pypi/malaya"><img alt="Pypi version" src="https://badge.fury.io/py/malaya.svg"></a>
        <a href="https://pypi.python.org/pypi/malaya"><img alt="Python3 version" src="https://img.shields.io/pypi/pyversions/malaya.svg"></a>
        <a href="https://github.com/huseinzol05/Malaya/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
        <a href="https://malaya.readthedocs.io/"><img alt="Documentation" src="https://readthedocs.org/projects/malaya/badge/?version=latest"></a>
        <a href="https://travis-ci.org/huseinzol05/Malaya"><img alt="Build status" src="https://travis-ci.org/huseinzol05/Malaya.svg?branch=master"></a>
    </p>

=========

**Malaya** is a Natural-Language-Toolkit library for bahasa Malaysia, powered by Deep Learning Tensorflow.

Documentation
--------------

Proper documentation is available at https://malaya.readthedocs.io/

Installing from the PyPI
----------------------------------

CPU version
::

    $ pip install malaya

GPU version
::

    $ pip install malaya-gpu

Only **Python 3.6.x and above** and **Tensorflow 1.X** are supported.

Features
--------

-  **Emotion Analysis**

   From fine-tuning BERT, Attention-Recurrent model, Sparse Tensorflow, Self-Attention to build deep emotion analysis models.
-  **Entities Recognition**

   Latest state-of-art CRF deep learning models to do Naming Entity Recognition.

-  **Language Detection**

   using Multinomial, SGD, XGB, Fast-text N-grams deep learning to distinguish Malay, English, and Indonesian.
-  **Normalizer**

   using local Malaysia NLP researches to normalize any
   bahasa texts.
-  **Num2Word**

   Convert from numbers to cardinal or ordinal representation.
-  **Part-of-Speech Recognition**

   Latest state-of-art CRF deep learning models to do Part-of-Speech Recognition.
-  **Dependency Parsing**

   Latest state-of-art CRF deep learning models to do analyzes the grammatical structure of a sentence, establishing relationships between words.
-  **ELMO (biLM)**

   Provide pretrained bahasa wikipedia and bahasa news ELMO, with easy interface and visualization.
-  **Relevancy Analysis**

   From Dilated Convolutional Neural Network and Self-Attention to build deep relevancy analysis models.
-  **Sentiment Analysis**

   From fine-tuning BERT, Attention-Recurrent model, Sparse Tensorflow and Self-Attention to build deep sentiment analysis models.
-  **Spell Correction**

   Using local Malaysia NLP researches to auto-correct any bahasa words.
-  **Stemmer**

   Use Character LSTM Seq2Seq with attention state-of-art to do Bahasa stemming.
-  **Subjectivity Analysis**

   From fine-tuning BERT, Attention-Recurrent model, Sparse Tensorflow and Self-Attention to build deep subjectivity analysis models.
-  **Similarity**

   Use deep LSTM siamese, deep Dilated CNN siamese, deep Self-Attention, siamese, Doc2Vec and BERT to build deep semantic similarity models.
-  **Summarization**

   Using skip-thought and residual-network with attention state-of-art, LDA, LSA and Doc2Vec to give precise unsupervised summarization, and TextRank as scoring algorithm.
-  **Topic Modelling**

   Provide LDA2Vec, LDA, NMF and LSA interface for easy topic modelling with topics visualization.
-  **Toxicity Analysis**

   From fine-tuning BERT, Attention-Recurrent model, Self-Attention to build deep toxicity analysis models.
-  **Word2Vec**

   Provide pretrained bahasa wikipedia and bahasa news Word2Vec, with easy interface and visualization.
-  **Fast-text**

   Provide pretrained bahasa wikipedia Fast-text, with easy interface and visualization.

References
-----------

If you use our software for research, please cite:

::

  @misc{Malaya, Natural-Language-Toolkit library for bahasa Malaysia, powered by Deep Learning Tensorflow,
    author = {Husein, Zolkepli},
    title = {Malaya},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/huseinzol05/malaya}}
  }


License
--------

.. |License| image:: https://app.fossa.io/api/projects/git%2Bgithub.com%2Fhuseinzol05%2FMalaya.svg?type=large
   :target: https://app.fossa.io/projects/git%2Bgithub.com%2Fhuseinzol05%2FMalaya?ref=badge_large

|License|
