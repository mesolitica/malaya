.. figure:: https://raw.githubusercontent.com/DevconX/Malaya/master/session/towns-of-malaya.jpg

Malaya
======

Natural-Language-Toolkit for bahasa Malaysia, powered by Deep Learning
Tensorflow.

.. figure:: https://img.shields.io/badge/build-passing-green.svg
   :alt: alt text

Features
--------

-  **Entities Recognition**, using latest state-of-art CRF deep learning
   model.
-  **Language Detection**, using Character-wise eXtreme Gradient
   Boosting to distinguish Malay, English, and Indonesian.
-  **Normalizer**, using local Malaysia NLP researches to normalize any
   bahasa texts.
-  Num2Word
-  **Part-of-Speech Recognition**, using latest state-of-art CRF deep
   learning model.
-  **Sentiment Analysis**, from BERT, Fast-Text, Dynamic-Memory Network,
   Attention to build deep sentiment analysis models.
-  **Spell Correction**, using local Malaysia NLP researches to
   auto-correct any bahasa words.
-  Stemmer
-  **Summarization**, using skip-thought state-of-art to give precise
   summarization.
-  Topic Modelling
-  **Topic and Influencers Analysis**, using deep and machine learning
   models to understand topics and Influencers similarity in sentences.
-  **Toxicity Analysis**, from Fast-Text, Stacking, Entity-Network to do
   multi-label classification.
-  Word2Vec

Installation
------------

The latest release of Malaya can be installed using ``pip``,

.. code:: bash

   pip install malaya

Also, if want to install GPU version, simply,

.. code:: bash

   pip install malaya-gpu

Documentation
-------------

All the documentations moved to `Malaya Wiki`_.

Contributors
------------

-  **Husein Zolkepli** - *Initial work* - `huseinzol05`_

-  **Sani** - *build PIP package* - `khursani8`_

.. _Malaya Wiki: https://github.com/DevconX/Malaya/wiki
.. _huseinzol05: https://github.com/huseinzol05
.. _khursani8: https://github.com/khursani8
