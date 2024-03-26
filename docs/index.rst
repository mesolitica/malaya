.. malaya documentation master file, created by
   sphinx-quickstart on Sat Dec  8 23:44:35 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Malaya's documentation!
======================================

.. include::
   README.rst

Contents:
=========

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   speech-toolkit
   installation
   Dataset
   running-on-windows
   Contributing
   Api

.. toctree::
   :maxdepth: 2
   :caption: GPU Environment

   gpu-environment
   different-precision

.. toctree::
   :maxdepth: 2
   :caption: Augmentation Module

   load-augmentation-abstractive
   load-augmentation-encoder
   load-augmentation-rules

.. toctree::
   :maxdepth: 2
   :caption: Dictionary Module

   dictionary-malay

.. toctree::
   :maxdepth: 2
   :caption: Tokenization Module

   load-tokenizer-word
   load-tokenizer-sentence
   load-tokenizer-syllable

.. toctree::
   :maxdepth: 2
   :caption: Language Model Module

   load-kenlm
   load-mlm
   load-gpt2-lm

.. toctree::
   :maxdepth: 2
   :caption: Spelling Correction Module
   
   load-spelling-correction-probability
   load-spelling-correction-probability-lm
   load-spelling-correction-jamspell
   load-spelling-correction-spylls
   load-spelling-correction-symspell
   load-compare-lm-spelling-correction

.. toctree::
   :maxdepth: 2
   :caption: Normalization Module
   
   load-preprocessing
   load-demoji
   load-stemmer
   load-true-case
   load-segmentation
   load-num2word
   load-word2num
   load-normalizer

.. toctree::
   :maxdepth: 2
   :caption: Jawi Module

   load-jawi

.. toctree::
   :maxdepth: 2
   :caption: Kesalahan Tatabahasa Module

   load-tatabahasa-tagging

.. toctree::
   :maxdepth: 2
   :caption: Generative Module

   load-isi-penting-generator-article-style
   load-isi-penting-generator-headline-news-style
   load-isi-penting-generator-karangan-style
   load-isi-penting-generator-news-style
   load-isi-penting-generator-product-description-style
   load-paraphrase
   load-llm

.. toctree::
   :maxdepth: 2
   :caption: Classification Module

   load-emotion
   load-language-detection
   language-detection-words
   load-nsfw
   load-sentiment

.. toctree::
   :maxdepth: 2
   :caption: Retrieval Module

   load-wordvector
   load-embedding
   load-reranker

.. toctree::
   :maxdepth: 2
   :caption: Similarity Module

   load-similarity-doc2vec
   load-similarity-semantic

.. toctree::
   :maxdepth: 2
   :caption: Tagging Module

   load-general-malaya-entities
   load-entities
   load-pos

.. toctree::
   :maxdepth: 2
   :caption: Parsing Module

   load-dependency
   load-constituency

.. toctree::
   :maxdepth: 2
   :caption: Summarization Module

   load-summarization-extractive
   load-summarization-abstractive

.. toctree::
   :maxdepth: 2
   :caption: Translation Module

   load-translation
   load-noisy-translation

.. toctree::
   :maxdepth: 2
   :caption: Question Answer Module

   load-qa-extractive

.. toctree::
   :maxdepth: 2
   :caption: Zeroshot Module

   load-zeroshot-classification

.. toctree::
   :maxdepth: 2
   :caption: Topic Modeling Module

   load-topic-model-decomposition
   load-topic-model-transformer
   load-topic-model-bertopic

.. toctree::
   :maxdepth: 2
   :caption: Keyword Module

   load-keyword-abstractive
   load-keyword-extractive

.. toctree::
   :maxdepth: 2
   :caption: Knowledge Graph

   text-to-kg
   
.. toctree::
   :maxdepth: 2
   :caption: Converter

   t5-ctranslate2

.. toctree::
   :maxdepth: 2
   :caption: Misc Module
   
   load-stack
   sentiment-bias-towards-countries
   sentiment-bias-towards-politicians
