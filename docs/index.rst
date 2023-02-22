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
   knowledge-graph-toolkit
   installation
   mock-tensorflow
   Dataset
   running-on-windows
   Contributing

.. toctree::
   :maxdepth: 2
   :caption: GPU Environment

   gpu-environment-tensorflow
   gpu-environment-pytorch
   different-precision-pytorch

.. toctree::
   :maxdepth: 2
   :caption: Pre-trained model

   load-transformer
   load-transformer-huggingface
   load-wordvector

.. toctree::
   :maxdepth: 2
   :caption: Alignment Module

   alignment-en-ms-eflomal
   alignment-en-ms-huggingface
   alignment-ms-en-eflomal
   alignment-ms-en-huggingface

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

   load-spelling-correction-jamspell
   load-spelling-correction-probability
   load-spelling-correction-probability-lm
   load-compare-lm-spelling-correction
   load-spelling-correction-spylls
   load-spelling-correction-symspell
   load-spelling-correction-encoder-transformer
   load-spelling-correction-transformer

.. toctree::
   :maxdepth: 2
   :caption: Normalization Module
   
   load-preprocessing
   load-demoji
   load-stemmer
   load-true-case
   load-true-case-huggingface
   load-segmentation
   load-segmentation-huggingface
   load-num2word
   load-word2num
   load-coreference-resolution
   load-normalizer-abstractive
   load-normalizer

.. toctree::
   :maxdepth: 2
   :caption: Convert Module

   load-phoneme
   load-phoneme-ipa
   load-rumi-jawi
   load-jawi-rumi

.. toctree::
   :maxdepth: 2
   :caption: Kesalahan Tatabahasa Module

   load-tatabahasa-tagging
   load-tatabahasa-tagging-huggingface

.. toctree::
   :maxdepth: 2
   :caption: Generative Module

   load-prefix-generator
   load-isi-penting-generator
   load-isi-penting-generator-huggingface-article-style
   load-isi-penting-generator-huggingface-headline-news-style
   load-isi-penting-generator-huggingface-karangan-style
   load-isi-penting-generator-huggingface-news-style
   load-isi-penting-generator-huggingface-product-description-style
   load-paraphrase
   load-paraphrase-huggingface

.. toctree::
   :maxdepth: 2
   :caption: Classification Module

   load-emotion
   load-language-detection
   language-detection-words
   load-nsfw
   load-relevancy
   load-sentiment
   load-subjectivity
   load-toxic

.. toctree::
   :maxdepth: 2
   :caption: Similarity Module

   load-doc2vec-similarity
   load-semantic-similarity
   load-semantic-similarity-huggingface

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
   load-dependency-huggingface
   load-constituency

.. toctree::
   :maxdepth: 2
   :caption: Summarization Module

   load-extractive
   load-abstractive
   load-abstractive-huggingface

.. toctree::
   :maxdepth: 2
   :caption: Translation Module

   load-translation-ms-en
   load-translation-ms-en-huggingface
   load-translation-noisy-ms-en-huggingface
   load-translation-en-ms
   load-translation-en-ms-huggingface
   load-translation-noisy-en-ms-huggingface
   load-translation-ind-ms-huggingface
   load-translation-jav-ms-huggingface
   load-translation-ms-ind-huggingface
   load-translation-ms-jav-huggingface

.. toctree::
   :maxdepth: 2
   :caption: Question Answer Module

   load-qa-extractive
   load-qa-extractive-huggingface

.. toctree::
   :maxdepth: 2
   :caption: Zeroshot Module

   load-zeroshot-classification
   load-zeroshot-classification-huggingface
   zeroshot-ner

.. toctree::
   :maxdepth: 2
   :caption: Topic Modeling Module

   load-topic-model-decomposition
   load-topic-model-lda2vec
   load-topic-model-transformer
   load-topic-model-bertopic

.. toctree::
   :maxdepth: 2
   :caption: Keyword Module

   load-abstractive-keyword-huggingface
   load-keyword-extractive

.. toctree::
   :maxdepth: 2
   :caption: Misc Module
   
   load-lexicon
   load-clustering
   load-stack

.. toctree::
   :maxdepth: 2
   :caption: Bias

   sentiment-bias-towards-countries
   sentiment-bias-towards-politicians

.. toctree::
   :maxdepth: 2
   :caption: Misc
   
   Api
   Donation
