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

.. toctree::
   :maxdepth: 2
   :caption: GPU Environment

   gpu-environment-tensorflow
   gpu-environment-pytorch

.. toctree::
   :maxdepth: 2
   :caption: Pre-trained model

   load-transformer
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
   load-segmentation
   load-num2word
   load-word2num
   load-coreference-resolution
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

   load-tatabahasa
   load-tatabahasa-tagging

.. toctree::
   :maxdepth: 2
   :caption: Generative Module

   load-augmentation
   load-prefix-generator
   load-isi-penting-generator
   load-lexicon
   load-paraphrase

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
   load-zeroshot-classification

.. toctree::
   :maxdepth: 2
   :caption: Similarity Module

   load-doc2vec-similarity
   load-semantic-similarity
   load-unsupervised-keyword-extraction
   load-keyphrase-similarity

.. toctree::
   :maxdepth: 2
   :caption: Tagging Module

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

   load-abstractive
   long-text-abstractive-summarization
   load-extractive

.. toctree::
   :maxdepth: 2
   :caption: Translation Module

   load-translation-ms-en
   ms-en-long-translation
   load-translation-ms-en-huggingface
   load-translation-noisy-ms-en
   load-translation-en-ms
   load-translation-en-ms-huggingface
   en-ms-long-translation
   load-translation-noisy-en-ms

.. toctree::
   :maxdepth: 2
   :caption: Question Answer Module

   load-qa-squad

.. toctree::
   :maxdepth: 2
   :caption: Misc Module
   
   load-topic-modeling
   load-clustering
   load-stack

.. toctree::
   :maxdepth: 2
   :caption: Bias

   sentiment-bias-towards-countries
   sentiment-bias-towards-politicians

.. toctree::
   :maxdepth: 2
   :caption: Finetune Pretrained Model

   tf-estimator-alxlnet
   tf-estimator-bert
   tf-estimator-xlnet

.. toctree::
   :maxdepth: 2
   :caption: Misc
   
   Api
   Donation
