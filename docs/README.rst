.. raw:: html

    <p align="center">
        <a href="#readme">
            <img alt="logo" width="50%" src="https://malaya-dataset.s3-ap-southeast-1.amazonaws.com/malaya-icon.png">
        </a>
    </p>
    <p align="center">
        <a href="https://pypi.python.org/pypi/malaya"><img alt="Pypi version" src="https://badge.fury.io/py/malaya.svg"></a>
        <a href="https://pypi.python.org/pypi/malaya"><img alt="Python3 version" src="https://img.shields.io/pypi/pyversions/malaya.svg"></a>
        <a href="https://github.com/huseinzol05/Malaya/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/huseinzol05/malaya.svg?color=blue"></a>
        <a href="https://malaya.readthedocs.io/"><img alt="Documentation" src="https://readthedocs.org/projects/malaya/badge/?version=latest"></a>
        <a href="https://pepy.tech/project/malaya"><img alt="total stats" src="https://static.pepy.tech/badge/malaya"></a>
        <a href="https://pepy.tech/project/malaya"><img alt="download stats / month" src="https://static.pepy.tech/badge/malaya/month"></a>
        <a href="https://pepy.tech/project/malaya-gpu"><img alt="total stats" src="https://static.pepy.tech/badge/malaya-gpu"></a>
        <a href="https://pepy.tech/project/malaya-gpu"><img alt="download stats / month" src="https://static.pepy.tech/badge/malaya-gpu/month"></a>
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

Only **Python 3.6.x and above** and **Tensorflow 1.10 and above but not 2.0** are supported.

Features
--------

-  **Augmentation**

   Augment any text using dictionary of synonym, Wordvector or Transformer-Bahasa.
-  **Dependency Parsing**

   Transfer learning on BERT-base-bahasa, Tiny-BERT-bahasa, Albert-base-bahasa, Albert-tiny-bahasa, XLNET-base-bahasa, ALXLNET-base-bahasa.
-  **Emotion Analysis**

   Transfer learning on BERT-base-bahasa, Tiny-BERT-bahasa, Albert-base-bahasa, Albert-tiny-bahasa, XLNET-base-bahasa, ALXLNET-base-bahasa.
-  **Entities Recognition**

   Transfer learning on BERT-base-bahasa, Tiny-BERT-bahasa, Albert-base-bahasa, Albert-tiny-bahasa, XLNET-base-bahasa, ALXLNET-base-bahasa.
-  **Generator**

   Generate any texts given a context using T5-Bahasa, GPT2-Bahasa or Transformer-Bahasa.
-  **Keyword Extraction**

   Provide RAKE, TextRank and Attention Mechanism hybrid with Transformer-Bahasa.
-  **Language Detection**

   using Fast-text and Sparse Deep learning Model to classify Malay (formal and social media), Indonesia (formal and social media), Rojak language and Manglish.
-  **Normalizer**

   using local Malaysia NLP researches hybrid with Transformer-Bahasa to normalize any bahasa texts.
-  **Num2Word**

   Convert from numbers to cardinal or ordinal representation.
-  **Paraphrase**

   Provide Abstractive Paraphrase using T5-Bahasa and Transformer-Bahasa.
-  **Part-of-Speech Recognition**

   Transfer learning on BERT-base-bahasa, Tiny-BERT-bahasa, Albert-base-bahasa, Albert-tiny-bahasa, XLNET-base-bahasa, ALXLNET-base-bahasa.
-  **Relevancy Analysis**

   Transfer learning on BERT-base-bahasa, Tiny-BERT-bahasa, Albert-base-bahasa, Albert-tiny-bahasa, XLNET-base-bahasa, ALXLNET-base-bahasa.
-  **Sentiment Analysis**

   Transfer learning on BERT-base-bahasa, Tiny-BERT-bahasa, Albert-base-bahasa, Albert-tiny-bahasa, XLNET-base-bahasa, ALXLNET-base-bahasa.
-  **Similarity**

   Using deep Encoder, Doc2Vec, BERT-base-bahasa, Tiny-BERT-bahasa, Albert-base-bahasa, Albert-tiny-bahasa, XLNET-base-bahasa and ALXLNET-base-bahasa to build deep semantic similarity models.
-  **Spell Correction**

   Using local Malaysia NLP researches hybrid with Transformer-Bahasa to auto-correct any bahasa words.
-  **Stemmer**

   Using BPE LSTM Seq2Seq with attention state-of-art to do Bahasa stemming.
-  **Subjectivity Analysis**

   Transfer learning on BERT-base-bahasa, Tiny-BERT-bahasa, Albert-base-bahasa, Albert-tiny-bahasa, XLNET-base-bahasa, ALXLNET-base-bahasa.
-  **Summarization**

   Provide Abstractive T5-Bahasa also Extractive interface using Transformer-Bahasa, skip-thought, LDA, LSA and Doc2Vec.
-  **Topic Modelling**

   Provide Transformer-Bahasa, LDA2Vec, LDA, NMF and LSA interface for easy topic modelling with topics visualization.
-  **Toxicity Analysis**

   Transfer learning on BERT-base-bahasa, Tiny-BERT-bahasa, Albert-base-bahasa, Albert-tiny-bahasa, XLNET-base-bahasa, ALXLNET-base-bahasa.
-  **Transformer**

   Provide easy interface to load BERT-base-bahasa, Tiny-BERT-bahasa, Albert-base-bahasa, Albert-tiny-bahasa, XLNET-base-bahasa, ALXLNET-base-bahasa, ELECTRA-base-bahasa and ELECTRA-small-bahasa.
-  **Word2Num**

   Convert from cardinal or ordinal representation to numbers.
-  **Word2Vec**

   Provide pretrained bahasa wikipedia and bahasa news Word2Vec, with easy interface and visualization.
-  **Zero-shot classification**

   Provide Zero-shot classification interface using Transformer-Bahasa to recognize texts without any labeled training data.

Pretrained Models
------------------

Malaya also released Bahasa pretrained models, simply check at `Malaya/pretrained-model <https://github.com/huseinzol05/Malaya/tree/master/pretrained-model>`_

Or can try use huggingface 🤗 Transformers library, https://huggingface.co/models?filter=malay

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

Acknowledgement
----------------

Thanks to `Im Big <https://www.facebook.com/imbigofficial/>`_, `LigBlou <https://www.facebook.com/ligblou>`_, `Mesolitica <https://mesolitica.com/>`_ and `KeyReply <https://www.keyreply.com/>`_ for sponsoring AWS, GCP and private cloud to train Malaya models.

.. raw:: html

    <a href="#readme">
        <img alt="logo" width="50%" src="https://malaya-dataset.s3-ap-southeast-1.amazonaws.com/ligblou-mesolitca-keyreply.png">
    </a>

Contributing
----------------

Thank you for contributing this library, really helps a lot. Feel free to contact me to suggest me anything or want to contribute other kind of forms, we accept everything, not just code!

.. raw:: html

    <a href="#readme">
        <img alt="logo" width="30%" src="https://contributors-img.firebaseapp.com/image?repo=huseinzol05/malaya">
    </a>

License
--------

.. |License| image:: https://app.fossa.io/api/projects/git%2Bgithub.com%2Fhuseinzol05%2FMalaya.svg?type=large
   :target: https://app.fossa.io/projects/git%2Bgithub.com%2Fhuseinzol05%2FMalaya?ref=badge_large

|License|
