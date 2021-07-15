.. raw:: html

    <p align="center">
        <a href="#readme">
            <img alt="logo" width="40%" src="https://f000.backblazeb2.com/file/huseinhouse-public/malaya.png">
        </a>
    </p>
    <p align="center">
        <a href="https://pypi.python.org/pypi/malaya"><img alt="Pypi version" src="https://badge.fury.io/py/malaya.svg"></a>
        <a href="https://pypi.python.org/pypi/malaya"><img alt="Python3 version" src="https://img.shields.io/pypi/pyversions/malaya.svg"></a>
        <a href="https://github.com/huseinzol05/Malaya/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/huseinzol05/malaya.svg?color=blue"></a>
        <a href="https://malaya.readthedocs.io/"><img alt="Documentation" src="https://readthedocs.org/projects/malaya/badge/?version=latest"></a>
        <a href="https://pepy.tech/project/malaya"><img alt="total stats" src="https://static.pepy.tech/badge/malaya"></a>
        <a href="https://pepy.tech/project/malaya"><img alt="download stats / month" src="https://static.pepy.tech/badge/malaya/month"></a>
        <a href="https://discord.gg/C3CsXAvzz5"><img alt="discord" src="https://img.shields.io/badge/discord%20server-malaya-rgb(118,138,212).svg"></a>
        
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

    $ pip install malaya[gpu]

Only **Python 3.6.0 and above** and **Tensorflow 1.15.0 and above** are supported.

We recommend to use **virtualenv** for development. All examples tested on Tensorflow version 1.15.4, 2.4.1 and 2.5.

Features
--------

-  **Augmentation**, augment any text using dictionary of synonym, Wordvector or Transformer-Bahasa.
-  **Constituency Parsing**, breaking a text into sub-phrases using finetuned Transformer-Bahasa.  
-  **Coreference Resolution**, finding all expressions that refer to the same entity in a text using Dependency Parsing models.
-  **Dependency Parsing**, extracting a dependency parse of a sentence using finetuned Transformer-Bahasa.
-  **Emotion Analysis**, detect and recognize 6 different emotions of texts using finetuned Transformer-Bahasa.
-  **Entities Recognition**, seeks to locate and classify named entities mentioned in text using finetuned Transformer-Bahasa.
-  **Generator**, generate any texts given a context using T5-Bahasa, GPT2-Bahasa or Transformer-Bahasa.
-  **Keyword Extraction**, provide RAKE, TextRank and Attention Mechanism hybrid with Transformer-Bahasa.
-  **Knowledge Graph**, generate Knowledge Graph using Transformer-Bahasa or parse from Dependency Parsing models.
-  **Language Detection**, using Fast-text and Sparse Deep learning Model to classify Malay (formal and social media), Indonesia (formal and social media), Rojak language and Manglish.
-  **Normalizer**, using local Malaysia NLP researches hybrid with Transformer-Bahasa to normalize any bahasa texts.
-  **Num2Word**, convert from numbers to cardinal or ordinal representation.
-  **Paraphrase**, provide Abstractive Paraphrase using T5-Bahasa and Transformer-Bahasa.
-  **Part-of-Speech Recognition**, grammatical tagging is the process of marking up a word in a text using finetuned Transformer-Bahasa.
-  **Question Answer**, reading comprehension using finetuned Transformer-Bahasa.
-  **Relevancy Analysis**, detect and recognize relevancy of texts using finetuned Transformer-Bahasa.
-  **Sentiment Analysis**, detect and recognize polarity of texts using finetuned Transformer-Bahasa.
-  **Text Similarity**, provide interface for lexical similarity deep semantic similarity using finetuned Transformer-Bahasa.
-  **Spell Correction**, using local Malaysia NLP researches hybrid with Transformer-Bahasa to auto-correct any bahasa words.
-  **Stemmer**, using BPE LSTM Seq2Seq with attention state-of-art to do Bahasa stemming.
-  **Subjectivity Analysis**, detect and recognize self-opinion polarity of texts using finetuned Transformer-Bahasa.
-  **Kesalahan Tatabahasa**, Fix kesalahan tatabahasa using TransformerTag-Bahasa.
-  **Summarization**, provide Abstractive T5-Bahasa also Extractive interface using Transformer-Bahasa, skip-thought and Doc2Vec.
-  **Topic Modelling**, provide Transformer-Bahasa, LDA2Vec, LDA, NMF and LSA interface for easy topic modelling with topics visualization.
-  **Toxicity Analysis**, detect and recognize 27 different toxicity patterns of texts using finetuned Transformer-Bahasa.
-  **Transformer**, provide easy interface to load Pretrained Language models Malaya.
-  **Translation**, provide Neural Machine Translation using Transformer for EN to MS and MS to EN.
-  **Word2Num**, convert from cardinal or ordinal representation to numbers.
-  **Word2Vec**, provide pretrained bahasa wikipedia and bahasa news Word2Vec, with easy interface and visualization.
-  **Zero-shot classification**, provide Zero-shot classification interface using Transformer-Bahasa to recognize texts without any labeled training data.
-  **Hybrid 8-bit Quantization**, provide hybrid 8-bit quantization for all models to reduce inference time up to 2x and model size up to 4x.
-  **Longer Sequences Transformer**, provide BigBird + Pegasus for longer Abstractive Summarization, Neural Machine Translation and Relevancy Analysis sequences.
-  **Distilled Transformer**, provide distilled transformer models for Abstractive Summarization.

Pretrained Models
------------------

Malaya also released Bahasa pretrained models, simply check at `Malaya/pretrained-model <https://github.com/huseinzol05/Malaya/tree/master/pretrained-model>`_

- **ALBERT**, a Lite BERT for Self-supervised Learning of Language Representations, https://arxiv.org/abs/1909.11942
- **ALXLNET**, a Lite XLNET, no paper produced.
- **BERT**, Pre-training of Deep Bidirectional Transformers for Language Understanding, https://arxiv.org/abs/1810.04805
- **BigBird**, Transformers for Longer Sequences, https://arxiv.org/abs/2007.14062
- **ELECTRA**, Pre-training Text Encoders as Discriminators Rather Than Generators, https://arxiv.org/abs/2003.10555
- **GPT2**, Language Models are Unsupervised Multitask Learners, https://github.com/openai/gpt-2
- **LM-Transformer**, Exactly like T5, but use Tensor2Tensor instead Mesh Tensorflow with little tweak, no paper produced.
- **PEGASUS**, Pre-training with Extracted Gap-sentences for Abstractive Summarization, https://arxiv.org/abs/1912.08777
- **T5**, Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, https://arxiv.org/abs/1910.10683
- **TinyBERT**, Distilling BERT for Natural Language Understanding, https://arxiv.org/abs/1909.10351
- **Word2Vec**, Efficient Estimation of Word Representations in Vector Space, https://arxiv.org/abs/1301.3781
- **XLNET**, Generalized Autoregressive Pretraining for Language Understanding, https://arxiv.org/abs/1906.08237
- **FNet**, FNet: Mixing Tokens with Fourier Transforms, https://arxiv.org/abs/2105.03824

Or can try use huggingface 🤗 Transformers library, https://huggingface.co/models?filter=ms

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

Thanks to `KeyReply <https://www.keyreply.com/>`_ for sponsoring private cloud to train Malaya models, without it, this library will collapse entirely.  

.. raw:: html

    <a href="#readme">
        <img alt="logo" width="20%" src="https://cdn.techinasia.com/data/images/16234a59ae3f218dc03815a08eaab483.png">
    </a>

Also, thanks to `Tensorflow Research Cloud <https://www.tensorflow.org/tfrc>`_ for free TPUs access.

.. raw:: html

    <a href="https://www.tensorflow.org/tfrc">
        <img alt="logo" width="20%" src="https://2.bp.blogspot.com/-xojf3dn8Ngc/WRubNXxUZJI/AAAAAAAAB1A/0W7o1hR_n20QcWyXHXDI1OTo7vXBR8f7QCLcB/s400/image2.png">
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
