.. raw:: html

    <p align="center">
        <a href="#readme">
            <img alt="logo" width="40%" src="https://i.imgur.com/yi6jwST.png">
        </a>
    </p>
    <p align="center">
        <a href="https://pypi.python.org/pypi/malaya"><img alt="Pypi version" src="https://badge.fury.io/py/malaya.svg"></a>
        <a href="https://pypi.python.org/pypi/malaya"><img alt="Python3 version" src="https://img.shields.io/pypi/pyversions/malaya.svg"></a>
        <a href="https://github.com/huseinzol05/Malaya/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/huseinzol05/malaya.svg?color=blue"></a>
        <a href="https://malaya.readthedocs.io/"><img alt="Documentation" src="https://readthedocs.org/projects/malaya/badge/?version=latest"></a>
        <a href="https://pepy.tech/project/malaya"><img alt="total stats" src="https://static.pepy.tech/badge/malaya"></a>
        <a href="https://pepy.tech/project/malaya"><img alt="download stats / month" src="https://static.pepy.tech/badge/malaya/month"></a>
        <a href="https://discord.gg/aNzbnRqt3A"><img alt="discord" src="https://img.shields.io/badge/discord%20server-malaya-rgb(118,138,212).svg"></a>
    </p>

=========

**Malaya** is a Natural-Language-Toolkit library for bahasa Malaysia, powered by Tensorflow and PyTorch.

Documentation
--------------

Proper documentation is available at https://malaya.readthedocs.io/

Installing from the PyPI
----------------------------------

::

    $ pip install malaya

It will automatically install all dependencies except for Tensorflow and PyTorch. So you can choose your own Tensorflow CPU / GPU version and PyTorch CPU / GPU version.

Only **Python >= 3.6.0**, **Tensorflow >= 1.15.0**, and **PyTorch >= 1.10** are supported.

If you are a Windows user, make sure read https://malaya.readthedocs.io/en/latest/running-on-windows.html

Development Release
---------------------------------

Install from `master` branch,

::

    $ pip install git+https://github.com/huseinzol05/malaya.git


We recommend to use **virtualenv** for development. 

Documentation at https://malaya.readthedocs.io/en/latest/

Features
--------

-  **Alignment**, translation word alignment using Eflomal.
-  **Abstractive text augmentation**, augment any text into social media text structure using T5-Bahasa.
-  **Encoder text augmentation**, augment any text Wordvector or Transformer-Bahasa word replacement technique.
-  **Rules based text augmentation**, augment any text using dictionary of synonym and rules based.
-  **Isi Penting Generator**, generate text from list of isi penting using T5-Bahasa.
-  **Prefix Generator**, generate text from prefix using GPT2-Bahasa.
-  **Abstractive Keyword**, provide abstractive keyword using T5-Bahasa.
-  **Extractive Keyword**, provide RAKE, TextRank and Attention Mechanism hybrid with Transformer-Bahasa.
-  **Abstractive Normalizer**, normalize any malay texts using T5-Bahasa.
-  **Rules based Normalizer**, using local Malaysia NLP researches hybrid with Transformer-Bahasa to normalize any malay texts.
-  **Extractive QA**, reading comprehension using T5-Bahasa and Flan-T5.
-  **Doc2Vec Similarity**, provide Word2Vec and Encoder interface for text similarity.
-  **Semantic Similarity**, provide semantic similarity using T5-Bahasa.
-  **Spelling Correction**, using local Malaysia NLP researches hybrid with Transformer-Bahasa to auto-correct any malay words and NeuSpell using T5-Bahasa.
-  **Abstractive Summarization**, provide abstractive summarization using T5-Bahasa.
-  **Extractive Summarization**, Extractive interface using Transformer-Bahasa and Doc2Vec.
-  **Text to Knowledge Graph**, Generate knowledge graph from human sentences.
-  **Topic Modeling**, provide Transformer-Bahasa, LDA2Vec, LDA, NMF, LSA interface and easy BERTopic integration.
-  **End-to-End Translation**, provide multilanguages translation including local languages to English or Malay using T5-Bahasa.
-  **Zero-shot classification**, provide Zero-shot classification interface using Transformer-Bahasa to recognize texts without any labeled training data.
-  **Zero-shot Entity Recognition**, provide Zero-shot entity tagging interface using Transformer-Bahasa to extract entities.
-  **Constituency Parsing**, breaking a text into sub-phrases using finetuned Transformer-Bahasa.  
-  **Coreference Resolution**, finding all expressions that refer to the same entity in a text using Dependency Parsing models.
-  **Dependency Parsing**, extracting a dependency parse of a sentence using finetuned Transformer-Bahasa and T5-Bahasa.
-  **Emotion Analysis**, detect and recognize 6 different emotions of texts using finetuned Transformer-Bahasa.
-  **Entity Recognition**, seeks to locate and classify named entities mentioned in text using finetuned Transformer-Bahasa.
-  **Jawi-to-Rumi**, convert from Jawi to Rumi using Transformer.
-  **Knowledge Graph to Text**, Generate human sentences from a knowledge graph.
-  **Language Detection**, using Fast-text and Sparse Deep learning Model to classify Malay (formal and social media), Indonesia (formal and social media), Rojak language and Manglish.
-  **Language Model**, using KenLM, Masked language model using BERT, ALBERT and RoBERTa, and GPT2 to do text scoring.
-  **NSFW Detection**, detect NSFW text using rules based and subwords Naive Bayes.
-  **Num2Word**, convert from numbers to cardinal or ordinal representation.
-  **Paraphrase**, provide Abstractive Paraphrase using T5-Bahasa and Transformer-Bahasa.
-  **Grapheme-to-Phoneme**, convert from Grapheme to Phoneme DBP or IPA using LSTM Seq2Seq with attention state-of-art.
-  **Part-of-Speech Recognition**, grammatical tagging is the process of marking up a word in a text using finetuned Transformer-Bahasa.
-  **Relevancy Analysis**, detect and recognize relevancy of texts using finetuned Transformer-Bahasa.
-  **Rumi-to-Jawi**, convert from Rumi to Jawi using Transformer.
-  **Text Segmentation**, dividing written text into meaningful words using T5-Bahasa.
-  **Sentiment Analysis**, detect and recognize polarity of texts using finetuned Transformer-Bahasa.
-  **Text Similarity**, provide interface for lexical similarity deep semantic similarity using finetuned Transformer-Bahasa.
-  **Stemmer**, using BPE LSTM Seq2Seq with attention state-of-art to do Bahasa stemming including local language structure.
-  **Subjectivity Analysis**, detect and recognize self-opinion polarity of texts using finetuned Transformer-Bahasa.
-  **Kesalahan Tatabahasa**, Fix kesalahan tatabahasa using TransformerTag-Bahasa.
-  **Tokenizer**, provide word, sentence and syllable tokenizers.
-  **Toxicity Analysis**, detect and recognize 27 different toxicity patterns of texts using finetuned Transformer-Bahasa.
-  **Transformer**, provide easy interface to load Pretrained Language Malaya models.
-  **True Case**, provide true casing utility using T5-Bahasa.
-  **Word2Num**, convert from cardinal or ordinal representation to numbers.
-  **Word2Vec**, provide pretrained malay wikipedia and malay news Word2Vec, with easy interface and visualization.

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
- **Fastformer**, Fastformer: Additive Attention Can Be All You Need, https://arxiv.org/abs/2108.09084
- **MLM Scoring**, Masked Language Model Scoring, https://arxiv.org/abs/1910.14659

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

Thanks to `KeyReply <https://www.keyreply.com/>`_ for private V100s cloud and `Mesolitica <https://mesolitica.com/>`_ for private RTXs cloud to train Malaya models,

.. raw:: html

    <a href="#readme">
        <img alt="logo" width="20%" src="https://image4.owler.com/logo/keyreply_owler_20191024_163259_original.png">
    </a>

.. raw:: html

    <a href="#readme">
        <img alt="logo" width="20%" src="https://i1.wp.com/mesolitica.com/wp-content/uploads/2019/06/Mesolitica_Logo_Only.png?fit=857%2C532&ssl=1">
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
