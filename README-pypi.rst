**Malaya** is a Natural-Language-Toolkit library for bahasa Malaysia, powered by Tensorflow and PyTorch.

Documentation
--------------

Proper documentation is available at https://malaya.readthedocs.io/

Installing from the PyPI
----------------------------------

::

    $ pip install malaya

It will automatically all dependencies except for Tensorflow and PyTorch. So you can choose your own Tensorflow CPU / GPU version and PyTorch CPU / GPU version.

Only **Python >= 3.6.0**, **Tensorflow >= 1.15.0**, and **PyTorch >= 1.10** are supported.

Development Release
---------------------------------

Install from `master` branch,
::

    $ pip install git+https://github.com/huseinzol05/malaya.git


We recommend to use **virtualenv** for development. 

Documentation at https://malaya.readthedocs.io/en/latest/

Features
--------

-  **Alignment**, translation word alignment using Eflomal and pretrained Transformer models.
-  **Augmentation**, augment any text using dictionary of synonym, Wordvector or Transformer-Bahasa.
-  **Constituency Parsing**, breaking a text into sub-phrases using finetuned Transformer-Bahasa.  
-  **Coreference Resolution**, finding all expressions that refer to the same entity in a text using Dependency Parsing models.
-  **Dependency Parsing**, extracting a dependency parse of a sentence using finetuned Transformer-Bahasa.
-  **Emotion Analysis**, detect and recognize 6 different emotions of texts using finetuned Transformer-Bahasa.
-  **Entities Recognition**, seeks to locate and classify named entities mentioned in text using finetuned Transformer-Bahasa.
-  **Generator**, generate any texts given a context using T5-Bahasa, GPT2-Bahasa or Transformer-Bahasa.
-  **Jawi-to-Rumi**, convert from Jawi to Rumi using Transformer.
-  **Keyword Extraction**, provide RAKE, TextRank and Attention Mechanism hybrid with Transformer-Bahasa.
-  **Knowledge Graph**, generate Knowledge Graph using T5-Bahasa or parse from Dependency Parsing models.
-  **Language Detection**, using Fast-text and Sparse Deep learning Model to classify Malay (formal and social media), Indonesia (formal and social media), Rojak language and Manglish.
-  **Normalizer**, using local Malaysia NLP researches hybrid with Transformer-Bahasa to normalize any bahasa texts.
-  **Num2Word**, convert from numbers to cardinal or ordinal representation.
-  **Paraphrase**, provide Abstractive Paraphrase using T5-Bahasa and Transformer-Bahasa.
-  **Grapheme-to-Phoneme**, convert from Grapheme to Phoneme DBP or IPA using LSTM Seq2Seq with attention state-of-art.
-  **Part-of-Speech Recognition**, grammatical tagging is the process of marking up a word in a text using finetuned Transformer-Bahasa.
-  **Question Answer**, reading comprehension using finetuned Transformer-Bahasa.
-  **Relevancy Analysis**, detect and recognize relevancy of texts using finetuned Transformer-Bahasa.
-  **Rumi-to-Jawi**, convert from Rumi to Jawi using Transformer.
-  **Sentiment Analysis**, detect and recognize polarity of texts using finetuned Transformer-Bahasa.
-  **Text Similarity**, provide interface for lexical similarity deep semantic similarity using finetuned Transformer-Bahasa.
-  **Spelling Correction**, using local Malaysia NLP researches hybrid with Transformer-Bahasa to auto-correct any bahasa words and NeuSpell using T5-Bahasa.
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
-  **Longer Sequences Transformer**, provide BigBird, BigBird + Pegasus and Fastformer for longer sequence tasks.

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

Also, thanks to `Tensorflow Research Cloud <https://www.tensorflow.org/tfrc>`_ for free TPUs access.

Contributing
----------------

Thank you for contributing this library, really helps a lot. Feel free to contact me to suggest me anything or want to contribute other kind of forms, we accept everything, not just code!
