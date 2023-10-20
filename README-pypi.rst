**Malaya** is a Natural-Language-Toolkit library for bahasa Malaysia, powered by PyTorch.

Documentation
--------------

Proper documentation is available at https://malaya.readthedocs.io/

Installing from the PyPI
----------------------------------

::

    $ pip install malaya

It will automatically install all dependencies except for PyTorch. So you can choose your own PyTorch CPU / GPU version.

Only **Python >= 3.6.0**,  and **PyTorch >= 1.10** are supported.

If you are a Windows user, make sure read https://malaya.readthedocs.io/en/latest/running-on-windows.html

Development Release
---------------------------------

Install from `master` branch,

::

    $ pip install git+https://github.com/huseinzol05/malaya.git


We recommend to use **virtualenv** for development. 

Documentation at https://malaya.readthedocs.io/en/latest/

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
- **Llama2**, Llama 2: Open Foundation and Fine-Tuned Chat Models, https://arxiv.org/abs/2307.09288

References
-----------

If you use our software for research, please cite:

::

  @misc{Malaya, Natural-Language-Toolkit library for bahasa Malaysia, powered by PyTorch,
    author = {Husein, Zolkepli},
    title = {Malaya},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/mesolitica/malaya}}
  }

Acknowledgement
----------------

Thanks to `KeyReply <https://www.keyreply.com/>`_ for private V100s cloud and `Mesolitica <https://mesolitica.com/>`_ for private RTXs cloud to train Malaya-Speech models.

Also, thanks to `Tensorflow Research Cloud <https://www.tensorflow.org/tfrc>`_ for free TPUs access.

Contributing
----------------

Thank you for contributing this library, really helps a lot. Feel free to contact me to suggest me anything or want to contribute other kind of forms, we accept everything, not just code!
