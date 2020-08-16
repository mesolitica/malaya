UnicodeDecodeError: ‘charmap’ codec can’t decode byte
-----------------------------------------------------

To solve this,

Windows Settings > Administrative language settings > Change system
locale.

Checked Beta: Use Unicode UTF-8 for worldwide language support.

Restarted, everything works well.

Full dicussion check `issue
25 <https://github.com/huseinzol05/Malaya/issues/25>`__.

youtokentome failed to build
----------------------------

`YouTokenToMe <https://github.com/VKCOM/YouTokenToMe>`__ required cython
to compile and usually Windows users will break on this part, so we need
to install Malaya without
`YouTokenToMe <https://github.com/VKCOM/YouTokenToMe>`__.

.. code:: bash

   pip install malaya --no-deps
   pip install tensorflow==1.15
   pip install bert-tensorflow albert-tensorflow

If we skipped `YouTokenToMe <https://github.com/VKCOM/YouTokenToMe>`__,
we not able to use languaged-detection module,
https://malaya.readthedocs.io/en/latest/Language.html

Unable to use any T5 models
---------------------------

T5 depends on tensorflow-text, currently there is no official
tensorflow-text binary released for Windows. So no T5 model for Windows
users.

List T5 models,

1. `malaya.summarization.abstractive.t5 <https://malaya.readthedocs.io/en/latest/Abstractive.html#load-t5>`__
2. `malaya.generator.t5 <https://malaya.readthedocs.io/en/latest/Generator.html#load-t5>`__
3. `malaya.paraphrase.t5 <https://malaya.readthedocs.io/en/latest/Paraphrase.html#load-t5-models>`__

Lack development on Windows
---------------------------

Malaya been tested on modern UNIX / LINUX systems that supported
tensorflow binary. Windows support not our development path right now
(we don’t have any Windows machine here).
