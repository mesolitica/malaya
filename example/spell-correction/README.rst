.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 3.48 s, sys: 539 ms, total: 4.02 s
    Wall time: 3.4 s


.. code:: ipython3

    # some text examples copied from Twitter
    
    string1 = 'krajaan patut bagi pencen awal skt kpd warga emas supaya emosi'
    string2 = 'Husein ska mkn aym dkat kampng Jawa'
    string3 = 'Melayu malas ni narration dia sama je macam men are trash. True to some, false to some.'
    string4 = 'Tapi tak pikir ke bahaya perpetuate myths camtu. Nanti kalau ada hiring discrimination despite your good qualifications because of your race tau pulak marah. Your kids will be victims of that too.'
    string5 = 'DrM cerita Melayu malas semenjak saya kat University (early 1980s) and now as i am edging towards retirement in 4-5 years time after a career of being an Engineer, Project Manager, General Manager'
    string6 = 'blh bntg dlm kls nlp sy, nnti intch'

Load probability speller
------------------------

The probability speller extends the functionality of the Peter Norvig’s,
http://norvig.com/spell-correct.html.

And improve it using some algorithms from Normalization of noisy texts
in Malaysian online reviews,
https://www.researchgate.net/publication/287050449_Normalization_of_noisy_texts_in_Malaysian_online_reviews.

Also added custom vowels and consonant augmentation to adapt with our
local shortform / typos.

.. code:: ipython3

    prob_corrector = malaya.spell.probability()

To correct a word
^^^^^^^^^^^^^^^^^

.. code:: ipython3

    prob_corrector.correct('sy')




.. parsed-literal::

    'saya'



.. code:: ipython3

    prob_corrector.correct('mhthir')




.. parsed-literal::

    'mahathir'



.. code:: ipython3

    prob_corrector.correct('mknn')




.. parsed-literal::

    'makanan'



List possible generated pool of words
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    prob_corrector.edit_candidates('mhthir')




.. parsed-literal::

    {'mahathir'}



.. code:: ipython3

    prob_corrector.edit_candidates('smbng')




.. parsed-literal::

    {'sambang',
     'sambong',
     'sambung',
     'sembang',
     'sembong',
     'sembung',
     'simbang',
     'smbg',
     'sombong',
     'sumbang',
     'sumbing'}



Now you can see, ``edit_candidates`` suggested quite a lot candidates
and some of candidates not an actual word like ``sambang``, to reduce
that, we can use
`sentencepiece <https://github.com/google/sentencepiece>`__ to check a
candidate a legit word for malaysia context or not.

.. code:: ipython3

    prob_corrector_sp = malaya.spell.probability(sentence_piece = True)
    prob_corrector_sp.edit_candidates('smbng')




.. parsed-literal::

    {'sambong',
     'sambung',
     'sembang',
     'sembong',
     'sembung',
     'smbg',
     'sombong',
     'sumbang',
     'sumbing'}



**So how does the model knows which words need to pick? highest counts
from wikipedia!**

To correct a sentence
^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    prob_corrector.correct_text(string1)




.. parsed-literal::

    'kerajaan patut bagi pencen awal sakit kepada warga emas supaya emosi'



.. code:: ipython3

    prob_corrector.correct_text(string2)




.. parsed-literal::

    'Husein suka makan ayam dekat kampung Jawa'



.. code:: ipython3

    prob_corrector.correct_text(string3)




.. parsed-literal::

    'Melayu malas ni narration dia sama sahaja macam men are trash. True to some, false to some.'



.. code:: ipython3

    prob_corrector.correct_text(string4)




.. parsed-literal::

    'Tapi tak fikir ke bahaya perpetuate myths macam itu. Nanti kalau ada hiring discrimination despite your good qualifications because of your race tahu pula marah. Your kids will be victims of that too.'



.. code:: ipython3

    prob_corrector.correct_text(string5)




.. parsed-literal::

    'DrM cerita Melayu malas semenjak saya kat University (early 1980s) and now as saya am edging towards retirement in 4-5 years time after a career of being an Engineer, Project Manager, General Manager'



.. code:: ipython3

    prob_corrector.correct_text(string6)




.. parsed-literal::

    'boleh bintang dalam kelas nlp saya, nanti intch'



Load transformer speller
------------------------

This spelling correction is a transformer based, improvement version of
``malaya.spell.probability``. Problem with ``malaya.spell.probability``,
it naively picked highest probability of word based on public sentences
(wiki, news and social media) without understand actual context,
example,

.. code:: python

   string = 'krajaan patut bagi pencen awal skt kpd warga emas supaya emosi'
   prob_corrector = malaya.spell.probability()
   prob_corrector.correct_text(string)
   -> 'kerajaan patut bagi pencen awal sakit kepada warga emas supaya emosi'

It supposely replaced ``skt`` with ``sikit``, a common word people use
in social media to give a little bit of attention to ``pencen``. So, to
fix that, we can use Transformer model! **Right now transformer speller
supported ``BERT`` and ``ALBERT`` only, XLNET is not that good**.

.. code:: ipython3

    model = malaya.transformer.load(model = 'bert', size = 'small')
    transformer_corrector = malaya.spell.transformer(model, sentence_piece = True)


.. parsed-literal::

    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/bert/modeling.py:93: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_transformer/_bert.py:48: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/bert/modeling.py:171: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/bert/modeling.py:409: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/bert/modeling.py:490: The name tf.assert_less_equal is deprecated. Please use tf.compat.v1.assert_less_equal instead.
    
    WARNING:tensorflow:
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
      * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/bert/modeling.py:671: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.Dense instead.
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow_core/python/layers/core.py:187: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.__call__` method instead.
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_transformer/_bert.py:85: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_transformer/_bert.py:86: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_transformer/_bert.py:87: The name tf.get_collection is deprecated. Please use tf.compat.v1.get_collection instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_transformer/_bert.py:88: The name tf.GraphKeys is deprecated. Please use tf.compat.v1.GraphKeys instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_transformer/_bert.py:93: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_transformer/_bert.py:95: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/bert-model/small/bert-small-v2/model.ckpt


.. code:: ipython3

    transformer_corrector.correct_text(string1)




.. parsed-literal::

    'kerajaan patut bagi pencen awal sikit kepada warga emas supaya emosi'



perfect! But again, transformer model is very expensive! You can compare
the time wall with probability based.

.. code:: ipython3

    %%time
    transformer_corrector.correct_text(string1)


.. parsed-literal::

    CPU times: user 28.7 s, sys: 1.5 s, total: 30.2 s
    Wall time: 6.11 s




.. parsed-literal::

    'kerajaan patut bagi pencen awal sikit kepada warga emas supaya emosi'



.. code:: ipython3

    %%time
    prob_corrector.correct_text(string1)


.. parsed-literal::

    CPU times: user 105 ms, sys: 7.19 ms, total: 112 ms
    Wall time: 112 ms




.. parsed-literal::

    'kerajaan patut bagi pencen awal sakit kepada warga emas supaya emosi'



.. code:: ipython3

    transformer_corrector.correct_text(string2)




.. parsed-literal::

    'Husein suke makan ayam dekat kampung Jawa'



Transformer did a mistake suggested ``suke`` instead ``suka``, this is
because Malaya Transformer trained more on local context (social media)
instead of standard context.

Load symspeller speller
-----------------------

This spelling correction is an improvement version for
`symspeller <https://github.com/mammothb/symspellpy>`__ to adapt with
our local shortform / typos. Before you able to use this spelling
correction, you need to install
`symspeller <https://github.com/mammothb/symspellpy>`__,

.. code:: bash

   pip install symspellpy

.. code:: ipython3

    symspell_corrector = malaya.spell.symspell()

To correct a word
^^^^^^^^^^^^^^^^^

.. code:: ipython3

    symspell_corrector.correct('bntng')




.. parsed-literal::

    'bintang'



.. code:: ipython3

    symspell_corrector.correct('kerajaan')




.. parsed-literal::

    'kerajaan'



.. code:: ipython3

    symspell_corrector.correct('mknn')




.. parsed-literal::

    'makanan'



List possible generated words
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    symspell_corrector.edit_step('mrh')




.. parsed-literal::

    {'marah': 12684.0,
     'merah': 21448.5,
     'arah': 15066.5,
     'darah': 10003.0,
     'mara': 7504.5,
     'malah': 7450.0,
     'zarah': 3753.5,
     'murah': 3575.5,
     'barah': 2707.5,
     'march': 2540.5,
     'martha': 390.0,
     'marsha': 389.0,
     'maratha': 88.5,
     'marcha': 22.5,
     'karaha': 13.5,
     'maraba': 13.5,
     'varaha': 11.5,
     'marana': 4.5,
     'marama': 4.5}



To correct a sentence
^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    symspell_corrector.correct_text(string1)




.. parsed-literal::

    'kerajaan patut bagi pencen awal saat kepada warga emas supaya emosi'



.. code:: ipython3

    symspell_corrector.correct_text(string2)




.. parsed-literal::

    'Husein sama makan ayam dapat kampung Jawa'



.. code:: ipython3

    symspell_corrector.correct_text(string3)




.. parsed-literal::

    'Melayu malas ni narration dia sama sahaja macam men are trash. True to some, false to some.'



.. code:: ipython3

    symspell_corrector.correct_text(string4)




.. parsed-literal::

    'Tapi tak fikir ke bahaya perpetuate maathai macam itu. Nanti kalau ada hiring discrimination despite your good qualifications because of your race tahu pula marah. Your kids will be victims of that too.'



.. code:: ipython3

    symspell_corrector.correct_text(string5)




.. parsed-literal::

    'DrM cerita Melayu malas semenjak saya kat University (early 1980s) and now as saya am edging towards retirement in 4-5 aras time after a career of being an Engineer, Project Manager, General Manager'



.. code:: ipython3

    symspell_corrector.correct_text(string6)




.. parsed-literal::

    'ialah bintang dalam kelas malaya saya, nanti mintalah'


