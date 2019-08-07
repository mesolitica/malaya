
Malaya provided basic interface for XLNET, specific to Malay language,
we called it
`XLNET-Bahasa <https://github.com/huseinzol05/Malaya/tree/master/xlnet>`__.
This interface not able us to use it to do custom training.

If you want to download pretrained model for
`XLNET-Bahasa <https://github.com/huseinzol05/Malaya/tree/master/xlnet>`__
and use it for custom transfer-learning, you can download it here,
https://github.com/huseinzol05/Malaya/tree/master/xlnet, some notebooks
to help you get started,

1. `Text
   classification <https://github.com/huseinzol05/Malaya/tree/master/xlnet/finetune-subjectivity>`__.

Why XLNET
---------

1. Transformer model learn the context of a word based on all of its
   surroundings (live string), bidirectionally. So it much better
   understand left and right hand side relationships, plus permutation
   combination of the sentence to understand more about the context.
2. Because of transformer able to leverage to context during live
   string, we dont need to capture available words in this world,
   instead capture substrings and build the attention after that. XLNET
   will never have Out-Of-Vocab problem.
3. XLNET achieved new state-of-art for modern NLP and able to outperform
   BERT, you can read more about the benchmark
   `here <https://github.com/zihangdai/xlnet#results-on-reading-comprehension>`__.

.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 5.97 s, sys: 1.63 s, total: 7.6 s
    Wall time: 13.3 s


list XLNET-Bahasa available
---------------------------

.. code:: ipython3

    malaya.xlnet.available_xlnet_model()




.. parsed-literal::

    ['base', 'small']



1. ``base`` is pretrained model size ``BASE`` trained on malay language,
   released by Malaya.
2. ``small`` is pretrained model size ``SMALL`` trained on malay
   language, released by Malaya.

load XLNET-Bahasa
-----------------

.. code:: ipython3

    %%time
    model = malaya.xlnet.xlnet(model = 'base')


.. parsed-literal::

    WARNING: Logging before flag parsing goes to stderr.
    W0806 19:02:47.416335 4414723520 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_xlnet/xlnet.py:70: The name tf.gfile.Open is deprecated. Please use tf.io.gfile.GFile instead.
    
    W0806 19:02:47.421904 4414723520 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/xlnet.py:62: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    W0806 19:02:47.439782 4414723520 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_xlnet/xlnet.py:253: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
    
    W0806 19:02:47.441549 4414723520 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_xlnet/xlnet.py:253: The name tf.AUTO_REUSE is deprecated. Please use tf.compat.v1.AUTO_REUSE instead.
    
    W0806 19:02:47.449923 4414723520 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_xlnet/modeling.py:686: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.
    
    W0806 19:02:47.453914 4414723520 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_xlnet/modeling.py:693: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.
    
    W0806 19:02:47.568154 4414723520 deprecation.py:323] From /Users/huseinzol/Documents/Malaya/malaya/_xlnet/modeling.py:797: dropout (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.dropout instead.
    W0806 19:02:49.158231 4414723520 deprecation.py:323] From /Users/huseinzol/Documents/Malaya/malaya/_xlnet/modeling.py:99: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.dense instead.
    W0806 19:02:59.735135 4414723520 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/xlnet.py:75: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    
    W0806 19:03:01.226956 4414723520 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/xlnet.py:81: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
    
    W0806 19:03:01.715428 4414723520 deprecation.py:323] From /usr/local/lib/python3.6/site-packages/tensorflow/python/training/saver.py:1276: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use standard file APIs to check for files with this prefix.


.. parsed-literal::

    CPU times: user 13.7 s, sys: 1.83 s, total: 15.5 s
    Wall time: 17.2 s


.. code:: ipython3

    strings = ['Kerajaan galakkan rakyat naik public transport tapi parking kat lrt ada 15. Reserved utk staff rapid je dah berpuluh. Park kereta tepi jalan kang kene saman dgn majlis perbandaran. Kereta pulak senang kene curi. Cctv pun tak ada. Naik grab dah 5-10 ringgit tiap hari. Gampang juga',
               'Alaa Tun lek ahhh npe muka masam cmni kn agong kata usaha kerajaan terdahulu sejak selepas merdeka',
               "Orang ramai cakap nurse kerajaan garang. So i tell u this. Most of our local ppl will treat us as hamba abdi and they don't respect us as a nurse"]

I have random sentences copied from Twitter, searched using ``kerajan``
keyword.

Vectorization
^^^^^^^^^^^^^

.. code:: ipython3

    v = model.vectorize(strings)
    v.shape




.. parsed-literal::

    (3, 512)



1. Vector size for ``base`` is 512.
2. Vector size for ``small`` is 256.

Attention
^^^^^^^^^

Attention is to get which part of the sentence give the impact. Method
available for attention,

-  ``'last'`` - attention from last layer.
-  ``'first'`` - attention from first layer.
-  ``'mean'`` - average attentions from all layers.

You can give list of strings or a string to get the attention, in this
documentation, I just want to use a string.

.. code:: ipython3

    model.attention(strings[1], method = 'last')




.. parsed-literal::

    [[('Alaa', 0.042688485),
      ('Tun', 0.057717055),
      ('lek', 0.06485453),
      ('ahhh', 0.045797937),
      ('npe', 0.07240139),
      ('muka', 0.06268131),
      ('masam', 0.045819648),
      ('cmni', 0.06796275),
      ('kn', 0.100742154),
      ('agong', 0.10299317),
      ('kata', 0.084064975),
      ('usaha', 0.035359822),
      ('kerajaan', 0.030469837),
      ('terdahulu', 0.04009748),
      ('sejak', 0.049386293),
      ('selepas', 0.049373068),
      ('merdeka', 0.04759017)]]



.. code:: ipython3

    model.attention(strings[1], method = 'first')




.. parsed-literal::

    [[('Alaa', 0.0208059),
      ('Tun', 0.02863956),
      ('lek', 0.03288769),
      ('ahhh', 0.053664364),
      ('npe', 0.060574025),
      ('muka', 0.06008208),
      ('masam', 0.071261086),
      ('cmni', 0.05584477),
      ('kn', 0.062477697),
      ('agong', 0.050815508),
      ('kata', 0.06935718),
      ('usaha', 0.06918364),
      ('kerajaan', 0.07442247),
      ('terdahulu', 0.06999181),
      ('sejak', 0.077083915),
      ('selepas', 0.07548738),
      ('merdeka', 0.067420855)]]



.. code:: ipython3

    model.attention(strings[1], method = 'mean')




.. parsed-literal::

    [[('Alaa', 0.06647704),
      ('Tun', 0.05647921),
      ('lek', 0.0548396),
      ('ahhh', 0.062701255),
      ('npe', 0.055179868),
      ('muka', 0.054572195),
      ('masam', 0.054664183),
      ('cmni', 0.06586684),
      ('kn', 0.056376744),
      ('agong', 0.06807073),
      ('kata', 0.06906264),
      ('usaha', 0.057989392),
      ('kerajaan', 0.05028565),
      ('terdahulu', 0.054037325),
      ('sejak', 0.06337146),
      ('selepas', 0.05514585),
      ('merdeka', 0.054879967)]]


