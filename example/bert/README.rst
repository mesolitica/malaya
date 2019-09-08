
Malaya provided basic interface for BERT, specific to Malay language, we
called it
`BERT-Bahasa <https://github.com/huseinzol05/Malaya/tree/master/bert>`__.
This interface not able us to use it to do custom training.

If you want to download pretrained model for
`BERT-Bahasa <https://github.com/huseinzol05/Malaya/tree/master/pretrained-model/bert>`__
and use it for custom transfer-learning, you can download it here,
https://github.com/huseinzol05/Malaya/tree/master/pretrained-model/bert,
some notebooks to help you get started,

1. `Text
   classification <https://github.com/huseinzol05/Malaya/tree/master/pretrained-model/bert/finetune/finetune-subjectivity>`__.
2. `Text
   similarity <https://github.com/huseinzol05/Malaya/tree/master/pretrained-model/bert/finetune/finetune-similarity>`__.

Why BERT
--------

1. Transformer model learn the context of a word based on all of its
   surroundings (live string), bidirectionally. So it much better
   understand left and right hand side relationships.
2. Because of transformer able to leverage to context during live
   string, we dont need to capture available words in this world,
   instead capture substrings and build the attention after that. BERT
   will never have Out-Of-Vocab problem.
3. BERT achieved new state-of-art for modern NLP, you can read more
   about the benchmark
   `here <https://github.com/google-research/bert#introduction>`__.

.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 5.42 s, sys: 953 ms, total: 6.37 s
    Wall time: 7.69 s


list BERT-Bahasa available
--------------------------

.. code:: ipython3

    malaya.bert.available_bert_model()




.. parsed-literal::

    ['multilanguage', 'base', 'small']



1. ``multilanguage`` is pretrained model size ``BASE`` trained on
   multilanguage, released by Google.
2. ``base`` is pretrained model size ``BASE`` trained on malay language,
   released by Malaya.
3. ``small`` is pretrained model size ``SMALL`` trained on malay
   language, released by Malaya.

load BERT-Bahasa
----------------

.. code:: ipython3

    %%time
    model = malaya.bert.bert(model = 'base')


.. parsed-literal::

    WARNING: Logging before flag parsing goes to stderr.
    W0908 23:12:58.925322 4632950208 deprecation_wrapper.py:119] From /usr/local/lib/python3.6/site-packages/bert/modeling.py:93: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    W0908 23:12:58.929401 4632950208 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/bert.py:50: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    W0908 23:12:58.949140 4632950208 deprecation_wrapper.py:119] From /usr/local/lib/python3.6/site-packages/bert/modeling.py:171: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
    
    W0908 23:12:58.953647 4632950208 deprecation_wrapper.py:119] From /usr/local/lib/python3.6/site-packages/bert/modeling.py:409: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.
    
    W0908 23:12:59.016011 4632950208 deprecation_wrapper.py:119] From /usr/local/lib/python3.6/site-packages/bert/modeling.py:490: The name tf.assert_less_equal is deprecated. Please use tf.compat.v1.assert_less_equal instead.
    
    W0908 23:12:59.100100 4632950208 deprecation.py:323] From /usr/local/lib/python3.6/site-packages/bert/modeling.py:671: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.dense instead.
    W0908 23:13:03.390677 4632950208 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/bert.py:62: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    
    W0908 23:13:05.371360 4632950208 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/bert.py:67: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
    
    W0908 23:13:05.747210 4632950208 deprecation.py:323] From /usr/local/lib/python3.6/site-packages/tensorflow/python/training/saver.py:1276: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use standard file APIs to check for files with this prefix.


.. parsed-literal::

    CPU times: user 10.1 s, sys: 1.72 s, total: 11.9 s
    Wall time: 8.94 s


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

    (3, 768)



1. Vector size for ``multilanguage`` is 768.
2. Vector size for ``base`` is 768.
3. Vector size for ``small`` is 512.

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

    [[('Alaa', 0.07662392),
      ('Tun', 0.06297707),
      ('lek', 0.05288772),
      ('ahhh', 0.12370589),
      ('npe', 0.031775884),
      ('muka', 0.07801706),
      ('masam', 0.04338463),
      ('cmni', 0.058882464),
      ('kn', 0.054250218),
      ('agong', 0.15547961),
      ('kata', 0.07811978),
      ('usaha', 0.05006122),
      ('kerajaan', 0.041515753),
      ('terdahulu', 0.0357033),
      ('sejak', 0.012836863),
      ('selepas', 0.01774636),
      ('merdeka', 0.02603223)]]



.. code:: ipython3

    model.attention(strings[1], method = 'first')




.. parsed-literal::

    [[('Alaa', 0.035761356),
      ('Tun', 0.049164597),
      ('lek', 0.027038181),
      ('ahhh', 0.07766667),
      ('npe', 0.041269727),
      ('muka', 0.021136),
      ('masam', 0.095767364),
      ('cmni', 0.033513222),
      ('kn', 0.019944489),
      ('agong', 0.17159887),
      ('kata', 0.029981887),
      ('usaha', 0.035313524),
      ('kerajaan', 0.039649293),
      ('terdahulu', 0.07565842),
      ('sejak', 0.031608675),
      ('selepas', 0.09213663),
      ('merdeka', 0.12279116)]]



.. code:: ipython3

    model.attention(strings[1], method = 'mean')




.. parsed-literal::

    [[('Alaa', 0.067627385),
      ('Tun', 0.05937794),
      ('lek', 0.062164135),
      ('ahhh', 0.05282652),
      ('npe', 0.05113411),
      ('muka', 0.050083853),
      ('masam', 0.05692221),
      ('cmni', 0.07740603),
      ('kn', 0.056951318),
      ('agong', 0.08621354),
      ('kata', 0.06419954),
      ('usaha', 0.057119563),
      ('kerajaan', 0.0406653),
      ('terdahulu', 0.06452254),
      ('sejak', 0.047204666),
      ('selepas', 0.050263014),
      ('merdeka', 0.055318326)]]



Visualize Attention
^^^^^^^^^^^^^^^^^^^

Before using attention visualization, we need to load D3 into our
jupyter notebook first.

.. code:: javascript

    %%javascript
    require.config({
      paths: {
          d3: '//cdnjs.cloudflare.com/ajax/libs/d3/3.4.8/d3.min',
          jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
      }
    });



.. parsed-literal::

    <IPython.core.display.Javascript object>


.. code:: ipython3

    model.visualize_attention('nak makan ayam dgn husein')



.. raw:: html

    
              <span style="user-select:none">
                Layer: <select id="layer"></select>
              </span>
              <div id='vis'></div>
            



.. parsed-literal::

    <IPython.core.display.Javascript object>



.. parsed-literal::

    <IPython.core.display.Javascript object>


.. code:: ipython3

    from IPython.core.display import Image, display
    
    display(Image('bert-attention.png', width=300))



.. image:: load-bert_files/load-bert_20_0.png
   :width: 300px

