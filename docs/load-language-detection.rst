.. code:: python

    %%time
    import malaya
    import fasttext


.. parsed-literal::

    CPU times: user 3.95 s, sys: 741 ms, total: 4.69 s
    Wall time: 4.1 s


List available language detected
--------------------------------

.. code:: python

    malaya.language_detection.label()




.. parsed-literal::

    {0: 'eng', 1: 'ind', 2: 'malay', 3: 'manglish', 4: 'other', 5: 'rojak'}



.. code:: python

    chinese_text = '今天是６月１８号，也是Muiriel的生日！'
    english_text = 'i totally love it man'
    indon_text = 'menjabat saleh perombakan menjabat periode komisi energi fraksi partai pengurus partai periode periode partai terpilih periode menjabat komisi perdagangan investasi persatuan periode'
    malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'
    socialmedia_malay_text = 'nti aku tengok dulu tiket dari kl pukul berapa ada nahh'
    socialmedia_indon_text = 'saking kangen papanya pas vc anakku nangis'
    rojak_text = 'jadi aku tadi bikin ini gengs dan dijual haha salad only k dan haha drinks only k'
    manglish_text = 'power lah even shopback come to edmw riao'

Load Fast-text model
--------------------

Make sure fast-text already installed, if not, simply,

.. code:: bash

   pip install fasttext

In this example, I am going to compare with pretrained fasttext from
Facebook. https://fasttext.cc/docs/en/language-identification.html

Simply download pretrained model,

.. code:: bash

   wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

.. code:: python

    model = fasttext.load_model('lid.176.ftz') 


.. parsed-literal::

    


.. code:: python

    fast_text = malaya.language_detection.fasttext()


.. parsed-literal::

    


.. code:: python

    fast_text.predict(indon_text,get_proba=True)




.. parsed-literal::

    {'eng': 0.0,
     'ind': 1.0000107288360596,
     'malay': 0.0,
     'manglish': 0.0,
     'other': 0.0,
     'rojak': 0.0}



.. code:: python

    model.predict(indon_text)




.. parsed-literal::

    (('__label__id',), array([0.40272361]))



.. code:: python

    fast_text.predict(malay_text,get_proba=True)




.. parsed-literal::

    {'eng': 0.0,
     'ind': 0.0,
     'malay': 0.9999417066574097,
     'manglish': 0.0,
     'other': 0.0,
     'rojak': 0.0}



.. code:: python

    model.predict(malay_text)




.. parsed-literal::

    (('__label__ms',), array([0.57101035]))



.. code:: python

    fast_text.predict(socialmedia_malay_text,get_proba=True)




.. parsed-literal::

    {'eng': 0.0,
     'ind': 0.0,
     'malay': 0.9999960660934448,
     'manglish': 0.0,
     'other': 0.0,
     'rojak': 0.0}



.. code:: python

    model.predict(socialmedia_malay_text)




.. parsed-literal::

    (('__label__id',), array([0.7870034]))



.. code:: python

    fast_text.predict(socialmedia_indon_text,get_proba=True)




.. parsed-literal::

    {'eng': 0.0,
     'ind': 1.0000200271606445,
     'malay': 0.0,
     'manglish': 0.0,
     'other': 0.0,
     'rojak': 0.0}



.. code:: python

    model.predict(socialmedia_indon_text)




.. parsed-literal::

    (('__label__fr',), array([0.2912012]))



.. code:: python

    fast_text.predict(rojak_text,get_proba=True)




.. parsed-literal::

    {'eng': 0.0,
     'ind': 0.0,
     'malay': 0.0,
     'manglish': 0.0,
     'other': 0.0,
     'rojak': 0.9999275207519531}



.. code:: python

    model.predict(rojak_text)




.. parsed-literal::

    (('__label__id',), array([0.87948251]))



.. code:: python

    fast_text.predict(manglish_text,get_proba=True)




.. parsed-literal::

    {'eng': 0.0,
     'ind': 0.0,
     'malay': 0.0,
     'manglish': 1.000038981437683,
     'other': 0.0,
     'rojak': 0.0}



.. code:: python

    model.predict(manglish_text)




.. parsed-literal::

    (('__label__en',), array([0.89707506]))



.. code:: python

    fast_text.predict(chinese_text, get_proba = True)




.. parsed-literal::

    {'eng': 0.0,
     'ind': 0.0,
     'malay': 0.0,
     'manglish': 0.0,
     'other': 0.5427265167236328,
     'rojak': 0.0}



.. code:: python

    model.predict(chinese_text)




.. parsed-literal::

    (('__label__zh',), array([0.97311586]))



.. code:: python

    fast_text.predict_batch([indon_text,malay_text],get_proba=True)




.. parsed-literal::

    [{'eng': 0.0,
      'ind': 1.0000107288360596,
      'malay': 0.0,
      'manglish': 0.0,
      'other': 0.0,
      'rojak': 0.0},
     {'eng': 0.0,
      'ind': 0.0,
      'malay': 0.9999417066574097,
      'manglish': 0.0,
      'other': 0.0,
      'rojak': 0.0}]



Load Deep learning model
------------------------

Deep learning model is slightly more accurate then fast-text model, but
the size is around 50MB, while fast-text just like 15MB. Can check
accuracy comparison at here,
https://malaya.readthedocs.io/en/latest/Accuracy.html#language-detection

.. code:: python

    deep = malaya.language_detection.deep_model()


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_models/_tensorflow_model.py:36: The name tf.sparse_placeholder is deprecated. Please use tf.compat.v1.sparse_placeholder instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_models/_tensorflow_model.py:38: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_models/_tensorflow_model.py:39: The name tf.truncated_normal is deprecated. Please use tf.random.truncated_normal instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow_core/python/ops/embedding_ops.py:515: div (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Deprecated in favor of operator or tf.math.divide.
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_models/_tensorflow_model.py:43: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.Dense instead.
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow_core/python/layers/core.py:187: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.__call__` method instead.
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_models/_tensorflow_model.py:65: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_models/_tensorflow_model.py:66: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_models/_tensorflow_model.py:67: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_models/_tensorflow_model.py:67: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.
    
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/language-detection/deep/model.ckpt


.. code:: python

    deep.predict(indon_text)


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_models/_tensorflow_model.py:29: The name tf.SparseTensorValue is deprecated. Please use tf.compat.v1.SparseTensorValue instead.
    




.. parsed-literal::

    'ind'



.. code:: python

    deep.predict(malay_text)




.. parsed-literal::

    'malay'



.. code:: python

    deep.predict_batch([indon_text,malay_text])




.. parsed-literal::

    ['ind', 'malay']



.. code:: python

    deep.predict(socialmedia_malay_text)




.. parsed-literal::

    'malay'



.. code:: python

    deep.predict(socialmedia_indon_text)




.. parsed-literal::

    'ind'



.. code:: python

    deep.predict(rojak_text, get_proba = True)




.. parsed-literal::

    {'eng': 7.730631e-08,
     'ind': 0.008739273,
     'malay': 0.00026563255,
     'manglish': 3.1339885e-05,
     'other': 7.3840456e-06,
     'rojak': 0.99095637}



.. code:: python

    deep.predict_batch([rojak_text, malay_text])




.. parsed-literal::

    ['rojak', 'malay']



