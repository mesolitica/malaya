.. code:: python

    %%time
    import malaya
    import fasttext


.. parsed-literal::

    CPU times: user 4.81 s, sys: 1.14 s, total: 5.94 s
    Wall time: 6.9 s


List available language detected
--------------------------------

.. code:: python

    malaya.language_detection.label




.. parsed-literal::

    ['eng', 'ind', 'malay', 'manglish', 'other', 'rojak']



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
    fast_text = malaya.language_detection.fasttext()


.. parsed-literal::

    
    


.. code:: python

    model.predict(['តើប្រព័ន្ធប្រតិបត្តិការណាដែលត្រូវគ្នាជាមួយកម្មវិធីធនាគារអេប៊ីអេ។'])




.. parsed-literal::

    ([['__label__km']], array([[0.99841499]]))



.. code:: python

    fast_text.predict(['តើប្រព័ន្ធប្រតិបត្តិការណាដែលត្រូវគ្នាជាមួយកម្មវិធីធនាគារអេប៊ីអេ។'])




.. parsed-literal::

    ['other']



**Language detection in Malaya is not trying to tackle possible
languages in this world, just towards to hyperlocal language.**

.. code:: python

    model.predict(['suka makan ayam dan daging'])




.. parsed-literal::

    ([['__label__id']], array([[0.6334154]]))



.. code:: python

    fast_text.predict_proba(['suka makan ayam dan daging'])




.. parsed-literal::

    [{'eng': 0.0,
      'ind': 0.0,
      'malay': 0.8817721009254456,
      'manglish': 0.0,
      'other': 0.0,
      'rojak': 0.0}]



.. code:: python

    model.predict(malay_text)




.. parsed-literal::

    (('__label__ms',), array([0.57101035]))



.. code:: python

    fast_text.predict_proba([malay_text])




.. parsed-literal::

    [{'eng': 0.0,
      'ind': 0.0,
      'malay': 0.9999504089355469,
      'manglish': 0.0,
      'other': 0.0,
      'rojak': 0.0}]



.. code:: python

    model.predict(socialmedia_malay_text)




.. parsed-literal::

    (('__label__id',), array([0.7870034]))



.. code:: python

    fast_text.predict_proba([socialmedia_malay_text])




.. parsed-literal::

    [{'eng': 0.0,
      'ind': 0.0,
      'malay': 0.9996305704116821,
      'manglish': 0.0,
      'other': 0.0,
      'rojak': 0.0}]



.. code:: python

    model.predict(socialmedia_indon_text)




.. parsed-literal::

    (('__label__fr',), array([0.2912012]))



.. code:: python

    fast_text.predict_proba([socialmedia_indon_text])




.. parsed-literal::

    [{'eng': 0.0,
      'ind': 1.0000293254852295,
      'malay': 0.0,
      'manglish': 0.0,
      'other': 0.0,
      'rojak': 0.0}]



.. code:: python

    model.predict(rojak_text)




.. parsed-literal::

    (('__label__id',), array([0.87948251]))



.. code:: python

    fast_text.predict_proba([rojak_text])




.. parsed-literal::

    [{'eng': 0.0,
      'ind': 0.0,
      'malay': 0.0,
      'manglish': 0.0,
      'other': 0.0,
      'rojak': 0.9994134306907654}]



.. code:: python

    model.predict(manglish_text)




.. parsed-literal::

    (('__label__en',), array([0.89707506]))



.. code:: python

    fast_text.predict_proba([manglish_text])




.. parsed-literal::

    [{'eng': 0.0,
      'ind': 0.0,
      'malay': 0.0,
      'manglish': 1.00004243850708,
      'other': 0.0,
      'rojak': 0.0}]



.. code:: python

    model.predict(chinese_text)




.. parsed-literal::

    (('__label__zh',), array([0.97311586]))



.. code:: python

    fast_text.predict_proba([chinese_text])




.. parsed-literal::

    [{'eng': 0.0,
      'ind': 0.0,
      'malay': 0.0,
      'manglish': 0.0,
      'other': 0.9921814203262329,
      'rojak': 0.0}]



.. code:: python

    fast_text.predict_proba([indon_text,malay_text])




.. parsed-literal::

    [{'eng': 0.0,
      'ind': 1.0000287294387817,
      'malay': 0.0,
      'manglish': 0.0,
      'other': 0.0,
      'rojak': 0.0},
     {'eng': 0.0,
      'ind': 0.0,
      'malay': 0.9999504089355469,
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

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/model/tf.py:28: The name tf.sparse_placeholder is deprecated. Please use tf.compat.v1.sparse_placeholder instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/model/tf.py:30: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/model/tf.py:31: The name tf.truncated_normal is deprecated. Please use tf.random.truncated_normal instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow_core/python/ops/embedding_ops.py:515: div (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Deprecated in favor of operator or tf.math.divide.
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/model/tf.py:35: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.Dense instead.
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow_core/python/layers/core.py:187: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.__call__` method instead.
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/model/tf.py:43: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/model/tf.py:44: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/model/tf.py:45: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/model/tf.py:45: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.
    
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/language-detection/deep/model.ckpt


.. code:: python

    deep.predict_proba([indon_text])


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/model/tf.py:21: The name tf.SparseTensorValue is deprecated. Please use tf.compat.v1.SparseTensorValue instead.
    




.. parsed-literal::

    [{'eng': 3.6145173e-06,
      'ind': 0.9998909,
      'malay': 5.4685403e-05,
      'manglish': 5.7687397e-09,
      'other': 5.8103406e-06,
      'rojak': 4.4987148e-05}]



.. code:: python

    deep.predict_proba([malay_text])




.. parsed-literal::

    [{'eng': 9.500837e-11,
      'ind': 0.0004703698,
      'malay': 0.9991295,
      'manglish': 1.602048e-13,
      'other': 1.9133091e-07,
      'rojak': 0.0004000054}]



.. code:: python

    deep.predict_proba([indon_text,malay_text])




.. parsed-literal::

    [{'eng': 3.6145207e-06,
      'ind': 0.9998909,
      'malay': 5.468535e-05,
      'manglish': 5.7687397e-09,
      'other': 5.8103406e-06,
      'rojak': 4.4987148e-05},
     {'eng': 9.500837e-11,
      'ind': 0.0004703698,
      'malay': 0.9991295,
      'manglish': 1.602048e-13,
      'other': 1.9133091e-07,
      'rojak': 0.0004000056}]



.. code:: python

    deep.predict_proba([socialmedia_malay_text])




.. parsed-literal::

    [{'eng': 1.4520887e-09,
      'ind': 0.0064318455,
      'malay': 0.9824693,
      'manglish': 2.1923141e-13,
      'other': 1.06363805e-05,
      'rojak': 0.0110881105}]



.. code:: python

    deep.predict_proba([socialmedia_indon_text])




.. parsed-literal::

    [{'eng': 4.0632068e-07,
      'ind': 0.9999995,
      'malay': 6.871639e-10,
      'manglish': 7.4285925e-11,
      'other': 1.5928721e-07,
      'rojak': 4.892652e-10}]



.. code:: python

    deep.predict_proba([rojak_text, malay_text])




.. parsed-literal::

    [{'eng': 0.0040922514,
      'ind': 0.02200061,
      'malay': 0.0027574676,
      'manglish': 9.336553e-06,
      'other': 0.00023811469,
      'rojak': 0.97090226},
     {'eng': 9.500837e-11,
      'ind': 0.0004703698,
      'malay': 0.9991295,
      'manglish': 1.602048e-13,
      'other': 1.9133091e-07,
      'rojak': 0.0004000056}]


