
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.4 s, sys: 1.76 s, total: 14.2 s
    Wall time: 19 s


List available language detected
--------------------------------

.. code:: python

    malaya.language_detection.label()




.. parsed-literal::

    {0: 'OTHER', 1: 'ENGLISH', 2: 'INDONESIA', 3: 'MALAY'}



.. code:: python

    chinese_text = '今天是６月１８号，也是Muiriel的生日！'
    english_text = 'i totally love it man'
    indon_text = 'menjabat saleh perombakan menjabat periode komisi energi fraksi partai pengurus partai periode periode partai terpilih periode menjabat komisi perdagangan investasi persatuan periode'
    malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'

Load multinomial model
----------------------

.. code:: python

    multinomial = malaya.language_detection.multinomial()
    multinomial.predict(chinese_text,get_proba=True)




.. parsed-literal::

    {'OTHER': 0.9994579061567993,
     'ENGLISH': 0.0005420938432138138,
     'INDONESIA': 7.676027325117918e-23,
     'MALAY': 8.781948877234918e-26}



.. code:: python

    multinomial.predict(english_text)




.. parsed-literal::

    'ENGLISH'



.. code:: python

    multinomial.predict(indon_text)




.. parsed-literal::

    'INDONESIA'



.. code:: python

    multinomial.predict(malay_text)




.. parsed-literal::

    'MALAY'



.. code:: python

    multinomial.predict(malay_text,get_proba=True)




.. parsed-literal::

    {'OTHER': 2.0343169760514329e-209,
     'ENGLISH': 0.0,
     'INDONESIA': 2.5947121007026616e-193,
     'MALAY': 1.0}



.. code:: python

    multinomial.predict_batch([english_text,malay_text])




.. parsed-literal::

    ['ENGLISH', 'MALAY']



.. code:: python

    multinomial.predict_batch([indon_text,malay_text],get_proba=True)




.. parsed-literal::

    [{'OTHER': 0.0, 'ENGLISH': 0.0, 'INDONESIA': 1.0, 'MALAY': 0.0},
     {'OTHER': 2.0343169760514329e-209,
      'ENGLISH': 0.0,
      'INDONESIA': 2.5947121007026616e-193,
      'MALAY': 1.0}]



Load SGD model
--------------

.. code:: python

    sgd = malaya.language_detection.sgd()
    sgd.predict(chinese_text,get_proba=True)




.. parsed-literal::

    {'OTHER': 1.0, 'ENGLISH': 0.0, 'INDONESIA': 0.0, 'MALAY': 0.0}



.. code:: python

    sgd.predict(english_text)




.. parsed-literal::

    'ENGLISH'



.. code:: python

    sgd.predict(indon_text)




.. parsed-literal::

    'INDONESIA'



.. code:: python

    sgd.predict(malay_text)




.. parsed-literal::

    'OTHER'



.. code:: python

    sgd.predict(malay_text,get_proba=True)




.. parsed-literal::

    {'OTHER': 0.25, 'ENGLISH': 0.25, 'INDONESIA': 0.25, 'MALAY': 0.25}



.. code:: python

    sgd.predict_batch([english_text,malay_text])




.. parsed-literal::

    ['ENGLISH', 'OTHER']



.. code:: python

    sgd.predict_batch([indon_text,malay_text],get_proba=True)




.. parsed-literal::

    [{'OTHER': 0.0, 'ENGLISH': 0.0, 'INDONESIA': 1.0, 'MALAY': 0.0},
     {'OTHER': 0.25, 'ENGLISH': 0.25, 'INDONESIA': 0.25, 'MALAY': 0.25}]



Load XGB model
--------------

.. code:: python

    xgb = malaya.language_detection.xgb()
    xgb.predict(chinese_text)




.. parsed-literal::

    'OTHER'



.. code:: python

    xgb.predict(indon_text,get_proba=True)




.. parsed-literal::

    {'OTHER': 1.7766696e-09,
     'ENGLISH': 4.3007767e-11,
     'INDONESIA': 1.0,
     'MALAY': 2.0483236e-08}



.. code:: python

    xgb.predict_batch([indon_text,malay_text],get_proba=True)




.. parsed-literal::

    [{'OTHER': 1.7766696e-09,
      'ENGLISH': 4.3007767e-11,
      'INDONESIA': 1.0,
      'MALAY': 2.0483236e-08},
     {'OTHER': 0.025863007,
      'ENGLISH': 6.506632e-07,
      'INDONESIA': 0.0044011325,
      'MALAY': 0.9697352}]



Load Deep learning model
------------------------

.. code:: python

    deep = malaya.language_detection.deep_model()
    deep.predict(chinese_text)


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from /Users/huseinzol/Malaya/language-detection/deep/model.ckpt




.. parsed-literal::

    'OTHER'



.. code:: python

    deep.predict(indon_text)




.. parsed-literal::

    'INDONESIA'



.. code:: python

    deep.predict(malay_text)




.. parsed-literal::

    'MALAY'



.. code:: python

    deep.predict_batch([indon_text,malay_text])




.. parsed-literal::

    ['INDONESIA', 'MALAY']
