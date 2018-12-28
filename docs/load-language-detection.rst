
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 13.3 s, sys: 1.62 s, total: 14.9 s
    Wall time: 19.3 s


List available language detected
--------------------------------

.. code:: ipython3

    malaya.language_detection.label()




.. parsed-literal::

    {0: 'OTHER', 1: 'ENGLISH', 2: 'INDONESIA', 3: 'MALAY'}



.. code:: ipython3

    chinese_text = '今天是６月１８号，也是Muiriel的生日！'
    english_text = 'i totally love it man'
    indon_text = 'berbicara dalam bahasa Indonesia membutuhkan teknologi yang baik untuk bekerja dengan baik, tetapi teknologi yang sulit didapat'
    malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'

Load multinomial model
----------------------

.. code:: ipython3

    multinomial = malaya.language_detection.multinomial()
    multinomial.predict(chinese_text,get_proba=True)




.. parsed-literal::

    {'OTHER': 0.9999999999998863,
     'ENGLISH': 1.0007472112772622e-13,
     'INDONESIA': 6.841882467097028e-20,
     'MALAY': 3.2100041975729288e-31}



.. code:: ipython3

    multinomial.predict(english_text)




.. parsed-literal::

    'ENGLISH'



.. code:: ipython3

    multinomial.predict(indon_text)




.. parsed-literal::

    'INDONESIA'



.. code:: ipython3

    multinomial.predict(malay_text)




.. parsed-literal::

    'MALAY'



.. code:: ipython3

    multinomial.predict(malay_text,get_proba=True)




.. parsed-literal::

    {'OTHER': 0.0,
     'ENGLISH': 0.0,
     'INDONESIA': 7.866819388410703e-125,
     'MALAY': 1.0}



.. code:: ipython3

    multinomial.predict_batch([english_text,malay_text])




.. parsed-literal::

    ['ENGLISH', 'MALAY']



.. code:: ipython3

    multinomial.predict_batch([english_text,malay_text],get_proba=True)




.. parsed-literal::

    [{'OTHER': 1.6169333516662691e-38,
      'ENGLISH': 1.0,
      'INDONESIA': 1.392662138457825e-49,
      'MALAY': 5.0056770790612016e-36},
     {'OTHER': 0.0,
      'ENGLISH': 0.0,
      'INDONESIA': 7.866819388410703e-125,
      'MALAY': 1.0}]



Load SGD model
--------------

.. code:: ipython3

    sgd = malaya.language_detection.sgd()
    sgd.predict(chinese_text,get_proba=True)




.. parsed-literal::

    {'OTHER': 0.971323012260486,
     'ENGLISH': 0.0,
     'INDONESIA': 0.0,
     'MALAY': 0.028676987739513916}



.. code:: ipython3

    sgd.predict(english_text)




.. parsed-literal::

    'ENGLISH'



.. code:: ipython3

    sgd.predict(indon_text)




.. parsed-literal::

    'INDONESIA'



.. code:: ipython3

    sgd.predict(malay_text)




.. parsed-literal::

    'MALAY'



.. code:: ipython3

    sgd.predict(malay_text,get_proba=True)




.. parsed-literal::

    {'OTHER': 0.0, 'ENGLISH': 0.0, 'INDONESIA': 0.0, 'MALAY': 1.0}



.. code:: ipython3

    sgd.predict_batch([english_text,malay_text])




.. parsed-literal::

    ['ENGLISH', 'MALAY']



.. code:: ipython3

    sgd.predict_batch([english_text,malay_text],get_proba=True)




.. parsed-literal::

    [{'OTHER': 0.0, 'ENGLISH': 1.0, 'INDONESIA': 0.0, 'MALAY': 0.0},
     {'OTHER': 0.0, 'ENGLISH': 0.0, 'INDONESIA': 0.0, 'MALAY': 1.0}]



Load XGB model
--------------

.. code:: ipython3

    xgb = malaya.language_detection.xgb()
    xgb.predict(chinese_text)




.. parsed-literal::

    'OTHER'



.. code:: ipython3

    xgb.predict(indon_text,get_proba=True)




.. parsed-literal::

    {'OTHER': 4.8431886e-08,
     'ENGLISH': 4.20957e-15,
     'INDONESIA': 0.9999635,
     'MALAY': 3.639226e-05}



.. code:: ipython3

    xgb.predict_batch([indon_text,malay_text],get_proba=True)




.. parsed-literal::

    [{'OTHER': 4.8431886e-08,
      'ENGLISH': 4.20957e-15,
      'INDONESIA': 0.9999635,
      'MALAY': 3.639226e-05},
     {'OTHER': 4.8667624e-12,
      'ENGLISH': 5.9805926e-16,
      'INDONESIA': 4.418073e-06,
      'MALAY': 0.9999956}]


