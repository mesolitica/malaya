
.. code:: ipython3

    import malaya

List available language detected
--------------------------------

.. code:: ipython3

    malaya.get_language_labels()




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

    multinomial = malaya.multinomial_detect_languages()
    multinomial.predict(chinese_text,get_proba=True)




.. parsed-literal::

    {'OTHER': 1.0,
     'ENGLISH': 2.157849898017918e-22,
     'INDONESIA': 4.2440922283612186e-30,
     'MALAY': 1.191161632678076e-41}



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
     'INDONESIA': 1.2874523558561307e-52,
     'MALAY': 1.0}



.. code:: ipython3

    multinomial.predict_batch([english_text,malay_text])




.. parsed-literal::

    ['ENGLISH', 'MALAY']



.. code:: ipython3

    multinomial.predict_batch([english_text,malay_text],get_proba=True)




.. parsed-literal::

    [{'OTHER': 1.807742600646247e-74,
      'ENGLISH': 1.0,
      'INDONESIA': 7.503596159299667e-77,
      'MALAY': 1.4742530879417279e-58},
     {'OTHER': 0.0,
      'ENGLISH': 0.0,
      'INDONESIA': 1.2874523558561307e-52,
      'MALAY': 1.0}]



Load SGD model
--------------

.. code:: ipython3

    sgd = malaya.sgd_detect_languages()
    sgd.predict(chinese_text,get_proba=True)




.. parsed-literal::

    {'OTHER': 1.0, 'ENGLISH': 0.0, 'INDONESIA': 0.0, 'MALAY': 0.0}



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

    xgb = malaya.xgb_detect_languages()
    xgb.predict(chinese_text)




.. parsed-literal::

    'OTHER'



.. code:: ipython3

    xgb.predict(indon_text,get_proba=True)




.. parsed-literal::

    {'OTHER': 1.980007e-07,
     'ENGLISH': 8.863334e-08,
     'INDONESIA': 0.8836274,
     'MALAY': 0.116372354}



.. code:: ipython3

    xgb.predict_batch([indon_text,malay_text],get_proba=True)




.. parsed-literal::

    [{'OTHER': 1.980007e-07,
      'ENGLISH': 8.863334e-08,
      'INDONESIA': 0.8836274,
      'MALAY': 0.116372354},
     {'OTHER': 4.3554013e-10,
      'ENGLISH': 3.5299177e-10,
      'INDONESIA': 0.00014907354,
      'MALAY': 0.99985087}]


