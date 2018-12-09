
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
    multinomial.predict(chinese_text)


.. parsed-literal::

    downloading LANGUAGE-DETECTION pickled bag-of-word multinomial


.. parsed-literal::

    46.0MB [00:16, 2.81MB/s]                          


.. parsed-literal::

    downloading LANGUAGE-DETECTION pickled multinomial model


.. parsed-literal::

    58.0MB [00:26, 2.15MB/s]                          




.. parsed-literal::

    'ENGLISH'



.. code:: ipython3

    multinomial.predict(english_text)




.. parsed-literal::

    'ENGLISH'



.. code:: ipython3

    multinomial.predict(indon_text)




.. parsed-literal::

    'MALAY'



.. code:: ipython3

    multinomial.predict(malay_text)




.. parsed-literal::

    'MALAY'



.. code:: ipython3

    multinomial.predict(malay_text,get_proba=True)




.. parsed-literal::

    {'OTHER': 0.0,
     'ENGLISH': 0.0,
     'INDONESIA': 1.485952831042105e-173,
     'MALAY': 1.0}



.. code:: ipython3

    multinomial.predict_batch([english_text,malay_text])




.. parsed-literal::

    ['ENGLISH', 'MALAY']



.. code:: ipython3

    multinomial.predict_batch([english_text,malay_text],get_proba=True)




.. parsed-literal::

    [{'OTHER': 5.0953089622773946e-58,
      'ENGLISH': 1.0,
      'INDONESIA': 3.1682621618878156e-60,
      'MALAY': 4.1605996684502836e-54},
     {'OTHER': 0.0,
      'ENGLISH': 0.0,
      'INDONESIA': 1.485952831042105e-173,
      'MALAY': 1.0}]



Load XGB model
--------------

.. code:: ipython3

    xgb = malaya.xgb_detect_languages()
    xgb.predict(chinese_text)


.. parsed-literal::

      0%|          | 0.00/37.8 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading LANGUAGE-DETECTION pickled bag-of-word XGB


.. parsed-literal::

    38.0MB [00:16, 3.37MB/s]                          


.. parsed-literal::

    downloading LANGUAGE-DETECTION pickled XGB model


.. parsed-literal::

    22.0MB [00:06, 3.81MB/s]                          




.. parsed-literal::

    'OTHER'



.. code:: ipython3

    xgb.predict(indon_text,get_proba=True)




.. parsed-literal::

    {'OTHER': 6.92337e-10,
     'ENGLISH': 3.507782e-11,
     'INDONESIA': 0.9995041,
     'MALAY': 0.0004959471}



.. code:: ipython3

    xgb.predict_batch([indon_text,malay_text],get_proba=True)




.. parsed-literal::

    [{'OTHER': 6.92337e-10,
      'ENGLISH': 3.507782e-11,
      'INDONESIA': 0.9995041,
      'MALAY': 0.0004959471},
     {'OTHER': 1.174448e-09,
      'ENGLISH': 1.4715874e-10,
      'INDONESIA': 0.001421933,
      'MALAY': 0.9985781}]


