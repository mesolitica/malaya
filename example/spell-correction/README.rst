
.. code:: ipython3

    import malaya

Load naive speller
------------------

.. code:: ipython3

    malays = malaya.load_malay_dictionary()
    corrector = malaya.naive_speller(malays)

.. code:: ipython3

    corrector.correct('mknn')


.. parsed-literal::

    [(('mini', False), 50), (('makan', False), 67), (('mana', False), 50), (('min', False), 57), (('makna', False), 67), (('makin', False), 67), (('menu', False), 50), (('maun', False), 50), (('mani', False), 50), (('main', False), 50), (('mena', False), 50), (('makanan', False), 73)] 
    




.. parsed-literal::

    'makanan'



List similar words
^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    corrector.correct('tmpat',debug=True)


.. parsed-literal::

    [(('tumpat', True), 91), (('tepat', False), 80), (('tempat', False), 91)] 
    




.. parsed-literal::

    'tempat'



Only pool based on first character
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%time
    corrector.correct('mknn',first_char=True)


.. parsed-literal::

    [(('mini', False), 50), (('makan', False), 67), (('mana', False), 50), (('min', False), 57), (('makna', False), 67), (('makin', False), 67), (('menu', False), 50), (('maun', False), 50), (('mani', False), 50), (('main', False), 50), (('mena', False), 50), (('makanan', False), 73)] 
    
    CPU times: user 272 ms, sys: 2.83 ms, total: 274 ms
    Wall time: 292 ms




.. parsed-literal::

    'makanan'



Pool on no condition
^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%time
    corrector.correct('mknn',first_char=False)


.. parsed-literal::

    [(('mini', False), 50), (('kon', False), 57), (('makan', False), 67), (('ikan', False), 50), (('mana', False), 50), (('min', False), 57), (('makna', False), 67), (('kun', False), 57), (('makin', False), 67), (('menu', False), 50), (('akan', False), 50), (('mani', False), 50), (('main', False), 50), (('mena', False), 50), (('ikon', False), 50), (('kan', False), 57), (('ken', False), 57), (('makanan', False), 73), (('maun', False), 50)] 
    
    CPU times: user 407 ms, sys: 3.95 ms, total: 411 ms
    Wall time: 417 ms




.. parsed-literal::

    'makanan'



.. code:: ipython3

    corrector.correct('tempat')


.. parsed-literal::

    [(('tempat', False), 100)] 
    




.. parsed-literal::

    'tempat'


