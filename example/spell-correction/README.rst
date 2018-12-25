
.. code:: ipython3

    import malaya

Load naive speller
------------------

.. code:: ipython3

    malays = malaya.load_malay_dictionary()
    corrector = malaya.spell.naive(malays)

.. code:: ipython3

    corrector.correct('mknn')


.. parsed-literal::

    [(('maun', False), 50), (('makan', False), 67), (('mana', False), 50), (('main', False), 50), (('makna', False), 67), (('menu', False), 50), (('mani', False), 50), (('makin', False), 67), (('min', False), 57), (('mini', False), 50), (('mena', False), 50), (('makanan', False), 73)] 
    




.. parsed-literal::

    'makanan'



List similar words
^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    corrector.correct('tmpat',debug=True)


.. parsed-literal::

    [(('tempat', False), 91), (('tumpat', True), 91), (('tepat', False), 80)] 
    




.. parsed-literal::

    'tempat'



Only pool based on first character
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%time
    corrector.correct('mknn',first_char=True)


.. parsed-literal::

    [(('maun', False), 50), (('makan', False), 67), (('mana', False), 50), (('main', False), 50), (('makna', False), 67), (('menu', False), 50), (('mani', False), 50), (('makin', False), 67), (('min', False), 57), (('mini', False), 50), (('mena', False), 50), (('makanan', False), 73)] 
    
    CPU times: user 276 ms, sys: 4.43 ms, total: 281 ms
    Wall time: 281 ms




.. parsed-literal::

    'makanan'



Pool on no condition
^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%time
    corrector.correct('mknn',first_char=False)


.. parsed-literal::

    [(('maun', False), 50), (('kan', False), 57), (('makan', False), 67), (('main', False), 50), (('mana', False), 50), (('makna', False), 67), (('mena', False), 50), (('menu', False), 50), (('mani', False), 50), (('ikon', False), 50), (('makin', False), 67), (('min', False), 57), (('akan', False), 50), (('ken', False), 57), (('kun', False), 57), (('mini', False), 50), (('kon', False), 57), (('ikan', False), 50), (('makanan', False), 73)] 
    
    CPU times: user 411 ms, sys: 4.14 ms, total: 415 ms
    Wall time: 415 ms




.. parsed-literal::

    'makanan'



.. code:: ipython3

    corrector.correct('tempat')


.. parsed-literal::

    [(('tempat', False), 100)] 
    




.. parsed-literal::

    'tempat'


