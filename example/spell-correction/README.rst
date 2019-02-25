
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 9.95 s, sys: 751 ms, total: 10.7 s
    Wall time: 10.8 s


Load naive speller
------------------

.. code:: ipython3

    malays = malaya.load_malay_dictionary()
    corrector = malaya.spell.naive(malays)

.. code:: ipython3

    corrector.correct('mknn')


.. parsed-literal::

    [(('maun', False), 50), (('makin', False), 67), (('main', False), 50), (('mena', False), 50), (('min', False), 57), (('mini', False), 50), (('mana', False), 50), (('makanan', False), 73), (('makan', False), 67), (('menu', False), 50), (('makna', False), 67), (('mani', False), 50)] 
    




.. parsed-literal::

    'makanan'



List similar words
^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    corrector.correct('tmpat',debug=True)


.. parsed-literal::

    [(('tepat', False), 80), (('tempat', False), 91), (('tumpat', True), 91)] 
    




.. parsed-literal::

    'tempat'



Only pool based on first character
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%time
    corrector.correct('mknn',first_char=True)


.. parsed-literal::

    [(('maun', False), 50), (('makin', False), 67), (('main', False), 50), (('mena', False), 50), (('min', False), 57), (('mini', False), 50), (('mana', False), 50), (('makanan', False), 73), (('makan', False), 67), (('menu', False), 50), (('makna', False), 67), (('mani', False), 50)] 
    
    CPU times: user 544 ms, sys: 4.74 ms, total: 549 ms
    Wall time: 551 ms




.. parsed-literal::

    'makanan'



Pool on no condition
^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%time
    corrector.correct('mknn',first_char=False)


.. parsed-literal::

    [(('maun', False), 50), (('kan', False), 57), (('akan', False), 50), (('makin', False), 67), (('main', False), 50), (('mena', False), 50), (('min', False), 57), (('kon', False), 57), (('mini', False), 50), (('mana', False), 50), (('makanan', False), 73), (('ikon', False), 50), (('ken', False), 57), (('menu', False), 50), (('makna', False), 67), (('makan', False), 67), (('kun', False), 57), (('mani', False), 50), (('ikan', False), 50)] 
    
    CPU times: user 840 ms, sys: 5.32 ms, total: 845 ms
    Wall time: 850 ms




.. parsed-literal::

    'makanan'



.. code:: ipython3

    corrector.correct('tempat')


.. parsed-literal::

    [(('tempat', False), 100)] 
    




.. parsed-literal::

    'tempat'


