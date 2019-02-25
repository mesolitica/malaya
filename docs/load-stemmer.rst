
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.1 s, sys: 1.36 s, total: 13.4 s
    Wall time: 17 s


Use Sastrawi stemmer
--------------------

.. code:: python

    malaya.stem.sastrawi('saya tengah berjalan')




.. parsed-literal::

    'saya tengah jalan'



.. code:: python

    malaya.stem.sastrawi('saya tengah berjalankan sangat-sangat')




.. parsed-literal::

    'saya tengah jalan sangat'



.. code:: python

    malaya.stem.sastrawi('menarik')




.. parsed-literal::

    'tarik'



List available deep learning stemming models
--------------------------------------------

.. code:: python

    malaya.stem.available_deep_model()




.. parsed-literal::

    ['lstm', 'bahdanau', 'luong']



Load deep learning model
------------------------

.. code:: python

    stemmer = malaya.stem.deep_model('bahdanau')
    stemmer.stem('saya sangat sukakan awak tetapi awak sangatlah sakai')




.. parsed-literal::

    'saya sangat suka awak tetap awak sangat saka'



.. code:: python

    stemmer = malaya.stem.deep_model('luong')
    stemmer.stem('saya sangat sukakan awak tetapi awak sangatlah sakai')




.. parsed-literal::

    'saya sangat suka awak tetap awak sangat saka'



.. code:: python

    stemmer = malaya.stem.deep_model('lstm')
    stemmer.stem('saya sangat sukakan awak tetapi awak sangatlah sakai')




.. parsed-literal::

    'saya sangat suka awak tetapi awak sangat sakai'



.. code:: python

    malaya.stem.sastrawi('saya sangat sukakan awak tetapi awak sangatlah sakai')




.. parsed-literal::

    'saya sangat suka awak tetapi awak sangat sakai'
