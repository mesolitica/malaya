
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.1 s, sys: 1.36 s, total: 13.4 s
    Wall time: 17 s


Use Sastrawi stemmer
--------------------

.. code:: ipython3

    malaya.stem.sastrawi('saya tengah berjalan')




.. parsed-literal::

    'saya tengah jalan'



.. code:: ipython3

    malaya.stem.sastrawi('saya tengah berjalankan sangat-sangat')




.. parsed-literal::

    'saya tengah jalan sangat'



.. code:: ipython3

    malaya.stem.sastrawi('menarik')




.. parsed-literal::

    'tarik'



List available deep learning stemming models
--------------------------------------------

.. code:: ipython3

    malaya.stem.available_deep_model()




.. parsed-literal::

    ['lstm', 'bahdanau', 'luong']



Load deep learning model
------------------------

.. code:: ipython3

    stemmer = malaya.stem.deep_model('bahdanau')
    stemmer.stem('saya sangat sukakan awak tetapi awak sangatlah sakai')




.. parsed-literal::

    'saya sangat suka awak tetap awak sangat saka'



.. code:: ipython3

    stemmer = malaya.stem.deep_model('luong')
    stemmer.stem('saya sangat sukakan awak tetapi awak sangatlah sakai')




.. parsed-literal::

    'saya sangat suka awak tetap awak sangat saka'



.. code:: ipython3

    stemmer = malaya.stem.deep_model('lstm')
    stemmer.stem('saya sangat sukakan awak tetapi awak sangatlah sakai')




.. parsed-literal::

    'saya sangat suka awak tetapi awak sangat sakai'



.. code:: ipython3

    malaya.stem.sastrawi('saya sangat sukakan awak tetapi awak sangatlah sakai')




.. parsed-literal::

    'saya sangat suka awak tetapi awak sangat sakai'


