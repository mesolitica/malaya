
.. code:: python

    import malaya

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



Load deep learning model
------------------------

I really not suggest you to use this model. Use Sastrawi instead. We are
adding our own rules into Sastrawi stemmer

.. code:: python

    stemmer = malaya.stem.deep_model()

.. code:: python

    stemmer.stem('saya tengah berjalankan sangat-sangat')




.. parsed-literal::

    'saya tengah jalan sangat sangat'



.. code:: python

    stemmer.stem('saya sangat sukakan awak')




.. parsed-literal::

    'saya sangat suka awak'



.. code:: python

    stemmer.stem('saya sangat suakkan awak')




.. parsed-literal::

    'saya sangat suak awak'
