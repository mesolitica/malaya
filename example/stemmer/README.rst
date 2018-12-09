
.. code:: ipython3

    import malaya

Use Sastrawi stemmer
--------------------

.. code:: ipython3

    malaya.sastrawi_stemmer('saya tengah berjalan')




.. parsed-literal::

    'saya tengah jalan'



.. code:: ipython3

    malaya.sastrawi_stemmer('saya tengah berjalankan sangat-sangat')




.. parsed-literal::

    'saya tengah jalan sangat'



.. code:: ipython3

    malaya.sastrawi_stemmer('menarik')




.. parsed-literal::

    'tarik'



Load deep learning model
------------------------

I really not suggest you to use this model. Use Sastrawi instead. We are
adding our own rules into Sastrawi stemmer

.. code:: ipython3

    stemmer = malaya.deep_stemmer()


.. parsed-literal::

    downloading JSON stemmer


.. parsed-literal::

    1.00MB [00:00, 260MB/s]                    
      0%|          | 0.00/21.4 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading stemmer graph


.. parsed-literal::

    22.0MB [00:06, 3.92MB/s]                          


.. code:: ipython3

    stemmer.stem('saya tengah berjalankan sangat-sangat')




.. parsed-literal::

    'saya tengah jalan sangat sangat'



.. code:: ipython3

    stemmer.stem('saya sangat sukakan awak')




.. parsed-literal::

    'saya sangat suka awak'



.. code:: ipython3

    stemmer.stem('saya sangat suakkan awak')




.. parsed-literal::

    'saya sangat suak awak'


