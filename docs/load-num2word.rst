
.. code:: ipython3

    import malaya


.. parsed-literal::

    1.00MB [00:00, 734MB/s]                    

.. parsed-literal::

    downloading stopwords


.. parsed-literal::

    
    Using TensorFlow backend.
    /usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    1.00MB [00:00, 39.9MB/s]                   


.. parsed-literal::

    downloading ZIP rules-based


.. code:: ipython3

    malaya.to_cardinal(123456789)




.. parsed-literal::

    'seratus dua puluh tiga juta empat ratus lima puluh enam ribu tujuh ratus lapan puluh sembilan'



.. code:: ipython3

    malaya.to_cardinal(10)




.. parsed-literal::

    'sepuluh'



.. code:: ipython3

    malaya.to_cardinal(12)




.. parsed-literal::

    'dua belas'



.. code:: ipython3

    malaya.to_ordinal(1)




.. parsed-literal::

    'pertama'



.. code:: ipython3

    malaya.to_cardinal(1)




.. parsed-literal::

    'satu'



.. code:: ipython3

    malaya.to_ordinal(10)




.. parsed-literal::

    'kesepuluh'



.. code:: ipython3

    malaya.to_ordinal(12)




.. parsed-literal::

    'kedua belas'



.. code:: ipython3

    malaya.to_cardinal(-123456789)




.. parsed-literal::

    'negatif seratus dua puluh tiga juta empat ratus lima puluh enam ribu tujuh ratus lapan puluh sembilan'



.. code:: ipython3

    malaya.to_cardinal(-1234567.89)




.. parsed-literal::

    'negatif satu juta dua ratus tiga puluh empat ribu lima ratus enam puluh tujuh perpuluhan lapan sembilan'



.. code:: ipython3

    malaya.to_ordinal(11)




.. parsed-literal::

    'kesebelas'


