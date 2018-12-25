
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 11.4 s, sys: 798 ms, total: 12.2 s
    Wall time: 12.4 s


.. code:: python

    string = 'y u xsuka makan HUSEIN kt situ tmpt'
    another = 'i mmg xska mknn HUSEIN kampng tempt'

Load basic normalizer
---------------------

.. code:: python

    malaya.normalize.basic(string)




.. parsed-literal::

    'kenapa awak xsuka makan Husein kt situ tmpt'



Load fuzzy normalizer
---------------------

.. code:: python

    malays = malaya.load_malay_dictionary()
    normalizer = malaya.normalize.fuzzy(malays)

.. code:: python

    normalizer.normalize(string)




.. parsed-literal::

    'kenapa awak tak suka makan Husein kat situ tempat'



.. code:: python

    normalizer.normalize(another)




.. parsed-literal::

    'saya memang tak saka makanan Husein kampung tempat'



Load spell normalizer
---------------------

.. code:: python

    normalizer = malaya.normalize.spell(malays)

To list all selected words during normalize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize(string,debug=True)


.. parsed-literal::

    [(('ampit', False), 20), (('tempat', False), 20), (('tempo', False), 15), (('tat', False), 11), (('tapa', False), 10), (('ampu', False), 15), (('impi', False), 15), (('amput', False), 20), (('taut', False), 15), (('tuat', False), 15), (('ampe', False), 15), (('top', False), 11), (('umut', False), 21), (('tampu', False), 15), (('timpa', False), 20), (('temut', False), 15), (('tut', False), 16), (('empat', True), 15), (('tumit', False), 20), (('amit', False), 21), (('topi', False), 10), (('umpat', True), 20), (('tepi', False), 10), (('tumpat', True), 24), (('umat', False), 21), (('ampo', False), 15), (('tepu', False), 10), (('tipu', False), 15), (('empu', False), 10), (('tip', False), 16), (('tempa', False), 15), (('tuit', False), 15), (('tampa', False), 15), (('tepet', False), 15), (('emat', False), 15), (('tamat', False), 20), (('taat', False), 15), (('amat', False), 21), (('tampi', True), 15), (('tapi', False), 10), (('tempe', False), 15), (('mat', False), 16), (('tumpu', False), 20), (('tepat', False), 15)]





.. parsed-literal::

    'kenapa awak tak suka makan Husein kat situ umat'



Load deep learning
------------------

This model is not perfect, really suggest you to use other models

.. code:: python

    normalizer = malaya.normalize.deep_model()
    normalizer.normalize(string)




.. parsed-literal::

    'eye uau tak suka makan unsein kati situ tumpat'
