
.. code:: python

    import malaya

.. code:: python

    string = 'y u xsuka makan HUSEIN kt situ tmpt'
    another = 'i mmg xska mknn HUSEIN kampng tempt'

Load basic normalizer
---------------------

.. code:: python

    malaya.basic_normalizer(string)




.. parsed-literal::

    'kenapa awak xsuka makan Husein kt situ tmpt'



Load fuzzy normalizer
---------------------

.. code:: python

    malays = malaya.load_malay_dictionary()
    normalizer = malaya.fuzzy_normalizer(malays)

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

    normalizer = malaya.spell_normalizer(malays)

To list all selected words during normalize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize(string,debug=True)


.. parsed-literal::

    downloading Topics, Influencers, Location data


.. parsed-literal::

    1.00MB [00:00, 12.9MB/s]


.. parsed-literal::

    [(('tempo', False), 15), (('impi', False), 15), (('amit', False), 21), (('tampu', False), 15), (('tampi', True), 15), (('tut', False), 16), (('ampit', False), 20), (('tipu', False), 15), (('tuat', False), 15), (('tumpu', False), 20), (('timpa', False), 20), (('umpat', True), 20), (('tampa', False), 15), (('topi', False), 10), (('empat', True), 15), (('tempa', False), 15), (('tip', False), 16), (('tapa', False), 10), (('top', False), 11), (('tepat', False), 15), (('tapi', False), 10), (('tamat', False), 20), (('emat', False), 15), (('tepu', False), 10), (('taut', False), 15), (('ampo', False), 15), (('mat', False), 16), (('ampu', False), 15), (('temut', False), 15), (('tempat', False), 20), (('tepet', False), 15), (('tuit', False), 15), (('umat', False), 21), (('amat', False), 21), (('ampe', False), 15), (('empu', False), 10), (('tempe', False), 15), (('tumpat', True), 24), (('umut', False), 21), (('taat', False), 15), (('tepi', False), 10), (('tat', False), 11), (('amput', False), 20), (('tumit', False), 20)]





.. parsed-literal::

    'kenapa awak tak suka makan Husein kat situ amit'



Load deep learning
------------------

This model is not perfect, really suggest you to use other models

.. code:: python

    normalizer = malaya.deep_normalizer()
    normalizer.normalize(string)


.. parsed-literal::

    1.00MB [00:00, 93.1MB/s]

.. parsed-literal::

    downloading JSON normalizer
    downloading normalizer graph


.. parsed-literal::


    22.0MB [00:07, 4.30MB/s]




.. parsed-literal::

    'eye uau tak suka makan unsein kati situ tumpat'
