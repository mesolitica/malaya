
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 11.4 s, sys: 1.54 s, total: 12.9 s
    Wall time: 16.7 s


.. code:: python

    string = 'xjdi ke, y u xsuke makan HUSEIN kt situ tmpt, i hate it. pelikle'
    another = 'i mmg xske mknn HUSEIN kampng tempt, i love them. pelikle'

Load basic normalizer
---------------------

.. code:: python

    malaya.normalize.basic(string)




.. parsed-literal::

    'xjdi ke kenapa awak xsuke makan Husein kt situ tmpt i hate it pelikle'



Load fuzzy normalizer
---------------------

.. code:: python

    malays = malaya.load_malay_dictionary()
    normalizer = malaya.normalize.fuzzy(malays)

.. code:: python

    normalizer.normalize(string)




.. parsed-literal::

    'tak jadi ke kenapa awak tak suka makan Husein kat situ tempat saya hate it pelik lah'



.. code:: python

    normalizer.normalize(another)




.. parsed-literal::

    'saya memang tak saka makanan Husein kampung tempt saya love them pelik lah'



Load spell normalizer
---------------------

.. code:: python

    normalizer = malaya.normalize.spell(malays)

To list all selected words during normalize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize(string,debug=True)


.. parsed-literal::

    [(('judi', False), 86), (('adi', False), 67), (('di', False), 80), (('jadi', False), 86)]

    [(('tepu', False), 50), (('amput', False), 67), (('tamat', False), 67), (('empat', True), 67), (('tumit', False), 67), (('ampe', False), 50), (('tipu', False), 50), (('tat', False), 57), (('top', False), 57), (('tampu', False), 67), (('topi', False), 50), (('tepi', False), 50), (('tempat', False), 80), (('umut', False), 50), (('ampo', False), 50), (('timpa', False), 67), (('impi', False), 50), (('tempe', False), 67), (('tapa', False), 50), (('taat', False), 50), (('tepet', False), 67), (('umat', False), 50), (('tepat', False), 67), (('tut', False), 57), (('tumpat', True), 80), (('tuat', False), 50), (('tampi', True), 67), (('umpat', True), 67), (('temut', False), 67), (('emat', False), 50), (('ampit', False), 67), (('amit', False), 50), (('tempo', False), 67), (('tumpu', False), 67), (('tempa', False), 67), (('empu', False), 50), (('amat', False), 50), (('taut', False), 50), (('mat', False), 57), (('tampa', False), 67), (('tuit', False), 50), (('tip', False), 57), (('ampu', False), 50), (('tapi', False), 50)]





.. parsed-literal::

    'tak jadi ke kenapa awak tak suka makan Husein kat situ tempat saya hate it pelik lah'



List available deep learning stemming models
--------------------------------------------

.. code:: python

    malaya.normalize.available_deep_model()




.. parsed-literal::

    ['lstm', 'bahdanau', 'luong']



Load deep learning
------------------

We experimenting a lot for ``seq2seq`` models, we try to do the best
normalizer deep learning models.

.. code:: python

    normalizer = malaya.normalize.deep_model(malays, 'bahdanau')
    print(normalizer.normalize(string))
    normalizer.normalize(another)


.. parsed-literal::

    jidiomik ke kenapa awak sukeesi makan Husein kat situ tempatmo saya hate it pelik lah




.. parsed-literal::

    'saya memang sikeuoi maknnkano Husein kampanga tempt saya love them pelik lah'



.. code:: python

    normalizer = malaya.normalize.deep_model(malays, 'luong')
    print(normalizer.normalize(string))
    normalizer.normalize(another)


.. parsed-literal::

    jadidilox ke kenapa awak sokeled makan Husein kat situ tampatgllah saya hate it pelik lah




.. parsed-literal::

    'saya memang skeflleh makafnnloh Husein kampangja tempt saya love them pelik lah'



.. code:: python

    normalizer = malaya.normalize.deep_model(malays, 'lstm')
    print(normalizer.normalize(string))
    normalizer.normalize(another)


.. parsed-literal::

    jajiodi ke kenapa awak sukeeia makan Husein kat situ tempatwa saya hate it pelik lah




.. parsed-literal::

    'saya memang sekeoia makankari Husein kampangi tempt saya love them pelik lah'
