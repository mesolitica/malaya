
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 13.6 s, sys: 1.56 s, total: 15.2 s
    Wall time: 20.3 s


.. code:: ipython3

    string = 'y u xsuke makan HUSEIN kt situ tmpt'
    another = 'i mmg xske mknn HUSEIN kampng tempt'

Load basic normalizer
---------------------

.. code:: ipython3

    malaya.normalize.basic(string)




.. parsed-literal::

    'kenapa awak xsuke makan Husein kt situ tmpt'



Load fuzzy normalizer
---------------------

.. code:: ipython3

    malays = malaya.load_malay_dictionary()
    normalizer = malaya.normalize.fuzzy(malays)

.. code:: ipython3

    normalizer.normalize(string)




.. parsed-literal::

    'kenapa awak tak suka makan Husein kat situ tempat'



.. code:: ipython3

    normalizer.normalize(another)




.. parsed-literal::

    'saya memang tak saka makanan Husein kampung tempat'



Load spell normalizer
---------------------

.. code:: ipython3

    normalizer = malaya.normalize.spell(malays)

To list all selected words during normalize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize(string,debug=True)


.. parsed-literal::

    [(('umut', False), 50), (('amit', False), 50), (('tepi', False), 50), (('tuit', False), 50), (('tempat', False), 80), (('tampa', False), 67), (('umpat', True), 67), (('taut', False), 50), (('amput', False), 67), (('tipu', False), 50), (('tapa', False), 50), (('temut', False), 67), (('mat', False), 57), (('empu', False), 50), (('tuat', False), 50), (('tempo', False), 67), (('tampi', True), 67), (('tepu', False), 50), (('tempe', False), 67), (('tampu', False), 67), (('ampo', False), 50), (('tut', False), 57), (('impi', False), 50), (('ampit', False), 67), (('tapi', False), 50), (('ampe', False), 50), (('tepat', False), 67), (('tumit', False), 67), (('ampu', False), 50), (('tumpu', False), 67), (('tamat', False), 67), (('tepet', False), 67), (('tempa', False), 67), (('tat', False), 57), (('amat', False), 50), (('emat', False), 50), (('umat', False), 50), (('tumpat', True), 80), (('tip', False), 57), (('empat', True), 67), (('taat', False), 50), (('timpa', False), 67), (('top', False), 57), (('topi', False), 50)] 
    




.. parsed-literal::

    'kenapa awak tak suka makan Husein kat situ tempat'



Load deep learning
------------------

**This model is not perfect, really suggest you to use other models.
Husein needs to read more!**

.. code:: ipython3

    normalizer = malaya.normalize.deep_model()
    normalizer.normalize(string)




.. parsed-literal::

    'eye uau tak suke makan unsein kati situ tumpat'


