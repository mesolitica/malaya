NSFW Detection
==============

.. container:: alert alert-info

   This tutorial is available as an IPython notebook at
   `Malaya/example/nsfw <https://github.com/huseinzol05/Malaya/tree/master/example/nsfw>`__.

Pretty simple and straightforward, just to detect whether a text is NSFW
or not.

.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 4.05 s, sys: 741 ms, total: 4.79 s
    Wall time: 4.59 s


Get label
~~~~~~~~~

.. code:: ipython3

    malaya.nsfw.label




.. parsed-literal::

    ['sex', 'gambling', 'negative']



Load lexicon model
~~~~~~~~~~~~~~~~~~

Pretty naive but really effective, lexicon gathered at
`Malay-Dataset/corpus/nsfw <https://github.com/huseinzol05/Malay-Dataset/tree/master/corpus/nsfw>`__.

Lexicon model only has,

.. code:: python

   model.predict(List[str])

.. code:: ipython3

    lexicon_model = malaya.nsfw.lexicon()

.. code:: ipython3

    string1 = 'xxx sgt panas, best weh'
    string2 = 'jmpa dekat kl sentral'
    string3 = 'Rolet Dengan Wang Sebenar'

.. code:: ipython3

    lexicon_model.predict([string1, string2, string3])




.. parsed-literal::

    ['sex', 'negative', 'gambling']



Load multinomial model
~~~~~~~~~~~~~~~~~~~~~~

All model interface will follow sklearn interface started v3.4,

.. code:: python

   model.predict(List[str])

   model.predict_proba(List[str])

.. code:: ipython3

    model = malaya.nsfw.multinomial()

.. code:: ipython3

    model.predict([string1, string2, string3])




.. parsed-literal::

    ['sex', 'negative', 'gambling']



.. code:: ipython3

    model.predict_proba([string1, string2, string3])




.. parsed-literal::

    [{'sex': 0.9357058034930408,
      'gambling': 0.02616353532998711,
      'negative': 0.03813066117697173},
     {'sex': 0.027541900360621846,
      'gambling': 0.03522626245360637,
      'negative': 0.9372318371857732},
     {'sex': 0.01865380888750343,
      'gambling': 0.9765340760395791,
      'negative': 0.004812115072918792}]


