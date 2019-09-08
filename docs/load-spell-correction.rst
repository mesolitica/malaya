
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 5.95 s, sys: 1.61 s, total: 7.56 s
    Wall time: 12.7 s


.. code:: python

    # some text examples copied from Twitter

    string1 = 'Kerajaan patut bagi pencen awal skit kpd warga emas supaya emosi'
    string2 = 'Husein ska mkn aym dkat kmpng Jawa'
    string3 = 'Melayu malas ni narration dia sama je macam men are trash. True to some, false to some.'
    string4 = 'Tapi tak pikir ke bahaya perpetuate myths camtu. Nanti kalau ada hiring discrimination despite your good qualifications because of your race tau pulak marah. Your kids will be victims of that too.'
    string5 = 'DrM cerita Melayu malas semenjak saya kat University (early 1980s) and now as i am edging towards retirement in 4-5 years time after a career of being an Engineer, Project Manager, General Manager'

Load probability speller
------------------------

.. code:: python

    prob_corrector = malaya.spell.probability()

To correct a word
^^^^^^^^^^^^^^^^^

.. code:: python

    prob_corrector.correct('mkn')




.. parsed-literal::

    'makan'



To correct a sentence
^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    prob_corrector.correct_text(string1)




.. parsed-literal::

    'Kerajaan patut bagi pencen awal sakit kepada warga emas supaya emosi'



.. code:: python

    prob_corrector.correct_text(string2)




.. parsed-literal::

    'Husein suka makan ayam dekat kampung Jawa'



.. code:: python

    prob_corrector.correct_text(string3)




.. parsed-literal::

    'Melayu malas ni narration dia sama sahaja macam men are trash. True to some, false to some.'



Load distance-based speller
---------------------------

Default is use JaroWrinkler algorithm, common algorithm to fix a typo,
can check implementation in
`malaya.texts._distance <https://github.com/huseinzol05/Malaya/blob/master/malaya/texts/_distance.py>`__

To use another string algorithm, just replace parameter ``distancer``
with algorithm object.

.. code:: python


   def distance(distancer = JaroWinkler, validate = True):
       """
       Train a String matching Spell Corrector.

       Parameters
       ----------
       distancer: object
           string matching object, default is malaya.texts._distance.JaroWinkler
       validate: bool, optional (default=True)
           if True, malaya will check model availability and download if not available.

       Returns
       -------
       _SPELL: Trained malaya.spell._SPELL class
       """


.. code:: python

    distance_corrector = malaya.spell.distance()

To correct a word
^^^^^^^^^^^^^^^^^

.. code:: python

    distance_corrector.correct('mkn')




.. parsed-literal::

    'mkpn'



To correct a sentence
^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    distance_corrector.correct_text(string1)




.. parsed-literal::

    'Kerajaan patut bagi pencen awal sakit kepada warga emas supaya emosi'



.. code:: python

    distance_corrector.correct_text(string2)




.. parsed-literal::

    'Husein skaf mkpn ayam dekat kumpang Jawa'



.. code:: python

    distance_corrector.correct_text(string3)




.. parsed-literal::

    'Melayu malas ni narration dia sama sahaja macam men are trash. True to some, false to some.'
