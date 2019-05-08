
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 11.9 s, sys: 1.55 s, total: 13.5 s
    Wall time: 17.7 s


Load fuzzy speller
------------------

.. code:: ipython3

    malays = malaya.load_malay_dictionary()
    corrector = malaya.spell.fuzzy(malays)

.. code:: ipython3

    corrector.correct('mknn')


.. parsed-literal::

    [(('makanan', False), 73)] 
    




.. parsed-literal::

    'makanan'



List similar words
^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    corrector.correct('tmpat',debug=True)


.. parsed-literal::

    [(('tepat', False), 80), (('tempat', False), 91), (('tumpat', True), 91)] 
    




.. parsed-literal::

    'tempat'



Only pool based on first character
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%time
    corrector.correct('mknn',first_char=True)


.. parsed-literal::

    [(('makanan', False), 73)] 
    
    CPU times: user 22.4 ms, sys: 1.46 ms, total: 23.9 ms
    Wall time: 24.5 ms




.. parsed-literal::

    'makanan'



Pool on no condition
^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%time
    corrector.correct('mknn',first_char=False)


.. parsed-literal::

    [(('makanan', False), 73)] 
    
    CPU times: user 27.4 ms, sys: 2.68 ms, total: 30 ms
    Wall time: 32.1 ms




.. parsed-literal::

    'makanan'



.. code:: ipython3

    corrector.correct('tempat')


.. parsed-literal::

    [(('tempat', False), 100)] 
    




.. parsed-literal::

    'tempat'



Load probability speller
------------------------

.. code:: ipython3

    corrector = malaya.spell.probability()

.. code:: ipython3

    %%time
    corrector.correct('mknn')


.. parsed-literal::

    CPU times: user 240 µs, sys: 17 µs, total: 257 µs
    Wall time: 263 µs




.. parsed-literal::

    'makanan'



Fast-mode
^^^^^^^^^

You can use fast mode, means we only search word pools from augmented
user input.

.. code:: ipython3

    %%time
    corrector.correct('mknn',fast = True)


.. parsed-literal::

    CPU times: user 8 µs, sys: 1e+03 ns, total: 9 µs
    Wall time: 14.1 µs




.. parsed-literal::

    'makanan'



If want more pool size, simply ``fast = False``, default is ``False``.

.. code:: ipython3

    %%time
    corrector.correct('tmpat')


.. parsed-literal::

    CPU times: user 424 µs, sys: 6 µs, total: 430 µs
    Wall time: 438 µs




.. parsed-literal::

    'tempat'



Assume wrong
^^^^^^^^^^^^

Sometime words inside the dictionary is not really correct, like example
below,

.. code:: ipython3

    corrector.correct('saye')




.. parsed-literal::

    'saye'



We got ``saye`` as output, because the algorithm assumed all words
inside the dictionary is correct.

So to solve this, you need to set ``assume_wrong`` parameter as
``False``.

.. code:: ipython3

    corrector.correct('saye', assume_wrong = True)




.. parsed-literal::

    'saya'



edit_step
^^^^^^^^^

You can check how augmented word been done, by simply
``corrector.edit_step``

.. code:: ipython3

    corrector.edit_step('mknn')




.. parsed-literal::

    {'aknn',
     'amknn',
     'bknn',
     'bmknn',
     'cknn',
     'cmknn',
     'dknn',
     'dmknn',
     'eknn',
     'emknn',
     'fknn',
     'fmknn',
     'gknn',
     'gmknn',
     'hknn',
     'hmknn',
     'iknn',
     'imknn',
     'jknn',
     'jmknn',
     'kknn',
     'kmknn',
     'kmnn',
     'knn',
     'lknn',
     'lmknn',
     'makanan',
     'maknn',
     'mann',
     'mbknn',
     'mbnn',
     'mcknn',
     'mcnn',
     'mdknn',
     'mdnn',
     'meknn',
     'menn',
     'mfknn',
     'mfnn',
     'mgknn',
     'mgnn',
     'mhknn',
     'mhnn',
     'mikinin',
     'miknn',
     'minn',
     'mjknn',
     'mjnn',
     'mkan',
     'mkann',
     'mkbn',
     'mkbnn',
     'mkcn',
     'mkcnn',
     'mkdn',
     'mkdnn',
     'mken',
     'mkenn',
     'mkfn',
     'mkfnn',
     'mkgn',
     'mkgnn',
     'mkhn',
     'mkhnn',
     'mkin',
     'mkinn',
     'mkjn',
     'mkjnn',
     'mkkn',
     'mkknn',
     'mkln',
     'mklnn',
     'mkmn',
     'mkmnn',
     'mkn',
     'mkna',
     'mknan',
     'mknb',
     'mknbn',
     'mknc',
     'mkncn',
     'mknd',
     'mkndn',
     'mkne',
     'mknen',
     'mknf',
     'mknfn',
     'mkng',
     'mkngn',
     'mknh',
     'mknhn',
     'mkni',
     'mknin',
     'mknj',
     'mknjn',
     'mknk',
     'mknkn',
     'mknl',
     'mknln',
     'mknm',
     'mknmn',
     'mknn',
     'mknna',
     'mknnb',
     'mknnc',
     'mknnd',
     'mknne',
     'mknnf',
     'mknng',
     'mknnh',
     'mknni',
     'mknnj',
     'mknnk',
     'mknnl',
     'mknnm',
     'mknnn',
     'mknno',
     'mknnp',
     'mknnq',
     'mknnr',
     'mknns',
     'mknnt',
     'mknnu',
     'mknnv',
     'mknnw',
     'mknnx',
     'mknny',
     'mknnz',
     'mkno',
     'mknon',
     'mknp',
     'mknpn',
     'mknq',
     'mknqn',
     'mknr',
     'mknrn',
     'mkns',
     'mknsn',
     'mknt',
     'mkntn',
     'mknu',
     'mknun',
     'mknv',
     'mknvn',
     'mknw',
     'mknwn',
     'mknx',
     'mknxn',
     'mkny',
     'mknyn',
     'mknz',
     'mknzn',
     'mkon',
     'mkonn',
     'mkpn',
     'mkpnn',
     'mkqn',
     'mkqnn',
     'mkrn',
     'mkrnn',
     'mksn',
     'mksnn',
     'mktn',
     'mktnn',
     'mkun',
     'mkunn',
     'mkvn',
     'mkvnn',
     'mkwn',
     'mkwnn',
     'mkxn',
     'mkxnn',
     'mkyn',
     'mkynn',
     'mkzn',
     'mkznn',
     'mlknn',
     'mlnn',
     'mmknn',
     'mmnn',
     'mnkn',
     'mnknn',
     'mnn',
     'mnnn',
     'moknn',
     'monn',
     'mpknn',
     'mpnn',
     'mqknn',
     'mqnn',
     'mrknn',
     'mrnn',
     'msknn',
     'msnn',
     'mtknn',
     'mtnn',
     'muknn',
     'mukunun',
     'munn',
     'mvknn',
     'mvnn',
     'mwknn',
     'mwnn',
     'mxknn',
     'mxnn',
     'myknn',
     'mynn',
     'mzknn',
     'mznn',
     'nknn',
     'nmknn',
     'oknn',
     'omknn',
     'pknn',
     'pmknn',
     'qknn',
     'qmknn',
     'rknn',
     'rmknn',
     'sknn',
     'smknn',
     'tknn',
     'tmknn',
     'uknn',
     'umknn',
     'vknn',
     'vmknn',
     'wknn',
     'wmknn',
     'xknn',
     'xmknn',
     'yknn',
     'ymknn',
     'zknn',
     'zmknn'}


