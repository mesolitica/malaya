
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.4 s, sys: 1.57 s, total: 14 s
    Wall time: 17.9 s


.. code:: ipython3

    string1 = 'xjdi ke, y u xsuke makan HUSEIN kt situ tmpt, i hate it. pelikle'
    string2 = 'i mmg xske mknn HUSEIN kampng tmpat, i love them. pelikle saye'
    string3 = 'perdana menteri ke11 sgt suka mkan ayam, harganya cuma rm15.50'
    string4 = 'pada 10/4, kementerian mengumumkan'
    string5 = 'Husein Zolkepli dapat tempat ke-12 lumba lari hari ni'
    string6 = 'Husein Zolkepli (2011 - 2019) adalah ketua kampng di kedah'

Load basic normalizer
---------------------

.. code:: ipython3

    print(malaya.normalize.basic(string1))
    print(malaya.normalize.basic(string2))
    print(malaya.normalize.basic(string3))
    print(malaya.normalize.basic(string4))
    print(malaya.normalize.basic(string5))
    print(malaya.normalize.basic(string6))


.. parsed-literal::

    xjdi ke kenapa awak xsuke makan Husein kt situ tmpt saya hate it pelikle
    saya mmg xske mknn Husein kampng tmpat saya love them pelikle saye
    perdana menteri ke sgt suka mkan ayam harganya cuma rm
    pada kementerian mengumumkan
    Husein Zolkepli dapat tempat ke lumba lari hari ni
    Husein Zolkepli adalah ketua kampng di kedah


Load spell normalizer
---------------------

.. code:: ipython3

    corrector = malaya.spell.probability()
    normalizer = malaya.normalize.spell(corrector)

.. code:: ipython3

    print(normalizer.normalize(string1))
    print(normalizer.normalize(string2))
    print(normalizer.normalize(string3))
    print(normalizer.normalize(string4))
    print(normalizer.normalize(string5))
    print(normalizer.normalize(string6))


.. parsed-literal::

    tak jadi ke , kenapa awak tak suka makan HUSEIN kat itu mpt , saya hate it . pelik lah
    saya memang tak suka makanan HUSEIN kampung tempat , saya love them . pelik lah sama
    perdana menteri ke-sebelas sangat suka makan awam , harganya cuma lima belas perpuluhan lima ringgit
    pada sepuluh hari bulan empat , kementerian mengumumkan
    Husein Zolkepli dapat tempat ke-dua belas lumba lari hari ni
    Husein Zolkepli ( dua ribu sebelas hingga dua ribu sembilan belas ) adalah ketua kampung di kedai


We can see that our normalizer normalize ``ayam`` become ``awam``, this
is because we force our spelling correction to predict correct word, to
disable that, simply ``assume_wrong = False``.

.. code:: ipython3

    %%time
    normalizer.normalize(string3, assume_wrong = False)


.. parsed-literal::

    CPU times: user 505 µs, sys: 1e+03 ns, total: 506 µs
    Wall time: 513 µs




.. parsed-literal::

    'perdana menteri ke-sebelas sangat suka makan ayam , harganya cuma lima belas perpuluhan lima ringgit'



.. code:: ipython3

    %%time
    normalizer.normalize(string2, assume_wrong = False)


.. parsed-literal::

    CPU times: user 1.54 ms, sys: 27 µs, total: 1.57 ms
    Wall time: 1.59 ms




.. parsed-literal::

    'saya memang tak ska makanan HUSEIN kampung tempat , saya love them . pelik lah saya'



.. code:: ipython3

    %%time
    normalizer.normalize(string6, assume_wrong = False)


.. parsed-literal::

    CPU times: user 450 µs, sys: 15 µs, total: 465 µs
    Wall time: 482 µs




.. parsed-literal::

    'Husein Zolkepli ( dua ribu sebelas hingga dua ribu sembilan belas ) adalah ketua kampung di kedah'



Load fuzzy normalizer
---------------------

.. code:: ipython3

    malays = malaya.load_malay_dictionary()
    normalizer = malaya.normalize.fuzzy(malays)

.. code:: ipython3

    %%time
    normalizer.normalize(string3)


.. parsed-literal::

    CPU times: user 7.54 s, sys: 83 ms, total: 7.63 s
    Wall time: 7.9 s




.. parsed-literal::

    'perdana menteri ke-sebelas sangat suka makan ayam , harganya cuma lima belas perpuluhan lima ringgit'



.. code:: ipython3

    %%time
    normalizer.normalize(string2)


.. parsed-literal::

    CPU times: user 7.43 s, sys: 65.9 ms, total: 7.49 s
    Wall time: 7.7 s




.. parsed-literal::

    'saya memang tak saka makanan HUSEIN kampung tempat , saya love them . pelik lah saya'


