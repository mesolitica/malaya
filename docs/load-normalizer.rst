
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 6.5 s, sys: 1.7 s, total: 8.2 s
    Wall time: 13.2 s


.. code:: python

    string1 = 'xjdi ke, y u xsuke makan HUSEIN kt situ tmpt, i hate it. pelikle, pada'
    string2 = 'i mmg xske mknn HUSEIN kampng tmpat, i love them. pelikle saye'
    string3 = 'perdana menteri ke11 sgt suka makn ayam, harganya cuma rm15.50'
    string4 = 'pada 10/4, kementerian mengumumkan, 1/100'
    string5 = 'Husein Zolkepli dapat tempat ke-12 lumba lari hari ni'
    string6 = 'Husein Zolkepli (2011 - 2019) adalah ketua kampng di kedah sekolah King Edward ke-IV'

Load spell normalizer
---------------------

.. code:: python

    corrector = malaya.spell.probability()
    normalizer = malaya.normalize.spell(corrector)

.. code:: python

    normalizer.normalize('boleh dtg 8pagi esok tak? 2 oktober 2019 2pm, tlong bayar rm 3.2k sekali tau')




.. parsed-literal::

    {'normalize': 'boleh dtg lapan pagi esok tak ? 2 oktober 2019 2pm , tolong bayar tiga ribu dua ratus perpuluhan kosong ringgit sekali tahu',
     'date': {'2 oktober 2019': datetime.datetime(2019, 10, 2, 0, 0),
      'esok': datetime.datetime(2019, 9, 15, 16, 9, 1, 141534),
      '2pm': datetime.datetime(2019, 9, 14, 14, 0),
      '8 AM': datetime.datetime(2019, 9, 14, 8, 0),
      '2 oktober 2019 2pm': datetime.datetime(2019, 10, 2, 14, 0)},
     'money': {'rm 3.2k': 'RM3200.0'}}



Here you can see, Malaya normalizer will normalize ``minggu depan`` to
datetime object, also ``3k ringgit`` to ``RM3000``

.. code:: python

    print(normalizer.normalize(string1))
    print(normalizer.normalize(string2))
    print(normalizer.normalize(string3))
    print(normalizer.normalize(string4))
    print(normalizer.normalize(string5))
    print(normalizer.normalize(string6))


.. parsed-literal::

    {'normalize': 'tak jadi ke , kenapa awak tak suka makan HUSEIN kat situ tmpat , saya hate it . pelik lah , pada', 'date': {}, 'money': {}}
    {'normalize': 'saya memang tak suka makanan HUSEIN kampung tempat , saya love them . pelik lah saya', 'date': {}, 'money': {}}
    {'normalize': 'perdana menteri kesebelas sangat suka makan ayam , harganya cuma lima belas perpuluhan lima ringgit', 'date': {}, 'money': {'rm15.50': 'RM15.5'}}
    {'normalize': 'pada sepuluh hari bulan empat , kementerian mengumumkan , satu per seratus', 'date': {}, 'money': {}}
    {'normalize': 'Husein Zolkepli dapat tempat kedua belas lumba lari hari ni', 'date': {}, 'money': {}}
    {'normalize': 'Husein Zolkepli ( dua ribu sebelas hingga dua ribu sembilan belas ) adalah ketua kampung di kedah sekolah King Edward keempat', 'date': {}, 'money': {}}


Normalizing rules
-----------------

1. Normalize title,
^^^^^^^^^^^^^^^^^^^

.. code:: python


   {
       'dr': 'Doktor',
       'yb': 'Yang Berhormat',
       'hj': 'Haji',
       'ybm': 'Yang Berhormat Mulia',
       'tyt': 'Tuan Yang Terutama',
       'yab': 'Yang Berhormat',
       'ybm': 'Yang Berhormat Mulia',
       'yabhg': 'Yang Amat Berbahagia',
       'ybhg': 'Yang Berbahagia',
       'miss': 'Cik',
   }

.. code:: python

    normalizer.normalize('Dr yahaya')




.. parsed-literal::

    {'normalize': 'Doktor yahaya', 'date': {}, 'money': {}}



2. expand ``x``
^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize('xtahu')




.. parsed-literal::

    {'normalize': 'tak tahu', 'date': {}, 'money': {}}



3. normalize ``ke -``
^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize('ke-12')




.. parsed-literal::

    {'normalize': 'kedua belas', 'date': {}, 'money': {}}



.. code:: python

    normalizer.normalize('ke - 12')




.. parsed-literal::

    {'normalize': 'kedua belas', 'date': {}, 'money': {}}



4. normalize ``ke - roman``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize('ke-XXI')




.. parsed-literal::

    {'normalize': 'kedua puluh satu', 'date': {}, 'money': {}}



.. code:: python

    normalizer.normalize('ke - XXI')




.. parsed-literal::

    {'normalize': 'kedua puluh satu', 'date': {}, 'money': {}}



5. normalize ``NUM - NUM``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize('2011 - 2019')




.. parsed-literal::

    {'normalize': 'dua ribu sebelas hingga dua ribu sembilan belas',
     'date': {},
     'money': {}}



.. code:: python

    normalizer.normalize('2011.01-2019')




.. parsed-literal::

    {'normalize': 'dua ribu sebelas perpuluhan kosong satu hingga dua ribu sembilan belas',
     'date': {},
     'money': {}}



6. normalize ``pada NUM (/ | -) NUM``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize('pada 10/4')




.. parsed-literal::

    {'normalize': 'pada sepuluh hari bulan empat', 'date': {}, 'money': {}}



.. code:: python

    normalizer.normalize('PADA 10 -4')




.. parsed-literal::

    {'normalize': 'pada sepuluh hari bulan empat', 'date': {}, 'money': {}}



7. normalize ``NUM / NUM``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize('10 /4')




.. parsed-literal::

    {'normalize': 'sepuluh per empat', 'date': {}, 'money': {}}



8. normalize ``rm NUM``
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize('RM 10.5')




.. parsed-literal::

    {'normalize': 'RM 10.5', 'date': {}, 'money': {'rm 10.5': 'RM10.5'}}



9. normalize ``rm NUM sen``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize('rm 10.5 sen')




.. parsed-literal::

    {'normalize': 'sepuluh perpuluhan lima ringgit sen',
     'date': {},
     'money': {'rm 10.5': 'RM10.5'}}



10. normalize ``NUM sen``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize('10.5 sen')




.. parsed-literal::

    {'normalize': 'kosong perpuluhan satu kosong lima ringgit',
     'date': {},
     'money': {'10.5 sen': 'RM0.105'}}



11. normalize money
^^^^^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize('rm10.4m')




.. parsed-literal::

    {'normalize': 'satu juta empat ratus ribu perpuluhan kosong ringgit',
     'date': {},
     'money': {'rm10.4m': 'RM10400000.0'}}



.. code:: python

    normalizer.normalize('$10.4K')




.. parsed-literal::

    {'normalize': 'seribu empat ratus perpuluhan kosong dollar',
     'date': {},
     'money': {'$10.4k': '$10400.0'}}



12. normalize cardinal
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize('123')




.. parsed-literal::

    {'normalize': 'seratus dua puluh tiga', 'date': {}, 'money': {}}



13. normalize ordinal
^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize('ke123')




.. parsed-literal::

    {'normalize': 'keseratus dua puluh tiga', 'date': {}, 'money': {}}



14. normalize date / time / datetime string to datetime.datetime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize('2 hari lepas')




.. parsed-literal::

    {'normalize': 'dua hari lepas',
     'date': {'2 hari lalu': datetime.datetime(2019, 9, 12, 16, 9, 1, 590144)},
     'money': {}}



.. code:: python

    normalizer.normalize('esok')




.. parsed-literal::

    {'normalize': 'esok',
     'date': {'esok': datetime.datetime(2019, 9, 15, 16, 9, 1, 611619)},
     'money': {}}



.. code:: python

    normalizer.normalize('okt 2019')




.. parsed-literal::

    {'normalize': 'okt 2019',
     'date': {'okt 2019': datetime.datetime(2019, 10, 14, 0, 0)},
     'money': {}}



.. code:: python

    normalizer.normalize('2pgi')




.. parsed-literal::

    {'normalize': 'dua pagi',
     'date': {'2 AM': datetime.datetime(2019, 9, 14, 2, 0)},
     'money': {}}



.. code:: python

    normalizer.normalize('pukul 8 malam')




.. parsed-literal::

    {'normalize': 'pukul lapan malam',
     'date': {'pukul 8': datetime.datetime(2019, 9, 8, 0, 0)},
     'money': {}}



.. code:: python

    normalizer.normalize('jan 2 2019 12:01pm')




.. parsed-literal::

    {'normalize': 'jan 2 2019 12:01pm',
     'date': {'jan 2 2019': datetime.datetime(2019, 1, 2, 0, 0),
      '12:01pm': datetime.datetime(2019, 9, 14, 12, 1),
      'jan 2 2019 12:01pm': datetime.datetime(2019, 1, 2, 12, 1)},
     'money': {}}



.. code:: python

    normalizer.normalize('2 ptg jan 2 2019')




.. parsed-literal::

    {'normalize': 'dua ptg jan 2 2019',
     'date': {'jan 2 2019': datetime.datetime(2019, 1, 2, 0, 0),
      '2 PM': datetime.datetime(2019, 9, 14, 14, 0),
      '2 PM jan 2 2019': datetime.datetime(2019, 1, 2, 14, 0)},
     'money': {}}



15. normalize money string to string number representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    normalizer.normalize('50 sen')




.. parsed-literal::

    {'normalize': 'kosong perpuluhan lima ringgit',
     'date': {},
     'money': {'50 sen': 'RM0.5'}}



.. code:: python

    normalizer.normalize('20.5 ringgit')




.. parsed-literal::

    {'normalize': 'dua puluh perpuluhan lima ringgit',
     'date': {},
     'money': {'20.5 ringgit': 'RM20.5'}}



.. code:: python

    normalizer.normalize('20m ringgit')




.. parsed-literal::

    {'normalize': 'dua juta perpuluhan kosong ringgit',
     'date': {},
     'money': {'20m ringgit': 'RM20000000.0'}}



.. code:: python

    normalizer.normalize('22.5123334k ringgit')




.. parsed-literal::

    {'normalize': 'dua ribu lima ratus dua belas perpuluhan tiga tiga tiga empat ringgit',
     'date': {},
     'money': {'22.5123334k ringgit': 'RM22512.3334'}}
