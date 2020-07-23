.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 5.03 s, sys: 1.24 s, total: 6.28 s
    Wall time: 7.11 s


.. code:: ipython3

    string1 = 'xjdi ke, y u xsuke makan HUSEIN kt situ tmpt, i hate it. pelikle, pada'
    string2 = 'i mmg2 xske mknn HUSEIN kampng tmpat, i love them. pelikle saye'
    string3 = 'perdana menteri ke11 sgt suka makn ayam, harganya cuma rm15.50'
    string4 = 'pada 10/4, kementerian mengumumkan, 1/100'
    string5 = 'Husein Zolkepli dapat tempat ke-12 lumba lari hari ni'
    string6 = 'Husein Zolkepli (2011 - 2019) adalah ketua kampng di kedah sekolah King Edward ke-IV'
    string7 = '2jam 30 minit aku tunggu kau, 60.1 kg kau ni, suhu harini 31.2c, aku dahaga minum 600ml'

Load normalizer
---------------

This normalizer can load any spelling correction model, eg,
``malaya.spell.probability``, or ``malaya.spell.transformer``.

.. code:: ipython3

    corrector = malaya.spell.probability()
    normalizer = malaya.normalize.normalizer(corrector)

normalize
^^^^^^^^^

.. code:: python

   def normalize(
       self, string: str, check_english: bool = True, normalize_entity = True
   ):
       """
       Normalize a string

       Parameters
       ----------
       string : str
       check_english: bool, (default=True)
           check a word in english dictionary.
       normalize_entity: bool, (default=True)
           normalize entities, only effect `date`, `datetime`, `time` and `money` patterns string only.

       Returns
       -------
       string: normalized string
       """

.. code:: ipython3

    string = 'boleh dtg 8pagi esok tak atau minggu depan? 2 oktober 2019 2pm, tlong bayar rm 3.2k sekali tau'

.. code:: ipython3

    normalizer.normalize(string)




.. parsed-literal::

    {'normalize': 'boleh datang lapan pagi esok tak atau minggu depan ? 02/10/2019 14:00:00 , tolong bayar tiga ribu dua ratus ringgit sekali tahu',
     'date': {'minggu depan': datetime.datetime(2020, 7, 30, 22, 16, 4, 460303),
      '8 AM esok': datetime.datetime(2020, 7, 24, 8, 0),
      '2 oktober 2019 2pm': datetime.datetime(2019, 10, 2, 14, 0)},
     'money': {'rm 3.2k': 'RM3200.0'}}



.. code:: ipython3

    normalizer.normalize(string, normalize_entity = False)




.. parsed-literal::

    {'normalize': 'boleh datang lapan pagi esok tak atau minggu depan ? 02/10/2019 14:00:00 , tolong bayar tiga ribu dua ratus ringgit sekali tahu',
     'date': {},
     'money': {}}



Here you can see, Malaya normalizer will normalize ``minggu depan`` to
datetime object, also ``3.2k ringgit`` to ``RM3200``

.. code:: ipython3

    print(normalizer.normalize(string1))
    print(normalizer.normalize(string2))
    print(normalizer.normalize(string3))
    print(normalizer.normalize(string4))
    print(normalizer.normalize(string5))
    print(normalizer.normalize(string6))
    print(normalizer.normalize(string7))


.. parsed-literal::

    {'normalize': 'tak jadi ke , kenapa awak tak suka makan HUSEIN kat situ tempat , saya hate it . pelik lah , pada', 'date': {}, 'money': {}}
    {'normalize': 'saya memang - memang tak suka makanan HUSEIN kampung tempat , saya love them . pelik lah saya', 'date': {}, 'money': {}}
    {'normalize': 'perdana menteri kesebelas sangat suka makan ayam , harganya cuma lima belas ringgit lima puluh sen', 'date': {}, 'money': {'rm15.50': 'RM15.50'}}
    {'normalize': 'pada sepuluh hari bulan empat , kementerian mengumumkan , satu per seratus', 'date': {}, 'money': {}}
    {'normalize': 'Husein Zolkepli dapat tempat kedua belas lumba lari hari ni', 'date': {}, 'money': {}}
    {'normalize': 'Husein Zolkepli ( dua ribu sebelas hingga dua ribu sembilan belas ) adalah ketua kampung di kedah sekolah King Edward keempat', 'date': {}, 'money': {}}
    {'normalize': 'dua jam tiga puluh minit aku tunggu kau , enam puluh perpuluhan satu kilogram kau ni , suhu hari ini tiga puluh satu perpuluhan dua celcius , aku dahaga minum enam ratus milliliter', 'date': {}, 'money': {}}


Normalizing rules
-----------------

**All these rules will ignore if first letter is capital.**

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

.. code:: ipython3

    normalizer.normalize('Dr yahaya')




.. parsed-literal::

    {'normalize': 'Doktor yahaya', 'date': {}, 'money': {}}



2. expand ``x``
^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('xtahu')




.. parsed-literal::

    {'normalize': 'tak tahu', 'date': {}, 'money': {}}



3. normalize ``ke -``
^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('ke-12')




.. parsed-literal::

    {'normalize': 'kedua belas', 'date': {}, 'money': {}}



.. code:: ipython3

    normalizer.normalize('ke - 12')




.. parsed-literal::

    {'normalize': 'kedua belas', 'date': {}, 'money': {}}



4. normalize ``ke - roman``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('ke-XXI')




.. parsed-literal::

    {'normalize': 'kedua puluh satu', 'date': {}, 'money': {}}



.. code:: ipython3

    normalizer.normalize('ke - XXI')




.. parsed-literal::

    {'normalize': 'kedua puluh satu', 'date': {}, 'money': {}}



5. normalize ``NUM - NUM``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('2011 - 2019')




.. parsed-literal::

    {'normalize': 'dua ribu sebelas hingga dua ribu sembilan belas',
     'date': {},
     'money': {}}



.. code:: ipython3

    normalizer.normalize('2011.01-2019')




.. parsed-literal::

    {'normalize': 'dua ribu sebelas perpuluhan kosong satu hingga dua ribu sembilan belas',
     'date': {},
     'money': {}}



6. normalize ``pada NUM (/ | -) NUM``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('pada 10/4')




.. parsed-literal::

    {'normalize': 'pada sepuluh hari bulan empat', 'date': {}, 'money': {}}



.. code:: ipython3

    normalizer.normalize('PADA 10 -4')




.. parsed-literal::

    {'normalize': 'pada sepuluh hari bulan empat', 'date': {}, 'money': {}}



7. normalize ``NUM / NUM``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('10 /4')




.. parsed-literal::

    {'normalize': 'sepuluh per empat', 'date': {}, 'money': {}}



8. normalize ``rm NUM``
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('RM10.5')




.. parsed-literal::

    {'normalize': 'sepuluh ringgit lima puluh sen',
     'date': {},
     'money': {'rm10.5': 'RM10.5'}}



9. normalize ``rm NUM sen``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('rm 10.5 sen')




.. parsed-literal::

    {'normalize': 'sepuluh ringgit lima puluh sen',
     'date': {},
     'money': {'rm 10.5': 'RM10.5'}}



10. normalize ``NUM sen``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('1015 sen')




.. parsed-literal::

    {'normalize': 'sepuluh ringgit lima belas sen',
     'date': {},
     'money': {'1015 sen': 'RM10.15'}}



11. normalize money
^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('rm10.4m')




.. parsed-literal::

    {'normalize': 'sepuluh juta empat ratus ribu ringgit',
     'date': {},
     'money': {'rm10.4m': 'RM10400000.0'}}



.. code:: ipython3

    normalizer.normalize('$10.4K')




.. parsed-literal::

    {'normalize': 'sepuluh ribu empat ratus dollar',
     'date': {},
     'money': {'$10.4k': '$10400.0'}}



12. normalize cardinal
^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('123')




.. parsed-literal::

    {'normalize': 'seratus dua puluh tiga', 'date': {}, 'money': {}}



13. normalize ordinal
^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('ke123')




.. parsed-literal::

    {'normalize': 'keseratus dua puluh tiga', 'date': {}, 'money': {}}



14. normalize date / time / datetime string to datetime.datetime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('2 hari lepas')




.. parsed-literal::

    {'normalize': 'dua hari lepas',
     'date': {'2 hari lalu': datetime.datetime(2020, 7, 19, 10, 41, 59, 113558)},
     'money': {}}



.. code:: ipython3

    normalizer.normalize('esok')




.. parsed-literal::

    {'normalize': 'esok',
     'date': {'esok': datetime.datetime(2020, 7, 22, 10, 41, 59, 309305)},
     'money': {}}



.. code:: ipython3

    normalizer.normalize('okt 2019')




.. parsed-literal::

    {'normalize': '21/10/2019',
     'date': {'okt 2019': datetime.datetime(2019, 10, 21, 0, 0)},
     'money': {}}



.. code:: ipython3

    normalizer.normalize('2pgi')




.. parsed-literal::

    {'normalize': 'dua pagi',
     'date': {'2 AM': datetime.datetime(2020, 7, 21, 2, 0)},
     'money': {}}



.. code:: ipython3

    normalizer.normalize('pukul 8 malam')




.. parsed-literal::

    {'normalize': 'pukul lapan malam',
     'date': {'pukul 8': datetime.datetime(2020, 7, 8, 0, 0)},
     'money': {}}



.. code:: ipython3

    normalizer.normalize('jan 2 2019 12:01pm')




.. parsed-literal::

    {'normalize': '02/01/2019 12:01:00',
     'date': {'jan 2 2019 12:01pm': datetime.datetime(2019, 1, 2, 12, 1)},
     'money': {}}



.. code:: ipython3

    normalizer.normalize('2 ptg jan 2 2019')




.. parsed-literal::

    {'normalize': 'dua petang 02/01/2019',
     'date': {'2 PM jan 2 2019': datetime.datetime(2019, 1, 2, 14, 0)},
     'money': {}}



15. normalize money string to string number representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('50 sen')




.. parsed-literal::

    {'normalize': 'lima puluh ringgit', 'date': {}, 'money': {'50 sen': 'RM0.5'}}



.. code:: ipython3

    normalizer.normalize('20.5 ringgit')




.. parsed-literal::

    {'normalize': 'dua puluh ringgit lima puluh sen',
     'date': {},
     'money': {'20.5 ringgit': 'RM20.5'}}



.. code:: ipython3

    normalizer.normalize('20m ringgit')




.. parsed-literal::

    {'normalize': 'dua puluh juta ringgit',
     'date': {},
     'money': {'20m ringgit': 'RM20000000.0'}}



.. code:: ipython3

    normalizer.normalize('22.5123334k ringgit')




.. parsed-literal::

    {'normalize': 'dua puluh dua ribu lima ratus dua belas ringgit tiga ribu tiga ratus tiga puluh empat sen',
     'date': {},
     'money': {'22.5123334k ringgit': 'RM22512.3334'}}



16. normalize date string to %d/%m/%y
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('1 nov 2019')




.. parsed-literal::

    {'normalize': '01/11/2019',
     'date': {'1 nov 2019': datetime.datetime(2019, 11, 1, 0, 0)},
     'money': {}}



.. code:: ipython3

    normalizer.normalize('januari 1 1996')




.. parsed-literal::

    {'normalize': '01/01/1996',
     'date': {'januari 1 1996': datetime.datetime(1996, 1, 1, 0, 0)},
     'money': {}}



.. code:: ipython3

    normalizer.normalize('januari 2019')




.. parsed-literal::

    {'normalize': '21/01/2019',
     'date': {'januari 2019': datetime.datetime(2019, 1, 21, 0, 0)},
     'money': {}}



17. normalize time string to %H:%M:%S
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('2pm')




.. parsed-literal::

    {'normalize': '14:00:00',
     'date': {'2pm': datetime.datetime(2020, 7, 21, 14, 0)},
     'money': {}}



.. code:: ipython3

    normalizer.normalize('2:01pm')




.. parsed-literal::

    {'normalize': '14:01:00',
     'date': {'2:01pm': datetime.datetime(2020, 7, 21, 14, 1)},
     'money': {}}



.. code:: ipython3

    normalizer.normalize('2AM')




.. parsed-literal::

    {'normalize': '02:00:00',
     'date': {'2am': datetime.datetime(2020, 7, 21, 2, 0)},
     'money': {}}



.. code:: ipython3

    ' - '.join(['h', 'h', 'h'])




.. parsed-literal::

    'h - h - h'



18. expand repetition shortform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('skit2')




.. parsed-literal::

    {'normalize': 'sakit - sakit', 'date': {}, 'money': {}}



.. code:: ipython3

    normalizer.normalize('xskit2')




.. parsed-literal::

    {'normalize': 'tak sakit - sakit', 'date': {}, 'money': {}}



.. code:: ipython3

    normalizer.normalize('xjdi2')




.. parsed-literal::

    {'normalize': 'tak jadi - jadi', 'date': {}, 'money': {}}



.. code:: ipython3

    normalizer.normalize('xjdi4')




.. parsed-literal::

    {'normalize': 'tak jadi - jadi - jadi - jadi', 'date': {}, 'money': {}}



.. code:: ipython3

    normalizer.normalize('xjdi0')




.. parsed-literal::

    {'normalize': 'tak jadi', 'date': {}, 'money': {}}



.. code:: ipython3

    normalizer.normalize('xjdi')




.. parsed-literal::

    {'normalize': 'tak jadi', 'date': {}, 'money': {}}



19. normalize ``NUM SI-UNIT``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('61.2 kg')




.. parsed-literal::

    {'normalize': 'enam puluh satu perpuluhan dua kilogram',
     'date': {},
     'money': {}}



.. code:: ipython3

    normalizer.normalize('61.2kg')




.. parsed-literal::

    {'normalize': 'enam puluh satu perpuluhan dua kilogram',
     'date': {},
     'money': {}}



.. code:: ipython3

    normalizer.normalize('61kg')




.. parsed-literal::

    {'normalize': 'enam puluh satu kilogram', 'date': {}, 'money': {}}



.. code:: ipython3

    normalizer.normalize('61ml')




.. parsed-literal::

    {'normalize': 'enam puluh satu milliliter', 'date': {}, 'money': {}}



.. code:: ipython3

    normalizer.normalize('61m')




.. parsed-literal::

    {'normalize': 'enam puluh satu meter', 'date': {}, 'money': {}}



.. code:: ipython3

    normalizer.normalize('61.3434km')




.. parsed-literal::

    {'normalize': 'enam puluh satu perpuluhan tiga empat tiga empat kilometer',
     'date': {},
     'money': {}}



.. code:: ipython3

    normalizer.normalize('61.3434c')




.. parsed-literal::

    {'normalize': 'enam puluh satu perpuluhan tiga empat tiga empat celcius',
     'date': {},
     'money': {}}



.. code:: ipython3

    normalizer.normalize('61.3434 c')




.. parsed-literal::

    {'normalize': 'enam puluh satu perpuluhan tiga empat tiga empat celcius',
     'date': {},
     'money': {}}



