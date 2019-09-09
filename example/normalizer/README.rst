
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 5.85 s, sys: 1.41 s, total: 7.26 s
    Wall time: 11.9 s


.. code:: ipython3

    string1 = 'xjdi ke, y u xsuke makan HUSEIN kt situ tmpt, i hate it. pelikle, pada'
    string2 = 'i mmg xske mknn HUSEIN kampng tmpat, i love them. pelikle saye'
    string3 = 'perdana menteri ke11 sgt suka makn ayam, harganya cuma rm15.50'
    string4 = 'pada 10/4, kementerian mengumumkan, 1/100'
    string5 = 'Husein Zolkepli dapat tempat ke-12 lumba lari hari ni'
    string6 = 'Husein Zolkepli (2011 - 2019) adalah ketua kampng di kedah sekolah King Edward ke-IV'

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

    tak jadi ke , kenapa awak tak suka makan HUSEIN kat situ tmpat , saya hate it . pelik lah , pada
    saya memang tak suka makanan HUSEIN kampung tempat , saya love them . pelik lah saya
    perdana menteri kesebelas sangat suka makan ayam , harganya cuma lima belas perpuluhan lima ringgit
    pada sepuluh hari bulan empat , kementerian mengumumkan , satu per seratus
    Husein Zolkepli dapat tempat kedua belas lumba lari hari ni
    Husein Zolkepli ( dua ribu sebelas hingga dua ribu sembilan belas ) adalah ketua kampung di kedah sekolah King Edward keempat


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

.. code:: ipython3

    normalizer.normalize('Dr yahaya')




.. parsed-literal::

    'Doktor yahaya'



2. expand ``x``
^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('xtahu')




.. parsed-literal::

    'tak tahu'



3. normalize ``ke -``
^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('ke-12')




.. parsed-literal::

    'kedua belas'



.. code:: ipython3

    normalizer.normalize('ke - 12')




.. parsed-literal::

    'kedua belas'



4. normalize ``ke - roman``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('ke-XXI')




.. parsed-literal::

    'kedua puluh satu'



.. code:: ipython3

    normalizer.normalize('ke - XXI')




.. parsed-literal::

    'kedua puluh satu'



5. normalize ``NUM - NUM``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('2011 - 2019')




.. parsed-literal::

    'dua ribu sebelas hingga dua ribu sembilan belas'



.. code:: ipython3

    normalizer.normalize('2011.01-2019')




.. parsed-literal::

    'dua ribu sebelas perpuluhan kosong satu hingga dua ribu sembilan belas'



6. normalize ``pada NUM (/ | -) NUM``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('pada 10/4')




.. parsed-literal::

    'pada sepuluh hari bulan empat'



.. code:: ipython3

    normalizer.normalize('PADA 10 -4')




.. parsed-literal::

    'pada sepuluh hari bulan empat'



7. normalize ``NUM / NUM``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('10 /4')




.. parsed-literal::

    'sepuluh per empat'



8. normalize ``rm NUM``
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('RM 10.5')




.. parsed-literal::

    'sepuluh perpuluhan lima ringgit'



9. normalize ``rm NUM sen``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('rm 10.5 sen')




.. parsed-literal::

    'sepuluh perpuluhan lima ringgit'



10. normalize ``NUM sen``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('10.5 sen')




.. parsed-literal::

    'sepuluh perpuluhan lima sen'



11. normalize money
^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('rm10.4m')




.. parsed-literal::

    'sepuluh perpuluhan empat juta ringgit'



.. code:: ipython3

    normalizer.normalize('$10.4M')




.. parsed-literal::

    'sepuluh perpuluhan empat juta dollar'



.. code:: ipython3

    normalizer.normalize('rm10.4b')




.. parsed-literal::

    'sepuluh perpuluhan empat billion ringgit'



12. normalize cardinal
^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('123')




.. parsed-literal::

    'seratus dua puluh tiga'



13. normalize ordinal
^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize('ke123')




.. parsed-literal::

    'keseratus dua puluh tiga'


