
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.7 s, sys: 1.49 s, total: 14.2 s
    Wall time: 17.9 s


List available deep learning NER models
---------------------------------------

.. code:: ipython3

    malaya.entity.available_deep_model()




.. parsed-literal::

    ['concat', 'bahdanau', 'luong', 'entity-network', 'attention']



Describe supported entities
---------------------------

.. code:: ipython3

    malaya.describe_entities()


.. parsed-literal::

    OTHER - Other
    law - law, regulation, related law documents, documents, etc
    location - location, place
    organization - organization, company, government, facilities, etc
    person - person, group of people, believes, etc
    quantity - numbers, quantity
    time - date, day, time, etc
    event - unique event happened, etc


.. code:: ipython3

    string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'

Load CRF model
--------------

.. code:: ipython3

    crf = malaya.entity.crf()
    crf.predict(string)




.. parsed-literal::

    [('kuala', 'location'),
     ('lumpur', 'location'),
     ('sempena', 'OTHER'),
     ('sambutan', 'event'),
     ('aidilfitri', 'event'),
     ('minggu', 'OTHER'),
     ('depan', 'OTHER'),
     ('perdana', 'person'),
     ('menteri', 'person'),
     ('tun', 'person'),
     ('dr', 'person'),
     ('mahathir', 'person'),
     ('mohamad', 'person'),
     ('dan', 'OTHER'),
     ('menteri', 'OTHER'),
     ('pengangkutan', 'OTHER'),
     ('anthony', 'person'),
     ('loke', 'person'),
     ('siew', 'person'),
     ('fook', 'person'),
     ('menitipkan', 'OTHER'),
     ('pesanan', 'OTHER'),
     ('khas', 'OTHER'),
     ('kepada', 'OTHER'),
     ('orang', 'OTHER'),
     ('ramai', 'OTHER'),
     ('yang', 'OTHER'),
     ('mahu', 'OTHER'),
     ('pulang', 'OTHER'),
     ('ke', 'OTHER'),
     ('kampung', 'location'),
     ('halaman', 'location'),
     ('masing-masing', 'OTHER'),
     ('dalam', 'OTHER'),
     ('video', 'OTHER'),
     ('pendek', 'OTHER'),
     ('terbitan', 'OTHER'),
     ('jabatan', 'organization'),
     ('keselamatan', 'organization'),
     ('jalan', 'organization'),
     ('raya', 'organization'),
     ('jkjr', 'organization'),
     ('itu', 'OTHER'),
     ('dr', 'person'),
     ('mahathir', 'person'),
     ('menasihati', 'OTHER'),
     ('mereka', 'OTHER'),
     ('supaya', 'OTHER'),
     ('berhenti', 'OTHER'),
     ('berehat', 'OTHER'),
     ('dan', 'OTHER'),
     ('tidur', 'OTHER'),
     ('sebentar', 'OTHER'),
     ('sekiranya', 'OTHER'),
     ('mengantuk', 'OTHER'),
     ('ketika', 'OTHER'),
     ('memandu', 'OTHER')]



Print important features from CRF model
---------------------------------------

.. code:: ipython3

    crf.print_features(10)


.. parsed-literal::

    Top-10 positive:
    15.295689 person   word:pengarah
    12.352726 location word:dibuat-buat
    11.206675 organization word:pas
    10.718764 person   word:solana
    10.579257 person   word:anggodo
    10.205311 location word:kenya
    10.178896 time     word:jumat
    10.138113 person   word:terpantas
    9.938075 OTHER    word:sudah
    9.896239 location word:pakistan
    
    Top-10 negative:
    -6.265843 OTHER    word:memintanya
    -6.318719 OTHER    prefix-3:pah
    -6.365330 time     next_word-suffix-3:nin
    -6.443976 person   is_numeric
    -6.508225 event    suffix-1:u
    -6.535034 OTHER    prefix-3:wir
    -7.109250 OTHER    prefix-3:di-
    -7.176552 OTHER    word:ramadan
    -7.470627 organization suffix-3:ari
    -7.846867 time     next_word-prefix-1:n


Print important transitions from CRF Model
------------------------------------------

.. code:: ipython3

    crf.print_transitions(10)


.. parsed-literal::

    Top-10 likely transitions:
    quantity -> quantity 4.731903
    location -> location 4.547566
    organization -> organization 4.322757
    OTHER  -> OTHER   4.267569
    event  -> event   3.796581
    law    -> law     3.234600
    person -> person  3.178005
    time   -> time    2.716374
    location -> OTHER   0.057188
    OTHER  -> location -0.033477
    
    Top-10 unlikely transitions:
    event  -> person  -4.618084
    event  -> quantity -4.649371
    time   -> law     -4.748618
    organization -> event   -4.763703
    event  -> location -4.995439
    organization -> law     -5.343159
    person -> law     -6.000496
    time   -> quantity -6.551308
    organization -> time    -6.602369
    quantity -> time    -8.047114


Load deep learning models
-------------------------

.. code:: ipython3

    for i in malaya.entity.available_deep_model():
        print('Testing %s model'%(i))
        model = malaya.entity.deep_model(i)
        print(model.predict(string))
        print()


.. parsed-literal::

    Testing concat model
    downloading frozen /Users/huseinzol/Malaya/entity/concat model


.. parsed-literal::

    8.00MB [00:01, 4.04MB/s]                          
      0%|          | 0.00/0.35 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading frozen /Users/huseinzol/Malaya/entity/concat setting


.. parsed-literal::

    1.00MB [00:00, 6.68MB/s]                   


.. parsed-literal::

    [('kuala', 'location'), ('lumpur', 'location'), ('sempena', 'OTHER'), ('sambutan', 'event'), ('aidilfitri', 'event'), ('minggu', 'time'), ('depan', 'time'), ('perdana', 'person'), ('menteri', 'person'), ('tun', 'person'), ('dr', 'person'), ('mahathir', 'person'), ('mohamad', 'OTHER'), ('dan', 'OTHER'), ('menteri', 'OTHER'), ('pengangkutan', 'organization'), ('anthony', 'person'), ('loke', 'person'), ('siew', 'person'), ('fook', 'person'), ('menitipkan', 'OTHER'), ('pesanan', 'OTHER'), ('khas', 'OTHER'), ('kepada', 'OTHER'), ('orang', 'OTHER'), ('ramai', 'OTHER'), ('yang', 'OTHER'), ('mahu', 'OTHER'), ('pulang', 'OTHER'), ('ke', 'OTHER'), ('kampung', 'OTHER'), ('halaman', 'location'), ('masing-masing', 'OTHER'), ('dalam', 'OTHER'), ('video', 'OTHER'), ('pendek', 'OTHER'), ('terbitan', 'OTHER'), ('jabatan', 'OTHER'), ('keselamatan', 'OTHER'), ('jalan', 'location'), ('raya', 'organization'), ('jkjr', 'time'), ('itu', 'OTHER'), ('dr', 'person'), ('mahathir', 'person'), ('menasihati', 'OTHER'), ('mereka', 'OTHER'), ('supaya', 'OTHER'), ('berhenti', 'OTHER'), ('berehat', 'OTHER'), ('dan', 'OTHER'), ('tidur', 'OTHER'), ('sebentar', 'OTHER'), ('sekiranya', 'OTHER'), ('mengantuk', 'OTHER'), ('ketika', 'OTHER'), ('memandu', 'OTHER')]
    
    Testing bahdanau model
    [('kuala', 'location'), ('lumpur', 'location'), ('sempena', 'OTHER'), ('sambutan', 'event'), ('aidilfitri', 'event'), ('minggu', 'time'), ('depan', 'time'), ('perdana', 'person'), ('menteri', 'person'), ('tun', 'person'), ('dr', 'person'), ('mahathir', 'person'), ('mohamad', 'person'), ('dan', 'OTHER'), ('menteri', 'organization'), ('pengangkutan', 'organization'), ('anthony', 'person'), ('loke', 'person'), ('siew', 'person'), ('fook', 'person'), ('menitipkan', 'OTHER'), ('pesanan', 'OTHER'), ('khas', 'OTHER'), ('kepada', 'OTHER'), ('orang', 'OTHER'), ('ramai', 'OTHER'), ('yang', 'OTHER'), ('mahu', 'OTHER'), ('pulang', 'OTHER'), ('ke', 'OTHER'), ('kampung', 'OTHER'), ('halaman', 'OTHER'), ('masing-masing', 'OTHER'), ('dalam', 'OTHER'), ('video', 'OTHER'), ('pendek', 'OTHER'), ('terbitan', 'OTHER'), ('jabatan', 'organization'), ('keselamatan', 'organization'), ('jalan', 'organization'), ('raya', 'organization'), ('jkjr', 'person'), ('itu', 'OTHER'), ('dr', 'person'), ('mahathir', 'person'), ('menasihati', 'OTHER'), ('mereka', 'OTHER'), ('supaya', 'OTHER'), ('berhenti', 'OTHER'), ('berehat', 'OTHER'), ('dan', 'OTHER'), ('tidur', 'OTHER'), ('sebentar', 'OTHER'), ('sekiranya', 'OTHER'), ('mengantuk', 'OTHER'), ('ketika', 'OTHER'), ('memandu', 'OTHER')]
    
    Testing luong model
    [('kuala', 'location'), ('lumpur', 'location'), ('sempena', 'OTHER'), ('sambutan', 'event'), ('aidilfitri', 'event'), ('minggu', 'time'), ('depan', 'time'), ('perdana', 'person'), ('menteri', 'person'), ('tun', 'person'), ('dr', 'person'), ('mahathir', 'person'), ('mohamad', 'person'), ('dan', 'OTHER'), ('menteri', 'person'), ('pengangkutan', 'organization'), ('anthony', 'person'), ('loke', 'person'), ('siew', 'person'), ('fook', 'person'), ('menitipkan', 'OTHER'), ('pesanan', 'OTHER'), ('khas', 'OTHER'), ('kepada', 'OTHER'), ('orang', 'OTHER'), ('ramai', 'OTHER'), ('yang', 'OTHER'), ('mahu', 'OTHER'), ('pulang', 'OTHER'), ('ke', 'OTHER'), ('kampung', 'OTHER'), ('halaman', 'location'), ('masing-masing', 'OTHER'), ('dalam', 'OTHER'), ('video', 'OTHER'), ('pendek', 'OTHER'), ('terbitan', 'OTHER'), ('jabatan', 'OTHER'), ('keselamatan', 'OTHER'), ('jalan', 'law'), ('raya', 'law'), ('jkjr', 'law'), ('itu', 'OTHER'), ('dr', 'person'), ('mahathir', 'person'), ('menasihati', 'OTHER'), ('mereka', 'OTHER'), ('supaya', 'OTHER'), ('berhenti', 'OTHER'), ('berehat', 'OTHER'), ('dan', 'OTHER'), ('tidur', 'OTHER'), ('sebentar', 'OTHER'), ('sekiranya', 'OTHER'), ('mengantuk', 'OTHER'), ('ketika', 'OTHER'), ('memandu', 'OTHER')]
    
    Testing entity-network model
    [('kuala', 'location'), ('lumpur', 'location'), ('sempena', 'OTHER'), ('sambutan', 'event'), ('aidilfitri', 'event'), ('minggu', 'OTHER'), ('depan', 'OTHER'), ('perdana', 'OTHER'), ('menteri', 'OTHER'), ('tun', 'person'), ('dr', 'person'), ('mahathir', 'person'), ('mohamad', 'person'), ('dan', 'OTHER'), ('menteri', 'OTHER'), ('pengangkutan', 'OTHER'), ('anthony', 'person'), ('loke', 'person'), ('siew', 'person'), ('fook', 'person'), ('menitipkan', 'OTHER'), ('pesanan', 'OTHER'), ('khas', 'OTHER'), ('kepada', 'OTHER'), ('orang', 'OTHER'), ('ramai', 'OTHER'), ('yang', 'OTHER'), ('mahu', 'OTHER'), ('pulang', 'OTHER'), ('ke', 'OTHER'), ('kampung', 'OTHER'), ('halaman', 'OTHER'), ('masing-masing', 'OTHER'), ('dalam', 'OTHER'), ('video', 'OTHER'), ('pendek', 'OTHER'), ('terbitan', 'OTHER'), ('jabatan', 'organization'), ('keselamatan', 'organization'), ('jalan', 'organization'), ('raya', 'organization'), ('jkjr', 'organization'), ('itu', 'OTHER'), ('dr', 'person'), ('mahathir', 'person'), ('menasihati', 'OTHER'), ('mereka', 'OTHER'), ('supaya', 'OTHER'), ('berhenti', 'OTHER'), ('berehat', 'OTHER'), ('dan', 'OTHER'), ('tidur', 'OTHER'), ('sebentar', 'OTHER'), ('sekiranya', 'OTHER'), ('mengantuk', 'OTHER'), ('ketika', 'OTHER'), ('memandu', 'OTHER')]
    
    Testing attention model
    [('kuala', 'location'), ('lumpur', 'location'), ('sempena', 'OTHER'), ('sambutan', 'event'), ('aidilfitri', 'event'), ('minggu', 'event'), ('depan', 'OTHER'), ('perdana', 'person'), ('menteri', 'OTHER'), ('tun', 'person'), ('dr', 'person'), ('mahathir', 'person'), ('mohamad', 'person'), ('dan', 'OTHER'), ('menteri', 'OTHER'), ('pengangkutan', 'OTHER'), ('anthony', 'person'), ('loke', 'person'), ('siew', 'person'), ('fook', 'person'), ('menitipkan', 'OTHER'), ('pesanan', 'OTHER'), ('khas', 'OTHER'), ('kepada', 'OTHER'), ('orang', 'OTHER'), ('ramai', 'OTHER'), ('yang', 'OTHER'), ('mahu', 'OTHER'), ('pulang', 'OTHER'), ('ke', 'OTHER'), ('kampung', 'location'), ('halaman', 'location'), ('masing-masing', 'OTHER'), ('dalam', 'OTHER'), ('video', 'OTHER'), ('pendek', 'OTHER'), ('terbitan', 'OTHER'), ('jabatan', 'OTHER'), ('keselamatan', 'OTHER'), ('jalan', 'location'), ('raya', 'location'), ('jkjr', 'location'), ('itu', 'OTHER'), ('dr', 'person'), ('mahathir', 'person'), ('menasihati', 'OTHER'), ('mereka', 'OTHER'), ('supaya', 'OTHER'), ('berhenti', 'OTHER'), ('berehat', 'OTHER'), ('dan', 'OTHER'), ('tidur', 'OTHER'), ('sebentar', 'OTHER'), ('sekiranya', 'OTHER'), ('mengantuk', 'OTHER'), ('ketika', 'OTHER'), ('memandu', 'OTHER')]
    


Print important features from deep learning model
-------------------------------------------------

.. code:: ipython3

    bahdanau = malaya.entity.deep_model('bahdanau')
    bahdanau.print_features(10)


.. parsed-literal::

    Top-10 positive:
    ton: 4.945406
    dollar: 4.345774
    disertai: 4.257772
    menjejaskan: 4.252921
    sesaat: 4.082481
    mata: 4.060701
    abul: 4.024586
    ruang: 3.983563
    orator: 3.899390
    universal: 3.866645
    
    Top-10 negative:
    abah: -4.691194
    raju: -4.370757
    dipengerusikan: -4.142600
    rs: -4.013050
    bacaan: -4.001595
    indonesia-malaysia: -3.921156
    nacp: -3.749232
    memprediksi: -3.659459
    ruhut: -3.620089
    pengesahan: -3.618848


Print important transitions from deep learning model
----------------------------------------------------

.. code:: ipython3

    bahdanau.print_transitions(10)


.. parsed-literal::

    Top-10 likely transitions:
    event -> event: 0.810878
    OTHER -> OTHER: 0.626205
    PAD -> OTHER: 0.519626
    PAD -> event: 0.512354
    law -> law: 0.460971
    person -> law: 0.448240
    person -> event: 0.407665
    location -> event: 0.402317
    organization -> PAD: 0.402057
    time -> person: 0.342275
    
    Top-10 unlikely transitions:
    person -> organization: -0.914907
    law -> event: -0.843547
    event -> law: -0.829639
    organization -> person: -0.810431
    time -> quantity: -0.783691
    person -> location: -0.712586
    quantity -> law: -0.663559
    law -> time: -0.656724
    quantity -> time: -0.640747
    organization -> quantity: -0.615018


Voting stack model
------------------

.. code:: ipython3

    entity_network = malaya.entity.deep_model('entity-network')
    bahdanau = malaya.entity.deep_model('bahdanau')
    luong = malaya.entity.deep_model('luong')
    malaya.stack.voting_stack([entity_network, bahdanau, crf], string)




.. parsed-literal::

    [('kuala', 'location'),
     ('lumpur', 'location'),
     ('sempena', 'OTHER'),
     ('sambutan', 'event'),
     ('aidilfitri', 'event'),
     ('minggu', 'time'),
     ('depan', 'OTHER'),
     ('perdana', 'person'),
     ('menteri', 'person'),
     ('tun', 'person'),
     ('dr', 'person'),
     ('mahathir', 'person'),
     ('mohamad', 'person'),
     ('dan', 'OTHER'),
     ('menteri', 'OTHER'),
     ('pengangkutan', 'OTHER'),
     ('anthony', 'person'),
     ('loke', 'person'),
     ('siew', 'person'),
     ('fook', 'person'),
     ('menitipkan', 'OTHER'),
     ('pesanan', 'OTHER'),
     ('khas', 'OTHER'),
     ('kepada', 'OTHER'),
     ('orang', 'OTHER'),
     ('ramai', 'OTHER'),
     ('yang', 'OTHER'),
     ('mahu', 'OTHER'),
     ('pulang', 'OTHER'),
     ('ke', 'OTHER'),
     ('kampung', 'OTHER'),
     ('halaman', 'OTHER'),
     ('masing-masing', 'OTHER'),
     ('dalam', 'OTHER'),
     ('video', 'OTHER'),
     ('pendek', 'OTHER'),
     ('terbitan', 'OTHER'),
     ('jabatan', 'organization'),
     ('keselamatan', 'organization'),
     ('jalan', 'organization'),
     ('raya', 'organization'),
     ('jkjr', 'organization'),
     ('itu', 'OTHER'),
     ('dr', 'person'),
     ('mahathir', 'person'),
     ('menasihati', 'OTHER'),
     ('mereka', 'OTHER'),
     ('supaya', 'OTHER'),
     ('berhenti', 'OTHER'),
     ('berehat', 'OTHER'),
     ('dan', 'OTHER'),
     ('tidur', 'OTHER'),
     ('sebentar', 'OTHER'),
     ('sekiranya', 'OTHER'),
     ('mengantuk', 'OTHER'),
     ('ketika', 'OTHER'),
     ('memandu', 'OTHER')]


