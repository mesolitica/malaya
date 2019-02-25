
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.6 s, sys: 1.48 s, total: 14.1 s
    Wall time: 17.7 s


List available deep learning POS models
---------------------------------------

.. code:: ipython3

    malaya.pos.available_deep_model()




.. parsed-literal::

    ['concat', 'bahdanau', 'luong', 'entity-network', 'attention']



Describe supported POS
----------------------

.. code:: ipython3

    malaya.describe_pos()


.. parsed-literal::

    ADJ - Adjective, kata sifat
    ADP - Adposition
    ADV - Adverb, kata keterangan
    ADX - Auxiliary verb, kata kerja tambahan
    CCONJ - Coordinating conjuction, kata hubung
    DET - Determiner, kata penentu
    NOUN - Noun, kata nama
    NUM - Number, nombor
    PART - Particle
    PRON - Pronoun, kata ganti
    PROPN - Proper noun, kata ganti nama khas
    SCONJ - Subordinating conjunction
    SYM - Symbol
    VERB - Verb, kata kerja
    X - Other


Load CRF Model
--------------

.. code:: ipython3

    crf = malaya.pos.crf()

.. code:: ipython3

    string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'

.. code:: ipython3

    crf.predict(string)




.. parsed-literal::

    [('Kuala', 'PROPN'),
     ('Lumpur', 'PROPN'),
     ('Sempena', 'SCONJ'),
     ('sambutan', 'NOUN'),
     ('Aidilfitri', 'PROPN'),
     ('minggu', 'NOUN'),
     ('depan', 'ADP'),
     ('Perdana', 'PROPN'),
     ('Menteri', 'PROPN'),
     ('Tun', 'PROPN'),
     ('Dr', 'PROPN'),
     ('Mahathir', 'PROPN'),
     ('Mohamad', 'PROPN'),
     ('dan', 'CCONJ'),
     ('Menteri', 'PROPN'),
     ('Pengangkutan', 'PROPN'),
     ('Anthony', 'PROPN'),
     ('Loke', 'PROPN'),
     ('Siew', 'PROPN'),
     ('Fook', 'PROPN'),
     ('menitipkan', 'VERB'),
     ('pesanan', 'NOUN'),
     ('khas', 'ADJ'),
     ('kepada', 'ADP'),
     ('orang', 'NOUN'),
     ('ramai', 'ADJ'),
     ('yang', 'PRON'),
     ('mahu', 'ADV'),
     ('pulang', 'VERB'),
     ('ke', 'ADP'),
     ('kampung', 'NOUN'),
     ('halaman', 'NOUN'),
     ('masing-masing', 'NOUN'),
     ('Dalam', 'ADP'),
     ('video', 'NOUN'),
     ('pendek', 'ADJ'),
     ('terbitan', 'NOUN'),
     ('Jabatan', 'PROPN'),
     ('Keselamatan', 'PROPN'),
     ('Jalan', 'PROPN'),
     ('Raya', 'PROPN'),
     ('Jkjr', 'PROPN'),
     ('itu', 'DET'),
     ('Dr', 'PROPN'),
     ('Mahathir', 'PROPN'),
     ('menasihati', 'VERB'),
     ('mereka', 'PRON'),
     ('supaya', 'SCONJ'),
     ('berhenti', 'VERB'),
     ('berehat', 'VERB'),
     ('dan', 'CCONJ'),
     ('tidur', 'VERB'),
     ('sebentar', 'ADP'),
     ('sekiranya', 'NOUN'),
     ('mengantuk', 'VERB'),
     ('ketika', 'ADV'),
     ('memandu', 'VERB')]



Print important features CRF model
----------------------------------

.. code:: ipython3

    crf.print_features(10)


.. parsed-literal::

    Top-10 positive:
    16.443463 DET      word:para
    15.494273 DET      word:berbagai
    14.856205 DET      word:tersebut
    14.426293 ADJ      word:menakjubkan
    14.319714 ADV      word:memang
    14.158206 ADP      word:tentang
    13.907366 VERB     word:percaya
    13.635634 VERB     word:integrasi
    13.630582 ADP      word:dengan
    13.562358 ADV      word:menurutnya
    
    Top-10 negative:
    -6.663068 PROPN    prefix-2:be
    -6.714450 ADV      next_word:menyatakan
    -6.862083 PROPN    next_word:Jepang
    -7.183600 PROPN    suffix-3:pun
    -7.264241 ADV      next_word-suffix-3:nai
    -7.676069 VERB     word:memuaskan
    -7.961231 ADP      prev_word:pernah
    -8.006671 NOUN     suffix-2:ke
    -8.135974 ADP      prev_word-prefix-3:pal
    -8.173493 PROPN    suffix-3:nya


Print important transitions CRF model
-------------------------------------

.. code:: ipython3

    crf.print_transitions(10)


.. parsed-literal::

    Top-10 likely transitions:
    PROPN  -> PROPN   5.767666
    NOUN   -> NOUN    4.291842
    DET    -> DET     3.723729
    NOUN   -> PROPN   3.035784
    CCONJ  -> CCONJ   2.545162
    X      -> X       2.476296
    ADP    -> NOUN    2.324735
    ADJ    -> ADJ     2.285807
    NOUN   -> ADJ     2.258407
    ADP    -> PROPN   2.181474
    
    Top-10 unlikely transitions:
    SCONJ  -> AUX     -3.341014
    PART   -> NUM     -3.406289
    SCONJ  -> ADJ     -3.447362
    SYM    -> ADV     -3.468094
    SYM    -> ADJ     -3.597291
    AUX    -> NUM     -3.657861
    PART   -> PART    -4.059430
    X      -> CCONJ   -4.929272
    ADP    -> SCONJ   -4.960199
    ADP    -> CCONJ   -6.236844


Load deep learning models
-------------------------

.. code:: ipython3

    for i in malaya.pos.available_deep_model():
        print('Testing %s model'%(i))
        model = malaya.pos.deep_model(i)
        print(model.predict(string))
        print()


.. parsed-literal::

    Testing concat model
    [('Kuala', 'PROPN'), ('Lumpur', 'PROPN'), ('Sempena', 'PROPN'), ('sambutan', 'NOUN'), ('Aidilfitri', 'PROPN'), ('minggu', 'NOUN'), ('depan', 'NOUN'), ('Perdana', 'PROPN'), ('Menteri', 'PROPN'), ('Tun', 'PROPN'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('Mohamad', 'NOUN'), ('dan', 'CCONJ'), ('Menteri', 'PROPN'), ('Pengangkutan', 'NOUN'), ('Anthony', 'PROPN'), ('Loke', 'PROPN'), ('Siew', 'NOUN'), ('Fook', 'PROPN'), ('menitipkan', 'VERB'), ('pesanan', 'NOUN'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'ADJ'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'NOUN'), ('Dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('Jabatan', 'NOUN'), ('Keselamatan', 'NOUN'), ('Jalan', 'PROPN'), ('Raya', 'PROPN'), ('Jkjr', 'NOUN'), ('itu', 'DET'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('menasihati', 'VERB'), ('mereka', 'PRON'), ('supaya', 'SCONJ'), ('berhenti', 'VERB'), ('berehat', 'NOUN'), ('dan', 'CCONJ'), ('tidur', 'NOUN'), ('sebentar', 'ADV'), ('sekiranya', 'NOUN'), ('mengantuk', 'VERB'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]
    
    Testing bahdanau model
    [('Kuala', 'PROPN'), ('Lumpur', 'PROPN'), ('Sempena', 'PROPN'), ('sambutan', 'NOUN'), ('Aidilfitri', 'PROPN'), ('minggu', 'VERB'), ('depan', 'ADJ'), ('Perdana', 'PROPN'), ('Menteri', 'PROPN'), ('Tun', 'PROPN'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('Mohamad', 'PROPN'), ('dan', 'CCONJ'), ('Menteri', 'PROPN'), ('Pengangkutan', 'PROPN'), ('Anthony', 'PROPN'), ('Loke', 'PROPN'), ('Siew', 'PROPN'), ('Fook', 'PROPN'), ('menitipkan', 'VERB'), ('pesanan', 'NOUN'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'ADJ'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'NOUN'), ('Dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('Jabatan', 'NOUN'), ('Keselamatan', 'NOUN'), ('Jalan', 'PROPN'), ('Raya', 'PROPN'), ('Jkjr', 'PROPN'), ('itu', 'DET'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('menasihati', 'VERB'), ('mereka', 'PRON'), ('supaya', 'SCONJ'), ('berhenti', 'VERB'), ('berehat', 'NOUN'), ('dan', 'CCONJ'), ('tidur', 'VERB'), ('sebentar', 'ADV'), ('sekiranya', 'NOUN'), ('mengantuk', 'VERB'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]
    
    Testing luong model
    [('Kuala', 'PROPN'), ('Lumpur', 'PROPN'), ('Sempena', 'PROPN'), ('sambutan', 'NOUN'), ('Aidilfitri', 'PROPN'), ('minggu', 'NOUN'), ('depan', 'ADJ'), ('Perdana', 'PROPN'), ('Menteri', 'PROPN'), ('Tun', 'PROPN'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('Mohamad', 'PROPN'), ('dan', 'CCONJ'), ('Menteri', 'PROPN'), ('Pengangkutan', 'PROPN'), ('Anthony', 'PROPN'), ('Loke', 'PROPN'), ('Siew', 'PROPN'), ('Fook', 'PROPN'), ('menitipkan', 'VERB'), ('pesanan', 'NOUN'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'ADJ'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'VERB'), ('Dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('Jabatan', 'PROPN'), ('Keselamatan', 'PROPN'), ('Jalan', 'PROPN'), ('Raya', 'PROPN'), ('Jkjr', 'PROPN'), ('itu', 'DET'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('menasihati', 'VERB'), ('mereka', 'PRON'), ('supaya', 'SCONJ'), ('berhenti', 'VERB'), ('berehat', 'NOUN'), ('dan', 'CCONJ'), ('tidur', 'VERB'), ('sebentar', 'ADV'), ('sekiranya', 'NOUN'), ('mengantuk', 'VERB'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]
    
    Testing entity-network model
    [('Kuala', 'NUM'), ('Lumpur', 'NUM'), ('Sempena', 'NUM'), ('sambutan', 'NUM'), ('Aidilfitri', 'NUM'), ('minggu', 'NOUN'), ('depan', 'NOUN'), ('Perdana', 'ADJ'), ('Menteri', 'CCONJ'), ('Tun', 'NUM'), ('Dr', 'NUM'), ('Mahathir', 'NUM'), ('Mohamad', 'NUM'), ('dan', 'CCONJ'), ('Menteri', 'NUM'), ('Pengangkutan', 'PROPN'), ('Anthony', 'PROPN'), ('Loke', 'PROPN'), ('Siew', 'PROPN'), ('Fook', 'PROPN'), ('menitipkan', 'NUM'), ('pesanan', 'NUM'), ('khas', 'NUM'), ('kepada', 'PROPN'), ('orang', 'PROPN'), ('ramai', 'VERB'), ('yang', 'DET'), ('mahu', 'NOUN'), ('pulang', 'NUM'), ('ke', 'PROPN'), ('kampung', 'VERB'), ('halaman', 'NUM'), ('masing-masing', 'NUM'), ('Dalam', 'NUM'), ('video', 'NOUN'), ('pendek', 'NUM'), ('terbitan', 'NUM'), ('Jabatan', 'NOUN'), ('Keselamatan', 'NOUN'), ('Jalan', 'NUM'), ('Raya', 'NUM'), ('Jkjr', 'NUM'), ('itu', 'SYM'), ('Dr', 'X'), ('Mahathir', 'NUM'), ('menasihati', 'NUM'), ('mereka', 'NUM'), ('supaya', 'NOUN'), ('berhenti', 'ADJ'), ('berehat', 'ADJ'), ('dan', 'CCONJ'), ('tidur', 'NOUN'), ('sebentar', 'NOUN'), ('sekiranya', 'NUM'), ('mengantuk', 'NUM'), ('ketika', 'NOUN'), ('memandu', 'NUM')]
    
    Testing attention model
    [('Kuala', 'PROPN'), ('Lumpur', 'PROPN'), ('Sempena', 'PROPN'), ('sambutan', 'NOUN'), ('Aidilfitri', 'PROPN'), ('minggu', 'NOUN'), ('depan', 'ADJ'), ('Perdana', 'PROPN'), ('Menteri', 'PROPN'), ('Tun', 'PROPN'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('Mohamad', 'PROPN'), ('dan', 'CCONJ'), ('Menteri', 'PROPN'), ('Pengangkutan', 'PROPN'), ('Anthony', 'PROPN'), ('Loke', 'PROPN'), ('Siew', 'PROPN'), ('Fook', 'PROPN'), ('menitipkan', 'VERB'), ('pesanan', 'NOUN'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'ADJ'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'VERB'), ('Dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('Jabatan', 'NOUN'), ('Keselamatan', 'PROPN'), ('Jalan', 'PROPN'), ('Raya', 'PROPN'), ('Jkjr', 'PROPN'), ('itu', 'DET'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('menasihati', 'VERB'), ('mereka', 'PRON'), ('supaya', 'SCONJ'), ('berhenti', 'VERB'), ('berehat', 'NOUN'), ('dan', 'CCONJ'), ('tidur', 'VERB'), ('sebentar', 'ADV'), ('sekiranya', 'ADJ'), ('mengantuk', 'VERB'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]
    


Print important features from deep learning model
-------------------------------------------------

.. code:: ipython3

    bahdanau = malaya.pos.deep_model('bahdanau')
    bahdanau.print_features(10)


.. parsed-literal::

    Top-10 positive:
    1971: 4.942553
    Puisi: 4.754801
    27: 4.659504
    buahan: 4.551769
    kaisarnya: 4.503439
    Kedua: 4.459490
    Times: 4.378673
    perlengkapan: 4.342615
    kelautan: 4.273527
    Persija: 4.260429
    
    Top-10 negative:
    Sakova: -5.102705
    engkau: -5.000618
    Cin: -4.962496
    bermesin: -4.823804
    Husm: -4.719638
    saatnya: -4.693280
    Vireta: -4.615777
    menjamu: -4.589007
    Aff: -4.437630
    dilahirkan: -4.422080


Print important transitions from deep learning model
----------------------------------------------------

.. code:: ipython3

    bahdanau.print_transitions(10)


.. parsed-literal::

    Top-10 likely transitions:
    SCONJ -> CCONJ: 0.688627
    SCONJ -> PRON: 0.539603
    ADV -> NUM: 0.517046
    PROPN -> PART: 0.479875
    ADP -> DET: 0.470052
    AUX -> ADV: 0.424240
    PRON -> NUM: 0.420834
    PAD -> AUX: 0.415958
    NUM -> ADV: 0.401860
    PART -> SYM: 0.395167
    
    Top-10 unlikely transitions:
    ADP -> CCONJ: -0.791846
    DET -> X: -0.675577
    SCONJ -> SCONJ: -0.665004
    VERB -> VERB: -0.646812
    PART -> NUM: -0.644018
    CCONJ -> CCONJ: -0.590792
    AUX -> NUM: -0.579523
    ADV -> SCONJ: -0.569171
    NUM -> VERB: -0.568291
    PRON -> SYM: -0.563159


Voting stack model
------------------

.. code:: ipython3

    entity_network = malaya.pos.deep_model('entity-network')
    bahdanau = malaya.pos.deep_model('bahdanau')
    luong = malaya.pos.deep_model('luong')
    malaya.stack.voting_stack([luong, bahdanau, crf, entity_network], string)




.. parsed-literal::

    [('Kuala', 'PROPN'),
     ('Lumpur', 'PROPN'),
     ('Sempena', 'PROPN'),
     ('sambutan', 'NOUN'),
     ('Aidilfitri', 'PROPN'),
     ('minggu', 'NOUN'),
     ('depan', 'ADP'),
     ('Perdana', 'PROPN'),
     ('Menteri', 'PROPN'),
     ('Tun', 'PROPN'),
     ('Dr', 'PROPN'),
     ('Mahathir', 'PROPN'),
     ('Mohamad', 'PROPN'),
     ('dan', 'CCONJ'),
     ('Menteri', 'PROPN'),
     ('Pengangkutan', 'PROPN'),
     ('Anthony', 'PROPN'),
     ('Loke', 'PROPN'),
     ('Siew', 'PROPN'),
     ('Fook', 'PROPN'),
     ('menitipkan', 'VERB'),
     ('pesanan', 'NOUN'),
     ('khas', 'ADJ'),
     ('kepada', 'ADP'),
     ('orang', 'NOUN'),
     ('ramai', 'ADJ'),
     ('yang', 'PRON'),
     ('mahu', 'ADV'),
     ('pulang', 'VERB'),
     ('ke', 'ADP'),
     ('kampung', 'NOUN'),
     ('halaman', 'NOUN'),
     ('masing-masing', 'NOUN'),
     ('Dalam', 'ADP'),
     ('video', 'NOUN'),
     ('pendek', 'ADJ'),
     ('terbitan', 'NOUN'),
     ('Jabatan', 'NOUN'),
     ('Keselamatan', 'PROPN'),
     ('Jalan', 'PROPN'),
     ('Raya', 'PROPN'),
     ('Jkjr', 'PROPN'),
     ('itu', 'DET'),
     ('Dr', 'PROPN'),
     ('Mahathir', 'PROPN'),
     ('menasihati', 'VERB'),
     ('mereka', 'PRON'),
     ('supaya', 'SCONJ'),
     ('berhenti', 'VERB'),
     ('berehat', 'NOUN'),
     ('dan', 'CCONJ'),
     ('tidur', 'NOUN'),
     ('sebentar', 'NOUN'),
     ('sekiranya', 'NOUN'),
     ('mengantuk', 'VERB'),
     ('ketika', 'SCONJ'),
     ('memandu', 'VERB')]


