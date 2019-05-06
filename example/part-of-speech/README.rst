
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.1 s, sys: 1.51 s, total: 13.6 s
    Wall time: 17.3 s


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

    [('KUALA', 'PROPN'),
     ('LUMPUR', 'PROPN'),
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
     ('JKJR', 'PROPN'),
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
    [('KUALA', 'PROPN'), ('LUMPUR', 'PROPN'), ('Sempena', 'PROPN'), ('sambutan', 'NOUN'), ('Aidilfitri', 'NOUN'), ('minggu', 'NOUN'), ('depan', 'NOUN'), ('Perdana', 'PROPN'), ('Menteri', 'PROPN'), ('Tun', 'PROPN'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('Mohamad', 'PROPN'), ('dan', 'CCONJ'), ('Menteri', 'PROPN'), ('Pengangkutan', 'NOUN'), ('Anthony', 'PROPN'), ('Loke', 'PROPN'), ('Siew', 'PROPN'), ('Fook', 'PROPN'), ('menitipkan', 'VERB'), ('pesanan', 'NOUN'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'NOUN'), ('yang', 'PRON'), ('mahu', 'PRON'), ('pulang', 'VERB'), ('ke', 'NUM'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'VERB'), ('Dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('Jabatan', 'NOUN'), ('Keselamatan', 'NOUN'), ('Jalan', 'PROPN'), ('Raya', 'PROPN'), ('JKJR', 'NOUN'), ('itu', 'DET'), ('Dr', 'ADV'), ('Mahathir', 'PROPN'), ('menasihati', 'VERB'), ('mereka', 'PRON'), ('supaya', 'SCONJ'), ('berhenti', 'VERB'), ('berehat', 'NOUN'), ('dan', 'CCONJ'), ('tidur', 'NOUN'), ('sebentar', 'ADV'), ('sekiranya', 'NOUN'), ('mengantuk', 'VERB'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]
    
    Testing bahdanau model
    [('KUALA', 'PROPN'), ('LUMPUR', 'PROPN'), ('Sempena', 'PROPN'), ('sambutan', 'NOUN'), ('Aidilfitri', 'PROPN'), ('minggu', 'NOUN'), ('depan', 'ADJ'), ('Perdana', 'PROPN'), ('Menteri', 'PROPN'), ('Tun', 'PROPN'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('Mohamad', 'PROPN'), ('dan', 'CCONJ'), ('Menteri', 'PROPN'), ('Pengangkutan', 'NOUN'), ('Anthony', 'PROPN'), ('Loke', 'PROPN'), ('Siew', 'PROPN'), ('Fook', 'PROPN'), ('menitipkan', 'VERB'), ('pesanan', 'NOUN'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'ADJ'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'NOUN'), ('Dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('Jabatan', 'NOUN'), ('Keselamatan', 'PROPN'), ('Jalan', 'PROPN'), ('Raya', 'PROPN'), ('JKJR', 'PROPN'), ('itu', 'DET'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('menasihati', 'NOUN'), ('mereka', 'PRON'), ('supaya', 'SCONJ'), ('berhenti', 'VERB'), ('berehat', 'NOUN'), ('dan', 'CCONJ'), ('tidur', 'NOUN'), ('sebentar', 'ADJ'), ('sekiranya', 'NOUN'), ('mengantuk', 'VERB'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]
    
    Testing luong model
    [('KUALA', 'PROPN'), ('LUMPUR', 'PROPN'), ('Sempena', 'PROPN'), ('sambutan', 'NOUN'), ('Aidilfitri', 'PROPN'), ('minggu', 'NOUN'), ('depan', 'ADJ'), ('Perdana', 'PROPN'), ('Menteri', 'PROPN'), ('Tun', 'PROPN'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('Mohamad', 'PROPN'), ('dan', 'CCONJ'), ('Menteri', 'PROPN'), ('Pengangkutan', 'PROPN'), ('Anthony', 'PROPN'), ('Loke', 'PROPN'), ('Siew', 'PROPN'), ('Fook', 'PROPN'), ('menitipkan', 'VERB'), ('pesanan', 'NOUN'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'ADJ'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'NOUN'), ('Dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('Jabatan', 'NOUN'), ('Keselamatan', 'PROPN'), ('Jalan', 'PROPN'), ('Raya', 'PROPN'), ('JKJR', 'PROPN'), ('itu', 'DET'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('menasihati', 'VERB'), ('mereka', 'PRON'), ('supaya', 'SCONJ'), ('berhenti', 'VERB'), ('berehat', 'NOUN'), ('dan', 'CCONJ'), ('tidur', 'VERB'), ('sebentar', 'ADJ'), ('sekiranya', 'NOUN'), ('mengantuk', 'VERB'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]
    
    Testing entity-network model
    [('KUALA', 'NUM'), ('LUMPUR', 'NUM'), ('Sempena', 'NUM'), ('sambutan', 'NUM'), ('Aidilfitri', 'NUM'), ('minggu', 'NOUN'), ('depan', 'NOUN'), ('Perdana', 'ADJ'), ('Menteri', 'CCONJ'), ('Tun', 'NUM'), ('Dr', 'NUM'), ('Mahathir', 'NUM'), ('Mohamad', 'NUM'), ('dan', 'CCONJ'), ('Menteri', 'ADV'), ('Pengangkutan', 'ADV'), ('Anthony', 'ADJ'), ('Loke', 'NUM'), ('Siew', 'SYM'), ('Fook', 'PROPN'), ('menitipkan', 'PROPN'), ('pesanan', 'PROPN'), ('khas', 'PROPN'), ('kepada', 'PROPN'), ('orang', 'PROPN'), ('ramai', 'VERB'), ('yang', 'PRON'), ('mahu', 'NOUN'), ('pulang', 'ADV'), ('ke', 'ADV'), ('kampung', 'VERB'), ('halaman', 'NUM'), ('masing-masing', 'PROPN'), ('Dalam', 'PROPN'), ('video', 'PROPN'), ('pendek', 'PROPN'), ('terbitan', 'ADJ'), ('Jabatan', 'ADP'), ('Keselamatan', 'PROPN'), ('Jalan', 'NUM'), ('Raya', 'NUM'), ('JKJR', 'NUM'), ('itu', 'SYM'), ('Dr', 'X'), ('Mahathir', 'X'), ('menasihati', 'NUM'), ('mereka', 'NOUN'), ('supaya', 'NOUN'), ('berhenti', 'ADJ'), ('berehat', 'ADJ'), ('dan', 'CCONJ'), ('tidur', 'NOUN'), ('sebentar', 'NOUN'), ('sekiranya', 'NUM'), ('mengantuk', 'PROPN'), ('ketika', 'CCONJ'), ('memandu', 'ADV')]
    
    Testing attention model
    [('KUALA', 'PROPN'), ('LUMPUR', 'PROPN'), ('Sempena', 'PROPN'), ('sambutan', 'NOUN'), ('Aidilfitri', 'PROPN'), ('minggu', 'NOUN'), ('depan', 'ADJ'), ('Perdana', 'PROPN'), ('Menteri', 'PROPN'), ('Tun', 'PROPN'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('Mohamad', 'PROPN'), ('dan', 'CCONJ'), ('Menteri', 'PROPN'), ('Pengangkutan', 'PROPN'), ('Anthony', 'PROPN'), ('Loke', 'PROPN'), ('Siew', 'PROPN'), ('Fook', 'PROPN'), ('menitipkan', 'VERB'), ('pesanan', 'NOUN'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'ADJ'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'VERB'), ('Dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('Jabatan', 'NOUN'), ('Keselamatan', 'PROPN'), ('Jalan', 'PROPN'), ('Raya', 'PROPN'), ('JKJR', 'PROPN'), ('itu', 'DET'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('menasihati', 'VERB'), ('mereka', 'PRON'), ('supaya', 'SCONJ'), ('berhenti', 'VERB'), ('berehat', 'ADJ'), ('dan', 'CCONJ'), ('tidur', 'NOUN'), ('sebentar', 'ADV'), ('sekiranya', 'NOUN'), ('mengantuk', 'VERB'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]
    


Print important features from deep learning model
-------------------------------------------------

.. code:: ipython3

    bahdanau = malaya.pos.deep_model('bahdanau')
    bahdanau.print_features(10)


.. parsed-literal::

    Top-10 positive:
    tahapan: 4.671836
    Shanksville: 4.510098
    merekamnya: 4.445672
    basket: 4.269119
    perkuliahan: 4.235321
    Juventus: 4.220717
    Cigugur: 4.194372
    sekutu: 4.154349
    dipelihara: 4.075409
    dipacu: 4.054930
    
    Top-10 negative:
    Kkp: -4.665946
    Tryphon: -4.562038
    Tidung: -4.405613
    Dane: -4.368353
    merasakan: -4.307473
    Ina: -4.235865
    sekelompok: -4.183155
    Lionel: -4.140708
    Kibo: -4.140357
    Quena: -4.000028


Print important transitions from deep learning model
----------------------------------------------------

.. code:: ipython3

    bahdanau.print_transitions(10)


.. parsed-literal::

    Top-10 likely transitions:
    CCONJ -> SCONJ: 0.761498
    NUM -> SYM: 0.649993
    ADV -> NUM: 0.587261
    SCONJ -> CCONJ: 0.556012
    ADP -> NOUN: 0.532615
    VERB -> ADP: 0.463013
    SYM -> X: 0.460407
    ADJ -> ADP: 0.443898
    DET -> SCONJ: 0.406774
    PAD -> SYM: 0.394821
    
    Top-10 unlikely transitions:
    PRON -> SCONJ: -0.733985
    DET -> X: -0.727224
    SYM -> ADJ: -0.684060
    X -> SCONJ: -0.642626
    PART -> PART: -0.641473
    ADJ -> SYM: -0.636572
    SYM -> ADV: -0.634957
    ADP -> X: -0.620329
    PART -> DET: -0.597990
    DET -> NUM: -0.563087


Visualize output alignment from attention
-----------------------------------------

This visualization only can call from ``bahdanau`` or ``luong`` model.

.. code:: ipython3

    d_object, predicted, state_fw, state_bw = bahdanau.get_alignment(string)

.. code:: ipython3

    d_object.to_graphvis()




.. image:: load-pos_files/load-pos_21_0.svg



Voting stack model
------------------

.. code:: ipython3

    entity_network = malaya.pos.crf()
    bahdanau = malaya.pos.deep_model('bahdanau')
    luong = malaya.pos.deep_model('luong')
    malaya.stack.voting_stack([luong, bahdanau, crf, entity_network], string)




.. parsed-literal::

    [('KUALA', 'PROPN'),
     ('LUMPUR', 'PROPN'),
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
     ('Pengangkutan', 'NOUN'),
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
     ('JKJR', 'PROPN'),
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
     ('sebentar', 'ADJ'),
     ('sekiranya', 'NOUN'),
     ('mengantuk', 'VERB'),
     ('ketika', 'ADV'),
     ('memandu', 'VERB')]


