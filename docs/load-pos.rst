
.. code:: python

    import malaya

List available deep learning POS models
---------------------------------------

.. code:: python

    malaya.get_available_pos_models()




.. parsed-literal::

    ['concat', 'bahdanau', 'luong', 'entity-network']



Describe supported POS
----------------------

.. code:: python

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

.. code:: python

    crf = malaya.crf_pos()

.. code:: python

    string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'

.. code:: python

    crf.predict(string)




.. parsed-literal::

    [('Kuala', 'PROPN'),
     ('Lumpur', 'PROPN'),
     ('Sempena', 'CCONJ'),
     ('sambutan', 'NOUN'),
     ('Aidilfitri', 'NOUN'),
     ('minggu', 'NOUN'),
     ('depan', 'ADJ'),
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
     ('Dalam', 'NOUN'),
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
     ('ketika', 'SCONJ'),
     ('memandu', 'VERB')]



Print important features CRF model
----------------------------------

.. code:: python

    crf.print_features(10)


.. parsed-literal::

    Top-10 positive:
    16.307872 DET      word:tersebut
    15.868179 DET      word:para
    15.590679 VERB     word:percaya
    15.520492 ADP      word:dari
    15.296975 DET      word:berbagai
    14.691924 ADJ      word:menakjubkan
    14.609917 ADJ      word:menyejukkan
    14.503045 PRON     word:kapan
    14.319357 DET      word:ini
    14.267956 ADV      word:pernah

    Top-10 negative:
    -7.217718 PROPN    word:bunga
    -7.258999 VERB     word:memuaskan
    -7.498110 ADP      prev_word:pernah
    -7.523901 ADV      next_word-suffix-3:nai
    -7.874955 NOUN     prev_word-prefix-3:arw
    -7.921689 NOUN     suffix-2:ke
    -8.049832 ADJ      prev_word:sunda
    -8.210202 PROPN    prefix-3:ora
    -8.524420 NUM      prev_word:perang
    -10.346546 CCONJ    prev_word-suffix-3:rja


Print important transitions CRF model
-------------------------------------

.. code:: python

    crf.print_transitions(10)


.. parsed-literal::

    Top-10 likely transitions:
    PROPN  -> PROPN   5.529614
    DET    -> DET     4.492123
    NOUN   -> NOUN    2.600533
    ADJ    -> ADJ     2.276762
    CCONJ  -> CCONJ   1.888801
    CCONJ  -> SCONJ   1.855106
    NOUN   -> ADJ     1.729610
    SCONJ  -> CCONJ   1.598273
    NUM    -> NUM     1.475505
    ADV    -> VERB    1.442607

    Top-10 unlikely transitions:
    SCONJ  -> AUX     -3.559017
    X      -> SCONJ   -3.566058
    SYM    -> ADJ     -3.720358
    PART   -> ADP     -3.744172
    X      -> CCONJ   -4.270577
    PART   -> PART    -4.543812
    ADV    -> X       -4.809254
    ADP    -> SCONJ   -5.157816
    ADP    -> CCONJ   -5.455725
    ADP    -> SYM     -6.841944


Load deep learning models
-------------------------

.. code:: python

    for i in malaya.get_available_pos_models():
        print('Testing %s model'%(i))
        model = malaya.deep_pos(i)
        print(model.predict(string))
        print()


.. parsed-literal::

    Testing concat model
    [('Kuala', 'PROPN'), ('Lumpur', 'PROPN'), ('Sempena', 'VERB'), ('sambutan', 'NOUN'), ('Aidilfitri', 'NOUN'), ('minggu', 'NOUN'), ('depan', 'ADJ'), ('Perdana', 'NOUN'), ('Menteri', 'PROPN'), ('Tun', 'PROPN'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('Mohamad', 'PROPN'), ('dan', 'CCONJ'), ('Menteri', 'NOUN'), ('Pengangkutan', 'PROPN'), ('Anthony', 'PROPN'), ('Loke', 'PROPN'), ('Siew', 'PROPN'), ('Fook', 'PROPN'), ('menitipkan', 'NOUN'), ('pesanan', 'VERB'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'ADJ'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'PROPN'), ('Dalam', 'NOUN'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('Jabatan', 'PROPN'), ('Keselamatan', 'PROPN'), ('Jalan', 'PROPN'), ('Raya', 'PROPN'), ('Jkjr', 'NOUN'), ('itu', 'DET'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('menasihati', 'VERB'), ('mereka', 'PRON'), ('supaya', 'SCONJ'), ('berhenti', 'VERB'), ('berehat', 'PROPN'), ('dan', 'CCONJ'), ('tidur', 'VERB'), ('sebentar', 'ADJ'), ('sekiranya', 'NOUN'), ('mengantuk', 'PROPN'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]

    Testing bahdanau model
    [('Kuala', 'PROPN'), ('Lumpur', 'PROPN'), ('Sempena', 'PRON'), ('sambutan', 'NOUN'), ('Aidilfitri', 'PROPN'), ('minggu', 'PROPN'), ('depan', 'ADJ'), ('Perdana', 'PROPN'), ('Menteri', 'PROPN'), ('Tun', 'PROPN'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('Mohamad', 'PROPN'), ('dan', 'CCONJ'), ('Menteri', 'PROPN'), ('Pengangkutan', 'PROPN'), ('Anthony', 'PROPN'), ('Loke', 'PROPN'), ('Siew', 'PROPN'), ('Fook', 'PRON'), ('menitipkan', 'PROPN'), ('pesanan', 'ADP'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'ADJ'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'PROPN'), ('Dalam', 'PROPN'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('Jabatan', 'NOUN'), ('Keselamatan', 'PROPN'), ('Jalan', 'PROPN'), ('Raya', 'PROPN'), ('Jkjr', 'PROPN'), ('itu', 'DET'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('menasihati', 'VERB'), ('mereka', 'PRON'), ('supaya', 'PART'), ('berhenti', 'VERB'), ('berehat', 'PROPN'), ('dan', 'CCONJ'), ('tidur', 'VERB'), ('sebentar', 'ADV'), ('sekiranya', 'PROPN'), ('mengantuk', 'PROPN'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]

    Testing luong model
    downloading POS frozen luong model


.. parsed-literal::

    13.0MB [00:04, 2.92MB/s]


.. parsed-literal::

    downloading POS luong dictionary


.. parsed-literal::

    1.00MB [00:00, 3.15MB/s]


.. parsed-literal::

    [('Kuala', 'PROPN'), ('Lumpur', 'PROPN'), ('Sempena', 'PROPN'), ('sambutan', 'PROPN'), ('Aidilfitri', 'PROPN'), ('minggu', 'PROPN'), ('depan', 'ADP'), ('Perdana', 'PROPN'), ('Menteri', 'PROPN'), ('Tun', 'PROPN'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('Mohamad', 'PROPN'), ('dan', 'CCONJ'), ('Menteri', 'PROPN'), ('Pengangkutan', 'PROPN'), ('Anthony', 'PROPN'), ('Loke', 'PROPN'), ('Siew', 'PROPN'), ('Fook', 'PROPN'), ('menitipkan', 'PROPN'), ('pesanan', 'PROPN'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'NOUN'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'NOUN'), ('Dalam', 'NOUN'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('Jabatan', 'PROPN'), ('Keselamatan', 'PROPN'), ('Jalan', 'PROPN'), ('Raya', 'PROPN'), ('Jkjr', 'PROPN'), ('itu', 'DET'), ('Dr', 'PROPN'), ('Mahathir', 'PROPN'), ('menasihati', 'PROPN'), ('mereka', 'PRON'), ('supaya', 'SCONJ'), ('berhenti', 'VERB'), ('berehat', 'NOUN'), ('dan', 'CCONJ'), ('tidur', 'NOUN'), ('sebentar', 'ADV'), ('sekiranya', 'PROPN'), ('mengantuk', 'NOUN'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]

    Testing entity-network model
    downloading POS frozen entity-network model


.. parsed-literal::

    11.0MB [00:03, 3.21MB/s]
      0%|          | 0.00/0.69 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading POS entity-network dictionary


.. parsed-literal::

    1.00MB [00:00, 2.44MB/s]


.. parsed-literal::

    [('Kuala', 'NOUN'), ('Lumpur', 'DET'), ('Sempena', 'NOUN'), ('sambutan', 'NOUN'), ('Aidilfitri', 'NOUN'), ('minggu', 'PROPN'), ('depan', 'SCONJ'), ('Perdana', 'VERB'), ('Menteri', 'PROPN'), ('Tun', 'VERB'), ('Dr', 'NOUN'), ('Mahathir', 'ADV'), ('Mohamad', 'VERB'), ('dan', 'CCONJ'), ('Menteri', 'ADV'), ('Pengangkutan', 'ADV'), ('Anthony', 'VERB'), ('Loke', 'X'), ('Siew', 'X'), ('Fook', 'DET'), ('menitipkan', 'VERB'), ('pesanan', 'DET'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'ADJ'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'NOUN'), ('Dalam', 'ADV'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'ADP'), ('Jabatan', 'NOUN'), ('Keselamatan', 'PROPN'), ('Jalan', 'PROPN'), ('Raya', 'PROPN'), ('Jkjr', 'PROPN'), ('itu', 'DET'), ('Dr', 'VERB'), ('Mahathir', 'PROPN'), ('menasihati', 'PROPN'), ('mereka', 'PRON'), ('supaya', 'ADV'), ('berhenti', 'VERB'), ('berehat', 'NOUN'), ('dan', 'CCONJ'), ('tidur', 'NOUN'), ('sebentar', 'ADV'), ('sekiranya', 'VERB'), ('mengantuk', 'ADV'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]



Voting stack model
------------------

.. code:: python

    entity_network = malaya.deep_pos('entity-network')
    bahdanau = malaya.deep_pos('bahdanau')
    malaya.voting_stack([entity_network, bahdanau, crf], string)




.. parsed-literal::

    [('Kuala', 'PROPN'),
     ('Lumpur', 'PROPN'),
     ('Sempena', 'SCONJ'),
     ('sambutan', 'NOUN'),
     ('Aidilfitri', 'NOUN'),
     ('minggu', 'PROPN'),
     ('depan', 'ADJ'),
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
     ('pesanan', 'VERB'),
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
     ('Dalam', 'NOUN'),
     ('video', 'NOUN'),
     ('pendek', 'ADJ'),
     ('terbitan', 'NOUN'),
     ('Jabatan', 'NOUN'),
     ('Keselamatan', 'PROPN'),
     ('Jalan', 'PROPN'),
     ('Raya', 'PROPN'),
     ('Jkjr', 'DET'),
     ('itu', 'DET'),
     ('Dr', 'VERB'),
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
     ('ketika', 'SCONJ'),
     ('memandu', 'VERB')]
