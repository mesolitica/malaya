
.. code:: python

    import malaya

List available deep learning POS models
---------------------------------------

.. code:: python

    malaya.pos.available_deep_model()




.. parsed-literal::

    ['concat', 'bahdanau', 'luong', 'entity-network', 'attention']



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

    crf = malaya.pos.crf()

.. code:: python

    string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'

.. code:: python

    crf.predict(string)




.. parsed-literal::

    [('kuala', 'PROPN'),
     ('lumpur', 'PROPN'),
     ('sempena', 'PROPN'),
     ('sambutan', 'NOUN'),
     ('aidilfitri', 'NOUN'),
     ('minggu', 'NOUN'),
     ('depan', 'ADJ'),
     ('perdana', 'PROPN'),
     ('menteri', 'PROPN'),
     ('tun', 'PROPN'),
     ('dr', 'PROPN'),
     ('mahathir', 'PROPN'),
     ('mohamad', 'PROPN'),
     ('dan', 'CCONJ'),
     ('menteri', 'VERB'),
     ('pengangkutan', 'PROPN'),
     ('anthony', 'PROPN'),
     ('loke', 'PROPN'),
     ('siew', 'PROPN'),
     ('fook', 'PROPN'),
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
     ('dalam', 'ADP'),
     ('video', 'NOUN'),
     ('pendek', 'ADJ'),
     ('terbitan', 'NOUN'),
     ('jabatan', 'NOUN'),
     ('keselamatan', 'PROPN'),
     ('jalan', 'PROPN'),
     ('raya', 'PROPN'),
     ('jkjr', 'PROPN'),
     ('itu', 'DET'),
     ('dr', 'PROPN'),
     ('mahathir', 'PROPN'),
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

    for i in malaya.pos.available_deep_model():
        print('Testing %s model'%(i))
        model = malaya.pos.deep_model(i)
        print(model.predict(string))
        print()


.. parsed-literal::

    Testing concat model
    [('kuala', 'PROPN'), ('lumpur', 'PROPN'), ('sempena', 'PROPN'), ('sambutan', 'NOUN'), ('aidilfitri', 'PROPN'), ('minggu', 'NOUN'), ('depan', 'ADJ'), ('perdana', 'ADJ'), ('menteri', 'NOUN'), ('tun', 'PROPN'), ('dr', 'PROPN'), ('mahathir', 'PROPN'), ('mohamad', 'PROPN'), ('dan', 'CCONJ'), ('menteri', 'NOUN'), ('pengangkutan', 'PROPN'), ('anthony', 'NOUN'), ('loke', 'NOUN'), ('siew', 'PROPN'), ('fook', 'PROPN'), ('menitipkan', 'PROPN'), ('pesanan', 'ADV'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'ADJ'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'NOUN'), ('dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('jabatan', 'NOUN'), ('keselamatan', 'NOUN'), ('jalan', 'PROPN'), ('raya', 'PROPN'), ('jkjr', 'PROPN'), ('itu', 'DET'), ('dr', 'PROPN'), ('mahathir', 'PROPN'), ('menasihati', 'NOUN'), ('mereka', 'PRON'), ('supaya', 'SCONJ'), ('berhenti', 'VERB'), ('berehat', 'PROPN'), ('dan', 'CCONJ'), ('tidur', 'NOUN'), ('sebentar', 'ADJ'), ('sekiranya', 'NOUN'), ('mengantuk', 'PROPN'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]

    Testing bahdanau model
    [('kuala', 'PROPN'), ('lumpur', 'PROPN'), ('sempena', 'PROPN'), ('sambutan', 'NOUN'), ('aidilfitri', 'PROPN'), ('minggu', 'PROPN'), ('depan', 'ADJ'), ('perdana', 'ADJ'), ('menteri', 'NOUN'), ('tun', 'PROPN'), ('dr', 'PROPN'), ('mahathir', 'PROPN'), ('mohamad', 'PROPN'), ('dan', 'CCONJ'), ('menteri', 'PROPN'), ('pengangkutan', 'PROPN'), ('anthony', 'PROPN'), ('loke', 'PROPN'), ('siew', 'PROPN'), ('fook', 'PROPN'), ('menitipkan', 'VERB'), ('pesanan', 'ADV'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'ADJ'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'PROPN'), ('dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('jabatan', 'NOUN'), ('keselamatan', 'NOUN'), ('jalan', 'PROPN'), ('raya', 'PROPN'), ('jkjr', 'PROPN'), ('itu', 'DET'), ('dr', 'PROPN'), ('mahathir', 'PROPN'), ('menasihati', 'VERB'), ('mereka', 'PRON'), ('supaya', 'SCONJ'), ('berhenti', 'VERB'), ('berehat', 'PROPN'), ('dan', 'CCONJ'), ('tidur', 'VERB'), ('sebentar', 'ADV'), ('sekiranya', 'PROPN'), ('mengantuk', 'PROPN'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]

    Testing luong model
    [('kuala', 'PROPN'), ('lumpur', 'PROPN'), ('sempena', 'PROPN'), ('sambutan', 'PROPN'), ('aidilfitri', 'PROPN'), ('minggu', 'PROPN'), ('depan', 'ADJ'), ('perdana', 'PROPN'), ('menteri', 'PROPN'), ('tun', 'PROPN'), ('dr', 'PROPN'), ('mahathir', 'PROPN'), ('mohamad', 'PROPN'), ('dan', 'CCONJ'), ('menteri', 'NOUN'), ('pengangkutan', 'NOUN'), ('anthony', 'PROPN'), ('loke', 'PROPN'), ('siew', 'PROPN'), ('fook', 'PROPN'), ('menitipkan', 'PROPN'), ('pesanan', 'NOUN'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'NOUN'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'NOUN'), ('dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('jabatan', 'NOUN'), ('keselamatan', 'NOUN'), ('jalan', 'NOUN'), ('raya', 'PROPN'), ('jkjr', 'PROPN'), ('itu', 'DET'), ('dr', 'PROPN'), ('mahathir', 'PROPN'), ('menasihati', 'PROPN'), ('mereka', 'PRON'), ('supaya', 'ADV'), ('berhenti', 'VERB'), ('berehat', 'NOUN'), ('dan', 'CCONJ'), ('tidur', 'VERB'), ('sebentar', 'ADV'), ('sekiranya', 'NOUN'), ('mengantuk', 'NOUN'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]

    Testing entity-network model
    [('kuala', 'PROPN'), ('lumpur', 'PROPN'), ('sempena', 'PROPN'), ('sambutan', 'PROPN'), ('aidilfitri', 'PROPN'), ('minggu', 'PROPN'), ('depan', 'PROPN'), ('perdana', 'PROPN'), ('menteri', 'PROPN'), ('tun', 'PROPN'), ('dr', 'PROPN'), ('mahathir', 'PROPN'), ('mohamad', 'PROPN'), ('dan', 'CCONJ'), ('menteri', 'PROPN'), ('pengangkutan', 'NOUN'), ('anthony', 'NOUN'), ('loke', 'NOUN'), ('siew', 'VERB'), ('fook', 'NOUN'), ('menitipkan', 'NOUN'), ('pesanan', 'VERB'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'ADJ'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'NOUN'), ('dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('jabatan', 'NOUN'), ('keselamatan', 'PROPN'), ('jalan', 'PROPN'), ('raya', 'PROPN'), ('jkjr', 'PROPN'), ('itu', 'DET'), ('dr', 'PROPN'), ('mahathir', 'VERB'), ('menasihati', 'PROPN'), ('mereka', 'PRON'), ('supaya', 'ADV'), ('berhenti', 'VERB'), ('berehat', 'NOUN'), ('dan', 'CCONJ'), ('tidur', 'NOUN'), ('sebentar', 'ADV'), ('sekiranya', 'NOUN'), ('mengantuk', 'ADJ'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]

    Testing attention model
    [('kuala', 'X'), ('lumpur', 'DET'), ('sempena', 'X'), ('sambutan', 'DET'), ('aidilfitri', 'X'), ('minggu', 'DET'), ('depan', 'X'), ('perdana', 'DET'), ('menteri', 'X'), ('tun', 'DET'), ('dr', 'X'), ('mahathir', 'DET'), ('mohamad', 'X'), ('dan', 'DET'), ('menteri', 'X'), ('pengangkutan', 'DET'), ('anthony', 'X'), ('loke', 'DET'), ('siew', 'X'), ('fook', 'DET'), ('menitipkan', 'X'), ('pesanan', 'DET'), ('khas', 'X'), ('kepada', 'DET'), ('orang', 'X'), ('ramai', 'DET'), ('yang', 'X'), ('mahu', 'DET'), ('pulang', 'X'), ('ke', 'DET'), ('kampung', 'X'), ('halaman', 'DET'), ('masing-masing', 'X'), ('dalam', 'DET'), ('video', 'X'), ('pendek', 'DET'), ('terbitan', 'X'), ('jabatan', 'DET'), ('keselamatan', 'X'), ('jalan', 'DET'), ('raya', 'X'), ('jkjr', 'DET'), ('itu', 'X'), ('dr', 'DET'), ('mahathir', 'X'), ('menasihati', 'DET'), ('mereka', 'X'), ('supaya', 'DET'), ('berhenti', 'X'), ('berehat', 'DET'), ('dan', 'X'), ('tidur', 'DET'), ('sebentar', 'X'), ('sekiranya', 'DET'), ('mengantuk', 'X'), ('ketika', 'DET'), ('memandu', 'VERB')]



Voting stack model
------------------

.. code:: python

    entity_network = malaya.pos.deep_model('entity-network')
    bahdanau = malaya.pos.deep_model('bahdanau')
    luong = malaya.pos.deep_model('luong')
    malaya.stack.voting_stack([entity_network, bahdanau, crf], string)




.. parsed-literal::

    [('kuala', 'PROPN'),
     ('lumpur', 'PROPN'),
     ('sempena', 'PROPN'),
     ('sambutan', 'NOUN'),
     ('aidilfitri', 'PROPN'),
     ('minggu', 'PROPN'),
     ('depan', 'ADJ'),
     ('perdana', 'PROPN'),
     ('menteri', 'PROPN'),
     ('tun', 'PROPN'),
     ('dr', 'PROPN'),
     ('mahathir', 'PROPN'),
     ('mohamad', 'PROPN'),
     ('dan', 'CCONJ'),
     ('menteri', 'NOUN'),
     ('pengangkutan', 'PROPN'),
     ('anthony', 'PROPN'),
     ('loke', 'PROPN'),
     ('siew', 'PROPN'),
     ('fook', 'NOUN'),
     ('menitipkan', 'PROPN'),
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
     ('dalam', 'ADP'),
     ('video', 'NOUN'),
     ('pendek', 'ADJ'),
     ('terbitan', 'NOUN'),
     ('jabatan', 'NOUN'),
     ('keselamatan', 'PROPN'),
     ('jalan', 'PROPN'),
     ('raya', 'PROPN'),
     ('jkjr', 'PROPN'),
     ('itu', 'DET'),
     ('dr', 'PROPN'),
     ('mahathir', 'PROPN'),
     ('menasihati', 'VERB'),
     ('mereka', 'PRON'),
     ('supaya', 'SCONJ'),
     ('berhenti', 'VERB'),
     ('berehat', 'VERB'),
     ('dan', 'CCONJ'),
     ('tidur', 'VERB'),
     ('sebentar', 'ADV'),
     ('sekiranya', 'NOUN'),
     ('mengantuk', 'PROPN'),
     ('ketika', 'SCONJ'),
     ('memandu', 'VERB')]
