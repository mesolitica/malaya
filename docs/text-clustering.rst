
.. code:: python

    import malaya

.. code:: python

    string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'

Cluster same word structure based on POS and Entities
-----------------------------------------------------

.. code:: python

    bahdanau_entities = malaya.entity.deep_model('bahdanau')
    bahdanau_pos = malaya.pos.deep_model('bahdanau')

.. code:: python

    result_entities = bahdanau_entities.predict(string)
    result_pos = bahdanau_pos.predict(string)

.. code:: python

    from malaya.cluster import cluster_words, pos_entities_ngram

.. code:: python

    result_pos




.. parsed-literal::

    [('Kuala', 'PROPN'),
     ('Lumpur', 'PROPN'),
     ('Sempena', 'PROPN'),
     ('sambutan', 'NOUN'),
     ('Aidilfitri', 'PROPN'),
     ('minggu', 'VERB'),
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
     ('mahu', 'VERB'),
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
     ('Keselamatan', 'NOUN'),
     ('Jalan', 'PROPN'),
     ('Raya', 'PROPN'),
     ('Jkjr', 'NOUN'),
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
     ('sebentar', 'ADV'),
     ('sekiranya', 'NOUN'),
     ('mengantuk', 'VERB'),
     ('ketika', 'SCONJ'),
     ('memandu', 'VERB')]



.. code:: python

    generated_grams = pos_entities_ngram(
        result_pos,
        result_entities,
        ngram = (1, 3),
        accept_pos = ['NOUN', 'PROPN', 'VERB'],
        accept_entities = ['law', 'location', 'organization', 'person', 'time'],
    )
    generated_grams




.. parsed-literal::

    ['terbitan Jabatan Keselamatan',
     'pesanan orang mahu',
     'Loke',
     'berehat tidur sekiranya',
     'sambutan Aidilfitri',
     'masing-masing',
     'terbitan',
     'halaman masing-masing',
     'Jabatan Keselamatan Jalan',
     'Dr Mahathir',
     'Dr Mahathir Mohamad',
     'pesanan',
     'Keselamatan',
     'Raya Jkjr',
     'terbitan Jabatan',
     'berhenti',
     'pesanan orang',
     'Aidilfitri minggu depan',
     'Kuala Lumpur Sempena',
     'Anthony Loke Siew',
     'Lumpur',
     'pulang kampung',
     'mahu pulang',
     'Mahathir',
     'Raya Jkjr Dr',
     'menasihati berhenti',
     'pulang kampung halaman',
     'Pengangkutan Anthony',
     'Mahathir menasihati',
     'Menteri',
     'Keselamatan Jalan',
     'Mohamad',
     'Dr',
     'Tun Dr Mahathir',
     'Fook menitipkan pesanan',
     'sekiranya mengantuk memandu',
     'Siew Fook menitipkan',
     'Sempena sambutan Aidilfitri',
     'Siew Fook',
     'mengantuk',
     'berhenti berehat',
     'sambutan',
     'Jalan Raya Jkjr',
     'Tun',
     'depan',
     'menasihati berhenti berehat',
     'Anthony Loke',
     'Pengangkutan',
     'Aidilfitri minggu',
     'Anthony',
     'Jkjr Dr',
     'Lumpur Sempena',
     'minggu depan Perdana',
     'Menteri Tun Dr',
     'Jkjr Dr Mahathir',
     'Fook',
     'Loke Siew',
     'tidur',
     'Menteri Tun',
     'Mahathir menasihati berhenti',
     'kampung halaman',
     'menitipkan',
     'Raya',
     'mahu',
     'Menteri Pengangkutan',
     'berehat tidur',
     'menitipkan pesanan orang',
     'menitipkan pesanan',
     'Sempena',
     'orang mahu pulang',
     'memandu',
     'kampung',
     'menasihati',
     'Kuala',
     'depan Perdana',
     'video',
     'Mohamad Menteri Pengangkutan',
     'Loke Siew Fook',
     'Mohamad Menteri',
     'masing-masing video terbitan',
     'halaman',
     'Menteri Pengangkutan Anthony',
     'orang mahu',
     'Siew',
     'sekiranya',
     'Perdana Menteri',
     'tidur sekiranya',
     'depan Perdana Menteri',
     'minggu depan',
     'video terbitan',
     'minggu',
     'Jabatan',
     'video terbitan Jabatan',
     'kampung halaman masing-masing',
     'sekiranya mengantuk',
     'mengantuk memandu',
     'Keselamatan Jalan Raya',
     'Tun Dr',
     'Kuala Lumpur',
     'Aidilfitri',
     'Mahathir Mohamad',
     'Mahathir Mohamad Menteri',
     'Fook menitipkan',
     'sambutan Aidilfitri minggu',
     'Jalan',
     'berehat',
     'Perdana',
     'Sempena sambutan',
     'mahu pulang kampung',
     'Perdana Menteri Tun',
     'Jkjr',
     'halaman masing-masing video',
     'tidur sekiranya mengantuk',
     'pulang',
     'Jabatan Keselamatan',
     'orang',
     'Dr Mahathir menasihati',
     'Pengangkutan Anthony Loke',
     'Lumpur Sempena sambutan',
     'masing-masing video',
     'berhenti berehat tidur',
     'Jalan Raya']



.. code:: python

    cluster_words(generated_grams)




.. parsed-literal::

    ['terbitan Jabatan Keselamatan',
     'orang mahu pulang',
     'video terbitan Jabatan',
     'pesanan orang mahu',
     'kampung halaman masing-masing',
     'Tun Dr Mahathir',
     'berehat tidur sekiranya',
     'Fook menitipkan pesanan',
     'sekiranya mengantuk memandu',
     'Siew Fook menitipkan',
     'Keselamatan Jalan Raya',
     'Jabatan Keselamatan Jalan',
     'Sempena sambutan Aidilfitri',
     'Dr Mahathir Mohamad',
     'Jalan Raya Jkjr',
     'Mohamad Menteri Pengangkutan',
     'Mahathir Mohamad Menteri',
     'menasihati berhenti berehat',
     'sambutan Aidilfitri minggu',
     'Aidilfitri minggu depan',
     'Kuala Lumpur Sempena',
     'Anthony Loke Siew',
     'Loke Siew Fook',
     'masing-masing video terbitan',
     'minggu depan Perdana',
     'Menteri Pengangkutan Anthony',
     'Menteri Tun Dr',
     'Jkjr Dr Mahathir',
     'mahu pulang kampung',
     'Mahathir menasihati berhenti',
     'Perdana Menteri Tun',
     'Raya Jkjr Dr',
     'halaman masing-masing video',
     'tidur sekiranya mengantuk',
     'depan Perdana Menteri',
     'pulang kampung halaman',
     'Dr Mahathir menasihati',
     'Pengangkutan Anthony Loke',
     'Lumpur Sempena sambutan',
     'berhenti berehat tidur',
     'menitipkan pesanan orang']



Cluster POS and Entities
------------------------

.. code:: python

    from malaya.cluster import cluster_pos, cluster_entities

.. code:: python

    cluster_pos(result_pos)




.. parsed-literal::

    {'ADJ': ['depan', 'khas', 'ramai', 'pendek'],
     'ADP': ['kepada', 'ke', 'Dalam'],
     'ADV': ['sebentar'],
     'ADX': [],
     'CCONJ': ['dan'],
     'DET': ['itu'],
     'NOUN': ['sambutan',
      'pesanan',
      'orang',
      'kampung halaman masing-masing',
      'video',
      'terbitan Jabatan Keselamatan',
      'Jkjr',
      'berehat',
      'tidur',
      'sekiranya'],
     'NUM': [],
     'PART': [],
     'PRON': ['yang', 'mereka'],
     'PROPN': ['Kuala Lumpur Sempena',
      'Aidilfitri',
      'Perdana Menteri Tun Dr Mahathir Mohamad',
      'Menteri Pengangkutan Anthony Loke Siew Fook',
      'Jalan Raya',
      'Dr Mahathir'],
     'SCONJ': ['supaya', 'ketika'],
     'SYM': [],
     'VERB': ['minggu',
      'menitipkan',
      'mahu pulang',
      'menasihati',
      'berhenti',
      'mengantuk'],
     'X': []}



.. code:: python

    cluster_entities(result_entities)




.. parsed-literal::

    {'OTHER': ['sempena',
      'dan',
      'menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing dalam video pendek terbitan',
      'itu'],
     'law': [],
     'location': ['kuala lumpur'],
     'organization': ['menteri pengangkutan', 'jabatan keselamatan jalan raya'],
     'person': ['perdana menteri tun dr mahathir mohamad',
      'anthony loke siew fook',
      'jkjr',
      'dr mahathir'],
     'quantity': [],
     'time': ['minggu depan'],
     'event': ['sambutan aidilfitri']}



Generate ngrams
---------------

.. code:: python

    from malaya.cluster import sentence_ngram

.. code:: python

    sentence_ngram(string, ngram = (3, 5))




.. parsed-literal::

    ['terbitan Jabatan Keselamatan',
     'sebentar sekiranya mengantuk',
     'mahu pulang ke',
     'Menteri Tun Dr Mahathir Mohamad',
     'Jabatan Keselamatan Jalan Raya (JKJR)',
     'Aidilfitri minggu depan, Perdana',
     'pulang ke kampung halaman masing-masing.',
     'Mahathir Mohamad dan Menteri',
     'tidur sebentar sekiranya mengantuk',
     'berhenti berehat dan tidur sebentar',
     'Jabatan Keselamatan Jalan',
     'menitipkan pesanan khas kepada orang',
     'Dalam video pendek terbitan',
     'Raya (JKJR) itu, Dr Mahathir',
     'Dr Mahathir Mohamad',
     'Anthony Loke Siew Fook',
     'masing-masing. Dalam video pendek',
     '(JKJR) itu, Dr',
     'pesanan khas kepada orang',
     'Dr Mahathir Mohamad dan Menteri',
     'Anthony Loke Siew',
     'tidur sebentar sekiranya',
     'LUMPUR: Sempena sambutan',
     'Pengangkutan Anthony Loke Siew Fook',
     'Aidilfitri minggu depan, Perdana Menteri',
     'Tun Dr Mahathir Mohamad',
     'sebentar sekiranya mengantuk ketika memandu.',
     'Pengangkutan Anthony Loke Siew',
     'supaya berhenti berehat',
     'halaman masing-masing. Dalam video',
     'Sempena sambutan Aidilfitri minggu',
     'masing-masing. Dalam video pendek terbitan',
     'khas kepada orang ramai',
     'dan tidur sebentar sekiranya mengantuk',
     'berhenti berehat dan',
     'Menteri Pengangkutan Anthony Loke',
     'Jalan Raya (JKJR) itu, Dr',
     'depan, Perdana Menteri',
     'Tun Dr Mahathir',
     'pulang ke kampung',
     'masing-masing. Dalam video',
     'berehat dan tidur',
     'KUALA LUMPUR: Sempena sambutan Aidilfitri',
     'KUALA LUMPUR: Sempena',
     'Fook menitipkan pesanan',
     'Dr Mahathir Mohamad dan',
     'terbitan Jabatan Keselamatan Jalan Raya',
     'Siew Fook menitipkan',
     'KUALA LUMPUR: Sempena sambutan',
     'Sempena sambutan Aidilfitri',
     'dan tidur sebentar',
     'halaman masing-masing. Dalam video pendek',
     'Mahathir menasihati mereka',
     'Aidilfitri minggu depan,',
     'sekiranya mengantuk ketika memandu.',
     'pulang ke kampung halaman',
     'pesanan khas kepada',
     'Keselamatan Jalan Raya (JKJR)',
     'Menteri Pengangkutan Anthony Loke Siew',
     'dan Menteri Pengangkutan Anthony',
     'Dalam video pendek',
     'Fook menitipkan pesanan khas',
     'Perdana Menteri Tun Dr',
     'ke kampung halaman masing-masing.',
     'Menteri Tun Dr',
     'kampung halaman masing-masing. Dalam video',
     'Mahathir Mohamad dan',
     'Mahathir menasihati mereka supaya',
     'LUMPUR: Sempena sambutan Aidilfitri minggu',
     '(JKJR) itu, Dr Mahathir menasihati',
     'Siew Fook menitipkan pesanan khas',
     'menitipkan pesanan khas kepada',
     'video pendek terbitan',
     'kampung halaman masing-masing. Dalam',
     'itu, Dr Mahathir menasihati mereka',
     'itu, Dr Mahathir',
     'orang ramai yang',
     'ke kampung halaman',
     'kepada orang ramai yang mahu',
     'yang mahu pulang',
     'Dr Mahathir menasihati mereka',
     'video pendek terbitan Jabatan Keselamatan',
     'ke kampung halaman masing-masing. Dalam',
     'orang ramai yang mahu',
     'Mohamad dan Menteri Pengangkutan',
     'khas kepada orang',
     'Mohamad dan Menteri',
     'berhenti berehat dan tidur',
     'ramai yang mahu pulang',
     'tidur sebentar sekiranya mengantuk ketika',
     'Loke Siew Fook menitipkan pesanan',
     'menasihati mereka supaya',
     'Menteri Tun Dr Mahathir',
     'pendek terbitan Jabatan Keselamatan',
     'Raya (JKJR) itu, Dr',
     'pesanan khas kepada orang ramai',
     'ramai yang mahu',
     'dan tidur sebentar sekiranya',
     'minggu depan, Perdana Menteri',
     'video pendek terbitan Jabatan',
     'dan Menteri Pengangkutan',
     'Tun Dr Mahathir Mohamad dan',
     'yang mahu pulang ke',
     'Mahathir Mohamad dan Menteri Pengangkutan',
     'Anthony Loke Siew Fook menitipkan',
     'sambutan Aidilfitri minggu depan, Perdana',
     'Loke Siew Fook',
     'supaya berhenti berehat dan',
     'Menteri Pengangkutan Anthony',
     'menitipkan pesanan khas',
     'Dr Mahathir menasihati mereka supaya',
     'Siew Fook menitipkan pesanan',
     'Loke Siew Fook menitipkan',
     'Mohamad dan Menteri Pengangkutan Anthony',
     'sebentar sekiranya mengantuk ketika',
     'menasihati mereka supaya berhenti',
     'Jalan Raya (JKJR) itu,',
     'itu, Dr Mahathir menasihati',
     'pendek terbitan Jabatan Keselamatan Jalan',
     'mahu pulang ke kampung halaman',
     'minggu depan, Perdana Menteri Tun',
     'Sempena sambutan Aidilfitri minggu depan,',
     'orang ramai yang mahu pulang',
     'halaman masing-masing. Dalam',
     'Dalam video pendek terbitan Jabatan',
     'Jalan Raya (JKJR)',
     'Keselamatan Jalan Raya',
     'mereka supaya berhenti',
     'mahu pulang ke kampung',
     'pendek terbitan Jabatan',
     'berehat dan tidur sebentar',
     'minggu depan, Perdana',
     'kampung halaman masing-masing.',
     'supaya berhenti berehat dan tidur',
     'sambutan Aidilfitri minggu depan,',
     'Perdana Menteri Tun Dr Mahathir',
     'sambutan Aidilfitri minggu',
     'Mahathir menasihati mereka supaya berhenti',
     'mereka supaya berhenti berehat dan',
     'berehat dan tidur sebentar sekiranya',
     'depan, Perdana Menteri Tun Dr',
     'depan, Perdana Menteri Tun',
     'mengantuk ketika memandu.',
     'sekiranya mengantuk ketika',
     'dan Menteri Pengangkutan Anthony Loke',
     'kepada orang ramai yang',
     'Keselamatan Jalan Raya (JKJR) itu,',
     'LUMPUR: Sempena sambutan Aidilfitri',
     'Perdana Menteri Tun',
     '(JKJR) itu, Dr Mahathir',
     'yang mahu pulang ke kampung',
     'Raya (JKJR) itu,',
     'terbitan Jabatan Keselamatan Jalan',
     'kepada orang ramai',
     'Jabatan Keselamatan Jalan Raya',
     'Fook menitipkan pesanan khas kepada',
     'Dr Mahathir menasihati',
     'menasihati mereka supaya berhenti berehat',
     'Pengangkutan Anthony Loke',
     'ramai yang mahu pulang ke',
     'khas kepada orang ramai yang',
     'mereka supaya berhenti berehat']
