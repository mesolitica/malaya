
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

    generated_grams = pos_entities_ngram(
        result_pos,
        result_entities,
        ngram = (1, 3),
        accept_pos = ['NOUN', 'PROPN', 'VERB'],
        accept_entities = ['law', 'location', 'organization', 'person', 'time'],
    )
    generated_grams




.. parsed-literal::

    ['pengangkutan anthony',
     'khas orang pulang',
     'terbitan jabatan keselamatan',
     'dr mahathir menasihati',
     'tidur sekiranya mengantuk',
     'kuala',
     'raya',
     'berehat',
     'jalan',
     'berhenti',
     'video',
     'pulang kampung',
     'dr',
     'menasihati',
     'mengantuk',
     'depan perdana menteri',
     'memandu',
     'mohamad menteri pengangkutan',
     'jabatan',
     'siew',
     'mahathir menasihati',
     'menteri tun dr',
     'kampung halaman masing-masing',
     'menteri tun',
     'sekiranya mengantuk memandu',
     'khas',
     'mohamad menteri',
     'halaman',
     'berhenti berehat',
     'khas orang',
     'jalan raya',
     'raya jkjr dr',
     'terbitan',
     'pesanan',
     'minggu depan',
     'siew fook menitipkan',
     'orang',
     'halaman masing-masing',
     'mengantuk memandu',
     'fook menitipkan pesanan',
     'mohamad',
     'kampung',
     'fook menitipkan',
     'video terbitan',
     'minggu',
     'video terbitan jabatan',
     'kuala lumpur sempena',
     'tun dr',
     'jkjr dr mahathir',
     'sambutan aidilfitri minggu',
     'berhenti berehat tidur',
     'loke siew fook',
     'loke siew',
     'kuala lumpur',
     'raya jkjr',
     'fook',
     'pengangkutan',
     'perdana menteri',
     'tun dr mahathir',
     'keselamatan jalan',
     'depan perdana',
     'orang pulang',
     'halaman masing-masing video',
     'jkjr',
     'tidur',
     'aidilfitri minggu depan',
     'sempena sambutan aidilfitri',
     'sekiranya mengantuk',
     'mahathir',
     'terbitan jabatan',
     'menitipkan pesanan khas',
     'menteri pengangkutan',
     'dr mahathir',
     'masing-masing video',
     'menteri',
     'sempena sambutan',
     'depan',
     'berehat tidur',
     'dr mahathir mohamad',
     'menasihati berhenti',
     'orang pulang kampung',
     'aidilfitri minggu',
     'masing-masing video terbitan',
     'masing-masing',
     'tidur sekiranya',
     'menteri pengangkutan anthony',
     'kampung halaman',
     'mahathir mohamad',
     'lumpur',
     'sempena',
     'perdana',
     'mahathir menasihati berhenti',
     'siew fook',
     'anthony',
     'sekiranya',
     'keselamatan',
     'sambutan aidilfitri',
     'minggu depan perdana',
     'pesanan khas',
     'tun',
     'jalan raya jkjr',
     'lumpur sempena sambutan',
     'jabatan keselamatan',
     'jkjr dr',
     'pengangkutan anthony loke',
     'pulang',
     'sambutan',
     'anthony loke',
     'jabatan keselamatan jalan',
     'menitipkan',
     'pulang kampung halaman',
     'berehat tidur sekiranya',
     'aidilfitri',
     'perdana menteri tun',
     'pesanan khas orang',
     'menasihati berhenti berehat',
     'loke',
     'anthony loke siew',
     'lumpur sempena',
     'keselamatan jalan raya',
     'mahathir mohamad menteri',
     'menitipkan pesanan']



.. code:: python

    cluster_words(generated_grams)




.. parsed-literal::

    ['khas orang pulang',
     'terbitan jabatan keselamatan',
     'sekiranya mengantuk memandu',
     'orang pulang kampung',
     'masing-masing video terbitan',
     'dr mahathir menasihati',
     'menteri pengangkutan anthony',
     'tidur sekiranya mengantuk',
     'mahathir menasihati berhenti',
     'tun dr mahathir',
     'minggu depan perdana',
     'raya jkjr dr',
     'halaman masing-masing video',
     'siew fook menitipkan',
     'jalan raya jkjr',
     'menteri tun dr',
     'lumpur sempena sambutan',
     'aidilfitri minggu depan',
     'sempena sambutan aidilfitri',
     'pengangkutan anthony loke',
     'fook menitipkan pesanan',
     'jabatan keselamatan jalan',
     'menitipkan pesanan khas',
     'depan perdana menteri',
     'pulang kampung halaman',
     'video terbitan jabatan',
     'kuala lumpur sempena',
     'berehat tidur sekiranya',
     'mohamad menteri pengangkutan',
     'menasihati berhenti berehat',
     'jkjr dr mahathir',
     'perdana menteri tun',
     'pesanan khas orang',
     'sambutan aidilfitri minggu',
     'anthony loke siew',
     'berhenti berehat tidur',
     'loke siew fook',
     'dr mahathir mohamad',
     'keselamatan jalan raya',
     'kampung halaman masing-masing',
     'mahathir mohamad menteri']



Cluster POS and Entities
------------------------

.. code:: python

    from malaya.cluster import cluster_pos, cluster_entities

.. code:: python

    cluster_pos(result_pos)




.. parsed-literal::

    {'ADJ': ['depan perdana', 'khas', 'ramai', 'pendek'],
     'ADP': ['kepada', 'ke', 'dalam'],
     'ADV': ['mahu', 'sebentar'],
     'ADX': [],
     'CCONJ': ['dan'],
     'DET': ['itu'],
     'NOUN': ['sambutan',
      'menteri',
      'menitipkan',
      'orang',
      'kampung halaman',
      'video',
      'terbitan jabatan keselamatan jalan raya jkjr'],
     'NUM': [],
     'PART': [],
     'PRON': ['yang', 'mereka'],
     'PROPN': ['kuala lumpur sempena',
      'aidilfitri minggu',
      'tun dr mahathir mohamad',
      'pengangkutan anthony loke siew fook',
      'masing-masing',
      'dr mahathir',
      'berehat',
      'sekiranya mengantuk'],
     'SCONJ': ['supaya', 'ketika'],
     'SYM': [],
     'VERB': ['pesanan', 'pulang', 'menasihati', 'berhenti', 'tidur'],
     'X': []}



.. code:: python

    cluster_entities(result_entities)




.. parsed-literal::

    {'OTHER': ['sempena',
      'dan',
      'pengangkutan',
      'menitipkan pesanan',
      'kepada',
      'ramai yang mahu pulang ke kampung',
      'masing-masing dalam video pendek terbitan',
      'itu'],
     'law': [],
     'location': ['kuala lumpur', 'halaman'],
     'organization': ['khas', 'orang', 'jabatan keselamatan jalan raya jkjr'],
     'person': ['perdana menteri tun dr mahathir mohamad',
      'menteri',
      'anthony loke siew fook',
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

    ['video pendek terbitan',
     'menitipkan pesanan khas kepada',
     'tidur sebentar sekiranya',
     'sekiranya mengantuk ketika',
     '(JKJR) itu, Dr',
     'supaya berhenti berehat dan',
     'yang mahu pulang ke',
     'halaman masing-masing. Dalam video pendek',
     'pendek terbitan Jabatan Keselamatan',
     'Loke Siew Fook',
     'ramai yang mahu',
     'mengantuk ketika memandu.',
     'Jabatan Keselamatan Jalan Raya',
     'ke kampung halaman masing-masing. Dalam',
     'berhenti berehat dan',
     'LUMPUR: Sempena sambutan Aidilfitri',
     'minggu depan, Perdana',
     'Raya (JKJR) itu, Dr',
     'Sempena sambutan Aidilfitri minggu depan,',
     'Mohamad dan Menteri Pengangkutan Anthony',
     'Dr Mahathir Mohamad',
     'masing-masing. Dalam video pendek terbitan',
     'depan, Perdana Menteri',
     'menasihati mereka supaya berhenti',
     'Dalam video pendek terbitan Jabatan',
     'sebentar sekiranya mengantuk ketika memandu.',
     'ke kampung halaman masing-masing.',
     'sebentar sekiranya mengantuk ketika',
     'Tun Dr Mahathir Mohamad',
     'Loke Siew Fook menitipkan',
     'tidur sebentar sekiranya mengantuk ketika',
     'orang ramai yang',
     'Mohamad dan Menteri Pengangkutan',
     'kepada orang ramai yang',
     'minggu depan, Perdana Menteri Tun',
     'video pendek terbitan Jabatan Keselamatan',
     'menitipkan pesanan khas kepada orang',
     'Pengangkutan Anthony Loke Siew',
     'Mahathir menasihati mereka supaya',
     'Siew Fook menitipkan',
     'LUMPUR: Sempena sambutan',
     'sambutan Aidilfitri minggu',
     'KUALA LUMPUR: Sempena sambutan Aidilfitri',
     'Aidilfitri minggu depan, Perdana',
     'Anthony Loke Siew Fook menitipkan',
     'KUALA LUMPUR: Sempena',
     'kampung halaman masing-masing. Dalam',
     'pesanan khas kepada',
     'dan tidur sebentar sekiranya mengantuk',
     'pendek terbitan Jabatan',
     'supaya berhenti berehat',
     'Mahathir menasihati mereka',
     'Tun Dr Mahathir Mohamad dan',
     'berehat dan tidur',
     'Siew Fook menitipkan pesanan',
     'Perdana Menteri Tun',
     'ke kampung halaman',
     'khas kepada orang ramai',
     'masing-masing. Dalam video pendek',
     'Keselamatan Jalan Raya',
     'pendek terbitan Jabatan Keselamatan Jalan',
     'Menteri Tun Dr Mahathir Mohamad',
     'kepada orang ramai',
     'berhenti berehat dan tidur sebentar',
     '(JKJR) itu, Dr Mahathir menasihati',
     'kampung halaman masing-masing.',
     'Jalan Raya (JKJR) itu,',
     'dan tidur sebentar',
     'mahu pulang ke kampung',
     'Siew Fook menitipkan pesanan khas',
     'itu, Dr Mahathir',
     'sebentar sekiranya mengantuk',
     'Sempena sambutan Aidilfitri minggu',
     'Perdana Menteri Tun Dr Mahathir',
     'menasihati mereka supaya berhenti berehat',
     'halaman masing-masing. Dalam',
     'KUALA LUMPUR: Sempena sambutan',
     'Tun Dr Mahathir',
     'video pendek terbitan Jabatan',
     'khas kepada orang',
     'Jabatan Keselamatan Jalan',
     '(JKJR) itu, Dr Mahathir',
     'Anthony Loke Siew',
     'Menteri Pengangkutan Anthony Loke Siew',
     'itu, Dr Mahathir menasihati mereka',
     'Mohamad dan Menteri',
     'orang ramai yang mahu',
     'dan Menteri Pengangkutan',
     'tidur sebentar sekiranya mengantuk',
     'mereka supaya berhenti',
     'Mahathir Mohamad dan Menteri',
     'khas kepada orang ramai yang',
     'Jalan Raya (JKJR)',
     'ramai yang mahu pulang ke',
     'LUMPUR: Sempena sambutan Aidilfitri minggu',
     'kampung halaman masing-masing. Dalam video',
     'itu, Dr Mahathir menasihati',
     'Anthony Loke Siew Fook',
     'Sempena sambutan Aidilfitri',
     'mahu pulang ke',
     'terbitan Jabatan Keselamatan Jalan Raya',
     'kepada orang ramai yang mahu',
     'Menteri Tun Dr',
     'depan, Perdana Menteri Tun Dr',
     'Dr Mahathir menasihati mereka supaya',
     'halaman masing-masing. Dalam video',
     'pesanan khas kepada orang ramai',
     'sekiranya mengantuk ketika memandu.',
     'dan tidur sebentar sekiranya',
     'yang mahu pulang',
     'terbitan Jabatan Keselamatan Jalan',
     'dan Menteri Pengangkutan Anthony Loke',
     'menitipkan pesanan khas',
     'ramai yang mahu pulang',
     'Loke Siew Fook menitipkan pesanan',
     'mereka supaya berhenti berehat dan',
     'pulang ke kampung',
     'dan Menteri Pengangkutan Anthony',
     'Raya (JKJR) itu, Dr Mahathir',
     'Dalam video pendek terbitan',
     'Jabatan Keselamatan Jalan Raya (JKJR)',
     'Fook menitipkan pesanan',
     'Raya (JKJR) itu,',
     'supaya berhenti berehat dan tidur',
     'Perdana Menteri Tun Dr',
     'Mahathir Mohamad dan',
     'Dr Mahathir Mohamad dan',
     'yang mahu pulang ke kampung',
     'minggu depan, Perdana Menteri',
     'orang ramai yang mahu pulang',
     'Mahathir menasihati mereka supaya berhenti',
     'berehat dan tidur sebentar',
     'Aidilfitri minggu depan, Perdana Menteri',
     'pesanan khas kepada orang',
     'Pengangkutan Anthony Loke',
     'Dalam video pendek',
     'sambutan Aidilfitri minggu depan, Perdana',
     'Mahathir Mohamad dan Menteri Pengangkutan',
     'Menteri Pengangkutan Anthony Loke',
     'berehat dan tidur sebentar sekiranya',
     'pulang ke kampung halaman',
     'menasihati mereka supaya',
     'Dr Mahathir menasihati',
     'Fook menitipkan pesanan khas',
     'Keselamatan Jalan Raya (JKJR)',
     'Fook menitipkan pesanan khas kepada',
     'Keselamatan Jalan Raya (JKJR) itu,',
     'Jalan Raya (JKJR) itu, Dr',
     'mereka supaya berhenti berehat',
     'Pengangkutan Anthony Loke Siew Fook',
     'Menteri Tun Dr Mahathir',
     'Aidilfitri minggu depan,',
     'pulang ke kampung halaman masing-masing.',
     'Dr Mahathir Mohamad dan Menteri',
     'mahu pulang ke kampung halaman',
     'Dr Mahathir menasihati mereka',
     'berhenti berehat dan tidur',
     'terbitan Jabatan Keselamatan',
     'Menteri Pengangkutan Anthony',
     'sambutan Aidilfitri minggu depan,',
     'masing-masing. Dalam video',
     'depan, Perdana Menteri Tun']
