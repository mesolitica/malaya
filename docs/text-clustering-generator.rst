
.. code:: python

    import malaya

.. code:: python

    string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'

Cluster same word structure based on POS and Entities
-----------------------------------------------------

.. code:: python

    bahdanau_entities = malaya.entities.deep_entities('bahdanau')
    bahdanau_pos = malaya.pos.deep_pos('bahdanau')

.. code:: python

    result_entities = bahdanau_entities.predict(string)
    result_pos = bahdanau_pos.predict(string)

.. code:: python

    from malaya.text_functions import cluster_words, generate_ngram

.. code:: python

    generated_grams = generate_ngram(
        result_pos,
        result_entities,
        ngram = (1, 3),
        accept_pos = ['NOUN', 'PROPN', 'VERB'],
        accept_entities = ['law', 'location', 'organization', 'person', 'time'],
    )
    generated_grams




.. parsed-literal::

    ['Jalan',
     'Aidilfitri minggu depan',
     'tidur sekiranya mengantuk',
     'Mahathir',
     'Jabatan',
     'Kuala Lumpur sambutan',
     'Jkjr Dr Mahathir',
     'Raya Jkjr',
     'Mohamad',
     'berhenti',
     'kampung halaman',
     'mengantuk memandu',
     'Aidilfitri',
     'Siew',
     'sambutan Aidilfitri',
     'Jkjr',
     'Lumpur sambutan',
     'Raya',
     'depan Perdana',
     'orang',
     'menitipkan orang pulang',
     'orang pulang',
     'minggu depan',
     'video terbitan',
     'Anthony Loke Siew',
     'pulang kampung',
     'Kuala',
     'depan Perdana Menteri',
     'Menteri Pengangkutan Anthony',
     'Pengangkutan',
     'Raya Jkjr Dr',
     'Jalan Raya',
     'mengantuk',
     'terbitan Jabatan',
     'Aidilfitri minggu',
     'orang pulang kampung',
     'menitipkan orang',
     'halaman',
     'Dalam video',
     'Dr Mahathir berhenti',
     'Pengangkutan Anthony Loke',
     'Fook',
     'memandu',
     'Menteri',
     'sekiranya',
     'tidur sekiranya',
     'berehat',
     'terbitan Jabatan Keselamatan',
     'Mahathir Mohamad',
     'Perdana Menteri',
     'Keselamatan Jalan',
     'Anthony Loke',
     'Jabatan Keselamatan Jalan',
     'kampung halaman Dalam',
     'Menteri Pengangkutan',
     'Dalam video terbitan',
     'Tun Dr',
     'depan',
     'Loke Siew Fook',
     'Lumpur sambutan Aidilfitri',
     'Anthony',
     'Dr Mahathir',
     'Keselamatan Jalan Raya',
     'Menteri Tun',
     'Loke',
     'Lumpur',
     'Mohamad Menteri',
     'Mahathir berhenti berehat',
     'halaman Dalam video',
     'Menteri Tun Dr',
     'minggu depan Perdana',
     'menitipkan',
     'Dr Mahathir Mohamad',
     'sambutan',
     'tidur',
     'Tun',
     'Dalam',
     'halaman Dalam',
     'Kuala Lumpur',
     'Mahathir Mohamad Menteri',
     'Jalan Raya Jkjr',
     'Keselamatan',
     'Mahathir berhenti',
     'terbitan',
     'berehat tidur sekiranya',
     'pulang kampung halaman',
     'Jabatan Keselamatan',
     'sekiranya mengantuk',
     'Fook menitipkan',
     'Jkjr Dr',
     'sekiranya mengantuk memandu',
     'sambutan Aidilfitri minggu',
     'berhenti berehat',
     'berehat tidur',
     'Perdana Menteri Tun',
     'Siew Fook menitipkan',
     'kampung',
     'video terbitan Jabatan',
     'Siew Fook',
     'Mohamad Menteri Pengangkutan',
     'berhenti berehat tidur',
     'Perdana',
     'Pengangkutan Anthony',
     'Loke Siew',
     'Dr',
     'pulang',
     'Fook menitipkan orang',
     'minggu',
     'video',
     'Tun Dr Mahathir']



.. code:: python

    cluster_words(generated_grams)




.. parsed-literal::

    ['kampung halaman Dalam',
     'Anthony Loke Siew',
     'Dalam video terbitan',
     'pulang kampung halaman',
     'Aidilfitri minggu depan',
     'depan Perdana Menteri',
     'tidur sekiranya mengantuk',
     'Loke Siew Fook',
     'Lumpur sambutan Aidilfitri',
     'Menteri Pengangkutan Anthony',
     'Raya Jkjr Dr',
     'Keselamatan Jalan Raya',
     'Mahathir berhenti berehat',
     'orang pulang kampung',
     'sekiranya mengantuk memandu',
     'halaman Dalam video',
     'Kuala Lumpur sambutan',
     'Menteri Tun Dr',
     'Jkjr Dr Mahathir',
     'minggu depan Perdana',
     'sambutan Aidilfitri minggu',
     'Dr Mahathir Mohamad',
     'Dr Mahathir berhenti',
     'Siew Fook menitipkan',
     'Perdana Menteri Tun',
     'Pengangkutan Anthony Loke',
     'video terbitan Jabatan',
     'Mohamad Menteri Pengangkutan',
     'berhenti berehat tidur',
     'Mahathir Mohamad Menteri',
     'menitipkan orang pulang',
     'Jalan Raya Jkjr',
     'Fook menitipkan orang',
     'terbitan Jabatan Keselamatan',
     'Jabatan Keselamatan Jalan',
     'berehat tidur sekiranya',
     'Tun Dr Mahathir']



Cluster POS and Entities
------------------------

.. code:: python

    from malaya.text_functions import cluster_pos, cluster_entities

.. code:: python

    cluster_pos(result_pos)




.. parsed-literal::

    {'ADJ': ['depan', 'khas', 'ramai', 'pendek'],
     'ADP': ['kepada', 'ke'],
     'ADV': ['pesanan', 'mahu', 'menasihati', 'sebentar'],
     'ADX': [],
     'CCONJ': ['dan'],
     'DET': ['itu'],
     'NOUN': ['sambutan', 'orang', 'kampung halaman', 'video', 'terbitan Jabatan'],
     'NUM': [],
     'PART': [],
     'PRON': ['Sempena', 'yang', 'masing-masing', 'mereka'],
     'PROPN': ['Kuala Lumpur',
      'Aidilfitri minggu',
      'Perdana Menteri Tun Dr Mahathir Mohamad',
      'Menteri Pengangkutan Anthony Loke Siew Fook menitipkan',
      'Keselamatan Jalan Raya Jkjr',
      'Dr Mahathir',
      'berehat',
      'sekiranya mengantuk'],
     'SCONJ': ['supaya', 'ketika'],
     'SYM': [],
     'VERB': ['pulang', 'Dalam', 'berhenti', 'tidur'],
     'X': []}



.. code:: python

    cluster_entities(result_entities)




.. parsed-literal::

    {'OTHER': ['sempena',
      'dan',
      'menitipkan pesanan khas kepada',
      'ramai yang mahu pulang ke',
      'masing-masing dalam video pendek terbitan',
      'itu'],
     'law': [],
     'location': ['kuala lumpur', 'kampung halaman'],
     'organization': ['orang', 'jabatan keselamatan jalan raya jkjr'],
     'person': ['perdana menteri tun dr mahathir mohamad',
      'menteri pengangkutan anthony loke siew fook',
      'dr mahathir'],
     'quantity': [],
     'time': ['minggu depan'],
     'event': ['sambutan aidilfitri']}



Generate ngrams
---------------

.. code:: python

    from malaya.text_functions import sentence_ngram

.. code:: python

    sentence_ngram(string, ngram = (3, 5))




.. parsed-literal::

    ['KUALA LUMPUR: Sempena sambutan Aidilfitri',
     'menasihati mereka supaya berhenti',
     'halaman masing-masing. Dalam video',
     'masing-masing. Dalam video pendek terbitan',
     'Menteri Tun Dr Mahathir Mohamad',
     'Perdana Menteri Tun Dr',
     'LUMPUR: Sempena sambutan Aidilfitri minggu',
     'sekiranya mengantuk ketika',
     'Sempena sambutan Aidilfitri',
     'Menteri Pengangkutan Anthony Loke',
     'Fook menitipkan pesanan khas kepada',
     'LUMPUR: Sempena sambutan',
     'Aidilfitri minggu depan, Perdana',
     'Anthony Loke Siew Fook',
     'video pendek terbitan',
     'ke kampung halaman masing-masing. Dalam',
     'sebentar sekiranya mengantuk',
     '(JKJR) itu, Dr Mahathir',
     'mereka supaya berhenti',
     'sambutan Aidilfitri minggu depan, Perdana',
     'supaya berhenti berehat',
     'supaya berhenti berehat dan',
     'Dr Mahathir menasihati mereka',
     'masing-masing. Dalam video',
     'Raya (JKJR) itu,',
     'menitipkan pesanan khas kepada',
     'Mohamad dan Menteri',
     'LUMPUR: Sempena sambutan Aidilfitri',
     'Dr Mahathir Mohamad dan',
     'Fook menitipkan pesanan khas',
     'KUALA LUMPUR: Sempena',
     'Dr Mahathir menasihati mereka supaya',
     'tidur sebentar sekiranya mengantuk ketika',
     'minggu depan, Perdana',
     'pulang ke kampung halaman masing-masing.',
     'Jalan Raya (JKJR) itu,',
     'KUALA LUMPUR: Sempena sambutan',
     'depan, Perdana Menteri Tun Dr',
     'Mahathir Mohamad dan Menteri Pengangkutan',
     'Mohamad dan Menteri Pengangkutan',
     'Anthony Loke Siew',
     'Dalam video pendek terbitan',
     'Keselamatan Jalan Raya (JKJR) itu,',
     'kampung halaman masing-masing.',
     'Dr Mahathir menasihati',
     'Menteri Pengangkutan Anthony',
     '(JKJR) itu, Dr',
     'sebentar sekiranya mengantuk ketika',
     'berehat dan tidur',
     'Fook menitipkan pesanan',
     'Raya (JKJR) itu, Dr Mahathir',
     'Keselamatan Jalan Raya (JKJR)',
     'itu, Dr Mahathir menasihati',
     'kepada orang ramai yang',
     'Perdana Menteri Tun Dr Mahathir',
     'dan Menteri Pengangkutan Anthony',
     'pesanan khas kepada orang',
     'Jabatan Keselamatan Jalan Raya (JKJR)',
     'dan tidur sebentar sekiranya mengantuk',
     'Mohamad dan Menteri Pengangkutan Anthony',
     'Pengangkutan Anthony Loke',
     'tidur sebentar sekiranya mengantuk',
     'orang ramai yang mahu pulang',
     'Sempena sambutan Aidilfitri minggu',
     'pesanan khas kepada orang ramai',
     'itu, Dr Mahathir',
     'Dalam video pendek',
     'ramai yang mahu pulang',
     'yang mahu pulang',
     'pulang ke kampung halaman',
     'Jalan Raya (JKJR)',
     'tidur sebentar sekiranya',
     'orang ramai yang',
     'terbitan Jabatan Keselamatan',
     'Jabatan Keselamatan Jalan',
     'Aidilfitri minggu depan,',
     'menasihati mereka supaya',
     'terbitan Jabatan Keselamatan Jalan Raya',
     'menitipkan pesanan khas',
     'depan, Perdana Menteri Tun',
     'ke kampung halaman masing-masing.',
     'Loke Siew Fook',
     'masing-masing. Dalam video pendek',
     'Pengangkutan Anthony Loke Siew',
     'Keselamatan Jalan Raya',
     'pulang ke kampung',
     'Tun Dr Mahathir Mohamad dan',
     'Tun Dr Mahathir Mohamad',
     'Menteri Tun Dr',
     'khas kepada orang',
     'Menteri Pengangkutan Anthony Loke Siew',
     'sebentar sekiranya mengantuk ketika memandu.',
     'dan tidur sebentar',
     'Dr Mahathir Mohamad',
     'dan Menteri Pengangkutan',
     'menitipkan pesanan khas kepada orang',
     'Siew Fook menitipkan pesanan khas',
     'Sempena sambutan Aidilfitri minggu depan,',
     'dan tidur sebentar sekiranya',
     'mengantuk ketika memandu.',
     'kampung halaman masing-masing. Dalam',
     'kepada orang ramai yang mahu',
     'Anthony Loke Siew Fook menitipkan',
     'yang mahu pulang ke',
     'ramai yang mahu',
     'pendek terbitan Jabatan Keselamatan',
     'yang mahu pulang ke kampung',
     'Jalan Raya (JKJR) itu, Dr',
     'kampung halaman masing-masing. Dalam video',
     'pendek terbitan Jabatan',
     'khas kepada orang ramai',
     'orang ramai yang mahu',
     '(JKJR) itu, Dr Mahathir menasihati',
     'Mahathir Mohamad dan Menteri',
     'Mahathir menasihati mereka supaya',
     'sekiranya mengantuk ketika memandu.',
     'Mahathir Mohamad dan',
     'terbitan Jabatan Keselamatan Jalan',
     'minggu depan, Perdana Menteri',
     'ke kampung halaman',
     'berhenti berehat dan tidur sebentar',
     'Siew Fook menitipkan pesanan',
     'mereka supaya berhenti berehat',
     'Dalam video pendek terbitan Jabatan',
     'halaman masing-masing. Dalam',
     'video pendek terbitan Jabatan Keselamatan',
     'mereka supaya berhenti berehat dan',
     'depan, Perdana Menteri',
     'supaya berhenti berehat dan tidur',
     'Loke Siew Fook menitipkan',
     'Mahathir menasihati mereka',
     'video pendek terbitan Jabatan',
     'sambutan Aidilfitri minggu',
     'berhenti berehat dan',
     'Menteri Tun Dr Mahathir',
     'mahu pulang ke kampung',
     'pendek terbitan Jabatan Keselamatan Jalan',
     'pesanan khas kepada',
     'itu, Dr Mahathir menasihati mereka',
     'berehat dan tidur sebentar',
     'berehat dan tidur sebentar sekiranya',
     'sambutan Aidilfitri minggu depan,',
     'ramai yang mahu pulang ke',
     'Perdana Menteri Tun',
     'Siew Fook menitipkan',
     'khas kepada orang ramai yang',
     'Raya (JKJR) itu, Dr',
     'halaman masing-masing. Dalam video pendek',
     'Aidilfitri minggu depan, Perdana Menteri',
     'menasihati mereka supaya berhenti berehat',
     'Jabatan Keselamatan Jalan Raya',
     'Dr Mahathir Mohamad dan Menteri',
     'Pengangkutan Anthony Loke Siew Fook',
     'berhenti berehat dan tidur',
     'minggu depan, Perdana Menteri Tun',
     'dan Menteri Pengangkutan Anthony Loke',
     'Mahathir menasihati mereka supaya berhenti',
     'Loke Siew Fook menitipkan pesanan',
     'mahu pulang ke kampung halaman',
     'mahu pulang ke',
     'kepada orang ramai',
     'Tun Dr Mahathir']
