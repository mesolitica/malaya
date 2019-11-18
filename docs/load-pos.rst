
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 6.57 s, sys: 1.44 s, total: 8.01 s
    Wall time: 12.5 s


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


List available Transformer POS models
-------------------------------------

.. code:: python

    malaya.pos.available_transformer_model()




.. parsed-literal::

    {'bert': ['base', 'small'], 'xlnet': ['base'], 'albert': ['base']}



Make sure you can check accuracy chart from here first before select a
model,
https://malaya.readthedocs.io/en/latest/Accuracy.html#pos-recognition

**You might want to use ALBERT, a very small size, 43MB, but the
accuracy is still on the top notch.**

.. code:: python

    string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'

Load ALBERT model
-----------------

.. code:: python

    model = malaya.pos.transformer(model = 'albert', size = 'base')


.. parsed-literal::

    WARNING: Logging before flag parsing goes to stderr.
    W1017 22:32:06.326681 4564137408 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:68: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

    W1017 22:32:06.327857 4564137408 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:69: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

    W1017 22:32:07.276805 4564137408 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:64: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.



.. code:: python

    model.predict(string)




.. parsed-literal::

    [('Kuala', 'PROPN'),
     ('Lumpur:', 'PROPN'),
     ('Sempena', 'ADP'),
     ('sambutan', 'NOUN'),
     ('Aidilfitri', 'PROPN'),
     ('minggu', 'NOUN'),
     ('depan,', 'ADJ'),
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
     ('masing-masing.', 'DET'),
     ('Dalam', 'ADP'),
     ('video', 'NOUN'),
     ('pendek', 'ADJ'),
     ('terbitan', 'NOUN'),
     ('Jabatan', 'PROPN'),
     ('Keselamatan', 'PROPN'),
     ('Jalan', 'PROPN'),
     ('Raya', 'PROPN'),
     ('(JKJR)', 'PUNCT'),
     ('itu,', 'DET'),
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
     ('sekiranya', 'SCONJ'),
     ('mengantuk', 'NOUN'),
     ('ketika', 'SCONJ'),
     ('memandu.', 'NOUN')]



.. code:: python

    model.analyze(string)




.. parsed-literal::

    {'words': ['Kuala',
      'Lumpur:',
      'Sempena',
      'sambutan',
      'Aidilfitri',
      'minggu',
      'depan,',
      'Perdana',
      'Menteri',
      'Tun',
      'Dr',
      'Mahathir',
      'Mohamad',
      'dan',
      'Menteri',
      'Pengangkutan',
      'Anthony',
      'Loke',
      'Siew',
      'Fook',
      'menitipkan',
      'pesanan',
      'khas',
      'kepada',
      'orang',
      'ramai',
      'yang',
      'mahu',
      'pulang',
      'ke',
      'kampung',
      'halaman',
      'masing-masing.',
      'Dalam',
      'video',
      'pendek',
      'terbitan',
      'Jabatan',
      'Keselamatan',
      'Jalan',
      'Raya',
      '(JKJR)',
      'itu,',
      'Dr',
      'Mahathir',
      'menasihati',
      'mereka',
      'supaya',
      'berhenti',
      'berehat',
      'dan',
      'tidur',
      'sebentar',
      'sekiranya',
      'mengantuk',
      'ketika',
      'memandu.'],
     'tags': [{'text': 'Kuala Lumpur:',
       'type': 'PROPN',
       'score': 1.0,
       'beginOffset': 0,
       'endOffset': 1},
      {'text': 'Sempena',
       'type': 'ADP',
       'score': 1.0,
       'beginOffset': 2,
       'endOffset': 2},
      {'text': 'sambutan',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 3,
       'endOffset': 3},
      {'text': 'Aidilfitri',
       'type': 'PROPN',
       'score': 1.0,
       'beginOffset': 4,
       'endOffset': 4},
      {'text': 'minggu',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 5,
       'endOffset': 5},
      {'text': 'depan,',
       'type': 'ADJ',
       'score': 1.0,
       'beginOffset': 6,
       'endOffset': 6},
      {'text': 'Perdana Menteri Tun Dr Mahathir Mohamad',
       'type': 'PROPN',
       'score': 1.0,
       'beginOffset': 7,
       'endOffset': 12},
      {'text': 'dan',
       'type': 'CCONJ',
       'score': 1.0,
       'beginOffset': 13,
       'endOffset': 13},
      {'text': 'Menteri Pengangkutan Anthony Loke Siew Fook',
       'type': 'PROPN',
       'score': 1.0,
       'beginOffset': 14,
       'endOffset': 19},
      {'text': 'menitipkan',
       'type': 'VERB',
       'score': 1.0,
       'beginOffset': 20,
       'endOffset': 20},
      {'text': 'pesanan',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 21,
       'endOffset': 21},
      {'text': 'khas',
       'type': 'ADJ',
       'score': 1.0,
       'beginOffset': 22,
       'endOffset': 22},
      {'text': 'kepada',
       'type': 'ADP',
       'score': 1.0,
       'beginOffset': 23,
       'endOffset': 23},
      {'text': 'orang',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 24,
       'endOffset': 24},
      {'text': 'ramai',
       'type': 'ADJ',
       'score': 1.0,
       'beginOffset': 25,
       'endOffset': 25},
      {'text': 'yang',
       'type': 'PRON',
       'score': 1.0,
       'beginOffset': 26,
       'endOffset': 26},
      {'text': 'mahu',
       'type': 'ADV',
       'score': 1.0,
       'beginOffset': 27,
       'endOffset': 27},
      {'text': 'pulang',
       'type': 'VERB',
       'score': 1.0,
       'beginOffset': 28,
       'endOffset': 28},
      {'text': 'ke',
       'type': 'ADP',
       'score': 1.0,
       'beginOffset': 29,
       'endOffset': 29},
      {'text': 'kampung halaman',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 30,
       'endOffset': 31},
      {'text': 'masing-masing.',
       'type': 'DET',
       'score': 1.0,
       'beginOffset': 32,
       'endOffset': 32},
      {'text': 'Dalam',
       'type': 'ADP',
       'score': 1.0,
       'beginOffset': 33,
       'endOffset': 33},
      {'text': 'video',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 34,
       'endOffset': 34},
      {'text': 'pendek',
       'type': 'ADJ',
       'score': 1.0,
       'beginOffset': 35,
       'endOffset': 35},
      {'text': 'terbitan',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 36,
       'endOffset': 36},
      {'text': 'Jabatan Keselamatan Jalan Raya',
       'type': 'PROPN',
       'score': 1.0,
       'beginOffset': 37,
       'endOffset': 40},
      {'text': '(JKJR)',
       'type': 'PUNCT',
       'score': 1.0,
       'beginOffset': 41,
       'endOffset': 41},
      {'text': 'itu,',
       'type': 'DET',
       'score': 1.0,
       'beginOffset': 42,
       'endOffset': 42},
      {'text': 'Dr Mahathir',
       'type': 'PROPN',
       'score': 1.0,
       'beginOffset': 43,
       'endOffset': 44},
      {'text': 'menasihati',
       'type': 'VERB',
       'score': 1.0,
       'beginOffset': 45,
       'endOffset': 45},
      {'text': 'mereka',
       'type': 'PRON',
       'score': 1.0,
       'beginOffset': 46,
       'endOffset': 46},
      {'text': 'supaya',
       'type': 'SCONJ',
       'score': 1.0,
       'beginOffset': 47,
       'endOffset': 47},
      {'text': 'berhenti',
       'type': 'VERB',
       'score': 1.0,
       'beginOffset': 48,
       'endOffset': 48},
      {'text': 'berehat',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 49,
       'endOffset': 49},
      {'text': 'dan',
       'type': 'CCONJ',
       'score': 1.0,
       'beginOffset': 50,
       'endOffset': 50},
      {'text': 'tidur',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 51,
       'endOffset': 51},
      {'text': 'sebentar',
       'type': 'ADV',
       'score': 1.0,
       'beginOffset': 52,
       'endOffset': 52},
      {'text': 'sekiranya',
       'type': 'SCONJ',
       'score': 1.0,
       'beginOffset': 53,
       'endOffset': 53},
      {'text': 'mengantuk',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 54,
       'endOffset': 54},
      {'text': 'ketika',
       'type': 'SCONJ',
       'score': 1.0,
       'beginOffset': 55,
       'endOffset': 55}]}



Voting stack model
------------------

.. code:: python

    xlnet = malaya.pos.transformer(model = 'xlnet', size = 'base')
    malaya.stack.voting_stack([model, xlnet, xlnet], string)




.. parsed-literal::

    [('Kuala', 'PROPN'),
     ('Lumpur:', 'PROPN'),
     ('Sempena', 'ADP'),
     ('sambutan', 'NOUN'),
     ('Aidilfitri', 'PROPN'),
     ('minggu', 'NOUN'),
     ('depan,', 'NOUN'),
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
     ('masing-masing.', 'NOUN'),
     ('Dalam', 'ADP'),
     ('video', 'NOUN'),
     ('pendek', 'ADJ'),
     ('terbitan', 'NOUN'),
     ('Jabatan', 'PROPN'),
     ('Keselamatan', 'PROPN'),
     ('Jalan', 'PROPN'),
     ('Raya', 'PROPN'),
     ('(JKJR)', 'PUNCT'),
     ('itu,', 'DET'),
     ('Dr', 'PROPN'),
     ('Mahathir', 'PROPN'),
     ('menasihati', 'VERB'),
     ('mereka', 'PRON'),
     ('supaya', 'SCONJ'),
     ('berhenti', 'VERB'),
     ('berehat', 'VERB'),
     ('dan', 'CCONJ'),
     ('tidur', 'VERB'),
     ('sebentar', 'ADV'),
     ('sekiranya', 'SCONJ'),
     ('mengantuk', 'NOUN'),
     ('ketika', 'SCONJ'),
     ('memandu.', 'NOUN')]
