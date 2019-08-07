
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 4.38 s, sys: 677 ms, total: 5.06 s
    Wall time: 5.11 s


BERT model
----------

BERT is the best POS model in term of accuracy, you can check POS
accuracy here,
https://malaya.readthedocs.io/en/latest/Accuracy.html#pos-recognition.
Question is, why BERT?

1. Transformer model learn the context of a word based on all of its
   surroundings (live string), bidirectionally. So it much better
   understand left and right hand side relationships.
2. Because of transformer able to leverage to context during live
   string, we dont need to capture available words in this world,
   instead capture substrings and build the attention after that. BERT
   will never have Out-Of-Vocab problem.

List available BERT POS models
------------------------------

.. code:: python

    malaya.pos.available_bert_model()




.. parsed-literal::

    ['multilanguage', 'base', 'small']



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


.. code:: python

    string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'

Load BERT models
----------------

.. code:: python

    model = malaya.pos.bert(model = 'base')


.. parsed-literal::

    WARNING: Logging before flag parsing goes to stderr.
    W0807 17:28:38.491054 4655850944 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:45: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

    W0807 17:28:38.492187 4655850944 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:46: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

    W0807 17:28:45.406651 4655850944 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:41: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.



.. code:: python

    model.predict(string)




.. parsed-literal::

    [('Kuala', 'PROPN'),
     ('Lumpur', 'PROPN'),
     (':', 'PUNCT'),
     ('Sempena', 'PROPN'),
     ('sambutan', 'NOUN'),
     ('Aidilfitri', 'NOUN'),
     ('minggu', 'NOUN'),
     ('depan', 'ADJ'),
     (',', 'PUNCT'),
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
     ('masing', 'DET'),
     ('-', 'PUNCT'),
     ('masing', 'DET'),
     ('.', 'PUNCT'),
     ('Dalam', 'ADP'),
     ('video', 'NOUN'),
     ('pendek', 'ADJ'),
     ('terbitan', 'NOUN'),
     ('Jabatan', 'NOUN'),
     ('Keselamatan', 'PROPN'),
     ('Jalan', 'PROPN'),
     ('Raya', 'PROPN'),
     ('(', 'PUNCT'),
     ('Jkjr', 'PROPN'),
     (')', 'PUNCT'),
     ('itu', 'DET'),
     (',', 'PUNCT'),
     ('Dr', 'PROPN'),
     ('Mahathir', 'PROPN'),
     ('menasihati', 'VERB'),
     ('mereka', 'PRON'),
     ('supaya', 'SCONJ'),
     ('berhenti', 'VERB'),
     ('berehat', 'ADJ'),
     ('dan', 'CCONJ'),
     ('tidur', 'VERB'),
     ('sebentar', 'ADV'),
     ('sekiranya', 'SCONJ'),
     ('mengantuk', 'ADJ'),
     ('ketika', 'SCONJ'),
     ('memandu', 'VERB'),
     ('.', 'PUNCT')]



.. code:: python

    model.analyze(string)




.. parsed-literal::

    {'words': ['Kuala',
      'Lumpur',
      ':',
      'Sempena',
      'sambutan',
      'Aidilfitri',
      'minggu',
      'depan',
      ',',
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
      'masing',
      '-',
      'masing',
      '.',
      'Dalam',
      'video',
      'pendek',
      'terbitan',
      'Jabatan',
      'Keselamatan',
      'Jalan',
      'Raya',
      '(',
      'Jkjr',
      ')',
      'itu',
      ',',
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
      'memandu',
      '.'],
     'tags': [{'text': 'Kuala Lumpur',
       'type': 'PROPN',
       'score': 1.0,
       'beginOffset': 0,
       'endOffset': 1},
      {'text': ':',
       'type': 'PUNCT',
       'score': 1.0,
       'beginOffset': 2,
       'endOffset': 2},
      {'text': 'Sempena',
       'type': 'PROPN',
       'score': 1.0,
       'beginOffset': 3,
       'endOffset': 3},
      {'text': 'sambutan Aidilfitri minggu',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 4,
       'endOffset': 6},
      {'text': 'depan',
       'type': 'ADJ',
       'score': 1.0,
       'beginOffset': 7,
       'endOffset': 7},
      {'text': ',',
       'type': 'PUNCT',
       'score': 1.0,
       'beginOffset': 8,
       'endOffset': 8},
      {'text': 'Perdana Menteri Tun Dr Mahathir Mohamad',
       'type': 'PROPN',
       'score': 1.0,
       'beginOffset': 9,
       'endOffset': 14},
      {'text': 'dan',
       'type': 'CCONJ',
       'score': 1.0,
       'beginOffset': 15,
       'endOffset': 15},
      {'text': 'Menteri Pengangkutan Anthony Loke Siew Fook',
       'type': 'PROPN',
       'score': 1.0,
       'beginOffset': 16,
       'endOffset': 21},
      {'text': 'menitipkan',
       'type': 'VERB',
       'score': 1.0,
       'beginOffset': 22,
       'endOffset': 22},
      {'text': 'pesanan',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 23,
       'endOffset': 23},
      {'text': 'khas',
       'type': 'ADJ',
       'score': 1.0,
       'beginOffset': 24,
       'endOffset': 24},
      {'text': 'kepada',
       'type': 'ADP',
       'score': 1.0,
       'beginOffset': 25,
       'endOffset': 25},
      {'text': 'orang',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 26,
       'endOffset': 26},
      {'text': 'ramai',
       'type': 'ADJ',
       'score': 1.0,
       'beginOffset': 27,
       'endOffset': 27},
      {'text': 'yang',
       'type': 'PRON',
       'score': 1.0,
       'beginOffset': 28,
       'endOffset': 28},
      {'text': 'mahu',
       'type': 'ADV',
       'score': 1.0,
       'beginOffset': 29,
       'endOffset': 29},
      {'text': 'pulang',
       'type': 'VERB',
       'score': 1.0,
       'beginOffset': 30,
       'endOffset': 30},
      {'text': 'ke',
       'type': 'ADP',
       'score': 1.0,
       'beginOffset': 31,
       'endOffset': 31},
      {'text': 'kampung halaman',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 32,
       'endOffset': 33},
      {'text': 'masing',
       'type': 'DET',
       'score': 1.0,
       'beginOffset': 34,
       'endOffset': 34},
      {'text': '-',
       'type': 'PUNCT',
       'score': 1.0,
       'beginOffset': 35,
       'endOffset': 35},
      {'text': 'masing',
       'type': 'DET',
       'score': 1.0,
       'beginOffset': 36,
       'endOffset': 36},
      {'text': '.',
       'type': 'PUNCT',
       'score': 1.0,
       'beginOffset': 37,
       'endOffset': 37},
      {'text': 'Dalam',
       'type': 'ADP',
       'score': 1.0,
       'beginOffset': 38,
       'endOffset': 38},
      {'text': 'video',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 39,
       'endOffset': 39},
      {'text': 'pendek',
       'type': 'ADJ',
       'score': 1.0,
       'beginOffset': 40,
       'endOffset': 40},
      {'text': 'terbitan',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 41,
       'endOffset': 41},
      {'text': 'Jabatan Keselamatan Jalan Raya',
       'type': 'PROPN',
       'score': 1.0,
       'beginOffset': 42,
       'endOffset': 45},
      {'text': '(',
       'type': 'PUNCT',
       'score': 1.0,
       'beginOffset': 46,
       'endOffset': 46},
      {'text': 'Jkjr',
       'type': 'PROPN',
       'score': 1.0,
       'beginOffset': 47,
       'endOffset': 47},
      {'text': ')',
       'type': 'PUNCT',
       'score': 1.0,
       'beginOffset': 48,
       'endOffset': 48},
      {'text': 'itu',
       'type': 'DET',
       'score': 1.0,
       'beginOffset': 49,
       'endOffset': 49},
      {'text': ',',
       'type': 'PUNCT',
       'score': 1.0,
       'beginOffset': 50,
       'endOffset': 50},
      {'text': 'Dr Mahathir',
       'type': 'PROPN',
       'score': 1.0,
       'beginOffset': 51,
       'endOffset': 52},
      {'text': 'menasihati',
       'type': 'VERB',
       'score': 1.0,
       'beginOffset': 53,
       'endOffset': 53},
      {'text': 'mereka',
       'type': 'PRON',
       'score': 1.0,
       'beginOffset': 54,
       'endOffset': 54},
      {'text': 'supaya',
       'type': 'SCONJ',
       'score': 1.0,
       'beginOffset': 55,
       'endOffset': 55},
      {'text': 'berhenti',
       'type': 'VERB',
       'score': 1.0,
       'beginOffset': 56,
       'endOffset': 56},
      {'text': 'berehat',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 57,
       'endOffset': 57},
      {'text': 'dan',
       'type': 'CCONJ',
       'score': 1.0,
       'beginOffset': 58,
       'endOffset': 58},
      {'text': 'tidur',
       'type': 'VERB',
       'score': 1.0,
       'beginOffset': 59,
       'endOffset': 59},
      {'text': 'sebentar',
       'type': 'ADV',
       'score': 1.0,
       'beginOffset': 60,
       'endOffset': 60},
      {'text': 'sekiranya',
       'type': 'SCONJ',
       'score': 1.0,
       'beginOffset': 61,
       'endOffset': 61},
      {'text': 'mengantuk',
       'type': 'ADJ',
       'score': 1.0,
       'beginOffset': 62,
       'endOffset': 62},
      {'text': 'ketika',
       'type': 'SCONJ',
       'score': 1.0,
       'beginOffset': 63,
       'endOffset': 63},
      {'text': 'memandu',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 64,
       'endOffset': 64}]}



List available deep learning models
-----------------------------------

.. code:: python

    malaya.pos.available_deep_model()




.. parsed-literal::

    ['concat', 'bahdanau', 'luong', 'entity-network', 'attention']



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
    [('Kuala', 'NOUN'), ('Lumpur', 'ADJ'), (':', 'PUNCT'), ('Sempena', 'NOUN'), ('sambutan', 'NOUN'), ('Aidilfitri', 'NOUN'), ('minggu', 'NOUN'), ('depan', 'ADJ'), (',', 'PUNCT'), ('Perdana', 'PROPN'), ('Menteri', 'PROPN'), ('Tun', 'PROPN'), ('Dr', 'DET'), ('Mahathir', 'NOUN'), ('Mohamad', 'NOUN'), ('dan', 'CCONJ'), ('Menteri', 'NOUN'), ('Pengangkutan', 'NOUN'), ('Anthony', 'NOUN'), ('Loke', 'NOUN'), ('Siew', 'NOUN'), ('Fook', 'NOUN'), ('menitipkan', 'NOUN'), ('pesanan', 'NOUN'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'NOUN'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing', 'DET'), ('-', 'PUNCT'), ('masing', 'DET'), ('.', 'PUNCT'), ('Dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('Jabatan', 'NOUN'), ('Keselamatan', 'NOUN'), ('Jalan', 'NOUN'), ('Raya', 'NOUN'), ('(', 'PUNCT'), ('Jkjr', 'NOUN'), (')', 'PUNCT'), ('itu', 'DET'), (',', 'PUNCT'), ('Dr', 'DET'), ('Mahathir', 'NOUN'), ('menasihati', 'NOUN'), ('mereka', 'PRON'), ('supaya', 'SCONJ'), ('berhenti', 'VERB'), ('berehat', 'NOUN'), ('dan', 'CCONJ'), ('tidur', 'NOUN'), ('sebentar', 'ADV'), ('sekiranya', 'NOUN'), ('mengantuk', 'NOUN'), ('ketika', 'SCONJ'), ('memandu', 'NOUN'), ('.', 'PUNCT')]

    Testing bahdanau model
    [('Kuala', 'PROPN'), ('Lumpur', 'PROPN'), (':', 'PUNCT'), ('Sempena', 'NOUN'), ('sambutan', 'NOUN'), ('Aidilfitri', 'NOUN'), ('minggu', 'NOUN'), ('depan', 'ADJ'), (',', 'PUNCT'), ('Perdana', 'PROPN'), ('Menteri', 'PROPN'), ('Tun', 'PROPN'), ('Dr', 'ADV'), ('Mahathir', 'NOUN'), ('Mohamad', 'NOUN'), ('dan', 'CCONJ'), ('Menteri', 'PROPN'), ('Pengangkutan', 'NOUN'), ('Anthony', 'NOUN'), ('Loke', 'NOUN'), ('Siew', 'NOUN'), ('Fook', 'NOUN'), ('menitipkan', 'NOUN'), ('pesanan', 'NOUN'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'NOUN'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing', 'ADV'), ('-', 'PUNCT'), ('masing', 'ADV'), ('.', 'PUNCT'), ('Dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('Jabatan', 'NOUN'), ('Keselamatan', 'NOUN'), ('Jalan', 'NOUN'), ('Raya', 'NOUN'), ('(', 'PUNCT'), ('Jkjr', 'NOUN'), (')', 'PUNCT'), ('itu', 'DET'), (',', 'PUNCT'), ('Dr', 'ADV'), ('Mahathir', 'VERB'), ('menasihati', 'NOUN'), ('mereka', 'PRON'), ('supaya', 'CCONJ'), ('berhenti', 'VERB'), ('berehat', 'NOUN'), ('dan', 'CCONJ'), ('tidur', 'NOUN'), ('sebentar', 'ADV'), ('sekiranya', 'NOUN'), ('mengantuk', 'NOUN'), ('ketika', 'SCONJ'), ('memandu', 'NOUN'), ('.', 'PUNCT')]

    Testing luong model
    [('Kuala', 'NOUN'), ('Lumpur', 'NOUN'), (':', 'PUNCT'), ('Sempena', 'CCONJ'), ('sambutan', 'NOUN'), ('Aidilfitri', 'NOUN'), ('minggu', 'NOUN'), ('depan', 'ADJ'), (',', 'PUNCT'), ('Perdana', 'PROPN'), ('Menteri', 'PROPN'), ('Tun', 'PROPN'), ('Dr', 'PROPN'), ('Mahathir', 'ADV'), ('Mohamad', 'ADV'), ('dan', 'CCONJ'), ('Menteri', 'PROPN'), ('Pengangkutan', 'CCONJ'), ('Anthony', 'PROPN'), ('Loke', 'PROPN'), ('Siew', 'PROPN'), ('Fook', 'PROPN'), ('menitipkan', 'PROPN'), ('pesanan', 'SCONJ'), ('khas', 'ADV'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'NOUN'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing', 'ADV'), ('-', 'PUNCT'), ('masing', 'ADV'), ('.', 'PUNCT'), ('Dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('Jabatan', 'NOUN'), ('Keselamatan', 'NOUN'), ('Jalan', 'NOUN'), ('Raya', 'NOUN'), ('(', 'PUNCT'), ('Jkjr', 'ADV'), (')', 'PUNCT'), ('itu', 'DET'), (',', 'PUNCT'), ('Dr', 'PROPN'), ('Mahathir', 'VERB'), ('menasihati', 'NOUN'), ('mereka', 'PRON'), ('supaya', 'SCONJ'), ('berhenti', 'VERB'), ('berehat', 'ADV'), ('dan', 'CCONJ'), ('tidur', 'NOUN'), ('sebentar', 'ADV'), ('sekiranya', 'ADV'), ('mengantuk', 'ADV'), ('ketika', 'SCONJ'), ('memandu', 'NOUN'), ('.', 'PUNCT')]



.. code:: python

    bahdanau = malaya.pos.deep_model('bahdanau')
    bahdanau.analyze(string)




.. parsed-literal::

    {'words': ['Kuala',
      'Lumpur',
      ':',
      'Sempena',
      'sambutan',
      'Aidilfitri',
      'minggu',
      'depan',
      ',',
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
      'masing',
      '-',
      'masing',
      '.',
      'Dalam',
      'video',
      'pendek',
      'terbitan',
      'Jabatan',
      'Keselamatan',
      'Jalan',
      'Raya',
      '(',
      'Jkjr',
      ')',
      'itu',
      ',',
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
      'memandu',
      '.'],
     'tags': [{'text': 'Kuala',
       'type': 'PROPN',
       'score': 1.0,
       'beginOffset': 0,
       'endOffset': 0},
      {'text': 'Lumpur',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 1,
       'endOffset': 1},
      {'text': ':',
       'type': 'PUNCT',
       'score': 1.0,
       'beginOffset': 2,
       'endOffset': 2},
      {'text': 'Sempena sambutan Aidilfitri minggu',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 3,
       'endOffset': 6},
      {'text': 'depan',
       'type': 'ADJ',
       'score': 1.0,
       'beginOffset': 7,
       'endOffset': 7},
      {'text': ',',
       'type': 'PUNCT',
       'score': 1.0,
       'beginOffset': 8,
       'endOffset': 8},
      {'text': 'Perdana Menteri Tun',
       'type': 'PROPN',
       'score': 1.0,
       'beginOffset': 9,
       'endOffset': 11},
      {'text': 'Dr',
       'type': 'PUNCT',
       'score': 1.0,
       'beginOffset': 12,
       'endOffset': 12},
      {'text': 'Mahathir',
       'type': 'VERB',
       'score': 1.0,
       'beginOffset': 13,
       'endOffset': 13},
      {'text': 'Mohamad',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 14,
       'endOffset': 14},
      {'text': 'dan',
       'type': 'CCONJ',
       'score': 1.0,
       'beginOffset': 15,
       'endOffset': 15},
      {'text': 'Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 16,
       'endOffset': 23},
      {'text': 'khas',
       'type': 'ADJ',
       'score': 1.0,
       'beginOffset': 24,
       'endOffset': 24},
      {'text': 'kepada',
       'type': 'ADP',
       'score': 1.0,
       'beginOffset': 25,
       'endOffset': 25},
      {'text': 'orang ramai',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 26,
       'endOffset': 27},
      {'text': 'yang',
       'type': 'PRON',
       'score': 1.0,
       'beginOffset': 28,
       'endOffset': 28},
      {'text': 'mahu',
       'type': 'ADV',
       'score': 1.0,
       'beginOffset': 29,
       'endOffset': 29},
      {'text': 'pulang',
       'type': 'VERB',
       'score': 1.0,
       'beginOffset': 30,
       'endOffset': 30},
      {'text': 'ke',
       'type': 'ADP',
       'score': 1.0,
       'beginOffset': 31,
       'endOffset': 31},
      {'text': 'kampung halaman',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 32,
       'endOffset': 33},
      {'text': 'masing',
       'type': 'ADV',
       'score': 1.0,
       'beginOffset': 34,
       'endOffset': 34},
      {'text': '-',
       'type': 'PUNCT',
       'score': 1.0,
       'beginOffset': 35,
       'endOffset': 35},
      {'text': 'masing',
       'type': 'ADV',
       'score': 1.0,
       'beginOffset': 36,
       'endOffset': 36},
      {'text': '.',
       'type': 'PUNCT',
       'score': 1.0,
       'beginOffset': 37,
       'endOffset': 37},
      {'text': 'Dalam',
       'type': 'ADP',
       'score': 1.0,
       'beginOffset': 38,
       'endOffset': 38},
      {'text': 'video',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 39,
       'endOffset': 39},
      {'text': 'pendek',
       'type': 'ADJ',
       'score': 1.0,
       'beginOffset': 40,
       'endOffset': 40},
      {'text': 'terbitan Jabatan Keselamatan Jalan',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 41,
       'endOffset': 44},
      {'text': 'Raya',
       'type': 'PROPN',
       'score': 1.0,
       'beginOffset': 45,
       'endOffset': 45},
      {'text': '(',
       'type': 'PUNCT',
       'score': 1.0,
       'beginOffset': 46,
       'endOffset': 46},
      {'text': 'Jkjr',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 47,
       'endOffset': 47},
      {'text': ')',
       'type': 'PUNCT',
       'score': 1.0,
       'beginOffset': 48,
       'endOffset': 48},
      {'text': 'itu',
       'type': 'DET',
       'score': 1.0,
       'beginOffset': 49,
       'endOffset': 49},
      {'text': ',',
       'type': 'PUNCT',
       'score': 1.0,
       'beginOffset': 50,
       'endOffset': 50},
      {'text': 'Dr',
       'type': 'PRON',
       'score': 1.0,
       'beginOffset': 51,
       'endOffset': 51},
      {'text': 'Mahathir',
       'type': 'VERB',
       'score': 1.0,
       'beginOffset': 52,
       'endOffset': 52},
      {'text': 'menasihati',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 53,
       'endOffset': 53},
      {'text': 'mereka',
       'type': 'PRON',
       'score': 1.0,
       'beginOffset': 54,
       'endOffset': 54},
      {'text': 'supaya',
       'type': 'CCONJ',
       'score': 1.0,
       'beginOffset': 55,
       'endOffset': 55},
      {'text': 'berhenti',
       'type': 'VERB',
       'score': 1.0,
       'beginOffset': 56,
       'endOffset': 56},
      {'text': 'berehat',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 57,
       'endOffset': 57},
      {'text': 'dan',
       'type': 'CCONJ',
       'score': 1.0,
       'beginOffset': 58,
       'endOffset': 58},
      {'text': 'tidur',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 59,
       'endOffset': 59},
      {'text': 'sebentar',
       'type': 'ADV',
       'score': 1.0,
       'beginOffset': 60,
       'endOffset': 60},
      {'text': 'sekiranya mengantuk',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 61,
       'endOffset': 62},
      {'text': 'ketika',
       'type': 'SCONJ',
       'score': 1.0,
       'beginOffset': 63,
       'endOffset': 63},
      {'text': 'memandu',
       'type': 'NOUN',
       'score': 1.0,
       'beginOffset': 64,
       'endOffset': 64}]}



Print important features from deep learning model
-------------------------------------------------

.. code:: python

    bahdanau.print_features(10)


.. parsed-literal::

    Top-10 positive:
    Mengorbit: 4.307532
    massa: 4.232903
    Terdapat: 4.153726
    office: 4.106574
    memberlakukan: 4.015248
    mengacau: 3.976333
    gigih: 3.964687
    dilalap: 3.940776
    rasio: 3.932958
    dilepaskan: 3.925436

    Top-10 negative:
    injili: -4.493387
    2013: -4.486157
    Hermann: -4.029718
    Redaksi: -4.018761
    menikahimu: -3.941787
    Bangsamoro: -3.782915
    oxlevanus: -3.771584
    Roundup: -3.732997
    George: -3.702288
    Pi: -3.677682


Print important transitions from deep learning model
----------------------------------------------------

.. code:: python

    bahdanau.print_transitions(10)


.. parsed-literal::

    Top-10 likely transitions:
    ADP -> NOUN: 0.630389
    PART -> ADV: 0.540397
    ADJ -> PUNCT: 0.538496
    ADJ -> ADP: 0.514535
    NOUN -> ADJ: 0.507203
    X -> VERB: 0.427052
    ADJ -> CCONJ: 0.408229
    PUNCT -> X: 0.398683
    NUM -> NOUN: 0.368446
    PROPN -> PROPN: 0.367188

    Top-10 unlikely transitions:
    X -> PRON: -1.137289
    DET -> X: -1.135260
    X -> ADP: -1.133719
    PROPN -> NOUN: -1.130220
    VERB -> X: -1.125563
    X -> CCONJ: -1.094884
    DET -> SYM: -1.031330
    AUX -> CCONJ: -1.014629
    PART -> DET: -0.970728
    ADP -> SCONJ: -0.969023


Voting stack model
------------------

.. code:: python

    bahdanau = malaya.pos.deep_model('bahdanau')
    bert_small = malaya.pos.bert('small')
    bert = malaya.pos.bert('base')
    malaya.stack.voting_stack([bert, bahdanau, bert_small], string)




.. parsed-literal::

    [('Kuala', 'PROPN'),
     ('Lumpur', 'PROPN'),
     (':', 'PUNCT'),
     ('Sempena', 'ADP'),
     ('sambutan', 'NOUN'),
     ('Aidilfitri', 'NOUN'),
     ('minggu', 'NOUN'),
     ('depan', 'ADJ'),
     (',', 'PUNCT'),
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
     ('masing', 'DET'),
     ('-', 'PUNCT'),
     ('masing', 'DET'),
     ('.', 'PUNCT'),
     ('Dalam', 'ADP'),
     ('video', 'NOUN'),
     ('pendek', 'ADJ'),
     ('terbitan', 'NOUN'),
     ('Jabatan', 'PROPN'),
     ('Keselamatan', 'PROPN'),
     ('Jalan', 'PROPN'),
     ('Raya', 'PROPN'),
     ('(', 'PUNCT'),
     ('Jkjr', 'PROPN'),
     (')', 'PUNCT'),
     ('itu', 'DET'),
     (',', 'PUNCT'),
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
     ('memandu', 'NOUN'),
     ('.', 'PUNCT')]
