
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 9 µs, sys: 1 µs, total: 10 µs
    Wall time: 17.2 µs


BERT model
----------

BERT is the best NER model in term of accuracy, you can check NER
accuracy here,
https://malaya.readthedocs.io/en/latest/Accuracy.html#entities-recognition.
Question is, why BERT?

1. Transformer model learn the context of a word based on all of its
   surroundings (live string), bidirectionally. So it much better
   understand left and right hand side relationships.
2. Because of transformer able to leverage to context during live
   string, we dont need to capture available words in this world,
   instead capture substrings and build the attention after that. BERT
   will never have Out-Of-Vocab problem.

List available BERT NER models
------------------------------

.. code:: ipython3

    malaya.entity.available_bert_model()




.. parsed-literal::

    ['multilanguage', 'base', 'small']



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

Load BERT models
----------------

.. code:: ipython3

    model = malaya.entity.bert(model = 'base')


.. parsed-literal::

    WARNING: Logging before flag parsing goes to stderr.
    W0807 17:19:59.994667 4422120896 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:45: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    W0807 17:19:59.995772 4422120896 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:46: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    W0807 17:20:09.183666 4422120896 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:41: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


.. code:: ipython3

    model.predict(string)




.. parsed-literal::

    [('Kuala', 'location'),
     ('Lumpur', 'location'),
     ('Sempena', 'OTHER'),
     ('sambutan', 'OTHER'),
     ('Aidilfitri', 'OTHER'),
     ('minggu', 'OTHER'),
     ('depan', 'OTHER'),
     ('Perdana', 'person'),
     ('Menteri', 'person'),
     ('Tun', 'person'),
     ('Dr', 'person'),
     ('Mahathir', 'person'),
     ('Mohamad', 'person'),
     ('dan', 'OTHER'),
     ('Menteri', 'person'),
     ('Pengangkutan', 'person'),
     ('Anthony', 'person'),
     ('Loke', 'person'),
     ('Siew', 'person'),
     ('Fook', 'person'),
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
     ('halaman', 'location'),
     ('masing-masing', 'OTHER'),
     ('Dalam', 'OTHER'),
     ('video', 'OTHER'),
     ('pendek', 'OTHER'),
     ('terbitan', 'OTHER'),
     ('Jabatan', 'organization'),
     ('Keselamatan', 'organization'),
     ('Jalan', 'organization'),
     ('Raya', 'organization'),
     ('(Jkjr)', 'organization'),
     ('itu', 'OTHER'),
     ('Dr', 'person'),
     ('Mahathir', 'person'),
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



.. code:: ipython3

    model.analyze(string)




.. parsed-literal::

    {'words': ['Kuala',
      'Lumpur',
      'Sempena',
      'sambutan',
      'Aidilfitri',
      'minggu',
      'depan',
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
      'masing-masing',
      'Dalam',
      'video',
      'pendek',
      'terbitan',
      'Jabatan',
      'Keselamatan',
      'Jalan',
      'Raya',
      '(Jkjr)',
      'itu',
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
      'memandu'],
     'tags': [{'text': 'Kuala Lumpur',
       'type': 'location',
       'score': 1.0,
       'beginOffset': 0,
       'endOffset': 1},
      {'text': 'Sempena sambutan Aidilfitri minggu depan',
       'type': 'OTHER',
       'score': 1.0,
       'beginOffset': 2,
       'endOffset': 6},
      {'text': 'Perdana Menteri Tun Dr Mahathir Mohamad',
       'type': 'person',
       'score': 1.0,
       'beginOffset': 7,
       'endOffset': 12},
      {'text': 'dan',
       'type': 'OTHER',
       'score': 1.0,
       'beginOffset': 13,
       'endOffset': 13},
      {'text': 'Menteri Pengangkutan Anthony Loke Siew Fook',
       'type': 'person',
       'score': 1.0,
       'beginOffset': 14,
       'endOffset': 19},
      {'text': 'menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing Dalam video pendek terbitan',
       'type': 'OTHER',
       'score': 1.0,
       'beginOffset': 20,
       'endOffset': 36},
      {'text': 'Jabatan Keselamatan Jalan Raya (Jkjr)',
       'type': 'organization',
       'score': 1.0,
       'beginOffset': 37,
       'endOffset': 41},
      {'text': 'itu',
       'type': 'OTHER',
       'score': 1.0,
       'beginOffset': 42,
       'endOffset': 42},
      {'text': 'Dr Mahathir',
       'type': 'person',
       'score': 1.0,
       'beginOffset': 43,
       'endOffset': 44}]}



Load general Malaya entity model
--------------------------------

This model required external entity tagging model like bert or deep
learning, and this model able to classify,

1.  date
2.  money
3.  temperature
4.  distance
5.  volume
6.  duration
7.  phone
8.  email
9.  url
10. time

.. code:: ipython3

    model = malaya.entity.bert(model = 'small')
    entity = malaya.entity.general_entity(model = model)


.. parsed-literal::

    WARNING: Logging before flag parsing goes to stderr.
    W0914 01:36:43.574370 4646454720 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:45: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    W0914 01:36:43.575525 4646454720 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:46: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    W0914 01:36:48.098227 4646454720 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:41: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


.. code:: ipython3

    entity.predict('Husein baca buku Perlembagaan yang berharga 3k ringgit dekat kfc sungai petani minggu lepas, pukul 2 petang , suhu 32 celcius')




.. parsed-literal::

    {'OTHER': ['baca buku',
      'yang berharga 3k ringgit dekat',
      'minggu lepas pukul',
      'suhu 32 celcius'],
     'law': ['Perlembagaan'],
     'location': ['sungai petani'],
     'organization': ['kfc'],
     'person': ['Husein'],
     'quantity': [],
     'time': ['2 petang'],
     'event': [],
     'date': {'minggu lalu': datetime.datetime(2019, 9, 7, 1, 37, 1, 489575)},
     'money': {'3k ringgit': 'RM3000'},
     'temperature': ['32 celcius'],
     'distance': [],
     'volume': [],
     'duration': [],
     'phone': [],
     'email': [],
     'url': []}



.. code:: ipython3

    entity.predict('contact Husein at husein.zol05@gmail.com')




.. parsed-literal::

    {'OTHER': ['contact', 'at', 'gmail com'],
     'law': [],
     'location': [],
     'organization': ['zol05'],
     'person': ['Husein', 'husein'],
     'quantity': [],
     'time': [],
     'event': [],
     'date': {},
     'money': {},
     'temperature': [],
     'distance': [],
     'volume': [],
     'duration': [],
     'phone': [],
     'email': ['husein.zol05@gmail.com'],
     'url': []}



.. code:: ipython3

    entity.predict('tolong tempahkan meja makan esok dekat Restoran Sebulek')




.. parsed-literal::

    {'OTHER': ['tolong tempahkan meja makan esok dekat'],
     'law': [],
     'location': ['Restoran Sebulek'],
     'organization': [],
     'person': [],
     'quantity': [],
     'time': [],
     'event': [],
     'date': {'esok': datetime.datetime(2019, 9, 15, 1, 37, 30, 850860)},
     'money': {},
     'temperature': [],
     'distance': [],
     'volume': [],
     'duration': [],
     'phone': [],
     'email': [],
     'url': []}



List available deep learning models
-----------------------------------

.. code:: ipython3

    malaya.entity.available_deep_model()




.. parsed-literal::

    ['concat', 'bahdanau', 'luong']



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

    19.0MB [00:03, 5.98MB/s]                          


.. parsed-literal::

    [('KUALA', 'location'), ('LUMPUR', 'location'), ('Sempena', 'OTHER'), ('sambutan', 'OTHER'), ('Aidilfitri', 'time'), ('minggu', 'time'), ('depan', 'time'), ('Perdana', 'person'), ('Menteri', 'person'), ('Tun', 'person'), ('Dr', 'person'), ('Mahathir', 'person'), ('Mohamad', 'person'), ('dan', 'OTHER'), ('Menteri', 'person'), ('Pengangkutan', 'person'), ('Anthony', 'person'), ('Loke', 'person'), ('Siew', 'person'), ('Fook', 'person'), ('menitipkan', 'person'), ('pesanan', 'OTHER'), ('khas', 'OTHER'), ('kepada', 'OTHER'), ('orang', 'OTHER'), ('ramai', 'OTHER'), ('yang', 'OTHER'), ('mahu', 'OTHER'), ('pulang', 'OTHER'), ('ke', 'OTHER'), ('kampung', 'location'), ('halaman', 'location'), ('masing-masing', 'OTHER'), ('Dalam', 'OTHER'), ('video', 'OTHER'), ('pendek', 'OTHER'), ('terbitan', 'OTHER'), ('Jabatan', 'organization'), ('Keselamatan', 'organization'), ('Jalan', 'organization'), ('Raya', 'organization'), ('(JKJR)', 'location'), ('itu', 'OTHER'), ('Dr', 'person'), ('Mahathir', 'person'), ('menasihati', 'OTHER'), ('mereka', 'OTHER'), ('supaya', 'OTHER'), ('berhenti', 'OTHER'), ('berehat', 'person'), ('dan', 'OTHER'), ('tidur', 'OTHER'), ('sebentar', 'OTHER'), ('sekiranya', 'OTHER'), ('mengantuk', 'OTHER'), ('ketika', 'OTHER'), ('memandu', 'OTHER')]
    
    Testing bahdanau model
    [('KUALA', 'location'), ('LUMPUR', 'location'), ('Sempena', 'OTHER'), ('sambutan', 'OTHER'), ('Aidilfitri', 'location'), ('minggu', 'time'), ('depan', 'time'), ('Perdana', 'location'), ('Menteri', 'person'), ('Tun', 'person'), ('Dr', 'person'), ('Mahathir', 'person'), ('Mohamad', 'person'), ('dan', 'OTHER'), ('Menteri', 'person'), ('Pengangkutan', 'person'), ('Anthony', 'person'), ('Loke', 'person'), ('Siew', 'person'), ('Fook', 'person'), ('menitipkan', 'person'), ('pesanan', 'OTHER'), ('khas', 'OTHER'), ('kepada', 'OTHER'), ('orang', 'OTHER'), ('ramai', 'OTHER'), ('yang', 'OTHER'), ('mahu', 'OTHER'), ('pulang', 'OTHER'), ('ke', 'OTHER'), ('kampung', 'location'), ('halaman', 'OTHER'), ('masing-masing', 'OTHER'), ('Dalam', 'OTHER'), ('video', 'OTHER'), ('pendek', 'OTHER'), ('terbitan', 'OTHER'), ('Jabatan', 'organization'), ('Keselamatan', 'organization'), ('Jalan', 'organization'), ('Raya', 'organization'), ('(JKJR)', 'OTHER'), ('itu', 'OTHER'), ('Dr', 'person'), ('Mahathir', 'person'), ('menasihati', 'OTHER'), ('mereka', 'OTHER'), ('supaya', 'OTHER'), ('berhenti', 'OTHER'), ('berehat', 'OTHER'), ('dan', 'OTHER'), ('tidur', 'OTHER'), ('sebentar', 'OTHER'), ('sekiranya', 'OTHER'), ('mengantuk', 'location'), ('ketika', 'OTHER'), ('memandu', 'OTHER')]
    
    Testing luong model
    [('KUALA', 'location'), ('LUMPUR', 'location'), ('Sempena', 'OTHER'), ('sambutan', 'OTHER'), ('Aidilfitri', 'organization'), ('minggu', 'time'), ('depan', 'time'), ('Perdana', 'person'), ('Menteri', 'person'), ('Tun', 'person'), ('Dr', 'person'), ('Mahathir', 'person'), ('Mohamad', 'person'), ('dan', 'OTHER'), ('Menteri', 'person'), ('Pengangkutan', 'person'), ('Anthony', 'person'), ('Loke', 'person'), ('Siew', 'person'), ('Fook', 'person'), ('menitipkan', 'OTHER'), ('pesanan', 'OTHER'), ('khas', 'OTHER'), ('kepada', 'OTHER'), ('orang', 'OTHER'), ('ramai', 'OTHER'), ('yang', 'OTHER'), ('mahu', 'OTHER'), ('pulang', 'OTHER'), ('ke', 'OTHER'), ('kampung', 'location'), ('halaman', 'location'), ('masing-masing', 'OTHER'), ('Dalam', 'OTHER'), ('video', 'OTHER'), ('pendek', 'OTHER'), ('terbitan', 'OTHER'), ('Jabatan', 'organization'), ('Keselamatan', 'organization'), ('Jalan', 'organization'), ('Raya', 'organization'), ('(JKJR)', 'location'), ('itu', 'OTHER'), ('Dr', 'person'), ('Mahathir', 'person'), ('menasihati', 'OTHER'), ('mereka', 'OTHER'), ('supaya', 'OTHER'), ('berhenti', 'OTHER'), ('berehat', 'OTHER'), ('dan', 'OTHER'), ('tidur', 'OTHER'), ('sebentar', 'OTHER'), ('sekiranya', 'OTHER'), ('mengantuk', 'organization'), ('ketika', 'OTHER'), ('memandu', 'OTHER')]
    


.. code:: ipython3

    bahdanau = malaya.entity.deep_model('bahdanau')
    bahdanau.analyze(string)




.. parsed-literal::

    {'words': ['KUALA',
      'LUMPUR',
      'Sempena',
      'sambutan',
      'Aidilfitri',
      'minggu',
      'depan',
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
      'masing-masing',
      'Dalam',
      'video',
      'pendek',
      'terbitan',
      'Jabatan',
      'Keselamatan',
      'Jalan',
      'Raya',
      '(JKJR)',
      'itu',
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
      'memandu'],
     'tags': [{'text': 'KUALA LUMPUR',
       'type': 'location',
       'score': 1.0,
       'beginOffset': 0,
       'endOffset': 1},
      {'text': 'Sempena sambutan',
       'type': 'OTHER',
       'score': 1.0,
       'beginOffset': 2,
       'endOffset': 3},
      {'text': 'Aidilfitri',
       'type': 'event',
       'score': 1.0,
       'beginOffset': 4,
       'endOffset': 4},
      {'text': 'minggu depan',
       'type': 'time',
       'score': 1.0,
       'beginOffset': 5,
       'endOffset': 6},
      {'text': 'Perdana Menteri Tun Dr Mahathir Mohamad',
       'type': 'person',
       'score': 1.0,
       'beginOffset': 7,
       'endOffset': 12},
      {'text': 'dan',
       'type': 'OTHER',
       'score': 1.0,
       'beginOffset': 13,
       'endOffset': 13},
      {'text': 'Menteri Pengangkutan Anthony Loke Siew Fook',
       'type': 'person',
       'score': 1.0,
       'beginOffset': 14,
       'endOffset': 19},
      {'text': 'menitipkan pesanan khas kepada orang ramai yang mahu pulang ke',
       'type': 'OTHER',
       'score': 1.0,
       'beginOffset': 20,
       'endOffset': 29},
      {'text': 'kampung',
       'type': 'location',
       'score': 1.0,
       'beginOffset': 30,
       'endOffset': 30},
      {'text': 'halaman masing-masing Dalam video pendek terbitan',
       'type': 'OTHER',
       'score': 1.0,
       'beginOffset': 31,
       'endOffset': 36},
      {'text': 'Jabatan Keselamatan Jalan Raya',
       'type': 'organization',
       'score': 1.0,
       'beginOffset': 37,
       'endOffset': 40},
      {'text': '(JKJR)',
       'type': 'person',
       'score': 1.0,
       'beginOffset': 41,
       'endOffset': 41},
      {'text': 'itu',
       'type': 'OTHER',
       'score': 1.0,
       'beginOffset': 42,
       'endOffset': 42},
      {'text': 'Dr Mahathir',
       'type': 'person',
       'score': 1.0,
       'beginOffset': 43,
       'endOffset': 44},
      {'text': 'menasihati mereka supaya',
       'type': 'OTHER',
       'score': 1.0,
       'beginOffset': 45,
       'endOffset': 47},
      {'text': 'berhenti berehat',
       'type': 'person',
       'score': 1.0,
       'beginOffset': 48,
       'endOffset': 49}]}



Print important features from deep learning model
-------------------------------------------------

.. code:: ipython3

    bahdanau = malaya.entity.deep_model('bahdanau')
    bahdanau.print_features(10)


.. parsed-literal::

    Top-10 positive:
    the: 10.046949
    (pdrm): 9.305849
    rahimah,: 7.731076
    giacc: 7.596547
    umno,: 7.465683
    Garuda: 7.419884
    nahdlatul: 7.338985
    dpa: 7.324896
    dikaji: 7.246631
    Giwangan: 7.170110
    
    Top-10 negative:
    tetangga: -9.680513
    independen: -9.539474
    302: -9.263991
    memandangkan: -9.190342
    ke-80: -8.990475
    keistimewaan: -8.617548
    pesan: -8.553379
    Sekjen: -8.510725
    rasa: -8.442114
    lepas.: -8.440548


Print important transitions from deep learning model
----------------------------------------------------

.. code:: ipython3

    bahdanau.print_transitions(10)


.. parsed-literal::

    Top-10 likely transitions:
    quantity -> quantity: 0.768479
    law -> law: 0.748858
    event -> event: 0.671466
    time -> time: 0.566861
    quantity -> PAD: 0.515885
    organization -> time: 0.430649
    PAD -> law: 0.396928
    time -> person: 0.387298
    time -> organization: 0.380183
    OTHER -> time: 0.346963
    
    Top-10 unlikely transitions:
    person -> law: -0.959066
    law -> person: -0.763240
    event -> organization: -0.744430
    person -> event: -0.647477
    time -> event: -0.640794
    law -> OTHER: -0.634643
    organization -> event: -0.629229
    organization -> OTHER: -0.606970
    OTHER -> law: -0.598875
    OTHER -> event: -0.598665


Voting stack model
------------------

.. code:: ipython3

    bahdanau = malaya.entity.deep_model('bahdanau')
    luong = malaya.entity.deep_model('luong')
    bert = malaya.entity.bert('base')
    malaya.stack.voting_stack([bert, bahdanau, luong], string)




.. parsed-literal::

    [('KUALA', 'location'),
     ('LUMPUR', 'location'),
     ('Sempena', 'OTHER'),
     ('sambutan', 'OTHER'),
     ('Aidilfitri', 'organization'),
     ('minggu', 'time'),
     ('depan', 'time'),
     ('Perdana', 'person'),
     ('Menteri', 'person'),
     ('Tun', 'person'),
     ('Dr', 'person'),
     ('Mahathir', 'person'),
     ('Mohamad', 'person'),
     ('dan', 'OTHER'),
     ('Menteri', 'person'),
     ('Pengangkutan', 'person'),
     ('Anthony', 'person'),
     ('Loke', 'person'),
     ('Siew', 'person'),
     ('Fook', 'person'),
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
     ('Dalam', 'OTHER'),
     ('video', 'OTHER'),
     ('pendek', 'OTHER'),
     ('terbitan', 'OTHER'),
     ('Jabatan', 'organization'),
     ('Keselamatan', 'organization'),
     ('Jalan', 'organization'),
     ('Raya', 'organization'),
     ('(JKJR)', 'person'),
     ('itu', 'OTHER'),
     ('Dr', 'person'),
     ('Mahathir', 'person'),
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


