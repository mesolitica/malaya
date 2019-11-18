
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 6.58 s, sys: 1.5 s, total: 8.08 s
    Wall time: 12.3 s


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


List available Transformer NER models
-------------------------------------

.. code:: ipython3

    malaya.entity.available_transformer_model()




.. parsed-literal::

    {'bert': ['base', 'small'], 'xlnet': ['base'], 'albert': ['base']}



Make sure you can check accuracy chart from here first before select a
model,
https://malaya.readthedocs.io/en/latest/Accuracy.html#entities-recognition

**You might want to use ALBERT, a very small size, 43MB, but the
accuracy is still on the top notch.**

.. code:: ipython3

    string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'

Load ALBERT model
-----------------

.. code:: ipython3

    model = malaya.entity.transformer(model = 'albert', size = 'base')


.. parsed-literal::

    WARNING: Logging before flag parsing goes to stderr.
    W1017 22:28:20.427351 4703032768 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:68: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    W1017 22:28:20.428478 4703032768 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:69: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    W1017 22:28:21.298430 4703032768 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:64: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


.. code:: ipython3

    model.predict(string)




.. parsed-literal::

    [('Kuala', 'location'),
     ('Lumpur:', 'location'),
     ('Sempena', 'OTHER'),
     ('sambutan', 'OTHER'),
     ('Aidilfitri', 'OTHER'),
     ('minggu', 'OTHER'),
     ('depan,', 'OTHER'),
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
     ('masing-masing.', 'OTHER'),
     ('Dalam', 'OTHER'),
     ('video', 'OTHER'),
     ('pendek', 'OTHER'),
     ('terbitan', 'OTHER'),
     ('Jabatan', 'organization'),
     ('Keselamatan', 'organization'),
     ('Jalan', 'organization'),
     ('Raya', 'organization'),
     ('(JKJR)', 'organization'),
     ('itu,', 'OTHER'),
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
     ('memandu.', 'OTHER')]



.. code:: ipython3

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
       'type': 'location',
       'score': 1.0,
       'beginOffset': 0,
       'endOffset': 1},
      {'text': 'Sempena sambutan Aidilfitri minggu depan,',
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
      {'text': 'menitipkan pesanan khas kepada orang ramai yang mahu pulang ke',
       'type': 'OTHER',
       'score': 1.0,
       'beginOffset': 20,
       'endOffset': 29},
      {'text': 'kampung halaman',
       'type': 'location',
       'score': 1.0,
       'beginOffset': 30,
       'endOffset': 31},
      {'text': 'masing-masing. Dalam video pendek terbitan',
       'type': 'OTHER',
       'score': 1.0,
       'beginOffset': 32,
       'endOffset': 36},
      {'text': 'Jabatan Keselamatan Jalan Raya (JKJR)',
       'type': 'organization',
       'score': 1.0,
       'beginOffset': 37,
       'endOffset': 41},
      {'text': 'itu,',
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

This model able to classify,

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
11. datetime
12. local and generic foods, can check available rules in
    malaya.texts._food
13. local and generic drinks, can check available rules in
    malaya.texts._food

We can insert BERT or any deep learning model by passing
``malaya.entity.general_entity(model = model)``, as long the model has
``predict`` method and return ``[(string, label), (string, label)]``.
This is an optional.

.. code:: ipython3

    entity = malaya.entity.general_entity(model = model)

.. code:: ipython3

    entity.predict('Husein baca buku Perlembagaan yang berharga 3k ringgit dekat kfc sungai petani minggu lepas, 2 ptg 2 oktober 2019 , suhu 32 celcius, sambil makan ayam goreng dan milo o ais')




.. parsed-literal::

    {'person': ['Husein'],
     'OTHER': ['baca buku',
      'yang berharga',
      'dekat',
      'lepas, 2 ptg',
      ', suhu 32 celcius, sambil makan ayam goreng dan milo o ais'],
     'law': ['Perlembagaan'],
     'quantity': ['3k ringgit'],
     'location': ['kfc sungai petani'],
     'time': {'2 oktober 2019': datetime.datetime(2019, 10, 2, 0, 0),
      '2 PM': datetime.datetime(2019, 10, 17, 14, 0),
      'minggu': None},
     'date': {'2 oktober 2019': datetime.datetime(2019, 10, 2, 0, 0),
      'minggu lalu': datetime.datetime(2019, 10, 10, 22, 28, 23, 292272)},
     'money': {'3k ringgit': 'RM3000.0'},
     'temperature': ['32 celcius'],
     'distance': [],
     'volume': [],
     'duration': [],
     'phone': [],
     'email': [],
     'url': [],
     'datetime': {'2 ptg 2 oktober 2019': datetime.datetime(2019, 10, 2, 14, 0)},
     'food': ['ayam goreng'],
     'drink': ['milo o ais']}



.. code:: ipython3

    entity.predict('contact Husein at husein.zol05@gmail.com')




.. parsed-literal::

    {'OTHER': ['contact Husein at'],
     'person': ['husein.zol05@gmail.com'],
     'date': {},
     'money': {},
     'temperature': [],
     'distance': [],
     'volume': [],
     'duration': [],
     'phone': [],
     'email': ['husein.zol05@gmail.com'],
     'url': [],
     'time': {},
     'datetime': {},
     'food': [],
     'drink': []}



.. code:: ipython3

    entity.predict('tolong tempahkan meja makan makan nasi dagang dan jus apple, milo tarik esok dekat Restoran Sebulek')




.. parsed-literal::

    {'OTHER': ['tolong tempahkan meja makan makan nasi',
      'dan',
      'tarik esok dekat Restoran'],
     'person': ['dagang', 'jus apple, milo', 'Sebulek'],
     'date': {'esok': datetime.datetime(2019, 10, 18, 22, 28, 26, 567487)},
     'money': {},
     'temperature': [],
     'distance': [],
     'volume': [],
     'duration': [],
     'phone': [],
     'email': [],
     'url': [],
     'time': {},
     'datetime': {},
     'food': ['nasi dagang'],
     'drink': ['milo tarik', 'jus apple']}



Voting stack model
------------------

.. code:: ipython3

    xlnet = malaya.entity.transformer(model = 'xlnet', size = 'base')
    malaya.stack.voting_stack([model, xlnet, xlnet],
    'tolong tempahkan meja makan makan nasi dagang dan jus apple, milo tarik esok dekat Restoran Sebulek')




.. parsed-literal::

    [('tolong', 'OTHER'),
     ('tempahkan', 'OTHER'),
     ('meja', 'OTHER'),
     ('makan', 'OTHER'),
     ('makan', 'OTHER'),
     ('nasi', 'OTHER'),
     ('dagang', 'OTHER'),
     ('dan', 'OTHER'),
     ('jus', 'OTHER'),
     ('apple,', 'OTHER'),
     ('milo', 'person'),
     ('tarik', 'OTHER'),
     ('esok', 'OTHER'),
     ('dekat', 'OTHER'),
     ('Restoran', 'organization'),
     ('Sebulek', 'person')]


