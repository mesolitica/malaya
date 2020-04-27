.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 4.21 s, sys: 798 ms, total: 5.01 s
    Wall time: 4.31 s


Describe supported entities
---------------------------

.. code:: ipython3

    malaya.entity.describe()


.. parsed-literal::

    OTHER - Other
    law - law, regulation, related law documents, documents, etc
    location - location, place
    organization - organization, company, government, facilities, etc
    person - person, group of people, believes, unique arts (eg; food, drink), etc
    quantity - numbers, quantity
    time - date, day, time, etc
    event - unique event happened, etc


List available Transformer NER models
-------------------------------------

.. code:: ipython3

    malaya.entity.available_transformer_model()




.. parsed-literal::

    {'bert': ['426.4 MB', 'accuracy: 0.994'],
     'tiny-bert': ['57.7 MB', 'accuracy: 0.986'],
     'albert': ['48.6 MB', 'accuracy: 0.984'],
     'tiny-albert': ['22.4 MB', 'accuracy: 0.971'],
     'xlnet': ['446.6 MB', 'accuracy: 0.992'],
     'alxlnet': ['46.8 MB', 'accuracy: 0.993']}



Make sure you can check accuracy chart from here first before select a
model,
https://malaya.readthedocs.io/en/latest/Accuracy.html#entities-recognition

**You might want to use Tiny-Albert, a very small size, 22.4MB, but the
accuracy is still on the top notch.**

.. code:: ipython3

    string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'

Load ALBERT model
-----------------

.. code:: ipython3

    model = malaya.entity.transformer(model = 'albert')


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:54: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:55: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/albert/tokenization.py:240: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.
    
    INFO:tensorflow:loading sentence piece model
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:49: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


.. code:: ipython3

    model.predict(string)




.. parsed-literal::

    [('Kuala', 'location'),
     ('Lumpur:', 'location'),
     ('Sempena', 'OTHER'),
     ('sambutan', 'OTHER'),
     ('Aidilfitri', 'event'),
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
     ('kampung', 'OTHER'),
     ('halaman', 'OTHER'),
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
      {'text': 'minggu depan,',
       'type': 'OTHER',
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
      {'text': 'menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan',
       'type': 'OTHER',
       'score': 1.0,
       'beginOffset': 20,
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
     'OTHER': ['baca buku Perlembagaan yang berharga 3k ringgit dekat',
      'minggu lepas,',
      ', suhu 32 celcius, sambil makan ayam goreng dan milo o ais'],
     'location': ['kfc sungai petani'],
     'time': {'2 PM': datetime.datetime(2020, 4, 26, 14, 0),
      '2 PM 2 oktober 2019': datetime.datetime(2019, 10, 2, 14, 0)},
     'date': {'2 oktober 2019': datetime.datetime(2019, 10, 2, 0, 0),
      'minggu lalu': datetime.datetime(2020, 4, 19, 23, 51, 23, 231714)},
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
     'drink': ['milo o ais'],
     'weight': []}



.. code:: ipython3

    entity.predict('contact Husein at husein.zol05@gmail.com')




.. parsed-literal::

    {'OTHER': ['contact', 'at'],
     'person': ['Husein', 'husein.zol05@gmail.com'],
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
     'drink': [],
     'weight': []}



.. code:: ipython3

    entity.predict('tolong tempahkan meja makan makan nasi dagang dan jus apple, milo tarik esok dekat Restoran Sebulek')




.. parsed-literal::

    {'OTHER': ['tolong tempahkan meja makan makan', 'dan', 'esok dekat'],
     'person': ['nasi dagang', 'jus apple, milo tarik', 'Restoran Sebulek'],
     'date': {'esok': datetime.datetime(2020, 4, 27, 23, 51, 23, 646172)},
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
     'drink': ['milo tarik', 'jus apple'],
     'weight': []}



Voting stack model
------------------

.. code:: ipython3

    alxlnet = malaya.entity.transformer(model = 'alxlnet')
    malaya.stack.voting_stack([model, alxlnet, alxlnet],
    'tolong tempahkan meja makan makan nasi dagang dan jus apple, milo tarik esok dekat Restoran Sebulek')




.. parsed-literal::

    [('tolong', 'OTHER'),
     ('tempahkan', 'OTHER'),
     ('meja', 'OTHER'),
     ('makan', 'OTHER'),
     ('makan', 'OTHER'),
     ('nasi', 'OTHER'),
     ('dagang', 'person'),
     ('dan', 'OTHER'),
     ('jus', 'OTHER'),
     ('apple,', 'person'),
     ('milo', 'person'),
     ('tarik', 'OTHER'),
     ('esok', 'OTHER'),
     ('dekat', 'OTHER'),
     ('Restoran', 'organization'),
     ('Sebulek', 'organization')]



