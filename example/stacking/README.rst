
.. code:: ipython3

    import malaya

.. code:: ipython3

    bahdanau = malaya.sentiment.deep_model('bahdanau')
    luong = malaya.sentiment.deep_model('luong')
    entity = malaya.sentiment.deep_model('entity-network')
    multinomial = malaya.sentiment.multinomial()

Stack multiple sentiment models
-------------------------------

.. code:: ipython3

    malaya.stack.predict_stack([bahdanau,
                                luong,
                                entity,
                                multinomial],
                              'harga minyak tak menentu')




.. parsed-literal::

    {'negative': 0.5589608851409659, 'positive': 0.40215390241162385}



Stack multiple toxic models
---------------------------

.. code:: ipython3

    bahdanau = malaya.toxic.deep_model('bahdanau')
    luong = malaya.toxic.deep_model('luong')
    entity = malaya.toxic.deep_model('entity-network')
    malaya.stack.predict_stack([bahdanau,
                                luong,
                                entity],
                              'harga minyak tak menentu')




.. parsed-literal::

    {'toxic': 0.1940847,
     'severe_toxic': 0.06477653,
     'obscene': 0.15512805,
     'threat': 0.13735601,
     'insult': 0.14242107,
     'identity_hate': 0.119892955}



Stack tagging models
--------------------

.. code:: ipython3

    string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'
    entity_network = malaya.entity.deep_model('entity-network')
    bahdanau = malaya.entity.deep_model('bahdanau')
    luong = malaya.entity.deep_model('luong')
    malaya.stack.voting_stack([entity_network, bahdanau, luong], string)




.. parsed-literal::

    [('kuala', 'location'),
     ('lumpur', 'location'),
     ('sempena', 'OTHER'),
     ('sambutan', 'event'),
     ('aidilfitri', 'event'),
     ('minggu', 'time'),
     ('depan', 'time'),
     ('perdana', 'person'),
     ('menteri', 'person'),
     ('tun', 'person'),
     ('dr', 'person'),
     ('mahathir', 'person'),
     ('mohamad', 'person'),
     ('dan', 'OTHER'),
     ('menteri', 'person'),
     ('pengangkutan', 'OTHER'),
     ('anthony', 'person'),
     ('loke', 'person'),
     ('siew', 'person'),
     ('fook', 'person'),
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
     ('masing-masing', 'OTHER'),
     ('dalam', 'OTHER'),
     ('video', 'OTHER'),
     ('pendek', 'OTHER'),
     ('terbitan', 'OTHER'),
     ('jabatan', 'organization'),
     ('keselamatan', 'organization'),
     ('jalan', 'organization'),
     ('raya', 'organization'),
     ('jkjr', 'organization'),
     ('itu', 'OTHER'),
     ('dr', 'person'),
     ('mahathir', 'person'),
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


