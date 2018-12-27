
Why Stacking?
-------------

Sometime a single model is not good enough. So, you need to use multiple
models to get a better result! It called stacking.

.. code:: ipython3

    import malaya

.. code:: ipython3

    bahdanau = malaya.sentiment.deep_model('bahdanau')
    luong = malaya.sentiment.deep_model('luong')
    entity = malaya.sentiment.deep_model('entity-network')
    multinomial = malaya.sentiment.multinomial()

Stack multiple sentiment models
-------------------------------

``malaya.stack.predict_stack`` provide an easy stacking solution for
Malaya models. Well, not just for sentiment models, any classification
models can use ``malaya.stack.predict_stack``.

.. code:: python

   def predict_stack(models, text, mode = 'gmean'):
       """
       Stacking for predictive models.

       Parameters
       ----------
       models: list
           list of models
       text: str
           string to predict
       mode : str, optional (default='gmean')
           Model architecture supported. Allowed values:

           * ``'gmean'`` - geometrical mean
           * ``'hmean'`` - harmonic mean
           * ``'mean'`` - mean
           * ``'min'`` - min
           * ``'max'`` - max
           * ``'median'`` - Harrell-Davis median


       Returns
       -------
       result: dict
       """

.. code:: ipython3

    malaya.stack.predict_stack([bahdanau,
                                luong,
                                entity,
                                multinomial],
                              'harga minyak tak menentu')




.. parsed-literal::

    {'negative': 0.5549062374008705, 'positive': 0.4072814056650461}



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

    {'toxic': 0.2057164,
     'severe_toxic': 0.06787095,
     'obscene': 0.15890868,
     'threat': 0.15786164,
     'insult': 0.15252964,
     'identity_hate': 0.12279783}



Stack language detection models
-------------------------------

.. code:: ipython3

    xgb = malaya.language_detection.xgb()
    multinomial = malaya.language_detection.multinomial()
    sgd = malaya.language_detection.sgd()
    malaya.stack.predict_stack([xgb,
                                multinomial,
                                sgd],
                              'didukungi secara natifnya')




.. parsed-literal::

    {'OTHER': 0.0, 'ENGLISH': 0.0, 'INDONESIA': 0.9305759540118518, 'MALAY': 0.0}



Stack tagging models
--------------------

For tagging models, we use majority voting stacking. So you need to need
have more than 2 models to make it perfect, or else, it will pick
randomly from 2 models. ``malaya.stack.voting_stack`` provides easy
interface for this kind of stacking. **But only can use for Entites and
POS recognition.**

.. code:: python

   def voting_stack(models, text):
       """
       Stacking for POS and Entities Recognition models.

       Parameters
       ----------
       models: list
           list of models
       text: str
           string to predict

       Returns
       -------
       result: list
       """

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


