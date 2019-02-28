
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 11.4 s, sys: 1.89 s, total: 13.3 s
    Wall time: 18.4 s


Text augmentation
-----------------

Let say you have a very limited labelled corpus, and you want to add
more, but labelling is very costly.

So, text augmentation! You can use word2vec to replace words with
similar semantics!

.. code:: ipython3

    string = 'saya suka makan ayam dan ikan'

.. code:: ipython3

    embedded_wiki = malaya.word2vec.load_wiki()
    word_vector_wiki = malaya.word2vec.word2vec(embedded_wiki['nce_weights'], 
                                                embedded_wiki['dictionary'])

.. code:: ipython3

    augmented = malaya.generator.w2v_augmentation(string, 
                                      word_vector_wiki,
                                      soft=True,
                                      augment_counts=3)
    augmented




.. parsed-literal::

    ['saya suka makan ayam ataupun daging',
     'saya suka makan ayam serta ikan',
     'saya suka makan ayam serta udang']



Let we compare word mover distance with original.

.. code:: ipython3

    malaya.word_mover.distance(string.split(), augmented[0].split(), word_vector_wiki)




.. parsed-literal::

    1.1325703463561534



.. code:: ipython3

    malaya.word_mover.distance(string.split(), augmented[1].split(), word_vector_wiki)




.. parsed-literal::

    0.5428173272147179



.. code:: ipython3

    malaya.word_mover.distance(string.split(), augmented[2].split(), word_vector_wiki)




.. parsed-literal::

    1.0979998614006043



They are pretty good in term of sentence similarity! **Distance that
higher than 2 ratios are assumed bad**.

.. code:: ipython3

    augmented = malaya.generator.w2v_augmentation('kerajaan sebenarnya sangat sayangkan rakyatnya', 
                                      word_vector_wiki,
                                      soft=True,
                                      augment_counts=5)
    augmented




.. parsed-literal::

    ['kerajaan sebenarnya amat sayangkan rakyatnya',
     'kerajaan sebenarnya agak sayangkan warganya',
     'kerajaan sebenarnya semakin sayangkan rakyatnya',
     'kerajaan sebenarnya sangat sayangkan penduduknya',
     'kerajaan sebenarnya agak sayangkan penduduknya']



.. code:: ipython3

    bahdanau_entities = malaya.entity.deep_model('bahdanau')
    bahdanau_pos = malaya.pos.deep_model('bahdanau')

.. code:: ipython3

    string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar sekiranya mengantuk ketika memandu.'

.. code:: ipython3

    result_entities = bahdanau_entities.predict(string)
    result_pos = bahdanau_pos.predict(string)

Generate ngram sentences
------------------------

.. code:: ipython3

    malaya.generator.sentence_ngram(string, ngram = (3, 5))




.. parsed-literal::

    ['Jalan Raya (JKJR) itu,',
     'Pengangkutan Anthony Loke Siew',
     'mengantuk ketika memandu.',
     'KUALA LUMPUR: Sempena sambutan Aidilfitri',
     'masing-masing. Dalam video pendek terbitan',
     'terbitan Jabatan Keselamatan Jalan Raya',
     'Anthony Loke Siew',
     'Jalan Raya (JKJR)',
     'Mohamad dan Menteri Pengangkutan',
     'ramai yang mahu pulang ke',
     'KUALA LUMPUR: Sempena',
     'tidur sebentar sekiranya mengantuk',
     'pesanan khas kepada',
     'Mahathir menasihati mereka supaya',
     'Raya (JKJR) itu, Dr',
     'KUALA LUMPUR: Sempena sambutan',
     'Sempena sambutan Aidilfitri minggu depan,',
     'LUMPUR: Sempena sambutan Aidilfitri minggu',
     'Loke Siew Fook menitipkan pesanan',
     'orang ramai yang mahu pulang',
     'pulang ke kampung',
     'berehat dan tidur',
     'mereka supaya berhenti berehat dan',
     'Mohamad dan Menteri',
     'Raya (JKJR) itu,',
     'Fook menitipkan pesanan khas',
     'kampung halaman masing-masing. Dalam video',
     'masing-masing. Dalam video',
     'pesanan khas kepada orang ramai',
     'halaman masing-masing. Dalam video',
     'terbitan Jabatan Keselamatan Jalan',
     'ke kampung halaman masing-masing.',
     'Jabatan Keselamatan Jalan',
     'halaman masing-masing. Dalam video pendek',
     'Mahathir menasihati mereka supaya berhenti',
     'Dr Mahathir Mohamad dan',
     'Dr Mahathir menasihati mereka',
     'pesanan khas kepada orang',
     'orang ramai yang',
     'yang mahu pulang ke kampung',
     'dan tidur sebentar sekiranya mengantuk',
     'video pendek terbitan Jabatan Keselamatan',
     'mereka supaya berhenti',
     'Dalam video pendek',
     'Sempena sambutan Aidilfitri',
     'video pendek terbitan',
     'Jabatan Keselamatan Jalan Raya (JKJR)',
     'Mohamad dan Menteri Pengangkutan Anthony',
     'Mahathir Mohamad dan Menteri',
     'mahu pulang ke kampung halaman',
     '(JKJR) itu, Dr Mahathir',
     'Dalam video pendek terbitan Jabatan',
     'berhenti berehat dan',
     'khas kepada orang',
     'menitipkan pesanan khas',
     'khas kepada orang ramai',
     'pendek terbitan Jabatan Keselamatan',
     'kepada orang ramai',
     'Anthony Loke Siew Fook menitipkan',
     'Keselamatan Jalan Raya (JKJR) itu,',
     'Dr Mahathir menasihati mereka supaya',
     'tidur sebentar sekiranya mengantuk ketika',
     'Mahathir menasihati mereka',
     'berhenti berehat dan tidur',
     'Menteri Pengangkutan Anthony Loke Siew',
     'Menteri Tun Dr Mahathir Mohamad',
     'sebentar sekiranya mengantuk',
     'kampung halaman masing-masing.',
     'orang ramai yang mahu',
     'berehat dan tidur sebentar',
     '(JKJR) itu, Dr Mahathir menasihati',
     'Loke Siew Fook',
     'Dr Mahathir Mohamad dan Menteri',
     'pendek terbitan Jabatan Keselamatan Jalan',
     'ke kampung halaman masing-masing. Dalam',
     'Aidilfitri minggu depan, Perdana',
     'sekiranya mengantuk ketika',
     'khas kepada orang ramai yang',
     'Menteri Tun Dr',
     'Sempena sambutan Aidilfitri minggu',
     'menasihati mereka supaya',
     'menitipkan pesanan khas kepada',
     'dan Menteri Pengangkutan Anthony',
     'mahu pulang ke',
     'kepada orang ramai yang mahu',
     'Siew Fook menitipkan pesanan',
     'itu, Dr Mahathir menasihati',
     'dan tidur sebentar',
     'menasihati mereka supaya berhenti berehat',
     'Fook menitipkan pesanan',
     'mereka supaya berhenti berehat',
     'ke kampung halaman',
     'menitipkan pesanan khas kepada orang',
     'menasihati mereka supaya berhenti',
     'Keselamatan Jalan Raya (JKJR)',
     'Keselamatan Jalan Raya',
     '(JKJR) itu, Dr',
     'Siew Fook menitipkan',
     'Anthony Loke Siew Fook',
     'Jabatan Keselamatan Jalan Raya',
     'Perdana Menteri Tun Dr Mahathir',
     'kepada orang ramai yang',
     'Pengangkutan Anthony Loke',
     'supaya berhenti berehat dan',
     'supaya berhenti berehat',
     'ramai yang mahu pulang',
     'halaman masing-masing. Dalam',
     'Aidilfitri minggu depan, Perdana Menteri',
     'pulang ke kampung halaman',
     'supaya berhenti berehat dan tidur',
     'Tun Dr Mahathir Mohamad dan',
     'yang mahu pulang ke',
     'Aidilfitri minggu depan,',
     'itu, Dr Mahathir menasihati mereka',
     'dan Menteri Pengangkutan Anthony Loke',
     'berehat dan tidur sebentar sekiranya',
     'Menteri Tun Dr Mahathir',
     'pendek terbitan Jabatan',
     'Fook menitipkan pesanan khas kepada',
     'masing-masing. Dalam video pendek',
     'depan, Perdana Menteri',
     'minggu depan, Perdana Menteri',
     'dan Menteri Pengangkutan',
     'Dr Mahathir menasihati',
     'LUMPUR: Sempena sambutan Aidilfitri',
     'Menteri Pengangkutan Anthony Loke',
     'kampung halaman masing-masing. Dalam',
     'Dalam video pendek terbitan',
     'Mahathir Mohamad dan',
     'video pendek terbitan Jabatan',
     'minggu depan, Perdana Menteri Tun',
     'minggu depan, Perdana',
     'ramai yang mahu',
     'Siew Fook menitipkan pesanan khas',
     'Jalan Raya (JKJR) itu, Dr',
     'Menteri Pengangkutan Anthony',
     'dan tidur sebentar sekiranya',
     'tidur sebentar sekiranya',
     'yang mahu pulang',
     'Tun Dr Mahathir Mohamad',
     'Tun Dr Mahathir',
     'itu, Dr Mahathir',
     'Dr Mahathir Mohamad',
     'Mahathir Mohamad dan Menteri Pengangkutan',
     'pulang ke kampung halaman masing-masing.',
     'sambutan Aidilfitri minggu',
     'Raya (JKJR) itu, Dr Mahathir',
     'berhenti berehat dan tidur sebentar',
     'terbitan Jabatan Keselamatan',
     'Perdana Menteri Tun Dr',
     'sekiranya mengantuk ketika memandu.',
     'sebentar sekiranya mengantuk ketika',
     'sebentar sekiranya mengantuk ketika memandu.',
     'mahu pulang ke kampung',
     'depan, Perdana Menteri Tun Dr',
     'depan, Perdana Menteri Tun',
     'Pengangkutan Anthony Loke Siew Fook',
     'Perdana Menteri Tun',
     'sambutan Aidilfitri minggu depan,',
     'Loke Siew Fook menitipkan',
     'LUMPUR: Sempena sambutan',
     'sambutan Aidilfitri minggu depan, Perdana']



Generate ngram sentences for selected POS and Entities
------------------------------------------------------

.. code:: ipython3

    generated_grams = malaya.generator.pos_entities_ngram(
        result_pos,
        result_entities,
        ngram = (1, 3),
        accept_pos = ['NOUN', 'PROPN', 'VERB'],
        accept_entities = ['law', 'location', 'organization', 'person', 'time'],
    )
    generated_grams




.. parsed-literal::

    ['Kuala Lumpur Sempena',
     'masing-masing video terbitan',
     'orang',
     'Mahathir Mohamad Menteri',
     'terbitan',
     'tidur',
     'Keselamatan Jalan',
     'Anthony Loke Siew',
     'minggu depan Perdana',
     'halaman masing-masing video',
     'sekiranya mengantuk',
     'Mohamad Menteri',
     'Tun',
     'menitipkan pesanan orang',
     'kampung halaman masing-masing',
     'masing-masing video',
     'Lumpur',
     'Kuala Lumpur',
     'orang pulang',
     'menitipkan',
     'minggu',
     'Jabatan Keselamatan Jalan',
     'berhenti',
     'Fook menitipkan',
     'Loke',
     'Menteri Tun',
     'Raya Jkjr',
     'Keselamatan',
     'Aidilfitri minggu',
     'Mohamad Menteri Pengangkutan',
     'Sempena sambutan Aidilfitri',
     'kampung halaman',
     'Raya Jkjr Dr',
     'Menteri Pengangkutan',
     'Anthony',
     'sambutan',
     'Mohamad',
     'Jalan',
     'halaman',
     'sekiranya',
     'Pengangkutan Anthony',
     'Pengangkutan',
     'Jkjr',
     'pulang',
     'berhenti berehat tidur',
     'berehat',
     'pulang kampung halaman',
     'Loke Siew Fook',
     'Mahathir',
     'Jabatan Keselamatan',
     'Jabatan',
     'berehat tidur',
     'video',
     'Jkjr Dr Mahathir',
     'mengantuk',
     'Menteri Tun Dr',
     'video terbitan',
     'Fook menitipkan pesanan',
     'pesanan',
     'Siew',
     'sekiranya mengantuk memandu',
     'Keselamatan Jalan Raya',
     'Siew Fook menitipkan',
     'minggu depan',
     'pulang kampung',
     'halaman masing-masing',
     'menasihati berhenti',
     'mengantuk memandu',
     'Pengangkutan Anthony Loke',
     'Jalan Raya Jkjr',
     'Aidilfitri minggu depan',
     'sambutan Aidilfitri',
     'depan Perdana Menteri',
     'Lumpur Sempena',
     'Mahathir menasihati',
     'video terbitan Jabatan',
     'Sempena sambutan',
     'Jkjr Dr',
     'Jalan Raya',
     'Loke Siew',
     'tidur sekiranya mengantuk',
     'depan Perdana',
     'memandu',
     'Mahathir Mohamad',
     'Dr Mahathir menasihati',
     'Fook',
     'Menteri',
     'Siew Fook',
     'Dr',
     'orang pulang kampung',
     'Menteri Pengangkutan Anthony',
     'terbitan Jabatan',
     'Aidilfitri',
     'masing-masing',
     'Tun Dr Mahathir',
     'tidur sekiranya',
     'Dr Mahathir Mohamad',
     'sambutan Aidilfitri minggu',
     'Tun Dr',
     'menitipkan pesanan',
     'menasihati',
     'berhenti berehat',
     'terbitan Jabatan Keselamatan',
     'menasihati berhenti berehat',
     'Lumpur Sempena sambutan',
     'Perdana Menteri',
     'Anthony Loke',
     'pesanan orang pulang',
     'Sempena',
     'depan',
     'Mahathir menasihati berhenti',
     'Perdana Menteri Tun',
     'Perdana',
     'Kuala',
     'Dr Mahathir',
     'berehat tidur sekiranya',
     'Raya',
     'pesanan orang',
     'kampung']


