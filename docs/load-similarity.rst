
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 13.8 s, sys: 1.58 s, total: 15.4 s
    Wall time: 19.6 s


.. code:: python

    news = 'najib razak dan mahathir mengalami masalah air di kemamam terengganu'
    second_news = 'ikat penyedia perkhidmatan jalur lebar Telekom Malaysia (TM) perlu mencari jalan penyelesaian bagi meningkatkan akses capaian Internet ke seluruh negara, kata Menteri Komunikasi dan Multimedia, Gobind Singh Deo. Beliau berkata menjadi dasar kerajaan untuk membekalkan akses Internet jalur lebar kepada semua dan memberi penekanan kepada kualiti perkhidmatan yang terbaik. "Dasar kerajaan untuk bekalkan akses kepada semua bukan sekadar pembekalan sahaja tetapi beri penekanan kepada kualiti perkhidmatan yang baik dan dapat bersaing dengan negara lain pada tahap antarabangsa," kata Gobind Singh menerusi catatan di laman rasmi Twitter beliau, malam tadi. Beliau berkata demikian sebagai respons terhadap aduan beberapa pengguna Twitter berhubung akses Internet yang masih tidak stabil serta harga yang tidak berpatutan di beberapa lokasi di seluruh negara.'

Topics provided by malaya
-------------------------

Topics
^^^^^^

.. code:: python

    malaya.topic.topic['sosial']




.. parsed-literal::

    ['sosial', 'kehidupan', 'taraf hidup', 'sosiologi', 'keusahawan', 'masyarakat']



Influencer
^^^^^^^^^^

.. code:: python

    malaya.topic.influencer['mahathir']




.. parsed-literal::

    ['tun mahathir',
     'madey',
     'dr mahathir',
     'tun m',
     'mahathir',
     'madir',
     'dr m',
     'mahathir muhamad']



location
^^^^^^^^

.. code:: python

    malaya.topic.location[0]




.. parsed-literal::

    {'negeri': 'JOHOR', 'parlimen': 'SEGAMAT', 'dun': 'BULOH KASAP'}



wakil rakyat
^^^^^^^^^^^^

.. code:: python

    malaya.topic.calon[0]




.. parsed-literal::

    {'KodN': 1,
     'KodParlimen': 1,
     'KodKawasan': 1,
     'JenisKawasan': 'PARLIMEN',
     'susunan': 2,
     'NamaCalon': 'DATUK ZAHIDI BIN ZAINUL ABIDIN',
     'parti': 'BN'}



Train fuzzy text similarity
---------------------------

I want to train topics related when given a string. You can give any
corpus, the format is,

.. code:: python

   {'left':['right1','right2']}

.. code:: python

    fuzzy = malaya.similarity.fuzzy(malaya.topic.topic)

.. code:: python

    fuzzy.get_similarity(news,fuzzy_ratio = 60)




.. parsed-literal::

    ['najib razak', 'masalah air', 'mahathir', 'tan sri mokhzani mahathir']



.. code:: python

    fuzzy.get_similarity(second_news,fuzzy_ratio = 90)




.. parsed-literal::

    ['telekom malaysia',
     'kerajaan',
     'internet',
     'twitter',
     'teknologi',
     'politik',
     'pendidikan',
     'sosial media']



Train bag-of-word text similarity
---------------------------------

I want to train topics related when given a string. You can give any
corpus, the format is,

.. code:: python

   {'left':['right1','right2']}

bag-of-word text similarity fitted by using character wised n-gram.

``vectorizer`` supported ``['tfidf','count','skip-gram']``.

.. code:: python

    tfidf = malaya.similarity.bow(malaya.topic.topic,vectorizer = 'tfidf')

.. code:: python

    tfidf.get_similarity(second_news)




.. parsed-literal::

    ['kkmm', 'universiti islam antarabangsa', 'perkhidmatan awam', 'twitter']



.. code:: python

    count = malaya.similarity.bow(malaya.topic.topic,vectorizer = 'count')

.. code:: python

    count.get_similarity(second_news)




.. parsed-literal::

    ['mic',
     'kerajaan',
     'majlis pakatan harapan',
     'jabatan bubar',
     '1malaysia',
     'kemelangan penumpang cedera',
     'pendidikan',
     'malaysian chinese association',
     'ppbm',
     'menyiasat skandal',
     'tentera malaysia',
     'pakatan harapan',
     'parti islam semalaysia',
     'jabatan agama islam wilayah persekutuan',
     'bursa malaysia',
     'rela',
     'undi pos',
     'twitter',
     'parti pribumi bersatu malaysia',
     'perkhidmatan awam',
     'hutang negara',
     'politik',
     'timbalan perdana menteri',
     'kkmm',
     'perdana menteri',
     'ptptn',
     'menteri pertahanan',
     'universiti islam antarabangsa',
     'gaji menteri',
     'teknologi',
     'bebas tahanan',
     'infrastruktur',
     'menteri kewangan']



.. code:: python

    skip = malaya.similarity.bow(malaya.topic.topic,vectorizer = 'skip-gram')

.. code:: python

    skip.get_similarity(second_news)




.. parsed-literal::

    []



Train siamese network text similarity
-------------------------------------

All parameters supported,

.. code:: python

       """
       Train a deep siamese network for text similarity

       Parameters
       ----------
       dictionary: dict
           format {'left':['right']}
       epoch: int, (default=5)
           iteration numbers
       batch_size: int, (default=32)
           batch size for every feed, batch size must <= size of corpus
       embedding_size: int, (default=256)
           vector size representation for a word
       output_size: int, (default=100)
           encoder output size, bigger means more vector definition
       maxlen: int, (default=100)
           max length of a string to be train
       ngram: tuple, (default=(1,4))
           n-grams size to train a corpus
       num_layers: int, (default=100)
           number of bidirectional rnn layers

       Returns
       -------
       _DEEP_SIAMESE_SIMILARITY: malaya.similarity._DEEP_SIAMESE_SIMILARITY class
       """

.. code:: python

    siamese = malaya.similarity.deep_siamese(malaya.topic.topic,epoch=3)
    siamese.get_similarity(news)


.. parsed-literal::

    minibatch loop: 100%|██████████| 137/137 [02:04<00:00,  1.35it/s, accuracy=0.5, cost=0.128]
    minibatch loop: 100%|██████████| 137/137 [01:58<00:00,  1.45it/s, accuracy=0.75, cost=0.11]
    minibatch loop: 100%|██████████| 137/137 [02:02<00:00,  1.38it/s, accuracy=1, cost=0.0455]




.. parsed-literal::

    ['isytihar darurat',
     'mic',
     'dewan rakyat',
     'agama',
     'majlis pakatan harapan',
     'cambridge analytica',
     'tabung haji',
     'ganja',
     'universiti',
     'isu kerugian',
     'isu dadah',
     'tun daim zainuddin',
     'menteri dalam negeri',
     'perkasa',
     'pengedar dadah',
     'anwar ibrahim',
     'sst',
     'saham dan komoditi',
     'amanah',
     'astro awani',
     'recep tayyip erdogan',
     'kementerian dalam negeri',
     'pakatan harapan',
     'parti islam semalaysia',
     'jabatan agama islam wilayah persekutuan',
     'undi pos',
     'pusat daerah mangundi',
     'programming language',
     'wan azizah',
     'rumah mampu milik',
     'kkmm',
     'menteri pertahanan',
     'universiti islam antarabangsa',
     'datuk seri abdul hadi awang',
     'donald trump',
     'gaji menteri',
     'bebas tahanan',
     'ask me a question',
     'ahli dewan undangan negeri']



You can speed up your training iteration by using
`malaya-gpu <https://pypi.org/project/malaya-gpu/>`__

After you trained, actually you save that model by using method
``save_model``. Just provide directory you want to save.

.. code:: python

    siamese.save_model('siamese')

.. code:: python

    !ls siamese


.. parsed-literal::

    checkpoint                     model.ckpt.meta
    model.ckpt.data-00000-of-00001 model.json
    model.ckpt.index


You can load your model but need to use interface provided by malaya,
``malaya.similarity.load_siamese``

.. code:: python

    siamese = malaya.similarity.load_siamese('siamese')


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from siamese/model.ckpt


.. code:: python

    siamese.get_similarity(news)




.. parsed-literal::

    ['isytihar darurat',
     'mic',
     'majlis pakatan harapan',
     'cambridge analytica',
     'ask me a question',
     'tabung harapan',
     'tabung haji',
     'ganja',
     'universiti',
     'isu kerugian',
     'isu dadah',
     'tun daim zainuddin',
     'menteri dalam negeri',
     'perkasa',
     'pengedar dadah',
     'anwar ibrahim',
     'sst',
     'saham dan komoditi',
     'amanah',
     'astro awani',
     'recep tayyip erdogan',
     'kementerian dalam negeri',
     'parti islam semalaysia',
     'jabatan agama islam wilayah persekutuan',
     'isu ecrl',
     'parti keadilan rakyat',
     'pusat daerah mangundi',
     'programming language',
     'wan azizah',
     'timbalan perdana menteri',
     'kkmm',
     'perdana menteri',
     'masalah air',
     'menteri pertahanan',
     'universiti islam antarabangsa',
     'datuk seri abdul hadi awang',
     'donald trump',
     'gaji menteri',
     'bebas tahanan',
     'datuk seri azmin ali',
     'ahli dewan undangan negeri']



Train skipthought text similarity
---------------------------------

All parameters supported,

.. code:: python

       """
       Train a deep skip-thought network for text similarity

       Parameters
       ----------
       dictionary: dict
           format {'left':['right']}
       epoch: int, (default=5)
           iteration numbers
       batch_size: int, (default=32)
           batch size for every feed, batch size must <= size of corpus
       embedding_size: int, (default=256)
           vector size representation for a word
       maxlen: int, (default=100)
           max length of a string to be train
       ngram: tuple, (default=(1,4))
           n-grams size to train a corpus

       Returns
       -------
       _DEEP_SIMILARITY: malaya.similarity._DEEP_SIMILARITY class
       """

.. code:: python

    skipthought = malaya.similarity.deep_skipthought(malaya.topic.topic,epoch=3)
    skipthought.get_similarity(news)


.. parsed-literal::

    minibatch loop: 100%|██████████| 137/137 [01:35<00:00,  1.83it/s, cost=3.05]
    minibatch loop: 100%|██████████| 137/137 [01:31<00:00,  1.69it/s, cost=0.428]
    minibatch loop: 100%|██████████| 137/137 [01:38<00:00,  1.71it/s, cost=0.164]




.. parsed-literal::

    ['malaysia-indonesia',
     'tunku ismail idris',
     'mikro-ekonomi',
     'tengku razaleigh hamzah',
     'k-pop',
     'kkmm',
     'pusat transformasi bandar',
     'hari raya',
     '#fakenews',
     'makro-ekonomi',
     'lee kuan yew',
     'pilihan raya umum ke-14',
     'undi rosak',
     'datuk seri azmin ali',
     'ahli dewan undangan negeri',
     'recep tayyip erdogan',
     'inisiatif peduli rakyat',
     'nga kor ming']



You can speed up your training iteration by using
`malaya-gpu <https://pypi.org/project/malaya-gpu/>`__

After you trained, actually you save that model by using method
``save_model``. Just provide directory you want to save.

.. code:: python

    skipthought.save_model('skipthought')

.. code:: python

    !ls skipthought


.. parsed-literal::

    checkpoint                     model.ckpt.meta
    model.ckpt.data-00000-of-00001 model.json
    model.ckpt.index


You can load your model but need to use interface provided by malaya,
``malaya.similarity.load_skipthought``

.. code:: python

    skipthought = malaya.similarity.load_skipthought('skipthought')


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from skipthought/model.ckpt


.. code:: python

    skipthought.get_similarity(news)




.. parsed-literal::

    ['malaysia-indonesia',
     'tunku ismail idris',
     'mikro-ekonomi',
     'tengku razaleigh hamzah',
     'k-pop',
     'kkmm',
     'pusat transformasi bandar',
     'hari raya',
     '#fakenews',
     'makro-ekonomi',
     'lee kuan yew',
     'pilihan raya umum ke-14',
     'undi rosak',
     'datuk seri azmin ali',
     'ahli dewan undangan negeri',
     'recep tayyip erdogan',
     'inisiatif peduli rakyat',
     'nga kor ming']



Using fuzzy for location
------------------------

.. code:: python

    malaya.similarity.fuzzy_location('saya suka makan sate di sungai petani')




.. parsed-literal::

    {'negeri': [], 'parlimen': ['sungai petani'], 'dun': []}



Check location from a string
----------------------------

.. code:: python

    malaya.similarity.is_location('sungai petani')




.. parsed-literal::

    True
