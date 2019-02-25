
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 10.7 s, sys: 906 ms, total: 11.6 s
    Wall time: 12 s


.. code:: ipython3

    news = 'najib razak dan mahathir mengalami masalah air di kemamam terengganu'
    second_news = 'ikat penyedia perkhidmatan jalur lebar Telekom Malaysia (TM) perlu mencari jalan penyelesaian bagi meningkatkan akses capaian Internet ke seluruh negara, kata Menteri Komunikasi dan Multimedia, Gobind Singh Deo. Beliau berkata menjadi dasar kerajaan untuk membekalkan akses Internet jalur lebar kepada semua dan memberi penekanan kepada kualiti perkhidmatan yang terbaik. "Dasar kerajaan untuk bekalkan akses kepada semua bukan sekadar pembekalan sahaja tetapi beri penekanan kepada kualiti perkhidmatan yang baik dan dapat bersaing dengan negara lain pada tahap antarabangsa," kata Gobind Singh menerusi catatan di laman rasmi Twitter beliau, malam tadi. Beliau berkata demikian sebagai respons terhadap aduan beberapa pengguna Twitter berhubung akses Internet yang masih tidak stabil serta harga yang tidak berpatutan di beberapa lokasi di seluruh negara.'

Topics provided by malaya
-------------------------

Topics
^^^^^^

.. code:: ipython3

    malaya.topic.topic['sosial']




.. parsed-literal::

    ['sosial', 'kehidupan', 'taraf hidup', 'sosiologi', 'keusahawan', 'masyarakat']



Influencer
^^^^^^^^^^

.. code:: ipython3

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

.. code:: ipython3

    malaya.topic.location[0]




.. parsed-literal::

    {'negeri': 'JOHOR', 'parlimen': 'SEGAMAT', 'dun': 'BULOH KASAP'}



wakil rakyat
^^^^^^^^^^^^

.. code:: ipython3

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

.. code:: ipython3

    fuzzy = malaya.similarity.fuzzy(malaya.topic.topic)

.. code:: ipython3

    fuzzy.get_similarity(news,fuzzy_ratio = 60)




.. parsed-literal::

    ['tan sri mokhzani mahathir', 'masalah air', 'mahathir', 'najib razak']



.. code:: ipython3

    fuzzy.get_similarity(second_news,fuzzy_ratio = 90)




.. parsed-literal::

    ['pendidikan',
     'sosial media',
     'politik',
     'kerajaan',
     'telekom malaysia',
     'twitter',
     'teknologi',
     'internet']



Train bag-of-word text similarity
---------------------------------

I want to train topics related when given a string. You can give any
corpus, the format is,

.. code:: python

   {'left':['right1','right2']}

bag-of-word text similarity fitted by using character wised n-gram.

``vectorizer`` supported ``['tfidf','count','skip-gram']``.

.. code:: ipython3

    tfidf = malaya.similarity.bow(malaya.topic.topic,vectorizer = 'tfidf')

.. code:: ipython3

    tfidf.get_similarity(second_news)




.. parsed-literal::

    ['perkhidmatan awam', 'kkmm', 'universiti islam antarabangsa', 'twitter']



.. code:: ipython3

    count = malaya.similarity.bow(malaya.topic.topic,vectorizer = 'count')

.. code:: ipython3

    count.get_similarity(second_news)




.. parsed-literal::

    ['timbalan perdana menteri',
     'parti islam semalaysia',
     'pendidikan',
     '1malaysia',
     'gaji menteri',
     'mic',
     'bebas tahanan',
     'twitter',
     'infrastruktur',
     'suruhanjaya pilihan raya malaysia',
     'perkasa',
     'pakatan harapan',
     'kerajaan',
     'datuk seri ti lian ker',
     'tentera malaysia',
     'gerakan',
     'universiti islam antarabangsa',
     'ptptn',
     'rela',
     'ahli dewan undangan negeri',
     'teknologi',
     'politik',
     'telekom malaysia',
     'kkmm',
     'kementerian dalam negeri',
     'perkhidmatan awam',
     'bursa malaysia',
     'parti pribumi bersatu malaysia',
     'ppbm',
     'hutang negara',
     'menyiasat skandal',
     'majlis pakatan harapan',
     'perdana menteri',
     'menteri pertahanan']



.. code:: ipython3

    skip = malaya.similarity.bow(malaya.topic.topic,vectorizer = 'skip-gram')

.. code:: ipython3

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

.. code:: ipython3

    siamese = malaya.similarity.deep_siamese(malaya.topic.topic,epoch=3)
    siamese.get_similarity(news)


.. parsed-literal::

    minibatch loop: 100%|██████████| 137/137 [01:42<00:00,  1.53it/s, accuracy=0.5, cost=0.129]  
    minibatch loop: 100%|██████████| 137/137 [01:40<00:00,  1.52it/s, accuracy=0.833, cost=0.108]
    minibatch loop: 100%|██████████| 137/137 [01:40<00:00,  1.54it/s, accuracy=1, cost=0.0514]    




.. parsed-literal::

    ['parti islam semalaysia',
     'pusat transformasi bandar',
     'malaysia baru',
     'mic',
     'bridge city park',
     'suruhanjaya pilihan raya malaysia',
     'kotak undi',
     'lgbt',
     'tentera malaysia',
     'dewan rakyat',
     'isu kemiskinan',
     'undi rosak',
     'produk berbahaya',
     'politik',
     'telekom malaysia',
     'bank negara',
     'kertas undi',
     'malay mail',
     'gaji minimum',
     'donald trump',
     'najib razak',
     'bank malaysia',
     'humanoid',
     'perkhidmatan awam',
     'rosmah mansur',
     'isu dadah',
     'stock market malaysia',
     'bursa malaysia',
     'pusat daerah mangundi',
     'undi pos',
     'universiti teknologi malaysia',
     'hutang negara',
     'makro-ekonomi',
     'rtm',
     'pengangkutan awam']



You can speed up your training iteration by using
`malaya-gpu <https://pypi.org/project/malaya-gpu/>`__

After you trained, actually you save that model by using method
``save_model``. Just provide directory you want to save.

.. code:: ipython3

    siamese.save_model('siamese')

.. code:: ipython3

    !ls siamese


.. parsed-literal::

    checkpoint                     model.ckpt.meta
    model.ckpt.data-00000-of-00001 model.json
    model.ckpt.index


You can load your model but need to use interface provided by malaya,
``malaya.similarity.load_siamese``

.. code:: ipython3

    siamese = malaya.similarity.load_siamese('siamese')


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from siamese/model.ckpt


.. code:: ipython3

    siamese.get_similarity(news)




.. parsed-literal::

    ['pilihan raya umum ke-14',
     'parti islam semalaysia',
     'malaysia baru',
     'pengedar dadah',
     'suruhanjaya pilihan raya malaysia',
     'kotak undi',
     'lgbt',
     'makanan',
     'tentera malaysia',
     'gerakan',
     'isu kemiskinan',
     'undi rosak',
     'produk berbahaya',
     'bloomberg',
     'telekom malaysia',
     'bank negara',
     'kertas undi',
     'malay mail',
     'gaji minimum',
     '1mdb',
     'najib razak',
     'bank malaysia',
     'humanoid',
     'perkhidmatan awam',
     'rosmah mansur',
     'isu dadah',
     'stock market malaysia',
     'bursa malaysia',
     'undi pos',
     'universiti teknologi malaysia',
     'hutang negara',
     'makro-ekonomi',
     'rtm',
     'pengangkutan awam']



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

.. code:: ipython3

    skipthought = malaya.similarity.deep_skipthought(malaya.topic.topic,epoch=3)
    skipthought.get_similarity(news)


.. parsed-literal::

    minibatch loop: 100%|██████████| 137/137 [01:20<00:00,  1.93it/s, cost=3.4] 
    minibatch loop: 100%|██████████| 137/137 [01:17<00:00,  1.91it/s, cost=0.793]
    minibatch loop: 100%|██████████| 137/137 [01:17<00:00,  1.90it/s, cost=0.342] 




.. parsed-literal::

    ['pilihan raya umum ke-14',
     'pusat transformasi bandar',
     'hari raya',
     'nga kor ming',
     'programming language',
     '#fakenews',
     'mikro-ekonomi',
     'datuk seri azmin ali',
     'recep tayyip erdogan',
     'k-pop',
     'malaysia-indonesia',
     'tengku razaleigh hamzah',
     'anthony loke siew fook',
     'lee kuan yew',
     'rais yatim',
     'undi rosak',
     'kkmm',
     'inisiatif peduli rakyat',
     'tunku ismail idris',
     'pusat daerah mangundi',
     'makro-ekonomi',
     'new straits times']



You can speed up your training iteration by using
`malaya-gpu <https://pypi.org/project/malaya-gpu/>`__

After you trained, actually you save that model by using method
``save_model``. Just provide directory you want to save.

.. code:: ipython3

    skipthought.save_model('skipthought')

.. code:: ipython3

    !ls skipthought


.. parsed-literal::

    checkpoint                     model.ckpt.meta
    model.ckpt.data-00000-of-00001 model.json
    model.ckpt.index


You can load your model but need to use interface provided by malaya,
``malaya.similarity.load_skipthought``

.. code:: ipython3

    skipthought = malaya.similarity.load_skipthought('skipthought')


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from skipthought/model.ckpt


.. code:: ipython3

    skipthought.get_similarity(news)




.. parsed-literal::

    ['pilihan raya umum ke-14',
     'pusat transformasi bandar',
     'hari raya',
     'nga kor ming',
     'programming language',
     '#fakenews',
     'mikro-ekonomi',
     'datuk seri azmin ali',
     'recep tayyip erdogan',
     'k-pop',
     'malaysia-indonesia',
     'tengku razaleigh hamzah',
     'anthony loke siew fook',
     'lee kuan yew',
     'rais yatim',
     'undi rosak',
     'kkmm',
     'inisiatif peduli rakyat',
     'tunku ismail idris',
     'pusat daerah mangundi',
     'makro-ekonomi',
     'new straits times']



Using fuzzy for location
------------------------

.. code:: ipython3

    malaya.similarity.fuzzy_location('saya suka makan sate di sungai petani')




.. parsed-literal::

    {'negeri': [], 'parlimen': ['sungai petani'], 'dun': []}



Check location from a string
----------------------------

.. code:: ipython3

    malaya.similarity.is_location('sungai petani')




.. parsed-literal::

    True


