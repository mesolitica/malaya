
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.5 s, sys: 1.77 s, total: 14.3 s
    Wall time: 19.5 s


Deep Siamese network
--------------------

Purpose of deep siamese network to study semantic similarity between 2
strings, near to 1.0 means more similar. Deep Siamese leverage the power
of word-vector, and we also implemented BERT to study semantic
similarity and BERT leverage the power of attention!

List deep siamese models
------------------------

.. code:: python

    malaya.similarity.available_deep_siamese()




.. parsed-literal::

    ['self-attention', 'bahdanau', 'dilated-cnn']



-  ``'self-attention'`` - Fast-text architecture, embedded and logits
   layers only with self attention.
-  ``'bahdanau'`` - LSTM with bahdanau attention architecture.
-  ``'dilated-cnn'`` - Pyramid Dilated CNN architecture.

Load deep siamese models
------------------------

.. code:: python

    string1 = 'Pemuda mogok lapar desak kerajaan prihatin isu iklim'
    string2 = 'Perbincangan isu pembalakan perlu babit kerajaan negeri'
    string3 = 'kerajaan perlu kisah isu iklim, pemuda mogok lapar'
    string4 = 'Kerajaan dicadang tubuh jawatankuasa khas tangani isu alam sekitar'

Load bahdanau model
^^^^^^^^^^^^^^^^^^^

.. code:: python

    model = malaya.similarity.deep_siamese('bahdanau')

Calculate similarity between 2 strings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``predict`` need to give 2 strings, left and right string

.. code:: python

    model.predict(string1, string2)




.. parsed-literal::

    0.4267301



.. code:: python

    model.predict(string1, string3)




.. parsed-literal::

    0.28711933



Calculate similarity more than 2 strings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``predict_batch`` need to give 2 lists of strings, left and right
strings

.. code:: python

    model.predict_batch([string1, string2], [string3, string4])




.. parsed-literal::

    array([0.39504164, 0.33375728], dtype=float32)



Load self-attention model
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    model = malaya.similarity.deep_siamese('self-attention')

.. code:: python

    model.predict_batch([string1, string2], [string3, string4])




.. parsed-literal::

    array([0.08130383, 0.09907728], dtype=float32)



Load dilated-cnn model
^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    model = malaya.similarity.deep_siamese('dilated-cnn')

.. code:: python

    model.predict_batch([string1, string2], [string3, string4])




.. parsed-literal::

    array([0.1886251 , 0.00937402], dtype=float32)



Calculate similarity using doc2vec
----------------------------------

We need to load word vector provided by Malaya.

Important parameters, 1. ``aggregation``, aggregation function to
accumulate word vectors. Default is ``mean``.

::

   * ``'mean'`` - mean.
   * ``'min'`` - min.
   * ``'max'`` - max.
   * ``'sum'`` - sum.
   * ``'sqrt'`` - square root.

2. ``similarity`` distance function to calculate similarity. Default is
   ``cosine``.

   -  ``'cosine'`` - cosine similarity.
   -  ``'euclidean'`` - euclidean similarity.
   -  ``'manhattan'`` - manhattan similarity.

Using word2vec
^^^^^^^^^^^^^^

I will use ``load_news``, word2vec from wikipedia took a very long time.
wikipedia much more accurate.

.. code:: python

    embedded_news = malaya.word2vec.load_news(64)
    w2v_wiki = malaya.word2vec.word2vec(embedded_news['nce_weights'],
                                        embedded_news['dictionary'])

.. code:: python

    malaya.similarity.doc2vec(w2v_wiki, string1, string2)




.. parsed-literal::

    0.9181415736675262



.. code:: python

    malaya.similarity.doc2vec(w2v_wiki, string1, string4)




.. parsed-literal::

    0.9550771713256836



.. code:: python

    malaya.similarity.doc2vec(w2v_wiki, string1, string4, similarity = 'euclidean')




.. parsed-literal::

    0.4642694249990522



Different similarity function different percentage.

**So you can try use fast-text and elmo to do the similarity study.**

Calculate similarity using summarizer
-------------------------------------

We can use extractive summarization model
``malaya.summarize.deep_extractive()`` to get strings embedded and
calculate similarity between the vectors.

.. code:: python

    deep_summary = malaya.summarize.deep_extractive(model = 'skip-thought')

.. code:: python

    malaya.similarity.summarizer(deep_summary, string1, string3)




.. parsed-literal::

    0.8722701370716095



BERT model
----------

BERT is the best similarity model in term of accuracy, you can check
similarity accuracy here,
https://malaya.readthedocs.io/en/latest/Accuracy.html#similarity. But
warning, the model size is 700MB! Make sure you have enough resources to
use BERT, and installed bert-tensorflow first,

.. code:: python

    model = malaya.similarity.bert()

.. code:: python

    model.predict(string1, string3)




.. parsed-literal::

    0.97767043



.. code:: python

    model.predict_batch([string1, string2], [string3, string4])




.. parsed-literal::

    array([0.9253927, 0.0317315], dtype=float32)



**BERT is the best!**

Topics similarity
-----------------

If you are interested in multiple topics searching inside a string when
giving set of topics to supervised, Malaya provided some interface and
topics related to political landscape in Malaysia

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

    ['tan sri mokhzani mahathir', 'masalah air', 'mahathir', 'najib razak']



.. code:: python

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

.. code:: python

    tfidf = malaya.similarity.bow(malaya.topic.topic,vectorizer = 'tfidf')

.. code:: python

    tfidf.get_similarity(second_news)




.. parsed-literal::

    ['perkhidmatan awam', 'kkmm', 'universiti islam antarabangsa', 'twitter']



.. code:: python

    count = malaya.similarity.bow(malaya.topic.topic,vectorizer = 'count')

.. code:: python

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

.. code:: python

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
