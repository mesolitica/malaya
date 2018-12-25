
.. code:: python

    import malaya

.. code:: python

    news = 'najib razak dan mahathir mengalami masalah air di kemamam terengganu'
    second_news = 'ikat penyedia perkhidmatan jalur lebar Telekom Malaysia (TM) perlu mencari jalan penyelesaian bagi meningkatkan akses capaian Internet ke seluruh negara, kata Menteri Komunikasi dan Multimedia, Gobind Singh Deo. Beliau berkata menjadi dasar kerajaan untuk membekalkan akses Internet jalur lebar kepada semua dan memberi penekanan kepada kualiti perkhidmatan yang terbaik. "Dasar kerajaan untuk bekalkan akses kepada semua bukan sekadar pembekalan sahaja tetapi beri penekanan kepada kualiti perkhidmatan yang baik dan dapat bersaing dengan negara lain pada tahap antarabangsa," kata Gobind Singh menerusi catatan di laman rasmi Twitter beliau, malam tadi. Beliau berkata demikian sebagai respons terhadap aduan beberapa pengguna Twitter berhubung akses Internet yang masih tidak stabil serta harga yang tidak berpatutan di beberapa lokasi di seluruh negara.'

Using fuzzy for topics
----------------------

.. code:: python

    malaya.topic_influencer.fuzzy_topic(news)




.. parsed-literal::

    ['najib razak', 'mahathir', 'masalah air']



.. code:: python

    malaya.topic_influencer.fuzzy_topic(second_news)




.. parsed-literal::

    ['politik',
     'kkmm',
     'telekom malaysia',
     'internet',
     'pendidikan',
     'perkhidmatan awam',
     'teknologi',
     'twitter',
     'kerajaan',
     'sosial media']



Using fuzzy for influencers
---------------------------

.. code:: python

    malaya.topic_influencer.fuzzy_influencer(news)




.. parsed-literal::

    ['najib razak', 'mahathir']



.. code:: python

    malaya.topic_influencer.fuzzy_influencer(second_news)




.. parsed-literal::

    ['gobind singh deo']



Using fuzzy for location
------------------------

.. code:: python

    malaya.topic_influencer.fuzzy_location('saya suka makan sate di sungai petani')




.. parsed-literal::

    {'negeri': [], 'parlimen': ['sungai petani'], 'dun': []}



Check location from a string
----------------------------

.. code:: python

    malaya.topic_influencer.is_location('sungai petani')




.. parsed-literal::

    True



Train TF-IDF for topics analysis
--------------------------------

.. code:: python

    topics_similarity = malaya.topic_influencer.fast_topic()

.. code:: python

    topics_similarity.get_similarity(news)




.. parsed-literal::

    ['najib razak',
     'mahathir',
     'tan sri mokhzani mahathir',
     'tengku razaleigh hamzah']



Train TF-IDF for influencers analysis
-------------------------------------

.. code:: python

    influencers_similarity = malaya.topic_influencer.fast_influencer()

.. code:: python

    influencers_similarity.get_similarity(news)




.. parsed-literal::

    ['najib razak',
     'mahathir',
     'tan sri mokhzani mahathir',
     'tengku razaleigh hamzah']



.. code:: python

    influencers_similarity.get_similarity(second_news)




.. parsed-literal::

    ['mic',
     'jabatan perancangan bandar dan desa',
     'pakatan harapan',
     'parti pribumi bersatu malaysia',
     'gobind singh deo',
     'ppbm',
     'parti islam semalaysia',
     'perkasa']



Train skip-thought model for topics analysis
--------------------------------------------

.. code:: python

    deep_topic = malaya.topic_influencer.skipthought_topic()


.. parsed-literal::

    minibatch loop: 100%|██████████| 157/157 [01:47<00:00,  1.67it/s, cost=0.447]
    minibatch loop: 100%|██████████| 157/157 [01:45<00:00,  1.71it/s, cost=0.00799]
    minibatch loop: 100%|██████████| 157/157 [01:45<00:00,  1.68it/s, cost=0.00315]
    minibatch loop: 100%|██████████| 157/157 [01:44<00:00,  1.60it/s, cost=0.00197]
    minibatch loop: 100%|██████████| 157/157 [01:44<00:00,  1.70it/s, cost=0.00152]


.. code:: python

    deep_topic.get_similarity(news, anchor = 0.5)




.. parsed-literal::

    ['kkmm',
     'k-pop',
     'mikro-ekonomi',
     'malaysia-indonesia',
     'makro-ekonomi',
     'pilihan raya umum ke-14',
     'programming language',
     '#fakenews',
     'undi rosak']



.. code:: python

    deep_topic.get_similarity(second_news, anchor = 0.5)




.. parsed-literal::

    ['datuk seri abdul hadi awang',
     'kkmm',
     'k-pop',
     'mikro-ekonomi',
     'malaysia-indonesia',
     'makro-ekonomi',
     'pilihan raya umum ke-14',
     'programming language',
     '#fakenews',
     'new straits times',
     'undi rosak']



Train skip-thought model for influencers analysis
-------------------------------------------------

.. code:: python

    deep_influencer = malaya.topic_influencer.skipthought_influencer()


.. parsed-literal::

    minibatch loop: 100%|██████████| 20/20 [00:13<00:00,  1.70it/s, cost=3.46]
    minibatch loop: 100%|██████████| 20/20 [00:13<00:00,  1.33it/s, cost=1.08]
    minibatch loop: 100%|██████████| 20/20 [00:13<00:00,  1.66it/s, cost=0.547]
    minibatch loop: 100%|██████████| 20/20 [00:12<00:00,  1.74it/s, cost=0.275]
    minibatch loop: 100%|██████████| 20/20 [00:12<00:00,  1.59it/s, cost=0.253]
    minibatch loop: 100%|██████████| 20/20 [00:12<00:00,  1.69it/s, cost=0.281]
    minibatch loop: 100%|██████████| 20/20 [00:12<00:00,  1.71it/s, cost=0.209]
    minibatch loop: 100%|██████████| 20/20 [00:12<00:00,  1.66it/s, cost=0.259]
    minibatch loop: 100%|██████████| 20/20 [00:12<00:00,  1.67it/s, cost=0.232]
    minibatch loop: 100%|██████████| 20/20 [00:12<00:00,  1.62it/s, cost=0.219]


.. code:: python

    deep_influencer.get_similarity(news, anchor = 0.5)




.. parsed-literal::

    ['najib razak', 'mahathir']



.. code:: python

    deep_influencer.get_similarity(second_news, anchor = 0.5)




.. parsed-literal::

    ['gobind singh deo']



Train siamese network for topics analysis
-----------------------------------------

.. code:: python

    deep_topic = malaya.topic_influencer.siamese_topic()
    print(deep_topic.get_similarity(news, anchor = 0.5))
    print(deep_topic.get_similarity(second_news, anchor = 0.5))


.. parsed-literal::

    minibatch loop: 100%|██████████| 157/157 [01:50<00:00,  1.67it/s, accuracy=1, cost=0.114]
    minibatch loop: 100%|██████████| 157/157 [01:49<00:00,  1.69it/s, accuracy=1, cost=0.0739]
    minibatch loop: 100%|██████████| 157/157 [01:49<00:00,  1.66it/s, accuracy=1, cost=0.0686]
    minibatch loop: 100%|██████████| 157/157 [01:50<00:00,  1.68it/s, accuracy=1, cost=0.0279]
    minibatch loop: 100%|██████████| 157/157 [01:49<00:00,  1.70it/s, accuracy=1, cost=0.0193]


.. parsed-literal::

    ['kesihatan', 'politik', 'wan azizah', 'kaum cina', 'tiga penjuru', 'pusat transformasi bandar', 'bumiputra', 'jabatan perancangan bandar dan desa', 'pusat daerah mangundi', 'menteri pertahanan', 'kewangan', 'gaza', 'kaum melayu', 'programming language', 'lgbt', 'infrastruktur', 'sinar harian', 'singapura', 'real madrid cf', 'anwar ibrahim']
    ['politik', 'kkmm', 'bumiputra', 'malaysia-indonesia', 'menteri pertahanan', 'motogp', 'programming language', 'twitter', 'lgbt', 'gaji menteri', 'singapura']


.. code:: python

    print(deep_topic.get_similarity(news, anchor = 0.7))
    print(deep_topic.get_similarity(second_news, anchor = 0.7))


.. parsed-literal::

    []
    []


Train siamese network for influencers analysis
----------------------------------------------

.. code:: python

    deep_influencer = malaya.topic_influencer.siamese_influencer()


.. parsed-literal::

    minibatch loop: 100%|██████████| 20/20 [00:14<00:00,  1.46it/s, accuracy=0.583, cost=0.129]
    minibatch loop: 100%|██████████| 20/20 [00:13<00:00,  1.48it/s, accuracy=0.542, cost=0.124]
    minibatch loop: 100%|██████████| 20/20 [00:13<00:00,  1.49it/s, accuracy=0.542, cost=0.121]
    minibatch loop: 100%|██████████| 20/20 [00:14<00:00,  1.49it/s, accuracy=0.833, cost=0.0885]
    minibatch loop: 100%|██████████| 20/20 [00:14<00:00,  1.47it/s, accuracy=0.875, cost=0.0637]


.. code:: python

    deep_influencer.get_similarity(news, anchor = 0.5)




.. parsed-literal::

    ['najib razak', 'mahathir']



.. code:: python

    deep_influencer.get_similarity(second_news, anchor = 0.5)




.. parsed-literal::

    ['gobind singh deo']
