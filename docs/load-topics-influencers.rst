
.. code:: ipython3

    import malaya

.. code:: ipython3

    news = 'najib razak dan mahathir mengalami masalah air di kemamam terengganu'
    second_news = 'ikat penyedia perkhidmatan jalur lebar Telekom Malaysia (TM) perlu mencari jalan penyelesaian bagi meningkatkan akses capaian Internet ke seluruh negara, kata Menteri Komunikasi dan Multimedia, Gobind Singh Deo. Beliau berkata menjadi dasar kerajaan untuk membekalkan akses Internet jalur lebar kepada semua dan memberi penekanan kepada kualiti perkhidmatan yang terbaik. "Dasar kerajaan untuk bekalkan akses kepada semua bukan sekadar pembekalan sahaja tetapi beri penekanan kepada kualiti perkhidmatan yang baik dan dapat bersaing dengan negara lain pada tahap antarabangsa," kata Gobind Singh menerusi catatan di laman rasmi Twitter beliau, malam tadi. Beliau berkata demikian sebagai respons terhadap aduan beberapa pengguna Twitter berhubung akses Internet yang masih tidak stabil serta harga yang tidak berpatutan di beberapa lokasi di seluruh negara.'

Using fuzzy for topics
----------------------

.. code:: ipython3

    malaya.fuzzy_get_topics(news)




.. parsed-literal::

    ['najib razak', 'masalah air', 'mahathir']



.. code:: ipython3

    malaya.fuzzy_get_topics(second_news)




.. parsed-literal::

    ['teknologi',
     'internet',
     'kkmm',
     'perkhidmatan awam',
     'twitter',
     'pendidikan',
     'politik',
     'sosial media',
     'telekom malaysia',
     'kerajaan']



Using fuzzy for influencers
---------------------------

.. code:: ipython3

    malaya.fuzzy_get_influencers(news)




.. parsed-literal::

    ['najib razak', 'mahathir']



.. code:: ipython3

    malaya.fuzzy_get_influencers(second_news)




.. parsed-literal::

    ['gobind singh deo']



Train TF-IDF for topics analysis
--------------------------------

.. code:: ipython3

    topics_similarity = malaya.fast_get_topics()

.. code:: ipython3

    topics_similarity.get_similarity(news)




.. parsed-literal::

    ['tan sri mokhzani mahathir', 'najib razak', 'masalah air', 'mahathir']



Train TF-IDF for influencers analysis
-------------------------------------

.. code:: ipython3

    influencers_similarity = malaya.fast_get_influencers()

.. code:: ipython3

    influencers_similarity.get_similarity(news)




.. parsed-literal::

    ['tan sri mokhzani mahathir', 'najib razak', 'zakir naik', 'mahathir']



.. code:: ipython3

    influencers_similarity.get_similarity(second_news)




.. parsed-literal::

    ['parti pribumi bersatu malaysia',
     'majlis pakatan harapan',
     'jabatan perancangan bandar dan desa',
     'pakatan harapan',
     'gobind singh deo',
     'parti islam semalaysia',
     'ppbm']



Train skip-thought model for topics analysis
--------------------------------------------

.. code:: ipython3

    deep_topic = malaya.deep_get_topics()


.. parsed-literal::

    minibatch loop: 100%|██████████| 168/168 [01:57<00:00,  1.62it/s, cost=3.04]
    minibatch loop: 100%|██████████| 168/168 [02:01<00:00,  1.57it/s, cost=0.0263]
    minibatch loop: 100%|██████████| 168/168 [02:00<00:00,  1.55it/s, cost=0.0103]
    minibatch loop: 100%|██████████| 168/168 [02:01<00:00,  1.58it/s, cost=0.00615]
    minibatch loop: 100%|██████████| 168/168 [01:59<00:00,  1.51it/s, cost=0.00474]


.. code:: ipython3

    deep_topic.get_similarity(news, anchor = 0.5)




.. parsed-literal::

    ['tan sri mokhzani mahathir',
     'najib razak',
     'pusat transformasi bandar',
     'anthony loke siew fook',
     '#fakenews',
     'survei institut darul ehsan',
     'makro-ekonomi',
     'pilihan raya umum ke-14',
     'malaysia-indonesia',
     'k-pop',
     'lee kuan yew',
     'tengku razaleigh hamzah',
     'tan sri dr rais yatim',
     'mikro-ekonomi']



.. code:: ipython3

    deep_topic.get_similarity(second_news, anchor = 0.5)




.. parsed-literal::

    ['tan sri mokhzani mahathir',
     'kkmm',
     'rais yatim',
     'datuk seri abdul hadi awang',
     'survei institut darul ehsan',
     'pilihan raya umum ke-14',
     'ahli dewan undangan negeri',
     'malaysia-indonesia',
     'datuk seri ti lian ker',
     'k-pop',
     'datuk seri azmin ali',
     'tengku razaleigh hamzah',
     'pusat daerah mangundi',
     'jabatan agama islam wilayah persekutuan',
     'pusat transformasi bandar',
     'inisiatif peduli rakyat',
     'makro-ekonomi',
     'anthony loke siew fook',
     'nga kor ming',
     'lee kuan yew',
     'tunku ismail idris',
     'tan sri dr rais yatim',
     '#fakenews',
     'mikro-ekonomi']



Train skip-thought model for influencers analysis
-------------------------------------------------

.. code:: ipython3

    deep_influencer = malaya.deep_get_influencers()


.. parsed-literal::

    minibatch loop: 100%|██████████| 24/24 [00:15<00:00,  1.55it/s, cost=3.64]
    minibatch loop: 100%|██████████| 24/24 [00:14<00:00,  1.68it/s, cost=1.45]
    minibatch loop: 100%|██████████| 24/24 [00:15<00:00,  1.40it/s, cost=0.55] 
    minibatch loop: 100%|██████████| 24/24 [00:14<00:00,  1.69it/s, cost=0.362]
    minibatch loop: 100%|██████████| 24/24 [00:15<00:00,  1.63it/s, cost=0.275]
    minibatch loop: 100%|██████████| 24/24 [00:14<00:00,  1.62it/s, cost=0.249]
    minibatch loop: 100%|██████████| 24/24 [00:15<00:00,  1.63it/s, cost=0.237] 
    minibatch loop: 100%|██████████| 24/24 [00:14<00:00,  1.64it/s, cost=0.207] 
    minibatch loop: 100%|██████████| 24/24 [00:16<00:00,  1.55it/s, cost=0.262]
    minibatch loop: 100%|██████████| 24/24 [00:15<00:00,  1.44it/s, cost=0.229] 


.. code:: ipython3

    deep_influencer.get_similarity(news, anchor = 0.5)




.. parsed-literal::

    ['najib razak', 'anthony loke siew fook', 'datuk seri azmin ali', 'mahathir']



.. code:: ipython3

    deep_influencer.get_similarity(second_news, anchor = 0.5)




.. parsed-literal::

    ['gobind singh deo']



Train siamese network for topics analysis
-----------------------------------------

.. code:: ipython3

    deep_topic = malaya.deep_siamese_get_topics()
    print(deep_topic.get_similarity(news, anchor = 0.5))
    print(deep_topic.get_similarity(second_news, anchor = 0.5))


.. parsed-literal::

    minibatch loop: 100%|██████████| 168/168 [02:03<00:00,  1.60it/s, accuracy=0.75, cost=0.113] 
    minibatch loop: 100%|██████████| 168/168 [02:01<00:00,  1.64it/s, accuracy=1, cost=0.0975]   
    minibatch loop: 100%|██████████| 168/168 [02:10<00:00,  1.65it/s, accuracy=1, cost=0.0539]   
    minibatch loop: 100%|██████████| 168/168 [01:59<00:00,  1.64it/s, accuracy=1, cost=0.057]     
    minibatch loop: 100%|██████████| 168/168 [01:58<00:00,  1.68it/s, accuracy=1, cost=0.0324]    


.. parsed-literal::

    ['tan sri mokhzani mahathir', 'najib razak', 'internet', 'rais yatim', 'anwar ibrahim', '1mdb', 'makanan', 'idealogi', 'recep tayyip erdogan', 'datuk seri abdul hadi awang', 'fc bayern munich', 'tsunami fitnah', 'thai cave', 'oppo smartphone', 'arsenal fc', 'jho low', 'datuk johari abdul', 'teknologi', 'syed saddiq', 'liverpool fc', 'isu bumiputera', 'startup companies', 'datuk seri ti lian ker', 'kadir jasin', 'datuk seri azmin ali', 'ptptn', 'tengku razaleigh hamzah', 'tabung haji', 'isu kemiskinan', 'pengangkutan awam', 'perkhidmatan awam', 'wanita', 'euro 2020', 'ganja', 'sosial', 'twitter', 'huawei smartphone', 'anthony loke siew fook', 'felda', 'fc barcelona', 'bebas tahanan', 'gst', 'ekonomi', 'lee kuan yew', 'tunku ismail idris', 'baling botol', 'masyarakat', 'sosial media', 'ariff md yusof', 'tan sri dr rais yatim', 'lenovo smartphone', 'chelsea fc', '#fakenews', 'umno', 'median salary', 'gaji minimum', 'juventus fc', 'kesihatan', 'mikro-ekonomi']
    ['anwar ibrahim', 'makanan', 'recep tayyip erdogan', 'datuk seri abdul hadi awang', 'fc bayern munich', 'tsunami fitnah', 'jho low', 'syed saddiq', 'liverpool fc', 'tabung haji', 'tengku razaleigh hamzah', 'pengangkutan awam', 'wanita', 'euro 2020', 'ganja', 'fc barcelona', 'felda', 'bung mokhtar', 'bebas tahanan', 'gst', 'ekonomi', 'lee kuan yew', 'baling botol', 'ariff md yusof', 'chelsea fc', 'median salary', 'gaji minimum', 'kesihatan']


.. code:: ipython3

    print(deep_topic.get_similarity(news, anchor = 0.7))
    print(deep_topic.get_similarity(second_news, anchor = 0.7))


.. parsed-literal::

    ['tan sri mokhzani mahathir', 'ganja', 'syed saddiq', 'sosial', 'chelsea fc', 'makanan', 'liverpool fc', 'felda', 'datuk seri abdul hadi awang', 'gaji minimum', 'juventus fc', 'baling botol', 'datuk seri azmin ali', 'masyarakat', 'arsenal fc', 'pengangkutan awam', 'perkhidmatan awam', 'euro 2020', 'jho low']
    []

