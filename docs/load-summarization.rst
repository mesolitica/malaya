
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 6.39 s, sys: 1.44 s, total: 7.82 s
    Wall time: 11.9 s


.. code:: python

    isu_kerajaan = [
        'Kenyataan kontroversi Setiausaha Agung Barisan Nasional (BN), Datuk Seri Mohamed Nazri Aziz berhubung sekolah vernakular merupakan pandangan peribadi beliau',
        'Timbalan Presiden UMNO, Datuk Seri Mohamad Hasan berkata, kenyataan tersebut tidak mewakili pendirian serta pandangan UMNO \n\nkerana parti itu menghormati serta memahami keperluan sekolah vernakular dalam negara',
        '"Saya ingin menegaskan dua perkara penting',
        'Pertama pendirian beliau tersebut adalah pandangan peribadi yang tidak mewakili pendirian dan pandangan UMNO',
        '"Kedua UMNO sebagai sebuah parti sangat menghormati dan memahami keperluan sekolah vernakular di Malaysia',
        'UMNO berpendirian sekolah jenis ini perlu terus wujud di negara kita," katanya dalam satu kenyataan akhbar malam ini',
        'Mohamed Nazri semalam menjelaskan, kenyataannya mengenai sekolah jenis kebangsaan Cina dan Tamil baru-baru ini disalah petik pihak media',
        'Kata Nazri dalam kenyataannya itu, beliau menekankan bahawa semua pihak perlu menghormati hak orang Melayu dan bumiputera',
        'Mohamad yang menjalankan tugas-tugas Presiden UMNO berkata, UMNO konsisten dengan pendirian itu dalam mengiktiraf kepelbagaian bangsa dan etnik termasuk hak untuk beragama serta mendapat pendidikan',
        'Menurut beliau, persefahaman dan keupayaan meraikan kepelbagaian itu menjadi kelebihan dan kekuatan UMNO dan BN selama ini',
        'Kata beliau, komitmen UMNO dan BN berhubung perkara itu dapat dilihat dengan jelas dalam bentuk sokongan infrastruktur, pengiktirafan dan pemberian peruntukan yang diperlukan',
        '"Saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar UMNO dan BN," katanya',
        'Beliau turut menegaskan Mohamed Nazri telah mengambil pertanggungjawaban dengan membuat penjelasan maksud sebenarnya ucapanny di Semenyih, Selangor tersebut',
    ]

.. code:: python

    isu_string = '\n\n\n\nDUA legenda hebat dan ‘The living legend’ ini sudah memartabatkan bidang muzik sejak lebih tiga dekad lalu. Jika Datuk Zainal Abidin, 59, dikenali sebagai penyanyi yang memperjuangkan konsep ‘world music’, Datuk Sheila Majid, 55, pula lebih dikenali dengan irama jazz dan R&B.\n\nNamun, ada satu persamaan yang mengeratkan hubungan mereka kerana sama-sama mencintai bidang muzik sejak dulu.\n\nKetika ditemui dalam sesi fotografi yang diatur di Balai Berita, baru-baru ini, Zainal berkata, dia lebih ‘senior’ daripada Sheila kerana bermula dengan kumpulan Headwind sebelum menempa nama sebagai penyanyi solo.\n\n“Saya mula berkawan rapat dengan Sheila ketika sama-sama bernaung di bawah pengurusan Roslan Aziz Productions (RAP) selepas membina karier sebagai artis solo.\n\n“Namun, selepas tidak lagi bernaung di bawah RAP, kami juga membawa haluan karier seni masing-masing selepas itu,” katanya.\n\nJusteru katanya, dia memang menanti peluang berganding dengan Sheila dalam satu konsert.\n\nPenyanyi yang popular dengan lagu Hijau dan Ikhlas Tapi Jauh itu mengakui mereka memang ada keserasian ketika bergandingan kerana membesar pada era muzik yang sama.\n\n“Kami memang meminati bidang muzik dan saling memahami antara satu sama lain. Mungkin kerana kami berdua sudah berada pada tahap di puncak karier muzik masing-masing.\n\n“Saya bersama Sheila serta Datuk Afdlin Shauki akan terbabit dalam satu segmen yang ditetapkan.\n\n“Selain persembahan solo, saya juga berduet dengan Sheila dan Afdlin dalam segmen interaktif ini. Setiap penyanyi akan menyampaikan enam hingga tujuh lagu setiap seorang sepanjang konsert yang berlangsung tiga hari ini,” katanya.\n\nBagi Sheila pula, dia memang ada terbabit dengan beberapa persembahan bersama Zainal cuma tiada publisiti ketika itu.\n\n“Kami pernah terbabit dengan showcase dan majlis korporat sebelum ini. Selain itu, Zainal juga terbabit dengan Konsert Legenda yang membabitkan jelajah empat lokasi sebelum ini.\n\n“Sebab itu, saya sukar menolak untuk bekerjasama dengannya dalam Festival KL Jamm yang dianjurkan buat julung kali dan berkongsi pentas dalam satu konsert bertaraf antarabangsa,” katanya.\n\n\n\nFESTIVAL KL Jamm bakal menggabungkan pelbagai genre muzik seperti rock, hip hop, jazz dan pop dengan lebih 100 persembahan, 20 ‘showcase’ dan pameran.\n\nKonsert berbayar\n\n\n\nMewakili golongan anak seni, Sheila menaruh harapan semoga Festival KL Jamm akan menjadi platform buat artis yang sudah ada nama dan artis muda untuk membuat persembahan, sekali gus sama-sama memartabatkan industri muzik tempatan.\n\nMenurut Sheila, dia juga mencadangkan lebih banyak tempat diwujudkan untuk menggalakkan artis muda membuat persembahan, sekali gus menggilap bakat mereka.\n\n“Berbanding pada zaman saya dulu, artis muda sekarang tidak banyak tempat khusus untuk mereka menyanyi dan menonjolkan bakat di tempat awam.\n\n“Rata-rata hanya sekadar menyanyi di laman Instagram dan cuma dikenali menerusi satu lagu. Justeru, bagaimana mereka mahu buat showcase kalau hanya dikenali dengan satu lagu?” katanya.\n\nPada masa sama, Sheila juga merayu peminat tempatan untuk sama-sama memberi sokongan pada penganjuran festival KL Jamm sekali gus mencapai objektifnya.\n\n“Peminat perlu ubah persepsi negatif mereka dengan menganggap persembahan artis tempatan tidak bagus.\n\n“Kemasukan artis luar juga perlu dilihat dari sudut yang positif kerana kita perlu belajar bagaimana untuk menjadi bagus seperti mereka,” katanya.\n\nSementara itu, Zainal pula berharap festival itu akan mendidik orang ramai untuk menonton konsert berbayar serta memberi sokongan pada artis tempatan.\n\n“Ramai yang hanya meminati artis tempatan tetapi tidak mahu mengeluarkan sedikit wang untuk membeli tiket konsert mereka.\n\n“Sedangkan artis juga menyanyi untuk kerjaya dan ia juga punca pendapatan bagi menyara hidup,” katanya.\n\nFestival KL Jamm bakal menghimpunkan barisan artis tempatan baru dan nama besar dalam konsert iaitu Datuk Ramli Sarip, Datuk Afdlin Shauki, Zamani, Amelina, Radhi OAG, Dr Burn, Santesh, Rabbit Mac, Sheezy, kumpulan Bunkface, Ruffedge, Pot Innuendo, artis dari Kartel (Joe Flizzow, Sona One, Ila Damia, Yung Raja, Faris Jabba dan Abu Bakarxli) dan Malaysia Pasangge (artis India tempatan).\n\nManakala, artis antarabangsa pula membabitkan J Arie (Hong Kong), NCT Dream (Korea Selatan) dan DJ Sura (Korea Selatan).\n\nKL Jamm dianjurkan Music Unlimited International Sdn Bhd dan bakal menggabungkan pelbagai genre muzik seperti rock, hip hop, jazz dan pop dengan lebih 100 persembahan, 20 ‘showcase’, pameran dan perdagangan berkaitan.\n\nFestival tiga hari itu bakal berlangsung di Pusat Pameran dan Perdagangan Antarabangsa Malaysia (MITEC), Kuala Lumpur pada 26 hingga 28 April ini.\n\nMaklumat mengenai pembelian tiket dan keterangan lanjut boleh melayari www.kljamm.com.'

We also can give a string, Malaya will always split a string by into
multiple sentences.

Important parameters,

1. ``top_k``, number of summarized strings.
2. ``important_words``, number of important words.

List available skip-thought models
----------------------------------

.. code:: python

    malaya.summarize.available_skipthought()




.. parsed-literal::

    ['lstm', 'residual-network']



-  ``'lstm'`` - LSTM skip-thought deep learning model trained on news
   dataset. Hopefully we can train on wikipedia dataset.
-  ``'residual-network'`` - CNN residual network with Bahdanau Attention
   skip-thought deep learning model trained on wikipedia dataset.

We use TextRank for scoring algorithm.

Encoder summarization
---------------------

We leverage the power of deep encoder models like skip-thought, BERT and
XLNET to do extractive summarization for us.

Use skip-thought
^^^^^^^^^^^^^^^^

.. code:: python

    lstm = malaya.summarize.deep_skipthought(model = 'lstm')
    encoder = malaya.summarize.encoder(lstm)


.. parsed-literal::

    downloading SUMMARIZE news frozen model


.. parsed-literal::

    56.0MB [00:13, 4.18MB/s]
      0%|          | 0.00/0.98 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading SUMMARIZE news dictionary


.. parsed-literal::

    1.00MB [00:00, 3.55MB/s]
    WARNING: Logging before flag parsing goes to stderr.
    W1018 00:37:31.562124 4502980032 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:68: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.

    W1018 00:37:31.563504 4502980032 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:69: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.

    W1018 00:37:32.268280 4502980032 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_models/_skip_thought.py:136: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.



.. code:: python

    encoder.summarize(isu_kerajaan, important_words = 10)




.. parsed-literal::

    {'summary': 'Pertama pendirian beliau tersebut adalah pandangan peribadi yang tidak mewakili pendirian dan pandangan UMNO. UMNO berpendirian sekolah jenis ini perlu terus wujud di negara kita," katanya dalam satu kenyataan akhbar malam ini. "Saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar UMNO dan BN," katanya.',
     'top-words': ['bersabdabarangsiapa',
      'kepulangan',
      'seliakekurangan',
      'poupart',
      'sharmini',
      'pembikin',
      'sakailah',
      'chusus',
      'mempunya',
      'diharap'],
     'cluster-top-words': ['sakailah',
      'chusus',
      'bersabdabarangsiapa',
      'sharmini',
      'seliakekurangan',
      'kepulangan',
      'pembikin',
      'poupart',
      'mempunya',
      'diharap']}



Problem with skip-thought models, ``top-words`` suggested are really not
good, because skip-thought trained to leverage sentence level, not word
level. How about XLNET or BERT? Lets we try XLNET.

.. code:: python

    xlnet = malaya.transformer.load(model = 'xlnet', size = 'base')
    encoder = malaya.summarize.encoder(xlnet)


.. parsed-literal::

    W1018 00:38:15.394328 4502980032 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet_model/xlnet.py:70: The name tf.gfile.Open is deprecated. Please use tf.io.gfile.GFile instead.

    W1018 00:38:15.397522 4502980032 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet.py:71: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

    W1018 00:38:15.409126 4502980032 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet_model/xlnet.py:253: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

    W1018 00:38:15.411124 4502980032 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet_model/xlnet.py:253: The name tf.AUTO_REUSE is deprecated. Please use tf.compat.v1.AUTO_REUSE instead.

    W1018 00:38:15.415867 4502980032 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet_model/modeling.py:686: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.

    W1018 00:38:15.514928 4502980032 deprecation.py:323] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet_model/modeling.py:797: dropout (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.dropout instead.
    W1018 00:38:16.088240 4502980032 deprecation.py:323] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet_model/modeling.py:99: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.dense instead.
    W1018 00:38:24.365406 4502980032 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet.py:90: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

    W1018 00:38:24.663450 4502980032 deprecation.py:323] From /usr/local/lib/python3.6/site-packages/tensorflow/python/training/saver.py:1276: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use standard file APIs to check for files with this prefix.


.. code:: python

    encoder.summarize(isu_kerajaan, important_words = 10, method = 'mean')




.. parsed-literal::

    {'summary': 'Kata Nazri dalam kenyataannya itu, beliau menekankan bahawa semua pihak perlu menghormati hak orang Melayu dan bumiputera. Mohamad yang menjalankan tugas-tugas Presiden UMNO berkata, UMNO konsisten dengan pendirian itu dalam mengiktiraf kepelbagaian bangsa dan etnik termasuk hak untuk beragama serta mendapat pendidikan. Mohamed Nazri semalam menjelaskan, kenyataannya mengenai sekolah jenis kebangsaan Cina dan Tamil baru-baru ini disalah petik pihak media.',
     'top-words': ['umno',
      'pendirian',
      'vernakular',
      'pandangan',
      'menghormati',
      'sekolah',
      'nazri',
      'kenyataan',
      'hak',
      'peribadi'],
     'cluster-top-words': ['pendirian',
      'kenyataan',
      'umno',
      'vernakular',
      'hak',
      'nazri',
      'sekolah',
      'peribadi',
      'menghormati',
      'pandangan']}



Much much better!

Train LSA model
---------------

Important parameters,

1. ``vectorizer``, vectorizer technique. Allowed values:

   -  ``'bow'`` - Bag of Word.
   -  ``'tfidf'`` - Term frequency inverse Document Frequency.
   -  ``'skip-gram'`` - Bag of Word with skipping certain n-grams.

2. ``ngram``, n-grams size to train a corpus.
3. ``important_words``, number of important words.
4. ``top_k``, number of summarized strings.

.. code:: python

    malaya.summarize.lsa(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'Timbalan Presiden UMNO, Datuk Seri Mohamad Hasan berkata, kenyataan tersebut tidak mewakili pendirian serta pandangan UMNO   kerana parti itu menghormati serta memahami keperluan sekolah vernakular dalam negara. "Saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar UMNO dan BN," katanya. UMNO berpendirian sekolah jenis ini perlu terus wujud di negara kita," katanya dalam satu kenyataan akhbar malam ini.',
     'top-words': ['umno',
      'nyata',
      'sekolah',
      'pandang',
      'vernakular',
      'hormat',
      'sekolah vernakular',
      'nazri',
      'hormat paham',
      'hak'],
     'cluster-top-words': ['sekolah vernakular',
      'umno',
      'nyata',
      'hak',
      'nazri',
      'hormat paham',
      'pandang']}



We can use ``tfidf`` as vectorizer.

.. code:: python

    malaya.summarize.lsa(isu_kerajaan,important_words=10, ngram = (1,3), vectorizer = 'tfidf')




.. parsed-literal::

    {'summary': 'Timbalan Presiden UMNO, Datuk Seri Mohamad Hasan berkata, kenyataan tersebut tidak mewakili pendirian serta pandangan UMNO   kerana parti itu menghormati serta memahami keperluan sekolah vernakular dalam negara. Mohamad yang menjalankan tugas-tugas Presiden UMNO berkata, UMNO konsisten dengan pendirian itu dalam mengiktiraf kepelbagaian bangsa dan etnik termasuk hak untuk beragama serta mendapat pendidikan. "Saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar UMNO dan BN," katanya.',
     'top-words': ['wakil pandang umno',
      'mohamed',
      'paham sekolah vernakular',
      'paham sekolah',
      'paham',
      'negara',
      'nazri nyata',
      'mohamed nazri',
      'mohamad',
      'pandang peribadi'],
     'cluster-top-words': ['wakil pandang umno',
      'mohamed nazri',
      'pandang peribadi',
      'mohamad',
      'paham sekolah vernakular',
      'nazri nyata',
      'negara']}



We can use ``skip-gram`` as vectorizer, and can override ``skip`` value.

.. code:: python

    malaya.summarize.lsa(isu_kerajaan,important_words=10, ngram = (1,3), vectorizer = 'skip-gram', skip = 3)




.. parsed-literal::

    {'summary': 'Timbalan Presiden UMNO, Datuk Seri Mohamad Hasan berkata, kenyataan tersebut tidak mewakili pendirian serta pandangan UMNO   kerana parti itu menghormati serta memahami keperluan sekolah vernakular dalam negara. "Saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar UMNO dan BN," katanya. Mohamad yang menjalankan tugas-tugas Presiden UMNO berkata, UMNO konsisten dengan pendirian itu dalam mengiktiraf kepelbagaian bangsa dan etnik termasuk hak untuk beragama serta mendapat pendidikan.',
     'top-words': ['umno',
      'sekolah',
      'nyata',
      'pandang',
      'nazri',
      'hormat',
      'vernakular',
      'pandang umno',
      'sekolah vernakular',
      'presiden umno'],
     'cluster-top-words': ['sekolah vernakular',
      'pandang umno',
      'nyata',
      'presiden umno',
      'hormat',
      'nazri']}



.. code:: python

    malaya.summarize.lsa(isu_string,important_words=10)




.. parsed-literal::

    {'summary': 'Konsert berbayar    Mewakili golongan anak seni, Sheila menaruh harapan semoga Festival KL Jamm akan menjadi platform buat artis yang sudah ada nama dan artis muda untuk membuat persembahan, sekali gus sama-sama memartabatkan industri muzik tempatan. Festival KL Jamm bakal menghimpunkan barisan artis tempatan baru dan nama besar dalam konsert iaitu Datuk Ramli Sarip, Datuk Afdlin Shauki, Zamani, Amelina, Radhi OAG, Dr Burn, Santesh, Rabbit Mac, Sheezy, kumpulan Bunkface, Ruffedge, Pot Innuendo, artis dari Kartel (Joe Flizzow, Sona One, Ila Damia, Yung Raja, Faris Jabba dan Abu Bakarxli) dan Malaysia Pasangge (artis India tempatan). "Sedangkan artis juga menyanyi untuk kerjaya dan ia juga punca pendapatan bagi menyara hidup," katanya.',
     'top-words': ['artis',
      'sheila',
      'konsert',
      'muzik',
      'nyanyi',
      'sembah',
      'festival',
      'jamm',
      'kl',
      'babit'],
     'cluster-top-words': ['muzik',
      'babit',
      'konsert',
      'nyanyi',
      'jamm',
      'artis',
      'kl',
      'sheila',
      'sembah',
      'festival']}



Train LDA model
---------------

.. code:: python

    malaya.summarize.lda(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': '"Saya ingin menegaskan dua perkara penting. "Saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar UMNO dan BN," katanya. Menurut beliau, persefahaman dan keupayaan meraikan kepelbagaian itu menjadi kelebihan dan kekuatan UMNO dan BN selama ini.',
     'top-words': ['umno',
      'nyata',
      'sekolah',
      'pandang',
      'vernakular',
      'hormat',
      'sekolah vernakular',
      'nazri',
      'hormat paham',
      'hak'],
     'cluster-top-words': ['sekolah vernakular',
      'umno',
      'nyata',
      'hak',
      'nazri',
      'hormat paham',
      'pandang']}



.. code:: python

    malaya.summarize.lda(isu_string,important_words=10, vectorizer = 'skip-gram')




.. parsed-literal::

    {'summary': '"Sedangkan artis juga menyanyi untuk kerjaya dan ia juga punca pendapatan bagi menyara hidup," katanya. Penyanyi yang popular dengan lagu Hijau dan Ikhlas Tapi Jauh itu mengakui mereka memang ada keserasian ketika bergandingan kerana membesar pada era muzik yang sama. "Kami memang meminati bidang muzik dan saling memahami antara satu sama lain.',
     'top-words': ['artis',
      'sheila',
      'konsert',
      'muzik',
      'festival',
      'sembah',
      'nyanyi',
      'kl',
      'kl jamm',
      'jamm'],
     'cluster-top-words': ['muzik',
      'konsert',
      'nyanyi',
      'artis',
      'sheila',
      'sembah',
      'festival',
      'kl jamm']}



Load doc2vec summarization
--------------------------

We need to load word vector provided by Malaya. ``doc2vec`` does not
return ``top-words``, so parameter ``important_words`` cannot be use.

Important parameters, 1. ``aggregation``, aggregation function to
accumulate word vectors. Default is ``mean``.

::

   * ``'mean'`` - mean.
   * ``'min'`` - min.
   * ``'max'`` - max.
   * ``'sum'`` - sum.
   * ``'sqrt'`` - square root.

Using word2vec
^^^^^^^^^^^^^^

I will use ``load_news``, word2vec from wikipedia took a very long time
for my noob laptop,

.. code:: python

    embedded_news = malaya.wordvector.load_news(256)
    w2v_wiki = malaya.wordvector.load(embedded_news['nce_weights'],
                                        embedded_news['dictionary'])

.. code:: python

    malaya.summarize.doc2vec(w2v_wiki, isu_kerajaan, soft = False, top_k = 5)




.. parsed-literal::

    'Timbalan Presiden UMNO, Datuk Seri Mohamad Hasan berkata, kenyataan tersebut tidak mewakili pendirian serta pandangan UMNO   kerana parti itu menghormati serta memahami keperluan sekolah vernakular dalam negara. Mohamad yang menjalankan tugas-tugas Presiden UMNO berkata, UMNO konsisten dengan pendirian itu dalam mengiktiraf kepelbagaian bangsa dan etnik termasuk hak untuk beragama serta mendapat pendidikan. Kata Nazri dalam kenyataannya itu, beliau menekankan bahawa semua pihak perlu menghormati hak orang Melayu dan bumiputera. Kata beliau, komitmen UMNO dan BN berhubung perkara itu dapat dilihat dengan jelas dalam bentuk sokongan infrastruktur, pengiktirafan dan pemberian peruntukan yang diperlukan. "Kedua UMNO sebagai sebuah parti sangat menghormati dan memahami keperluan sekolah vernakular di Malaysia.'
