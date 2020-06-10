.. code:: python

    %%time
    import malaya
    from pprint import pprint


.. parsed-literal::

    CPU times: user 4.4 s, sys: 908 ms, total: 5.31 s
    Wall time: 4.63 s


List available skip-thought models
----------------------------------

.. code:: python

    malaya.summarization.extractive.available_skipthought()




.. parsed-literal::

    ['lstm', 'residual-network']



-  ``'lstm'`` - LSTM skip-thought deep learning model trained on news
   dataset. Hopefully we can train on wikipedia dataset.
-  ``'residual-network'`` - CNN residual network with Bahdanau Attention
   skip-thought deep learning model trained on wikipedia dataset.

**We use TextRank for scoring algorithm.**

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

Encoder summarization
---------------------

We leverage the power of deep encoder models like skip-thought or
Transformer to do extractive summarization for us.

Extractive summarization from Malaya 100% unsupervised, we simply feed
output from models into TextRank score.

.. code:: python

   def summarize(
       self, corpus, top_k: int = 3, important_words: int = 3, **kwargs
   ):
       """
       Summarize list of strings / corpus

       Parameters
       ----------
       corpus: str, list

       top_k: int, (default=3)
           number of summarized strings.
       important_words: int, (default=3)
           number of important words.

       Returns
       -------
       string: summarized string
       """

We also can give a string, Malaya will always split a string by into
multiple sentences.

Important parameters,

1. ``top_k``, number of summarized strings.
2. ``important_words``, number of important words.

Load skip-thought
^^^^^^^^^^^^^^^^^

.. code:: python

    lstm = malaya.summarization.extractive.deep_skipthought(model = 'lstm')
    encoder = malaya.summarization.extractive.encoder(lstm)


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:54: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:55: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/model/skip_thought.py:136: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


.. code:: python

    encoder.summarize(isu_kerajaan, important_words = 10)




.. parsed-literal::

    {'summary': 'Pertama pendirian beliau tersebut adalah pandangan peribadi yang tidak mewakili pendirian dan pandangan UMNO. Beliau turut menegaskan Mohamed Nazri telah mengambil pertanggungjawaban dengan membuat penjelasan maksud sebenarnya ucapanny di Semenyih, Selangor tersebut. UMNO berpendirian sekolah jenis ini perlu terus wujud di negara kita," katanya dalam satu kenyataan akhbar malam ini.',
     'top-words': ['bersabdabarangsiapa',
      'kepulangan',
      'seliakekurangan',
      'pembikin',
      'poupart',
      'sharmini',
      'mempunya',
      'chusus',
      'sakailah',
      'diharap'],
     'cluster-top-words': ['chusus',
      'mempunya',
      'sakailah',
      'seliakekurangan',
      'sharmini',
      'bersabdabarangsiapa',
      'diharap',
      'kepulangan',
      'poupart',
      'pembikin']}



Problem with skip-thought models, ``top-words`` suggested are really not
good, because skip-thought trained to leverage sentence level, not word
level. How about Transformer model? Lets we try ALXLNET.

Load Transformer
^^^^^^^^^^^^^^^^

.. code:: python

    alxlnet = malaya.transformer.load(model = 'alxlnet')
    encoder = malaya.summarization.extractive.encoder(alxlnet)


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/alxlnet/xlnet.py:70: The name tf.gfile.Open is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/alxlnet/__init__.py:81: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/alxlnet/xlnet.py:253: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/alxlnet/xlnet.py:253: The name tf.AUTO_REUSE is deprecated. Please use tf.compat.v1.AUTO_REUSE instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/alxlnet/modeling.py:697: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.
    
    INFO:tensorflow:memory input None
    INFO:tensorflow:Use float type <dtype: 'float32'>
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/alxlnet/modeling.py:704: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/alxlnet/modeling.py:809: dropout (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.dropout instead.
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow_core/python/layers/core.py:271: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.__call__` method instead.
    WARNING:tensorflow:
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
      * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/alxlnet/modeling.py:109: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.Dense instead.
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/alxlnet/__init__.py:95: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/alxlnet/__init__.py:96: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/alxlnet/__init__.py:100: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/alxlnet/__init__.py:103: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/alxlnet-model/base/alxlnet-base/model.ckpt


.. code:: python

    encoder.summarize(isu_kerajaan, important_words = 10, method = 'mean')




.. parsed-literal::

    {'summary': 'Kata Nazri dalam kenyataannya itu, beliau menekankan bahawa semua pihak perlu menghormati hak orang Melayu dan bumiputera. Pertama pendirian beliau tersebut adalah pandangan peribadi yang tidak mewakili pendirian dan pandangan UMNO. Mohamed Nazri semalam menjelaskan, kenyataannya mengenai sekolah jenis kebangsaan Cina dan Tamil baru-baru ini disalah petik pihak media.',
     'top-words': ['umno',
      'malaysia',
      'bumiputera',
      'media',
      'negara',
      'sekolah',
      'pendidikan',
      'pendirian',
      'nazri',
      'pandangan'],
     'cluster-top-words': ['sekolah',
      'bumiputera',
      'umno',
      'pendidikan',
      'malaysia',
      'media',
      'pendirian',
      'negara',
      'pandangan',
      'nazri']}



Much much better!

Feed LSA model
--------------

.. code:: python


   def lda(
       corpus,
       top_k = 3,
       important_words = 10,
       max_df = 0.95,
       min_df = 2,
       ngram = (1, 3),
       vectorizer = 'bow',
       **kwargs,
   ):
       """
       summarize a list of strings using LDA, scoring using TextRank.

       Parameters
       ----------
       corpus: list
       top_k: int, (default=3)
           number of summarized strings.
       important_words: int, (default=10)
           number of important words.
       max_df: float, (default=0.95)
           maximum of a word selected based on document frequency.
       min_df: int, (default=2)
           minimum of a word selected on based on document frequency.
       ngram: tuple, (default=(1,3))
           n-grams size to train a corpus.
       vectorizer: str, (default='bow')
           vectorizer technique. Allowed values:

           * ``'bow'`` - Bag of Word.
           * ``'tfidf'`` - Term frequency inverse Document Frequency.
           * ``'skip-gram'`` - Bag of Word with skipping certain n-grams.

Important parameters,

1. ``vectorizer``, vectorizer technique. Allowed values:

   -  ``'bow'`` - Bag of Word.
   -  ``'tfidf'`` - Term frequency inverse Document Frequency.
   -  ``'skip-gram'`` - Bag of Word with skipping certain n-grams.

2. ``ngram``, n-grams size to train a corpus.
3. ``important_words``, number of important words.
4. ``top_k``, number of summarized strings.

.. code:: python

    malaya.summarization.extractive.lsa(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'Timbalan Presiden UMNO, Datuk Seri Mohamad Hasan berkata, kenyataan tersebut tidak mewakili pendirian serta pandangan UMNO   kerana parti itu menghormati serta memahami keperluan sekolah vernakular dalam negara. Mohamad yang menjalankan tugas-tugas Presiden UMNO berkata, UMNO konsisten dengan pendirian itu dalam mengiktiraf kepelbagaian bangsa dan etnik termasuk hak untuk beragama serta mendapat pendidikan. "Saya ingin menegaskan dua perkara penting.',
     'top-words': ['umno',
      'nyata',
      'sekolah',
      'nazri',
      'pandang',
      'hormat',
      'sekolah vernakular',
      'vernakular',
      'mohamed',
      'mohamed nazri'],
     'cluster-top-words': ['nyata',
      'umno',
      'hormat',
      'mohamed nazri',
      'sekolah vernakular',
      'pandang']}



We can use ``tfidf`` as vectorizer.

.. code:: python

    malaya.summarization.extractive.lsa(isu_kerajaan,important_words=10, ngram = (1,3), vectorizer = 'tfidf')




.. parsed-literal::

    {'summary': 'Timbalan Presiden UMNO, Datuk Seri Mohamad Hasan berkata, kenyataan tersebut tidak mewakili pendirian serta pandangan UMNO   kerana parti itu menghormati serta memahami keperluan sekolah vernakular dalam negara. Mohamad yang menjalankan tugas-tugas Presiden UMNO berkata, UMNO konsisten dengan pendirian itu dalam mengiktiraf kepelbagaian bangsa dan etnik termasuk hak untuk beragama serta mendapat pendidikan. Kenyataan kontroversi Setiausaha Agung Barisan Nasional (BN), Datuk Seri Mohamed Nazri Aziz berhubung sekolah vernakular merupakan pandangan peribadi beliau.',
     'top-words': ['wakil pandang umno',
      'pandang umno',
      'hak',
      'hormat paham',
      'hormat paham sekolah',
      'hubung',
      'iktiraf',
      'jenis',
      'mohamad',
      'nazri nyata'],
     'cluster-top-words': ['jenis',
      'nazri nyata',
      'iktiraf',
      'wakil pandang umno',
      'hak',
      'mohamad',
      'hormat paham sekolah',
      'hubung']}



We can use ``skip-gram`` as vectorizer, and can override ``skip`` value.

.. code:: python

    malaya.summarization.extractive.lsa(isu_kerajaan,important_words=10, ngram = (1,3), vectorizer = 'skip-gram', skip = 3)




.. parsed-literal::

    {'summary': '"Saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar UMNO dan BN," katanya. Timbalan Presiden UMNO, Datuk Seri Mohamad Hasan berkata, kenyataan tersebut tidak mewakili pendirian serta pandangan UMNO   kerana parti itu menghormati serta memahami keperluan sekolah vernakular dalam negara. Mohamad yang menjalankan tugas-tugas Presiden UMNO berkata, UMNO konsisten dengan pendirian itu dalam mengiktiraf kepelbagaian bangsa dan etnik termasuk hak untuk beragama serta mendapat pendidikan.',
     'top-words': ['umno',
      'nyata',
      'sekolah',
      'pandang',
      'nazri',
      'mohamed',
      'mohamed nazri',
      'pandang umno',
      'sekolah vernakular',
      'presiden umno'],
     'cluster-top-words': ['pandang umno',
      'nyata',
      'mohamed nazri',
      'presiden umno',
      'sekolah vernakular']}



Feed LDA model
--------------

.. code:: python

    malaya.summarization.extractive.lda(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': '"Saya ingin menegaskan dua perkara penting. "Saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar UMNO dan BN," katanya. Kata beliau, komitmen UMNO dan BN berhubung perkara itu dapat dilihat dengan jelas dalam bentuk sokongan infrastruktur, pengiktirafan dan pemberian peruntukan yang diperlukan.',
     'top-words': ['umno',
      'nyata',
      'sekolah',
      'nazri',
      'pandang',
      'hormat',
      'sekolah vernakular',
      'vernakular',
      'mohamed',
      'mohamed nazri'],
     'cluster-top-words': ['nyata',
      'umno',
      'hormat',
      'mohamed nazri',
      'sekolah vernakular',
      'pandang']}



.. code:: python

    malaya.summarization.extractive.lda(isu_string,important_words=10, vectorizer = 'skip-gram')




.. parsed-literal::

    {'summary': 'katanya. Justeru, bagaimana mereka mahu buat showcase kalau hanya dikenali dengan satu lagu"? "Rata-rata hanya sekadar menyanyi di laman Instagram dan cuma dikenali menerusi satu lagu.',
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
     'cluster-top-words': ['sheila',
      'kl jamm',
      'nyanyi',
      'festival',
      'muzik',
      'konsert',
      'artis',
      'sembah']}



Load doc2vec summarization
--------------------------

We need to load word vector provided by Malaya. ``doc2vec`` does not
return ``top-words``, so parameter ``important_words`` cannot be use.

.. code:: python


   def doc2vec(
       vectorizer,
       corpus,
       top_k: int = 3,
       aggregation: int = 'mean',
       soft: bool = True,
   ):
       """
       summarize a list of strings using doc2vec, scoring using TextRank.

       Parameters
       ----------
       vectorizer : object
           fast-text or word2vec interface object.
       corpus: list
       top_k: int, (default=3)
           number of summarized strings.
       aggregation : str, optional (default='mean')
           Aggregation supported. Allowed values:

           * ``'mean'`` - mean.
           * ``'min'`` - min.
           * ``'max'`` - max.
           * ``'sum'`` - sum.
           * ``'sqrt'`` - square root.
       soft: bool, optional (default=True)
           word not inside vectorizer will replace with nearest word if True, else, will skip.

**We use TextRank for scoring algorithm.**

Using word2vec
^^^^^^^^^^^^^^

I will use ``load_news``, you can try embedded from wikipedia.

.. code:: python

    vocab_news, embedded_news = malaya.wordvector.load_news()
    w2v_wiki = malaya.wordvector.load(embedded_news, vocab_news)

.. code:: python

    malaya.summarization.extractive.doc2vec(w2v_wiki, isu_kerajaan, soft = False, top_k = 5)




.. parsed-literal::

    'Timbalan Presiden UMNO, Datuk Seri Mohamad Hasan berkata, kenyataan tersebut tidak mewakili pendirian serta pandangan UMNO   kerana parti itu menghormati serta memahami keperluan sekolah vernakular dalam negara. Mohamad yang menjalankan tugas-tugas Presiden UMNO berkata, UMNO konsisten dengan pendirian itu dalam mengiktiraf kepelbagaian bangsa dan etnik termasuk hak untuk beragama serta mendapat pendidikan. Kata Nazri dalam kenyataannya itu, beliau menekankan bahawa semua pihak perlu menghormati hak orang Melayu dan bumiputera. Kata beliau, komitmen UMNO dan BN berhubung perkara itu dapat dilihat dengan jelas dalam bentuk sokongan infrastruktur, pengiktirafan dan pemberian peruntukan yang diperlukan. "Saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar UMNO dan BN," katanya.'


