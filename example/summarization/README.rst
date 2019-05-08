
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 11.9 s, sys: 1.46 s, total: 13.4 s
    Wall time: 17 s


.. code:: ipython3

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

.. code:: ipython3

    isu_string = '\n\n\n\nDUA legenda hebat dan ‘The living legend’ ini sudah memartabatkan bidang muzik sejak lebih tiga dekad lalu. Jika Datuk Zainal Abidin, 59, dikenali sebagai penyanyi yang memperjuangkan konsep ‘world music’, Datuk Sheila Majid, 55, pula lebih dikenali dengan irama jazz dan R&B.\n\nNamun, ada satu persamaan yang mengeratkan hubungan mereka kerana sama-sama mencintai bidang muzik sejak dulu.\n\nKetika ditemui dalam sesi fotografi yang diatur di Balai Berita, baru-baru ini, Zainal berkata, dia lebih ‘senior’ daripada Sheila kerana bermula dengan kumpulan Headwind sebelum menempa nama sebagai penyanyi solo.\n\n“Saya mula berkawan rapat dengan Sheila ketika sama-sama bernaung di bawah pengurusan Roslan Aziz Productions (RAP) selepas membina karier sebagai artis solo.\n\n“Namun, selepas tidak lagi bernaung di bawah RAP, kami juga membawa haluan karier seni masing-masing selepas itu,” katanya.\n\nJusteru katanya, dia memang menanti peluang berganding dengan Sheila dalam satu konsert.\n\nPenyanyi yang popular dengan lagu Hijau dan Ikhlas Tapi Jauh itu mengakui mereka memang ada keserasian ketika bergandingan kerana membesar pada era muzik yang sama.\n\n“Kami memang meminati bidang muzik dan saling memahami antara satu sama lain. Mungkin kerana kami berdua sudah berada pada tahap di puncak karier muzik masing-masing.\n\n“Saya bersama Sheila serta Datuk Afdlin Shauki akan terbabit dalam satu segmen yang ditetapkan.\n\n“Selain persembahan solo, saya juga berduet dengan Sheila dan Afdlin dalam segmen interaktif ini. Setiap penyanyi akan menyampaikan enam hingga tujuh lagu setiap seorang sepanjang konsert yang berlangsung tiga hari ini,” katanya.\n\nBagi Sheila pula, dia memang ada terbabit dengan beberapa persembahan bersama Zainal cuma tiada publisiti ketika itu.\n\n“Kami pernah terbabit dengan showcase dan majlis korporat sebelum ini. Selain itu, Zainal juga terbabit dengan Konsert Legenda yang membabitkan jelajah empat lokasi sebelum ini.\n\n“Sebab itu, saya sukar menolak untuk bekerjasama dengannya dalam Festival KL Jamm yang dianjurkan buat julung kali dan berkongsi pentas dalam satu konsert bertaraf antarabangsa,” katanya.\n\n\n\nFESTIVAL KL Jamm bakal menggabungkan pelbagai genre muzik seperti rock, hip hop, jazz dan pop dengan lebih 100 persembahan, 20 ‘showcase’ dan pameran.\n\nKonsert berbayar\n\n\n\nMewakili golongan anak seni, Sheila menaruh harapan semoga Festival KL Jamm akan menjadi platform buat artis yang sudah ada nama dan artis muda untuk membuat persembahan, sekali gus sama-sama memartabatkan industri muzik tempatan.\n\nMenurut Sheila, dia juga mencadangkan lebih banyak tempat diwujudkan untuk menggalakkan artis muda membuat persembahan, sekali gus menggilap bakat mereka.\n\n“Berbanding pada zaman saya dulu, artis muda sekarang tidak banyak tempat khusus untuk mereka menyanyi dan menonjolkan bakat di tempat awam.\n\n“Rata-rata hanya sekadar menyanyi di laman Instagram dan cuma dikenali menerusi satu lagu. Justeru, bagaimana mereka mahu buat showcase kalau hanya dikenali dengan satu lagu?” katanya.\n\nPada masa sama, Sheila juga merayu peminat tempatan untuk sama-sama memberi sokongan pada penganjuran festival KL Jamm sekali gus mencapai objektifnya.\n\n“Peminat perlu ubah persepsi negatif mereka dengan menganggap persembahan artis tempatan tidak bagus.\n\n“Kemasukan artis luar juga perlu dilihat dari sudut yang positif kerana kita perlu belajar bagaimana untuk menjadi bagus seperti mereka,” katanya.\n\nSementara itu, Zainal pula berharap festival itu akan mendidik orang ramai untuk menonton konsert berbayar serta memberi sokongan pada artis tempatan.\n\n“Ramai yang hanya meminati artis tempatan tetapi tidak mahu mengeluarkan sedikit wang untuk membeli tiket konsert mereka.\n\n“Sedangkan artis juga menyanyi untuk kerjaya dan ia juga punca pendapatan bagi menyara hidup,” katanya.\n\nFestival KL Jamm bakal menghimpunkan barisan artis tempatan baru dan nama besar dalam konsert iaitu Datuk Ramli Sarip, Datuk Afdlin Shauki, Zamani, Amelina, Radhi OAG, Dr Burn, Santesh, Rabbit Mac, Sheezy, kumpulan Bunkface, Ruffedge, Pot Innuendo, artis dari Kartel (Joe Flizzow, Sona One, Ila Damia, Yung Raja, Faris Jabba dan Abu Bakarxli) dan Malaysia Pasangge (artis India tempatan).\n\nManakala, artis antarabangsa pula membabitkan J Arie (Hong Kong), NCT Dream (Korea Selatan) dan DJ Sura (Korea Selatan).\n\nKL Jamm dianjurkan Music Unlimited International Sdn Bhd dan bakal menggabungkan pelbagai genre muzik seperti rock, hip hop, jazz dan pop dengan lebih 100 persembahan, 20 ‘showcase’, pameran dan perdagangan berkaitan.\n\nFestival tiga hari itu bakal berlangsung di Pusat Pameran dan Perdagangan Antarabangsa Malaysia (MITEC), Kuala Lumpur pada 26 hingga 28 April ini.\n\nMaklumat mengenai pembelian tiket dan keterangan lanjut boleh melayari www.kljamm.com.'

We also can give a string, Malaya will always split a string by into
multiple sentences.

Load Pretrained News summarization deep learning
------------------------------------------------

.. code:: ipython3

    deep_summary = malaya.summarize.deep_model_news()

.. code:: ipython3

    deep_summary.summarize(isu_string,important_words=10)




.. parsed-literal::

    {'summary': 'Namun, ada satu persamaan yang mengeratkan hubungan mereka kerana sama-sama mencintai bidang muzik sejak dulu. "Kami pernah terbabit dengan showcase dan majlis korporat sebelum ini. "Sedangkan artis juga menyanyi untuk kerjaya dan ia juga punca pendapatan bagi menyara hidup," katanya.',
     'top-words': ['dumex',
      'unchallenged',
      'yussoffkaunsel',
      'sharmini',
      'merotan',
      'vienna',
      'pancaroba',
      'kepulangan',
      'mandat',
      'kelembaban'],
     'cluster-top-words': ['kelembaban',
      'merotan',
      'pancaroba',
      'yussoffkaunsel',
      'dumex',
      'unchallenged',
      'vienna',
      'mandat',
      'sharmini',
      'kepulangan']}



.. code:: ipython3

    deep_summary.summarize(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': '"Kedua UMNO sebagai sebuah parti sangat menghormati dan memahami keperluan sekolah vernakular di Malaysia. Kenyataan kontroversi Setiausaha Agung Barisan Nasional (BN), Datuk Seri Mohamed Nazri Aziz berhubung sekolah vernakular merupakan pandangan peribadi beliau. Pertama pendirian beliau tersebut adalah pandangan peribadi yang tidak mewakili pendirian dan pandangan UMNO.',
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
     'cluster-top-words': ['seliakekurangan',
      'bersabdabarangsiapa',
      'poupart',
      'chusus',
      'sakailah',
      'pembikin',
      'sharmini',
      'mempunya',
      'kepulangan',
      'diharap']}



You also can change sentences to vector representation using
``vectorize()``.

.. code:: ipython3

    deep_summary.vectorize(isu_kerajaan).shape




.. parsed-literal::

    (12, 128)



.. code:: ipython3

    deep_summary.vectorize(isu_kerajaan).shape




.. parsed-literal::

    (12, 128)



Load Pretrained Wikipedia summarization deep learning
-----------------------------------------------------

.. code:: ipython3

    deep_summary = malaya.summarize.deep_model_wiki()


.. parsed-literal::

    WARNING: this model is using convolutional based, Tensorflow-GPU above 1.10 may got a problem. Please downgrade to Tensorflow-GPU v1.8 if got any cuDNN error.


.. code:: ipython3

    deep_summary.summarize(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'Mohamed Nazri semalam menjelaskan, kenyataannya mengenai sekolah jenis kebangsaan Cina dan Tamil baru-baru ini disalah petik pihak media. "Kedua UMNO sebagai sebuah parti sangat menghormati dan memahami keperluan sekolah vernakular di Malaysia. "Saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar UMNO dan BN," katanya.',
     'top-words': ['jagaannya',
      'ferdy',
      'hoe',
      'zulkifli',
      'televisyen',
      'lanun',
      'laksmi',
      'ongr',
      'kawan',
      'diimbau'],
     'cluster-top-words': ['televisyen',
      'jagaannya',
      'diimbau',
      'zulkifli',
      'lanun',
      'laksmi',
      'kawan',
      'ongr',
      'hoe',
      'ferdy']}



You also can change sentences to vector representation using
``vectorize()``.

.. code:: ipython3

    deep_summary.vectorize(isu_kerajaan).shape




.. parsed-literal::

    (12, 64)



.. code:: ipython3

    deep_summary.vectorize(isu_kerajaan).shape




.. parsed-literal::

    (12, 64)



Train skip-thought summarization deep learning model
----------------------------------------------------

.. code:: ipython3

    deep_summary = malaya.summarize.train_skip_thought(isu_kerajaan, batch_size = 2)


.. parsed-literal::

    minibatch loop: 100%|██████████| 5/5 [00:01<00:00,  2.94it/s, cost=9.45]
    minibatch loop: 100%|██████████| 5/5 [00:01<00:00,  4.56it/s, cost=7.99]
    minibatch loop: 100%|██████████| 5/5 [00:01<00:00,  4.67it/s, cost=6.61]
    minibatch loop: 100%|██████████| 5/5 [00:01<00:00,  4.62it/s, cost=5.34]
    minibatch loop: 100%|██████████| 5/5 [00:01<00:00,  4.55it/s, cost=4.17]


.. code:: ipython3

    deep_summary.summarize(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'Pertama pendirian beliau tersebut adalah pandangan peribadi yang tidak mewakili pendirian dan pandangan UMNO. Kenyataan kontroversi Setiausaha Agung Barisan Nasional (BN), Datuk Seri Mohamed Nazri Aziz berhubung sekolah vernakular merupakan pandangan peribadi beliau. Kata beliau, komitmen UMNO dan BN berhubung perkara itu dapat dilihat dengan jelas dalam bentuk sokongan infrastruktur, pengiktirafan dan pemberian peruntukan yang diperlukan.',
     'top-words': ['vernakular',
      'bentuk',
      'parti',
      'jelas',
      'pertama',
      'disalah',
      'adalah',
      'kekuatan',
      'bahawa',
      'penting'],
     'cluster-top-words': ['adalah',
      'penting',
      'bentuk',
      'pertama',
      'bahawa',
      'parti',
      'disalah',
      'kekuatan',
      'jelas',
      'vernakular']}



Train LSA model
---------------

.. code:: ipython3

    malaya.summarize.lsa(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'Menurut beliau, persefahaman dan keupayaan meraikan kepelbagaian itu menjadi kelebihan dan kekuatan UMNO dan BN selama ini. Kata beliau, komitmen UMNO dan BN berhubung perkara itu dapat dilihat dengan jelas dalam bentuk sokongan infrastruktur, pengiktirafan dan pemberian peruntukan yang diperlukan. "Saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar UMNO dan BN," katanya.',
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
     'cluster-top-words': ['negara',
      'mohamad',
      'pandang peribadi',
      'wakil pandang umno',
      'mohamed nazri',
      'nazri nyata',
      'paham sekolah vernakular']}



.. code:: ipython3

    malaya.summarize.lsa(isu_string,important_words=10)




.. parsed-literal::

    {'summary': "KL Jamm dianjurkan Music Unlimited International Sdn Bhd dan bakal menggabungkan pelbagai genre muzik seperti rock, hip hop, jazz dan pop dengan lebih 100 persembahan, 20 'showcase', pameran dan perdagangan berkaitan. Festival tiga hari itu bakal berlangsung di Pusat Pameran dan Perdagangan Antarabangsa Malaysia (MITEC), Kuala Lumpur pada 26 hingga 28 April ini. Maklumat mengenai pembelian tiket dan keterangan lanjut boleh melayari www.kljamm.com.",
     'top-words': ['zaman',
      'jamm anjur',
      'genre muzik rock',
      'hip',
      'hip hop',
      'hip hop jazz',
      'hop',
      'hop jazz',
      'hop jazz pop',
      'jazz pop'],
     'cluster-top-words': ['hip hop jazz',
      'genre muzik rock',
      'hop jazz pop',
      'jamm anjur',
      'zaman']}



Train NMF model
---------------

.. code:: ipython3

    malaya.summarize.nmf(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'Menurut beliau, persefahaman dan keupayaan meraikan kepelbagaian itu menjadi kelebihan dan kekuatan UMNO dan BN selama ini. Kata beliau, komitmen UMNO dan BN berhubung perkara itu dapat dilihat dengan jelas dalam bentuk sokongan infrastruktur, pengiktirafan dan pemberian peruntukan yang diperlukan. "Saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar UMNO dan BN," katanya.',
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
     'cluster-top-words': ['negara',
      'mohamad',
      'pandang peribadi',
      'wakil pandang umno',
      'mohamed nazri',
      'nazri nyata',
      'paham sekolah vernakular']}



Train LDA model
---------------

.. code:: ipython3

    malaya.summarize.lda(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'Menurut beliau, persefahaman dan keupayaan meraikan kepelbagaian itu menjadi kelebihan dan kekuatan UMNO dan BN selama ini. Kata beliau, komitmen UMNO dan BN berhubung perkara itu dapat dilihat dengan jelas dalam bentuk sokongan infrastruktur, pengiktirafan dan pemberian peruntukan yang diperlukan. "Saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar UMNO dan BN," katanya.',
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
     'cluster-top-words': ['negara',
      'mohamad',
      'pandang peribadi',
      'wakil pandang umno',
      'mohamed nazri',
      'nazri nyata',
      'paham sekolah vernakular']}



Not clustering important words
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    malaya.summarize.lda(isu_kerajaan,important_words=10,return_cluster=False)




.. parsed-literal::

    {'summary': 'Menurut beliau, persefahaman dan keupayaan meraikan kepelbagaian itu menjadi kelebihan dan kekuatan UMNO dan BN selama ini. Kata beliau, komitmen UMNO dan BN berhubung perkara itu dapat dilihat dengan jelas dalam bentuk sokongan infrastruktur, pengiktirafan dan pemberian peruntukan yang diperlukan. "Saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar UMNO dan BN," katanya.',
     'top-words': ['wakil pandang umno',
      'mohamed',
      'paham sekolah vernakular',
      'paham sekolah',
      'paham',
      'negara',
      'nazri nyata',
      'mohamed nazri',
      'mohamad',
      'pandang peribadi']}


