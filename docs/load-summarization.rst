
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.6 s, sys: 1.86 s, total: 14.4 s
    Wall time: 20.6 s


.. code:: python

    isu_kerajaan = ['Institusi raja khususnya Yang di-Pertuan Agong adalah kedaulatan negara dengan kedudukan dan peranannya termaktub dalam Perlembagaan Persekutuan yang perlu disokong dan didukung oleh kerajaan serta rakyat.',
                   'Pensyarah Kulliyah Undang-Undang Ahmad Ibrahim, Universiti Islam Antarabangsa Malaysia (UIAM) Prof Madya Dr Shamrahayu Ab Aziz berkata perubahan kerajaan, susulan kemenangan Pakatan Harapan pada Pilihan Raya Umum Ke-14 pada Mei lepas, tidak memberi kesan dari segi peranan, fungsi dan kedudukan Yang di-Pertuan Agong.',
                   'Peralihan kerajaan itu menyaksikan Sultan Muhammad V mencatat sejarah tersendiri dengan menjadi Yang di-Pertuan Agong Malaysia yang pertama memerintah dalam era dua kerajaan berbeza.',
                   'Semasa dilantik sebagai Yang di-Pertuan Agong Ke-15 pada 13 Dis 2016, kerajaan ketika itu diterajui oleh Barisan Nasional dan pada 10 Mei lepas, kepimpinan negara diambil alih oleh Pakatan Harapan yang memenangi Pilihan Raya Umum Ke-14.',
                   'Ketika merasmikan Istiadat Pembukaan Penggal Pertama, Parlimen ke-14 pada 17 Julai lepas, Seri Paduka bertitah mengalu-alukan pendekatan kerajaan Pakatan Harapan dalam menegakkan ketelusan terutamanya dengan mendedahkan kedudukan kewangan negara yang sebenar serta mengkaji semula perbelanjaan, kos projek dan mengurus kewangan secara berhemat bagi menangani kos sara hidup.',
                   'Pada Jun lepas, Sultan Muhammad V memperkenankan supaya peruntukan gaji dan emolumen Yang di-Pertuan Agong dikurangkan sebanyak 10 peratus sepanjang pemerintahan sehingga 2021 berikutan keprihatinan Seri Paduka terhadap tahap hutang dan keadaan ekonomi negara.',
                   'Seri Paduka turut menitahkan supaya Majlis Rumah Terbuka Aidilfitri tahun ini tidak diadakan di Istana Negara dengan peruntukan majlis itu digunakan bagi membantu golongan yang kurang bernasib baik.']

Load Pretrained News summarization deep learning
------------------------------------------------

.. code:: python

    deep_summary = malaya.summarize.deep_model_news()

.. code:: python

    deep_summary.summarize(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'pensyarah kulliyah undang-undang ahmad ibrahim, universiti islam antarabangsa malaysia uiam prof madya dr shamrahayu ab aziz berkata perubahan kerajaan, susulan kemenangan pakatan harapan pada pilihan raya umum ke-14 pada mei lepas, tidak memberi kesan dari segi peranan, fungsi dan kedudukan yang di-pertuan agong. semasa dilantik sebagai yang di-pertuan agong ke-15 pada 13 dis 2016, kerajaan ketika itu diterajui oleh barisan nasional dan pada 10 mei lepas, kepimpinan negara diambil alih oleh pakatan harapan yang memenangi pilihan raya umum ke-14. ketika merasmikan istiadat pembukaan penggal pertama, parlimen ke-14 pada 17 julai lepas, seri paduka bertitah mengalu-alukan pendekatan kerajaan pakatan harapan dalam menegakkan ketelusan terutamanya dengan mendedahkan kedudukan kewangan negara yang sebenar serta mengkaji semula perbelanjaan, kos projek dan mengurus kewangan secara berhemat bagi menangani kos sara hidup',
     'top-words': ['sharmini',
      'bielefeld',
      'taksi',
      'diharap',
      'unchallenged',
      'bkkm',
      'chusus',
      'menjebaknya',
      'diiringi',
      'ibubapanya'],
     'cluster-top-words': ['bielefeld',
      'sharmini',
      'diiringi',
      'unchallenged',
      'menjebaknya',
      'taksi',
      'diharap',
      'bkkm',
      'ibubapanya',
      'chusus']}



You also can change sentences to vector representation using
``vectorize()``.

.. code:: python

    deep_summary.vectorize(isu_kerajaan).shape




.. parsed-literal::

    (7, 128)



Load Pretrained Wikipedia summarization deep learning
-----------------------------------------------------

.. code:: python

    deep_summary = malaya.summarize.deep_model_wiki()


.. parsed-literal::

    WARNING: this model is using convolutional based, Tensorflow-GPU above 1.10 may got a problem. Please downgrade to Tensorflow-GPU v1.8 if got any cuDNN error.


.. code:: python

    deep_summary.summarize(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'peralihan kerajaan itu menyaksikan sultan muhammad v mencatat sejarah tersendiri dengan menjadi yang di-pertuan agong malaysia yang pertama memerintah dalam era dua kerajaan berbeza. pensyarah kulliyah undang-undang ahmad ibrahim, universiti islam antarabangsa malaysia uiam prof madya dr shamrahayu ab aziz berkata perubahan kerajaan, susulan kemenangan pakatan harapan pada pilihan raya umum ke-14 pada mei lepas, tidak memberi kesan dari segi peranan, fungsi dan kedudukan yang di-pertuan agong. pada jun lepas, sultan muhammad v memperkenankan supaya peruntukan gaji dan emolumen yang di-pertuan agong dikurangkan sebanyak 10 peratus sepanjang pemerintahan sehingga 2021 berikutan keprihatinan seri paduka terhadap tahap hutang dan keadaan ekonomi negara',
     'top-words': ['jagaannya',
      'ferdy',
      'sharidake',
      'televisyen',
      'zulkifli',
      'hoe',
      'luteum',
      'kawan',
      'diimbau',
      'brunei'],
     'cluster-top-words': ['luteum',
      'kawan',
      'brunei',
      'jagaannya',
      'sharidake',
      'televisyen',
      'zulkifli',
      'hoe',
      'diimbau',
      'ferdy']}



You also can change sentences to vector representation using
``vectorize()``.

.. code:: python

    deep_summary.vectorize(isu_kerajaan).shape




.. parsed-literal::

    (7, 64)



Train skip-thought summarization deep learning model
----------------------------------------------------

.. code:: python

    deep_summary = malaya.summarize.train_skip_thought(isu_kerajaan, batch_size = 2)


.. parsed-literal::

    minibatch loop: 100%|██████████| 3/3 [00:00<00:00,  2.68it/s, cost=9.68]
    minibatch loop: 100%|██████████| 3/3 [00:00<00:00,  4.15it/s, cost=6.97]
    minibatch loop: 100%|██████████| 3/3 [00:00<00:00,  4.05it/s, cost=6.22]
    minibatch loop: 100%|██████████| 3/3 [00:00<00:00,  3.89it/s, cost=5.27]
    minibatch loop: 100%|██████████| 3/3 [00:00<00:00,  4.07it/s, cost=4.49]


.. code:: python

    deep_summary.summarize(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'peralihan kerajaan itu menyaksikan sultan muhammad v mencatat sejarah tersendiri dengan menjadi yang di-pertuan agong malaysia yang pertama memerintah dalam era dua kerajaan berbeza. seri paduka turut menitahkan supaya majlis rumah terbuka aidilfitri tahun ini tidak diadakan di istana negara dengan peruntukan majlis itu digunakan bagi membantu golongan yang kurang bernasib baik. ketika merasmikan istiadat pembukaan penggal pertama, parlimen ke-14 pada 17 julai lepas, seri paduka bertitah mengalu-alukan pendekatan kerajaan pakatan harapan dalam menegakkan ketelusan terutamanya dengan mendedahkan kedudukan kewangan negara yang sebenar serta mengkaji semula perbelanjaan, kos projek dan mengurus kewangan secara berhemat bagi menangani kos sara hidup',
     'top-words': ['pilihan',
      'mengurus',
      'keprihatinan',
      '10',
      'undang',
      'UNK',
      'antarabangsa',
      'merasmikan',
      '14',
      'itu'],
     'cluster-top-words': ['pilihan',
      'merasmikan',
      'undang',
      'mengurus',
      'UNK',
      'itu',
      '14',
      '10',
      'antarabangsa',
      'keprihatinan']}



Train LSA model
---------------

.. code:: python

    malaya.summarize.lsa(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'merasmikan istiadat pembukaan penggal parlimen julai seri paduka bertitah mengalu alukan pendekatan kerajaan pakatan harapan menegakkan ketelusan terutamanya mendedahkan kedudukan kewangan negara sebenar mengkaji perbelanjaan kos projek mengurus kewangan berhemat menangani kos sara hidup. jun sultan muhammad v memperkenankan peruntukan gaji emolumen pertuan agong dikurangkan peratus pemerintahan berikutan keprihatinan seri paduka tahap hutang ekonomi negara. seri paduka menitahkan majlis rumah terbuka aidilfitri diadakan istana negara peruntukan majlis membantu golongan bernasib',
     'top-words': ['titah',
      'perintah',
      'alih',
      'buka',
      'malaysia',
      'mei',
      'muhammad',
      'paduka titah',
      'sultan muhammad',
      'peran'],
     'cluster-top-words': ['buka',
      'peran',
      'perintah',
      'mei',
      'sultan muhammad',
      'alih',
      'malaysia',
      'paduka titah']}



Maintain original
^^^^^^^^^^^^^^^^^

.. code:: python

    malaya.summarize.lsa(isu_kerajaan, important_words=10,maintain_original=True)




.. parsed-literal::

    {'summary': 'ketika merasmikan istiadat pembukaan penggal pertama, parlimen ke-14 pada 17 julai lepas, seri paduka bertitah mengalu-alukan pendekatan kerajaan pakatan harapan dalam menegakkan ketelusan terutamanya dengan mendedahkan kedudukan kewangan negara yang sebenar serta mengkaji semula perbelanjaan, kos projek dan mengurus kewangan secara berhemat bagi menangani kos sara hidup. pada jun lepas, sultan muhammad v memperkenankan supaya peruntukan gaji dan emolumen yang di-pertuan agong dikurangkan sebanyak 10 peratus sepanjang pemerintahan sehingga 2021 berikutan keprihatinan seri paduka terhadap tahap hutang dan keadaan ekonomi negara. seri paduka turut menitahkan supaya majlis rumah terbuka aidilfitri tahun ini tidak diadakan di istana negara dengan peruntukan majlis itu digunakan bagi membantu golongan yang kurang bernasib baik',
     'top-words': ['titah',
      'pilih',
      'alih',
      'buka',
      'malaysia',
      'mei',
      'muhammad',
      'paduka titah',
      'peran',
      'sultan muhammad'],
     'cluster-top-words': ['buka',
      'pilih',
      'peran',
      'mei',
      'sultan muhammad',
      'alih',
      'malaysia',
      'paduka titah']}



Train NMF model
---------------

.. code:: python

    malaya.summarize.nmf(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'merasmikan istiadat pembukaan penggal parlimen julai seri paduka bertitah mengalu alukan pendekatan kerajaan pakatan harapan menegakkan ketelusan terutamanya mendedahkan kedudukan kewangan negara sebenar mengkaji perbelanjaan kos projek mengurus kewangan berhemat menangani kos sara hidup. jun sultan muhammad v memperkenankan peruntukan gaji emolumen pertuan agong dikurangkan peratus pemerintahan berikutan keprihatinan seri paduka tahap hutang ekonomi negara. seri paduka menitahkan majlis rumah terbuka aidilfitri diadakan istana negara peruntukan majlis membantu golongan bernasib',
     'top-words': ['titah',
      'perintah',
      'alih',
      'buka',
      'malaysia',
      'mei',
      'muhammad',
      'paduka titah',
      'sultan muhammad',
      'peran'],
     'cluster-top-words': ['buka',
      'peran',
      'perintah',
      'mei',
      'sultan muhammad',
      'alih',
      'malaysia',
      'paduka titah']}



Train LDA model
---------------

.. code:: python

    malaya.summarize.lda(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'merasmikan istiadat pembukaan penggal parlimen julai seri paduka bertitah mengalu alukan pendekatan kerajaan pakatan harapan menegakkan ketelusan terutamanya mendedahkan kedudukan kewangan negara sebenar mengkaji perbelanjaan kos projek mengurus kewangan berhemat menangani kos sara hidup. jun sultan muhammad v memperkenankan peruntukan gaji emolumen pertuan agong dikurangkan peratus pemerintahan berikutan keprihatinan seri paduka tahap hutang ekonomi negara. seri paduka menitahkan majlis rumah terbuka aidilfitri diadakan istana negara peruntukan majlis membantu golongan bernasib',
     'top-words': ['titah',
      'perintah',
      'alih',
      'buka',
      'malaysia',
      'mei',
      'muhammad',
      'paduka titah',
      'sultan muhammad',
      'peran'],
     'cluster-top-words': ['buka',
      'peran',
      'perintah',
      'mei',
      'sultan muhammad',
      'alih',
      'malaysia',
      'paduka titah']}



Not clustering important words
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    malaya.summarize.lda(isu_kerajaan,important_words=10,return_cluster=False)




.. parsed-literal::

    {'summary': 'merasmikan istiadat pembukaan penggal parlimen julai seri paduka bertitah mengalu alukan pendekatan kerajaan pakatan harapan menegakkan ketelusan terutamanya mendedahkan kedudukan kewangan negara sebenar mengkaji perbelanjaan kos projek mengurus kewangan berhemat menangani kos sara hidup. jun sultan muhammad v memperkenankan peruntukan gaji emolumen pertuan agong dikurangkan peratus pemerintahan berikutan keprihatinan seri paduka tahap hutang ekonomi negara. seri paduka menitahkan majlis rumah terbuka aidilfitri diadakan istana negara peruntukan majlis membantu golongan bernasib',
     'top-words': ['titah',
      'perintah',
      'alih',
      'buka',
      'malaysia',
      'mei',
      'muhammad',
      'paduka titah',
      'sultan muhammad',
      'peran']}
