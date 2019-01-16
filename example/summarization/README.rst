
.. code:: ipython3

    import malaya

.. code:: ipython3

    isu_kerajaan = ['Institusi raja khususnya Yang di-Pertuan Agong adalah kedaulatan negara dengan kedudukan dan peranannya termaktub dalam Perlembagaan Persekutuan yang perlu disokong dan didukung oleh kerajaan serta rakyat.',
                   'Pensyarah Kulliyah Undang-Undang Ahmad Ibrahim, Universiti Islam Antarabangsa Malaysia (UIAM) Prof Madya Dr Shamrahayu Ab Aziz berkata perubahan kerajaan, susulan kemenangan Pakatan Harapan pada Pilihan Raya Umum Ke-14 pada Mei lepas, tidak memberi kesan dari segi peranan, fungsi dan kedudukan Yang di-Pertuan Agong.',
                   'Peralihan kerajaan itu menyaksikan Sultan Muhammad V mencatat sejarah tersendiri dengan menjadi Yang di-Pertuan Agong Malaysia yang pertama memerintah dalam era dua kerajaan berbeza.',
                   'Semasa dilantik sebagai Yang di-Pertuan Agong Ke-15 pada 13 Dis 2016, kerajaan ketika itu diterajui oleh Barisan Nasional dan pada 10 Mei lepas, kepimpinan negara diambil alih oleh Pakatan Harapan yang memenangi Pilihan Raya Umum Ke-14.',
                   'Ketika merasmikan Istiadat Pembukaan Penggal Pertama, Parlimen ke-14 pada 17 Julai lepas, Seri Paduka bertitah mengalu-alukan pendekatan kerajaan Pakatan Harapan dalam menegakkan ketelusan terutamanya dengan mendedahkan kedudukan kewangan negara yang sebenar serta mengkaji semula perbelanjaan, kos projek dan mengurus kewangan secara berhemat bagi menangani kos sara hidup.',
                   'Pada Jun lepas, Sultan Muhammad V memperkenankan supaya peruntukan gaji dan emolumen Yang di-Pertuan Agong dikurangkan sebanyak 10 peratus sepanjang pemerintahan sehingga 2021 berikutan keprihatinan Seri Paduka terhadap tahap hutang dan keadaan ekonomi negara.',
                   'Seri Paduka turut menitahkan supaya Majlis Rumah Terbuka Aidilfitri tahun ini tidak diadakan di Istana Negara dengan peruntukan majlis itu digunakan bagi membantu golongan yang kurang bernasib baik.']

Load Pretrained News summarization deep learning
------------------------------------------------

.. code:: ipython3

    deep_summary = malaya.summarize.deep_model_news()

.. code:: ipython3

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
      'ibubapanya']}



Load Pretrained Wikipedia summarization deep learning
-----------------------------------------------------

.. code:: ipython3

    deep_summary = malaya.summarize.deep_model_wiki()

.. code:: ipython3

    deep_summary.summarize(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'pensyarah kulliyah undang-undang ahmad ibrahim, universiti islam antarabangsa malaysia uiam prof madya dr shamrahayu ab aziz berkata perubahan kerajaan, susulan kemenangan pakatan harapan pada pilihan raya umum ke-14 pada mei lepas, tidak memberi kesan dari segi peranan, fungsi dan kedudukan yang di-pertuan agong. institusi raja khususnya yang di-pertuan agong adalah kedaulatan negara dengan kedudukan dan peranannya termaktub dalam perlembagaan persekutuan yang perlu disokong dan didukung oleh kerajaan serta rakyat. ketika merasmikan istiadat pembukaan penggal pertama, parlimen ke-14 pada 17 julai lepas, seri paduka bertitah mengalu-alukan pendekatan kerajaan pakatan harapan dalam menegakkan ketelusan terutamanya dengan mendedahkan kedudukan kewangan negara yang sebenar serta mengkaji semula perbelanjaan, kos projek dan mengurus kewangan secara berhemat bagi menangani kos sara hidup',
     'top-words': ['jagaannya',
      'ferdy',
      'sharidake',
      'televisyen',
      'zulkifli',
      'hoe',
      'luteum',
      'diimbau',
      'kawan',
      'brunei']}



Train skip-thought summarization deep learning model
----------------------------------------------------

.. code:: ipython3

    deep_summary = malaya.summarize.train_skip_thought(isu_kerajaan, batch_size = 2)


.. parsed-literal::

    minibatch loop: 100%|██████████| 3/3 [00:00<00:00,  2.79it/s, cost=9.4] 
    minibatch loop: 100%|██████████| 3/3 [00:00<00:00,  3.83it/s, cost=6.9] 
    minibatch loop: 100%|██████████| 3/3 [00:01<00:00,  2.80it/s, cost=5.64]
    minibatch loop: 100%|██████████| 3/3 [00:00<00:00,  3.01it/s, cost=4.43]
    minibatch loop: 100%|██████████| 3/3 [00:00<00:00,  3.38it/s, cost=3.54]


.. code:: ipython3

    deep_summary.summarize(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'seri paduka turut menitahkan supaya majlis rumah terbuka aidilfitri tahun ini tidak diadakan di istana negara dengan peruntukan majlis itu digunakan bagi membantu golongan yang kurang bernasib baik. peralihan kerajaan itu menyaksikan sultan muhammad v mencatat sejarah tersendiri dengan menjadi yang di-pertuan agong malaysia yang pertama memerintah dalam era dua kerajaan berbeza. pada jun lepas, sultan muhammad v memperkenankan supaya peruntukan gaji dan emolumen yang di-pertuan agong dikurangkan sebanyak 10 peratus sepanjang pemerintahan sehingga 2021 berikutan keprihatinan seri paduka terhadap tahap hutang dan keadaan ekonomi negara',
     'top-words': ['oleh',
      'mencatat',
      ',',
      'tahap',
      'berikutan',
      'harapan',
      'projek',
      'ke',
      'madya',
      'alukan']}



Train LSA model
---------------

.. code:: ipython3

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
      'malaysia',
      'perintah',
      'alih',
      'sultan muhammad',
      'paduka titah',
      'mei',
      'peran']}



Maintain original
^^^^^^^^^^^^^^^^^

.. code:: ipython3

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
      'malaysia',
      'pilih',
      'alih',
      'sultan muhammad',
      'paduka titah',
      'mei',
      'peran']}



Train NMF model
---------------

.. code:: ipython3

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
      'malaysia',
      'perintah',
      'alih',
      'sultan muhammad',
      'paduka titah',
      'mei',
      'peran']}



Train LDA model
---------------

.. code:: ipython3

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
      'malaysia',
      'perintah',
      'alih',
      'sultan muhammad',
      'paduka titah',
      'mei',
      'peran']}



Not clustering important words
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

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


