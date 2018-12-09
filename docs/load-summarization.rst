
.. code:: python

    import malaya

.. code:: python

    isu_kerajaan = ['Institusi raja khususnya Yang di-Pertuan Agong adalah kedaulatan negara dengan kedudukan dan peranannya termaktub dalam Perlembagaan Persekutuan yang perlu disokong dan didukung oleh kerajaan serta rakyat.',
                   'Pensyarah Kulliyah Undang-Undang Ahmad Ibrahim, Universiti Islam Antarabangsa Malaysia (UIAM) Prof Madya Dr Shamrahayu Ab Aziz berkata perubahan kerajaan, susulan kemenangan Pakatan Harapan pada Pilihan Raya Umum Ke-14 pada Mei lepas, tidak memberi kesan dari segi peranan, fungsi dan kedudukan Yang di-Pertuan Agong.',
                   'Peralihan kerajaan itu menyaksikan Sultan Muhammad V mencatat sejarah tersendiri dengan menjadi Yang di-Pertuan Agong Malaysia yang pertama memerintah dalam era dua kerajaan berbeza.',
                   'Semasa dilantik sebagai Yang di-Pertuan Agong Ke-15 pada 13 Dis 2016, kerajaan ketika itu diterajui oleh Barisan Nasional dan pada 10 Mei lepas, kepimpinan negara diambil alih oleh Pakatan Harapan yang memenangi Pilihan Raya Umum Ke-14.',
                   'Ketika merasmikan Istiadat Pembukaan Penggal Pertama, Parlimen ke-14 pada 17 Julai lepas, Seri Paduka bertitah mengalu-alukan pendekatan kerajaan Pakatan Harapan dalam menegakkan ketelusan terutamanya dengan mendedahkan kedudukan kewangan negara yang sebenar serta mengkaji semula perbelanjaan, kos projek dan mengurus kewangan secara berhemat bagi menangani kos sara hidup.',
                   'Pada Jun lepas, Sultan Muhammad V memperkenankan supaya peruntukan gaji dan emolumen Yang di-Pertuan Agong dikurangkan sebanyak 10 peratus sepanjang pemerintahan sehingga 2021 berikutan keprihatinan Seri Paduka terhadap tahap hutang dan keadaan ekonomi negara.',
                   'Seri Paduka turut menitahkan supaya Majlis Rumah Terbuka Aidilfitri tahun ini tidak diadakan di Istana Negara dengan peruntukan majlis itu digunakan bagi membantu golongan yang kurang bernasib baik.']

Train LSA model
---------------

.. code:: python

    malaya.summarize_lsa(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'ketika merasmikan istiadat pembukaan penggal pertama parlimen julai lepas seri paduka bertitah mengalu alukan pendekatan kerajaan pakatan harapan menegakkan ketelusan terutamanya mendedahkan kedudukan kewangan negara sebenar mengkaji semula perbelanjaan kos projek mengurus kewangan secara berhemat menangani kos sara hidup. jun lepas sultan muhammad v memperkenankan peruntukan gaji emolumen pertuan agong dikurangkan sebanyak peratus sepanjang pemerintahan sehingga berikutan keprihatinan seri paduka tahap hutang keadaan ekonomi negara. seri paduka turut menitahkan majlis rumah terbuka aidilfitri tahun diadakan istana negara peruntukan majlis digunakan membantu golongan kurang bernasib baik',
     'top-words': ['umum',
      'pilih',
      'alih',
      'buka',
      'ketika',
      'kurang',
      'malaysia',
      'mei',
      'mei lepas',
      'muhammad'],
     'cluster-top-words': ['pilih',
      'kurang',
      'umum',
      'malaysia',
      'ketika',
      'mei lepas',
      'buka',
      'muhammad',
      'alih']}



Maintain original
^^^^^^^^^^^^^^^^^

.. code:: python

    malaya.summarize_lsa(isu_kerajaan, important_words=10,maintain_original=True)




.. parsed-literal::

    {'summary': 'ketika merasmikan istiadat pembukaan penggal pertama, parlimen ke-14 pada 17 julai lepas, seri paduka bertitah mengalu-alukan pendekatan kerajaan pakatan harapan dalam menegakkan ketelusan terutamanya dengan mendedahkan kedudukan kewangan negara yang sebenar serta mengkaji semula perbelanjaan, kos projek dan mengurus kewangan secara berhemat bagi menangani kos sara hidup. pada jun lepas, sultan muhammad v memperkenankan supaya peruntukan gaji dan emolumen yang di-pertuan agong dikurangkan sebanyak 10 peratus sepanjang pemerintahan sehingga 2021 berikutan keprihatinan seri paduka terhadap tahap hutang dan keadaan ekonomi negara. seri paduka turut menitahkan supaya majlis rumah terbuka aidilfitri tahun ini tidak diadakan di istana negara dengan peruntukan majlis itu digunakan bagi membantu golongan yang kurang bernasib baik',
     'top-words': ['umum 14',
      'pertama',
      'ada',
      'alih',
      'buka',
      'ketika',
      'kurang',
      'malaysia',
      'mei',
      'mei lepas'],
     'cluster-top-words': ['kurang',
      'pertama',
      'ada',
      'malaysia',
      'ketika',
      'umum 14',
      'mei lepas',
      'buka',
      'alih']}



Train NMF model
---------------

.. code:: python

    malaya.summarize_nmf(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'ketika merasmikan istiadat pembukaan penggal pertama parlimen julai lepas seri paduka bertitah mengalu alukan pendekatan kerajaan pakatan harapan menegakkan ketelusan terutamanya mendedahkan kedudukan kewangan negara sebenar mengkaji semula perbelanjaan kos projek mengurus kewangan secara berhemat menangani kos sara hidup. jun lepas sultan muhammad v memperkenankan peruntukan gaji emolumen pertuan agong dikurangkan sebanyak peratus sepanjang pemerintahan sehingga berikutan keprihatinan seri paduka tahap hutang keadaan ekonomi negara. seri paduka turut menitahkan majlis rumah terbuka aidilfitri tahun diadakan istana negara peruntukan majlis digunakan membantu golongan kurang bernasib baik',
     'top-words': ['umum',
      'pilih',
      'alih',
      'buka',
      'ketika',
      'kurang',
      'malaysia',
      'mei',
      'mei lepas',
      'muhammad'],
     'cluster-top-words': ['pilih',
      'kurang',
      'umum',
      'malaysia',
      'ketika',
      'mei lepas',
      'buka',
      'muhammad',
      'alih']}



Train LDA model
---------------

.. code:: python

    malaya.summarize_lda(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'ketika merasmikan istiadat pembukaan penggal pertama parlimen julai lepas seri paduka bertitah mengalu alukan pendekatan kerajaan pakatan harapan menegakkan ketelusan terutamanya mendedahkan kedudukan kewangan negara sebenar mengkaji semula perbelanjaan kos projek mengurus kewangan secara berhemat menangani kos sara hidup. jun lepas sultan muhammad v memperkenankan peruntukan gaji emolumen pertuan agong dikurangkan sebanyak peratus sepanjang pemerintahan sehingga berikutan keprihatinan seri paduka tahap hutang keadaan ekonomi negara. seri paduka turut menitahkan majlis rumah terbuka aidilfitri tahun diadakan istana negara peruntukan majlis digunakan membantu golongan kurang bernasib baik',
     'top-words': ['umum',
      'pilih',
      'alih',
      'buka',
      'ketika',
      'kurang',
      'malaysia',
      'mei',
      'mei lepas',
      'muhammad'],
     'cluster-top-words': ['pilih',
      'kurang',
      'umum',
      'malaysia',
      'ketika',
      'mei lepas',
      'buka',
      'muhammad',
      'alih']}



Not clustering important words
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    malaya.summarize_lda(isu_kerajaan,important_words=10,return_cluster=False)




.. parsed-literal::

    {'summary': 'ketika merasmikan istiadat pembukaan penggal pertama parlimen julai lepas seri paduka bertitah mengalu alukan pendekatan kerajaan pakatan harapan menegakkan ketelusan terutamanya mendedahkan kedudukan kewangan negara sebenar mengkaji semula perbelanjaan kos projek mengurus kewangan secara berhemat menangani kos sara hidup. jun lepas sultan muhammad v memperkenankan peruntukan gaji emolumen pertuan agong dikurangkan sebanyak peratus sepanjang pemerintahan sehingga berikutan keprihatinan seri paduka tahap hutang keadaan ekonomi negara. seri paduka turut menitahkan majlis rumah terbuka aidilfitri tahun diadakan istana negara peruntukan majlis digunakan membantu golongan kurang bernasib baik',
     'top-words': ['umum',
      'pilih',
      'alih',
      'buka',
      'ketika',
      'kurang',
      'malaysia',
      'mei',
      'mei lepas',
      'muhammad']}



Load deep learning model
------------------------

.. code:: python

    deep_summary = malaya.summarize_deep_learning()


.. parsed-literal::

    downloading SUMMARIZE skip-thought frozen model


.. parsed-literal::

    119MB [00:39, 3.88MB/s]
      0%|          | 0.00/0.98 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading SUMMARIZE skip-thought dictionary


.. parsed-literal::

    1.00MB [00:00, 2.39MB/s]


.. code:: python

    deep_summary.summarize(isu_kerajaan)




.. parsed-literal::

    'peralihan kerajaan itu menyaksikan sultan muhammad v mencatat sejarah tersendiri dengan menjadi yang di-pertuan agong malaysia yang pertama memerintah dalam era dua kerajaan berbeza. semasa dilantik sebagai yang di-pertuan agong ke-15 pada 13 dis 2016, kerajaan ketika itu diterajui oleh barisan nasional dan pada 10 mei lepas, kepimpinan negara diambil alih oleh pakatan harapan yang memenangi pilihan raya umum ke-14. seri paduka turut menitahkan supaya majlis rumah terbuka aidilfitri tahun ini tidak diadakan di istana negara dengan peruntukan majlis itu digunakan bagi membantu golongan yang kurang bernasib baik'
