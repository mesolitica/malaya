
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12 s, sys: 1.61 s, total: 13.6 s
    Wall time: 17.9 s


.. code:: ipython3

    isu_kerajaan = [
        'Kenyataan kontroversi Setiausaha Agung Barisan Nasional (BN), Datuk Seri Mohamed Nazri Aziz berhubung sekolah vernakular merupakan pandangan peribadi beliau',
        'Timbalan Presiden UMNO, Datuk Seri Mohamad Hasan berkata, kenyataan tersebut tidak mewakili pendirian serta pandangan UMNO kerana parti itu menghormati serta memahami keperluan sekolah vernakular dalam negara',
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

We also can give a string, but remember, Malaya will always split a
string by ``.`` for summarization task.

.. code:: ipython3

    isu_kerajaan_combined = '. '.join(isu_kerajaan)

Load Pretrained News summarization deep learning
------------------------------------------------

.. code:: ipython3

    deep_summary = malaya.summarize.deep_model_news()

.. code:: ipython3

    deep_summary.summarize(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'timbalan presiden umno, datuk seri mohamad hasan berkata, kenyataan tersebut tidak mewakili pendirian serta pandangan umno kerana parti itu menghormati serta memahami keperluan sekolah vernakular dalam negara. menurut beliau, persefahaman dan keupayaan meraikan kepelbagaian itu menjadi kelebihan dan kekuatan umno dan bn selama ini. umno berpendirian sekolah jenis ini perlu terus wujud di negara kita," katanya dalam satu kenyataan akhbar malam ini',
     'top-words': ['bersabdabarangsiapa',
      'pembikin',
      'sharmini',
      'kepulangan',
      'sakailah',
      'klon',
      'seliakekurangan',
      'poupart',
      'chusus',
      'mempunya'],
     'cluster-top-words': ['chusus',
      'seliakekurangan',
      'pembikin',
      'sakailah',
      'kepulangan',
      'klon',
      'poupart',
      'mempunya',
      'sharmini',
      'bersabdabarangsiapa']}



.. code:: ipython3

    deep_summary.summarize(isu_kerajaan_combined,important_words=10)




.. parsed-literal::

    {'summary': 'timbalan presiden umno, datuk seri mohamad hasan berkata, kenyataan tersebut tidak mewakili pendirian serta pandangan umno kerana parti itu menghormati serta memahami keperluan sekolah vernakular dalam negara. menurut beliau, persefahaman dan keupayaan meraikan kepelbagaian itu menjadi kelebihan dan kekuatan umno dan bn selama ini. umno berpendirian sekolah jenis ini perlu terus wujud di negara kita," katanya dalam satu kenyataan akhbar malam ini',
     'top-words': ['bersabdabarangsiapa',
      'pembikin',
      'sharmini',
      'kepulangan',
      'sakailah',
      'klon',
      'seliakekurangan',
      'poupart',
      'chusus',
      'mempunya'],
     'cluster-top-words': ['chusus',
      'seliakekurangan',
      'pembikin',
      'sakailah',
      'kepulangan',
      'klon',
      'poupart',
      'mempunya',
      'sharmini',
      'bersabdabarangsiapa']}



You also can change sentences to vector representation using
``vectorize()``.

.. code:: ipython3

    deep_summary.vectorize(isu_kerajaan_combined).shape




.. parsed-literal::

    (13, 128)



.. code:: ipython3

    deep_summary.vectorize(isu_kerajaan).shape




.. parsed-literal::

    (13, 128)



Load Pretrained Wikipedia summarization deep learning
-----------------------------------------------------

.. code:: ipython3

    deep_summary = malaya.summarize.deep_model_wiki()


.. parsed-literal::

    WARNING: this model is using convolutional based, Tensorflow-GPU above 1.10 may got a problem. Please downgrade to Tensorflow-GPU v1.8 if got any cuDNN error.


.. code:: ipython3

    deep_summary.summarize(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': '"saya ingin menegaskan dua perkara penting. umno berpendirian sekolah jenis ini perlu terus wujud di negara kita," katanya dalam satu kenyataan akhbar malam ini. "saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar umno dan bn," katanya',
     'top-words': ['jagaannya',
      'ferdy',
      'hoe',
      'lanun',
      'laksmi',
      'zulkifli',
      'televisyen',
      'ongr',
      'kawan',
      'sharidake'],
     'cluster-top-words': ['lanun',
      'hoe',
      'televisyen',
      'jagaannya',
      'ongr',
      'laksmi',
      'zulkifli',
      'kawan',
      'ferdy',
      'sharidake']}



You also can change sentences to vector representation using
``vectorize()``.

.. code:: ipython3

    deep_summary.vectorize(isu_kerajaan).shape




.. parsed-literal::

    (13, 64)



.. code:: ipython3

    deep_summary.vectorize(isu_kerajaan_combined).shape




.. parsed-literal::

    (13, 64)



Train skip-thought summarization deep learning model
----------------------------------------------------

.. code:: ipython3

    deep_summary = malaya.summarize.train_skip_thought(isu_kerajaan, batch_size = 2)


.. parsed-literal::

    minibatch loop: 100%|██████████| 6/6 [00:01<00:00,  3.79it/s, cost=9.18]
    minibatch loop: 100%|██████████| 6/6 [00:01<00:00,  5.26it/s, cost=6.16]
    minibatch loop: 100%|██████████| 6/6 [00:01<00:00,  5.02it/s, cost=4.82]
    minibatch loop: 100%|██████████| 6/6 [00:01<00:00,  5.14it/s, cost=3.59]
    minibatch loop: 100%|██████████| 6/6 [00:01<00:00,  5.28it/s, cost=2.63]


.. code:: ipython3

    deep_summary.summarize(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'pertama pendirian beliau tersebut adalah pandangan peribadi yang tidak mewakili pendirian dan pandangan umno. "kedua umno sebagai sebuah parti sangat menghormati dan memahami keperluan sekolah vernakular di malaysia. kenyataan kontroversi setiausaha agung barisan nasional bn , datuk seri mohamed nazri aziz berhubung sekolah vernakular merupakan pandangan peribadi beliau',
     'top-words': ['persefahaman',
      'kekuatan',
      'pendirian',
      'isu',
      'tugas',
      'presiden',
      'menjalankan',
      'sokongan',
      'bentuk',
      'tidak'],
     'cluster-top-words': ['persefahaman',
      'sokongan',
      'bentuk',
      'tidak',
      'isu',
      'menjalankan',
      'presiden',
      'pendirian',
      'kekuatan',
      'tugas']}



Train LSA model
---------------

.. code:: ipython3

    malaya.summarize.lsa(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'komitmen umno berhubung bentuk sokongan infrastruktur pengiktirafan pemberian peruntukan. berharap isu dipolitikkan bertanggungjawab menggambarkan pendirian sebenar umno. mohamed nazri mengambil pertanggungjawaban penjelasan maksud ucapanny semenyih selangor',
     'top-words': ['wakil pandang umno',
      'parti',
      'hak',
      'hormat paham',
      'hormat paham sekolah',
      'hubung',
      'iktiraf',
      'jenis',
      'mohamad',
      'nazri nyata'],
     'cluster-top-words': ['wakil pandang umno',
      'nazri nyata',
      'parti',
      'mohamad',
      'hak',
      'jenis',
      'hubung',
      'hormat paham sekolah',
      'iktiraf']}



.. code:: ipython3

    malaya.summarize.lsa(isu_kerajaan_combined,important_words=10)




.. parsed-literal::

    {'summary': 'komitmen umno berhubung bentuk sokongan infrastruktur pengiktirafan pemberian peruntukan. berharap isu dipolitikkan bertanggungjawab menggambarkan pendirian sebenar umno. mohamed nazri mengambil pertanggungjawaban penjelasan maksud ucapanny semenyih selangor',
     'top-words': ['wakil pandang umno',
      'parti',
      'hak',
      'hormat paham',
      'hormat paham sekolah',
      'hubung',
      'iktiraf',
      'jenis',
      'mohamad',
      'nazri nyata'],
     'cluster-top-words': ['wakil pandang umno',
      'nazri nyata',
      'parti',
      'mohamad',
      'hak',
      'jenis',
      'hubung',
      'hormat paham sekolah',
      'iktiraf']}



Maintain original
^^^^^^^^^^^^^^^^^

.. code:: ipython3

    malaya.summarize.lsa(isu_kerajaan, important_words=10,maintain_original=True)




.. parsed-literal::

    {'summary': 'kata beliau, komitmen umno dan bn berhubung perkara itu dapat dilihat dengan jelas dalam bentuk sokongan infrastruktur, pengiktirafan dan pemberian peruntukan yang diperlukan. "saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar umno dan bn," katanya. beliau turut menegaskan mohamed nazri telah mengambil pertanggungjawaban dengan membuat penjelasan maksud sebenarnya ucapanny di semenyih, selangor tersebut',
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
     'cluster-top-words': ['wakil pandang umno',
      'nazri nyata',
      'mohamad',
      'hak',
      'jenis',
      'hubung',
      'hormat paham sekolah',
      'iktiraf']}



.. code:: ipython3

    malaya.summarize.lsa(isu_kerajaan_combined, important_words=10,maintain_original=True)




.. parsed-literal::

    {'summary': 'kata beliau, komitmen umno dan bn berhubung perkara itu dapat dilihat dengan jelas dalam bentuk sokongan infrastruktur, pengiktirafan dan pemberian peruntukan yang diperlukan. "saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar umno dan bn," katanya. beliau turut menegaskan mohamed nazri telah mengambil pertanggungjawaban dengan membuat penjelasan maksud sebenarnya ucapanny di semenyih, selangor tersebut',
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
     'cluster-top-words': ['wakil pandang umno',
      'nazri nyata',
      'mohamad',
      'hak',
      'jenis',
      'hubung',
      'hormat paham sekolah',
      'iktiraf']}



Train NMF model
---------------

.. code:: ipython3

    malaya.summarize.nmf(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'komitmen umno berhubung bentuk sokongan infrastruktur pengiktirafan pemberian peruntukan. berharap isu dipolitikkan bertanggungjawab menggambarkan pendirian sebenar umno. mohamed nazri mengambil pertanggungjawaban penjelasan maksud ucapanny semenyih selangor',
     'top-words': ['wakil pandang umno',
      'parti',
      'hak',
      'hormat paham',
      'hormat paham sekolah',
      'hubung',
      'iktiraf',
      'jenis',
      'mohamad',
      'nazri nyata'],
     'cluster-top-words': ['wakil pandang umno',
      'nazri nyata',
      'parti',
      'mohamad',
      'hak',
      'jenis',
      'hubung',
      'hormat paham sekolah',
      'iktiraf']}



Train LDA model
---------------

.. code:: ipython3

    malaya.summarize.lda(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': 'komitmen umno berhubung bentuk sokongan infrastruktur pengiktirafan pemberian peruntukan. berharap isu dipolitikkan bertanggungjawab menggambarkan pendirian sebenar umno. mohamed nazri mengambil pertanggungjawaban penjelasan maksud ucapanny semenyih selangor',
     'top-words': ['wakil pandang umno',
      'parti',
      'hak',
      'hormat paham',
      'hormat paham sekolah',
      'hubung',
      'iktiraf',
      'jenis',
      'mohamad',
      'nazri nyata'],
     'cluster-top-words': ['wakil pandang umno',
      'nazri nyata',
      'parti',
      'mohamad',
      'hak',
      'jenis',
      'hubung',
      'hormat paham sekolah',
      'iktiraf']}



Not clustering important words
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    malaya.summarize.lda(isu_kerajaan,important_words=10,return_cluster=False)




.. parsed-literal::

    {'summary': 'komitmen umno berhubung bentuk sokongan infrastruktur pengiktirafan pemberian peruntukan. berharap isu dipolitikkan bertanggungjawab menggambarkan pendirian sebenar umno. mohamed nazri mengambil pertanggungjawaban penjelasan maksud ucapanny semenyih selangor',
     'top-words': ['wakil pandang umno',
      'parti',
      'hak',
      'hormat paham',
      'hormat paham sekolah',
      'hubung',
      'iktiraf',
      'jenis',
      'mohamad',
      'nazri nyata']}


