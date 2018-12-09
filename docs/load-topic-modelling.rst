
.. code:: python

    import pandas as pd
    import malaya

.. code:: python

    df = pd.read_csv('tests/02032018.csv',sep=';')
    df = df.iloc[3:,1:]
    df.columns = ['text','label']
    corpus = df.text.tolist()

Train LDA model
---------------

.. code:: python

    lda = malaya.lda_topic_modelling(corpus,10,stemming=False)


Print topics
^^^^^^^^^^^^

.. code:: python

    lda.print_topics(5)


.. parsed-literal::

    topic 0       topic 1       topic 2       topic 3       topic 4
    --------      --------      --------      --------      --------
    masalah       parti         ia            perlu         kerajaan
    selesaikan    semua         hutang        pilihan       projek
    termasuk      sewa          mdb           raya          sebagai
    perniagaan    tanah         projek        masing        bahasa
    mempunyai     hati          ada           faktor        syarikat
    pencemaran    kelulusan     lebih         umum          menjadi
    rakyat        masa          masa          masa          jadi
    sebagai       nak           diselesaikan  membuat       dilakukan
    jppm          rumah         tahun         parti         swasta
    kerja         terus         kewangan      diri          indonesia




Important sentences based on topics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    lda.get_sentences(5)




.. parsed-literal::

    ['kerja kerja naik taraf mengatasi masalah banjir masalah pencemaran air sungai benut menjadi nadi utama bekalan air bersih giat dilaksanakan',
     'sumbangan segmen penting menyumbang seperlima hasil selain pendapatan sampingan lain iaitu makanan penerbangan perkhidmatan daftar masuk lapangan terbang perniagaan logistik',
     'mempunyai perniagaan lain termasuk perniagaan digital santan santan menerima minat seluruh asia beberapa pihak menghubungi aktiviti francais sebagai restoran makanan segera asia',
     'berharap kerjasama diteruskan seterusnya menjadikan kementerian sebagai kementerian membantu rakyat',
     'soal demokrasi semua berhak bertanding penting amanah rakyat dijalankan']



Get topics
^^^^^^^^^^

.. code:: python

    lda.get_topics(10)




.. parsed-literal::

    [(0,
      'masalah selesaikan termasuk perniagaan mempunyai pencemaran rakyat sebagai jppm kerja'),
     (1, 'parti semua sewa tanah hati kelulusan masa nak rumah terus'),
     (2, 'ia hutang mdb projek ada lebih masa diselesaikan tahun kewangan'),
     (3, 'perlu pilihan raya masing faktor umum masa membuat parti diri'),
     (4,
      'kerajaan projek sebagai bahasa syarikat menjadi jadi dilakukan swasta indonesia'),
     (5,
      'orang asli pekan menjadi takut bertanggungjawab perdana menteri semua tahu'),
     (6,
      'awam kerajaan memastikan malaysia seri datuk menteri pimpinan ahli perdana'),
     (7, 'ada soal bulan ahli baiah pernah status tn undi buat'),
     (8, 'malaysia orang asli negara rakyat lain perniagaan awam kerajaan sama'),
     (9,
      'negara bahasa rakyat besar kalau tindakan orang mengambil malaysia bank')]



Train NMF model
---------------

.. code:: python

    nmf = malaya.nmf_topic_modelling(corpus,10)
    nmf.print_topics(5)


.. parsed-literal::

    topic 0       topic 1       topic 2       topic 3       topic 4
    --------      --------      --------      --------      --------
    negara        ada           ia            ros           menteri
    malaysia      kalau         jalan         tangguh       perdana
    bangun        raja          lihat         pilih         jelas
    rakyat        pas           lancar        parti         datuk
    kongsi        parti         kembang       umno          seri
    alam          sama          jual          lembaga       terima
    penting       buat          gembira       dah           kena
    sedia         baiah         projek        putus         isu
    selatan       politik       baik          lebih         jemaah
    lebih         bn            beli          tempoh        nyata




.. code:: python

    nmf.get_sentences(5)




.. parsed-literal::

    ['sedia kongsi alam tahu bangun ekonomi sosial negara bangun lain rangka program kerjasama teknikal malaysia mtcp tunjuk sedia malaysia kongsi alam bangun negara negara selatan selatan',
     'sedia kongsi alam tahu bangun ekonomi sosial negara bangun lain rangka program kerjasama teknikal malaysia mtcp tunjuk sedia malaysia kongsi alam bangun negara negara selatan selatan',
     'bagai negara lebih bangun malaysia main peran penting kongsi alam mahir bangun kawasan bandar',
     'bagai negara lebih bangun malaysia main peran penting kongsi alam mahir bangun kawasan bandar',
     'semua rakyat malaysia punya penting langsung negara']



.. code:: python

    nmf.get_topics(10)




.. parsed-literal::

    [(0, 'negara malaysia bangun rakyat kongsi alam penting sedia selatan lebih'),
     (1, 'ada kalau raja pas parti sama buat baiah politik bn'),
     (2, 'ia jalan lihat lancar kembang jual gembira projek baik beli'),
     (3, 'ros tangguh pilih parti umno lembaga dah putus lebih tempoh'),
     (4, 'menteri perdana jelas datuk seri terima kena isu jemaah nyata'),
     (5, 'orang asli jadi masyarakat percaya dasar jakoa rana sumber temiar'),
     (6, 'hutang mdb selesai perlu masa wang wujud ia tempoh pendek'),
     (7, 'ajar masa laku maju bidang didik tingkat terus proses raja'),
     (8, 'kapal jho low rampas doj niaga dah dakwa keluar sivil'),
     (9, 'undi bulan pakat impak bagai wujud keluar tuju pihak catat')]



Train LSA model
---------------

.. code:: python

    lsa = malaya.lsa_topic_modelling(corpus,10)
    lsa.print_topics(5)


.. parsed-literal::

    topic 0       topic 1       topic 2       topic 3       topic 4
    --------      --------      --------      --------      --------
    ada           negara        ia            pilih         menteri
    ia            malaysia      hutang        tangguh       jelas
    malaysia      bangun        mdb           ros           perdana
    baik          rakyat        projek        masa          rakyat
    negara        kongsi        masa          lebih         terima
    rakyat        alam          lihat         umno          datuk
    jadi          penting       wang          raya          kena
    raja          selatan       perlu         hutang        nyata
    masa          kawasan       selesai       parti         seri
    menteri       perlu         jual          lembaga       selesai




.. code:: python

    lsa.get_sentences(5)




.. parsed-literal::

    ['perdana menteri beri jelas isu kena nyata pihak raja terima baik jelas',
     'nak tutup hutang ada projek lain jadi ia makan masa',
     'jual syarikat paja pagi wang terima tingkat ringgit jadi ia cermin lebih baik kuat daya tahan ekonomi malaysia',
     'ia lihat dasar sambut orang ramai hadap ancang bangun kawasan tempat ia jelas tunjuk laksana terima',
     'semua rakyat malaysia punya penting langsung negara']



.. code:: python

    lsa.get_topics(10)




.. parsed-literal::

    [(0, 'ada ia malaysia baik negara rakyat jadi raja masa menteri'),
     (1,
      'negara malaysia bangun rakyat kongsi alam penting selatan kawasan perlu'),
     (2, 'ia hutang mdb projek masa lihat wang perlu selesai jual'),
     (3, 'pilih tangguh ros masa lebih umno raya hutang parti lembaga'),
     (4, 'menteri jelas perdana rakyat terima datuk kena nyata seri selesai'),
     (5, 'orang asli jadi dasar jalan ia baik undi ros lancar'),
     (6, 'perlu rakyat masa orang jadi selesai laku dasar masalah wujud'),
     (7, 'undi ajar laku masa terus maju bidang bulan didik bagai'),
     (8, 'perlu orang tumbuh undi asli rana nyata dakwa keluar sumber'),
     (9, 'undi ambil pihak baik putus semua buat jalan bulan cara')]
