.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 4.89 s, sys: 1.16 s, total: 6.06 s
    Wall time: 7.11 s


List available T5 models
------------------------

.. code:: python

    malaya.summarize.available_t5()




.. parsed-literal::

    {'small': ['122MB',
      'ROUGE-1: 0.33854',
      'ROUGE-2: 0.14588',
      'ROUGE-L: 0.23528'],
     'base': ['448MB', 'ROUGE-1: 0.34103', 'ROUGE-2: 0.14994', 'ROUGE-L: 0.23655']}



For now we do not have a good metrics to calculate ‘abstract’ of
summaries generated from the model. ROUGE is simply calculate same
N-GRAM overlap each others, sometime summaries generated are almost
human perfect but not close to as baseline summaries, so ROUGE score
will become lower.

Load T5
-------

T5 is a transformer model that capable to generate abstractive
summarization. In this example, we are going to use ``base`` model, feel
free to use ``small`` if you find ``base`` is too slow.

.. code:: python

    model = malaya.summarize.t5(model = 'base')


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/summarize.py:562: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/summarize.py:564: load (from tensorflow.python.saved_model.loader_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.loader.load or tf.compat.v1.saved_model.load. There will be a new function for importing SavedModels in Tensorflow 2.0.
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/summarize-argmax/t5/base/model/variables/variables


summarization mode
^^^^^^^^^^^^^^^^^^

T5 in Malaya provided 3 different mode for summarization,

1. generate summary,

.. code:: python

   model.summarize(string, mode = 'ringkasan')

2. generate title,

.. code:: python

   model.summarize(string, mode = 'tajuk')

3. generate short body (this is simply summarize every sentences in our
   string, splitted by fullstop),

.. code:: python

   model.summarize(string, mode = 'perenggan')

default is ``ringkasan``,

.. code:: python

   def summarize(self, string: str, mode: str = 'ringkasan')

I am going to simply copy paste some local news into this notebook. I
will search about ``isu mahathir`` in google news, `link
here <https://www.google.com/search?q=isu+mahathir&sxsrf=ALeKk02V_bAJC3sSrV38JQgGYWL_mE0biw:1589951900053&source=lnms&tbm=nws&sa=X&ved=2ahUKEwjapNmx2MHpAhVp_XMBHRt7BEQQ_AUoAnoECCcQBA&biw=1440&bih=648&dpr=2>`__.

**link**:
https://www.hmetro.com.my/mutakhir/2020/05/580438/peletakan-jawatan-tun-m-ditolak-bukan-lagi-isu

**Title**: Peletakan jawatan Tun M ditolak, bukan lagi isu.

**Body**: PELETAKAN jawatan Tun Dr Mahathir Mohamad sebagai Pengerusi
Parti Pribumi Bersatu Malaysia (Bersatu) ditolak di dalam mesyuarat khas
Majlis Pimpinan Tertinggi (MPT) pada 24 Februari lalu.

Justeru, tidak timbul soal peletakan jawatan itu sah atau tidak kerana
ia sudah pun diputuskan pada peringkat parti yang dipersetujui semua
termasuk Presiden, Tan Sri Muhyiddin Yassin.

Bekas Setiausaha Agung Bersatu Datuk Marzuki Yahya berkata, pada
mesyuarat itu MPT sebulat suara menolak peletakan jawatan Dr Mahathir.

"Jadi ini agak berlawanan dengan keputusan yang kita sudah buat. Saya
tak faham bagaimana Jabatan Pendaftar Pertubuhan Malaysia (JPPM) kata
peletakan jawatan itu sah sedangkan kita sudah buat keputusan di dalam
mesyuarat, bukan seorang dua yang buat keputusan.

"Semua keputusan mesti dibuat melalui parti. Walau apa juga perbincangan
dibuat di luar daripada keputusan mesyuarat, ini bukan keputusan parti.

“Apa locus standy yang ada pada Setiausaha Kerja untuk membawa perkara
ini kepada JPPM. Seharusnya ia dibawa kepada Setiausaha Agung sebagai
pentadbir kepada parti,” katanya kepada Harian Metro.

Beliau mengulas laporan media tempatan hari ini mengenai pengesahan JPPM
bahawa Dr Mahathir tidak lagi menjadi Pengerusi Bersatu berikutan
peletakan jawatannya di tengah-tengah pergolakan politik pada akhir
Februari adalah sah.

Laporan itu juga menyatakan, kedudukan Muhyiddin Yassin memangku jawatan
itu juga sah.

Menurutnya, memang betul Dr Mahathir menghantar surat peletakan jawatan,
tetapi ditolak oleh MPT.

“Fasal yang disebut itu terpakai sekiranya berhenti atau diberhentikan,
tetapi ini mesyuarat sudah menolak,” katanya.

Marzuki turut mempersoal kenyataan media yang dibuat beberapa pimpinan
parti itu hari ini yang menyatakan sokongan kepada Perikatan Nasional.

“Kenyataan media bukanlah keputusan rasmi. Walaupun kita buat 1,000
kenyataan sekali pun ia tetap tidak merubah keputusan yang sudah dibuat
di dalam mesyuarat. Kita catat di dalam minit apa yang berlaku di dalam
mesyuarat,” katanya.

.. code:: python

    string = """
    PELETAKAN jawatan Tun Dr Mahathir Mohamad sebagai Pengerusi Parti Pribumi Bersatu Malaysia (Bersatu) ditolak di dalam mesyuarat khas Majlis Pimpinan Tertinggi (MPT) pada 24 Februari lalu.
    
    Justeru, tidak timbul soal peletakan jawatan itu sah atau tidak kerana ia sudah pun diputuskan pada peringkat parti yang dipersetujui semua termasuk Presiden, Tan Sri Muhyiddin Yassin.
    
    Bekas Setiausaha Agung Bersatu Datuk Marzuki Yahya berkata, pada mesyuarat itu MPT sebulat suara menolak peletakan jawatan Dr Mahathir.
    
    "Jadi ini agak berlawanan dengan keputusan yang kita sudah buat. Saya tak faham bagaimana Jabatan Pendaftar Pertubuhan Malaysia (JPPM) kata peletakan jawatan itu sah sedangkan kita sudah buat keputusan di dalam mesyuarat, bukan seorang dua yang buat keputusan.
    
    "Semua keputusan mesti dibuat melalui parti. Walau apa juga perbincangan dibuat di luar daripada keputusan mesyuarat, ini bukan keputusan parti.
    
    "Apa locus standy yang ada pada Setiausaha Kerja untuk membawa perkara ini kepada JPPM. Seharusnya ia dibawa kepada Setiausaha Agung sebagai pentadbir kepada parti," katanya kepada Harian Metro.
    
    Beliau mengulas laporan media tempatan hari ini mengenai pengesahan JPPM bahawa Dr Mahathir tidak lagi menjadi Pengerusi Bersatu berikutan peletakan jawatannya di tengah-tengah pergolakan politik pada akhir Februari adalah sah.
    
    Laporan itu juga menyatakan, kedudukan Muhyiddin Yassin memangku jawatan itu juga sah.
    
    Menurutnya, memang betul Dr Mahathir menghantar surat peletakan jawatan, tetapi ditolak oleh MPT.
    
    "Fasal yang disebut itu terpakai sekiranya berhenti atau diberhentikan, tetapi ini mesyuarat sudah menolak," katanya.
    
    Marzuki turut mempersoal kenyataan media yang dibuat beberapa pimpinan parti itu hari ini yang menyatakan sokongan kepada Perikatan Nasional.
    
    "Kenyataan media bukanlah keputusan rasmi. Walaupun kita buat 1,000 kenyataan sekali pun ia tetap tidak merubah keputusan yang sudah dibuat di dalam mesyuarat. Kita catat di dalam minit apa yang berlaku di dalam mesyuarat," katanya.
    """

.. code:: python

    import re
    
    # minimum cleaning, just simply to remove newlines.
    def cleaning(string):
        string = string.replace('\n', ' ')
        string = re.sub(r'[ ]+', ' ', string).strip()
        return string
    
    string = cleaning(string)

generate ringkasan
^^^^^^^^^^^^^^^^^^

.. code:: python

    model.summarize(string, mode = 'ringkasan')




.. parsed-literal::

    'Kenyataan media yang dibuat oleh kepimpinan parti adalah sah. Tidak ada persoalan peletakan jawatan Dr Mahathir adalah sah atau tidak. Ia sudah diputuskan oleh semua pihak termasuk Presiden, Tan Sri Muhyiddin Yassin'



generate tajuk
^^^^^^^^^^^^^^

.. code:: python

    model.summarize(string, mode = 'tajuk')




.. parsed-literal::

    'Bukan pertikai keputusan MPT - Marzuki'



generate perenggan
^^^^^^^^^^^^^^^^^^

This mode is not really good as ``ringkasan`` and ``tajuk``, it is
pretty hard to hard to supervised summaries for each sentences. We
applied ``#`` to mask sensitive issues.

.. code:: python

    model.summarize(string, mode = 'perenggan')




.. parsed-literal::

    'Peletakan jawatan dr mahathir di mesyuarat khas. Tidak ada persoalan mengenai peletakan jawatan presiden. Bekas ketua un menolak peletakan jawatan dr m. Keputusan kami mengenai keputusan ####. Malaysia mengatakan peletakan jawatan adalah sah. "Semua keputusan mesti dibuat melalui parti.. Perbincangan mengenai keputusan mesyuarat parti tidak ada keputusan parti. Locus standy untuk membawa perkara ini kepada jppm. Ketua parti mengatakan bahawa dia harus menjadi pentadbir. Pm mengatakan bahawa dia tidak lagi menjadi ketua bersatu. Laporan mengatakan bahawa kedudukan muhyiddin yassin disahkan. Pm menolak peletakan jawatan tetapi menolak surat peletakan jawatan. #### - #### - ####. Marzuki menolak kenyataan media yang menyokong parti. "Kenyataan media bukanlah keputusan rasmi.. Keputusan mengenai pertemuan afghanistan tetap tidak berubah. Kami catat dalam minit yang berlaku di mesyuarat'



**Link**: https://www.malaysiakini.com/news/525953

**Title**: Mahathir jangan hipokrit isu kes mahkamah Riza, kata
Takiyuddin

**Body**: Menteri undang-undang Takiyuddin Hassan berkata kerajaan
berharap Dr Mahathir Mohamad tidak bersikap hipokrit dengan mengatakan
beliau tertanya-tanya dan tidak faham dengan keputusan mahkamah melepas
tanpa membebaskan (DNAA) Riza Aziz, anak tiri bekas perdana menteri
Najib Razak, dalam kes pengubahan wang haram membabitkan dana 1MDB.

Pemimpin PAS itu berkata ini kerana keputusan itu dibuat oleh peguam
negara dan dilaksanakan oleh timbalan pendakwa raya yang mengendalikan
kes tersebut pada akhir 2019.

“Saya merujuk kepada kenyataan Dr Mahathir tentang tindakan Mahkamah
Sesyen memberikan pelepasan tanpa pembebasan (discharge not amounting to
acquittal) kepada Riza Aziz baru-baru ini.

“Kerajaan berharap Dr Mahathir tidak bersikap hipokrit dengan mengatakan
beliau ‘tertanya-tanya’, keliru dan tidak faham terhadap suatu keputusan
yang dibuat oleh Peguam Negara dan dilaksanakan oleh Timbalan Pendakwa
Raya yang mengendalikan kes ini pada akhir tahun 2019,” katanya dalam
satu kenyataan hari ini.

Riza pada Khamis dilepas tanpa dibebaskan daripada lima pertuduhan
pengubahan wang berjumlah AS$248 juta (RM1.08 bilion).

Dalam persetujuan yang dicapai antara pihak Riza dan pendakwaan, beliau
dilepas tanpa dibebaskan atas pertuduhan itu dengan syarat memulangkan
semula aset dari luar negara dengan nilai anggaran AS$107.3 juta
(RM465.3 juta).

Ekoran itu, Mahathir antara lain menyuarakan kekhuatirannya berkenaan
persetujuan itu dan mempersoalkan jika pihak yang didakwa atas tuduhan
mencuri boleh terlepas daripada tindakan jika memulangkan semula apa
yang dicurinya.

“Dia curi berbilion-bilion…Dia bagi balik kepada kerajaan. Dia kata
kepada kerajaan, ‘Nah, duit yang aku curi. Sekarang ini, jangan ambil
tindakan terhadap aku.’ Kita pun kata, ‘Sudah bagi balik duit okey
lah’,” katanya.

Menjelaskan bahawa beliau tidak mempersoalkan keputusan mahkamah,
Mahathir pada masa sama berkata ia menunjukkan undang-undang mungkin
perlu dipinda.

Mengulas lanjut, Takiyuddin yang juga setiausaha agung PAS berkata
kenyataan Mahathir tidak munasabah sebagai bekas perdana menteri.

"Kerajaan berharap Dr Mahathir tidak terus bertindak mengelirukan rakyat
dengan mengatakan beliau ‘keliru’.

“Kerajaan PN akan terus bertindak mengikut undang-undang dan berpegang
kepada prinsip kebebasan badan kehakiman dan proses perundangan yang
sah,” katanya.

.. code:: python

    string = """
    Menteri undang-undang Takiyuddin Hassan berkata kerajaan berharap Dr Mahathir Mohamad tidak bersikap hipokrit dengan mengatakan beliau tertanya-tanya dan tidak faham dengan keputusan mahkamah melepas tanpa membebaskan (DNAA) Riza Aziz, anak tiri bekas perdana menteri Najib Razak, dalam kes pengubahan wang haram membabitkan dana 1MDB.
    
    Pemimpin PAS itu berkata ini kerana keputusan itu dibuat oleh peguam negara dan dilaksanakan oleh timbalan pendakwa raya yang mengendalikan kes tersebut pada akhir 2019.
    
    “Saya merujuk kepada kenyataan Dr Mahathir tentang tindakan Mahkamah Sesyen memberikan pelepasan tanpa pembebasan (discharge not amounting to acquittal) kepada Riza Aziz baru-baru ini.
    
    “Kerajaan berharap Dr Mahathir tidak bersikap hipokrit dengan mengatakan beliau ‘tertanya-tanya’, keliru dan tidak faham terhadap suatu keputusan yang dibuat oleh Peguam Negara dan dilaksanakan oleh Timbalan Pendakwa Raya yang mengendalikan kes ini pada akhir tahun 2019,” katanya dalam satu kenyataan hari ini.
    
    Riza pada Khamis dilepas tanpa dibebaskan daripada lima pertuduhan pengubahan wang berjumlah AS$248 juta (RM1.08 bilion).
    
    Dalam persetujuan yang dicapai antara pihak Riza dan pendakwaan, beliau dilepas tanpa dibebaskan atas pertuduhan itu dengan syarat memulangkan semula aset dari luar negara dengan nilai anggaran AS$107.3 juta (RM465.3 juta).
    
    Ekoran itu, Mahathir antara lain menyuarakan kekhuatirannya berkenaan persetujuan itu dan mempersoalkan jika pihak yang didakwa atas tuduhan mencuri boleh terlepas daripada tindakan jika memulangkan semula apa yang dicurinya.
    
    "Dia curi berbilion-bilion...Dia bagi balik kepada kerajaan. Dia kata kepada kerajaan, 'Nah, duit yang aku curi. Sekarang ini, jangan ambil tindakan terhadap aku.' Kita pun kata, 'Sudah bagi balik duit okey lah'," katanya.
    
    Menjelaskan bahawa beliau tidak mempersoalkan keputusan mahkamah, Mahathir pada masa sama berkata ia menunjukkan undang-undang mungkin perlu dipinda.
    
    Mengulas lanjut, Takiyuddin yang juga setiausaha agung PAS berkata
    kenyataan Mahathir tidak munasabah sebagai bekas perdana menteri.
    
    "Kerajaan berharap Dr Mahathir tidak terus bertindak mengelirukan rakyat dengan mengatakan beliau ‘keliru’.
    
    “Kerajaan PN akan terus bertindak mengikut undang-undang dan berpegang kepada prinsip kebebasan badan kehakiman dan proses perundangan yang sah,” katanya.
    """
    
    string = cleaning(string)

.. code:: python

    model.summarize(string, mode = 'ringkasan')




.. parsed-literal::

    '"Kerajaan berharap Dr Mahathir tidak hipokrit," kata menteri undang-undang. Riza Aziz, anak tiri Najib Razak, dilepas tanpa dibebaskan atas tuduhan pengubahan wang haram. Mahathir mengatakan dia mempersoalkan jika pihak yang didakwa mencuri boleh terlepas tindakan'



.. code:: python

    model.summarize(string, mode = 'tajuk')




.. parsed-literal::

    'Kerajaan harap Dr M tak hipokrit'



.. code:: python

    model.summarize(string, mode = 'perenggan')




.. parsed-literal::

    'Menteri mengatakan bahawa dia tertanya-tanya dengan keputusan mahkamah untuk membebaskan anak tiri najib. Pas mengatakan peguam negara akan dilantik pada akhir tahun. Pm merujuk kepada pembebasan tanpa pembebasan kepada aig. Pm berharap tidak ada yang hipokrit dengan keputusan pendakwaan. Riza dilepas tanpa dibebaskan dari tuduhan pengubahan wang. Pihak pendakwaan brazil bersetuju untuk mengembalikan aset luar negara. Pm mempersoalkan sama ada pihak yang dituduh mencuri boleh terlepas tindakan. "Dia curi berbilion-bilion...Dia bagi balik kepada kerajaan.. Britain mengatakan duit yang dicuri adalah wang yang dicuri. Sekarang ini, jangan ambil tindakan terhadap aku.. Aig mengatakan kita \'terus memberi balik duit okey lah\'. Mahathir mengatakan undang-undang mungkin perlu dipinda. Afghanistan mengatakan bahawa kenyataan pm tidak wajar. Pm berharap pm tidak akan berbohong. Pm malaysia mengatakan bahawa ia akan mematuhi undang-undang'



List available skip-thought models
----------------------------------

.. code:: python

    malaya.summarize.available_skipthought()




.. parsed-literal::

    ['lstm', 'residual-network']



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

-  ``'lstm'`` - LSTM skip-thought deep learning model trained on news
   dataset. Hopefully we can train on wikipedia dataset.
-  ``'residual-network'`` - CNN residual network with Bahdanau Attention
   skip-thought deep learning model trained on wikipedia dataset.

We use TextRank for scoring algorithm.

Encoder summarization
---------------------

We leverage the power of deep encoder models like skip-thought or
Transformer to do extractive summarization for us.

Load skip-thought
^^^^^^^^^^^^^^^^^

.. code:: python

    lstm = malaya.summarize.deep_skipthought(model = 'lstm')
    encoder = malaya.summarize.encoder(lstm)


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:54: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:55: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/model/skip_thought.py:136: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


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
     'cluster-top-words': ['sharmini',
      'seliakekurangan',
      'diharap',
      'sakailah',
      'pembikin',
      'poupart',
      'mempunya',
      'bersabdabarangsiapa',
      'kepulangan',
      'chusus']}



Problem with skip-thought models, ``top-words`` suggested are really not
good, because skip-thought trained to leverage sentence level, not word
level. How about Transformer model? Lets we try ALXLNET.

Load Transformer
^^^^^^^^^^^^^^^^

.. code:: python

    alxlnet = malaya.transformer.load(model = 'alxlnet')
    encoder = malaya.summarize.encoder(alxlnet)


.. parsed-literal::

    INFO:tensorflow:memory input None
    INFO:tensorflow:Use float type <dtype: 'float32'>
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
      'pandangan',
      'kenyataan'],
     'cluster-top-words': ['pandangan',
      'bumiputera',
      'umno',
      'negara',
      'kenyataan',
      'pendidikan',
      'pendirian',
      'media',
      'malaysia',
      'sekolah']}



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
     'cluster-top-words': ['hormat paham',
      'hak',
      'umno',
      'sekolah vernakular',
      'nyata',
      'nazri',
      'pandang']}



We can use ``tfidf`` as vectorizer.

.. code:: python

    malaya.summarize.lsa(isu_kerajaan,important_words=10, ngram = (1,3), vectorizer = 'tfidf')




.. parsed-literal::

    {'summary': 'Timbalan Presiden UMNO, Datuk Seri Mohamad Hasan berkata, kenyataan tersebut tidak mewakili pendirian serta pandangan UMNO   kerana parti itu menghormati serta memahami keperluan sekolah vernakular dalam negara. "Saya ingin menegaskan dua perkara penting. "Saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar UMNO dan BN," katanya.',
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
      'mohamed nazri',
      'pandang peribadi',
      'paham sekolah vernakular',
      'mohamad',
      'wakil pandang umno',
      'nazri nyata']}



We can use ``skip-gram`` as vectorizer, and can override ``skip`` value.

.. code:: python

    malaya.summarize.lsa(isu_kerajaan,important_words=10, ngram = (1,3), vectorizer = 'skip-gram', skip = 3)




.. parsed-literal::

    {'summary': 'Timbalan Presiden UMNO, Datuk Seri Mohamad Hasan berkata, kenyataan tersebut tidak mewakili pendirian serta pandangan UMNO   kerana parti itu menghormati serta memahami keperluan sekolah vernakular dalam negara. Mohamad yang menjalankan tugas-tugas Presiden UMNO berkata, UMNO konsisten dengan pendirian itu dalam mengiktiraf kepelbagaian bangsa dan etnik termasuk hak untuk beragama serta mendapat pendidikan. "Saya berharap isu ini tidak dipolitikkan secara tidak bertanggungjawab oleh mana-mana pihak terutama dengan cara yang tidak menggambarkan pendirian sebenar UMNO dan BN," katanya.',
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
     'cluster-top-words': ['nyata',
      'sekolah vernakular',
      'hormat',
      'nazri',
      'pandang umno',
      'presiden umno']}



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
     'cluster-top-words': ['nyanyi',
      'artis',
      'muzik',
      'sembah',
      'konsert',
      'festival',
      'kl',
      'jamm',
      'babit',
      'sheila']}



Train LDA model
---------------

.. code:: python

    malaya.summarize.lda(isu_kerajaan,important_words=10)




.. parsed-literal::

    {'summary': '"Saya ingin menegaskan dua perkara penting. Kata beliau, komitmen UMNO dan BN berhubung perkara itu dapat dilihat dengan jelas dalam bentuk sokongan infrastruktur, pengiktirafan dan pemberian peruntukan yang diperlukan. Timbalan Presiden UMNO, Datuk Seri Mohamad Hasan berkata, kenyataan tersebut tidak mewakili pendirian serta pandangan UMNO   kerana parti itu menghormati serta memahami keperluan sekolah vernakular dalam negara.',
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
     'cluster-top-words': ['hormat paham',
      'hak',
      'umno',
      'sekolah vernakular',
      'nyata',
      'nazri',
      'pandang']}



.. code:: python

    malaya.summarize.lda(isu_string,important_words=10, vectorizer = 'skip-gram')




.. parsed-literal::

    {'summary': '"Kami pernah terbabit dengan showcase dan majlis korporat sebelum ini. Manakala, artis antarabangsa pula membabitkan J Arie (Hong Kong), NCT Dream (Korea Selatan) dan DJ Sura (Korea Selatan). DUA legenda hebat dan \'The living legend\' ini sudah memartabatkan bidang muzik sejak lebih tiga dekad lalu.',
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
     'cluster-top-words': ['sembah',
      'artis',
      'muzik',
      'nyanyi',
      'festival',
      'konsert',
      'kl jamm',
      'sheila']}



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

I will use ``load_news``, you can try embedded from wikipedia.

.. code:: python

    vocab_news, embedded_news = malaya.wordvector.load_news()
    w2v_wiki = malaya.wordvector.load(embedded_news, vocab_news)

.. code:: python

    malaya.summarize.doc2vec(w2v_wiki, isu_kerajaan, soft = False, top_k = 5)




.. parsed-literal::

    'Timbalan Presiden UMNO, Datuk Seri Mohamad Hasan berkata, kenyataan tersebut tidak mewakili pendirian serta pandangan UMNO   kerana parti itu menghormati serta memahami keperluan sekolah vernakular dalam negara. Mohamad yang menjalankan tugas-tugas Presiden UMNO berkata, UMNO konsisten dengan pendirian itu dalam mengiktiraf kepelbagaian bangsa dan etnik termasuk hak untuk beragama serta mendapat pendidikan. Kata Nazri dalam kenyataannya itu, beliau menekankan bahawa semua pihak perlu menghormati hak orang Melayu dan bumiputera. Kata beliau, komitmen UMNO dan BN berhubung perkara itu dapat dilihat dengan jelas dalam bentuk sokongan infrastruktur, pengiktirafan dan pemberian peruntukan yang diperlukan. "Kedua UMNO sebagai sebuah parti sangat menghormati dan memahami keperluan sekolah vernakular di Malaysia.'


