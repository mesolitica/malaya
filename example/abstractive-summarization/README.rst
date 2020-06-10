.. code:: ipython3

    %%time
    import malaya
    from pprint import pprint


.. parsed-literal::

    CPU times: user 4.54 s, sys: 945 ms, total: 5.49 s
    Wall time: 4.65 s


List available T5 models
------------------------

.. code:: ipython3

    malaya.summarization.abstractive.available_t5()




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

   def t5(model: str = 'base', **kwargs):

       """
       Load T5 model to generate a summarization given a string.

       Parameters
       ----------
       model : str, optional (default='base')
           Model architecture supported. Allowed values:

           * ``'base'`` - T5 Base parameters.
           * ``'small'`` - T5 Small parameters.

       Returns
       -------
       result: malaya.model.t5.SUMMARIZATION class
       """

.. code:: ipython3

    model = malaya.summarization.abstractive.t5(model = 'base')


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/summarization/abstractive.py:74: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/summarization/abstractive.py:76: load (from tensorflow.python.saved_model.loader_impl) is deprecated and will be removed in a future version.
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

   def summarize(self, string: str, mode: str = 'ringkasan'):
       """
       Summarize a string.

       Parameters
       ----------
       string: str
       mode: str
           mode for summarization. Allowed values:

           * ``'ringkasan'`` - summarization for long sentence, eg, news summarization.
           * ``'tajuk'`` - title summarization for long sentence, eg, news title.
           * ``'perenggan'`` - summarization for each perenggan. This will automatically split sentences by EOS.

       Returns
       -------
       result: str
       """

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

.. code:: ipython3

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

.. code:: ipython3

    import re
    
    # minimum cleaning, just simply to remove newlines.
    def cleaning(string):
        string = string.replace('\n', ' ')
        string = re.sub(r'[ ]+', ' ', string).strip()
        return string
    
    string = cleaning(string)

generate ringkasan
^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    pprint(model.summarize(string, mode = 'ringkasan'))


.. parsed-literal::

    ('Kenyataan media yang dibuat oleh kepimpinan parti adalah sah. Tidak ada '
     'persoalan peletakan jawatan Dr Mahathir adalah sah atau tidak. Ia sudah '
     'diputuskan oleh semua pihak termasuk Presiden, Tan Sri Muhyiddin Yassin')


generate tajuk
^^^^^^^^^^^^^^

.. code:: ipython3

    model.summarize(string, mode = 'tajuk')




.. parsed-literal::

    'Bukan pertikai keputusan MPT - Marzuki'



generate perenggan
^^^^^^^^^^^^^^^^^^

This mode is not really good as ``ringkasan`` and ``tajuk``, it is
pretty hard to hard to supervised summaries for each sentences. We
applied ``#`` to mask sensitive issues.

.. code:: ipython3

    pprint(model.summarize(string, mode = 'perenggan'))


.. parsed-literal::

    ('Peletakan jawatan dr mahathir di mesyuarat khas. Tidak ada persoalan '
     'mengenai peletakan jawatan presiden. Bekas ketua un menolak peletakan '
     'jawatan dr m. Keputusan kami mengenai keputusan ####. Malaysia mengatakan '
     'peletakan jawatan adalah sah. "Semua keputusan mesti dibuat melalui parti.. '
     'Perbincangan mengenai keputusan mesyuarat parti tidak ada keputusan parti. '
     'Locus standy untuk membawa perkara ini kepada jppm. Ketua parti mengatakan '
     'bahawa dia harus menjadi pentadbir. Pm mengatakan bahawa dia tidak lagi '
     'menjadi ketua bersatu. Laporan mengatakan bahawa kedudukan muhyiddin yassin '
     'disahkan. Pm menolak peletakan jawatan tetapi menolak surat peletakan '
     'jawatan. #### - #### - ####. Marzuki menolak kenyataan media yang menyokong '
     'parti. "Kenyataan media bukanlah keputusan rasmi.. Keputusan mengenai '
     'pertemuan afghanistan tetap tidak berubah. Kami catat dalam minit yang '
     'berlaku di mesyuarat')


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

.. code:: ipython3

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

.. code:: ipython3

    pprint(model.summarize(string, mode = 'ringkasan'))


.. parsed-literal::

    ('"Kerajaan berharap Dr Mahathir tidak hipokrit," kata menteri undang-undang. '
     'Riza Aziz, anak tiri Najib Razak, dilepas tanpa dibebaskan atas tuduhan '
     'pengubahan wang haram. Mahathir mengatakan dia mempersoalkan jika pihak yang '
     'didakwa mencuri boleh terlepas tindakan')


.. code:: ipython3

    model.summarize(string, mode = 'tajuk')




.. parsed-literal::

    'Kerajaan harap Dr M tak hipokrit'



.. code:: ipython3

    pprint(model.summarize(string, mode = 'perenggan'))


.. parsed-literal::

    ('Menteri mengatakan bahawa dia tertanya-tanya dengan keputusan mahkamah untuk '
     'membebaskan anak tiri najib. Pas mengatakan peguam negara akan dilantik pada '
     'akhir tahun. Pm merujuk kepada pembebasan tanpa pembebasan kepada aig. Pm '
     'berharap tidak ada yang hipokrit dengan keputusan pendakwaan. Riza dilepas '
     'tanpa dibebaskan dari tuduhan pengubahan wang. Pihak pendakwaan brazil '
     'bersetuju untuk mengembalikan aset luar negara. Pm mempersoalkan sama ada '
     'pihak yang dituduh mencuri boleh terlepas tindakan. "Dia curi '
     'berbilion-bilion...Dia bagi balik kepada kerajaan.. Britain mengatakan duit '
     'yang dicuri adalah wang yang dicuri. Sekarang ini, jangan ambil tindakan '
     "terhadap aku.. Aig mengatakan kita 'terus memberi balik duit okey lah'. "
     'Mahathir mengatakan undang-undang mungkin perlu dipinda. Afghanistan '
     'mengatakan bahawa kenyataan pm tidak wajar. Pm berharap pm tidak akan '
     'berbohong. Pm malaysia mengatakan bahawa ia akan mematuhi undang-undang')

