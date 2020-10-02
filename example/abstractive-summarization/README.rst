Abstractive
===========

.. container:: alert alert-info

   This tutorial is available as an IPython notebook at
   `Malaya/example/abstractive-summarization <https://github.com/huseinzol05/Malaya/tree/master/example/abstractive-summarization>`__.

.. code:: ipython3

    %%time
    import malaya
    from pprint import pprint


.. parsed-literal::

    CPU times: user 4.99 s, sys: 666 ms, total: 5.65 s
    Wall time: 4.69 s


List available T5 models
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    malaya.summarization.abstractive.available_t5()


.. parsed-literal::

    INFO:root:tested on 5k CNN test set.




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Size (MB)</th>
          <th>Uncompressed Size (MB)</th>
          <th>ROUGE-1</th>
          <th>ROUGE-2</th>
          <th>ROUGE-L</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>small</th>
          <td>122.0</td>
          <td>355.6</td>
          <td>0.33854</td>
          <td>0.14588</td>
          <td>0.23528</td>
        </tr>
        <tr>
          <th>base</th>
          <td>448.0</td>
          <td>1300.0</td>
          <td>0.34103</td>
          <td>0.14994</td>
          <td>0.23655</td>
        </tr>
      </tbody>
    </table>
    </div>



Load T5
~~~~~~~

T5 is a transformer model that capable to generate abstractive
summarization. In this example, we are going to use ``base`` model, feel
free to use ``small`` if you find ``base`` is too slow.

.. code:: python

   def t5(model: str = 'base', compressed: bool = True, **kwargs):

       """
       Load T5 model to generate a summary given a string.

       Parameters
       ----------
       model : str, optional (default='base')
           Model architecture supported. Allowed values:

           * ``'base'`` - T5 BASE parameters.
           * ``'small'`` - T5 SMALL parameters.

       compressed: bool, optional (default=True)
           Load compressed model, but this not able to utilize malaya-gpu function. 
           This only compressed model size, but when loaded into VRAM / RAM, size uncompressed and compressed are the same.
           We prefer un-compressed model due to compressed model prone to error.

       Returns
       -------
       result: malaya.model.t5.SUMMARIZATION class
       """

**For malaya-gpu user, compressed t5 very fragile and we suggest use
``compressed=False``. Uncompressed model also can utilise GPU usage more
efficient**.

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

T5 in Malaya provided 2 different modes for summarization,

1. generate summary,

.. code:: python

   model.summarize(string, mode = 'ringkasan')

2. generate title,

.. code:: python

   model.summarize(string, mode = 'tajuk')

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



List available LM Transformer models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Problem with T5 models, it built on top of mesh-tensorflow, so the input
must size of 1. So we use Tensor2Tensor library to train exact model as
T5 with dynamic size of batch.

**But, we found out, our pretrained LM Transformer not good as T5**, we
might skipped some literature in t5 papers.

.. code:: ipython3

    malaya.summarization.abstractive.available_transformer()


.. parsed-literal::

    INFO:root:tested on 5k CNN test set.




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Size (MB)</th>
          <th>ROUGE-1</th>
          <th>ROUGE-2</th>
          <th>ROUGE-L</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>base</th>
          <td>832.0</td>
          <td>0.31863</td>
          <td>0.12150</td>
          <td>0.22023</td>
        </tr>
        <tr>
          <th>small</th>
          <td>379.0</td>
          <td>0.32215</td>
          <td>0.12741</td>
          <td>0.23528</td>
        </tr>
      </tbody>
    </table>
    </div>



Load Transformer
~~~~~~~~~~~~~~~~

.. code:: ipython3

    model = malaya.summarization.abstractive.transformer(model = 'small')


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:73: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:75: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:68: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


summarization mode
^^^^^^^^^^^^^^^^^^

T5 in Malaya provided 2 different modes for summarization,

1. generate summary,

.. code:: python

   model.summarize(string, mode = 'ringkasan')

2. generate title,

.. code:: python

   model.summarize(string, mode = 'tajuk')

default is ``ringkasan``,

.. code:: python

   def summarize(
       self,
       strings: List[str],
       mode: str = 'ringkasan',
       decoder: str = 'greedy',
       top_p: float = 0.7,
   ):
       """
       Summarize strings.

       Parameters
       ----------
       strings: List[str]
       mode: str
           mode for summarization. Allowed values:

           * ``'ringkasan'`` - summarization for long sentence, eg, news summarization.
           * ``'tajuk'`` - title summarization for long sentence, eg, news title.

decoder mode
^^^^^^^^^^^^

LM Transformer provided 3 different decoder for summarization,

1. greedy decoder, simply argmax,

.. code:: python

   model.summarization([string], decoder = 'greedy')

2. beam decoder, Beam width size 3, alpha 0.5 .

.. code:: python

   model.summarization([string], decoder = 'beam')

3. nucleus sampling decoder, Beam width size 1, with nucleus sampling.

.. code:: python

   model.summarization([string], decoder = 'nucleus', top_p = 0.7)

default is ``greedy``,

.. code:: python

   def summarize(
       self,
       strings: List[str],
       mode: str = 'ringkasan',
       decoder: str = 'greedy',
       top_p: float = 0.7,
   ):
       """
       Summarize strings.

       Parameters
       ----------

       decoder: str
           mode for summarization decoder. Allowed values:

           * ``'greedy'`` - Beam width size 1, alpha 0.
           * ``'beam'`` - Beam width size 3, alpha 0.5 .
           * ``'nucleus'`` - Beam width size 1, with nucleus sampling.

       top_p: float, (default=0.7)
           cumulative distribution and cut off as soon as the CDF exceeds `top_p`.
           this is only useful if use `nucleus` decoder.

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

generate tajuk
^^^^^^^^^^^^^^

.. code:: ipython3

    pprint(model.summarize([string], mode = 'tajuk'))


.. parsed-literal::

    ['Tun M letak jawatan Pengerusi Bersatu']


.. code:: ipython3

    pprint(model.summarize([string], mode = 'tajuk', decoder = 'beam'))


.. parsed-literal::

    ['Mengapa letak jawatan itu sah?']


.. code:: ipython3

    pprint(model.summarize([string], mode = 'tajuk', decoder = 'nucleus', top_p = 0.7))


.. parsed-literal::

    ['Tun M letak jawatan Pengerusi Bersatu secara sah']


generate ringkasan
^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    pprint(model.summarize([string], mode = 'ringkasan'))


.. parsed-literal::

    ['- Tun M tidak lagi menjadi Pengerusi Bersatu, tetapi dia masih menjadi ketua '
     'parti itu. Bekas setiausaha Agung Bersatu, Marzuki Yahya mengatakan bahawa '
     'dia tidak faham bagaimana JPPM mengatakan peletakan jawatan itu sah, Times. '
     '"Jadi ini agak berlawanan dengan keputusan yang kita sudah buat," katanya. '
     '"Saya tidak faham bagaimana Jabatan Pendaftar Pertubuhan Malaysia mengatakan '
     'peletakan jawatan itu sah sedangkan kita sudah membuat keputusan di dalam '
     'mesyuarat, bukan seorang dua yang membuat keputusan. " Marzuki mengatakan '
     'bahawa dia tidak faham bagaimana JPPM mengatakan peletakan jawatan itu sah, '
     'tetapi "apa pun perbincangan dibuat di luar dari keputusan mesyuarat, ini '
     'bukan keputusan parti. " (Dalam berita lain, seorang lelaki yang mengatakan '
     'bahawa dia adalah "pembersihan" dengan J. Seharusnya membawa kepada '
     'Setiausaha Agung.']


.. code:: ipython3

    pprint(model.summarize([string], mode = 'ringkasan', decoder = 'beam'))


.. parsed-literal::

    ['- Tun M tidak lagi menjadi Pengerusi Bersatu, tetapi dia masih mempunyai '
     'pekerjaan. Bekas setiausaha Agung Bersatu, Marzuki Yahya mengatakan bahawa '
     'dia tidak faham bagaimana JPPM mengatakan peletakan jawatan itu sah, Times. '
     '"Jadi ini agak berlawanan dengan keputusan yang kita sudah buat," katanya. '
     '"Saya tidak faham bagaimana Jabatan Pendaftar Pertubuhan Malaysia mengatakan '
     'peletakan jawatan itu sah sedangkan kita sudah membuat keputusan di dalam '
     'mesyuarat, bukan seorang dua yang membuat keputusan. " Marzuki mengatakan '
     'bahawa dia tidak faham bagaimana JPPM mengatakan peletakan jawatan itu sah. '
     '"Apa locus standy yang ada pada Setiausaha Kerja untuk membawa perkara ini '
     'kepada J. Seharusnya ia dibawa kepada Setiausaha Agung sebagai pentadbir '
     'kepada parti," katanya.']


.. code:: ipython3

    pprint(model.summarize([string], mode = 'ringkasan', decoder = 'nucleus', top_p = 0.7))


.. parsed-literal::

    ['- Seorang bekas setiausaha agung Uc yang mengatakan bahawa dia tidak mahu '
     'jawatan Tun Dr Mahathir sebagai Pengerusi Bersatu pada mesyuarat khas Majlis '
     'Pimpinan Tertinggi pada 24 Februari mengatakan bahawa dia "agak berlawanan '
     'dengan keputusan yang kita sudah buat. " Dia mengatakan bahawa tidak ada '
     '"ketidakmampuan" Jabatan Kelayakan Pertubuhan Malaysia - yang menyatakan '
     'bahawa Dr Mahathir berhenti atau dipecat - kerana ia sudah diputuskan pada '
     'peringkat parti yang dipersetujui semua termasuk Presiden, Tan Sri Muhyiddin '
     'Yassin, laporan Daily Metro. Sekiranya peletakan jawatan itu sah, itu harus '
     'diputuskan pada peringkat parti yang dipersetujui semua termasuk Presiden, '
     'Tan Sri Muhyiddin Yassin. "Saya tidak faham bagaimana Jabatan Pendaftar '
     'Pertubuhan Malaysia (JPPM) kata peletakan jawatan itu sah sedangkan kita '
     'sudah membuat keputusan di dalam mesyuarat, bukan seorang dua yang membuat '
     'keputusan," kata Marzuki Yahya. "Semua keputusan mesti dibuat melalui parti. '
     'Walau apa pun perbincangan dibuat di luar dari keputusan mesyuarat, ini '
     'bukan keputusan parti. " Dia menambahkan bahawa "apa locus standy yang ada '
     'pada Setiausaha Kerja untuk membawa perkara ini kepada J. Seharusnya ia '
     'dibawa kepada Setiausaha Agung sebagai pentadbir kepada parti. " Di tengah '
     'pergolakan politik pada akhir Februari adalah sah, Marzuki mengatakan '
     'kedudukan Dr Mahathir yang sebenarnya ditolak adalah sah. "Kenyataan media '
     'bukanlah keputusan rasmi," katanya. "Walaupun kita membuat 1,000 kenyataan '
     'sekali pun ia tetap tidak pendahuluan keputusan yang sudah dibuat di dalam '
     'mesyuarat. Kami catat di dalam minit apa yang berlaku di dalam mesyuarat.']

