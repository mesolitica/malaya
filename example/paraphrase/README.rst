.. code:: ipython3

    %%time
    
    import malaya


.. parsed-literal::

    CPU times: user 4.88 s, sys: 1.29 s, total: 6.17 s
    Wall time: 7.82 s


List available T5 models
------------------------

.. code:: ipython3

    malaya.paraphrase.available_t5()




.. parsed-literal::

    {'small': ['122MB', '*BLEU: 0.953'], 'base': ['448MB', '*BLEU: 0.953']}



\* We purposely peaked test set for T5 models because we tested T5
models are more powerful than encoder-decoder transformer, so we train
on 100% dataset.

List available Transformer models
---------------------------------

.. code:: ipython3

    malaya.paraphrase.available_transformer()




.. parsed-literal::

    {'tiny': ['18.4MB', 'BLEU: 0.594'],
     'small': ['43MB', 'BLEU: 0.737'],
     'base': ['234MB', 'BLEU: 0.792'],
     'tiny-bert': ['60.6MB', 'BLEU: 0.609'],
     'bert': ['449MB', 'BLUE: 0.696']}



We tested on 5k paraphrases.

Load Transformer models
-----------------------

.. code:: ipython3

    transformer = malaya.paraphrase.transformer()
    transformer_tiny = malaya.paraphrase.transformer(model = 'tiny')
    transformer_small = malaya.paraphrase.transformer(model = 'small')


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:54: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:55: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:49: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


Load T5 models
--------------

.. code:: ipython3

    t5 = malaya.paraphrase.t5()


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/paraphrase.py:84: load (from tensorflow.python.saved_model.loader_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.loader.load or tf.compat.v1.saved_model.load. There will be a new function for importing SavedModels in Tensorflow 2.0.
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/paraphrase/t5/base/model/variables/variables


Paraphrase simple string
------------------------

To paraphrase, simply use ``paraphrase`` method.

.. code:: ipython3

    from pprint import pprint

.. code:: ipython3

    string = "Beliau yang juga saksi pendakwaan kesembilan berkata, ia bagi mengelak daripada wujud isu digunakan terhadap Najib."
    pprint(string)


.. parsed-literal::

    ('Beliau yang juga saksi pendakwaan kesembilan berkata, ia bagi mengelak '
     'daripada wujud isu digunakan terhadap Najib.')


.. code:: ipython3

    %%time
    
    pprint(transformer.paraphrase(string))


.. parsed-literal::

    ('Dia yang juga merupakan seorang saksi pendakwaan kesembilan berkata, bagi '
     'mengelak daripada isu digunakan terhadap Najib.')
    CPU times: user 20 s, sys: 7.43 s, total: 27.4 s
    Wall time: 7.14 s


.. code:: ipython3

    %%time
    
    pprint(transformer_tiny.paraphrase(string))


.. parsed-literal::

    ('Dia juga seorang saksi pendakwaan kesembilan berkata, ia bagi mengelak '
     'daripada wujud isu digunakan terhadap Najib.')
    CPU times: user 1.54 s, sys: 1.17 s, total: 2.7 s
    Wall time: 896 ms


.. code:: ipython3

    %%time
    
    pprint(transformer_small.paraphrase(string))


.. parsed-literal::

    ('Dia juga seorang saksi pendakwaan kesembilan berkata, ia bagi mengelak '
     'daripada wujud isu digunakan terhadap Najib.')
    CPU times: user 3.41 s, sys: 2.52 s, total: 5.94 s
    Wall time: 1.72 s


.. code:: ipython3

    pprint(t5.paraphrase(string))


.. parsed-literal::

    ('Ini juga bagi mengelakkan wujud isu yang digunakan terhadap Najib, kata '
     'saksi kesembilan.')


Paraphrase longer string
------------------------

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
    pprint(string)


.. parsed-literal::

    ('PELETAKAN jawatan Tun Dr Mahathir Mohamad sebagai Pengerusi Parti Pribumi '
     'Bersatu Malaysia (Bersatu) ditolak di dalam mesyuarat khas Majlis Pimpinan '
     'Tertinggi (MPT) pada 24 Februari lalu. Justeru, tidak timbul soal peletakan '
     'jawatan itu sah atau tidak kerana ia sudah pun diputuskan pada peringkat '
     'parti yang dipersetujui semua termasuk Presiden, Tan Sri Muhyiddin Yassin. '
     'Bekas Setiausaha Agung Bersatu Datuk Marzuki Yahya berkata, pada mesyuarat '
     'itu MPT sebulat suara menolak peletakan jawatan Dr Mahathir. "Jadi ini agak '
     'berlawanan dengan keputusan yang kita sudah buat. Saya tak faham bagaimana '
     'Jabatan Pendaftar Pertubuhan Malaysia (JPPM) kata peletakan jawatan itu sah '
     'sedangkan kita sudah buat keputusan di dalam mesyuarat, bukan seorang dua '
     'yang buat keputusan. "Semua keputusan mesti dibuat melalui parti. Walau apa '
     'juga perbincangan dibuat di luar daripada keputusan mesyuarat, ini bukan '
     'keputusan parti. "Apa locus standy yang ada pada Setiausaha Kerja untuk '
     'membawa perkara ini kepada JPPM. Seharusnya ia dibawa kepada Setiausaha '
     'Agung sebagai pentadbir kepada parti," katanya kepada Harian Metro. Beliau '
     'mengulas laporan media tempatan hari ini mengenai pengesahan JPPM bahawa Dr '
     'Mahathir tidak lagi menjadi Pengerusi Bersatu berikutan peletakan jawatannya '
     'di tengah-tengah pergolakan politik pada akhir Februari adalah sah. Laporan '
     'itu juga menyatakan, kedudukan Muhyiddin Yassin memangku jawatan itu juga '
     'sah. Menurutnya, memang betul Dr Mahathir menghantar surat peletakan '
     'jawatan, tetapi ditolak oleh MPT. "Fasal yang disebut itu terpakai sekiranya '
     'berhenti atau diberhentikan, tetapi ini mesyuarat sudah menolak," katanya. '
     'Marzuki turut mempersoal kenyataan media yang dibuat beberapa pimpinan parti '
     'itu hari ini yang menyatakan sokongan kepada Perikatan Nasional. "Kenyataan '
     'media bukanlah keputusan rasmi. Walaupun kita buat 1,000 kenyataan sekali '
     'pun ia tetap tidak merubah keputusan yang sudah dibuat di dalam mesyuarat. '
     'Kita catat di dalam minit apa yang berlaku di dalam mesyuarat," katanya.')


Transformer model
^^^^^^^^^^^^^^^^^

For transformer model,

.. code:: python

   def paraphrase(
       self, string: str, beam_search: bool = True, split_fullstop: bool = True
   ):
       """
       Paraphrase a string.

       Parameters
       ----------
       string : str
       beam_search : bool, (optional=True)
           If True, use beam search decoder, else use greedy decoder.
       split_fullstop: bool, (default=True)
           if True, will generate paraphrase for each strings splitted by fullstop.

       Returns
       -------
       result: str
       """
       

We can choose to use greedy decoder or beam decoder. Again, beam decoder
is really slow.

.. code:: ipython3

    pprint(transformer.paraphrase(string, beam_search = False))


.. parsed-literal::

    ('PELETAKAN Tun Dr. Mahathir Mohamad sebagai ketua Parti Pribumi Bersatu '
     'Malaysia (Bersatu) ditolak pada 24 Februari lalu di dalam mesyuarat khas '
     'Majlis Pimpinan Tertinggi (MPT) . Justeru, tidak timbul peletakan jawatan '
     'itu sah atau tidak kerana ia sudah pun diputuskan pada peringkat parti yang '
     'dibenarkan semua termasuk Presiden, Sri Muhyiddin Yassin. Bekas Setiausaha '
     'Agung Bersatu Datuk Marzuki Yahya, berkata pada mesyuarat tahun 1970-an, '
     'Msebulat suara telah menolak peletakan jawatan Dr Mahathir. "Jadi ini agak '
     'berlawanan dengan tegas keputusan yang kita sudah buat." peletakan jawatan '
     'itu sah, sementara kita sudah buat keputusan di dalam mesyuarat, bukan '
     'seorang dua yang buat keputusan, kata Demokrat ejen Jabatan Tuntutan '
     'menolegar dari Malaysia. "Semua keputusan" mesti dibuat melalui parti. '
     'Namun, apa juga perbincangan di luar daripada keputusan berikutnya, ini '
     'bukan keputusan Parti. Setiausaha kerja itu bunga untuk membawa perkara itu '
     'kepada JPPM. Seharusnya dilaporkan kepada Setiausaha Negara sebagai '
     'pentadbir parti itu, kata Harian kepada Setiausaha Agung. Dia mengulas '
     'mengenai rakan-rakan pelajar media tempatan pada akhir Februari, mengenai '
     'pengesahan JPM bahawa Dr. Mahathir tidak lagi menjadi Pengerusi Bersatu '
     'setelah peletakan jawatan di tengah-tengah pergolakan politik. Di sini '
     'Laporan itu juga, kedudukan Muhyiddin Yassin memangku jawatan itu juga sah. '
     '"Pada hari Khamis, Dr. Mahathir menghantar surat peletakan jawatan, tetapi '
     'ditolak oleh MPT. "Fasal yang disebut itu digunakan sekiranya berhenti atau '
     'diberhentikan, tetapi ini mesyuarat sudah menolak," katanya. Marzuki turut '
     'mempersoal kenyataan media, yang dibuat beberapa pimpinan Parti hari ini '
     'yang menyatakan sokongan kepada Perikatan Nasional. `` Kenyataan media '
     'bukanlah keputusan rasmi. Walaupun kita buat 1,000 kenyataan sekali pun ia '
     'tetap tidak membenarkan keputusan yang sudah dibuat di perjumpaan. Tetapi '
     'kita catat di dalam minit apa yang berlaku di dalam mesyuarat, "kata Gemi.')


You can see ``Gemi`` out-of-context, this is because the model trying to
predict who is ``katanya``, so it simply pulled random name from
training set. To solve this problem, you need to do sliding windows. If
we have 5 strings, simply give [s1, s2], [s2, s3] and so on the model,
at least the model got some context from previous string.

.. code:: ipython3

    pprint(transformer_small.paraphrase(string, beam_search = False))


.. parsed-literal::

    ('PELETAKAN jawatan Tun Dr Mahathir Mohamad sebagai Pengerusi Parti Pribumi '
     'Bersatu Malaysia (Bersatu) ditolak di dalam mesyuarat khas Majlis Pimpinan '
     'Tertinggi (Mahathir) pada 24 Februari lalu. Justeru, tidak timbul pertanyaan '
     'mengenai jawatan itu sah atau tidak kerana ia sudah pun diputuskan pada '
     'peringkat parti yang secara umum kepemimpinan termasuk Presiden, Tan Sri '
     'Muhyiddin Yassin. Bekas Setiausaha Agung Bersatu walikota Marzuki Yahya, '
     'berkata pada mesyuarat itu MPT sebulat suara menolak perselasi Mahathir. '
     '"Jadi ini agak menangkan dengan keputusan yang kita sudah buat. Saya tak '
     'bagaimana seorang Demokrat untuk memindahkan memindahkan memindahkan '
     'memindahkan memindahkan memindahkan memindahkan memindahkan kata-kata pada '
     'hari Isnin, sementara kita sudah buat keputusan di dalam pertemuan, bukan '
     'seorang dua yang buat keputusan. "Semua keputusan harus dibuat melalui '
     'parti. Namun, apa juga perbincangan dibuat di luar daripada keputusan parti, '
     'ini bukan keputusan. "Apa locus standy, yang ada pada setiausaha ketaran, '
     'untuk membawa perkara ini kepada JPM. Seharusnya, seorang pentadbir Amerika '
     'Syarikat untuk parti, seorang pentadbir bekerja sebagai Harian, seorang '
     'lelaki. Dia mengatakan seorang lelaki bernama mengulas laporan media '
     'tempatan mengenai pengesahan JPPM bahawa Dr. Mahathir tidak lagi menjadi '
     'ketua Bersatu setelah peletakan jawatan di tengah-tengah amalan politik pada '
     'akhir Februari adalah sah. Ia juga menyatakan, kedudukan Muhyiddin Yassin '
     'memangku jawatan itu juga sah. Sepertasi Dr. Mahathir, dia membuat surat '
     'perganti dengan Mahathir, tetapi ditolak oleh MPT. "Fasal yang disebut '
     'terpakai jika berhenti atau diberhentikan, tetapi ini pertemuan sudah '
     'menolak," katanya. Marzuki mengambil penjelasan media yang beberapa pimpinan '
     'parti itu pada hari ini, yang mengumumkan sokongan kepada Perikatan '
     'Nasional. "Kenyataan media bukanlah keputusan rasmi. Walaupun kita buat '
     '1,000 kenyataan di mana ia tetap tidak merubah keputusan yang dibuat di '
     'dalam mesyuarat. Kami catat di dalam minit apa yang berlaku di dalam '
     'pertemuan ini.')


We can see transformer-model brings ``Amerika Syarikat`` to the context,
which is not really make sense.

What if I tried to paraphrase entire string without split it into
substrings?

.. code:: ipython3

    pprint(transformer.paraphrase(string, beam_search = False, split_fullstop = False))


.. parsed-literal::

    ('Tetapi, pada mesyuarat Sabtu, dia mendakwa bahawa peletakan jawatan '
     'Pengerusi dan membawanya ke parti-300, setelah keputusan itu tidak sah, dan '
     'panggilan pengawas berlangsung sekaligus oleh pihak berkuasa tetapi '
     'keputusan berpecah belah di sini, walikota Belongnya, tidak akan pernah '
     'dilihat sebagai alasan, tetapi jika tidak ada, dia tidak dapat dilihat pada '
     'mesyuarat penetapan lapisan media di sini untuk jawatan lain.')


It pulled out-of-context related to the string from the training set,
which is not make any sense.

T5 model
^^^^^^^^

In T5, we cannot choose to use greedy decoder or beam decoder.

.. code:: ipython3

    pprint(t5.paraphrase(string))


.. parsed-literal::

    ('Peletakan jawatan Tun Dr. Mahathir sebagai Pengerusi Parti Pribumi Bersatu '
     'Malaysia ditolak di dalam mesyuarat khas MPT (Parti Pimpinan Tertinggi) pada '
     '24 Februari lalu. Tidak kira sama ada peletakan jawatan itu sah atau tidak, '
     'kerana sudah pun dinyatakan pada peringkat parti yang menyetujui semua, '
     'termasuk Presiden Tan Sri Muhyiddin Yassin. Saya telah mencadangkan kepada '
     'MPT untuk sebulat suara, kata Marzuki. " Ini semua dibuat berlawanan dengan '
     'keputusan yang kita sudah membuat. Saya tak faham bagaimana Jabatan '
     'Pendaftaran Pertubuhan Malaysia (JPPM) mengatakan bahawa peletakan jawatan '
     'itu sah, padahal kita sudah membuat keputusan di dalam pertemuan, bukan dua '
     'orang yang membuat keputusan. " Semua keputusan mesti dibuat oleh parti. '
     'Tetapi keputusan untuk memberi kesan kepada perbincangan, itu bukan '
     'keputusan parti. " Apa yang berlaku di locus standy Setiausaha Kerja untuk '
     'membawa perkara ini ke JPPM. Seharusnya diserahkan kepada Setiausaha Agung '
     'sebagai pentadbir kepada parti, kata Bruno. Namun, dia menambah laporan '
     'media tempatan hari ini mengenai pengesahan JPPM bahawa Dr Mahathir tidak '
     'lagi menjadi pemimpin Bersatu setelah peletakan jawatan di tengah-tengah '
     'pergolakan politik pada akhir Februari. Kedudukan Muhyiddin juga sah, kata '
     'laporan itu. Dia mengatakan bahawa memang betul Tun Mahathir menyerahkan '
     'surat peletakan jawatan tetapi ditolak oleh MPT. "Fasal yang disebut itu '
     'terpakai sekiranya berhenti atau diberhentikan, tetapi ini pertemuan sudah '
     'menolak," katanya. Sebaliknya, Marzuki mempersoalkan kenyataan media '
     'beberapa pimpinan parti hari ini yang menyatakan sokongan kepada Perikatan '
     'Nasional. Akhbar bebas sepenuhnya menentukan keputusan rasmi mereka. '
     'Walaupun kami membuat 1,000 kenyataan, ia tetap tidak mengubah keputusan '
     'yang sudah dibuat di dalam mesyuarat. Kami mengambil minit apa yang berlaku '
     'di dalam mesyuarat ini, "kata Griffin.')


You can see ``Griffin`` out-of-context, this is because the model trying
to predict who is ``katanya``, so it simply pulled random name from
training set. To solve this problem, you need to do sliding windows. If
we have 5 strings, simply give [s1, s2], [s2, s3] and so on the model,
at least the model got some context from previous string.

.. code:: ipython3

    pprint(t5.paraphrase(string, split_fullstop = False))


.. parsed-literal::

    ('Kedudukan Dr. Mahathir sebagai Pengerusi Parti Pribumi Bersatu Malaysia '
     '(Bersatu) ditolak di dalam mesyuarat khas Majlis Pimpinan Tertinggi (MPT) '
     'pada 24 Februari, dan bahawa posisi beliau memangku melalui parti, dan '
     'bahawa semua keputusan harus diambil oleh MPT, dan itu bukan keputusan '
     'parti, tetapi sebagai kenyataan media, kata Marzuki, pemimpin yang '
     'berpangkalan di Perlis, yang ditemui pada hari Jumaat. "')


When you try to paraphrase entire string, the output is quite good, a
summary!

