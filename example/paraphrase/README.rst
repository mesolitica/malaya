Paraphrase
==========

.. container:: alert alert-info

   This tutorial is available as an IPython notebook at
   `Malaya/example/paraphrase <https://github.com/huseinzol05/Malaya/tree/master/example/paraphrase>`__.

.. code:: ipython3

    %%time
    
    import malaya
    from pprint import pprint


.. parsed-literal::

    CPU times: user 5.17 s, sys: 1.02 s, total: 6.19 s
    Wall time: 7.38 s


List available T5 models
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    malaya.paraphrase.available_t5()


.. parsed-literal::

    INFO:root:tested on 1k paraphrase texts.




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
          <th>BLEU</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>small</th>
          <td>122.0</td>
          <td>355.6</td>
          <td>0.81801</td>
        </tr>
        <tr>
          <th>base</th>
          <td>448.0</td>
          <td>1300.0</td>
          <td>0.86698</td>
        </tr>
      </tbody>
    </table>
    </div>



Load T5 models
~~~~~~~~~~~~~~

.. code:: python

   def t5(model: str = 'base', compressed: bool = True, **kwargs):

       """
       Load T5 model to generate a paraphrase given a string.

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
       result: malaya.model.t5.PARAPHRASE class
       """

**For malaya-gpu user, compressed t5 very fragile and we suggest use
``compressed=False``. Uncompressed model also can utilise GPU usage more
efficient**.

.. code:: ipython3

    t5 = malaya.paraphrase.t5()


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/paraphrase.py:84: load (from tensorflow.python.saved_model.loader_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.loader.load or tf.compat.v1.saved_model.load. There will be a new function for importing SavedModels in Tensorflow 2.0.
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/paraphrase/t5/base/model/variables/variables


Paraphrase simple string
~~~~~~~~~~~~~~~~~~~~~~~~

To paraphrase, simply use ``paraphrase`` method.

.. code:: ipython3

    string = "Beliau yang juga saksi pendakwaan kesembilan berkata, ia bagi mengelak daripada wujud isu digunakan terhadap Najib."
    pprint(string)


.. parsed-literal::

    ('Beliau yang juga saksi pendakwaan kesembilan berkata, ia bagi mengelak '
     'daripada wujud isu digunakan terhadap Najib.')


.. code:: ipython3

    pprint(t5.paraphrase(string))


.. parsed-literal::

    ('Ini juga bagi mengelakkan wujud isu yang digunakan terhadap Najib, kata '
     'saksi kesembilan.')


Paraphrase longer string
~~~~~~~~~~~~~~~~~~~~~~~~

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


T5 model
^^^^^^^^

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

List available LM Transformer models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Problem with T5 models, it built on top of mesh-tensorflow, so the input
must size of 1. So we use Tensor2Tensor library to train exact model as
T5 with dynamic size of batch.

**But, we found out, our pretrained LM Transformer not good as T5**, we
might skipped some literature in t5 papers.

.. code:: ipython3

    malaya.paraphrase.available_transformer()


.. parsed-literal::

    INFO:root:tested on 1k paraphrase texts.




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
          <th>BLEU</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>small</th>
          <td>379.0</td>
          <td>0.5534</td>
        </tr>
        <tr>
          <th>base</th>
          <td>832.0</td>
          <td>0.5970</td>
        </tr>
      </tbody>
    </table>
    </div>



Load Transformer
~~~~~~~~~~~~~~~~

.. code:: ipython3

    model = malaya.paraphrase.transformer()

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

   def paraphrase(
       self,
       strings: List[str],
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
    splitted = malaya.text.function.split_into_sentences(string)

.. code:: ipython3

    model.paraphrase([' '.join(splitted[:2])], decoder = 'greedy')




.. parsed-literal::

    ['PELETAKAN pengunduran Tun Dr. Mahathir sebagai ketua Parti Pribumi Bersatu Malaysia (Bersatu) dibincangkan pada 24 Februari lalu di dalam mesyuarat khas Majlis Pimpinan Tertinggi (MPT), dan tidak ada keraguan bahawa peletakan jawatan itu sah atau tidak, kerana ia sudah diputuskan pada peringkat parti yang menyetujui semua Presiden, Tan Sri Muhyiddin Yassin.']



.. code:: ipython3

    model.paraphrase([' '.join(splitted[:2])], decoder = 'beam')




.. parsed-literal::

    ['PELETAKAN pengunduran Tun Dr. Mahathir sebagai ketua Parti Pribumi Bersatu Malaysia (Bersatu) dibincangkan pada 24 Februari lalu di dalam mesyuarat khas Majlis Pimpinan Tertinggi (MPT), dan tentu saja tidak ada keraguan bahawa peletakan jawatan itu sah atau tidak dibuat pada peringkat parti yang menyetujui semua, termasuk Presiden, Tan Sri Muhyiddin Yassin.']



.. code:: ipython3

    model.paraphrase([' '.join(splitted[:2])], decoder = 'nucleus', top_p = 0.7)




.. parsed-literal::

    ['PELETAKAN pengunduran Tun Dr. Mahathir sebagai ketua Parti Pribumi Bersatu Malaysia (Bersatu) dibincangkan pada 24 Februari lalu di dalam mesyuarat Majlis Pimpinan Tertinggi (MPT), dan tidak ada persoalan bahawa peletakan jawatan itu sah atau tidak, kerana telah diputuskan pada peringkat parti yang menyetujui semua, termasuk Presiden, Tan Sri Muhyiddin Yassin.']


