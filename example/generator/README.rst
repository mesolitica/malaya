Generator
=========

.. container:: alert alert-info

   This tutorial is available as an IPython notebook at
   `Malaya/example/generator <https://github.com/huseinzol05/Malaya/tree/master/example/generator>`__.

.. code:: ipython3

    %%time
    import malaya
    from pprint import pprint


.. parsed-literal::

    CPU times: user 5 s, sys: 718 ms, total: 5.72 s
    Wall time: 4.87 s


List available T5 Model
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    malaya.generator.available_t5()




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
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>small</th>
          <td>122</td>
        </tr>
        <tr>
          <th>base</th>
          <td>448</td>
        </tr>
      </tbody>
    </table>
    </div>



Load T5
~~~~~~~

T5 in Malaya is quite unique, most of the text generative model we found
on the internet like GPT2 or Markov, simply just continue prefix input
from user, but not for T5 Malaya. We want to generate an article or
karangan like high school when the users give ‘isi penting’.

.. code:: python

   def t5(model: str = 'base', **kwargs):

       """
       Load T5 model to generate a string given a isu penting.

       Parameters
       ----------
       model : str, optional (default='base')
           Model architecture supported. Allowed values:

           * ``'base'`` - T5 Base parameters.
           * ``'small'`` - T5 Small parameters.

       Returns
       -------
       result: malaya.model.t5.GENERATOR class
       """

.. code:: ipython3

    model = malaya.generator.t5()


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/generator.py:510: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/generator.py:512: load (from tensorflow.python.saved_model.loader_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.loader.load or tf.compat.v1.saved_model.load. There will be a new function for importing SavedModels in Tensorflow 2.0.
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/generator-sample/t5/base/model/variables/variables


.. code:: ipython3

    isi_penting = ['Dr M perlu dikekalkan sebagai perdana menteri',
                  'Muhyiddin perlulah menolong Dr M',
                  'rakyat perlu menolong Muhyiddin']

I just want to test the model given this isi penting, because we all
know, Dr M and Muhyiddin are not supporting each others in the real
world.

generate
^^^^^^^^

``model.generate`` accepts list of strings.

.. code:: python

   def generate(self, strings: List[str]):
       """
       generate a long text given a isi penting.

       Parameters
       ----------
       strings: List[str]

       Returns
       -------
       result: str
       """

.. code:: ipython3

    pprint(model.generate(isi_penting))


.. parsed-literal::

    (': Presiden Bersatu, Tan Sri Muhyiddin Yassin perlu mengekalkan Tun Dr '
     'Mahathir Mohamad sebagai perdana menteri berbanding Datuk Seri Anwar Ibrahim '
     'yang hanya minta bantuan untuk menyelesaikan kemelut kedudukan '
     'negara.Muhyiddin berkata, ini kerana semua pihak tahu masalah yang dihadapi '
     'oleh Perdana Menteri adalah di luar bidang kuasa beliau sendiri.Katanya, '
     'Muhyiddin perlu membantu beliau kerana beliau percaya rakyat Malaysia tahu '
     'apa yang berlaku di luar bidang kuasa beliau."Apa yang berlaku di luar '
     'bidang kuasa Dr Mahathir... semua tahu bahawa ini berlaku di bawah '
     'kepimpinan Anwar."Muhyiddin dan seluruh rakyat yang tahu apa yang berlaku di '
     'Johor."Ini kerana di Johor ini, majoriti menteri-menteri dalam Pakatan '
     'Harapan banyak sangat ketua-ketua parti."Jadi Muhyiddin perlu bantu Dr '
     'Mahathir sebab rakyat tahu apa yang berlaku di Johor Bahru," katanya dalam '
     'satu kenyataan di sini, pada Jumaat.Dalam pada itu, Muhyiddin berkata, '
     'rakyat juga perlu menolong Muhyiddin untuk menyelesaikan masalah yang '
     'melanda negara ketika ini.Menurutnya, Muhyiddin perlu menggalas tugas dengan '
     'baik dan memastikan keadaan negara berada dalam keadaan baik.')


Pretty good!

.. code:: ipython3

    isi_penting = ['Neelofa tetap dengan keputusan untuk berkahwin akhir tahun ini',
                  'Long Tiger sanggup membantu Neelofa',
                  'Tiba-tiba Long Tiger bergaduh dengan Husein']

We also can give any isi penting even does not make any sense.

.. code:: ipython3

    pprint(model.generate(isi_penting))


.. parsed-literal::

    ('Kuala Lumpur: Pelakon, Neelofa tetap dengan keputusan dibuat untuk berkahwin '
     'penutup tahun ini, selepas mengadakan pertemuan dengan Long Tiger. Neelofa '
     'atau nama sebenarnya, Mohd Neelofa Ahmad Noor berkata, dia tidak pernah '
     'merancang untuk berkahwin, namun menegaskan dirinya lebih mengutamakan masa '
     'depan. "Saya seronok bersama keluarga. Kalau kami berkahwin awal tahun ini, '
     'ia mengambil masa yang lama. Itu impian saya tetapi biarlah, selepas setahun '
     'saya berehat, saya akan mula bekerja. "Jadi, apabila sering sesi pertemuan '
     'dengan Long Tiger, saya kena tegas mengenai perkara ini. Bukan soal nak '
     'memalukan diri sendiri tetapi siapa yang boleh menghentam saya," katanya '
     'kepada Bh Online. Dalam sesi pertemuan itu, Neelofa yang juga pengacara '
     'acara Top 5, bergaduh dengan Husein, dalam pergaduhan yang berlaku di '
     'Kompleks Mahkamah Tinggi Syariah di sini, baru-baru ini. Ditanya mengenai '
     'hubungannya dengan wanita itu, Neelofa berkata, mereka masih belum '
     'menyelesaikan perkara itu dengan baik. "Saya tidak tahu pasal semua ini, '
     'tetapi ia akan diselesaikan menerusi cara baik. Tidak kiralah apa yang kami '
     'tidak cakap pun. "Pada mulanya kami hanya mahu membebaskan mereka daripada '
     'sebarang isu, namun selepas beberapa hari bergaduh, kami akhirnya mengambil '
     'keputusan untuk berkahwin dengan Hadiza Aziz. "Jika mereka mahu, kami akan '
     'membendung, namun pada masa yang sama, kami tidak mahu bergaduh dengan '
     'lelaki yang digelar Long Tiger," katanya.')


How about karangan like high school?

.. code:: ipython3

    # http://mieadham86.blogspot.com/2016/09/isi-isi-penting-karangan-bahasa-melayu.html
    # KEBAIKAN AMALAN BERGOTONG-ROYONG
    
    isi_penting = ['Dapat memupuk semangat kerjasama',
                   'Dapat mengeratkan hubungan silaturahim.',
                   'Kebersihan kawasan persekitaran terpelihara.',
                   'Terhindar daripada wabak penyakit seperti Denggi',
                   'Mengisi masa lapang',
                   'Menerapkan nilai-nilai murni dalam kehidupan']

.. code:: ipython3

    pprint(model.generate(isi_penting))


.. parsed-literal::

    ('Dewasa ini, kes-kes seumpama denggi semakin menular di kalangan masyarakat. '
     'Justeru, individu yang bertanggungjawab dan berkesan perlu memainkan peranan '
     'penting dalam memastikan persekitaran dalam komuniti terjamin. Persis kata '
     'peribahasa Melayu, melentur buluh biarlah dari rebungnya. Oleh itu, tindakan '
     'yang wajar perlu diambil terutamanya jika kita mengamalkan sikap-sikap di '
     'dalam komuniti supaya kehidupan kita tidak terjejas. Oleh itu, kita perlu '
     'mengamalkan sikap bekerjasama dengan masyarakat dalam memastikan '
     'persekitaran kita selamat. Jika kita sehati, sikap bekerjasama dapat dipupuk '
     'dan dibudayakan dalam masyarakat. Maka, amalan ini secara tidak langsung '
     'mampu membantu kita supaya tidak hidup lebih sejahtera. Pada masa yang sama, '
     'ia juga dapat mengelakkan berlakunya sebarang masalah kesihatan dan '
     'seterusnya membantu yang mungkin akan berlaku pada masa akan datang. '
     'Masyarakat yang prihatin perlu meluahkan perasaan dan menitik beratkan soal '
     'kebersihan kawasan persekitaran. Bak kata peribahasa Melayu, mencegah lebih '
     'baik daripada merawat. Tamsilnya, pihak kerajaan perlu menjalankan usaha '
     'yang bersungguh-sungguh sebagai tanggungjawab yang diamanahkan. Selain itu, '
     'sikap masyarakat yang mengambil berat tentang kebersihan kawasan '
     'persekitaran dapat membantu mengurangkan masalah kesihatan yang kian '
     'menular. Secara tidak langsung, masyarakat awam akan melahirkan masyarakat '
     'yang peka dan menghargai keberadaan anggota masyarakat di sekeliling mereka. '
     'Bagi memastikan kebersihan kawasan persekitaran terjamin, kita perlu '
     'memastikan komuniti yang berada ditaarapkan dalam keadaan bersih dan terurus '
     'agar keselamatan masyarakat terjamin. Para pekerja dan ahli peniaga perlu '
     'memastikan kebersihan kawasan mereka dijaga dengan baik. Hal ini kerana, '
     'kita akan berhadapan dengan pelbagai masalah kesihatan yang mengakibatkan '
     'Malaysia menjadi negara ketiga yang paling teruk terkena jangkitan demam '
     'denggi pada tahun lepas. Sekiranya kita mempraktikkan amalan berkenaan, kita '
     'akan berhadapan dengan bahaya. Sekiranya aktiviti ini diteruskan, kita akan '
     'terencat daripada jumlah kes penyakit yang menyerang. Secara tidak langsung, '
     'kita akan dapat membendung penularan wabak penyakit di kalangan masyarakat. '
     'Sebagai contoh, wabak denggi di Malaysia berkemungkinan boleh menularkan '
     'jangkitan kepada penduduk di negeri-negeri yang lain. Oleh itu, langkah ini '
     'wajar dan mempunyai sistem pengurusan kebersihan yang terbaik bagi '
     'membolehkan jumlah pesakit yang dirawat di hospital meningkat. Kesannya, ia '
     'dapat membantu kita untuk mengamalkan kaedah yang betul dan matang dalam '
     'kehidupan. Selain itu, sekiranya kita mengamalkan sikap kerja, kita akan '
     'sentiasa berusaha supaya kita terhindar daripada wabak penyakit yang '
     'menyerang penduduk di sekeliling kita. Bak kata peribahasa Melayu, mencegah '
     'lebih baik daripada merawat. Semua pihak perlu berganding bahu bagai aur '
     'dengan tebing untuk menjaga kesihatan dan keselamatan para pekerja dalam '
     'kawasan yang sangat rentan. Kebersihan kawasan persekitaran merupakan elemen '
     'yang penting dalam memastikan persekitaran kita selamat daripada jangkitan '
     'wabak seperti denggi. Kita tentunya tidak mahu ada tempat yang kotor dan '
     'busuk namun kita tidak boleh berbuat demikian kerana ia merupakan elemen '
     'yang tidak boleh dijual beli. Oleh itu, jika kita mengamalkan sikap kerja '
     "yang 'membersihkan', kita akan menjadi lebih baik dan selamat daripada wabak "
     'penyakit seperti denggi. Jika kita mengamalkan sikap ini, kita akan menjadi '
     'lebih baik dan selamat daripada ancaman penyakit-penyakit yang berbahaya. '
     'Tidak kira apabila kita sudah terbiasa dengan amalan ini, sudah pasti '
     'keselamatan kita akan terjamin. Selain itu, kita perlulah dirikan amalan '
     'seperti rajin mencuci tangan menggunakan sabun atau segala benda lain kerana '
     'kita juga mempunyai tempat yang sesuai untuk membasuh tangan dengan baik. '
     'Perkara ini boleh menjadi perubahan kepada amalan kita dalam kehidupan '
     'apabila kita berusaha untuk membersihkan kawasan yang telah dikenal pasti. '
     'Secara tidak langsung, kita dapat bertukar-tukar fikiran dan mengamalkan '
     'nilai-nilai murni dalam kehidupan. Hal ini demikian kerana, kita antara '
     'mereka yang merancang untuk melakukan sesuatu bagi mengelakkan berlakunya '
     'kemalangan. Hakikatnya, amalan membasuh tangan menggunakan sabun atau benda '
     'lain adalah berniat buruk kerana akan dapat mengganggu kelancaran proses '
     'pemanduan terutamanya apabila tidur. Kesannya, kita akan mewujudkan '
     'masyarakat yang bertimbang rasa dan bergantung kepada orang lain untuk '
     'melakukan kerja mereka walaupun di mana mereka berada. Selain itu, kita '
     'dapat mengamalkan cara yang betul dalam memastikan kebersihan kawasan '
     'persekitaran adalah terjamin. Kita tidak boleh menyembunyikan diri daripada '
     'pengetahuan umum seperti di tempat awam seperti tempat letak kereta yang '
     'sering digunakan oleh orang ramai. Jika kita menggunakan tandas awam dan '
     'menggunakan botol air untuk membersihkan kawasan berkenaan, kita akan mudah '
     'terdedah dengan wabak penyakit yang membahayakan kesihatan. Selain itu, kita '
     'juga perlu sentiasa berjaga-jaga dengan memakai penutup mulut dan hidung '
     'jika ada demam. Jika kita tidak mengamalkan kebersihan, besar kemungkinan ia '
     'boleh mengundang kepada penularan wabak penyakit. Bak kata peribahasa '
     'Melayu, mencegah lebih baik daripada merawat. Jika kita membuat keputusan '
     'untuk menutup mulut atau hidung dengan pakaian yang bersih dan bijak, kita '
     'akan menjadi lebih baik daripada menyelamatkan diri sendiri daripada '
     'jangkitan penyakit. Andai kata, pengamal media dapat menggunakan telefon '
     'pintar ketika membuat liputan di media massa, proses ini akan membuatkan '
     'kehidupan mereka lebih mudah dan sukar. Selain itu, proses nyah kuman juga '
     'dapat memastikan kebersihan di kawasan rumah kita terjamin. Contohnya, semua '
     'stesen minyak dan restoran makanan segera perlu memakai penutup mulut dan '
     'hidung secara betul agar penularan wabak penyakit dapat dihentikan. Penonton '
     'yang berada di dalam juga wajar digalakkan untuk menggunakan penutup mulut '
     'dan hidung agar mudah terkena jangkitan kuman. Selain itu, pengisian masa '
     'lapang yang terdapat di kawasan tempat awam dapat mendidik masyarakat untuk '
     'mengamalkan nilai-nilai murni seperti rajin mencuci tangan menggunakan sabun '
     'dan air supaya tidak terdedah kepada virus denggi. Walaupun kita mempunyai '
     'ramai kenalan yang ramai tetapi tidak dapat mengamalkannya kerana kita perlu '
     'adalah rakan yang sedar dan memahami tugas masing-masing. Pelbagai cara yang '
     'boleh kita lakukan bagi memastikan hospital atau klinik-klinik kerajaan '
     'menjadi')


.. code:: ipython3

    # http://mieadham86.blogspot.com/2016/09/isi-isi-penting-karangan-bahasa-melayu.html
    # CARA MENJADI MURID CEMERLANG
    
    isi_penting = ['Rajin berusaha – tidak mudah putus asa',
                   'Menghormati orang yang lebih tua – mendapat keberkatan',
                   'Melibatkan diri secara aktif dalam bidang kokurikulum',
                   'Memberi tumpuan ketika guru mengajar.',
                   'Berdisiplin – menepati jadual yang disediakan.',
                   'Bercita-cita tinggi – mempunyai keazaman yang tinggi untuk berjaya']

.. code:: ipython3

    pprint(model.generate(isi_penting))


.. parsed-literal::

    ('Sejak akhir-akhir ini, pelbagai isu yang hangat diperkatakan oleh masyarakat '
     'yang berkait dengan sambutan Hari Raya Aidilfitri. Pelbagai faktor yang '
     'melatari perkara yang berlaku dalam kalangan masyarakat hari ini, khususnya '
     'bagi golongan muda. Dikatakan bahawa kehidupan kita hari ini semakin '
     'mencabar terutamanya kesibukan dalam menjalankan tugas dan mengajar. '
     'Justeru, tidak dinafikan apabila semakin jauh kita, semakin ramai yang '
     'memilih untuk lalai atau tidak mematuhi arahan yang telah ditetapkan. '
     'Mendepani cabaran ini, golongan muda terpaksa menempuhi segala cabaran untuk '
     'menjadi lebih baik dan lebih baik. Minda yang perlu diterapkan, terutama di '
     'dalam kelas untuk mempelajari ilmu pengetahuan. Jika tidak, kita akan '
     'menjadi lebih mudah untuk menilai dan menyelesaikan masalah yang dihadapi. '
     'Oleh itu, kita perlu berfikir untuk menetapkan langkah yang patut atau perlu '
     'dilaksanakan bagi mengatasi masalah yang berlaku. Selain itu, guru-guru juga '
     'harus mendidik peserta-peserta dalam kelas supaya dapat menjalankan kegiatan '
     'dengan lebih serius dan berkesan. Guru-Guru juga seharusnya berusaha untuk '
     'meningkatkan kemahiran mereka dalam kalangan pelajar. Seperti peribahasa '
     'Melayu, melentur buluh biarlah dari rebungnya. Setiap insan mempunyai '
     'peranan masing-masing dan tanggungjawab yang masing-masing. Kesempatan untuk '
     'memberikan nasihat dan teguran adalah lebih penting dan membantu secara '
     'halus dan bijaksana dalam melakukan sesuatu. Selain itu, guru-guru hendaklah '
     'berani untuk melakukan sesuatu perkara yang memberi manfaat kepada para '
     'pelajar yang lain. Cara ini adalah dengan melakukan aktiviti-aktiviti yang '
     'boleh memberi manfaat kepada para pelajar. Selain itu, guru-guru juga '
     'perlulah menjaga disiplin mereka dengan sebaik-baiknya. Dalam menyampaikan '
     'nasihat dan teguran secara berterusan, pelajar juga boleh melakukan perkara '
     'yang boleh mendatangkan mudarat. Anak-Anak awal pelajar dan rakan-rakan '
     'mereka juga boleh melakukan tugas yang bermanfaat. Keadaan ini membolehkan '
     'mereka untuk lebih berusaha dan memberikan nasihat yang berguna kepada kaum '
     'lain. Oleh itu, mereka perlu sentiasa mengingati dan mendidik pelajar dengan '
     'nilai-nilai yang murni. Setiap orang mempunyai impian yang tinggi untuk '
     'berjaya. Sama ada kita berjaya atau tidak, pencapaian yang diperoleh setelah '
     'tamat belajar akan memberikan kita nilai yang baik dan perlu menjadi contoh '
     'yang baik untuk negara kita.')


Load GPT2
~~~~~~~~~

Malaya provided Pretrained GPT2 model, specific to Malay, we called it
GPT2-Bahasa. This interface not able us to use it to do custom training.

GPT2-Bahasa was pretrained on ~0.9 billion words, and below is the list
of dataset we trained,

1. `dumping wikipedia
   (222MB) <https://github.com/huseinzol05/Malaya-Dataset#wikipedia-1>`__.
2. `local news
   (257MB) <https://github.com/huseinzol05/Malaya-Dataset#public-news>`__.
3. `local parliament text
   (45MB) <https://github.com/huseinzol05/Malaya-Dataset#parliament>`__.
4. `IIUM Confession
   (74MB) <https://github.com/huseinzol05/Malaya-Dataset#iium-confession>`__.
5. `Wattpad
   (74MB) <https://github.com/huseinzol05/Malaya-Dataset#wattpad>`__.
6. `Academia PDF
   (42MB) <https://github.com/huseinzol05/Malaya-Dataset#academia-pdf>`__.
7. `Common-Crawl
   (3GB) <https://github.com/huseinzol05/malaya-dataset#common-crawl>`__.

If you want to download pretrained model for GPT2-Bahasa and use it for
custom transfer-learning, you can download it here,
https://github.com/huseinzol05/Malaya/tree/master/pretrained-model/gpt2,
some notebooks to help you get started.

**Here we hope these models are not use to finetune for spreading fake
news**.

Or you can simply use
`Transformers <https://huggingface.co/models?filter=malay&search=gpt2>`__
to try GPT2-Bahasa models from Malaya, simply check available models
from here, https://huggingface.co/models?filter=malay&search=gpt2

.. code:: ipython3

    from IPython.core.display import Image, display
    
    display(Image('gpt2.png', width=500))



.. image:: load-generator_files/load-generator_23_0.png
   :width: 500px


load model
^^^^^^^^^^

GPT2-Bahasa only available ``117M`` and ``345M`` models.

1. ``117M`` size around 442MB.
2. ``345M`` is around 1.2GB.

.. code:: python

   def gpt2(
       model: str = '345M',
       generate_length: int = 256,
       temperature: float = 1.0,
       top_k: int = 40,
       **kwargs
   ):

       """
       Load GPT2 model to generate a string given a prefix string.

       Parameters
       ----------
       model : str, optional (default='345M')
           Model architecture supported. Allowed values:

           * ``'117M'`` - GPT2 117M parameters.
           * ``'345M'`` - GPT2 345M parameters.

       generate_length : int, optional (default=256)
           length of sentence to generate.
       temperature : float, optional (default=1.0)
           temperature value, value should between 0 and 1.
       top_k : int, optional (default=40)
           top-k in nucleus sampling selection.

       Returns
       -------
       result: malaya.transformers.gpt2.Model class
       """

.. code:: ipython3

    model = malaya.generator.gpt2(model = '117M')


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/gpt2/__init__.py:19: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/gpt2/__init__.py:140: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/gpt2/__init__.py:141: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/gpt2/__init__.py:142: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/gpt2/__init__.py:142: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.
    
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/gpt2/117M/gpt2-bahasa-117M/model.ckpt


.. code:: ipython3

    string = 'ceritanya sebegini, aku bangun pagi baca surat khabar berita harian, tetiba aku nampak cerita seram, '

generate
^^^^^^^^

``model.generate`` accepts a string.

.. code:: python

   def generate(self, string: str):
       """
       generate a text given an initial string.

       Parameters
       ----------
       string : str

       Returns
       -------
       result: str
       """

.. code:: ipython3

    print(model.generate(string))


.. parsed-literal::

    ceritanya sebegini, aku bangun pagi baca surat khabar berita harian, tetiba aku nampak cerita seram, ara aku yang lain keluar, aku pandang cerita tapi tak ingat, aku takut dan bimbang aku terpaksa marah kerana hati aku yang berada di sekeliling aku tadi tak putus-putus.
    Dalam diam, aku juga merasa kagum dan terharu bila aku bangun pagi untuk bangun dan tengok kisah seram ni, masa tu aku terus pandang, bila aku berada dalam bilik yang indah, aku tahu tentang benda yang nak diperkatakan.
    “Tu sikit, dengan banyak masa aku nak keluar dan keluar aku dah mula bangun pagi, aku nak keluar lagi, lepas tu nanti terus masuk ke bilik sambil nampak benda yang tak ada yang nak diperkatakan.
    Tak tau cerita tu macam benda yang boleh aku buat kalau rasa macam cerita.
    Sampai di bilik, aku pun rasa macam, benda yang nak diperkatakan tu bukan benda yang perlu aku buat.
    Macam tak percaya apa yang aku buat ni?
    Mungkin benda yang nak diperkatakan itu boleh buat aku jugak, cuma benda yang boleh bagi aku kata tak logik atau memang betul.
    Cuma yang paling aku nak cakap ni adalah benda pelik yang aku fikir nak nampak yang tak boleh dan kalau tak logik pun tak patut.
    So, apa kata dorang mainkan benda yang aku cakap ni.
    Rasa pelik dan amat pelik kan?
    Macam nak buat orang lain jadi macam benda pelik dan susah sangat nak buat


.. code:: ipython3

    model = malaya.generator.gpt2(model = '345M')


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/gpt2/345M/gpt2-bahasa-345M/model.ckpt


.. code:: ipython3

    string = 'ceritanya sebegini, aku bangun pagi baca surat khabar berita harian, tetiba aku nampak cerita seram, '
    print(model.generate(string))


.. parsed-literal::

    ceritanya sebegini, aku bangun pagi baca surat khabar berita harian, tetiba aku nampak cerita seram, omputeh-uteh cerita lama-lama, seram tak boleh bayang
    Sebelum kejadian, dalam 2 jam aku buat panggilan polis , lepas tu kira la sendiri nak ke lokasi.
    Tengok cerita lama..
    Sekarang ni, apa yang aku lalui, kita yang jaga diri, kita yang jaga kesihatan dan juga kita yang jaga minda dalam hidup.
    Maka, inilah jalan penyelesaian terbaiknya.
    Jangan lupakan manusia
    Orang yang paling ditakuti untuk berjaya dalam hidup, tidak akan jumpa yang tersayang!
    Jangan rosakkan masa depannya, ingatlah apa yang kita nak buat, walaupun pahit untuk ditelan.
    Jangan lupakan orang lain - masa depan mereka.
    Jangan lupakan orang - masa itulah kita yang lebih dicintai.
    Jangan lupakan orang - orang yang kita sayang, mereka bukan orang yang tersayang!
    Jangan lupakan orang - orang yang kita cinta, mereka cinta pada kita.
    Jangan lupakan diri - diri kita - yang kita punya, yang kita tinggal adalah masa lalu kita.
    Jangan lupakan orang lain - orang yang kita cinta, lebih indah dari masa lalu kita.
    Jangan lupakan semua orang - orang yang tinggal ataupun hidup.
    Jangan cuba lupakan diri kita - kerja keras dan selalu ada masa depan kita.
    Jangan pernah putus rasa - kecewa kerana kita telah banyak berubah.
    Jangan pernah putus putus asa kerana kita


Load Transformer
~~~~~~~~~~~~~~~~

We also can generate a text like GPT2 using Transformer-Bahasa. Right
now only supported BERT, ALBERT and ELECTRA.

.. code:: python

   def transformer(
       string: str,
       model,
       generate_length: int = 30,
       leed_out_len: int = 1,
       temperature: float = 1.0,
       top_k: int = 100,
       burnin: int = 15,
       batch_size: int = 5,
   ):
       """
       Use pretrained transformer models to generate a string given a prefix string.
       https://github.com/nyu-dl/bert-gen, https://arxiv.org/abs/1902.04094

       Parameters
       ----------
       string: str
       model: object
           transformer interface object. Right now only supported BERT, ALBERT.
       generate_length : int, optional (default=256)
           length of sentence to generate.
       leed_out_len : int, optional (default=1)
           length of extra masks for each iteration. 
       temperature: float, optional (default=1.0)
           logits * temperature.
       top_k: int, optional (default=100)
           k for top-k sampling.
       burnin: int, optional (default=15)
           for the first burnin steps, sample from the entire next word distribution, instead of top_k.
       batch_size: int, optional (default=5)
           generate sentences size of batch_size.

       Returns
       -------
       result: List[str]
       """

.. code:: ipython3

    electra = malaya.transformer.load(model = 'electra')


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:56: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/modeling.py:240: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.Dense instead.
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow_core/python/layers/core.py:187: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.__call__` method instead.
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:79: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:93: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/sampling.py:26: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:114: multinomial (from tensorflow.python.ops.random_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.random.categorical` instead.
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:117: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:118: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:120: The name tf.get_collection is deprecated. Please use tf.compat.v1.get_collection instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:121: The name tf.GraphKeys is deprecated. Please use tf.compat.v1.GraphKeys instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:127: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:129: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/electra-model/base/electra-base/model.ckpt


.. code:: ipython3

    malaya.generator.transformer(string, electra)


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/babble.py:30: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
    




.. parsed-literal::

    ['ceritanya sebegini , aku bangun pagi baca surat khabar berita harian , tetiba aku nampak cerita seram , seriuslah Allah tarik balik rezeki aku untuk kau berjumpa balik . patutlah terpentak apabila tiba masa kita baru perasan kejadian begitu , tapi nyata rupanya . Begitulah kehidupan',
     'ceritanya sebegini , aku bangun pagi baca surat khabar berita harian , tetiba aku nampak cerita seram , rupanya ada segelintir pihak yang tak faham bahasa Melayu berbalas budi . Kisah ringkas , Kisah ringkas , Kisah kisah ringkas , Kisah akhir cerita , Kisah kematian .',
     'ceritanya sebegini , aku bangun pagi baca surat khabar berita harian , tetiba aku nampak cerita seram , kenapa la cara bunuh diri tu mangkuk , orang baru terpengaruh dengan isu kononnya anak anak mangsa bunuh diri , mana tahu tau apa sebenar dosa orang itu sebenar .',
     'ceritanya sebegini , aku bangun pagi baca surat khabar berita harian , tetiba aku nampak cerita seram , diri yang gelap , menyedihkan , menyedihkan , remaja yang kaya , miskin , berkelulusan SPM dan masih hidup lagi . Alhamdulillah Allah berikan kekuatan kami semua sahabat semua',
     'ceritanya sebegini , aku bangun pagi baca surat khabar berita harian , tetiba aku nampak cerita seram , filem yang kerap diorang tayang dalam bahasa sedih . Lagi - lagi , aku rasa seram sebab aku tak sangka kalau korang cerita seram dia hilang tu cerita seram .']



ngrams
~~~~~~

You can generate ngrams pretty easy using this interface,

.. code:: python

   def ngrams(
       sequence,
       n: int,
       pad_left = False,
       pad_right = False,
       left_pad_symbol = None,
       right_pad_symbol = None,
   ):
       """
       generate ngrams.

       Parameters
       ----------
       sequence : List[str]
           list of tokenize words.
       n : int
           ngram size

       Returns
       -------
       ngram: list
       """

.. code:: ipython3

    string = 'saya suka makan ayam'
    
    list(malaya.generator.ngrams(string.split(), n = 2))




.. parsed-literal::

    [('saya', 'suka'), ('suka', 'makan'), ('makan', 'ayam')]



.. code:: ipython3

    list(malaya.generator.ngrams(string.split(), n = 2, pad_left = True, pad_right = True))




.. parsed-literal::

    [(None, 'saya'),
     ('saya', 'suka'),
     ('suka', 'makan'),
     ('makan', 'ayam'),
     ('ayam', None)]



.. code:: ipython3

    list(malaya.generator.ngrams(string.split(), n = 2, pad_left = True, pad_right = True,
                                left_pad_symbol = 'START'))




.. parsed-literal::

    [('START', 'saya'),
     ('saya', 'suka'),
     ('suka', 'makan'),
     ('makan', 'ayam'),
     ('ayam', None)]



.. code:: ipython3

    list(malaya.generator.ngrams(string.split(), n = 2, pad_left = True, pad_right = True,
                                left_pad_symbol = 'START', right_pad_symbol = 'END'))




.. parsed-literal::

    [('START', 'saya'),
     ('saya', 'suka'),
     ('suka', 'makan'),
     ('makan', 'ayam'),
     ('ayam', 'END')]


