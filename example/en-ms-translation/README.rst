EN to MS
========

.. container:: alert alert-info

   This tutorial is available as an IPython notebook at
   `Malaya/example/en-ms-translation <https://github.com/huseinzol05/Malaya/tree/master/example/en-ms-translation>`__.

.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 5.01 s, sys: 749 ms, total: 5.75 s
    Wall time: 5.09 s


List available Transformer models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    malaya.translation.en_ms.available_transformer()




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
          <td>42.7</td>
          <td>0.142</td>
        </tr>
        <tr>
          <th>base</th>
          <td>234.0</td>
          <td>0.696</td>
        </tr>
        <tr>
          <th>large</th>
          <td>817.0</td>
          <td>0.699</td>
        </tr>
      </tbody>
    </table>
    </div>



We tested on 77k EN-MY sentences. **``small`` actually already good
enough if you have strings shorter than 300 words. You can check more
examples below. Our test set consist very long texts that ``small`` not
able to predict it very good.**

Load Transformer models
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    transformer = malaya.translation.en_ms.transformer()
    transformer_small = malaya.translation.en_ms.transformer(model = 'small')
    transformer_large = malaya.translation.en_ms.transformer(model = 'large')


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:73: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:75: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:68: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


Translate
^^^^^^^^^

.. code:: python

   def translate(self, strings: List[str], beam_search: bool = True):
       """
       translate list of strings.

       Parameters
       ----------
       strings : List[str]
       beam_search : bool, (optional=True)
           If True, use beam search decoder, else use greedy decoder.

       Returns
       -------
       result: List[str]
       """

**For better results, we suggest do not to try translate more than 200
words, always split by end of sentences**.

.. code:: ipython3

    from pprint import pprint

.. code:: ipython3

    # https://www.malaymail.com/news/malaysia/2020/07/01/dr-mahathir-again-claims-anwar-lacks-popularity-with-malays-to-be-pakatans/1880420
    
    string_news1 = 'KUALA LUMPUR, July 1 - Datuk Seri Anwar Ibrahim is not suitable to as the prime minister candidate as he is allegedly not "popular" among the Malays, Tun Dr Mahathir Mohamad claimed. The former prime minister reportedly said the PKR president needs someone like himself in order to acquire support from the Malays and win the election.'
    pprint(string_news1)


.. parsed-literal::

    ('KUALA LUMPUR, July 1 - Datuk Seri Anwar Ibrahim is not suitable to as the '
     'prime minister candidate as he is allegedly not "popular" among the Malays, '
     'Tun Dr Mahathir Mohamad claimed. The former prime minister reportedly said '
     'the PKR president needs someone like himself in order to acquire support '
     'from the Malays and win the election.')


.. code:: ipython3

    # https://edition.cnn.com/2020/07/06/politics/new-york-attorney-general-blm/index.html
    
    string_news2 = '(CNN)New York Attorney General Letitia James on Monday ordered the Black Lives Matter Foundation -- which she said is not affiliated with the larger Black Lives Matter movement -- to stop collecting donations in New York. "I ordered the Black Lives Matter Foundation to stop illegally accepting donations that were intended for the #BlackLivesMatter movement. This foundation is not affiliated with the movement, yet it accepted countless donations and deceived goodwill," James tweeted.'
    pprint(string_news2)


.. parsed-literal::

    ('(CNN)New York Attorney General Letitia James on Monday ordered the Black '
     'Lives Matter Foundation -- which she said is not affiliated with the larger '
     'Black Lives Matter movement -- to stop collecting donations in New York. "I '
     'ordered the Black Lives Matter Foundation to stop illegally accepting '
     'donations that were intended for the #BlackLivesMatter movement. This '
     'foundation is not affiliated with the movement, yet it accepted countless '
     'donations and deceived goodwill," James tweeted.')


.. code:: ipython3

    # https://www.thestar.com.my/business/business-news/2020/07/04/malaysia-worries-new-eu-food-rules-could-hurt-palm-oil-exports
    
    string_news3 = 'Amongst the wide-ranging initiatives proposed are a sustainable food labelling framework, a reformulation of processed foods, and a sustainability chapter in all EU bilateral trade agreements. The EU also plans to publish a proposal for a legislative framework for sustainable food systems by 2023 to ensure all foods on the EU market become increasingly sustainable.'
    pprint(string_news3)


.. parsed-literal::

    ('Amongst the wide-ranging initiatives proposed are a sustainable food '
     'labelling framework, a reformulation of processed foods, and a '
     'sustainability chapter in all EU bilateral trade agreements. The EU also '
     'plans to publish a proposal for a legislative framework for sustainable food '
     'systems by 2023 to ensure all foods on the EU market become increasingly '
     'sustainable.')


.. code:: ipython3

    # https://jamesclear.com/articles
    
    string_article1 = 'This page shares my best articles to read on topics like health, happiness, creativity, productivity and more. The central question that drives my work is, “How can we live better?” To answer that question, I like to write about science-based ways to solve practical problems.'
    pprint(string_article1)


.. parsed-literal::

    ('This page shares my best articles to read on topics like health, happiness, '
     'creativity, productivity and more. The central question that drives my work '
     'is, “How can we live better?” To answer that question, I like to write about '
     'science-based ways to solve practical problems.')


.. code:: ipython3

    # https://towardsdatascience.com/fuzzy-matching-at-scale-84f2bfd0c536
    
    string_article2 = 'Fuzzy matching at scale. From 3.7 hours to 0.2 seconds. How to perform intelligent string matching in a way that can scale to even the biggest data sets. Data in the real world is messy. Dealing with messy data sets is painful and burns through time which could be spent analysing the data itself.'
    pprint(string_article2)


.. parsed-literal::

    ('Fuzzy matching at scale. From 3.7 hours to 0.2 seconds. How to perform '
     'intelligent string matching in a way that can scale to even the biggest data '
     'sets. Data in the real world is messy. Dealing with messy data sets is '
     'painful and burns through time which could be spent analysing the data '
     'itself.')


.. code:: ipython3

    random_string1 = 'i am in medical school.'
    random_string2 = 'Emmerdale is the debut studio album,songs were not released in the U.S <> These songs were not released in the U.S. edition of said album and were previously unavailable on any U.S. release.'
    pprint(random_string2)


.. parsed-literal::

    ('Emmerdale is the debut studio album,songs were not released in the U.S <> '
     'These songs were not released in the U.S. edition of said album and were '
     'previously unavailable on any U.S. release.')


Comparing with Google Translate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These printscreens taken on 7th July 2020, Google always update model,
so Google Translate in the future might improved.

**string_news1**

.. code:: ipython3

    from IPython.core.display import Image, display
    
    display(Image('en-string1.png', width=450))



.. image:: load-translation-en-ms_files/load-translation-en-ms_18_0.png
   :width: 450px


KUALA LUMPUR, 1 Julai - Anwar Ibrahim tidak sesuai menjadi calon perdana
menteri kerana dia dikatakan tidak “popular” di kalangan orang Melayu,
kata Tun Dr Mahathir Mohamad. Bekas perdana menteri itu dilaporkan
mengatakan bahawa presiden PKR memerlukan seseorang seperti dirinya
untuk mendapatkan sokongan orang Melayu dan memenangi pilihan raya.

**string_news2**

.. code:: ipython3

    display(Image('en-string2.png', width=450))



.. image:: load-translation-en-ms_files/load-translation-en-ms_21_0.png
   :width: 450px


(CNN) Peguam Negara New York, Letitia James pada hari Isnin
memerintahkan Yayasan Black Lives Matter - yang menurutnya tidak
berafiliasi dengan gerakan Black Lives Matter yang lebih besar - untuk
berhenti mengumpulkan derma di New York. “Saya memerintahkan Black Lives
Matter Foundation untuk berhenti secara haram menerima sumbangan yang
ditujukan untuk gerakan #BlackLivesMatter. Yayasan ini tidak berafiliasi
dengan gerakan itu, namun ia menerima banyak sumbangan dan menipu
muhibah,” tweet James.

**string_news3**

.. code:: ipython3

    display(Image('en-string3.png', width=450))



.. image:: load-translation-en-ms_files/load-translation-en-ms_24_0.png
   :width: 450px


Di antara inisiatif luas yang dicadangkan adalah kerangka pelabelan
makanan yang berkelanjutan, penyusunan semula makanan yang diproses, dan
bab keberlanjutan dalam semua perjanjian perdagangan dua hala EU. EU
juga berencana untuk menerbitkan proposal untuk kerangka perundangan
untuk sistem makanan lestari pada tahun 2023 untuk memastikan semua
makanan di pasar EU menjadi semakin

**random_string2**

.. code:: ipython3

    display(Image('en-string4.png', width=450))



.. image:: load-translation-en-ms_files/load-translation-en-ms_27_0.png
   :width: 450px


Emmerdale adalah album studio sulung, lagu-lagu tidak dirilis di A.S.

Translate transformer base
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%time
    
    pprint(transformer.translate([string_news1, string_news2, string_news3], beam_search = False))


.. parsed-literal::

    ['KUALA LUMPUR 1 Julai - Datuk Seri Anwar Ibrahim tidak sesuai menjadi calon '
     'Perdana Menteri kerana beliau didakwa tidak "popular" dalam kalangan orang '
     'Melayu, Tun Dr Mahathir Mohamad mendakwa, bekas Perdana Menteri itu '
     'dilaporkan berkata Presiden PKR itu memerlukan seseorang seperti dirinya '
     'bagi mendapatkan sokongan daripada orang Melayu dan memenangi pilihan raya.',
     '(CNN) Peguam Negara New York Letitia James pada hari Isnin memerintahkan '
     'Black Lives Matter Foundation - yang menurutnya tidak berafiliasi dengan '
     'gerakan Black Lives Matter yang lebih besar - untuk berhenti mengumpulkan '
     'sumbangan di New York. "Saya memerintahkan Black Lives Matter Foundation '
     'untuk berhenti menerima sumbangan secara haram yang bertujuan untuk gerakan '
     '#BlackLivesMatter. Yayasan ini tidak berafiliasi dengan gerakan itu, namun '
     'ia menerima banyak sumbangan dan muhibah yang ditipu," tweet James.',
     'Di antara inisiatif luas yang diusulkan adalah kerangka pelabelan makanan '
     'yang berkelanjutan, reformulasi makanan yang diproses, dan bab keberlanjutan '
     'dalam semua perjanjian perdagangan dua hala EU. EU juga berencana untuk '
     'menerbitkan proposal untuk kerangka perundangan untuk sistem makanan lestari '
     'pada tahun 2023 untuk memastikan semua makanan di pasar EU menjadi semakin '
     'lestari.']
    CPU times: user 25.3 s, sys: 13.7 s, total: 38.9 s
    Wall time: 11.2 s


.. code:: ipython3

    %%time
    
    pprint(transformer.translate([string_article1, string_article2], beam_search = False))


.. parsed-literal::

    ['Halaman ini berkongsi artikel terbaik saya untuk dibaca mengenai topik '
     'seperti kesihatan, kebahagiaan, kreativiti, produktiviti dan banyak lagi. '
     'Soalan utama yang mendorong kerja saya adalah, "Bagaimana kita dapat hidup '
     'lebih baik?" Untuk menjawab soalan itu, saya suka menulis mengenai kaedah '
     'berasaskan sains untuk menyelesaikan masalah praktikal.',
     'Pemadanan kabur pada skala. Dari 3.7 jam hingga 0.2 saat. Cara melakukan '
     'pemadanan rentetan pintar dengan cara yang dapat meningkatkan bahkan set '
     'data terbesar. Data di dunia nyata tidak kemas. Berurusan dengan set data '
     'yang tidak kemas menyakitkan dan terbakar sepanjang masa yang dapat '
     'dihabiskan untuk menganalisis data itu sendiri.']
    CPU times: user 16.9 s, sys: 9.43 s, total: 26.3 s
    Wall time: 6.03 s


.. code:: ipython3

    %%time
    
    pprint(transformer.translate([random_string1, random_string2], beam_search = False))


.. parsed-literal::

    ['saya di sekolah perubatan.',
     'Emmerdale adalah album studio debut, lagu-lagu tidak dikeluarkan di A.S <> '
     'Lagu-lagu ini tidak dikeluarkan dalam edisi A.S. album tersebut dan '
     'sebelumnya tidak tersedia pada sebarang pelepasan A.S.']
    CPU times: user 10.9 s, sys: 5.79 s, total: 16.7 s
    Wall time: 3.9 s


Translate transformer small
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%time
    
    pprint(transformer_small.translate([string_news1, string_news2, string_news3], beam_search = False))


.. parsed-literal::

    ['KUALA LUMPUR 1 Julai - Datuk Seri Anwar Ibrahim tidak sesuai kerana calon '
     'perdana menteri kerana didakwa tidak "popular" dalam kalangan orang Melayu, '
     'Tun Dr Mahathir Mohamad mendakwa. Bekas perdana menteri itu dilaporkan '
     'berkata, presiden PKR itu memerlukan seseorang seperti dirinya sendiri untuk '
     'memperoleh sokongan daripada orang Melayu dan memenangi pilihan raya.hari '
     'ini, Datuk Seri Anwar Ibrahim tidak sesuai untuk menjadi calon',
     '(CNN) Peguam Negara New York Letitia James pada hari Isnin memerintahkan '
     'Yayasan Black Lives Matter - yang menurutnya tidak berafiliasi dengan '
     'gerakan Black Lives Matter yang lebih besar - untuk berhenti mengumpulkan '
     'sumbangan di New York. "Saya memerintahkan Yayasan Black Lives Matter untuk '
     'berhenti menerima sumbangan secara haram yang bertujuan untuk gerakan '
     '#BlackLivesMatter. Yayasan ini tidak berafiliasi dengan gerakan itu, namun '
     'ia menerima banyak sumbangan dan muhibah yang menipu," tweet James.',
     'Amongst inisiatif luas yang dicadangkan adalah kerangka kerja kerja kerja '
     'makanan yang berkelanjutan, penyusunan semula makanan yang diproses, dan bab '
     'kelestarian dalam semua perjanjian perdagangan dua hala EU. EU juga '
     'merancang untuk menerbitkan cadangan kerangka perundangan untuk sistem '
     'makanan lestari pada tahun 2023 untuk memastikan semua makanan di pasaran EU '
     'semakin lestari.']
    CPU times: user 3.83 s, sys: 796 ms, total: 4.63 s
    Wall time: 1.69 s


.. code:: ipython3

    %%time
    
    pprint(transformer_small.translate([string_article1, string_article2], beam_search = False))


.. parsed-literal::

    ['Halaman ini berkongsi artikel terbaik saya untuk membaca topik seperti '
     'kesihatan, kebahagiaan, kreativiti, produktiviti dan banyak lagi. Soalan '
     'pusat yang mendorong karya saya adalah, "Bagaimana kita dapat hidup lebih '
     'baik?" Untuk menjawab soalan itu, saya suka menulis mengenai cara berasaskan '
     'sains untuk menyelesaikan masalah praktikal.',
     'Pemadanan Fuzzy pada skala. Dari 3.7 jam hingga 0.2 saat. Cara melakukan '
     'pemadanan rentetan pintar dengan cara yang dapat meningkatkan set data '
     'terbesar bahkan. Data di dunia nyata tidak kemas. Berurusan dengan set data '
     'yang tidak kemas menyakitkan dan terbakar melalui masa yang dapat dihabiskan '
     'untuk menganalisis data itu sendiri.']
    CPU times: user 2.61 s, sys: 424 ms, total: 3.04 s
    Wall time: 840 ms


.. code:: ipython3

    %%time
    
    pprint(transformer_small.translate([random_string1, random_string2], beam_search = False))


.. parsed-literal::

    ['saya berada di sekolah perubatan.',
     'Emmerdale adalah album studio sulung, lagu-lagu tidak dikeluarkan di A.S <> '
     'Lagu-lagu ini tidak dikeluarkan di edisi A.S. yang dikatakan album dan '
     'sebelumnya tidak tersedia di mana-mana pelepasan A.S.']
    CPU times: user 1.66 s, sys: 310 ms, total: 1.97 s
    Wall time: 588 ms


Translate transformer large
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%time
    
    pprint(transformer_large.translate([string_news1, string_news2, string_news3], beam_search = False))


.. parsed-literal::

    ['KUALA LUMPUR 1 Julai - Datuk Seri Anwar Ibrahim tidak sesuai menjadi calon '
     'Perdana Menteri kerana beliau didakwa tidak "popular" dalam kalangan orang '
     'Melayu, kata Tun Dr Mahathir Mohamad. Bekas Perdana Menteri itu dilaporkan '
     'berkata, Presiden PKR memerlukan seseorang seperti dirinya bagi mendapatkan '
     'sokongan daripada orang Melayu dan memenangi pilihan raya.',
     '(CNN) Peguam Negara New York Letitia James pada hari Isnin memerintahkan '
     'Black Lives Matter Foundation - yang menurutnya tidak berafiliasi dengan '
     'gerakan Black Lives Matter yang lebih besar - untuk berhenti mengumpulkan '
     'sumbangan di New York. "Saya memerintahkan Black Lives Matter Foundation '
     'untuk berhenti menerima sumbangan secara haram yang bertujuan untuk gerakan '
     '#BlackLivesMatter. Yayasan ini tidak berafiliasi dengan gerakan itu, namun '
     'ia menerima banyak sumbangan dan muhibah yang ditipu," tweet James.',
     'Di antara inisiatif luas yang diusulkan adalah kerangka pelabelan makanan '
     'berkelanjutan, penyusunan semula makanan yang diproses, dan bab '
     'keberlanjutan dalam semua perjanjian perdagangan dua hala EU. EU juga '
     'berencana untuk menerbitkan proposal untuk kerangka perundangan untuk sistem '
     'makanan berkelanjutan pada tahun 2023 untuk memastikan semua makanan di '
     'pasar EU menjadi semakin berkelanjutan.']
    CPU times: user 1min 2s, sys: 27 s, total: 1min 29s
    Wall time: 25.9 s


.. code:: ipython3

    %%time
    
    pprint(transformer_large.translate([string_article1, string_article2], beam_search = False))


.. parsed-literal::

    ['Halaman ini berkongsi artikel terbaik saya untuk membaca topik seperti '
     'kesihatan, kebahagiaan, kreativiti, produktiviti dan banyak lagi. Soalan '
     'utama yang mendorong karya saya adalah, "Bagaimana kita dapat hidup lebih '
     'baik?" Untuk menjawab soalan itu, saya suka menulis mengenai kaedah '
     'berasaskan sains untuk menyelesaikan masalah praktikal.',
     'Pemadanan kabur pada skala. Dari 3.7 jam hingga 0.2 saat. Cara melakukan '
     'pemadanan rentetan pintar dengan cara yang dapat meningkatkan skala ke set '
     'data terbesar. Data di dunia nyata tidak kemas. Berurusan dengan set data '
     'yang tidak kemas menyakitkan dan terbakar sepanjang masa yang dapat '
     'dihabiskan untuk menganalisis data itu sendiri.']
    CPU times: user 42.6 s, sys: 17.5 s, total: 1min
    Wall time: 13.4 s


.. code:: ipython3

    %%time
    
    pprint(transformer_large.translate([random_string1, random_string2], beam_search = False))


.. parsed-literal::

    ['saya di sekolah perubatan.',
     'Emmerdale adalah album studio debut, lagu-lagu tidak dikeluarkan di AS <> '
     'Lagu-lagu ini tidak dikeluarkan dalam edisi A.S. album tersebut dan '
     'sebelumnya tidak tersedia untuk sebarang pelepasan A.S.']
    CPU times: user 26.2 s, sys: 10.9 s, total: 37.1 s
    Wall time: 7.88 s


