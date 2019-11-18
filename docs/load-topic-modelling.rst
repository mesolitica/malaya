
.. code:: python

    import pandas as pd
    import malaya

.. code:: python

    df = pd.read_csv('tests/02032018.csv',sep=';')
    df = df.iloc[3:,1:]
    df.columns = ['text','label']
    corpus = df.text.tolist()

Load attention model
--------------------

We can use BERT or XLNET model to build topic modeling for corpus we
have, the power of attention!

.. code:: python

    xlnet = malaya.transformer.load(model = 'xlnet', size = 'base')
    attention = malaya.topic_model.attention(corpus, n_topics = 10, vectorizer = xlnet)


.. parsed-literal::

    WARNING: Logging before flag parsing goes to stderr.
    W1018 00:42:06.978917 4487972288 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet_model/xlnet.py:70: The name tf.gfile.Open is deprecated. Please use tf.io.gfile.GFile instead.

    W1018 00:42:06.981930 4487972288 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet.py:71: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

    W1018 00:42:06.992113 4487972288 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet_model/xlnet.py:253: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

    W1018 00:42:06.993095 4487972288 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet_model/xlnet.py:253: The name tf.AUTO_REUSE is deprecated. Please use tf.compat.v1.AUTO_REUSE instead.

    W1018 00:42:06.995229 4487972288 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet_model/modeling.py:686: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.

    W1018 00:42:06.997777 4487972288 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet_model/modeling.py:693: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.

    W1018 00:42:07.099963 4487972288 deprecation.py:323] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet_model/modeling.py:797: dropout (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.dropout instead.
    W1018 00:42:07.776345 4487972288 deprecation.py:323] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet_model/modeling.py:99: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.dense instead.
    W1018 00:42:15.209581 4487972288 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet.py:84: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.

    W1018 00:42:16.381231 4487972288 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_transformer/_xlnet.py:90: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

    W1018 00:42:16.752380 4487972288 deprecation.py:323] From /usr/local/lib/python3.6/site-packages/tensorflow/python/training/saver.py:1276: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use standard file APIs to check for files with this prefix.


Get topics
^^^^^^^^^^

.. code:: python

    attention.top_topics(5, top_n = 10, return_df = True)




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
          <th>topic 0</th>
          <th>topic 1</th>
          <th>topic 2</th>
          <th>topic 3</th>
          <th>topic 4</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ambil</td>
          <td>nyata</td>
          <td>rana</td>
          <td>menteri</td>
          <td>malaysia</td>
        </tr>
        <tr>
          <th>1</th>
          <td>putus</td>
          <td>dasar</td>
          <td>negara</td>
          <td>laku</td>
          <td>negara</td>
        </tr>
        <tr>
          <th>2</th>
          <td>undi</td>
          <td>tulis</td>
          <td>laksana</td>
          <td>jalan</td>
          <td>pimpin</td>
        </tr>
        <tr>
          <th>3</th>
          <td>rakyat</td>
          <td>laksana</td>
          <td>menteri</td>
          <td>gaji</td>
          <td>sasar</td>
        </tr>
        <tr>
          <th>4</th>
          <td>raja</td>
          <td>parti</td>
          <td>mdb</td>
          <td>perdana</td>
          <td>jalan</td>
        </tr>
        <tr>
          <th>5</th>
          <td>lembaga</td>
          <td>rana</td>
          <td>terima</td>
          <td>perdana menteri</td>
          <td>antarabangsa</td>
        </tr>
        <tr>
          <th>6</th>
          <td>ros</td>
          <td>catat</td>
          <td>urus</td>
          <td>tingkat</td>
          <td>hidup</td>
        </tr>
        <tr>
          <th>7</th>
          <td>kerja</td>
          <td>pas</td>
          <td>dakwa</td>
          <td>usaha</td>
          <td>undang</td>
        </tr>
        <tr>
          <th>8</th>
          <td>teknikal</td>
          <td>tangguh</td>
          <td>tuntut</td>
          <td>raja</td>
          <td>menteri</td>
        </tr>
        <tr>
          <th>9</th>
          <td>jalan</td>
          <td>umno</td>
          <td>sivil</td>
          <td>rakyat</td>
          <td>serius</td>
        </tr>
      </tbody>
    </table>
    </div>



Get topics as string
^^^^^^^^^^^^^^^^^^^^

.. code:: python

    attention.get_topics(10)




.. parsed-literal::

    [(0, 'ambil putus undi rakyat raja lembaga ros kerja teknikal jalan'),
     (1, 'nyata dasar tulis laksana parti rana catat pas tangguh umno'),
     (2, 'rana negara laksana menteri mdb terima urus dakwa tuntut sivil'),
     (3,
      'menteri laku jalan gaji perdana perdana menteri tingkat usaha raja rakyat'),
     (4,
      'malaysia negara pimpin sasar jalan antarabangsa hidup undang menteri serius'),
     (5, 'malaysia bangun raja negara laku kongsi niaga pelbagai bina tumbuh'),
     (6,
      'jppm mdb tangguh daftar selesai selesai jppm tutup mdb bayar tutup mdb pimpin sokong'),
     (7, 'mdb selesai laku hutang rana projek pendek wang tempoh pelbagai'),
     (8, 'negara punca mca malaysia pilih percaya malu kebaji jaga kebaji jaga'),
     (9, 'malaysia rakyat parti negara bangun program alam ajar raja resolusi')]



Train LDA2Vec model
-------------------

.. code:: python

    lda2vec = malaya.topic_model.lda2vec(corpus, 10, vectorizer = 'skip-gram', skip = 4)


.. parsed-literal::

    minibatch loop: 100%|██████████| 287/287 [00:02<00:00, 135.60it/s, cost=-7.4e+3, epoch=1]
    minibatch loop: 100%|██████████| 287/287 [00:01<00:00, 145.11it/s, cost=-8.01e+3, epoch=2]
    minibatch loop: 100%|██████████| 287/287 [00:02<00:00, 143.11it/s, cost=-8.62e+3, epoch=3]
    minibatch loop: 100%|██████████| 287/287 [00:01<00:00, 144.70it/s, cost=-9.24e+3, epoch=4]
    minibatch loop: 100%|██████████| 287/287 [00:01<00:00, 147.26it/s, cost=-9894.22, epoch=5]
    minibatch loop: 100%|██████████| 287/287 [00:01<00:00, 145.32it/s, cost=-1.06e+4, epoch=6]
    minibatch loop: 100%|██████████| 287/287 [00:01<00:00, 139.67it/s, cost=-1.13e+4, epoch=7]
    minibatch loop: 100%|██████████| 287/287 [00:01<00:00, 148.40it/s, cost=-1.2e+4, epoch=8]
    minibatch loop: 100%|██████████| 287/287 [00:01<00:00, 149.87it/s, cost=-1.27e+4, epoch=9]
    minibatch loop: 100%|██████████| 287/287 [00:01<00:00, 150.03it/s, cost=-1.33e+4, epoch=10]


Get topics
^^^^^^^^^^

You able to set to return as Pandas Dataframe or not by using
``return_df`` parameter

.. code:: python

    lda2vec.top_topics(5, top_n = 10, return_df = True)




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
          <th>topic 0</th>
          <th>topic 1</th>
          <th>topic 2</th>
          <th>topic 3</th>
          <th>topic 4</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>bidang didik proses</td>
          <td>bahagian</td>
          <td>dagang didik latih</td>
          <td>guru</td>
          <td>takut</td>
        </tr>
        <tr>
          <th>1</th>
          <td>kandung</td>
          <td>gagal</td>
          <td>projek jalan lancar</td>
          <td>muda</td>
          <td>dagang didik latih</td>
        </tr>
        <tr>
          <th>2</th>
          <td>wang dagang tani</td>
          <td>alam negara</td>
          <td>april</td>
          <td>program kerjasama malaysia</td>
          <td>tatakelakuan</td>
        </tr>
        <tr>
          <th>3</th>
          <td>langkah</td>
          <td>sokong</td>
          <td>tanding</td>
          <td>terima tumbuh</td>
          <td>bahagian</td>
        </tr>
        <tr>
          <th>4</th>
          <td>laku raja tingkat</td>
          <td>mahkamah</td>
          <td>awam</td>
          <td>ubah</td>
          <td>program teknikal</td>
        </tr>
        <tr>
          <th>5</th>
          <td>negara maju ancang</td>
          <td>alam ekonomi</td>
          <td>main peran kongsi</td>
          <td>tatakelakuan</td>
          <td>pindah kampung baharu</td>
        </tr>
        <tr>
          <th>6</th>
          <td>dagang didik latih</td>
          <td>bekal bersih</td>
          <td>rasmi</td>
          <td>serang</td>
          <td>nama</td>
        </tr>
        <tr>
          <th>7</th>
          <td>kaya</td>
          <td>negara negara maju</td>
          <td>gagal</td>
          <td>awam</td>
          <td>gagal</td>
        </tr>
        <tr>
          <th>8</th>
          <td>sivil doj</td>
          <td>ajar ajar</td>
          <td>alam ekonomi</td>
          <td>bahagian</td>
          <td>ikan</td>
        </tr>
        <tr>
          <th>9</th>
          <td>raja tingkat maju</td>
          <td>tani didik teknikal</td>
          <td>ikan</td>
          <td>jho low kapal</td>
          <td>ekonomi dagang</td>
        </tr>
      </tbody>
    </table>
    </div>



Important sentences based on topics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    lda2vec.get_sentences(5)




.. parsed-literal::

    ['teknikal',
     'konon fokus pilih raya',
     'sedia kongsi alam bangun ekonomi sosial negara bangun rangka program kerjasama teknikal malaysia mtcp sedia malaysia kongsi alam bangun negara negara selatan selatan',
     'kerja diploma kerja jawat kerani',
     'niaga masuk niaga digital santan santan terima minat asia hubung aktiviti francais restoran makan asia']



Get topics as string
^^^^^^^^^^^^^^^^^^^^

.. code:: python

    lda2vec.get_topics(10)




.. parsed-literal::

    [(0,
      'bidang didik proses kandung wang dagang tani langkah laku raja tingkat negara maju ancang dagang didik latih kaya sivil doj raja tingkat maju'),
     (1,
      'bahagian gagal alam negara sokong mahkamah alam ekonomi bekal bersih negara negara maju ajar ajar tani didik teknikal'),
     (2,
      'dagang didik latih projek jalan lancar april tanding awam main peran kongsi rasmi gagal alam ekonomi ikan'),
     (3,
      'guru muda program kerjasama malaysia terima tumbuh ubah tatakelakuan serang awam bahagian jho low kapal'),
     (4,
      'takut dagang didik latih tatakelakuan bahagian program teknikal pindah kampung baharu nama gagal ikan ekonomi dagang'),
     (5,
      'demokrasi kongsi maklumat main peran tatakelakuan bank seri razak pendek laku raja tingkat alam bangun nama'),
     (6,
      'melayu cina atur teknikal mtcp sedia serang aabar latih teknikal diplomasi ekonomi wajar nyata pengaruh suasana'),
     (7,
      'industri diplomasi kongsi alam bangun rangka sumber baca ekonomi sosial bangun pelopor jppm kukuh bantu teknikal industri diplomasi'),
     (8,
      'berita aabar main peran kongsi sesuai huni menteri najib dagang tani maju bidang fasal alam negara'),
     (9,
      'sebahagian alam ekonomi program teknikal huni takut saudi bahagian bawa nama generasi')]



Visualize topics
^^^^^^^^^^^^^^^^

.. code:: python

    lda2vec.visualize_topics(notebook_mode = True)




.. raw:: html


    <link rel="stylesheet" type="text/css" href="https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css">


    <div id="ldavis_el298249953790166480605917"></div>
    <script type="text/javascript">

    var ldavis_el298249953790166480605917_data = {"mdsDat": {"x": [0.00036729527095732137, 0.00013062522207853386, 0.00012120805155089045, 1.8581504929226857e-06, -1.756749301306377e-05, 9.133900718855546e-05, -1.2915079926905381e-05, -0.00011363289060742937, -0.00021221114037791724, -0.0003559990983429081], "y": [4.611508446377833e-05, -0.00027289789664000186, 0.00015772512117665402, 1.039206169505258e-05, -5.001234304925466e-06, 8.995320655417934e-05, -9.428057571577143e-05, -5.7208691605491514e-06, 0.00010879440476858412, -3.507930283700073e-05], "topics": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "cluster": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Freq": [14.230677604675293, 13.979503631591797, 11.322186470031738, 10.722183227539062, 10.681981086730957, 10.051714897155762, 8.206633567810059, 8.027471542358398, 7.565258979797363, 5.212383270263672]}, "tinfo": {"Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10"], "Freq": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.5275194644927979, 0.5247675776481628, 0.50400310754776, 0.5004177093505859, 0.521309494972229, 0.5083129405975342, 0.5064933896064758, 0.5125532746315002, 0.4945676028728485, 0.5065138936042786, 0.5078061819076538, 0.4875238239765167, 0.49688705801963806, 0.4962652623653412, 0.5066617727279663, 0.4885111451148987, 0.4974117577075958, 0.47920674085617065, 0.4857404828071594, 0.4957547187805176, 0.49749791622161865, 0.48526647686958313, 0.502259373664856, 0.506951093673706, 0.5001589059829712, 0.48929253220558167, 0.49400225281715393, 0.4951671361923218, 0.509636640548706, 0.48090454936027527, 0.4919571876525879, 0.5046827793121338, 0.5067011713981628, 0.5033771395683289, 0.5050286650657654, 0.4986009895801544, 0.5023398399353027, 0.5041957497596741, 0.5011716485023499, 0.5042726397514343, 0.5043779611587524, 0.5010722279548645, 0.5040551424026489, 0.49939265847206116, 0.49940159916877747, 0.49839484691619873, 0.5074851512908936, 0.49655914306640625, 0.5103943347930908, 0.5172608494758606, 0.5219040513038635, 0.5142205953598022, 0.5004667639732361, 0.49996212124824524, 0.49642306566238403, 0.4945739507675171, 0.4997965395450592, 0.4788416922092438, 0.48904451727867126, 0.4777865707874298, 0.4998224377632141, 0.47058460116386414, 0.4991624653339386, 0.5044231414794922, 0.50034499168396, 0.501396656036377, 0.47607845067977905, 0.5075699090957642, 0.49506106972694397, 0.5122635960578918, 0.49059945344924927, 0.49336379766464233, 0.5026562213897705, 0.4893145263195038, 0.49111923575401306, 0.497896671295166, 0.5030560493469238, 0.5014657378196716, 0.49450916051864624, 0.5015655755996704, 0.49321264028549194, 0.4943033456802368, 0.49915802478790283, 0.4968954622745514, 0.494005411863327, 0.49761486053466797, 0.49987223744392395, 0.49827805161476135, 0.49547523260116577, 0.4949560761451721, 0.38782477378845215, 0.3865618109703064, 0.4032517671585083, 0.3877080976963043, 0.39007997512817383, 0.38838037848472595, 0.37064018845558167, 0.3940030038356781, 0.3827779293060303, 0.38837164640426636, 0.3979876637458801, 0.3828448951244354, 0.3887055516242981, 0.3868727385997772, 0.37709948420524597, 0.37495139241218567, 0.38356971740722656, 0.3871244192123413, 0.38379713892936707, 0.3871367275714874, 0.3881930112838745, 0.38864654302597046, 0.38141125440597534, 0.38840383291244507, 0.3834998309612274, 0.39698556065559387, 0.38480910658836365, 0.38613271713256836, 0.3941294252872467, 0.3808521628379822, 0.391391783952713, 0.3982695937156677, 0.392273485660553, 0.3935680389404297, 0.3895891308784485, 0.38996651768684387, 0.38985520601272583, 0.3965434730052948, 0.39721980690956116, 0.38812538981437683, 0.3929317593574524, 0.391459196805954, 0.3915291130542755, 0.3953103721141815, 0.39036911725997925, 0.3910071849822998, 0.3915422856807709, 0.3914490044116974, 0.3918246328830719, 0.39207205176353455, 0.3929504156112671, 0.3905414044857025, 0.39055660367012024, 0.38997891545295715, 0.38951390981674194, 0.38994544744491577, 0.3818492293357849, 0.3872268497943878, 0.36776992678642273, 0.38126030564308167, 0.37470561265945435, 0.37264129519462585, 0.37993016839027405, 0.3809441328048706, 0.3826698064804077, 0.3631887435913086, 0.3729683458805084, 0.3624376356601715, 0.36629828810691833, 0.37203648686408997, 0.3765391409397125, 0.3599027991294861, 0.357854425907135, 0.3540274202823639, 0.359149694442749, 0.3633568584918976, 0.36483240127563477, 0.3641008734703064, 0.36766186356544495, 0.3774735629558563, 0.3593643307685852, 0.36669930815696716, 0.3658859431743622, 0.36402106285095215, 0.35916638374328613, 0.3507174849510193, 0.36484068632125854, 0.376287579536438, 0.37983864545822144, 0.3794609308242798, 0.3697313964366913, 0.37486618757247925, 0.37644675374031067, 0.3701556921005249, 0.38067564368247986, 0.37998443841934204, 0.37323397397994995, 0.37757888436317444, 0.3786185085773468, 0.36967241764068604, 0.37083256244659424, 0.3733384311199188, 0.37165480852127075, 0.37183094024658203, 0.37148746848106384, 0.37147730588912964, 0.37080228328704834, 0.3704317510128021, 0.38219305872917175, 0.37335482239723206, 0.38646140694618225, 0.3776058852672577, 0.36768394708633423, 0.3854677975177765, 0.37796756625175476, 0.3647611737251282, 0.368236243724823, 0.3651108741760254, 0.3668617308139801, 0.3778987228870392, 0.38164037466049194, 0.38028690218925476, 0.3766491115093231, 0.37198495864868164, 0.37120872735977173, 0.38445258140563965, 0.3686044216156006, 0.3675602972507477, 0.36069563031196594, 0.37869998812675476, 0.37779366970062256, 0.36824437975883484, 0.3671937882900238, 0.36661770939826965, 0.3696352243423462, 0.3612649440765381, 0.37393659353256226, 0.37183454632759094, 0.37251585721969604, 0.3790668547153473, 0.3771462142467499, 0.37490442395210266, 0.3808903396129608, 0.37518778443336487, 0.3758373558521271, 0.376625657081604, 0.3748056888580322, 0.3726683259010315, 0.37479302287101746, 0.3725692629814148, 0.3730581998825073, 0.374947190284729, 0.3746517300605774, 0.37311431765556335, 0.3694708049297333, 0.3568474352359772, 0.3670026957988739, 0.35984787344932556, 0.35584062337875366, 0.35265859961509705, 0.35664263367652893, 0.3557986617088318, 0.36691731214523315, 0.3646205961704254, 0.3508114516735077, 0.35050535202026367, 0.360371857881546, 0.3620525598526001, 0.34805792570114136, 0.34639817476272583, 0.3573734164237976, 0.3540729582309723, 0.33647316694259644, 0.3469923138618469, 0.3392474055290222, 0.35085272789001465, 0.3640170097351074, 0.3374669849872589, 0.34844598174095154, 0.3390522599220276, 0.3508561849594116, 0.35864099860191345, 0.34074386954307556, 0.3437782824039459, 0.35625624656677246, 0.35711777210235596, 0.3517040014266968, 0.35848015546798706, 0.3577231466770172, 0.350302129983902, 0.35901203751564026, 0.3538017272949219, 0.3511720299720764, 0.3539183735847473, 0.3535681664943695, 0.35429051518440247, 0.3529168367385864, 0.3513585329055786, 0.3513423502445221, 0.28562578558921814, 0.2977355718612671, 0.2877656817436218, 0.2801203429698944, 0.2895995080471039, 0.2904042899608612, 0.3000859320163727, 0.27812665700912476, 0.28271356225013733, 0.2793373465538025, 0.27927637100219727, 0.2842130959033966, 0.28641191124916077, 0.2910221219062805, 0.27975109219551086, 0.2788815498352051, 0.2894541621208191, 0.2872353196144104, 0.28863033652305603, 0.281374454498291, 0.2873436212539673, 0.27465957403182983, 0.28080737590789795, 0.28082600235939026, 0.27693843841552734, 0.28278183937072754, 0.2794572114944458, 0.279264360666275, 0.2806682288646698, 0.2812654376029968, 0.2824579179286957, 0.28437215089797974, 0.28378909826278687, 0.2868350148200989, 0.2825130820274353, 0.28299853205680847, 0.2919161319732666, 0.28338173031806946, 0.2837882936000824, 0.2861934006214142, 0.2865530550479889, 0.28446057438850403, 0.2842855453491211, 0.28559383749961853, 0.2867198884487152, 0.2842567265033722, 0.28296521306037903, 0.2834913432598114, 0.28311339020729065, 0.28310757875442505, 0.2831561863422394, 0.29345595836639404, 0.2910703718662262, 0.30175068974494934, 0.2858620882034302, 0.2924703061580658, 0.288622111082077, 0.2933351695537567, 0.2836005389690399, 0.2905733585357666, 0.2837408781051636, 0.286594033241272, 0.2848608195781708, 0.28102216124534607, 0.2700227200984955, 0.277927964925766, 0.28469935059547424, 0.2776638865470886, 0.2793947458267212, 0.281704306602478, 0.28519803285598755, 0.28656548261642456, 0.28108271956443787, 0.28924012184143066, 0.2738184630870819, 0.2852907180786133, 0.28252553939819336, 0.29072803258895874, 0.28094789385795593, 0.27954569458961487, 0.2815594971179962, 0.2872244715690613, 0.2851552367210388, 0.2868538200855255, 0.2820433974266052, 0.28219708800315857, 0.28400418162345886, 0.28305667638778687, 0.2826562225818634, 0.28376322984695435, 0.2829289138317108, 0.28373709321022034, 0.2832607626914978, 0.2896236181259155, 0.28572791814804077, 0.280870646238327, 0.282777339220047, 0.2612730860710144, 0.268709272146225, 0.2849855124950409, 0.2843151390552521, 0.2749160826206207, 0.27331918478012085, 0.2693324387073517, 0.2708789110183716, 0.2643619179725647, 0.2791226804256439, 0.2765505909919739, 0.27206405997276306, 0.2776900827884674, 0.2767797112464905, 0.27461594343185425, 0.2666652798652649, 0.2765456736087799, 0.2747349143028259, 0.2725893557071686, 0.2726787328720093, 0.28049927949905396, 0.2708202600479126, 0.2705048620700836, 0.2627657949924469, 0.26425886154174805, 0.26984351873397827, 0.26565220952033997, 0.271453320980072, 0.2687094509601593, 0.27210745215415955, 0.273377925157547, 0.2727624475955963, 0.272938072681427, 0.2756541669368744, 0.2715807855129242, 0.27213478088378906, 0.2713797986507416, 0.2063463032245636, 0.2036699503660202, 0.20323550701141357, 0.19654583930969238, 0.1968422532081604, 0.19062693417072296, 0.1907961666584015, 0.18822535872459412, 0.18770189583301544, 0.18945874273777008, 0.18875578045845032, 0.18510442972183228, 0.19169928133487701, 0.1926242560148239, 0.18803437054157257, 0.1851411759853363, 0.18661579489707947, 0.18724635243415833, 0.18649135529994965, 0.19335336983203888, 0.19367137551307678, 0.18722842633724213, 0.1880875527858734, 0.18345223367214203, 0.18640795350074768, 0.18735508620738983, 0.1807815283536911, 0.18947705626487732, 0.19318673014640808, 0.18655343353748322, 0.19530510902404785, 0.18810848891735077, 0.19118274748325348, 0.19355371594429016, 0.19020722806453705, 0.1891212910413742, 0.18894559144973755, 0.18816320598125458, 0.1878604292869568, 0.1875845342874527, 0.18740880489349365], "Term": ["alam ekonomi", "takut", "program teknikal", "sebahagian", "huni", "bahagian", "dagang tani", "berbeza", "gagal", "saudi", "tahap", "untung", "didik latih teknikal", "dagang didik latih", "galak", "negara negara maju", "pindah kampung baharu", "urus niaga low", "teruk", "low", "bawa", "nama", "sedia kongsi", "tatakelakuan", "niaga low", "fasal", "perintah", "tani didik", "capai", "inggeris bahasa", "industri diplomasi", "rangka", "kukuh", "pelbagai", "kongsi alam bangun", "teknikal industri diplomasi", "program teknikal malaysia", "sumber baca", "rakyat malaysia", "bantu", "ekonomi sosial bangun", "peringkat", "sepenuh", "didik latih", "kongsi alam negara", "malaysia mtcp malaysia", "projek tanah", "kena", "seri", "negara bidang", "malaysia mtcp sedia", "catat", "latih teknikal", "bangun bandar", "malu", "timbang wajar", "sosial negara", "kuok", "pelopor", "tingkat bidang", "pelabur", "masyarakat asli", "jppm", "prestasi", "angkat", "kongsi", "dasar", "fasal lembaga", "kongsi mahir bangun", "jual beli", "dunia", "serius pimpin", "alam bangun", "raja tingkat bidang", "tawar", "mahir bangun bandar", "latih teknikal diplomasi", "tangguh parti", "serang", "teknikal mtcp sedia", "melayu cina", "atur", "maju didik", "selatan selatan", "menteri perdana", "bantu", "proses ajar", "rakyat", "politik arah", "politik", "rakyat pimpin", "teknikal", "komprehensif", "wajar nyata", "suasana", "isi", "malaysia", "ekonomi", "hutang mdb", "aabar", "bahan", "rangka", "fasal lembaga", "alam bangun sosial", "rangka kerjasama", "aset", "baca", "tanah", "pelihara", "pengaruh", "menteri najib razak", "industri", "kerjasama teknikal", "urus jho low", "lihat potensi", "ekonomi sosial", "tani didik teknikal", "laku tingkat", "komitmen", "april", "projek", "selesa", "bidang didik proses", "buku", "perdana", "nyata laku", "raja", "kaya", "maklumat", "pelbagai", "kandung", "catat", "bahasa inggeris bahasa", "bayar", "teknikal", "kena", "negara malaysia", "ancang wang dagang", "nyata", "allahyarham nik", "projek tanah", "jemaah", "selesai", "malaysia mtcp sedia", "rumah", "langkah", "pilih umno", "malaysia peran", "sivil doj", "ros", "serah", "wang dagang tani", "bekal", "raja tingkat maju", "teknikal industri diplomasi", "rampas", "islam", "negara maju ancang", "laku raja tingkat", "maju", "matlamat", "pendek", "bangkit", "dagang didik latih", "ahli", "bangun rangka program", "pelopor", "buah syarikat", "sosial bangun rangka", "demokrasi", "kongsi maklumat", "dana seleweng", "serius pimpin", "daging", "giat hidup", "bahagian", "ubah", "guru", "mdb", "program kerjasama malaysia", "senarai", "sabah", "serang", "terima tumbuh", "muda", "alam bangun negara", "alam bangun kawasan", "muka", "wujud", "wang mdb", "angkat", "mtcp sedia malaysia", "anti", "terima", "timbang wajar", "bimbang", "majlis", "lonjak", "ekonomi sosial negara", "perintah", "tulis", "kembang", "anak", "bentuk", "projek jalan", "hidup duduk", "suara", "bertanggungjawab", "ekonomi dagang", "awam", "mahir bangun bandar", "pindah kampung baharu", "wajar tulis nyata", "negara negara maju", "tatakelakuan", "bahagian", "individu", "jho low kapal", "nama", "mahkamah", "makna", "program teknikal", "giat", "atur", "kerjasama teknikal", "ekonomi sosial", "bangun main peran", "generasi", "tanding", "keluarga", "dagang didik latih", "takut", "anak", "projek jalan lancar", "alam ekonomi", "musnah", "perdana seri", "isu politik", "anak muda", "gagal", "awam", "cabar", "rujuk", "undang", "sekira", "april", "alam bangun kawasan", "hadap", "syarikat terbang", "ikan", "wajar tulis nyata", "idea", "mtcp malaysia", "latih industri", "menang", "sah", "hebat", "trx", "luas", "rasmi", "beli", "laku raja tingkat", "main peran kongsi", "pelan", "aabar", "ancang ekonomi", "ekonomi dagang", "wajar tulis", "kongsi maklumat", "bidang didik", "bawa", "tani didik teknikal", "bahagian", "nama", "demokrasi", "negara bangun program", "main peran", "pendek", "sivil doj", "mesyuarat", "bangkit", "ph", "kongsi maklumat", "bank", "timbang", "umno pilih", "laku raja tingkat", "seri razak", "kongsi alam ekonomi", "imbang", "bidang ancang", "menteri seri", "kuasa", "kongsi alam negara", "bayar", "murah", "tatakelakuan", "lihat", "ltat", "kasih", "serius pimpin", "alam bangun", "pilihanraya", "didik ajar ajar", "negara rangka", "dagang didik latih", "trx", "melayu cina", "rasmi", "matlamat", "nama", "sedia kongsi", "strategi", "ekonomi dagang", "jabat perdana", "april", "awam", "wajar tulis", "bidang didik", "belia", "berita", "maju bidang", "sumbang", "dagang tani", "menteri najib", "aabar", "sumber", "najib razak", "kerjasama malaysia mtcp", "status jawat", "aabar bvi", "ilmu", "sesuai", "bahan uji", "muka", "huni", "fasal", "mahukan", "hubung", "kerjasama malaysia", "niaga jho", "didik proses ajar", "lancar", "gera sosial", "masuk", "percaya", "maju ancang", "baharu", "bangun arus perdana", "konsep", "peran alam mahir", "hasil makan", "alam negara", "arus", "syarikat", "main peran kongsi", "mtcp malaysia", "sosial", "dedah", "alam ekonomi", "negara negara maju", "ulang kali", "wajar nyata", "tani didik teknikal", "bangun rangka kerjasama", "ubah", "muda", "minimum", "pindah kampung baharu", "nama", "alam negara", "mahkamah", "bahagian", "timbang", "sokong", "negara negara maju", "gagal", "didik", "alam ekonomi", "dagang tani", "rupa", "fasal", "didik latih teknikal", "razak", "urus niaga low", "main kongsi", "undi", "peran kongsi mahir", "bantu negara", "huni", "terima tumbuh", "tuntut doj", "ajar ajar", "wang tani", "takut", "nila", "bekal bersih", "masyarakat manfaat", "arab", "inggeris bahasa", "program teknikal", "pelan", "tani didik teknikal", "sesuai", "timbang tulis nyata", "laku tingkat", "rendah", "cawangan", "kongsi maklumat", "ikan", "tatakelakuan", "projek jalan lancar", "takut", "program teknikal", "pindah kampung baharu", "dagang didik latih", "razak", "urus niaga low", "bahagian", "tatakelakuan", "saudi", "penambahbaikan", "senarai", "timbang wajar nyata", "bidang didik ajar", "ekonomi dagang", "gagal", "berbeza", "ikan", "sedia kongsi", "alam ekonomi", "musuh", "bawa", "tahap", "luas", "duduk sewa", "nama", "sebahagian", "huni", "low", "bahasa ilmu", "indonesia", "niaga low", "bangun malaysia main", "negara bangun malaysia", "cawangan", "generasi", "negara rangka", "pelan", "april", "sedia malaysia alam", "bekal bersih", "kongsi maklumat", "sebahagian", "alam ekonomi", "program teknikal", "huni", "takut", "teknikal diplomasi", "dagang tani", "galak", "low", "teknikal malaysia mtcp", "untung", "faktor", "berbeza", "saudi", "capai", "bahan uji", "khidmat", "bilion", "laku raja", "perintah", "tahap", "tangguh", "niaga low", "laksana nhrap", "impak", "isu", "imej", "tani didik", "bawa", "senarai", "bahagian", "teruk", "generasi", "nama", "gagal", "dedah", "pindah kampung", "alam negara", "sedia kongsi", "sokong", "dagang didik latih"], "Total": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.2614126205444336, 3.2996842861175537, 3.2170767784118652, 3.2004268169403076, 3.3388891220092773, 3.261245012283325, 3.256657600402832, 3.296323537826538, 3.1835343837738037, 3.2683193683624268, 3.2802038192749023, 3.1515157222747803, 3.2131712436676025, 3.209266424179077, 3.279754638671875, 3.169693946838379, 3.2297041416168213, 3.111567735671997, 3.1545302867889404, 3.219851493835449, 3.2367470264434814, 3.1670327186584473, 3.278716564178467, 3.3133726119995117, 3.273667812347412, 3.2045645713806152, 3.238116979598999, 3.245950222015381, 3.3423073291778564, 3.155674934387207, 3.234269618988037, 3.3450024127960205, 3.363895893096924, 3.334451198577881, 3.3520028591156006, 3.2928056716918945, 3.3352909088134766, 3.3620057106018066, 3.3277218341827393, 3.373617172241211, 3.3907434940338135, 3.334747076034546, 3.4101481437683105, 3.3327033519744873, 3.381711006164551, 3.322736978530884, 3.2593774795532227, 3.24655818939209, 3.3444695472717285, 3.3897690773010254, 3.428666353225708, 3.3818888664245605, 3.299715042114258, 3.2981693744659424, 3.2792603969573975, 3.2683193683624268, 3.3045623302459717, 3.170931100845337, 3.2441165447235107, 3.175095796585083, 3.3225717544555664, 3.1286227703094482, 3.3195443153381348, 3.358809232711792, 3.334124803543091, 3.342416286468506, 3.1754026412963867, 3.3855443000793457, 3.3027162551879883, 3.4221134185791016, 3.277651309967041, 3.2996842861175537, 3.3620057106018066, 3.2731924057006836, 3.2889740467071533, 3.3347702026367188, 3.3729934692382812, 3.3693490028381348, 3.3165462017059326, 3.3832671642303467, 3.310098886489868, 3.3211276531219482, 3.374274492263794, 3.3586041927337646, 3.322991371154785, 3.38773775100708, 3.453176736831665, 3.4148895740509033, 3.3728134632110596, 3.448843240737915, 3.1154072284698486, 3.127290964126587, 3.262721300125122, 3.1707828044891357, 3.191270589828491, 3.178704261779785, 3.039858102798462, 3.2365524768829346, 3.150461435317993, 3.2004268169403076, 3.2912487983703613, 3.1670327186584473, 3.2187306880950928, 3.208446979522705, 3.1286227703094482, 3.111567735671997, 3.183821678161621, 3.2141106128692627, 3.190596342086792, 3.2204079627990723, 3.2297041416168213, 3.2345080375671387, 3.174467086791992, 3.2367470264434814, 3.198535442352295, 3.311596632003784, 3.210357427597046, 3.2228164672851562, 3.290379762649536, 3.1805808544158936, 3.2715952396392822, 3.3440051078796387, 3.2874538898468018, 3.3022572994232178, 3.261245012283325, 3.266855478286743, 3.2667815685272217, 3.3583219051361084, 3.385415554046631, 3.2550301551818848, 3.340965986251831, 3.3184213638305664, 3.323636293411255, 3.4058470726013184, 3.301791191101074, 3.321244478225708, 3.3423073291778564, 3.3407106399536133, 3.3590145111083984, 3.369706630706787, 3.4265925884246826, 3.312105417251587, 3.334747076034546, 3.3208911418914795, 3.311812162399292, 3.4505791664123535, 3.3155910968780518, 3.372711181640625, 3.215350389480591, 3.339108467102051, 3.2883076667785645, 3.2792611122131348, 3.3444695472717285, 3.361055850982666, 3.3778021335601807, 3.208826780319214, 3.3072211742401123, 3.2196810245513916, 3.256948947906494, 3.308292865753174, 3.3520028591156006, 3.206418752670288, 3.1889684200286865, 3.157473564147949, 3.2045645713806152, 3.242283344268799, 3.255955457687378, 3.2526822090148926, 3.2890307903289795, 3.3771960735321045, 3.2160871028900146, 3.2818446159362793, 3.2751951217651367, 3.26155161857605, 3.219144582748413, 3.1446828842163086, 3.2724294662475586, 3.376162052154541, 3.417975902557373, 3.4203989505767822, 3.322736978530884, 3.3771417140960693, 3.3941874504089355, 3.328075647354126, 3.451411008834839, 3.4505791664123535, 3.368391990661621, 3.4259848594665527, 3.4653308391571045, 3.3282837867736816, 3.347262144088745, 3.397280693054199, 3.375467300415039, 3.3818888664245605, 3.374274492263794, 3.38773775100708, 3.369900703430176, 3.400038719177246, 3.3541464805603027, 3.287994146347046, 3.4058470726013184, 3.354159355163574, 3.2751951217651367, 3.43603777885437, 3.37471604347229, 3.2594950199127197, 3.2955470085144043, 3.269991159439087, 3.285809278488159, 3.3867080211639404, 3.4203989505767822, 3.408329963684082, 3.3759605884552, 3.3343794345855713, 3.3292555809020996, 3.448843240737915, 3.3072211742401123, 3.2992990016937256, 3.2378437519073486, 3.4017865657806396, 3.3941874504089355, 3.308532953262329, 3.303297519683838, 3.2996084690093994, 3.32773756980896, 3.2528395652770996, 3.368967294692993, 3.351266384124756, 3.3579280376434326, 3.4209585189819336, 3.4036290645599365, 3.385415554046631, 3.450990676879883, 3.39542555809021, 3.4221134185791016, 3.4400227069854736, 3.417975902557373, 3.3833396434783936, 3.4265925884246826, 3.390904426574707, 3.404442548751831, 3.453176736831665, 3.4505791664123535, 3.4653308391571045, 3.369706630706787, 3.2605185508728027, 3.375667095184326, 3.3184213638305664, 3.290379762649536, 3.270775079727173, 3.323636293411255, 3.3163230419158936, 3.4265925884246826, 3.407994031906128, 3.279313802719116, 3.2911016941070557, 3.385415554046631, 3.403563976287842, 3.2775826454162598, 3.262408494949341, 3.366917371749878, 3.3380229473114014, 3.177687406539917, 3.279754638671875, 3.208446979522705, 3.323298692703247, 3.451411008834839, 3.2011587619781494, 3.3060266971588135, 3.218822479248047, 3.334747076034546, 3.4101481437683105, 3.2412149906158447, 3.2703983783721924, 3.390270709991455, 3.4058470726013184, 3.351266384124756, 3.428666353225708, 3.4209585189819336, 3.340965986251831, 3.4653308391571045, 3.395324468612671, 3.3580105304718018, 3.417975902557373, 3.4170873165130615, 3.448843240737915, 3.4203989505767822, 3.3833396434783936, 3.390904426574707, 3.217219591140747, 3.3602824211120605, 3.267794132232666, 3.1854958534240723, 3.295940637588501, 3.3071799278259277, 3.4221134185791016, 3.1799826622009277, 3.242982864379883, 3.213456630706787, 3.2145166397094727, 3.271660566329956, 3.300809383392334, 3.3541712760925293, 3.224299192428589, 3.2196810245513916, 3.3422975540161133, 3.3168718814849854, 3.3370490074157715, 3.255669355392456, 3.3262481689453125, 3.1795105934143066, 3.251161813735962, 3.2546651363372803, 3.211374521255493, 3.281071186065674, 3.2425551414489746, 3.2406413555145264, 3.2598047256469727, 3.2668113708496094, 3.2811341285705566, 3.3076553344726562, 3.301853895187378, 3.346710205078125, 3.287346839904785, 3.295532464981079, 3.450990676879883, 3.303297519683838, 3.311985731124878, 3.3637893199920654, 3.37471604347229, 3.328075647354126, 3.3251218795776367, 3.358809232711792, 3.453176736831665, 3.357025146484375, 3.3155910968780518, 3.3778021335601807, 3.360265016555786, 3.3771417140960693, 3.4653308391571045, 3.346710205078125, 3.3282837867736816, 3.4505791664123535, 3.279313802719116, 3.361767530441284, 3.328075647354126, 3.3867080211639404, 3.281555414199829, 3.37471604347229, 3.295940637588501, 3.331963062286377, 3.3168718814849854, 3.2753522396087646, 3.149803400039673, 3.2453413009643555, 3.326073169708252, 3.2475457191467285, 3.26833438873291, 3.299551010131836, 3.3422975540161133, 3.361055850982666, 3.2968835830688477, 3.393296241760254, 3.21858286857605, 3.354159355163574, 3.324071168899536, 3.4223780632019043, 3.3085782527923584, 3.2964279651641846, 3.322315216064453, 3.397280693054199, 3.39542555809021, 3.453176736831665, 3.3541712760925293, 3.360870361328125, 3.4148895740509033, 3.391838788986206, 3.37917423248291, 3.4265925884246826, 3.4017865657806396, 3.451411008834839, 3.43603777885437, 3.354159355163574, 3.397280693054199, 3.3771417140960693, 3.4058470726013184, 3.149803400039673, 3.2453413009643555, 3.4505791664123535, 3.451411008834839, 3.3508260250091553, 3.336015462875366, 3.2883076667785645, 3.310777425765991, 3.235142946243286, 3.417975902557373, 3.3867080211639404, 3.3324332237243652, 3.4017865657806396, 3.395324468612671, 3.37471604347229, 3.2796924114227295, 3.404442548751831, 3.3843348026275635, 3.3579280376434326, 3.367417335510254, 3.4653308391571045, 3.3461802005767822, 3.3422975540161133, 3.251826763153076, 3.2706542015075684, 3.340662956237793, 3.2897849082946777, 3.367091655731201, 3.3305983543395996, 3.37917423248291, 3.400038719177246, 3.390270709991455, 3.39542555809021, 3.448843240737915, 3.3804404735565186, 3.4223780632019043, 3.4265925884246826, 3.3461802005767822, 3.37471604347229, 3.397280693054199, 3.3422975540161133, 3.354159355163574, 3.269162893295288, 3.295940637588501, 3.257636308670044, 3.251826763153076, 3.2868471145629883, 3.2756526470184326, 3.214609146118164, 3.3324332237243652, 3.3508260250091553, 3.2741942405700684, 3.224299192428589, 3.2517635822296143, 3.2631921768188477, 3.2541861534118652, 3.3771960735321045, 3.3843348026275635, 3.2726969718933105, 3.2897849082946777, 3.2113664150238037, 3.2741994857788086, 3.2956039905548096, 3.182115077972412, 3.3361432552337646, 3.404442548751831, 3.2883076667785645, 3.4505791664123535, 3.3241496086120605, 3.400038719177246, 3.4653308391571045, 3.3867080211639404, 3.3637893199920654, 3.365328788757324, 3.346710205078125, 3.395324468612671, 3.361767530441284, 3.4058470726013184], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.12800000607967377, 0.11110000312328339, 0.09610000252723694, 0.094200000166893, 0.09269999712705612, 0.09099999815225601, 0.08879999816417694, 0.08860000222921371, 0.0877000018954277, 0.08529999852180481, 0.08420000225305557, 0.08349999785423279, 0.08309999853372574, 0.08309999853372574, 0.08209999650716782, 0.07970000058412552, 0.07900000363588333, 0.07900000363588333, 0.07880000025033951, 0.07880000025033951, 0.07699999958276749, 0.0738999992609024, 0.07370000332593918, 0.07249999791383743, 0.07100000232458115, 0.07039999961853027, 0.06960000097751617, 0.06949999928474426, 0.06909999996423721, 0.06849999725818634, 0.066600002348423, 0.05849999934434891, 0.0568000003695488, 0.05900000035762787, 0.057100001722574234, 0.06210000067949295, 0.056699998676776886, 0.052400000393390656, 0.056699998676776886, 0.04910000041127205, 0.04430000111460686, 0.0544000007212162, 0.03790000081062317, 0.051600001752376556, 0.03700000047683716, 0.05260000005364418, 0.10779999941587448, 0.08990000188350677, 0.0877000018954277, 0.08760000020265579, 0.08510000258684158, 0.08399999886751175, 0.08150000125169754, 0.08100000023841858, 0.07959999889135361, 0.07919999957084656, 0.0786999985575676, 0.07720000296831131, 0.07540000230073929, 0.07360000163316727, 0.07329999655485153, 0.07320000231266022, 0.07289999723434448, 0.07169999927282333, 0.07090000063180923, 0.07050000131130219, 0.07000000029802322, 0.06989999860525131, 0.0697999969124794, 0.06840000301599503, 0.06830000132322311, 0.06719999760389328, 0.06719999760389328, 0.06710000336170197, 0.06589999794960022, 0.0658000037074089, 0.06469999998807907, 0.06260000169277191, 0.06449999660253525, 0.05869999900460243, 0.06379999965429306, 0.06270000338554382, 0.05660000070929527, 0.056699998676776886, 0.061500001698732376, 0.0494999997317791, 0.0348999984562397, 0.04280000180006027, 0.04960000142455101, 0.02630000002682209, 0.09480000287294388, 0.08780000358819962, 0.0877000018954277, 0.07689999788999557, 0.07660000026226044, 0.07620000094175339, 0.07410000264644623, 0.07249999791383743, 0.0706000030040741, 0.06930000334978104, 0.0658000037074089, 0.06549999862909317, 0.06449999660253525, 0.06300000101327896, 0.06260000169277191, 0.062300000339746475, 0.06210000067949295, 0.061799999326467514, 0.060600001364946365, 0.05990000069141388, 0.05979999899864197, 0.05939999967813492, 0.05939999967813492, 0.05810000002384186, 0.05730000138282776, 0.057100001722574234, 0.05700000002980232, 0.05660000070929527, 0.056299999356269836, 0.0560000017285347, 0.05510000139474869, 0.050599999725818634, 0.05249999836087227, 0.05130000039935112, 0.053599998354911804, 0.05290000140666962, 0.05260000005364418, 0.041999999433755875, 0.0357000008225441, 0.05180000141263008, 0.03799999877810478, 0.04100000113248825, 0.03970000147819519, 0.024800000712275505, 0.043299999088048935, 0.039000000804662704, 0.0340999998152256, 0.034299999475479126, 0.02979999966919422, 0.027300000190734863, 0.012799999676644802, 0.0406000018119812, 0.033799998462200165, 0.0364999994635582, 0.038100000470876694, -0.0019000000320374966, 0.07150000333786011, 0.06840000301599503, 0.06459999829530716, 0.06289999932050705, 0.0608999989926815, 0.05810000002384186, 0.05779999867081642, 0.05550000071525574, 0.054999999701976776, 0.054099999368190765, 0.05050000175833702, 0.048700001090765, 0.04780000075697899, 0.04769999906420708, 0.04659999907016754, 0.04580000042915344, 0.045499999076128006, 0.04470000043511391, 0.04430000111460686, 0.044199999421834946, 0.04410000145435333, 0.04309999942779541, 0.04170000180602074, 0.041600000113248825, 0.04129999876022339, 0.041200000792741776, 0.04100000113248825, 0.04010000079870224, 0.039799999445676804, 0.039400000125169754, 0.039000000804662704, 0.03869999945163727, 0.03579999879002571, 0.0340999998152256, 0.03709999844431877, 0.03460000082850456, 0.033799998462200165, 0.03660000115633011, 0.028300000354647636, 0.02669999934732914, 0.03290000185370445, 0.027499999850988388, 0.018799999728798866, 0.03530000150203705, 0.03269999846816063, 0.02459999918937683, 0.026499999687075615, 0.025100000202655792, 0.026399999856948853, 0.02239999920129776, 0.02590000070631504, 0.01600000075995922, 0.06459999829530716, 0.06109999865293503, 0.06040000170469284, 0.05249999836087227, 0.04969999939203262, 0.04899999871850014, 0.04740000143647194, 0.04650000110268593, 0.04500000178813934, 0.04430000111460686, 0.044199999421834946, 0.04360000044107437, 0.04360000044107437, 0.04360000044107437, 0.04349999874830246, 0.04340000078082085, 0.042899999767541885, 0.04259999841451645, 0.042500000447034836, 0.041999999433755875, 0.041999999433755875, 0.04129999876022339, 0.041099999099969864, 0.041099999099969864, 0.039799999445676804, 0.039400000125169754, 0.03909999877214432, 0.03889999911189079, 0.03830000013113022, 0.03799999877810478, 0.03779999911785126, 0.03660000115633011, 0.03660000115633011, 0.03610000014305115, 0.03269999846816063, 0.033900000154972076, 0.027799999341368675, 0.02459999918937683, 0.026200000196695328, 0.030700000002980232, 0.02370000071823597, 0.028200000524520874, 0.025499999523162842, 0.016300000250339508, 0.016300000250339508, 0.007899999618530273, 0.0869000032544136, 0.08510000258684158, 0.07840000092983246, 0.07590000331401825, 0.07320000231266022, 0.07010000199079514, 0.06530000269412994, 0.06520000100135803, 0.06319999694824219, 0.06239999830722809, 0.062300000339746475, 0.05779999867081642, 0.05730000138282776, 0.05660000070929527, 0.05490000173449516, 0.05480000004172325, 0.054499998688697815, 0.05380000174045563, 0.052000001072883606, 0.05119999870657921, 0.050599999725818634, 0.04910000041127205, 0.04809999838471413, 0.047600001096725464, 0.04740000143647194, 0.046799998730421066, 0.04569999873638153, 0.04520000144839287, 0.04490000009536743, 0.04479999840259552, 0.04439999908208847, 0.0421999990940094, 0.04309999942779541, 0.039400000125169754, 0.039500001817941666, 0.0421999990940094, 0.03020000085234642, 0.035999998450279236, 0.03959999978542328, 0.02969999983906746, 0.028999999165534973, 0.021800000220537186, 0.026100000366568565, 0.032600000500679016, 0.030300000682473183, 0.07859999686479568, 0.07670000195503235, 0.07050000131130219, 0.06909999996423721, 0.06830000132322311, 0.06769999861717224, 0.06629999727010727, 0.06369999796152115, 0.06040000170469284, 0.057500001043081284, 0.05700000002980232, 0.05689999833703041, 0.05570000037550926, 0.05570000037550926, 0.05570000037550926, 0.05400000140070915, 0.05380000174045563, 0.05380000174045563, 0.05249999836087227, 0.05180000141263008, 0.05130000039935112, 0.05130000039935112, 0.051100000739097595, 0.05009999871253967, 0.04960000142455101, 0.04899999871850014, 0.04899999871850014, 0.048900000751018524, 0.04800000041723251, 0.04800000041723251, 0.04780000075697899, 0.04650000110268593, 0.04619999974966049, 0.04340000078082085, 0.04610000178217888, 0.04529999941587448, 0.030300000682473183, 0.04430000111460686, 0.04320000112056732, 0.03610000014305115, 0.0340999998152256, 0.040699999779462814, 0.04089999943971634, 0.03550000116229057, 0.011699999682605267, 0.031300000846385956, 0.03920000046491623, 0.02239999920129776, 0.02630000002682209, 0.021299999207258224, -0.00430000014603138, 0.08829999715089798, 0.08569999784231186, 0.08560000360012054, 0.08240000158548355, 0.0803999975323677, 0.07729999721050262, 0.07599999755620956, 0.0737999975681305, 0.07010000199079514, 0.06989999860525131, 0.06909999996423721, 0.06750000268220901, 0.066600002348423, 0.065700002014637, 0.06469999998807907, 0.06419999897480011, 0.06310000270605087, 0.06289999932050705, 0.06159999966621399, 0.06109999865293503, 0.06030000001192093, 0.06019999831914902, 0.05999999865889549, 0.05810000002384186, 0.05790000036358833, 0.057100001722574234, 0.05660000070929527, 0.05620000138878822, 0.05490000173449516, 0.05420000106096268, 0.05180000141263008, 0.045099999755620956, 0.03420000150799751, 0.04639999940991402, 0.04500000178813934, 0.03539999946951866, 0.03880000114440918, 0.041099999099969864, 0.031099999323487282, 0.03539999946951866, 0.023800000548362732, 0.026599999517202377, 0.13220000267028809, 0.10589999705553055, 0.09470000118017197, 0.09300000220537186, 0.09210000187158585, 0.09030000120401382, 0.0877000018954277, 0.08510000258684158, 0.08110000193119049, 0.07970000058412552, 0.07940000295639038, 0.07829999923706055, 0.0771000012755394, 0.07649999856948853, 0.07639999687671661, 0.07620000094175339, 0.07609999924898148, 0.0746999979019165, 0.07289999723434448, 0.07209999859333038, 0.07109999656677246, 0.07050000131130219, 0.07050000131130219, 0.06800000369548798, 0.06759999692440033, 0.06750000268220901, 0.06750000268220901, 0.06589999794960022, 0.0658000037074089, 0.06549999862909317, 0.06520000100135803, 0.06360000371932983, 0.06430000066757202, 0.06239999830722809, 0.0608999989926815, 0.061500001698732376, 0.06069999933242798, 0.054999999701976776, 0.060100000351667404, 0.049800001084804535, 0.04580000042915344, 0.1680999994277954, 0.14659999310970306, 0.13779999315738678, 0.12060000002384186, 0.11860000342130661, 0.11219999939203262, 0.10490000247955322, 0.10300000011920929, 0.10199999809265137, 0.1005999967455864, 0.10029999911785126, 0.09960000216960907, 0.09860000014305115, 0.09790000319480896, 0.09690000116825104, 0.09679999947547913, 0.09619999676942825, 0.09610000252723694, 0.09480000287294388, 0.09390000253915787, 0.0934000015258789, 0.09309999644756317, 0.0925000011920929, 0.09160000085830688, 0.08820000290870667, 0.0868000015616417, 0.08609999716281891, 0.08579999953508377, 0.08500000089406967, 0.08470000326633453, 0.08240000158548355, 0.08219999819993973, 0.07580000162124634, 0.06909999996423721, 0.07460000365972519, 0.07569999992847443, 0.07429999858140945, 0.07569999992847443, 0.059700001031160355, 0.06809999793767929, 0.05420000106096268], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -6.604100227355957, -6.609300136566162, -6.649700164794922, -6.656799793243408, -6.615900039672852, -6.64109992980957, -6.644700050354004, -6.632800102233887, -6.668600082397461, -6.644700050354004, -6.642099857330322, -6.6828999519348145, -6.663899898529053, -6.66510009765625, -6.644400119781494, -6.6809000968933105, -6.662799835205078, -6.700099945068359, -6.686600208282471, -6.666200160980225, -6.662600040435791, -6.6875, -6.65310001373291, -6.643799781799316, -6.657299995422363, -6.679299831390381, -6.6697001457214355, -6.667300224304199, -6.638500213623047, -6.696599960327148, -6.673799991607666, -6.6483001708984375, -6.6442999839782715, -6.650899887084961, -6.647600173950195, -6.660399913787842, -6.6529998779296875, -6.6493000984191895, -6.655300140380859, -6.649099826812744, -6.648900032043457, -6.6554999351501465, -6.649600028991699, -6.65880012512207, -6.65880012512207, -6.660799980163574, -6.625, -6.646699905395508, -6.619200229644775, -6.605899810791016, -6.59689998626709, -6.611800193786621, -6.638899803161621, -6.639900207519531, -6.646999835968018, -6.650700092315674, -6.640200138092041, -6.68310022354126, -6.6620001792907715, -6.685299873352051, -6.640200138092041, -6.700500011444092, -6.641499996185303, -6.63100004196167, -6.639100074768066, -6.63700008392334, -6.688799858093262, -6.624800205230713, -6.649700164794922, -6.615600109100342, -6.65880012512207, -6.653200149536133, -6.634500026702881, -6.661399841308594, -6.657700061798096, -6.644000053405762, -6.633699893951416, -6.636899948120117, -6.650899887084961, -6.63670015335083, -6.653500080108643, -6.651299953460693, -6.641499996185303, -6.645999908447266, -6.651899814605713, -6.644599914550781, -6.640100002288818, -6.6433000564575195, -6.648900032043457, -6.650000095367432, -6.683000087738037, -6.686299800872803, -6.644000053405762, -6.683300018310547, -6.677199840545654, -6.681600093841553, -6.728400230407715, -6.667200088500977, -6.696100234985352, -6.681600093841553, -6.657199859619141, -6.696000099182129, -6.680799961090088, -6.685500144958496, -6.711100101470947, -6.716800212860107, -6.6940999031066895, -6.684899806976318, -6.69350004196167, -6.684800148010254, -6.68209981918335, -6.6809000968933105, -6.699699878692627, -6.681600093841553, -6.694300174713135, -6.6596999168396, -6.690899848937988, -6.687399864196777, -6.666900157928467, -6.701200008392334, -6.673900127410889, -6.656499862670898, -6.671599864959717, -6.668300151824951, -6.678500175476074, -6.677499771118164, -6.677800178527832, -6.660799980163574, -6.65910005569458, -6.682300090789795, -6.670000076293945, -6.673699855804443, -6.673500061035156, -6.663899898529053, -6.676499843597412, -6.674900054931641, -6.673500061035156, -6.673699855804443, -6.672800064086914, -6.6722002029418945, -6.669899940490723, -6.67609977722168, -6.676000118255615, -6.677499771118164, -6.678699970245361, -6.677599906921387, -6.644100189208984, -6.630099773406982, -6.681700229644775, -6.645699977874756, -6.663000106811523, -6.668499946594238, -6.649199962615967, -6.646500110626221, -6.642000198364258, -6.694200038909912, -6.667699813842773, -6.696300029754639, -6.685699939727783, -6.670199871063232, -6.658100128173828, -6.7032999992370605, -6.709000110626221, -6.719799995422363, -6.705399990081787, -6.69379997253418, -6.689700126647949, -6.691699981689453, -6.682000160217285, -6.655700206756592, -6.704800128936768, -6.684599876403809, -6.686800003051758, -6.69189977645874, -6.705399990081787, -6.7291998863220215, -6.689700126647949, -6.65880012512207, -6.649400234222412, -6.650400161743164, -6.676400184631348, -6.662600040435791, -6.658400058746338, -6.67519998550415, -6.647200107574463, -6.64900016784668, -6.666900157928467, -6.655399799346924, -6.652599811553955, -6.676499843597412, -6.673399925231934, -6.6666998863220215, -6.671199798583984, -6.6707000732421875, -6.671599864959717, -6.6717000007629395, -6.673500061035156, -6.674499988555908, -6.639500141143799, -6.662899971008301, -6.628399848937988, -6.651500225067139, -6.678199768066406, -6.630899906158447, -6.650599956512451, -6.686200141906738, -6.676700115203857, -6.685200214385986, -6.6803998947143555, -6.6508002281188965, -6.640900135040283, -6.644499778747559, -6.654099941253662, -6.666500091552734, -6.668600082397461, -6.633600234985352, -6.6757001876831055, -6.678500175476074, -6.697400093078613, -6.64870023727417, -6.651000022888184, -6.676599979400635, -6.679500102996826, -6.681099891662598, -6.672900199890137, -6.695799827575684, -6.661300182342529, -6.666900157928467, -6.66510009765625, -6.64769983291626, -6.6528000831604, -6.658699989318848, -6.642899990081787, -6.6579999923706055, -6.656199932098389, -6.654099941253662, -6.658999919891357, -6.664700031280518, -6.658999919891357, -6.664999961853027, -6.663700103759766, -6.658599853515625, -6.65939998626709, -6.66349983215332, -6.612500190734863, -6.647299766540527, -6.619200229644775, -6.638899803161621, -6.650100231170654, -6.65910005569458, -6.647799968719482, -6.650199890136719, -6.6194000244140625, -6.625699996948242, -6.664299964904785, -6.665200233459473, -6.637400150299072, -6.632800102233887, -6.6722002029418945, -6.677000045776367, -6.6458001136779785, -6.655099868774414, -6.706099987030029, -6.675300121307373, -6.69789981842041, -6.6641998291015625, -6.627399921417236, -6.703100204467773, -6.67110013961792, -6.698400020599365, -6.6641998291015625, -6.642300128936768, -6.693399906158447, -6.684599876403809, -6.648900032043457, -6.646500110626221, -6.661799907684326, -6.6427001953125, -6.644800186157227, -6.665800094604492, -6.641200065612793, -6.655799865722656, -6.663300037384033, -6.6554999351501465, -6.656499862670898, -6.6545000076293945, -6.658299922943115, -6.662799835205078, -6.662799835205078, -6.667099952697754, -6.6255998611450195, -6.659599781036377, -6.686600208282471, -6.653299808502197, -6.6504998207092285, -6.617700099945068, -6.693699836730957, -6.677299976348877, -6.6894001960754395, -6.689599990844727, -6.672100067138672, -6.664299964904785, -6.648399829864502, -6.687900066375732, -6.690999984741211, -6.653800010681152, -6.661499977111816, -6.656599998474121, -6.68209981918335, -6.661099910736084, -6.706200122833252, -6.684100151062012, -6.684000015258789, -6.697999954223633, -6.67710018157959, -6.688899993896484, -6.689599990844727, -6.684599876403809, -6.682499885559082, -6.678199768066406, -6.671500205993652, -6.673500061035156, -6.662899971008301, -6.678100109100342, -6.676300048828125, -6.645299911499023, -6.675000190734863, -6.673600196838379, -6.66510009765625, -6.663899898529053, -6.671199798583984, -6.671800136566162, -6.667200088500977, -6.663300037384033, -6.671899795532227, -6.676499843597412, -6.674600124359131, -6.675899982452393, -6.676000118255615, -6.67579984664917, -6.618000030517578, -6.626100063323975, -6.590099811553955, -6.644199848175049, -6.621300220489502, -6.6346001625061035, -6.6184000968933105, -6.652100086212158, -6.627900123596191, -6.651599884033203, -6.641600131988525, -6.64769983291626, -6.661300182342529, -6.701200008392334, -6.672299861907959, -6.6483001708984375, -6.673299789428711, -6.667099952697754, -6.65880012512207, -6.646500110626221, -6.64169979095459, -6.661099910736084, -6.632400035858154, -6.68720006942749, -6.646200180053711, -6.655900001525879, -6.627299785614014, -6.661499977111816, -6.666500091552734, -6.65939998626709, -6.639400005340576, -6.646699905395508, -6.640699863433838, -6.657599925994873, -6.657100200653076, -6.650700092315674, -6.654099941253662, -6.6554999351501465, -6.651599884033203, -6.6545000076293945, -6.651700019836426, -6.653299808502197, -6.571800231933594, -6.585400104522705, -6.602499961853027, -6.595699787139893, -6.674799919128418, -6.6468000411987305, -6.5879998207092285, -6.5903000831604, -6.623899936676025, -6.629799842834473, -6.644499778747559, -6.638700008392334, -6.663099765777588, -6.608799934387207, -6.618000030517578, -6.634399890899658, -6.613900184631348, -6.617199897766113, -6.625, -6.654399871826172, -6.618000030517578, -6.624599933624268, -6.632400035858154, -6.6321001052856445, -6.603799819946289, -6.638899803161621, -6.640100002288818, -6.669099807739258, -6.66349983215332, -6.642600059509277, -6.658199787139893, -6.636600017547607, -6.6468000411987305, -6.634200096130371, -6.629499912261963, -6.631800174713135, -6.631199836730957, -6.621300220489502, -6.636099815368652, -6.634099960327148, -6.636899948120117, -6.538300037384033, -6.551400184631348, -6.553500175476074, -6.586999893188477, -6.5854997634887695, -6.617599964141846, -6.616700172424316, -6.630199909210205, -6.632999897003174, -6.623700141906738, -6.627399921417236, -6.646999835968018, -6.6118998527526855, -6.607100009918213, -6.631199836730957, -6.6468000411987305, -6.638800144195557, -6.63539981842041, -6.639500141143799, -6.603400230407715, -6.6016998291015625, -6.635499954223633, -6.63100004196167, -6.655900001525879, -6.639900207519531, -6.634900093078613, -6.670599937438965, -6.623600006103516, -6.6041998863220215, -6.639200210571289, -6.593299865722656, -6.630899906158447, -6.61460018157959, -6.60230016708374, -6.619800090789795, -6.625500202178955, -6.626399993896484, -6.6305999755859375, -6.632199764251709, -6.633600234985352, -6.6346001625061035]}, "token.table": {"Topic": [2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2], "Freq": [0.29221707582473755, 0.2932423949241638, 0.29832911491394043, 0.2956927418708801, 0.29647254943847656, 0.3018072843551636, 0.30596765875816345, 0.29982390999794006, 0.29492056369781494, 0.29537346959114075, 0.30485910177230835, 0.2974414825439453, 0.2974414825439453, 0.30661559104919434, 0.29918476939201355, 0.2972743511199951, 0.2964177429676056, 0.29950082302093506, 0.3049008548259735, 0.3005058765411377, 0.3108412027359009, 0.30499738454818726, 0.3068070411682129, 0.3030564785003662, 0.30546775460243225, 0.29895344376564026, 0.29165858030319214, 0.312458336353302, 0.2991945147514343, 0.29557228088378906, 0.2998994290828705, 0.3070632815361023, 0.30305930972099304, 0.2990010678768158, 0.29987281560897827, 0.29992878437042236, 0.3033682703971863, 0.29679325222969055, 0.3066313564777374, 0.29500535130500793, 0.29772454500198364], "Term": ["aabar", "alam bangun", "angkat", "atur", "baca", "bangun bandar", "bantu", "dasar", "dunia", "ekonomi", "ekonomi sosial bangun", "fasal lembaga", "fasal lembaga", "industri diplomasi", "isi", "jppm", "jual beli", "kongsi alam bangun", "kongsi alam negara", "kongsi mahir bangun", "kukuh", "latih teknikal", "latih teknikal diplomasi", "maju didik", "malu", "masyarakat asli", "melayu cina", "pelbagai", "pelopor", "pengaruh", "prestasi", "program teknikal malaysia", "rangka", "serang", "serius pimpin", "suasana", "sumber baca", "tanah", "teknikal industri diplomasi", "teknikal mtcp sedia", "wajar nyata"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [8, 7, 1, 4, 3, 6, 9, 2, 5, 10]};

    function LDAvis_load_lib(url, callback){
      var s = document.createElement('script');
      s.src = url;
      s.async = true;
      s.onreadystatechange = s.onload = callback;
      s.onerror = function(){console.warn("failed to load library " + url);};
      document.getElementsByTagName("head")[0].appendChild(s);
    }

    if(typeof(LDAvis) !== "undefined"){
       // already loaded: just create the visualization
       !function(LDAvis){
           new LDAvis("#" + "ldavis_el298249953790166480605917", ldavis_el298249953790166480605917_data);
       }(LDAvis);
    }else if(typeof define === "function" && define.amd){
       // require.js is available: use it to load d3/LDAvis
       require.config({paths: {d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min"}});
       require(["d3"], function(d3){
          window.d3 = d3;
          LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
            new LDAvis("#" + "ldavis_el298249953790166480605917", ldavis_el298249953790166480605917_data);
          });
        });
    }else{
        // require.js not available: dynamically load d3 & LDAvis
        LDAvis_load_lib("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js", function(){
             LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
                     new LDAvis("#" + "ldavis_el298249953790166480605917", ldavis_el298249953790166480605917_data);
                })
             });
    }
    </script>



Train LDA model
---------------

.. code:: python

    lda = malaya.topic_model.lda(corpus,10,stemming=None,vectorizer='skip-gram',ngram=(1,4),skip=3)


.. parsed-literal::

    /usr/local/lib/python3.6/site-packages/sklearn/decomposition/online_lda.py:536: DeprecationWarning: The default value for 'learning_method' will be changed from 'online' to 'batch' in the release 0.20. This warning was introduced in 0.18.
      DeprecationWarning)


Print topics
^^^^^^^^^^^^

.. code:: python

    lda.top_topics(5, top_n = 10, return_df = True)




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
          <th>topic 0</th>
          <th>topic 1</th>
          <th>topic 2</th>
          <th>topic 3</th>
          <th>topic 4</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>maju bidang ajar</td>
          <td>parti</td>
          <td>bangun negara selatan</td>
          <td>awam</td>
          <td>projek</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ajar</td>
          <td>tangguh</td>
          <td>alam negara selatan</td>
          <td>terima</td>
          <td>laku</td>
        </tr>
        <tr>
          <th>2</th>
          <td>maju bidang proses ajar</td>
          <td>ros</td>
          <td>negara selatan</td>
          <td>pilih</td>
          <td>asli</td>
        </tr>
        <tr>
          <th>3</th>
          <td>maju proses ajar</td>
          <td>jppm</td>
          <td>alam bangun negara selatan</td>
          <td>putus</td>
          <td>mdb</td>
        </tr>
        <tr>
          <th>4</th>
          <td>didik ajar</td>
          <td>rana</td>
          <td>kongsi bangun negara selatan</td>
          <td>bayar</td>
          <td>hutang</td>
        </tr>
        <tr>
          <th>5</th>
          <td>maju bidang didik ajar</td>
          <td>nyata</td>
          <td>kongsi alam negara selatan</td>
          <td>ambil</td>
          <td>wang</td>
        </tr>
        <tr>
          <th>6</th>
          <td>maju didik ajar</td>
          <td>pilih</td>
          <td>bangun</td>
          <td>menteri</td>
          <td>raja</td>
        </tr>
        <tr>
          <th>7</th>
          <td>tingkat didik proses ajar</td>
          <td>daftar</td>
          <td>kongsi</td>
          <td>isu</td>
          <td>selesai</td>
        </tr>
        <tr>
          <th>8</th>
          <td>didik proses ajar</td>
          <td>umno</td>
          <td>negara</td>
          <td>politik</td>
          <td>malaysia</td>
        </tr>
        <tr>
          <th>9</th>
          <td>bidang proses ajar</td>
          <td>kelulus</td>
          <td>alam negara</td>
          <td>tindak</td>
          <td>masyarakat</td>
        </tr>
      </tbody>
    </table>
    </div>



Important sentences based on topics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    lda.get_sentences(5)




.. parsed-literal::

    ['bantu negara negara maju bidang ancang ekonomi wang dagang tani didik latih teknikal industri diplomasi',
     'bantu negara negara maju bidang ancang ekonomi wang dagang tani didik latih teknikal industri diplomasi',
     'laku raja tingkat maju bidang didik proses ajar ajar',
     'laku raja tingkat maju bidang didik proses ajar ajar',
     'tolak sebarang percubaan merosak musnah tanah suci jaga tumpu islam']



Get topics
^^^^^^^^^^

.. code:: python

    lda.get_topics(10)




.. parsed-literal::

    [(0,
      'maju bidang ajar ajar maju bidang proses ajar maju proses ajar didik ajar maju bidang didik ajar maju didik ajar tingkat didik proses ajar didik proses ajar bidang proses ajar'),
     (1, 'parti tangguh ros jppm rana nyata pilih daftar umno kelulus'),
     (2,
      'bangun negara selatan alam negara selatan negara selatan alam bangun negara selatan kongsi bangun negara selatan kongsi alam negara selatan bangun kongsi negara alam negara'),
     (3, 'awam terima pilih putus bayar ambil menteri isu politik tindak'),
     (4, 'projek laku asli mdb hutang wang raja selesai malaysia masyarakat'),
     (5,
      'terjemah ambil ilmu bahasa bahasa ilmu kaji ambil langkah langkah bahasa bahasa tunggu'),
     (6,
      'malaysia bangun negara negara malaysia pesawat nyata kawasan bandar kawasan bandar dakwa'),
     (7, 'duduk urus takut giat hidup arab saudi bidang arab saudi pekan nilai'),
     (8,
      'rakyat bahasa rakyat malaysia malaysia niaga negara bahasa ilmu inggeris bahasa inggeris jual'),
     (9,
      'tanggung hutang capai matlamat dunia main negara teknikal silap individu')]



Visualize topics
^^^^^^^^^^^^^^^^

.. code:: python

    lda.visualize_topics(notebook_mode = True)




.. raw:: html


    <link rel="stylesheet" type="text/css" href="https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.css">


    <div id="ldavis_el298249993929764917568162"></div>
    <script type="text/javascript">

    var ldavis_el298249993929764917568162_data = {"mdsDat": {"x": [-0.112945341112951, -0.1689033372518357, 0.21532184758649506, 0.11658831672845305, -0.04337148173146946, 0.06613800567890737, 0.011288262026154227, -0.025624593854212726, -0.024016004470275833, -0.03447567359926506], "y": [0.2073235034530079, -0.21270714863488124, -0.044162053743126324, -0.00933312043906155, 0.0397996020445293, 0.00030305401717365863, 0.0080528663500681, 0.005624129836059783, 0.0020649875664019604, 0.00303417954982805], "topics": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "cluster": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Freq": [24.61762364031246, 19.87993212718284, 16.560476297657836, 11.411956116013226, 9.43787868526767, 7.197704665850548, 6.11269423507769, 2.018684275114424, 1.8269052474839484, 0.9361447100393413]}, "tinfo": {"Category": ["Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Default", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic1", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic2", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic3", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic4", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic5", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic6", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic7", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic8", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic9", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10", "Topic10"], "Freq": [13.0, 15.0, 8.0, 8.0, 16.0, 13.0, 26.0, 16.0, 16.0, 16.0, 17.0, 8.0, 11.0, 16.0, 6.0, 12.0, 9.0, 15.0, 9.0, 4.0, 9.0, 12.0, 9.0, 5.0, 4.0, 8.0, 22.0, 5.0, 5.0, 2.0, 6.587899404331065, 6.582723527638028, 6.57958412512532, 6.580586396083964, 6.5788823339169165, 6.580725323183667, 4.967627014945321, 4.968881838112019, 4.963816999690098, 4.965221444619736, 4.965381830747468, 4.963691508277804, 4.970795603357066, 4.962668101705472, 4.967794098649532, 3.3621164386661953, 3.36361945797512, 3.3599085054276343, 3.3615317138809377, 3.3599457549423017, 3.361213358044136, 3.359558046761562, 3.3619520898677657, 3.355655537666853, 3.3621743304424796, 3.3527327963498585, 3.36227303082802, 3.3586371892595106, 3.3645705006025777, 3.3556812003300105, 3.3590163269266915, 5.877860202394956, 3.3632909579753503, 3.3623630633323898, 3.3632730368253627, 4.201841445355887, 6.083930969060346, 4.196287685700658, 4.200460404416833, 4.984363267812663, 3.781485284649263, 4.368323178306798, 3.3703224284757676, 3.3646017742281167, 3.3627097234021224, 5.089930682650278, 5.091892475027403, 5.086189347776762, 5.084361178312137, 5.0894226437776, 5.089595279201713, 5.088129844345109, 5.089005995954251, 5.089295072763944, 5.089263784452562, 5.0870160245040985, 5.09067851924933, 5.086127110168578, 5.091312533928023, 5.088464016161044, 5.087134568291664, 5.086161528276182, 5.086443381490528, 5.086338768430236, 3.8773873543049766, 2.679741041965723, 2.6737255303623155, 2.6786654293185097, 2.671541126631945, 2.6706774995660667, 2.6706849921451035, 2.675790811154981, 2.6760187361310286, 2.6714939000964955, 2.6702828166697063, 2.6719819941616563, 3.872766705626779, 5.0916613131522, 3.8951440511405577, 2.676562399922336, 3.8788026496830996, 2.675515310248589, 3.2402311116631997, 2.8120082163433118, 12.42783138719525, 5.888171638551086, 5.888386385497008, 5.877706890912964, 5.873449104080366, 5.85995860413394, 4.941837146585609, 4.437014392091414, 4.3847625668108865, 4.4008080807433565, 3.971490572215974, 8.107240141923306, 3.9268618932184647, 3.9575908936145363, 3.480931687910566, 3.4560911507837884, 3.4349625657427754, 3.4367929361440464, 4.25800893930458, 4.23456445136074, 3.452936073657296, 3.3952282969796297, 2.984015908016706, 10.214424375465615, 2.986934790692615, 2.9864982836934555, 2.9854905842142845, 2.9730602950767633, 2.9845447488815893, 2.979294450652365, 12.707056337309735, 11.665946089996316, 12.471516976784207, 11.902203691680349, 6.641170308364637, 10.823010951456677, 4.874538523949325, 10.716526872596836, 6.609502267957336, 3.9771585445029918, 3.9440363648920176, 10.16591873400249, 4.464102707294019, 6.269698362133015, 6.87298986871185, 5.462250106532181, 4.8277594118286995, 5.9905880335868655, 4.577786493840777, 4.480296678786164, 4.581299328986283, 7.809227740902209, 4.983247742612174, 4.490437387526597, 4.498970839744315, 3.9345852354474737, 3.4203175623388105, 2.902433434186112, 2.8062303920122753, 3.7567251515830504, 4.494109446880232, 2.3691121232638452, 2.3379821269113954, 2.3349008869831294, 2.3395423121292374, 2.2950438541006375, 2.296059377855335, 2.288676661600015, 2.2937566890882746, 2.3137831463553957, 2.315374462538667, 2.3097905696728853, 2.2929987476903193, 2.2646549581208033, 1.8365412597452548, 4.608375957908966, 4.37213185147388, 1.8320517646516041, 1.7922676840104137, 1.8325560216285353, 1.7479078354110833, 5.665678180255352, 7.546120587571593, 2.655033729081886, 5.594318686377827, 5.0663680804861295, 5.785505161564424, 4.519970807767709, 5.643671462874178, 3.5812614985270907, 3.7964920687684796, 5.332451462345192, 3.804392861714599, 4.154581588455318, 2.860170048603657, 2.857112984298248, 2.5095740141452008, 2.86963321553088, 3.83628557183592, 2.878154706481576, 3.742728740756947, 3.3767555609083972, 3.0106839591124057, 3.0300170005821423, 1.9174910252046322, 1.7574958947689978, 1.3349450097755264, 1.3303262071708726, 1.3320585672947973, 1.332485121221161, 1.3312869792670614, 1.3349441595213833, 1.3321562530779743, 1.3296564761809375, 1.3330447693606957, 1.3301454927299154, 1.3301488727991788, 1.3293012109134295, 1.328784151947977, 1.333069454452846, 1.3277164836349094, 1.3321986322269401, 1.327759880736643, 1.3371068581357073, 1.3292698543029997, 1.3282748067233652, 1.3302939021540785, 1.3292869045997389, 1.3277850741586616, 1.3337079294606502, 1.3293774074969265, 1.3292149173107815, 1.3255110580352876, 1.3346999780987503, 1.3315807263121888, 1.3360478610643631, 1.334253730386213, 1.334792160696383, 1.3353995820250375, 1.3341713037024505, 1.3364401058362052, 1.3309921654936936, 1.969472288860406, 1.7488199905185173, 1.3351225427088835, 1.3317798854622793, 1.3322374509622543, 1.3356310082976035, 1.33475014673644, 3.2267280475631868, 1.332930091615594, 2.55506266089682, 1.3330450156112637, 1.550819680175982, 1.7800960660227196, 1.9838407539670126, 1.3855644470431558, 1.5203239278036587, 1.3330476645667115, 1.3345490270889369, 1.3339878021325546, 1.333352458738177, 4.011241389054285, 4.0097021725532, 5.503510765077369, 3.5832285072188013, 3.270717087725006, 3.1980164433434743, 11.130799252482571, 3.1141032046668284, 2.7220315262564534, 2.6763384347926245, 2.67699916567153, 6.614015391532916, 4.055526030935617, 2.233203251187492, 1.8927440056105116, 1.8610172735427737, 1.9147905924284703, 1.863477514769805, 1.8628501737483405, 1.8644458149236134, 1.8598773480048878, 1.82446334907242, 3.053690146736867, 1.429762553452319, 1.4286336605852823, 1.4198217012092547, 2.677232219296458, 1.3647339971561228, 2.713992387292839, 2.081986767155667, 3.1137250679870063, 2.7275840740936275, 4.576771282760608, 3.0714874580964375, 3.854716583354715, 4.927254757470159, 3.289684093181144, 3.1819297790630205, 4.113343346205371, 2.7593913488412922, 2.8198200316462096, 2.327284618683963, 2.7209256534985617, 3.864191109743847, 4.292104652796378, 2.0396925802591412, 2.0417207010298113, 2.0744557777727266, 2.0308285765817793, 3.6236549820420842, 1.881780709853121, 1.9341444029876953, 1.63632203459514, 1.6370532361315429, 1.5980861279487204, 1.4261493624710477, 1.4296363319161052, 1.4049558354568168, 1.3912044185028367, 1.1194010428619032, 1.121052872521199, 1.1219869132454603, 1.1229455361637004, 1.1197341859527778, 1.1178066405405744, 1.1163900410118777, 1.117007768598014, 1.1175275058631455, 1.120157574653292, 1.1198442902188992, 1.082344434468002, 2.636546718389065, 1.082940334275198, 1.922407151640541, 1.5597433181257554, 1.5532025644634433, 1.5480393757324609, 2.4903091330528717, 5.627026409963397, 1.8397726362001388, 3.0433484201732166, 1.5993225948007022, 2.5667892957248313, 1.5576898747784145, 2.7882764079524467, 1.679571626910225, 3.3085487842635692, 1.7561266691665265, 1.523624605864228, 1.5623488123550373, 1.5485773550628428, 1.0122839609384215, 0.8077663821415587, 0.8078094053107714, 0.807395220974718, 0.7672492060823172, 0.7708540835869334, 0.7692792050049022, 0.5468207703787976, 0.5467187382314828, 0.5266887846887849, 0.7684331374654894, 1.0354668032740637, 1.416134594652619, 0.5420015000972773, 0.30421371829285443, 0.2851198925391222, 0.7839245646659093, 1.0528074466609243, 0.5443008257707821, 0.3044324641304879, 0.2856599570881951, 0.28506722865998285, 0.2862553535722956, 0.28570148714721855, 0.285294573621606, 0.3044253952629582, 0.3028231885484728, 0.79234935506984, 0.7855440734547965, 0.28370923144113025, 0.8076224842445688, 0.5457211977275662, 0.3770190323973858, 0.5277515959717224, 0.5458937902812505, 0.5280645636839886, 0.5514807757329944, 0.30428061569799253, 0.3123362020071678, 0.30537188396151194, 0.3051054724859864, 2.2406826679969294, 0.915172306607691, 0.679586083788233, 0.5002569234407308, 0.4991651110908264, 0.49797662309036095, 0.500356245726789, 0.4984327872611163, 0.49944172276466003, 1.3638090937564782, 0.48308236105763946, 0.46682350232554626, 0.9132600776438833, 0.9277388248687217, 1.3622131943692093, 0.46604675452512073, 0.501454226520924, 0.6995203564516578, 0.6994042344834774, 0.25206135943190044, 0.2691841605363134, 0.5007809002815462, 0.2518546302677462, 0.6998523928670947, 0.25337843288193623, 0.2679277001292999, 0.2541237417147152, 0.2520466791527724, 1.380582516116005, 1.3628040270781043, 0.68504362709251, 0.6987883942390298, 0.5012403466628341, 0.4669607706616052, 0.48320507144736796, 0.48367951096125783, 0.490225429852289, 0.4675876123466894, 0.5642983445438465, 0.29699034878394076, 0.2971583263616182, 0.15574211442762126, 0.3721555152055454, 0.16535301436140049, 0.15551210673534122, 0.16562923606820082, 0.15541787788543598, 0.15627115241275946, 0.1660963171656266, 0.16490405800461996, 0.2863217497062884, 0.1553314866154913, 0.165838744889676, 0.16641036400466397, 0.16540694800205905, 0.1561065578645198, 0.10596726279990992, 0.16544587816235226, 0.16508959040588161, 0.16805790881243576, 0.1548898183405615, 0.16500815125483717, 0.4313197601255078, 0.15472862240489058, 0.15542686466437078, 0.023643232519096398, 0.023255442378560517, 0.022677136486882566, 0.16860657538561494, 0.03523932167821428, 0.026047349445789698, 0.03413809821135821], "Term": ["rakyat", "hutang", "bahasa", "ambil", "parti", "asli", "malaysia", "projek", "mdb", "rana", "raja", "awam", "terima", "laku", "rakyat malaysia", "selesai", "pilih", "wang", "bayar", "ilmu", "niaga", "menteri", "jual", "bahasa ilmu", "duduk", "putus", "negara", "tangguh", "urus", "terjemah", "bangun negara selatan", "alam negara selatan", "kongsi bangun negara selatan", "alam bangun negara selatan", "kongsi alam negara selatan", "negara selatan", "kongsi alam bangun negara", "kongsi alam negara", "alam bangun negara", "bangun negara", "kongsi bangun negara", "sedia kongsi bangun negara", "alam negara", "sedia alam bangun negara", "sedia kongsi alam negara", "malaysia alam negara", "malaysia kongsi negara selatan", "sedia malaysia bangun negara", "kongsi negara selatan", "bangun selatan", "kongsi alam bangun selatan", "kongsi negara selatan selatan", "malaysia kongsi bangun negara", "sedia alam", "malaysia alam negara selatan", "kongsi negara", "malaysia alam bangun negara", "program kerjasama teknikal malaysia", "sedia bangun", "alam bangun selatan", "kerjasama teknikal malaysia kongsi", "kongsi", "malaysia bangun negara", "selatan", "kerjasama teknikal malaysia", "kongsi alam", "bangun", "alam", "sedia", "negara", "undi", "malaysia", "kongsi bangun", "kongsi alam bangun", "alam bangun", "didik ajar", "maju bidang ajar", "proses ajar", "tingkat bidang didik ajar", "maju didik ajar", "maju bidang didik ajar", "tingkat maju bidang ajar", "bidang proses ajar", "tingkat didik proses ajar", "didik proses ajar", "bidang didik proses ajar", "maju proses ajar", "bidang didik ajar", "maju bidang proses ajar", "tingkat maju proses ajar", "tingkat bidang proses ajar", "maju didik proses ajar", "tingkat maju didik ajar", "bidang ajar", "maju bidang", "negara maju bidang ekonomi", "bantu negara bidang ekonomi", "negara bidang ancang wang", "negara bidang ekonomi", "bantu negara maju bidang", "bantu negara ancang", "negara bidang ancang", "negara ancang ekonomi wang", "negara bidang ancang ekonomi", "negara ancang", "bantu negara bidang ancang", "bidang", "ajar", "maju", "negara maju ancang", "didik", "negara maju", "wang", "negara", "asli", "tempoh hutang", "projek projek", "wang hutang hutang", "hutang hutang", "hutang selesai", "wang mdb", "hutang pendek", "najib", "hutang mdb", "kerja", "masyarakat", "beli", "indonesia", "rana malaysia", "jual syarikat", "air", "swasta", "jakoa", "masyarakat asli", "bank", "wujud laku", "selesai tempoh", "selesai", "mdb selesai", "pendek", "mdb hutang", "hak", "mdb hutang pendek", "putrajaya", "projek", "hutang", "laku", "mdb", "wujud", "wang", "tempoh", "raja", "syarikat", "kait", "khusus", "malaysia", "wang hutang", "menteri", "rana", "kena", "perdana", "negara", "jual", "laksana", "bangun", "awam", "politik", "gagal", "pilih raya", "serah", "jawat awam", "ambil tindak", "rendah", "kumpul", "raya", "jalan lancar", "seleweng", "dana seleweng", "ph putus", "angkat", "politik arah", "gagal dana", "parti putus", "semak imbang", "imbang", "semak", "berbeza", "jawat ahli", "gembira", "tindak", "jawat", "gembira lihat projek", "menteri menteri", "gembira lihat", "niaga jho", "putus", "terima", "janji", "ambil", "isu", "pilih", "ahli", "bayar", "tuju", "hadap", "menteri", "jalan", "kena", "low", "perdana menteri", "hubung", "dana", "parti", "sokong", "mdb", "rana", "perdana", "syarikat", "pesawat", "kawasan bandar", "bangun malaysia main mahir", "negara bangun peran", "negara bangun main peran", "main peran alam mahir", "bangun malaysia kongsi", "main kongsi mahir", "malaysia peran kongsi mahir", "bangun main peran alam", "kongsi bangun kawasan", "bangun kawasan bandar", "main kongsi alam kawasan", "negara malaysia peran", "peran kongsi bangun kawasan", "kongsi alam kawasan bandar", "peran kongsi bangun bandar", "peran kongsi alam bandar", "kongsi kawasan", "alam kawasan", "negara malaysia main", "kongsi alam bandar", "negara malaysia main peran", "peran mahir bangun bandar", "bangun malaysia main alam", "bangun main peran kongsi", "alam mahir bangun", "negara main peran kongsi", "negara bangun malaysia alam", "kongsi mahir kawasan bandar", "bangun malaysia kongsi alam", "mac", "kongsi alam kawasan", "negara main peran alam", "peran kongsi mahir", "main kongsi mahir kawasan", "malaysia main alam bangun", "peran mahir", "negara malaysia", "kawasan", "peran mahir kawasan bandar", "main peran bangun kawasan", "main kongsi bangun", "bangun kongsi alam mahir", "negara bangun main", "malaysia", "negara malaysia kongsi alam", "bangun", "main peran", "bandar", "nyata", "negara", "anti", "dakwa", "kongsi mahir bangun", "malaysia kongsi", "kongsi", "malaysia alam", "inggeris", "bahasa inggeris", "rakyat malaysia", "makan", "harga", "hasil", "rakyat", "kembang", "hutang makan", "bahasa inggeris bahasa", "inggeris bahasa", "bahasa", "bahasa ilmu", "untung", "percaya", "malaysia niaga", "positif", "bekerjasama", "milik sewa", "sewa tanah", "faham", "hasil makan", "babit", "pimpin negara", "jual harga", "rakyat pimpin", "bahasa bahasa", "rumah sewa", "nilai", "sumbang", "sewa", "ilmu", "niaga", "masuk", "jual", "malaysia", "tingkat", "bayar", "negara", "lihat", "mudah", "tanah", "hutang", "ros", "tangguh", "tangguh pilih parti", "pilih parti", "parti jppm", "tangguh pilih", "jppm", "umno tangguh", "tangguh parti", "jppm daftar", "daftar ph", "gambar", "allahyarham nik", "nik", "parti lembaga", "umno pilih", "wajar tulis", "timbang tulis nyata", "wajar nyata", "timbang nyata", "timbang wajar tulis", "timbang wajar", "wajar tulis nyata", "timbang wajar nyata", "timbang wajar tulis nyata", "tulis nyata", "timbang tulis", "parti daftar", "daftar", "ikan", "tulis", "majlis", "murah", "murah hati", "kelulus", "parti", "hilang", "nyata", "berita", "umno", "takut", "pilih", "hati", "rana", "lembaga", "salah", "sedia", "malaysia", "giat hidup", "saudi", "arab", "arab saudi", "giat hidup duduk", "giat duduk", "sukan", "cari jalan", "bidang selamat", "hidup duduk", "giat", "takut", "duduk", "jaga", "hala", "sukan rana", "hidup", "urus", "cari", "langgar", "hebat", "turun", "malaysia khusus", "generasi alam", "generasi", "suci", "perintah", "pekan", "nilai", "ganas", "bidang", "selamat", "kukuh", "khusus", "jalan", "alam", "malaysia", "komprehensif", "asli", "negara kerjasama", "atur", "terjemah", "ambil langkah", "pelabur", "bahan uji", "bahan", "bahan uji kaji", "bahan kaji", "uji kaji", "uji", "ilmu bahasa", "dasar kaji", "pelabur urus", "langkah", "kaji", "ilmu", "lapor", "ambil putus", "tunggu", "ilmu ilmu", "sukan rana", "bawa", "anak", "muka", "bahasa bahasa", "titik", "pergi", "nila", "dar ambil", "ambil", "bahasa", "urus", "bahasa ilmu", "tumbuh", "kelulus", "dasar", "putus", "rana", "raja", "tanggung", "dunia", "matlamat", "hala", "capai", "teruk", "sasar", "jejas", "serang", "arah capai", "individu", "global", "main", "lonjak", "strategi", "silap", "hapus", "tn", "baiah", "pandang", "tubuh", "teknikal", "arah", "malu", "hutang", "tuju", "pelbagai", "dasar kaji", "bahan uji kaji", "pelabur urus", "negara", "bukti", "benda", "sedia"], "Total": [13.0, 15.0, 8.0, 8.0, 16.0, 13.0, 26.0, 16.0, 16.0, 16.0, 17.0, 8.0, 11.0, 16.0, 6.0, 12.0, 9.0, 15.0, 9.0, 4.0, 9.0, 12.0, 9.0, 5.0, 4.0, 8.0, 22.0, 5.0, 5.0, 2.0, 7.2500730988904305, 7.246068642830178, 7.2441016715005, 7.247084375646174, 7.247162370854246, 7.256174136713333, 5.625321879523678, 5.627270220773654, 5.621835178569613, 5.624539087965815, 5.62758454646506, 5.626221349823348, 5.635936393094136, 5.63365677755797, 5.639990585242188, 4.00961940440138, 4.011521954234006, 4.009452634246562, 4.011688326793887, 4.009924352696506, 4.0125281848641885, 4.0115743476628225, 4.014471955824728, 4.0073162668051285, 4.015767211037369, 4.004527501433507, 4.016086330116874, 4.011806351645928, 4.019441370827525, 4.009130537695374, 4.013700636318861, 8.471570816388814, 4.021416978975496, 4.020049974604771, 4.0220210211411915, 6.330777745102499, 15.130922161933059, 7.5499169242356645, 9.678759227794117, 22.646409593029144, 7.652460437255961, 26.19648207139505, 5.244894901752475, 5.240625343556451, 5.238272412980409, 5.761575795093861, 5.764542163059032, 5.75891856124354, 5.757151399018973, 5.764575048974629, 5.766159795466551, 5.764680245481533, 5.766368014663683, 5.767542835494789, 5.767759324230902, 5.765582165878107, 5.770535767678592, 5.765422304469884, 5.773306701159448, 5.770122413887817, 5.770075795894056, 5.769372445854322, 5.770554118417029, 5.770512717496402, 4.557815316040813, 3.348311474922673, 3.346129191826616, 3.354071440113198, 3.3459083149893454, 3.347923709862187, 3.3485333241018083, 3.355380812725941, 3.3557997295121504, 3.350197742049352, 3.349229950400682, 3.35180075648597, 5.309908978711119, 7.794482606882831, 5.913568428507018, 3.3596521704974576, 6.273352181307998, 3.357950636738612, 15.93893821310551, 22.646409593029144, 13.575331033438616, 6.583274177370791, 6.586719103482559, 6.580786899139469, 6.590090287963833, 6.59347844150953, 5.6424779907543465, 5.146607684688937, 5.097940929975649, 5.1378975847820865, 4.668007989840379, 9.540819736341703, 4.663957565196378, 4.709368881559598, 4.178913285467143, 4.154640896240258, 4.141716372368433, 4.155683017835963, 5.150527720291642, 5.15063827236634, 4.2014013010942, 4.151578921392557, 3.677591800754059, 12.608419900770093, 3.687684502334691, 3.6879530958981617, 3.6878112037312665, 3.6727482361713415, 3.6869477206375274, 3.68519569222932, 16.359283873372558, 15.459318889605143, 16.990646099364536, 16.73182987327172, 8.878849095839099, 15.93893821310551, 6.520640045116741, 17.402004337137296, 10.289690109689507, 5.195726569292305, 5.152803866800269, 26.19648207139505, 6.461085512352684, 12.895332912130089, 16.963063372766413, 12.266117091714376, 8.91108372266689, 22.646409593029144, 9.183957210938631, 8.578462950595501, 15.130922161933059, 8.506993848651161, 5.723585311476426, 5.185643298399064, 5.224297822144511, 4.64960278819383, 4.115459209429333, 3.601122174853009, 3.510638285426513, 4.750921485325945, 5.7363449236015684, 3.055817616404134, 3.020285871122625, 3.024102507077934, 3.0322027262122164, 2.9802744006407536, 2.9829682062129024, 2.979484595453752, 2.9883792041177357, 3.029404379325238, 3.033100479185104, 3.026615799519652, 3.016911502723161, 2.9911599477222603, 2.512809895978968, 6.306382257368366, 5.988108729261357, 2.5224888101145955, 2.47359632806641, 2.5298151952564627, 2.434468995519947, 8.219018348036832, 11.370949860613818, 3.7571622521397976, 8.791409689403157, 8.154845667337764, 9.710991756329848, 7.201666354686313, 9.94546793621467, 5.813327779658336, 6.699531107505759, 12.895332912130089, 7.593603466569354, 12.266117091714376, 5.030982970115623, 5.447442782489048, 3.807390485720636, 5.957378041403843, 16.4149704815227, 6.189831348343904, 16.73182987327172, 16.963063372766413, 8.91108372266689, 10.289690109689507, 2.613602398566299, 2.5605164068344255, 2.002606136437702, 2.00107925010256, 2.004129201441751, 2.005877879415064, 2.0042275662519256, 2.00993609896529, 2.0057472312820708, 2.002024109418482, 2.007455638182207, 2.0034418290278615, 2.003507275752618, 2.002785426023437, 2.0023096778056657, 2.009503663777716, 2.0014636157265695, 2.008581658578467, 2.0019936654851893, 2.016133488255219, 2.004586661337274, 2.0032182233114213, 2.006282970695239, 2.004802602414152, 2.0026903318513196, 2.0119237369965006, 2.005409590436346, 2.005228336320067, 1.999643068358947, 2.013529747007022, 2.0089667085970913, 2.0162253059927653, 2.013425437018377, 2.014763911598786, 2.0159418077110427, 2.0138196347313917, 2.018656073354587, 2.0083653236505414, 3.566437487601689, 3.027883371694623, 2.020007475097646, 2.0108347250493463, 2.012309334768383, 2.0247177309224065, 2.0224958155837816, 26.19648207139505, 2.0157419624100363, 15.130922161933059, 2.017010736793534, 4.29229063189338, 8.88383478835963, 22.646409593029144, 2.64473923619005, 7.624541580796911, 2.0203191172713155, 5.248050197772073, 8.471570816388814, 3.6348016709013096, 4.768411036996617, 4.769841817759892, 6.596579431573219, 4.299715170828975, 3.9772768019663896, 3.9078487407156923, 13.617029146090001, 3.8550365111295455, 3.439624957733023, 3.4386416623041693, 3.4397383420555707, 8.748736427450389, 5.421625873001931, 3.0163090326961806, 2.58734791478384, 2.5622420036988376, 2.640669309828748, 2.572597040191658, 2.5748484240838176, 2.57890990722891, 2.5726615644632123, 2.5345409205397216, 4.413364525114444, 2.125361408874042, 2.135169140352627, 2.1227119393669605, 4.099757462007438, 2.1811618260254257, 4.4319802985164065, 3.442437273751006, 5.417256895384312, 4.7685543444519025, 9.29454955490204, 5.842729509530975, 9.183957210938631, 26.19648207139505, 10.204590948079025, 9.94546793621467, 22.646409593029144, 7.5853168208511335, 8.151727665476692, 5.142416771108244, 15.459318889605143, 4.61474896883033, 5.617285899015783, 2.732230642634979, 2.7450077039641854, 2.789424480536498, 2.7395887741029066, 4.9887444520295805, 2.6462431672060385, 2.7404299560888594, 2.331451283290232, 2.3357247664457095, 2.2852179638525163, 2.2079157203213247, 2.225697910963028, 2.2332218215897135, 2.2363376425164345, 1.8111645886321208, 1.8138513573752202, 1.8169141394740793, 1.8204799071480433, 1.8153426907739572, 1.813066539691765, 1.8130042648634137, 1.814874006930757, 1.8172748294772711, 1.8233205844279003, 1.8254738324572979, 1.7685232814424976, 4.343041290303471, 1.7847770074786538, 3.2692042967878865, 2.7152025332325516, 2.7247470953459585, 2.723498075703033, 5.300488942094431, 16.4149704815227, 3.649287423541875, 8.88383478835963, 3.2876919104705524, 7.238422731359822, 3.25195288652763, 9.710991756329848, 3.843567415825716, 16.963063372766413, 6.389189451360129, 3.6755543827096053, 9.678759227794117, 26.19648207139505, 1.7378334795226156, 1.5401467671699427, 1.5445798241545166, 1.5447025283076412, 1.5026633886824208, 1.512069229053147, 1.7170428920623126, 1.2825189715752723, 1.299026918409571, 1.2676900337298722, 1.9724695744001242, 3.25195288652763, 4.897349099956955, 1.949338699960266, 1.175763116219621, 1.2391856609300196, 3.4232056366274684, 5.048323931413676, 2.760092557016664, 1.5459385705800088, 1.4868002136429157, 1.4881532211057726, 1.5038609555809286, 1.5046541816907733, 1.505112778120852, 1.6876248682079384, 1.681128184545899, 4.433982813734191, 4.4319802985164065, 1.6083869571975786, 5.309908978711119, 3.9526550889634566, 2.5980046502719536, 5.152803866800269, 7.593603466569354, 7.5499169242356645, 26.19648207139505, 1.8151749030689668, 13.575331033438616, 2.6692743319763577, 2.5014148856936265, 2.9765370974631633, 1.6592029951122105, 1.4226681190892214, 1.2401725241184545, 1.2374937450540098, 1.2412857772847896, 1.2477281194177345, 1.245672689750819, 1.250037125852934, 3.4336327842461394, 1.227547242324942, 1.2140885274527609, 2.549211915343648, 3.1238574129984285, 4.7685543444519025, 1.6861954267157715, 1.8254322835906918, 2.57081089436261, 2.7697794636198876, 1.2391856609300196, 1.4985390653181507, 2.802889853980536, 1.4728177465668142, 4.099757462007438, 1.531290350138376, 1.6524491779351658, 1.5762769663972893, 1.5818077881167463, 8.791409689403157, 8.748736427450389, 5.048323931413676, 5.421625873001931, 3.904291500112131, 5.300488942094431, 6.5499027695043015, 8.219018348036832, 16.963063372766413, 17.402004337137296, 1.3262969877718085, 1.0504455388488005, 1.0569905772444423, 1.175763116219621, 2.8436650284023197, 1.4305720193598703, 1.3807982430684018, 1.4958261353001119, 1.403932272145097, 1.4240344062845818, 1.5151032644619795, 1.5068783474952547, 2.864920340082658, 1.8522767003301739, 2.0137317818772864, 2.379721384231504, 2.37135238695189, 2.4786872873714456, 2.037261672858694, 3.503117577073849, 3.6062039170319165, 3.812122200804151, 3.623636643152755, 4.450234936360434, 15.459318889605143, 5.813327779658336, 5.987884051615016, 1.227547242324942, 1.2412857772847896, 1.2140885274527609, 22.646409593029144, 4.139137394440636, 1.8015464624489559, 9.678759227794117], "loglift": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.3059, 1.3057, 1.3055, 1.3052, 1.305, 1.304, 1.2774, 1.2773, 1.2772, 1.277, 1.2765, 1.2764, 1.2761, 1.2749, 1.2748, 1.2256, 1.2256, 1.225, 1.2249, 1.2249, 1.2246, 1.2243, 1.2243, 1.2242, 1.2241, 1.2241, 1.224, 1.224, 1.2239, 1.2238, 1.2236, 1.0362, 1.223, 1.2231, 1.2228, 0.9918, 0.4906, 0.8144, 0.567, -0.112, 0.6968, -0.3895, 0.9595, 0.9586, 0.9585, 1.4915, 1.4914, 1.4912, 1.4912, 1.4909, 1.4907, 1.4906, 1.4905, 1.4904, 1.4903, 1.4902, 1.4901, 1.4901, 1.4898, 1.4897, 1.4895, 1.4894, 1.4893, 1.4893, 1.4538, 1.3927, 1.3911, 1.3906, 1.3904, 1.3895, 1.3893, 1.3891, 1.3891, 1.3891, 1.3889, 1.3888, 1.2999, 1.1896, 1.1979, 1.3882, 1.1347, 1.3883, 0.0223, -0.4706, 1.7098, 1.6866, 1.6861, 1.6852, 1.683, 1.6802, 1.6656, 1.6498, 1.6475, 1.6433, 1.6366, 1.6353, 1.6261, 1.6242, 1.6154, 1.6141, 1.611, 1.6082, 1.6079, 1.6023, 1.602, 1.597, 1.5892, 1.5876, 1.5874, 1.5872, 1.5869, 1.5868, 1.5868, 1.5855, 1.5455, 1.5166, 1.4889, 1.4576, 1.5078, 1.4111, 1.5072, 1.3134, 1.3555, 1.5309, 1.5308, 0.8516, 1.4284, 1.077, 0.8947, 0.9892, 1.1852, 0.4683, 1.1019, 1.1486, 0.6034, 2.0849, 2.032, 2.0266, 2.021, 2.0035, 1.9855, 1.9548, 1.9466, 1.9357, 1.9265, 1.916, 1.9144, 1.9119, 1.9112, 1.9092, 1.9088, 1.9067, 1.906, 1.901, 1.9005, 1.9002, 1.8961, 1.8923, 1.857, 1.8568, 1.856, 1.8507, 1.8483, 1.8481, 1.8392, 1.7985, 1.7605, 1.8233, 1.7185, 1.6945, 1.6526, 1.7047, 1.6039, 1.6861, 1.6025, 1.2875, 1.4794, 1.0879, 1.6058, 1.5252, 1.7537, 1.4401, 0.7168, 1.4048, 0.673, 0.5564, 1.0854, 0.9479, 2.0507, 1.9841, 1.9549, 1.9522, 1.952, 1.9514, 1.9513, 1.9512, 1.9512, 1.9512, 1.951, 1.9509, 1.9508, 1.9506, 1.9504, 1.95, 1.95, 1.9498, 1.9498, 1.9498, 1.9496, 1.9496, 1.9496, 1.9495, 1.9495, 1.9493, 1.9493, 1.9493, 1.9493, 1.9493, 1.9492, 1.9489, 1.949, 1.9487, 1.9486, 1.9487, 1.948, 1.949, 1.7666, 1.8115, 1.9464, 1.9484, 1.948, 1.9444, 1.9449, 0.2663, 1.9468, 0.5818, 1.9463, 1.3424, 0.7529, -0.0745, 1.714, 0.748, 1.9447, 0.9912, 0.5119, 1.3576, 2.4585, 2.4578, 2.4502, 2.4491, 2.4358, 2.431, 2.4298, 2.418, 2.3974, 2.3808, 2.3807, 2.3517, 2.3411, 2.3308, 2.3188, 2.3116, 2.31, 2.3089, 2.3077, 2.307, 2.307, 2.3027, 2.2631, 2.235, 2.2296, 2.2292, 2.2053, 2.1625, 2.141, 2.1286, 2.0776, 2.0728, 1.923, 1.9884, 1.7632, 0.9606, 1.4994, 1.4918, 0.9256, 1.6202, 1.5699, 1.8386, 0.8942, 2.6173, 2.5257, 2.5025, 2.4988, 2.4987, 2.4954, 2.4751, 2.4539, 2.4464, 2.4408, 2.4394, 2.4371, 2.3577, 2.3522, 2.3314, 2.3201, 2.3136, 2.3136, 2.3128, 2.3117, 2.3116, 2.3112, 2.3099, 2.3094, 2.3086, 2.3076, 2.3062, 2.3038, 2.2957, 2.2952, 2.2638, 2.2405, 2.2327, 2.2299, 2.0394, 1.7242, 2.1099, 1.7235, 2.0742, 1.7581, 2.0588, 1.547, 1.9669, 1.1603, 1.5033, 1.9142, 0.9711, -0.0335, 3.3623, 3.2574, 3.2545, 3.254, 3.2305, 3.229, 3.0998, 3.0503, 3.0373, 3.0244, 2.96, 2.7583, 2.662, 2.6227, 2.5508, 2.4334, 2.4287, 2.3351, 2.2792, 2.2778, 2.2531, 2.2502, 2.2438, 2.2414, 2.2396, 2.1901, 2.1887, 2.1807, 2.1725, 2.1677, 2.0195, 1.9227, 1.9725, 1.6241, 1.2701, 1.2427, 0.042, 2.1167, 0.1308, 1.7347, 1.7988, 3.7186, 3.4076, 3.2637, 3.0947, 3.0946, 3.0892, 3.0888, 3.0866, 3.0851, 3.0792, 3.07, 3.0467, 2.976, 2.7885, 2.7496, 2.7166, 2.7105, 2.701, 2.6263, 2.41, 2.2857, 2.2803, 2.2365, 2.2347, 2.2036, 2.1833, 2.1775, 2.1658, 2.1513, 2.1432, 2.0052, 1.9537, 1.9498, 1.5732, 1.3958, 1.1698, 0.4586, 0.3858, 3.8166, 3.4079, 3.4022, 2.6497, 2.6376, 2.5134, 2.4875, 2.4705, 2.4702, 2.4615, 2.4605, 2.4587, 2.368, 2.1925, 2.1744, 2.0109, 2.0083, 1.9062, 1.7149, 1.6184, 1.5872, 1.5495, 1.5186, 1.3764, 1.092, 1.0449, 1.0198, 0.7215, 0.6938, 0.6908, -0.229, -0.0949, 0.4347, -0.9761], "logprob": [30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, -5.3512, -5.352, -5.3525, -5.3523, -5.3526, -5.3523, -5.6335, -5.6333, -5.6343, -5.634, -5.634, -5.6343, -5.6329, -5.6345, -5.6335, -6.0239, -6.0234, -6.0245, -6.0241, -6.0245, -6.0242, -6.0247, -6.0239, -6.0258, -6.0239, -6.0267, -6.0238, -6.0249, -6.0232, -6.0258, -6.0248, -5.4653, -6.0235, -6.0238, -6.0235, -5.8009, -5.4308, -5.8023, -5.8013, -5.6302, -5.9063, -5.7621, -6.0215, -6.0232, -6.0237, -5.3954, -5.3951, -5.3962, -5.3965, -5.3955, -5.3955, -5.3958, -5.3956, -5.3956, -5.3956, -5.396, -5.3953, -5.3962, -5.3952, -5.3957, -5.396, -5.3962, -5.3961, -5.3962, -5.6676, -6.037, -6.0392, -6.0374, -6.0401, -6.0404, -6.0404, -6.0385, -6.0384, -6.0401, -6.0405, -6.0399, -5.6687, -5.3951, -5.663, -6.0382, -5.6672, -6.0386, -5.8471, -5.9888, -4.3201, -5.0671, -5.067, -5.0689, -5.0696, -5.0719, -5.2423, -5.35, -5.3619, -5.3582, -5.4609, -4.7473, -5.4722, -5.4644, -5.5927, -5.5999, -5.606, -5.6055, -5.3912, -5.3967, -5.6008, -5.6176, -5.7467, -4.5162, -5.7458, -5.7459, -5.7463, -5.7504, -5.7466, -5.7483, -4.2979, -4.3833, -4.3166, -4.3633, -4.9467, -4.4583, -5.256, -4.4682, -4.9515, -5.4595, -5.4678, -4.521, -5.344, -5.0043, -4.9124, -5.1422, -5.2656, -5.0498, -5.3188, -5.3403, -5.318, -4.4124, -4.8616, -4.9657, -4.9638, -5.0979, -5.2379, -5.4021, -5.4358, -5.1441, -4.9649, -5.6051, -5.6184, -5.6197, -5.6177, -5.6369, -5.6365, -5.6397, -5.6375, -5.6288, -5.6281, -5.6305, -5.6378, -5.6502, -5.8598, -4.9398, -4.9924, -5.8622, -5.8842, -5.862, -5.9092, -4.7332, -4.4466, -5.4912, -4.7459, -4.845, -4.7123, -4.9592, -4.7371, -5.1919, -5.1336, -4.7939, -5.1315, -5.0435, -5.4168, -5.4179, -5.5475, -5.4135, -5.1232, -5.4105, -5.1478, -5.2507, -5.3655, -5.3591, -5.6267, -5.7138, -5.9888, -5.9923, -5.991, -5.9907, -5.9916, -5.9888, -5.9909, -5.9928, -5.9903, -5.9924, -5.9924, -5.9931, -5.9935, -5.9902, -5.9943, -5.9909, -5.9942, -5.9872, -5.9931, -5.9939, -5.9923, -5.9931, -5.9942, -5.9898, -5.993, -5.9931, -5.9959, -5.989, -5.9914, -5.988, -5.9894, -5.989, -5.9885, -5.9894, -5.9877, -5.9918, -5.6, -5.7188, -5.9887, -5.9912, -5.9909, -5.9883, -5.989, -5.1063, -5.9904, -5.3397, -5.9903, -5.8389, -5.7011, -5.5927, -5.9516, -5.8588, -5.9903, -5.9891, -5.9896, -5.99, -4.6177, -4.618, -4.3014, -4.7305, -4.8218, -4.8442, -3.597, -4.8708, -5.0054, -5.0223, -5.0221, -4.1176, -4.6067, -5.2033, -5.3687, -5.3856, -5.3572, -5.3843, -5.3847, -5.3838, -5.3863, -5.4055, -4.8904, -5.6493, -5.65, -5.6562, -5.022, -5.6958, -5.0083, -5.2734, -4.8709, -5.0033, -4.4858, -4.8846, -4.6575, -4.412, -4.816, -4.8493, -4.5925, -4.9918, -4.9701, -5.1621, -5.0058, -4.4916, -4.3866, -5.1306, -5.1296, -5.1137, -5.1349, -4.5559, -5.2112, -5.1837, -5.3509, -5.3505, -5.3746, -5.4884, -5.4859, -5.5034, -5.5132, -5.7306, -5.7291, -5.7283, -5.7274, -5.7303, -5.732, -5.7333, -5.7327, -5.7322, -5.7299, -5.7302, -5.7642, -4.8739, -5.7637, -5.1898, -5.3988, -5.403, -5.4064, -4.931, -4.1158, -5.2337, -4.7304, -5.3738, -4.9007, -5.4002, -4.8179, -5.3248, -4.6469, -5.2803, -5.4223, -5.3972, -5.406, -4.7232, -4.9489, -4.9489, -4.9494, -5.0004, -4.9957, -4.9977, -5.3391, -5.3393, -5.3766, -4.9988, -4.7006, -4.3875, -5.3479, -5.9255, -5.9903, -4.9789, -4.684, -5.3437, -5.9248, -5.9884, -5.9905, -5.9863, -5.9883, -5.9897, -5.9248, -5.9301, -4.9682, -4.9768, -5.9953, -4.9491, -5.3411, -5.7109, -5.3746, -5.3408, -5.374, -5.3306, -5.9253, -5.8991, -5.9217, -5.9225, -3.8288, -4.7243, -5.0219, -5.3283, -5.3304, -5.3328, -5.3281, -5.3319, -5.3299, -4.3253, -5.3632, -5.3974, -4.7264, -4.7106, -4.3265, -5.3991, -5.3259, -4.993, -4.9932, -6.0137, -5.948, -5.3272, -6.0145, -4.9925, -6.0085, -5.9527, -6.0056, -6.0138, -4.3131, -4.3261, -5.0139, -4.994, -5.3263, -5.3971, -5.3629, -5.362, -5.3485, -5.3958, -4.5392, -5.1811, -5.1805, -5.8266, -4.9555, -5.7667, -5.828, -5.765, -5.8287, -5.8232, -5.7622, -5.7694, -5.2177, -5.8292, -5.7638, -5.7603, -5.7664, -5.8242, -6.2116, -5.7661, -5.7683, -5.7505, -5.8321, -5.7688, -4.8079, -5.8331, -5.8286, -7.7117, -7.7282, -7.7534, -5.7472, -7.3126, -7.6149, -7.3444]}, "token.table": {"Topic": [3, 4, 7, 3, 2, 3, 1, 3, 5, 7, 8, 1, 5, 1, 1, 1, 5, 5, 1, 1, 7, 3, 4, 9, 9, 4, 9, 4, 3, 4, 6, 9, 4, 4, 5, 8, 8, 3, 4, 3, 3, 3, 4, 6, 7, 9, 9, 6, 9, 6, 9, 6, 9, 6, 6, 3, 4, 5, 6, 1, 3, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 3, 2, 2, 2, 2, 7, 4, 6, 7, 6, 3, 4, 7, 4, 3, 7, 2, 8, 2, 2, 2, 2, 8, 3, 7, 3, 4, 6, 7, 8, 8, 4, 6, 7, 7, 3, 4, 5, 3, 4, 7, 4, 4, 1, 3, 4, 6, 2, 3, 6, 2, 2, 3, 6, 8, 6, 4, 4, 7, 4, 4, 4, 4, 7, 7, 3, 8, 8, 8, 8, 4, 3, 4, 7, 3, 3, 6, 6, 6, 1, 6, 7, 6, 3, 8, 8, 6, 7, 4, 3, 6, 3, 6, 3, 3, 3, 7, 6, 9, 6, 9, 6, 9, 4, 5, 3, 6, 6, 3, 4, 6, 2, 8, 3, 3, 4, 7, 8, 4, 4, 1, 4, 4, 4, 4, 5, 7, 7, 3, 6, 6, 3, 3, 6, 3, 9, 3, 5, 5, 4, 7, 6, 1, 3, 4, 6, 7, 3, 1, 1, 3, 8, 1, 1, 5, 6, 1, 5, 5, 1, 5, 1, 1, 5, 5, 1, 1, 1, 5, 5, 1, 1, 5, 5, 5, 1, 1, 1, 1, 3, 4, 3, 4, 6, 2, 3, 4, 7, 3, 3, 6, 9, 3, 3, 4, 7, 3, 4, 6, 3, 3, 4, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 7, 2, 6, 2, 2, 2, 2, 2, 2, 2, 6, 1, 3, 4, 5, 6, 7, 8, 1, 5, 1, 1, 1, 1, 3, 1, 5, 1, 1, 5, 6, 5, 3, 4, 3, 6, 3, 4, 3, 3, 4, 7, 3, 3, 3, 3, 4, 5, 4, 6, 1, 3, 6, 3, 6, 7, 6, 7, 3, 1, 2, 3, 4, 5, 6, 2, 2, 5, 5, 5, 5, 2, 2, 2, 2, 1, 5, 5, 2, 2, 2, 5, 6, 5, 5, 5, 5, 1, 3, 4, 6, 4, 7, 4, 6, 8, 2, 3, 4, 5, 7, 4, 6, 7, 1, 3, 4, 6, 7, 7, 7, 7, 4, 3, 8, 9, 1, 4, 7, 3, 5, 5, 5, 5, 5, 5, 5, 6, 3, 4, 5, 3, 4, 2, 5, 5, 4, 4, 6, 7, 7, 4, 6, 4, 4, 6, 1, 3, 4, 6, 3, 2, 3, 3, 4, 7, 2, 3, 4, 4, 6, 6, 6, 1, 3, 4, 6, 7, 3, 4, 7, 4, 7, 6, 3, 7, 3, 8, 1, 3, 7, 1, 1, 1, 1, 1, 1, 4, 6, 8, 1, 3, 4, 3, 4, 4, 4, 4, 7, 3, 6, 6, 3, 3, 4, 4, 2, 8, 3, 6, 3, 3, 4, 7, 8, 2, 3, 6, 3, 7, 7, 7, 7, 10, 1, 2, 3, 7, 3, 3, 4, 9, 3, 7, 7, 7, 7, 7, 7, 7, 3, 4, 6, 2, 3, 6, 2, 2, 2, 2, 2, 2, 4, 1, 3, 7, 3, 4, 6, 6, 7, 7, 4, 6, 9, 4, 9, 3, 1, 3, 4, 7, 7, 7, 1, 3, 4, 6, 6, 3, 4, 8, 9, 7, 7, 7, 2, 3, 6, 3, 6, 3, 3, 3, 6, 3], "Freq": [0.1388567521389371, 0.6942837606946856, 0.1388567521389371, 0.7243373834129679, 0.6414793966676898, 0.2565917586670759, 0.5298071541899715, 0.13245178854749287, 0.13245178854749287, 0.13245178854749287, 0.13245178854749287, 0.5727079012855493, 0.19090263376184974, 0.8893892903619722, 0.9659056852606132, 0.7482919230972541, 0.49599890375582695, 0.4986512504821599, 0.8871640223134232, 0.9660410831087479, 0.45291583858756485, 0.11374740062510832, 0.68248440375065, 0.11374740062510832, 0.6026990084672376, 0.5478154456833444, 0.5478154456833444, 0.833073651582636, 0.35677463335915455, 0.35677463335915455, 0.35677463335915455, 0.35677463335915455, 0.6710791461249352, 0.3781091104620873, 0.3781091104620873, 0.6474252637265849, 0.6473738352041081, 0.2759658592948622, 0.5519317185897245, 0.7022302239235069, 0.88395634481706, 0.3997737463382474, 0.9404027018625918, 0.6797535039148407, 0.22658450130494687, 0.8014566510424246, 0.8063394250012311, 0.8001155433185205, 0.11430222047407436, 0.7317506042250256, 0.2439168680750085, 0.7377860615426821, 0.18444651538567053, 0.8386022331194537, 0.8724375188282213, 0.49085496150172786, 0.49085496150172786, 0.46595167278264576, 0.46595167278264576, 0.39653895088397356, 0.33044912573664464, 0.19826947544198678, 0.13217965029465784, 0.4991410209725102, 0.49389600571356046, 0.4994944842549699, 0.4970367323628526, 0.4989453377642561, 0.4977683282259682, 0.4993283205574692, 0.49934931377910935, 0.8889617303394567, 0.9655075065479406, 0.7481437892918419, 0.7140474772592394, 0.8959146317603702, 0.8950412682480571, 0.89655832994372, 0.8960777663967412, 0.6673166039803525, 0.6032898641352065, 0.30164493206760323, 0.10054831068920107, 0.7774245125661033, 0.8576407362384693, 0.5550786620516225, 0.5550786620516225, 0.6629296213013659, 0.3041647536422823, 0.6083295072845646, 0.7533085813781548, 0.1883271453445387, 0.8664741323313994, 0.8672391606289693, 0.8672151148224062, 0.8670969295204826, 0.7698069884682015, 0.7247886972849377, 0.24159623242831255, 0.3516588592580605, 0.3516588592580605, 0.36230669057014614, 0.36230669057014614, 0.36230669057014614, 0.7797155614561675, 0.23025339460452715, 0.23025339460452715, 0.6907601838135815, 0.8562652709477476, 0.5246217044804841, 0.13115542612012102, 0.26231085224024203, 0.3357181609258264, 0.5035772413887396, 0.1678590804629132, 0.6613532429271116, 0.6321880619835426, 0.15267402207188494, 0.15267402207188494, 0.30534804414376987, 0.30534804414376987, 0.6376176379700713, 0.15940440949251783, 0.15940440949251783, 0.8678181417412988, 0.8668877667961157, 0.40838420116253893, 0.20419210058126946, 0.20419210058126946, 0.7774050141792752, 0.771360421422526, 0.6712570365531344, 0.8751900394780356, 0.621740928403436, 0.7959217301716404, 0.7905715815724824, 0.7928677391869742, 0.6644020398581093, 0.664604539812799, 0.5069786692674965, 0.5069786692674965, 0.6613453807443703, 0.5754291258531289, 0.6654850364570533, 0.6636235776180659, 0.2985283548813316, 0.5970567097626632, 0.1492641774406658, 0.8168270208271482, 0.8434005890498535, 0.7542849415250108, 0.7676858033790154, 0.7890975378586932, 0.2601749603460954, 0.2601749603460954, 0.5203499206921908, 0.6725853217022537, 0.5842476942081674, 0.2921238471040837, 0.7888363664560344, 0.2740260998760777, 0.5480521997521554, 0.7879412451261041, 0.7762308343395911, 0.19405770858489776, 0.9104579357521738, 0.8721881126183101, 0.7785285584219468, 0.7772109795545377, 0.909990083872382, 0.560294084812699, 0.6291214869954093, 0.2097071623318031, 0.29123673462931243, 0.29123673462931243, 0.36103957485953675, 0.36103957485953675, 0.6593912775805354, 0.6600210186697109, 0.8493707119998047, 0.8388538590665203, 0.8721593626238485, 0.24525295530858543, 0.6131323882714635, 0.12262647765429271, 0.5129944837294736, 0.5129944837294736, 0.7766194489626987, 0.26337956792252176, 0.5267591358450435, 0.13168978396126088, 0.13168978396126088, 0.6544893220274892, 0.7984749655917641, 0.16699763568310352, 0.6679905427324141, 0.6686369284674933, 0.728958749761486, 0.66852689386883, 0.20045123770434223, 0.8018049508173689, 0.8578347805653157, 0.5444276236440547, 0.4355420989152438, 0.4683469712543935, 0.722084068135672, 0.7698634534851645, 0.19246586337129112, 0.6402340874067949, 0.32011704370339744, 0.3302637113926642, 0.6605274227853284, 0.7810924369247086, 0.37732368123943705, 0.37732368123943705, 0.7782027462876051, 0.08152539165596982, 0.4076269582798491, 0.3261015666238793, 0.08152539165596982, 0.08152539165596982, 0.8568965624535657, 0.7458936649587158, 0.7474398994418853, 0.7762763930861345, 0.19406909827153362, 0.550911098599519, 0.708251176793872, 0.11804186279897866, 0.11804186279897866, 0.6318339011497295, 0.15795847528743237, 0.49919673671246323, 0.5724507674811393, 0.1908169224937131, 0.8888380268869829, 0.7476582996516797, 0.49666602080922895, 0.49763532061448207, 0.8885302826834189, 0.9658952900174759, 0.5719847692272367, 0.19066158974241226, 0.498143012966165, 0.888480654305003, 0.9663033896306513, 0.4995020799716901, 0.4949713099535585, 0.4966402912529271, 0.7491520532512477, 0.7478148239889758, 0.74783607132891, 0.38491078139345214, 0.38491078139345214, 0.8419419290246539, 0.4662839978486271, 0.23314199892431356, 0.11657099946215678, 0.1765677410061647, 0.7062709640246588, 0.05885591366872157, 0.05885591366872157, 0.6468562328610623, 0.3922780973919912, 0.3922780973919912, 0.3922780973919912, 0.593051068788459, 0.31302875196074215, 0.46954312794111325, 0.31302875196074215, 0.26366729923557547, 0.26366729923557547, 0.39550094885336323, 0.5398761425988606, 0.39753662691369346, 0.5963049403705402, 0.4959763162518248, 0.34904984477549145, 0.34904984477549145, 0.49912471599303265, 0.4969414904170785, 0.49752825501009584, 0.49656880028055866, 0.4957831813972948, 0.4985348361743792, 0.4973059135804708, 0.3682966510824009, 0.7365933021648018, 0.6764105376235359, 0.16910263440588397, 0.8776134447401515, 0.8673715723759549, 0.8671282408668385, 0.8660548033929767, 0.867366624169352, 0.8666453842120788, 0.8664706712339523, 0.9302941802139907, 0.15269225803291178, 0.38173064508227944, 0.038173064508227945, 0.11451919352468383, 0.19086532254113972, 0.07634612901645589, 0.038173064508227945, 0.5502363487975581, 0.27511817439877906, 0.7469958943618366, 0.7482006887503798, 0.7470552555323614, 0.7460057028864204, 0.6649550919511097, 0.5716408736474309, 0.19054695788247697, 0.74729629027479, 0.747845838618337, 0.49537908571924677, 0.7805663934604193, 0.498567309182225, 0.6741217133254341, 0.22470723777514473, 0.34230576595022794, 0.5134586489253419, 0.8385023741227806, 0.10481279676534758, 0.7766027797875803, 0.7171959128731886, 0.23906530429106287, 0.05976632607276572, 0.8134906681135546, 0.8136811876142512, 0.8135186180110271, 0.46528461427746903, 0.3877371785645575, 0.0775474357129115, 0.8085393632369205, 0.7767447517659762, 0.12267338177096997, 0.49069352708387987, 0.36802014531290994, 0.6789706345751416, 0.3670065385914398, 0.7340130771828796, 0.36717485094674207, 0.7343497018934841, 0.7846305116012999, 0.22078555010941178, 0.13247133006564707, 0.26494266013129414, 0.08831422004376471, 0.08831422004376471, 0.17662844008752943, 0.895728285136438, 0.8939746831781661, 0.49443860021601865, 0.49896982653643773, 0.5000892488381303, 0.4997303329934323, 0.8940862952490849, 0.8954695307521962, 0.8944353313770655, 0.8966175153575758, 0.7492673106099139, 0.4963360690764333, 0.4986963239484083, 0.8934020551635419, 0.892949581609752, 0.8959739924044202, 0.5607836971635619, 0.28039184858178096, 0.49609524366124347, 0.498855958331526, 0.4984341763382805, 0.49930461197009823, 0.964695701634668, 0.21517987377292314, 0.21517987377292314, 0.5379496844323078, 0.821534389503632, 0.4492972721384791, 0.634406275875224, 0.6768983158621535, 0.22563277195405118, 0.11256400235068381, 0.11256400235068381, 0.11256400235068381, 0.22512800470136762, 0.33769200705205143, 0.28546001611378946, 0.28546001611378946, 0.28546001611378946, 0.06091999989434261, 0.18275999968302784, 0.24367999957737044, 0.12183999978868522, 0.36551999936605567, 0.565443503341584, 0.7169937791667098, 0.44778355214537197, 0.669259107828139, 0.676592608953636, 0.22553086965121202, 0.7029046244743226, 0.5010117053270025, 0.1670039017756675, 0.1670039017756675, 0.8134593694634237, 0.49786375163244784, 0.4996343636439181, 0.4994232466058405, 0.49604606451186617, 0.49791737998260793, 0.498802225613542, 0.4950476730051014, 0.7729922939903852, 0.5610989814046558, 0.3366593888427935, 0.11221979628093116, 0.3671447466009288, 0.5507171199013932, 0.6051623331917293, 0.5948386382387116, 0.7652273356869841, 0.6595865054505676, 0.617856564041367, 0.1029760940068945, 0.3089282820206835, 0.7285954050736225, 0.765653133909209, 0.47050821372058904, 0.8735783128757499, 0.6704731199730577, 0.7573837407644592, 0.7477928237411525, 0.7946558113805734, 0.061127370106197954, 0.12225474021239591, 0.9109239221735832, 0.8682185634033928, 0.8140680307224559, 0.12166903122181943, 0.7300141873309166, 0.12166903122181943, 0.17239393473760695, 0.6321110940378921, 0.17239393473760695, 0.07343745755931942, 0.8078120331525136, 0.9095623060767206, 0.47109548001045404, 0.05895161610994533, 0.41266131276961726, 0.17685484832983597, 0.11790323221989066, 0.17685484832983597, 0.7178899860001864, 0.6973081384179731, 0.17432703460449328, 0.8545454575749679, 0.8667860434050552, 0.4584712551210508, 0.5441356028925376, 0.5441356028925376, 0.7242187662245324, 0.649288769951142, 0.413276113792908, 0.309957085344681, 0.206638056896454, 0.7486307045068292, 0.8875230063566913, 0.746372374473112, 0.8865263025585873, 0.888695927357531, 0.7482318095930683, 0.5059890010601658, 0.2529945005300829, 0.2529945005300829, 0.7462593795976238, 0.7931207937791811, 0.15862415875583621, 0.8157512205092678, 0.6621889732764303, 0.6608040572303283, 0.6601957842437248, 0.860288541239848, 0.712285072321957, 0.36919054027215664, 0.5537858104082349, 0.7755214691268683, 0.8404345202982115, 0.32311058047407093, 0.4846658707111064, 0.49659046403278073, 0.5925487463703256, 0.5823966335511375, 0.2904918580870359, 0.5809837161740719, 0.721902990946173, 0.6802925963152476, 0.29155396984939186, 0.6150150601153265, 0.30750753005766324, 0.19446109572804807, 0.38892219145609613, 0.38892219145609613, 0.17802191627369585, 0.7120876650947834, 0.729812486378743, 0.7300365729724932, 0.7320026240797843, 0.7539789422880387, 0.524642153280949, 0.2623210766404745, 0.7667958920297192, 0.15335917840594385, 0.9114005946500412, 0.26383020211805386, 0.7035472056481437, 0.6719217448035691, 0.6990210814045309, 0.5493057056403309, 0.5478029770790441, 0.5513130918550434, 0.5515517374061779, 0.5510024366325905, 0.55086017922801, 0.5502745010162521, 0.1585695188127237, 0.7928475940636185, 0.1585695188127237, 0.29398532633635244, 0.3919804351151366, 0.29398532633635244, 0.8684850637855394, 0.8665397434740743, 0.8669203060320327, 0.8673507960687145, 0.8664679158007089, 0.8665327425230619, 0.6530440160545871, 0.8068787096257409, 0.5545998080014561, 0.27729990400072807, 0.17201851295898757, 0.6880740518359503, 0.17201851295898757, 0.30588482982924525, 0.6117696596584905, 0.5484499042793224, 0.5122568332673316, 0.2561284166336658, 0.2561284166336658, 0.3889823254572497, 0.3889823254572497, 0.6719738168203875, 0.2763032879159127, 0.13815164395795634, 0.13815164395795634, 0.414454931873869, 0.44715966900004955, 0.755788441812641, 0.5227077007188462, 0.13067692517971155, 0.13067692517971155, 0.2613538503594231, 0.663062033206944, 0.1980855455366889, 0.3961710910733778, 0.1980855455366889, 0.1980855455366889, 0.5503837403618083, 0.5521309362365838, 0.5515706826400306, 0.1882183091426569, 0.6901338001897419, 0.06273943638088562, 0.6190910168813839, 0.15477275422034598, 0.9117450681748386, 0.8861354901504094, 0.7883904686791461, 0.22525441962261317, 0.7226166373813545], "Term": ["ahli", "ahli", "ahli", "air", "ajar", "ajar", "alam", "alam", "alam", "alam", "alam", "alam bangun", "alam bangun", "alam bangun negara", "alam bangun negara selatan", "alam bangun selatan", "alam kawasan", "alam mahir bangun", "alam negara", "alam negara selatan", "allahyarham nik", "ambil", "ambil", "ambil", "ambil langkah", "ambil putus", "ambil putus", "ambil tindak", "anak", "anak", "anak", "anak", "angkat", "anti", "anti", "arab", "arab saudi", "arah", "arah", "arah capai", "asli", "atur", "awam", "babit", "babit", "bahan kaji", "bahan uji", "bahasa", "bahasa", "bahasa bahasa", "bahasa bahasa", "bahasa ilmu", "bahasa ilmu", "bahasa inggeris", "bahasa inggeris bahasa", "baiah", "baiah", "bandar", "bandar", "bangun", "bangun", "bangun", "bangun", "bangun kawasan bandar", "bangun kongsi alam mahir", "bangun main peran alam", "bangun main peran kongsi", "bangun malaysia kongsi", "bangun malaysia kongsi alam", "bangun malaysia main alam", "bangun malaysia main mahir", "bangun negara", "bangun negara selatan", "bangun selatan", "bank", "bantu negara ancang", "bantu negara bidang ancang", "bantu negara bidang ekonomi", "bantu negara maju bidang", "bawa", "bayar", "bayar", "bayar", "bekerjasama", "beli", "benda", "benda", "berbeza", "berita", "berita", "bidang", "bidang", "bidang ajar", "bidang didik ajar", "bidang didik proses ajar", "bidang proses ajar", "bidang selamat", "bukti", "bukti", "capai", "capai", "cari", "cari", "cari", "cari jalan", "daftar", "daftar", "daftar", "daftar ph", "dakwa", "dakwa", "dakwa", "dana", "dana", "dana", "dana seleweng", "dar ambil", "dasar", "dasar", "dasar", "dasar", "didik", "didik", "didik", "didik ajar", "didik proses ajar", "duduk", "duduk", "duduk", "faham", "gagal", "gagal dana", "gambar", "ganas", "gembira", "gembira lihat", "gembira lihat projek", "generasi", "generasi alam", "giat", "giat", "giat duduk", "giat hidup", "giat hidup duduk", "global", "hadap", "hadap", "hadap", "hak", "hapus", "harga", "hasil", "hasil makan", "hati", "hati", "hati", "hebat", "hidup", "hidup", "hidup duduk", "hilang", "hilang", "hubung", "hutang", "hutang", "hutang hutang", "hutang makan", "hutang mdb", "hutang pendek", "hutang selesai", "ikan", "ilmu", "ilmu", "ilmu bahasa", "ilmu bahasa", "ilmu ilmu", "ilmu ilmu", "imbang", "individu", "indonesia", "inggeris", "inggeris bahasa", "isu", "isu", "isu", "jaga", "jaga", "jakoa", "jalan", "jalan", "jalan", "jalan", "jalan lancar", "janji", "jawat", "jawat", "jawat ahli", "jawat awam", "jejas", "jppm", "jppm", "jppm daftar", "jual", "jual", "jual harga", "jual syarikat", "kait", "kait", "kaji", "kaji", "kawasan", "kawasan", "kawasan bandar", "kelulus", "kelulus", "kembang", "kena", "kena", "kena", "kena", "kena", "kerja", "kerjasama teknikal malaysia", "kerjasama teknikal malaysia kongsi", "khusus", "khusus", "komprehensif", "kongsi", "kongsi", "kongsi", "kongsi alam", "kongsi alam", "kongsi alam bandar", "kongsi alam bangun", "kongsi alam bangun", "kongsi alam bangun negara", "kongsi alam bangun selatan", "kongsi alam kawasan", "kongsi alam kawasan bandar", "kongsi alam negara", "kongsi alam negara selatan", "kongsi bangun", "kongsi bangun", "kongsi bangun kawasan", "kongsi bangun negara", "kongsi bangun negara selatan", "kongsi kawasan", "kongsi mahir bangun", "kongsi mahir kawasan bandar", "kongsi negara", "kongsi negara selatan", "kongsi negara selatan selatan", "kukuh", "kukuh", "kumpul", "laksana", "laksana", "laksana", "laku", "laku", "laku", "laku", "langgar", "langkah", "langkah", "langkah", "lapor", "lembaga", "lembaga", "lembaga", "lihat", "lihat", "lihat", "lonjak", "low", "low", "mac", "main", "main", "main kongsi alam kawasan", "main kongsi bangun", "main kongsi mahir", "main kongsi mahir kawasan", "main peran", "main peran alam mahir", "main peran bangun kawasan", "majlis", "majlis", "maju", "maju", "maju bidang", "maju bidang ajar", "maju bidang didik ajar", "maju bidang proses ajar", "maju didik ajar", "maju didik proses ajar", "maju proses ajar", "makan", "malaysia", "malaysia", "malaysia", "malaysia", "malaysia", "malaysia", "malaysia", "malaysia alam", "malaysia alam", "malaysia alam bangun negara", "malaysia alam negara", "malaysia alam negara selatan", "malaysia bangun negara", "malaysia khusus", "malaysia kongsi", "malaysia kongsi", "malaysia kongsi bangun negara", "malaysia kongsi negara selatan", "malaysia main alam bangun", "malaysia niaga", "malaysia peran kongsi mahir", "malu", "malu", "masuk", "masuk", "masyarakat", "masyarakat", "masyarakat asli", "mdb", "mdb", "mdb", "mdb hutang", "mdb hutang pendek", "mdb selesai", "menteri", "menteri", "menteri", "menteri menteri", "milik sewa", "mudah", "mudah", "mudah", "muka", "murah", "murah", "murah hati", "murah hati", "najib", "negara", "negara", "negara", "negara", "negara", "negara", "negara ancang", "negara ancang ekonomi wang", "negara bangun main", "negara bangun main peran", "negara bangun malaysia alam", "negara bangun peran", "negara bidang ancang", "negara bidang ancang ekonomi", "negara bidang ancang wang", "negara bidang ekonomi", "negara kerjasama", "negara main peran alam", "negara main peran kongsi", "negara maju", "negara maju ancang", "negara maju bidang ekonomi", "negara malaysia", "negara malaysia", "negara malaysia kongsi alam", "negara malaysia main", "negara malaysia main peran", "negara malaysia peran", "negara selatan", "niaga", "niaga", "niaga", "niaga jho", "nik", "nila", "nilai", "nilai", "nyata", "nyata", "nyata", "nyata", "nyata", "pandang", "pandang", "pandang", "parti", "parti", "parti", "parti", "parti", "parti daftar", "parti jppm", "parti lembaga", "parti putus", "pekan", "pekan", "pelabur", "pelbagai", "pelbagai", "pelbagai", "pendek", "peran kongsi alam bandar", "peran kongsi bangun bandar", "peran kongsi bangun kawasan", "peran kongsi mahir", "peran mahir", "peran mahir bangun bandar", "peran mahir kawasan bandar", "percaya", "perdana", "perdana", "perdana", "perdana menteri", "perdana menteri", "pergi", "perintah", "pesawat", "ph putus", "pilih", "pilih", "pilih", "pilih parti", "pilih raya", "pimpin negara", "politik", "politik arah", "positif", "program kerjasama teknikal malaysia", "projek", "projek", "projek", "projek projek", "proses ajar", "putrajaya", "putus", "putus", "putus", "raja", "raja", "raja", "rakyat", "rakyat", "rakyat malaysia", "rakyat pimpin", "rana", "rana", "rana", "rana", "rana", "rana malaysia", "raya", "raya", "rendah", "ros", "rumah sewa", "salah", "salah", "sasar", "saudi", "sedia", "sedia", "sedia", "sedia alam", "sedia alam bangun negara", "sedia bangun", "sedia kongsi alam negara", "sedia kongsi bangun negara", "sedia malaysia bangun negara", "selamat", "selamat", "selamat", "selatan", "selesai", "selesai", "selesai tempoh", "seleweng", "semak", "semak imbang", "serah", "serang", "sewa", "sewa", "sewa tanah", "silap", "sokong", "sokong", "strategi", "suci", "sukan", "sumbang", "sumbang", "swasta", "syarikat", "syarikat", "takut", "takut", "tanah", "tanah", "tanah", "tangguh", "tangguh", "tangguh parti", "tangguh pilih", "tangguh pilih parti", "tanggung", "teknikal", "teknikal", "tempoh", "tempoh", "tempoh hutang", "terima", "terima", "terjemah", "teruk", "timbang nyata", "timbang tulis", "timbang tulis nyata", "timbang wajar", "timbang wajar nyata", "timbang wajar tulis", "timbang wajar tulis nyata", "tindak", "tindak", "tindak", "tingkat", "tingkat", "tingkat", "tingkat bidang didik ajar", "tingkat bidang proses ajar", "tingkat didik proses ajar", "tingkat maju bidang ajar", "tingkat maju didik ajar", "tingkat maju proses ajar", "titik", "tn", "tubuh", "tubuh", "tuju", "tuju", "tuju", "tulis", "tulis", "tulis nyata", "tumbuh", "tumbuh", "tumbuh", "tunggu", "tunggu", "turun", "umno", "umno", "umno", "umno", "umno pilih", "umno tangguh", "undi", "undi", "undi", "undi", "untung", "urus", "urus", "urus", "urus", "wajar nyata", "wajar tulis", "wajar tulis nyata", "wang", "wang", "wang", "wang hutang", "wang hutang", "wang hutang hutang", "wang mdb", "wujud", "wujud", "wujud laku"]}, "R": 30, "lambda.step": 0.01, "plot.opts": {"xlab": "PC1", "ylab": "PC2"}, "topic.order": [3, 1, 5, 4, 7, 9, 2, 8, 6, 10]};

    function LDAvis_load_lib(url, callback){
      var s = document.createElement('script');
      s.src = url;
      s.async = true;
      s.onreadystatechange = s.onload = callback;
      s.onerror = function(){console.warn("failed to load library " + url);};
      document.getElementsByTagName("head")[0].appendChild(s);
    }

    if(typeof(LDAvis) !== "undefined"){
       // already loaded: just create the visualization
       !function(LDAvis){
           new LDAvis("#" + "ldavis_el298249993929764917568162", ldavis_el298249993929764917568162_data);
       }(LDAvis);
    }else if(typeof define === "function" && define.amd){
       // require.js is available: use it to load d3/LDAvis
       require.config({paths: {d3: "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min"}});
       require(["d3"], function(d3){
          window.d3 = d3;
          LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
            new LDAvis("#" + "ldavis_el298249993929764917568162", ldavis_el298249993929764917568162_data);
          });
        });
    }else{
        // require.js not available: dynamically load d3 & LDAvis
        LDAvis_load_lib("https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js", function(){
             LDAvis_load_lib("https://cdn.rawgit.com/bmabey/pyLDAvis/files/ldavis.v1.0.0.js", function(){
                     new LDAvis("#" + "ldavis_el298249993929764917568162", ldavis_el298249993929764917568162_data);
                })
             });
    }
    </script>



Train NMF model
---------------

.. code:: python

    nmf = malaya.topic_model.nmf(corpus,10)
    nmf.top_topics(5, top_n = 10, return_df = True)




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
          <th>topic 0</th>
          <th>topic 1</th>
          <th>topic 2</th>
          <th>topic 3</th>
          <th>topic 4</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>negara</td>
          <td>negara</td>
          <td>menteri</td>
          <td>mdb</td>
          <td>projek</td>
        </tr>
        <tr>
          <th>1</th>
          <td>bangun</td>
          <td>wang</td>
          <td>perdana</td>
          <td>niaga</td>
          <td>jual</td>
        </tr>
        <tr>
          <th>2</th>
          <td>sedia</td>
          <td>ancang</td>
          <td>perdana menteri</td>
          <td>doj</td>
          <td>syarikat</td>
        </tr>
        <tr>
          <th>3</th>
          <td>kongsi</td>
          <td>maju</td>
          <td>seri</td>
          <td>urus</td>
          <td>sewa</td>
        </tr>
        <tr>
          <th>4</th>
          <td>alam</td>
          <td>bidang</td>
          <td>najib</td>
          <td>low</td>
          <td>jual syarikat</td>
        </tr>
        <tr>
          <th>5</th>
          <td>malaysia</td>
          <td>ekonomi</td>
          <td>najib razak</td>
          <td>jho</td>
          <td>swasta</td>
        </tr>
        <tr>
          <th>6</th>
          <td>kongsi alam</td>
          <td>industri</td>
          <td>razak</td>
          <td>urus niaga</td>
          <td>indonesia</td>
        </tr>
        <tr>
          <th>7</th>
          <td>selatan</td>
          <td>latih</td>
          <td>menteri seri</td>
          <td>jho low</td>
          <td>tanah</td>
        </tr>
        <tr>
          <th>8</th>
          <td>kongsi alam bangun</td>
          <td>dagang</td>
          <td>menteri seri najib</td>
          <td>tuntut</td>
          <td>rana</td>
        </tr>
        <tr>
          <th>9</th>
          <td>alam bangun</td>
          <td>didik</td>
          <td>seri najib razak</td>
          <td>sivil</td>
          <td>kena</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    nmf.get_sentences(5)




.. parsed-literal::

    ['sedia kongsi alam bangun ekonomi sosial negara bangun rangka program kerjasama teknikal malaysia mtcp sedia malaysia kongsi alam bangun negara negara selatan selatan',
     'sedia kongsi alam bangun ekonomi sosial negara bangun rangka program kerjasama teknikal malaysia mtcp sedia malaysia kongsi alam bangun negara negara selatan selatan',
     'kali kongsi maklumat kena pelbagai khidmat biaya program sedia usahawan wanita iks sabah kongsi idea alam aspirasi promosi produk peringkat luas',
     'mou memorandum persefahaman arab saudi bidang selamat kongsi alam pakar malaysia deradikalisasi ganas khusus daesh',
     'terusi bentang bajet raja sedia promosi tingkat mudah lancong negara']



.. code:: python

    nmf.get_topics(10)




.. parsed-literal::

    [(0,
      'negara bangun sedia kongsi alam malaysia kongsi alam selatan kongsi alam bangun alam bangun'),
     (1, 'negara wang ancang maju bidang ekonomi industri latih dagang didik'),
     (2,
      'menteri perdana perdana menteri seri najib najib razak razak menteri seri menteri seri najib seri najib razak'),
     (3, 'mdb niaga doj urus low jho urus niaga jho low tuntut sivil'),
     (4,
      'projek jual syarikat sewa jual syarikat swasta indonesia tanah rana kena'),
     (5,
      'rakyat malaysia negara rakyat malaysia pimpin pimpin negara maklumat kait asas pandang'),
     (6,
      'parti umno tangguh pilih lembaga putus jalan tangguh pilih pilih parti tangguh pilih parti'),
     (7, 'ajar raja tingkat laku ajar ajar proses didik bidang maju didik proses'),
     (8,
      'bangun malaysia kawasan alam bangun kawasan main bandar kongsi kongsi alam peran'),
     (9,
      'asli masyarakat jakoa bangun perdana bangun arus perdana pelopor arus arus perdana ganti')]



Train LSA model
---------------

.. code:: python

    lsa = malaya.topic_model.lsa(corpus,10)
    lsa.top_topics(5, top_n = 10, return_df = True)




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
          <th>topic 0</th>
          <th>topic 1</th>
          <th>topic 2</th>
          <th>topic 3</th>
          <th>topic 4</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>negara</td>
          <td>negara</td>
          <td>negara</td>
          <td>mdb</td>
          <td>projek</td>
        </tr>
        <tr>
          <th>1</th>
          <td>bangun</td>
          <td>wang</td>
          <td>maju</td>
          <td>niaga</td>
          <td>jual</td>
        </tr>
        <tr>
          <th>2</th>
          <td>malaysia</td>
          <td>menteri</td>
          <td>maju bidang</td>
          <td>doj</td>
          <td>malaysia</td>
        </tr>
        <tr>
          <th>3</th>
          <td>kongsi</td>
          <td>mdb</td>
          <td>bidang</td>
          <td>urus</td>
          <td>raja</td>
        </tr>
        <tr>
          <th>4</th>
          <td>alam</td>
          <td>raja</td>
          <td>teknikal</td>
          <td>jho</td>
          <td>syarikat</td>
        </tr>
        <tr>
          <th>5</th>
          <td>kongsi alam</td>
          <td>didik</td>
          <td>didik</td>
          <td>urus niaga</td>
          <td>tingkat</td>
        </tr>
        <tr>
          <th>6</th>
          <td>sedia</td>
          <td>maju</td>
          <td>negara negara</td>
          <td>low</td>
          <td>ajar</td>
        </tr>
        <tr>
          <th>7</th>
          <td>selatan</td>
          <td>bidang</td>
          <td>tani</td>
          <td>jho low</td>
          <td>sewa</td>
        </tr>
        <tr>
          <th>8</th>
          <td>alam bangun</td>
          <td>maju bidang</td>
          <td>negara maju bidang</td>
          <td>tuntut</td>
          <td>jual syarikat</td>
        </tr>
        <tr>
          <th>9</th>
          <td>kongsi alam bangun</td>
          <td>rakyat</td>
          <td>tani didik latih</td>
          <td>tuntut sivil</td>
          <td>rakyat</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    lsa.get_sentences(5)




.. parsed-literal::

    ['sedia kongsi alam bangun ekonomi sosial negara bangun rangka program kerjasama teknikal malaysia mtcp sedia malaysia kongsi alam bangun negara negara selatan selatan',
     'sedia kongsi alam bangun ekonomi sosial negara bangun rangka program kerjasama teknikal malaysia mtcp sedia malaysia kongsi alam bangun negara negara selatan selatan',
     'negara bangun malaysia main peran kongsi alam mahir bangun kawasan bandar',
     'negara bangun malaysia main peran kongsi alam mahir bangun kawasan bandar',
     'bantu negara negara maju bidang ancang ekonomi wang dagang tani didik latih teknikal industri diplomasi']



.. code:: python

    lsa.get_topics(10)




.. parsed-literal::

    [(0,
      'negara bangun malaysia kongsi alam kongsi alam sedia selatan alam bangun kongsi alam bangun'),
     (1, 'negara wang menteri mdb raja didik maju bidang maju bidang rakyat'),
     (2,
      'negara maju maju bidang bidang teknikal didik negara negara tani negara maju bidang tani didik latih'),
     (3, 'mdb niaga doj urus jho urus niaga low jho low tuntut tuntut sivil'),
     (4,
      'projek jual malaysia raja syarikat tingkat ajar sewa jual syarikat rakyat'),
     (5,
      'parti pilih rakyat tangguh umno pimpin negara malaysia rakyat malaysia tangguh pilih'),
     (6,
      'rakyat malaysia menteri bangun asli perdana kawasan negara bangun kawasan main'),
     (7,
      'ajar tingkat ajar ajar proses raja bidang didik tingkat maju raja tingkat maju laku raja tingkat didik proses'),
     (8,
      'bangun projek bandar kawasan bangun kawasan main mahir peran kongsi alam peran kongsi kawasan bandar'),
     (9,
      'asli masyarakat jakoa bangun arus perdana pelopor ganti arus arus perdana bangun arus masyarakat asli')]
