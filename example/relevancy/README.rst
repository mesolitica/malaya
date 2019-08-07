
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 6.05 s, sys: 1.62 s, total: 7.66 s
    Wall time: 13 s


Explanation
-----------

Positive relevancy: The article or piece of text is relevant, tendency
is high to become not a fake news. Can be a positive or negative
sentiment.

Negative relevancy: The article or piece of text is not relevant,
tendency is high to become a fake news. Can be a positive or negative
sentiment.

Right now relevancy module only support deep learning model.

.. code:: ipython3

    negative_text = 'Roti Massimo Mengandungi DNA Babi. Roti produk Massimo keluaran Syarikat The Italian Baker mengandungi DNA babi. Para pengguna dinasihatkan supaya tidak memakan produk massimo. Terdapat pelbagai produk roti keluaran syarikat lain yang boleh dimakan dan halal. Mari kita sebarkan berita ini supaya semua rakyat Malaysia sedar dengan apa yang mereka makna setiap hari. Roti tidak halal ada DNA babi jangan makan ok.'
    positive_text = 'Jabatan Kemajuan Islam Malaysia memperjelaskan dakwaan sebuah mesej yang dikitar semula, yang mendakwa kononnya kod E dikaitkan dengan kandungan lemak babi sepertimana yang tular di media sosial. . Tular: November 2017 . Tular: Mei 2014 JAKIM ingin memaklumkan kepada masyarakat berhubung maklumat yang telah disebarkan secara meluas khasnya melalui media sosial berhubung kod E yang dikaitkan mempunyai lemak babi. Untuk makluman, KOD E ialah kod untuk bahan tambah (aditif) dan ianya selalu digunakan pada label makanan di negara Kesatuan Eropah. Menurut JAKIM, tidak semua nombor E yang digunakan untuk membuat sesuatu produk makanan berasaskan dari sumber yang haram. Sehubungan itu, sekiranya sesuatu produk merupakan produk tempatan dan mendapat sijil Pengesahan Halal Malaysia, maka ia boleh digunakan tanpa was-was sekalipun mempunyai kod E-kod. Tetapi sekiranya produk tersebut bukan produk tempatan serta tidak mendapat sijil pengesahan halal Malaysia walaupun menggunakan e-kod yang sama, pengguna dinasihatkan agar berhati-hati dalam memilih produk tersebut.'

BERT model
----------

BERT is the best relevancy model in term of accuracy, you can check
relevancy accuracy here,
https://malaya.readthedocs.io/en/latest/Accuracy.html#relevancy-analysis.
Question is, why BERT?

1. Transformer model learn the context of a word based on all of its
   surroundings (live string), bidirectionally. So it much better
   understand left and right hand side relationships.
2. Because of transformer able to leverage to context during live
   string, we dont need to capture available words in this world,
   instead capture substrings and build the attention after that. BERT
   will never have Out-Of-Vocab problem.

List available BERT models
--------------------------

.. code:: ipython3

    malaya.relevancy.available_bert_model()




.. parsed-literal::

    ['multilanguage', 'base']



Load BERT models
----------------

.. code:: ipython3

    model = malaya.relevancy.bert(model = 'base')


.. parsed-literal::

    downloading frozen /Users/huseinzol/Malaya/relevancy/base model


.. parsed-literal::

    447MB [01:19, 6.44MB/s]                          
    WARNING: Logging before flag parsing goes to stderr.
    W0807 17:39:06.005466 4628391360 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:45: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    W0807 17:39:06.006567 4628391360 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:46: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    W0807 17:39:11.612503 4628391360 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:41: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


Predict single string
^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model.predict(negative_text,get_proba=True)




.. parsed-literal::

    {'positive': 0.49505678, 'negative': 0.5049432}



Predict batch of strings
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model.predict_batch([negative_text, positive_text],get_proba=True)




.. parsed-literal::

    [{'positive': 0.9720986, 'negative': 0.027901381},
     {'positive': 0.00013454593, 'negative': 0.9998654}]



Problem with ``predict_batch``, short string need to pad with empty
token to make sure the length is same with longer text. This can lead to
overfeat, so, beware.

Open relevancy visualization dashboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Default when you call ``predict_words`` it will open a browser with
visualization dashboard, you can disable by ``visualization=False``.

.. code:: ipython3

    model.predict_words(negative_text)

.. code:: ipython3

    from IPython.core.display import Image, display
    
    display(Image('relevancy-negative.png', width=800))



.. image:: load-relevancy_files/load-relevancy_15_0.png
   :width: 800px


List available deep learning models
-----------------------------------

.. code:: ipython3

    malaya.relevancy.available_deep_model()




.. parsed-literal::

    ['self-attention', 'dilated-cnn']



As you can see, we are not using recurrent architecture for relevancy
classification. Article or news can be really a long text, and when
talking about recurrent on long text dependency, we might have a problem
about gradient vanishing or long term dependency, plus it is very
expensive to calculate.

Load deep learning models
-------------------------

Good thing about deep learning models from Malaya, it returns
``Attention`` result, means, which part of words give the high impact to
the results. But to get ``Attention``, you need to set
``get_proba=True``.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

Load dilated-cnn model
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    model = malaya.relevancy.deep_model('dilated-cnn')

Predict single string
^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model.predict(positive_text)




.. parsed-literal::

    'positive'



.. code:: ipython3

    result = model.predict(positive_text,get_proba=True)
    result




.. parsed-literal::

    {'positive': 0.9892117,
     'negative': 0.010788266,
     'attention': {'Jabatan': 0.011842445,
      'Kemajuan': 0.0004990599,
      'Islam': 0.0105412435,
      'Malaysia': 1.8620213e-07,
      'memperjelaskan': 0.00090273784,
      'dakwaan': 0.009680763,
      'sebuah': 0.34631768,
      'mesej': 0.0032939622,
      'yang': 7.391781e-06,
      'dikitar': 0.009578085,
      'semula': 0.25352243,
      ',': 0.0,
      'mendakwa': 0.039328944,
      'kononnya': 0.0005362838,
      'kod': 1.7578169e-06,
      'E': 0.0,
      'dikaitkan': 0.000102088605,
      'dengan': 0.00080459507,
      'kandungan': 0.0044339728,
      'lemak': 3.484632e-05,
      'babi': 4.280925e-06,
      'sepertimana': 0.0030541394,
      'tular': 0.0010676038,
      'di': 0.0,
      'media': 3.1618736e-06,
      'sosial': 1.3851669e-05,
      '.': 0.0,
      'Tular': 0.00011631558,
      ':': 0.0,
      'November 2017': 0.00085097033,
      'Mei 2014': 4.920134e-05,
      'JAKIM': 0.0012241085,
      'ingin': 0.0881682,
      'memaklumkan': 7.052896e-05,
      'kepada': 0.021570362,
      'masyarakat': 0.00023094988,
      'berhubung': 1.1057678e-05,
      'maklumat': 0.00026868083,
      'telah': 7.2317605e-05,
      'disebarkan': 9.50331e-07,
      'secara': 5.8421193e-07,
      'meluas': 0.00023408467,
      'khasnya': 0.00048214395,
      'melalui': 0.0052656163,
      'mempunyai': 7.862253e-06,
      'Untuk': 2.5891845e-06,
      'makluman': 0.00015925457,
      'KOD': 0.0005097225,
      'ialah': 6.233322e-05,
      'untuk': 0.0025757798,
      'bahan': 2.664205e-07,
      'tambah': 3.5512012e-06,
      '(': 0.0,
      'aditif': 8.97466e-05,
      ')': 0.0,
      'dan': 0.0,
      'ianya': 2.8783128e-05,
      'selalu': 3.586144e-05,
      'digunakan': 1.0551267e-05,
      'pada': 5.0717895e-06,
      'label': 4.0046502e-06,
      'makanan': 0.0010502156,
      'negara': 1.742862e-05,
      'Kesatuan': 6.244018e-06,
      'Eropah': 2.1515665e-05,
      'Menurut': 9.812214e-06,
      'tidak': 4.3439675e-07,
      'semua': 0.000835331,
      'nombor': 6.793219e-05,
      'membuat': 0.001359321,
      'sesuatu': 1.5853509e-06,
      'produk': 1.0694217e-07,
      'berasaskan': 5.729283e-07,
      'dari': 3.824304e-05,
      'sumber': 9.358e-06,
      'haram': 3.41809e-06,
      'Sehubungan': 7.6232165e-05,
      'itu': 7.043718e-06,
      'sekiranya': 9.1800786e-08,
      'merupakan': 4.6512785e-05,
      'tempatan': 0.0005657043,
      'mendapat': 2.7107068e-05,
      'sijil': 7.168571e-06,
      'Pengesahan': 1.9922183e-05,
      'Halal': 4.70037e-05,
      'maka': 3.5196587e-07,
      'ia': 1.0549179e-06,
      'boleh': 2.1123176e-06,
      'tanpa': 2.927889e-06,
      'was': 3.6377685e-06,
      '-': 0.0,
      'sekalipun': 4.786906e-07,
      'Tetapi': 1.3105414e-07,
      'tersebut': 7.7244266e-08,
      'bukan': 1.510455e-06,
      'serta': 1.9440507e-05,
      'pengesahan': 0.00013690614,
      'halal': 4.2278854e-05,
      'walaupun': 2.7934178e-07,
      'menggunakan': 9.947374e-05,
      'e': 0.0,
      'sama': 2.0695973e-07,
      'pengguna': 1.64596e-07,
      'dinasihatkan': 1.1992488e-08,
      'agar': 2.787569e-06,
      'berhati': 1.4157456e-07,
      'hati': 9.918707e-06,
      'dalam': 5.582234e-07,
      'memilih': 4.5280743e-08}}



.. code:: ipython3

    plt.figure(figsize = (15, 5))
    keys = result['attention'].keys()
    values = result['attention'].values()
    aranged = [i for i in range(len(keys))]
    plt.bar(aranged, values)
    plt.xticks(aranged, keys, rotation = 'vertical')
    plt.show()



.. image:: load-relevancy_files/load-relevancy_26_0.png


Open relevancy visualization dashboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Default when you call ``predict_words`` it will open a browser with
visualization dashboard, you can disable by ``visualization=False``.

.. code:: ipython3

    model.predict_words(positive_text)

.. code:: ipython3

    from IPython.core.display import Image, display
    
    display(Image('relevancy-dilated-cnn.png', width=800))



.. image:: load-relevancy_files/load-relevancy_29_0.png
   :width: 800px


Predict batch of strings
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model.predict_batch([negative_text, positive_text],get_proba=True)




.. parsed-literal::

    [{'positive': 0.8330916, 'negative': 0.16690837},
     {'positive': 0.9961637, 'negative': 0.00383623}]



**You might want to try ``self-attention`` by yourself.**

Stacking models
---------------

More information, you can read at
https://malaya.readthedocs.io/en/latest/Stack.html

.. code:: ipython3

    cnn = malaya.relevancy.deep_model('dilated-cnn')
    attention = malaya.relevancy.deep_model('self-attention')

.. code:: ipython3

    malaya.stack.predict_stack([cnn, attention], [positive_text, negative_text])




.. parsed-literal::

    [{'positive': 0.83132225, 'negative': 0.02162251},
     {'positive': 0.3027468, 'negative': 0.54677427}]


