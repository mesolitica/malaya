
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 13.2 s, sys: 1.4 s, total: 14.6 s
    Wall time: 18.2 s


Explanation
-----------

Positive subjectivity: based on or influenced by personal feelings,
tastes, or opinions. Can be a positive or negative sentiment.

Negative subjectivity: based on a report or a fact. Can be a positive or
negative sentiment.

.. code:: python

    negative_text = 'Kerajaan negeri Kelantan mempersoalkan motif kenyataan Menteri Kewangan Lim Guan Eng yang hanya menyebut Kelantan penerima terbesar bantuan kewangan dari Kerajaan Persekutuan. Sedangkan menurut Timbalan Menteri Besarnya, Datuk Mohd Amar Nik Abdullah, negeri lain yang lebih maju dari Kelantan turut mendapat pembiayaan dan pinjaman.'
    positive_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'

Load multinomial model
----------------------

.. code:: python

    model = malaya.subjective.multinomial()
    print(model.predict(positive_text,get_proba=True))
    print(model.predict(negative_text,get_proba=True))
    model.predict_batch([negative_text,negative_text],get_proba=True)


.. parsed-literal::

    {'negative': 0.46203835811002486, 'positive': 0.5379616418899767}
    {'negative': 0.8698758314042119, 'positive': 0.13012416859579023}




.. parsed-literal::

    [{'negative': 0.8698758314042119, 'positive': 0.13012416859579023},
     {'negative': 0.8698758314042119, 'positive': 0.13012416859579023}]



Load xgb model
--------------

.. code:: python

    model = malaya.subjective.xgb()
    print(model.predict(positive_text,get_proba=True))
    print(model.predict(negative_text,get_proba=True))
    model.predict_batch([negative_text,negative_text],get_proba=True)


.. parsed-literal::

    {'negative': 0.4284472, 'positive': 0.5715528}
    {'negative': 0.9249991, 'positive': 0.07500088}




.. parsed-literal::

    [{'negative': 0.9249991, 'positive': 0.07500088},
     {'negative': 0.9249991, 'positive': 0.07500088}]



List available deep learning models
-----------------------------------

.. code:: python

    malaya.subjective.available_deep_model()




.. parsed-literal::

    ['fast-text',
     'hierarchical',
     'bahdanau',
     'luong',
     'bidirectional',
     'bert',
     'entity-network']



.. code:: python

    for i in malaya.subjective.available_deep_model():
        print('Testing %s model'%(i))
        model = malaya.subjective.deep_model(i)
        print(model.predict(negative_text))
        print(model.predict_batch([negative_text, positive_text]))
        print()


.. parsed-literal::

    Testing fast-text model
    {'negative': 0.9999857, 'positive': 1.4311945e-05}
    [{'negative': 0.99998176, 'positive': 1.8284805e-05}, {'negative': 0.6197021, 'positive': 0.3802979}]

    Testing hierarchical model
    {'negative': 0.9999937, 'positive': 6.295898e-06, 'attention': [['kerajaan', 0.0012118855], ['negeri', 0.0020442759], ['kelantan', 0.0021679106], ['mempersoalkan', 0.0022253846], ['motif', 0.0033358238], ['kenyataan', 0.008720655], ['menteri', 0.04487104], ['kewangan', 0.10065087], ['lim', 0.059503824], ['guan', 0.15100963], ['eng', 0.04026543], ['yang', 0.043928813], ['hanya', 0.01824422], ['menyebut', 0.022727199], ['kelantan', 0.016984997], ['penerima', 0.024233121], ['terbesar', 0.011635249], ['bantuan', 0.001963468], ['kewangan', 0.0072085205], ['dari', 0.0021965506], ['kerajaan', 0.0027234056], ['persekutuan', 0.0014449719], ['sedangkan', 0.0021539854], ['menurut', 0.002655797], ['timbalan', 0.0115157785], ['menteri', 0.005335992], ['besarnya', 0.028362982], ['datuk', 0.009886651], ['mohd', 0.02055805], ['amar', 0.04487915], ['nik', 0.17517959], ['abdullah', 0.02919604], ['negeri', 0.041412108], ['lain', 0.030245796], ['yang', 0.0063164025], ['lebih', 0.006518348], ['maju', 0.001788858], ['dari', 0.008994939], ['kelantan', 0.0024882965], ['turut', 0.00038583783], ['mendapat', 0.0010022834], ['pembiayaan', 0.0012560145], ['pinjaman', 0.000569819]]}
    [{'negative': 0.99999154, 'positive': 8.507095e-06}, {'negative': 0.3101697, 'positive': 0.6898303}]

    Testing bahdanau model
    {'negative': 0.9999198, 'positive': 8.020081e-05, 'attention': [['kerajaan', 0.0050431504], ['negeri', 0.009287076], ['kelantan', 0.0040232316], ['mempersoalkan', 0.021462629], ['motif', 0.0053027985], ['kenyataan', 0.010193134], ['menteri', 0.033055097], ['kewangan', 0.0049648904], ['lim', 0.0040454715], ['guan', 0.0040232316], ['eng', 0.0040232316], ['yang', 0.004409306], ['hanya', 0.0053563444], ['menyebut', 0.0076429546], ['kelantan', 0.0040232316], ['penerima', 0.0068015233], ['terbesar', 0.013676326], ['bantuan', 0.009508641], ['kewangan', 0.0049648904], ['dari', 0.00456691], ['kerajaan', 0.0050431504], ['persekutuan', 0.022685695], ['sedangkan', 0.0034438083], ['menurut', 0.0051381364], ['timbalan', 0.0040232316], ['menteri', 0.033055097], ['besarnya', 0.013676326], ['datuk', 0.017668026], ['mohd', 0.0040232316], ['amar', 0.0043330067], ['nik', 0.0036893887], ['abdullah', 0.0040232316], ['negeri', 0.009287076], ['lain', 0.003357054], ['yang', 0.004409306], ['lebih', 0.005977192], ['maju', 0.0037889858], ['dari', 0.00456691], ['kelantan', 0.0040232316], ['turut', 0.60284543], ['mendapat', 0.01692126], ['pembiayaan', 0.01374086], ['pinjaman', 0.043906275]]}
    [{'negative': 0.9999558, 'positive': 4.4271066e-05}, {'negative': 0.64894015, 'positive': 0.35105988}]

    Testing luong model
    {'negative': 0.051370017, 'positive': 0.94863, 'attention': [['kerajaan', 0.032675084], ['negeri', 0.016353898], ['kelantan', 0.019432731], ['mempersoalkan', 0.0119247725], ['motif', 0.01284633], ['kenyataan', 0.01961166], ['menteri', 0.008495899], ['kewangan', 0.018757388], ['lim', 0.018171076], ['guan', 0.019432731], ['eng', 0.019432731], ['yang', 0.009098401], ['hanya', 0.0077451197], ['menyebut', 0.053905513], ['kelantan', 0.019432731], ['penerima', 0.0129601145], ['terbesar', 0.018512659], ['bantuan', 0.0077343024], ['kewangan', 0.018757388], ['dari', 0.009422931], ['kerajaan', 0.032675084], ['persekutuan', 0.008676352], ['sedangkan', 0.06508403], ['menurut', 0.0082393885], ['timbalan', 0.019432731], ['menteri', 0.008495899], ['besarnya', 0.018512659], ['datuk', 0.009194007], ['mohd', 0.019432731], ['amar', 0.014017649], ['nik', 0.014721154], ['abdullah', 0.019432731], ['negeri', 0.016353898], ['lain', 0.024851179], ['yang', 0.009098401], ['lebih', 0.008518517], ['maju', 0.00883592], ['dari', 0.009422931], ['kelantan', 0.019432731], ['turut', 0.008133899], ['mendapat', 0.016000977], ['pembiayaan', 0.028110703], ['pinjaman', 0.2586229]]}
    [{'negative': 0.034302887, 'positive': 0.9656971}, {'negative': 0.7590918, 'positive': 0.24090825}]

    Testing bidirectional model
    {'negative': 0.9999918, 'positive': 8.242413e-06}
    [{'negative': 0.99999166, 'positive': 8.287703e-06}, {'negative': 0.992438, 'positive': 0.007561967}]

    Testing bert model
    {'negative': 0.98487025, 'positive': 0.0151297115}
    [{'negative': 0.98487025, 'positive': 0.015129704}, {'negative': 0.98668575, 'positive': 0.013314218}]

    Testing entity-network model
    {'negative': 0.6470485, 'positive': 0.35295156}
    [{'negative': 0.6470482, 'positive': 0.35295185}, {'negative': 0.65467215, 'positive': 0.34532788}]



Unsupervised important words learning
-------------------------------------

.. code:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set() # i just really like seaborn colors

Visualizing bahdanau model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    model = malaya.subjective.deep_model('bahdanau')
    result = model.predict(negative_text)['attention']

    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-subjectivity_files/load-subjectivity_13_0.png


Visualizing luong model
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    model = malaya.subjective.deep_model('luong')
    result = model.predict(negative_text)['attention']

    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-subjectivity_files/load-subjectivity_15_0.png


Visualizing hierarchical model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    model = malaya.subjective.deep_model('hierarchical')
    result = model.predict(negative_text)['attention']

    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-subjectivity_files/load-subjectivity_17_0.png


Load Sparse deep learning models
--------------------------------

What happen if a word not included in the dictionary of the models? like
``setan``, what if ``setan`` appeared in text we want to classify? We
found this problem when classifying social media texts / posts. Words
used not really a vocabulary-based contextual.

Malaya will treat **unknown words** as ``<UNK>``, so, to solve this
problem, we need to use N-grams character based. Malaya chose tri-grams
until fifth-grams.

.. code:: python

   setan = ['set', 'eta', 'tan']

Sklearn provided easy interface to use n-grams, problem is, it is very
sparse, a lot of zeros and not memory efficient. Sklearn returned sparse
matrix for the result, lucky Tensorflow already provided some sparse
function.

malaya.subjective.available_sparse_deep_model()

Right now Malaya only provide 1 sparse model, ``fast-text-char``. We
will try to evolve it.

.. code:: python

    sparse_model = malaya.subjective.sparse_deep_model()


.. parsed-literal::

    downloading frozen /Users/huseinzol/Malaya/subjective/fast-text-char model


.. parsed-literal::

    16.0MB [00:04, 3.16MB/s]
    1.00MB [00:00, 536MB/s]
      0%|          | 0.00/0.05 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading frozen /Users/huseinzol/Malaya/subjective/fast-text-char index
    downloading frozen /Users/huseinzol/Malaya/subjective/fast-text-char meta


.. parsed-literal::

    1.00MB [00:00, 15.9MB/s]
      0%|          | 0.00/2.77 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading frozen /Users/huseinzol/Malaya/subjective/fast-text-char vector


.. parsed-literal::

    3.00MB [00:00, 4.00MB/s]


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from /Users/huseinzol/Malaya/subjective/fast-text-char/model.ckpt


.. code:: python

    sparse_model.predict(positive_text)




.. parsed-literal::

    {'negative': 0.06594214, 'positive': 0.93405783}



.. code:: python

    sparse_model.predict_batch([positive_text, negative_text])




.. parsed-literal::

    [{'negative': 0.06594214, 'positive': 0.93405783},
     {'negative': 0.9535811, 'positive': 0.046418883}]
