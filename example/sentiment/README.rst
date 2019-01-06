
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.9 s, sys: 1.45 s, total: 14.4 s
    Wall time: 18 s


.. code:: ipython3

    positive_text = 'Kerajaan negeri Kelantan mempersoalkan motif kenyataan Menteri Kewangan Lim Guan Eng yang hanya menyebut Kelantan penerima terbesar bantuan kewangan dari Kerajaan Persekutuan. Sedangkan menurut Timbalan Menteri Besarnya, Datuk Mohd Amar Nik Abdullah, negeri lain yang lebih maju dari Kelantan turut mendapat pembiayaan dan pinjaman.'
    negative_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'

Load multinomial model
----------------------

.. code:: ipython3

    model = malaya.sentiment.multinomial()
    print(model.predict(positive_text,get_proba=True))
    print(model.predict(negative_text,get_proba=True))
    model.predict_batch([negative_text,negative_text],get_proba=True)


.. parsed-literal::

    {'negative': 0.27780816431771815, 'positive': 0.7221918356822792}
    {'negative': 0.4230539695981826, 'positive': 0.5769460304018175}




.. parsed-literal::

    [{'negative': 0.4230539695981826, 'positive': 0.5769460304018175},
     {'negative': 0.4230539695981826, 'positive': 0.5769460304018175}]



Load xgb model
--------------

.. code:: ipython3

    model = malaya.sentiment.xgb()
    print(model.predict(positive_text,get_proba=True))
    print(model.predict(negative_text,get_proba=True))
    model.predict_batch([negative_text,negative_text],get_proba=True)


.. parsed-literal::

    {'negative': 0.44467238, 'positive': 0.5553276}
    {'negative': 0.47532737, 'positive': 0.5246726}




.. parsed-literal::

    [{'negative': 0.47532737, 'positive': 0.5246726},
     {'negative': 0.47532737, 'positive': 0.5246726}]



List available deep learning models
-----------------------------------

.. code:: ipython3

    malaya.sentiment.available_deep_model()




.. parsed-literal::

    ['fast-text',
     'hierarchical',
     'bahdanau',
     'luong',
     'bidirectional',
     'bert',
     'entity-network']



Load deep learning models
-------------------------

.. code:: ipython3

    for i in malaya.sentiment.available_deep_model():
        print('Testing %s model'%(i))
        model = malaya.sentiment.deep_model(i)
        print(model.predict(negative_text))
        print(model.predict_batch([negative_text, positive_text]))
        print()


.. parsed-literal::

    Testing fast-text model
    {'negative': 0.99185514, 'positive': 0.008144839}
    [{'negative': 0.8494132, 'positive': 0.15058675}, {'negative': 0.04582213, 'positive': 0.95417786}]
    
    Testing hierarchical model
    {'negative': 0.110631816, 'positive': 0.8893682, 'attention': [['kerajaan', 0.10388958], ['sebenarnya', 0.4929846], ['sangat', 0.29071146], ['bencikan', 0.07854219], ['rakyatnya', 0.019831425], ['minyak', 0.010329048], ['naik', 0.002903083], ['segalanya', 0.0008086153]]}
    [{'negative': 0.096144125, 'positive': 0.9038559}, {'negative': 0.12305506, 'positive': 0.87694496}]
    
    Testing bahdanau model
    {'negative': 0.27703816, 'positive': 0.72296184, 'attention': [['kerajaan', 0.13535759], ['sebenarnya', 0.023817956], ['sangat', 0.030500164], ['bencikan', 0.637391], ['rakyatnya', 0.04856573], ['minyak', 0.036034647], ['naik', 0.060078725], ['segalanya', 0.028254228]]}
    [{'negative': 0.60924244, 'positive': 0.3907576}, {'negative': 0.2196782, 'positive': 0.78032184}]
    
    Testing luong model
    {'negative': 0.60044205, 'positive': 0.3995579, 'attention': [['kerajaan', 0.15034309], ['sebenarnya', 0.08993225], ['sangat', 0.068059266], ['bencikan', 0.122634426], ['rakyatnya', 0.096616365], ['minyak', 0.102913655], ['naik', 0.18173374], ['segalanya', 0.18776724]]}
    [{'negative': 0.98223615, 'positive': 0.017763875}, {'negative': 0.114074536, 'positive': 0.88592553}]
    
    Testing bidirectional model
    {'negative': 0.11350883, 'positive': 0.8864912}
    [{'negative': 0.24156873, 'positive': 0.7584313}, {'negative': 0.337586, 'positive': 0.66241395}]
    
    Testing bert model
    {'negative': 0.992415, 'positive': 0.007585052}
    [{'negative': 0.992415, 'positive': 0.007585059}, {'negative': 0.9923813, 'positive': 0.0076187113}]
    
    Testing entity-network model
    {'negative': 0.5229405, 'positive': 0.4770595}
    [{'negative': 0.5229405, 'positive': 0.4770595}, {'negative': 0.6998231, 'positive': 0.3001769}]
    


Unsupervised important words learning
-------------------------------------

.. code:: ipython3

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set() # i just really like seaborn colors

Visualizing bahdanau model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model = malaya.sentiment.deep_model('bahdanau')
    result = model.predict(positive_text)['attention']
    
    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-sentiment_files/load-sentiment_13_0.png


Visualizing luong model
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model = malaya.sentiment.deep_model('luong')
    result = model.predict(positive_text)['attention']
    
    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-sentiment_files/load-sentiment_15_0.png


Visualizing hierarchical model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model = malaya.sentiment.deep_model('hierarchical')
    result = model.predict(positive_text)['attention']
    
    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-sentiment_files/load-sentiment_17_0.png


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

.. code:: ipython3

    malaya.sentiment.available_sparse_deep_model()




.. parsed-literal::

    ['fast-text-char']



Right now Malaya only provide 1 sparse model, ``fast-text-char``. We
will try to evolve it.

.. code:: ipython3

    sparse_model = malaya.sentiment.sparse_deep_model()


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from /Users/huseinzol/Malaya/sentiment/fast-text-char/model.ckpt


.. code:: ipython3

    sparse_model.predict(positive_text)




.. parsed-literal::

    {'negative': 0.38546535, 'positive': 0.6145346}



.. code:: ipython3

    sparse_model.predict_batch([positive_text, negative_text])




.. parsed-literal::

    [{'negative': 0.38546535, 'positive': 0.6145346},
     {'negative': 0.50480145, 'positive': 0.49519858}]



**Not bad huh, but the polarity is not really high as word-based models.
Word-based models can get negative / positive value really near to 1.0**

Train a multinomial model using custom dataset
----------------------------------------------

.. code:: ipython3

    import pandas as pd
    df = pd.read_csv('tests/02032018.csv',sep=';')
    df = df.iloc[3:,1:]
    df.columns = ['text','label']
    corpus = df.text.tolist()

corpus should be [(text, label)]

.. code:: ipython3

    dataset = [[df.iloc[i,0],df.iloc[i,1]] for i in range(df.shape[0])]
    bayes = malaya.sentiment.train_multinomial(dataset)


.. parsed-literal::

                 precision    recall  f1-score   support
    
       Negative       0.00      0.00      0.00        15
        Neutral       0.50      0.06      0.11        17
       Positive       0.49      1.00      0.66        29
    
    avg / total       0.37      0.49      0.34        61
    


You also able to feed directory location
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   directory
       |
       |- adidas
       |- apple
       |- hungry

.. code:: ipython3

    bayes = malaya.sentiment.train_multinomial('tests/local')


.. parsed-literal::

                 precision    recall  f1-score   support
    
         adidas       0.95      0.59      0.73       311
          apple       0.97      0.61      0.75       458
         hungry       0.79      0.92      0.85      1079
       kerajaan       0.84      0.82      0.83      1388
           nike       0.96      0.50      0.66       325
    pembangkang       0.71      0.86      0.78      1498
    
    avg / total       0.82      0.80      0.79      5059
    


.. code:: ipython3

    bayes.predict('saya suka kerajaan dan anwar ibrahim', get_proba = True)




.. parsed-literal::

    {'adidas': 0.0005687282506828817,
     'apple': 0.000662907603798319,
     'hungry': 0.009170662067707083,
     'kerajaan': 0.06985529223854361,
     'nike': 0.0006071193644321936,
     'pembangkang': 0.9191352904748373}



Train a multinomial using skip-gram vectorization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    bayes = malaya.sentiment.train_multinomial(
        'tests/local', vector = 'skip-gram', ngram_range = (1, 3), skip = 5
    )


.. parsed-literal::

                 precision    recall  f1-score   support
    
         adidas       0.38      0.84      0.52       328
          apple       0.50      0.89      0.64       499
         hungry       0.84      0.93      0.88      1083
       kerajaan       0.89      0.61      0.72      1332
           nike       0.57      0.78      0.66       323
    pembangkang       0.89      0.53      0.66      1494
    
    avg / total       0.79      0.71      0.71      5059
    


.. code:: ipython3

    bayes.predict('saya suka kerajaan dan anwar ibrahim', get_proba = True)




.. parsed-literal::

    {'adidas': 7.916087875403519e-14,
     'apple': 3.823879121188251e-14,
     'hungry': 2.319120520022076e-10,
     'kerajaan': 8.978991657701227e-07,
     'nike': 1.1627344175225e-13,
     'pembangkang': 0.999999101868701}


