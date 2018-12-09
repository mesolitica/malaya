
.. code:: ipython3

    import malaya

.. code:: ipython3

    positive_text = 'Kerajaan negeri Kelantan mempersoalkan motif kenyataan Menteri Kewangan Lim Guan Eng yang hanya menyebut Kelantan penerima terbesar bantuan kewangan dari Kerajaan Persekutuan. Sedangkan menurut Timbalan Menteri Besarnya, Datuk Mohd Amar Nik Abdullah, negeri lain yang lebih maju dari Kelantan turut mendapat pembiayaan dan pinjaman.'
    negative_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'

Load multinomial model
----------------------

.. code:: ipython3

    model = malaya.pretrained_bayes_sentiment()
    print(model.predict(positive_text,get_proba=True))
    print(model.predict(negative_text,get_proba=True))
    model.predict_batch([negative_text,negative_text],get_proba=True)


.. parsed-literal::

    downloading SENTIMENT pickled multinomial model


.. parsed-literal::

    2.00MB [00:00, 4.86MB/s]                          
      0%|          | 0.00/9.08 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading SENTIMENT pickled multinomial tfidf vectorization


.. parsed-literal::

    10.0MB [00:03, 3.22MB/s]                          


.. parsed-literal::

    {'negative': 0.27780816431771815, 'positive': 0.7221918356822792}
    {'negative': 0.4230539695981826, 'positive': 0.5769460304018175}




.. parsed-literal::

    [{'negative': 0.4230539695981826, 'positive': 0.5769460304018175},
     {'negative': 0.4230539695981826, 'positive': 0.5769460304018175}]



Load xgb model
--------------

.. code:: ipython3

    model = malaya.pretrained_xgb_sentiment()
    print(model.predict(positive_text,get_proba=True))
    print(model.predict(negative_text,get_proba=True))
    model.predict_batch([negative_text,negative_text],get_proba=True)


.. parsed-literal::

      0%|          | 0.00/1.78 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading SENTIMENT pickled XGB model


.. parsed-literal::

    2.00MB [00:00, 3.88MB/s]                          
      0%|          | 0.00/9.08 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading SENTIMENT pickled XGB tfidf vectorization


.. parsed-literal::

    10.0MB [00:03, 3.28MB/s]                          


.. parsed-literal::

    {'negative': 0.44467238, 'positive': 0.5553276}
    {'negative': 0.47532737, 'positive': 0.5246726}




.. parsed-literal::

    [{'negative': 0.47532737, 'positive': 0.5246726},
     {'negative': 0.47532737, 'positive': 0.5246726}]



List available deep learning models
-----------------------------------

.. code:: ipython3

    malaya.get_available_sentiment_models()




.. parsed-literal::

    ['fast-text',
     'hierarchical',
     'bahdanau',
     'luong',
     'bidirectional',
     'bert',
     'entity-network']



.. code:: ipython3

    for i in malaya.get_available_sentiment_models():
        print('Testing %s model'%(i))
        model = malaya.deep_sentiment(i)
        print(model.predict(negative_text))
        print(model.predict_batch([negative_text, positive_text]))
        print()


.. parsed-literal::

    Testing fast-text model
    downloading SENTIMENT frozen fast-text model


.. parsed-literal::

    127MB [00:43, 2.92MB/s]                          
      0%|          | 0.00/0.45 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading SENTIMENT fast-text dictionary


.. parsed-literal::

    1.00MB [00:00, 5.89MB/s]                   
      0%|          | 0.00/1.68 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading SENTIMENT fast-text bigrams


.. parsed-literal::

    2.00MB [00:00, 4.01MB/s]                          


.. parsed-literal::

    {'negative': 0.99185514, 'positive': 0.008144839}


.. parsed-literal::

      0%|          | 0.00/23.6 [00:00<?, ?MB/s]

.. parsed-literal::

    [{'negative': 0.8494132, 'positive': 0.15058675}, {'negative': 0.04582213, 'positive': 0.95417786}]
    
    Testing hierarchical model
    downloading SENTIMENT frozen hierarchical model


.. parsed-literal::

    24.0MB [00:07, 3.14MB/s]                          
      0%|          | 0.00/0.45 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading SENTIMENT hierarchical dictionary


.. parsed-literal::

    1.00MB [00:00, 5.45MB/s]                   


.. parsed-literal::

    {'negative': 0.119958304, 'positive': 0.88004166, 'attention': [['kerajaan', 0.07279364], ['sebenarnya', 0.26620612], ['sangat', 0.39088085], ['bencikan', 0.18420841], ['rakyatnya', 0.077199794], ['minyak', 0.0068039955], ['naik', 0.0014752398], ['segalanya', 0.0004320148]]}
    [{'negative': 0.036293767, 'positive': 0.96370625}, {'negative': 0.0425552, 'positive': 0.95744485}]
    
    Testing bahdanau model
    downloading SENTIMENT frozen bahdanau model


.. parsed-literal::

    20.0MB [00:09, 2.16MB/s]                          
      0%|          | 0.00/0.45 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading SENTIMENT bahdanau dictionary


.. parsed-literal::

    1.00MB [00:00, 5.57MB/s]                   


.. parsed-literal::

    {'negative': 0.35867092, 'positive': 0.64132905, 'attention': [['kerajaan', 0.13515094], ['sebenarnya', 0.023719592], ['sangat', 0.030418796], ['bencikan', 0.63889986], ['rakyatnya', 0.048370756], ['minyak', 0.03579358], ['naik', 0.059553757], ['segalanya', 0.028092839]]}


.. parsed-literal::

      0%|          | 0.00/18.8 [00:00<?, ?MB/s]

.. parsed-literal::

    [{'negative': 0.6422382, 'positive': 0.35776183}, {'negative': 0.42549333, 'positive': 0.5745067}]
    
    Testing luong model
    downloading SENTIMENT frozen luong model


.. parsed-literal::

    19.0MB [00:05, 4.02MB/s]                          
      0%|          | 0.00/0.45 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading SENTIMENT luong dictionary


.. parsed-literal::

    1.00MB [00:00, 5.64MB/s]                   


.. parsed-literal::

    {'negative': 0.9633553, 'positive': 0.036644693, 'attention': [['kerajaan', 0.125], ['sebenarnya', 0.125], ['sangat', 0.125], ['bencikan', 0.125], ['rakyatnya', 0.125], ['minyak', 0.125], ['naik', 0.125], ['segalanya', 0.125]]}


.. parsed-literal::

      0%|          | 0.00/23.1 [00:00<?, ?MB/s]

.. parsed-literal::

    [{'negative': 0.22811669, 'positive': 0.77188325}, {'negative': 0.9460423, 'positive': 0.053957727}]
    
    Testing bidirectional model
    downloading SENTIMENT frozen bidirectional model


.. parsed-literal::

    24.0MB [00:06, 3.54MB/s]                          
      0%|          | 0.00/0.45 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading SENTIMENT bidirectional dictionary


.. parsed-literal::

    1.00MB [00:00, 5.10MB/s]                   


.. parsed-literal::

    {'negative': 0.14917508, 'positive': 0.85082495}


.. parsed-literal::

      0%|          | 0.00/17.9 [00:00<?, ?MB/s]

.. parsed-literal::

    [{'negative': 0.20314497, 'positive': 0.7968551}, {'negative': 0.24708004, 'positive': 0.75292}]
    
    Testing bert model
    downloading SENTIMENT frozen bert model


.. parsed-literal::

    18.0MB [00:05, 4.06MB/s]                          
      0%|          | 0.00/0.45 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading SENTIMENT bert dictionary


.. parsed-literal::

    1.00MB [00:00, 5.30MB/s]                   


.. parsed-literal::

    {'negative': 0.992415, 'positive': 0.007585052}


.. parsed-literal::

      0%|          | 0.00/14.1 [00:00<?, ?MB/s]

.. parsed-literal::

    [{'negative': 0.992415, 'positive': 0.007585059}, {'negative': 0.9923813, 'positive': 0.0076187113}]
    
    Testing entity-network model
    downloading SENTIMENT frozen entity-network model


.. parsed-literal::

    15.0MB [00:03, 3.83MB/s]                          
      0%|          | 0.00/0.45 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading SENTIMENT entity-network dictionary


.. parsed-literal::

    1.00MB [00:00, 3.59MB/s]                   


.. parsed-literal::

    {'negative': 0.5229405, 'positive': 0.4770595}
    [{'negative': 0.5229405, 'positive': 0.4770595}, {'negative': 0.6998231, 'positive': 0.3001769}]
    


Train a multinomial model using custom dataset
----------------------------------------------

.. code:: ipython3

    import pandas as pd

.. code:: ipython3

    df = pd.read_csv('tests/02032018.csv',sep=';')
    df = df.iloc[3:,1:]
    df.columns = ['text','label']
    corpus = df.text.tolist()

corpus should be [(text, label)]

.. code:: ipython3

    dataset = [[df.iloc[i,0],df.iloc[i,1]] for i in range(df.shape[0])]
    bayes = malaya.bayes_sentiment(dataset)


.. parsed-literal::

                 precision    recall  f1-score   support
    
       Negative       0.00      0.00      0.00        13
        Neutral       0.60      0.18      0.27        17
       Positive       0.54      0.97      0.69        31
    
    avg / total       0.44      0.54      0.43        61
    


You also able to feed directory location
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   directory
       |
       |- adidas
       |- apple
       |- hungry

.. code:: ipython3

    bayes = malaya.bayes_sentiment('tests/local')


.. parsed-literal::

                 precision    recall  f1-score   support
    
         adidas       0.95      0.59      0.73       338
          apple       0.98      0.63      0.77       460
         hungry       0.77      0.92      0.84      1050
       kerajaan       0.83      0.82      0.82      1336
           nike       0.96      0.47      0.63       349
    pembangkang       0.72      0.85      0.78      1526
    
    avg / total       0.81      0.79      0.79      5059
    


.. code:: ipython3

    bayes.predict('saya suka kerajaan dan anwar ibrahim', get_proba = True)




.. parsed-literal::

    {'adidas': 0.0005067492499547119,
     'apple': 0.0006219599505916614,
     'hungry': 0.010011663300494363,
     'kerajaan': 0.07345851543210191,
     'nike': 0.0005615484860834134,
     'pembangkang': 0.9148395635807751}



Train a multinomial using skip-gram vectorization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    bayes = malaya.bayes_sentiment(
        'tests/local', vector = 'skip-gram', ngram_range = (1, 3), skip = 5
    )


.. parsed-literal::

                 precision    recall  f1-score   support
    
         adidas       0.35      0.86      0.50       286
          apple       0.50      0.87      0.63       484
         hungry       0.81      0.91      0.86      1016
       kerajaan       0.89      0.58      0.70      1400
           nike       0.54      0.79      0.64       330
    pembangkang       0.87      0.55      0.67      1543
    
    avg / total       0.78      0.69      0.70      5059
    


.. code:: ipython3

    bayes.predict('saya suka kerajaan dan anwar ibrahim', get_proba = True)




.. parsed-literal::

    {'adidas': 2.850507725739238e-13,
     'apple': 1.3603929607881664e-13,
     'hungry': 1.0435702854645526e-09,
     'kerajaan': 3.4176860121376738e-06,
     'nike': 2.749534983926924e-13,
     'pembangkang': 0.9999965812697159}



Visualize malaya attention deep learning models
-----------------------------------------------

.. code:: ipython3

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set() # i just really like seaborn colors

Visualizing bahdanau model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model = malaya.deep_sentiment('bahdanau')
    result = model.predict(positive_text)['attention']
    
    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-sentiment_files/load-sentiment_23_0.png


Visualizing luong model
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model = malaya.deep_sentiment('luong')
    result = model.predict(positive_text)['attention']
    
    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-sentiment_files/load-sentiment_25_0.png


Visualizing hierarchical model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model = malaya.deep_sentiment('hierarchical')
    result = model.predict(positive_text)['attention']
    
    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-sentiment_files/load-sentiment_27_0.png


.. code:: ipython3

    malaya.sentiment.deep_sentiment('hi')


::


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-23-02523991663a> in <module>
    ----> 1 malaya.sentiment.deep_sentiment('hi')
    

    ~/Documents/Malaya/malaya/sentiment.py in deep_sentiment(model)
        251     else:
        252         raise Exception(
    --> 253             'model sentiment not supported, please check supported models from malaya.get_available_sentiment_models()'
        254         )
        255 


    Exception: model sentiment not supported, please check supported models from malaya.get_available_sentiment_models()

