
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 13.4 s, sys: 1.49 s, total: 14.9 s
    Wall time: 18.6 s


.. code:: ipython3

    positive_text = 'Kerajaan negeri Kelantan mempersoalkan motif kenyataan Menteri Kewangan Lim Guan Eng yang hanya menyebut Kelantan penerima terbesar bantuan kewangan dari Kerajaan Persekutuan. Sedangkan menurut Timbalan Menteri Besarnya, Datuk Mohd Amar Nik Abdullah, negeri lain yang lebih maju dari Kelantan turut mendapat pembiayaan dan pinjaman.'
    negative_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'

All models got ``get_proba`` parameters. If True, it will returned
probability every classes. Else, it will return highest probability
class. **Default is False.**

Load multinomial model
----------------------

.. code:: ipython3

    model = malaya.sentiment.multinomial()
    print(model.predict(positive_text,get_proba=True))
    print(model.predict(negative_text,get_proba=True))
    model.predict_batch([negative_text,negative_text],get_proba=True)


.. parsed-literal::

    {'negative': 0.2422829560944563, 'positive': 0.7577170439055456}
    {'negative': 0.7385102541701198, 'positive': 0.26148974582987783}




.. parsed-literal::

    [{'negative': 0.7385102541701198, 'positive': 0.26148974582987783},
     {'negative': 0.7385102541701198, 'positive': 0.26148974582987783}]



Load xgb model
--------------

.. code:: ipython3

    model = malaya.sentiment.xgb()
    print(model.predict(positive_text,get_proba=True))
    print(model.predict(negative_text,get_proba=True))
    model.predict_batch([negative_text,negative_text],get_proba=True)


.. parsed-literal::

    {'negative': 0.24086821, 'positive': 0.7591318}
    {'negative': 0.844284, 'positive': 0.15571605}




.. parsed-literal::

    [{'negative': 0.844284, 'positive': 0.15571605},
     {'negative': 0.844284, 'positive': 0.15571605}]



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
        print(model.predict_batch([negative_text, positive_text],get_proba=True))
        print()


.. parsed-literal::

    Testing fast-text model
    negative
    ['negative', 'positive']
    [{'negative': 0.8405276, 'positive': 0.15947239}, {'negative': 1.8619101e-05, 'positive': 0.9999814}]
    
    Testing hierarchical model
    negative
    ['negative', 'positive']
    [{'negative': 0.94964343, 'positive': 0.050356578}, {'negative': 0.018990694, 'positive': 0.98100936}]
    
    Testing bahdanau model
    negative
    ['negative', 'positive']
    [{'negative': 0.99957806, 'positive': 0.0004218969}, {'negative': 0.06250752, 'positive': 0.9374925}]
    
    Testing luong model
    negative
    ['negative', 'positive']
    [{'negative': 0.8572259, 'positive': 0.14277408}, {'negative': 0.010074314, 'positive': 0.9899256}]
    
    Testing bidirectional model
    negative
    ['negative', 'positive']
    [{'negative': 0.99718577, 'positive': 0.0028142664}, {'negative': 0.0021885026, 'positive': 0.9978115}]
    
    Testing bert model
    positive
    ['positive', 'negative']
    [{'negative': 0.37042966, 'positive': 0.62957036}, {'negative': 0.84760416, 'positive': 0.15239581}]
    
    Testing entity-network model
    positive
    ['positive', 'positive']
    [{'negative': 0.44306344, 'positive': 0.55693656}, {'negative': 0.32117522, 'positive': 0.6788247}]
    


Unsupervised important words learning
-------------------------------------

.. code:: ipython3

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set() # i just really like seaborn colors

We need to set ``get_proba`` become True to get the ‘attention’.

Visualizing bahdanau model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model = malaya.sentiment.deep_model('bahdanau')
    result = model.predict(positive_text,get_proba=True)['attention']
    
    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-sentiment_files/load-sentiment_15_0.png


Visualizing luong model
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model = malaya.sentiment.deep_model('luong')
    result = model.predict(positive_text,get_proba=True)['attention']
    
    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-sentiment_files/load-sentiment_17_0.png


Visualizing hierarchical model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model = malaya.sentiment.deep_model('hierarchical')
    result = model.predict(positive_text,get_proba=True)['attention']
    
    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-sentiment_files/load-sentiment_19_0.png


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

    'positive'



.. code:: ipython3

    sparse_model.predict_batch([positive_text, negative_text])




.. parsed-literal::

    ['positive', 'negative']



.. code:: ipython3

    sparse_model.predict_batch([positive_text, negative_text],get_proba=True)




.. parsed-literal::

    [{'negative': 0.41368636, 'positive': 0.58631366},
     {'negative': 0.6855174, 'positive': 0.31448266}]



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
    
       Negative       0.00      0.00      0.00        14
        Neutral       1.00      0.16      0.27        19
       Positive       0.48      1.00      0.65        28
    
    avg / total       0.53      0.51      0.38        61
    


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
    
         adidas       0.96      0.60      0.73       307
          apple       0.98      0.61      0.75       452
         hungry       0.81      0.92      0.86      1080
       kerajaan       0.85      0.84      0.85      1368
           nike       0.97      0.59      0.73       337
    pembangkang       0.71      0.86      0.78      1515
    
    avg / total       0.83      0.81      0.81      5059
    


.. code:: ipython3

    bayes.predict('saya suka kerajaan dan anwar ibrahim', get_proba = True)




.. parsed-literal::

    {'adidas': 0.0009457320283650909,
     'apple': 0.0015201576315820337,
     'hungry': 0.010995398743342504,
     'kerajaan': 0.052672712669506426,
     'nike': 0.0010803916911456771,
     'pembangkang': 0.9327856072360596}



Train a multinomial using skip-gram vectorization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    bayes = malaya.sentiment.train_multinomial(
        'tests/local', vector = 'skip-gram', ngram_range = (1, 3), skip = 5
    )


.. parsed-literal::

                 precision    recall  f1-score   support
    
         adidas       0.39      0.85      0.54       327
          apple       0.53      0.90      0.67       485
         hungry       0.82      0.92      0.87      1006
       kerajaan       0.90      0.62      0.73      1440
           nike       0.55      0.79      0.65       311
    pembangkang       0.88      0.56      0.68      1490
    
    avg / total       0.79      0.71      0.72      5059
    


.. code:: ipython3

    bayes.predict('saya suka kerajaan dan anwar ibrahim', get_proba = True)




.. parsed-literal::

    {'adidas': 5.862202104675177e-13,
     'apple': 1.2083814498363065e-12,
     'hungry': 5.524552079715613e-10,
     'kerajaan': 3.502009688789598e-08,
     'nike': 9.084892985507942e-13,
     'pembangkang': 0.9999999644247459}


