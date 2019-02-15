
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.1 s, sys: 1.33 s, total: 13.5 s
    Wall time: 16.7 s


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
    [{'negative': 0.9479493, 'positive': 0.052050766}, {'negative': 0.027757995, 'positive': 0.972242}]
    
    Testing bahdanau model
    negative
    ['negative', 'positive']
    [{'negative': 0.9993224, 'positive': 0.00067757705}, {'negative': 0.11770468, 'positive': 0.8822953}]
    
    Testing luong model
    negative
    ['negative', 'positive']
    [{'negative': 0.9428673, 'positive': 0.057132788}, {'negative': 0.009072777, 'positive': 0.9909272}]
    
    Testing bidirectional model
    negative
    ['negative', 'positive']
    [{'negative': 0.993769, 'positive': 0.006230943}, {'negative': 0.0032205353, 'positive': 0.9967795}]
    
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
    
       Negative       0.00      0.00      0.00        11
        Neutral       1.00      0.10      0.18        20
       Positive       0.51      1.00      0.67        30
    
    avg / total       0.58      0.52      0.39        61
    


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
    
         adidas       0.94      0.61      0.74       313
          apple       0.98      0.62      0.76       416
         hungry       0.82      0.92      0.87      1082
       kerajaan       0.85      0.82      0.83      1392
           nike       0.95      0.55      0.70       351
    pembangkang       0.71      0.86      0.78      1505
    
    avg / total       0.82      0.80      0.80      5059
    


.. code:: ipython3

    bayes.predict('saya suka kerajaan dan anwar ibrahim', get_proba = True)




.. parsed-literal::

    {'adidas': 0.000975603010411438,
     'apple': 0.0016314798511805502,
     'hungry': 0.012531351707188306,
     'kerajaan': 0.05689689670169819,
     'nike': 0.001125104249853524,
     'pembangkang': 0.9268395644796663}



Train a multinomial using skip-gram vectorization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    bayes = malaya.sentiment.train_multinomial(
        'tests/local', vector = 'skip-gram', ngram_range = (1, 3), skip = 5
    )


.. parsed-literal::

                 precision    recall  f1-score   support
    
         adidas       0.39      0.84      0.53       317
          apple       0.53      0.93      0.67       480
         hungry       0.83      0.94      0.88      1048
       kerajaan       0.90      0.60      0.72      1345
           nike       0.58      0.83      0.68       334
    pembangkang       0.88      0.56      0.69      1535
    
    avg / total       0.79      0.72      0.72      5059
    


.. code:: ipython3

    bayes.predict('saya suka kerajaan dan anwar ibrahim', get_proba = True)




.. parsed-literal::

    {'adidas': 1.5075646514607906e-12,
     'apple': 3.3125444042691963e-12,
     'hungry': 7.508118534366736e-10,
     'kerajaan': 4.03710500865345e-07,
     'nike': 2.5065260905002984e-12,
     'pembangkang': 0.9999995955313602}


