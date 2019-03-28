
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.3 s, sys: 1.5 s, total: 13.8 s
    Wall time: 17.7 s


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
    [{'negative': 0.9504666, 'positive': 0.049533408}, {'negative': 0.041675426, 'positive': 0.9583246}]
    
    Testing bahdanau model
    negative
    ['negative', 'positive']
    [{'negative': 0.9993631, 'positive': 0.0006369345}, {'negative': 0.10564381, 'positive': 0.89435613}]
    
    Testing luong model
    negative
    ['negative', 'positive']
    [{'negative': 0.8851047, 'positive': 0.11489531}, {'negative': 0.0025337301, 'positive': 0.9974663}]
    
    Testing bidirectional model
    negative
    ['negative', 'positive']
    [{'negative': 0.97722447, 'positive': 0.02277552}, {'negative': 0.007992058, 'positive': 0.992008}]
    
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
