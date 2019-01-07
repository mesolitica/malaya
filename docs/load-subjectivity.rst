
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 10.4 s, sys: 630 ms, total: 11 s
    Wall time: 11 s


Explanation
-----------

Positive subjectivity: based on or influenced by personal feelings,
tastes, or opinions. Can be a positive or negative sentiment.

Negative subjectivity: based on a report or a fact. Can be a positive or
negative sentiment.

.. code:: ipython3

    negative_text = 'Kerajaan negeri Kelantan mempersoalkan motif kenyataan Menteri Kewangan Lim Guan Eng yang hanya menyebut Kelantan penerima terbesar bantuan kewangan dari Kerajaan Persekutuan. Sedangkan menurut Timbalan Menteri Besarnya, Datuk Mohd Amar Nik Abdullah, negeri lain yang lebih maju dari Kelantan turut mendapat pembiayaan dan pinjaman.'
    positive_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'

All models got ``get_proba`` parameters. If True, it will returned
probability every classes. Else, it will return highest probability
class. **Default is False.**

Load multinomial model
----------------------

.. code:: ipython3

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

.. code:: ipython3

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

.. code:: ipython3

    malaya.subjective.available_deep_model()




.. parsed-literal::

    ['fast-text',
     'hierarchical',
     'bahdanau',
     'luong',
     'bidirectional',
     'bert',
     'entity-network']



.. code:: ipython3

    for i in malaya.subjective.available_deep_model():
        print('Testing %s model'%(i))
        model = malaya.subjective.deep_model(i)
        print(model.predict(negative_text))
        print(model.predict_batch([negative_text, positive_text]))
        print(model.predict_batch([negative_text, positive_text],get_proba=True))
        print()


.. parsed-literal::

    Testing fast-text model
    negative
    ['negative', 'negative']
    [{'negative': 0.99998176, 'positive': 1.8284805e-05}, {'negative': 0.6197021, 'positive': 0.3802979}]
    
    Testing hierarchical model
    negative
    ['negative', 'positive']
    [{'negative': 0.9999926, 'positive': 7.3815904e-06}, {'negative': 0.17152506, 'positive': 0.82847494}]
    
    Testing bahdanau model
    negative
    ['negative', 'negative']
    [{'negative': 0.99988747, 'positive': 0.000112509806}, {'negative': 0.41513687, 'positive': 0.5848631}]
    
    Testing luong model
    positive
    ['positive', 'negative']
    [{'negative': 0.03757865, 'positive': 0.96242136}, {'negative': 0.3540184, 'positive': 0.64598167}]
    
    Testing bidirectional model
    negative
    ['negative', 'negative']
    [{'negative': 0.99999166, 'positive': 8.3508085e-06}, {'negative': 0.9939621, 'positive': 0.0060379254}]
    
    Testing bert model
    negative
    ['negative', 'negative']
    [{'negative': 0.98487025, 'positive': 0.015129704}, {'negative': 0.98668575, 'positive': 0.013314218}]
    
    Testing entity-network model
    negative
    ['negative', 'negative']
    [{'negative': 0.6470482, 'positive': 0.35295185}, {'negative': 0.65467215, 'positive': 0.34532788}]
    


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

    model = malaya.subjective.deep_model('bahdanau')
    result = model.predict(negative_text, get_proba=True)['attention']
    
    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-subjectivity_files/load-subjectivity_15_0.png


Visualizing luong model
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model = malaya.subjective.deep_model('luong')
    result = model.predict(negative_text, get_proba=True)['attention']
    
    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-subjectivity_files/load-subjectivity_17_0.png


Visualizing hierarchical model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model = malaya.subjective.deep_model('hierarchical')
    result = model.predict(negative_text, get_proba=True)['attention']
    
    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-subjectivity_files/load-subjectivity_19_0.png


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

    malaya.subjective.available_sparse_deep_model()




.. parsed-literal::

    ['fast-text-char']



Right now Malaya only provide 1 sparse model, ``fast-text-char``. We
will try to evolve it.

.. code:: ipython3

    sparse_model = malaya.subjective.sparse_deep_model()


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from /Users/huseinzol/Malaya/subjective/fast-text-char/model.ckpt


.. code:: ipython3

    sparse_model.predict(positive_text)




.. parsed-literal::

    'positive'



.. code:: ipython3

    sparse_model.predict_batch([positive_text, negative_text])




.. parsed-literal::

    ['positive', 'negative']



.. code:: ipython3

    sparse_model.predict_batch([positive_text, negative_text], get_proba=True)




.. parsed-literal::

    [{'negative': 0.06594214, 'positive': 0.93405783},
     {'negative': 0.9535811, 'positive': 0.046418883}]


