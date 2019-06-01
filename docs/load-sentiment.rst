
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.8 s, sys: 1.72 s, total: 14.5 s
    Wall time: 19.2 s


.. code:: python

    positive_text = 'Kerajaan negeri Kelantan mempersoalkan motif kenyataan Menteri Kewangan Lim Guan Eng yang hanya menyebut Kelantan penerima terbesar bantuan kewangan dari Kerajaan Persekutuan sebanyak RM50 juta. Sedangkan menurut Timbalan Menteri Besarnya, Datuk Mohd Amar Nik Abdullah, negeri lain yang lebih maju dari Kelantan turut mendapat pembiayaan dan pinjaman.'
    negative_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'

All models have ``get_proba`` parameters. If True, it will returned
probability every classes. Else, it will return highest probability
class. **Default is False.**

All models have ``add_neutral`` parameters. If True, it will add
``neutral`` probability, Else, default probabilities. **Default is
True.**

Load multinomial model
----------------------

.. code:: python

    model = malaya.sentiment.multinomial()
    print(model.predict(positive_text,get_proba=True))
    print(model.predict(negative_text,get_proba=True))
    model.predict_batch([negative_text,negative_text],get_proba=True)


.. parsed-literal::

    {'negative': 0.003559988321312934, 'positive': 0.6440011678687021, 'neutral': 0.352438843809985}
    {'negative': 0.4770205083402397, 'positive': 0.005229794916597557, 'neutral': 0.5177496967431627}




.. parsed-literal::

    [{'negative': 0.4770205083402397,
      'positive': 0.005229794916597557,
      'neutral': 0.5177496967431627},
     {'negative': 0.4770205083402397,
      'positive': 0.005229794916597557,
      'neutral': 0.5177496967431627}]



Disable ``neutral`` probability,

.. code:: python

    print(model.predict(negative_text,get_proba=True,add_neutral=True))
    print(model.predict(negative_text,get_proba=True,add_neutral=False))


.. parsed-literal::

    {'negative': 0.4770205083402397, 'positive': 0.005229794916597557, 'neutral': 0.5177496967431627}
    {'negative': 0.7385102541701198, 'positive': 0.26148974582987783}


Load xgb model
--------------

.. code:: python

    model = malaya.sentiment.xgb()
    print(model.predict(positive_text,get_proba=True))
    print(model.predict(negative_text,get_proba=True))
    model.predict_batch([negative_text,negative_text],get_proba=True)


.. parsed-literal::

    {'negative': 0.0045786616, 'positive': 0.5421338, 'neutral': 0.45328754}
    {'negative': 0.688568, 'positive': 0.0031143208, 'neutral': 0.30831766}




.. parsed-literal::

    [{'negative': 0.688568, 'positive': 0.0031143208, 'neutral': 0.30831766},
     {'negative': 0.688568, 'positive': 0.0031143208, 'neutral': 0.30831766}]



List available deep learning models
-----------------------------------

.. code:: python

    malaya.sentiment.available_deep_model()




.. parsed-literal::

    ['self-attention', 'bahdanau', 'luong']



Load deep learning models
-------------------------

Good thing about deep learning models from Malaya, it returns
``Attention`` result, means, which part of words give the high impact to
the results. But to get ``Attention``, you need to set
``get_proba=True``.

.. code:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

Load bahdanau model
~~~~~~~~~~~~~~~~~~~

.. code:: python

    model = malaya.sentiment.deep_model('bahdanau')

Predict single string
^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    model.predict(positive_text)




.. parsed-literal::

    'neutral'



.. code:: python

    result = model.predict(positive_text,get_proba=True,add_neutral=False)
    result




.. parsed-literal::

    {'negative': 0.29423502,
     'positive': 0.70576495,
     'attention': {'Kerajaan': 0.0019730187,
      'negeri': 0.0016380441,
      'Kelantan': 0.52261657,
      'mempersoalkan': 0.0041695302,
      'motif': 0.009157478,
      'kenyataan': 0.0020427739,
      'Menteri': 0.0026452087,
      'Kewangan': 0.0017612759,
      'Lim': 0.046150286,
      'Guan': 0.046651356,
      'Eng': 0.014238223,
      'yang': 0.0014762171,
      'hanya': 0.0030002387,
      'menyebut': 0.0025070142,
      'penerima': 0.001477954,
      'terbesar': 0.0014683361,
      'bantuan': 0.0020200813,
      'kewangan': 0.0015684298,
      'dari': 0.001558458,
      'Persekutuan': 0.0021011133,
      'sebanyak': 0.001435061,
      'RM50': 0.037767526,
      'juta': 0.0031749196,
      '.': 0.0,
      'Sedangkan': 0.0015534447,
      'menurut': 0.0014812354,
      'Timbalan': 0.0020608688,
      'Besarnya': 0.001435703,
      ',': 0.0,
      'Datuk': 0.0014482451,
      'Mohd': 0.0014422016,
      'Amar': 0.0014641153,
      'Nik': 0.0015784851,
      'Abdullah': 0.0014410047,
      'lain': 0.0016714201,
      'lebih': 0.0037415246,
      'maju': 0.019784313,
      'turut': 0.011382608,
      'mendapat': 0.0025349073,
      'pembiayaan': 0.0020161376,
      'dan': 0.0,
      'pinjaman': 0.009653877}}



.. code:: python

    plt.figure(figsize = (15, 5))
    keys = result['attention'].keys()
    values = result['attention'].values()
    aranged = [i for i in range(len(keys))]
    plt.bar(aranged, values)
    plt.xticks(aranged, keys, rotation = 'vertical')
    plt.show()



.. image:: load-sentiment_files/load-sentiment_18_0.png


Open sentiment visualization dashboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Default when you call ``predict_words`` it will open a browser with
visualization dashboard, you can disable by ``visualization=False``.

.. code:: python

    model.predict_words(positive_text)


.. parsed-literal::

    Serving to http://127.0.0.1:8889/    [Ctrl-C to exit]


.. parsed-literal::

    127.0.0.1 - - [30/May/2019 11:13:59] "GET / HTTP/1.1" 200 -
    127.0.0.1 - - [30/May/2019 11:13:59] "GET /static/admin-materialize.min.css HTTP/1.1" 200 -
    127.0.0.1 - - [30/May/2019 11:13:59] "GET /static/echarts.min.js HTTP/1.1" 200 -


.. parsed-literal::


    stopping Server...


.. code:: python

    from IPython.core.display import Image, display

    display(Image('sentiment-visualization.png', width=800))



.. image:: load-sentiment_files/load-sentiment_21_0.png
   :width: 800px


I tried to put the html and javascript inside a notebook cell, pretty
hard you know and a lot of weird bugs. Let stick to HTTP serving ya.

.. code:: python

    display(Image('sentiment-negative.png', width=800))



.. image:: load-sentiment_files/load-sentiment_23_0.png
   :width: 800px


``predict_words`` only accept a single string. You can’t predict
multiple texts.

Predict batch of strings
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    model.predict_batch([negative_text, positive_text],get_proba=True)




.. parsed-literal::

    [{'negative': 0.94391596, 'positive': 0.000560839, 'neutral': 0.055523217},
     {'negative': 0.004329388, 'positive': 0.5670612, 'neutral': 0.42860943}]



**You might want to try ``luong`` and ``self-attention`` by yourself.**

Stacking models
---------------

More information, you can read at
https://malaya.readthedocs.io/en/latest/Stack.html

.. code:: python

    multinomial = malaya.sentiment.multinomial()
    xgb = malaya.sentiment.xgb()
    bahdanau = malaya.sentiment.deep_model('bahdanau')

.. code:: python

    malaya.stack.predict_stack([multinomial, xgb, bahdanau], positive_text)




.. parsed-literal::

    {'negative': 0.0037063136821626594,
     'positive': 0.6215181632979583,
     'neutral': 0.3669251238766725}



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

.. code:: python

    malaya.sentiment.available_sparse_deep_model()




.. parsed-literal::

    ['fast-text-char']



Right now Malaya only provide 1 sparse model, ``fast-text-char``. We
will try to evolve it.

.. code:: python

    sparse_model = malaya.sentiment.sparse_deep_model()


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from /Users/huseinzol/Malaya/sentiment/fast-text-char/model.ckpt


.. code:: python

    sparse_model.predict(positive_text)




.. parsed-literal::

    'positive'



.. code:: python

    sparse_model.predict_batch([positive_text, negative_text])




.. parsed-literal::

    ['positive', 'negative']



.. code:: python

    sparse_model.predict_batch([positive_text, negative_text],get_proba=True)




.. parsed-literal::

    [{'negative': 0.42412993, 'positive': 0.5758701},
     {'negative': 0.6855174, 'positive': 0.31448266}]



Right now sparse models does not have ``neutral`` class.
