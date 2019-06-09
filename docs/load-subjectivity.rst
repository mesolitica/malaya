
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 10.9 s, sys: 920 ms, total: 11.8 s
    Wall time: 12.1 s


Explanation
-----------

Positive subjectivity: based on or influenced by personal feelings,
tastes, or opinions. Can be a positive or negative sentiment.

Negative subjectivity: based on a report or a fact. Can be a positive or
negative sentiment.

.. code:: python

    negative_text = 'Kerajaan negeri Kelantan mempersoalkan motif kenyataan Menteri Kewangan Lim Guan Eng yang hanya menyebut Kelantan penerima terbesar bantuan kewangan dari Kerajaan Persekutuan. Sedangkan menurut Timbalan Menteri Besarnya, Datuk Mohd Amar Nik Abdullah, negeri lain yang lebih maju dari Kelantan turut mendapat pembiayaan dan pinjaman.'
    positive_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'

All models got ``get_proba`` parameters. If True, it will returned
probability every classes. Else, it will return highest probability
class. **Default is False.**

Load multinomial model
----------------------

.. code:: python

    model = malaya.subjective.multinomial()
    print(model.predict(positive_text,get_proba=True))
    print(model.predict(negative_text,get_proba=True))
    model.predict_batch([negative_text,negative_text],get_proba=True)


.. parsed-literal::

    {'negative': 0.009240767162200498, 'positive': 0.0759232837799535, 'neutral': 0.914835949057846}
    {'negative': 0.7214589553228845, 'positive': 0.0027854104467711456, 'neutral': 0.2757556342303443}




.. parsed-literal::

    [{'negative': 0.7214589553228845,
      'positive': 0.0027854104467711456,
      'neutral': 0.2757556342303443},
     {'negative': 0.7214589553228845,
      'positive': 0.0027854104467711456,
      'neutral': 0.2757556342303443}]



Load xgb model
--------------

.. code:: python

    model = malaya.subjective.xgb()
    print(model.predict(positive_text,get_proba=True))
    print(model.predict(negative_text,get_proba=True))
    model.predict_batch([negative_text,negative_text],get_proba=True)


.. parsed-literal::

    {'negative': 0.0085689435, 'positive': 0.14310563, 'neutral': 0.84832543}
    {'negative': 0.84999824, 'positive': 0.0015000176, 'neutral': 0.14850175}




.. parsed-literal::

    [{'negative': 0.84999824, 'positive': 0.0015000176, 'neutral': 0.14850175},
     {'negative': 0.84999824, 'positive': 0.0015000176, 'neutral': 0.14850175}]



List available deep learning models
-----------------------------------

.. code:: python

    malaya.subjective.available_deep_model()




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

    model = malaya.subjective.deep_model('bahdanau')


.. parsed-literal::

    downloading frozen /Users/huseinzol/Malaya/subjective/bahdanau model


.. parsed-literal::

    20.0MB [00:07, 2.85MB/s]
      0%|          | 0.00/0.45 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading frozen /Users/huseinzol/Malaya/subjective/bahdanau setting


.. parsed-literal::

    1.00MB [00:00, 5.75MB/s]


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

    {'negative': 0.3413489,
     'positive': 0.6586511,
     'attention': {'kerajaan': 0.02428512,
      'sebenarnya': 0.05316463,
      'sangat': 0.7279027,
      'bencikan': 0.07460431,
      'rakyatnya': 0.026773913,
      ',': 0.0,
      'minyak': 0.048565686,
      'naik': 0.023328593,
      'dan': 0.0,
      'segalanya': 0.021375034}}



.. code:: python

    plt.figure(figsize = (15, 5))
    keys = result['attention'].keys()
    values = result['attention'].values()
    aranged = [i for i in range(len(keys))]
    plt.bar(aranged, values)
    plt.xticks(aranged, keys, rotation = 'vertical')
    plt.show()



.. image:: load-subjectivity_files/load-subjectivity_17_0.png


Open subjectivity visualization dashboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Default when you call ``predict_words`` it will open a browser with
visualization dashboard, you can disable by ``visualization=False``.

.. code:: python

    model.predict_words(negative_text)


.. parsed-literal::

    Serving to http://127.0.0.1:8889/    [Ctrl-C to exit]


.. parsed-literal::

    127.0.0.1 - - [01/Jun/2019 12:16:49] "GET / HTTP/1.1" 200 -
    127.0.0.1 - - [01/Jun/2019 12:16:49] "GET /static/admin-materialize.min.css HTTP/1.1" 200 -
    127.0.0.1 - - [01/Jun/2019 12:16:49] "GET /static/echarts.min.js HTTP/1.1" 200 -
    127.0.0.1 - - [01/Jun/2019 12:16:49] "GET /favicon.ico HTTP/1.1" 200 -
    ----------------------------------------
    Exception happened during processing of request from ('127.0.0.1', 61989)
    Traceback (most recent call last):
      File "/usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/lib/python3.6/socketserver.py", line 317, in _handle_request_noblock
        self.process_request(request, client_address)
      File "/usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/lib/python3.6/socketserver.py", line 348, in process_request
        self.finish_request(request, client_address)
      File "/usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/lib/python3.6/socketserver.py", line 361, in finish_request
        self.RequestHandlerClass(request, client_address, self)
      File "/usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/lib/python3.6/socketserver.py", line 696, in __init__
        self.handle()
      File "/usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/lib/python3.6/http/server.py", line 418, in handle
        self.handle_one_request()
      File "/usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/lib/python3.6/http/server.py", line 406, in handle_one_request
        method()
      File "/Users/huseinzol/Documents/Malaya/malaya/_utils/_server.py", line 32, in do_GET
        with open(filepath, 'rb') as fh:
    FileNotFoundError: [Errno 2] No such file or directory: '/Users/huseinzol/Documents/Malaya/malaya/_utils/web/favicon.ico'
    ----------------------------------------


.. parsed-literal::


    stopping Server...


.. code:: python

    from IPython.core.display import Image, display

    display(Image('subjective-bahdanau.png', width=800))



.. image:: load-subjectivity_files/load-subjectivity_20_0.png
   :width: 800px


I tried to put the html and javascript inside a notebook cell, pretty
hard you know and a lot of weird bugs. Let stick to HTTP serving ya.

``predict_words`` only accept a single string. You can’t predict
multiple texts.

Predict batch of strings
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    model.predict_batch([negative_text, positive_text],get_proba=True)




.. parsed-literal::

    [{'negative': 0.83364284, 'positive': 0.0016635716, 'neutral': 0.1646936},
     {'negative': 0.003325577, 'positive': 0.6674423, 'neutral': 0.3292321}]



**You might want to try ``luong`` and ``self-attention`` by yourself.**

BERT model
----------

BERT is the best subjectivity model in term of accuracy, you can check
subjectivity accuracy here,
https://malaya.readthedocs.io/en/latest/Accuracy.html#subjectivity-analysis.
But warning, the model size is 700MB! Make sure you have enough
resources to use BERT, and installed ``bert-tensorflow`` first,

.. code:: bash

   pip3 install bert-tensorflow

.. code:: python

    model = malaya.subjective.bert()
    model.predict_batch([negative_text, positive_text],get_proba=True)


.. parsed-literal::

    Found old version of /Users/huseinzol/Malaya/subjective/bert, deleting..
    Done.
    downloading frozen /Users/huseinzol/Malaya/subjective/bert model


.. parsed-literal::

    679MB [03:17, 4.01MB/s]




.. parsed-literal::

    [{'negative': 0.9999628, 'positive': 3.7092312e-07, 'neutral': 3.683567e-05},
     {'negative': 0.99188435, 'positive': 8.11561e-05, 'neutral': 0.008034468}]



Stacking models
---------------

More information, you can read at
https://malaya.readthedocs.io/en/latest/Stack.html

.. code:: python

    multinomial = malaya.subjective.multinomial()
    xgb = malaya.subjective.xgb()
    bahdanau = malaya.subjective.deep_model('bahdanau')

.. code:: python

    malaya.stack.predict_stack([multinomial, xgb, bahdanau], positive_text)




.. parsed-literal::

    {'negative': 0.008627402242055781,
     'positive': 0.12711225500695544,
     'neutral': 0.8541128287159148}



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

    malaya.subjective.available_sparse_deep_model()




.. parsed-literal::

    ['fast-text-char']



Right now Malaya only provide 1 sparse model, ``fast-text-char``. We
will try to evolve it.

.. code:: python

    sparse_model = malaya.subjective.sparse_deep_model()


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from /Users/huseinzol/Malaya/subjective/fast-text-char/model.ckpt


.. code:: python

    sparse_model.predict(positive_text)




.. parsed-literal::

    'positive'



.. code:: python

    sparse_model.predict_batch([positive_text, negative_text])




.. parsed-literal::

    ['positive', 'negative']



.. code:: python

    sparse_model.predict_batch([positive_text, negative_text], get_proba=True)




.. parsed-literal::

    [{'negative': 0.054842573, 'positive': 0.94515747},
     {'negative': 0.95071983, 'positive': 0.04928014}]



Right now sparse models does not have ``neutral`` class.
