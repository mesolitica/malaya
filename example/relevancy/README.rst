
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 6.45 s, sys: 1.46 s, total: 7.91 s
    Wall time: 11.9 s


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

List available Transformer models
---------------------------------

.. code:: ipython3

    malaya.relevancy.available_transformer_model()




.. parsed-literal::

    {'bert': ['base'], 'xlnet': ['base'], 'albert': ['base']}



Make sure you can check accuracy chart from here first before select a
model, https://malaya.readthedocs.io/en/latest/Accuracy.html#relevancy

**You might want to use ALBERT, a very small size, 43MB, but the
accuracy is still on the top notch.**

Load ALBERT model
-----------------

.. code:: ipython3

    model = malaya.relevancy.transformer(model = 'albert', size = 'base')


.. parsed-literal::

    WARNING: Logging before flag parsing goes to stderr.
    W1017 22:41:08.638995 4600153536 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:68: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    W1017 22:41:08.640387 4600153536 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:69: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    W1017 22:41:11.770990 4600153536 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:64: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


Predict single string
^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model.predict(negative_text,get_proba=True)




.. parsed-literal::

    {'negative': 0.9698777, 'positive': 0.03012232}



Predict batch of strings
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model.predict_batch([negative_text, positive_text],get_proba=True)




.. parsed-literal::

    [{'negative': 0.9202206, 'positive': 0.079779394},
     {'negative': 1.6046162e-05, 'positive': 0.9999839}]



Open relevancy visualization dashboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Default when you call ``predict_words`` it will open a browser with
visualization dashboard, you can disable by ``visualization=False``.

.. code:: ipython3

    model.predict_words(negative_text)

.. code:: ipython3

    from IPython.core.display import Image, display
    
    display(Image('relevancy-negative.png', width=800))



.. image:: load-relevancy_files/load-relevancy_14_0.png
   :width: 800px


Stacking models
---------------

More information, you can read at
https://malaya.readthedocs.io/en/latest/Stack.html

.. code:: ipython3

    bert = malaya.relevancy.transformer(model = 'bert', size = 'base')

.. code:: ipython3

    malaya.stack.predict_stack([bert, model], [positive_text, negative_text])




.. parsed-literal::

    [{'negative': 5.5796554e-05, 'positive': 0.9998949},
     {'negative': 0.9592692, 'positive': 0.0014194043}]


