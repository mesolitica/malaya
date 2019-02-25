
Why fast-text?
--------------

FastText (which is essentially an extension of word2vec model), treats
each word as composed of character ngrams. So the vector for a word is
made of the sum of this character n grams.

N-gram feature is the most significant improvement in FastText, it’s
designed to solve OOV(Out-of-Vocabulary) issue.

For example, the word ``aquarium`` can be split into
``<aq/aqu/qua/uar/ari/riu/ium/um>``, ``<`` and ``>`` means SOW and EOW.

As Word Embedder encounter word ``aquarius``, it might not recognize it,
but it can guess by the share part in ``aquarium`` and ``aquarius``, to
embed aquarius near aquarium

Pretrained fast-text
--------------------

You can download Malaya pretrained without need to import malaya.

word2vec from wikipedia
^^^^^^^^^^^^^^^^^^^^^^^

`size-1024 <https://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/v16/fasttext/fasttext-wiki-1024.p>`__

You will get a pickle file, contains ``['nce_weights', 'dictionary']``.

To load that pickle file,

.. code:: python

   import pickle
   with open('file.p', 'rb') as fopen:
       word2vec = pickle.load(fopen)

But If you don’t know what to do with malaya fast-text, Malaya provided
some useful functions for you!

.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 11.5 s, sys: 1.44 s, total: 13 s
    Wall time: 16.5 s


Load malaya wikipedia fast-text
-------------------------------

.. code:: ipython3

    wiki, ngrams = malaya.fast_text.load_wiki()

Load fast-text interface
------------------------

**But problem with fast-text, dictionary only have ngrams words, we need
to provide actual words to compare semantic similarity**

.. code:: ipython3

    fast_text_nce = malaya.fast_text.fast_text(wiki['nce_weights'],wiki['dictionary'],ngrams)
    fast_text_embed = malaya.fast_text.fast_text(wiki['embed_weights'],wiki['dictionary'],ngrams)

Check top-k similar semantics based on a word
---------------------------------------------

.. code:: ipython3

    fast_text_nce.n_closest('najib',['najib razak','mahathir','1mdb','rosmah',
                                 'kerajaan','penjara','cerdik','kipas','husein'])




.. parsed-literal::

    [['najib razak', 0.9890886545181274],
     ['husein', 0.9856827259063721],
     ['mahathir', 0.9842776656150818],
     ['kipas', 0.9831240177154541],
     ['rosmah', 0.9758304357528687]]



.. code:: ipython3

    fast_text_embed.n_closest('najib',['najib razak','mahathir','1mdb','rosmah',
                                 'kerajaan','penjara','cerdik','kipas','husein'])




.. parsed-literal::

    [['najib razak', 0.6421577334403992],
     ['mahathir', 0.12914645671844482],
     ['1mdb', 0.0642591118812561],
     ['rosmah', 0.04841345548629761],
     ['kerajaan', 0.03990912437438965]]



Based on the results, ``embed_weights`` learnt better than
``nce_weights``. It is really depends during training session.

.. code:: ipython3

    fast_text_embed.n_closest('mkn',['makan','mukun','makin','gejala','mahathir'])




.. parsed-literal::

    [['mukun', 0.4251323938369751],
     ['makan', 0.3479733467102051],
     ['makin', 0.33818352222442627],
     ['mahathir', 0.24181222915649414],
     ['gejala', -0.0475466251373291]]



.. code:: ipython3

    fast_text_embed.n_closest('mkn',['makan','mukun','makin','gejala','mahathir'],return_similarity = False)




.. parsed-literal::

    ['mukun', 'makan', 'makin', 'mahathir', 'gejala']



Calculate vb - va + vc
----------------------

.. code:: ipython3

    fast_text_embed.analogy('makan','kfc','mikin',
                            ['makan','mukun','makin','gejala','mahathir'])




.. parsed-literal::

    ['mikin']



Fast-text calculator
--------------------

You can put any equation you wanted.

.. code:: ipython3

    fast_text_embed.calculator('anwar + amerika + mahathir',
                               ['makan','mukun','makin','gejala','mahathir'],
                               return_similarity = False)




.. parsed-literal::

    ['mahathir', 'makan', 'makin', 'gejala', 'mukun']



.. code:: ipython3

    fast_text_embed.calculator('(anwar + amerika) / mahathir',
                               ['makan','mukun','makin','gejala','mahathir'],
                               return_similarity = True)




.. parsed-literal::

    [['makan', 0.11451129709969454],
     ['gejala', 0.11409430208988958],
     ['mahathir', 0.070956003373157],
     ['makin', 0.00728455196402511],
     ['mukun', -0.012139292144108138]]



Visualize scatter-plot
----------------------

.. code:: ipython3

    result = fast_text_embed.n_closest('mkn',['makan','mukun','makin','gejala','mahathir'])
    fast_text_embed.scatter_plot(result, centre = 'mkn', notebook_mode = True)



.. parsed-literal::

    <Figure size 700x700 with 1 Axes>


Visualize tree-plot
-------------------

.. code:: ipython3

    fast_text_embed.tree_plot(result, notebook_mode = True)



.. parsed-literal::

    <Figure size 504x504 with 0 Axes>



.. image:: load-fast-text_files/load-fast-text_22_1.png


Train on custom corpus
----------------------

.. code:: ipython3

    isu_kerajaan = ['Institusi raja khususnya Yang di-Pertuan Agong adalah kedaulatan negara dengan kedudukan dan peranannya termaktub dalam Perlembagaan Persekutuan yang perlu disokong dan didukung oleh kerajaan serta rakyat.',
                   'Pensyarah Kulliyah Undang-Undang Ahmad Ibrahim, Universiti Islam Antarabangsa Malaysia (UIAM) Prof Madya Dr Shamrahayu Ab Aziz berkata perubahan kerajaan, susulan kemenangan Pakatan Harapan pada Pilihan Raya Umum Ke-14 pada Mei lepas, tidak memberi kesan dari segi peranan, fungsi dan kedudukan Yang di-Pertuan Agong.',
                   'Peralihan kerajaan itu menyaksikan Sultan Muhammad V mencatat sejarah tersendiri dengan menjadi Yang di-Pertuan Agong Malaysia yang pertama memerintah dalam era dua kerajaan berbeza.',
                   'Semasa dilantik sebagai Yang di-Pertuan Agong Ke-15 pada 13 Dis 2016, kerajaan ketika itu diterajui oleh Barisan Nasional dan pada 10 Mei lepas, kepimpinan negara diambil alih oleh Pakatan Harapan yang memenangi Pilihan Raya Umum Ke-14.',
                   'Ketika merasmikan Istiadat Pembukaan Penggal Pertama, Parlimen ke-14 pada 17 Julai lepas, Seri Paduka bertitah mengalu-alukan pendekatan kerajaan Pakatan Harapan dalam menegakkan ketelusan terutamanya dengan mendedahkan kedudukan kewangan negara yang sebenar serta mengkaji semula perbelanjaan, kos projek dan mengurus kewangan secara berhemat bagi menangani kos sara hidup.',
                   'Pada Jun lepas, Sultan Muhammad V memperkenankan supaya peruntukan gaji dan emolumen Yang di-Pertuan Agong dikurangkan sebanyak 10 peratus sepanjang pemerintahan sehingga 2021 berikutan keprihatinan Seri Paduka terhadap tahap hutang dan keadaan ekonomi negara.',
                   'Seri Paduka turut menitahkan supaya Majlis Rumah Terbuka Aidilfitri tahun ini tidak diadakan di Istana Negara dengan peruntukan majlis itu digunakan bagi membantu golongan yang kurang bernasib baik.']

.. code:: ipython3

    embed_weights, nce_weights, dictionary, ngrams = malaya.fast_text.train(isu_kerajaan, ngrams= (3,4))


.. parsed-literal::

    WARNING:tensorflow:From /usr/local/lib/python3.6/site-packages/tensorflow/python/ops/nn_impl.py:1124: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.


.. parsed-literal::

    train minibatch loop:  25%|██▌       | 2/8 [00:00<00:00, 17.00it/s, cost=31.4]

.. parsed-literal::

    model built, vocab size 162, document length 239


.. parsed-literal::

    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 28.86it/s, cost=33.2]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 26.27it/s, cost=28.8]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 30.99it/s, cost=30.2]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 72.80it/s, cost=21.5]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 34.06it/s, cost=18]  
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 99.32it/s, cost=16.6]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 31.45it/s, cost=4.56]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 52.21it/s, cost=10.6]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 34.21it/s, cost=8.77]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 72.56it/s, cost=11.3]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 33.21it/s, cost=8.4] 
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 77.31it/s, cost=14.4]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 32.58it/s, cost=7.85]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 83.67it/s, cost=6.94]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 35.13it/s, cost=13.1]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 104.69it/s, cost=4.54]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 34.91it/s, cost=4.55]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 123.10it/s, cost=9.08]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 34.58it/s, cost=6.03]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 86.25it/s, cost=7.39]


.. code:: ipython3

    fast_text_embed = malaya.fast_text.fast_text(nce_weights,dictionary,ngrams)

.. code:: ipython3

    word = 'paduka'
    print(fast_text_embed.n_closest(word, ['raja','agong','universiti','mahathir',
                                            'najib','husein','malaysia','pertama','sultan'],
                                     num_closest=8, metric='cosine'))


.. parsed-literal::

    [['raja', 0.8651032447814941], ['pertama', 0.8305785655975342], ['sultan', 0.7715829610824585], ['malaysia', 0.7260801792144775], ['mahathir', 0.7086304426193237], ['husein', 0.6739279627799988], ['universiti', 0.673927903175354], ['najib', 0.673927903175354]]

