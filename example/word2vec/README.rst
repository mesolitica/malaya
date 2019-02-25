
Pretrained word2vec
-------------------

You can download Malaya pretrained without need to import malaya.

word2vec from news
^^^^^^^^^^^^^^^^^^

`size-32 <https://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/v7/word2vec/word2vec-32.p>`__

`size-64 <https://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/v7/word2vec/word2vec-64.p>`__

`size-128 <https://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/v7/word2vec/word2vec-128.p>`__

`size-256 <https://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/v7/word2vec/word2vec-256.p>`__

`size-512 <https://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/v7/word2vec/word2vec-512.p>`__

word2vec from wikipedia
^^^^^^^^^^^^^^^^^^^^^^^

`size-256 <https://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/v13/word2vec/word2vec-wiki-nce-256.p>`__

You will get a pickle file, contains ``['nce_weights', 'dictionary']``.

To load that pickle file,

.. code:: python

   import pickle
   with open('file.p', 'rb') as fopen:
       word2vec = pickle.load(fopen)

But If you don’t know what to do with malaya word2vec, Malaya provided
some useful functions for you!

.. code:: ipython3

    %%time
    import malaya
    %matplotlib inline


.. parsed-literal::

    CPU times: user 12.1 s, sys: 1.52 s, total: 13.6 s
    Wall time: 17.3 s


Load malaya news word2vec
-------------------------

.. code:: ipython3

    embedded_news = malaya.word2vec.load_news(256)

Load malaya wikipedia word2vec
------------------------------

.. code:: ipython3

    embedded_wiki = malaya.word2vec.load_wiki()

Load word2vec interface
-----------------------

.. code:: ipython3

    word_vector_news = malaya.word2vec.word2vec(embedded_news['nce_weights'], embedded_news['dictionary'])
    word_vector_wiki = malaya.word2vec.word2vec(embedded_wiki['nce_weights'], embedded_wiki['dictionary'])

Check top-k similar semantics based on a word
---------------------------------------------

.. code:: ipython3

    word = 'anwar'
    print("Embedding layer: 8 closest words to: '%s'"%(word))
    print(word_vector_news.n_closest(word=word, num_closest=8, metric='cosine'))


.. parsed-literal::

    Embedding layer: 8 closest words to: 'anwar'
    [['mahathir', 0.44774019718170166], ['beliau', 0.44170427322387695], ['zaid', 0.43993180990219116], ['hishammuddin', 0.4343132972717285], ['kuok', 0.43307822942733765], ['husam', 0.43213725090026855], ['anifah', 0.4307258129119873], ['pesakit', 0.4262162446975708]]


.. code:: ipython3

    word = 'anwar'
    print("Embedding layer: 8 closest words to: '%s'"%(word))
    print(word_vector_wiki.n_closest(word=word, num_closest=8, metric='cosine'))


.. parsed-literal::

    Embedding layer: 8 closest words to: 'anwar'
    [['zaid', 0.7285637855529785], ['khairy', 0.6839416027069092], ['zabidi', 0.6709405183792114], ['nizar', 0.6695379018783569], ['harussani', 0.6595045328140259], ['shahidan', 0.6565827131271362], ['azalina', 0.6541041135787964], ['shahrizat', 0.6538639068603516]]


Check batch top-k similar semantics based on a word
---------------------------------------------------

.. code:: ipython3

    words = ['anwar', 'mahathir']
    word_vector_news.batch_n_closest(words, num_closest=8,
                                     return_similarity=False)




.. parsed-literal::

    [['anwar',
      'mahathir',
      'beliau',
      'zaid',
      'hishammuddin',
      'kuok',
      'husam',
      'anifah'],
     ['mahathir',
      'najib',
      'obama',
      'subramaniam',
      'anwar',
      'zamihan',
      'mujahid',
      'dzulkefly']]



What happen if a word not in the dictionary?

You can set parameter ``soft`` to ``True`` or ``False``. Default is
``True``.

if ``True``, a word not in the dictionary will be replaced with nearest
fuzzywuzzy ratio.

if ``False``, it will throw an exception if a word not in the
dictionary.

.. code:: ipython3

    words = ['anwar', 'mahathir','husein-comel']
    word_vector_news.batch_n_closest(words, num_closest=8,
                                     return_similarity=False,soft=False)


::


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-8-4be8160131f5> in <module>
          1 words = ['anwar', 'mahathir','husein-comel']
          2 word_vector_news.batch_n_closest(words, num_closest=8,
    ----> 3                                  return_similarity=False,soft=False)
    

    ~/Documents/Malaya/malaya/word2vec.py in batch_n_closest(self, words, num_closest, return_similarity, soft)
        628                     raise Exception(
        629                         '%s not in dictionary, please use another word or set `soft` = True'
    --> 630                         % (words[i])
        631                     )
        632         batches = np.array([self.get_vector_by_name(w) for w in words])


    Exception: husein-comel not in dictionary, please use another word or set `soft` = True


.. code:: ipython3

    words = ['anwar', 'mahathir','husein-comel']
    word_vector_news.batch_n_closest(words, num_closest=8,
                                     return_similarity=False,soft=True)




.. parsed-literal::

    [['anwar',
      'mahathir',
      'beliau',
      'zaid',
      'hishammuddin',
      'kuok',
      'husam',
      'anifah'],
     ['mahathir',
      'najib',
      'obama',
      'subramaniam',
      'anwar',
      'zamihan',
      'mujahid',
      'dzulkefly'],
     ['income',
      'wishes',
      'styles',
      'devices',
      'holographic',
      'proper',
      'refined',
      'moves']]



Calculate vb - va + vc
----------------------

.. code:: ipython3

    print(word_vector_news.analogy('anwar', 'penjara', 'kerajaan', 5))


.. parsed-literal::

    ['penjara', 'kerajaan', 'kkm', 'kabinet', 'tuju']


.. code:: ipython3

    print(word_vector_wiki.analogy('anwar', 'penjara', 'kerajaan', 5))


.. parsed-literal::

    ['penjara', 'kerajaan', 'kemaharajaan', 'pemerintah', 'pelabuhan']


Word2vec calculator
-------------------

You can put any equation you wanted.

.. code:: ipython3

    word_vector_news.calculator('anwar + amerika + mahathir', num_closest=8, metric='cosine',
                          return_similarity=False)




.. parsed-literal::

    ['mahathir',
     'anwar',
     'amerika',
     'UNK',
     'najib',
     'husam',
     'kuok',
     'azalina',
     'mujahid']



.. code:: ipython3

    word_vector_wiki.calculator('anwar + amerika + mahathir', num_closest=8, metric='cosine',
                          return_similarity=False)




.. parsed-literal::

    ['anwar',
     'mahathir',
     'hishammuddin',
     'sukarno',
     'khairy',
     'suffian',
     'ahmadinejad',
     'davutoglu',
     'shahrizat']



Word2vec batch calculator
-------------------------

We can use Tensorflow computational graph to do distributed
nearest-neighbors for us for fast suggestion!

.. code:: ipython3

    equations = ['mahathir + najib'] * 10 # assumed we have 10 different equations

.. code:: ipython3

    %%time
    for eq in equations:
        word_vector_wiki.calculator(eq, num_closest=8, metric='cosine',return_similarity=False)


.. parsed-literal::

    CPU times: user 1min 29s, sys: 7.02 s, total: 1min 36s
    Wall time: 1min 37s


.. code:: ipython3

    %%time
    results = word_vector_wiki.batch_calculator(equations,num_closest=8)


.. parsed-literal::

    CPU times: user 1min 33s, sys: 3.24 s, total: 1min 36s
    Wall time: 1min 26s


If you are using GPU, or many cores, this will definitely speed up this
process.

Visualize scatter-plot
----------------------

.. code:: ipython3

    word = 'anwar'
    result = word_vector_news.n_closest(word=word, num_closest=8, metric='cosine')
    word_vector_news.scatter_plot(result, centre = word, notebook_mode = True)



.. image:: load-word2vec_files/load-word2vec_29_0.png


.. code:: ipython3

    word = 'anwar'
    result = word_vector_wiki.n_closest(word=word, num_closest=8, metric='cosine')
    word_vector_wiki.scatter_plot(result, centre = word, notebook_mode = True)



.. image:: load-word2vec_files/load-word2vec_30_0.png


Visualize tree-plot
-------------------

.. code:: ipython3

    word = 'anwar'
    result = word_vector_news.n_closest(word=word, num_closest=8, metric='cosine')
    word_vector_news.tree_plot(result, notebook_mode = True)



.. parsed-literal::

    <Figure size 504x504 with 0 Axes>



.. image:: load-word2vec_files/load-word2vec_32_1.png


.. code:: ipython3

    word = 'anwar'
    result = word_vector_wiki.n_closest(word=word, num_closest=8, metric='cosine')
    word_vector_wiki.tree_plot(result, notebook_mode = True)



.. parsed-literal::

    <Figure size 504x504 with 0 Axes>



.. image:: load-word2vec_files/load-word2vec_33_1.png


Get embedding from a word
-------------------------

If a word not found in the vocabulary, it will throw an exception with
top-5 nearest words

.. code:: ipython3

    word_vector_news.get_vector_by_name('husein-comel')


::


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-21-b4f84915c530> in <module>
    ----> 1 word_vector_news.get_vector_by_name('husein-comel')
    

    ~/Documents/Malaya/malaya/word2vec.py in get_vector_by_name(self, word)
        292             raise Exception(
        293                 'input not found in dictionary, here top-5 nearest words [%s]'
    --> 294                 % (strings)
        295             )
        296         return np.ravel(self._embed_matrix[self._dictionary[word], :])


    Exception: input not found in dictionary, here top-5 nearest words [income, husein, incomes, hussein, husseiny]


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

    embed_weights, nce_weights, dictionary = malaya.word2vec.train(isu_kerajaan)


.. parsed-literal::

    WARNING:tensorflow:From /usr/local/lib/python3.6/site-packages/tensorflow/python/ops/nn_impl.py:1124: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.


.. parsed-literal::

    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 53.59it/s, cost=37]  
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 37.88it/s, cost=32.5]
    train minibatch loop:   0%|          | 0/8 [00:00<?, ?it/s, cost=30.8]

.. parsed-literal::

    model built, vocab size 157, document length 239


.. parsed-literal::

    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 114.23it/s, cost=33]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 104.38it/s, cost=29.7]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 110.13it/s, cost=26.7]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 172.65it/s, cost=22.3]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 117.81it/s, cost=21.6]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 159.72it/s, cost=14.8]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 131.81it/s, cost=13.9]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 181.36it/s, cost=16.8]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 136.84it/s, cost=13.1]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 164.23it/s, cost=14.8]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 144.23it/s, cost=11.3]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 171.15it/s, cost=7.84]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 128.12it/s, cost=10.6]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 175.80it/s, cost=8.91]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 133.73it/s, cost=10.7]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 185.70it/s, cost=15.2]
    train minibatch loop: 100%|██████████| 8/8 [00:00<00:00, 147.33it/s, cost=7.85]
    test minibatch loop: 100%|██████████| 1/1 [00:00<00:00, 186.12it/s, cost=5.2]


.. code:: ipython3

    trained_word2vec = malaya.word2vec.word2vec(nce_weights, dictionary)

.. code:: ipython3

    word = 'paduka'
    print("Embedding layer: 8 closest words to: '%s'"%(word))
    print(trained_word2vec.n_closest(word=word, num_closest=8, metric='cosine'))


.. parsed-literal::

    Embedding layer: 8 closest words to: 'paduka'
    [['itu', 0.5166147947311401], ['END', 0.5115543007850647], ['nasional', 0.5072782039642334], ['sebagai', 0.5061907768249512], ['kos', 0.504166841506958], ['UNK', 0.49253714084625244], ['antarabangsa', 0.4919373393058777], ['ketika', 0.48901939392089844]]

