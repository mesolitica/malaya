
.. code:: python

    import malaya

Load malaya news word2vec
-------------------------

.. code:: python

    embedded = malaya.malaya_word2vec(256)


.. parsed-literal::

    downloading word2vec-256 embedded


.. parsed-literal::

    109MB [00:44, 2.82MB/s]


Load word2vec model
-------------------

.. code:: python

    word_vector = malaya.Word2Vec(embedded['nce_weights'], embedded['dictionary'])

.. code:: python

    word = 'anwar'
    print("Embedding layer: 8 closest words to: '%s'"%(word))
    print(word_vector.n_closest(word=word, num_closest=8, metric='cosine'))


.. parsed-literal::

    Embedding layer: 8 closest words to: 'anwar'
    [['mahathir', 0.44774019718170166], ['beliau', 0.44170427322387695], ['zaid', 0.43993180990219116], ['hishammuddin', 0.4343132972717285], ['kuok', 0.43307822942733765], ['husam', 0.43213725090026855], ['anifah', 0.4307258129119873], ['pesakit', 0.4262162446975708]]


.. code:: python

    print(word_vector.analogy('anwar', 'penjara', 'kerajaan', 5))


.. parsed-literal::

    ['penjara', 'kerajaan', 'kkm', 'kabinet', 'tuju']


.. code:: python

    word_vector.calculator('anwar + amerika + mahathir', num_closest=8, metric='cosine',
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



.. code:: python

    word_vector.calculator('anwar * amerika', num_closest=8, metric='cosine',
                          return_similarity=False)




.. parsed-literal::

    ['turut',
     'pengajian',
     'tangan',
     'beli',
     'terus',
     'susulan',
     'pengetahuan',
     'tujuan',
     'meter']
