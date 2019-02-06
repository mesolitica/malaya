
.. code:: ipython3

    import malaya

What is word mover distance?
----------------------------

between two documents in a meaningful way, even when they have no words
in common. It uses vector embeddings of words. It been shown to
outperform many of the state-of-the-art methods in k-nearest neighbors
classification.

You can read more about word mover distance from `Word Distance between
Word
Embeddings <https://towardsdatascience.com/word-distance-between-word-embeddings-cc3e9cf1d632>`__.

**Closest to 0 is better**.

.. code:: ipython3

    left_sentence = 'saya suka makan ayam'
    right_sentence = 'saya suka makan ikan'
    left_token = left_sentence.split()
    right_token = right_sentence.split()

.. code:: ipython3

    w2v_wiki = malaya.word2vec.load_wiki()
    w2v_wiki = malaya.word2vec.word2vec(w2v_wiki['nce_weights'],w2v_wiki['dictionary'])

.. code:: ipython3

    fasttext_wiki, ngrams = malaya.fast_text.load_wiki()
    fasttext_wiki = malaya.fast_text.fast_text(fasttext_wiki['embed_weights'],fasttext_wiki['dictionary'], ngrams)

Using word2vec
--------------

.. code:: ipython3

    malaya.word_mover.distance(left_token, right_token, w2v_wiki)




.. parsed-literal::

    0.8225146532058716



.. code:: ipython3

    malaya.word_mover.distance(left_token, left_token, w2v_wiki)




.. parsed-literal::

    0.0



Using fast-text
---------------

.. code:: ipython3

    malaya.word_mover.distance(left_token, right_token, fasttext_wiki)




.. parsed-literal::

    2.82466983795166



.. code:: ipython3

    malaya.word_mover.distance(left_token, left_token, fasttext_wiki)




.. parsed-literal::

    0.0


