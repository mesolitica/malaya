
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 11.4 s, sys: 1.39 s, total: 12.8 s
    Wall time: 16.2 s


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
    fasttext_wiki = malaya.fast_text.fast_text(fasttext_wiki['embed_weights'],
                                               fasttext_wiki['dictionary'], ngrams)

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



Why word mover distance?
------------------------

Maybe you heard about skipthought or siamese network to train sentences
similarity, but both required a good corpus plus really slow to train.
Malaya provided both models to train your own text similarity, can check
here, `Malaya
text-similarity <https://malaya.readthedocs.io/en/latest/Similarity.html>`__

``word2vec`` or ``fast-text`` are really good to know semantic
definitions between 2 words, like below,

.. code:: ipython3

    w2v_wiki.n_closest(word = 'anwar', num_closest=8, metric='cosine')




.. parsed-literal::

    [['zaid', 0.7285637855529785],
     ['khairy', 0.6839416027069092],
     ['zabidi', 0.6709405183792114],
     ['nizar', 0.6695379018783569],
     ['harussani', 0.6595045328140259],
     ['shahidan', 0.6565827131271362],
     ['azalina', 0.6541041135787964],
     ['shahrizat', 0.6538639068603516]]



So we got some suggestion from the interface included distance between
0-1, closest to 1 is better.

Now let say I want to compare similarity between 2 sentences, and using
vectors representation from our word2vec and fast-text.

I got, ``rakyat sebenarnya sukakan mahathir``, and
``rakyat sebenarnya sukakan najib``

.. code:: ipython3

    mahathir = 'rakyat sebenarnya sukakan mahathir'
    najib = 'rakyat sebenarnya sukakan najib'
    malaya.word_mover.distance(mahathir.split(), najib.split(), w2v_wiki)




.. parsed-literal::

    0.9017602205276489



0.9, quite good. What happen if we make our sentence quite polarity
ambigious for najib? (Again, this is just example)

.. code:: ipython3

    mahathir = 'rakyat sebenarnya sukakan mahathir'
    najib = 'rakyat sebenarnya gilakan najib'
    malaya.word_mover.distance(mahathir.split(), najib.split(), w2v_wiki)




.. parsed-literal::

    1.7690724730491638



We just changed ``sukakan`` with ``gilakan``, but our word2vec
representation based on ``rakyat sebenarnya <word> <person>`` not able
to correlate same polarity, real definition of ``gilakan`` is positive
polarity, but word2vec learnt ``gilakan`` is negative or negate.

Soft mode
---------

What happened if a word is not inside vectorizer dictionary?
``malaya.word_mover.distance`` will throw an exception.

.. code:: ipython3

    left = 'tyi'
    right = 'qwe'
    malaya.word_mover.distance(left.split(), right.split(), w2v_wiki)


::


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    ~/Documents/Malaya/malaya/word_mover.py in _word_mover(left_token, right_token, vectorizer, soft)
         45         try:
    ---> 46             wordvecs[token] = vectorizer.get_vector_by_name(token)
         47         except Exception as e:


    ~/Documents/Malaya/malaya/word2vec.py in get_vector_by_name(self, word)
        289                 'input not found in dictionary, here top-5 nearest words [%s]'
    --> 290                 % (strings)
        291             )


    Exception: input not found in dictionary, here top-5 nearest words [qw, qe, we, qwest, qwabe]

    
    During handling of the above exception, another exception occurred:


    Exception                                 Traceback (most recent call last)

    <ipython-input-13-4acdc71ff70d> in <module>
          1 left = 'tyi'
          2 right = 'qwe'
    ----> 3 malaya.word_mover.distance(left.split(), right.split(), w2v_wiki)
    

    ~/Documents/Malaya/malaya/word_mover.py in distance(left_token, right_token, vectorizer, soft)
        111     if not hasattr(vectorizer, 'get_vector_by_name'):
        112         raise ValueError('vectorizer must has `get_vector_by_name` method')
    --> 113     prob = _word_mover(left_token, right_token, vectorizer, soft = soft)
        114     return pulp.value(prob.objective)
        115 


    ~/Documents/Malaya/malaya/word_mover.py in _word_mover(left_token, right_token, vectorizer, soft)
         47         except Exception as e:
         48             if not soft:
    ---> 49                 raise Exception(e)
         50             else:
         51                 arr = np.array([fuzz.ratio(token, k) for k in vectorizer.words])


    Exception: input not found in dictionary, here top-5 nearest words [qw, qe, we, qwest, qwabe]


So if use ``soft = True``, if the word is not inside vectorizer, it will
find the nearest word.

.. code:: ipython3

    left = 'tyi'
    right = 'qwe'
    malaya.word_mover.distance(left.split(), right.split(), w2v_wiki, soft = True)




.. parsed-literal::

    1.273216962814331



Load expander
-------------

We want to expand shortforms based on ``malaya.normalize.spell`` by
using word mover distance. If our vector knows that ``mkn`` semantically
similar to ``makan`` based on ``saya suka mkn ayam`` sentence, word
mover distance will become closer.

It is really depends on our vector, and word2vec may not able to
understand shortform, so we will use fast-text to fix ``OUT-OF-VOCAB``
problem.

.. code:: ipython3

    malays = malaya.load_malay_dictionary()
    wiki, ngrams = malaya.fast_text.load_wiki()
    fast_text_embed = malaya.fast_text.fast_text(wiki['embed_weights'],wiki['dictionary'],ngrams)
    expander = malaya.word_mover.expander(malays, fast_text_embed)


.. parsed-literal::

    downloading Malay texts


.. parsed-literal::

    1.00MB [00:00, 1.70MB/s]                   


.. code:: ipython3

    string = 'y u xsuka makan HUSEIN kt situ tmpt'
    another = 'i mmg xska mknn HUSEIN kampng tempt'

.. code:: ipython3

    expander.expand(string)




.. parsed-literal::

    [[('tmpt',
       'kenapa awak tak suka makan Husein kat situ tut',
       0.8088938253521919),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tuit',
       0.863929785296917),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tat',
       0.8680638003787995),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ top',
       0.8688952446055412),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tip',
       0.8978437346220016),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ taat',
       0.936883625289917),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ topi',
       0.9442774548711776),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tumit',
       0.9495834815340042),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tempe',
       0.9758907731723786),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ ampe',
       0.9821926467533112),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tempo',
       0.9836614096956253),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tepet',
       0.994007917971611),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ amit',
       0.9999424153804779),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tuat',
       1.0002889167022706),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ mat',
       1.0071370331926346),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ temut',
       1.011553812426567),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ ampit',
       1.022653616695404),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ ampo',
       1.0231078831071854),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tipu',
       1.0246861065587998),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tepi',
       1.0285266551542283),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ umut',
       1.0287358275117875),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ emat',
       1.0357482937116622),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ empat',
       1.0431590774860382),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tapi',
       1.0562509994459153),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tepu',
       1.0601519473543166),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tumpat',
       1.074669928882599),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ impi',
       1.078846170501709),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ umat',
       1.0791117155513763),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tampi',
       1.0883281208925248),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tumpu',
       1.091578345676422),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ umpat',
       1.092372225769043),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tepat',
       1.0979607516746521),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tampa',
       1.1118229238204955),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ amput',
       1.1226389572820663),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tapa',
       1.129335333744049),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ timpa',
       1.1353471846590042),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ empu',
       1.1459274488725661),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tempa',
       1.164648480837822),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tampu',
       1.1812463180065156),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tempat',
       1.1856716803007126),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tamat',
       1.2068403679332733),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ amat',
       1.2214121790246963),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ ampu',
       1.2350379461402894),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ taut',
       1.2796957146606445)]]



.. code:: ipython3

    expander.expand(another)




.. parsed-literal::

    [[('ska', 'saya memang tak soka mknn Husein kampng tempt', 0.7199365496635437),
      ('ska', 'saya memang tak suka mknn Husein kampng tempt', 0.8050327301025391),
      ('ska', 'saya memang tak sika mknn Husein kampng tempt', 0.8729341626167297),
      ('ska', 'saya memang tak saka mknn Husein kampng tempt', 0.875930666923523),
      ('ska', 'saya memang tak spa mknn Husein kampng tempt', 0.8995948433876038),
      ('ska', 'saya memang tak sua mknn Husein kampng tempt', 0.9496822357177734),
      ('ska', 'saya memang tak seka mknn Husein kampng tempt', 0.9891390204429626),
      ('ska', 'saya memang tak ski mknn Husein kampng tempt', 1.1318669319152832),
      ('ska', 'saya memang tak sia mknn Husein kampng tempt', 1.1666431427001953)],
     [('mknn', 'saya memang tak ska min Husein kampng tempt', 0.8653836846351624),
      ('mknn', 'saya memang tak ska maun Husein kampng tempt', 1.045318603515625),
      ('mknn', 'saya memang tak ska kun Husein kampng tempt', 1.0710314512252808),
      ('mknn', 'saya memang tak ska ken Husein kampng tempt', 1.0728274583816528),
      ('mknn', 'saya memang tak ska kon Husein kampng tempt', 1.0992072820663452),
      ('mknn', 'saya memang tak ska ikon Husein kampng tempt', 1.1365187168121338),
      ('mknn', 'saya memang tak ska makin Husein kampng tempt', 1.180336833000183),
      ('mknn', 'saya memang tak ska main Husein kampng tempt', 1.182568907737732),
      ('mknn', 'saya memang tak ska makan Husein kampng tempt', 1.183489203453064),
      ('mknn', 'saya memang tak ska makna Husein kampng tempt', 1.184565544128418),
      ('mknn', 'saya memang tak ska kan Husein kampng tempt', 1.2368937730789185),
      ('mknn', 'saya memang tak ska akan Husein kampng tempt', 1.2527291774749756),
      ('mknn', 'saya memang tak ska mani Husein kampng tempt', 1.266147494316101),
      ('mknn', 'saya memang tak ska ikan Husein kampng tempt', 1.2773109674453735),
      ('mknn', 'saya memang tak ska mini Husein kampng tempt', 1.3020210266113281),
      ('mknn', 'saya memang tak ska mana Husein kampng tempt', 1.3099677562713623),
      ('mknn', 'saya memang tak ska menu Husein kampng tempt', 1.3974181413650513),
      ('mknn', 'saya memang tak ska mena Husein kampng tempt', 1.404064655303955),
      ('mknn',
       'saya memang tak ska makanan Husein kampng tempt',
       1.4473483562469482)],
     [('kampng',
       'saya memang tak ska mknn Husein kampung tempt',
       0.9272603988647461)],
     [('tempt',
       'saya memang tak ska mknn Husein kampng tempo',
       0.7405402660369873),
      ('tempt',
       'saya memang tak ska mknn Husein kampng tempe',
       0.7510019540786743),
      ('tempt', 'saya memang tak ska mknn Husein kampng tempa', 0.885798454284668),
      ('tempt',
       'saya memang tak ska mknn Husein kampng temut',
       0.9036741256713867),
      ('tempt',
       'saya memang tak ska mknn Husein kampng tempat',
       0.9161624312400818)]]


