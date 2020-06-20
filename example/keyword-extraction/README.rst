.. code:: ipython3

    import malaya

.. code:: ipython3

    # https://www.bharian.com.my/berita/nasional/2020/06/698386/isu-bersatu-tun-m-6-yang-lain-saman-muhyiddin
    
    string = """
    Dalam saman itu, plaintif memohon perisytiharan, antaranya mereka adalah ahli BERSATU yang sah, masih lagi memegang jawatan dalam parti (bagi pemegang jawatan) dan layak untuk bertanding pada pemilihan parti.
    
    Mereka memohon perisytiharan bahawa semua surat pemberhentian yang ditandatangani Muhammad Suhaimi bertarikh 28 Mei lalu dan pengesahan melalui mesyuarat Majlis Pimpinan Tertinggi (MPT) parti bertarikh 4 Jun lalu adalah tidak sah dan terbatal.
    
    Plaintif juga memohon perisytiharan bahawa keahlian Muhyiddin, Hamzah dan Muhammad Suhaimi di dalam BERSATU adalah terlucut, berkuat kuasa pada 28 Februari 2020 dan/atau 29 Februari 2020, menurut Fasal 10.2.3 perlembagaan parti.
    
    Yang turut dipohon, perisytiharan bahawa Seksyen 18C Akta Pertubuhan 1966 adalah tidak terpakai untuk menghalang pelupusan pertikaian berkenaan oleh mahkamah.
    
    Perisytiharan lain ialah Fasal 10.2.6 Perlembagaan BERSATU tidak terpakai di atas hal melucutkan/ memberhentikan keahlian semua plaintif.
    """

.. code:: ipython3

    import re
    
    # minimum cleaning, just simply to remove newlines.
    def cleaning(string):
        string = string.replace('\n', ' ')
        string = re.sub('[^A-Za-z\-() ]+', ' ', string).strip()
        string = re.sub(r'[ ]+', ' ', string).strip()
        return string
    
    string = cleaning(string)

Use RAKE algorithm
------------------

Original implementation from https://github.com/aneesha/RAKE. Malaya
added attention mechanism into RAKE algorithm.

.. code:: python

   def rake(
       string: str,
       model = None,
       top_k: int = 5,
       auto_ngram: bool = True,
       ngram_method: str = 'bow',
       ngram: Tuple[int, int] = (1, 1),
       atleast: int = 1,
       stop_words: List[str] = STOPWORDS,
       **kwargs
   ):
       """
       Extract keywords using Rake algorithm.

       Parameters
       ----------
       string: str
       model: Object, optional (default='None')
           Transformer model or any model has `attention` method.
       top_k: int, optional (default=5)
           return top-k results.
       auto_ngram: bool, optional (default=True)
           If True, will generate keyword candidates using N suitable ngram. Else use `ngram_method`.
       ngram_method: str, optional (default='bow')
           Only usable if `auto_ngram` is False. supported ngram generator:

           * ``'bow'`` - bag-of-word.
           * ``'skipgram'`` - bag-of-word with skip technique.
       ngram: tuple, optional (default=(1,1))
           n-grams size.
       atleast: int, optional (default=1)
           at least count appeared in the string to accept as candidate.
       stop_words: list, (default=malaya.text.function.STOPWORDS)
           list of stop words to remove. 

       Returns
       -------
       result: Tuple[float, str]
       """

auto-ngram
^^^^^^^^^^

This will auto generated N-size ngram for keyword candidates.

.. code:: ipython3

    malaya.keyword_extraction.rake(string)




.. parsed-literal::

    [(0.11666666666666665, 'ditandatangani Muhammad Suhaimi bertarikh Mei'),
     (0.08888888888888888, 'mesyuarat Majlis Pimpinan Tertinggi'),
     (0.08888888888888888, 'Seksyen C Akta Pertubuhan'),
     (0.05138888888888888, 'parti bertarikh Jun'),
     (0.04999999999999999, 'keahlian Muhyiddin Hamzah')]



auto-gram with Attention
^^^^^^^^^^^^^^^^^^^^^^^^

This will use attention mechanism as the scores. I will use
``small-electra`` in this example.

.. code:: ipython3

    electra = malaya.transformer.load(model = 'small-electra')


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:56: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/modeling.py:240: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.Dense instead.
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow_core/python/layers/core.py:187: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.__call__` method instead.
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:79: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:93: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/sampling.py:26: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:115: multinomial (from tensorflow.python.ops.random_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.random.categorical` instead.
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:118: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:119: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:121: The name tf.get_collection is deprecated. Please use tf.compat.v1.get_collection instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:122: The name tf.GraphKeys is deprecated. Please use tf.compat.v1.GraphKeys instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:128: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:130: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/electra-model/small/electra-small/model.ckpt


.. code:: ipython3

    malaya.keyword_extraction.rake(string, model = electra)




.. parsed-literal::

    [(0.2113546236771915, 'ditandatangani Muhammad Suhaimi bertarikh Mei'),
     (0.1707678455680971, 'terlucut berkuat kuasa'),
     (0.16650756665229807, 'Muhammad Suhaimi'),
     (0.1620429894692799, 'mesyuarat Majlis Pimpinan Tertinggi'),
     (0.08333952583953884, 'Seksyen C Akta Pertubuhan')]



fixed-ngram
^^^^^^^^^^^

.. code:: ipython3

    malaya.keyword_extraction.rake(string, auto_ngram = False, ngram = (1, 3), 
                                   ngram_method = 'skipgram', skip = 3)




.. parsed-literal::

    [(0.0010991603139160087, 'parti memohon perisytiharan'),
     (0.0010989640254270869, 'memohon perisytiharan Muhammad'),
     (0.0010985209375133323, 'perisytiharan Muhammad Suhaimi'),
     (0.0010972572356757605, 'memohon perisytiharan BERSATU'),
     (0.0010970435210070695, 'memohon perisytiharan sah')]



fixed-ngram with Attention
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    malaya.keyword_extraction.rake(string, model = electra, auto_ngram = False, ngram = (1, 3), 
                                   ngram_method = 'skipgram', skip = 3)




.. parsed-literal::

    [(0.007511555412415397, 'Suhaimi terlucut kuasa'),
     (0.00726812348703141, 'Suhaimi terlucut Februari'),
     (0.00725420955956774, 'Suhaimi terlucut berkuat'),
     (0.007235384019369932, 'Muhyiddin Suhaimi terlucut'),
     (0.00721164037502389, 'Hamzah Suhaimi terlucut')]



Use Textrank algorithm
----------------------

Malaya simply use textrank algorithm from networkx library.

.. code:: python

   def textrank(
       string: str,
       vectorizer,
       top_k: int = 5,
       auto_ngram: bool = True,
       ngram_method: str = 'bow',
       ngram: Tuple[int, int] = (1, 1),
       atleast: int = 1,
       stop_words: List[str] = STOPWORDS,
       **kwargs
   ):
       """
       Extract keywords using Textrank algorithm.

       Parameters
       ----------
       string: str
       vectorizer: Object, optional (default='None')
           model has `fit_transform` or `vectorize` method.
       top_k: int, optional (default=5)
           return top-k results.
       auto_ngram: bool, optional (default=True)
           If True, will generate keyword candidates using N suitable ngram. Else use `ngram_method`.
       ngram_method: str, optional (default='bow')
           Only usable if `auto_ngram` is False. supported ngram generator:

           * ``'bow'`` - bag-of-word.
           * ``'skipgram'`` - bag-of-word with skip technique.
       ngram: tuple, optional (default=(1,1))
           n-grams size.
       atleast: int, optional (default=1)
           at least count appeared in the string to accept as candidate.
       stop_words: list, (default=malaya.text.function.STOPWORDS)
           list of stop words to remove. 

       Returns
       -------
       result: Tuple[float, str]
       """

.. code:: ipython3

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer()

auto-ngram with TFIDF
^^^^^^^^^^^^^^^^^^^^^

This will auto generated N-size ngram for keyword candidates.

.. code:: ipython3

    malaya.keyword_extraction.textrank(string, vectorizer = tfidf)




.. parsed-literal::

    [(0.00015733542115111895, 'plaintif memohon perisytiharan'),
     (0.00012558589872969095, 'Fasal perlembagaan parti'),
     (0.00011512878779574369, 'Fasal Perlembagaan BERSATU'),
     (0.00011505807280697136, 'parti'),
     (0.00010763518916902933, 'memohon perisytiharan')]



auto-ngram with Attention
^^^^^^^^^^^^^^^^^^^^^^^^^

This will auto generated N-size ngram for keyword candidates.

.. code:: ipython3

    electra = malaya.transformer.load(model = 'small-electra')
    albert = malaya.transformer.load(model = 'albert')


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/electra-model/small/electra-small/model.ckpt
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/albert/tokenization.py:240: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.
    
    INFO:tensorflow:loading sentence piece model
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/albert/modeling.py:116: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/albert/modeling.py:588: The name tf.assert_less_equal is deprecated. Please use tf.compat.v1.assert_less_equal instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/albert/modeling.py:1025: The name tf.AUTO_REUSE is deprecated. Please use tf.compat.v1.AUTO_REUSE instead.
    
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/albert-model/base/albert-base/model.ckpt


.. code:: ipython3

    malaya.keyword_extraction.textrank(string, vectorizer = electra)




.. parsed-literal::

    [(6.3182663025223e-05, 'dipohon perisytiharan'),
     (6.31674674645778e-05, 'pemegang jawatan'),
     (6.316119389302752e-05, 'parti bertarikh Jun'),
     (6.316104723812124e-05, 'Februari'),
     (6.315819355276039e-05, 'plaintif')]



.. code:: ipython3

    malaya.keyword_extraction.textrank(string, vectorizer = albert)




.. parsed-literal::

    [(7.94645241452814e-05, 'Fasal Perlembagaan BERSATU'),
     (7.728400390215039e-05, 'mesyuarat Majlis Pimpinan Tertinggi'),
     (7.506390584039057e-05, 'Muhammad Suhaimi'),
     (7.503252483650059e-05, 'pengesahan'),
     (7.502407753712274e-05, 'terbatal Plaintif')]



fixed-ngram with Attention
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    malaya.keyword_extraction.textrank(string, vectorizer = electra, auto_ngram = False,
                                       ngram = (1, 3), ngram_method = 'skipgram', skip = 3)




.. parsed-literal::

    [(1.7071539462023998e-09, 'perisytiharan ahli sah'),
     (1.7071528386679705e-09, 'Fasal parti perisytiharan'),
     (1.7071498274826471e-09, 'Plaintif perisytiharan keahlian'),
     (1.7071355361007092e-09, 'Fasal dipohon perisytiharan'),
     (1.707130673312775e-09, 'plaintif perisytiharan')]



.. code:: ipython3

    malaya.keyword_extraction.textrank(string, vectorizer = albert, auto_ngram = False,
                                       ngram = (1, 3), ngram_method = 'skipgram', skip = 3)




.. parsed-literal::

    [(2.1995491577326747e-09, 'Perisytiharan Fasal melucutkan'),
     (2.1990164283127147e-09, 'Pimpinan Tertinggi (MPT)'),
     (2.1981574699825158e-09, 'Majlis Pimpinan (MPT)'),
     (2.1980610020130363e-09, 'Perisytiharan Fasal BERSATU'),
     (2.1973393621296214e-09, 'Perisytiharan Perlembagaan')]



Load Attention mechanism
------------------------

Use attention mechanism to get important keywords.

auto-ngram
^^^^^^^^^^

This will auto generated N-size ngram for keyword candidates.

.. code:: ipython3

    malaya.keyword_extraction.attention(string, model = electra)




.. parsed-literal::

    [(0.9452064568002397, 'menghalang pelupusan pertikaian'),
     (0.007486688404188947, 'Fasal Perlembagaan BERSATU'),
     (0.005130747276971111, 'ahli BERSATU'),
     (0.005036595631722718, 'melucutkan memberhentikan keahlian'),
     (0.004883706288857347, 'BERSATU')]



.. code:: ipython3

    malaya.keyword_extraction.attention(string, model = albert)




.. parsed-literal::

    [(0.16196368022187793, 'plaintif memohon perisytiharan'),
     (0.09294065744319371, 'memohon perisytiharan'),
     (0.06902302277868422, 'plaintif'),
     (0.05584840295920779, 'ditandatangani Muhammad Suhaimi bertarikh Mei'),
     (0.05206225590337424, 'dipohon perisytiharan')]



fixed-ngram
^^^^^^^^^^^

.. code:: ipython3

    malaya.keyword_extraction.attention(string, model = electra, auto_ngram = False,
                                       ngram = (1, 3), ngram_method = 'bow')




.. parsed-literal::

    [(0.15667043125587973, 'pelupusan pertikaian mahkamah'),
     (0.15665311872357476, 'pertikaian mahkamah Perisytiharan'),
     (0.15657934237804905, 'pertikaian mahkamah'),
     (0.1563242367855659, 'menghalang pelupusan pertikaian'),
     (0.1562270516451705, 'pelupusan pertikaian')]



.. code:: ipython3

    malaya.keyword_extraction.attention(string, model = albert, auto_ngram = False,
                                       ngram = (1, 3), ngram_method = 'bow')




.. parsed-literal::

    [(0.031264380566934015, 'saman plaintif memohon'),
     (0.02621530292963218, 'plaintif memohon perisytiharan'),
     (0.02573609954868083, 'Dalam saman plaintif'),
     (0.022935623722179672, 'plaintif memohon'),
     (0.019724791761830188, 'Mereka memohon perisytiharan')]


