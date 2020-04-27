.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 4.91 s, sys: 1.19 s, total: 6.1 s
    Wall time: 7.16 s


Wordvector augmentation
-----------------------

Let say you have a very limited labelled corpus, and you want to add
more, but labelling is very costly.

So, text augmentation! You can use wordvector to replace words with
similar semantics!

.. code:: python

   def wordvector_augmentation(
       string,
       wordvector,
       threshold = 0.5,
       top_n = 5,
       soft = False,
       cleaning_function = None,
   ):
       """
       augmenting a string using wordvector.

       Parameters
       ----------
       string: str
       wordvector: object
           wordvector interface object.
       threshold: float, optional (default=0.5)
           random selection for a word.
       soft: bool, optional (default=False)
           if True, a word not in the dictionary will be replaced with nearest jarowrinkler ratio.
           if False, it will throw an exception if a word not in the dictionary.
       top_n: int, (default=5)
           number of nearest neighbors returned.
       cleaning_function: function, (default=None)
           function to clean text.

       Returns
       -------
       result: list
       """

.. code:: ipython3

    string = 'saya suka makan ayam dan ikan'

.. code:: ipython3

    vocab_wiki, embedded_wiki = malaya.wordvector.load_wiki()
    word_vector_wiki = malaya.wordvector.load(embedded_wiki, vocab_wiki)


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/wordvector.py:113: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/wordvector.py:124: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


.. code:: ipython3

    augmented = malaya.generator.wordvector_augmentation(string,
                                      word_vector_wiki,
                                      soft=True)
    augmented




.. parsed-literal::

    ['saya suka makan ayam dan ikan',
     'saya suka makan ayam serta ayam',
     'saya suka makan ayam atau ular',
     'saya suka makan ayam mahupun keju',
     'saya suka makan ayam tetapi lembu']



.. code:: ipython3

    text = 'Perdana Menteri berkata, beliau perlu memperoleh maklumat terperinci berhubung isu berkenaan sebelum kerajaan dapat mengambil sebarang tindakan lanjut. Bagaimanapun, beliau yakin masalah itu dapat diselesaikan dan pentadbiran kerajaan boleh berfungsi dengan baik.'

.. code:: ipython3

    augmented = malaya.generator.wordvector_augmentation(text,
                                      word_vector_wiki,
                                      soft=True)
    augmented




.. parsed-literal::

    ['Perdana Menteri berkata , beliau perlu memperoleh maklumat terperinci berhubung isu berkenaan sebelum kerajaan dapat mengambil sebarang tindakan lanjut . Bagaimanapun , beliau yakin masalah itu dapat diselesaikan dan pentadbiran kerajaan boleh berfungsi dengan baik .',
     'Perdana Menteri berkata , beliau perlu memperoleh data terperinci berhubung isu berkaitan sebelum pemerintah dapat mendapat segala tindakan lanjut . Bagaimanapun , dia sedar masalah itu boleh diselesaikan serta pentadbiran pemerintah boleh berfungsi dengan baik .',
     'Perdana Menteri berkata , beliau perlu memperoleh bacaan terperinci berhubung isu tertentu sebelum perlembagaan dapat menghabiskan sesuatu tindakan lanjut . Bagaimanapun , baginda bimbang masalah itu harus diselesaikan atau pentadbiran perlembagaan boleh berfungsi dengan baik .',
     'Perdana Menteri berkata , beliau perlu memperoleh penjelasan terperinci berhubung isu tersebut sebelum kesultanan dapat mengubah suatu tindakan lanjut . Bagaimanapun , mereka menyangka masalah itu perlu diselesaikan mahupun pentadbiran kesultanan boleh berfungsi dengan baik .',
     'Perdana Menteri berkata , beliau perlu memperoleh informasi terperinci berhubung isu berlainan sebelum pemerintahan dapat memakan pelbagai tindakan lanjut . Bagaimanapun , saya takut masalah itu mampu diselesaikan tetapi pentadbiran pemerintahan boleh berfungsi dengan baik .']



Transformer augmentation
------------------------

Problem with wordvector, it just replaced a word for near synonym
without understood the whole sentence context, so, Transformer comes to
the rescue!

.. code:: python

   def transformer_augmentation(
       string,
       model,
       threshold = 0.5,
       top_p = 0.8,
       top_k = 100,
       temperature = 0.8,
       top_n = 5,
       cleaning_function = None,
   ):

       """
       augmenting a string using transformer + nucleus sampling / top-k sampling.

       Parameters
       ----------
       string: str
       model: object
           transformer interface object. Right now only supported BERT.
       threshold: float, optional (default=0.5)
           random selection for a word.
       top_p: float, optional (default=0.8)
           cumulative sum of probabilities to sample a word. If top_n bigger than 0, the model will use nucleus sampling, else top-k sampling.
       top_k: int, optional (default=100)
           k for top-k sampling.
       temperature: float, optional (default=0.8)
           logits * temperature.
       top_n: int, (default=5)
           number of nearest neighbors returned.
       cleaning_function: function, (default=None)
           function to clean text.

       Returns
       -------
       result: list
       """

.. code:: ipython3

    model = malaya.transformer.load(model = 'albert')


.. parsed-literal::

    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/albert/tokenization.py:240: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.
    
    INFO:tensorflow:loading sentence piece model
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/albert/modeling.py:116: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/albert/modeling.py:194: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/albert/modeling.py:507: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/albert/modeling.py:588: The name tf.assert_less_equal is deprecated. Please use tf.compat.v1.assert_less_equal instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/albert/modeling.py:1025: The name tf.AUTO_REUSE is deprecated. Please use tf.compat.v1.AUTO_REUSE instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/albert/modeling.py:253: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.Dense instead.
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow_core/python/layers/core.py:187: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.__call__` method instead.
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/sampling.py:26: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/albert/__init__.py:118: multinomial (from tensorflow.python.ops.random_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.random.categorical` instead.
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/albert/__init__.py:122: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/albert/__init__.py:123: The name tf.get_collection is deprecated. Please use tf.compat.v1.get_collection instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/albert/__init__.py:124: The name tf.GraphKeys is deprecated. Please use tf.compat.v1.GraphKeys instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/albert/__init__.py:129: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/albert/__init__.py:131: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/albert-model/base/albert-base/model.ckpt


.. code:: ipython3

    augmented = malaya.generator.transformer_augmentation(text, model)
    augmented




.. parsed-literal::

    ['Perdana Menteri berkata, pihaknya perlu memperoleh maklumat terperinci berhubung isu berkenaan sebelum kerajaan tidak menyelesaikan alih tindakan lanjut. Bagaimanapun, beliau yakin masalah itu dapat diselesaikan dan semua kerajaan boleh pulih dengan baik.',
     'Perdana Menteri berkata, kerajaan perlu memperoleh maklumat terperinci daripada masalah berkenaan sebelum kerajaan tidak mengendalikan alih tindakan lanjut. Bagaimanapun, beliau berharap masalah itu dapat diselesaikan kerana semua kerajaan boleh diselesaikan dengan baik.',
     'Perdana Menteri berkata, kerajaan perlu memperoleh maklumat terperinci daripada masalah berkenaan sebelum kerajaan tidak menyelesaikan alih tindakan lanjut. Bagaimanapun, beliau yakin masalah itu dapat diselesaikan supaya semua kerajaan boleh diselesaikan dengan baik.',
     'Perdana Menteri berkata, pihaknya perlu memperoleh maklumat terperinci mengenai isu berkenaan sebelum kerajaan tidak mengendalikan alih tindakan lanjut. Bagaimanapun, beliau berharap masalah itu dapat diselesaikan dan semua kerajaan boleh pulih dengan baik.',
     'Perdana Menteri berkata, kerajaan perlu memperoleh maklumat terperinci berhubung isu berkenaan sebelum kerajaan tidak menyelesaikan alih tindakan lanjut. Bagaimanapun, beliau berharap masalah itu dapat diselesaikan kerana semua kerajaan boleh pulih dengan baik.']


