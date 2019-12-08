.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 4.21 s, sys: 879 ms, total: 5.08 s
    Wall time: 5.46 s


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

.. code:: python

    string = 'saya suka makan ayam dan ikan'

.. code:: python

    embedded_wiki = malaya.wordvector.load_wiki()
    word_vector_wiki = malaya.wordvector.load(embedded_wiki['nce_weights'], embedded_wiki['dictionary'])


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/wordvector.py:85: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/wordvector.py:96: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


.. code:: python

    augmented = malaya.generator.wordvector_augmentation(string,
                                      word_vector_wiki,
                                      soft=True)
    augmented




.. parsed-literal::

    ['saya suka makan ayam dan ikan',
     'anda gemar minum itik atau kerang',
     'kami pandai mengeram kambing serta daging',
     'kamu sanggup mengunyah lembu ataupun ayam',
     'kita terpesona memakan arnab mahupun udang']



.. code:: python

    text = 'Perdana Menteri berkata, beliau perlu memperoleh maklumat terperinci berhubung isu berkenaan sebelum kerajaan dapat mengambil sebarang tindakan lanjut. Bagaimanapun, beliau yakin masalah itu dapat diselesaikan dan pentadbiran kerajaan boleh berfungsi dengan baik.'

.. code:: python

    augmented = malaya.generator.wordvector_augmentation(text,
                                      word_vector_wiki,
                                      soft=True)
    augmented




.. parsed-literal::

    ['Perdana Menteri berkata , beliau perlu memperoleh maklumat terperinci berhubung isu berkenaan sebelum kerajaan dapat mengambil sebarang tindakan lanjut . Bagaimanapun , beliau yakin masalah itu dapat diselesaikan dan pentadbiran kerajaan boleh berfungsi dengan baik .',
     'Perdana Menteri berkata , dia perlu memperoleh informasi teliti berhubung isu berkenaan sebelum kerajaan boleh mengambil suatu perbuatan lanjut . Bagaimanapun , dia yakin masaalah itu dapat diselesaikan atau pentadbiran kerajaan boleh beroperasi dengan baiknya .',
     'Perdana Menteri berkata , mereka perlu memperoleh data sistematik berhubung isu berkenaan sebelum kerajaan harus mengambil sesuatu perlakuan lanjut . Bagaimanapun , mereka yakin permasalahan itu dapat diselesaikan serta pentadbiran kerajaan boleh bertindak dengan hebat .',
     'Perdana Menteri berkata , baginda perlu memperoleh perincian ekstensif berhubung isu berkenaan sebelum kerajaan mampu mengambil pelbagai sikap lanjut . Bagaimanapun , baginda yakin kesulitan itu dapat diselesaikan ataupun pentadbiran kerajaan boleh bergetar dengan kuat .',
     'Perdana Menteri berkata , saya perlu memperoleh info menyeluruh berhubung isu berkenaan sebelum kerajaan perlu mengambil segala kelakuan lanjut . Bagaimanapun , saya yakin kesukaran itu dapat diselesaikan mahupun pentadbiran kerajaan boleh dimanfaatkan dengan kukuh .']



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

.. code:: python

    model = malaya.transformer.load(model = 'bert', size = 'small')


.. parsed-literal::

    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/bert/modeling.py:93: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/bert/modeling.py:171: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/bert/modeling.py:409: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/bert/modeling.py:490: The name tf.assert_less_equal is deprecated. Please use tf.compat.v1.assert_less_equal instead.
    
    WARNING:tensorflow:
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
      * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.
    
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/bert/modeling.py:671: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.Dense instead.
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow_core/python/layers/core.py:187: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.__call__` method instead.
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_transformer/_sampling.py:26: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_transformer/_bert.py:102: multinomial (from tensorflow.python.ops.random_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.random.categorical` instead.
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_transformer/_bert.py:106: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_transformer/_bert.py:107: The name tf.get_collection is deprecated. Please use tf.compat.v1.get_collection instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_transformer/_bert.py:108: The name tf.GraphKeys is deprecated. Please use tf.compat.v1.GraphKeys instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_transformer/_bert.py:113: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/_transformer/_bert.py:115: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/bert-model/small/bert-small-v2/model.ckpt


.. code:: python

    augmented = malaya.generator.transformer_augmentation(text, model)
    augmented




.. parsed-literal::

    ['Perdana Menteri berkata, beliau telah mendapatkan maklumat terperinci mengenai perkara berkenaan supaya kerajaan tidak mengambil sebarang tindakan.. Bagaimanapun, beliau yakin isu itu dapat diselesaikan dan pastinya kerajaan boleh berfungsi dengan baik.',
     'Perdana Menteri berkata, beliau akan mendapatkan maklumat terperinci berhubung perkara berkenaan supaya kerajaan tidak mengambil sebarang tindakan segera. Bagaimanapun, beliau yakin isu itu dapat diselesaikan dan diharap kerajaan boleh berfungsi dengan baik.',
     'Perdana Menteri berkata, beliau akan memberikan maklumat terperinci berhubung isu berkenaan agar kerajaan perlu mengambil sebarang tindakan.. Bagaimanapun, beliau yakin perkara itu dapat diselesaikan dan berharap kerajaan boleh berfungsi dengan baik.',
     'Perdana Menteri berkata, beliau akan memberikan maklumat terperinci berhubung perkara berkenaan dan kerajaan akan mengambil sebarang tindakan sewajarnya. Bagaimanapun, beliau yakin perkara itu dapat diselesaikan dan berharap kerajaan boleh berfungsi dengan baik.',
     'Perdana Menteri berkata, beliau telah mendapatkan maklumat terperinci berhubung isu berkenaan supaya kerajaan tidak mengambil sebarang tindakan.. Bagaimanapun, beliau yakin isu itu dapat diselesaikan dan berharap kerajaan boleh berfungsi dengan baik.']



