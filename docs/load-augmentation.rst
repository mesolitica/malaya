.. code:: python

    %%time
    
    import malaya


.. parsed-literal::

    CPU times: user 4.36 s, sys: 840 ms, total: 5.2 s
    Wall time: 4.77 s


Why augmentation
----------------

Let say you have a very limited labelled corpus, and you want to add
more, but labelling is very costly.

So, text augmentation! We provided few augmentation interfaces in
Malaya.

Load Synonym
------------

Use dictionary of synonym to replace words with it synonyms. Synonym
data from
`Malaya-Dataset/90k-synonym <https://github.com/huseinzol05/Malaya-Dataset#90k-synonym>`__.

.. code:: python

   def synonym(
       string: str,
       threshold: float = 0.5,
       top_n = 5,
       cleaning_function: Callable = augmentation_textcleaning,
       **kwargs
   ):
       """
       augmenting a string using synonym, https://github.com/huseinzol05/Malaya-Dataset#90k-synonym

       Parameters
       ----------
       string: str
       threshold: float, optional (default=0.5)
           random selection for a word.
       top_n: int, (default=5)
           number of nearest neighbors returned. Length of returned result should as top_n.
       cleaning_function: function, (default=malaya.text.function.augmentation_textcleaning)
           function to clean text.

       Returns
       -------
       result: List[str]
       """

.. code:: python

    string = 'saya suka makan ayam dan ikan'
    text = 'Perdana Menteri berkata, beliau perlu memperoleh maklumat terperinci berhubung isu berkenaan sebelum kerajaan dapat mengambil sebarang tindakan lanjut. Bagaimanapun, beliau yakin masalah itu dapat diselesaikan dan pentadbiran kerajaan boleh berfungsi dengan baik.'

.. code:: python

    malaya.augmentation.synonym(string)




.. parsed-literal::

    ['saya suka makan ayam dan ikan',
     'saya mencinta makan ayam jantan dan ikan',
     'saya mencinta makan makan ayam jantan dan ikan',
     'saya suka makan makan ayam dan ikan',
     'saya suka makan ayam dan ikan']



.. code:: python

    malaya.augmentation.synonym(text)




.. parsed-literal::

    ['Perdana menteri menunjukkan beliau perlu memperoleh maklumat Terperinci menghayati isu berkenaan sebelum kerajaan dapat mengawali sebarang nombor lanjut bagaimanapun beliau beramanah sedih itu dapat diselesaikan dan pengurusannya kerajaan berupaya berfungsi dengan baik',
     'Perdana menteri menunjukkan beliau wajib mengusahakan data Terperinci menghayati penerbitan berkenaan di hadapan jajahan menggunakannya mendapatkan sebarang digit tertua bagaimanapun beliau beramanah suram itu dapatkan diselesaikan dan pengurusannya kabinet boleh mengangkut dengan baik',
     'Ulung uskup merupakan beliau wajib mengusahakan data Terperinci mempunyai penerbitan berkenaan di hadapan kerajaan menggunakannya berkumpul sebarang nombor gelap bagaimanapun beliau beramanah daif itu memperoleh diselesaikan dan pengurusannya kerajaan boleh mencari dengan baik',
     'Ulung menteri merupakan beliau wajib memupuk dokumen Terperinci mempunyai pengeluaran berkenaan sebelum kerajaan menangani berkumpul sebarang nombor gelap masih beliau yakin daif itu tiba diselesaikan dan pengurusannya pemerintah boleh mengesani dengan baik',
     'Perdana uskup menunjukkan beliau wajib pelihara dokumen Terperinci mempunyai keluaran berkenaan sebelumnya kerajaan menangani berkumpul sebarang nombor jahat Bagaimana pun beliau yakin daif itu maju diselesaikan dan pengurusannya komandan boleh mengesani dengan baik']



Load Wordvector
---------------

dictionary of synonym is quite hard to populate, required some domain
experts to help us. So we can use wordvector to find nearest words.

.. code:: python

   def wordvector(
       string: str,
       wordvector,
       threshold: float = 0.5,
       top_n: int = 5,
       soft: bool = False,
       cleaning_function: Callable = augmentation_textcleaning,
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
           number of nearest neighbors returned. Length of returned result should as top_n.
       cleaning_function: function, (default=malaya.text.function.augmentation_textcleaning)
           function to clean text.

       Returns
       -------
       result: List[str]
       """

.. code:: python

    vocab_wiki, embedded_wiki = malaya.wordvector.load_wiki()
    word_vector_wiki = malaya.wordvector.load(embedded_wiki, vocab_wiki)


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/wordvector.py:114: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/wordvector.py:125: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


.. code:: python

    malaya.augmentation.wordvector(
        string, word_vector_wiki, soft = True
    )




.. parsed-literal::

    ['saya suka makan ayam dan ikan',
     'kamu gemar minum ayam serta ayam',
     'anda pandai tidur ayam atau ular',
     'kami senang mandi ayam mahupun keju',
     'aku ingin berehat ayam tetapi lembu']



.. code:: python

    malaya.augmentation.wordvector(
        text, word_vector_wiki, soft = True
    )




.. parsed-literal::

    ['perdana menteri berkata beliau perlu memperoleh maklumat terperinci berhubung isu berkenaan sebelum kerajaan dapat mengambil sebarang tindakan lanjut bagaimanapun beliau yakin masalah itu dapat diselesaikan dan pentadbiran kerajaan boleh berfungsi dengan baik',
     'perdana kementerian menyatakan beliau perlu memperoleh maklumat terperinci berkaitan persoalan berkaitan selepas kerajaan dapat mendapat sebarang tindakan terperinci walaupun dia sedar gangguan itu boleh dibuktikan serta pentadbiran kerajaan dapat dikelaskan dengan baik',
     'perdana setiausaha mengatakan beliau perlu memperoleh maklumat terperinci berhadapan prosedur tertentu setelah kerajaan dapat menghabiskan sebarang tindakan lanjutan namun baginda bimbang kelemahan itu harus dilaksanakan atau pentadbiran kerajaan harus bertindak dengan baik',
     'perdana jabatan mendapati beliau perlu memperoleh maklumat terperinci sejajar artikel tersebut ketika kerajaan dapat mengubah sebarang tindakan ringkas maka mereka menyangka gejala itu perlu dikesan mahupun pentadbiran kerajaan perlu dirujuk dengan baik',
     'perdana duta mencadangkan beliau perlu memperoleh maklumat terperinci bertentangan kontroversi berlainan sejak kerajaan dapat memakan sebarang tindakan positif tetapi saya takut risiko itu mampu diperhatikan tetapi pentadbiran kerajaan akan dikira dengan baik']



Load Transformer
----------------

Problem with wordvector, it just replaced a word for near synonym
without understood the whole sentence context, so, Transformer comes to
the rescue!

.. code:: python

   def transformer(
       string: str,
       model,
       threshold: float = 0.5,
       top_p: float = 0.9,
       top_k: int = 100,
       temperature: float = 1.0,
       top_n: int = 5,
       cleaning_function: Callable = None,
   ):

       """
       augmenting a string using transformer + nucleus sampling / top-k sampling.

       Parameters
       ----------
       string: str
       model: object
           transformer interface object. Right now only supported BERT, ALBERT and ELECTRA.
       threshold: float, optional (default=0.5)
           random selection for a word.
       top_p: float, optional (default=0.8)
           cumulative sum of probabilities to sample a word. 
           If top_n bigger than 0, the model will use nucleus sampling, else top-k sampling.
       top_k: int, optional (default=100)
           k for top-k sampling.
       temperature: float, optional (default=0.8)
           logits * temperature.
       top_n: int, (default=5)
           number of nearest neighbors returned. Length of returned result should as top_n.
       cleaning_function: function, (default=None)
           function to clean text.

       Returns
       -------
       result: List[str]
       """

.. code:: python

    electra = malaya.transformer.load(model = 'electra')


.. parsed-literal::

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
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:114: multinomial (from tensorflow.python.ops.random_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.random.categorical` instead.
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:118: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:120: The name tf.get_collection is deprecated. Please use tf.compat.v1.get_collection instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:121: The name tf.GraphKeys is deprecated. Please use tf.compat.v1.GraphKeys instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:127: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:129: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/electra-model/base/electra-base/model.ckpt


.. code:: python

    malaya.augmentation.transformer(text, electra)




.. parsed-literal::

    ['Perdana Menteri berkata , kerajaan sudah memperoleh maklumat terperinci berhubung masalah berkenaan supaya kerajaan dapat mengambil pelbagai tindakan sewajarnya . Bagaimanapun , beliau yakin masalah itu berjaya diselesaikan dan akhirnya terdahulu boleh diselesaikan dengan baik .',
     'Perdana Menteri berkata , kerajaan perlu memperoleh maklumat terperinci berhubung isu berkenaan supaya kerajaan dapat mengambil serius tindakan segera . Bagaimanapun , beliau berharap masalah itu boleh diselesaikan dan akhirnya kementerian boleh diselesaikan dengan baik .',
     'Perdana Menteri berkata , kerajaan telah memperoleh maklumat terperinci berhubung isu berkenaan supaya kerajaan dapat mengambil beberapa tindakan sewajarnya . Bagaimanapun , beliau berharap masalah itu perlu diselesaikan dan siasatan BN boleh diselesaikan dengan baik .',
     'Perdana Menteri berkata , kerajaan akan memperoleh maklumat terperinci berhubung isu berkenaan supaya kerajaan dapat mengambil sebarang tindakan susulan . Bagaimanapun , beliau mengharapkan masalah itu dapat diselesaikan dan membolehkan tidak boleh ditangani dengan baik .',
     'Perdana Menteri berkata , kerajaan sudah memperoleh maklumat terperinci berhubung isu berkenaan supaya kerajaan dapat mengambil sebarang tindakan lanjut . Bagaimanapun , beliau berharap masalah itu dapat diselesaikan dan hanya masih boleh diselesaikan dengan baik .']



