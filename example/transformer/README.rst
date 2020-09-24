Transformer
===========

.. container:: alert alert-info

   This tutorial is available as an IPython notebook at
   `Malaya/example/transformer <https://github.com/huseinzol05/Malaya/tree/master/example/transformer>`__.

Malaya provided basic interface for Pretrained Transformer encoder
models, specific to Malay, local social media slang and Manglish
language, we called it Transformer-Bahasa. Below are the list of dataset
we pretrained,

Standard Bahasa dataset,

1. `Malay-dataset/dumping <https://github.com/huseinzol05/Malay-Dataset/tree/master/dumping>`__.
2. `Malay-dataset/pure-text <https://github.com/huseinzol05/Malay-Dataset/tree/master/pure-text>`__.

Bahasa social media,

1. `Malay-dataset/dumping/instagram <https://github.com/huseinzol05/Malay-Dataset/tree/master/dumping/instagram>`__.
2. `Malay-dataset/dumping/twitter <https://github.com/huseinzol05/Malay-Dataset/tree/master/dumping/twitter>`__.

Singlish / Manglish,

1. `Malay-dataset/dumping/singlish <https://github.com/huseinzol05/Malay-Dataset/tree/master/dumping/singlish-text>`__.
2. `Malay-dataset/dumping/singapore-news <https://github.com/huseinzol05/Malay-Dataset/tree/master/dumping/singapore-news>`__.

**This interface not able us to use it to do custom training**.

If you want to download pretrained model for Transformer-Bahasa and use
it for custom transfer-learning, you can download it here,
https://github.com/huseinzol05/Malaya/tree/master/pretrained-model/,
some notebooks to help you get started.

Or you can simply use `hugging-face
transformers <https://huggingface.co/models?filter=ms>`__ to try
transformer models from Malaya, simply check available models from here,
https://huggingface.co/models?filter=ms

.. code:: ipython3

    from IPython.core.display import Image, display
    
    display(Image('huggingface.png', width=500))



.. image:: load-transformer_files/load-transformer_4_0.png
   :width: 500px


.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 4.88 s, sys: 641 ms, total: 5.52 s
    Wall time: 4.5 s


list Transformer-Bahasa available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    malaya.transformer.available_transformer()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Size (MB)</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>bert</th>
          <td>425.6</td>
          <td>Google BERT BASE parameters</td>
        </tr>
        <tr>
          <th>tiny-bert</th>
          <td>57.4</td>
          <td>Google BERT TINY parameters</td>
        </tr>
        <tr>
          <th>albert</th>
          <td>48.6</td>
          <td>Google ALBERT BASE parameters</td>
        </tr>
        <tr>
          <th>tiny-albert</th>
          <td>22.4</td>
          <td>Google ALBERT TINY parameters</td>
        </tr>
        <tr>
          <th>xlnet</th>
          <td>446.6</td>
          <td>Google XLNET BASE parameters</td>
        </tr>
        <tr>
          <th>alxlnet</th>
          <td>46.8</td>
          <td>Malaya ALXLNET BASE parameters</td>
        </tr>
        <tr>
          <th>electra</th>
          <td>443</td>
          <td>Google ELECTRA BASE parameters</td>
        </tr>
        <tr>
          <th>small-electra</th>
          <td>55</td>
          <td>Google ELECTRA SMALL parameters</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    strings = ['Kerajaan galakkan rakyat naik public transport tapi parking kat lrt ada 15. Reserved utk staff rapid je dah berpuluh. Park kereta tepi jalan kang kene saman dgn majlis perbandaran. Kereta pulak senang kene curi. Cctv pun tak ada. Naik grab dah 5-10 ringgit tiap hari. Gampang juga',
               'Alaa Tun lek ahhh npe muka masam cmni kn agong kata usaha kerajaan terdahulu sejak selepas merdeka',
               "Orang ramai cakap nurse kerajaan garang. So i tell u this. Most of our local ppl will treat us as hamba abdi and they don't respect us as a nurse"]

Load XLNET-Bahasa
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    xlnet = malaya.transformer.load(model = 'xlnet')


.. parsed-literal::

    WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/malaya/transformers/xlnet/xlnet.py:70: The name tf.gfile.Open is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/malaya/transformers/xlnet/__init__.py:81: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/malaya/transformers/xlnet/xlnet.py:253: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
    
    WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/malaya/transformers/xlnet/xlnet.py:253: The name tf.AUTO_REUSE is deprecated. Please use tf.compat.v1.AUTO_REUSE instead.
    
    WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/malaya/transformers/xlnet/modeling.py:686: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.
    
    INFO:tensorflow:memory input None
    INFO:tensorflow:Use float type <dtype: 'float32'>
    WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/malaya/transformers/xlnet/modeling.py:693: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.
    
    WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/malaya/transformers/xlnet/modeling.py:797: dropout (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.dropout instead.
    WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/tensorflow_core/python/layers/core.py:271: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `layer.__call__` method instead.
    WARNING:tensorflow:
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
      * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.
    
    WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/malaya/transformers/xlnet/modeling.py:99: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use keras.layers.Dense instead.
    WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/malaya/transformers/xlnet/__init__.py:94: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    
    WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/malaya/transformers/xlnet/__init__.py:95: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
    
    WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/malaya/transformers/xlnet/__init__.py:96: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.
    
    WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/malaya/transformers/xlnet/__init__.py:100: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
    
    WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/malaya/transformers/xlnet/__init__.py:103: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/xlnet-model/base/xlnet-base/model.ckpt


I have random sentences copied from Twitter, searched using ``kerajaan``
keyword.

Vectorization
^^^^^^^^^^^^^

Change a string or batch of strings to latent space / vectors
representation.

.. code:: ipython3

    v = xlnet.vectorize(strings)
    v.shape




.. parsed-literal::

    (3, 768)



Attention
^^^^^^^^^

Attention is to get which part of the sentence give the impact. Method
available for attention,

-  ``'last'`` - attention from last layer.
-  ``'first'`` - attention from first layer.
-  ``'mean'`` - average attentions from all layers.

You can give list of strings or a string to get the attention, in this
documentation, I just want to use a string.

.. code:: ipython3

    xlnet.attention([strings[1]], method = 'last')




.. parsed-literal::

    [[('Alaa', 0.062061824),
      ('Tun', 0.051056776),
      ('lek', 0.13115405),
      ('ahhh', 0.08195943),
      ('npe', 0.06210695),
      ('muka', 0.04706182),
      ('masam', 0.058289353),
      ('cmni', 0.026094284),
      ('kn', 0.056146827),
      ('agong', 0.033949938),
      ('kata', 0.052644122),
      ('usaha', 0.07063393),
      ('kerajaan', 0.046773836),
      ('terdahulu', 0.057166394),
      ('sejak', 0.045712817),
      ('selepas', 0.047048207),
      ('merdeka', 0.07013944)]]



.. code:: ipython3

    xlnet.attention([strings[1]], method = 'first')




.. parsed-literal::

    [[('Alaa', 0.045956098),
      ('Tun', 0.040094823),
      ('lek', 0.0611072),
      ('ahhh', 0.07029096),
      ('npe', 0.048513662),
      ('muka', 0.056670234),
      ('masam', 0.04088071),
      ('cmni', 0.08728454),
      ('kn', 0.047778472),
      ('agong', 0.081243224),
      ('kata', 0.03866041),
      ('usaha', 0.058326427),
      ('kerajaan', 0.055446573),
      ('terdahulu', 0.077162124),
      ('sejak', 0.05951431),
      ('selepas', 0.05385498),
      ('merdeka', 0.07721528)]]



.. code:: ipython3

    xlnet.attention([strings[1]], method = 'mean')




.. parsed-literal::

    [[('Alaa', 0.06978634),
      ('Tun', 0.0517442),
      ('lek', 0.059642658),
      ('ahhh', 0.055883657),
      ('npe', 0.05339206),
      ('muka', 0.06806306),
      ('masam', 0.0489921),
      ('cmni', 0.0698193),
      ('kn', 0.057752036),
      ('agong', 0.065566674),
      ('kata', 0.059152905),
      ('usaha', 0.063305095),
      ('kerajaan', 0.050608452),
      ('terdahulu', 0.05888331),
      ('sejak', 0.057429556),
      ('selepas', 0.042058233),
      ('merdeka', 0.067920305)]]



Visualize Attention
^^^^^^^^^^^^^^^^^^^

Before using attention visualization, we need to load D3 into our
jupyter notebook first. This visualization borrow from
https://github.com/jessevig/bertviz .

.. code:: javascript

    %%javascript
    require.config({
      paths: {
          d3: '//cdnjs.cloudflare.com/ajax/libs/d3/3.4.8/d3.min',
          jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
      }
    });



.. parsed-literal::

    <IPython.core.display.Javascript object>


.. code:: ipython3

    xlnet.visualize_attention('nak makan ayam dgn husein')



.. raw:: html

    
    <span style="user-select:none">
      Layer: <select id="layer"></select>
    </span>
    <div id='vis'></div>




.. parsed-literal::

    <IPython.core.display.Javascript object>



.. parsed-literal::

    <IPython.core.display.Javascript object>


*I attached a printscreen, readthedocs cannot visualize the javascript.*

.. code:: ipython3

    from IPython.core.display import Image, display
    
    display(Image('xlnet-attention.png', width=300))



.. image:: load-transformer_files/load-transformer_25_0.png
   :width: 300px


**All attention models able to use these interfaces.**

Load ELECTRA-Bahasa
~~~~~~~~~~~~~~~~~~~

Feel free to use another models.

.. code:: ipython3

    electra = malaya.transformer.load(model = 'electra')


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
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:114: multinomial (from tensorflow.python.ops.random_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.random.categorical` instead.
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:117: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:118: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:120: The name tf.get_collection is deprecated. Please use tf.compat.v1.get_collection instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:121: The name tf.GraphKeys is deprecated. Please use tf.compat.v1.GraphKeys instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:127: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/transformers/electra/__init__.py:129: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    INFO:tensorflow:Restoring parameters from /Users/huseinzolkepli/Malaya/electra-model/base/electra-base/model.ckpt


.. code:: ipython3

    electra.attention([strings[1]], method = 'last')




.. parsed-literal::

    [[('Alaa', 0.059817147),
      ('Tun', 0.075028375),
      ('lek', 0.057848394),
      ('ahhh', 0.046973262),
      ('npe', 0.05160833),
      ('muka', 0.06221234),
      ('masam', 0.058585588),
      ('cmni', 0.054711323),
      ('kn', 0.06741887),
      ('agong', 0.056326747),
      ('kata', 0.054182768),
      ('usaha', 0.07986903),
      ('kerajaan', 0.05559596),
      ('terdahulu', 0.052879248),
      ('sejak', 0.049992196),
      ('selepas', 0.053916205),
      ('merdeka', 0.06303418)]]


