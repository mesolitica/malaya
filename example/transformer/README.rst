Malaya provided basic interface for Pretrained Transformer encoder
models, specific to Malay, local social media slang and Manglish
language, we called it Transformer-Bahasa. This interface not able us to
use it to do custom training.

If you want to download pretrained model for Transformer-Bahasa and use
it for custom transfer-learning, you can download it here,
https://github.com/huseinzol05/Malaya/tree/master/pretrained-model/,
some notebooks to help you get started.

Or you can simply use `hugging-face
transformers <https://huggingface.co/models?filter=malay>`__ to try
transformer models from Malaya, simply check available models from here,
https://huggingface.co/models?filter=malay

.. code:: ipython3

    from IPython.core.display import Image, display
    
    display(Image('huggingface.png', width=500))



.. image:: load-transformer_files/load-transformer_2_0.png
   :width: 500px


.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 4.93 s, sys: 1.31 s, total: 6.25 s
    Wall time: 8 s


list Transformer-Bahasa available
---------------------------------

.. code:: ipython3

    malaya.transformer.available_transformer()




.. parsed-literal::

    ['bert',
     'tiny-bert',
     'albert',
     'tiny-albert',
     'xlnet',
     'alxlnet',
     'electra',
     'small-electra']



1. ``bert`` - BERT architecture from google.

2. ``tiny-bert`` - BERT architecture from google with smaller
   parameters.

3. ``albert`` - ALBERT architecture from google.

4. ``tiny-albert`` - ALBERT architecture from google with smaller
   parameters.

5. ``xlnet`` - XLNET architecture from google.

6. ``alxlnet`` Malaya architecture, unpublished model, A-lite XLNET.

7. ``electra`` ELECTRA architecture from google.

8. ``small-electra`` ELECTRA architecture from google with smaller
   parameters.

.. code:: ipython3

    strings = ['Kerajaan galakkan rakyat naik public transport tapi parking kat lrt ada 15. Reserved utk staff rapid je dah berpuluh. Park kereta tepi jalan kang kene saman dgn majlis perbandaran. Kereta pulak senang kene curi. Cctv pun tak ada. Naik grab dah 5-10 ringgit tiap hari. Gampang juga',
               'Alaa Tun lek ahhh npe muka masam cmni kn agong kata usaha kerajaan terdahulu sejak selepas merdeka',
               "Orang ramai cakap nurse kerajaan garang. So i tell u this. Most of our local ppl will treat us as hamba abdi and they don't respect us as a nurse"]

Load XLNET-Bahasa
-----------------

.. code:: ipython3

    xlnet = malaya.transformer.load(model = 'xlnet')


.. parsed-literal::

    INFO:tensorflow:memory input None
    INFO:tensorflow:Use float type <dtype: 'float32'>
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



.. image:: load-transformer_files/load-transformer_24_0.png
   :width: 300px


**All attention models able to use these interfaces.**

Load ELECTRA-Bahasa
-------------------

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


