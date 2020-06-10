.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 4.9 s, sys: 1.26 s, total: 6.16 s
    Wall time: 7.39 s


what is zero-shot classification
--------------------------------

Commonly we supervised a machine learning on specific labels, negative /
positive for sentiment, anger / happy / sadness for emotion and etc. The
model cannot give an output if we want to know how much percentage of
‘jealous’ in emotion analysis model because supported labels are only
{anger, happy, sadness}. Imagine, for example, trying to identify a text
without ever having seen one ‘jealous’ label before, impossible. **So,
zero-shot trying to solve this problem.**

zero-shot learning refers to the process by which a machine learns how
to recognize objects (image, text, any features) without any labeled
training data to help in the classification.

`Yin et al. (2019) <https://arxiv.org/abs/1909.00161>`__ stated in his
paper, any pretrained language model finetuned on text similarity
actually can acted as an out-of-the-box zero-shot text classifier.

So, we are going to use transformer models from
``malaya.similarity.transformer`` with a little tweaks.

List available Transformer models
---------------------------------

.. code:: python

    malaya.zero_shot.classification.available_transformer()




.. parsed-literal::

    {'bert': ['423.4 MB', 'accuracy: 0.885'],
     'tiny-bert': ['56.6 MB', 'accuracy: 0.873'],
     'albert': ['46.3 MB', 'accuracy: 0.873'],
     'tiny-albert': ['21.9 MB', 'accuracy: 0.824'],
     'xlnet': ['448.7 MB', 'accuracy: 0.784'],
     'alxlnet': ['49.0 MB', 'accuracy: 0.888']}



We trained on `Quora Question
Pairs <https://github.com/huseinzol05/Malay-Dataset#quora>`__,
`translated SNLI <https://github.com/huseinzol05/Malay-Dataset#snli>`__
and `translated
MNLI <https://github.com/huseinzol05/Malay-Dataset#mnli>`__

Make sure you can check accuracy chart from here first before select a
model, https://malaya.readthedocs.io/en/latest/Accuracy.html#similarity

**You might want to use ALXLNET, a very small size, 49MB, but the
accuracy is still on the top notch.**

Load transformer model
----------------------

In this example, I am going to load ``alxlnet``, feel free to use any
available models above.

.. code:: python

    model = malaya.zero_shot.classification.transformer(model = 'alxlnet')


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:54: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:55: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:49: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


predict batch
^^^^^^^^^^^^^

.. code:: python

   def predict_proba(self, strings: List[str], labels: List[str]):
       """
       classify list of strings and return probability.

       Parameters
       ----------
       strings : List[str]
       labels : List[str]

       Returns
       -------
       list: list of float
       """

Because it is a zero-shot, we need to give labels for the model.

.. code:: python

    # copy from twitter
    
    string = 'gov macam bengong, kami nk pilihan raya, gov backdoor, sakai'

.. code:: python

    model.predict_proba([string], labels = ['najib razak', 'mahathir', 'kerajaan', 'PRU', 'anarki'])




.. parsed-literal::

    [{'najib razak': 0.011697772,
      'mahathir': 0.030579083,
      'kerajaan': 0.038274202,
      'PRU': 0.74709743,
      'anarki': 0.054001417}]



Quite good.

.. code:: python

    string = 'tolong order foodpanda jab, lapar'

.. code:: python

    model.predict_proba([string], labels = ['makan', 'makanan', 'novel', 'buku', 'kerajaan', 'food delivery'])




.. parsed-literal::

    [{'makan': 0.4262973,
      'makanan': 0.94525576,
      'novel': 0.0016873145,
      'buku': 0.00282516,
      'kerajaan': 0.0013985565,
      'food delivery': 0.9190869}]



the model understood ``order foodpanda`` got close relationship with
``makan``, ``makanan`` and ``food delivery``.

.. code:: python

    string = 'kerajaan sebenarnya sangat prihatin dengan rakyat, bagi duit bantuan'

.. code:: python

    model.predict_proba([string], labels = ['makan', 'makanan', 'novel', 'buku', 'kerajaan', 'food delivery',
                                           'kerajaan jahat', 'kerajaan prihatin', 'bantuan rakyat'])




.. parsed-literal::

    [{'makan': 0.0010322841,
      'makanan': 0.0059771817,
      'novel': 0.0068290858,
      'buku': 0.00083946186,
      'kerajaan': 0.9823078,
      'food delivery': 0.017137317,
      'kerajaan jahat': 0.4863779,
      'kerajaan prihatin': 0.96803045,
      'bantuan rakyat': 0.94919217}]



Stacking models
---------------

More information, you can read at
https://malaya.readthedocs.io/en/latest/Stack.html

If you want to stack zero-shot classification models, you need to pass
labels using keyword parameter,

.. code:: python

   malaya.stack.predict_stack([model1, model2], List[str], labels = List[str])

We will passed ``labels`` as ``**kwargs``.

.. code:: python

    alxlnet = malaya.zero_shot.classification.transformer(model = 'alxlnet')
    albert = malaya.zero_shot.classification.transformer(model = 'albert')
    tiny_bert = malaya.zero_shot.classification.transformer(model = 'tiny-bert')


.. parsed-literal::

    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/albert/tokenization.py:240: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.
    
    INFO:tensorflow:loading sentence piece model


.. code:: python

    string = 'kerajaan sebenarnya sangat prihatin dengan rakyat, bagi duit bantuan'
    labels = ['makan', 'makanan', 'novel', 'buku', 'kerajaan', 'food delivery', 
     'kerajaan jahat', 'kerajaan prihatin', 'bantuan rakyat']
    malaya.stack.predict_stack([alxlnet, albert, tiny_bert], [string], 
                               labels = labels)




.. parsed-literal::

    [{'makan': 0.0044827852,
      'makanan': 0.0027062024,
      'novel': 0.0020867025,
      'buku': 0.013082165,
      'kerajaan': 0.8859287,
      'food delivery': 0.0028363755,
      'kerajaan jahat': 0.018133936,
      'kerajaan prihatin': 0.9922408,
      'bantuan rakyat': 0.909674}]



