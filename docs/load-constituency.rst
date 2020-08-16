.. code:: python

    %%time
    
    import malaya


.. parsed-literal::

    CPU times: user 4.75 s, sys: 1.24 s, total: 5.99 s
    Wall time: 7.27 s


what is constituency parsing
----------------------------

Assign a sentence into its own syntactic structure, defined by certain
standardization. For example,

.. code:: python

    from IPython.core.display import Image, display
    
    display(Image('constituency.png', width=500))



.. image:: load-constituency_files/load-constituency_2_0.png
   :width: 500px


Read more at Stanford notes,
https://web.stanford.edu/~jurafsky/slp3/13.pdf

The context free grammar totally depends on language, so for Bahasa, we
follow https://github.com/famrashel/idn-treebank

List available transformer Constituency models
----------------------------------------------

.. code:: python

    malaya.constituency.available_transformer()




.. parsed-literal::

    {'bert': ['470.0 MB',
      'Recall: 78.96',
      'Precision: 81.78',
      'FScore: 80.35',
      'CompleteMatch: 10.37',
      'TaggingAccuracy: 91.59'],
     'tiny-bert': ['125 MB',
      'Recall: 74.89',
      'Precision: 78.79',
      'FScore: 76.79',
      'CompleteMatch: 9.01',
      'TaggingAccuracy: 91.17'],
     'albert': ['180.0 MB',
      'Recall: 77.57',
      'Precision: 80.50',
      'FScore: 79.01',
      'CompleteMatch: 5.77',
      'TaggingAccuracy: 90.30'],
     'tiny-albert': ['56.7 MB',
      'Recall: 67.21',
      'Precision: 74.89',
      'FScore: 70.84',
      'CompleteMatch: 2.11',
      'TaggingAccuracy: 87.75'],
     'xlnet': ['498.0 MB',
      'Recall: 80.65',
      'Precision: 82.22',
      'FScore: 81.43',
      'CompleteMatch: 11.08',
      'TaggingAccuracy: 92.12']}



Make sure you can check accuracy chart from here first before select a
model,
https://malaya.readthedocs.io/en/latest/Accuracy.html#constituency-parsing

The best model in term of accuracy is **XLNET**.

.. code:: python

    string = 'Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar sekiranya mengantuk ketika memandu.'

Load xlnet constituency model
-----------------------------

.. code:: python

    model = malaya.constituency.transformer(model = 'xlnet')


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:73: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:75: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:68: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


Parse into NLTK Tree
^^^^^^^^^^^^^^^^^^^^

Make sure you already installed ``nltk``, if not, simply,

.. code:: bash

   pip install nltk

We preferred to parse into NLTK tree, so we can play around with
children / subtrees.

.. code:: python

    tree = model.parse_nltk_tree(string)

.. code:: python

    tree




.. image:: load-constituency_files/load-constituency_12_0.png



Parse into Tree
^^^^^^^^^^^^^^^

This is a simple Tree object defined at
`malaya.text.trees <https://github.com/huseinzol05/Malaya/blob/master/malaya/text/trees.py>`__.

.. code:: python

    tree = model.parse_tree(string)
