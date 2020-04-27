.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 5.1 s, sys: 1.3 s, total: 6.4 s
    Wall time: 7.82 s


Describe supported dependencies
-------------------------------

.. code:: python

    malaya.dependency.describe()


.. parsed-literal::

    acl - clausal modifier of noun
    advcl - adverbial clause modifier
    advmod - adverbial modifier
    amod - adjectival modifier
    appos - appositional modifier
    aux - auxiliary
    case - case marking
    ccomp - clausal complement
    compound - compound
    compound:plur - plural compound
    conj - conjunct
    cop - cop
    csubj - clausal subject
    dep - dependent
    det - determiner
    fixed - multi-word expression
    flat - name
    iobj - indirect object
    mark - marker
    nmod - nominal modifier
    nsubj - nominal subject
    obj - direct object
    parataxis - parataxis
    root - root
    xcomp - open clausal complement
    you can read more from https://universaldependencies.org/treebanks/id_pud/index.html


.. code:: python

    string = 'Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar sekiranya mengantuk ketika memandu.'

List available transformer Dependency models
--------------------------------------------

.. code:: python

    malaya.dependency.available_transformer_model()




.. parsed-literal::

    {'bert': ['426.0 MB',
      'arc accuracy: 0.855',
      'types accuracy: 0.848',
      'root accuracy: 0.920'],
     'tiny-bert': ['59.5 MB',
      'arc accuracy: 0.718',
      'types accuracy: 0.694',
      'root accuracy: 0.886'],
     'albert': ['50.0 MB',
      'arc accuracy: 0.811',
      'types accuracy: 0.793',
      'root accuracy: 0.879'],
     'tiny-albert': ['24.8 MB',
      'arc accuracy: 0.708',
      'types accuracy: 0.673',
      'root accuracy: 0.817'],
     'xlnet': ['450.2 MB',
      'arc accuracy: 0.931',
      'types accuracy: 0.925',
      'root accuracy: 0.947'],
     'alxlnet': ['50.0 MB',
      'arc accuracy: 0.894',
      'types accuracy: 0.886',
      'root accuracy: 0.942']}



Make sure you can check accuracy chart from here first before select a
model,
https://malaya.readthedocs.io/en/latest/Accuracy.html#dependency-parsing

**The best model in term of accuracy is XLNET**.

Load xlnet dependency model
---------------------------

.. code:: python

    model = malaya.dependency.transformer(model = 'xlnet')


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:54: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:55: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:49: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


.. code:: python

    d_object, tagging, indexing = model.predict(string)
    d_object.to_graphvis()




.. image:: load-dependency_files/load-dependency_9_0.svg



Voting stack model
------------------

.. code:: python

    alxlnet = malaya.dependency.transformer(model = 'alxlnet')
    tagging, indexing = malaya.stack.voting_stack([model, alxlnet, model], string)
    malaya.dependency.dependency_graph(tagging, indexing).to_graphvis()


.. parsed-literal::

    downloading frozen /Users/huseinzolkepli/Malaya/dependency/alxlnet/base model


.. parsed-literal::

    51.0MB [00:50, 1.01MB/s]                          




.. image:: load-dependency_files/load-dependency_11_2.svg



Dependency graph object
-----------------------

To initiate a dependency graph from dependency models, you need to call
``malaya.dependency.dependency_graph``.

.. code:: python

    graph = malaya.dependency.dependency_graph(tagging, indexing)
    graph




.. parsed-literal::

    <malaya.function.parse_dependency.DependencyGraph at 0x164e67e90>



generate graphvis
^^^^^^^^^^^^^^^^^

.. code:: python

    graph.to_graphvis()




.. image:: load-dependency_files/load-dependency_15_0.svg



Get nodes
^^^^^^^^^

.. code:: python

    graph.nodes




.. parsed-literal::

    defaultdict(<function malaya.function.parse_dependency.DependencyGraph.__init__.<locals>.<lambda>()>,
                {0: {'address': 0,
                  'word': None,
                  'lemma': None,
                  'ctag': 'TOP',
                  'tag': 'TOP',
                  'feats': None,
                  'head': None,
                  'deps': defaultdict(list, {'root': [3]}),
                  'rel': None},
                 1: {'address': 1,
                  'word': 'Dr',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 3,
                  'deps': defaultdict(list, {'flat': [2]}),
                  'rel': 'nsubj'},
                 3: {'address': 3,
                  'word': 'menasihati',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 0,
                  'deps': defaultdict(list,
                              {'nsubj': [1], 'obj': [4], 'ccomp': [6]}),
                  'rel': 'root'},
                 2: {'address': 2,
                  'word': 'Mahathir',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 1,
                  'deps': defaultdict(list, {}),
                  'rel': 'flat'},
                 4: {'address': 4,
                  'word': 'mereka',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 3,
                  'deps': defaultdict(list, {}),
                  'rel': 'obj'},
                 5: {'address': 5,
                  'word': 'supaya',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 6,
                  'deps': defaultdict(list, {}),
                  'rel': 'case'},
                 6: {'address': 6,
                  'word': 'berhenti',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 3,
                  'deps': defaultdict(list,
                              {'case': [5], 'ccomp': [7], 'conj': [9]}),
                  'rel': 'ccomp'},
                 7: {'address': 7,
                  'word': 'berehat',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 6,
                  'deps': defaultdict(list, {}),
                  'rel': 'ccomp'},
                 8: {'address': 8,
                  'word': 'dan',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 9,
                  'deps': defaultdict(list, {}),
                  'rel': 'cc'},
                 9: {'address': 9,
                  'word': 'tidur',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 6,
                  'deps': defaultdict(list,
                              {'cc': [8],
                               'advmod': [10],
                               'amod': [12],
                               'advcl': [14]}),
                  'rel': 'conj'},
                 10: {'address': 10,
                  'word': 'sebentar',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 9,
                  'deps': defaultdict(list, {}),
                  'rel': 'advmod'},
                 11: {'address': 11,
                  'word': 'sekiranya',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 12,
                  'deps': defaultdict(list, {}),
                  'rel': 'advmod'},
                 12: {'address': 12,
                  'word': 'mengantuk',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 9,
                  'deps': defaultdict(list, {'advmod': [11]}),
                  'rel': 'amod'},
                 13: {'address': 13,
                  'word': 'ketika',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 14,
                  'deps': defaultdict(list, {}),
                  'rel': 'case'},
                 14: {'address': 14,
                  'word': 'memandu.',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 9,
                  'deps': defaultdict(list, {'case': [13]}),
                  'rel': 'advcl'}})



Flat the graph
^^^^^^^^^^^^^^

.. code:: python

    list(graph.triples())




.. parsed-literal::

    [(('menasihati', '_'), 'nsubj', ('Dr', '_')),
     (('Dr', '_'), 'flat', ('Mahathir', '_')),
     (('menasihati', '_'), 'obj', ('mereka', '_')),
     (('menasihati', '_'), 'ccomp', ('berhenti', '_')),
     (('berhenti', '_'), 'case', ('supaya', '_')),
     (('berhenti', '_'), 'ccomp', ('berehat', '_')),
     (('berhenti', '_'), 'conj', ('tidur', '_')),
     (('tidur', '_'), 'cc', ('dan', '_')),
     (('tidur', '_'), 'advmod', ('sebentar', '_')),
     (('tidur', '_'), 'amod', ('mengantuk', '_')),
     (('mengantuk', '_'), 'advmod', ('sekiranya', '_')),
     (('tidur', '_'), 'advcl', ('memandu.', '_')),
     (('memandu.', '_'), 'case', ('ketika', '_'))]



Check the graph contains cycles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    graph.contains_cycle()




.. parsed-literal::

    False



Generate networkx
^^^^^^^^^^^^^^^^^

Make sure you already installed networkx, ``pip install networkx``

.. code:: python

    digraph = graph.to_networkx()
    digraph




.. parsed-literal::

    <networkx.classes.multidigraph.MultiDiGraph at 0x1a875a110>



.. code:: python

    import networkx as nx
    import matplotlib.pyplot as plt
    nx.draw_networkx(digraph)
    plt.show()



.. parsed-literal::

    <Figure size 640x480 with 1 Axes>


.. code:: python

    digraph.edges()




.. parsed-literal::

    OutMultiEdgeDataView([(1, 3), (2, 1), (4, 3), (5, 6), (6, 3), (7, 6), (8, 9), (9, 6), (10, 9), (11, 12), (12, 9), (13, 14), (14, 9)])



.. code:: python

    digraph.nodes()




.. parsed-literal::

    NodeView((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))



.. code:: python

    labels = {i:graph.get_by_address(i)['word'] for i in digraph.nodes()}
    labels




.. parsed-literal::

    {1: 'Dr',
     2: 'Mahathir',
     3: 'menasihati',
     4: 'mereka',
     5: 'supaya',
     6: 'berhenti',
     7: 'berehat',
     8: 'dan',
     9: 'tidur',
     10: 'sebentar',
     11: 'sekiranya',
     12: 'mengantuk',
     13: 'ketika',
     14: 'memandu.'}



.. code:: python

    plt.figure(figsize=(15,5))
    nx.draw_networkx(digraph,labels=labels)
    plt.show()



.. image:: load-dependency_files/load-dependency_28_0.png


