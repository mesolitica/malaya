
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 6.4 s, sys: 1.6 s, total: 7.99 s
    Wall time: 12.8 s


Describe supported dependencies
-------------------------------

.. code:: ipython3

    malaya.describe_dependency()


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


.. code:: ipython3

    string = 'Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar sekiranya mengantuk ketika memandu.'

List available transformer Dependency models
--------------------------------------------

.. code:: ipython3

    malaya.dependency.available_transformer_model()




.. parsed-literal::

    {'bert': ['base'], 'xlnet': ['base'], 'albert': ['base']}



Make sure you can check accuracy chart from here first before select a
model,
https://malaya.readthedocs.io/en/latest/Accuracy.html#dependency-parsing

**The best model in term of accuracy is XLNET**.

Load xlnet dependency model
---------------------------

.. code:: ipython3

    model = malaya.dependency.transformer(model = 'xlnet', size = 'base')


.. parsed-literal::

    WARNING: Logging before flag parsing goes to stderr.
    W1017 22:05:38.566553 4363621824 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:68: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    W1017 22:05:38.568574 4363621824 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:69: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    W1017 22:05:43.521211 4363621824 deprecation_wrapper.py:119] From /Users/huseinzol/Documents/Malaya/malaya/_utils/_utils.py:64: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


.. code:: ipython3

    d_object, tagging, indexing = model.predict(string)
    d_object.to_graphvis()




.. image:: load-dependency_files/load-dependency_9_0.svg



Voting stack model
------------------

.. code:: ipython3

    bert = malaya.dependency.transformer(model = 'bert', size = 'base')
    tagging, indexing = malaya.stack.voting_stack([model, bert, model], string)
    malaya.dependency.dependency_graph(tagging, indexing).to_graphvis()




.. image:: load-dependency_files/load-dependency_11_0.svg



Dependency graph object
-----------------------

To initiate a dependency graph from dependency models, you need to call
``malaya.dependency.dependency_graph``.

.. code:: ipython3

    graph = malaya.dependency.dependency_graph(tagging, indexing)
    graph




.. parsed-literal::

    <malaya._utils._parse_dependency.DependencyGraph at 0x142fddf28>



generate graphvis
^^^^^^^^^^^^^^^^^

.. code:: ipython3

    graph.to_graphvis()




.. image:: load-dependency_files/load-dependency_15_0.svg



Get nodes
^^^^^^^^^

.. code:: ipython3

    graph.nodes




.. parsed-literal::

    defaultdict(<function malaya._utils._parse_dependency.DependencyGraph.__init__.<locals>.<lambda>()>,
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
                  'deps': defaultdict(list, {}),
                  'rel': 'nsubj'},
                 3: {'address': 3,
                  'word': 'menasihati',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 0,
                  'deps': defaultdict(list,
                              {'nsubj': [1],
                               'flat': [2],
                               'obj': [4],
                               'xcomp': [6]}),
                  'rel': 'root'},
                 2: {'address': 2,
                  'word': 'Mahathir',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 3,
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
                              {'case': [5], 'advmod': [7], 'conj': [9]}),
                  'rel': 'xcomp'},
                 7: {'address': 7,
                  'word': 'berehat',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 6,
                  'deps': defaultdict(list, {}),
                  'rel': 'advmod'},
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
                              {'cc': [8], 'advmod': [10], 'amod': [12]}),
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
                  'deps': defaultdict(list, {'advmod': [11], 'amod': [14]}),
                  'rel': 'amod'},
                 13: {'address': 13,
                  'word': 'ketika',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 13,
                  'deps': defaultdict(list, {'mark': [13]}),
                  'rel': 'mark'},
                 14: {'address': 14,
                  'word': 'memandu.',
                  'lemma': '_',
                  'ctag': '_',
                  'tag': '_',
                  'feats': '_',
                  'head': 12,
                  'deps': defaultdict(list, {}),
                  'rel': 'amod'}})



Flat the graph
^^^^^^^^^^^^^^

.. code:: ipython3

    list(graph.triples())




.. parsed-literal::

    [(('menasihati', '_'), 'nsubj', ('Dr', '_')),
     (('menasihati', '_'), 'flat', ('Mahathir', '_')),
     (('menasihati', '_'), 'obj', ('mereka', '_')),
     (('menasihati', '_'), 'xcomp', ('berhenti', '_')),
     (('berhenti', '_'), 'case', ('supaya', '_')),
     (('berhenti', '_'), 'advmod', ('berehat', '_')),
     (('berhenti', '_'), 'conj', ('tidur', '_')),
     (('tidur', '_'), 'cc', ('dan', '_')),
     (('tidur', '_'), 'advmod', ('sebentar', '_')),
     (('tidur', '_'), 'amod', ('mengantuk', '_')),
     (('mengantuk', '_'), 'advmod', ('sekiranya', '_')),
     (('mengantuk', '_'), 'amod', ('memandu.', '_'))]



Check the graph contains cycles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    graph.contains_cycle()




.. parsed-literal::

    False



Generate networkx
^^^^^^^^^^^^^^^^^

Make sure you already installed networkx, ``pip install networkx``

.. code:: ipython3

    digraph = graph.to_networkx()
    digraph




.. parsed-literal::

    <networkx.classes.multidigraph.MultiDiGraph at 0x142fec1d0>



.. code:: ipython3

    import networkx as nx
    import matplotlib.pyplot as plt
    nx.draw_networkx(digraph)
    plt.show()



.. parsed-literal::

    <Figure size 640x480 with 1 Axes>


.. code:: ipython3

    digraph.edges()




.. parsed-literal::

    OutMultiEdgeDataView([(1, 3), (2, 3), (4, 3), (5, 6), (6, 3), (7, 6), (8, 9), (9, 6), (10, 9), (11, 12), (12, 9), (13, 13), (14, 12)])



.. code:: ipython3

    digraph.nodes()




.. parsed-literal::

    NodeView((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))



.. code:: ipython3

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



.. code:: ipython3

    plt.figure(figsize=(15,5))
    nx.draw_networkx(digraph,labels=labels)
    plt.show()



.. image:: load-dependency_files/load-dependency_28_0.png

