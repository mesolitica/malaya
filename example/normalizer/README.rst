
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 13.2 s, sys: 1.41 s, total: 14.6 s
    Wall time: 18.2 s


.. code:: ipython3

    string = 'y u xsuka makan HUSEIN kt situ tmpt'
    another = 'i mmg xska mknn HUSEIN kampng tempt'

Load basic normalizer
---------------------

.. code:: ipython3

    malaya.normalize.basic(string)




.. parsed-literal::

    'kenapa awak xsuka makan Husein kt situ tmpt'



Load fuzzy normalizer
---------------------

.. code:: ipython3

    malays = malaya.load_malay_dictionary()
    normalizer = malaya.normalize.fuzzy(malays)

.. code:: ipython3

    normalizer.normalize(string)




.. parsed-literal::

    'kenapa awak tak suka makan Husein kat situ tempat'



.. code:: ipython3

    normalizer.normalize(another)




.. parsed-literal::

    'saya memang tak saka makanan Husein kampung tempat'



Load deep expander
------------------

.. code:: ipython3

    wiki, ngrams = malaya.fast_text.load_wiki()
    fast_text_embed = malaya.fast_text.fast_text(wiki['embed_weights'],wiki['dictionary'],ngrams)
    normalizer = malaya.normalize.deep_expander(malays, fast_text_embed)

.. code:: ipython3

    normalizer.normalize(string)




.. parsed-literal::

    [[('tmpt',
       'kenapa awak tak suka makan Husein kat situ tut',
       0.8088938253521919),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tuit',
       0.863929785296917),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tat',
       0.8680638003787995),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ top',
       0.8688952446055412),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tip',
       0.8978437346220016),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ taat',
       0.936883625289917),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ topi',
       0.9442774548711776),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tumit',
       0.9495834815340042),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tempe',
       0.9758907731723786),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ ampe',
       0.9821926467533112),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tempo',
       0.9836614096956253),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tepet',
       0.994007917971611),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ amit',
       0.9999424153804779),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tuat',
       1.0002889167022706),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ mat',
       1.0071370331926346),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ temut',
       1.011553812426567),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ ampit',
       1.022653616695404),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ ampo',
       1.0231078831071854),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tipu',
       1.0246861065587998),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tepi',
       1.0285266551542283),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ umut',
       1.0287358275117875),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ emat',
       1.0357482937116622),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ empat',
       1.0431590774860382),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tapi',
       1.0562509994459153),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tepu',
       1.0601519473543166),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tumpat',
       1.074669928882599),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ impi',
       1.078846170501709),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ umat',
       1.0791117155513763),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tampi',
       1.0883281208925248),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tumpu',
       1.091578345676422),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ umpat',
       1.092372225769043),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tepat',
       1.0979607516746521),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tampa',
       1.1118229238204955),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ amput',
       1.1226389572820663),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tapa',
       1.129335333744049),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ timpa',
       1.1353471846590042),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ empu',
       1.1459274488725661),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tempa',
       1.164648480837822),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tampu',
       1.1812463180065156),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tempat',
       1.1856716803007126),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ tamat',
       1.2068403679332733),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ amat',
       1.2214121790246963),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ ampu',
       1.2350379461402894),
      ('tmpt',
       'kenapa awak tak suka makan Husein kat situ taut',
       1.2796957146606445)]]



.. code:: ipython3

    normalizer.normalize(another)




.. parsed-literal::

    [[('ska', 'saya memang tak soka mknn Husein kampng tempt', 0.7199365496635437),
      ('ska', 'saya memang tak suka mknn Husein kampng tempt', 0.8050327301025391),
      ('ska', 'saya memang tak sika mknn Husein kampng tempt', 0.8729341626167297),
      ('ska', 'saya memang tak saka mknn Husein kampng tempt', 0.875930666923523),
      ('ska', 'saya memang tak spa mknn Husein kampng tempt', 0.8995948433876038),
      ('ska', 'saya memang tak sua mknn Husein kampng tempt', 0.9496822357177734),
      ('ska', 'saya memang tak seka mknn Husein kampng tempt', 0.9891390204429626),
      ('ska', 'saya memang tak ski mknn Husein kampng tempt', 1.1318669319152832),
      ('ska', 'saya memang tak sia mknn Husein kampng tempt', 1.1666431427001953)],
     [('mknn', 'saya memang tak ska min Husein kampng tempt', 0.8653836846351624),
      ('mknn', 'saya memang tak ska maun Husein kampng tempt', 1.045318603515625),
      ('mknn', 'saya memang tak ska kun Husein kampng tempt', 1.0710314512252808),
      ('mknn', 'saya memang tak ska ken Husein kampng tempt', 1.0728274583816528),
      ('mknn', 'saya memang tak ska kon Husein kampng tempt', 1.0992072820663452),
      ('mknn', 'saya memang tak ska ikon Husein kampng tempt', 1.1365187168121338),
      ('mknn', 'saya memang tak ska makin Husein kampng tempt', 1.180336833000183),
      ('mknn', 'saya memang tak ska main Husein kampng tempt', 1.182568907737732),
      ('mknn', 'saya memang tak ska makan Husein kampng tempt', 1.183489203453064),
      ('mknn', 'saya memang tak ska makna Husein kampng tempt', 1.184565544128418),
      ('mknn', 'saya memang tak ska kan Husein kampng tempt', 1.2368937730789185),
      ('mknn', 'saya memang tak ska akan Husein kampng tempt', 1.2527291774749756),
      ('mknn', 'saya memang tak ska mani Husein kampng tempt', 1.266147494316101),
      ('mknn', 'saya memang tak ska ikan Husein kampng tempt', 1.2773109674453735),
      ('mknn', 'saya memang tak ska mini Husein kampng tempt', 1.3020210266113281),
      ('mknn', 'saya memang tak ska mana Husein kampng tempt', 1.3099677562713623),
      ('mknn', 'saya memang tak ska menu Husein kampng tempt', 1.3974181413650513),
      ('mknn', 'saya memang tak ska mena Husein kampng tempt', 1.404064655303955),
      ('mknn',
       'saya memang tak ska makanan Husein kampng tempt',
       1.4473483562469482)],
     [('kampng',
       'saya memang tak ska mknn Husein kampung tempt',
       0.9272603988647461)],
     [('tempt',
       'saya memang tak ska mknn Husein kampng tempo',
       0.7405402660369873),
      ('tempt',
       'saya memang tak ska mknn Husein kampng tempe',
       0.7510019540786743),
      ('tempt', 'saya memang tak ska mknn Husein kampng tempa', 0.885798454284668),
      ('tempt',
       'saya memang tak ska mknn Husein kampng temut',
       0.9036741256713867),
      ('tempt',
       'saya memang tak ska mknn Husein kampng tempat',
       0.9161624312400818)]]



``deep_expander`` will suggest nearest distance based on Word Mover
distance. This distance algorithm really depends on vector definition.

Load spell normalizer
---------------------

.. code:: ipython3

    normalizer = malaya.normalize.spell(malays)

To list all selected words during normalize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    normalizer.normalize(string,debug=True)


.. parsed-literal::

    [(('tepu', False), 10), (('tuat', False), 15), (('taut', False), 15), (('tapa', False), 10), (('timpa', False), 20), (('tampi', True), 15), (('tumpat', True), 24), (('temut', False), 15), (('tut', False), 16), (('umpat', True), 20), (('amat', False), 21), (('top', False), 11), (('ampit', False), 20), (('tampa', False), 15), (('tipu', False), 15), (('tat', False), 11), (('tepi', False), 10), (('ampu', False), 15), (('impi', False), 15), (('umut', False), 21), (('umat', False), 21), (('amit', False), 21), (('tampu', False), 15), (('tumit', False), 20), (('tempa', False), 15), (('tempat', False), 20), (('empu', False), 10), (('tapi', False), 10), (('topi', False), 10), (('tempo', False), 15), (('tuit', False), 15), (('tip', False), 16), (('tamat', False), 20), (('tepet', False), 15), (('tepat', False), 15), (('amput', False), 20), (('mat', False), 16), (('tumpu', False), 20), (('tempe', False), 15), (('emat', False), 15), (('ampo', False), 15), (('empat', True), 15), (('ampe', False), 15), (('taat', False), 15)] 
    




.. parsed-literal::

    'kenapa awak tak suka makan Husein kat situ amit'



Load deep learning
------------------

This model is not perfect, really suggest you to use other models.
Husein needs to read more!

.. code:: ipython3

    normalizer = malaya.normalize.deep_model()
    normalizer.normalize(string)




.. parsed-literal::

    'eye uau tak suka makan unsein kati situ tumpat'


