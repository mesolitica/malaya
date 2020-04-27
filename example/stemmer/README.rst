.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 4.85 s, sys: 1.29 s, total: 6.14 s
    Wall time: 7.76 s


Use deep learning model
-----------------------

.. code:: ipython3

    model = malaya.stem.deep_model()


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:54: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:55: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:49: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


.. code:: ipython3

    string = 'Benda yg SALAH ni, jgn lah didebatkan. Yg SALAH xkan jadi betul. Ingat tu. Mcm mana kesat sekalipun org sampaikan mesej, dan memang benda tu salah, diam je. Xyah nk tunjuk kau open sangat nk tegur cara org lain berdakwah'
    another_string = 'melayu bodoh, dah la gay, sokong lgbt lagi, memang tak guna, http://twitter.com'

.. code:: ipython3

    model.stem(string)




.. parsed-literal::

    'Benda yg SALAH ni , jgn lah debat . Yg SALAH xkan jadi betul . Ingat tu . Mcm mana kesat sekalipun org sampai mesej , dan memang benda tu salah , diam je . Xyah nk tunjuk kau open sangat nk tegur cara org lain dakwah'



.. code:: ipython3

    model.stem(another_string)




.. parsed-literal::

    'layu bodoh , dah la gay , sokong lgbt lagi , memang tak guna , http://twitter.com'



.. code:: ipython3

    model.stem('saya menyerukanlah')




.. parsed-literal::

    'saya seru'



Use Sastrawi stemmer
--------------------

Malaya also included interface for `Sastrawi
stemmer <https://pypi.org/project/PySastrawi/>`__. We also use it for
internal purpose. To use it, simply,

.. code:: python

   malaya.stem.sastrawi(str)

But it not able to maintain words like url, hashtag, money, datetime and
user mention.

.. code:: ipython3

    malaya.stem.sastrawi(another_string)




.. parsed-literal::

    'melayu bodoh dah la gay sokong lgbt lagi memang tak guna http twitter com'



.. code:: ipython3

    malaya.stem.sastrawi('saya menyerukanlah')




.. parsed-literal::

    'saya seru'



.. code:: ipython3

    malaya.stem.sastrawi('menarik')




.. parsed-literal::

    'tarik'


