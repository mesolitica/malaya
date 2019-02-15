
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12 s, sys: 1.41 s, total: 13.4 s
    Wall time: 17.1 s


.. code:: python

    anger_text = 'aku cukup tak suka budak gemuk tu'
    fear_text = 'saya takut dengan hantu'
    joy_text = 'gembiranya hari ni, dapat jumpa crush'
    love_text = 'saya terlalu cintakan dia'
    sadness_text = 'kawan rapat aku putuskan hubungan'
    surprise_text = 'terharu aku harini, semua orang cakap selamat hari jadi'

All models got ``get_proba`` parameters. If True, it will returned
probability every classes. Else, it will return highest probability
class. **Default is True.**

Load multinomial model
----------------------

.. code:: python

    model = malaya.emotion.multinomial()
    print(model.predict(anger_text))
    print(model.predict(anger_text,get_proba=True))
    model.predict_batch([anger_text,
                        fear_text,
                        joy_text,
                        love_text,
                        sadness_text,
                        surprise_text])


.. parsed-literal::

    anger
    {'anger': 0.30367763926253094, 'fear': 0.16709964152193366, 'joy': 0.17026521921403184, 'love': 0.18405977732934192, 'sadness': 0.1388341895665479, 'surprise': 0.03606353310561458}




.. parsed-literal::

    ['anger', 'fear', 'joy', 'joy', 'sadness', 'fear']



Load xgb model
--------------

.. code:: python

    model = malaya.emotion.xgb()
    print(model.predict(anger_text))
    print(model.predict(anger_text,get_proba=True))
    model.predict_batch([anger_text,
                        fear_text,
                        joy_text,
                        love_text,
                        sadness_text,
                        surprise_text],get_proba=True)


.. parsed-literal::

    love
    {'anger': 0.22918181, 'fear': 0.089252785, 'joy': 0.1318236, 'love': 0.46476611, 'sadness': 0.07200217, 'surprise': 0.012973559}




.. parsed-literal::

    [{'anger': 0.22918181,
      'fear': 0.089252785,
      'joy': 0.1318236,
      'love': 0.46476611,
      'sadness': 0.07200217,
      'surprise': 0.012973559},
     {'anger': 0.013483193,
      'fear': 0.939588,
      'joy': 0.01674833,
      'love': 0.003220023,
      'sadness': 0.022906518,
      'surprise': 0.0040539484},
     {'anger': 0.10506946,
      'fear': 0.025150253,
      'joy': 0.725915,
      'love': 0.05211037,
      'sadness': 0.078554265,
      'surprise': 0.013200594},
     {'anger': 0.11640434,
      'fear': 0.097485565,
      'joy': 0.24893147,
      'love': 0.25440857,
      'sadness': 0.2650988,
      'surprise': 0.01767122},
     {'anger': 0.27124837,
      'fear': 0.15662362,
      'joy': 0.131251,
      'love': 0.022184724,
      'sadness': 0.41255626,
      'surprise': 0.006135965},
     {'anger': 0.07513438,
      'fear': 0.2525073,
      'joy': 0.024355419,
      'love': 0.002638406,
      'sadness': 0.0059716892,
      'surprise': 0.6393928}]



List available deep learning models
-----------------------------------

.. code:: python

    malaya.emotion.available_deep_model()




.. parsed-literal::

    ['fast-text',
     'hierarchical',
     'bahdanau',
     'luong',
     'bidirectional',
     'bert',
     'entity-network']



.. code:: python

    for i in malaya.emotion.available_deep_model():
        print('Testing %s model'%(i))
        model = malaya.emotion.deep_model(i)
        print(model.predict(anger_text))
        print(model.predict_batch([anger_text,
                        fear_text,
                        joy_text,
                        love_text,
                        sadness_text,
                        surprise_text]))
        print(model.predict_batch([anger_text,
                        fear_text,
                        joy_text,
                        love_text,
                        sadness_text,
                        surprise_text], get_proba = True))
        print()


.. parsed-literal::

    Testing fast-text model
    love
    ['love', 'fear', 'joy', 'love', 'sadness', 'surprise']
    [{'anger': 2.538603e-07, 'fear': 4.1372344e-13, 'joy': 1.0892472e-08, 'love': 0.99999976, 'sadness': 3.8994935e-16, 'surprise': 2.439655e-08}, {'anger': 4.4489467e-24, 'fear': 1.0, 'joy': 1.3903143e-28, 'love': 1.7920514e-33, 'sadness': 1.01771616e-26, 'surprise': 6.799581e-18}, {'anger': 9.583714e-26, 'fear': 1.5029816e-24, 'joy': 1.0, 'love': 3.7527533e-13, 'sadness': 8.348174e-24, 'surprise': 2.080897e-16}, {'anger': 1.7409228e-13, 'fear': 3.2279754e-12, 'joy': 0.0005876841, 'love': 0.9994123, 'sadness': 1.8902605e-11, 'surprise': 9.9256076e-11}, {'anger': 1.2737708e-11, 'fear': 5.882562e-10, 'joy': 9.112171e-13, 'love': 7.7659496e-20, 'sadness': 1.0, 'surprise': 1.6035637e-16}, {'anger': 5.5730725e-37, 'fear': 0.16033638, 'joy': 1.2999706e-30, 'love': 0.0, 'sadness': 0.0, 'surprise': 0.8396636}]

    Testing hierarchical model
    anger
    ['anger', 'fear', 'joy', 'joy', 'sadness', 'joy']
    [{'anger': 0.22394963, 'fear': 0.35022292, 'joy': 0.19895941, 'love': 0.013231089, 'sadness': 0.20033234, 'surprise': 0.013304558}, {'anger': 0.0056565125, 'fear': 0.9885886, 'joy': 0.0034398232, 'love': 0.00018917819, 'sadness': 0.0012037805, 'surprise': 0.00092218135}, {'anger': 0.01764421, 'fear': 0.01951682, 'joy': 0.8797468, 'love': 0.041130837, 'sadness': 0.013527576, 'surprise': 0.028433735}, {'anger': 0.028772388, 'fear': 0.07343067, 'joy': 0.48502314, 'love': 0.28668693, 'sadness': 0.10576224, 'surprise': 0.020324599}, {'anger': 0.021873059, 'fear': 0.014633018, 'joy': 0.01073073, 'love': 0.0012993184, 'sadness': 0.94936466, 'surprise': 0.0020992015}, {'anger': 0.020028168, 'fear': 0.17150529, 'joy': 0.3734562, 'love': 0.19241562, 'sadness': 0.008164915, 'surprise': 0.23442967}]

    Testing bahdanau model
    love
    ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    [{'anger': 0.53818357, 'fear': 0.14104106, 'joy': 0.010708541, 'love': 0.2570674, 'sadness': 0.047102023, 'surprise': 0.005897305}, {'anger': 0.0005677081, 'fear': 0.9770825, 'joy': 0.005677423, 'love': 0.0007302013, 'sadness': 0.0017472907, 'surprise': 0.014194911}, {'anger': 0.06975506, 'fear': 0.0069800974, 'joy': 0.5717373, 'love': 0.30618504, 'sadness': 0.011454151, 'surprise': 0.033888407}, {'anger': 0.0038130684, 'fear': 0.0053994465, 'joy': 0.10317592, 'love': 0.8656706, 'sadness': 0.0056833136, 'surprise': 0.016257582}, {'anger': 0.01122868, 'fear': 0.019208057, 'joy': 0.0024597098, 'love': 0.0002851458, 'sadness': 0.965973, 'surprise': 0.00084543176}, {'anger': 0.00083102344, 'fear': 0.23240082, 'joy': 0.033536877, 'love': 0.0011026214, 'sadness': 0.00037630452, 'surprise': 0.7317524}]

    Testing luong model
    love
    ['joy', 'fear', 'joy', 'sadness', 'sadness', 'surprise']
    [{'anger': 0.057855386, 'fear': 0.040447887, 'joy': 0.29915547, 'love': 0.5720974, 'sadness': 0.00927453, 'surprise': 0.02116932}, {'anger': 0.0063275485, 'fear': 0.9673098, 'joy': 0.0065225014, 'love': 0.0008387138, 'sadness': 0.00706696, 'surprise': 0.011934649}, {'anger': 0.0014677589, 'fear': 0.0020899512, 'joy': 0.88741183, 'love': 0.076111265, 'sadness': 0.0038936164, 'surprise': 0.029025558}, {'anger': 0.013268307, 'fear': 0.0035831807, 'joy': 0.056010414, 'love': 0.21701123, 'sadness': 0.69225526, 'surprise': 0.017871574}, {'anger': 0.0018013288, 'fear': 0.0012173079, 'joy': 5.611221e-05, 'love': 9.00831e-05, 'sadness': 0.9967213, 'surprise': 0.000113809925}, {'anger': 0.00015200193, 'fear': 0.36670414, 'joy': 0.0003732592, 'love': 0.00011813393, 'sadness': 0.000118975, 'surprise': 0.63253355}]

    Testing bidirectional model
    love
    ['fear', 'fear', 'anger', 'joy', 'sadness', 'surprise']
    [{'anger': 0.031539902, 'fear': 0.44634053, 'joy': 0.0022038615, 'love': 0.24390388, 'sadness': 0.00030186496, 'surprise': 0.27570996}, {'anger': 0.0028205896, 'fear': 0.9787958, 'joy': 0.016622344, 'love': 0.00041048063, 'sadness': 0.0004424488, 'surprise': 0.00090834824}, {'anger': 0.4523394, 'fear': 0.32489082, 'joy': 0.04712723, 'love': 0.01679146, 'sadness': 0.039135754, 'surprise': 0.1197153}, {'anger': 0.04196525, 'fear': 0.08604635, 'joy': 0.65291435, 'love': 0.049389884, 'sadness': 0.077201255, 'surprise': 0.09248292}, {'anger': 0.06327597, 'fear': 0.058998022, 'joy': 0.041568566, 'love': 0.002343863, 'sadness': 0.8224733, 'surprise': 0.011340328}, {'anger': 1.5136379e-05, 'fear': 0.002162331, 'joy': 3.5301118e-06, 'love': 0.006482973, 'sadness': 2.4173462e-06, 'surprise': 0.99133366}]

    Testing bert model
    anger
    ['anger', 'anger', 'anger', 'anger', 'anger', 'anger']
    [{'anger': 0.79530007, 'fear': 0.043149363, 'joy': 0.050191555, 'love': 0.0028053583, 'sadness': 0.1083552, 'surprise': 0.0001983959}, {'anger': 0.7761929, 'fear': 0.02267685, 'joy': 0.08533038, 'love': 0.019361326, 'sadness': 0.09622978, 'surprise': 0.00020885638}, {'anger': 0.724599, 'fear': 0.021534633, 'joy': 0.14938025, 'love': 0.009412263, 'sadness': 0.09488238, 'surprise': 0.0001914676}, {'anger': 0.8217926, 'fear': 0.009756618, 'joy': 0.061514165, 'love': 0.03527268, 'sadness': 0.07142815, 'surprise': 0.00023569519}, {'anger': 0.9093987, 'fear': 0.00811897, 'joy': 0.024754424, 'love': 0.003218321, 'sadness': 0.054415427, 'surprise': 9.422473e-05}, {'anger': 0.9215124, 'fear': 0.009484482, 'joy': 0.023237498, 'love': 0.0027847919, 'sadness': 0.042906344, 'surprise': 7.447611e-05}]

    Testing entity-network model
    joy
    ['joy', 'sadness', 'joy', 'sadness', 'sadness', 'joy']
    [{'anger': 0.11245817, 'fear': 0.09678851, 'joy': 0.29964533, 'love': 0.07372399, 'sadness': 0.26239878, 'surprise': 0.15498528}, {'anger': 0.12070423, 'fear': 0.13202831, 'joy': 0.22073878, 'love': 0.031163175, 'sadness': 0.3202514, 'surprise': 0.175114}, {'anger': 0.11448454, 'fear': 0.10408847, 'joy': 0.2848294, 'love': 0.059466686, 'sadness': 0.27815202, 'surprise': 0.1589789}, {'anger': 0.12346853, 'fear': 0.15664044, 'joy': 0.17575133, 'love': 0.019622162, 'sadness': 0.33732292, 'surprise': 0.18719462}, {'anger': 0.117459856, 'fear': 0.115517266, 'joy': 0.25831792, 'love': 0.044844825, 'sadness': 0.2980614, 'surprise': 0.16579871}, {'anger': 0.11082334, 'fear': 0.09062623, 'joy': 0.30381778, 'love': 0.097978726, 'sadness': 0.24158238, 'surprise': 0.15517157}]



Unsupervised important words learning
-------------------------------------

.. code:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set() # i just really like seaborn colors

We need to set ``get_proba`` become True to get the ‘attention’.

Visualizing bahdanau model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    model = malaya.emotion.deep_model('bahdanau')
    result = model.predict(surprise_text, get_proba = True)['attention']

    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-emotion_files/load-emotion_14_0.png


Visualizing luong model
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    model = malaya.emotion.deep_model('luong')
    result = model.predict(surprise_text, get_proba = True)['attention']

    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-emotion_files/load-emotion_16_0.png


Visualizing hierarchical model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    model = malaya.emotion.deep_model('hierarchical')
    result = model.predict(surprise_text, get_proba=True)['attention']

    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-emotion_files/load-emotion_18_0.png


Load Sparse deep learning models
--------------------------------

What happen if a word not included in the dictionary of the models? like
``setan``, what if ``setan`` appeared in text we want to classify? We
found this problem when classifying social media texts / posts. Words
used not really a vocabulary-based contextual.

Malaya will treat **unknown words** as ``<UNK>``, so, to solve this
problem, we need to use N-grams character based. Malaya chose tri-grams
until fifth-grams.

.. code:: python

   setan = ['set', 'eta', 'tan']

Sklearn provided easy interface to use n-grams, problem is, it is very
sparse, a lot of zeros and not memory efficient. Sklearn returned sparse
matrix for the result, lucky Tensorflow already provided some sparse
function.

.. code:: python

    malaya.emotion.available_sparse_deep_model()




.. parsed-literal::

    ['fast-text-char']



Right now Malaya only provide 1 sparse model, ``fast-text-char``. We
will try to evolve it.

.. code:: python

    sparse_model = malaya.emotion.sparse_deep_model()


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from /Users/huseinzol/Malaya/emotion/fast-text-char/model.ckpt


.. code:: python

    sparse_model.predict(sadness_text)




.. parsed-literal::

    'sadness'



.. code:: python

    sparse_model.predict_batch([anger_text,
                        fear_text,
                        joy_text,
                        love_text,
                        sadness_text,
                        surprise_text])




.. parsed-literal::

    ['love', 'fear', 'joy', 'love', 'sadness', 'fear']



.. code:: python

    sparse_model.predict_batch([anger_text,
                        fear_text,
                        joy_text,
                        love_text,
                        sadness_text,
                        surprise_text], get_proba = True)




.. parsed-literal::

    [{'anger': 0.07479232,
      'fear': 0.012134718,
      'joy': 0.034137156,
      'love': 0.85221285,
      'sadness': 0.006336733,
      'surprise': 0.020386234},
     {'anger': 1.6892743e-08,
      'fear': 0.99999964,
      'joy': 6.260633e-08,
      'love': 3.2111713e-10,
      'sadness': 3.542872e-08,
      'surprise': 2.2207877e-07},
     {'anger': 0.00012469916,
      'fear': 9.6892345e-06,
      'joy': 0.9917463,
      'love': 0.006561422,
      'sadness': 0.00040069615,
      'surprise': 0.0011572224},
     {'anger': 5.0021445e-05,
      'fear': 0.0010109642,
      'joy': 0.049688663,
      'love': 0.94577587,
      'sadness': 0.0032941191,
      'surprise': 0.00018034693},
     {'anger': 0.0010146926,
      'fear': 0.00020020001,
      'joy': 5.2909185e-05,
      'love': 2.640257e-06,
      'sadness': 0.99870074,
      'surprise': 2.8823646e-05},
     {'anger': 0.0057854424,
      'fear': 0.8317998,
      'joy': 0.017287944,
      'love': 0.008883897,
      'sadness': 0.0070799366,
      'surprise': 0.12916291}]
