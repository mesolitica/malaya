
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.8 s, sys: 1.44 s, total: 14.2 s
    Wall time: 17.9 s


.. code:: ipython3

    anger_text = 'aku cukup tak suka budak gemuk tu'
    fear_text = 'saya takut dengan hantu'
    joy_text = 'gembiranya hari ni, dapat jumpa crush'
    love_text = 'saya terlalu cintakan dia'
    sadness_text = 'kawan rapat aku putuskan hubungan'
    surprise_text = 'terharu aku harini, semua orang cakap selamat hari jadi'

Load multinomial model
----------------------

.. code:: ipython3

    model = malaya.emotion.multinomial()
    print(model.predict(anger_text,get_proba=True))
    print(model.predict(anger_text,get_proba=True))
    model.predict_batch([anger_text,
                        fear_text,
                        joy_text,
                        love_text,
                        sadness_text,
                        surprise_text],get_proba=True)


.. parsed-literal::

    {'anger': 0.27993946463423486, 'fear': 0.1482931513658756, 'joy': 0.1880009584798728, 'love': 0.21711876657658918, 'sadness': 0.1296730712078804, 'surprise': 0.03697458773554805}
    {'anger': 0.27993946463423486, 'fear': 0.1482931513658756, 'joy': 0.1880009584798728, 'love': 0.21711876657658918, 'sadness': 0.1296730712078804, 'surprise': 0.03697458773554805}




.. parsed-literal::

    [{'anger': 0.27993946463423486,
      'fear': 0.1482931513658756,
      'joy': 0.1880009584798728,
      'love': 0.21711876657658918,
      'sadness': 0.1296730712078804,
      'surprise': 0.03697458773554805},
     {'anger': 0.021771118261238547,
      'fear': 0.8872341260634127,
      'joy': 0.03047156512889429,
      'love': 0.014657222836008465,
      'sadness': 0.031140846706145215,
      'surprise': 0.014725121004301253},
     {'anger': 0.0396561464598951,
      'fear': 0.035597422143487406,
      'joy': 0.8005688936302132,
      'love': 0.07202283399889338,
      'sadness': 0.038207026719665686,
      'surprise': 0.013947677047845667},
     {'anger': 0.13066816251128913,
      'fear': 0.13412252332273905,
      'joy': 0.310998514493142,
      'love': 0.2331246309240758,
      'sadness': 0.14322815141407566,
      'surprise': 0.047858017334676865},
     {'anger': 0.1527725857957935,
      'fear': 0.1365898809847078,
      'joy': 0.1754707153173015,
      'love': 0.06602453529124905,
      'sadness': 0.447275616555693,
      'surprise': 0.021866666055256587},
     {'anger': 0.13493174968699168,
      'fear': 0.47761628067734224,
      'joy': 0.18298570800638353,
      'love': 0.0732126431050908,
      'sadness': 0.06633124837003398,
      'surprise': 0.06492237015415897}]



Load xgb model
--------------

.. code:: ipython3

    model = malaya.emotion.xgb()
    print(model.predict(anger_text,get_proba=True))
    print(model.predict(anger_text,get_proba=True))
    model.predict_batch([anger_text,
                        fear_text,
                        joy_text,
                        love_text,
                        sadness_text,
                        surprise_text],get_proba=True)


.. parsed-literal::

    {'anger': 0.21755809, 'fear': 0.090371706, 'joy': 0.13347618, 'love': 0.47302967, 'sadness': 0.0770047, 'surprise': 0.008559667}
    {'anger': 0.21755809, 'fear': 0.090371706, 'joy': 0.13347618, 'love': 0.47302967, 'sadness': 0.0770047, 'surprise': 0.008559667}




.. parsed-literal::

    [{'anger': 0.21755809,
      'fear': 0.090371706,
      'joy': 0.13347618,
      'love': 0.47302967,
      'sadness': 0.0770047,
      'surprise': 0.008559667},
     {'anger': 0.013483193,
      'fear': 0.939588,
      'joy': 0.01674833,
      'love': 0.003220023,
      'sadness': 0.022906518,
      'surprise': 0.0040539484},
     {'anger': 0.09142393,
      'fear': 0.029400537,
      'joy': 0.78257465,
      'love': 0.02881839,
      'sadness': 0.058004435,
      'surprise': 0.009778041},
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
     {'anger': 0.0714585,
      'fear': 0.19790031,
      'joy': 0.037659157,
      'love': 0.0025473926,
      'sadness': 0.00772799,
      'surprise': 0.6827066}]



List available deep learning models
-----------------------------------

.. code:: ipython3

    malaya.emotion.available_deep_model()




.. parsed-literal::

    ['fast-text',
     'hierarchical',
     'bahdanau',
     'luong',
     'bidirectional',
     'bert',
     'entity-network']



.. code:: ipython3

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
        print()


.. parsed-literal::

    Testing fast-text model
    {'anger': 7.548095e-08, 'fear': 2.7052243e-13, 'joy': 2.273731e-12, 'love': 0.9999999, 'sadness': 3.0124282e-12, 'surprise': 1.9232537e-11}
    [{'anger': 2.978304e-06, 'fear': 1.8461518e-10, 'joy': 1.0204276e-09, 'love': 0.999997, 'sadness': 1.3693535e-09, 'surprise': 2.6386826e-09}, {'anger': 1.2210384e-18, 'fear': 1.0, 'joy': 1.0015556e-19, 'love': 1.8750202e-24, 'sadness': 6.976661e-21, 'surprise': 3.2600536e-15}, {'anger': 2.47199e-19, 'fear': 2.3032567e-22, 'joy': 1.0, 'love': 5.1478095e-14, 'sadness': 4.464682e-20, 'surprise': 1.588908e-15}, {'anger': 4.1249185e-11, 'fear': 1.7474476e-10, 'joy': 0.00022258118, 'love': 0.9997774, 'sadness': 1.6592432e-11, 'surprise': 4.1854236e-09}, {'anger': 4.3972154e-08, 'fear': 2.1118221e-06, 'joy': 3.4898858e-07, 'love': 4.5489975e-12, 'sadness': 0.9999975, 'surprise': 4.8414757e-09}, {'anger': 1.1130476e-23, 'fear': 0.0003273876, 'joy': 5.694222e-17, 'love': 1.9363045e-25, 'sadness': 1.4252974e-26, 'surprise': 0.99967265}]
    
    Testing hierarchical model
    {'anger': 0.21870466, 'fear': 0.0723397, 'joy': 0.25246364, 'love': 0.23216271, 'sadness': 0.11244629, 'surprise': 0.11188304, 'attention': [['aku', 0.19288461], ['cukup', 0.25843868], ['tak', 0.108033694], ['suka', 0.07043509], ['budak', 0.080554284], ['gemuk', 0.11672647], ['tu', 0.17292716]]}
    [{'anger': 0.41256806, 'fear': 0.13891418, 'joy': 0.13064316, 'love': 0.02137984, 'sadness': 0.28263724, 'surprise': 0.013857505}, {'anger': 0.006116035, 'fear': 0.9831093, 'joy': 0.0070863734, 'love': 0.00031651792, 'sadness': 0.0019696557, 'surprise': 0.0014021696}, {'anger': 0.0034238265, 'fear': 0.0028603936, 'joy': 0.9739377, 'love': 0.0059432993, 'sadness': 0.005335002, 'surprise': 0.008499798}, {'anger': 0.037748642, 'fear': 0.09834084, 'joy': 0.47098926, 'love': 0.25399926, 'sadness': 0.109694675, 'surprise': 0.029227324}, {'anger': 0.012461308, 'fear': 0.012679063, 'joy': 0.005910567, 'love': 0.0016031803, 'sadness': 0.96547556, 'surprise': 0.0018702189}, {'anger': 0.018574355, 'fear': 0.11572055, 'joy': 0.46149588, 'love': 0.21992558, 'sadness': 0.014294567, 'surprise': 0.16998905}]
    
    Testing bahdanau model
    {'anger': 0.2166525, 'fear': 0.025524562, 'joy': 0.0151598565, 'love': 0.71086437, 'sadness': 0.024761798, 'surprise': 0.007036926, 'attention': [['aku', 0.078082114], ['cukup', 0.06909147], ['tak', 0.047952086], ['suka', 0.24055175], ['budak', 0.13882484], ['gemuk', 0.38366398], ['tu', 0.041833848]]}
    [{'anger': 0.4382134, 'fear': 0.08935531, 'joy': 0.01401889, 'love': 0.41449428, 'sadness': 0.041711926, 'surprise': 0.002206273}, {'anger': 0.001070517, 'fear': 0.9847227, 'joy': 0.004814546, 'love': 0.00042387607, 'sadness': 0.0027839406, 'surprise': 0.006184482}, {'anger': 0.020542638, 'fear': 0.0024991473, 'joy': 0.83662474, 'love': 0.131418, 'sadness': 0.0009067217, 'surprise': 0.00800883}, {'anger': 0.0022841198, 'fear': 0.0043074405, 'joy': 0.06361495, 'love': 0.9130883, 'sadness': 0.006188171, 'surprise': 0.010517066}, {'anger': 0.03213852, 'fear': 0.049160797, 'joy': 0.0064736363, 'love': 0.0005698313, 'sadness': 0.9094809, 'surprise': 0.002176242}, {'anger': 0.0014524, 'fear': 0.30045557, 'joy': 0.030223431, 'love': 0.0032396903, 'sadness': 0.0008275905, 'surprise': 0.66380125}]
    
    Testing luong model
    {'anger': 0.0016777685, 'fear': 0.029341456, 'joy': 0.15955624, 'love': 0.7718347, 'sadness': 0.0005396353, 'surprise': 0.037050243, 'attention': [['aku', 0.22837706], ['cukup', 0.08437486], ['tak', 0.098626174], ['suka', 0.09288791], ['budak', 0.183754], ['gemuk', 0.15856884], ['tu', 0.15341121]]}
    [{'anger': 0.06511979, 'fear': 0.0786746, 'joy': 0.34283832, 'love': 0.47801033, 'sadness': 0.013086433, 'surprise': 0.022270598}, {'anger': 0.010881466, 'fear': 0.95245326, 'joy': 0.011857338, 'love': 0.001075954, 'sadness': 0.00922352, 'surprise': 0.0145085305}, {'anger': 0.0044609196, 'fear': 0.0004858748, 'joy': 0.9796047, 'love': 0.010732659, 'sadness': 0.0011361868, 'surprise': 0.0035795663}, {'anger': 0.018199386, 'fear': 0.01024426, 'joy': 0.06503831, 'love': 0.28378096, 'sadness': 0.60309285, 'surprise': 0.019644177}, {'anger': 0.0012908528, 'fear': 0.0015431962, 'joy': 0.00025829085, 'love': 0.0001731802, 'sadness': 0.99648786, 'surprise': 0.00024661073}, {'anger': 0.00016957898, 'fear': 0.28888798, 'joy': 0.00024510975, 'love': 0.00014600258, 'sadness': 0.00012586307, 'surprise': 0.7104255}]
    
    Testing bidirectional model
    {'anger': 0.2834959, 'fear': 0.25807646, 'joy': 0.0024420407, 'love': 0.04385324, 'sadness': 0.113708116, 'surprise': 0.29842427}
    [{'anger': 0.6034175, 'fear': 0.29230013, 'joy': 0.00010138063, 'love': 0.00661169, 'sadness': 0.0017976413, 'surprise': 0.09577158}, {'anger': 0.81299293, 'fear': 0.13775633, 'joy': 0.00064284174, 'love': 0.0014932072, 'sadness': 0.012328662, 'surprise': 0.03478606}, {'anger': 0.76455575, 'fear': 0.15335076, 'joy': 0.00013497456, 'love': 0.0037875765, 'sadness': 0.002890406, 'surprise': 0.07528054}, {'anger': 0.79790246, 'fear': 0.11822995, 'joy': 0.0012087246, 'love': 0.0027527318, 'sadness': 0.030970545, 'surprise': 0.048935644}, {'anger': 0.8012364, 'fear': 0.13785931, 'joy': 0.00030905145, 'love': 0.0021522753, 'sadness': 0.007178641, 'surprise': 0.051264346}, {'anger': 0.29625162, 'fear': 0.29746345, 'joy': 0.0005198121, 'love': 0.028240409, 'sadness': 0.028906224, 'surprise': 0.34861845}]
    
    Testing bert model
    {'anger': 0.7953001, 'fear': 0.043149337, 'joy': 0.050191533, 'love': 0.0028053573, 'sadness': 0.108355165, 'surprise': 0.00019839591}
    [{'anger': 0.79530007, 'fear': 0.043149363, 'joy': 0.050191555, 'love': 0.0028053583, 'sadness': 0.1083552, 'surprise': 0.0001983959}, {'anger': 0.7761929, 'fear': 0.02267685, 'joy': 0.08533038, 'love': 0.019361326, 'sadness': 0.09622978, 'surprise': 0.00020885638}, {'anger': 0.724599, 'fear': 0.021534633, 'joy': 0.14938025, 'love': 0.009412263, 'sadness': 0.09488238, 'surprise': 0.0001914676}, {'anger': 0.8217926, 'fear': 0.009756618, 'joy': 0.061514165, 'love': 0.03527268, 'sadness': 0.07142815, 'surprise': 0.00023569519}, {'anger': 0.9093987, 'fear': 0.00811897, 'joy': 0.024754424, 'love': 0.003218321, 'sadness': 0.054415427, 'surprise': 9.422473e-05}, {'anger': 0.9215124, 'fear': 0.009484482, 'joy': 0.023237498, 'love': 0.0027847919, 'sadness': 0.042906344, 'surprise': 7.447611e-05}]
    
    Testing entity-network model
    {'anger': 0.11245816, 'fear': 0.09678849, 'joy': 0.29964545, 'love': 0.07372402, 'sadness': 0.26239866, 'surprise': 0.1549853}
    [{'anger': 0.11245817, 'fear': 0.09678851, 'joy': 0.29964533, 'love': 0.07372399, 'sadness': 0.26239878, 'surprise': 0.15498528}, {'anger': 0.12070423, 'fear': 0.13202831, 'joy': 0.22073878, 'love': 0.031163175, 'sadness': 0.3202514, 'surprise': 0.175114}, {'anger': 0.11448454, 'fear': 0.10408847, 'joy': 0.2848294, 'love': 0.059466686, 'sadness': 0.27815202, 'surprise': 0.1589789}, {'anger': 0.12346853, 'fear': 0.15664044, 'joy': 0.17575133, 'love': 0.019622162, 'sadness': 0.33732292, 'surprise': 0.18719462}, {'anger': 0.117459856, 'fear': 0.115517266, 'joy': 0.25831792, 'love': 0.044844825, 'sadness': 0.2980614, 'surprise': 0.16579871}, {'anger': 0.11082334, 'fear': 0.09062623, 'joy': 0.30381778, 'love': 0.097978726, 'sadness': 0.24158238, 'surprise': 0.15517157}]
    


Unsupervised important words learning
-------------------------------------

.. code:: ipython3

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set() # i just really like seaborn colors

Visualizing bahdanau model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model = malaya.emotion.deep_model('bahdanau')
    result = model.predict(surprise_text)['attention']
    
    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-emotion_files/load-emotion_12_0.png


Visualizing luong model
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model = malaya.emotion.deep_model('luong')
    result = model.predict(surprise_text)['attention']
    
    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-emotion_files/load-emotion_14_0.png


Visualizing hierarchical model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model = malaya.emotion.deep_model('hierarchical')
    result = model.predict(surprise_text)['attention']
    
    plt.figure(figsize = (15, 7))
    labels = [r[0] for r in result]
    val = [r[1] for r in result]
    aranged = [i for i in range(len(labels))]
    plt.bar(aranged, val)
    plt.xticks(aranged, labels, rotation = 'vertical')
    plt.show()



.. image:: load-emotion_files/load-emotion_16_0.png


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

.. code:: ipython3

    malaya.emotion.available_sparse_deep_model()




.. parsed-literal::

    ['fast-text-char']



Right now Malaya only provide 1 sparse model, ``fast-text-char``. We
will try to evolve it.

.. code:: ipython3

    sparse_model = malaya.emotion.sparse_deep_model()


.. parsed-literal::

    downloading frozen /Users/huseinzol/Malaya/emotion/fast-text-char model


.. parsed-literal::

    17.0MB [00:06, 3.33MB/s]                          
    1.00MB [00:00, 1.10kMB/s]                  


.. parsed-literal::

    downloading frozen /Users/huseinzol/Malaya/emotion/fast-text-char index
    downloading frozen /Users/huseinzol/Malaya/emotion/fast-text-char meta


.. parsed-literal::

    1.00MB [00:00, 16.5MB/s]                   
      0%|          | 0.00/2.93 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading frozen /Users/huseinzol/Malaya/emotion/fast-text-char vector


.. parsed-literal::

    3.00MB [00:00, 3.95MB/s]                          


.. parsed-literal::

    INFO:tensorflow:Restoring parameters from /Users/huseinzol/Malaya/emotion/fast-text-char/model.ckpt


.. code:: ipython3

    sparse_model.predict(sadness_text)




.. parsed-literal::

    {'anger': 0.0077239843,
     'fear': 0.014800851,
     'joy': 0.008525367,
     'love': 0.0013007817,
     'sadness': 0.9655128,
     'surprise': 0.0021361646}



.. code:: ipython3

    sparse_model.predict_batch([anger_text,
                        fear_text,
                        joy_text,
                        love_text,
                        sadness_text,
                        surprise_text])




.. parsed-literal::

    [{'anger': 0.055561937,
      'fear': 0.034661848,
      'joy': 0.20765074,
      'love': 0.65774184,
      'sadness': 0.0210206,
      'surprise': 0.023363067},
     {'anger': 1.5065236e-05,
      'fear': 0.9998666,
      'joy': 6.3056427e-06,
      'love': 2.9068442e-06,
      'sadness': 3.6798014e-05,
      'surprise': 7.235542e-05},
     {'anger': 0.00097060547,
      'fear': 5.1922354e-05,
      'joy': 0.99052715,
      'love': 0.0024538564,
      'sadness': 0.0005109437,
      'surprise': 0.005485538},
     {'anger': 0.00014133049,
      'fear': 0.0004463539,
      'joy': 0.12486383,
      'love': 0.87307847,
      'sadness': 0.0013382707,
      'surprise': 0.0001317923},
     {'anger': 0.0077239843,
      'fear': 0.014800851,
      'joy': 0.008525367,
      'love': 0.0013007816,
      'sadness': 0.9655128,
      'surprise': 0.0021361646},
     {'anger': 0.0003960413,
      'fear': 0.6634573,
      'joy': 0.0014801685,
      'love': 0.00056572456,
      'sadness': 0.000516784,
      'surprise': 0.33358407}]


