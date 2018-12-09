
.. code:: ipython3

    import malaya
    print(malaya.version, malaya.bump_version)


.. parsed-literal::

    0.8 0.8.0.0


.. code:: ipython3

    string = 'Benda yg SALAH ni, jgn lah didebatkan. Yg SALAH xkan jadi betul. Ingat tu. Mcm mana kesat sekalipun org sampaikan mesej, dan memang benda tu salah, diam je. Xyah nk tunjuk kau open sangat nk tegur cara org lain berdakwah. '
    another_string = 'bodoh, dah la gay, sokong lgbt lagi, memang tak guna'

Load multinomial model
----------------------

.. code:: ipython3

    model = malaya.toxic.multinomial_detect_toxic()

.. code:: ipython3

    model.predict(string)




.. parsed-literal::

    {'toxic': 0,
     'severe_toxic': 0,
     'obscene': 0,
     'threat': 0,
     'insult': 0,
     'identity_hate': 0}



.. code:: ipython3

    model.predict(string,get_proba=True)




.. parsed-literal::

    {'toxic': 0.14165235977019472,
     'severe_toxic': 1.9272487152616215e-06,
     'obscene': 0.011323038998473341,
     'threat': 8.249039905334012e-08,
     'insult': 0.008620760536227347,
     'identity_hate': 4.703244329372946e-06}



.. code:: ipython3

    model.predict(another_string)




.. parsed-literal::

    {'toxic': 1,
     'severe_toxic': 0,
     'obscene': 0,
     'threat': 0,
     'insult': 1,
     'identity_hate': 0}



.. code:: ipython3

    model.predict(another_string,get_proba=True)




.. parsed-literal::

    {'toxic': 0.97624511869432,
     'severe_toxic': 0.0004143925305717536,
     'obscene': 0.48936571876841484,
     'threat': 5.809081616106756e-06,
     'insult': 0.7853970362543069,
     'identity_hate': 0.002109806847753244}



.. code:: ipython3

    model.predict_batch([string,another_string])




.. parsed-literal::

    {'toxic': [0, 1],
     'severe_toxic': [0, 0],
     'obscene': [0, 0],
     'threat': [0, 0],
     'insult': [0, 1],
     'identity_hate': [0, 0]}



.. code:: ipython3

    model.predict_batch([string,another_string],get_proba=True)




.. parsed-literal::

    {'toxic': [0.14165235977019472, 0.97624511869432],
     'severe_toxic': [1.9272487152616215e-06, 0.0004143925305717536],
     'obscene': [0.011323038998473341, 0.48936571876841484],
     'threat': [8.249039905334012e-08, 5.809081616106756e-06],
     'insult': [0.008620760536227347, 0.7853970362543069],
     'identity_hate': [4.703244329372946e-06, 0.002109806847753244]}



Load logistics model
--------------------

.. code:: ipython3

    model = malaya.toxic.logistics_detect_toxic()

.. code:: ipython3

    model.predict(string)




.. parsed-literal::

    {'toxic': 0,
     'severe_toxic': 0,
     'obscene': 0,
     'threat': 0,
     'insult': 0,
     'identity_hate': 0}



.. code:: ipython3

    model.predict_batch([string,another_string],get_proba=True)




.. parsed-literal::

    {'toxic': [0.10299208923447233, 0.6297643126911581],
     'severe_toxic': [0.010195223990855215, 0.019551370640497476],
     'obscene': [0.04834509566263489, 0.1995748012804703],
     'threat': [0.003488478318883341, 0.004014463652898358],
     'insult': [0.04528784776538583, 0.3354069432946268],
     'identity_hate': [0.011326619000125776, 0.052626041879065236]}



List available deep learning models
-----------------------------------

.. code:: ipython3

    malaya.toxic.get_available_toxic_models()




.. parsed-literal::

    ['bahdanau', 'hierarchical', 'luong', 'fast-text', 'entity-network']



Load deep learning model
------------------------

.. code:: ipython3

    deep_model = malaya.toxic.deep_toxic()

.. code:: ipython3

    deep_model.predict(string)




.. parsed-literal::

    {'toxic': 0.99533576,
     'severe_toxic': 0.49553683,
     'obscene': 0.9582162,
     'threat': 0.07477511,
     'insult': 0.93286234,
     'identity_hate': 0.602743,
     'attention': [['benda', 0.027777778],
      ['yg', 0.027777778],
      ['salah', 0.027777778],
      ['ni', 0.027777778],
      ['jgn', 0.027777778],
      ['didebatkan', 0.027777778],
      ['yg', 0.027777778],
      ['salah', 0.027777778],
      ['jadi', 0.027777778],
      ['betul', 0.027777778],
      ['ingat', 0.027777778],
      ['tu', 0.027777778],
      ['mcm', 0.027777778],
      ['mana', 0.027777778],
      ['kesat', 0.027777778],
      ['sekalipun', 0.027777778],
      ['org', 0.027777778],
      ['sampaikan', 0.027777778],
      ['mesej', 0.027777778],
      ['memang', 0.027777778],
      ['benda', 0.027777778],
      ['tu', 0.027777778],
      ['salah', 0.027777778],
      ['diam', 0.027777778],
      ['je', 0.027777778],
      ['xyah', 0.027777778],
      ['nk', 0.027777778],
      ['tunjuk', 0.027777778],
      ['kau', 0.027777778],
      ['sangat', 0.027777778],
      ['nk', 0.027777778],
      ['tegur', 0.027777778],
      ['cara', 0.027777778],
      ['org', 0.027777778],
      ['lain', 0.027777778],
      ['berdakwah', 0.027777778]]}



.. code:: ipython3

    deep_model.predict_batch([string, another_string])




.. parsed-literal::

    {'toxic': [0.9979735, 0.9994906],
     'severe_toxic': [0.46729267, 0.13444535],
     'obscene': [0.96576005, 0.9766732],
     'threat': [0.05801873, 0.016751291],
     'insult': [0.94788307, 0.8914509],
     'identity_hate': [0.6173997, 0.11204952]}



.. code:: ipython3

    for model in malaya.toxic.get_available_toxic_models():
        deep_model = malaya.toxic.deep_toxic(model = model)
        print(deep_model.predict(string))
        print(deep_model.predict_batch([string, another_string]),'\n')
        


.. parsed-literal::

    {'toxic': 0.0014104632, 'severe_toxic': 9.9826e-06, 'obscene': 0.0001435599, 'threat': 1.2577166e-05, 'insult': 6.3974294e-05, 'identity_hate': 1.5297384e-05, 'attention': [['benda', 0.08840476], ['yg', 0.010839531], ['salah', 0.009628137], ['ni', 0.030507175], ['jgn', 0.060949128], ['didebatkan', 0.009529242], ['yg', 0.010839531], ['salah', 0.009628137], ['jadi', 0.010453292], ['betul', 0.008679067], ['ingat', 0.016919741], ['tu', 0.010233691], ['mcm', 0.0074331276], ['mana', 0.00834874], ['kesat', 0.022738086], ['sekalipun', 0.024935191], ['org', 0.03207217], ['sampaikan', 0.013176771], ['mesej', 0.012281337], ['memang', 0.007963687], ['benda', 0.08840476], ['tu', 0.010233691], ['salah', 0.009628137], ['diam', 0.008944328], ['je', 0.13689674], ['xyah', 0.060949128], ['nk', 0.086197734], ['tunjuk', 0.0068813916], ['kau', 0.011481011], ['sangat', 0.007749557], ['nk', 0.086197734], ['tegur', 0.016609907], ['cara', 0.013878295], ['org', 0.03207217], ['lain', 0.010712912], ['berdakwah', 0.007601946]]}
    {'toxic': [0.0008679644, 0.8282658], 'severe_toxic': [7.966964e-06, 0.0015704108], 'obscene': [0.000105044324, 0.035478123], 'threat': [1.0944875e-05, 0.0003193759], 'insult': [3.9492985e-05, 0.06402249], 'identity_hate': [1.2289709e-05, 0.056409597]} 
    
    {'toxic': 0.013686066, 'severe_toxic': 0.00015248382, 'obscene': 0.0036878092, 'threat': 0.0003602789, 'insult': 0.0016004957, 'identity_hate': 0.00065138243, 'attention': [['benda', 0.033093613], ['yg', 0.020311492], ['salah', 0.031071305], ['ni', 0.035885986], ['jgn', 0.025950985], ['didebatkan', 0.022656968], ['yg', 0.020546723], ['salah', 0.01839146], ['jadi', 0.023359463], ['betul', 0.045144785], ['ingat', 0.10163837], ['tu', 0.034831], ['mcm', 0.04458493], ['mana', 0.0069026393], ['kesat', 0.012962426], ['sekalipun', 0.034700137], ['org', 0.009746097], ['sampaikan', 0.026797928], ['mesej', 0.022329712], ['memang', 0.01396555], ['benda', 0.01942503], ['tu', 0.093634725], ['salah', 0.05520428], ['diam', 0.040790573], ['je', 0.11458255], ['xyah', 0.047265194], ['nk', 0.02154637], ['tunjuk', 0.007742938], ['kau', 0.009726809], ['sangat', 0.0012139411], ['nk', 0.0013947687], ['tegur', 0.0015051698], ['cara', 0.00012083069], ['org', 0.0001231546], ['lain', 0.00029581765], ['berdakwah', 0.00055633666]]}
    {'toxic': [0.014301307, 0.5686806], 'severe_toxic': [0.00013729886, 0.0018517369], 'obscene': [0.0037747642, 0.0343218], 'threat': [0.0003370598, 0.002325343], 'insult': [0.0015627467, 0.15438789], 'identity_hate': [0.0005754034, 0.037985425]} 
    
    {'toxic': 0.9975069, 'severe_toxic': 0.5357675, 'obscene': 0.9547516, 'threat': 0.053325806, 'insult': 0.93040293, 'identity_hate': 0.5089189, 'attention': [['benda', 0.027777778], ['yg', 0.027777778], ['salah', 0.027777778], ['ni', 0.027777778], ['jgn', 0.027777778], ['didebatkan', 0.027777778], ['yg', 0.027777778], ['salah', 0.027777778], ['jadi', 0.027777778], ['betul', 0.027777778], ['ingat', 0.027777778], ['tu', 0.027777778], ['mcm', 0.027777778], ['mana', 0.027777778], ['kesat', 0.027777778], ['sekalipun', 0.027777778], ['org', 0.027777778], ['sampaikan', 0.027777778], ['mesej', 0.027777778], ['memang', 0.027777778], ['benda', 0.027777778], ['tu', 0.027777778], ['salah', 0.027777778], ['diam', 0.027777778], ['je', 0.027777778], ['xyah', 0.027777778], ['nk', 0.027777778], ['tunjuk', 0.027777778], ['kau', 0.027777778], ['sangat', 0.027777778], ['nk', 0.027777778], ['tegur', 0.027777778], ['cara', 0.027777778], ['org', 0.027777778], ['lain', 0.027777778], ['berdakwah', 0.027777778]]}
    {'toxic': [0.996828, 0.9994374], 'severe_toxic': [0.3273979, 0.1762022], 'obscene': [0.9528909, 0.9758203], 'threat': [0.02051198, 0.041396763], 'insult': [0.8984617, 0.9271479], 'identity_hate': [0.38474312, 0.09764755]} 
    
    downloading TOXIC fast-text bigrams


.. parsed-literal::

    8.00MB [00:03, 2.07MB/s]                          


.. parsed-literal::

    {'toxic': 0.0020534173, 'severe_toxic': 0.0050337594, 'obscene': 3.7653503e-05, 'threat': 0.7628687, 'insult': 9.012385e-05, 'identity_hate': 0.22991635}
    {'toxic': [4.6989637e-08, 0.07565687], 'severe_toxic': [2.8443527e-08, 0.005023106], 'obscene': [4.1618722e-10, 0.0053009894], 'threat': [3.280739e-06, 0.0040464187], 'insult': [7.941728e-10, 0.043121953], 'identity_hate': [8.946894e-07, 0.016103525]} 
    
    downloading TOXIC frozen entity-network model


.. parsed-literal::

    56.0MB [00:26, 2.55MB/s]                          
      0%|          | 0.00/1.98 [00:00<?, ?MB/s]

.. parsed-literal::

    downloading TOXIC entity-network dictionary


.. parsed-literal::

    2.00MB [00:00, 2.13MB/s]                          


.. parsed-literal::

    {'toxic': 0.501814, 'severe_toxic': 0.03271238, 'obscene': 0.15100613, 'threat': 0.028492289, 'insult': 0.24221319, 'identity_hate': 0.043762065}
    {'toxic': [0.7704032, 0.23564923], 'severe_toxic': [0.1794783, 0.009002773], 'obscene': [0.50242037, 0.14901799], 'threat': [0.16002978, 0.030735493], 'insult': [0.61826205, 0.12641545], 'identity_hate': [0.2263789, 0.019457512]} 
    

