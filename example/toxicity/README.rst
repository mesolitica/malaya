.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 4.54 s, sys: 975 ms, total: 5.52 s
    Wall time: 5.61 s


get labels
----------

.. code:: ipython3

    malaya.toxic.label




.. parsed-literal::

    ['severe toxic',
     'obscene',
     'identity attack',
     'insult',
     'threat',
     'asian',
     'atheist',
     'bisexual',
     'buddhist',
     'christian',
     'female',
     'heterosexual',
     'indian',
     'homosexual, gay or lesbian',
     'intellectual or learning disability',
     'male',
     'muslim',
     'other disability',
     'other gender',
     'other race or ethnicity',
     'other religion',
     'other sexual orientation',
     'physical disability',
     'psychiatric or mental illness',
     'transgender',
     'malay',
     'chinese']



.. code:: ipython3

    string = 'Benda yg SALAH ni, jgn lah didebatkan. Yg SALAH xkan jadi betul. Ingat tu. Mcm mana kesat sekalipun org sampaikan mesej, dan memang benda tu salah, diam je. Xyah nk tunjuk kau open sangat nk tegur cara org lain berdakwah. '
    another_string = 'melayu bodoh, dah la gay, sokong lgbt lagi, memang tak guna'

Load multinomial model
----------------------

All model interface will follow sklearn interface started v3.4,

.. code:: python

   model.predict(List[str])

   model.predict_proba(List[str])

.. code:: ipython3

    model = malaya.toxic.multinomial()

.. code:: ipython3

    model.predict_proba([string])




.. parsed-literal::

    [{'severe toxic': 0.9983866471486633,
      'obscene': 0.9609727610993377,
      'identity attack': 0.8695613508984636,
      'insult': 0.5893315709933827,
      'threat': 0.022178387416617994,
      'asian': 0.020300810205187092,
      'atheist': 0.011794932510638331,
      'bisexual': 0.002584488616645158,
      'buddhist': 0.004570410474229619,
      'christian': 0.03405075979783316,
      'female': 0.03787090649113612,
      'heterosexual': 0.008360866566466152,
      'indian': 0.9206507865140837,
      'homosexual, gay or lesbian': 0.03492931132214706,
      'intellectual or learning disability': 0.00158322379679834,
      'male': 0.06432988855860852,
      'muslim': 0.06722155678421161,
      'other disability': 0.0,
      'other gender': 0.0,
      'other race or ethnicity': 0.0017973269863205566,
      'other religion': 0.0017937047323945308,
      'other sexual orientation': 0.0012965120040433268,
      'physical disability': 0.001553693991766015,
      'psychiatric or mental illness': 0.024938805254016427,
      'transgender': 0.011663162911194878,
      'malay': 0.9995238230425324,
      'chinese': 0.9912614436972298}]



List available Transformer models
---------------------------------

.. code:: ipython3

    malaya.toxic.available_transformer_model()




.. parsed-literal::

    {'bert': ['425.7 MB', 'accuracy: 0.814'],
     'tiny-bert': ['57.4 MB', 'accuracy: 0.815'],
     'albert': ['48.7 MB', 'accuracy: 0.812'],
     'tiny-albert': ['22.4 MB', 'accuracy: 0.808'],
     'xlnet': ['446.5 MB', 'accuracy: 0.807'],
     'alxlnet': ['46.8 MB', 'accuracy: 0.817']}



Load ALXLNET model
------------------

All model interface will follow sklearn interface started v3.4,

.. code:: python

   model.predict(List[str])

   model.predict_proba(List[str])

.. code:: ipython3

    model = malaya.toxic.transformer(model = 'alxlnet')

Predict batch of strings
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    model.predict_proba([string,another_string])




.. parsed-literal::

    [{'severe toxic': 0.201493,
      'obscene': 0.12493244,
      'identity attack': 0.005829394,
      'insult': 0.08384159,
      'threat': 0.0010293126,
      'asian': 0.0004298091,
      'atheist': 0.0005042255,
      'bisexual': 0.0007214546,
      'buddhist': 0.00031352043,
      'christian': 0.001463592,
      'female': 0.095250845,
      'heterosexual': 0.00018996,
      'indian': 0.029991329,
      'homosexual, gay or lesbian': 0.00020930171,
      'intellectual or learning disability': 0.00018399954,
      'male': 0.017134428,
      'muslim': 0.0050880015,
      'other disability': 0.000233531,
      'other gender': 4.813075e-05,
      'other race or ethnicity': 0.00010916591,
      'other religion': 0.00031152368,
      'other sexual orientation': 0.00026413798,
      'physical disability': 0.000107735395,
      'psychiatric or mental illness': 3.6627054e-05,
      'transgender': 0.00016203523,
      'malay': 0.08275634,
      'chinese': 0.001092732},
     {'severe toxic': 0.9906007,
      'obscene': 0.90202737,
      'identity attack': 0.9795381,
      'insult': 0.6345859,
      'threat': 0.015953332,
      'asian': 0.014682382,
      'atheist': 0.0035497844,
      'bisexual': 0.0326609,
      'buddhist': 0.0101745725,
      'christian': 0.025312841,
      'female': 0.00968048,
      'heterosexual': 0.029808193,
      'indian': 0.011105597,
      'homosexual, gay or lesbian': 0.13856784,
      'intellectual or learning disability': 0.04939267,
      'male': 0.014529228,
      'muslim': 0.024640262,
      'other disability': 0.0009796321,
      'other gender': 0.037679344,
      'other race or ethnicity': 0.033878565,
      'other religion': 0.003752023,
      'other sexual orientation': 0.103711344,
      'physical disability': 0.00469586,
      'psychiatric or mental illness': 0.001588594,
      'transgender': 0.003436562,
      'malay': 0.9901147,
      'chinese': 0.1126565}]



Open toxicity visualization dashboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Default when you call ``predict_words`` it will open a browser with
visualization dashboard, you can disable by ``visualization=False``.

.. code:: ipython3

    model.predict_words(another_string)

.. code:: ipython3

    from IPython.core.display import Image, display
    
    display(Image('toxicity-dashboard.png', width=800))



.. image:: load-toxic_files/load-toxic_15_0.png
   :width: 800px


Stacking models
---------------

More information, you can read at
https://malaya.readthedocs.io/en/latest/Stack.html

.. code:: ipython3

    albert = malaya.toxic.transformer(model = 'albert')


.. parsed-literal::

    INFO:tensorflow:loading sentence piece model


.. code:: ipython3

    malaya.stack.predict_stack([model, albert], [another_string])




.. parsed-literal::

    [{'severe toxic': 0.9968317,
      'obscene': 0.43022493,
      'identity attack': 0.90531594,
      'insult': 0.42289576,
      'threat': 0.0058603976,
      'asian': 0.000983668,
      'atheist': 0.0005495089,
      'bisexual': 0.0009623809,
      'buddhist': 0.0003632398,
      'christian': 0.0018632574,
      'female': 0.006050684,
      'heterosexual': 0.0025569045,
      'indian': 0.0056869243,
      'homosexual, gay or lesbian': 0.012232827,
      'intellectual or learning disability': 0.00091394753,
      'male': 0.011594971,
      'muslim': 0.0042621437,
      'other disability': 0.00027529505,
      'other gender': 0.0010361207,
      'other race or ethnicity': 0.0012320877,
      'other religion': 0.00091365684,
      'other sexual orientation': 0.0027996385,
      'physical disability': 0.00010540871,
      'psychiatric or mental illness': 0.000815311,
      'transgender': 0.0016718076,
      'malay': 0.96644485,
      'chinese': 0.05199418}]



