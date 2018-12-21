
Entities Recognition
--------------------

Trained on 80% of dataset, tested on 20% of dataset. All training
sessions stored in
`session/entities <https://github.com/huseinzol05/Malaya/tree/master/session/entities>`__

.. figure:: https://raw.githubusercontent.com/huseinzol05/Malaya/master/accuracy/ner-accuracy.png
   :alt: alt text

   alt text

Concat
^^^^^^

.. code:: text

                 precision    recall  f1-score   support

          OTHER       1.00      1.00      1.00    498279
          event       0.98      0.99      0.99      2217
            law       0.99      0.99      0.99      1610
       location       0.99      1.00      1.00     20194
   organization       0.99      0.99      0.99     26093
         person       1.00      0.99      1.00     43377
       quantity       1.00      1.00      1.00     13180
           time       0.99      1.00      0.99     12750

    avg / total       1.00      1.00      1.00    617700

Bahdanau
^^^^^^^^

.. code:: text

                 precision    recall  f1-score   support

          OTHER       1.00      1.00      1.00    498587
          event       0.98      0.99      0.98      2212
            law       1.00      0.99      0.99      1746
       location       0.99      1.00      1.00     20387
   organization       0.99      1.00      1.00     25376
         person       1.00      1.00      1.00     43158
       quantity       1.00      1.00      1.00     13581
           time       0.99      1.00      0.99     12653

    avg / total       1.00      1.00      1.00    617700

Luong
^^^^^

.. code:: text

                 precision    recall  f1-score   support

          OTHER       1.00      1.00      1.00    497138
          event       0.99      0.99      0.99      2331
            law       0.99      0.99      0.99      1872
       location       0.99      1.00      1.00     20671
   organization       0.99      1.00      0.99     25942
         person       0.99      1.00      1.00     43511
       quantity       1.00      1.00      1.00     13376
           time       1.00      1.00      1.00     12859

    avg / total       1.00      1.00      1.00    617700

Entity-Network
^^^^^^^^^^^^^^

.. code:: text

                 precision    recall  f1-score   support

          OTHER       1.00      1.00      1.00    497198
          event       0.98      0.95      0.96      2381
            law       0.99      0.97      0.98      1881
       location       0.99      0.99      0.99     20305
   organization       0.99      0.98      0.98     26036
         person       0.99      0.99      0.99     43470
       quantity       0.99      0.99      0.99     13608
           time       0.98      0.99      0.98     12821

    avg / total       1.00      1.00      1.00    617700

CRF
^^^

.. code:: text

                 precision    recall  f1-score   support

       quantity      0.991     0.991     0.991     13891
       location      0.987     0.989     0.988     20798
           time      0.987     0.977     0.982     13264
         person      0.993     0.987     0.990     43590
   organization      0.974     0.973     0.973     25426
          event      0.995     0.983     0.989      2417
            law      0.994     0.988     0.991      1686

    avg / total      0.987     0.983     0.985    121072

Attention
^^^^^^^^^

.. code:: text

                 precision    recall  f1-score   support

          OTHER       1.00      1.00      1.00    497073
          event       0.99      0.97      0.98      2426
            law       1.00      0.99      0.99      1806
       location       1.00      1.00      1.00     20176
   organization       1.00      1.00      1.00     26044
         person       1.00      1.00      1.00     44346
       quantity       1.00      1.00      1.00     13155
           time       0.99      1.00      1.00     12674

    avg / total       1.00      1.00      1.00    617700
