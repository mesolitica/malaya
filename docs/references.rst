
Recurrent neural network
------------------------

Malaya use
`Long-Short-Term-Memory <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`__
for all RNN gates.

LSTM References:

1. Hochreiter, Sepp; Schmidhuber, Jürgen (1997-11-01). “Long Short-Term
   Memory”. Neural Computation. 9 (8): 1735–1780.
   doi:10.1162/neco.1997.9.8.1735.

Malaya use recurrent neural network architecture on some models.

Sentiment Analysis
^^^^^^^^^^^^^^^^^^

1. ``malaya.deep_sentiment('luong')``
2. ``malaya.deep_sentiment('bahdanau')``
3. ``malaya.deep_sentiment('hierarchical')``

Toxicity Analysis
^^^^^^^^^^^^^^^^^

1. ``malaya.deep_toxic('luong')``
2. ``malaya.deep_toxic('bahdanau')``
3. ``malaya.deep_toxic('hierarchical')``

Entities Recognition
^^^^^^^^^^^^^^^^^^^^

1. ``malaya.deep_entities('entity-network')``

POS Recognition
^^^^^^^^^^^^^^^

1. ``malaya.deep_pos('entity-network')``

Stemmer
^^^^^^^

1. ``malaya.deep_stemmer()``

You can read more about `Recurrent Neural Network
here <http://karpathy.github.io/2015/05/21/rnn-effectiveness/>`__.

References
^^^^^^^^^^

1. Li, Xiangang; Wu, Xihong (2014-10-15). “Constructing Long Short-Term
   Memory based Deep Recurrent Neural Networks for Large Vocabulary
   Speech Recognition”. arXiv:1410.4281 [cs.CL].

2. Hochreiter, Sepp; Schmidhuber, Jürgen (1997-11-01). “Long Short-Term
   Memory”. Neural Computation. 9 (8): 1735–1780.
   doi:10.1162/neco.1997.9.8.1735.

3. Schmidhuber, Jürgen (January 2015). “Deep Learning in Neural
   Networks: An Overview”. Neural Networks. 61: 85–117. arXiv:1404.7828.
   doi:10.1016/j.neunet.2014.09.003. PMID 25462637.

Bidirectional recurrent neural network
--------------------------------------

Malaya use
`Long-Short-Term-Memory <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`__
for all BiRNN gates.

LSTM References:

1. Hochreiter, Sepp; Schmidhuber, Jürgen (1997-11-01). “Long Short-Term
   Memory”. Neural Computation. 9 (8): 1735–1780.
   doi:10.1162/neco.1997.9.8.1735.

Malaya use bidirectional recurrent neural network in some models.

Sentiment Analysis
^^^^^^^^^^^^^^^^^^

1. ``malaya.deep_sentiment('bidirectional')``

Entities Recognition
^^^^^^^^^^^^^^^^^^^^

1. ``malaya.deep_entities('concat')``
2. ``malaya.deep_entities('bahdanau')``
3. ``malaya.deep_entities('luong')``

POS Recognition
^^^^^^^^^^^^^^^

1. ``malaya.deep_pos('concat')``
2. ``malaya.deep_pos('bahdanau')``
3. ``malaya.deep_pos('luong')``

Normalizer
^^^^^^^^^^

1. ``malaya.deep_normalizer()``

Topics & Influencers Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. ``malaya.deep_siamese_get_topics()``
2. ``malaya.deep_siamese_get_influencers()``
3. ``malaya.deep_get_topics()``
4. ``malaya.deep_get_influencers()``

Summarization
^^^^^^^^^^^^^

1. ``malaya.summarize_deep_learning()``

You can read more about `Bidirectional Recurrent Neural Network
here <https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks>`__.

References
^^^^^^^^^^

1. M. Schuster, K.K. Paliwal. Bidirectional recurrent neural networks
   (November 1997). https://ieeexplore.ieee.org/document/650093

Seq2Seq
-------

Malaya use seq2seq in some models.

Normalizer
^^^^^^^^^^

1. ``malaya.deep_normalizer()``

Stemmer
^^^^^^^

1. ``malaya.deep_stemmer()``

You can read more about `Seq2Seq
here <https://google.github.io/seq2seq/>`__.

References
^^^^^^^^^^

1. Ilya Sutskever, Oriol Vinyals: “Sequence to Sequence Learning with
   Neural Networks”, 2014; [http://arxiv.org/abs/1409.3215
   arXiv:1409.3215].

Conditional Random Field
------------------------

Malaya use CRF in some models.

Entities Recognition
^^^^^^^^^^^^^^^^^^^^

1. ``malaya.deep_entities('concat')``
2. ``malaya.deep_entities('bahdanau')``
3. ``malaya.deep_entities('luong')``
4. ``malaya.deep_entities('entity-network')``

POS Recognition
^^^^^^^^^^^^^^^

1. ``malaya.deep_pos('concat')``
2. ``malaya.deep_pos('bahdanau')``
3. ``malaya.deep_pos('luong')``
4. ``malaya.deep_pos('entity-network')``

You can read more about `CRF
here <http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/>`__

References
^^^^^^^^^^

1. Zhiheng Huang, Wei Xu: “Bidirectional LSTM-CRF Models for Sequence
   Tagging”, 2015; [http://arxiv.org/abs/1508.01991 arXiv:1508.01991].

BERT (Deep Bidirectional Transformers)
--------------------------------------

Malaya use BERT in some models.

Sentiment Analysis
^^^^^^^^^^^^^^^^^^

1. ``malaya.deep_sentiment('bert')``

References
^^^^^^^^^^

1. Jacob Devlin, Ming-Wei Chang, Kenton Lee: “BERT: Pre-training of Deep
   Bidirectional Transformers for Language Understanding”, 2018;
   [http://arxiv.org/abs/1810.04805 arXiv:1810.04805].

Entity-Network
--------------

Malaya use Entity-Network in some models.

Sentiment Analysis
^^^^^^^^^^^^^^^^^^

1. ``malaya.deep_sentiment('entity-network')``

Toxicity Analysis
^^^^^^^^^^^^^^^^^

1. ``malaya.deep_toxic('entity-network')``

Entities Recognition
^^^^^^^^^^^^^^^^^^^^

1. ``malaya.deep_entities('entity-network')``

POS Recognition
^^^^^^^^^^^^^^^

1. ``malaya.deep_pos('entity-network')``

References
^^^^^^^^^^

1. Andrea Madotto: “Question Dependent Recurrent Entity Network for
   Question Answering”, 2017; [http://arxiv.org/abs/1707.07922
   arXiv:1707.07922].

Skip-thought Vector
-------------------

Malaya use skip-thought in some models.

Summarization
^^^^^^^^^^^^^

1. ``malaya.summarize_deep_learning()``

Topics & Influencers Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. ``malaya.deep_get_topics()``
2. ``malaya.deep_get_influencers()``

References
^^^^^^^^^^

1. Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel,
   Antonio Torralba, Raquel Urtasun: “Skip-Thought Vectors”, 2015;
   [http://arxiv.org/abs/1506.06726 arXiv:1506.06726].

Siamese Network
---------------

Malaya use siamese network in some models.

Topics & Influencers Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. ``malaya.deep_siamese_get_topics()``
2. ``malaya.deep_siamese_get_influencers()``

References
^^^^^^^^^^

1. Anfeng He, Chong Luo, Xinmei Tian: “A Twofold Siamese Network for
   Real-Time Object Tracking”, 2018; [http://arxiv.org/abs/1802.08817
   arXiv:1802.08817].

Normalizer
----------

References
^^^^^^^^^^

1. N. Samsudin, Mazidah Puteh, Abdul Razak Hamdan, Mohd Zakree Ahmad
   Nazri, Normalization of noisy texts in Malaysian online reviews;
   https://www.researchgate.net/publication/287050449_Normalization_of_noisy_texts_in_Malaysian_online_reviews

XGBoost
-------

Malaya use XGBoost in some models.

Sentiment Analysis
^^^^^^^^^^^^^^^^^^

1. ``malaya.sentiment.pretrained_xgb_sentiment()``

Language Detection
^^^^^^^^^^^^^^^^^^

1. ``malaya.xgb_detect_languages()``

References
^^^^^^^^^^

1. Tianqi Chen: “XGBoost: A Scalable Tree Boosting System”, 2016;
   [http://arxiv.org/abs/1603.02754 arXiv:1603.02754]. DOI:
   [https://dx.doi.org/10.1145/2939672.2939785 10.1145/2939672.2939785].

Multinomial
-----------

Malaya use multinomial in some models.

Sentiment Analysis
^^^^^^^^^^^^^^^^^^

1. ``malaya.sentiment.pretrained_bayes_sentiment()``

Language Detection
^^^^^^^^^^^^^^^^^^

1. ``malaya.multinomial_detect_languages()``

Toxicity Analysis
^^^^^^^^^^^^^^^^^

1. ``malaya.multinomial_detect_toxic()``

References
^^^^^^^^^^

1. https://medium.com/@johnm.kovachi/implementing-a-multinomial-naive-bayes-classifier-from-scratch-with-python-e70de6a3b92e

Logistic Regression
-------------------

Malaya use logistic regression in some models.

Toxicity Analysis
^^^^^^^^^^^^^^^^^

1. ``malaya.logistics_detect_toxic()``

References
^^^^^^^^^^

1. https://itnext.io/machine-learning-sentiment-analysis-of-movie-reviews-using-logisticregression-62e9622b4532
