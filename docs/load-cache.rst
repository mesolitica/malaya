
Cache location
--------------

You actually can know where is your Malaya caching folder.

.. code:: python

    import malaya

.. code:: python

    malaya.home




.. parsed-literal::

    '/Users/huseinzol/Malaya'



Cache subdirectories
--------------------

Start from version 1.0, Malaya put models in subdirectories, you can
print it by simply,

.. code:: python

    malaya.print_cache()


.. parsed-literal::

    Malaya/
    ├── dependency/
    │   ├── bahdanau/
    │   │   ├── bahdanau-dependency.json
    │   │   ├── bahdanau-dependency.pb
    │   │   └── version
    │   ├── concat/
    │   │   ├── concat-dependency.json
    │   │   ├── concat-dependency.pb
    │   │   └── version
    │   ├── crf/
    │   │   ├── crf-depend.pkl
    │   │   ├── crf-label.pkl
    │   │   └── version
    │   └── luong/
    │       ├── luong-dependency.json
    │       ├── luong-dependency.pb
    │       └── version
    ├── dictionary/
    │   └── malay-text.txt
    ├── emotion/
    │   ├── bahdanau/
    │   │   ├── bahdanau-emotion.json
    │   │   ├── bahdanau-emotion.pb
    │   │   └── version
    │   ├── bert/
    │   │   ├── bert-emotion.json
    │   │   ├── bert-emotion.pb
    │   │   └── version
    │   ├── bidirectional/
    │   │   ├── bidirectional-emotion.json
    │   │   ├── bidirectional-emotion.pb
    │   │   └── version
    │   ├── entity-network/
    │   │   ├── entity-emotion.json
    │   │   ├── entity-emotion.pb
    │   │   └── version
    │   ├── fast-text/
    │   │   ├── fasttext-emotion.json
    │   │   ├── fasttext-emotion.pb
    │   │   └── version
    │   ├── fast-text-char/
    │   │   ├── model.ckpt.data-00000-of-00001
    │   │   ├── model.ckpt.index
    │   │   ├── model.ckpt.meta
    │   │   ├── vectorizer-sparse-emotion.pkl
    │   │   └── version
    │   ├── hierarchical/
    │   │   ├── hierarchical-emotion.json
    │   │   ├── hierarchical-emotion.pb
    │   │   └── version
    │   ├── luong/
    │   │   ├── luong-emotion.json
    │   │   ├── luong-emotion.pb
    │   │   └── version
    │   ├── multinomial/
    │   │   ├── multinomial-emotion-tfidf.pkl
    │   │   ├── multinomial-emotion.pkl
    │   │   └── version
    │   └── xgb/
    │       ├── version
    │       ├── xgboost-emotion-tfidf.pkl
    │       └── xgboost-emotion.pkl
    ├── english.json
    ├── entity/
    │   ├── attention/
    │   │   ├── attention-entities.json
    │   │   ├── attention-entities.pb
    │   │   └── version
    │   ├── bahdanau/
    │   │   ├── bahdanau-entities.json
    │   │   ├── bahdanau-entities.pb
    │   │   └── version
    │   ├── concat/
    │   ├── crf/
    │   │   ├── crf-entities.pkl
    │   │   └── version
    │   ├── entity-network/
    │   │   ├── entity-entities.json
    │   │   ├── entity-entities.pb
    │   │   └── version
    │   └── luong/
    │       ├── luong-entities.json
    │       ├── luong-entities.pb
    │       └── version
    ├── fasttext-wiki/
    │   └── word2vec.p
    ├── language-detection/
    │   ├── deep/
    │   │   ├── model.ckpt.data-00000-of-00001
    │   │   ├── model.ckpt.index
    │   │   ├── model.ckpt.meta
    │   │   └── version
    │   ├── multinomial/
    │   │   ├── multinomial-language-detection.pkl
    │   │   ├── vectorizer-language-detection.pkl
    │   │   └── version
    │   ├── sgd/
    │   │   ├── sgd-language-detection.pkl
    │   │   └── version
    │   └── xgb/
    │       ├── version
    │       └── xgb-language-detection.pkl
    ├── normalizer/
    │   ├── normalizer-deep.json
    │   ├── normalizer-deep.pb
    │   └── version
    ├── pos/
    │   ├── attention/
    │   │   ├── attention-pos.json
    │   │   ├── attention-pos.pb
    │   │   └── version
    │   ├── bahdanau/
    │   │   ├── bahdanau-pos.json
    │   │   ├── bahdanau-pos.pb
    │   │   └── version
    │   ├── concat/
    │   │   ├── concat-pos.json
    │   │   ├── concat-pos.pb
    │   │   └── version
    │   ├── crf/
    │   │   ├── crf-pos.pkl
    │   │   └── version
    │   ├── entity-network/
    │   │   ├── entity-pos.pb
    │   │   └── version
    │   └── luong/
    │       ├── luong-pos.json
    │       ├── luong-pos.pb
    │       └── version
    ├── rules-based/
    │   ├── calon.csv
    │   ├── negeri.csv
    │   ├── person-normalized
    │   ├── short-normalized
    │   └── topic-normalized
    ├── rules-based.zip
    ├── sentiment/
    │   ├── bahdanau/
    │   │   ├── bahdanau-sentiment.json
    │   │   ├── bahdanau-sentiment.pb
    │   │   └── version
    │   ├── bert/
    │   │   ├── bert-sentiment.json
    │   │   ├── bert-sentiment.pb
    │   │   └── version
    │   ├── bidirectional/
    │   │   ├── bidirectional-sentiment.json
    │   │   ├── bidirectional-sentiment.pb
    │   │   └── version
    │   ├── entity-network/
    │   │   ├── entity-sentiment.json
    │   │   ├── entity-sentiment.pb
    │   │   └── version
    │   ├── fast-text/
    │   │   ├── fasttext-sentiment.json
    │   │   ├── fasttext-sentiment.pb
    │   │   └── version
    │   ├── fast-text-char/
    │   │   ├── model.ckpt.data-00000-of-00001
    │   │   ├── model.ckpt.index
    │   │   ├── model.ckpt.meta
    │   │   ├── vectorizer-sparse-sentiment.pkl
    │   │   └── version
    │   ├── hierarchical/
    │   │   ├── hierarchical-sentiment.json
    │   │   ├── hierarchical-sentiment.pb
    │   │   └── version
    │   ├── luong/
    │   │   ├── luong-sentiment.json
    │   │   ├── luong-sentiment.pb
    │   │   └── version
    │   ├── multinomial/
    │   │   ├── multinomial-sentiment-tfidf.pkl
    │   │   ├── multinomial-sentiment.pkl
    │   │   └── version
    │   └── xgb/
    │       ├── version
    │       ├── xgboost-sentiment-tfidf.pkl
    │       └── xgboost-sentiment.pkl
    ├── stem/
    │   ├── bahdanau/
    │   │   ├── bahdanau-stem.json
    │   │   ├── bahdanau-stem.pb
    │   │   └── version
    │   ├── lstm/
    │   │   ├── lstm-stem.json
    │   │   ├── lstm-stem.pb
    │   │   └── version
    │   ├── luong/
    │   │   ├── luong-stem.json
    │   │   ├── luong-stem.pb
    │   │   └── version
    │   ├── stemmer-deep.json
    │   ├── stemmer-deep.pb
    │   └── version
    ├── stop-word-kerulnet
    ├── subjective/
    │   ├── bahdanau/
    │   │   ├── bahdanau-subjective.json
    │   │   ├── bahdanau-subjective.pb
    │   │   └── version
    │   ├── bert/
    │   │   ├── bert-subjective.json
    │   │   ├── bert-subjective.pb
    │   │   └── version
    │   ├── bidirectional/
    │   │   ├── bidirectional-subjective.json
    │   │   ├── bidirectional-subjective.pb
    │   │   └── version
    │   ├── entity-network/
    │   │   ├── entity-subjective.json
    │   │   ├── entity-subjective.pb
    │   │   └── version
    │   ├── fast-text/
    │   │   ├── fasttext-subjective.json
    │   │   ├── fasttext-subjective.pb
    │   │   └── version
    │   ├── fast-text-char/
    │   │   ├── model.ckpt.data-00000-of-00001
    │   │   ├── model.ckpt.index
    │   │   ├── model.ckpt.meta
    │   │   ├── vectorizer-sparse-subjective.pkl
    │   │   └── version
    │   ├── hierarchical/
    │   │   ├── hierarchical-subjective.json
    │   │   ├── hierarchical-subjective.pb
    │   │   └── version
    │   ├── luong/
    │   │   ├── luong-subjective.json
    │   │   ├── luong-subjective.pb
    │   │   └── version
    │   ├── multinomial/
    │   │   ├── multinomial-subjective-tfidf.pkl
    │   │   ├── multinomial-subjective.pkl
    │   │   └── version
    │   └── xgb/
    │       ├── version
    │       ├── xgboost-subjective-tfidf.pkl
    │       └── xgboost-subjective.pkl
    ├── summarize/
    │   ├── dictionary-summary.json
    │   ├── summary-news.json
    │   ├── summary-news.pb
    │   ├── summary-wiki.json
    │   ├── summary-wiki.pb
    │   └── summary_frozen_model.pb
    ├── toxic/
    │   ├── bahdanau/
    │   │   ├── bahdanau-toxic.json
    │   │   ├── bahdanau-toxic.pb
    │   │   └── version
    │   ├── entity-network/
    │   │   ├── entity-toxic.json
    │   │   ├── entity-toxic.pb
    │   │   └── version
    │   ├── fast-text/
    │   │   ├── fasttext-toxic.json
    │   │   ├── fasttext-toxic.pb
    │   │   ├── fasttext-toxic.pkl
    │   │   └── version
    │   ├── hierarchical/
    │   │   ├── hierarchical-toxic.json
    │   │   ├── hierarchical-toxic.pb
    │   │   └── version
    │   ├── logistic/
    │   │   ├── logistics-toxic.pkl
    │   │   ├── vectorizer-toxic.pkl
    │   │   └── version
    │   ├── luong/
    │   │   ├── luong-toxic.json
    │   │   ├── luong-toxic.pb
    │   │   └── version
    │   └── multinomial/
    │       ├── multinomials-toxic.pkl
    │       ├── vectorizer-toxic.pkl
    │       └── version
    ├── version
    ├── word2vec-128/
    ├── word2vec-256/
    │   └── word2vec.p
    ├── word2vec-256.p
    └── word2vec-wiki/
        └── word2vec.p


Deleting specific model
-----------------------

Let say you want to clear some spaces, start from version 1.0, you can
specifically choose which model you want to delete.

.. code:: python

    malaya.clear_cache('word2vec-128')




.. parsed-literal::

    True



What happen if a directory does not exist?

.. code:: python

    malaya.clear_cache('word2vec-300')


::


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-5-715b00624659> in <module>
    ----> 1 malaya.clear_cache('word2vec-300')


    ~/Documents/Malaya/malaya/__init__.py in clear_cache(location)
        109     if not os.path.exists(location):
        110         raise Exception(
    --> 111             'folder not exist, please check path from malaya.print_cache()'
        112         )
        113     if not os.path.isdir(location):


    Exception: folder not exist, please check path from malaya.print_cache()


Purge cache
-----------

You can simply delete all models, totally purge it. By simply,

.. code:: python

    malaya.clear_all_cache




.. parsed-literal::

    <function malaya.clear_all_cache()>



I am not gonna to run it, because I prefer to keep it for now?
