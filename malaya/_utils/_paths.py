from .. import home

MALAY_TEXT = home + '/dictionary/malay-text.txt'
MALAY_TEXT_200K = home + '/dictionary-200k/malay-text.txt'

PATH_NGRAM = {
    1: {'model': home + '/preprocessing/bm_1grams.json', 'version': 'v23'},
    2: {'model': home + '/preprocessing/bm_2grams.json', 'version': 'v23'},
}

S3_PATH_NGRAM = {
    1: {'model': 'v23/preprocessing/bm_1grams.json'},
    2: {'model': 'v23/preprocessing/bm_2grams.json'},
}

PATH_PREPROCESSING = {
    1: {'model': home + '/preprocessing/counts_1grams.json', 'version': 'v23'},
    2: {'model': home + '/preprocessing/counts_2grams.json', 'version': 'v23'},
    'english-malay': {
        'model': home + '/preprocessing/english-malay-200k.json',
        'version': 'v23',
    },
}

S3_PATH_PREPROCESSING = {
    1: {'model': 'v23/preprocessing/counts_1grams.json'},
    2: {'model': 'v23/preprocessing/counts_2grams.json'},
    'english-malay': {'model': 'v23/preprocessing/english-malay-200k.json'},
}

PATH_ELMO = {
    128: {
        'model': home + '/elmo-wiki/elmo-128.pb',
        'setting': home + '/elmo-wiki/elmo-128.pkl',
        'version': 'v20',
    },
    256: {
        'model': home + '/elmo-wiki/elmo-256.pb',
        'setting': home + '/elmo-wiki/elmo-256.pkl',
        'version': 'v20',
    },
}

S3_PATH_ELMO = {
    128: {'model': 'v20/elmo/elmo-128.pb', 'setting': 'v20/elmo/elmo-128.pkl'},
    256: {'model': 'v20/elmo/elmo-256.pb', 'setting': 'v20/elmo/elmo-256.pkl'},
}

PATH_STEM = {
    'lstm': {
        'model': home + '/stem/lstm/lstm-stem.pb',
        'setting': home + '/stem/lstm/lstm-stem.json',
        'version': 'v15',
    },
    'bahdanau': {
        'model': home + '/stem/bahdanau/bahdanau-stem.pb',
        'setting': home + '/stem/bahdanau/bahdanau-stem.json',
        'version': 'v15',
    },
    'luong': {
        'model': home + '/stem/luong/luong-stem.pb',
        'setting': home + '/stem/luong/luong-stem.json',
        'version': 'v15',
    },
}

S3_PATH_STEM = {
    'lstm': {
        'model': 'v15/stem/lstm-stem.pb',
        'setting': 'v15/stem/lstm-stem.json',
    },
    'bahdanau': {
        'model': 'v15/stem/bahdanau-stem.pb',
        'setting': 'v15/stem/bahdanau-stem.json',
    },
    'luong': {
        'model': 'v15/stem/luong-stem.pb',
        'setting': 'v15/stem/luong-stem.json',
    },
}

PATH_SUMMARIZE = {
    'news': {
        'model': home + '/summarize/summary-news.pb',
        'setting': home + '/summarize/summary-news.json',
        'version': 'v13',
    },
    'wiki': {
        'model': home + '/summarize/summary-wiki.pb',
        'setting': home + '/summarize/summary-wiki.json',
        'version': 'v13',
    },
}

S3_PATH_SUMMARIZE = {
    'news': {
        'model': 'v13/summarize/summary-news.pb',
        'setting': 'v13/summarize/summary-news.json',
    },
    'wiki': {
        'model': 'v13/summarize/summary-wiki.pb',
        'setting': 'v13/summarize/summary-wiki.json',
    },
}

PATH_TOXIC = {
    'multinomial': {
        'model': home + '/toxic/multinomial/multinomials-toxic.pkl',
        'vector': home + '/toxic/multinomial/vectorizer-toxic.pkl',
        'version': 'v6',
    },
    'logistic': {
        'model': home + '/toxic/logistic/logistics-toxic.pkl',
        'vector': home + '/toxic/logistic/vectorizer-toxic.pkl',
        'version': 'v6',
    },
    'luong': {
        'model': home + '/toxic/luong/luong-toxic.pb',
        'setting': home + '/toxic/toxicity-dictionary.json',
        'version': 'v24',
    },
    'bahdanau': {
        'model': home + '/toxic/bahdanau/bahdanau-toxic.pb',
        'setting': home + '/toxic/toxicity-dictionary.json',
        'version': 'v24',
    },
    'self-attention': {
        'model': home + '/toxic/self-attention/self-attention-toxic.pb',
        'setting': home + '/toxic/toxicity-dictionary.json',
        'version': 'v24',
    },
    'fast-text-char': {
        'model': home + '/toxic/fast-text-char/model.ckpt.data-00000-of-00001',
        'index': home + '/toxic/fast-text-char/model.ckpt.index',
        'meta': home + '/toxic/fast-text-char/model.ckpt.meta',
        'vector': home + '/toxic/fast-text-char/vectorizer-sparse-toxicity.pkl',
        'version': 'v24',
    },
    'bert': {
        'model': home + '/toxic/bert/bert-toxic.pb',
        'vocab': home + '/bert/multilanguage-vocab.txt',
        'version': 'v24',
    },
}

S3_PATH_TOXIC = {
    'multinomial': {
        'model': 'v6/multinomials-toxic.pkl',
        'vector': 'v6/vectorizer-toxic.pkl',
    },
    'logistic': {
        'model': 'v6/logistics-toxic.pkl',
        'vector': 'v6/vectorizer-toxic.pkl',
    },
    'luong': {
        'model': 'v24/toxicity/luong-toxic.pb',
        'setting': 'v24/toxicity/toxicity-dictionary.json',
    },
    'bahdanau': {
        'model': 'v24/toxicity/bahdanau-toxic.pb',
        'setting': 'v24/toxicity/toxicity-dictionary.json',
    },
    'self-attention': {
        'model': 'v24/toxicity/self-attention-toxic.pb',
        'setting': 'v24/toxicity/toxicity-dictionary.json',
    },
    'fast-text-char': {
        'model': 'v24/toxicity/model.ckpt.data-00000-of-00001',
        'index': 'v24/toxicity/model.ckpt.index',
        'meta': 'v24/toxicity/model.ckpt.meta',
        'vector': 'v24/toxicity/vectorizer-sparse-toxicity.pkl',
    },
    'bert': {
        'model': 'v24/toxicity/bert-toxic.pb',
        'vocab': 'v24/multilanguage-vocab.txt',
    },
}

PATH_POS = {
    'crf': {'model': home + '/pos/crf/crf-pos.pkl', 'version': 'v14'},
    'concat': {
        'model': home + '/pos/concat/concat-pos.pb',
        'setting': home + '/pos/concat/concat-pos.json',
        'version': 'v14',
    },
    'luong': {
        'model': home + '/pos/luong/luong-pos.pb',
        'setting': home + '/pos/luong/luong-pos.json',
        'version': 'v21',
    },
    'bahdanau': {
        'model': home + '/pos/bahdanau/bahdanau-pos.pb',
        'setting': home + '/pos/bahdanau/bahdanau-pos.json',
        'version': 'v21',
    },
    'entity-network': {
        'model': home + '/pos/entity-network/entity-pos.pb',
        'setting': home + 'pos/entity-network/entity-pos.json',
        'version': 'v14',
    },
    'attention': {
        'model': home + '/pos/attention/attention-pos.pb',
        'setting': home + '/pos/attention/attention-pos.json',
        'version': 'v14',
    },
}

S3_PATH_POS = {
    'crf': {'model': 'v14/pos/crf-pos.pkl'},
    'concat': {
        'model': 'v14/pos/concat-pos.pb',
        'setting': 'v14/pos/concat-pos.json',
    },
    'luong': {
        'model': 'v21/pos/luong-pos.pb',
        'setting': 'v21/pos/luong-pos.json',
    },
    'bahdanau': {
        'model': 'v21/pos/bahdanau-pos.pb',
        'setting': 'v21/pos/bahdanau-pos.json',
    },
    'entity-network': {
        'model': 'v14/pos/entity-pos.pb',
        'setting': 'v14/pos/entity-pos.json',
    },
    'attention': {
        'model': 'v14/pos/attention-pos.pb',
        'setting': 'v14/pos/attention-pos.json',
    },
}

PATH_LANG_DETECTION = {
    'multinomial': {
        'model': home
        + '/language-detection/multinomial/multinomial-language-detection.pkl',
        'vector': home
        + '/language-detection/multinomial/vectorizer-language-detection.pkl',
        'version': 'v10.2',
    },
    'xgb': {
        'model': home + '/language-detection/xgb/xgb-language-detection.pkl',
        'vector': home
        + '/language-detection/multinomial/vectorizer-language-detection.pkl',
        'version': 'v10.2',
    },
    'sgd': {
        'model': home + '/language-detection/sgd/sgd-language-detection.pkl',
        'vector': home
        + '/language-detection/multinomial/vectorizer-language-detection.pkl',
        'version': 'v10.2',
    },
    'deep': {
        'model': home
        + '/language-detection/deep/model.ckpt.data-00000-of-00001',
        'index': home + '/language-detection/deep/model.ckpt.index',
        'meta': home + '/language-detection/deep/model.ckpt.meta',
        'vector': home
        + '/language-detection/multinomial/vectorizer-language-detection.pkl',
        'version': 'v10.2',
    },
}

S3_PATH_LANG_DETECTION = {
    'multinomial': {
        'model': 'v10/language-detection/multinomial-language-detection.pkl',
        'vector': 'v10/language-detection/language-detection-vectorizer.pkl',
    },
    'xgb': {
        'model': 'v10/language-detection/xgboost-language-detection.pkl',
        'vector': 'v10/language-detection/language-detection-vectorizer.pkl',
    },
    'sgd': {
        'model': 'v10/language-detection/sgd-language-detection.pkl',
        'vector': 'v10/language-detection/language-detection-vectorizer.pkl',
    },
    'deep': {
        'model': 'v10/language-detection/model.ckpt.data-00000-of-00001',
        'index': 'v10/language-detection/model.ckpt.index',
        'meta': 'v10/language-detection/model.ckpt.meta',
        'vector': 'v10/language-detection/language-detection-vectorizer.pkl',
    },
}

PATH_ENTITIES = {
    'crf': {'model': home + '/entity/crf/crf-entities.pkl', 'version': 'v8'},
    'concat': {
        'model': home + '/entity/concat/concat-entities.pb',
        'setting': home + '/entity/concat/concat-entities.json',
        'version': 'v14',
    },
    'luong': {
        'model': home + '/entity/luong/luong-entities.pb',
        'setting': home + '/entity/luong/luong-entities.json',
        'version': 'v21',
    },
    'bahdanau': {
        'model': home + '/entity/bahdanau/bahdanau-entities.pb',
        'setting': home + '/entity/bahdanau/bahdanau-entities.json',
        'version': 'v21',
    },
    'entity-network': {
        'model': home + '/entity/entity-network/entity-entities.pb',
        'setting': home + '/entity/entity-network/entity-entities.json',
        'version': 'v21',
    },
    'attention': {
        'model': home + '/entity/attention/attention-entities.pb',
        'setting': home + '/entity/attention/attention-entities.json',
        'version': 'v14',
    },
}

PATH_ENTITIES_SENSITIVE = {
    'crf': {
        'model': home + '/entity-sensitive/crf/crf-entities.pkl',
        'version': 'v21',
    },
    'concat': {
        'model': home + '/entity-sensitive/concat/concat-entities.pb',
        'setting': home + '/entity-sensitive/concat/concat-entities.json',
        'version': 'v21',
    },
    'luong': {
        'model': home + '/entity-sensitive/luong/luong-entities.pb',
        'setting': home + '/entity-sensitive/luong/luong-entities.json',
        'version': 'v21',
    },
    'bahdanau': {
        'model': home + '/entity-sensitive/bahdanau/bahdanau-entities.pb',
        'setting': home + '/entity-sensitive/bahdanau/bahdanau-entities.json',
        'version': 'v21',
    },
    'entity-network': {
        'model': home + '/entity-sensitive/entity-network/entity-entities.pb',
        'setting': home
        + '/entity-sensitive/entity-network/entity-entities.json',
        'version': 'v21',
    },
    'attention': {
        'model': home + '/entity-sensitive/attention/attention-entities.pb',
        'setting': home + '/entity-sensitive/attention/attention-entities.json',
        'version': 'v21',
    },
}

S3_PATH_ENTITIES = {
    'crf': {'model': 'v8/entities/crf-entities.pkl'},
    'concat': {
        'model': 'v14/entities/concat-entities.pb',
        'setting': 'v14/entities/concat-entities.json',
    },
    'luong': {
        'model': 'v21/entities/luong-entities.pb',
        'setting': 'v21/entities/luong-entities.json',
    },
    'bahdanau': {
        'model': 'v21/entities/bahdanau-entities.pb',
        'setting': 'v21/entities/bahdanau-entities.json',
    },
    'entity-network': {
        'model': 'v21/entities/entity-entities.pb',
        'setting': 'v21/entities/entity-entities.json',
    },
    'attention': {
        'model': 'v14/entities/attention-entities.pb',
        'setting': 'v14/entities/attention-entities.json',
    },
}

S3_PATH_ENTITIES_SENSITIVE = {
    'crf': {'model': 'v21/entities-sensitive/crf-entities.pkl'},
    'concat': {
        'model': 'v21/entities-sensitive/concat-entities.pb',
        'setting': 'v21/entities-sensitive/concat-entities.json',
    },
    'luong': {
        'model': 'v21/entities-sensitive/luong-entities.pb',
        'setting': 'v21/entities-sensitive/luong-entities.json',
    },
    'bahdanau': {
        'model': 'v21/entities-sensitive/bahdanau-entities.pb',
        'setting': 'v21/entities-sensitive/bahdanau-entities.json',
    },
    'entity-network': {
        'model': 'v21/entities-sensitive/entity-entities.pb',
        'setting': 'v21/entities-sensitive/entity-entities.json',
    },
    'attention': {
        'model': 'v21/entities-sensitive/attention-entities.pb',
        'setting': 'v21/entities-sensitive/attention-entities.json',
    },
}

PATH_SENTIMENT = {
    'bahdanau': {
        'model': home + '/sentiment/bahdanau/bahdanau-sentiment.pb',
        'setting': home + '/sentiment/sentiment-dictionary.json',
        'version': 'v24',
    },
    'luong': {
        'model': home + '/sentiment/luong/luong-sentiment.pb',
        'setting': home + '/sentiment/sentiment-dictionary.json',
        'version': 'v24',
    },
    'self-attention': {
        'model': home + '/sentiment/self-attention/self-attention-sentiment.pb',
        'setting': home + '/sentiment/sentiment-dictionary.json',
        'version': 'v24',
    },
    'multinomial': {
        'model': home + '/sentiment/multinomial/multinomial-sentiment.pkl',
        'vector': home
        + '/sentiment/multinomial/multinomial-sentiment-tfidf.pkl',
        'version': 'v17',
    },
    'xgb': {
        'model': home + '/sentiment/xgb/xgboost-sentiment.pkl',
        'vector': home + '/sentiment/xgb/xgboost-sentiment-tfidf.pkl',
        'version': 'v17',
    },
    'fast-text-char': {
        'model': home
        + '/sentiment/fast-text-char/model.ckpt.data-00000-of-00001',
        'index': home + '/sentiment/fast-text-char/model.ckpt.index',
        'meta': home + '/sentiment/fast-text-char/model.ckpt.meta',
        'vector': home
        + '/sentiment/fast-text-char/vectorizer-sparse-sentiment.pkl',
        'version': 'v24',
    },
    'bert': {
        'model': home + '/sentiment/bert/bert-sentiment.pb',
        'vocab': home + '/bert/multilanguage-vocab.txt',
        'version': 'v24',
    },
}

S3_PATH_SENTIMENT = {
    'bahdanau': {
        'model': 'v24/sentiment/bahdanau-sentiment.pb',
        'setting': 'v24/sentiment/sentiment-dictionary.json',
    },
    'luong': {
        'model': 'v24/sentiment/luong-sentiment.pb',
        'setting': 'v24/sentiment/sentiment-dictionary.json',
    },
    'self-attention': {
        'model': 'v24/sentiment/self-attention-sentiment.pb',
        'setting': 'v24/sentiment/sentiment-dictionary.json',
    },
    'multinomial': {
        'model': 'v17/sentiment/multinomial-sentiment.pkl',
        'vector': 'v17/sentiment/multinomial-sentiment-tfidf.pkl',
    },
    'xgb': {
        'model': 'v17/sentiment/xgboost-sentiment.pkl',
        'vector': 'v17/sentiment/xgboost-sentiment-tfidf.pkl',
    },
    'fast-text-char': {
        'model': 'v24/sentiment/model.ckpt.data-00000-of-00001',
        'index': 'v24/sentiment/model.ckpt.index',
        'meta': 'v24/sentiment/model.ckpt.meta',
        'vector': 'v24/sentiment/vectorizer-sparse-sentiment.pkl',
    },
    'bert': {
        'model': 'v24/sentiment/bert-sentiment.pb',
        'vocab': 'v24/multilanguage-vocab.txt',
    },
}

PATH_SUBJECTIVE = {
    'bahdanau': {
        'model': home + '/subjective/bahdanau/bahdanau-subjective.pb',
        'setting': home + '/subjective/subjective-dictionary.json',
        'version': 'v24',
    },
    'luong': {
        'model': home + '/subjective/luong/luong-subjective.pb',
        'setting': home + '/subjective/subjective-dictionary.json',
        'version': 'v24',
    },
    'self-attention': {
        'model': home + '/subjective/luong/luong-subjective.pb',
        'setting': home + '/subjective/subjective-dictionary.json',
        'version': 'v24',
    },
    'multinomial': {
        'model': home + '/subjective/multinomial/multinomial-subjective.pkl',
        'vector': home
        + '/subjective/multinomial/multinomial-subjective-tfidf.pkl',
        'version': 'v10',
    },
    'xgb': {
        'model': home + '/subjective/xgb/xgboost-subjective.pkl',
        'vector': home + '/subjective/xgb/xgboost-subjective-tfidf.pkl',
        'version': 'v10',
    },
    'fast-text-char': {
        'model': home
        + '/subjective/fast-text-char/model.ckpt.data-00000-of-00001',
        'index': home + '/subjective/fast-text-char/model.ckpt.index',
        'meta': home + '/subjective/fast-text-char/model.ckpt.meta',
        'vector': home
        + '/subjective/fast-text-char/vectorizer-sparse-subjective.pkl',
        'version': 'v17',
    },
    'bert': {
        'model': home + '/subjective/bert/bert-subjective.pb',
        'vocab': home + '/bert/multilanguage-vocab.txt',
        'version': 'v24',
    },
}

S3_PATH_SUBJECTIVE = {
    'bahdanau': {
        'model': 'v24/subjective/bahdanau-subjective.pb',
        'setting': 'v24/subjective/subjective-dictionary.json',
    },
    'luong': {
        'model': 'v24/subjective/luong-subjective.pb',
        'setting': 'v24/subjective/subjective-dictionary.json',
    },
    'self-attention': {
        'model': 'v24/subjective/self-attention-subjective.pb',
        'setting': 'v24/subjective/subjective-dictionary.json',
    },
    'multinomial': {
        'model': 'v10/subjective/multinomial-subjective.pkl',
        'vector': 'v10/subjective/multinomial-subjective-tfidf.pkl',
    },
    'xgb': {
        'model': 'v10/subjective/xgboost-subjective.pkl',
        'vector': 'v10/subjective/xgboost-subjective-tfidf.pkl',
    },
    'fast-text-char': {
        'model': 'v17/subjective/model.ckpt.data-00000-of-00001',
        'index': 'v17/subjective/model.ckpt.index',
        'meta': 'v17/subjective/model.ckpt.meta',
        'vector': 'v17/subjective/vectorizer-sparse-subjective.pkl',
    },
    'bert': {
        'model': 'v24/subjective/bert-subjective.pb',
        'vocab': 'v24/multilanguage-vocab.txt',
    },
}

PATH_EMOTION = {
    'bahdanau': {
        'model': home + '/emotion/bahdanau/bahdanau-emotion.pb',
        'setting': home + '/emotion/emotion-dictionary.json',
        'version': 'v24',
    },
    'luong': {
        'model': home + '/emotion/luong/luong-emotion.pb',
        'setting': home + '/emotion/emotion-dictionary.json',
        'version': 'v24',
    },
    'self-attention': {
        'model': home + '/emotion/luong/self-attention-emotion.pb',
        'setting': home + '/emotion/emotion-dictionary.json',
        'version': 'v24',
    },
    'multinomial': {
        'model': home + '/emotion/multinomial/multinomial-emotion.pkl',
        'vector': home + '/emotion/multinomial/multinomial-emotion-tfidf.pkl',
        'version': 'v24',
    },
    'xgb': {
        'model': home + '/emotion/xgb/xgboost-emotion.pkl',
        'vector': home + '/emotion/xgb/xgboost-emotion-tfidf.pkl',
        'version': 'v24',
    },
    'fast-text-char': {
        'model': home
        + '/emotion/fast-text-char/model.ckpt.data-00000-of-00001',
        'index': home + '/emotion/fast-text-char/model.ckpt.index',
        'meta': home + '/emotion/fast-text-char/model.ckpt.meta',
        'vector': home
        + '/emotion/fast-text-char/vectorizer-sparse-emotion.pkl',
        'version': 'v24',
    },
    'bert': {
        'model': home + '/emotion/bert/bert-emotion.pb',
        'vocab': home + '/bert/multilanguage-vocab.txt',
        'version': 'v24',
    },
}

S3_PATH_EMOTION = {
    'bahdanau': {
        'model': 'v24/emotion/bahdanau-emotion.pb',
        'setting': 'v24/emotion/emotion-dictionary.json',
    },
    'luong': {
        'model': 'v24/emotion/luong-emotion.pb',
        'setting': 'v24/emotion/emotion-dictionary.json',
    },
    'self-attention': {
        'model': 'v24/emotion/self-attention-emotion.pb',
        'setting': 'v24/emotion/emotion-dictionary.json',
    },
    'multinomial': {
        'model': 'v24/emotion/multinomial-emotion.pkl',
        'vector': 'v24/emotion/multinomial-emotion-tfidf.pkl',
    },
    'xgb': {
        'model': 'v24/emotion/xgboost-emotion.pkl',
        'vector': 'v24/emotion/xgboost-emotion-tfidf.pkl',
    },
    'fast-text-char': {
        'model': 'v24/emotion/model.ckpt.data-00000-of-00001',
        'index': 'v24/emotion/model.ckpt.index',
        'meta': 'v24/emotion/model.ckpt.meta',
        'vector': 'v24/emotion/vectorizer-sparse-emotion.pkl',
    },
    'bert': {
        'model': 'v24/emotion/bert-emotion.pb',
        'vocab': 'v24/multilanguage-vocab.txt',
    },
}

PATH_DEPEND = {
    'crf': {
        'model': home + '/dependency/crf/crf-label.pkl',
        'depend': home + '/dependency/crf/crf-depend.pkl',
        'version': 'v22',
    },
    'concat': {
        'model': home + '/dependency/concat/concat-dependency.pb',
        'setting': home + '/dependency/concat/concat-dependency.json',
        'version': 'v22',
    },
    'luong': {
        'model': home + '/dependency/luong/luong-dependency.pb',
        'setting': home + '/dependency/luong/luong-dependency.json',
        'version': 'v22',
    },
    'bahdanau': {
        'model': home + '/dependency/bahdanau/bahdanau-dependency.pb',
        'setting': home + '/dependency/bahdanau/bahdanau-dependency.json',
        'version': 'v22',
    },
    'attention-is-all-you-need': {
        'model': home + '/dependency/attention/attention-dependency.pb',
        'setting': home + '/dependency/attention/attention-dependency.json',
        'version': 'v22',
    },
}

S3_PATH_DEPEND = {
    'crf': {
        'model': 'v22/dependency/crf-label.pkl',
        'depend': 'v22/dependency/crf-depend.pkl',
    },
    'concat': {
        'model': 'v22/dependency/concat-dependency.pb',
        'setting': 'v22/dependency/concat-dependency.json',
    },
    'luong': {
        'model': 'v22/dependency/luong-dependency.pb',
        'setting': 'v22/dependency/luong-dependency.json',
    },
    'bahdanau': {
        'model': 'v22/dependency/bahdanau-dependency.pb',
        'setting': 'v22/dependency/bahdanau-dependency.json',
    },
    'attention-is-all-you-need': {
        'model': 'v22/dependency/attention-dependency.pb',
        'setting': 'v22/dependency/attention-dependency.json',
    },
}

PATH_RELEVANCY = {
    'self-attention': {
        'model': home + '/relevancy/self-attention/self-attention-relevancy.pb',
        'setting': home + '/relevancy/relevancy-dictionary.json',
        'version': 'v24',
    },
    'dilated-cnn': {
        'model': home + '/relevancy/dilated-cnn/dilated-cnn-relevancy.pb',
        'setting': home + '/relevancy/relevancy-dictionary.json',
        'version': 'v24',
    },
}
S3_PATH_RELEVANCY = {
    'self-attention': {
        'model': 'v24/relevancy/self-attention-relevancy.pb',
        'setting': 'v24/relevancy/relevancy-dictionary.json',
    },
    'dilated-cnn': {
        'model': 'v24/relevancy/dilated-cnn-relevancy.pb',
        'setting': 'v24/relevancy/relevancy-dictionary.json',
    },
}

PATH_SIMILARITY = {
    'bahdanau': {
        'model': home + '/similarity/bahdanau/bahdanau-similarity.pb',
        'setting': home + '/similarity/similarity-dictionary.json',
        'version': 'v26',
    },
    'dilated-cnn': {
        'model': home + '/similarity/luong/dilated-cnn-similarity.pb',
        'setting': home + '/similarity/similarity-dictionary.json',
        'version': 'v26',
    },
    'self-attention': {
        'model': home + '/similarity/luong/self-attention-similarity.pb',
        'setting': home + '/similarity/similarity-dictionary.json',
        'version': 'v26',
    },
    'bert': {
        'model': home + '/similarity/bert/bert-similarity.pb',
        'vocab': home + '/bert/multilanguage-vocab.txt',
        'version': 'v26',
    },
}

S3_PATH_SIMILARITY = {
    'bahdanau': {
        'model': 'v26/similarity/bahdanau-similarity.pb',
        'setting': 'v26/similarity/similarity-dictionary.json',
    },
    'self-attention': {
        'model': 'v26/similarity/self-attention-similarity.pb',
        'setting': 'v26/similarity/similarity-dictionary.json',
    },
    'dilated-cnn': {
        'model': 'v26/similarity/dilated-cnn-similarity.pb',
        'setting': 'v26/similarity/similarity-dictionary.json',
    },
    'bert': {
        'model': 'v26/similarity/bert-similarity.pb',
        'vocab': 'v24/multilanguage-vocab.txt',
    },
}
