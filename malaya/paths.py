from . import home

MALAY_TEXT = home + '/malay-text.txt'

PATH_SUMMARIZE = {
    'model': home + '/summary_frozen_model.pb',
    'setting': home + '/dictionary-summary.json',
}

S3_PATH_SUMMARIZE = {
    'model': 'v7/summary/summary_frozen_model.pb',
    'setting': 'v7/summary/dictionary-summary.json',
}

PATH_TOXIC = {
    'multinomial': {
        'model': home + '/multinomials-toxic.pkl',
        'vector': home + '/vectorizer-toxic.pkl',
    },
    'logistic': {
        'model': home + '/logistics-toxic.pkl',
        'vector': home + '/vectorizer-toxic.pkl',
    },
    'luong': {
        'model': home + '/luong-toxic.pb',
        'setting': home + '/luong-toxic.json',
    },
    'bahdanau': {
        'model': home + '/bahdanau-toxic.pb',
        'setting': home + '/bahdanau-toxic.json',
    },
    'hierarchical': {
        'model': home + '/hierarchical-toxic.pb',
        'setting': home + '/hierarchical-toxic.json',
    },
    'fast-text': {
        'model': home + '/fasttext-toxic.pb',
        'setting': home + '/fasttext-toxic.json',
        'pickle': home + '/fasttext-toxic.pkl',
    },
    'entity-network': {
        'model': home + '/entity-toxic.pb',
        'setting': home + '/entity-toxic.json',
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
        'model': 'v8/toxic/luong-toxic.pb',
        'setting': 'v8/toxic/luong-toxic.json',
    },
    'bahdanau': {
        'model': 'v8/toxic/bahdanau-toxic.pb',
        'setting': 'v8/toxic/bahdanau-toxic.json',
    },
    'hierarchical': {
        'model': 'v8/toxic/hierarchical-toxic.pb',
        'setting': 'v8/toxic/hierarchical-toxic.json',
    },
    'fast-text': {
        'model': 'v8/toxic/fasttext-toxic.pb',
        'setting': 'v8/toxic/fasttext-toxic.json',
        'pickle': 'v8/toxic/fasttext-toxic.pkl',
    },
    'entity-network': {
        'model': 'v8/toxic/entity-toxic.pb',
        'setting': 'v8/toxic/entity-toxic.json',
    },
}

PATH_POS = {
    'crf': {'model': home + '/crf-pos.pkl'},
    'xgb': {
        'model': home + '/xgb-pos.pkl',
        'vector': home + '/xgb-bow-pos.pkl',
    },
    'concat': {
        'model': home + '/concat-pos.pb',
        'setting': home + '/concat-pos.json',
    },
    'luong': {
        'model': home + '/luong-pos.pb',
        'setting': home + '/luong-pos.json',
    },
    'bahdanau': {
        'model': home + '/bahdanau-pos.pb',
        'setting': home + '/bahdanau-pos.json',
    },
    'entity-network': {
        'model': home + '/entity-pos.pb',
        'setting': home + '/entity-pos.json',
    },
}

S3_PATH_POS = {
    'crf': {'model': 'v8/pos/crf-pos.pkl'},
    'concat': {
        'model': 'v8/pos/concat-pos.pb',
        'setting': 'v8/pos/concat-pos.json',
    },
    'luong': {
        'model': 'v8/pos/luong-pos.pb',
        'setting': 'v8/pos/luong-pos.json',
    },
    'bahdanau': {
        'model': 'v8/pos/bahdanau-pos.pb',
        'setting': 'v8/pos/bahdanau-pos.json',
    },
    'entity-network': {
        'model': 'v8/pos/entity-pos.pb',
        'setting': 'v8/pos/entity-pos.json',
    },
}

PATH_NORMALIZER = {
    'deep': {
        'model': home + '/normalizer-deep.pb',
        'setting': home + '/normalizer-deep.json',
    }
}

S3_PATH_NORMALIZER = {
    'deep': {
        'model': 'v6/normalizer-deep.pb',
        'setting': 'v6/normalizer-deep.json',
    }
}

PATH_LANG_DETECTION = {
    'multinomial': {
        'model': home + '/multinomial-language-detection.pkl',
        'vector': home + '/vectorizer-language-detection.pkl',
    },
    'xgb': {
        'model': home + '/xgb-language-detection.pkl',
        'vector': home + '/vectorizer-language-detection.pkl',
    },
    'sgd': {
        'model': home + '/sgd-language-detection.pkl',
        'vector': home + '/vectorizer-language-detection.pkl',
    },
}

S3_PATH_LANG_DETECTION = {
    'multinomial': {
        'model': 'v8/language-detection/multinomial-language-detection.pkl',
        'vector': 'v8/language-detection/language-detection-vectorizer.pkl',
    },
    'xgb': {
        'model': 'v8/language-detection/xgboost-language-detection.pkl',
        'vector': 'v8/language-detection/language-detection-vectorizer.pkl',
    },
    'sgd': {
        'model': 'v8/language-detection/sgd-language-detection.pkl',
        'vector': 'v8/language-detection/language-detection-vectorizer.pkl',
    },
}

PATH_ENTITIES = {
    'crf': {'model': home + '/crf-entities.pkl'},
    'concat': {
        'model': home + '/concat-entities.pb',
        'setting': home + '/concat-entities.json',
    },
    'luong': {
        'model': home + '/luong-entities.pb',
        'setting': home + '/luong-entities.json',
    },
    'bahdanau': {
        'model': home + '/bahdanau-entities.pb',
        'setting': home + '/bahdanau-entities.json',
    },
    'entity-network': {
        'model': home + '/entity-entities.pb',
        'setting': home + '/entity-entities.json',
    },
    'attention': {
        'model': home + '/attention-entities.pb',
        'setting': home + '/attention-entities.json',
    },
}

S3_PATH_ENTITIES = {
    'crf': {'model': 'v8/entities/crf-entities.pkl'},
    'concat': {
        'model': 'v8/entities/concat-entities.pb',
        'setting': 'v8/entities/concat-entities.json',
    },
    'luong': {
        'model': 'v8/entities/luong-entities.pb',
        'setting': 'v8/entities/luong-entities.json',
    },
    'bahdanau': {
        'model': 'v8/entities/bahdanau-entities.pb',
        'setting': 'v8/entities/bahdanau-entities.json',
    },
    'entity-network': {
        'model': 'v8/entities/entity-entities.pb',
        'setting': 'v8/entities/entity-entities.json',
    },
    'attention': {
        'model': 'v8/entities/attention-entities.pb',
        'setting': 'v8/entities/attention-entities.json',
    },
}

PATH_SENTIMENTS = {
    'bert': {
        'model': home + '/bert-sentiment.pb',
        'setting': home + '/bert-sentiment.json',
    },
    'entity-network': {
        'model': home + '/entity-sentiment.pb',
        'setting': home + '/entity-sentiment.json',
    },
    'hierarchical': {
        'model': home + '/hierarchical-sentiment.pb',
        'setting': home + '/hierarchical-sentiment.json',
    },
    'bahdanau': {
        'model': home + '/bahdanau-sentiment.pb',
        'setting': home + '/bahdanau-sentiment.json',
    },
    'luong': {
        'model': home + '/luong-sentiment.pb',
        'setting': home + '/luong-sentiment.json',
    },
    'bidirectional': {
        'model': home + '/bidirectional-sentiment.pb',
        'setting': home + '/bidirectional-sentiment.json',
    },
    'fast-text': {
        'model': home + '/fasttext-sentiment.pb',
        'setting': home + '/fasttext-sentiment.json',
        'pickle': home + '/fasttext-sentiment.pkl',
    },
    'multinomial': {
        'model': home + '/multinomial-sentiment.pkl',
        'vector': home + '/multinomial-sentiment-tfidf.pkl',
    },
    'xgb': {
        'model': home + '/xgboost-sentiment.pkl',
        'vector': home + '/xgboost-sentiment-tfidf.pkl',
    },
}

S3_PATH_SENTIMENTS = {
    'bert': {
        'model': 'v8/sentiment/bert-sentiment.pb',
        'setting': 'v8/sentiment/bert-sentiment.json',
    },
    'entity-network': {
        'model': 'v8/sentiment/entity-sentiment.pb',
        'setting': 'v8/sentiment/entity-sentiment.json',
    },
    'hierarchical': {
        'model': 'v8/sentiment/hierarchical-sentiment.pb',
        'setting': 'v8/sentiment/hierarchical-sentiment.json',
    },
    'bahdanau': {
        'model': 'v8/sentiment/bahdanau-sentiment.pb',
        'setting': 'v8/sentiment/bahdanau-sentiment.json',
    },
    'luong': {
        'model': 'v8/sentiment/luong-sentiment.pb',
        'setting': 'v8/sentiment/luong-sentiment.json',
    },
    'bidirectional': {
        'model': 'v8/sentiment/bidirectional-sentiment.pb',
        'setting': 'v8/sentiment/bidirectional-sentiment.json',
    },
    'fast-text': {
        'model': 'v8/sentiment/fasttext-sentiment.pb',
        'setting': 'v8/sentiment/fasttext-sentiment.json',
        'pickle': 'v8/sentiment/fasttext-sentiment.pkl',
    },
    'multinomial': {
        'model': 'v8/sentiment/multinomial-sentiment.pkl',
        'vector': 'v8/sentiment/multinomial-sentiment-tfidf.pkl',
    },
    'xgb': {
        'model': 'v8/sentiment/xgboost-sentiment.pkl',
        'vector': 'v8/sentiment/xgboost-sentiment-tfidf.pkl',
    },
}
