from .. import home

MALAY_TEXT = home + '/dictionary/malay-text.txt'

PATH_STEM = {
    'model': home + '/stem/stemmer-deep.pb',
    'setting': home + '/stem/stemmer-deep.json',
    'version': 'v5',
}

S3_PATH_STEM = {
    'model': 'v5/stemmer-frozen-model.pb',
    'setting': 'v5/stemmer-deep.json',
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
        'setting': home + '/toxic/luong/luong-toxic.json',
        'version': 'v8',
    },
    'bahdanau': {
        'model': home + '/toxic/bahdanau/bahdanau-toxic.pb',
        'setting': home + '/toxic/bahdanau/bahdanau-toxic.json',
        'version': 'v8',
    },
    'hierarchical': {
        'model': home + '/toxic/hierarchical/hierarchical-toxic.pb',
        'setting': home + '/toxic/hierarchical/hierarchical-toxic.json',
        'version': 'v8',
    },
    'fast-text': {
        'model': home + '/toxic/fast-text/fasttext-toxic.pb',
        'setting': home + '/toxic/fast-text/fasttext-toxic.json',
        'pickle': home + '/toxic/fast-text/fasttext-toxic.pkl',
        'version': 'v8',
    },
    'entity-network': {
        'model': home + '/toxic/entity-network/entity-toxic.pb',
        'setting': home + '/toxic/entity-network/entity-toxic.json',
        'version': 'v8',
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
    'crf': {'model': home + '/pos/crf/crf-pos.pkl', 'version': 'v8'},
    'concat': {
        'model': home + '/pos/concat/concat-pos.pb',
        'setting': home + '/pos/concat/concat-pos.json',
        'version': 'v8',
    },
    'luong': {
        'model': home + '/pos/luong/luong-pos.pb',
        'setting': home + '/pos/luong/luong-pos.json',
        'version': 'v8',
    },
    'bahdanau': {
        'model': home + '/pos/bahdanau/bahdanau-pos.pb',
        'setting': home + '/pos/bahdanau/bahdanau-pos.json',
        'version': 'v8',
    },
    'entity-network': {
        'model': home + '/pos/entity-network/entity-pos.pb',
        'setting': home + 'pos/entity-network/entity-pos.json',
        'version': 'v8',
    },
    'attention': {
        'model': home + '/pos/attention/attention-pos.pb',
        'setting': home + '/pos/attention/attention-pos.json',
        'version': 'v8',
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
    'attention': {
        'model': 'v8/pos/attention-pos.pb',
        'setting': 'v8/pos/attention-pos.json',
    },
}

PATH_NORMALIZER = {
    'deep': {
        'model': home + '/normalizer/normalizer-deep.pb',
        'setting': home + '/normalizer/normalizer-deep.json',
        'version': 'v6',
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
        'version': 'v8',
    },
    'luong': {
        'model': home + '/entity/luong/luong-entities.pb',
        'setting': home + '/entity/luong/luong-entities.json',
        'version': 'v8',
    },
    'bahdanau': {
        'model': home + '/entity/bahdanau/bahdanau-entities.pb',
        'setting': home + '/entity/bahdanau/bahdanau-entities.json',
        'version': 'v8',
    },
    'entity-network': {
        'model': home + '/entity/entity-network/entity-entities.pb',
        'setting': home + '/entity/entity-network/entity-entities.json',
        'version': 'v8',
    },
    'attention': {
        'model': home + '/entity/attention/attention-entities.pb',
        'setting': home + '/entity/attention/attention-entities.json',
        'version': 'v8',
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
        'model': home + '/sentiment/bert/bert-sentiment.pb',
        'setting': home + '/sentiment/bert/bert-sentiment.json',
        'version': 'v8',
    },
    'entity-network': {
        'model': home + '/sentiment/entity-network/entity-sentiment.pb',
        'setting': home + '/sentiment/entity-network/entity-sentiment.json',
        'version': 'v8',
    },
    'hierarchical': {
        'model': home + '/sentiment/hierarchical/hierarchical-sentiment.pb',
        'setting': home + '/sentiment/hierarchical/hierarchical-sentiment.json',
        'version': 'v8',
    },
    'bahdanau': {
        'model': home + '/sentiment/bahdanau/bahdanau-sentiment.pb',
        'setting': home + '/sentiment/bahdanau/bahdanau-sentiment.json',
        'version': 'v8',
    },
    'luong': {
        'model': home + '/sentiment/luong/luong-sentiment.pb',
        'setting': home + '/sentiment/luong/luong-sentiment.json',
        'version': 'v8',
    },
    'bidirectional': {
        'model': home + '/sentiment/bidirectional/bidirectional-sentiment.pb',
        'setting': home
        + '/sentiment/bidirectional/bidirectional-sentiment.json',
        'version': 'v8',
    },
    'fast-text': {
        'model': home + '/sentiment/fast-text/fasttext-sentiment.pb',
        'setting': home + '/sentiment/fast-text/fasttext-sentiment.json',
        'pickle': home + '/sentiment/fast-text/fasttext-sentiment.pkl',
        'version': 'v8',
    },
    'multinomial': {
        'model': home + '/sentiment/multinomial/multinomial-sentiment.pkl',
        'vector': home
        + '/sentiment/multinomial/multinomial-sentiment-tfidf.pkl',
        'version': 'v8',
    },
    'xgb': {
        'model': home + '/sentiment/xgb/xgboost-sentiment.pkl',
        'vector': home + '/sentiment/xgb/xgboost-sentiment-tfidf.pkl',
        'version': 'v8',
    },
    'fast-text-char': {
        'model': home
        + '/sentiment/fast-text-char/model.ckpt.data-00000-of-00001',
        'index': home + '/sentiment/fast-text-char/model.ckpt.index',
        'meta': home + '/sentiment/fast-text-char/model.ckpt.meta',
        'vector': home
        + '/sentiment/fast-text-char/vectorizer-sparse-sentiment.pkl',
        'version': 'v12',
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
    'fast-text-char': {
        'model': 'v12/sentiment/model.ckpt.data-00000-of-00001',
        'index': 'v12/sentiment/model.ckpt.index',
        'meta': 'v12/sentiment/model.ckpt.meta',
        'vector': 'v12/sentiment/vectorizer-sparse-sentiment.pkl',
    },
}

PATH_SUBJECTIVE = {
    'bert': {
        'model': home + '/subjective/bert/bert-subjective.pb',
        'setting': home + '/subjective/bert/bert-subjective.json',
        'version': 'v10',
    },
    'entity-network': {
        'model': home + '/subjective/entity-network/entity-subjective.pb',
        'setting': home + '/subjective/entity-network/entity-subjective.json',
        'version': 'v10',
    },
    'hierarchical': {
        'model': home + '/subjective/hierarchical/hierarchical-subjective.pb',
        'setting': home
        + '/subjective/hierarchical/hierarchical-subjective.json',
        'version': 'v10',
    },
    'bahdanau': {
        'model': home + '/subjective/bahdanau/bahdanau-subjective.pb',
        'setting': home + '/subjective/bahdanau/bahdanau-subjective.json',
        'version': 'v10',
    },
    'luong': {
        'model': home + '/subjective/luong/luong-subjective.pb',
        'setting': home + '/subjective/luong/luong-subjective.json',
        'version': 'v10',
    },
    'bidirectional': {
        'model': home + '/subjective/bidirectional/bidirectional-subjective.pb',
        'setting': home
        + '/subjective/bidirectional/bidirectional-subjective.json',
        'version': 'v10',
    },
    'fast-text': {
        'model': home + '/subjective/fast-text/fasttext-subjective.pb',
        'setting': home + '/subjective/fast-text/fasttext-subjective.json',
        'pickle': home + '/subjective/fast-text/fasttext-subjective.pkl',
        'version': 'v10',
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
        'version': 'v12',
    },
}

S3_PATH_SUBJECTIVE = {
    'bert': {
        'model': 'v10/subjective/bert-subjective.pb',
        'setting': 'v10/subjective/bert-subjective.json',
    },
    'entity-network': {
        'model': 'v10/subjective/entity-subjective.pb',
        'setting': 'v10/subjective/entity-subjective.json',
    },
    'hierarchical': {
        'model': 'v10/subjective/hierarchical-subjective.pb',
        'setting': 'v10/subjective/hierarchical-subjective.json',
    },
    'bahdanau': {
        'model': 'v10/subjective/bahdanau-subjective.pb',
        'setting': 'v10/subjective/bahdanau-subjective.json',
    },
    'luong': {
        'model': 'v10/subjective/luong-subjective.pb',
        'setting': 'v10/subjective/luong-subjective.json',
    },
    'bidirectional': {
        'model': 'v10/subjective/bidirectional-subjective.pb',
        'setting': 'v10/subjective/bidirectional-subjective.json',
    },
    'fast-text': {
        'model': 'v10/subjective/fasttext-subjective.pb',
        'setting': 'v10/subjective/fasttext-subjective.json',
        'pickle': 'v10/subjective/fasttext-subjective.pkl',
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
        'model': 'v12/subjective/model.ckpt.data-00000-of-00001',
        'index': 'v12/subjective/model.ckpt.index',
        'meta': 'v12/subjective/model.ckpt.meta',
        'vector': 'v12/subjective/vectorizer-sparse-subjective.pkl',
    },
}

PATH_EMOTION = {
    'bert': {
        'model': home + '/emotion/bert/bert-emotion.pb',
        'setting': home + '/emotion/bert/bert-emotion.json',
        'version': 'v12',
    },
    'entity-network': {
        'model': home + '/emotion/entity-network/entity-emotion.pb',
        'setting': home + '/emotion/entity-network/entity-emotion.json',
        'version': 'v12',
    },
    'hierarchical': {
        'model': home + '/emotion/hierarchical/hierarchical-emotion.pb',
        'setting': home + '/emotion/hierarchical/hierarchical-emotion.json',
        'version': 'v12',
    },
    'bahdanau': {
        'model': home + '/emotion/bahdanau/bahdanau-emotion.pb',
        'setting': home + '/emotion/bahdanau/bahdanau-emotion.json',
        'version': 'v12',
    },
    'luong': {
        'model': home + '/emotion/luong/luong-emotion.pb',
        'setting': home + '/emotion/luong/luong-emotion.json',
        'version': 'v12',
    },
    'bidirectional': {
        'model': home + '/emotion/bidirectional/bidirectional-emotion.pb',
        'setting': home + '/emotion/bidirectional/bidirectional-emotion.json',
        'version': 'v12',
    },
    'fast-text': {
        'model': home + '/emotion/fast-text/fasttext-emotion.pb',
        'setting': home + '/emotion/fast-text/fasttext-emotion.json',
        'pickle': home + '/emotion/fast-text/fasttext-emotion.pkl',
        'version': 'v12',
    },
    'multinomial': {
        'model': home + '/emotion/multinomial/multinomial-emotion.pkl',
        'vector': home + '/emotion/multinomial/multinomial-emotion-tfidf.pkl',
        'version': 'v12',
    },
    'xgb': {
        'model': home + '/emotion/xgb/xgboost-emotion.pkl',
        'vector': home + '/emotion/xgb/xgboost-emotion-tfidf.pkl',
        'version': 'v12',
    },
    'fast-text-char': {
        'model': home
        + '/emotion/fast-text-char/model.ckpt.data-00000-of-00001',
        'index': home + '/emotion/fast-text-char/model.ckpt.index',
        'meta': home + '/emotion/fast-text-char/model.ckpt.meta',
        'vector': home
        + '/emotion/fast-text-char/vectorizer-sparse-emotion.pkl',
        'version': 'v12',
    },
}

S3_PATH_EMOTION = {
    'bert': {
        'model': 'v12/emotion/bert-emotion.pb',
        'setting': 'v12/emotion/bert-emotion.json',
    },
    'entity-network': {
        'model': 'v12/emotion/entity-emotion.pb',
        'setting': 'v12/emotion/entity-emotion.json',
    },
    'hierarchical': {
        'model': 'v12/emotion/hierarchical-emotion.pb',
        'setting': 'v12/emotion/hierarchical-emotion.json',
    },
    'bahdanau': {
        'model': 'v12/emotion/bahdanau-emotion.pb',
        'setting': 'v12/emotion/bahdanau-emotion.json',
    },
    'luong': {
        'model': 'v12/emotion/luong-emotion.pb',
        'setting': 'v12/emotion/luong-emotion.json',
    },
    'bidirectional': {
        'model': 'v12/emotion/bidirectional-emotion.pb',
        'setting': 'v12/emotion/bidirectional-emotion.json',
    },
    'fast-text': {
        'model': 'v12/emotion/fasttext-emotion.pb',
        'setting': 'v12/emotion/fasttext-emotion.json',
        'pickle': 'v12/emotion/fasttext-emotion.pkl',
    },
    'multinomial': {
        'model': 'v12/emotion/multinomial-emotion.pkl',
        'vector': 'v12/emotion/multinomial-emotion-tfidf.pkl',
    },
    'xgb': {
        'model': 'v12/emotion/xgboost-emotion.pkl',
        'vector': 'v12/emotion/xgboost-emotion-tfidf.pkl',
    },
    'fast-text-char': {
        'model': 'v12/emotion/model.ckpt.data-00000-of-00001',
        'index': 'v12/emotion/model.ckpt.index',
        'meta': 'v12/emotion/model.ckpt.meta',
        'vector': 'v12/emotion/vectorizer-sparse-emotion.pkl',
    },
}
