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
        'version': 'v27',
    },
    256: {
        'model': home + '/elmo-wiki/elmo-256.pb',
        'setting': home + '/elmo-wiki/elmo-256.pkl',
        'version': 'v27',
    },
}

S3_PATH_ELMO = {
    128: {'model': 'v27/elmo/elmo-128.pb', 'setting': 'v27/elmo/elmo-128.pkl'},
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
    'multilanguage': {
        'model': home + '/toxic/multilanguage/bert-toxic.pb',
        'vocab': home + '/bert/multilanguage-vocab.txt',
        'version': 'v27',
    },
    'base': {
        'model': home + '/toxic/base/bert-toxic.pb',
        'vocab': home + '/bert/sp10m.cased.v4.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v4.model',
        'version': 'v27',
    },
    'small': {
        'model': home + '/toxic/small/bert-toxic.pb',
        'vocab': home + '/bert/sp10m.cased.v4.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v4.model',
        'version': 'v27',
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
    'multilanguage': {
        'model': 'v27/toxicity/bert-multilanguage-toxicity.pb',
        'vocab': 'v24/multilanguage-vocab.txt',
    },
    'base': {
        'model': 'v27/toxicity/bert-base-toxicity.pb',
        'vocab': 'v27/sp10m.cased.v4.vocab',
        'tokenizer': 'v27/sp10m.cased.v4.model',
    },
    'small': {
        'model': 'v27/toxicity/bert-small-toxicity.pb',
        'vocab': 'v27/sp10m.cased.v4.vocab',
        'tokenizer': 'v27/sp10m.cased.v4.model',
    },
}

PATH_POS = {
    'concat': {
        'model': home + '/pos/concat/concat-pos.pb',
        'setting': home + '/pos/dictionary-pos.json',
        'version': 'v27',
    },
    'luong': {
        'model': home + '/pos/luong/luong-pos.pb',
        'setting': home + '/pos/dictionary-pos.json',
        'version': 'v27',
    },
    'bahdanau': {
        'model': home + '/pos/bahdanau/bahdanau-pos.pb',
        'setting': home + '/pos/dictionary-pos.json',
        'version': 'v27',
    },
    'multilanguage': {
        'model': home + '/pos/multilanguage/bert-multilanguage-pos.pb',
        'vocab': home + '/bert/multilanguage-vocab.txt',
        'setting': home + '/entity/dictionary-pos.json',
        'version': 'v27',
    },
    'base': {
        'model': home + '/pos/base/bert-pos.pb',
        'vocab': home + '/bert/sp10m.cased.v4.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v4.model',
        'setting': home + '/pos/dictionary-pos.json',
        'version': 'v27',
    },
    'small': {
        'model': home + '/pos/small/bert-pos.pb',
        'vocab': home + '/bert/sp10m.cased.v4.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v4.model',
        'setting': home + '/pos/dictionary-pos.json',
        'version': 'v27',
    },
}

S3_PATH_POS = {
    'concat': {
        'model': 'v27/pos/concat-pos.pb',
        'setting': 'bert-bahasa/dictionary-pos.json',
    },
    'luong': {
        'model': 'v27/pos/luong-pos.pb',
        'setting': 'bert-bahasa/dictionary-pos.json',
    },
    'bahdanau': {
        'model': 'v27/pos/bahdanau-pos.pb',
        'setting': 'bert-bahasa/dictionary-pos.json',
    },
    'multilanguage': {
        'model': 'v27/pos/bert-multilanguage-pos.pb',
        'vocab': 'v24/multilanguage-vocab.txt',
        'setting': 'bert-bahasa/dictionary-pos.json',
    },
    'base': {
        'model': 'v27/pos/bert-base-pos.pb',
        'vocab': 'v27/sp10m.cased.v4.vocab',
        'tokenizer': 'v27/sp10m.cased.v4.model',
        'setting': 'bert-bahasa/dictionary-pos.json',
    },
    'small': {
        'model': 'v27/pos/bert-small-pos.pb',
        'vocab': 'v27/sp10m.cased.v4.vocab',
        'tokenizer': 'v27/sp10m.cased.v4.model',
        'setting': 'bert-bahasa/dictionary-pos.json',
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
    'concat': {
        'model': home + '/entity/concat/concat-entities.pb',
        'setting': home + '/entity/dictionary-entities.json',
        'version': 'v27',
    },
    'luong': {
        'model': home + '/entity/luong/luong-entities.pb',
        'setting': home + '/entity/dictionary-entities.json',
        'version': 'v27',
    },
    'bahdanau': {
        'model': home + '/entity/bahdanau/bahdanau-entities.pb',
        'setting': home + '/entity/dictionary-entities.json',
        'version': 'v27',
    },
    'multilanguage': {
        'model': home + '/entity/multilanguage/bert-multilanguage-entity.pb',
        'vocab': home + '/bert/multilanguage-vocab.txt',
        'setting': home + '/entity/dictionary-entities.json',
        'version': 'v27',
    },
    'base': {
        'model': home + '/entity/base/bert-entity.pb',
        'vocab': home + '/bert/sp10m.cased.v4.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v4.model',
        'setting': home + '/entity/dictionary-entities.json',
        'version': 'v27',
    },
    'small': {
        'model': home + '/entity/small/bert-entity.pb',
        'vocab': home + '/bert/sp10m.cased.v4.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v4.model',
        'setting': home + '/entity/dictionary-entities.json',
        'version': 'v27',
    },
}

S3_PATH_ENTITIES = {
    'concat': {
        'model': 'v27/entities/concat-entities.pb',
        'setting': 'bert-bahasa/dictionary-entities.json',
    },
    'luong': {
        'model': 'v27/entities/luong-entities.pb',
        'setting': 'bert-bahasa/dictionary-entities.json',
    },
    'bahdanau': {
        'model': 'v27/entities/bahdanau-entities.pb',
        'setting': 'bert-bahasa/dictionary-entities.json',
    },
    'multilanguage': {
        'model': 'v27/entities/bert-multilanguage-entities.pb',
        'vocab': 'v24/multilanguage-vocab.txt',
        'setting': 'bert-bahasa/dictionary-entities.json',
    },
    'base': {
        'model': 'v27/entities/bert-base-entities.pb',
        'vocab': 'v27/sp10m.cased.v4.vocab',
        'tokenizer': 'v27/sp10m.cased.v4.model',
        'setting': 'bert-bahasa/dictionary-entities.json',
    },
    'small': {
        'model': 'v27/entities/bert-small-entities.pb',
        'vocab': 'v27/sp10m.cased.v4.vocab',
        'tokenizer': 'v27/sp10m.cased.v4.model',
        'setting': 'bert-bahasa/dictionary-entities.json',
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
    'multilanguage': {
        'model': home
        + '/sentiment/multilanguage/bert-multilanguage-sentiment.pb',
        'vocab': home + '/bert/multilanguage-vocab.txt',
        'version': 'v27',
    },
    'base': {
        'model': home + '/sentiment/base/bert-sentiment.pb',
        'vocab': home + '/bert/sp10m.cased.v4.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v4.model',
        'version': 'v27',
    },
    'small': {
        'model': home + '/sentiment/small/bert-sentiment.pb',
        'vocab': home + '/bert/sp10m.cased.v4.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v4.model',
        'version': 'v27',
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
    'multilanguage': {
        'model': 'v27/sentiment/bert-multilanguage-sentiment.pb',
        'vocab': 'v24/multilanguage-vocab.txt',
    },
    'base': {
        'model': 'v27/sentiment/bert-base-sentiment.pb',
        'vocab': 'v27/sp10m.cased.v4.vocab',
        'tokenizer': 'v27/sp10m.cased.v4.model',
    },
    'small': {
        'model': 'v27/sentiment/bert-small-sentiment.pb',
        'vocab': 'v27/sp10m.cased.v4.vocab',
        'tokenizer': 'v27/sp10m.cased.v4.model',
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
    'multilanguage': {
        'model': home
        + '/subjective/multilanguage/bert-multilanguage-subjective.pb',
        'vocab': home + '/bert/multilanguage-vocab.txt',
        'version': 'v27',
    },
    'base': {
        'model': home + '/subjective/base/bert-subjective.pb',
        'vocab': home + '/bert/sp10m.cased.v4.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v4.model',
        'version': 'v27',
    },
    'small': {
        'model': home + '/subjective/small/bert-subjective.pb',
        'vocab': home + '/bert/sp10m.cased.v4.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v4.model',
        'version': 'v27',
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
    'multilanguage': {
        'model': 'v27/subjective/bert-multilanguage-subjective.pb',
        'vocab': 'v24/multilanguage-vocab.txt',
    },
    'base': {
        'model': 'v27/subjective/bert-base-subjective.pb',
        'vocab': 'v27/sp10m.cased.v4.vocab',
        'tokenizer': 'v27/sp10m.cased.v4.model',
    },
    'small': {
        'model': 'v27/subjective/bert-small-subjective.pb',
        'vocab': 'v27/sp10m.cased.v4.vocab',
        'tokenizer': 'v27/sp10m.cased.v4.model',
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
    'multilanguage': {
        'model': home + '/emotion/multilanguage/bert-multilanguage-emotion.pb',
        'vocab': home + '/bert/multilanguage-vocab.txt',
        'version': 'v27',
    },
    'base': {
        'model': home + '/emotion/base/bert-emotion.pb',
        'vocab': home + '/bert/sp10m.cased.v4.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v4.model',
        'version': 'v27',
    },
    'small': {
        'model': home + '/emotion/small/bert-emotion.pb',
        'vocab': home + '/bert/sp10m.cased.v4.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v4.model',
        'version': 'v27',
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
    'multilanguage': {
        'model': 'v27/emotion/bert-multilanguage-emotion.pb',
        'vocab': 'v24/multilanguage-vocab.txt',
    },
    'base': {
        'model': 'v27/emotion/bert-base-emotion.pb',
        'vocab': 'v27/sp10m.cased.v4.vocab',
        'tokenizer': 'v27/sp10m.cased.v4.model',
    },
    'small': {
        'model': 'v27/emotion/bert-small-emotion.pb',
        'vocab': 'v27/sp10m.cased.v4.vocab',
        'tokenizer': 'v27/sp10m.cased.v4.model',
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
    'multilanguage': {
        'model': home + '/relevancy/multilanguage/bert-relevancy.pb',
        'vocab': home + '/bert/multilanguage-vocab.txt',
        'version': 'v27',
    },
    'base': {
        'model': home + '/relevancy/base/bert-relevancy.pb',
        'vocab': home + '/bert/sp10m.cased.v4.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v4.model',
        'version': 'v27',
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
    'multilanguage': {
        'model': 'v27/relevancy/bert-multilanguage-relevancy.pb',
        'vocab': 'v24/multilanguage-vocab.txt',
    },
    'base': {
        'model': 'v27/relevancy/bert-base-relevancy.pb',
        'vocab': 'v27/sp10m.cased.v4.vocab',
        'tokenizer': 'v27/sp10m.cased.v4.model',
    },
}

PATH_SIMILARITY = {
    'multilanguage': {
        'model': home + '/similarity/multilanguage/bert.pb',
        'vocab': home + '/bert/multilanguage-vocab.txt',
        'version': 'v24',
    },
    'base': {
        'model': home + '/base/base/bert.pb',
        'vocab': home + '/bert/sp10m.cased.v4.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v4.model',
        'version': 'v27',
    },
    'small': {
        'model': home + '/small/small/bert.pb',
        'vocab': home + '/bert/sp10m.cased.v4.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v4.model',
        'version': 'v27',
    },
}

S3_PATH_SIMILARITY = {
    'multilanguage': {
        'model': 'v26/similarity/bert-similarity.pb',
        'vocab': 'v24/multilanguage-vocab.txt',
    },
    'base': {
        'model': 'v27/similarity/bert-base-similarity.pb',
        'vocab': 'v27/sp10m.cased.v4.vocab',
        'tokenizer': 'v27/sp10m.cased.v4.model',
    },
    'small': {
        'model': 'v27/similarity/bert-small-similarity.pb',
        'vocab': 'v27/sp10m.cased.v4.vocab',
        'tokenizer': 'v27/sp10m.cased.v4.model',
    },
}

PATH_BERT = {
    'base': {
        'path': home + '/bert-model/base',
        'directory': home + '/bert-model/base/bert-bahasa-base/',
        'model': {
            'model': home + '/bert-model/base/bert-bahasa-base.tar.gz',
            'version': 'v27',
        },
    },
    'small': {
        'path': home + '/bert-model/small',
        'directory': home + '/bert-model/small/bert-bahasa-small/',
        'model': {
            'model': home + '/bert-model/small/bert-bahasa-small.tar.gz',
            'version': 'v27',
        },
    },
    'multilanguage': {
        'path': home + '/bert-model/multilanguage',
        'directory': home
        + '/bert-model/multilanguage/multi_cased_L-12_H-768_A-12/',
        'model': {
            'model': home
            + '/bert-model/multilanguage/multi_cased_L-12_H-768_A-12.zip',
            'version': 'v27',
        },
    },
}

S3_PATH_BERT = {
    'base': {'model': 'bert-bahasa/bert-bahasa-base.tar.gz'},
    'small': {'model': 'bert-bahasa/bert-bahasa-small.tar.gz'},
    'multilanguage': {
        'model': 'https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip'
    },
}

PATH_XLNET = {
    'base': {
        'path': home + '/xlnet-model/base',
        'directory': home + '/xlnet-model/base/9-july-2019/',
        'model': {
            'model': home + '/xlnet-model/base/xlnet-9-july-2019-v2.tar.gz',
            'version': 'v27',
        },
    },
    'small': {
        'path': home + '/xlnet-model/small',
        'directory': home + '/xlnet-model/small/xlnet-bahasa-small/',
        'model': {
            'model': home + '/xlnet-model/small/xlnet-bahasa-small.tar.gz',
            'version': 'v27',
        },
    },
}

S3_PATH_XLNET = {
    'base': {'model': 'bert-bahasa/xlnet-9-july-2019-v2.tar.gz'},
    'small': {'model': 'bert-bahasa/xlnet-bahasa-small.tar.gz'},
}
