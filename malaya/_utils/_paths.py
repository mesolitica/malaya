from .. import home

MALAY_TEXT = home + '/dictionary/malay-text.txt'
MALAY_TEXT_200K = home + '/dictionary-200k/malay-text.txt'

PATH_NGRAM = {
    1: {
        'model': home + '/preprocessing/ngram1/bm_1grams.json',
        'version': 'v28',
    },
    2: {
        'model': home + '/preprocessing/ngram2/bm_2grams.json',
        'version': 'v23',
    },
    'symspell': {
        'model': home + '/preprocessing/symspell/bm_1grams.txt',
        'version': 'v28',
    },
    'sentencepiece': {
        'vocab': home + '/bert/sp10m.cased.v4.vocab',
        'model': home + '/bert/sp10m.cased.v4.model',
        'version': 'v31',
    },
}

S3_PATH_NGRAM = {
    1: {'model': 'v27/preprocessing/bm_1grams.json'},
    2: {'model': 'v23/preprocessing/bm_2grams.json'},
    'symspell': {'model': 'v27/preprocessing/bm_1grams.txt'},
    'sentencepiece': {
        'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
        'model': 'bert-bahasa/sp10m.cased.v4.model',
    },
}

PATH_PREPROCESSING = {
    1: {
        'model': home + '/preprocessing/count1/1counts_1grams.json',
        'version': 'v23',
    },
    2: {
        'model': home + '/preprocessing/count2/counts_2grams.json',
        'version': 'v23',
    },
    'english-malay': {
        'model': home + '/preprocessing/english-malay/english-malay-200k.json',
        'version': 'v23',
    },
}

S3_PATH_PREPROCESSING = {
    1: {'model': 'v23/preprocessing/counts_1grams.json'},
    2: {'model': 'v23/preprocessing/counts_2grams.json'},
    'english-malay': {'model': 'v23/preprocessing/english-malay-200k.json'},
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
        'model': home + '/toxicity/multinomial/multinomial-toxicity.pkl',
        'vector': home + '/toxicity/multinomial/multinomial-toxicity-tfidf.pkl',
        'version': 'v30',
    },
    'bert': {
        'base': {
            'model': home + '/toxicity/bert/base/bert-toxicity.pb',
            'vocab': home + '/bert/sp10m.cased.v4.vocab',
            'tokenizer': home + '/bert/sp10m.cased.v4.model',
            'version': 'v30',
        },
        'small': {
            'model': home + '/toxicity/bert/small/bert-toxicity.pb',
            'vocab': home + '/bert/sp10m.cased.v4.vocab',
            'tokenizer': home + '/bert/sp10m.cased.v4.model',
            'version': 'v30',
        },
    },
    'albert': {
        'base': {
            'model': home + '/toxicity/albert/base/albert-toxicity.pb',
            'vocab': home + '/albert/sp10m.cased.v6.vocab',
            'tokenizer': home + '/albert/sp10m.cased.v6.model',
            'version': 'v30',
        }
    },
    'xlnet': {
        'base': {
            'model': home + '/toxicity/xlnet/base/albert-toxicity.pb',
            'vocab': home + '/xlnet/sp10m.cased.v5.vocab',
            'tokenizer': home + '/xlnet/sp10m.cased.v5.model',
            'version': 'v30',
        }
    },
}

S3_PATH_TOXIC = {
    'multinomial': {
        'model': 'v30/toxicity/multinomial-toxicity.pkl',
        'vector': 'v30/toxicity/multinomial-toxicity-tfidf.pkl',
    },
    'bert': {
        'base': {
            'model': 'v30/toxicity/bert-base-toxicity.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v4.model',
        },
        'small': {
            'model': 'v30/toxicity/bert-small-toxicity.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v4.model',
        },
    },
    'albert': {
        'base': {
            'model': 'v30/toxicity/albert-base-toxicity.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v6.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v6.model',
        }
    },
    'xlnet': {
        'base': {
            'model': 'v30/toxicity/xlnet-base-toxicity.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v5.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v5.model',
        }
    },
}

PATH_POS = {
    'bert': {
        'base': {
            'model': home + '/pos/bert/base/bert-pos.pb',
            'vocab': home + '/bert/sp10m.cased.v4.vocab',
            'tokenizer': home + '/bert/sp10m.cased.v4.model',
            'setting': home + '/pos/dictionary-pos.json',
            'version': 'v30',
        },
        'small': {
            'model': home + '/pos/bert/small/bert-pos.pb',
            'vocab': home + '/bert/sp10m.cased.v4.vocab',
            'tokenizer': home + '/bert/sp10m.cased.v4.model',
            'setting': home + '/pos/dictionary-pos.json',
            'version': 'v30',
        },
    },
    'xlnet': {
        'base': {
            'model': home + '/pos/xlnet/base/xlnet-pos.pb',
            'vocab': home + '/xlnet/sp10m.cased.v5.vocab',
            'tokenizer': home + '/xlnet/sp10m.cased.v5.model',
            'setting': home + '/pos/dictionary-pos.json',
            'version': 'v30',
        }
    },
    'albert': {
        'base': {
            'model': home + '/pos/albert/base/xlnet-pos.pb',
            'vocab': home + '/albert/sp10m.cased.v6.vocab',
            'tokenizer': home + '/albert/sp10m.cased.v6.model',
            'setting': home + '/pos/dictionary-pos.json',
            'version': 'v30',
        }
    },
}

S3_PATH_POS = {
    'bert': {
        'base': {
            'model': 'v30/pos/bert-base-pos.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v4.model',
            'setting': 'bert-bahasa/dictionary-pos.json',
        },
        'small': {
            'model': 'v30/pos/bert-small-pos.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v4.model',
            'setting': 'bert-bahasa/dictionary-pos.json',
        },
    },
    'xlnet': {
        'base': {
            'model': 'v30/pos/xlnet-base-pos.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v5.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v5.model',
            'setting': 'bert-bahasa/dictionary-pos.json',
        }
    },
    'albert': {
        'base': {
            'model': 'v30/pos/albert-base-pos.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v6.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v6.model',
            'setting': 'bert-bahasa/dictionary-pos.json',
        }
    },
}


PATH_LANG_DETECTION = {
    'fasttext-original': {
        'model': home + '/language-detection/fasttext-original/fasstext.bin',
        'version': 'v33',
    },
    'fasttext-quantized': {
        'model': home + '/language-detection/fasttext-quantized/fasstext.tfz',
        'version': 'v33',
    },
    'deep': {
        'model': home
        + '/language-detection/deep/model.ckpt.data-00000-of-00001',
        'index': home + '/language-detection/deep/model.ckpt.index',
        'meta': home + '/language-detection/deep/model.ckpt.meta',
        'vector': home
        + '/language-detection/deep/vectorizer-language-detection.pkl',
        'bpe': home + '/language-detection/deep/bpe.model',
        'version': 'v33',
    },
}

S3_PATH_LANG_DETECTION = {
    'fasttext-original': {
        'model': 'v33/language-detection/fasttext-malaya.bin'
    },
    'fasttext-quantized': {
        'model': 'v33/language-detection/fasttext-malaya.ftz'
    },
    'deep': {
        'model': 'v33/language-detection/model.ckpt.data-00000-of-00001',
        'index': 'v33/language-detection/model.ckpt.index',
        'meta': 'v33/language-detection/model.ckpt.meta',
        'vector': 'v33/language-detection/bow-language-detection.pkl',
        'bpe': 'v33/language-detection/language-detection.model',
    },
}

PATH_ENTITIES = {
    'bert': {
        'base': {
            'model': home + '/entity/bert/base/bert-entity.pb',
            'vocab': home + '/bert/sp10m.cased.v4.vocab',
            'tokenizer': home + '/bert/sp10m.cased.v4.model',
            'setting': home + '/entity/dictionary-entities.json',
            'version': 'v30',
        },
        'small': {
            'model': home + '/entity/bert/small/bert-entity.pb',
            'vocab': home + '/bert/sp10m.cased.v4.vocab',
            'tokenizer': home + '/bert/sp10m.cased.v4.model',
            'setting': home + '/entity/dictionary-entities.json',
            'version': 'v30',
        },
    },
    'xlnet': {
        'base': {
            'model': home + '/entity/xlnet/base/xlnet-entity.pb',
            'vocab': home + '/xlnet/sp10m.cased.v5.vocab',
            'tokenizer': home + '/xlnet/sp10m.cased.v5.model',
            'setting': home + '/entity/dictionary-entities.json',
            'version': 'v30',
        }
    },
    'albert': {
        'base': {
            'model': home + '/entity/albert/base/xlnet-entity.pb',
            'vocab': home + '/albert/sp10m.cased.v6.vocab',
            'tokenizer': home + '/albert/sp10m.cased.v6.model',
            'setting': home + '/entity/dictionary-entities.json',
            'version': 'v30',
        }
    },
}

S3_PATH_ENTITIES = {
    'bert': {
        'base': {
            'model': 'v30/entity/bert-base-entity.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v4.model',
            'setting': 'bert-bahasa/dictionary-entities.json',
        },
        'small': {
            'model': 'v30/entity/bert-small-entity.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v4.model',
            'setting': 'bert-bahasa/dictionary-entities.json',
        },
    },
    'xlnet': {
        'base': {
            'model': 'v30/entity/xlnet-base-entity.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v5.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v5.model',
            'setting': 'bert-bahasa/dictionary-entities.json',
        }
    },
    'albert': {
        'base': {
            'model': 'v30/entity/albert-base-entity.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v6.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v6.model',
            'setting': 'bert-bahasa/dictionary-entities.json',
        }
    },
}

PATH_SENTIMENT = {
    'multinomial': {
        'model': home + '/sentiment/multinomial/multinomial-sentiment.pkl',
        'vector': home
        + '/sentiment/multinomial/multinomial-sentiment-tfidf.pkl',
        'version': 'v30',
    },
    'bert': {
        'base': {
            'model': home + '/sentiment/bert/base/bert-sentiment.pb',
            'vocab': home + '/bert/sp10m.cased.v4.vocab',
            'tokenizer': home + '/bert/sp10m.cased.v4.model',
            'version': 'v30',
        },
        'small': {
            'model': home + '/sentiment/bert/small/bert-sentiment.pb',
            'vocab': home + '/bert/sp10m.cased.v4.vocab',
            'tokenizer': home + '/bert/sp10m.cased.v4.model',
            'version': 'v30',
        },
    },
    'albert': {
        'base': {
            'model': home + '/sentiment/albert/base/albert-sentiment.pb',
            'vocab': home + '/albert/sp10m.cased.v6.vocab',
            'tokenizer': home + '/albert/sp10m.cased.v6.model',
            'version': 'v30',
        }
    },
    'xlnet': {
        'base': {
            'model': home + '/sentiment/xlnet/base/albert-sentiment.pb',
            'vocab': home + '/xlnet/sp10m.cased.v5.vocab',
            'tokenizer': home + '/xlnet/sp10m.cased.v5.model',
            'version': 'v30',
        }
    },
}

S3_PATH_SENTIMENT = {
    'multinomial': {
        'model': 'v30/sentiment/multinomial-sentiment.pkl',
        'vector': 'v30/sentiment/multinomial-sentiment-tfidf.pkl',
    },
    'bert': {
        'base': {
            'model': 'v30/sentiment/bert-base-sentiment.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v4.model',
        },
        'small': {
            'model': 'v30/sentiment/bert-small-sentiment.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v4.model',
        },
    },
    'albert': {
        'base': {
            'model': 'v30/sentiment/albert-base-sentiment.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v6.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v6.model',
        }
    },
    'xlnet': {
        'base': {
            'model': 'v30/sentiment/xlnet-base-sentiment.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v5.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v5.model',
        }
    },
}

PATH_SUBJECTIVE = {
    'multinomial': {
        'model': home + '/subjective/multinomial/multinomial-subjective.pkl',
        'vector': home
        + '/subjective/multinomial/multinomial-subjective-tfidf.pkl',
        'version': 'v30',
    },
    'bert': {
        'base': {
            'model': home + '/subjective/bert/base/bert-subjective.pb',
            'vocab': home + '/bert/sp10m.cased.v4.vocab',
            'tokenizer': home + '/bert/sp10m.cased.v4.model',
            'version': 'v30',
        },
        'small': {
            'model': home + '/subjective/bert/small/bert-subjective.pb',
            'vocab': home + '/bert/sp10m.cased.v4.vocab',
            'tokenizer': home + '/bert/sp10m.cased.v4.model',
            'version': 'v30',
        },
    },
    'albert': {
        'base': {
            'model': home + '/subjective/albert/base/albert-subjective.pb',
            'vocab': home + '/albert/sp10m.cased.v6.vocab',
            'tokenizer': home + '/albert/sp10m.cased.v6.model',
            'version': 'v30',
        }
    },
    'xlnet': {
        'base': {
            'model': home + '/subjective/xlnet/base/albert-subjective.pb',
            'vocab': home + '/xlnet/sp10m.cased.v5.vocab',
            'tokenizer': home + '/xlnet/sp10m.cased.v5.model',
            'version': 'v30',
        }
    },
}

S3_PATH_SUBJECTIVE = {
    'multinomial': {
        'model': 'v10/subjective/multinomial-subjective.pkl',
        'vector': 'v10/subjective/multinomial-subjective-tfidf.pkl',
    },
    'bert': {
        'base': {
            'model': 'v30/subjective/bert-base-subjective.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v4.model',
        },
        'small': {
            'model': 'v30/subjective/bert-small-subjective.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v4.model',
        },
    },
    'albert': {
        'base': {
            'model': 'v30/subjective/albert-base-subjective.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v6.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v6.model',
        }
    },
    'xlnet': {
        'base': {
            'model': 'v30/subjective/xlnet-base-subjective.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v5.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v5.model',
        }
    },
}

PATH_EMOTION = {
    'multinomial': {
        'model': home + '/emotion/multinomial/multinomial-emotion.pkl',
        'vector': home + '/emotion/multinomial/multinomial-emotion-tfidf.pkl',
        'version': 'v30',
    },
    'bert': {
        'base': {
            'model': home + '/emotion/bert/base/bert-emotion.pb',
            'vocab': home + '/bert/sp10m.cased.v4.vocab',
            'tokenizer': home + '/bert/sp10m.cased.v4.model',
            'version': 'v30',
        },
        'small': {
            'model': home + '/emotion/bert/small/bert-emotion.pb',
            'vocab': home + '/bert/sp10m.cased.v4.vocab',
            'tokenizer': home + '/bert/sp10m.cased.v4.model',
            'version': 'v30',
        },
    },
    'albert': {
        'base': {
            'model': home + '/emotion/albert/base/albert-emotion.pb',
            'vocab': home + '/albert/sp10m.cased.v6.vocab',
            'tokenizer': home + '/albert/sp10m.cased.v6.model',
            'version': 'v30',
        }
    },
    'xlnet': {
        'base': {
            'model': home + '/emotion/xlnet/base/albert-emotion.pb',
            'vocab': home + '/xlnet/sp10m.cased.v5.vocab',
            'tokenizer': home + '/xlnet/sp10m.cased.v5.model',
            'version': 'v30',
        }
    },
}

S3_PATH_EMOTION = {
    'multinomial': {
        'model': 'v30/emotion/multinomial-emotion.pkl',
        'vector': 'v30/emotion/multinomial-emotion-tfidf.pkl',
    },
    'bert': {
        'base': {
            'model': 'v30/emotion/bert-base-emotion.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v4.model',
        },
        'small': {
            'model': 'v30/emotion/bert-small-emotion.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v4.model',
        },
    },
    'albert': {
        'base': {
            'model': 'v30/emotion/albert-base-emotion.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v6.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v6.model',
        }
    },
    'xlnet': {
        'base': {
            'model': 'v30/emotion/xlnet-base-emotion.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v5.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v5.model',
        }
    },
}

PATH_DEPEND = {
    'bert': {
        'base': {
            'model': home + '/dependency/bert/base/bert-dependency.pb',
            'vocab': home + '/bert/sp10m.cased.v4.vocab',
            'tokenizer': home + '/bert/sp10m.cased.v4.model',
            'version': 'v30',
        }
    },
    'albert': {
        'base': {
            'model': home + '/dependency/albert/base/albert-dependency.pb',
            'vocab': home + '/albert/sp10m.cased.v6.vocab',
            'tokenizer': home + '/albert/sp10m.cased.v6.model',
            'version': 'v30',
        }
    },
    'xlnet': {
        'base': {
            'model': home + '/dependency/xlnet/base/xlnet-dependency.pb',
            'vocab': home + '/xlnet/sp10m.cased.v5.vocab',
            'tokenizer': home + '/xlnet/sp10m.cased.v5.model',
            'version': 'v30',
        }
    },
}

S3_PATH_DEPEND = {
    'bert': {
        'base': {
            'model': 'v30/dependency/bert-base-dependency.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v4.model',
        }
    },
    'albert': {
        'base': {
            'model': 'v30/dependency/albert-base-dependency.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v6.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v6.model',
        }
    },
    'xlnet': {
        'base': {
            'model': 'v30/dependency/xlnet-base-dependency.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v5.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v5.model',
        }
    },
}

PATH_RELEVANCY = {
    'bert': {
        'base': {
            'model': home + '/relevancy/bert/base/bert-relevancy.pb',
            'vocab': home + '/bert/sp10m.cased.v4.vocab',
            'tokenizer': home + '/bert/sp10m.cased.v4.model',
            'version': 'v30',
        }
    },
    'albert': {
        'base': {
            'model': home + '/relevancy/albert/base/albert-relevancy.pb',
            'vocab': home + '/albert/sp10m.cased.v6.vocab',
            'tokenizer': home + '/albert/sp10m.cased.v6.model',
            'version': 'v30',
        }
    },
    'xlnet': {
        'base': {
            'model': home + '/relevancy/xlnet/base/albert-relevancy.pb',
            'vocab': home + '/xlnet/sp10m.cased.v5.vocab',
            'tokenizer': home + '/xlnet/sp10m.cased.v5.model',
            'version': 'v30',
        }
    },
}
S3_PATH_RELEVANCY = {
    'bert': {
        'base': {
            'model': 'v30/relevancy/bert-base-relevancy.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v4.model',
        }
    },
    'albert': {
        'base': {
            'model': 'v30/relevancy/albert-base-relevancy.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v6.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v6.model',
        }
    },
    'xlnet': {
        'base': {
            'model': 'v30/relevancy/xlnet-base-relevancy.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v5.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v5.model',
        }
    },
}

PATH_SIMILARITY = {
    'bert': {
        'base': {
            'model': home + '/similarity/bert/base/bert-similarity.pb',
            'vocab': home + '/bert/sp10m.cased.v4.vocab',
            'tokenizer': home + '/bert/sp10m.cased.v4.model',
            'version': 'v30',
        }
    },
    'albert': {
        'base': {
            'model': home + '/similarity/albert/base/albert-similarity.pb',
            'vocab': home + '/albert/sp10m.cased.v6.vocab',
            'tokenizer': home + '/albert/sp10m.cased.v6.model',
            'version': 'v30',
        }
    },
    'xlnet': {
        'base': {
            'model': home + '/similarity/xlnet/base/albert-similarity.pb',
            'vocab': home + '/xlnet/sp10m.cased.v5.vocab',
            'tokenizer': home + '/xlnet/sp10m.cased.v5.model',
            'version': 'v30',
        }
    },
}

S3_PATH_SIMILARITY = {
    'bert': {
        'base': {
            'model': 'v30/similarity/bert-base-similarity.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v4.model',
        }
    },
    'albert': {
        'base': {
            'model': 'v30/similarity/albert-base-similarity.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v6.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v6.model',
        }
    },
    'xlnet': {
        'base': {
            'model': 'v30/similarity/xlnet-base-similarity.pb',
            'vocab': 'bert-bahasa/sp10m.cased.v5.vocab',
            'tokenizer': 'bert-bahasa/sp10m.cased.v5.model',
        }
    },
}

PATH_BERT = {
    'base': {
        'path': home + '/bert-model/base',
        'directory': home + '/bert-model/base/bert-base-v2/',
        'model': {
            'model': home + '/bert-model/base/bert-bahasa-base.tar.gz',
            'version': 'v31',
        },
    },
    'small': {
        'path': home + '/bert-model/small',
        'directory': home + '/bert-model/small/bert-small-v2/',
        'model': {
            'model': home + '/bert-model/small/bert-bahasa-small.tar.gz',
            'version': 'v31',
        },
    },
}

S3_PATH_BERT = {
    'base': {'model': 'bert-bahasa/bert-base-2-12-2019.tar.gz'},
    'small': {'model': 'bert-bahasa/bert-small-4-12-2019.tar.gz'},
}

PATH_XLNET = {
    'base': {
        'path': home + '/xlnet-model/base',
        'directory': home + '/xlnet-model/base/xlnet-base/',
        'model': {
            'model': home + '/xlnet-model/base/xlnet-base.tar.gz',
            'version': 'v30',
        },
    }
}

S3_PATH_XLNET = {'base': {'model': 'bert-bahasa/xlnet-base-30-9-2019.tar.gz'}}

PATH_ALBERT = {
    'base': {
        'path': home + '/albert-model/base',
        'directory': home + '/albert-model/base/albert-base/',
        'model': {
            'model': home + '/albert-model/base/albert-base.tar.gz',
            'version': 'v32',
        },
    },
    'large': {
        'path': home + '/albert-model/large',
        'directory': home + '/albert-model/large/albert-large/',
        'model': {
            'model': home + '/albert-model/base/albert-large.tar.gz',
            'version': 'v32',
        },
    },
}

S3_PATH_ALBERT = {
    'base': {'model': 'bert-bahasa/albert-base-15-12-2019.tar.gz'},
    'large': {'model': 'bert-bahasa/albert-large-28-12-2019.tar.gz'},
}

PATH_ALXLNET = {
    'base': {
        'path': home + '/alxlnet-model/base',
        'directory': home + '/alxlnet-model/base/alxlnet-base/',
        'model': {
            'model': home + '/alxlnet-model/base/alxlnet-base.tar.gz',
            'version': 'v31',
        },
    }
}

S3_PATH_ALXLNET = {
    'base': {'model': 'bert-bahasa/alxlnet-base-6-11-2019.tar.gz'}
}

PATH_WORDVECTOR = {
    'news': {
        'vocab': home + '/wordvector/news/wordvector.json',
        'model': home + '/wordvector/news/wordvector.npy',
        'version': 'v31',
    },
    'wikipedia': {
        'vocab': home + '/wordvector/wikipedia/wordvector.json',
        'model': home + '/wordvector/wikipedia/wordvector.npy',
        'version': 'v31',
    },
    'socialmedia': {
        'vocab': home + '/wordvector/socialmedia/wordvector.json',
        'model': home + '/wordvector/socialmedia/wordvector.npy',
        'version': 'v31',
    },
}

S3_PATH_WORDVECTOR = {
    'news': {
        'vocab': 'bert-bahasa/word2vec-news-ms-256.json',
        'model': 'bert-bahasa/word2vec-news-ms-256.npy',
    },
    'wikipedia': {
        'vocab': 'bert-bahasa/word2vec-wiki-ms-256.json',
        'model': 'bert-bahasa/word2vec-wiki-ms-256.npy',
    },
    'socialmedia': {
        'vocab': 'bert-bahasa/word2vec-ms-socialmedia-256.json',
        'model': 'bert-bahasa/word2vec-ms-socialmedia-256.npy',
    },
}
