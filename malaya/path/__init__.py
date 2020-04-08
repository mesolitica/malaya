from malaya import home

MALAY_TEXT = home + '/dictionary/malay-text.txt'
MALAY_TEXT_200K = home + '/dictionary-200k/malay-text.txt'

# sorted based on modules, started from cluster until toxic

# dependency.py
PATH_DEPENDENCY = {
    'bert': {
        'model': home + '/dependency/bert/base/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'tiny-bert': {
        'model': home + '/dependency/bert/tiny/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'albert': {
        'model': home + '/dependency/albert/base/model.pb',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'version': 'v34',
    },
    'tiny-albert': {
        'model': home + '/dependency/albert/tiny/model.pb',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'version': 'v34',
    },
    'xlnet': {
        'model': home + '/dependency/xlnet/base/model.pb',
        'vocab': home + '/bert/sp10m.cased.v9.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v9.model',
        'version': 'v34',
    },
    'alxlnet': {
        'model': home + '/dependency/alxlnet/base/model.pb',
        'vocab': home + '/bert/sp10m.cased.v9.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v9.model',
        'version': 'v34',
    },
}

S3_PATH_DEPENDENCY = {
    'bert': {
        'model': 'v34/dependency/bert-base-dependency.pb',
        'vocab': 'bert-bahasa/sp10m.cased.bert.vocab',
        'tokenizer': 'bert-bahasa/sp10m.cased.bert.model',
    },
    'tiny-bert': {
        'model': 'v34/dependency/bert-tiny-dependency.pb',
        'vocab': 'bert-bahasa/sp10m.cased.bert.vocab',
        'tokenizer': 'bert-bahasa/sp10m.cased.bert.model',
    },
    'albert': {
        'model': 'v34/dependency/albert-base-dependency.pb',
        'vocab': 'bert-bahasa/sp10m.cased.v10.vocab',
        'tokenizer': 'bert-bahasa/sp10m.cased.v10.model',
    },
    'tiny-albert': {
        'model': 'v34/dependency/albert-tiny-dependency.pb',
        'vocab': 'bert-bahasa/sp10m.cased.v10.vocab',
        'tokenizer': 'bert-bahasa/sp10m.cased.v10.model',
    },
    'xlnet': {
        'model': 'v34/dependency/xlnet-base-dependency.pb',
        'vocab': 'bert-bahasa/sp10m.cased.v9.vocab',
        'tokenizer': 'bert-bahasa/sp10m.cased.v9.model',
    },
    'alxlnet': {
        'model': 'v34/dependency/alxlnet-base-dependency.pb',
        'vocab': 'bert-bahasa/sp10m.cased.v9.vocab',
        'tokenizer': 'bert-bahasa/sp10m.cased.v9.model',
    },
}


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
        'model': home + '/toxicity/multinomial/model.pkl',
        'vector': home + '/toxicity/multinomial/tfidf.pkl',
        'version': 'v34',
    },
    'bert': {
        'model': home + '/toxicity/bert/base/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'tiny-bert': {
        'model': home + '/toxicity/bert/tiny/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'albert': {
        'model': home + '/toxicity/albert/base/model.pb',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'version': 'v34',
    },
    'tiny-albert': {
        'model': home + '/toxicity/albert/tiny/model.pb',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'version': 'v34',
    },
    'xlnet': {
        'model': home + '/toxicity/xlnet/base/model.pb',
        'vocab': home + '/xlnet/sp10m.cased.v9.vocab',
        'tokenizer': home + '/xlnet/sp10m.cased.v9.model',
        'version': 'v34',
    },
    'alxlnet': {
        'model': home + '/toxicity/alxlnet/base/model.pb',
        'vocab': home + '/alxlnet/sp10m.cased.v9.vocab',
        'tokenizer': home + '/alxlnet/sp10m.cased.v9.model',
        'version': 'v34',
    },
}

S3_PATH_TOXIC = {
    'multinomial': {
        'model': 'v34/toxicity/multinomial-toxicity.pkl',
        'vector': 'v34/toxicity/multinomial-toxicity-tfidf.pkl',
    },
    'bert': {
        'model': 'v34/toxicity/bert-base-toxicity.pb',
        'vocab': 'bert-bahasa/sp10m.cased.bert.vocab',
        'tokenizer': 'bert-bahasa/sp10m.cased.bert.model',
    },
    'tiny-bert': {
        'model': 'v34/toxicity/tiny-bert-base-toxicity.pb',
        'vocab': 'bert-bahasa/sp10m.cased.bert.vocab',
        'tokenizer': 'bert-bahasa/sp10m.cased.bert.model',
    },
    'albert': {
        'model': 'v30/toxicity/albert-base-toxicity.pb',
        'vocab': 'bert-bahasa/sp10m.cased.v10.vocab',
        'tokenizer': 'bert-bahasa/sp10m.cased.v10.model',
    },
    'tiny-albert': {
        'model': 'v30/toxicity/albert-base-toxicity.pb',
        'vocab': 'bert-bahasa/sp10m.cased.v10.vocab',
        'tokenizer': 'bert-bahasa/sp10m.cased.v10.model',
    },
    'xlnet': {
        'model': 'v30/toxicity/xlnet-base-toxicity.pb',
        'vocab': 'bert-bahasa/sp10m.cased.v9.vocab',
        'tokenizer': 'bert-bahasa/sp10m.cased.v9.model',
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
        'version': 'v34',
    },
    'fasttext-quantized': {
        'model': home + '/language-detection/fasttext-quantized/fasstext.tfz',
        'version': 'v34',
    },
    'deep': {
        'model': home
        + '/language-detection/deep/model.ckpt.data-00000-of-00001',
        'index': home + '/language-detection/deep/model.ckpt.index',
        'meta': home + '/language-detection/deep/model.ckpt.meta',
        'vector': home
        + '/language-detection/deep/vectorizer-language-detection.pkl',
        'bpe': home + '/language-detection/deep/bpe.model',
        'version': 'v34',
    },
}

S3_PATH_LANG_DETECTION = {
    'fasttext-original': {
        'model': 'v34/language-detection/fasttext-malaya.bin'
    },
    'fasttext-quantized': {
        'model': 'v34/language-detection/fasttext-malaya.ftz'
    },
    'deep': {
        'model': 'v34/language-detection/model.ckpt.data-00000-of-00001',
        'index': 'v34/language-detection/model.ckpt.index',
        'meta': 'v34/language-detection/model.ckpt.meta',
        'vector': 'v34/language-detection/bow-language-detection.pkl',
        'bpe': 'v34/language-detection/language-detection.model',
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
        'model': home + '/sentiment/multinomial/multinomial.pkl',
        'vector': home + '/sentiment/multinomial/tfidf.pkl',
        'bpe': home + '/sentiment/multinomial/bpe.model',
        'version': 'v34',
    },
    'bert': {
        'model': home + '/sentiment/bert/base/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'tiny-bert': {
        'model': home + '/sentiment/bert/tiny/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'albert': {
        'model': home + '/sentiment/albert/base/model.pb',
        'vocab': home + '/albert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/albert/sp10m.cased.v10.model',
        'version': 'v34',
    },
    'tiny-albert': {
        'model': home + '/sentiment/albert/tiny/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'xlnet': {
        'model': home + '/sentiment/xlnet/base/model.pb',
        'vocab': home + '/xlnet/sp10m.cased.v9.vocab',
        'tokenizer': home + '/xlnet/sp10m.cased.v9.model',
        'version': 'v34',
    },
    'alxlnet': {
        'model': home + '/sentiment/alxlnet/base/model.pb',
        'vocab': home + '/xlnet/sp10m.cased.v9.vocab',
        'tokenizer': home + '/xlnet/sp10m.cased.v9.model',
        'version': 'v34',
    },
}

S3_PATH_SENTIMENT = {
    'multinomial': {
        'model': 'v34/sentiment/multinomial.pkl',
        'vector': 'v34/sentiment/tfidf.pkl',
        'bpe': 'v34/sentiment/bpe.model',
    },
    'bert': {
        'model': 'v34/sentiment/bert-base-sentiment.pb',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
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
        'model': home + '/sentiment/emotion/multinomial.pkl',
        'vector': home + '/sentiment/emotion/tfidf.pkl',
        'bpe': home + '/sentiment/emotion/bpe.model',
        'version': 'v34',
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
        'model': 'v34/emotion/multinomial.pkl',
        'vector': 'v34/emotion/tfidf.pkl',
        'bpe': 'v34/emotion/bpe.model',
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
        'directory': home + '/bert-model/base/bert-base-v3/',
        'model': {
            'model': home + '/bert-model/base/bert-bahasa-base.tar.gz',
            'version': 'v34',
        },
    }
}

S3_PATH_BERT = {'base': {'model': 'bert-bahasa/bert-base-23-03-2020.tar.gz'}}

PATH_XLNET = {
    'base': {
        'path': home + '/xlnet-model/base',
        'directory': home + '/xlnet-model/base/xlnet-base/',
        'model': {
            'model': home + '/xlnet-model/base/xlnet-base.tar.gz',
            'version': 'v34',
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
    'combine': {
        'vocab': home + '/wordvector/combine/wordvector.json',
        'model': home + '/wordvector/combine/wordvector.npy',
        'version': 'v34',
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
    'combine': {
        'vocab': 'bert-bahasa/word2vec-combined-256.json',
        'model': 'bert-bahasa/word2vec-combined-256.npy',
    },
}
