from malaya import home

MALAY_TEXT = home + '/dictionary/malay-text.txt'
MALAY_TEXT_200K = home + '/dictionary-200k/malay-text.txt'

# sorted based on modules, started from augmentation until toxic

PATH_AUGMENTATION = {
    'synonym': {
        'model': home + '/synonym/synonym0.json',
        'model2': home + '/synonym/synonym1.json',
        'version': 'v35',
    }
}

S3_PATH_AUGMENTATION = {
    'synonym': {
        'model': 'https://raw.githubusercontent.com/huseinzol05/Malaya-Dataset/master/dictionary/synonym/synonym0.json',
        'model2': 'https://raw.githubusercontent.com/huseinzol05/Malaya-Dataset/master/dictionary/synonym/synonym1.json',
    }
}

PATH_CONSTITUENCY = {
    'bert': {
        'model': home + '/constituency/bert/base/model.pb',
        'quantized': home + '/constituency/bert/base/quantized/model.pb',
        'dictionary': home + '/constituency/bert/base/vocab.json',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v38',
    },
    'tiny-bert': {
        'model': home + '/constituency/bert/tiny/model.pb',
        'quantized': home + '/constituency/bert/tiny/quantized/model.pb',
        'dictionary': home + '/constituency/bert/tiny/vocab.json',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v38',
    },
    'albert': {
        'model': home + '/constituency/albert/base/model.pb',
        'quantized': home + '/constituency/albert/base/quantized/model.pb',
        'dictionary': home + '/constituency/albert/base/vocab.json',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'version': 'v38',
    },
    'tiny-albert': {
        'model': home + '/constituency/albert/tiny/model.pb',
        'quantized': home + '/constituency/albert/tiny/quantized/model.pb',
        'dictionary': home + '/constituency/albert/tiny/vocab.json',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'version': 'v38',
    },
    'xlnet': {
        'model': home + '/constituency/xlnet/base/model.pb',
        'quantized': home + '/constituency/xlnet/base/quantized/model.pb',
        'dictionary': home + '/constituency/xlnet/base/vocab.json',
        'vocab': home + '/bert/sp10m.cased.v9.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v9.model',
        'version': 'v38',
    },
}

S3_PATH_CONSTITUENCY = {
    'bert': {
        'model': 'v38/constituency/bert-base.pb',
        'quantized': 'v40/constituency/bert-base.pb.quantized',
        'dictionary': 'v38/constituency/vocab-bert-base.json',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
    },
    'tiny-bert': {
        'model': 'v38/constituency/tiny-bert.pb',
        'quantized': 'v40/constituency/tiny-bert.pb.quantized',
        'dictionary': 'v38/constituency/vocab-tiny-bert.json',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
    },
    'albert': {
        'model': 'v38/constituency/albert-base.pb',
        'quantized': 'v40/constituency/albert-base.pb.quantized',
        'dictionary': 'v38/constituency/vocab-albert-base.json',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
    },
    'tiny-albert': {
        'model': 'v38/constituency/albert-tiny.pb',
        'quantized': 'v40/constituency/albert-tiny.pb.quantized',
        'dictionary': 'v38/constituency/vocab-albert-tiny.json',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
    },
    'xlnet': {
        'model': 'v40/constituency/xlnet-base.pb',
        'quantized': 'v40/constituency/xlnet-base.pb.quantized',
        'dictionary': 'v40/constituency/vocab-xlnet-base.json',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
    },
}

PATH_DEPENDENCY = {
    'bert': {
        'model': home + '/dependency/bert/base/model.pb',
        'quantized': home + '/dependency/bert/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'tiny-bert': {
        'model': home + '/dependency/bert/tiny/model.pb',
        'quantized': home + '/dependency/bert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'albert': {
        'model': home + '/dependency/albert/base/model.pb',
        'quantized': home + '/dependency/albert/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'version': 'v34',
    },
    'tiny-albert': {
        'model': home + '/dependency/albert/tiny/model.pb',
        'quantized': home + '/dependency/albert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'version': 'v34',
    },
    'xlnet': {
        'model': home + '/dependency/xlnet/base/model.pb',
        'quantized': home + '/dependency/xlnet/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v9.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v9.model',
        'version': 'v34',
    },
    'alxlnet': {
        'model': home + '/dependency/alxlnet/base/model.pb',
        'quantized': home + '/dependency/alxlnet/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v9.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v9.model',
        'version': 'v34',
    },
}

S3_PATH_DEPENDENCY = {
    'bert': {
        'model': 'v34/dependency/bert-base-dependency.pb',
        'quantized': 'v40/dependency/bert-base-dependency.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
    },
    'tiny-bert': {
        'model': 'v34/dependency/tiny-bert-dependency.pb',
        'quantized': 'v40/dependency/tiny-bert-dependency.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
    },
    'albert': {
        'model': 'v34/dependency/albert-base-dependency.pb',
        'quantized': 'v40/dependency/albert-base-dependency.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
    },
    'tiny-albert': {
        'model': 'v34/dependency/albert-tiny-dependency.pb',
        'quantized': 'v40/dependency/albert-tiny-dependency.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
    },
    'xlnet': {
        'model': 'v34/dependency/xlnet-base-dependency.pb',
        'quantized': 'v40/dependency/xlnet-base-dependency.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
    },
    'alxlnet': {
        'model': 'v34/dependency/alxlnet-base-dependency.pb',
        'quantized': 'v34/dependency/alxlnet-base-dependency.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
    },
}

PATH_EMOTION = {
    'multinomial': {
        'model': home + '/emotion/multinomial/multinomial.pkl',
        'vector': home + '/emotion/multinomial/tfidf.pkl',
        'bpe': home + '/emotion/multinomial/bpe.model',
        'version': 'v34',
    },
    'bert': {
        'model': home + '/emotion/bert/base/model.pb',
        'quantized': home + '/emotion/bert/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'tiny-bert': {
        'model': home + '/emotion/bert/tiny/model.pb',
        'quantized': home + '/emotion/bert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'albert': {
        'model': home + '/emotion/albert/base/model.pb',
        'quantized': home + '/emotion/albert/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'version': 'v34',
    },
    'tiny-albert': {
        'model': home + '/emotion/albert/tiny/model.pb',
        'quantized': home + '/emotion/albert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'version': 'v34',
    },
    'xlnet': {
        'model': home + '/emotion/xlnet/base/model.pb',
        'quantized': home + '/emotion/xlnet/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v9.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v9.model',
        'version': 'v34',
    },
    'alxlnet': {
        'model': home + '/emotion/alxlnet/base/model.pb',
        'quantized': home + '/emotion/alxlnet/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v9.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v9.model',
        'version': 'v34',
    },
}

S3_PATH_EMOTION = {
    'multinomial': {
        'model': 'v34/emotion/multinomial.pkl',
        'vector': 'v34/emotion/tfidf.pkl',
        'bpe': 'v34/emotion/bpe.model',
    },
    'bert': {
        'model': 'v34/emotion/bert-base-emotion.pb',
        'quantized': 'v40/emotion/bert-base-emotion.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
    },
    'tiny-bert': {
        'model': 'v34/emotion/tiny-bert-emotion.pb',
        'quantized': 'v40/emotion/tiny-bert-emotion.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
    },
    'albert': {
        'model': 'v34/emotion/albert-base-emotion.pb',
        'quantized': 'v40/emotion/albert-base-emotion.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
    },
    'tiny-albert': {
        'model': 'v34/emotion/albert-tiny-emotion.pb',
        'quantized': 'v40/emotion/albert-tiny-emotion.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
    },
    'xlnet': {
        'model': 'v34/emotion/xlnet-base-emotion.pb',
        'quantized': 'v40/emotion/xlnet-base-emotion.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
    },
    'alxlnet': {
        'model': 'v34/emotion/alxlnet-base-emotion.pb',
        'quantized': 'v40/emotion/alxlnet-base-emotion.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
    },
}

PATH_ENTITIES = {
    'bert': {
        'model': home + '/entity/bert/base/model.pb',
        'quantized': home + '/entity/bert/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'setting': home + '/entity/dictionary-entities.json',
        'version': 'v34',
    },
    'tiny-bert': {
        'model': home + '/entity/bert/tiny/model.pb',
        'quantized': home + '/entity/bert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'setting': home + '/entity/dictionary-entities.json',
        'version': 'v34',
    },
    'albert': {
        'model': home + '/entity/albert/base/model.pb',
        'quantized': home + '/entity/albert/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'setting': home + '/entity/dictionary-entities.json',
        'version': 'v34',
    },
    'tiny-albert': {
        'model': home + '/entity/albert/tiny/model.pb',
        'quantized': home + '/entity/albert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'setting': home + '/entity/dictionary-entities.json',
        'version': 'v34',
    },
    'xlnet': {
        'model': home + '/entity/xlnet/base/model.pb',
        'quantized': home + '/entity/xlnet/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v9.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v9.model',
        'setting': home + '/entity/dictionary-entities.json',
        'version': 'v34',
    },
    'alxlnet': {
        'model': home + '/entity/alxlnet/base/model.pb',
        'quantized': home + '/entity/alxlnet/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v9.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v9.model',
        'setting': home + '/entity/dictionary-entities.json',
        'version': 'v34',
    },
}

S3_PATH_ENTITIES = {
    'bert': {
        'model': 'v34/entity/bert-base-entity.pb',
        'quantized': 'v40/entity/bert-base-entity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
        'setting': 'bert-bahasa/dictionary-entities.json',
    },
    'tiny-bert': {
        'model': 'v34/entity/tiny-bert-entity.pb',
        'quantized': 'v40/entity/tiny-bert-entity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
        'setting': 'bert-bahasa/dictionary-entities.json',
    },
    'albert': {
        'model': 'v34/entity/albert-base-entity.pb',
        'quantized': 'v40/entity/albert-base-entity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
        'setting': 'bert-bahasa/dictionary-entities.json',
    },
    'tiny-albert': {
        'model': 'v34/entity/albert-tiny-entity.pb',
        'quantized': 'v40/entity/albert-tiny-entity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
        'setting': 'bert-bahasa/dictionary-entities.json',
    },
    'xlnet': {
        'model': 'v34/entity/xlnet-base-entity.pb',
        'quantized': 'v40/entity/xlnet-base-entity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
        'setting': 'bert-bahasa/dictionary-entities.json',
    },
    'alxlnet': {
        'model': 'v34/entity/alxlnet-base-entity.pb',
        'quantized': 'v40/entity/alxlnet-base-entity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
        'setting': 'bert-bahasa/dictionary-entities.json',
    },
}

PATH_GENERATOR = {
    't5-compressed': {
        'base': {
            'path': home + '/generator/t5-compressed/base',
            'directory': home + '/generator/t5-compressed/base/model/',
            'model': {
                'model': home
                + '/generator/t5-compressed/base/generator-t5-base.tar.gz',
                'version': 'v35',
            },
        },
        'small': {
            'path': home + '/generator/t5-compressed/small',
            'directory': home + '/generator/t5-compressed/small/model/',
            'model': {
                'model': home
                + '/generator/t5-compressed/small/generator-t5-small.tar.gz',
                'version': 'v35',
            },
        },
    },
    't5': {
        'base': {
            'model': home + '/generator/t5/base/model.pb',
            'quantized': home + '/generator/t5/base/quantized/model.pb',
            'version': 'v38',
        },
        'small': {
            'model': home + '/generator/t5/small/model.pb',
            'quantized': home + '/generator/t5/small/quantized/model.pb',
            'version': 'v38',
        },
    },
}

S3_PATH_GENERATOR = {
    't5-compressed': {
        'base': {'model': 'v35/generator/sample-generator-t5-base.tar.gz'},
        'small': {'model': 'v35/generator/sample-generator-t5-small.tar.gz'},
    },
    't5': {
        'base': {
            'model': 'v38/generator/base.pb',
            'quantized': 'v40/generator/base.pb.quantized',
        },
        'small': {
            'model': 'v38/generator/small.pb',
            'quantized': 'v40/generator/small.pb.quantized',
        },
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
        'vocab': home + '/preprocessing/sentencepiece/sp10m.cased.v4.vocab',
        'model': home + '/preprocessing/sentencepiece/sp10m.cased.v4.model',
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

PATH_NSFW = {
    'lexicon': {'model': home + '/nsfw/lexicon.json', 'version': 'v39'},
    'multinomial': {
        'model': home + '/nsfw/multinomial/multinomial.pkl',
        'vector': home + '/nsfw/multinomial/tfidf.pkl',
        'bpe': home + '/nsfw/multinomial/bpe.model',
        'version': 'v39',
    },
}

S3_PATH_NSFW = {
    'lexicon': {'model': 'v39/nsfw/nsfw-corpus.json'},
    'multinomial': {
        'model': 'v39/nsfw/multinomial-nsfw.pkl',
        'vector': 'v39/nsfw/tfidf-nsfw.pkl',
        'bpe': 'v39/nsfw/nsfw.model',
    },
}

PATH_PARAPHRASE = {
    't5-compressed': {
        'base': {
            'path': home + '/paraphrase/t5-compressed/base',
            'directory': home + '/paraphrase/t5-compressed/base/model/',
            'model': {
                'model': home
                + '/paraphrase/t5-compressed/base/paraphrase-t5-base.tar.gz',
                'version': 'v36',
            },
        },
        'small': {
            'path': home + '/paraphrase/t5-compressed/small',
            'directory': home + '/paraphrase/t5-compressed/small/model/',
            'model': {
                'model': home
                + '/paraphrase/t5-compressed/small/paraphrase-t5-small.tar.gz',
                'version': 'v36',
            },
        },
    },
    't5': {
        'base': {
            'model': home + '/paraphrase/t5/base/model.pb',
            'quantized': home + '/paraphrase/t5/base/quantized/model.pb',
            'version': 'v38',
        },
        'small': {
            'model': home + '/paraphrase/t5/small/model.pb',
            'quantized': home + '/paraphrase/t5/small/quantized/model.pb',
            'version': 'v38',
        },
    },
    'transformer': {
        'base': {
            'model': home + '/paraphrase/transformer/base/model.pb',
            'quantized': home
            + '/paraphrase/transformer/base/quantized/model.pb',
            'vocab': home + '/paraphrase/sp10m.cased.t5.model',
            'version': 'v39',
        },
        'small': {
            'model': home + '/paraphrase/transformer/small/model.pb',
            'quantized': home
            + '/paraphrase/transformer/small/quantized/model.pb',
            'vocab': home + '/paraphrase/sp10m.cased.t5.model',
            'version': 'v39',
        },
    },
}
S3_PATH_PARAPHRASE = {
    't5-compressed': {
        'base': {'model': 'v36/paraphrase/paraphrase-t5-base.tar.gz'},
        'small': {'model': 'v36/paraphrase/paraphrase-t5-small.tar.gz'},
    },
    't5': {
        'base': {
            'model': 'v38/paraphrase/base.pb',
            'quantized': 'v40/paraphrase/base.pb.t5.quantized',
        },
        'small': {
            'model': 'v38/paraphrase/small.pb',
            'quantized': 'v40/paraphrase/small.pb.t5.quantized',
        },
    },
    'transformer': {
        'base': {
            'model': 'v39/paraphrase/base.pb',
            'quantized': 'v40/paraphrase/base.pb.quantized',
            'vocab': 'tokenizer/sp10m.cased.t5.model',
        },
        'small': {
            'model': 'v39/paraphrase/small.pb',
            'quantized': 'v40/paraphrase/small.pb.quantized',
            'vocab': 'tokenizer/sp10m.cased.t5.model',
        },
    },
}

PATH_POS = {
    'bert': {
        'model': home + '/pos/bert/base/model.pb',
        'quantized': home + '/pos/bert/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'setting': home + '/pos/dictionary-pos.json',
        'version': 'v34',
    },
    'tiny-bert': {
        'model': home + '/pos/bert/tiny/model.pb',
        'quantized': home + '/pos/bert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'setting': home + '/pos/dictionary-pos.json',
        'version': 'v34',
    },
    'albert': {
        'model': home + '/pos/albert/base/model.pb',
        'quantized': home + '/pos/albert/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'setting': home + '/pos/dictionary-pos.json',
        'version': 'v34',
    },
    'tiny-albert': {
        'model': home + '/pos/albert/tiny/model.pb',
        'quantized': home + '/pos/albert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'setting': home + '/pos/dictionary-pos.json',
        'version': 'v34',
    },
    'xlnet': {
        'model': home + '/pos/xlnet/base/model.pb',
        'quantized': home + '/pos/xlnet/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v9.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v9.model',
        'setting': home + '/pos/dictionary-pos.json',
        'version': 'v34',
    },
    'alxlnet': {
        'model': home + '/pos/alxlnet/base/model.pb',
        'quantized': home + '/pos/alxlnet/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v9.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v9.model',
        'setting': home + '/pos/dictionary-pos.json',
        'version': 'v34',
    },
}

S3_PATH_POS = {
    'bert': {
        'model': 'v34/pos/bert-base-pos.pb',
        'quantized': 'v40/pos/bert-base-pos.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
        'setting': 'bert-bahasa/dictionary-pos.json',
    },
    'tiny-bert': {
        'model': 'v34/pos/tiny-bert-pos.pb',
        'quantized': 'v40/pos/tiny-bert-pos.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
        'setting': 'bert-bahasa/dictionary-pos.json',
    },
    'albert': {
        'model': 'v34/pos/albert-base-pos.pb',
        'quantized': 'v40/pos/albert-base-pos.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
        'setting': 'bert-bahasa/dictionary-pos.json',
    },
    'tiny-albert': {
        'model': 'v34/pos/albert-tiny-pos.pb',
        'quantized': 'v40/pos/albert-tiny-pos.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
        'setting': 'bert-bahasa/dictionary-pos.json',
    },
    'xlnet': {
        'model': 'v34/pos/xlnet-base-pos.pb',
        'quantized': 'v40/pos/xlnet-base-pos.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
        'setting': 'bert-bahasa/dictionary-pos.json',
    },
    'alxlnet': {
        'model': 'v34/pos/alxlnet-base-pos.pb',
        'quantized': 'v40/pos/alxlnet-base-pos.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
        'setting': 'bert-bahasa/dictionary-pos.json',
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

PATH_RELEVANCY = {
    'bert': {
        'model': home + '/relevancy/bert/base/model.pb',
        'quantized': home + '/relevancy/bert/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v40',
    },
    'tiny-bert': {
        'model': home + '/relevancy/bert/tiny/model.pb',
        'quantized': home + '/relevancy/bert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v40',
    },
    'albert': {
        'model': home + '/relevancy/albert/base/model.pb',
        'quantized': home + '/relevancy/albert/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'version': 'v40',
    },
    'tiny-albert': {
        'model': home + '/relevancy/albert/tiny/model.pb',
        'quantized': home + '/relevancy/albert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'version': 'v40',
    },
    'xlnet': {
        'model': home + '/relevancy/xlnet/base/model.pb',
        'quantized': home + '/relevancy/xlnet/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v9.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v9.model',
        'version': 'v40',
    },
    'alxlnet': {
        'model': home + '/relevancy/alxlnet/base/model.pb',
        'quantized': home + '/relevancy/alxlnet/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v9.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v9.model',
        'version': 'v40',
    },
}
S3_PATH_RELEVANCY = {
    'bert': {
        'model': 'v40/relevancy/bert-base-relevancy.pb',
        'quantized': 'v40/relevancy/bert-base-relevancy.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
    },
    'tiny-bert': {
        'model': 'v40/relevancy/tiny-bert-relevancy.pb',
        'quantized': 'v40/relevancy/tiny-bert-relevancy.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
    },
    'albert': {
        'model': 'v40/relevancy/albert-base-relevancy.pb',
        'quantized': 'v40/relevancy/albert-base-relevancy.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
    },
    'tiny-albert': {
        'model': 'v40/relevancy/albert-tiny-relevancy.pb',
        'quantized': 'v40/relevancy/albert-tiny-relevancy.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
    },
    'xlnet': {
        'model': 'v40/relevancy/xlnet-base-relevancy.pb',
        'quantized': 'v40/relevancy/xlnet-base-relevancy.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
    },
    'alxlnet': {
        'model': 'v40/relevancy/alxlnet-base-relevancy.pb',
        'quantized': 'v40/relevancy/alxlnet-base-relevancy.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
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
        'quantized': home + '/sentiment/bert/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'tiny-bert': {
        'model': home + '/sentiment/bert/tiny/model.pb',
        'quantized': home + '/sentiment/bert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'albert': {
        'model': home + '/sentiment/albert/base/model.pb',
        'quantized': home + '/sentiment/albert/base/quantized/model.pb',
        'vocab': home + '/albert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/albert/sp10m.cased.v10.model',
        'version': 'v34',
    },
    'tiny-albert': {
        'model': home + '/sentiment/albert/tiny/model.pb',
        'quantized': home + '/sentiment/albert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'xlnet': {
        'model': home + '/sentiment/xlnet/base/model.pb',
        'quantized': home + '/sentiment/xlnet/base/quantized/model.pb',
        'vocab': home + '/xlnet/sp10m.cased.v9.vocab',
        'tokenizer': home + '/xlnet/sp10m.cased.v9.model',
        'version': 'v34',
    },
    'alxlnet': {
        'model': home + '/sentiment/alxlnet/base/model.pb',
        'quantized': home + '/sentiment/alxlnet/base/quantized/model.pb',
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
        'quantized': 'v40/sentiment/bert-base-sentiment.pb.quantized',
    },
    'tiny-bert': {
        'model': 'v34/sentiment/tiny-bert-sentiment.pb',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
        'quantized': 'v40/sentiment/tiny-bert-sentiment.pb.quantized',
    },
    'albert': {
        'model': 'v34/sentiment/albert-base-sentiment.pb',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
        'quantized': 'v40/sentiment/albert-base-sentiment.pb.quantized',
    },
    'tiny-albert': {
        'model': 'v34/sentiment/albert-tiny-sentiment.pb',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
        'quantized': 'v40/sentiment/albert-tiny-sentiment.pb.quantized',
    },
    'xlnet': {
        'model': 'v34/sentiment/xlnet-base-sentiment.pb',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
        'quantized': 'v40/sentiment/xlnet-base-sentiment.pb.quantized',
    },
    'alxlnet': {
        'model': 'v34/sentiment/alxlnet-base-sentiment.pb',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
        'quantized': 'v40/sentiment/alxlnet-base-sentiment.pb.quantized',
    },
}

PATH_SIMILARITY = {
    'bert': {
        'model': home + '/similarity/bert/base/model.pb',
        'quantized': home + '/similarity/bert/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v36',
    },
    'tiny-bert': {
        'model': home + '/similarity/bert/tiny/model.pb',
        'quantized': home + '/similarity/bert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v36',
    },
    'albert': {
        'model': home + '/similarity/albert/base/model.pb',
        'quantized': home + '/similarity/albert/base/quantized/model.pb',
        'vocab': home + '/albert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/albert/sp10m.cased.v10.model',
        'version': 'v36',
    },
    'tiny-albert': {
        'model': home + '/similarity/albert/tiny/model.pb',
        'quantized': home + '/similarity/albert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v36',
    },
    'xlnet': {
        'model': home + '/similarity/xlnet/base/model.pb',
        'quantized': home + '/similarity/xlnet/base/quantized/model.pb',
        'vocab': home + '/xlnet/sp10m.cased.v9.vocab',
        'tokenizer': home + '/xlnet/sp10m.cased.v9.model',
        'version': 'v36',
    },
    'alxlnet': {
        'model': home + '/similarity/alxlnet/base/model.pb',
        'quantized': home + '/similarity/alxlnet/base/quantized/model.pb',
        'vocab': home + '/xlnet/sp10m.cased.v9.vocab',
        'tokenizer': home + '/xlnet/sp10m.cased.v9.model',
        'version': 'v36',
    },
}

S3_PATH_SIMILARITY = {
    'bert': {
        'model': 'v36/similarity/bert-base-similarity.pb',
        'quantized': 'v40/similarity/bert-base-similarity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
    },
    'tiny-bert': {
        'model': 'v36/similarity/tiny-bert-similarity.pb',
        'quantized': 'v40/similarity/tiny-bert-similarity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
    },
    'albert': {
        'model': 'v36/similarity/albert-base-similarity.pb',
        'quantized': 'v40/similarity/albert-base-similarity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
    },
    'tiny-albert': {
        'model': 'v36/similarity/albert-tiny-similarity.pb',
        'quantized': 'v40/similarity/albert-tiny-similarity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
    },
    'xlnet': {
        'model': 'v36/similarity/xlnet-base-similarity.pb',
        'quantized': 'v40/similarity/xlnet-base-similarity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
    },
    'alxlnet': {
        'model': 'v36/similarity/alxlnet-base-similarity.pb',
        'quantized': 'v40/similarity/alxlnet-base-similarity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
    },
}

PATH_STEM = {
    'deep': {
        'model': home + '/stem/lstm/model.pb',
        'quantized': home + '/stem/lstm/quantized/model.pb',
        'bpe': home + '/stem/lstm/bpe.model',
        'version': 'v34',
    }
}

S3_PATH_STEM = {
    'deep': {
        'model': 'v34/stem/model.pb',
        'quantized': 'v40/stem/model.pb.quantized',
        'bpe': 'v34/stem/bpe.model',
    }
}

PATH_SUBJECTIVE = {
    'multinomial': {
        'model': home + '/subjective/multinomial/multinomial.pkl',
        'vector': home + '/subjective/multinomial/tfidf.pkl',
        'bpe': home + '/subjective/multinomial/bpe.model',
        'version': 'v34',
    },
    'bert': {
        'model': home + '/subjective/bert/base/model.pb',
        'quantized': home + '/subjective/bert/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'tiny-bert': {
        'model': home + '/subjective/bert/tiny/model.pb',
        'quantized': home + '/subjective/bert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'albert': {
        'model': home + '/subjective/albert/base/model.pb',
        'quantized': home + '/subjective/albert/base/quantized/model.pb',
        'vocab': home + '/albert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/albert/sp10m.cased.v10.model',
        'version': 'v34',
    },
    'tiny-albert': {
        'model': home + '/subjective/albert/tiny/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'xlnet': {
        'model': home + '/subjective/xlnet/base/model.pb',
        'quantized': home + '/subjective/xlnet/base/quantized/model.pb',
        'vocab': home + '/xlnet/sp10m.cased.v9.vocab',
        'tokenizer': home + '/xlnet/sp10m.cased.v9.model',
        'version': 'v34',
    },
    'alxlnet': {
        'model': home + '/subjective/alxlnet/base/model.pb',
        'quantized': home + '/subjective/alxlnet/base/quantized/model.pb',
        'vocab': home + '/xlnet/sp10m.cased.v9.vocab',
        'tokenizer': home + '/xlnet/sp10m.cased.v9.model',
        'version': 'v34',
    },
}

S3_PATH_SUBJECTIVE = {
    'multinomial': {
        'model': 'v34/subjective/multinomial.pkl',
        'vector': 'v34/subjective/tfidf.pkl',
        'bpe': 'v34/subjective/bpe.model',
    },
    'bert': {
        'model': 'v34/subjective/bert-base-subjective.pb',
        'quantized': 'v40/subjective/bert-base-subjective.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
    },
    'tiny-bert': {
        'model': 'v34/subjective/tiny-bert-subjective.pb',
        'quantized': 'v40/subjective/tiny-bert-subjective.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
    },
    'albert': {
        'model': 'v34/subjective/albert-base-subjective.pb',
        'quantized': 'v40/subjective/albert-base-subjective.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
    },
    'tiny-albert': {
        'model': 'v34/subjective/albert-tiny-subjective.pb',
        'quantized': 'v40/subjective/albert-tiny-subjective.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
    },
    'xlnet': {
        'model': 'v34/subjective/xlnet-base-subjective.pb',
        'quantized': 'v40/subjective/xlnet-base-subjective.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
    },
    'alxlnet': {
        'model': 'v34/subjective/alxlnet-base-subjective.pb',
        'quantized': 'v40/subjective/alxlnet-base-subjective.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
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
    't5-compressed': {
        'base': {
            'path': home + '/summarize/t5-compressed/base',
            'directory': home + '/summarize/t5-compressed/base/model/',
            'model': {
                'model': home
                + '/summarize/t5-compressed/base/summarize-t5-base.tar.gz',
                'version': 'v35',
            },
        },
        'small': {
            'path': home + '/summarize/t5-compressed/small',
            'directory': home + '/summarize/t5-compressed/base/model/',
            'model': {
                'model': home
                + '/summarize/t5-compressed/small/summarize-t5-base.tar.gz',
                'version': 'v35',
            },
        },
    },
    't5': {
        'base': {
            'model': home + '/summarize/t5/base/model.pb',
            'quantized': home + '/summarize/t5/base/quantized/model.pb',
            'version': 'v38',
        },
        'small': {
            'model': home + '/summarize/t5/small/model.pb',
            'quantized': home + '/summarize/t5/small/quantized/model.pb',
            'version': 'v38',
        },
    },
    'transformer': {
        'base': {
            'model': home + '/summarize/transformer/base/model.pb',
            'quantized': home
            + '/summarize/transformer/base/quantized/model.pb',
            'vocab': home + '/summarize/sp10m.cased.t5.model',
            'version': 'v39',
        },
        'small': {
            'model': home + '/summarize/transformer/small/model.pb',
            'quantized': home
            + '/summarize/transformer/small/quantized/model.pb',
            'vocab': home + '/summarize/sp10m.cased.t5.model',
            'version': 'v39',
        },
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
    't5-compressed': {
        'base': {'model': 'v35/summarize/argmax-summarize-t5-base.tar.gz'},
        'small': {'model': 'v35/summarize/argmax-summarize-t5-small.tar.gz'},
    },
    't5': {
        'base': {
            'model': 'v38/summarize/base.pb',
            'quantized': 'v40/summarize/base.pb.quantized',
        },
        'small': {
            'model': 'v38/summarize/small.pb',
            'quantized': 'v40/summarize/small.pb.quantized',
        },
    },
    'transformer': {
        'base': {
            'model': 'v39/summarization/base.pb',
            'quantized': 'v40/summarize/transformer-base.pb.quantized',
            'vocab': 'tokenizer/sp10m.cased.t5.model',
        },
        'small': {
            'model': 'v39/summarization/small.pb',
            'quantized': 'v40/summarize/transformer-small.pb.quantized',
            'vocab': 'tokenizer/sp10m.cased.t5.model',
        },
    },
}

PATH_TOXIC = {
    'multinomial': {
        'model': home + '/toxicity/multinomial/multinomial.pkl',
        'vector': home + '/toxicity/multinomial/tfidf.pkl',
        'bpe': home + '/toxicity/multinomial/bpe.model',
        'version': 'v34',
    },
    'bert': {
        'model': home + '/toxicity/bert/base/model.pb',
        'quantized': home + '/toxicity/bert/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'tiny-bert': {
        'model': home + '/toxicity/bert/tiny/model.pb',
        'quantized': home + '/toxicity/bert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v34',
    },
    'albert': {
        'model': home + '/toxicity/albert/base/model.pb',
        'quantized': home + '/toxicity/albert/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'version': 'v34',
    },
    'tiny-albert': {
        'model': home + '/toxicity/albert/tiny/model.pb',
        'quantized': home + '/toxicity/albert/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.v10.vocab',
        'tokenizer': home + '/bert/sp10m.cased.v10.model',
        'version': 'v34',
    },
    'xlnet': {
        'model': home + '/toxicity/xlnet/base/model.pb',
        'quantized': home + '/toxicity/xlnet/base/quantized/model.pb',
        'vocab': home + '/xlnet/sp10m.cased.v9.vocab',
        'tokenizer': home + '/xlnet/sp10m.cased.v9.model',
        'version': 'v34',
    },
    'alxlnet': {
        'model': home + '/toxicity/alxlnet/base/model.pb',
        'quantized': home + '/toxicity/alxlnet/base/quantized/model.pb',
        'vocab': home + '/alxlnet/sp10m.cased.v9.vocab',
        'tokenizer': home + '/alxlnet/sp10m.cased.v9.model',
        'version': 'v34',
    },
}

S3_PATH_TOXIC = {
    'multinomial': {
        'model': 'v34/toxicity/multinomial.pkl',
        'vector': 'v34/toxicity/tfidf.pkl',
        'bpe': 'v34/toxicity/bpe.model',
    },
    'bert': {
        'model': 'v34/toxicity/bert-base-toxicity.pb',
        'quantized': 'v40/toxicity/bert-base-toxicity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
    },
    'tiny-bert': {
        'model': 'v34/toxicity/tiny-bert-toxicity.pb',
        'quantized': 'v40/toxicity/tiny-bert-toxicity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
    },
    'albert': {
        'model': 'v34/toxicity/albert-base-toxicity.pb',
        'quantized': 'v40/toxicity/albert-base-toxicity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
    },
    'tiny-albert': {
        'model': 'v34/toxicity/albert-tiny-toxicity.pb',
        'quantized': 'v40/toxicity/albert-tiny-toxicity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v10.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v10.model',
    },
    'xlnet': {
        'model': 'v34/toxicity/xlnet-base-toxicity.pb',
        'quantized': 'v40/toxicity/xlnet-base-toxicity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
    },
    'alxlnet': {
        'model': 'v34/toxicity/alxlnet-base-toxicity.pb',
        'quantized': 'v40/toxicity/alxlnet-base-toxicity.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.v9.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.v9.model',
    },
}

PATH_TRANSLATION = {
    'ms-en': {
        'base': {
            'model': home + '/translation/ms-en/base/model.pb',
            'quantized': home + '/translation/ms-en/base/quantized/model.pb',
            'vocab': home + '/translation/ms-en/base/vocab.subwords',
            'version': 'v37',
        },
        'large': {
            'model': home + '/translation/ms-en/large/model.pb',
            'quantized': home + '/translation/ms-en/large/quantized/model.pb',
            'vocab': home + '/translation/ms-en/large/vocab.subwords',
            'version': 'v37',
        },
        'small': {
            'model': home + '/translation/ms-en/small/model.pb',
            'quantized': home + '/translation/ms-en/small/quantized/model.pb',
            'vocab': home + '/translation/ms-en/small/vocab.subwords',
            'version': 'v37',
        },
    },
    'en-ms': {
        'base': {
            'model': home + '/translation/en-ms/base/model.pb',
            'quantized': home + '/translation/en-ms/base/quantized/model.pb',
            'vocab': home + '/translation/en-ms/base/vocab.subwords',
            'version': 'v38',
        },
        'large': {
            'model': home + '/translation/en-ms/large/model.pb',
            'quantized': home + '/translation/en-ms/large/quantized/model.pb',
            'vocab': home + '/translation/en-ms/large/vocab.subwords',
            'version': 'v38',
        },
        'small': {
            'model': home + '/translation/en-ms/small/model.pb',
            'quantized': home + '/translation/en-ms/small/quantized/model.pb',
            'vocab': home + '/translation/en-ms/small/vocab.subwords',
            'version': 'v38',
        },
    },
}
S3_PATH_TRANSLATION = {
    'ms-en': {
        'base': {
            'model': 'v37/translation/ms-en/base-translation.pb',
            'quantized': 'v40/translation/ms-en/base-translation.pb.quantized',
            'vocab': 'v37/translation/ms-en/vocab.subwords',
        },
        'large': {
            'model': 'v37/translation/ms-en/large-translation.pb',
            'quantized': 'v40/translation/ms-en/large-translation.pb.quantized',
            'vocab': 'v37/translation/ms-en/vocab.subwords',
        },
        'small': {
            'model': 'v37/translation/ms-en/small-translation.pb',
            'quantized': 'v40/translation/ms-en/small-translation.pb.quantized',
            'vocab': 'v37/translation/ms-en/vocab.subwords',
        },
    },
    'en-ms': {
        'base': {
            'model': 'v38/translation/en-ms/base-translation.pb',
            'quantized': 'v40/translation/en-ms/base-translation.pb.quantized',
            'vocab': 'v38/translation/en-ms/vocab.subwords',
        },
        'large': {
            'model': 'v38/translation/en-ms/large-translation.pb',
            'quantized': 'v40/translation/en-ms/large-translation.pb.quantized',
            'vocab': 'v38/translation/en-ms/vocab.subwords',
        },
        'small': {
            'model': 'v38/translation/en-ms/small-translation.pb',
            'quantized': 'v38/translation/en-ms/small-translation.pb.quantized',
            'vocab': 'v38/translation/en-ms/vocab.subwords',
        },
    },
}

PATH_TRUE_CASE = {
    'base': {
        'model': home + '/true-case/base.pb',
        'quantized': home + '/true-case/quantized/base.pb',
        'vocab': home + '/true-case/vocab.yttm',
        'version': 'v39',
    },
    'small': {
        'model': home + '/true-case/small.pb',
        'quantized': home + '/true-case/quantized/small.pb',
        'vocab': home + '/true-case/vocab.yttm',
        'version': 'v39',
    },
}
S3_PATH_TRUE_CASE = {
    'base': {
        'model': 'v39/true-case/base.pb',
        'quantized': 'v40/true-case/base.pb.quantized',
        'vocab': 'tokenizer/truecase.yttm',
    },
    'small': {
        'model': 'v39/true-case/small.pb',
        'quantized': 'v40/true-case/small.pb.quantized',
        'vocab': 'tokenizer/truecase.yttm',
    },
}

PATH_ELECTRA = {
    'electra': {
        'path': home + '/electra-model/base',
        'directory': home + '/electra-model/base/electra-base/',
        'model': {
            'model': home + '/electra-model/base/electra-bahasa-base.tar.gz',
            'version': 'v34',
        },
    },
    'small-electra': {
        'path': home + '/electra-model/small',
        'directory': home + '/electra-model/small/electra-small/',
        'model': {
            'model': home + '/electra-model/small/electra-bahasa-small.tar.gz',
            'version': 'v34',
        },
    },
}

S3_PATH_ELECTRA = {
    'electra': {'model': 'v34/pretrained-model/electra-base.tar.gz'},
    'small-electra': {'model': 'v34/pretrained-model/electra-small.tar.gz'},
}

PATH_BERT = {
    'bert': {
        'path': home + '/bert-model/base',
        'directory': home + '/bert-model/base/bert-base-v3/',
        'model': {
            'model': home + '/bert-model/base/bert-bahasa-base.tar.gz',
            'version': 'v34',
        },
    },
    'tiny-bert': {
        'path': home + '/bert-model/tiny',
        'directory': home + '/bert-model/tiny/tiny-bert-v1/',
        'model': {
            'model': home + '/bert-model/tiny/tiny-bert-bahasa.tar.gz',
            'version': 'v34',
        },
    },
}

S3_PATH_BERT = {
    'bert': {'model': 'v34/pretrained-model/bert-base.tar.gz'},
    'tiny-bert': {'model': 'v34/pretrained-model/tiny-bert.tar.gz'},
}

PATH_ALBERT = {
    'albert': {
        'path': home + '/albert-model/base',
        'directory': home + '/albert-model/base/albert-base/',
        'model': {
            'model': home + '/albert-model/base/albert-bahasa-base.tar.gz',
            'version': 'v34',
        },
    },
    'tiny-albert': {
        'path': home + '/albert-model/tiny',
        'directory': home + '/albert-model/tiny/albert-tiny/',
        'model': {
            'model': home + '/albert-model/tiny/albert-bahasa-tiny.tar.gz',
            'version': 'v34',
        },
    },
}

S3_PATH_ALBERT = {
    'albert': {'model': 'v34/pretrained-model/albert-base.tar.gz'},
    'tiny-albert': {'model': 'v34/pretrained-model/albert-tiny.tar.gz'},
}

PATH_XLNET = {
    'xlnet': {
        'path': home + '/xlnet-model/base',
        'directory': home + '/xlnet-model/base/xlnet-base/',
        'model': {
            'model': home + '/xlnet-model/base/xlnet-base.tar.gz',
            'version': 'v34',
        },
    }
}

S3_PATH_XLNET = {'xlnet': {'model': 'v34/pretrained-model/xlnet-base.tar.gz'}}

PATH_ALXLNET = {
    'alxlnet': {
        'path': home + '/alxlnet-model/base',
        'directory': home + '/alxlnet-model/base/alxlnet-base/',
        'model': {
            'model': home + '/alxlnet-model/base/alxlnet-base.tar.gz',
            'version': 'v34',
        },
    }
}

S3_PATH_ALXLNET = {
    'alxlnet': {'model': 'v34/pretrained-model/alxlnet-base.tar.gz'}
}

PATH_GPT2 = {
    '117M': {
        'path': home + '/gpt2/117M/',
        'directory': home + '/gpt2/117M/gpt2-bahasa-117M/',
        'model': {
            'model': home + '/gpt2/117M/gpt2-117M.tar.gz',
            'version': 'v34',
        },
    },
    '345M': {
        'path': home + '/gpt2/345M/',
        'directory': home + '/gpt2/345M/gpt2-bahasa-345M/',
        'model': {
            'model': home + '/gpt2/345M/gpt2-345M.tar.gz',
            'version': 'v34',
        },
    },
}

S3_PATH_GPT2 = {
    '117M': {'model': 'v34/pretrained-model/gpt2-bahasa-117M.tar.gz'},
    '345M': {'model': 'v34/pretrained-model/gpt2-bahasa-345M.tar.gz'},
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
