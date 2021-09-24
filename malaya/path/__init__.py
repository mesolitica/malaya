from malaya_boilerplate.utils import _get_home
from malaya import package

home, _ = _get_home(package=package)

MALAY_TEXT = home + '/dictionary/malay-text.txt'
MALAY_TEXT_200K = home + '/dictionary/200k-malay-text.txt'

BERT_BPE_VOCAB = 'bpe/sp10m.cased.bert.vocab'
BERT_BPE_MODEL = 'bpe/sp10m.cased.bert.model'
BERT_WORDPIECE_VOCAB = 'bpe/BERT.wordpiece.vocab'
BERT_WORDPIECE_MODEL = 'bpe/BERT.wordpiece'
ALBERT_BPE_VOCAB = 'bpe/sp10m.cased.v10.vocab'
ALBERT_BPE_MODEL = 'bpe/sp10m.cased.v10.model'
XLNET_BPE_VOCAB = 'bpe/sp10m.cased.v9.vocab'
XLNET_BPE_MODEL = 'bpe/sp10m.cased.v9.model'
T2T_BPE_VOCAB = 'bpe/sp10m.cased.t5.vocab'
T2T_BPE_MODEL = 'bpe/sp10m.cased.t5.model'
MS_EN_BPE_VOCAB = 'bpe/sp10m.cased.ms-en.vocab'
MS_EN_BPE_MODEL = 'bpe/sp10m.cased.ms-en.model'
TRANSLATION_BPE_VOCAB = 'bpe/sp10m.cased.translation.vocab'
TRANSLATION_BPE_MODEL = 'bpe/sp10m.cased.translation.model'
TRANSLATION_EN_MS_VOCAB = 'bpe/en-ms.subwords'
TRANSLATION_MS_EN_VOCAB = 'bpe/ms-en.subwords'
TRUE_CASE_VOCAB = 'bpe/true-case.yttm'
SEGMENTATION_VOCAB = 'bpe/segmentation.yttm'
PEGASUS_BPE_MODEL = 'bpe/pegasus.wordpiece'

ENTITY_SETTING = 'setting/entities.json'
ENTITY_ONTONOTES5_SETTING = 'setting/entities-ontonotes5.json'
POS_SETTING = 'setting/pos.json'

LANGUAGE_DETECTION_BOW = 'bpe/bow-language-detection.pkl'
LANGUAGE_DETECTION_VOCAB = 'bpe/language-detection.yttm'

STEMMER_VOCAB = 'bpe/stemmer.yttm'

CONSTITUENCY_SETTING = 'setting/constituency.json'

GPT2_ENCODER = 'bpe/gpt2-encoder.json'
GPT2_VOCAB = 'bpe/gpt2-vocab.bpe'

MODEL_VOCAB = {
    'albert': ALBERT_BPE_VOCAB,
    'bert': BERT_BPE_VOCAB,
    'tiny-albert': ALBERT_BPE_VOCAB,
    'tiny-bert': BERT_BPE_VOCAB,
    'xlnet': XLNET_BPE_VOCAB,
    'alxlnet': XLNET_BPE_VOCAB,
    'bigbird': BERT_BPE_VOCAB,
    'tiny-bigbird': BERT_BPE_VOCAB,
    'fnet': BERT_WORDPIECE_VOCAB,
    'fnet-large': BERT_WORDPIECE_VOCAB,
}

MODEL_BPE = {
    'albert': ALBERT_BPE_MODEL,
    'bert': BERT_BPE_MODEL,
    'tiny-albert': ALBERT_BPE_MODEL,
    'tiny-bert': BERT_BPE_MODEL,
    'xlnet': XLNET_BPE_MODEL,
    'alxlnet': XLNET_BPE_MODEL,
    'bigbird': BERT_BPE_MODEL,
    'tiny-bigbird': BERT_BPE_MODEL,
    'fnet': BERT_WORDPIECE_MODEL,
    'fnet-large': BERT_WORDPIECE_MODEL,
}

LM_VOCAB = {
    'translation-en-ms': TRANSLATION_EN_MS_VOCAB,
    'translation-ms-en': TRANSLATION_MS_EN_VOCAB,
    'true-case': TRUE_CASE_VOCAB,
    'segmentation': SEGMENTATION_VOCAB,
    'knowledge-graph-generator': MS_EN_BPE_MODEL,
}

TAGGING_SETTING = {
    'entity': ENTITY_SETTING,
    'pos': POS_SETTING,
    'entity-ontonotes5': ENTITY_ONTONOTES5_SETTING,
}

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
    }}

PATH_EMOTION = {
    'multinomial': {
        'model': home + '/emotion/multinomial/multinomial.pkl',
        'vector': home + '/emotion/multinomial/tfidf.pkl',
        'bpe': home + '/emotion/multinomial/bpe.model',
        'version': 'v34',
    }
}

S3_PATH_EMOTION = {
    'multinomial': {
        'model': 'v34/emotion/multinomial.pkl',
        'vector': 'v34/emotion/tfidf.pkl',
        'bpe': 'v34/emotion/bpe.model',
    }
}

PATH_LANG_DETECTION = {
    'fasttext-original': {
        'model': home + '/language-detection/fasttext-original/fasttext.bin',
        'version': 'v34',
    },
    'fasttext-quantized': {
        'model': home + '/language-detection/fasttext-quantized/fasttext.tfz',
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
    'jamspell': {
        'wiki+news': {
            'model': home + '/preprocessing/jamspell/wiki-news/model.bin',
            'version': 'v46',
        },
        'news': {
            'model': home + '/preprocessing/jamspell/news/model.bin',
            'version': 'v46',
        },
        'wiki': {
            'model': home + '/preprocessing/jamspell/wiki/model.bin',
            'version': 'v46',
        }
    },
    'spylls': {
        'libreoffice-pejam': {
            'model': home + '/preprocessing/spylls/libreoffice-pejam/model.oxt',
            'version': 'v46'
        }
    }
}

S3_PATH_NGRAM = {
    1: {'model': 'v27/preprocessing/bm_1grams.json'},
    2: {'model': 'v23/preprocessing/bm_2grams.json'},
    'symspell': {'model': 'v27/preprocessing/bm_1grams.txt'},
    'sentencepiece': {
        'vocab': 'bert-bahasa/sp10m.cased.v4.vocab',
        'model': 'bert-bahasa/sp10m.cased.v4.model',
    },
    'jamspell': {
        'wiki+news': {
            'model': 'v46/preprocessing/wiki-news.bin',
        },
        'news': {
            'model': 'v46/preprocessing/news.bin',
        },
        'wiki': {
            'model': 'v46/preprocessing/wiki.bin',
        }
    },
    'spylls': {
        'libreoffice-pejam': {
            'model': 'https://extensions.libreoffice.org/assets/downloads/741/pEJAm21z.oxt',
        }
    }
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

PATH_SENTIMENT = {
    'multinomial': {
        'model': home + '/sentiment/multinomial/multinomial.pkl',
        'vector': home + '/sentiment/multinomial/tfidf.pkl',
        'bpe': home + '/sentiment/multinomial/bpe.model',
        'version': 'v34',
    }
}

S3_PATH_SENTIMENT = {
    'multinomial': {
        'model': 'v34/sentiment/multinomial.pkl',
        'vector': 'v34/sentiment/tfidf.pkl',
        'bpe': 'v34/sentiment/bpe.model',
    }
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
    }
}

S3_PATH_SUBJECTIVE = {
    'multinomial': {
        'model': 'v34/subjective/multinomial.pkl',
        'vector': 'v34/subjective/tfidf.pkl',
        'bpe': 'v34/subjective/bpe.model',
    }
}

PATH_TOXIC = {
    'multinomial': {
        'model': home + '/toxicity/multinomial/multinomial.pkl',
        'vector': home + '/toxicity/multinomial/tfidf.pkl',
        'bpe': home + '/toxicity/multinomial/bpe.model',
        'version': 'v34',
    }
}

S3_PATH_TOXIC = {
    'multinomial': {
        'model': 'v34/toxicity/multinomial.pkl',
        'vector': 'v34/toxicity/tfidf.pkl',
        'bpe': 'v34/toxicity/bpe.model',
    }
}

PATH_TRUE_CASE = {
    'sacremoses': {
        'model': home + '/true-case/sacremoses/model.truecasemodel',
        'version': 'v43',
    }
}

S3_PATH_TRUE_CASE = {
    'sacremoses': {'model': 'v43/true-case/sacremoses/model.truecasemodel'}
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
