from malaya import home

MALAY_TEXT = home + '/dictionary/malay-text.txt'
MALAY_TEXT_200K = home + '/dictionary/200k-malay-text.txt'

BERT_BPE_VOCAB = 'bpe/sp10m.cased.bert.vocab'
BERT_BPE_MODEL = 'bpe/sp10m.cased.bert.model'
ALBERT_BPE_VOCAB = 'bpe/sp10m.cased.v10.vocab'
ALBERT_BPE_MODEL = 'bpe/sp10m.cased.v10.model'
XLNET_BPE_VOCAB = 'bpe/sp10m.cased.v9.vocab'
XLNET_BPE_MODEL = 'bpe/sp10m.cased.v9.model'
T2T_BPE_VOCAB = 'bpe/sp10m.cased.t5.vocab'
T2T_BPE_MODEL = 'bpe/sp10m.cased.t5.model'
TRANSLATION_BPE_VOCAB = 'bpe/sp10m.cased.translation.vocab'
TRANSLATION_BPE_MODEL = 'bpe/sp10m.cased.translation.model'
TRANSLATION_EN_MS_VOCAB = 'bpe/en-ms.subwords'
TRANSLATION_MS_EN_VOCAB = 'bpe/ms-en.subwords'
TRUE_CASE_VOCAB = 'bpe/true-case.yttm'
SEGMENTATION_VOCAB = 'bpe/segmentation.yttm'

ENTITY_SETTING = 'setting/entities.json'
ENTITY_ONTONOTES5_SETTING = 'setting/entities-ontonotes5.json'
POS_SETTING = 'setting/pos.json'

CONSTITUENCY_SETTING = 'setting/constituency.json'

MODEL_VOCAB = {
    'albert': ALBERT_BPE_VOCAB,
    'bert': BERT_BPE_VOCAB,
    'tiny-albert': ALBERT_BPE_VOCAB,
    'tiny-bert': BERT_BPE_VOCAB,
    'xlnet': XLNET_BPE_VOCAB,
    'alxlnet': XLNET_BPE_VOCAB,
    'bigbird': BERT_BPE_VOCAB,
    'tiny-bigbird': BERT_BPE_VOCAB,
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
}

LM_VOCAB = {
    'translation-en-ms': TRANSLATION_EN_MS_VOCAB,
    'translation-ms-en': TRANSLATION_MS_EN_VOCAB,
    'true-case': TRUE_CASE_VOCAB,
    'segmentation': SEGMENTATION_VOCAB,
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
    }
}

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
    'bigbird': {
        'model': home + '/relevancy/bigbird/base/model.pb',
        'quantized': home + '/relevancy/bigbird/base/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
        'version': 'v40',
    },
    'tiny-bigbird': {
        'model': home + '/relevancy/bigbird/tiny/model.pb',
        'quantized': home + '/relevancy/bigbird/tiny/quantized/model.pb',
        'vocab': home + '/bert/sp10m.cased.bert.vocab',
        'tokenizer': home + '/bert/sp10m.cased.bert.model',
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
    'bigbird': {
        'model': 'v42/relevancy/bigbird-base-relevancy.pb',
        'quantized': 'v42/relevancy/bigbird-base-relevancy.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
    },
    'tiny-bigbird': {
        'model': 'v42/relevancy/tiny-bigbird-relevancy.pb',
        'quantized': 'v42/relevancy/tiny-bigbird-relevancy.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.bert.vocab',
        'tokenizer': 'tokenizer/sp10m.cased.bert.model',
    },
}

PATH_SEGMENTATION = {
    'base': {
        'model': home + '/segmentation/base/model.pb',
        'quantized': home + '/segmentation/base/quantized/model.pb',
        'vocab': home + '/segmentation/vocab.yttm',
        'version': 'v41',
    },
    'small': {
        'model': home + '/segmentation/small/model.pb',
        'quantized': home + '/segmentation/quantized/small/model.pb',
        'vocab': home + '/segmentation/vocab.yttm',
        'version': 'v40',
    },
}
S3_PATH_SEGMENTATION = {
    'base': {
        'model': 'v41/segmentation/base.pb',
        'quantized': 'v41/segmentation/base.pb.quantized',
        'vocab': 'tokenizer/segmentation.yttm',
    },
    'small': {
        'model': 'v40/segmentation/small.pb',
        'quantized': 'v40/segmentation/small.pb.quantized',
        'vocab': 'tokenizer/segmentation.yttm',
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

PATH_TATABAHASA = {
    'small': {
        'model': home + '/tatabahasa/transformertag/small/model.pb',
        'quantized': home
        + '/tatabahasa/transformertag/small/quantized/model.pb',
        'vocab': home + '/tatabahasa/sp10m.cased.t5.model',
        'version': 'v41',
    },
    'base': {
        'model': home + '/tatabahasa/transformertag/base/model.pb',
        'quantized': home + '/tatabahasa/transformer/base/quantized/model.pb',
        'vocab': home + '/tatabahasa/sp10m.cased.t5.model',
        'version': 'v41',
    },
}

S3_PATH_TATABAHASA = {
    'base': {
        'model': 'v41/tatabahasa/base.pb',
        'quantized': 'v41/tatabahasa/base.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.t5.model',
    },
    'small': {
        'model': 'v41/tatabahasa/small.pb',
        'quantized': 'v41/tatabahasa/small.pb.quantized',
        'vocab': 'tokenizer/sp10m.cased.t5.model',
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
