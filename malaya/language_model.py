from malaya.function import describe_availability, check_file
from malaya.torch_model.gpt2_lm import LM as GPT2LM
from malaya.torch_model.mask_lm import (
    BertForMaskedLMOptimized,
    AlbertForMaskedLMOptimized,
    RobertaForMaskedLMOptimized,
    MLMScorer,
)
from transformers import AutoTokenizer
from herpetologist import check_type

_kenlm_availability = {
    'bahasa-wiki': {
        'Size (MB)': 70.5,
        'LM order': 3,
        'Description': 'MS wikipedia.',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'bahasa-news': {
        'Size (MB)': 107,
        'LM order': 3,
        'Description': 'local news.',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'bahasa-wiki-news': {
        'Size (MB)': 165,
        'LM order': 3,
        'Description': 'MS wikipedia + local news.',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'bahasa-wiki-news-iium-stt': {
        'Size (MB)': 416,
        'LM order': 3,
        'Description': 'MS wikipedia + local news + IIUM + STT',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'dump-combined': {
        'Size (MB)': 310,
        'LM order': 3,
        'Description': 'Academia + News + IIUM + Parliament + Watpadd + Wikipedia + Common Crawl + training set from https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'redape-community': {
        'Size (MB)': 887.1,
        'LM order': 4,
        'Description': 'Mirror for https://github.com/redapesolutions/suara-kami-community',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 4 --prune 0 1 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
}

_gpt2_availability = {
    'mesolitica/gpt2-117m-bahasa-cased': {
        'Size (MB)': 454,
    },
}

_mlm_availability = {
    'malay-huggingface/bert-base-bahasa-cased': {
        'Size (MB)': 310,
    },
    'malay-huggingface/bert-tiny-bahasa-cased': {
        'Size (MB)': 66.1,
    },
    'malay-huggingface/albert-base-bahasa-cased': {
        'Size (MB)': 45.9,
    },
    'malay-huggingface/albert-tiny-bahasa-cased': {
        'Size (MB)': 22.6,
    },
    'mesolitica/roberta-base-standard-bahasa-cased': {
        'Size (MB)': 443,
    },
    'mesolitica/roberta-tiny-standard-bahasa-cased': {
        'Size (MB)': 66.1,
    },
}


def available_kenlm():
    """
    List available KenLM Language Model.
    """

    return describe_availability(_kenlm_availability)


def available_gpt2():
    """
    List available GPT2 Language Model.
    """

    return describe_availability(_gpt2_availability)


def available_mlm():
    """
    List available MLM Language Model.
    """

    return describe_availability(_mlm_availability)


@check_type
def kenlm(model: str = 'dump-combined', **kwargs):
    """
    Load KenLM language model.

    Parameters
    ----------
    model: str, optional (default='dump-combined')
        Check available models at `malaya.language_model.available_models()`.
    Returns
    -------
    result: kenlm.Model class
    """

    try:
        import kenlm
    except:
        raise ModuleNotFoundError(
            'kenlm not installed. Please install it by `pip install pypi-kenlm` and try again.'
        )

    model = model.lower()
    if model not in _kenlm_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.kenlm.available_models()`.'
        )

    path = check_file(
        file=model,
        module='language-model',
        keys={
            'model': 'model.klm',
        },
        quantized=False,
        **kwargs,
    )
    return kenlm.Model(path['model'])


@check_type
def gpt2(model: str = 'mesolitica/gpt2-117m-bahasa-cased', force_check: bool = True, **kwargs):
    """
    Load GPT2 language model.

    Parameters
    ----------
    model: str, optional (default='mesolitica/gpt2-117m-bahasa-cased')
        Check available models at `malaya.language_model.available_gpt2()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.gpt2_lm.LM class
    """

    if model not in _gpt2_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.language_model.available_gpt2()`.'
        )
    model = GPT2LM.from_pretrained(model)
    model.load_tokenizer()
    return model


@check_type
def mlm(model: str = 'malay-huggingface/bert-tiny-bahasa-cased', force_check: bool = True, **kwargs):
    """
    Load Masked language model.

    Parameters
    ----------
    model: str, optional (default='malay-huggingface/bert-tiny-bahasa-cased')
        Check available models at `malaya.language_model.available_mlm()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.mask_lm.MLMScorer class
    """

    if model not in _mlm_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.language.available_mlm()`.'
        )

    splitted = model.lower().replace('/', '-').split('-')
    if 'bert' in splitted:
        model_class = BertForMaskedLMOptimized
    elif 'albert' in splitted:
        model_class = AlbertForMaskedLMOptimized
    elif 'roberta' in splitted:
        model_class = RobertaForMaskedLMOptimized
    else:
        raise ValueError(f'cannot determined model class for {model}, only supported BERT, ALBERT and RoBERTa for now.')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    model = model_class.from_pretrained(model)

    return MLMScorer(model, tokenizer)
