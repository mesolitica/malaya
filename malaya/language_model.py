from malaya.function import check_file
from malaya.torch_model.gpt2_lm import LM as GPT2LM

available_kenlm = {
    'bahasa-wiki': {'Size (MB)': 70.5,
                    'LM order': 3,
                    'Description': 'MS wikipedia.',
                    'Command': ['./lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
                                './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
                                ],
                    },
    'bahasa-news': {'Size (MB)': 107,
                    'LM order': 3,
                    'Description': 'local news.',
                    'Command': ['./lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
                                './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
                                ],
                    },
    'bahasa-wiki-news': {'Size (MB)': 165,
                         'LM order': 3,
                         'Description': 'MS wikipedia + local news.',
                         'Command': ['./lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
                                     './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
                                     ],
                         },
    'bahasa-wiki-news-iium-stt': {'Size (MB)': 416,
                                  'LM order': 3,
                                  'Description': 'MS wikipedia + local news + IIUM + STT',
                                  'Command': ['./lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
                                              './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
                                              ],
                                  },
    'dump-combined': {'Size (MB)': 310,
                      'LM order': 3,
                      'Description': 'Academia + News + IIUM + Parliament + Watpadd + Wikipedia + Common Crawl + training set from https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt',
                      'Command': ['./lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
                                  './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
                                  ],
                      },
    'redape-community': {'Size (MB)': 887.1,
                         'LM order': 4,
                         'Description': 'Mirror for https://github.com/redapesolutions/suara-kami-community',
                         'Command': ['./lmplz --text text.txt --arpa out.arpa -o 4 --prune 0 1 1 1',
                                     './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
                                     ],
                         },
}

available_gpt2 = {
    'mesolitica/gpt2-117m-bahasa-cased': {
        'Size (MB)': 454,
    },
}

available_mlm = {
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
    except BaseException:
        raise ModuleNotFoundError(
            'kenlm not installed. Please install it by `pip install pypi-kenlm` and try again.'
        )

    model = model.lower()
    if model not in available_kenlm:
        raise ValueError(
            'model not supported, please check supported models from `malaya.language_model.available_kenlm`.'
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


def gpt2(
    model: str = 'mesolitica/gpt2-117m-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
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

    if model not in available_gpt2 and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.language_model.available_gpt2`.'
        )
    model = GPT2LM.from_pretrained(model, **kwargs)
    model.load_tokenizer()
    return model


def mlm(
    model: str = 'malay-huggingface/bert-tiny-bahasa-cased',
    force_check: bool = True,
    **kwargs
):
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

    if model not in available_mlm and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.language_model.available_mlm`.'
        )

    return MLMScorer(model, **kwargs)
