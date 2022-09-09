from malaya.function import describe_availability, check_file
from herpetologist import check_type

_kenlm_availability = {
    'bahasa-news': {
        'Size (MB)': 24,
        'LM order': 3,
        'Description': 'local news.',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'bahasa-wiki': {
        'Size (MB)': 29,
        'LM order': 3,
        'Description': 'MS wikipedia.',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
    'bahasa-wiki-news': {
        'Size (MB)': 29,
        'LM order': 3,
        'Description': 'MS wikipedia + local news.',
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
    'dump-combined': {
        'Size (MB)': 310,
        'LM order': 3,
        'Description': 'Academia + News + IIUM + Parliament + Watpadd + Wikipedia + Common Crawl + training set from https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt',
        'Command': [
            './lmplz --text text.txt --arpa out.arpa -o 3 --prune 0 1 1',
            './build_binary -q 8 -b 7 -a 256 trie out.arpa out.trie.klm',
        ],
    },
}


def available_kenlm():
    """
    List available KenLM Language Model.
    """

    return describe_availability(_kenlm_availability)


@check_type
def kenlm(
    model: str = 'dump-combined', **kwargs
):
    """
    Load KenLM language model.

    Parameters
    ----------
    model : str, optional (default='dump-combined')
        Model architecture supported. Allowed values:

        * ``'bahasa-news'`` - Gathered from malaya-speech ASR bahasa transcript + Bahasa News (Random sample 300k sentences).
        * ``'bahasa-combined'`` - Gathered from malaya-speech ASR bahasa transcript + Bahasa News (Random sample 300k sentences) + Bahasa Wikipedia (Random sample 150k sentences).
        * ``'redape-community'`` - Mirror for https://github.com/redapesolutions/suara-kami-community
        * ``'dump-combined'`` - Academia + News + IIUM + Parliament + Watpadd + Wikipedia + Common Crawl + training set from https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model/prepare-stt.

    Returns
    -------
    result : kenlm.Model class
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
