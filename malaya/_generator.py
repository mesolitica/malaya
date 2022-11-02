import tensorflow as tf
import logging
from malaya.text.ngram import ngrams as generate_ngrams
from malaya.supervised import t5 as t5_load
from malaya.supervised import gpt2 as gpt2_load
from malaya.model.t5 import Generator
from herpetologist import check_type
from typing import List, Tuple

_accepted_pos = [
    'ADJ',
    'ADP',
    'ADV',
    'ADX',
    'CCONJ',
    'DET',
    'NOUN',
    'NUM',
    'PART',
    'PRON',
    'PROPN',
    'SCONJ',
    'SYM',
    'VERB',
    'X',
]
_accepted_entities = [
    'OTHER',
    'law',
    'location',
    'organization',
    'person',
    'quantity',
    'time',
    'event',
]


_isi_penting_availability = {
    't5': {'Size (MB)': 1250, 'Quantized Size (MB)': 481, 'Maximum Length': 1024},
    'small-t5': {'Size (MB)': 355.6, 'Quantized Size (MB)': 195, 'Maximum Length': 1024},
}

_gpt2_availability = {
    '117M': {'Size (MB)': 499, 'Quantized Size (MB)': 126, 'Perplexity': 6.232461},
    '345M': {'Size (MB)': 1420, 'Quantized Size (MB)': 357, 'Perplexity': 6.1040115},
}


def available_gpt2():
    """
    List available gpt2 generator models.
    """
    from malaya.function import describe_availability

    return describe_availability(_gpt2_availability,
                                 text='calculate perplexity on never seen malay karangan.')


def available_isi_penting():
    """
    List available transformer models for isi penting generator.
    """
    from malaya.function import describe_availability

    return describe_availability(_isi_penting_availability)
