import random
import tensorflow as tf
import logging
from malaya.text.bpe import SentencePieceTokenizer
from malaya.text.ngram import ngrams as generate_ngrams
from malaya.supervised import t5 as t5_load
from malaya.supervised import gpt2 as gpt2_load
from malaya.model.t5 import Generator, CommonGen
from malaya.path import PATH_NGRAM, S3_PATH_NGRAM
from malaya.function import check_file
from herpetologist import check_type
from typing import List, Dict, Tuple, Callable

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


@check_type
def ngrams(
    sequence,
    n: int,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    """
    generate ngrams.

    Parameters
    ----------
    sequence : List[str]
        list of tokenize words.
    n : int
        ngram size

    Returns
    -------
    result: List[Tuple[str, str]]
    """
    return generate_ngrams(
        sequence=sequence,
        n=n,
        pad_left=pad_left,
        pad_right=pad_right,
        left_pad_symbol=left_pad_symbol,
        right_pad_symbol=right_pad_symbol,
    )


@check_type
def pos_entities_ngram(
    result_pos: List[Tuple[str, str]],
    result_entities: List[Tuple[str, str]],
    ngram: Tuple[int, int] = (1, 3),
    accept_pos: List[str] = ['NOUN', 'PROPN', 'VERB'],
    accept_entities: List[str] = [
        'law',
        'location',
        'organization',
        'person',
        'time',
    ],
):
    """
    generate ngrams.

    Parameters
    ----------
    result_pos : List[Tuple[str, str]]
        result from POS recognition.
    result_entities : List[Tuple[str, str]]
        result of Entities recognition.
    ngram : Tuple[int, int]
        ngram sizes.
    accept_pos : List[str]
        accepted POS elements.
    accept_entities : List[str]
        accept entities elements.

    Returns
    -------
    result: list
    """
    if not all([i in _accepted_pos for i in accept_pos]):
        raise ValueError(
            'accept_pos must be a subset or equal of supported POS, please run malaya.describe_pos() to get supported POS'
        )
    if not all([i in _accepted_entities for i in accept_entities]):
        raise ValueError(
            'accept_entites must be a subset or equal of supported entities, please run malaya.describe_entities() to get supported entities'
        )

    words = []
    sentences = []
    for no in range(len(result_pos)):
        if (
            result_pos[no][1] in accept_pos
            or result_entities[no][1] in accept_entities
        ):
            words.append(result_pos[no][0])
    for gram in range(ngram[0], ngram[1] + 1, 1):
        gram_words = list(ngrams(words, gram))
        for sentence in gram_words:
            sentences.append(' '.join(sentence))
    return list(set(sentences))


@check_type
def sentence_ngram(sentence: str, ngram: Tuple[int, int] = (1, 3)):
    """
    generate ngram for a text

    Parameters
    ----------
    sentence: str
    ngram : tuple
        ngram sizes.

    Returns
    -------
    result: list
    """

    words = sentence.split()
    sentences = []
    for gram in range(ngram[0], ngram[1] + 1, 1):
        gram_words = list(ngrams(words, gram))
        for sentence in gram_words:
            sentences.append(' '.join(sentence))
    return list(set(sentences))


@check_type
def babble(
    string: str,
    model,
    generate_length: int = 30,
    leed_out_len: int = 1,
    temperature: float = 1.0,
    top_k: int = 100,
    burnin: int = 15,
    batch_size: int = 5,
):
    """
    Use pretrained transformer models to generate a string given a prefix string.
    https://github.com/nyu-dl/bert-gen, https://arxiv.org/abs/1902.04094

    Parameters
    ----------
    string: str
    model: object
        transformer interface object. Right now only supported BERT, ALBERT and ELECTRA.
    generate_length : int, optional (default=256)
        length of sentence to generate.
    leed_out_len : int, optional (default=1)
        length of extra masks for each iteration.
    temperature: float, optional (default=1.0)
        logits * temperature.
    top_k: int, optional (default=100)
        k for top-k sampling.
    burnin: int, optional (default=15)
        for the first burnin steps, sample from the entire next word distribution, instead of top_k.
    batch_size: int, optional (default=5)
        generate sentences size of batch_size.

    Returns
    -------
    result: List[str]
    """

    if not hasattr(model, 'samples'):
        raise ValueError('model must have `samples` attribute')
    if generate_length < 10:
        raise ValueError('generate_length must bigger than 10')
    if not 0 < temperature <= 1.0:
        raise ValueError('temperature must, 0 < temperature <= 1.0')
    if not top_k > 0:
        raise ValueError('top_k must be bigger than 0')
    if not burnin > 0:
        raise ValueError('burnin must be bigger than 0')
    if leed_out_len >= generate_length:
        raise ValueError('leed_out_len must be smaller than generate_length')
    if burnin >= generate_length:
        raise ValueError('burnin must be smaller than generate_length')

    from malaya.transformers.babble import sequential_generation

    if tf.executing_eagerly():
        logging.warning(
            'malaya.generator.babble will disable eager execution.'
        )
        tf.compat.v1.disable_eager_execution()

    return sequential_generation(
        string,
        model,
        batch_size=batch_size,
        max_len=generate_length,
        leed_out_len=leed_out_len,
        temperature=temperature,
        top_k=top_k,
        burnin=burnin,
    )


def available_gpt2():
    """
    List available gpt2 generator models.
    """
    from malaya.function import describe_availability

    return describe_availability(_gpt2_availability,
                                 text='calculate perplexity on never seen malay karangan.')


@check_type
def gpt2(model: str = '345M', quantized: bool = False, **kwargs):
    """
    Load GPT2 model to generate a string given a prefix string.

    Parameters
    ----------
    model : str, optional (default='345M')
        Model architecture supported. Allowed values:

        * ``'117M'`` - GPT2 117M parameters.
        * ``'345M'`` - GPT2 345M parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.tf.GPT2 class
    """

    model = model.upper()
    if model not in _gpt2_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.generator.available_gpt2()`.'
        )

    return gpt2_load.load(
        model=model,
        quantized=quantized,
        **kwargs,
    )


def available_isi_penting():
    """
    List available transformer models for isi penting generator.
    """
    from malaya.function import describe_availability

    return describe_availability(_isi_penting_availability)


@check_type
def isi_penting(model: str = 't5', quantized: bool = False, **kwargs):
    """
    Load Transformer model to generate a string given a isu penting.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'t5'`` - T5 BASE parameters.
        * ``'small-t5'`` - T5 SMALL parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.t5.Generator class
    """

    model = model.lower()
    if model not in _isi_penting_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.generator.available_isi_penting()`.'
        )

    return t5_load.load(
        module='generator',
        model=model,
        model_class=Generator,
        quantized=quantized,
        **kwargs,
    )
