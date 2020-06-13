import itertools
import os
import random
import numpy as np
from malaya.text.function import simple_textcleaning
from malaya.text.bpe import sentencepiece_tokenizer_bert as load_sentencepiece
from malaya.text.tatabahasa import alphabet, consonants, vowels
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

_t5_availability = {'small': ['122MB'], 'base': ['448MB']}
_gpt2_availability = {
    '117M': ['441.6MB, perplexity: 5.47394'],
    '345M': ['1.2GB, perplexity: 2.4596'],
}


def _check_digit(string):
    return any(i.isdigit() for i in string)


def _pad_sequence(
    sequence,
    n,
    pad_left = False,
    pad_right = False,
    left_pad_symbol = None,
    right_pad_symbol = None,
):
    sequence = iter(sequence)
    if pad_left:
        sequence = itertools.chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = itertools.chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


@check_type
def ngrams(
    sequence,
    n: int,
    pad_left = False,
    pad_right = False,
    left_pad_symbol = None,
    right_pad_symbol = None,
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
    sequence = _pad_sequence(
        sequence, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol
    )

    history = []
    while n > 1:
        try:
            next_item = next(sequence)
        except StopIteration:
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


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
def shortform(
    word: str,
    augment_vowel: bool = True,
    augment_consonant: bool = True,
    prob_delete_vowel: float = 0.5,
    **kwargs
):
    """
    augmenting a formal word into socialmedia form. Purposely typo, purposely delete some vowels, 
    purposely replaced some subwords into slang subwords.

    Parameters
    ----------
    word: str
    augment_vowel: bool, (default=True)
        if True, will augment vowels for each samples generated.
    augment_consonant: bool, (default=True)
        if True, will augment consonants for each samples generated.
    prob_delete_vowel: float, (default=0.5)
        probability to delete a vowel.

    Returns
    -------
    result: list
    """

    if not 0 < prob_delete_vowel < 1:
        raise Exception(
            'prob_delete_vowel must be bigger than 0 and less than 1'
        )
    word = simple_textcleaning(word)
    if not len(word):
        raise Exception('word is too short to augment shortform.')

    check_file(
        PATH_NGRAM['sentencepiece'], S3_PATH_NGRAM['sentencepiece'], **kwargs
    )

    vocab = PATH_NGRAM['sentencepiece']['vocab']
    vocab_model = PATH_NGRAM['sentencepiece']['model']
    tokenizer = load_sentencepiece(vocab, vocab_model)

    replace_consonants = {
        'n': 'm',
        't': 'y',
        'r': 't',
        'g': 'h',
        'j': 'k',
        'k': 'l',
        'd': 's',
        'd': 'f',
        'g': 'f',
        'b': 'n',
    }

    replace_vowels = {'u': 'i', 'i': 'o', 'o': 'u'}

    results = [word]

    if len(word) > 1:

        if word[-1] == 'a' and word[-2] in consonants:
            results.append(word[:-1] + 'e')

        if word[0] == 'f' and word[-1] == 'r':
            results.append('p' + words[1:])

        if word[-2] in consonants and word[-1] in vowels:
            results.append(word + 'k')

        if word[-2] in vowels and word[-1] == 'h':
            results.append(word[:-1])

    if len(word) > 2:
        if word[-3] in consonants and word[-2:] == 'ar':
            results.append(words[:-2] + 'o')

        if word[0] == 'h' and word[1] in vowels and word[2] in consonants:
            results.append(word[1:])

        if word[-3] in consonants and word[-2:] == 'ng':
            results.append(word[:-2] + 'g')

        if word[1:3] == 'ng':
            results.append(word[:1] + x[2:])

    if augment_consonant:
        result_consonants = []
        for k, v in replace_consonants.items():
            for r in results:
                result_consonants.extend([r.replace(k, v), r.replace(v, k)])
        results.extend(result_consonants)

    if augment_vowel:
        result_vowels = []
        for k, v in replace_vowels.items():
            for r in results:
                result_vowels.extend([r.replace(k, v), r.replace(v, k)])
        results.extend(result_vowels)

    result_deleted = []
    for s in results:
        deleted = []
        for c in s:
            if random.random() > prob_delete_vowel and c in vowels:
                continue
            else:
                deleted.append(c)
        result_deleted.append(''.join(deleted))
    results.extend(result_deleted)

    filtered = []
    for s in results:
        t = tokenizer.tokenize(s)
        if len(t) == 1:
            filtered.append(s)
            continue
        if t[0] == '▁':
            continue
        if any([len(w) < 3 for w in t]):
            continue
        filtered.append(s)

    return list(set(filtered))


@check_type
def transformer(
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
        transformer interface object. Right now only supported BERT, ALBERT.
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
        raise ValueError('model must has `samples` attribute')
    if generate_length < 10:
        raise Exception('generate_length must bigger than 10')
    if not 0 < temperature <= 1.0:
        raise Exception('temperature must, 0 < temperature <= 1.0')
    if not top_k > 0:
        raise ValueError('top_k must be bigger than 0')
    if not burnin > 0:
        raise ValueError('burnin must be bigger than 0')
    if leed_out_len >= generate_length:
        raise ValueError('leed_out_len must be smaller than generate_length')
    if burnin >= generate_length:
        raise ValueError('burnin must be smaller than generate_length')

    from malaya.transformers.babble import sequential_generation

    return sequential_generation(
        string,
        model,
        batch_size = batch_size,
        max_len = generate_length,
        leed_out_len = leed_out_len,
        temperature = temperature,
        top_k = top_k,
        burnin = burnin,
    )


def available_gpt2():
    """
    List available gpt2 generator models.
    """
    return _gpt2_availability


@check_type
def gpt2(
    model: str = '345M',
    generate_length: int = 256,
    temperature: float = 1.0,
    top_k: int = 40,
    **kwargs
):

    """
    Load GPT2 model to generate a string given a prefix string.

    Parameters
    ----------
    model : str, optional (default='345M')
        Model architecture supported. Allowed values:

        * ``'117M'`` - GPT2 117M parameters.
        * ``'345M'`` - GPT2 345M parameters.

    generate_length : int, optional (default=256)
        length of sentence to generate.
    temperature : float, optional (default=1.0)
        temperature value, value should between 0 and 1.
    top_k : int, optional (default=40)
        top-k in nucleus sampling selection.

    Returns
    -------
    result: malaya.transformers.gpt2.Model class
    """

    model = model.upper()
    if model not in ['117M', '345M']:
        raise Exception(
            "model not supported, for now only supported ['117M', '345M']"
        )

    if generate_length < 10:
        raise Exception('generate_length must bigger than 10')
    if not 0 < temperature <= 1.0:
        raise Exception('temperature must, 0 < temperature <= 1.0')
    if top_k < 5:
        raise Exception('top_k must bigger than 5')
    from malaya.transformers.gpt2 import load

    return load(
        model = model,
        generate_length = generate_length,
        temperature = temperature,
        top_k = top_k,
        **kwargs
    )


def available_t5():
    """
    List available T5 models.
    """
    return _t5_availability


@check_type
def t5(model: str = 'base', **kwargs):

    """
    Load T5 model to generate a string given a isu penting.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'base'`` - T5 Base parameters.
        * ``'small'`` - T5 Small parameters.

    Returns
    -------
    result: malaya.model.t5.GENERATOR class
    """

    model = model.lower()
    if model not in _t5_availability:
        raise Exception(
            'model not supported, please check supported models from malaya.generator.available_t5()'
        )

    from malaya.path import PATH_GENERATOR, S3_PATH_GENERATOR

    path = PATH_GENERATOR['sample']
    s3_path = S3_PATH_GENERATOR['sample']

    from malaya.function import check_file

    try:
        import tensorflow_text
        import tf_sentencepiece
        import tensorflow as tf
    except:
        raise Exception(
            'tensorflow-text and tf-sentencepiece not installed. Please install it by `pip install tensorflow-text tf-sentencepiece` and try again. Also, make sure tensorflow-text version same as tensorflow version.'
        )

    check_file(path[model]['model'], s3_path[model], **kwargs)

    if not os.path.exists(path[model]['directory'] + 'saved_model.pb'):
        import tarfile

        with tarfile.open(path[model]['model']['model']) as tar:
            tar.extractall(path = path[model]['path'])

    sess = tf.InteractiveSession()
    meta_graph_def = tf.compat.v1.saved_model.load(
        sess, ['serve'], path[model]['directory']
    )
    signature_def = meta_graph_def.signature_def['serving_default']
    pred = lambda x: sess.run(
        fetches = signature_def.outputs['outputs'].name,
        feed_dict = {signature_def.inputs['input'].name: x},
    )

    from malaya.model.t5 import GENERATOR

    return GENERATOR(pred)
