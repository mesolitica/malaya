import random
from malaya.text.function import simple_textcleaning
from malaya.text.bpe import sentencepiece_tokenizer_bert as load_sentencepiece
from malaya.text.tatabahasa import alphabet, consonants, vowels
from malaya.text.ngram import ngrams as generate_ngrams
from malaya.supervised import t5 as t5_load
from malaya.model.t5 import Generator
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


_transformer_availability = {
    't5': {'Size (MB)': 1250, 'Quantized Size (MB)': 481},
    'small-t5': {'Size (MB)': 355.6, 'Quantized Size (MB)': 195},
}

_gpt2_availability = {
    '117M': {'Size (MB)': 441.6, 'Perplexity': 5.47394},
    '345M': {'Size (MB)': 1200, 'Perplexity': 2.4596},
}


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
    return generate_ngrams(
        sequence = sequence,
        n = n,
        pad_left = pad_left,
        pad_right = pad_right,
        left_pad_symbol = left_pad_symbol,
        right_pad_symbol = right_pad_symbol,
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
def shortform(
    word: str,
    augment_vowel: bool = True,
    augment_consonant: bool = True,
    prob_delete_vowel: float = 0.5,
    **kwargs,
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
        raise ValueError(
            'prob_delete_vowel must be bigger than 0 and less than 1'
        )
    word = simple_textcleaning(word)
    if not len(word):
        raise ValueError('word is too short to augment shortform.')

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
    from malaya.function import describe_availability

    return describe_availability(_gpt2_availability)


@check_type
def gpt2(
    model: str = '345M',
    generate_length: int = 256,
    temperature: float = 1.0,
    top_k: int = 40,
    **kwargs,
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
    if model not in _gpt2_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.generator.available_gpt2()`.'
        )

    if generate_length < 10:
        raise ValueError('generate_length must bigger than 10')
    if not 0 < temperature <= 1.0:
        raise ValueError('temperature must, 0 < temperature <= 1.0')
    if top_k < 5:
        raise ValueError('top_k must bigger than 5')
    from malaya.transformers.gpt2 import load

    return load(
        model = model,
        generate_length = generate_length,
        temperature = temperature,
        top_k = top_k,
        **kwargs,
    )


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(_transformer_availability)


@check_type
def transformer(model: str = 't5', quantized: bool = False, **kwargs):

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
    result: model
        List of model classes:
        
        * if `t5` in model, will return `malaya.model.t5.Generator`.
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.generator.available_transformer()`.'
        )

    return t5_load.load(
        module = 'generator',
        model = model,
        model_class = Generator,
        quantized = quantized,
        **kwargs,
    )
