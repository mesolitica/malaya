import itertools
import random
import inspect
from .texts._text_functions import simple_textcleaning

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


def ngrams(
    sequence,
    n,
    pad_left = False,
    pad_right = False,
    left_pad_symbol = None,
    right_pad_symbol = None,
):
    """
    generate ngrams.

    Parameters
    ----------
    sequence : list of str
        list of tokenize words.
    n : int
        ngram size

    Returns
    -------
    ngram: list
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


def pos_entities_ngram(
    result_pos,
    result_entities,
    ngram = (1, 3),
    accept_pos = ['NOUN', 'PROPN', 'VERB'],
    accept_entities = ['law', 'location', 'organization', 'person', 'time'],
):
    """
    generate ngrams.

    Parameters
    ----------
    result_pos : list of tuple
        result from POS recognition.
    result_entities : list of tuple
        result of Entities recognition.
    ngram : tuple
        ngram sizes.
    accept_pos : list of str
        accepted POS elements.
    accept_entities : list of str
        accept entities elements.

    Returns
    -------
    result: list
    """
    if not isinstance(result_pos, list):
        raise ValueError('result_pos must be a list')
    if not isinstance(result_pos[0], tuple):
        raise ValueError('result_pos must be a list of tuple')
    if not isinstance(ngram, tuple):
        raise ValueError('ngram must be a tuple')
    if not len(ngram) == 2:
        raise ValueError('ngram size must equal to 2')
    if not isinstance(accept_pos, list):
        raise ValueError('accept_pos must be a list')
    if not isinstance(accept_entities, list):
        raise ValueError('accept_entites must be a list')
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


def sentence_ngram(sentence, ngram = (1, 3)):
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
    if not isinstance(sentence, str):
        raise ValueError('sentence must be a string')
    if not isinstance(ngram, tuple):
        raise ValueError('ngram must be a tuple')
    if not len(ngram) == 2:
        raise ValueError('ngram size must equal to 2')

    words = sentence.split()
    sentences = []
    for gram in range(ngram[0], ngram[1] + 1, 1):
        gram_words = list(ngrams(words, gram))
        for sentence in gram_words:
            sentences.append(' '.join(sentence))
    return list(set(sentences))


def w2v_augmentation(
    string,
    w2v,
    threshold = 0.5,
    soft = False,
    random_select = True,
    augment_counts = 1,
    top_n = 5,
    cleaning_function = simple_textcleaning,
):
    """
    augmenting a string using word2vec

    Parameters
    ----------
    string: str
    w2v: object
        word2vec interface object.
    threshold: float, optional (default=0.5)
        random selection for a word.
    soft: bool, optional (default=False)
        if True, a word not in the dictionary will be replaced with nearest fuzzywuzzy ratio.
        if False, it will throw an exception if a word not in the dictionary.
    random_select: bool, (default=True)
        if True, a word randomly selected in the pool.
        if False, based on the index
    augment_counts: int, (default=1)
        augmentation count for a string.
    top_n: int, (default=5)
        number of nearest neighbors returned.
    cleaning_function: function, (default=simple_textcleaning)


    Returns
    -------
    result: list
    """
    if not isinstance(string, str):
        raise ValueError('string must be a string')
    if not isinstance(threshold, float):
        raise ValueError('threshold must be a float')
    if not (threshold > 0 and threshold < 1):
        raise ValueError('threshold must be bigger than 0 and less than 1')
    if not isinstance(soft, bool):
        raise ValueError('soft must be a boolean')
    if not hasattr(w2v, 'batch_n_closest'):
        raise ValueError('word2vec must has `batch_n_closest` method')
    if not isinstance(random_select, bool):
        raise ValueError('random_select must be a boolean')
    if not isinstance(top_n, int):
        raise ValueError('top_n must be an integer')
    if not isinstance(augment_counts, int):
        raise ValueError('augment_counts must be an integer')
    if not random_select:
        if augment_counts > top_n:
            raise ValueError(
                'if random_select is False, augment_counts need to be less than or equal to top_n'
            )
    if cleaning_function:
        string = cleaning_function(string)
    string = string.split()
    selected = []
    while not len(selected):
        selected = [
            (no, w)
            for no, w in enumerate(string)
            if random.random() > threshold
        ]
    indices, words = [i[0] for i in selected], [i[1] for i in selected]
    batch_parameters = list(
        inspect.signature(w2v.batch_n_closest).parameters.keys()
    )
    if 'soft' in batch_parameters:
        results = w2v.batch_n_closest(words, num_closest = top_n, soft = soft)
    else:
        results = w2v.batch_n_closest(words, num_closest = top_n)
    augmented = []
    for i in range(augment_counts):
        string_ = string[:]
        for no in range(len(results)):
            if random_select:
                index = random.randint(0, len(results[no]) - 1)
            else:
                index = i
            string_[indices[no]] = results[no][index]
        augmented.append(' '.join(string_))
    return augmented
