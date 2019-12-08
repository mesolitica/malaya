import itertools
import random
import inspect
import numpy as np
import string as string_function
from .preprocessing import _tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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


def _check_digit(string):
    return any(i.isdigit() for i in string)


def _make_upper(p, o):
    p_split = p.split()
    o_split = o.split()
    return ' '.join(
        [
            s.title() if o_split[no][0].isupper() else s
            for no, s in enumerate(p_split)
        ]
    )


def _simple_textcleaning(string, lowering = True):
    """
    use by topic modelling
    only accept A-Z, a-z
    """
    string = unidecode(string)
    string = re.sub(
        '[^A-Za-z0-9\-\\/\'"!@#$%^&*()-+={}\[\];:<>,.? ]+', ' ', string
    )
    return re.sub(r'[ ]+', ' ', string.lower() if lowering else string).strip()


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


def to_ids(string, tokenizer):
    words = []
    for no, word in enumerate(string):
        if word == '<mask>':
            words.append(word)
        else:
            words.extend(tokenizer.tokenize(word))
    masked_tokens = ['<cls>'] + words + ['<sep>']
    masked_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    return masked_ids, masked_ids.index(6)


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


def wordvector_augmentation(
    string,
    wordvector,
    threshold = 0.5,
    top_n = 5,
    soft = False,
    cleaning_function = None,
):
    """
    augmenting a string using wordvector.

    Parameters
    ----------
    string: str
    wordvector: object
        wordvector interface object.
    threshold: float, optional (default=0.5)
        random selection for a word.
    soft: bool, optional (default=False)
        if True, a word not in the dictionary will be replaced with nearest jarowrinkler ratio.
        if False, it will throw an exception if a word not in the dictionary.
    top_n: int, (default=5)
        number of nearest neighbors returned.
    cleaning_function: function, (default=None)
        function to clean text.

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
    if not hasattr(wordvector, 'batch_n_closest'):
        raise ValueError('wordvector must has `batch_n_closest` method')
    if not hasattr(wordvector, '_dictionary'):
        raise ValueError('wordvector must has `_dictionary` attribute')
    if not isinstance(top_n, int):
        raise ValueError('top_n must be an integer')

    original_string = string
    if cleaning_function:
        string = cleaning_function(string)
    string = _tokenizer(string)
    original_string = string[:]
    selected = []
    for no, w in enumerate(string):
        if w in string_function.punctuation:
            continue
        if w[0].isupper():
            continue
        if random.random() > threshold:
            selected.append((no, w))

    if not len(selected):
        raise ValueError(
            'no words can augmented, make sure words available are not punctuation or proper nouns.'
        )

    indices, words = [i[0] for i in selected], [i[1] for i in selected]
    batch_parameters = list(
        inspect.signature(wordvector.batch_n_closest).parameters.keys()
    )
    if 'soft' in batch_parameters:
        results = wordvector.batch_n_closest(
            words, num_closest = top_n, soft = soft
        )
    else:
        results = wordvector.batch_n_closest(words, num_closest = top_n)

    augmented = []
    for i in range(top_n):
        string_ = string[:]
        for no in range(len(results)):
            string_[indices[no]] = results[no][i]
        augmented.append(
            _make_upper(' '.join(string_), ' '.join(original_string))
        )
    return augmented


def transformer_augmentation(
    string,
    model,
    threshold = 0.5,
    top_p = 0.8,
    top_k = 100,
    temperature = 0.8,
    top_n = 5,
    cleaning_function = None,
):

    """
    augmenting a string using transformer + nucleus sampling / top-k sampling.

    Parameters
    ----------
    string: str
    model: object
        transformer interface object. Right now only supported BERT.
    threshold: float, optional (default=0.5)
        random selection for a word.
    top_p: float, optional (default=0.8)
        cumulative sum of probabilities to sample a word. If top_n bigger than 0, the model will use nucleus sampling, else top-k sampling.
    top_k: int, optional (default=100)
        k for top-k sampling.
    temperature: float, optional (default=0.8)
        logits * temperature.
    top_n: int, (default=5)
        number of nearest neighbors returned.
    cleaning_function: function, (default=None)
        function to clean text.

    Returns
    -------
    result: list
    """

    if not isinstance(string, str):
        raise ValueError('string must be a string')
    if not hasattr(model, 'samples'):
        raise ValueError('model must has `samples` attribute')
    if not isinstance(threshold, float):
        raise ValueError('threshold must be a float')
    if not (threshold > 0 and threshold < 1):
        raise ValueError('threshold must be bigger than 0 and less than 1')
    if not isinstance(top_p, float):
        raise ValueError('top_p must be a float')
    if not top_p > 0:
        raise ValueError('top_p must be bigger than 0')
    if not isinstance(top_k, int):
        raise ValueError('top_k must be an integer')
    if not top_k > 0:
        raise ValueError('top_k must be bigger than 0')
    if not isinstance(temperature, float):
        raise ValueError('temperature must be a float')
    if not (temperature > 0 and threshold < 1):
        raise ValueError('temperature must be bigger than 0 and less than 1')
    if not isinstance(top_n, int):
        raise ValueError('top_n must be an integer')
    if not top_n > 0:
        raise ValueError('top_n must be bigger than 0')
    if top_n > top_k:
        raise ValueError('top_k must be bigger than top_n')

    original_string = string
    if cleaning_function:
        string = cleaning_function(string)
    string = _tokenizer(string)
    results = []
    for token_idx, token in enumerate(string):
        if token in string_function.punctuation:
            continue
        if token[0].isupper():
            continue
        if token.isdigit():
            continue
        if random.random() > threshold:
            results.append(token_idx)

    if not len(results):
        raise ValueError(
            'no words can augmented, make sure words available are not punctuation or proper nouns.'
        )

    maskeds, indices = [], []
    for index in results:
        new = string[:]
        new[index] = '<mask>'
        mask, ind = to_ids(new, model._tokenizer)
        maskeds.append(mask)
        indices.append(ind)

    masked_padded = pad_sequences(maskeds, padding = 'post')
    batch_indices = np.array([np.arange(len(indices)), indices]).T
    samples = model._sess.run(
        model.samples,
        feed_dict = {
            model.X: masked_padded,
            model.top_p: top_p,
            model.top_k: top_k,
            model.temperature: temperature,
            model.indices: batch_indices,
            model.k: top_n,
        },
    )
    outputs = []
    for i in range(samples.shape[1]):
        sample_i = samples[:, i]
        samples_tokens = model._tokenizer.convert_ids_to_tokens(
            sample_i.tolist()
        )
        new_splitted = ['▁' + w if len(w) > 1 else w for w in string]
        for no, index in enumerate(results):
            new_splitted[index] = samples_tokens[no]
        new = ''.join(model._tokenizer.sp_model.DecodePieces(new_splitted))
        outputs.append(new)
    return outputs
