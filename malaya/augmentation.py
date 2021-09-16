import random
import json
import inspect
import numpy as np
import string as string_function
from collections import defaultdict
from malaya.function import check_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from malaya.text.tatabahasa import alphabet, consonants, vowels
from malaya.text.function import augmentation_textcleaning, simple_textcleaning
from malaya.path import PATH_AUGMENTATION, S3_PATH_AUGMENTATION
from herpetologist import check_type
from typing import List, Dict, Tuple, Callable

_synonym_dict = None


def to_ids(string, tokenizer):
    words = []
    for no, word in enumerate(string):
        if word == '[MASK]':
            words.append(word)
        else:
            words.extend(tokenizer.tokenize(word))
    masked_tokens = ['[CLS]'] + words + ['[SEP]']
    masked_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    return masked_ids, masked_ids.index(tokenizer.vocab['[MASK]'])


def replace(string, threshold):
    for no, word in enumerate(string):
        if word in _synonym_dict and random.random() > threshold:
            w = random.choice(_synonym_dict[word])
            string[no] = w
    return string


def _make_upper(p, o):
    p_split = p.split()
    o_split = o.split()
    return ' '.join(
        [
            s.title() if o_split[no][0].isupper() else s
            for no, s in enumerate(p_split)
        ]
    )


@check_type
def synonym(
    string: str,
    threshold: float = 0.5,
    top_n=5,
    cleaning=augmentation_textcleaning,
    **kwargs
):
    """
    augmenting a string using synonym, https://github.com/huseinzol05/Malaya-Dataset#90k-synonym

    Parameters
    ----------
    string: str
    threshold: float, optional (default=0.5)
        random selection for a word.
    top_n: int, (default=5)
        number of nearest neighbors returned. Length of returned result should as top_n.
    cleaning: function, (default=malaya.text.function.augmentation_textcleaning)
        function to clean text.

    Returns
    -------
    result: List[str]
    """
    if not isinstance(cleaning, Callable) and cleaning is not None:
        raise ValueError('cleaning must be a callable type or None')

    global _synonym_dict

    if _synonym_dict is None:
        check_file(
            PATH_AUGMENTATION['synonym'],
            S3_PATH_AUGMENTATION['synonym'],
            **kwargs
        )
        synonyms = defaultdict(list)
        files = [
            PATH_AUGMENTATION['synonym']['model'],
            PATH_AUGMENTATION['synonym']['model2'],
        ]
        for file in files:
            with open(file) as fopen:
                data = json.load(fopen)

            for i in data:
                if not len(i[1]):
                    continue
                synonyms[i[0]].extend(i[1])
                for r in i[1]:
                    synonyms[r].append(i[0])
        for k, v in synonyms.items():
            synonyms[k] = list(set(v))
        _synonym_dict = synonyms

    original_string = string
    if cleaning:
        string = cleaning(string).split()

    augmented = []
    for i in range(top_n):
        string_ = replace(string, threshold)
        augmented.append(
            _make_upper(' '.join(string_), ' '.join(original_string))
        )
    return augmented


@check_type
def wordvector(
    string: str,
    wordvector,
    threshold: float = 0.5,
    top_n: int = 5,
    soft: bool = False,
    cleaning=augmentation_textcleaning,
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
        number of nearest neighbors returned. Length of returned result should as top_n.
    cleaning: function, (default=malaya.text.function.augmentation_textcleaning)
        function to clean text.

    Returns
    -------
    result: List[str]
    """

    if not isinstance(cleaning, Callable) and cleaning is not None:
        raise ValueError('cleaning must be a callable type or None')
    if not hasattr(wordvector, 'batch_n_closest'):
        raise ValueError('wordvector must have `batch_n_closest` method')
    if not hasattr(wordvector, '_dictionary'):
        raise ValueError('wordvector must have `_dictionary` attribute')

    from malaya.preprocessing import _tokenizer

    original_string = string
    if cleaning:
        string = cleaning(string)
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
            words, num_closest=top_n, soft=soft
        )
    else:
        results = wordvector.batch_n_closest(words, num_closest=top_n)

    augmented = []
    for i in range(top_n):
        string_ = string[:]
        for no in range(len(results)):
            string_[indices[no]] = results[no][i]
        augmented.append(
            _make_upper(' '.join(string_), ' '.join(original_string))
        )
    return augmented


@check_type
def transformer(
    string: str,
    model,
    threshold: float = 0.5,
    top_p: float = 0.9,
    top_k: int = 100,
    temperature: float = 1.0,
    top_n: int = 5,
    cleaning=None,
):
    """
    augmenting a string using transformer + nucleus sampling / top-k sampling.

    Parameters
    ----------
    string: str
    model: object
        transformer interface object. Right now only supported BERT, ALBERT and ELECTRA.
    threshold: float, optional (default=0.5)
        random selection for a word.
    top_p: float, optional (default=0.8)
        cumulative sum of probabilities to sample a word.
        If top_n bigger than 0, the model will use nucleus sampling, else top-k sampling.
    top_k: int, optional (default=100)
        k for top-k sampling.
    temperature: float, optional (default=0.8)
        logits * temperature.
    top_n: int, (default=5)
        number of nearest neighbors returned. Length of returned result should as top_n.
    cleaning: function, (default=None)
        function to clean text.

    Returns
    -------
    result: List[str]
    """
    if not isinstance(cleaning, Callable) and cleaning is not None:
        raise ValueError('cleaning must be a callable type or None')
    if not hasattr(model, 'samples'):
        raise ValueError('model must have `samples` attribute')
    if not (threshold > 0 and threshold < 1):
        raise ValueError('threshold must be bigger than 0 and less than 1')
    if not top_p > 0:
        raise ValueError('top_p must be bigger than 0')
    if not top_k > 0:
        raise ValueError('top_k must be bigger than 0')
    if not 0 < temperature <= 1.0:
        raise ValueError('temperature must, 0 < temperature <= 1.0')
    if not top_n > 0:
        raise ValueError('top_n must be bigger than 0')
    if top_n > top_k:
        raise ValueError('top_k must be bigger than top_n')

    from malaya.preprocessing import _tokenizer

    original_string = string
    if cleaning:
        string = cleaning(string)
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

    maskeds, indices, input_masks, input_segments = [], [], [], []
    for index in results:
        new = string[:]
        new[index] = '[MASK]'
        mask, ind = to_ids(new, model._tokenizer)
        maskeds.append(mask)
        indices.append(ind)
        input_masks.append([1] * len(mask))
        input_segments.append([0] * len(mask))

    masked_padded = pad_sequences(maskeds, padding='post')
    input_masks = pad_sequences(input_masks, padding='post')
    input_segments = pad_sequences(input_segments, padding='post')
    batch_indices = np.array([np.arange(len(indices)), indices]).T
    samples = model._sess.run(
        model.samples,
        feed_dict={
            model.X: masked_padded,
            model.MASK: input_masks,
            model.top_p: top_p,
            model.top_k: top_k,
            model.temperature: temperature,
            model.indices: batch_indices,
            model.k: top_n,
            model.segment_ids: input_segments,
        },
    )

    outputs = []
    for i in range(samples.shape[1]):
        sample_i = samples[:, i]
        samples_tokens = model._tokenizer.convert_ids_to_tokens(
            sample_i.tolist()
        )
        if hasattr(model._tokenizer, 'sp_model'):
            new_splitted = ['▁' + w if len(w) > 1 else w for w in string]
        else:
            new_splitted = [w if len(w) > 1 else w for w in string]
        for no, index in enumerate(results):
            new_splitted[index] = samples_tokens[no]
        if hasattr(model._tokenizer, 'sp_model'):
            new = ''.join(model._tokenizer.sp_model.DecodePieces(new_splitted))
        else:
            new = ' '.join(new_splitted)
        outputs.append(new)
    return outputs


def _replace(word, replace_dict, threshold=0.5):
    word = list(word[:])
    for i in range(len(word)):
        if word[i] in replace_dict and random.random() >= threshold:
            word[i] = replace_dict[word[i]]
    return ''.join(word)


def replace_similar_consonants(word: str, threshold: float = 0.8):
    """
    Naively replace consonants into similar consonants in a word.

    Parameters
    ----------
    word: str
    threshold: float, optional (default=0.8)

    Returns
    -------
    result: List[str]
    """
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
    return _replace(word=word, replace_dict=replace_consonants, threshold=threshold)


def replace_similar_vowels(word: str, threshold: float = 0.8):
    """
    Naively replace vowels into similar vowels in a word.
    Parameters
    ----------
    word: str
    threshold: float, optional (default=0.8)

    Returns
    -------
    result: List[str]
    """
    replace_vowels = {'u': 'i', 'i': 'o', 'o': 'u'}
    return _replace(word=word, replace_dict=replace_vowels, threshold=threshold)


@check_type
def socialmedia_form(word: str):
    """
    augmenting a word into socialmedia form.

    Parameters
    ----------
    word: str

    Returns
    -------
    result: List[str]
    """

    word = simple_textcleaning(word)
    if not len(word):
        raise ValueError('word is too short to augment shortform.')

    results = []

    if len(word) > 1:

        if word[-1] == 'a' and word[-2] in consonants:
            results.append(word[:-1] + 'e')

        if word[0] == 'f' and word[-1] == 'r':
            results.append('p' + word[1:])

        if word[-2] in consonants and word[-1] in vowels:
            results.append(word + 'k')

        if word[-2] in vowels and word[-1] == 'h':
            results.append(word[:-1])

    if len(word) > 2:
        if word[-3] in consonants and word[-2:] == 'ar':
            results.append(word[:-2] + 'o')

        if word[0] == 'h' and word[1] in vowels and word[2] in consonants:
            results.append(word[1:])

        if word[-3] in consonants and word[-2:] == 'ng':
            results.append(word[:-2] + 'g')

        if word[1:3] == 'ng':
            results.append(word[:1] + x[2:])

    return list(set(results))


def vowel_alternate(word: str, threshold: float = 0.5):
    """
    augmenting a word into vowel alternate.

    vowel_alternate('singapore')
    -> sngpore

    vowel_alternate('kampung')
    -> kmpng

    vowel_alternate('ayam')
    -> aym

    Parameters
    ----------
    word: str
    threshold: float, optional (default=0.5)

    Returns
    -------
    result: str
    """

    word = simple_textcleaning(word)
    if not len(word):
        raise ValueError('word is too short to augment shortform.')

    word = list(word[:])
    i = 0
    while i < len(word) - 2:
        subword = word[i: i + 3]
        if subword[0] in consonants and subword[1] in vowels and subword[2] in consonants \
                and random.random() >= threshold:
            word.pop(i + 1)
        i += 1
    return ''.join(word)
