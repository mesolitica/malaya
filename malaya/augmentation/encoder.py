import random
import inspect
import numpy as np
import re
import string as string_function
from tensorflow.keras.preprocessing.sequence import pad_sequences
from malaya.augmentation.base import _make_upper, _replace
from malaya.text.tatabahasa import consonants, vowels
from malaya.text.function import is_emoji
from malaya.text.regex import _expressions, _money, _date
from herpetologist import check_type
from typing import Callable, Dict, List
import warnings


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


@check_type
def wordvector(
    string: str,
    wordvector,
    threshold: float = 0.5,
    top_n: int = 5,
    soft: bool = False,
):
    """
    augmenting a string using wordvector.

    Parameters
    ----------
    string: str
        this string input assumed been properly tokenized and cleaned.
    wordvector: object
        wordvector interface object.
    threshold: float, optional (default=0.5)
        random selection for a word.
    soft: bool, optional (default=False)
        if True, a word not in the dictionary will be replaced with nearest jarowrinkler ratio.
        if False, it will throw an exception if a word not in the dictionary.
    top_n: int, (default=5)
        number of nearest neighbors returned. Length of returned result should as top_n.

    Returns
    -------
    result: List[str]
    """
    if not hasattr(wordvector, 'batch_n_closest'):
        raise ValueError('wordvector must have `batch_n_closest` method')
    if not hasattr(wordvector, '_dictionary'):
        raise ValueError('wordvector must have `_dictionary` attribute')

    original_string = string
    string = string.split()
    original_string = string[:]
    selected = []
    for no, w in enumerate(string):
        word = w
        if w in string_function.punctuation:
            continue
        if w[0].isupper():
            continue
        if (
            re.findall(_money, word.lower())
            or re.findall(_date, word.lower())
            or re.findall(_expressions['email'], word.lower())
            or re.findall(_expressions['url'], word.lower())
            or re.findall(_expressions['hashtag'], word.lower())
            or re.findall(_expressions['phone'], word.lower())
            or re.findall(_expressions['money'], word.lower())
            or re.findall(_expressions['date'], word.lower())
            or re.findall(_expressions['time'], word.lower())
            or re.findall(_expressions['ic'], word.lower())
            or re.findall(_expressions['user'], word.lower())
            or is_emoji(word.lower())
        ):
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
):
    """
    augmenting a string using transformer + nucleus sampling / top-k sampling.

    Parameters
    ----------
    string: str
        this string input assumed been properly tokenized and cleaned.
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

    Returns
    -------
    result: List[str]
    """
    warnings.warn(
        '`malaya.augmentation.encoder.transformer` is deprecated.', DeprecationWarning)

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

    original_string = string
    string = string.split()
    results = []
    for token_idx, token in enumerate(string):
        word = token
        if token in string_function.punctuation:
            continue
        if token[0].isupper():
            continue
        if token.isdigit():
            continue
        if (
            re.findall(_money, word.lower())
            or re.findall(_date, word.lower())
            or re.findall(_expressions['email'], word.lower())
            or re.findall(_expressions['url'], word.lower())
            or re.findall(_expressions['hashtag'], word.lower())
            or re.findall(_expressions['phone'], word.lower())
            or re.findall(_expressions['money'], word.lower())
            or re.findall(_expressions['date'], word.lower())
            or re.findall(_expressions['time'], word.lower())
            or re.findall(_expressions['ic'], word.lower())
            or re.findall(_expressions['user'], word.lower())
            or is_emoji(word.lower())
        ):
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
