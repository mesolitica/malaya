import re
import os
import random
import numpy as np
from fuzzywuzzy import fuzz
import json
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)
from ._utils._utils import (
    check_file,
    load_graph,
    check_available,
    generate_session,
)
from .preprocessing import _tokenizer
from ._utils._paths import PATH_SIMILARITY, S3_PATH_SIMILARITY
from ._models._tensorflow_model import SIAMESE, SIAMESE_BERT


def doc2vec(
    vectorizer,
    left_string,
    right_string,
    aggregation = 'mean',
    similarity = 'cosine',
    tokenizer = _tokenizer,
    soft = True,
):
    """
    Calculate similarity between 2 documents using doc2vec.

    Parameters
    ----------
    vectorizer : object
        fast-text or word2vec interface object.
    left_string: str
        first string to predict.
    right_string: str
        second string to predict.
    aggregation : str, optional (default='mean')
        Aggregation supported. Allowed values:

        * ``'mean'`` - mean.
        * ``'min'`` - min.
        * ``'max'`` - max.
        * ``'sum'`` - sum.
        * ``'sqrt'`` - square root.
    similarity : str, optional (default='mean')
        similarity supported. Allowed values:

        * ``'cosine'`` - cosine similarity.
        * ``'euclidean'`` - euclidean similarity.
        * ``'manhattan'`` - manhattan similarity.
    tokenizer : object
        default is tokenizer from malaya.preprocessing._SocialTokenizer
    soft: bool, optional (default=True)
        word not inside vectorizer will replace with nearest word if True, else, will skip.

    Returns
    -------
    result: float
    """

    if not hasattr(vectorizer, 'get_vector_by_name'):
        raise ValueError('vectorizer must has `get_vector_by_name` method')
    if not isinstance(left_string, str):
        raise ValueError('left_string must be a string')
    if not isinstance(right_string, str):
        raise ValueError('right_string must be a string')
    if not isinstance(aggregation, str):
        raise ValueError('aggregation must be a string')
    if not isinstance(similarity, str):
        raise ValueError('similarity must be a string')

    aggregation = aggregation.lower()
    if aggregation == 'mean':
        aggregation_function = np.mean
    elif aggregation == 'min':
        aggregation_function = np.min
    elif aggregation == 'max':
        aggregation_function = np.max
    elif aggregation == 'sum':
        aggregation_function = np.sum
    elif aggregation == 'sqrt':
        aggregation_function = np.sqrt
    else:
        raise ValueError(
            'aggregation only supports `mean`, `min`, `max`, `sum` and `sqrt`'
        )

    similarity = similarity.lower()
    if similarity == 'cosine':
        similarity_function = cosine_similarity
    elif similarity == 'euclidean':
        similarity_function = euclidean_distances
    elif similarity == 'manhattan':
        similarity_function = manhattan_distances
    else:
        raise ValueError(
            'similarity only supports `cosine`, `euclidean`, and `manhattan`'
        )

    left_tokenized = _tokenizer(left_string)
    if not len(left_tokenized):
        raise ValueError('insert not empty left string')
    right_tokenized = _tokenizer(right_string)
    if not len(right_tokenized):
        raise ValueError('insert not empty right string')

    left_vectors, right_vectors = [], []
    for token in left_tokenized:
        try:
            left_vectors.append(vectorizer.get_vector_by_name(token))
        except:
            if not soft:
                pass
            else:
                arr = np.array([fuzz.ratio(token, k) for k in vectorizer.words])
                idx = (-arr).argsort()[0]
                left_vectors.append(
                    vectorizer.get_vector_by_name(vectorizer.words[idx])
                )
    for token in right_tokenized:
        try:
            right_vectors.append(vectorizer.get_vector_by_name(token))
        except:
            if not soft:
                pass
            else:
                arr = np.array([fuzz.ratio(token, k) for k in vectorizer.words])
                idx = (-arr).argsort()[0]
                right_vectors.append(
                    vectorizer.get_vector_by_name(vectorizer.words[idx])
                )
    left_vectors = [aggregation_function(left_vectors, axis = 0)]
    right_vectors = [aggregation_function(right_vectors, axis = 0)]
    similar = similarity_function(left_vectors, right_vectors)[0, 0]
    if similarity == 'cosine':
        return (similar + 1) / 2
    else:
        return 1 / (similar + 1)


def available_deep_siamese():
    """
    List available deep siamese models.
    """
    return ['self-attention', 'bahdanau', 'dilated-cnn']


def deep_siamese(model = 'bahdanau', validate = True):
    """
    Load deep siamese model.

    Parameters
    ----------
    model : str, optional (default='bahdanau')
        Model architecture supported. Allowed values:

        * ``'self-attention'`` - Fast-text architecture, embedded and logits layers only with self attention.
        * ``'bahdanau'`` - LSTM with bahdanau attention architecture.
        * ``'dilated-cnn'`` - Pyramid Dilated CNN architecture.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    SIAMESE: malaya._models._tensorflow_model.SIAMESE class
    """

    if not isinstance(model, str):
        raise ValueError('model must be a string')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')
    model = model.lower()
    if model not in available_deep_siamese():
        raise Exception(
            'model is not supported, please check supported models from malaya.similarity.available_deep_siamese()'
        )
    if validate:
        check_file(PATH_SIMILARITY[model], S3_PATH_SIMILARITY[model])
    else:
        if not check_available(PATH_SIMILARITY[model]):
            raise Exception(
                'similarity/%s is not available, please `validate = True`'
                % (model)
            )
    try:
        with open(PATH_SIMILARITY[model]['setting'], 'r') as fopen:
            dictionary = json.load(fopen)['dictionary']
        g = load_graph(PATH_SIMILARITY[model]['model'])
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('similarity/%s') and try again"
            % (model)
        )
    return SIAMESE(
        g.get_tensor_by_name('import/Placeholder:0'),
        g.get_tensor_by_name('import/Placeholder_1:0'),
        g.get_tensor_by_name('import/logits:0'),
        generate_session(graph = g),
        dictionary,
    )


def bert(validate = True):
    """
    Load BERT similarity model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    SIMILARITY_BERT : malaya._models._tensorflow_model.SIAMESE_BERT class
    """
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')
    try:
        from bert import tokenization
    except:
        raise Exception(
            'bert-tensorflow not installed. Please install it using `pip3 install bert-tensorflow` and try again.'
        )
    if validate:
        check_file(PATH_SIMILARITY['bert'], S3_PATH_SIMILARITY['bert'])
    else:
        if not check_available(PATH_SIMILARITY['bert']):
            raise Exception(
                'toxic/bert is not available, please `validate = True`'
            )

    tokenization.validate_case_matches_checkpoint(True, '')
    tokenizer = tokenization.FullTokenizer(
        vocab_file = PATH_SIMILARITY['bert']['vocab'], do_lower_case = True
    )
    try:
        g = load_graph(PATH_SIMILARITY['bert']['model'])
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('similarity/bert') and try again"
        )

    return SIAMESE_BERT(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        segment_ids = g.get_tensor_by_name('import/Placeholder_1:0'),
        input_masks = g.get_tensor_by_name('import/Placeholder_2:0'),
        logits = g.get_tensor_by_name('import/logits:0'),
        sess = generate_session(graph = g),
        tokenizer = tokenizer,
        maxlen = 100,
        label = ['not similar', 'similar'],
    )
