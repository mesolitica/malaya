import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import re
import os
import random
import numpy as np
from fuzzywuzzy import fuzz
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from .texts.vectorizer import SkipGramVectorizer
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)
from sklearn.utils import shuffle
from .texts._text_functions import (
    STOPWORDS,
    STOPWORD_CALON,
    simple_textcleaning,
    str_idx,
)
from .generator import sentence_ngram
from . import home, _delete_macos
from ._utils._utils import (
    check_file,
    load_graph,
    check_available,
    generate_session,
)
from ._models._skip_thought import (
    train_model as skip_train,
    batch_sequence,
    load_skipthought as load_deep_skipthought,
)
from ._models._siamese_lstm import (
    train_model as siamese_train,
    load_siamese as load_deep_siamese,
)
from .preprocessing import _tokenizer
from .topic import calon
from .topic import location
from ._utils._paths import PATH_SIMILARITY, S3_PATH_SIMILARITY
from ._models._tensorflow_model import SIAMESE, SIAMESE_BERT


def _apply_stopwords_calon(string):
    string = re.sub('[^A-Za-z ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string.lower()).strip()
    return ' '.join(
        [i for i in string.split() if i not in STOPWORD_CALON and len(i) > 1]
    )


_namacalon = [
    _apply_stopwords_calon(val['NamaCalon'].lower()) for _, val in calon.items()
]
_namacalon = list(set(_namacalon))
_negeri = list(set([val['negeri'].lower() for _, val in location.items()]))
_parlimen = list(set([val['parlimen'].lower() for _, val in location.items()]))
_dun = list(set([val['dun'].lower() for _, val in location.items()]))
_location = list(set(_negeri + _parlimen + _dun))


class _DEEP_SIAMESE_SIMILARITY:
    def __init__(
        self,
        sess,
        model,
        keys,
        dictionary,
        maxlen,
        saver,
        embedding_size,
        num_layers,
        output_size,
        dropout,
        is_influencers = False,
    ):
        self._sess = sess
        self._model = model
        self.keys = keys
        self.dictionary = dictionary
        self.maxlen = maxlen
        self._saver = saver
        self._embedding_size = embedding_size
        self._num_layers = num_layers
        self._output_size = output_size
        self._dropout = dropout

    def save_model(self, location):
        """
        save model to tensorflow checkpoint file.

        Parameters
        ----------
        string: str

        """
        if not isinstance(location, str):
            raise ValueError('location must be a string')
        self._saver.save(self._sess, '%s/model.ckpt' % (location))
        with open('%s/model.json' % (location), 'w') as fopen:
            json.dump(
                {
                    'dictionary': self.dictionary,
                    'maxlen': self.maxlen,
                    'keys': self.keys,
                    'embedding_size': self._embedding_size,
                    'num_layers': self._num_layers,
                    'output_size': self._output_size,
                    'dropout': self._dropout,
                },
                fopen,
            )

    def get_similarity(self, string, anchor = 0.5):
        """
        Return similar topics / influencers.

        Parameters
        ----------
        string: str
        anchor: float, (default=0.5)
            baseline similarity.

        Returns
        -------
        results: list of strings
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(anchor, float):
            raise ValueError('anchor must be a float')
        if not (anchor > 0 and anchor < 1):
            raise ValueError('anchor must be bigger than 0, less than 1')
        original_string = simple_textcleaning(string)
        strings = [original_string] * len(self.keys)
        left = str_idx(strings, self.dictionary, self.maxlen, UNK = 3)
        right = str_idx(self.keys, self.dictionary, self.maxlen, UNK = 3)
        distances = self._sess.run(
            1 - self._model.distance,
            feed_dict = {self._model.X_left: left, self._model.X_right: right},
        )
        where = np.where(distances > anchor)[0]
        results = [self.keys[i].lower() for i in where]
        return list(set(results))


class _DEEP_SIMILARITY:
    def __init__(
        self,
        sess,
        model,
        vectorized,
        keys,
        dictionary,
        maxlen,
        saver,
        embedding_size,
        is_influencers = False,
    ):
        self._sess = sess
        self._model = model
        self.vectorized = vectorized
        self.keys = keys
        self.dictionary = dictionary
        self.maxlen = maxlen
        self._saver = saver
        self._embedding_size = embedding_size

    def save_model(self, location):
        """
        save model to tensorflow checkpoint file.

        Parameters
        ----------
        string: str

        """
        if not isinstance(location, str):
            raise ValueError('location must be a string')
        self._saver.save(self._sess, '%s/model.ckpt' % (location))
        with open('%s/model.json' % (location), 'w') as fopen:
            json.dump(
                {
                    'dictionary': self.dictionary,
                    'maxlen': self.maxlen,
                    'keys': self.keys,
                    'vectorized': self.vectorized.tolist(),
                    'embedding_size': self._embedding_size,
                },
                fopen,
            )

    def get_similarity(self, string, anchor = 0.5):
        """
        Return similar topics / influencers.

        Parameters
        ----------
        string: str
        anchor: float, (default=0.5)
            baseline similarity.

        Returns
        -------
        results: list of strings
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(anchor, float):
            raise ValueError('anchor must be a float')
        if not (anchor > 0 and anchor < 1):
            raise ValueError('anchor must be bigger than 0, less than 1')
        original_string = simple_textcleaning(string)
        string = ' '.join(set(original_string.split()))
        encoded = self._sess.run(
            self._model.get_thought,
            feed_dict = {
                self._model.INPUT: batch_sequence(
                    [string], self.dictionary, maxlen = self.maxlen
                )
            },
        )
        where = np.where(
            cosine_similarity(self.vectorized, encoded)[:, 0] > anchor
        )[0]
        results = [self.keys[i].lower() for i in where]
        return list(set(results))


class _FAST_SIMILARITY:
    def __init__(self, vectorizer, vectorized, keys):
        self.vectorizer = vectorizer
        self.vectorized = vectorized
        self.keys = keys

    def get_similarity(self, string, anchor = 0.1):
        """
        Return similar topics / influencers.

        Parameters
        ----------
        string: str
        anchor: float, (default=0.5)
            baseline similarity.

        Returns
        -------
        results: list of strings
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(anchor, float):
            raise ValueError('anchor must be a float')
        if not (anchor > 0 and anchor < 1):
            raise ValueError('anchor must be bigger than 0, less than 1')
        original_string = simple_textcleaning(string)
        string = ' '.join(set(original_string.split()))
        where = np.where(
            cosine_similarity(
                self.vectorized, self.vectorizer.transform([string])
            )[:, 0]
            > anchor
        )[0]
        results = [self.keys[i].lower() for i in where]
        return list(set(results))


class _FUZZY:
    def __init__(self, dictionary):
        self._dictionary = dictionary

    def get_similarity(self, string, fuzzy_ratio = 90):

        """
        check whether a string is a topic based on dictionary given.

        Parameters
        ----------
        string: str
        fuzzy_ratio: int, (default=90)
            ratio of similar characters by positions, if 90, means 90%.

        Returns
        -------
        list: list of topics
        """

        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(fuzzy_ratio, int):
            raise ValueError('fuzzy_ratio must be an integer')
        if not (fuzzy_ratio > 0 and fuzzy_ratio < 100):
            raise ValueError(
                'fuzzy_ratio must be bigger than 0 and less than 100'
            )
        string = string.lower()
        topics = []
        for key, vals in self._dictionary.items():
            for v in vals:
                if fuzz.token_set_ratio(v, string) >= fuzzy_ratio:
                    topics.append(key.lower())
                    break
        return list(set(topics))


def fuzzy(dictionary):
    """
    Return similar topics.

    Parameters
    ----------
    dictionary: dict
        format {'left':['right']}

    Returns
    -------
    results: _FUZZY object
    """
    if not isinstance(dictionary, dict):
        raise ValueError('dictionary must be a dictionary')
    if not isinstance(list(dictionary.keys())[0], str):
        raise ValueError('keys dictionary must be a string')
    if not isinstance(list(dictionary.values())[0], list):
        raise ValueError('values dictionary must be a list')
    if not isinstance(list(dictionary.values())[0][0], str):
        raise ValueError('elements of values dictionary must be a string')
    return _FUZZY(dictionary)


def is_location(string, fuzzy_ratio = 90, location = None):
    """
    check whether a string is a location, default is malaysia location.

    Parameters
    ----------
    string: str
    fuzzy_ratio: int, (default=90)
        ratio of similar characters by positions, if 90, means 90%.
    location: list, (default=None)
        list of locations. If None, will use malaya.similarity._location

    Returns
    -------
    boolean: bool
    """
    if not isinstance(string, str):
        raise ValueError('input must be a string')
    if not isinstance(fuzzy_ratio, int):
        raise ValueError('fuzzy_ratio must be an integer')
    if location is None:
        location = _location
    if not (fuzzy_ratio > 0 and fuzzy_ratio < 100):
        raise ValueError('fuzzy_ratio must be bigger than 0 and less than 100')
    for loc in location:
        if fuzz.token_set_ratio(loc.lower(), string) >= fuzzy_ratio:
            return True
    return False


def fuzzy_location(string, fuzzy_ratio = 90):
    """
    Return malaysia locations splitted by dun, parlimen and state from a string.

    Parameters
    ----------
    string: str
    fuzzy_ratio: int, (default=90)
        ratio of similar characters by positions, if 90, means 90%.

    Returns
    -------
    results: list of strings
    """
    if not isinstance(string, str):
        raise ValueError('input must be a string')
    if not isinstance(fuzzy_ratio, int):
        raise ValueError('fuzzy_ratio must be an integer')
    if not (fuzzy_ratio > 0 and fuzzy_ratio < 100):
        raise ValueError('fuzzy_ratio must be bigger than 0 and less than 100')
    negeri_list = list(
        set(
            [
                i
                for i in _negeri
                if fuzz.token_set_ratio(i, string) >= fuzzy_ratio
            ]
        )
    )
    parlimen_list = list(
        set(
            [
                i
                for i in _parlimen
                if fuzz.token_set_ratio(i, string) >= fuzzy_ratio
            ]
        )
    )
    dun_list = list(
        set([i for i in _dun if fuzz.token_set_ratio(i, string) >= fuzzy_ratio])
    )
    return {'negeri': negeri_list, 'parlimen': parlimen_list, 'dun': dun_list}


def _generate_topics(dictionary):
    texts = [' '.join(words) for _, words in dictionary.items()]
    keys = [key for key, _ in dictionary.items()]
    texts = [' '.join(list(set(text.split()))) for text in texts]
    output = []
    for text in texts:
        output.append(
            ' '.join([word for word in text.split() if word not in STOPWORDS])
        )
    return output, keys


def deep_siamese(
    dictionary,
    epoch = 5,
    batch_size = 32,
    embedding_size = 256,
    output_size = 100,
    maxlen = 100,
    ngram = (1, 4),
    num_layers = 1,
    **kwargs
):
    """
    Train a deep siamese network for text similarity

    Parameters
    ----------
    dictionary: dict
        format {'left':['right']}
    epoch: int, (default=5)
        iteration numbers.
    batch_size: int, (default=32)
        batch size for every feed, batch size must <= size of corpus.
    embedding_size: int, (default=256)
        vector size representation for a word.
    output_size: int, (default=100)
        encoder output size, bigger means more vector definition.
    maxlen: int, (default=100)
        max length of a string to be train.
    ngram: tuple, (default=(1,4))
        n-grams size to train a corpus.
    num_layers: int, (default=100)
        number of bidirectional rnn layers.

    Returns
    -------
    _DEEP_SIAMESE_SIMILARITY: malaya.similarity._DEEP_SIAMESE_SIMILARITY class
    """
    if not isinstance(dictionary, dict):
        raise ValueError('dictionary must be a dictionary')
    if not isinstance(list(dictionary.keys())[0], str):
        raise ValueError('keys dictionary must be a string')
    if not isinstance(list(dictionary.values())[0], list):
        raise ValueError('values dictionary must be a list')
    if not isinstance(list(dictionary.values())[0][0], str):
        raise ValueError('elements of values dictionary must be a string')
    if not isinstance(epoch, int):
        raise ValueError('epoch must be an integer')
    if not isinstance(batch_size, int):
        raise ValueError('batch_size must be an integer')
    if not isinstance(embedding_size, int):
        raise ValueError('embedding_size must be an integer')
    if not isinstance(output_size, int):
        raise ValueError('output_size must be an integer')
    if not isinstance(maxlen, int):
        raise ValueError('maxlen must be an integer')
    if not isinstance(ngram, tuple):
        raise ValueError('ngram must be a tuple')
    if not isinstance(num_layers, int):
        raise ValueError('num_layers must be an integer')
    if not len(ngram) == 2:
        raise ValueError('ngram size must equal to 2')
    output, keys = _generate_topics(dictionary)
    batch_x_left, batch_x_right, batch_y = [], [], []
    for i in range(len(output)):
        augmentation = sentence_ngram(output[i])
        batch_x_right.extend([keys[i]] * len(augmentation))
        batch_x_left.extend(augmentation)
        batch_y.extend([1] * len(augmentation))
    neg_batch_x_left, neg_batch_x_right = [], []
    for i in range(len(batch_x_left)):
        random_num = random.randint(0, len(batch_x_left) - 1)
        while random_num == i:
            random_num = random.randint(0, len(batch_x_left) - 1)
        neg_batch_x_right.append(batch_x_right[random_num])
        neg_batch_x_left.append(batch_x_left[i])
        batch_y.append(0)
    batch_x_left.extend(neg_batch_x_left)
    batch_x_right.extend(neg_batch_x_right)
    batch_x_left, batch_x_right, batch_y = shuffle(
        batch_x_left, batch_x_right, batch_y
    )
    sess, model, dictionary, saver, dropout = siamese_train(
        batch_x_left,
        batch_x_right,
        batch_y,
        epoch = epoch,
        batch_size = batch_size,
        embedding_size = embedding_size,
        output_size = output_size,
        maxlen = maxlen,
        num_layers = num_layers,
        **kwargs
    )
    return _DEEP_SIAMESE_SIMILARITY(
        sess,
        model,
        keys,
        dictionary,
        maxlen,
        saver,
        embedding_size,
        num_layers,
        output_size,
        dropout,
    )


def load_siamese(location):
    if not os.path.isfile('%s/model.json' % (location)) or not os.path.isfile(
        '%s/model.ckpt.meta' % (location)
    ):
        raise Exception(
            'siamese text similarity model at %s/ is not available, please train and save the model'
            % (location)
        )
    with open('%s/model.json' % (location)) as fopen:
        json_ = json.load(fopen)
    sess, model, saver = load_deep_siamese(location, json_)
    return _DEEP_SIAMESE_SIMILARITY(
        sess,
        model,
        json_['keys'],
        json_['dictionary'],
        json_['maxlen'],
        saver,
        json_['embedding_size'],
        json_['num_layers'],
        json_['output_size'],
        json_['dropout'],
    )


def deep_skipthought(
    dictionary,
    epoch = 5,
    batch_size = 16,
    embedding_size = 256,
    maxlen = 100,
    ngram = (1, 4),
):
    """
    Train a deep skip-thought network for text similarity

    Parameters
    ----------
    dictionary: dict
        format {'left':['right']}
    epoch: int, (default=5)
        iteration numbers.
    batch_size: int, (default=32)
        batch size for every feed, batch size must <= size of corpus.
    embedding_size: int, (default=256)
        vector size representation for a word.
    maxlen: int, (default=100)
        max length of a string to be train.
    ngram: tuple, (default=(1,4))
        n-grams size to train a corpus.

    Returns
    -------
    _DEEP_SIMILARITY: malaya.similarity._DEEP_SIMILARITY class
    """
    if not isinstance(dictionary, dict):
        raise ValueError('dictionary must be a dictionary')
    if not isinstance(list(dictionary.keys())[0], str):
        raise ValueError('keys dictionary must be a string')
    if not isinstance(list(dictionary.values())[0], list):
        raise ValueError('values dictionary must be a list')
    if not isinstance(list(dictionary.values())[0][0], str):
        raise ValueError('elements of values dictionary must be a string')
    if not isinstance(epoch, int):
        raise ValueError('epoch must be an integer')
    if not isinstance(batch_size, int):
        raise ValueError('batch_size must be an integer')
    if not isinstance(embedding_size, int):
        raise ValueError('embedding_size must be an integer')
    if not isinstance(maxlen, int):
        raise ValueError('maxlen must be an integer')
    if not isinstance(ngram, tuple):
        raise ValueError('ngram must be a tuple')
    if not len(ngram) == 2:
        raise ValueError('ngram size must equal to 2')
    output, keys = _generate_topics(dictionary)
    batch_x, batch_y = [], []
    for i in range(len(output)):
        augmentation = sentence_ngram(output[i])
        batch_y.extend([keys[i]] * len(augmentation))
        batch_x.extend(augmentation)
    batch_x, batch_y = shuffle(batch_x, batch_y)
    sess, model, dictionary, saver = skip_train(
        batch_x,
        batch_y,
        batch_y,
        epoch = epoch,
        batch_size = batch_size,
        embedding_size = embedding_size,
        maxlen = maxlen,
    )
    vectorized = sess.run(
        model.get_thought,
        feed_dict = {
            model.INPUT: batch_sequence(output, dictionary, maxlen = maxlen)
        },
    )
    return _DEEP_SIMILARITY(
        sess, model, vectorized, keys, dictionary, maxlen, saver, embedding_size
    )


def load_skipthought(location):
    if not os.path.isfile('%s/model.json' % (location)) or not os.path.isfile(
        '%s/model.ckpt.meta' % (location)
    ):
        raise Exception(
            'skipthought text similarity model at %s/ is not available, please train and save the model'
        )
    with open('%s/model.json' % (location)) as fopen:
        json_ = json.load(fopen)
    sess, model, saver = load_deep_skipthought(location, json_)
    return _DEEP_SIMILARITY(
        sess,
        model,
        np.array(json_['vectorized']),
        json_['keys'],
        json_['dictionary'],
        json_['maxlen'],
        saver,
        json_['embedding_size'],
    )


def bow(dictionary, vectorizer = 'tfidf', ngram = (3, 10)):
    """
    Train a bow for text similarity

    Parameters
    ----------
    dictionary: dict
        format {'left':['right']}
    vectorizer: str, (default='tfidf')
        vectorization technique for a corpus.
    ngram: tuple, (default=(3,10))
        n-grams size to train a corpus.

    Returns
    -------
    _FAST_SIMILARITY: malaya.similarity._FAST_SIMILARITY class
    """
    if not isinstance(dictionary, dict):
        raise ValueError('dictionary must be a dictionary')
    if not isinstance(list(dictionary.keys())[0], str):
        raise ValueError('keys dictionary must be a string')
    if not isinstance(list(dictionary.values())[0], list):
        raise ValueError('values dictionary must be a list')
    if not isinstance(list(dictionary.values())[0][0], str):
        raise ValueError('elements of values dictionary must be a string')
    if not isinstance(vectorizer, str):
        raise ValueError('vectorizer must be a string')
    if not isinstance(ngram, tuple):
        raise ValueError('ngram must be a tuple')
    if not len(ngram) == 2:
        raise ValueError('ngram size must equal to 2')
    if 'tfidf' in vectorizer.lower():
        char_vectorizer = TfidfVectorizer(
            sublinear_tf = True,
            strip_accents = 'unicode',
            analyzer = 'char',
            ngram_range = ngram,
        )
    elif 'count' in vectorizer.lower():
        char_vectorizer = CountVectorizer(
            strip_accents = 'unicode', analyzer = 'char', ngram_range = ngram
        )
    elif 'skip-gram' in vectorizer.lower():
        char_vectorizer = SkipGramVectorizer(
            strip_accents = 'unicode', analyzer = 'char', ngram_range = ngram
        )
    else:
        raise Exception('model not supported')
    output, keys = _generate_topics(dictionary)
    vectorizer = char_vectorizer.fit(output)
    vectorized = vectorizer.transform(output)
    return _FAST_SIMILARITY(vectorizer, vectorized, keys)


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


def summarizer(summarizer, left_string, right_string, similarity = 'cosine'):
    """
    Calculate similarity between 2 documents using summarizer.

    Parameters
    ----------
    vectorizer : object
        fast-text or word2vec interface object.
    left_string: str
        first string to predict.
    right_string: str
        second string to predict.
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

    if not hasattr(summarizer, 'vectorize'):
        raise ValueError('summarizer must has `vectorize` method')
    if not isinstance(left_string, str):
        raise ValueError('left_string must be a string')
    if not isinstance(right_string, str):
        raise ValueError('right_string must be a string')
    if not isinstance(similarity, str):
        raise ValueError('similarity must be a string')

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

    left_vectorized = summarizer.vectorize(left_string + '.')
    right_vectorized = summarizer.vectorize(right_string + '.')
    similar = similarity_function(left_vectorized, right_vectorized)[0, 0]
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
