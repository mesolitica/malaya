import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import re
import os
import random
import numpy as np
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle
from .texts._text_functions import (
    STOPWORDS,
    STOPWORD_CALON,
    simple_textcleaning,
    str_idx,
)
from .cluster import sentence_ngram
from . import home, _delete_macos
from ._utils._utils import download_file
from ._models._skip_thought import train_model as skip_train, batch_sequence
from ._models._siamese_lstm import train_model as siamese_train

_zip_location = home + '/rules-based.zip'
_namacalon = None
_parlimen = None
_dun = None
_negeri = None
_location = None
_person_dict = None
_topic_dict = None
_short_dict = None


def _apply_stopwords_calon(string):
    string = re.sub('[^A-Za-z ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string.lower()).strip()
    return ' '.join(
        [i for i in string.split() if i not in STOPWORD_CALON and len(i) > 1]
    )


class _DEEP_SIAMESE_SIMILARITY:
    def __init__(
        self, sess, model, keys, dictionary, maxlen, is_influencers = False
    ):
        self._sess = sess
        self._model = model
        self.keys = keys
        self.dictionary = dictionary
        self.maxlen = maxlen
        self._is_influencers = is_influencers

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
        assert isinstance(string, str), 'input must be a string'
        assert isinstance(anchor, float), 'anchor must be a float'
        assert (
            anchor > 0 and anchor < 1
        ), 'anchor must be bigger than 0, less than 1'
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
        for key, vals in _short_dict.items():
            for v in vals:
                if v in original_string.split():
                    results.append(key.lower())
                    break
        if self._is_influencers:
            for index in np.where(
                np.array(
                    [
                        fuzz.token_set_ratio(i, original_string)
                        for i in _namacalon
                    ]
                )
                >= 80
            )[0]:
                results.append(_namacalon[index].lower())
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
        is_influencers = False,
    ):
        self._sess = sess
        self._model = model
        self.vectorized = vectorized
        self.keys = keys
        self.dictionary = dictionary
        self.maxlen = maxlen
        self._is_influencers = is_influencers

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
        assert isinstance(string, str), 'input must be a string'
        assert isinstance(anchor, float), 'anchor must be a float'
        assert (
            anchor > 0 and anchor < 1
        ), 'anchor must be bigger than 0, less than 1'
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
        for key, vals in _short_dict.items():
            for v in vals:
                if v in original_string.split():
                    results.append(key.lower())
                    break
        if self._is_influencers:
            for index in np.where(
                np.array(
                    [
                        fuzz.token_set_ratio(i, original_string)
                        for i in _namacalon
                    ]
                )
                >= 80
            )[0]:
                results.append(_namacalon[index].lower())
        return list(set(results))


class _FAST_SIMILARITY:
    def __init__(self, vectorizer, vectorized, keys, is_influencers = False):
        self.vectorizer = vectorizer
        self.vectorized = vectorized
        self.keys = keys
        self._is_influencers = is_influencers

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
        assert isinstance(string, str), 'input must be a string'
        assert isinstance(anchor, float), 'anchor must be a float'
        assert (
            anchor > 0 and anchor < 1
        ), 'anchor must be bigger than 0, less than 1'
        original_string = simple_textcleaning(string)
        string = ' '.join(set(original_string.split()))
        where = np.where(
            cosine_similarity(
                self.vectorized, self.vectorizer.transform([string])
            )[:, 0]
            > anchor
        )[0]
        results = [self.keys[i].lower() for i in where]
        for key, vals in _short_dict.items():
            for v in vals:
                if v in original_string.split():
                    results.append(key.lower())
                    break
        if self._is_influencers:
            for index in np.where(
                np.array(
                    [
                        fuzz.token_set_ratio(i, original_string)
                        for i in _namacalon
                    ]
                )
                >= 80
            )[0]:
                results.append(_namacalon[index].lower())
        return list(set(results))


def _load_internal_data():
    import pandas as pd
    import zipfile

    global _namacalon, _parlimen, _dun, _negeri, _location, _person_dict, _topic_dict, _short_dict
    if not os.path.isfile(_zip_location):
        print('downloading Topics, Influencers, location data')
        download_file('rules-based.zip', _zip_location)

    if not os.path.exists(home + '/rules-based/calon.csv'):
        with zipfile.ZipFile(_zip_location, 'r') as zip_ref:
            zip_ref.extractall(home)

    _delete_macos()
    df = pd.read_csv(home + '/rules-based/calon.csv')
    _namacalon = df.NamaCalon.str.lower().unique().tolist()
    for i in range(len(_namacalon)):
        _namacalon[i] = _apply_stopwords_calon(_namacalon[i])
    _namacalon = list(set(_namacalon))

    df = pd.read_csv(home + '/rules-based/negeri.csv')
    _negeri = df.negeri.str.lower().unique().tolist()
    _parlimen = df.parlimen.str.lower().unique().tolist()
    _dun = df.dun.str.lower().unique().tolist()[:-1]

    _location = _negeri + _parlimen + _dun

    with open(home + '/rules-based/person-normalized', 'r') as fopen:
        person = list(filter(None, fopen.read().split('\n')))

    _person_dict = {}
    for i in range(len(person)):
        splitted = person[i].split(':')
        uniques = list(
            filter(
                None,
                (
                    set(
                        [k.strip().lower() for k in splitted[1].split(', ')]
                        + [splitted[0].lower()]
                    )
                ),
            )
        )
        _person_dict[splitted[0]] = uniques

    with open(home + '/rules-based/topic-normalized', 'r') as fopen:
        topic = list(filter(None, fopen.read().split('\n')))

    _topic_dict = {}
    for i in range(len(topic)):
        splitted = topic[i].split(':')
        uniques = list(
            filter(
                None,
                (
                    set(
                        [k.strip().lower() for k in splitted[1].split(', ')]
                        + [splitted[0].lower()]
                    )
                ),
            )
        )
        _topic_dict[splitted[0]] = uniques

    with open(home + '/rules-based/short-normalized', 'r') as fopen:
        short = list(filter(None, fopen.read().split('\n')))

    _short_dict = {}
    for i in range(len(short)):
        splitted = short[i].split(':')
        uniques = list(
            filter(
                None,
                (
                    set(
                        [k.strip().lower() for k in splitted[1].split(', ')]
                        + [splitted[0].lower()]
                    )
                ),
            )
        )
        _short_dict[splitted[0]] = uniques


def fuzzy_influencer(string):
    """
    Return similar influencers.

    Parameters
    ----------
    string: str

    Returns
    -------
    results: list of strings
    """
    if not _person_dict:
        _load_internal_data()
    assert isinstance(string, str), 'input must be a string'
    string = string.lower()
    influencers = []
    for key, vals in _person_dict.items():
        for v in vals:
            if fuzz.token_set_ratio(v, string) >= 80:
                influencers.append(key.lower())
                break
    for key, vals in _short_dict.items():
        for v in vals:
            if v in string.split():
                influencers.append(key.lower())
                break

    for index in np.where(
        np.array([fuzz.token_set_ratio(i, string) for i in _namacalon]) >= 80
    )[0]:
        influencers.append(_namacalon[index].lower())
    return list(set(influencers))


def fuzzy_topic(string):
    """
    Return similar topics.

    Parameters
    ----------
    string: str

    Returns
    -------
    results: list of strings
    """
    if not _person_dict:
        _load_internal_data()
    assert isinstance(string, str), 'input must be a string'
    string = string.lower()
    topics = []
    for key, vals in _topic_dict.items():
        for v in vals:
            if fuzz.token_set_ratio(v, string) >= 80:
                topics.append(key.lower())
                break
    for key, vals in _person_dict.items():
        for v in vals:
            if fuzz.token_set_ratio(v, string) >= 80:
                topics.append(key.lower())
                break
    for key, vals in _short_dict.items():
        for v in vals:
            if v in string.split():
                topics.append(key.lower())
                break

    return list(set(topics))


def is_location(string):
    """
    check whether a string is a malaysia location.

    Parameters
    ----------
    string: str

    Returns
    -------
    boolean: bool
    """
    if not _person_dict:
        _load_internal_data()
    for loc in _location:
        if fuzz.token_set_ratio(loc.lower(), string) >= 90:
            return True
    return False


def fuzzy_location(string):
    """
    Return similar _location.

    Parameters
    ----------
    string: str

    Returns
    -------
    results: list of strings
    """
    if not _person_dict:
        _load_internal_data()
    assert isinstance(string, str), 'input must be a string'
    negeri_list = list(
        set([i for i in _negeri if fuzz.token_set_ratio(i, string) >= 80])
    )
    parlimen_list = list(
        set([i for i in _parlimen if fuzz.token_set_ratio(i, string) >= 80])
    )
    dun_list = list(
        set([i for i in _dun if fuzz.token_set_ratio(i, string) >= 80])
    )
    return {'negeri': negeri_list, 'parlimen': parlimen_list, 'dun': dun_list}


def _generate_topics():
    if not _person_dict:
        _load_internal_data()
    texts = [' '.join(words) for _, words in _topic_dict.items()]
    keys = [key for key, _ in _topic_dict.items()]
    texts += [' '.join(words) for _, words in _person_dict.items()]
    keys += [key for key, _ in _person_dict.items()]
    texts = [' '.join(list(set(text.split()))) for text in texts]
    output = []
    for text in texts:
        output.append(
            ' '.join([word for word in text.split() if word not in STOPWORDS])
        )
    return output, keys


def _generate_influencers():
    if not _person_dict:
        _load_internal_data()
    texts = [' '.join(words) for _, words in _person_dict.items()]
    keys = [key for key, _ in _person_dict.items()]
    texts = [' '.join(list(set(text.split()))) for text in texts]
    output = []
    for text in texts:
        output.append(
            ' '.join([word for word in text.split() if word not in STOPWORDS])
        )
    return output, keys


def siamese_topic(
    epoch = 5,
    batch_size = 32,
    embedding_size = 256,
    output_size = 100,
    maxlen = 100,
    ngram = (1, 4),
):
    """
    Train a deep siamese network for topics similarity

    Parameters
    ----------
    epoch: int, (default=5)
        iteration numbers
    batch_size: int, (default=32)
        batch size for every feed, batch size must <= size of corpus
    embedding_size: int, (default=256)
        vector size representation for a word
    output_size: int, (default=100)
        encoder output size, bigger means more vector definition
    maxlen: int, (default=100)
        max length of a string to be train
    ngram: tuple, (default=(1,4))
        n-grams size to train a corpus

    Returns
    -------
    _DEEP_SIAMESE_SIMILARITY: malaya.topics_influencers._DEEP_SIAMESE_SIMILARITY class
    """
    if not _person_dict:
        _load_internal_data()
    assert isinstance(epoch, int), 'epoch must be an integer'
    assert isinstance(batch_size, int), 'batch_size must be an integer'
    assert isinstance(embedding_size, int), 'embedding_size must be an integer'
    assert isinstance(output_size, int), 'output_size must be an integer'
    assert isinstance(maxlen, int), 'maxlen must be an integer'
    assert isinstance(ngram, tuple), 'ngram must be a tuple'
    assert len(ngram) == 2, 'ngram size must equal to 2'
    output, keys = _generate_topics()
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
    sess, model, dictionary = siamese_train(
        batch_x_left,
        batch_x_right,
        batch_y,
        epoch = epoch,
        batch_size = batch_size,
        embedding_size = embedding_size,
        output_size = output_size,
        maxlen = maxlen,
    )
    return _DEEP_SIAMESE_SIMILARITY(sess, model, keys, dictionary, maxlen)


def siamese_influencer(
    epoch = 5,
    batch_size = 32,
    embedding_size = 256,
    output_size = 100,
    maxlen = 100,
    ngram = (1, 4),
):
    """
    Train a deep siamese network for influencers similarity

    Parameters
    ----------
    epoch: int, (default=5)
        iteration numbers
    batch_size: int, (default=32)
        batch size for every feed, batch size must <= size of corpus
    embedding_size: int, (default=256)
        vector size representation for a word
    output_size: int, (default=100)
        encoder output size, bigger means more vector definition
    maxlen: int, (default=100)
        max length of a string to be train
    ngram: tuple, (default=(1,4))
        n-grams size to train a corpus

    Returns
    -------
    _DEEP_SIAMESE_SIMILARITY: malaya.topic_influencers._DEEP_SIAMESE_SIMILARITY class
    """
    if not _person_dict:
        _load_internal_data()
    assert isinstance(epoch, int), 'epoch must be an integer'
    assert isinstance(batch_size, int), 'batch_size must be an integer'
    assert isinstance(embedding_size, int), 'embedding_size must be an integer'
    assert isinstance(output_size, int), 'output_size must be an integer'
    assert isinstance(maxlen, int), 'maxlen must be an integer'
    assert isinstance(ngram, tuple), 'ngram must be a tuple'
    assert len(ngram) == 2, 'ngram size must equal to 2'
    output, keys = _generate_influencers()
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
    sess, model, dictionary = siamese_train(
        batch_x_left,
        batch_x_right,
        batch_y,
        epoch = epoch,
        batch_size = batch_size,
        embedding_size = embedding_size,
        output_size = output_size,
        maxlen = maxlen,
    )
    return _DEEP_SIAMESE_SIMILARITY(
        sess, model, keys, dictionary, maxlen, is_influencers = True
    )


def skipthought_topic(
    epoch = 5,
    batch_size = 16,
    embedding_size = 256,
    maxlen = 100,
    ngram = (1, 4),
):
    """
    Train a deep skip-thought network for topics similarity

    Parameters
    ----------
    epoch: int, (default=5)
        iteration numbers
    batch_size: int, (default=32)
        batch size for every feed, batch size must <= size of corpus
    embedding_size: int, (default=256)
        vector size representation for a word
    maxlen: int, (default=100)
        max length of a string to be train
    ngram: tuple, (default=(1,4))
        n-grams size to train a corpus

    Returns
    -------
    _DEEP_SIMILARITY: malaya.topic_influencers._DEEP_SIMILARITY class
    """
    if not _person_dict:
        _load_internal_data()
    assert isinstance(epoch, int), 'epoch must be an integer'
    assert isinstance(batch_size, int), 'batch_size must be an integer'
    assert isinstance(embedding_size, int), 'embedding_size must be an integer'
    assert isinstance(maxlen, int), 'maxlen must be an integer'
    assert isinstance(ngram, tuple), 'ngram must be a tuple'
    assert len(ngram) == 2, 'ngram size must equal to 2'
    output, keys = _generate_topics()
    batch_x, batch_y = [], []
    for i in range(len(output)):
        augmentation = sentence_ngram(output[i])
        batch_y.extend([keys[i]] * len(augmentation))
        batch_x.extend(augmentation)
    batch_x, batch_y = shuffle(batch_x, batch_y)
    sess, model, dictionary = skip_train(
        batch_x,
        batch_y,
        batch_y,
        epoch = epoch,
        batch_size = batch_size,
        embedding_size = embedding_size,
        maxlen = maxlen,
    )
    encoded = sess.run(
        model.get_thought,
        feed_dict = {
            model.INPUT: batch_sequence(output, dictionary, maxlen = maxlen)
        },
    )
    return _DEEP_SIMILARITY(sess, model, encoded, keys, dictionary, maxlen)


def skipthought_influencer(
    epoch = 10,
    batch_size = 16,
    embedding_size = 256,
    maxlen = 100,
    ngram = (1, 4),
):
    """
    Train a deep skip-thought network for influencers similarity

    Parameters
    ----------
    epoch: int, (default=5)
        iteration numbers
    batch_size: int, (default=32)
        batch size for every feed, batch size must <= size of corpus
    embedding_size: int, (default=256)
        vector size representation for a word
    maxlen: int, (default=100)
        max length of a string to be train
    ngram: tuple, (default=(1,4))
        n-grams size to train a corpus

    Returns
    -------
    _DEEP_SIMILARITY: malaya.topics_influencers._DEEP_SIMILARITY class
    """
    if not _person_dict:
        _load_internal_data()
    assert isinstance(epoch, int), 'epoch must be an integer'
    assert isinstance(batch_size, int), 'batch_size must be an integer'
    assert isinstance(embedding_size, int), 'embedding_size must be an integer'
    assert isinstance(maxlen, int), 'maxlen must be an integer'
    assert isinstance(ngram, tuple), 'ngram must be a tuple'
    assert len(ngram) == 2, 'ngram size must equal to 2'
    output, keys = _generate_influencers()
    batch_x, batch_y = [], []
    for i in range(len(output)):
        augmentation = sentence_ngram(output[i])
        batch_y.extend([keys[i]] * len(augmentation))
        batch_x.extend(augmentation)
    assert batch_size < len(batch_x), 'batch size must smaller with corpus size'
    batch_x, batch_y = shuffle(batch_x, batch_y)
    sess, model, dictionary = skip_train(
        batch_x,
        batch_y,
        batch_y,
        epoch = epoch,
        batch_size = batch_size,
        embedding_size = embedding_size,
        maxlen = maxlen,
    )
    encoded = sess.run(
        model.get_thought,
        feed_dict = {
            model.INPUT: batch_sequence(output, dictionary, maxlen = maxlen)
        },
    )
    return _DEEP_SIMILARITY(
        sess, model, encoded, keys, dictionary, maxlen, is_influencers = True
    )


def fast_topic(vectorizer = 'tfidf', ngram = (3, 10)):
    """
    Train a deep siamese network for topics similarity

    Parameters
    ----------
    vectorizer: str, (default='tfidf')
        vectorization technique for a corpus
    ngram: tuple, (default=(1,4))
        n-grams size to train a corpus

    Returns
    -------
    _FAST_SIMILARITY: malaya.siamese_lstm._FAST_SIMILARITY class
    """
    if not _person_dict:
        _load_internal_data()
    assert isinstance(vectorizer, str), 'vectorizer must be a string'
    assert isinstance(ngram, tuple), 'ngram must be a tuple'
    assert len(ngram) == 2, 'ngram size must equal to 2'
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
    else:
        raise Exception('model not supported')
    output, keys = _generate_topics()
    vectorizer = char_vectorizer.fit(output)
    vectorized = vectorizer.transform(output)
    return _FAST_SIMILARITY(vectorizer, vectorized, keys)


def fast_influencer(vectorizer = 'tfidf', ngram = (3, 10)):
    """
    Train a deep siamese network for influencers similarity

    Parameters
    ----------
    vectorizer: str, (default='tfidf')
        vectorization technique for a corpus
    ngram: tuple, (default=(1,4))
        n-grams size to train a corpus

    Returns
    -------
    _FAST_SIMILARITY: malaya.siamese_lstm._FAST_SIMILARITY class
    """
    if not _person_dict:
        _load_internal_data()
    assert isinstance(vectorizer, str), 'vectorizer must be a string'
    assert isinstance(ngram, tuple), 'ngram must be a tuple'
    assert len(ngram) == 2, 'ngram size must equal to 2'
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
    else:
        raise Exception('model not supported')
    output, keys = _generate_influencers()
    vectorizer = char_vectorizer.fit(output)
    vectorized = vectorizer.transform(output)
    return _FAST_SIMILARITY(vectorizer, vectorized, keys, is_influencers = True)
