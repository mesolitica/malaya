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
from .text_functions import (
    STOPWORDS,
    sentence_ngram,
    simple_textcleaning,
    str_idx,
)
from . import home
from .utils import download_file
from .skip_thought import train_model as skip_train, batch_sequence
from .siamese_lstm import train_model as siamese_train

zip_location = home + '/rules-based.zip'
NAMACALON = None
PARLIMEN = None
DUN = None
NEGERI = None
LOCATION = None
PERSON_DICT = None
TOPIC_DICT = None
SHORT_DICT = None

STOPWORD_CALON = [
    'datuk',
    'bin',
    'hj',
    'haji',
    'bn',
    'bnt',
    'prof',
    'binti',
    'dr',
    'ustaz',
    'mejar',
    'ir',
    'md',
    'tuan',
    'puan',
    'yb',
    'ustazah',
    'cikgu',
    'dato',
    'dsp',
]


def apply_stopwords_calon(string):
    string = re.sub('[^A-Za-z ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string.lower()).strip()
    return ' '.join(
        [i for i in string.split() if i not in STOPWORD_CALON and len(i) > 1]
    )


class DEEP_SIAMESE_SIMILARITY:
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
        for key, vals in SHORT_DICT.items():
            for v in vals:
                if v in original_string.split():
                    results.append(key.lower())
                    break
        if self._is_influencers:
            for index in np.where(
                np.array(
                    [
                        fuzz.token_set_ratio(i, original_string)
                        for i in NAMACALON
                    ]
                )
                >= 80
            )[0]:
                results.append(NAMACALON[index].lower())
        return list(set(results))


class DEEP_SIMILARITY:
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
        for key, vals in SHORT_DICT.items():
            for v in vals:
                if v in original_string.split():
                    results.append(key.lower())
                    break
        if self._is_influencers:
            for index in np.where(
                np.array(
                    [
                        fuzz.token_set_ratio(i, original_string)
                        for i in NAMACALON
                    ]
                )
                >= 80
            )[0]:
                results.append(NAMACALON[index].lower())
        return list(set(results))


class FAST_SIMILARITY:
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
        for key, vals in SHORT_DICT.items():
            for v in vals:
                if v in original_string.split():
                    results.append(key.lower())
                    break
        if self._is_influencers:
            for index in np.where(
                np.array(
                    [
                        fuzz.token_set_ratio(i, original_string)
                        for i in NAMACALON
                    ]
                )
                >= 80
            )[0]:
                results.append(NAMACALON[index].lower())
        return list(set(results))


def load_internal_data():
    import pandas as pd
    import zipfile

    global NAMACALON, PARLIMEN, DUN, NEGERI, LOCATION, PERSON_DICT, TOPIC_DICT, SHORT_DICT
    if not os.path.isfile(zip_location):
        print('downloading Topics, Influencers, Location data')
        download_file('rules-based.zip', zip_location)

    if not os.path.exists(home + '/rules-based'):
        with zipfile.ZipFile(zip_location, 'r') as zip_ref:
            zip_ref.extractall(home)

    df = pd.read_csv(home + '/rules-based/calon.csv')
    NAMACALON = df.NamaCalon.str.lower().unique().tolist()
    for i in range(len(NAMACALON)):
        NAMACALON[i] = apply_stopwords_calon(NAMACALON[i])
    NAMACALON = list(set(NAMACALON))

    df = pd.read_csv(home + '/rules-based/negeri.csv')
    NEGERI = df.negeri.str.lower().unique().tolist()
    PARLIMEN = df.parlimen.str.lower().unique().tolist()
    DUN = df.dun.str.lower().unique().tolist()[:-1]

    LOCATION = NEGERI + PARLIMEN + DUN

    with open(home + '/rules-based/person-normalized', 'r') as fopen:
        person = list(filter(None, fopen.read().split('\n')))

    PERSON_DICT = {}
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
        PERSON_DICT[splitted[0]] = uniques

    with open(home + '/rules-based/topic-normalized', 'r') as fopen:
        topic = list(filter(None, fopen.read().split('\n')))

    TOPIC_DICT = {}
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
        TOPIC_DICT[splitted[0]] = uniques

    with open(home + '/rules-based/short-normalized', 'r') as fopen:
        short = list(filter(None, fopen.read().split('\n')))

    SHORT_DICT = {}
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
        SHORT_DICT[splitted[0]] = uniques


def fuzzy_get_influencers(string):
    """
    Return similar influencers.

    Parameters
    ----------
    string: str

    Returns
    -------
    results: list of strings
    """
    if not PERSON_DICT:
        load_internal_data()
    assert isinstance(string, str), 'input must be a string'
    string = string.lower()
    influencers = []
    for key, vals in PERSON_DICT.items():
        for v in vals:
            if fuzz.token_set_ratio(v, string) >= 80:
                influencers.append(key.lower())
                break
    for key, vals in SHORT_DICT.items():
        for v in vals:
            if v in string.split():
                influencers.append(key.lower())
                break

    for index in np.where(
        np.array([fuzz.token_set_ratio(i, string) for i in NAMACALON]) >= 80
    )[0]:
        influencers.append(NAMACALON[index].lower())
    return list(set(influencers))


def fuzzy_get_topics(string):
    """
    Return similar topics.

    Parameters
    ----------
    string: str

    Returns
    -------
    results: list of strings
    """
    if not PERSON_DICT:
        load_internal_data()
    assert isinstance(string, str), 'input must be a string'
    string = string.lower()
    topics = []
    for key, vals in TOPIC_DICT.items():
        for v in vals:
            if fuzz.token_set_ratio(v, string) >= 80:
                topics.append(key.lower())
                break
    for key, vals in PERSON_DICT.items():
        for v in vals:
            if fuzz.token_set_ratio(v, string) >= 80:
                topics.append(key.lower())
                break
    for key, vals in SHORT_DICT.items():
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
    if not PERSON_DICT:
        load_internal_data()
    for loc in LOCATION:
        if fuzz.token_set_ratio(loc.lower(), string) >= 90:
            return True
    return False


def fuzzy_get_location(string):
    """
    Return similar location.

    Parameters
    ----------
    string: str

    Returns
    -------
    results: list of strings
    """
    if not PERSON_DICT:
        load_internal_data()
    assert isinstance(string, str), 'input must be a string'
    negeri_list = list(
        set([i for i in NEGERI if fuzz.token_set_ratio(i, string) >= 80])
    )
    parlimen_list = list(
        set([i for i in PARLIMEN if fuzz.token_set_ratio(i, string) >= 80])
    )
    dun_list = list(
        set([i for i in DUN if fuzz.token_set_ratio(i, string) >= 80])
    )
    return {'negeri': negeri_list, 'parlimen': parlimen_list, 'dun': dun_list}


def generate_topics():
    if not PERSON_DICT:
        load_internal_data()
    texts = [' '.join(words) for _, words in TOPIC_DICT.items()]
    keys = [key for key, _ in TOPIC_DICT.items()]
    texts += [' '.join(words) for _, words in PERSON_DICT.items()]
    keys += [key for key, _ in PERSON_DICT.items()]
    texts = [' '.join(list(set(text.split()))) for text in texts]
    output = []
    for text in texts:
        output.append(
            ' '.join([word for word in text.split() if word not in STOPWORDS])
        )
    return output, keys


def generate_influencers():
    if not PERSON_DICT:
        load_internal_data()
    texts = [' '.join(words) for _, words in PERSON_DICT.items()]
    keys = [key for key, _ in PERSON_DICT.items()]
    texts = [' '.join(list(set(text.split()))) for text in texts]
    output = []
    for text in texts:
        output.append(
            ' '.join([word for word in text.split() if word not in STOPWORDS])
        )
    return output, keys


def deep_siamese_get_topics(
    epoch = 5,
    batch_size = 32,
    embedding_size = 256,
    output_size = 300,
    maxlen = 100,
    ngrams = (1, 4),
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
    ngrams: tuple, (default=(1,4))
        n-grams size to train a corpus

    Returns
    -------
    DEEP_SIAMESE_SIMILARITY: malaya.topics_influencers.DEEP_SIAMESE_SIMILARITY class
    """
    if not PERSON_DICT:
        load_internal_data()
    assert isinstance(epoch, int), 'epoch must be an integer'
    assert isinstance(batch_size, int), 'batch_size must be an integer'
    assert isinstance(embedding_size, int), 'embedding_size must be an integer'
    assert isinstance(output_size, int), 'output_size must be an integer'
    assert isinstance(maxlen, int), 'maxlen must be an integer'
    assert isinstance(ngrams, tuple), 'ngrams must be a tuple'
    output, keys = generate_topics()
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
    return DEEP_SIAMESE_SIMILARITY(sess, model, keys, dictionary, maxlen)


def deep_siamese_get_influencers(
    epoch = 5,
    batch_size = 32,
    embedding_size = 256,
    output_size = 300,
    maxlen = 100,
    ngrams = (1, 4),
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
    ngrams: tuple, (default=(1,4))
        n-grams size to train a corpus

    Returns
    -------
    DEEP_SIAMESE_SIMILARITY: malaya.topic_influencers.DEEP_SIAMESE_SIMILARITY class
    """
    if not PERSON_DICT:
        load_internal_data()
    assert isinstance(epoch, int), 'epoch must be an integer'
    assert isinstance(batch_size, int), 'batch_size must be an integer'
    assert isinstance(embedding_size, int), 'embedding_size must be an integer'
    assert isinstance(output_size, int), 'output_size must be an integer'
    assert isinstance(maxlen, int), 'maxlen must be an integer'
    assert isinstance(ngrams, tuple), 'ngrams must be a tuple'
    output, keys = generate_influencers()
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
    return DEEP_SIAMESE_SIMILARITY(
        sess, model, keys, dictionary, maxlen, is_influencers = True
    )


def deep_get_topics(
    epoch = 5,
    batch_size = 16,
    embedding_size = 256,
    output_size = 300,
    maxlen = 100,
    ngrams = (1, 4),
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
    output_size: int, (default=100)
        encoder output size, bigger means more vector definition
    maxlen: int, (default=100)
        max length of a string to be train
    ngrams: tuple, (default=(1,4))
        n-grams size to train a corpus

    Returns
    -------
    DEEP_SIMILARITY: malaya.topic_influencers.DEEP_SIMILARITY class
    """
    if not PERSON_DICT:
        load_internal_data()
    assert isinstance(epoch, int), 'epoch must be an integer'
    assert isinstance(batch_size, int), 'batch_size must be an integer'
    assert isinstance(embedding_size, int), 'embedding_size must be an integer'
    assert isinstance(output_size, int), 'output_size must be an integer'
    assert isinstance(maxlen, int), 'maxlen must be an integer'
    assert isinstance(ngrams, tuple), 'ngrams must be a tuple'
    output, keys = generate_topics()
    batch_x, batch_y = [], []
    for i in range(len(output)):
        augmentation = sentence_ngram(output[i])
        batch_y.extend([keys[i]] * len(augmentation))
        batch_x.extend(augmentation)
    batch_x, batch_y = shuffle(batch_x, batch_y)
    sess, model, dictionary = skip_train(
        batch_x,
        batch_y,
        epoch = epoch,
        batch_size = batch_size,
        embedding_size = embedding_size,
        output_size = output_size,
        maxlen = maxlen,
    )
    encoded = sess.run(
        model.get_thought,
        feed_dict = {
            model.INPUT: batch_sequence(output, dictionary, maxlen = maxlen)
        },
    )
    return DEEP_SIMILARITY(sess, model, encoded, keys, dictionary, maxlen)


def deep_get_influencers(
    epoch = 10,
    batch_size = 16,
    embedding_size = 256,
    output_size = 300,
    maxlen = 100,
    ngrams = (1, 4),
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
    output_size: int, (default=100)
        encoder output size, bigger means more vector definition
    maxlen: int, (default=100)
        max length of a string to be train
    ngrams: tuple, (default=(1,4))
        n-grams size to train a corpus

    Returns
    -------
    DEEP_SIMILARITY: malaya.topics_influencers.DEEP_SIMILARITY class
    """
    if not PERSON_DICT:
        load_internal_data()
    assert isinstance(epoch, int), 'epoch must be an integer'
    assert isinstance(batch_size, int), 'batch_size must be an integer'
    assert isinstance(embedding_size, int), 'embedding_size must be an integer'
    assert isinstance(output_size, int), 'output_size must be an integer'
    assert isinstance(maxlen, int), 'maxlen must be an integer'
    assert isinstance(ngrams, tuple), 'ngrams must be a tuple'
    output, keys = generate_influencers()
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
        epoch = epoch,
        batch_size = batch_size,
        embedding_size = embedding_size,
        output_size = output_size,
        maxlen = maxlen,
    )
    encoded = sess.run(
        model.get_thought,
        feed_dict = {
            model.INPUT: batch_sequence(output, dictionary, maxlen = maxlen)
        },
    )
    return DEEP_SIMILARITY(
        sess, model, encoded, keys, dictionary, maxlen, is_influencers = True
    )


def fast_get_topics(vectorizer = 'tfidf', ngrams = (3, 10)):
    """
    Train a deep siamese network for topics similarity

    Parameters
    ----------
    vectorizer: str, (default='tfidf')
        vectorization technique for a corpus
    ngrams: tuple, (default=(1,4))
        n-grams size to train a corpus

    Returns
    -------
    FAST_SIMILARITY: malaya.siamese_lstm.FAST_SIMILARITY class
    """
    if not PERSON_DICT:
        load_internal_data()
    assert isinstance(vectorizer, str), 'vectorizer must be a string'
    assert isinstance(ngrams, tuple), 'ngrams must be a tuple'
    if 'tfidf' in vectorizer.lower():
        char_vectorizer = TfidfVectorizer(
            sublinear_tf = True,
            strip_accents = 'unicode',
            analyzer = 'char',
            ngram_range = ngrams,
        )
    elif 'count' in vectorizer.lower():
        char_vectorizer = CountVectorizer(
            strip_accents = 'unicode', analyzer = 'char', ngram_range = ngrams
        )
    else:
        raise Exception('model not supported')
    output, keys = generate_topics()
    vectorizer = char_vectorizer.fit(output)
    vectorized = vectorizer.transform(output)
    return FAST_SIMILARITY(vectorizer, vectorized, keys)


def fast_get_influencers(vectorizer = 'tfidf', ngrams = (3, 10)):
    """
    Train a deep siamese network for influencers similarity

    Parameters
    ----------
    vectorizer: str, (default='tfidf')
        vectorization technique for a corpus
    ngrams: tuple, (default=(1,4))
        n-grams size to train a corpus

    Returns
    -------
    FAST_SIMILARITY: malaya.siamese_lstm.FAST_SIMILARITY class
    """
    if not PERSON_DICT:
        load_internal_data()
    assert isinstance(vectorizer, str), 'vectorizer must be a string'
    assert isinstance(ngrams, tuple), 'ngrams must be a tuple'
    if 'tfidf' in vectorizer.lower():
        char_vectorizer = TfidfVectorizer(
            sublinear_tf = True,
            strip_accents = 'unicode',
            analyzer = 'char',
            ngram_range = ngrams,
        )
    elif 'count' in vectorizer.lower():
        char_vectorizer = CountVectorizer(
            strip_accents = 'unicode', analyzer = 'char', ngram_range = ngrams
        )
    else:
        raise Exception('model not supported')
    output, keys = generate_influencers()
    vectorizer = char_vectorizer.fit(output)
    vectorized = vectorizer.transform(output)
    return FAST_SIMILARITY(vectorizer, vectorized, keys, is_influencers = True)
