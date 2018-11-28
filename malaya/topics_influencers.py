import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import re
from fuzzywuzzy import fuzz
import zipfile
import os
import random
from .text_functions import STOPWORDS, sentence_ngram, simple_textcleaning
from . import home
from .utils import download_file
from .skip_thought import train_model, batch_sequence

zip_location = home + '/rules-based.zip'

if not os.path.isfile(zip_location):
    print('downloading ZIP rules-based')
    download_file('rules-based.zip', zip_location)
    with zipfile.ZipFile(zip_location, 'r') as zip_ref:
        zip_ref.extractall(home)


def apply_stopwords_calon(string):
    string = re.sub('[^A-Za-z ]+', '', string)
    return ' '.join(
        [i for i in string.split() if i not in STOPWORDS and len(i) > 1]
    )


df = pd.read_csv(home + '/rules-based/calon.csv')
namacalon = df.NamaCalon.str.lower().unique().tolist()
for i in range(len(namacalon)):
    namacalon[i] = apply_stopwords_calon(namacalon[i])

df = pd.read_csv(home + '/rules-based/negeri.csv')
negeri = df.negeri.str.lower().unique().tolist()
parlimen = df.parlimen.str.lower().unique().tolist()
dun = df.dun.str.lower().unique().tolist()[:-1]

location = negeri + parlimen + dun

with open(home + '/rules-based/person-normalized', 'r') as fopen:
    person = list(filter(None, fopen.read().split('\n')))

person_dict = {}
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
    person_dict[splitted[0]] = uniques

with open(home + '/rules-based/topic-normalized', 'r') as fopen:
    topic = list(filter(None, fopen.read().split('\n')))

topic_dict = {}
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
    topic_dict[splitted[0]] = uniques

with open(home + '/rules-based/short-normalized', 'r') as fopen:
    short = list(filter(None, fopen.read().split('\n')))

short_dict = {}
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
    short_dict[splitted[0]] = uniques


def fuzzy_get_influencers(string):
    assert isinstance(string, str), 'input must be a string'
    string = string.lower()
    influencers = []
    for key, vals in person_dict.items():
        for v in vals:
            if fuzz.token_set_ratio(v, string) >= 80:
                influencers.append(key.lower())
                break
    for key, vals in short_dict.items():
        for v in vals:
            if v in string.split():
                influencers.append(key.lower())
                break

    for index in np.where(
        np.array([fuzz.token_set_ratio(i, string) for i in namacalon]) >= 80
    )[0]:
        influencers.append(namacalon[index].lower())
    return list(set(influencers))


def fuzzy_get_topics(string):
    assert isinstance(string, str), 'input must be a string'
    string = string.lower()
    topics = []
    for key, vals in topic_dict.items():
        for v in vals:
            if fuzz.token_set_ratio(v, string) >= 80:
                topics.append(key.lower())
                break
    for key, vals in person_dict.items():
        for v in vals:
            if fuzz.token_set_ratio(v, string) >= 80:
                topics.append(key.lower())
                break
    for key, vals in short_dict.items():
        for v in vals:
            if v in string.split():
                topics.append(key.lower())
                break

    return list(set(topics))


def is_location(string):
    for loc in location:
        if fuzz.token_set_ratio(loc.lower(), string) >= 90:
            return True
    return False


def fuzzy_get_location(string):
    assert isinstance(string, str), 'input must be a string'
    negeri_list = list(
        set([i for i in negeri if fuzz.token_set_ratio(i, string) >= 80])
    )
    parlimen_list = list(
        set([i for i in parlimen if fuzz.token_set_ratio(i, string) >= 80])
    )
    dun_list = list(
        set([i for i in dun if fuzz.token_set_ratio(i, string) >= 80])
    )
    return {'negeri': negeri_list, 'parlimen': parlimen_list, 'dun': dun_list}


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
        assert isinstance(string, str), 'input must be a string'
        assert isinstance(anchor, float), 'anchor must be a float'
        assert (
            anchor > 0 and anchor < 1
        ), 'anchor must be bigger than 0, less than 1'
        original_string = simple_textcleaning(string, decode = True)
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
        for key, vals in short_dict.items():
            for v in vals:
                if v in original_string.split():
                    results.append(key.lower())
                    break
        if self._is_influencers:
            for index in np.where(
                np.array(
                    [
                        fuzz.token_set_ratio(i, original_string)
                        for i in namacalon
                    ]
                )
                >= 80
            )[0]:
                results.append(namacalon[index].lower())
        return list(set(results))


class FAST_SIMILARITY:
    def __init__(self, vectorizer, vectorized, keys, is_influencers = False):
        self.vectorizer = vectorizer
        self.vectorized = vectorized
        self.keys = keys
        self._is_influencers = is_influencers

    def get_similarity(self, string, anchor = 0.1):
        assert isinstance(string, str), 'input must be a string'
        assert isinstance(anchor, float), 'anchor must be a float'
        assert (
            anchor > 0 and anchor < 1
        ), 'anchor must be bigger than 0, less than 1'
        original_string = simple_textcleaning(string, decode = True)
        string = ' '.join(set(original_string.split()))
        where = np.where(
            cosine_similarity(
                self.vectorized, self.vectorizer.transform([string])
            )[:, 0]
            > anchor
        )[0]
        results = [self.keys[i].lower() for i in where]
        for key, vals in short_dict.items():
            for v in vals:
                if v in original_string.split():
                    results.append(key.lower())
                    break
        if self._is_influencers:
            for index in np.where(
                np.array(
                    [
                        fuzz.token_set_ratio(i, original_string)
                        for i in namacalon
                    ]
                )
                >= 80
            )[0]:
                results.append(namacalon[index].lower())
        return list(set(results))


def generate_topics():
    texts = [' '.join(words) for _, words in topic_dict.items()]
    keys = [key for key, _ in topic_dict.items()]
    texts += [' '.join(words) for _, words in person_dict.items()]
    keys += [key for key, _ in person_dict.items()]
    texts = [' '.join(list(set(text.split()))) for text in texts]
    output = []
    for text in texts:
        output.append(
            ' '.join([word for word in text.split() if word not in STOPWORDS])
        )
    return output, keys


def generate_influencers():
    texts = [' '.join(words) for _, words in person_dict.items()]
    keys = [key for key, _ in person_dict.items()]
    texts = [' '.join(list(set(text.split()))) for text in texts]
    output = []
    for text in texts:
        output.append(
            ' '.join([word for word in text.split() if word not in STOPWORDS])
        )
    return output, keys


def deep_get_topics(
    epoch = 5,
    batch_size = 16,
    embedding_size = 256,
    output_size = 300,
    maxlen = 100,
    ngrams = (1, 4),
):
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
    sess, model, dictionary = train_model(
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
    sess, model, dictionary = train_model(
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
