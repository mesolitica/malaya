import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import re
import os
import json
import tensorflow as tf
from unidecode import unidecode
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from .tatabahasa import permulaan, hujung
from .utils import load_graph, download_file
from .text_functions import (
    pad_sentence_batch,
    stemmer_str_idx,
    classification_textcleaning,
)
from . import home

factory = StemmerFactory()
sastrawi = factory.create_stemmer()
stemmer_json = home + '/stemmer-deep.json'
stemmer_graph = home + '/stemmer-deep.pb'
GO = 0
PAD = 1
EOS = 2
UNK = 3


def classification_textcleaning_stemmer_attention(string):
    string = re.sub(
        'http\S+|www.\S+',
        '',
        ' '.join(
            [i for i in string.split() if i.find('#') < 0 and i.find('@') < 0]
        ),
    )
    string = unidecode(string).replace('.', ' . ').replace(',', ' , ')
    string = re.sub('[^A-Za-z ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string).strip()
    string = ' '.join(
        [i for i in re.findall('[\\w\']+|[;:\-\(\)&.,!?"]', string) if len(i)]
    )
    string = string.lower().split()
    string = [(naive_stemmer(word), word) for word in string]
    return (
        ' '.join([word[0] for word in string if len(word[0]) > 1]),
        ' '.join([word[1] for word in string if len(word[0]) > 1]),
    )


def classification_textcleaning_stemmer(string):
    string = re.sub(
        'http\S+|www.\S+',
        '',
        ' '.join(
            [i for i in string.split() if i.find('#') < 0 and i.find('@') < 0]
        ),
    )
    string = unidecode(string).replace('.', ' . ').replace(',', ' , ')
    string = re.sub('[^A-Za-z ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string).strip()
    string = ' '.join(
        [i for i in re.findall('[\\w\']+|[;:\-\(\)&.,!?"]', string) if len(i)]
    )
    string = string.lower().split()
    string = [(naive_stemmer(word), word) for word in string]
    return ' '.join([word[0] for word in string if len(word[0]) > 1])


class DEEP_STEMMER:
    def __init__(self, x, logits, sess, dicts):
        self._sess = sess
        self._x = x
        self._logits = logits
        self._dicts = dicts
        self._dicts['rev_dictionary_to'] = {
            int(k): v for k, v in self._dicts['rev_dictionary_to'].items()
        }

    def stem(self, string):
        """
        Stem a string.

        Parameters
        ----------
        string : str

        Returns
        -------
        string: stemmed string
        """
        assert isinstance(string, str), 'input must be a string'
        token_strings = classification_textcleaning(string, True).split()
        idx = stemmer_str_idx(token_strings, self._dicts['dictionary_from'])
        predicted = self._sess.run(
            self._logits, feed_dict = {self._x: pad_sentence_batch(idx, PAD)[0]}
        )
        results = []
        for word in predicted:
            results.append(
                ''.join(
                    [
                        self._dicts['rev_dictionary_to'][c]
                        for c in word
                        if c not in [GO, PAD, EOS, UNK]
                    ]
                )
            )
        return ' '.join(results)


def naive_stemmer(word):
    """
    Stem a string using Regex.

    Parameters
    ----------
    string : str

    Returns
    -------
    string: stemmed string
    """
    assert isinstance(word, str), 'input must be a string'
    hujung_result = re.findall(r'^(.*?)(%s)$' % ('|'.join(hujung)), word)
    word = hujung_result[0][0] if len(hujung_result) else word
    permulaan_result = re.findall(
        r'^(.*?)(%s)' % ('|'.join(permulaan[::-1])), word
    )
    permulaan_result.extend(
        re.findall(r'^(.*?)(%s)' % ('|'.join(permulaan)), word)
    )
    mula = permulaan_result if len(permulaan_result) else ''
    if len(mula):
        mula = mula[1][1] if len(mula[1][1]) > len(mula[0][1]) else mula[0][1]
    return word.replace(mula, '')


def sastrawi_stemmer(string):
    """
    Stem a string using Sastrawi.

    Parameters
    ----------
    string : str

    Returns
    -------
    string: stemmed string.
    """
    assert isinstance(string, str), 'input must be a string'
    return sastrawi.stem(string)


def deep_stemmer():
    """
    Load seq2seq stemmer deep learning model.

    Returns
    -------
    DEEP_STEMMER: malaya.stemmer.DEEP_STEMMER class
    """
    if not os.path.isfile(stemmer_json):
        print('downloading JSON stemmer')
        download_file('v5/stemmer-deep.json', stemmer_json)
    with open(stemmer_json, 'r') as fopen:
        dic_stemmer = json.load(fopen)
    if not os.path.isfile(stemmer_graph):
        print('downloading stemmer graph')
        download_file('v5/stemmer-frozen-model.pb', stemmer_graph)
    g = load_graph(stemmer_graph)
    return DEEP_STEMMER(
        g.get_tensor_by_name('import/Placeholder:0'),
        g.get_tensor_by_name('import/logits:0'),
        tf.InteractiveSession(graph = g),
        dic_stemmer,
    )
