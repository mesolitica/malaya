import numpy as np
from fuzzywuzzy import fuzz
import pickle
import os
import json
import tensorflow as tf
from collections import Counter
from . import home
from .utils import load_graph, download_file
from .num2word import to_cardinal
from .text_functions import normalizer_textcleaning, stemmer_str_idx, pad_sentence_batch
from .tatabahasa import rules_normalizer
from .spell import return_possible, edit_normalizer, build_dicts, return_known

consonants = 'bcdfghjklmnpqrstvwxyz'
vowels = 'aeiou'
sounds = {'x':'tidak','y':'kenapa','n':'dan','g':'pergi','s':'seperti','d':'di','k':'ok','u':'awak',
        't':'nanti','p':'pergi','wai':'kenapa'}
GO = 0
PAD = 1
EOS = 2
UNK = 3
malay_text = home+'/malay-text.txt'
normalizer_json = home+'/normalizer-deep.json'
normalizer_graph = home+'/normalizer-deep.pb'

class DEEP_NORMALIZER:
    def __init__(self, x, logits, sess, dicts):
        self._sess = sess
        self._x = x
        self._logits = logits
        self._dicts = dicts
        self._dicts['rev_dictionary_to'] = {int(k):v for k,v in self._dicts['rev_dictionary_to'].items()}

    def normalize(self,string):
        assert (isinstance(string, str)), "input must be a string"
        token_strings = normalizer_textcleaning(string).split()
        idx = stemmer_str_idx(token_strings,self._dicts['dictionary_from'])
        predicted = self._sess.run(self._logits, feed_dict={self._x:pad_sentence_batch(idx, PAD)[0]})
        results = []
        for word in predicted:
            results.append(''.join([self._dicts['rev_dictionary_to'][c] for c in word if c not in[GO,PAD,EOS,UNK]]))
        return ' '.join(results)

class SPELL_NORMALIZE:
    def __init__(self,corpus):
        self.corpus = Counter(corpus)

    def normalize(self, string, debug=True):
        assert (isinstance(string, str)), "input must be a string"
        result = []
        for word in normalizer_textcleaning(string).split():
            if word[0] == 'x' and len(word) > 1:
                result_string = 'tak '
                word = word[1:]
            else:
                result_string = ''
            if word[-2:] == 'la':
                end_result_string = ' lah'
                word = word[:-2]
            elif word[-3:] == 'lah':
                end_result_string = ' lah'
                word = word[:-3]
            else:
                end_result_string = ''
            if word in sounds:
                result.append(result_string + sounds[word] + end_result_string)
                continue
            if word in rules_normalizer:
                result.append(result_string + rules_normalizer[word] + end_result_string)
                continue
            if word in self.corpus:
                result.append(result_string + word + end_result_string)
                continue
            candidates = return_known([word], self.corpus) or return_known(edit_normalizer(word), self.corpus) or return_possible(word, self.corpus, edit_normalizer) or [word]
            candidates = list(candidates)
            if debug:
                print([(k,fuzz.ratio(word, k)) for k in candidates])
            result.append(result_string + candidates[np.argmax([fuzz.ratio(word, k) for k in candidates])] + end_result_string)
        return ' '.join(result)

class FUZZY_NORMALIZE:
    def __init__(self,normalized,corpus):
        self.normalized = normalized
        self.corpus = corpus

    def normalize(self,string):
        assert (isinstance(string, str)), "input must be a string"
        result = []
        for word in normalizer_textcleaning(string).split():
            if word[0] == 'x' and len(word) > 1:
                result_string = 'tak '
                word = word[1:]
            else:
                result_string = ''
            if word[-2:] == 'la':
                end_result_string = ' lah'
                word = word[:-2]
            elif word[-3:] == 'lah':
                end_result_string = ' lah'
                word = word[:-3]
            else:
                end_result_string = ''
            if word in sounds:
                result.append(result_string + sounds[word] + end_result_string)
                continue
            if word in rules_normalizer:
                result.append(result_string + rules_normalizer[word] + end_result_string)
                continue
            if word in self.corpus:
                result.append(result_string + word + end_result_string)
                continue
            results = []
            for i in range(len(self.normalized)):
                results.append(np.mean([fuzz.ratio(word, k) for k in self.normalized[i]]))
            if len(np.where(np.array(results) > 70)[0]) < 1:
                result.append(result_string + word + end_result_string)
                continue
            result.append(result_string + self.corpus[np.argmax(results)] + end_result_string)
        return ' '.join(result)

def fuzzy_normalizer(corpus):
    assert (isinstance(corpus, list) and isinstance(corpus[0], str)), 'input must be list of strings'
    transform = []
    for i in corpus:
        i = i.lower()
        result = []
        result.append(i)
        result.append(''.join(char for char in i if char not in vowels))
        if i[0] in consonants and i[-1] in consonants:
            result.append(i[0]+i[-1])
        if i[-1] == 'a':
            result.append(i[:-1]+'e')
            result.append(i+'k')
        if i[1] in vowels and i[0] in consonants:
            result.append(i[0]+i[2:])
        if i[-2] in vowels and i[-1] in consonants:
            result.append(i[:-2]+i[-1])
        result.append(i[0]+i[-1])
        if i[-2:] == 'ar':
            result.append(i[:-2]+'o')
        if i[:2] == 'ha':
            result.append(i[1:])
        transform.append(list(set(result)))
    return FUZZY_NORMALIZE(transform,corpus)

def spell_normalizer(corpus):
    assert (isinstance(corpus, list) and isinstance(corpus[0], str)), 'input must be list of strings'
    return SPELL_NORMALIZE(corpus)

def load_malay_dictionary():
    if not os.path.isfile(malay_text):
        print('downloading Malay texts')
        download_file('v6/malay-text.txt', malay_text)
    with open(malay_text,'r') as fopen:
        return [text.lower() for text in (list(filter(None, fopen.read().split('\n'))))]

def basic_normalizer(string):
    assert (isinstance(string, str)), "input must be a string"
    result = []
    for i in normalizer_textcleaning(string).split():
        if i in sounds:
            result.append(sounds[i])
        elif i[-1] == '2':
            result.append(i[:-1])
        else:
            result.append(i)
    return ' '.join(result)

def deep_normalizer():
    if not os.path.isfile(normalizer_json):
        print('downloading JSON normalizer')
        download_file("v6/normalizer-deep.json", normalizer_json)
    with open(normalizer_json,'r') as fopen:
        dic_normalizer = json.load(fopen)
    if not os.path.isfile(normalizer_graph):
        print('downloading normalizer graph')
        download_file("v6/normalizer-deep.pb", normalizer_graph)
    g=load_graph(normalizer_graph)
    return DEEP_NORMALIZER(g.get_tensor_by_name('import/Placeholder:0'),
                        g.get_tensor_by_name('import/logits:0'),
                        tf.InteractiveSession(graph=g),
                        dic_normalizer)
