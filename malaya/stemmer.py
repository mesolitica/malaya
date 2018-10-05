import re
import os
import json
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from .tatabahasa import permulaan, hujung
from .utils import load_graph, download_file
from .text_functions import pad_sentence_batch, stemmer_str_idx, classification_textcleaning
from . import home

factory = StemmerFactory()
sastrawi = factory.create_stemmer()
stemmer_json = home+'/stemmer-deep.json'
stemmer_graph = home+'/stemmer-deep.pb'
GO = 0
PAD = 1
EOS = 2
UNK = 3

class DEEP_STEMMER:
    def __init__(self, x, logits, sess, dicts):
        self._sess = sess
        self._x = x
        self._logits = logits
        self._dicts = dicts
        self._dicts['rev_dictionary_to'] = {int(k):v for k,v in self._dicts['rev_dictionary_to'].items()}

    def stem(self,string):
        assert (isinstance(string, str)), "input must be a string"
        token_strings = classification_textcleaning(string,True).split()
        idx = stemmer_str_idx(token_strings,self._dicts['dictionary_from'])
        predicted = self._sess.run(self._logits, feed_dict={self._x:pad_sentence_batch(idx, PAD)[0]})
        results = []
        for word in predicted:
            results.append(''.join([self._dicts['rev_dictionary_to'][c] for c in word if c not in[GO,PAD,EOS,UNK]]))
        return ' '.join(results)

def naive_stemmer(word):
    assert (isinstance(word, str)), "input must be a string"
    try:
        word = re.findall(r'^(.*?)(%s)$'%('|'.join(hujung)), word)[0][0]
        mula = re.findall(r'^(.*?)(%s)'%('|'.join(permulaan[::-1])), word)[0][1]
        return word.replace(mula,'')
    except:
        return word

def sastrawi_stemmer(string):
    assert (isinstance(string, str)), "input must be a string"
    return sastrawi.stem(string)

def deep_stemmer():
    if not os.path.isfile(stemmer_json):
        print('downloading JSON stemmer')
        download_file("v5/stemmer-deep.json", stemmer_json)
    with open(stemmer_json,'r') as fopen:
        dic_stemmer = json.load(fopen)
    if not os.path.isfile(stemmer_graph):
        print('downloading stemmer graph')
        download_file("v5/stemmer-frozen-model.pb", stemmer_graph)
    g=load_graph(stemmer_graph)
    return DEEP_STEMMER(g.get_tensor_by_name('import/Placeholder:0'),
                        g.get_tensor_by_name('import/logits:0'),
                        tf.InteractiveSession(graph=g),
                        dic_stemmer)
