from nltk.tokenize import word_tokenize
import os
import re
import pickle
import json
import xgboost as xgb
import numpy as np
from . import home
from .tatabahasa import tatabahasa_dict, hujung, permulaan
from .utils import download_file
from .text_functions import entities_textcleaning
from .keras_model import get_char_bidirectional, get_crf_lstm_bidirectional, get_crf_lstm_concat_bidirectional
from .keras_model import CONCAT_MODEL, WORD_MODEL, CHAR_MODEL

bow_pkl = home+'/bow-pos.pkl'
multinomial_pkl = home+'/multinomial-pos.pkl'
xgb_bow_pkl = home+'/xgb-bow-pos.pkl'
xgb_pkl = home+'/xgb-pos.pkl'

MULTINOMIAL, BOW = None, None
XGB, BOW_XGB = None, None

pos_labels = {0: 'ADJ', 1: 'ADP', 2: 'ADV', 3: 'AUX', 4: 'CCONJ', 5: 'DET',
6: 'NOUN', 7: 'NUM', 8: 'PART', 9: 'PRON', 10: 'PROPN',
11: 'SCONJ', 12: 'SYM', 13: 'VERB', 14: 'X'}

def naive_POS_word(word):
    for key, vals in tatabahasa_dict.items():
        if word in vals:
            return (key,word)
    try:
        if len(re.findall(r'^(.*?)(%s)$'%('|'.join(hujung[:1])), i)[0]) > 1:
            return ('KJ',word)
    except:
        pass
    try:
        if len(re.findall(r'^(.*?)(%s)'%('|'.join(permulaan[:-4])), word)[0]) > 1:
            return ('KJ',word)
    except Exception as e:
        pass
    if len(word) > 2:
        return ('KN',word)
    else:
        return ('',word)

def naive_pos(string):
    assert (isinstance(string, str)), "input must be a string"
    string = string.lower()
    results = []
    for i in word_tokenize(string):
        results.append(naive_POS_word(i))
    return results

def multinomial_pos(string):
    assert (isinstance(string, str)), "input must be a string"
    global MULTINOMIAL, BOW
    string = entities_textcleaning(string)
    if MULTINOMIAL is None and BOW is None:
        if not os.path.isfile(bow_pkl):
            print('downloading pickled bag-of-word multinomial POS')
            download_file("bow-entities.pkl", bow_pkl)
        if not os.path.isfile(multinomial_pkl):
            print('downloading pickled multinomial POS model')
            download_file("multinomial-entities.pkl", multinomial_pkl)
        with open(bow_pkl,'rb') as fopen:
            BOW = pickle.load(fopen)
        with open(multinomial_pkl,'rb') as fopen:
            MULTINOMIAL = pickle.load(fopen)
    return [(string[no],pos_labels[i]) for no, i in enumerate(MULTINOMIAL.predict(BOW.transform(string)))]

def xgb_pos(string):
    assert (isinstance(string, str)), "input must be a string"
    global XGB, BOW_XGB
    string = entities_textcleaning(string)
    if XGB is None and BOW_XGB is None:
        if not os.path.isfile(xgb_bow_pkl):
            print('downloading pickled bag-of-word XGB POS')
            download_file("xgb-bow-pos.pkl", xgb_bow_pkl)
        if not os.path.isfile(xgb_pkl):
            print('downloading pickled xgb POS model')
            download_file("xgb-pos.pkl", xgb_pkl)
        with open(xgb_bow_pkl,'rb') as fopen:
            BOW_XGB = pickle.load(fopen)
        with open(xgb_pkl,'rb') as fopen:
            XGB = pickle.load(fopen)
    dmatrix = xgb.DMatrix(BOW_XGB.transform(string))
    results = np.argmax(XGB.predict(dmatrix,ntree_limit=XGB.best_ntree_limit),axis=1)
    return [(string[no],pos_labels[i]) for no, i in enumerate(results)]

def get_available_pos_models():
    return ['char', 'word', 'concat']

char_frozen = home+'/pos-char-frozen.h5'
word_frozen = home+'/pos-word-frozen.h5'
concat_frozen = home+'/pos-concat-frozen.h5'

char_settings = home+'/pos-char-settings.json'
word_settings = home+'/pos-word-settings.json'
concat_settings = home+'/pos-concat-settings.json'

def deep_pos(model='concat'):
    if model == 'char':
        if not os.path.isfile(char_settings):
            print('downloading POS char settings')
            download_file("char-bidirectional-pos.json", char_settings)
        with open(char_settings,'r') as fopen:
            nodes = json.loads(fopen.read())
        if not os.path.isfile(char_frozen):
            print('downloading POS frozen char model')
            download_file("char-bidirectional-pos.h5", char_frozen)
        model = get_char_bidirectional(nodes['char2idx'],nodes['tag2idx'])
        model.load_weights(char_frozen)
        return CHAR_MODEL(model,nodes)
    elif model == 'word':
        if not os.path.isfile(word_settings):
            print('downloading POS word settings')
            download_file("crf-lstm-bidirectional-pos.json", word_settings)
        with open(word_settings,'r') as fopen:
            nodes = json.loads(fopen.read())
        if not os.path.isfile(word_frozen):
            print('downloading POS frozen word model')
            download_file("crf-lstm-bidirectional-pos.h5", word_frozen)
        model = get_crf_lstm_bidirectional(nodes['word2idx'],nodes['tag2idx'])
        model.load_weights(word_frozen)
        return WORD_MODEL(model,nodes)
    elif model == 'concat':
        if not os.path.isfile(concat_settings):
            print('downloading POS concat settings')
            download_file("crf-lstm-concat-bidirectional-pos.json", concat_settings)
        with open(concat_settings,'r') as fopen:
            nodes = json.loads(fopen.read())
            nodes['idx2word'] = {int(k):v for k,v in nodes['idx2word'].items()}
        if not os.path.isfile(concat_frozen):
            print('downloading POS frozen concat model')
            download_file("crf-lstm-concat-bidirectional-pos.h5", concat_frozen)
        model = get_crf_lstm_concat_bidirectional(nodes['char2idx'],nodes['word2idx'],nodes['tag2idx'])
        model.load_weights(concat_frozen)
        return CONCAT_MODEL(model,nodes)
    else:
        raise Exception('model not supported')
