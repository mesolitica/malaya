import pickle
import os
import json
import xgboost as xgb
import numpy as np
from .utils import download_file
from . import home
from .text_functions import entities_textcleaning
from .keras_model import get_char_bidirectional, get_crf_lstm_bidirectional, get_crf_lstm_concat_bidirectional
from .keras_model import CONCAT_MODEL, WORD_MODEL, CHAR_MODEL

bow_pkl = home+'/bow-entities.pkl'
multinomial_pkl = home+'/multinomial-entities.pkl'
xgb_bow_pkl = home+'/xgb-bow-entities.pkl'
xgb_pkl = home+'/xgb-entities.pkl'

MULTINOMIAL, BOW = None, None
XGB, BOW_XGB = None, None

entities_labels = {0:'OTHER', 1:'law', 2:'location', 3:'organization',
4:'person', 5:'quantity', 6:'time'}

def multinomial_entities(string):
    assert (isinstance(string, str)), "input must be a string"
    global MULTINOMIAL, BOW
    string = entities_textcleaning(string)
    if MULTINOMIAL is None and BOW is None:
        if not os.path.isfile(bow_pkl):
            print('downloading ENTITIES pickled bag-of-word multinomial')
            download_file("bow-entities.pkl", bow_pkl)
        if not os.path.isfile(multinomial_pkl):
            print('downloading ENTITIES pickled multinomial model')
            download_file("multinomial-entities.pkl", multinomial_pkl)
        with open(bow_pkl,'rb') as fopen:
            BOW = pickle.load(fopen)
        with open(multinomial_pkl,'rb') as fopen:
            MULTINOMIAL = pickle.load(fopen)
    return [(string[no],entities_labels[i]) for no, i in enumerate(MULTINOMIAL.predict(BOW.transform(string)))]

def xgb_entities(string):
    assert (isinstance(string, str)), "input must be a string"
    global XGB, BOW_XGB
    string = entities_textcleaning(string)
    if XGB is None and BOW_XGB is None:
        if not os.path.isfile(xgb_bow_pkl):
            print('downloading ENTITIES pickled bag-of-word XGB')
            download_file("xgb-bow-entities.pkl", xgb_bow_pkl)
        if not os.path.isfile(xgb_pkl):
            print('downloading ENTITIES pickled XGB model')
            download_file("xgb-entities.pkl", xgb_pkl)
        with open(xgb_bow_pkl,'rb') as fopen:
            BOW_XGB = pickle.load(fopen)
        with open(xgb_pkl,'rb') as fopen:
            XGB = pickle.load(fopen)
    dmatrix = xgb.DMatrix(BOW_XGB.transform(string))
    results = np.argmax(XGB.predict(dmatrix,ntree_limit=XGB.best_ntree_limit),axis=1)
    return [(string[no],entities_labels[i]) for no, i in enumerate(results)]

def get_available_entities_models():
    return ['char', 'word', 'concat']

char_frozen = home+'/entities-char-frozen.h5'
word_frozen = home+'/entities-word-frozen.h5'
concat_frozen = home+'/entities-concat-frozen.h5'

char_settings = home+'/entities-char-settings.json'
word_settings = home+'/entities-word-settings.json'
concat_settings = home+'/entities-concat-settings.json'

def deep_entities(model='concat'):
    if model == 'char':
        if not os.path.isfile(char_settings):
            print('downloading ENTITIES char settings')
            download_file("char-bidirectional.json", char_settings)
        with open(char_settings,'r') as fopen:
            nodes = json.loads(fopen.read())
        if not os.path.isfile(char_frozen):
            print('downloading ENTITIES frozen char model')
            download_file("char-bidirectional.h5", char_frozen)
        model = get_char_bidirectional(nodes['char2idx'],nodes['tag2idx'])
        model.load_weights(char_frozen)
        return CHAR_MODEL(model,nodes)
    elif model == 'word':
        if not os.path.isfile(word_settings):
            print('downloading ENTITIES word settings')
            download_file("crf-lstm-bidirectional.json", word_settings)
        with open(word_settings,'r') as fopen:
            nodes = json.loads(fopen.read())
        if not os.path.isfile(word_frozen):
            print('downloading ENTITIES frozen word model')
            download_file("crf-lstm-bidirectional.h5", word_frozen)
        model = get_crf_lstm_bidirectional(nodes['word2idx'],nodes['tag2idx'])
        model.load_weights(word_frozen)
        return WORD_MODEL(model,nodes)
    elif model == 'concat':
        if not os.path.isfile(concat_settings):
            print('downloading ENTITIES concat settings')
            download_file("crf-lstm-concat-bidirectional.json", concat_settings)
        with open(concat_settings,'r') as fopen:
            nodes = json.loads(fopen.read())
            nodes['idx2word'] = {int(k):v for k,v in nodes['idx2word'].items()}
        if not os.path.isfile(concat_frozen):
            print('downloading ENTITIES frozen concat model')
            download_file("crf-lstm-concat-bidirectional.h5", concat_frozen)
        model = get_crf_lstm_concat_bidirectional(nodes['char2idx'],nodes['word2idx'],nodes['tag2idx'])
        model.load_weights(concat_frozen)
        return CONCAT_MODEL(model,nodes)
    else:
        raise Exception('model not supported')
