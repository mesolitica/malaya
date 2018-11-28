import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import re
import tensorflow as tf
import numpy as np
import os
import json
from . import home
from .text_functions import (
    process_word_pos_entities,
    char_str_idx,
    generate_char_seq,
)
from .utils import load_graph, download_file

char_settings = home + '/char-settings.json'
char_frozen = home + '/char_frozen_model.pb'
concat_settings = home + '/concat-settings.json'
concat_frozen = home + '/concat_frozen_model.pb'
attention_settings = home + '/attention-settings.json'
attention_frozen = home + '/attention_pos_frozen_model.pb'


class DEEP_MODELS:
    def __init__(self, nodes, sess, predict):
        self.nodes = nodes
        self.sess = sess
        self.__predict = predict

    def predict(self, string):
        assert isinstance(string, str), 'input must be a string'
        return self.__predict(string, self.sess, self.nodes)


def get_entity_char(string, sess, model):
    batch_x = char_str_idx(
        [process_word_pos_entities(w) for w in string.split()],
        model['char2idx'],
        0,
    )
    logits, logits_pos = sess.run(
        [tf.argmax(model['logits'], 1), tf.argmax(model['logits_pos'], 1)],
        feed_dict = {model['X']: batch_x},
    )
    results = []
    for no, i in enumerate(string.split()):
        results.append(
            (
                i,
                model['idx2tag'][str(logits[no])],
                model['idx2pos'][str(logits_pos[no])],
            )
        )
    return results


def get_entity_concat(string, sess, model):
    array_X = char_str_idx(
        [[process_word_pos_entities(w) for w in string.split()]],
        model['word2idx'],
        2,
    )
    batch_x_char = generate_char_seq(
        array_X, model['idx2word'], model['char2idx']
    )
    Y_pred, Y_pos = sess.run(
        [model['crf_decode'], model['crf_decode_pos']],
        feed_dict = {
            model['word_ids']: array_X,
            model['char_ids']: batch_x_char,
        },
    )
    results = []
    for no, i in enumerate(string.split()):
        results.append(
            (
                i,
                model['idx2tag'][str(Y_pred[0, no])],
                model['idx2pos'][str(Y_pos[0, no])],
            )
        )
    return results


def get_available_pos_entities_models():
    return ['char', 'concat', 'attention']


def deep_pos_entities(model = 'attention'):
    assert isinstance(model, str), 'model must be a string'
    if model == 'char':
        if not os.path.isfile(char_settings):
            print('downloading POS-ENTITIES char settings')
            download_file('char-settings-pos-entities.json', char_settings)
        with open(char_settings, 'r') as fopen:
            nodes = json.loads(fopen.read())
        if not os.path.isfile(char_frozen):
            print('downloading POS-ENTITIES frozen char model')
            download_file('char-frozen-model-pos-entities.pb', char_frozen)
        g = load_graph(char_frozen)
        nodes['X'] = g.get_tensor_by_name('import/Placeholder:0')
        nodes['logits'] = g.get_tensor_by_name('import/logits:0')
        nodes['logits_pos'] = g.get_tensor_by_name('import/logits_pos:0')
        return DEEP_MODELS(
            nodes, tf.InteractiveSession(graph = g), get_entity_char
        )
    elif model == 'concat':
        if not os.path.isfile(concat_settings):
            print('downloading POS-ENTITIES concat settings')
            download_file('concat-settings-pos-entities.json', concat_settings)
        with open(concat_settings, 'r') as fopen:
            nodes = json.loads(fopen.read())
        if not os.path.isfile(concat_frozen):
            print('downloading POS-ENTITIES frozen concat model')
            download_file('concat-frozen-model-pos-entities.pb', concat_frozen)
        g = load_graph(concat_frozen)
        nodes['word_ids'] = g.get_tensor_by_name('import/Placeholder:0')
        nodes['char_ids'] = g.get_tensor_by_name('import/Placeholder_1:0')
        nodes['crf_decode'] = g.get_tensor_by_name(
            'import/entity-logits/cond/Merge:0'
        )
        nodes['crf_decode_pos'] = g.get_tensor_by_name(
            'import/pos-logits/cond/Merge:0'
        )
        nodes['idx2word'] = {int(k): v for k, v in nodes['idx2word'].items()}
        return DEEP_MODELS(
            nodes, tf.InteractiveSession(graph = g), get_entity_concat
        )
    elif model == 'attention':
        if not os.path.isfile(attention_settings):
            print('downloading POS-ENTITIES attention settings')
            download_file(
                'attention-settings-pos-entities.json', attention_settings
            )
        with open(attention_settings, 'r') as fopen:
            nodes = json.loads(fopen.read())
        if not os.path.isfile(attention_frozen):
            print('downloading POS-ENTITIES frozen attention model')
            download_file(
                'attention-frozen-model-pos-entities.pb', attention_frozen
            )
        g = load_graph(attention_frozen)
        nodes['word_ids'] = g.get_tensor_by_name('import/Placeholder:0')
        nodes['char_ids'] = g.get_tensor_by_name('import/Placeholder_1:0')
        nodes['crf_decode'] = g.get_tensor_by_name(
            'import/entity-logits/cond/Merge:0'
        )
        nodes['crf_decode_pos'] = g.get_tensor_by_name(
            'import/pos-logits/cond/Merge:0'
        )
        nodes['idx2word'] = {int(k): v for k, v in nodes['idx2word'].items()}
        return DEEP_MODELS(
            nodes, tf.InteractiveSession(graph = g), get_entity_concat
        )
    else:
        raise Exception('model not supported')
