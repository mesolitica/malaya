import re
import tensorflow as tf
import numpy as np
from nltk.tokenize import word_tokenize
from .tatabahasa import tatabahasa_dict, hujung, permulaan
from . import home
from .utils import load_graph, download_file

char_settings = home+'/char-settings.json'
char_frozen = home+'/char_frozen_model.pb'
concat_settings = home+'/concat-settings.json'
concat_frozen = home+'/concat_frozen_model.pb'
attention_settings = home+'/attention-settings.json'
attention_frozen = home+'/attention_pos_frozen_model.pb'

class DEEP_MODELS:
    def __init__(self,nodes,sess,predict):
        self.nodes = nodes
        self.sess = sess
        self.__predict = predict
    def predict(self,string):
        assert (isinstance(string, str)), "input must be a string"
        return self.__predict(string,self.sess,self.nodes)

def process_word(word, lower=True):
    if lower:
        word = word.lower()
    else:
        if word.isupper():
            word = word.title()
    word = re.sub('[^A-Za-z0-9\- ]+', '', word)
    if word.isdigit():
        word = 'NUM'
    return word

def str_idx(corpus, dic, UNK=3):
    maxlen = max([len(i) for i in corpus])
    X = np.zeros((len(corpus),maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i][:maxlen][::-1]):
            try:
                X[i,-1 - no]=dic[k]
            except Exception as e:
                X[i,-1 - no]=UNK
    return X

def generate_char_seq(batch,idx2word,char2idx):
    x = [[len(idx2word[i]) for i in k] for k in batch]
    maxlen = max([j for i in x for j in i])
    temp = np.zeros((batch.shape[0],batch.shape[1],maxlen),dtype=np.int32)
    for i in range(batch.shape[0]):
        for k in range(batch.shape[1]):
            for no, c in enumerate(idx2word[batch[i,k]]):
                temp[i,k,-1-no] = char2idx[c]
    return temp

def get_entity_char(string,sess,model):
    batch_x = str_idx([process_word(w) for w in string.split()],model['char2idx'])
    logits, logits_pos = sess.run([tf.argmax(model['logits'],1),tf.argmax(model['logits_pos'],1)],feed_dict={model['X']:batch_x})
    results = []
    for no, i in enumerate(string.split()):
        results.append((i,model['idx2tag'][str(logits[no])],model['idx2pos'][str(logits_pos[no])]))
    return results

def get_entity_concat(string,sess,model):
    test_X = []
    for w in string.split():
        w = process_word(w)
        try:
            test_X.append(model['word2idx'][w])
        except:
            test_X.append(2)
    array_X = np.array([test_X])
    batch_x_char = generate_char_seq(array_X,model['idx2word'],model['char2idx'])
    Y_pred,Y_pos = sess.run([model['crf_decode'],model['crf_decode_pos']],feed_dict={model['word_ids']:array_X,
                                              model['char_ids']:batch_x_char})
    results = []
    for no, i in enumerate(string.split()):
        results.append((i,model['idx2tag'][str(Y_pred[0,no])],model['idx2pos'][str(Y_pos[0,no])]))
    return results

def deep_pos_entities(model='attention'):
    if model == 'char':
        if not os.path.isfile(char_settings):
            print('downloading char settings')
            download_file("https://raw.githubusercontent.com/DevconX/Malaya/master/data/char-settings.json", char_settings)
        with open(char_settings,'r') as fopen:
            nodes = json.loads(fopen.read())
        if not os.path.isfile(char_frozen):
            print('downloading frozen char model')
            download_file("https://raw.githubusercontent.com/DevconX/Malaya/master/data/char_frozen_model.pb", char_frozen)
        g=load_graph(char_frozen)
        nodes['X'] = g.get_tensor_by_name('import/Placeholder:0')
        nodes['logits'] = g.get_tensor_by_name('import/logits:0')
        nodes['logits_pos'] = g.get_tensor_by_name('import/logits_pos:0')
        return DEEP_MODELS(nodes,tf.InteractiveSession(graph=g),get_entity_char)
    elif model == 'concat':
        if not os.path.isfile(concat_settings):
            print('downloading concat settings')
            download_file("https://raw.githubusercontent.com/DevconX/Malaya/master/data/concat-settings.json", concat_settings)
        with open(concat_settings,'r') as fopen:
            nodes = json.loads(fopen.read())
        if not os.path.isfile(concat_frozen):
            print('downloading frozen concat model')
            download_file("https://raw.githubusercontent.com/DevconX/Malaya/master/data/concat_frozen_model.pb", concat_frozen)
        g=load_graph(concat_frozen)
        nodes['word_ids'] = g.get_tensor_by_name('import/Placeholder:0')
        nodes['char_ids'] = g.get_tensor_by_name('import/Placeholder_1:0')
        nodes['crf_decode'] = g.get_tensor_by_name('import/entity-logits/cond/Merge:0')
        nodes['crf_decode_pos'] = g.get_tensor_by_name('import/pos-logits/cond/Merge:0')
        nodes['idx2word'] = {int(k):v for k,v in nodes['idx2word'].items()}
        return DEEP_MODELS(nodes,tf.InteractiveSession(graph=g),get_entity_concat)
    elif model == 'attention':
        if not os.path.isfile(attention_settings):
            print('downloading attention settings')
            download_file("https://raw.githubusercontent.com/DevconX/Malaya/master/data/attention-settings.json", attention_settings)
        with open(attention_settings,'r') as fopen:
            nodes = json.loads(fopen.read())
        if not os.path.isfile(attention_frozen):
            print('downloading frozen attention model')
            download_file("https://raw.githubusercontent.com/DevconX/Malaya/master/data/attention_frozen_model.pb", attention_frozen)
        g=load_graph(attention_frozen)
        nodes['word_ids'] = g.get_tensor_by_name('import/Placeholder:0')
        nodes['char_ids'] = g.get_tensor_by_name('import/Placeholder_1:0')
        nodes['crf_decode'] = g.get_tensor_by_name('import/entity-logits/cond/Merge:0')
        nodes['crf_decode_pos'] = g.get_tensor_by_name('import/pos-logits/cond/Merge:0')
        nodes['idx2word'] = {int(k):v for k,v in nodes['idx2word'].items()}
        return DEEP_MODELS(nodes,tf.InteractiveSession(graph=g),get_entity_concat)
    else:
        raise Exception('model not supported')

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
