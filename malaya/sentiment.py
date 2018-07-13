import tensorflow as tf
import re
import numpy as np
import os
from pathlib import Path
import itertools
from urllib.request import urlretrieve
from unidecode import unidecode
import pickle

home = str(Path.home())+'/Malaya'
path_attention = home+'/attention_frozen_model.pb'
path_normal = home+'/normal_frozen_model.pb'

def textcleaning(string):
    string = re.sub('http\S+|www.\S+', '',' '.join([i for i in string.split() if i.find('#')<0 and i.find('@')<0]))
    string = unidecode(string).replace('.', '. ').replace(',', ', ')
    string = re.sub('[^\'\"A-Za-z\- ]+', '', string)
    return ' '.join([i for i in re.findall("[\\w']+|[;:\-\(\)&.,!?\"]", string) if len(i)>1]).lower()

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph

def str_idx(corpus, dic, maxlen, UNK=0):
    X = np.zeros((len(corpus),maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i].split()[:maxlen][::-1]):
            try:
                X[i,-1 - no]=dic[k]
            except Exception as e:
                X[i,-1 - no]=UNK
    return X

class deep_sentiment:
    def __init__(self,model='attention'):
        size = 256
        if not os.path.isfile('%s/word2vec-%d.p'%(home,size)):
            print('downloading word2vec-%d embedded'%(size))
            urlretrieve('http://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/word2vec-%d.p'%(size), '%s/word2vec-%d.p'%(home,size))
        with open('%s/word2vec-%d.p'%(home,size), 'rb') as fopen:
            self.embedded = pickle.load(fopen)
        if model == 'attention':
            if not os.path.isfile(path_attention):
                print('downloading frozen attention model')
                urlretrieve('http://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/attention_frozen_model.pb', path_attention)
            g=load_graph(path_attention)
            self.X = g.get_tensor_by_name('import/Placeholder:0')
            self.logits = g.get_tensor_by_name('import/logits:0')
            self.alphas = g.get_tensor_by_name('import/alphas:0')
            self.sess = tf.InteractiveSession(graph=g)
            self.mode = 'attention'
        elif model == 'normal':
            if not os.path.isfile(path_normal):
                print('downloading frozen normal model')
                urlretrieve('http://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/normal_frozen_model.pb', path_normal)
            g=load_graph(path_normal)
            self.X = g.get_tensor_by_name('import/Placeholder:0')
            self.logits = g.get_tensor_by_name('import/logits:0')
            self.sess = tf.InteractiveSession(graph=g)
            self.mode = 'normal'
        else:
            print('model not supported,exit.')
    def predict(self,string):
        string = textcleaning(string)
        splitted = string.split()
        batch_x = str_idx([string], self.embedded['dictionary'], len(splitted), UNK=0)
        if self.mode == 'attention':
            probs, alphas = self.sess.run([tf.nn.softmax(self.logits),self.alphas],feed_dict={self.X:batch_x})
            words = []
            for i in range(alphas.shape[1]):
                words.append([splitted[i],alphas[0,i]])
            return {'negative':probs[0,0],'positive':probs[0,1],'attention':words}
        if self.mode == 'normal':
            probs = self.sess.run(tf.nn.softmax(self.logits),feed_dict={self.X:batch_x})
            return {'negative':probs[0,0],'positive':probs[0,1]}
    def predict_batch(self,strings):
        if self.mode == 'attention':
            print('predict_batch only support normal model')
        if self.mode == 'normal':
            strings = [textcleaning(i) for i in strings]
            maxlen = [len(i.split()) for i in strings]
            batch_x = str_idx(strings, self.embedded['dictionary'], maxlen, UNK=0)
            probs = self.sess.run(tf.nn.softmax(self.logits),feed_dict={self.X:batch_x})
            dicts = []
            for i in range(probs.shape[0]):
                dicts.append({'negative':probs[i,0],'positive':probs[i,1]})
            return dicts
