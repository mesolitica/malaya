from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from .text_functions import separate_dataset, deep_sentiment_textcleaning
from .stemmer import naive_stemmer
from sklearn.cross_validation import train_test_split
from sklearn import metrics, datasets
import tensorflow as tf
import re
import numpy as np
import os
import itertools
from unidecode import unidecode
import pickle
import random
from .language_detection import USER_XGB, USER_BAYES
from .utils import download_file, load_graph
from . import home

path_attention = home+'/attention_frozen_model.pb'
path_bahdanau = home+'/bahdanau_frozen_model.pb'
path_luong = home+'/luong_frozen_model.pb'
path_normal = home+'/normal_frozen_model.pb'
bayes_location = home + '/bayes-news.pkl'
tfidf_location = home + '/tfidf-news.pkl'
xgb_location = home + '/xgboost-sentiment.pkl'
xgb_tfidf_location = home + '/xgboost-tfidf.pkl'

def str_idx(corpus, dic, maxlen, UNK=0):
    X = np.zeros((len(corpus),maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i].split()[:maxlen][::-1]):
            try:
                X[i,-1 - no]=dic[k]
            except Exception as e:
                X[i,-1 - no]=UNK
    return X

def get_available_sentiment_models():
    return ['bahdanau','attention','luong','normal']

class deep_sentiment:
    def __init__(self,model='bahdanau'):
        size = 256
        if not os.path.isfile('%s/word2vec-%d.p'%(home,size)):
            print('downloading SENTIMENT word2vec-%d embedded'%(size))
            download_file('word2vec-%d.p'%(size),'%s/word2vec-%d.p'%(home,size))
        with open('%s/word2vec-%d.p'%(home,size), 'rb') as fopen:
            self.embedded = pickle.load(fopen)
        if model == 'attention':
            if not os.path.isfile(path_attention):
                print('downloading SENTIMENT frozen attention model')
                download_file('attention_frozen_model.pb',path_attention)
            g=load_graph(path_attention)
            self.X = g.get_tensor_by_name('import/Placeholder:0')
            self.logits = g.get_tensor_by_name('import/logits:0')
            self.alphas = g.get_tensor_by_name('import/alphas:0')
            self.sess = tf.InteractiveSession(graph=g)
            self.mode = 'attention'
        elif model == 'bahdanau':
            if not os.path.isfile(path_bahdanau):
                print('downloading SENTIMENT frozen bahdanau model')
                download_file('bahdanau_frozen_model.pb',path_bahdanau)
            g=load_graph(path_bahdanau)
            self.X = g.get_tensor_by_name('import/Placeholder:0')
            self.logits = g.get_tensor_by_name('import/logits:0')
            self.alphas = g.get_tensor_by_name('import/alphas:0')
            self.sess = tf.InteractiveSession(graph=g)
            self.mode = 'bahdanau'
        elif model == 'luong':
            if not os.path.isfile(path_luong):
                print('downloading SENTIMENT frozen luong model')
                download_file('luong_frozen_model.pb',path_luong)
            g=load_graph(path_luong)
            self.X = g.get_tensor_by_name('import/Placeholder:0')
            self.logits = g.get_tensor_by_name('import/logits:0')
            self.alphas = g.get_tensor_by_name('import/alphas:0')
            self.sess = tf.InteractiveSession(graph=g)
            self.mode = 'luong'
        elif model == 'normal':
            if not os.path.isfile(path_normal):
                print('downloading SENTIMENT frozen normal model')
                download_file('normal_frozen_model.pb',path_normal)
            g=load_graph(path_normal)
            self.X = g.get_tensor_by_name('import/Placeholder:0')
            self.logits = g.get_tensor_by_name('import/logits:0')
            self.sess = tf.InteractiveSession(graph=g)
            self.mode = 'normal'
        else:
            raise Exception('model sentiment not supported')
    def predict(self,string):
        assert (isinstance(string, str)), "input must be a string"
        string = deep_sentiment_textcleaning(string)
        splitted = string.split()
        batch_x = str_idx([string], self.embedded['dictionary'], len(splitted), UNK=0)
        if self.mode == 'attention':
            probs, alphas = self.sess.run([tf.nn.softmax(self.logits),self.alphas],feed_dict={self.X:batch_x})
            words = []
            for i in range(alphas.shape[1]):
                words.append([splitted[i],alphas[0,i]])
            return {'negative':probs[0,0],'positive':probs[0,1],'attention':words}
        if self.mode == 'luong' or self.mode == 'bahdanau':
            probs, alphas = self.sess.run([tf.nn.softmax(self.logits),self.alphas],feed_dict={self.X:batch_x})
            words = []
            for i in range(alphas.shape[0]):
                words.append([splitted[i],alphas[i]])
            return {'negative':probs[0,0],'positive':probs[0,1],'attention':words}
        if self.mode == 'normal':
            probs = self.sess.run(tf.nn.softmax(self.logits),feed_dict={self.X:batch_x})
            return {'negative':probs[0,0],'positive':probs[0,1]}
    def predict_batch(self,strings):
        assert (isinstance(strings, list) and isinstance(strings[0], str)), "input must be list of strings"
        strings = [deep_sentiment_textcleaning(i) for i in strings]
        maxlen = max([len(i.split()) for i in strings])
        batch_x = str_idx(strings, self.embedded['dictionary'], maxlen, UNK=0)
        probs = self.sess.run(tf.nn.softmax(self.logits),feed_dict={self.X:batch_x})
        dicts = []
        for i in range(probs.shape[0]):
            dicts.append({'negative':probs[i,0],'positive':probs[i,1]})
        return dicts

def bayes_sentiment(
                corpus,
                cleaning = True,
                stemming = True,
                vector = 'tfidf',
                split_size = 0.2, **kwargs):
    multinomial,labels,vectorize = None, None, None
    if vector.lower().find('tfidf') < 0 and vector.lower().find('bow') < 0:
        raise Exception('Invalid vectorization technique')
    if isinstance(corpus, str):
        trainset = datasets.load_files(container_path = corpus, encoding = 'UTF-8')
        trainset.data, trainset.target = separate_dataset(trainset)
        data, target = trainset.data, trainset.target
        labels = trainset.target_names
    if isinstance(corpus, list) or isinstance(corpus, tuple):
        corpus = np.array(corpus)
        data, target = corpus[:,0].tolist(), corpus[:,1].tolist()
        labels = np.unique(target).tolist()
        target = LabelEncoder().fit_transform(target)
    c = list(zip(data, target))
    random.shuffle(c)
    data, target = zip(*c)
    data, target = list(data), list(target)
    if stemming:
        for i in range(len(data)): data[i] = ' '.join([naive_stemmer(k) for k in data[i].split()])
    if cleaning:
        for i in range(len(data)): data[i] = deep_sentiment_textcleaning(data[i])
    if vector.lower().find('tfidf') >= 0:
        vectorize = TfidfVectorizer(**kwargs).fit(data)
        vectors = vectorize.transform(data)
    else:
        vectorize = CountVectorizer(**kwargs).fit(data)
        vectors = vectorize.transform(data)
    multinomial = MultinomialNB(**kwargs)
    if split_size:
        train_X, test_X, train_Y, test_Y = train_test_split(vectors, target, test_size = split_size)
        multinomial.partial_fit(train_X, train_Y,classes=np.unique(target))
        predicted = multinomial.predict(test_X)
        print(metrics.classification_report(test_Y, predicted, target_names = labels))
    else:
        multinomial.partial_fit(vectors,target,classes=np.unique(target))
        predicted = multinomial.predict(vectors)
        print(metrics.classification_report(target, predicted, target_names = labels))
    return USER_BAYES(multinomial, labels, vectorize)

def pretrained_bayes_sentiment():
    if not os.path.isfile(bayes_location):
        print('downloading SENTIMENT pickled multinomial model')
        download_file("multinomial-sentiment-news.pkl", bayes_location)
    if not os.path.isfile(tfidf_location):
        print('downloading SENTIMENT pickled tfidf vectorizations')
        download_file("tfidf-news.pkl", tfidf_location)
    with open(bayes_location,'rb') as fopen:
        multinomial = pickle.load(fopen)
    with open(tfidf_location,'rb') as fopen:
        vectorize = pickle.load(fopen)
    return USER_BAYES(multinomial, ['negative','positive'], vectorize)

def pretrained_xgb_sentiment():
    if not os.path.isfile(xgb_location):
        print('downloading SENTIMENT pickled XGB model')
        download_file("xgboost-sentiment.pkl", xgb_location)
    if not os.path.isfile(xgb_tfidf_location):
        print('downloading SENTIMENT pickled tfidf vectorizations')
        download_file("xgboost-tfidf.pkl", xgb_tfidf_location)
    with open(xgb_location,'rb') as fopen:
        xgb = pickle.load(fopen)
    with open(xgb_tfidf_location,'rb') as fopen:
        vectorize = pickle.load(fopen)
    return USER_XGB(xgb, ['negative','positive'], vectorize)
