from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import metrics, datasets
from keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import json
from unidecode import unidecode
import pickle
import random
from .text_functions import separate_dataset, classification_textcleaning, str_idx, add_ngram, fasttext_str_idx
from .stemmer import naive_stemmer
from .language_detection import USER_XGB, USER_BAYES
from .utils import download_file, load_graph
from . import home
from .keras_model import CLASSIFIER

path_stack = home+'/stack-sentiment.hdf5'
path_stack_setting = home+'/stack-sentiment.json'
path_hierarchical = home+'/hierarchical-sentiment.pb'
path_hierarchical_setting = home+'/hierarchical-sentiment.json'
path_bahdanau = home+'/bahdanau-sentiment.pb'
path_bahdanau_setting = home+'/bahdanau-sentiment.json'
path_luong = home+'/luong-sentiment.pb'
path_luong_setting = home+'/luong-sentiment.json'
path_bidirectional = home+'/bidirectional-sentiment.pb'
path_bidirectional_setting = home+'/bidirectional-sentiment.json'
path_fasttext = home+'/fasttext-sentiment.pb'
path_fasttext_setting = home+'/fasttext-sentiment.json'
path_fasttext_pickle = home+'/fasttext-sentiment.pkl'
bayes_location = home + '/bayes-news.pkl'
tfidf_location = home + '/tfidf-news.pkl'
xgb_location = home + '/xgboost-sentiment.pkl'
xgb_tfidf_location = home + '/xgboost-tfidf.pkl'

class SENTIMENT:
    def __init__(self, X, logits, sess, mode, dictionary, ngram=None, alphas=None):
        self._X = X
        self._logits = logits
        self._sess = sess
        self._mode = mode
        self._dictionary = dictionary
        self._ngram = ngram
        self._alphas = alphas

    def predict(self,string):
        assert (isinstance(string, str)), 'input must be a string'
        string = classification_textcleaning(string,True)
        splitted = string.split()
        if self._mode == 'fast-text':
            batch_x = fasttext_str_idx([string], self._dictionary)
            batch_x = add_ngram(batch_x, self._ngram)
        else:
            batch_x = str_idx([string], self._dictionary, len(splitted), UNK=3)
        if self._mode in ['luong', 'bahdanau', 'hierarchical']:
            probs, alphas = self._sess.run([tf.nn.softmax(self._logits),self._alphas],feed_dict={self._X:batch_x})
            if self._mode == 'hierarchical':
                alphas = alphas[0]
            words = []
            for i in range(alphas.shape[0]):
                words.append([splitted[i],alphas[i]])
            return {'negative':probs[0,0],'positive':probs[0,1],'attention':words}
        if self._mode in ['bidirectional','fast-text']:
            probs = self._sess.run(tf.nn.softmax(self._logits),feed_dict={self._X:batch_x})
            return {'negative':probs[0,0],'positive':probs[0,1]}

    def predict_batch(self,strings):
        assert (isinstance(strings, list) and isinstance(strings[0], str)), 'input must be list of strings'
        strings = [classification_textcleaning(i,True) for i in strings]
        maxlen = max([len(i.split()) for i in strings])
        if self._mode == 'fast-text':
            batch_x = fasttext_str_idx(strings, self._dictionary)
            batch_x = add_ngram(batch_x, self._ngram)
            batch_x = tf.keras.preprocessing.sequence.pad_sequences(batch_x, maxlen)
        else:
            batch_x = str_idx(strings, self._dictionary, maxlen, UNK=3)
        probs = self._sess.run(tf.nn.softmax(self._logits),feed_dict={self._X:batch_x})
        dicts = []
        for i in range(probs.shape[0]):
            dicts.append({'negative':probs[i,0],'positive':probs[i,1]})
        return dicts

def get_available_sentiment_models():
    return ['bahdanau','hierarchical','luong','bidirectional','fast-text','stack']

def deep_sentiment(model='luong'):
    if model == 'stack':
        if not os.path.isfile(path_stack):
            print('downloading SENTIMENT frozen stack model')
            download_file('v6/stack-sentiment.hdf5',path_stack)
        if not os.path.isfile(path_stack_setting):
            print('downloading SENTIMENT stack dictionary')
            download_file('v6/stack-sentiment.json',path_stack_setting)
        with open(path_stack_setting,'r') as fopen:
            dictionary = json.load(fopen)['dictionary']
        return CLASSIFIER(load_model(path_stack),dictionary,['negative','positive'])
    elif model == 'fast-text':
        if not os.path.isfile(path_fasttext):
            print('downloading SENTIMENT frozen fast-text model')
            download_file('v5/fasttext-sentiment.pb',path_fasttext)
        if not os.path.isfile(path_fasttext_setting):
            print('downloading SENTIMENT fast-text dictionary')
            download_file('v5/fasttext-sentiment.json',path_fasttext_setting)
        if not os.path.isfile(path_fasttext_pickle):
            print('downloading SENTIMENT fast-text pickle')
            download_file('v5/fasttext-sentiment.pkl',path_fasttext_pickle)
        with open(path_fasttext_setting,'r') as fopen:
            dictionary = json.load(fopen)['dictionary']
        with open(path_fasttext_pickle, 'rb') as fopen:
            ngram = pickle.load(fopen)
        g=load_graph(path_fasttext)
        return SENTIMENT(g.get_tensor_by_name('import/Placeholder:0'),
                         g.get_tensor_by_name('import/logits:0'),
                         tf.InteractiveSession(graph=g),
                         model,
                         dictionary,
                         ngram=ngram)
    elif model == 'hierarchical':
        if not os.path.isfile(path_hierarchical):
            print('downloading SENTIMENT frozen hierarchical model')
            download_file('v5/hierarchical-sentiment.pb',path_hierarchical)
        if not os.path.isfile(path_hierarchical_setting):
            print('downloading SENTIMENT hierarchical dictionary')
            download_file('v5/hierarchical-sentiment.json',path_hierarchical_setting)
        with open(path_hierarchical_setting,'r') as fopen:
            dictionary = json.load(fopen)['dictionary']
        g=load_graph(path_hierarchical)
        return SENTIMENT(g.get_tensor_by_name('import/Placeholder:0'),
                         g.get_tensor_by_name('import/logits:0'),
                         tf.InteractiveSession(graph=g),
                         model,
                         dictionary,
                         alphas=g.get_tensor_by_name('import/alphas:0'))
    elif model == 'bahdanau':
        if not os.path.isfile(path_bahdanau):
            print('downloading SENTIMENT frozen bahdanau model')
            download_file('v5/bahdanau-sentiment.pb',path_bahdanau)
        if not os.path.isfile(path_bahdanau_setting):
            print('downloading SENTIMENT bahdanau dictionary')
            download_file('v5/bahdanau-sentiment.json',path_bahdanau_setting)
        with open(path_bahdanau_setting,'r') as fopen:
            dictionary = json.load(fopen)['dictionary']
        g=load_graph(path_bahdanau)
        return SENTIMENT(g.get_tensor_by_name('import/Placeholder:0'),
                         g.get_tensor_by_name('import/logits:0'),
                         tf.InteractiveSession(graph=g),
                         model,
                         dictionary,
                         alphas=g.get_tensor_by_name('import/alphas:0'))
    elif model == 'luong':
        if not os.path.isfile(path_luong):
            print('downloading SENTIMENT frozen luong model')
            download_file('v5/luong-sentiment.pb',path_luong)
        if not os.path.isfile(path_luong_setting):
            print('downloading SENTIMENT luong dictionary')
            download_file('v5/luong-sentiment.json',path_luong_setting)
        with open(path_luong_setting,'r') as fopen:
            dictionary = json.load(fopen)
        g=load_graph(path_luong)
        return SENTIMENT(g.get_tensor_by_name('import/Placeholder:0'),
                         g.get_tensor_by_name('import/logits:0'),
                         tf.InteractiveSession(graph=g),
                         model,
                         dictionary,
                         alphas=g.get_tensor_by_name('import/alphas:0'))
    elif model == 'bidirectional':
        if not os.path.isfile(path_bidirectional):
            print('downloading SENTIMENT frozen bidirectional model')
            download_file('v5/bidirectional-sentiment.pb',path_bidirectional)
        if not os.path.isfile(path_bidirectional_setting):
            print('downloading SENTIMENT bidirectional dictionary')
            download_file('v5/bidirectional-sentiment.json',path_bidirectional_setting)
        with open(path_bidirectional_setting,'r') as fopen:
            dictionary = json.load(fopen)
        g=load_graph(path_bidirectional)
        return SENTIMENT(g.get_tensor_by_name('import/Placeholder:0'),
                         g.get_tensor_by_name('import/logits:0'),
                         tf.InteractiveSession(graph=g),
                         model,
                         dictionary)
    else:
        raise Exception('model sentiment not supported')

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
        for i in range(len(data)): data[i] = classification_textcleaning(data[i],True)
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
        download_file('v5/multinomial-sentiment.pkl', bayes_location)
    if not os.path.isfile(tfidf_location):
        print('downloading SENTIMENT pickled tfidf vectorizations')
        download_file('v5/tfidf-multinomial-sentiment.pkl', tfidf_location)
    with open(bayes_location,'rb') as fopen:
        multinomial = pickle.load(fopen)
    with open(tfidf_location,'rb') as fopen:
        vectorize = pickle.load(fopen)
    return USER_BAYES(multinomial, ['negative','positive'], vectorize)

def pretrained_xgb_sentiment():
    if not os.path.isfile(xgb_location):
        print('downloading SENTIMENT pickled XGB model')
        download_file('v5/xgboost-sentiment.pkl', xgb_location)
    if not os.path.isfile(xgb_tfidf_location):
        print('downloading SENTIMENT pickled tfidf vectorizations')
        download_file('v5/tfidf-xgboost-sentiment.pkl', xgb_tfidf_location)
    with open(xgb_location,'rb') as fopen:
        xgb = pickle.load(fopen)
    with open(xgb_tfidf_location,'rb') as fopen:
        vectorize = pickle.load(fopen)
    return USER_XGB(xgb, ['negative','positive'], vectorize)
