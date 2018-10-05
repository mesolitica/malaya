import pickle
import os
import xgboost as xgb
import numpy as np
from .text_functions import simple_textcleaning
from .utils import download_file
from . import home

bow_pkl = home+'/bow-language-detection.pkl'
multinomial_pkl = home+'/multinomial-language-detection.pkl'

xgb_bow_pkl = home+'/bow-xgb-language-detection.pkl'
xgb_pkl = home+'/xgb-language-detection.pkl'

lang_labels = {0: 'OTHER',1: 'ENGLISH',2: 'INDONESIA',3: 'MALAY'}

def get_language_labels():
    return lang_labels

class USER_XGB:
    def __init__(self, xgb, label, vectorize):
        self.xgb = xgb
        self.label = label
        self.vectorize = vectorize
    def predict(self, string, get_proba=False):
        assert (isinstance(string, str)), "input must be a string"
        vectors = self.vectorize.transform([simple_textcleaning(string,True)])
        result = self.xgb.predict(xgb.DMatrix(vectors),ntree_limit=self.xgb.best_ntree_limit)[0]
        if get_proba:
            return {self.label[i]:result[i] for i in range(len(result))}
        else:
            return self.label[np.argmax(result)]
    def predict_batch(self, strings, get_proba=False):
        assert (isinstance(strings, list) and isinstance(strings[0], str)), "input must be list of strings"
        strings = [simple_textcleaning(string,True) for string in strings]
        vectors = self.vectorize.transform(strings)
        results = self.xgb.predict(xgb.DMatrix(vectors),ntree_limit=self.xgb.best_ntree_limit)
        if get_proba:
            outputs = []
            for result in results:
                outputs.append({self.label[i]:result[i] for i in range(len(result))})
            return outputs
        else:
            return [self.label[i] for i in np.argmax(results,axis=1)]

class USER_BAYES:
    def __init__(self,multinomial, label, vectorize):
        self.multinomial = multinomial
        self.label = label
        self.vectorize = vectorize
    def predict(self, string, get_proba=False):
        assert (isinstance(string, str)), "input must be a string"
        vectors = self.vectorize.transform([simple_textcleaning(string,True)])
        if get_proba:
            result = self.multinomial.predict_proba(vectors)[0]
            return {self.label[i]:result[i] for i in range(len(result))}
        else:
            return self.label[self.multinomial.predict(vectors)[0]]
    def predict_batch(self, strings, get_proba=False):
        assert (isinstance(strings, list) and isinstance(strings[0], str)), "input must be list of strings"
        strings = [simple_textcleaning(string,True) for string in strings]
        vectors = self.vectorize.transform(strings)
        if get_proba:
            results = self.multinomial.predict_proba(vectors)
            outputs = []
            for result in results:
                outputs.append({self.label[i]:result[i] for i in range(len(result))})
            return outputs
        else:
            return [self.label[result] for result in self.multinomial.predict(vectors)]

def multinomial_detect_languages():
    if not os.path.isfile(bow_pkl):
        print('downloading LANGUAGE-DETECTION pickled bag-of-word')
        download_file("bow-language-detection.pkl", bow_pkl)
    if not os.path.isfile(multinomial_pkl):
        print('downloading LANGUAGE-DETECTION pickled multinomial model')
        download_file("multinomial-language-detection.pkl", multinomial_pkl)
    with open(bow_pkl,'rb') as fopen:
        BOW = pickle.load(fopen)
    with open(multinomial_pkl,'rb') as fopen:
        MULTINOMIAL = pickle.load(fopen)
    return USER_BAYES(MULTINOMIAL, lang_labels, BOW)

def xgb_detect_languages():
    if not os.path.isfile(xgb_bow_pkl):
        print('downloading LANGUAGE-DETECTION pickled bag-of-word XGB')
        download_file("bow-xgboost-language-detection.pkl", xgb_bow_pkl)
    if not os.path.isfile(xgb_pkl):
        print('downloading LANGUAGE-DETECTION pickled XGB model')
        download_file("xgboost-language-detection.pkl", xgb_pkl)
    with open(xgb_bow_pkl,'rb') as fopen:
        BOW = pickle.load(fopen)
    with open(xgb_pkl,'rb') as fopen:
        XGB = pickle.load(fopen)
    return USER_XGB(XGB, lang_labels, BOW)
