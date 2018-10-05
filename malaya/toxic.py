import json
import pickle
import os
from .utils import download_file
from . import home
from keras.models import load_model
from scipy.sparse import hstack
from .keras_model import CLASSIFIER
from .text_functions import classification_textcleaning, str_idx

path_stack = home+'/stack-toxic.hdf5'
path_stack_setting = home+'/stack-toxic.json'
path_logistics = home+'/logistics-toxic.pkl'
path_multinomials = home+'/multinomials-toxic.pkl'
path_vectorizer_setting = home+'/vectorizer-toxic.pkl'
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

class TOXIC:
    def __init__(self,models,vectors):
        self._models = models
        self._vectors = vectors

    def _stack(self, strings):
        char_features = self._vectors['char'].transform(strings)
        word_features = self._vectors['word'].transform(strings)
        return hstack([char_features, word_features])

    def predict(self, string, get_proba=False):
        assert (isinstance(string, str)), "input must be a string"
        stacked = self._stack([classification_textcleaning(string,True)])
        result = {}
        for no, label in enumerate(class_names):
            if get_proba:
                result[label] = self._models[no].predict_proba(stacked)[0,1]
            else:
                result[label] = self._models[no].predict(stacked)[0]
        return result

    def predict_batch(self, strings, get_proba=False):
        assert (isinstance(strings, list) and isinstance(strings[0], str)), "input must be list of strings"
        stacked = self._stack([classification_textcleaning(i,True) for i in strings])
        result = {}
        for no, label in enumerate(class_names):
            if get_proba:
                result[label] = self._models[no].predict_proba(stacked)[:,1].tolist()
            else:
                result[label] = self._models[no].predict(stacked).tolist()
        return result

def multinomial_detect_toxic():
    if not os.path.isfile(path_multinomials):
        print('downloading TOXIC pickled multinomial model')
        download_file('v6/multinomials-toxic.pkl', path_multinomials)
    if not os.path.isfile(path_vectorizer_setting):
        print('downloading TOXIC pickled tfidf vectorizations')
        download_file('v6/vectorizer-toxic.pkl', path_vectorizer_setting)
    with open(path_multinomials,'rb') as fopen:
        multinomial = pickle.load(fopen)
    with open(path_vectorizer_setting,'rb') as fopen:
        vectorize = pickle.load(fopen)
    return TOXIC(multinomial, vectorize)

def logistics_detect_toxic():
    if not os.path.isfile(path_logistics):
        print('downloading TOXIC pickled logistics model')
        download_file('v6/logistics-toxic.pkl', path_logistics)
    if not os.path.isfile(path_vectorizer_setting):
        print('downloading TOXIC pickled tfidf vectorizations')
        download_file('v6/vectorizer-toxic.pkl', path_vectorizer_setting)
    with open(path_logistics,'rb') as fopen:
        logistic = pickle.load(fopen)
    with open(path_vectorizer_setting,'rb') as fopen:
        vectorize = pickle.load(fopen)
    return TOXIC(logistic, vectorize)

def deep_toxic():
    if not os.path.isfile(path_stack):
        print('downloading TOXIC frozen stack model')
        download_file('v6/stack-toxic.hdf5',path_stack)
    if not os.path.isfile(path_stack_setting):
        print('downloading TOXIC stack dictionary')
        download_file('v6/stack-toxic.json',path_stack_setting)
    with open(path_stack_setting,'r') as fopen:
        dictionary = json.load(fopen)['dictionary']
    return CLASSIFIER(load_model(path_stack),dictionary,class_names)
