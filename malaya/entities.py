import pickle
import os
from .utils import download_file
from . import home
from .text_functions import entities_textcleaning

bow_pkl = home+'/bow-entities.pkl'
multinomial_pkl = home+'/multinomial-entities.pkl'

MULTINOMIAL, BOW = None, None

entities_labels = {0:'OTHER', 1:'law', 2:'location', 3:'organization',
4:'person', 5:'quantity', 6:'time'}

def multinomial_entities(string):
    assert (isinstance(string, str)), "input must be a string"
    global MULTINOMIAL, BOW
    string = entities_textcleaning(string)
    if MULTINOMIAL is None and BOW is None:
        if not os.path.isfile(bow_pkl):
            print('downloading pickled bag-of-word')
            download_file("https://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/bow-entities.pkl", bow_pkl)
        if not os.path.isfile(multinomial_pkl):
            print('downloading pickled multinomial model')
            download_file("http://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/multinomial-entities.pkl", multinomial_pkl)
        with open(bow_pkl,'rb') as fopen:
            BOW = pickle.load(fopen)
        with open(multinomial_pkl,'rb') as fopen:
            MULTINOMIAL = pickle.load(fopen)
    return [(string[no],entities_labels[i]) for no, i in enumerate(MULTINOMIAL.predict(BOW.transform(string)))]
