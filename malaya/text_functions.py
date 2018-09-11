import re
import os
import numpy as np
import sklearn.datasets
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from unidecode import unidecode
import itertools
import random
import json
import tensorflow as tf
from .utils import download_file
from .tatabahasa import stopword_tatabahasa
from . import home

STOPWORDS = None
stopwords_location = home+'/stop-word-kerulnet'
if not os.path.isfile(stopwords_location):
    print('downloading stopwords')
    download_file("http://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/stop-word-kerulnet", stopwords_location)
with open(stopwords_location,'r') as fopen:
    STOPWORDS = list(filter(None, fopen.read().split('\n')))

STOPWORDS = list(set(STOPWORDS + stopword_tatabahasa))
VOWELS = "aeiou"
PHONES = ['sh', 'ch', 'ph', 'sz', 'cz', 'sch', 'rz', 'dz']

def isWord(word):
    if word:
        consecutiveVowels = 0
        consecutiveConsonents = 0
        for idx, letter in enumerate(word.lower()):
            vowel = True if letter in VOWELS else False
            if idx:
                prev = word[idx-1]
                prevVowel = True if prev in VOWELS else False
                if not vowel and letter == 'y' and not prevVowel:
                    vowel = True
                if prevVowel != vowel:
                    consecutiveVowels = 0
                    consecutiveConsonents = 0
            if vowel:
                consecutiveVowels += 1
            else:
                consecutiveConsonents +=1
            if consecutiveVowels >= 3 or consecutiveConsonents > 3:
                return False
            if consecutiveConsonents == 3:
                subStr = word[idx-2:idx+1]
                if any(phone in subStr for phone in PHONES):
                    consecutiveConsonents -= 1
                    continue
                return False
    return True

list_laughing = ['huhu','haha','gaga','hihi','wkawka','wkwk','kiki','keke','rt']
def malaya_textcleaning(string):
    string = re.sub('http\S+|www.\S+', '',' '.join([i for i in string.split() if i.find('#')<0 and i.find('@')<0]))
    string = unidecode(string).replace('.', '. ')
    string = string.replace(',', ', ')
    string = re.sub('[^\'\"A-Za-z\- ]+', '', unidecode(string))
    string = [y.strip() for y in word_tokenize(string.lower()) if isWord(y.strip())]
    string = [y for y in string if all([y.find(k) < 0 for k in list_laughing]) and y[:len(y)//2] != y[len(y)//2:]]
    string = ' '.join(string).lower()
    string = (''.join(''.join(s)[:2] for _, s in itertools.groupby(string))).split()
    return ' '.join([y for y in string if y not in STOPWORDS])

def simple_textcleaning(string):
    string = unidecode(string)
    string = re.sub('[^A-Za-z ]+', ' ', string)
    string = filter(None, string.split())
    string = [y.strip() for y in string if len(y) > 1]
    return ' '.join(string).lower()

def simple_textcleaning_language_detection(string):
    string = re.sub('[^A-Za-z ]+', ' ', string)
    string = filter(None, string.split())
    string = [y.strip() for y in string if len(y) > 1]
    return ' '.join(string).lower()

def deep_sentiment_textcleaning(string):
    string = re.sub('http\S+|www.\S+', '',' '.join([i for i in string.split() if i.find('#')<0 and i.find('@')<0]))
    string = unidecode(string).replace('.', '. ').replace(',', ', ')
    string = re.sub('[^\'\"A-Za-z\- ]+', '', string)
    return ' '.join([i for i in re.findall("[\\w']+|[;:\-\(\)&.,!?\"]", string) if len(i)>1 and i not in STOPWORDS]).lower()

def variant(word):
    word = word.lower()
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    return np.unique(deletes + transposes + replaces + inserts, return_counts=True)

def separate_dataset(trainset):
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split('\n')
        data_ = list(filter(None, data_))
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget
