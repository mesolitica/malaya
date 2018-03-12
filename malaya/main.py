import re
import numpy as np
import sklearn.datasets
from fuzzywuzzy import fuzz
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from unidecode import unidecode
import random
from tatabahasa import *

stopword_tatabahasa = list(set(tanya_list+perintah_list+pangkal_list+bantu_list+penguat_list+\
                penegas_list+nafi_list+pemeri_list+sendi_list+pembenar_list+nombor_list+\
                suku_bilangan_list+pisahan_list+keterangan_list+arah_list+hubung_list+gantinama_list))

with open('stop-word-kerulnet','r') as fopen:
    stopword_kerulnet = fopen.read().split()
    
USER_BAYES = None
USER_NORMALIZE = None
VECTORIZE = None
    
def tokenizer(string):
    return [word_tokenize(t) for t in sent_tokenize(s)]

def naive_POS(word):
    for key, vals in tatabahasa_dict:
        if word in vals:
            return (key,word)
    try:
        if len(re.findall(r'^(.*?)(%s)$'%('|'.join(hujung[:1])), i)[0]) > 1:
            return ('KJ',word)
    except:
        return ('KN',word)
    try:
        if len(re.findall(r'^(.*?)(%s)'%('|'.join(permulaan[:-1])), word)[0]) > 1:
            return ('KJ',word)
    except:
        return ('KN',word)
    return ('KN',word)

def naive_POS_string(string):
    results = []
    for i in word_tokenize(string):
        results.append(naive_POS(i))
    return results

def stemming(word):
    try:
        word = re.findall(r'^(.*?)(%s)$'%('|'.join(hujung)), word)[0][0]
        mula = re.findall(r'^(.*?)(%s)'%('|'.join(permulaan)), word)[0][1]
        return word.replace(mula,'')
    except:
        return word
    
def clearstring(string,tokenize=True):
    string = unidecode(string)
    string = re.sub('^[ivxlcmIVXLCM]+','',string)
    string = re.sub('[^A-Za-z ]+', '', string)
    string = word_tokenize(string) if tokenize else string.split(' ')
    string = filter(None, string)
    string = [y.strip() for y in string]
    return ' '.join(string).lower()
    
def variant(word):
    word = word.lower()
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    return np.unique(deletes + transposes + replaces + inserts, return_counts=True)

def basic_normalize(string):
    result = []
    for i in string.lower().split():
        if i == 'x':
            result.append('tidak')
        elif i[-1] == '2':
            result.append(i[:-1]+'-'+i[:-1])
        else:
            result.append(i)
    return ' '.join(result)

#def train_normalize(corpus)

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

def train_bayes(corpus,tokenizing=True,cleaning=True,normalizing=True,stem=True,vector='tfidf',split=0.2):
    if vector.lower().find('tfidf') < 0 and vector.lower().find('bow'):
        raise Exception('Invalid vectorization technique')
    if isinstance(corpus, str):
        trainset = sklearn.datasets.load_files(container_path = corpus, encoding = 'UTF-8')
        trainset.data, trainset.target = separate_dataset(trainset)
    if isinstance(corpus, list) or isinstance(corpus, tuple):
        corpus = np.array(corpus)
        trainset.data, trainset.target = corpus[:,0].tolist(),corpus[:,1].tolist()
    c = list(zip(trainset.data, trainset.target))
    random.shuffle(c)
    trainset.data, trainset.target = zip(*c)
    if stem: 
        for i in range(len(trainset.data)): trainset.data[i] = ' '.join([stemming(k) for k in trainset.data[i].split()])
    if cleaning: 
        for i in range(len(trainset.data)): trainset.data[i] = clearstring(trainset.data[i],tokenizing)
    if vector.lower().find('tfidf') >= 0:
        VECTORIZE = TfidfVectorizer().fit(trainset.data)
        vectors = VECTORIZE.transform(trainset.data)
    else:
        VECTORIZE = CountVectorizer().fit(trainset.data)
        vectors = VECTORIZE.transform(trainset.data)
    USER_BAYES = MultinomialNB()
    if split:
        train_X, test_X, train_Y, test_Y = train_test_split(vectors, trainset.target, test_size = split)
        USER_BAYES.partial_fit(train_X, train_Y)
        predicted = USER_BAYES.predict(test_X)
        print(metrics.classification_report(test_Y, predicted, target_names = trainset.target_names))
    else:
        USER_BAYES.partial_fit(vectors, trainset.target)
        predicted = USER_BAYES.predict(vectors)
        print(metrics.classification_report(trainset.target, predicted, target_names = trainset.target_names))
        
def classify_bayes(string):
    if USER_BAYES is None or VECTORIZE is None:
        raise Exception('you need to train the classifier first, train_bayes')
    vectors = VECTORIZE.transform([string])
    return USER_BAYES.predict(vectors)[0]