import re
import os
import numpy as np
import sklearn.datasets
from fuzzywuzzy import fuzz
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from unidecode import unidecode
import random
from .tatabahasa import *

stopword_tatabahasa = list(set(tanya_list+perintah_list+pangkal_list+bantu_list+penguat_list+\
                penegas_list+nafi_list+pemeri_list+sendi_list+pembenar_list+nombor_list+\
                suku_bilangan_list+pisahan_list+keterangan_list+arah_list+hubung_list+gantinama_list))

LOC = os.path.dirname(os.path.abspath(__file__))
#with open(LOC+'/stop-word-kerulnet','r') as fopen:
#    stopword_kerulnet = fopen.read().split()

class USER_BAYES:
    multinomial = None
    label = None
    vectorize = None

class NORMALIZE:
    user = None
    corpus = None

def tokenizer(string):
    return [word_tokenize(t) for t in sent_tokenize(s)]

def naive_POS(word):
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

def naive_POS_string(string):
    string = string.lower()
    results = []
    for i in word_tokenize(string):
        results.append(naive_POS(i))
    return results

def stemming(word):
    try:
        word = re.findall(r'^(.*?)(%s)$'%('|'.join(hujung)), word)[0][0]
        mula = re.findall(r'^(.*?)(%s)'%('|'.join(permulaan[::-1])), word)[0][1]
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

def train_normalize(corpus):
    if not isinstance(corpus, list) and not isinstance(corpus, tuple):
        raise Exception('a list or a tuple of word needed for the corpus')
    transform = []
    for i in corpus:
        i = i.lower()
        result = []
        result.append(''.join(char for char in i if char not in 'aeiou'))
        if i[-1] == 'a':
            result.append(i[:-1]+'e')
            result.append(i+'k')
        if i[-2:] == 'ar':
            result.append(i[:-2]+'o')
        if i[:2] == 'ha':
            result.append(i[1:])
        splitted_double = i.split('-')
        if len(splitted_double) > 1 and splitted_double[0] == splitted_double[1]:
            result.append(splitted_double[0]+'2')
        transform.append(result)
    NORMALIZE.user = transform
    NORMALIZE.corpus = corpus
    print('done train normalizer')

def user_normalize(string):
    if NORMALIZE.user is None or NORMALIZE.corpus is None:
        raise Exception('you need to train the normalizer first, train_normalize')
    original_string = string
    string = string.lower()
    if string[0] == 'x':
        if len(string) == 1:
            return 'tak'
        result_string = 'tak '
        string = string[1:]
    else:
        result_string = ''
    results = []
    for i in range(len(NORMALIZE.user)):
        total = 0
        for k in NORMALIZE.user[i]: total += fuzz.ratio(string, k)
        results.append(total)
    if len(np.where(np.array(results) > 60)[0]) < 1:
        return original_string
    ids = np.argmax(results)
    return result_string + NORMALIZE.corpus[ids]

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
        data, target = trainset.data, trainset.target
        labels = trainset.target_names
        USER_BAYES.label = labels
    if isinstance(corpus, list) or isinstance(corpus, tuple):
        corpus = np.array(corpus)
        data, target = corpus[:,0].tolist(),corpus[:,1].tolist()
        labels = np.unique(target).tolist()
        USER_BAYES.label = labels
        target = LabelEncoder().fit_transform(target)
    c = list(zip(data, target))
    random.shuffle(c)
    data, target = zip(*c)
    data, target = list(data), list(target)
    if stem:
        for i in range(len(data)): data[i] = ' '.join([stemming(k) for k in data[i].split()])
    if cleaning:
        for i in range(len(data)): data[i] = clearstring(data[i],tokenizing)
    if vector.lower().find('tfidf') >= 0:
        USER_BAYES.vectorize = TfidfVectorizer().fit(data)
        vectors = USER_BAYES.vectorize.transform(data)
    else:
        USER_BAYES.vectorize = CountVectorizer().fit(data)
        vectors = USER_BAYES.vectorize.transform(data)
    USER_BAYES.multinomial = MultinomialNB()
    if split:
        train_X, test_X, train_Y, test_Y = train_test_split(vectors, target, test_size = split)
        USER_BAYES.multinomial.partial_fit(train_X, train_Y,classes=np.unique(target))
        predicted = USER_BAYES.multinomial.predict(test_X)
        print(metrics.classification_report(test_Y, predicted, target_names = labels))
    else:
        USER_BAYES.multinomial.partial_fit(vectors,target,classes=np.unique(target))
        predicted = USER_BAYES.multinomial.predict(vectors)
        print(metrics.classification_report(target, predicted, target_names = labels))

def classify_bayes(string):
    if USER_BAYES.multinomial is None or USER_BAYES.vectorize is None:
        raise Exception('you need to train the classifier first, train_bayes')
    vectors = USER_BAYES.vectorize.transform([string])
    results = USER_BAYES.multinomial.predict_proba(vectors)[0]
    out = []
    for no, i in enumerate(USER_BAYES.label):
        out.append((i,results[no]))
    return out
