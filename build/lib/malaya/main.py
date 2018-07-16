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
import itertools
import random
import sys
import json
import tensorflow as tf
from pathlib import Path
from urllib.request import urlretrieve
from .tatabahasa import *
from .utils import *

home = str(Path.home())+'/Malaya'
stopwords_location = home+'/stop-word-kerulnet'
char_settings = home+'/char-settings.json'
char_frozen = home+'/char_frozen_model.pb'
concat_settings = home+'/concat-settings.json'
concat_frozen = home+'/concat_frozen_model.pb'
attention_settings = home+'/attention-settings.json'
attention_frozen = home+'/attention_pos_frozen_model.pb'
STOPWORDS = None

stopword_tatabahasa = list(set(tanya_list+perintah_list+pangkal_list+bantu_list+penguat_list+\
                penegas_list+nafi_list+pemeri_list+sendi_list+pembenar_list+nombor_list+\
                suku_bilangan_list+pisahan_list+keterangan_list+arah_list+hubung_list+gantinama_list))

LOC = os.path.dirname(os.path.abspath(__file__))

try:
    if not os.path.exists(home):
        os.makedirs(home)
except:
    print('cannot make directory for cache, exiting.')
    sys.exit(1)

if not os.path.isfile(stopwords_location):
    print('downloading stopwords')
    download_file("https://raw.githubusercontent.com/DevconX/Malaya/master/data/stop-word-kerulnet", stopwords_location)
with open(stopwords_location,'r') as fopen:
    STOPWORDS = list(filter(None, fopen.read().split('\n')))

class USER_BAYES:
    def __init__(self,multinomial,label,vectorize):
        self.multinomial = multinomial
        self.label = label
        self.vectorize = vectorize
    def predict(self, string):
        vectors = self.vectorize.transform([string])
        results = self.multinomial.predict_proba(vectors)[0]
        out = []
        for no, i in enumerate(self.label):
            out.append((i,results[no]))
        return out

class NORMALIZE:
    def __init__(self,user,corpus):
        self.user = user
        self.corpus = corpus
    def normalize(self,string):
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
        for i in range(len(self.user)):
            total = 0
            for k in self.user[i]: total += fuzz.ratio(string, k)
            results.append(total)
        if len(np.where(np.array(results) > 60)[0]) < 1:
            return original_string
        ids = np.argmax(results)
        return result_string + self.corpus[ids]

class DEEP_MODELS:
    def __init__(self,nodes,sess,predict):
        self.nodes = nodes
        self.sess = sess
        self.__predict = predict
    def predict(self,string):
        return self.__predict(string,self.sess,self.nodes)

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
def textcleaning(string):
    string = re.sub('http\S+|www.\S+', '',' '.join([i for i in string.split() if i.find('#')<0 and i.find('@')<0]))
    string = unidecode(string).replace('.', '. ')
    string = string.replace(',', ', ')
    string = re.sub('[^\'\"A-Za-z\- ]+', '', unidecode(string))
    string = [y.strip() for y in word_tokenize(string.lower()) if isWord(y.strip())]
    string = [y for y in string if all([y.find(k) < 0 for k in list_laughing]) and y[:len(y)//2] != y[len(y)//2:]]
    string = ' '.join(string).lower()
    string = (''.join(''.join(s)[:2] for _, s in itertools.groupby(string))).split()
    return ' '.join([y for y in string if y not in STOPWORDS])

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

def clearstring(string):
    string = unidecode(string)
    string = re.sub('[^A-Za-z ]+', '', string)
    string = word_tokenize(string)
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = ' '.join(string).lower()
    string = ''.join(''.join(s)[:2] for _, s in itertools.groupby(string))
    return ' '.join([i for i in string.split() if i not in STOPWORDS])

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

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph

def deep_learning(model='attention'):
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
    return NORMALIZE(transform,corpus)

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
    multinomial,labels,vectorize = None, None, None
    if vector.lower().find('tfidf') < 0 and vector.lower().find('bow'):
        raise Exception('Invalid vectorization technique')
    if isinstance(corpus, str):
        trainset = sklearn.datasets.load_files(container_path = corpus, encoding = 'UTF-8')
        trainset.data, trainset.target = separate_dataset(trainset)
        data, target = trainset.data, trainset.target
        labels = trainset.target_names
    if isinstance(corpus, list) or isinstance(corpus, tuple):
        corpus = np.array(corpus)
        data, target = corpus[:,0].tolist(),corpus[:,1].tolist()
        labels = np.unique(target).tolist()
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
        vectorize = TfidfVectorizer().fit(data)
        vectors = vectorize.transform(data)
    else:
        vectorize = CountVectorizer().fit(data)
        vectors = vectorize.transform(data)
    multinomial = MultinomialNB()
    if split:
        train_X, test_X, train_Y, test_Y = train_test_split(vectors, target, test_size = split)
        multinomial.partial_fit(train_X, train_Y,classes=np.unique(target))
        predicted = multinomial.predict(test_X)
        print(metrics.classification_report(test_Y, predicted, target_names = labels))
    else:
        multinomial.partial_fit(vectors,target,classes=np.unique(target))
        predicted = multinomial.predict(vectors)
        print(metrics.classification_report(target, predicted, target_names = labels))
    return USER_BAYES(multinomial,labels,vectorize)
