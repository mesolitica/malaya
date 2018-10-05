import re
import os
import numpy as np
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import itertools
from .utils import download_file
from .tatabahasa import stopword_tatabahasa
from . import home

STOPWORDS = None
stopwords_location = home+'/stop-word-kerulnet'
if not os.path.isfile(stopwords_location):
    print('downloading stopwords')
    download_file("stop-word-kerulnet", stopwords_location)
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

list_laughing = ['huhu','haha','gaga','hihi','wkawka','wkwk','kiki','keke','huehue']
def malaya_textcleaning(string):
    string = re.sub('http\S+|www.\S+', '',' '.join([i for i in string.split() if i.find('#')<0 and i.find('@')<0]))
    string = unidecode(string).replace('.', '. ')
    string = string.replace(',', ', ')
    string = re.sub('[^\'\"A-Za-z\- ]+', ' ', string)
    string = [y.strip() for y in word_tokenize(string.lower()) if isWord(y.strip())]
    string = [y for y in string if all([y.find(k) < 0 for k in list_laughing]) and y[:len(y)//2] != y[len(y)//2:]]
    string = ' '.join(string)
    string = (''.join(''.join(s)[:2] for _, s in itertools.groupby(string))).split()
    return ' '.join([y for y in string if y not in STOPWORDS])

def normalizer_textcleaning(string):
    string = re.sub('http\S+|www.\S+', '',' '.join([i for i in string.split() if i.find('#')<0 and i.find('@')<0]))
    string = re.sub('[^A-Za-z ]+', ' ', string)
    string = [y.strip() for y in string.lower().split() if len(y.strip())]
    string = [y for y in string if all([y.find(k) < 0 for k in list_laughing])]
    string = ' '.join(string).lower()
    return ''.join(''.join(s)[:2] for _, s in itertools.groupby(string))

def simple_textcleaning(string, decode=False):
    '''
    use by language_detection, topic modelling
    only accept A-Z, a-z
    '''
    string = unidecode(string) if decode else string
    string = re.sub('[^A-Za-z ]+', ' ', string)
    string = [y.strip() for y in string if len(y)]
    return ' '.join(string).lower()

def entities_textcleaning(string):
    string = re.sub('[^A-Za-z0-9\-\/ ]+', ' ', string).split()
    return [y.strip() for y in string if len(y)]

def summary_textcleaning(string):
    string = re.sub('[^A-Za-z0-9\-\/\'\"\.\, ]+', ' ', string).split()
    return ' '.join([y.strip() for y in string if len(y)])

def classification_textcleaning(string,no_stopwords=False):
    '''
    use by our text classifiers, stemmer, summarization, topic-modelling
    remove links, hashtags, alias
    '''
    string = re.sub('http\S+|www.\S+', '',' '.join([i for i in string.split() if i.find('#')<0 and i.find('@')<0]))
    string = unidecode(string).replace('.', '. ').replace(',', ', ')
    string = re.sub('[^A-Za-z\- ]+', ' ', string)
    if no_stopwords:
        return ' '.join([i for i in re.findall("[\\w']+|[;:\-\(\)&.,!?\"]", string) if len(i)>1]).lower()
    else:
        return ' '.join([i for i in re.findall("[\\w']+|[;:\-\(\)&.,!?\"]", string) if len(i)>1 and i not in STOPWORDS]).lower()

def process_word_pos_entities(word):
    word = word.lower()
    word = re.sub('[^A-Za-z0-9\- ]+', '', word)
    if word.isdigit():
        word = 'NUM'
    return word

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

def print_topics_modelling(topics, feature_names, sorting, topics_per_chunk=6, n_words=20):
    for i in range(0, len(topics), topics_per_chunk):
        these_topics = topics[i: i + topics_per_chunk]
        len_this_chunk = len(these_topics)
        print(("topic {:<8}" * len_this_chunk).format(*these_topics))
        print(("-------- {0:<5}" * len_this_chunk).format(""))
        for i in range(n_words):
            print(("{:<14}" * len_this_chunk).format(*feature_names[sorting[these_topics, i]]))
        print("\n")

def cluster_words(list_words):
    dict_words = {}
    for word in list_words:
        found = False
        for key in dict_words.keys():
            if word in key or any([word in inside for inside in dict_words[key]]):
                dict_words[key].append(word)
                found = True
            if key in word:
                dict_words[key].append(word)
        if not found:
            dict_words[word] = [word]
    results = []
    for key, words in dict_words.items():
        results.append(max(list(set([key] + words)), key=len))
    return list(set(results))

def str_idx(corpus, dic, maxlen, UNK=0):
    X = np.zeros((len(corpus),maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i].split()[:maxlen][::-1]):
            try:
                X[i,-1 - no]=dic[k]
            except Exception as e:
                X[i,-1 - no]=UNK
    return X

def stemmer_str_idx(corpus, dic, UNK=3):
    X = []
    for i in corpus:
        ints = []
        for k in i:
            try:
                ints.append(dic[k])
            except Exception as e:
                ints.append(UNK)
        X.append(ints)
    return X

def fasttext_str_idx(texts, dictionary):
    idx_trainset = []
    for text in texts:
        idx = []
        for t in text.split():
            try:
                idx.append(dictionary[t])
            except:
                pass
        idx_trainset.append(idx)
    return idx_trainset

def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens

def add_ngram(sequences, token_indice):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, 3):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences

def char_str_idx(corpus, dic, UNK=0):
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
            for no, c in enumerate(idx2word[batch[i,k]].lower()):
                temp[i,k,-1-no] = char2idx[c]
    return temp
