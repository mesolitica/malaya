from nltk.tokenize import word_tokenize
import re
import pickle
from . import home
from .tatabahasa import tatabahasa_dict, hujung, permulaan
from .utils import download_file

def naive_POS_word(word):
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

def naive_pos(string):
    assert (isinstance(string, str)), "input must be a string"
    string = string.lower()
    results = []
    for i in word_tokenize(string):
        results.append(naive_POS_word(i))
    return results
