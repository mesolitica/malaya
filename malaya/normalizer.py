import numpy as np
from fuzzywuzzy import fuzz

class NAIVE_NORMALIZE:
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

def naive_normalizer(corpus):
    assert (isinstance(corpus, list) and isinstance(corpus[0], str)), 'input must be list of strings'
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
    return NAIVE_NORMALIZE(transform,corpus)

def basic_normalizer(string):
    assert (isinstance(string, str)), "input must be a string"
    result = []
    for i in string.lower().split():
        if i == 'x':
            result.append('tidak')
        elif i[-1] == '2':
            result.append(i[:-1]+'-'+i[:-1])
        else:
            result.append(i)
    return ' '.join(result)
