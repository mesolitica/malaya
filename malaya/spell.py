from collections import Counter, defaultdict
from fuzzywuzzy import fuzz
import numpy as np
from .text_functions import normalizer_textcleaning

alphabet = 'abcdefghijklmnopqrstuvwxyz'
consonants = 'bcdfghjklmnpqrstvwxyz'
vowels = 'aeiou'

def build_dicts(words):
    occurences = {}
    for l in alphabet:
        occurences[l] = defaultdict(lambda: 0)
    for word in words:
        occurences[word[0]][word] += 1
    return occurences

def edit_normalizer(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
    replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    pseudo = []
    for c in vowels:
        pseudo.append(''.join([w+c for w in word]))
    fuzziness = []
    try:
        if word[-1] == 'e':
            fuzziness.append(word[:-1]+'a')
    except:
        pass
    try:
        if word[-2:] == 'ak':
            fuzziness.append(word[:-1])
    except:
        pass
    try:
        if word[0] in consonants and word[1] in consonants and word[2] in vowels:
            fuzziness.append(word[0]+word[2]+word[1:])
    except:
        pass
    try:
        if word[0] in vowels and word[-1] in consonants and word[2] in consonants:
            fuzziness.append(word[:-2]+word[0]+word[-1])
    except:
        pass
    try:
        if word[-1] == 'o' and word[-3] in vowels and word[-2] in consonants:
            fuzziness.append(word[:-1]+'ar')
    except:
        pass
    try:
        if word[0] == 'a' and word[1] in consonants:
            fuzziness.append('h'+word)
    except:
        pass
    return set(deletes + transposes + replaces + inserts + fuzziness + pseudo)

def return_possible(word, dicts, edits):
    return set(e2 for e1 in edits(word) for e2 in edits(e1) if e2 in dicts)

def return_known(word, dicts):
    return set(w for w in word if w in dicts)

class SPELL:
    def __init__(self,corpus):
        self.corpus = corpus
        self.occurences = build_dicts(self.corpus)
        self.corpus = Counter(corpus)

    def correct(self, string, first_char=True,debug=True):
        assert (isinstance(string, str)) and not string.count(' '), "input must be a single word"
        assert len(string) > 1, "input must be long than 1 characters"
        string = normalizer_textcleaning(string)
        if not string:
            return string
        if first_char:
            candidates = return_known([string], self.occurences[string[0]]) or return_known(edit_normalizer(string), self.occurences[string[0]]) or return_possible(string, self.occurences[string[0]], edit_normalizer) or [string]
        else:
            candidates = return_known([string], self.corpus) or return_known(edit_normalizer(string), self.corpus) or return_possible(string, self.corpus, edit_normalizer) or [string]
        candidates = list(candidates)
        if debug:
            print([(k,fuzz.ratio(string, k)) for k in candidates])
        return candidates[np.argmax([fuzz.ratio(string, k) for k in candidates])]

def naive_speller(corpus):
    assert (isinstance(corpus, list) and isinstance(corpus[0], str)), 'input must be list of strings'
    return SPELL(corpus)
