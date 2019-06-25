import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import re
import os
import numpy as np
import itertools
import collections
from unidecode import unidecode
from .._utils._utils import download_file
from ._tatabahasa import stopword_tatabahasa, stopwords, stopwords_calon
from ._english_words import _english_words
from .. import home
import json

STOPWORDS = set(stopwords + stopword_tatabahasa + stopwords_calon)
STOPWORD_CALON = set(stopwords_calon)
VOWELS = 'aeiou'
PHONES = ['sh', 'ch', 'ph', 'sz', 'cz', 'sch', 'rz', 'dz']
ENGLISH_WORDS = _english_words


def _isWord(word):
    if word:
        consecutiveVowels = 0
        consecutiveConsonents = 0
        for idx, letter in enumerate(word.lower()):
            vowel = True if letter in VOWELS else False
            if idx:
                prev = word[idx - 1]
                prevVowel = True if prev in VOWELS else False
                if not vowel and letter == 'y' and not prevVowel:
                    vowel = True
                if prevVowel != vowel:
                    consecutiveVowels = 0
                    consecutiveConsonents = 0
            if vowel:
                consecutiveVowels += 1
            else:
                consecutiveConsonents += 1
            if consecutiveVowels >= 3 or consecutiveConsonents > 3:
                return False
            if consecutiveConsonents == 3:
                subStr = word[idx - 2 : idx + 1]
                if any(phone in subStr for phone in PHONES):
                    consecutiveConsonents -= 1
                    continue
                return False
    return True


_list_laughing = {
    'huhu',
    'haha',
    'gaga',
    'hihi',
    'wkawka',
    'wkwk',
    'kiki',
    'keke',
    'huehue',
}


def malaya_textcleaning(string):
    """
    use by normalizer, spell
    remove links, hashtags, alias
    only accept A-Z, a-z
    remove any laugh
    remove any repeated char more than 2 times
    remove most of nonsense words
    """
    string = re.sub(
        'http\S+|www.\S+',
        '',
        ' '.join(
            [
                word
                for word in string.split()
                if word.find('#') < 0 and word.find('@') < 0
            ]
        ),
    )
    string = unidecode(string).replace('.', '. ').replace(',', ' , ')
    string = re.sub('[^\'"A-Za-z\- ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string.lower()).strip()
    string = [word for word in string.lower().split() if _isWord(word)]
    string = [
        word
        for word in string
        if not any([laugh in word for laugh in _list_laughing])
        and word[: len(word) // 2] != word[len(word) // 2 :]
    ]
    string = ' '.join(string)
    string = (
        ''.join(''.join(s)[:2] for _, s in itertools.groupby(string))
    ).split()
    return ' '.join([word for word in string if word not in STOPWORDS])


def normalizer_textcleaning(string):
    """
    use by normalizer, spell
    remove links, hashtags, alias
    only accept A-Z, a-z
    remove any laugh
    remove any repeated char more than 2 times
    """
    string = re.sub(
        'http\S+|www.\S+',
        '',
        ' '.join(
            [
                word
                for word in string.split()
                if word.find('#') < 0 and word.find('@') < 0
            ]
        ),
    )
    string = re.sub('[^A-Za-z ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string).strip()
    string = [
        word.title() if word.isupper() else word
        for word in string.split()
        if len(word)
    ]
    string = [
        word
        for word in string
        if not any([laugh in word for laugh in _list_laughing])
    ]
    string = ' '.join(string)
    return ''.join(''.join(s)[:2] for _, s in itertools.groupby(string))


def simple_textcleaning(string, lowering = True):
    """
    use by topic modelling
    only accept A-Z, a-z
    """
    string = unidecode(string)
    string = re.sub('[^A-Za-z ]+', ' ', string)
    return re.sub(r'[ ]+', ' ', string.lower() if lowering else string).strip()


def entities_textcleaning(string, lowering = True):
    """
    use by entities recognition, pos recognition and dependency parsing
    """
    string = re.sub('[^A-Za-z0-9\-\/ ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string).strip()
    original_string = string.split()
    if lowering:
        string = string.lower()
    string = [
        (original_string[no], word.title() if word.isupper() else word)
        for no, word in enumerate(string.split())
        if len(word)
    ]
    return [s[0] for s in string], [s[1] for s in string]


def summary_textcleaning(string):
    original_string = string
    string = re.sub('[^A-Za-z0-9\-\/\'"\.\, ]+', ' ', unidecode(string))
    return original_string, re.sub(r'[ ]+', ' ', string.lower()).strip()


def get_hashtags(string):
    return [hash.lower() for hash in re.findall('#(\w+)', string)]


def split_by_dot(string):
    string = re.sub(
        r'(?<!\d)\.(?!\d)',
        'SPLITTT',
        string.replace('\n', '').replace('/', ' '),
    )
    string = string.split('SPLITTT')
    return [re.sub(r'[ ]+', ' ', sentence).strip() for sentence in string]


def language_detection_textcleaning(string):
    string = re.sub(
        u'[0-9!@#$%^&*()_\-+{}|\~`\'";:?/.>,<]',
        ' ',
        string.lower(),
        flags = re.UNICODE,
    )
    return re.sub(r'[ ]+', ' ', string).strip()


def pos_entities_textcleaning(string):
    """
    use by text entities and pos
    remove links, hashtags, alias
    """
    string = re.sub(
        'http\S+|www.\S+',
        '',
        ' '.join(
            [i for i in string.split() if i.find('#') < 0 and i.find('@') < 0]
        ),
    )
    string = unidecode(string).replace('.', ' . ').replace(',', ' , ')
    string = re.sub('[^A-Za-z\- ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string).strip()
    return ' '.join(
        [
            word.title() if word.isupper() else word
            for word in string.split()
            if len(word)
        ]
    )


def classification_textcleaning(string, no_stopwords = False, lowering = True):
    """
    stemmer, summarization, topic-modelling
    remove links, hashtags, alias
    """
    string = re.sub(
        'http\S+|www.\S+',
        '',
        ' '.join(
            [i for i in string.split() if i.find('#') < 0 and i.find('@') < 0]
        ),
    )
    string = unidecode(string).replace('.', ' . ').replace(',', ' , ')
    string = re.sub('[^A-Za-z ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string).strip()
    if no_stopwords:
        string = ' '.join(
            [
                i
                for i in re.findall('[\\w\']+|[;:\-\(\)&.,!?"]', string)
                if len(i)
            ]
        )
    else:
        string = ' '.join(
            [
                i
                for i in re.findall('[\\w\']+|[;:\-\(\)&.,!?"]', string)
                if len(i) and i not in STOPWORDS
            ]
        )
    if lowering:
        return string.lower()
    else:
        return ' '.join(
            [
                word.title() if word.isupper() else word
                for word in string.split()
                if len(word)
            ]
        )


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


def print_topics_modelling(
    topics, feature_names, sorting, n_words = 20, return_df = True
):
    if return_df:
        try:
            import pandas as pd
        except:
            raise Exception(
                'pandas not installed. Please install it and try again or set `return_df = False`'
            )
    df = {}
    for i in range(topics):
        words = []
        for k in range(n_words):
            words.append(feature_names[sorting[i, k]])
        df['topic %d' % (i)] = words
    if return_df:
        return pd.DataFrame.from_dict(df)
    else:
        return df


def str_idx(corpus, dic, maxlen, UNK = 0):
    X = np.zeros((len(corpus), maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i].split()[:maxlen][::-1]):
            X[i, -1 - no] = dic.get(k, UNK)
    return X


def stemmer_str_idx(corpus, dic, UNK = 3):
    X = []
    for i in corpus:
        ints = []
        for k in i:
            ints.append(dic.get(k, UNK))
        X.append(ints)
    return X


def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(
            sentence + [pad_int] * (max_sentence_len - len(sentence))
        )
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens


def char_str_idx(corpus, dic, UNK = 2):
    maxlen = max([len(i) for i in corpus])
    X = np.zeros((len(corpus), maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i][:maxlen]):
            X[i, no] = dic.get(k, UNK)
    return X


def generate_char_seq(batch, dic, UNK = 2):
    maxlen_c = max([len(k) for k in batch])
    x = [[len(i) for i in k] for k in batch]
    maxlen = max([j for i in x for j in i])
    temp = np.zeros((len(batch), maxlen_c, maxlen), dtype = np.int32)
    for i in range(len(batch)):
        for k in range(len(batch[i])):
            for no, c in enumerate(batch[i][k][::-1]):
                temp[i, k, -1 - no] = dic.get(c, UNK)
    return temp


def build_dataset(words, n_words, included_prefix = True):
    count = (
        [['GO', 0], ['PAD', 1], ['EOS', 2], ['UNK', 3]]
        if included_prefix
        else []
    )
    count.extend(collections.Counter(words).most_common(n_words))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    for word in words:
        index = dictionary.get(word, 3)
        data.append(index)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


alphabets = '([A-Za-z])'
prefixes = '(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]'
suffixes = '(Inc|Ltd|Jr|Sr|Co)'
starters = '(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)'
acronyms = '([A-Z][.][A-Z][.](?:[A-Z][.])?)'
websites = '[.](com|net|org|io|gov|me|edu|my)'
another_websites = '(www|http|https)[.]'
digits = '([0-9])'


def split_into_sentences(text):
    text = unidecode(text)
    text = ' ' + text + '  '
    text = text.replace('\n', ' ')
    text = re.sub(prefixes, '\\1<prd>', text)
    text = re.sub(websites, '<prd>\\1', text)
    text = re.sub(another_websites, '\\1<prd>', text)
    if '...' in text:
        text = text.replace('...', '<prd><prd><prd>')
    if 'Ph.D' in text:
        text = text.replace('Ph.D.', 'Ph<prd>D<prd>')
    text = re.sub('\s' + alphabets + '[.] ', ' \\1<prd> ', text)
    text = re.sub(acronyms + ' ' + starters, '\\1<stop> \\2', text)
    text = re.sub(
        alphabets + '[.]' + alphabets + '[.]' + alphabets + '[.]',
        '\\1<prd>\\2<prd>\\3<prd>',
        text,
    )
    text = re.sub(
        alphabets + '[.]' + alphabets + '[.]', '\\1<prd>\\2<prd>', text
    )
    text = re.sub(' ' + suffixes + '[.] ' + starters, ' \\1<stop> \\2', text)
    text = re.sub(' ' + suffixes + '[.]', ' \\1<prd>', text)
    text = re.sub(' ' + alphabets + '[.]', ' \\1<prd>', text)
    text = re.sub(digits + '[.]' + digits, '\\1<prd>\\2', text)
    if '”' in text:
        text = text.replace('.”', '”.')
    if '"' in text:
        text = text.replace('."', '".')
    if '!' in text:
        text = text.replace('!"', '"!')
    if '?' in text:
        text = text.replace('?"', '"?')
    text = text.replace('.', '.<stop>')
    text = text.replace('?', '?<stop>')
    text = text.replace('!', '!<stop>')
    text = text.replace('<prd>', '.')
    sentences = text.split('<stop>')
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences if len(s) > 10]
    return sentences


def end_of_chunk(prev_tag, tag):
    if not len(prev_tag):
        return False
    if prev_tag != tag:
        return True


def start_of_chunk(prev_tag, tag):
    if not len(prev_tag):
        return True
    if prev_tag != tag:
        return False


def tag_chunk(seq):
    words = [i[0] for i in seq]
    seq = [i[1] for i in seq]
    prev_tag = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq):
        if end_of_chunk(prev_tag, chunk):
            chunks.append((prev_tag, begin_offset, i - 1))
            prev_tag = ''
        if start_of_chunk(prev_tag, chunk):
            begin_offset = i
        prev_tag = chunk
    res = {'words': words, 'tags': []}
    for chunk_type, chunk_start, chunk_end in chunks:
        tag = {
            'text': ' '.join(words[chunk_start : chunk_end + 1]),
            'type': chunk_type,
            'score': 1.0,
            'beginOffset': chunk_start,
            'endOffset': chunk_end,
        }
        res['tags'].append(tag)
    return res


def bert_tokenization(tokenizer, texts, maxlen):

    input_ids, input_masks, segment_ids = [], [], []
    for text in texts:
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a) > maxlen - 2:
            tokens_a = tokens_a[: (maxlen - 2)]
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_id = [0] * len(tokens)
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_id)
        padding = [0] * (maxlen - len(input_id))
        input_id += padding
        input_mask += padding
        segment_id += padding

        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)

    return input_ids, input_masks, segment_ids


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def bert_tokenization_siamese(tokenizer, left, right, maxlen):
    input_ids, input_masks, segment_ids = [], [], []
    for i in range(len(left)):
        tokens_a = tokenizer.tokenize(left[i])
        tokens_b = tokenizer.tokenize(right[i])
        _truncate_seq_pair(tokens_a, tokens_b, maxlen - 3)

        tokens = []
        segment_id = []
        tokens.append('[CLS]')
        segment_id.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_id.append(0)

        tokens.append('[SEP]')
        segment_id.append(0)
        for token in tokens_b:
            tokens.append(token)
            segment_id.append(1)
        tokens.append('[SEP]')
        segment_id.append(1)
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_id)

        while len(input_id) < maxlen:
            input_id.append(0)
            input_mask.append(0)
            segment_id.append(0)

        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)

    return input_ids, input_masks, segment_ids
