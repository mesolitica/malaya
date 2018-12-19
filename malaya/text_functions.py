import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import re
import os
import numpy as np
import itertools
import collections
import random
from unidecode import unidecode
from .utils import download_file
from .tatabahasa import stopword_tatabahasa
from . import home

STOPWORDS = None
stopwords_location = home + '/stop-word-kerulnet'
if not os.path.isfile(stopwords_location):
    print('downloading stopwords')
    download_file('stop-word-kerulnet', stopwords_location)
with open(stopwords_location, 'r') as fopen:
    STOPWORDS = list(filter(None, fopen.read().split('\n')))

STOPWORDS = list(set(STOPWORDS + stopword_tatabahasa))
VOWELS = 'aeiou'
PHONES = ['sh', 'ch', 'ph', 'sz', 'cz', 'sch', 'rz', 'dz']


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


_list_laughing = [
    'huhu',
    'haha',
    'gaga',
    'hihi',
    'wkawka',
    'wkwk',
    'kiki',
    'keke',
    'huehue',
]


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
    string = unidecode(string).replace('.', '. ')
    string = string.replace(',', ', ')
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


def simple_textcleaning(string):
    """
    use by topic modelling
    only accept A-Z, a-z
    """
    string = unidecode(string)
    string = re.sub('[^A-Za-z ]+', ' ', string)
    return re.sub(r'[ ]+', ' ', string.lower()).strip()


def entities_textcleaning(string):
    """
    use by xgb entities, multinomial entities,
    xgb pos, xgb entities, char model, word model, concat model
    """
    string = re.sub('[^A-Za-z0-9\-\/ ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string).strip()
    return [
        word.title() if word.isupper() else word
        for word in string.split()
        if len(word)
    ]


def summary_textcleaning(string):
    string = re.sub('[^A-Za-z0-9\-\/\'"\.\, ]+', ' ', string)
    return re.sub(r'[ ]+', ' ', string.lower()).strip()


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


def cluster_pos(result):
    output = {
        'ADJ': [],
        'ADP': [],
        'ADV': [],
        'ADX': [],
        'CCONJ': [],
        'DET': [],
        'NOUN': [],
        'NUM': [],
        'PART': [],
        'PRON': [],
        'PROPN': [],
        'SCONJ': [],
        'SYM': [],
        'VERB': [],
        'X': [],
    }
    last_label, words = None, []
    for word, label in result:
        if last_label != label and last_label:
            joined = ' '.join(words)
            if joined not in output[last_label]:
                output[last_label].append(joined)
            words = []
            last_label = label
            words.append(word)

        else:
            if not last_label:
                last_label = label
            words.append(word)
    return output


def cluster_entities(result):
    output = {
        'OTHER': [],
        'law': [],
        'location': [],
        'organization': [],
        'person': [],
        'quantity': [],
        'time': [],
        'event': [],
    }
    last_label, words = None, []
    for word, label in result:
        if last_label != label and last_label:
            joined = ' '.join(words)
            if joined not in output[last_label]:
                output[last_label].append(joined)
            words = []
            last_label = label
            words.append(word)

        else:
            if not last_label:
                last_label = label
            words.append(word)
    return output


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
    string = unidecode(string).replace('.', '. ').replace(',', ', ')
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
    use by our text classifiers, stemmer, summarization, topic-modelling
    remove links, hashtags, alias
    """
    string = re.sub(
        'http\S+|www.\S+',
        '',
        ' '.join(
            [i for i in string.split() if i.find('#') < 0 and i.find('@') < 0]
        ),
    )
    string = unidecode(string).replace('.', '. ').replace(',', ', ')
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


def print_topics_modelling(
    topics, feature_names, sorting, topics_per_chunk = 6, n_words = 20
):
    for i in range(0, len(topics), topics_per_chunk):
        these_topics = topics[i : i + topics_per_chunk]
        len_this_chunk = len(these_topics)
        print(('topic {:<8}' * len_this_chunk).format(*these_topics))
        print(('-------- {0:<5}' * len_this_chunk).format(''))
        for i in range(n_words):
            print(
                ('{:<14}' * len_this_chunk).format(
                    *feature_names[sorting[these_topics, i]]
                )
            )
        print('\n')


def cluster_words(list_words):
    dict_words = {}
    for word in list_words:
        found = False
        for key in dict_words.keys():
            if word in key or any(
                [word in inside for inside in dict_words[key]]
            ):
                dict_words[key].append(word)
                found = True
            if key in word:
                dict_words[key].append(word)
        if not found:
            dict_words[word] = [word]
    results = []
    for key, words in dict_words.items():
        results.append(max(list(set([key] + words)), key = len))
    return list(set(results))


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


def fasttext_str_idx(texts, dictionary):
    idx_trainset = []
    for text in texts:
        idx = []
        for t in text.split():
            if t in dictionary:
                idx.append(dictionary[t])
        idx_trainset.append(idx)
    return idx_trainset


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


def pad_sequence(
    sequence,
    n,
    pad_left = False,
    pad_right = False,
    left_pad_symbol = None,
    right_pad_symbol = None,
):
    sequence = iter(sequence)
    if pad_left:
        sequence = itertools.chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = itertools.chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(
    sequence,
    n,
    pad_left = False,
    pad_right = False,
    left_pad_symbol = None,
    right_pad_symbol = None,
):
    sequence = pad_sequence(
        sequence, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol
    )

    history = []
    while n > 1:
        try:
            next_item = next(sequence)
        except StopIteration:
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def add_ngram(sequences, token_indice, ngram = (2, 3)):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(ngram[0], ngram[1]):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i : i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences


def generate_ngram(
    result_pos,
    result_entities,
    ngram = (1, 3),
    accept_pos = ['NOUN', 'PROPN', 'VERB'],
    accept_entities = ['law', 'location', 'organization', 'person', 'time'],
):
    words = []
    sentences = []
    for no in range(len(result_pos)):
        if (
            result_pos[no][1] in accept_pos
            or result_entities[no][1] in accept_entities
        ):
            words.append(result_pos[no][0])
    for gram in range(ngram[0], ngram[1] + 1, 1):
        gram_words = list(ngrams(words, gram))
        for sentence in gram_words:
            sentences.append(' '.join(sentence))
    return list(set(sentences))


def sentence_ngram(sentence, ngram = (1, 3)):
    words = sentence.split()
    sentences = []
    for gram in range(ngram[0], ngram[1] + 1, 1):
        gram_words = list(ngrams(words, gram))
        for sentence in gram_words:
            sentences.append(' '.join(sentence))
    return list(set(sentences))


def char_str_idx(corpus, dic, UNK = 0):
    maxlen = max([len(i) for i in corpus])
    X = np.zeros((len(corpus), maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i][:maxlen][::-1]):
            X[i, -1 - no] = dic.get(k, UNK)
    return X


def generate_char_seq(batch, idx2word, char2idx):
    x = [[len(idx2word[i]) for i in k] for k in batch]
    maxlen = max([j for i in x for j in i])
    temp = np.zeros((batch.shape[0], batch.shape[1], maxlen), dtype = np.int32)
    for i in range(batch.shape[0]):
        for k in range(batch.shape[1]):
            for no, c in enumerate(idx2word[batch[i, k]].lower()):
                temp[i, k, -1 - no] = char2idx[c]
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


def features_crf(sentence, index):
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'prev_word-prefix-1': '' if index == 0 else sentence[index - 1][0],
        'prev_word-prefix-2': '' if index == 0 else sentence[index - 1][:2],
        'prev_word-prefix-3': '' if index == 0 else sentence[index - 1][:3],
        'prev_word-suffix-1': '' if index == 0 else sentence[index - 1][-1],
        'prev_word-suffix-2': '' if index == 0 else sentence[index - 1][-2:],
        'prev_word-suffix-3': '' if index == 0 else sentence[index - 1][-3:],
        'next_word-prefix-1': ''
        if index == len(sentence) - 1
        else sentence[index + 1][0],
        'next_word-prefix-2': ''
        if index == len(sentence) - 1
        else sentence[index + 1][:2],
        'next_word-prefix-3': ''
        if index == len(sentence) - 1
        else sentence[index + 1][:3],
        'next_word-suffix-1': ''
        if index == len(sentence) - 1
        else sentence[index + 1][-1],
        'next_word-suffix-2': ''
        if index == len(sentence) - 1
        else sentence[index + 1][-2:],
        'next_word-suffix-3': ''
        if index == len(sentence) - 1
        else sentence[index + 1][-3:],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
    }


def most_common(l):
    return max(set(l), key = l.count)


def voting_stack(models, text):
    assert isinstance(models, list), 'models must be a list'
    assert isinstance(text, str), 'vectorizer must be a string'
    results, texts, votes = [], [], []
    for i in range(len(models)):
        assert 'predict' in dir(models[i]), 'all models must able to predict'
        predicted = np.array(models[i].predict(text))
        results.append(predicted[:, 1:2])
        texts.append(predicted[:, 0])
    concatenated = np.concatenate(results, axis = 1)
    for row in concatenated:
        votes.append(most_common(row.tolist()))
    return list(map(lambda X: (X[0], X[1]), list(zip(texts[-1], votes))))


def skipgrams(
    sequence,
    vocabulary_size,
    window_size = 4,
    negative_samples = 1.0,
    shuffle = True,
    categorical = False,
    sampling_table = None,
    seed = None,
):
    couples = []
    labels = []
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

        window_start = max(0, i - window_size)
        window_end = min(len(sequence), i + window_size + 1)
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                couples.append([wi, wj])
                if categorical:
                    labels.append([0, 1])
                else:
                    labels.append(1)

    if negative_samples > 0:
        num_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)

        couples += [
            [words[i % len(words)], random.randint(1, vocabulary_size - 1)]
            for i in range(num_negative_samples)
        ]
        if categorical:
            labels += [[1, 0]] * num_negative_samples
        else:
            labels += [0] * num_negative_samples

    if shuffle:
        if seed is None:
            seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)

    return couples, labels
