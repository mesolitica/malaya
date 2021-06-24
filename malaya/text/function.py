import re
import os
import numpy as np
import itertools
import collections
from unidecode import unidecode
from itertools import permutations, combinations
from malaya.text.tatabahasa import (
    stopword_tatabahasa,
    stopwords,
    stopwords_calon,
    laughing,
    mengeluh,
)
from malaya.text.rules import normalized_chars
from malaya.text.english.words import words as _english_words
from malaya.text.bahasa.words import words as _malay_words
import json

STOPWORDS = set(stopwords + stopword_tatabahasa + stopwords_calon)
STOPWORD_CALON = set(stopwords_calon)
VOWELS = 'aeiou'
PHONES = ['sh', 'ch', 'ph', 'sz', 'cz', 'sch', 'rz', 'dz']
PUNCTUATION = '!"#$%&\'()*+,./:;<=>?@[\]^_`{|}~'
ENGLISH_WORDS = _english_words
MALAY_WORDS = _malay_words

alphabets = '([A-Za-z])'
prefixes = (
    '(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt|Puan|puan|Tuan|tuan|sir|Sir)[.]'
)
suffixes = '(Inc|Ltd|Jr|Sr|Co|Mo)'
starters = '(Mr|Mrs|Ms|Dr|He\\s|She\\s|It\\s|They\\s|Their\\s|Our\\s|We\\s|But\\s|However\\s|That\\s|This\\s|Wherever|Dia|Mereka|Tetapi|Kita|Itu|Ini|Dan|Kami|Beliau|Seri|Datuk|Dato|Datin|Tuan|Puan)'
acronyms = '([A-Z][.][A-Z][.](?:[A-Z][.])?)'
emails = r'(?:^|(?<=[^\w@.)]))(?:[\w+-](?:\.(?!\.))?)*?[\w+-]@(?:\w-?)*?\w+(?:\.(?:[a-z]{2,})){1,3}(?:$|(?=\b))'
websites = '[.](com|net|org|io|gov|me|edu|my)'
another_websites = '(www|http|https)[.]'
digits = '([0-9])'
before_digits = '([Nn]o|[Nn]ombor|[Nn]umber|[Kk]e|=|al|[Pp]ukul)'
month = '([Jj]an(?:uari)?|[Ff]eb(?:ruari)?|[Mm]a(?:c)?|[Aa]pr(?:il)?|Mei|[Jj]u(?:n)?|[Jj]ula(?:i)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]kt(?:ober)?|[Nn]ov(?:ember)?|[Dd]is(?:ember)?)'


def get_stopwords():
    return list(STOPWORDS)


def get_stopwords_calon():
    return list(STOPWORD_CALON)


def generate_compound(word):
    combs = {word}
    for i in range(1, len(word) + 1):
        for c in combinations(word, i):
            cs = ''.join(c)
            r = []

            for no in range(len(word)):
                for c in cs:

                    if word[no] == c:
                        p = c + c
                    else:
                        p = word[no]
                    r.append(p)

            s = ''.join(
                ''.join(s)[i - 1: i + 1]
                for _, s in itertools.groupby(''.join(r))
            )

            combs.add(s)

    combs.add(''.join([c + c for c in word]))
    return list(combs)


MENGELUH = []
for word in mengeluh:
    MENGELUH.extend(generate_compound(word))

MENGELUH = set([word for word in MENGELUH if len(word) > 2])


def isword_malay(word):
    if re.sub('[^0-9!@#$%\\^&*()-=_\\+{}\\[\\];\':",./<>? ]+', '', word) == word:
        return True
    if not any([c in VOWELS for c in word]):
        return False
    return True


def isword_english(word):
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
                subStr = word[idx - 2: idx + 1]
                if any(phone in subStr for phone in PHONES):
                    consecutiveConsonents -= 1
                    continue
                return False
    return True


def make_cleaning(s, c_dict):
    s = s.translate(c_dict)
    return s


def upperfirst(string):
    return string[0].upper() + string[1:]


def translation_textcleaning(string):
    return re.sub(r'[ ]+', ' ', unidecode(string)).strip()


def split_nya(string):
    string = re.sub(f'([{PUNCTUATION}])', r' \1 ', string)
    string = re.sub('\s{2,}', ' ', string)
    result = []
    for word in string.split():
        if word.endswith('nya'):
            result.extend([word[:-3], 'nya'])
        else:
            result.append(word)
    return ' '.join(result)


def transformer_textcleaning(string, space_after_punct=False):
    """
    use by any transformer model before tokenization
    """
    string = unidecode(string)
    string = ' '.join(
        [make_cleaning(w, normalized_chars) for w in string.split()]
    )
    string = re.sub('\\(dot\\)', '.', string)
    string = (
        re.sub(re.findall(r'\<a(.*?)\>', string)[0], '', string)
        if (len(re.findall(r'\<a (.*?)\>', string)) > 0)
        and ('href' in re.findall(r'\<a (.*?)\>', string)[0])
        else string
    )
    string = re.sub(
        r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', string
    )
    string = re.sub(r'[ ]+', ' ', string).strip().split()
    string = [w for w in string if w[0] != '@']
    string = ' '.join(string)
    if space_after_punct:
        string = re.sub(f'([{PUNCTUATION}])', r' \1 ', string)
        string = re.sub('\s{2,}', ' ', string)
    return string


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
        'http\\S+|www.\\S+',
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
    string = re.sub('[^\'"A-Za-z\\- ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string.lower()).strip()
    string = [word for word in string.lower().split() if isword_english(word)]
    string = [
        word
        for word in string
        if not any([laugh in word for laugh in laughing])
        and word[: len(word) // 2] != word[len(word) // 2:]
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
        'http\\S+|www.\\S+',
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
        if not any([laugh in word for laugh in laughing])
    ]
    string = ' '.join(string)
    return ''.join(''.join(s)[:2] for _, s in itertools.groupby(string))


def remove_newlines(string):
    string = string.replace('\n', ' ')
    string = re.sub(r'[ ]+', ' ', string).strip()
    return string


def question_cleaning(string):
    string = remove_newlines(string)
    string = upperfirst(string)
    return string


def simple_textcleaning(string, lowering=True):
    """
    use by topic modelling
    only accept A-Z, a-z
    """
    string = unidecode(string)
    string = re.sub('[^A-Za-z ]+', ' ', string)
    return re.sub(r'[ ]+', ' ', string.lower() if lowering else string).strip()


def entities_textcleaning(string, lowering=True):
    """
    use by entities recognition, pos recognition and dependency parsing
    """
    string = re.sub('[^A-Za-z0-9\\-() ]+', ' ', string)
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
    string = re.sub('[^A-Za-z0-9\\-\\/\'"\\.\\, ]+', ' ', unidecode(string))
    return original_string, re.sub(r'[ ]+', ' ', string.lower()).strip()


def get_hashtags(string):
    return [hash.lower() for hash in re.findall('#(\\w+)', string)]


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
        'http\\S+|www.\\S+',
        '',
        ' '.join(
            [i for i in string.split() if i.find('#') < 0 and i.find('@') < 0]
        ),
    )

    chars = ',.()!:\'"/;=-'
    for c in chars:
        string = string.replace(c, f' {c} ')
    string = string.replace('\n', '').replace('\t', '')

    string = re.sub(
        '[0-9!@#$%^&*()_\\-+{}|\\~`\'";:?/.>,<]', ' ', string, flags=re.UNICODE
    )
    string = re.sub(r'[ ]+', ' ', string).strip()

    return string.lower()


def augmentation_textcleaning(string):
    string = re.sub(
        'http\\S+|www.\\S+',
        '',
        ' '.join(
            [i for i in string.split() if i.find('#') < 0 and i.find('@') < 0]
        ),
    )
    chars = ',.()!:\'"/;=-'
    for c in chars:
        string = string.replace(c, f' {c} ')
    string = re.sub(
        '[0-9!@#$%^&*()_\\-+{}|\\~`\'";:?/.>,<]', ' ', string, flags=re.UNICODE
    )
    string = re.sub(r'[ ]+', ' ', string).strip()
    return string.lower()


def pos_entities_textcleaning(string):
    """
    use by text entities and pos
    remove links, hashtags, alias
    """
    string = re.sub(
        'http\\S+|www.\\S+',
        '',
        ' '.join(
            [i for i in string.split() if i.find('#') < 0 and i.find('@') < 0]
        ),
    )
    string = unidecode(string).replace('.', ' . ').replace(',', ' , ')
    string = re.sub('[^A-Za-z\\- ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string).strip()
    return ' '.join(
        [
            word.title() if word.isupper() else word
            for word in string.split()
            if len(word)
        ]
    )


def classification_textcleaning(string, no_stopwords=False, lowering=True):
    """
    stemmer, summarization, topic-modelling
    remove links, hashtags, alias
    """
    string = re.sub(
        'http\\S+|www.\\S+',
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
                for i in re.findall('[\\w\']+|[;:\\-\\(\\)&.,!?"]', string)
                if len(i)
            ]
        )
    else:
        string = ' '.join(
            [
                i
                for i in re.findall('[\\w\']+|[;:\\-\\(\\)&.,!?"]', string)
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


def summarization_textcleaning(string):
    return re.sub(r'[ ]+', ' ', string).strip()


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
    topics, feature_names, sorting, n_words=20, return_df=True
):
    if return_df:
        try:
            import pandas as pd
        except BaseException:
            raise ModuleNotFoundError(
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


def build_dataset(words, n_words, included_prefix=True):
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


def multireplace(string, replacements):
    substrs = sorted(replacements, key=len, reverse=True)
    regexp = re.compile('|'.join(map(re.escape, substrs)))
    return regexp.sub(lambda match: replacements[match.group(0)], string)


def case_of(text):
    return (
        str.upper
        if text.isupper()
        else str.lower
        if text.islower()
        else str.title
        if text.istitle()
        else str
    )


def replace_any(string, lists, replace_with):
    result = []
    for word in string.split():
        word_lower = word.lower()
        if any([e in word_lower for e in lists]):
            result.append(case_of(word)(replace_with))
        else:
            result.append(word)
    return ' '.join(result)


def replace_laugh(string, replace_with='haha'):
    return replace_any(string, laughing, replace_with)


def replace_mengeluh(string, replace_with='aduh'):
    return replace_any(string, MENGELUH, replace_with)


def split_into_sentences(text, minimum_length=5):
    """
    Sentence tokenizer.

    Parameters
    ----------
    text: str
    minimum_length: int, optional (default=5)
        minimum length to assume a string is a string, default 5 characters.

    Returns
    -------
    result: List[str]
    """

    def replace_sub(pattern, text):
        alls = re.findall(pattern, text)
        for a in alls:
            text = text.replace(a, a.replace('.', '<prd>'))
        return text

    text = text.replace('\x97', '\n')
    text = '. '.join([s for s in text.split('\n') if len(s)])
    text = text + '.'
    text = unidecode(text)
    text = ' ' + text + '  '
    text = text.replace('\n', ' ')
    text = re.sub(prefixes, '\\1<prd>', text)
    text = replace_sub(emails, text)
    text = re.sub(websites, '<prd>\\1', text)
    text = re.sub(another_websites, '\\1<prd>', text)
    text = re.sub('[,][.]+', '<prd>', text)
    if '...' in text:
        text = text.replace('...', '<prd><prd><prd>')
    if 'Ph.D' in text:
        text = text.replace('Ph.D.', 'Ph<prd>D<prd>')
    text = re.sub('[.]\\s*[,]', '<prd>,', text)
    text = re.sub(before_digits + '\\s*[.]\\s*' + digits, '\\1<prd>\\2', text)
    text = re.sub(month + '[.]\\s*' + digits, '\\1<prd>\\2', text)
    text = re.sub('\\s' + alphabets + '[.][ ]+', ' \\1<prd> ', text)
    text = re.sub(acronyms + ' ' + starters, '\\1<stop> \\2', text)
    text = re.sub(
        alphabets + '[.]' + alphabets + '[.]' + alphabets + '[.]',
        '\\1<prd>\\2<prd>\\3<prd>',
        text,
    )
    text = re.sub(
        alphabets + '[.]' + alphabets + '[.]', '\\1<prd>\\2<prd>', text
    )
    text = re.sub(' ' + suffixes + '[.][ ]+' + starters, ' \\1<stop> \\2', text)
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
    sentences = [s.strip() for s in sentences if len(s) > minimum_length]
    return sentences


def tag_chunk(seq):
    results = []
    last_no, last_label, tokens = 0, None, []
    for no in range(len(seq)):
        if last_label is None:
            tokens.append(seq[no][0])
            last_label = seq[no][1]
            last_no = no
        elif seq[no][1] == last_label:
            tokens.append(seq[no][0])
        else:
            tag = {
                'text': tokens,
                'type': last_label,
                'score': 1.0,
                'beginOffset': last_no,
                'endOffset': no,
            }
            results.append(tag)
            last_label = seq[no][1]
            last_no = no
            tokens = [seq[no][0]]

    if len(tokens):
        tag = {
            'text': tokens,
            'type': last_label,
            'score': 1.0,
            'beginOffset': last_no,
            'endOffset': no + 1,
        }
        results.append(tag)

    return results
