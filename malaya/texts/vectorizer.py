import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

from itertools import combinations
from toolz import compose
import random
from sklearn.feature_extraction.text import CountVectorizer


class SkipGramVectorizer(CountVectorizer):
    def __init__(self, skip = 1, **kwds):
        super(SkipGramVectorizer, self).__init__(**kwds)
        self.skip = skip

    def build_sent_analyzer(self, preprocess, stop_words, tokenize):
        return lambda sent: self._word_skip_grams(
            compose(tokenize, preprocess, self.decode)(sent), stop_words
        )

    def build_analyzer(self):
        preprocess = self.build_preprocessor()
        stop_words = self.get_stop_words()
        tokenize = self.build_tokenizer()
        sent_analyze = self.build_sent_analyzer(
            preprocess, stop_words, tokenize
        )

        return lambda doc: self._sent_skip_grams(doc, sent_analyze)

    def _sent_skip_grams(self, doc, sent_analyze):
        skip_grams = []
        for sent in doc.split('\n'):
            skip_grams.extend(sent_analyze(sent))
        return skip_grams

    def _word_skip_grams(self, tokens, stop_words = None):
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]
        min_n, max_n = self.ngram_range
        skip = self.skip
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)
            tokens_append = tokens.append
            space_join = ' '.join

            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    head = [original_tokens[i]]
                    for skip_tail in combinations(
                        original_tokens[i + 1 : i + n + skip], n - 1
                    ):
                        tokens_append(space_join(head + list(skip_tail)))
        return tokens


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


def features_crf_dependency(sentence, tag, index):
    return {
        'word': sentence[index],
        'tag': tag[index],
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
