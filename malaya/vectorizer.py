import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

from itertools import combinations
from toolz import compose
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
