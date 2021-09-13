from collections import Counter, defaultdict
from itertools import product
import tensorflow as tf
import numpy as np
import json
import re
from malaya.text.function import case_of, ENGLISH_WORDS, MALAY_WORDS
from malaya.text.tatabahasa import (
    alphabet,
    consonants,
    vowels,
    permulaan,
    hujung,
    stopword_tatabahasa,
    quad_vowels,
    group_compound,
)
from malaya.text.rules import rules_normalizer
from malaya.text.bpe import SentencePieceTokenizer
from malaya.path import PATH_NGRAM, S3_PATH_NGRAM
from malaya.function import check_file
from malaya.supervised import t5 as t5_load
from malaya.model.t5 import Spell as T5_Spell
from herpetologist import check_type

_transformer_availability = {
    'small-t5': {
        'Size (MB)': 355.6,
        'Quantized Size (MB)': 195,
        'WER': 0.0156248,
        'Suggested length': 256,
    },
    'tiny-t5': {
        'Size (MB)': 208,
        'Quantized Size (MB)': 103,
        'WER': 0.023712,
        'Suggested length': 256,
    },
    'super-tiny-t5': {
        'Size (MB)': 81.8,
        'Quantized Size (MB)': 27.1,
        'WER': 0.038001,
        'Suggested length': 256,
    },
}


def tokens_to_masked_ids(tokens, mask_ind, tokenizer):
    masked_tokens = tokens[:]
    masked_tokens[mask_ind] = '[MASK]'
    masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
    masked_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    return masked_ids


def generate_ids(mask, tokenizer):
    tokens = tokenizer.tokenize(mask)
    input_ids = [
        tokens_to_masked_ids(tokens, i, tokenizer) for i in range(len(tokens))
    ]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, input_ids, tokens_ids


def _build_dicts(words):
    occurences = {}
    for l in alphabet:
        occurences[l] = defaultdict(lambda: 0)
    for word in words:
        occurences[word[0]][word] += 1
    return occurences


def _get_indices(string, c='a'):
    return [i for i in range(len(string)) if string[i] == c]


def _permutate(string, indices):
    p = [''.join(_set) for _set in product(list(vowels), repeat=len(indices))]
    p = [p_ for p_ in p if not all([a in p_ for a in quad_vowels])]
    mutate = []
    for p_ in p:
        s = list(string[:])
        for i in range(len(indices)):
            s[indices[i]] = p_[i]
        mutate.append(''.join(s))
    return mutate


def _permutate_sp(string, indices, sp_tokenizer):
    p = [''.join(_set) for _set in product(list(vowels), repeat=len(indices))]
    p = [p_ for p_ in p if not all([a in p_ for a in quad_vowels])]
    mutate = []
    for p_ in p:
        s = list(string[:])
        for i in range(len(indices)):
            s[indices[i]] = p_[i]
        s = ''.join(s)
        if sp_tokenizer.tokenize(s)[0] == '▁':
            continue
        mutate.append(s)
    return mutate


def _augment_vowel_alternate(string):
    """
    malaya.spell._augment_vowel_alternate('sngpore')
    -> ('sangapor', 'sangapora')

    malaya.spell._augment_vowel_alternate('kmpung')
    -> ('kmpung', 'kmpunga')

    malaya.spell._augment_vowel_alternate('aym')
    -> ('ayam', 'ayama')
    """
    r = []
    # a flag to not duplicate
    last_time = False
    for i, c in enumerate(string[:-1], 1):
        last = i - 2
        if last < 0:
            last = 0

        # we only want to put a vowel after consonant if next that consonant if not a wovel
        if c in consonants and string[i] not in vowels:
            if c + string[i] in group_compound and not last_time:
                r.append(c + string[i])
                last_time = True
            elif string[last] + c in group_compound and not last_time:
                r.append(string[last] + c)
                last_time = True
            else:
                last_time = False
                if len(r):
                    # ['ng'] gg
                    if (
                        r[-1] in group_compound
                        and c + string[i] == r[-1][-1] * 2
                    ):
                        r.append('^')
                        continue
                    elif r[-1] in group_compound and c == r[-1][-1]:
                        if c + string[i] in group_compound:
                            continue
                        else:
                            r.append('a')
                            continue
                r.append(c + 'a')

        else:
            if len(r):
                if r[-1] in group_compound and c == r[-1][-1]:
                    continue
            r.append(c)

    if len(r):

        if r[-1][-1] in vowels and string[-1] in consonants:
            r.append(string[-1])

        elif (
            r[-1] in group_compound
            and string[-2] in vowels
            and string[-1] in consonants
        ):
            r.append(string[-2:])

    left = ''.join(r).replace('^', '')
    right = left + 'a'
    return left, right


def _augment_vowel_prob(word, **kwargs):
    l, r = _augment_vowel_alternate(word)
    return list(
        set(_permutate(l, _get_indices(l)) + _permutate(r, _get_indices(r)))
    )


def _augment_vowel_prob_sp(word, sp_tokenizer):
    l, r = _augment_vowel_alternate(word)
    return list(
        set(
            _permutate_sp(l, _get_indices(l), sp_tokenizer)
            + _permutate_sp(r, _get_indices(r), sp_tokenizer)
        )
    )


def _augment_vowel(
    string, selected=['a', 'u', 'i', 'e'], included_end=True
):
    pseudo = []
    if included_end:
        end = string[-1]
    else:
        end = ''
    for c in selected:
        pseudo.append(''.join([w + c for w in string[:-1]] + [end]))
    return pseudo


def _return_possible(word, dicts, edits):
    return set(e2 for e1 in edits(word) for e2 in edits(e1) if e2 in dicts)


def _return_known(word, dicts):
    return set(w for w in word if w in dicts)


class Spell:
    def __init__(self, sp_tokenizer, corpus, add_norvig_method=True):
        self._sp_tokenizer = sp_tokenizer
        if self._sp_tokenizer:
            self._augment = _augment_vowel_prob_sp
        else:
            self._augment = _augment_vowel_prob
        self._add_norvig_method = add_norvig_method
        self._corpus = corpus
        self.WORDS = Counter(self._corpus)
        self.N = sum(self.WORDS.values())

    def edit_step(self, word):
        """
        All edits that are one edit away from `word`.
        """
        if self._add_norvig_method:
            splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
            deletes = [L + R[1:] for L, R in splits if R]
            transposes = [
                L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1
            ]
            inserts = [L + c + R for L, R in splits for c in alphabet]
        pseudo = _augment_vowel(word)
        pseudo.extend(self._augment(word, sp_tokenizer=self._sp_tokenizer))
        fuzziness = []
        if len(word):

            # berape -> berapa, mne -> mna
            if word[-1] == 'e':
                inner = word[:-1] + 'a'
                fuzziness.append(inner)
                pseudo.extend(
                    self._augment(inner, sp_tokenizer=self._sp_tokenizer)
                )

            # pikir -> fikir
            if word[0] == 'p':
                inner = 'f' + word[1:]
                fuzziness.append(inner)
                pseudo.extend(
                    self._augment(inner, sp_tokenizer=self._sp_tokenizer)
                )

        if len(word) > 2:
            # bapak -> bapa, mintak -> minta, mntak -> mnta
            if word[-2:] == 'ak':
                inner = word[:-1]
                fuzziness.append(word[:-1])
                pseudo.extend(
                    self._augment(word[:-1], sp_tokenizer=self._sp_tokenizer)
                )

            # hnto -> hantar, bako -> bkar, sabo -> sabar
            if word[-1] == 'o' and word[-2] in consonants:
                inner = word[:-1] + 'ar'
                fuzziness.append(inner)
                pseudo.extend(
                    self._augment(inner, sp_tokenizer=self._sp_tokenizer)
                )

            # antu -> hantu, antar -> hantar
            if word[0] == 'a' and word[1] in consonants:
                inner = 'h' + word
                fuzziness.append(inner)
                pseudo.extend(_augment_vowel(inner))
                pseudo.extend(
                    self._augment(inner, sp_tokenizer=self._sp_tokenizer)
                )

            # ptg -> ptng, dtg -> dtng
            if (
                word[-3] in consonants
                and word[-2] in consonants
                and word[-1] == 'g'
            ):
                inner = word[:-1] + 'ng'
                fuzziness.append(inner)
                pseudo.extend(
                    self._augment(inner, sp_tokenizer=self._sp_tokenizer)
                )

            # igt -> ingt
            if word[1] == 'g' and word[2] in consonants:
                inner = word[0] + 'n' + word[1:]
                fuzziness.append(inner)
                pseudo.extend(
                    self._augment(inner, sp_tokenizer=self._sp_tokenizer)
                )

        if self._add_norvig_method:
            return set(deletes + transposes + inserts + fuzziness + pseudo)
        else:
            return set(fuzziness + pseudo)

    def edits2(self, word):
        """
        All edits that are two edits away from `word`.
        """
        return (e2 for e1 in self.edit_step(word) for e2 in self.edit_step(e1))

    def known(self, words):
        """
        The subset of `words` that appear in the dictionary of WORDS.
        """
        return set(w for w in words if w in self.WORDS)

    def edit_candidates(self, word):
        """
        Generate candidates given a word.

        Parameters
        ----------
        word: str

        Returns
        -------
        result: List[str]
        """

        ttt = self.known(self.edit_step(word)) or {word}
        ttt = {i for i in ttt if len(i) > 3 and i not in ENGLISH_WORDS}
        ttt = self.known([word]) | ttt
        if not len(ttt):
            ttt = {word}
        return list(ttt)

    @check_type
    def correct_text(self, text: str):
        """
        Correct all the words within a text, returning the corrected text.

        Parameters
        ----------
        text: str

        Returns
        -------
        result: str
        """

        return re.sub('[a-zA-Z]+', self.correct_match, text)

    def correct_match(self, match: str):
        """
        Spell-correct word in match, and preserve proper upper/lower/title case.

        Parameters
        ----------
        match: str

        Returns
        -------
        result: str
        """

        word = match.group()
        if word[0].isupper():
            return word
        return case_of(word)(self.correct(word.lower()))

    @check_type
    def correct_word(self, word: str):
        """
        Spell-correct word in match, and preserve proper upper/lower/title case.

        Parameters
        ----------
        word: str

        Returns
        -------
        result: str
        """

        return case_of(word)(self.correct(word.lower()))


class Probability(Spell):
    """
    The SpellCorrector extends the functionality of the Peter Norvig's
    spell-corrector in http://norvig.com/spell-correct.html
    And improve it using some algorithms from Normalization of noisy texts in Malaysian online reviews,
    https://www.researchgate.net/publication/287050449_Normalization_of_noisy_texts_in_Malaysian_online_reviews
    Added custom vowels augmentation
    """

    def __init__(self, corpus, sp_tokenizer=None):
        Spell.__init__(self, sp_tokenizer, corpus)

    def tokens(text):
        return REGEX_TOKEN.findall(text.lower())

    def P(self, word):
        """
        Probability of `word`.
        """
        return self.WORDS[word] / self.N

    def most_probable(self, words):
        _known = self.known(words)
        if _known:
            return max(_known, key=self.P)
        else:
            return []

    @check_type
    def correct(self, word: str, **kwargs):
        """
        Most probable spelling correction for word.

        Parameters
        ----------
        word: str

        Returns
        -------
        result: str
        """

        if word in ENGLISH_WORDS:
            return word
        if self._corpus.get(word, 0) > 5000:
            return word
        if word in MALAY_WORDS:
            return word
        if word in stopword_tatabahasa:
            return word

        cp_word = word[:]
        hujung_result = [v for k, v in hujung.items() if word.endswith(k)]
        if len(hujung_result):
            hujung_result = max(hujung_result, key=len)
            if len(hujung_result):
                word = word[: -len(hujung_result)]
        permulaan_result = [
            v for k, v in permulaan.items() if word.startswith(k)
        ]
        if len(permulaan_result):
            permulaan_result = max(permulaan_result, key=len)
            if len(permulaan_result):
                word = word[len(permulaan_result):]

        combined = True
        if len(word):
            if word in rules_normalizer:
                word = rules_normalizer[word]
            elif self._corpus.get(word, 0) > 1000:
                pass
            else:
                candidates1 = self.edit_candidates(word)
                candidates2 = self.edit_candidates(cp_word)
                word1 = max(candidates1, key=self.P)
                word2 = max(candidates2, key=self.P)

                if self.WORDS[word1] > self.WORDS[word2]:
                    word = word1
                else:
                    word = word2
                    combined = False

            if (
                len(hujung_result)
                and not word.endswith(hujung_result)
                and combined
            ):
                word = word + hujung_result
            if (
                len(permulaan_result)
                and not word.startswith(permulaan_result)
                and combined
            ):
                word = permulaan_result + word

        else:
            if len(hujung_result) and not word.endswith(hujung_result):
                word = word + hujung_result
            if len(permulaan_result) and not word.startswith(permulaan_result):
                word = permulaan_result + word

        return word

    def elong_normalized_candidates(self, word, acc=None):
        if acc is None:
            acc = []
        candidates = [w for w in set(word) if word.count(w) > 1]
        for c in candidates:
            _w = word.replace(c + c, c)
            if _w in acc:
                continue
            acc.append(_w)
            self.elong_normalized_candidates(_w, acc)
        return acc + [word]

    def best_elong_candidate(self, word):
        candidates = self.elong_normalized_candidates(word)
        best = self.most_probable(candidates)
        return best or word

    def normalize_elongated(self, word):
        return case_of(word)(self.best_elong_candidate(word.lower()))


class Symspell:
    """
    The SymspellCorrector extends the functionality of symspeller, https://github.com/mammothb/symspellpy
    And improve it using some algorithms from Normalization of noisy texts in Malaysian online reviews,
    https://www.researchgate.net/publication/287050449_Normalization_of_noisy_texts_in_Malaysian_online_reviews
    Added custom vowels augmentation
    """

    def __init__(self, model, verbosity, corpus, k=10):
        self._model = model
        self._verbosity = verbosity
        self._corpus = corpus
        self.k = k

    def predict(self, word):
        max_edit_distance_lookup = 2
        suggestion_verbosity = self._verbosity
        suggestions = self._model.lookup(
            word, suggestion_verbosity, max_edit_distance_lookup
        )[: self.k]
        return suggestions

    def edit_step(self, word):
        """
        Generate candidates given a word.

        Parameters
        ----------
        word: str

        Returns
        -------
        result: {candidate1, candidate2}
        """

        result = list(_augment_vowel_alternate(word))

        if len(word):

            # berape -> berapa, mne -> mna
            if word[-1] == 'e':
                inner = word[:-1] + 'a'
                result.extend(list(_augment_vowel_alternate(inner)))

            # pikir -> fikir
            if word[0] == 'p':
                inner = 'f' + word[1:]
                result.extend(list(_augment_vowel_alternate(inner)))

        if len(word) > 2:
            # bapak -> bapa, mintak -> minta, mntak -> mnta
            if word[-2:] == 'ak':
                inner = word[:-1]
                fuzziness.append(word[:-1])
                result.extend(list(_augment_vowel_alternate(word[:-1])))

            # hnto -> hantar, bako -> bkar, sabo -> sabar
            if word[-1] == 'o' and word[-2] in consonants:
                inner = word[:-1] + 'ar'
                result.extend(list(_augment_vowel_alternate(inner)))

            # antu -> hantu, antar -> hantar
            if word[0] == 'a' and word[1] in consonants:
                inner = 'h' + word
                result.extend(list(_augment_vowel_alternate(inner)))

            # ptg -> ptng, dtg -> dtng
            if (
                word[-3] in consonants
                and word[-2] in consonants
                and word[-1] == 'g'
            ):
                inner = word[:-1] + 'ng'
                result.extend(list(_augment_vowel_alternate(inner)))

            # igt -> ingt
            if word[1] == 'g' and word[2] in consonants:
                inner = word[0] + 'n' + word[1:]
                result.extend(list(_augment_vowel_alternate(inner)))

        words = {}
        for r in result:
            suggestions = self.predict(r)
            for s in suggestions:
                words[s.term] = words.get(s.term, 0) + (
                    s.count / (s.distance + 1)
                )

        return words

    def edit_candidates(self, word):
        """
        Generate candidates given a word.

        Parameters
        ----------
        word: str

        Returns
        -------
        result: List[str]
        """

        ttt = self.edit_step(word) or {word: 10}
        ttt = {
            k: v
            for k, v in ttt.items()
            if len(k) > 3 and k not in ENGLISH_WORDS
        }
        ttt[word] = ttt.get(word, 0) + 10
        if not len(ttt):
            ttt = {word: 10}
        return list(ttt)

    @check_type
    def correct(self, word: str, **kwargs):
        """
        Most probable spelling correction for word.

        Parameters
        ----------
        word: str

        Returns
        -------
        result: str
        """

        if word in ENGLISH_WORDS:
            return word
        if self._corpus.get(word, 0) > 5000:
            return word
        if word in MALAY_WORDS:
            return word
        if word in stopword_tatabahasa:
            return word

        cp_word = word[:]
        hujung_result = [v for k, v in hujung.items() if word.endswith(k)]
        if len(hujung_result):
            hujung_result = max(hujung_result, key=len)
            if len(hujung_result):
                word = word[: -len(hujung_result)]
        permulaan_result = [
            v for k, v in permulaan.items() if word.startswith(k)
        ]
        if len(permulaan_result):
            permulaan_result = max(permulaan_result, key=len)
            if len(permulaan_result):
                word = word[len(permulaan_result):]

        combined = True
        if len(word):
            if word in rules_normalizer:
                word = rules_normalizer[word]
            else:
                candidates1 = self.edit_candidates(word)
                candidates2 = self.edit_candidates(cp_word)
                word1 = max(candidates1, key=candidates1.get)
                word2 = max(candidates2, key=candidates2.get)

                if candidates1[word1] > candidates2[word2]:
                    word = word1
                else:
                    word = word2
                    combined = False

        if len(hujung_result) and not word.endswith(hujung_result) and combined:
            word = word + hujung_result
        if (
            len(permulaan_result)
            and not word.startswith(permulaan_result)
            and combined
        ):
            word = permulaan_result + word
        return word

    @check_type
    def correct_text(self, text: str):
        """
        Correct all the words within a text, returning the corrected text.

        Parameters
        ----------
        text: str

        Returns
        -------
        result: str
        """

        return re.sub('[a-zA-Z]+', self.correct_match, text)

    def correct_match(self, match):
        """
        Spell-correct word in match, and preserve proper upper/lower/title case.
        """

        word = match.group()
        if word[0].isupper():
            return word
        return case_of(word)(self.correct(word.lower()))


class JamSpell:
    def __init__(self, corrector):
        self._corrector = corrector

    def _validate(self, word: str, string: str, index: int = -1):
        string = string.split()
        if word not in string:
            raise ValueError('word not in string after split by spaces')
        if index < 0:
            index = string.index(word)
        return string, index

    @check_type
    def correct(self, word: str, string: str, index: int = -1):
        """
        Correct a word within a text, returning the corrected word.

        Parameters
        ----------
        word: str
        string: str
            Entire string, `word` must a word inside `string`.
        index: int, optional(default=-1)
            index of word in the string, if -1, will try to use `string.index(word)`.

        Returns
        -------
        result: str
        """

        candidates = self.edit_candidates(word=word, string=string, index=index)
        return candidates[0]

    @check_type
    def correct_text(self, text: str):
        """
        Correct all the words within a text, returning the corrected text.

        Parameters
        ----------
        text: str

        Returns
        -------
        result: str
        """

        return self._corrector.FixFragment(text)

    def edit_candidates(self, word: str, string: str, index: int = -1):
        """
        Generate candidates given a word.

        Parameters
        ----------
        word: str
        string: str
            Entire string, `word` must a word inside `string`.
        index: int, optional(default=-1)
            index of word in the string, if -1, will try to use `string.index(word)`.

        Returns
        -------
        result: List[str]
        """

        string, index = self._validate(word=word, string=string, index=index)
        return self._corrector.GetCandidates(string, index)


class Spylls(Spell):
    def __init__(self, dictionary):
        self._dictionary = dictionary

    @check_type
    def correct(self, word: str):
        """
        Correct a word within a text, returning the corrected word.

        Parameters
        ----------
        word: str

        Returns
        -------
        result: str
        """
        r = self.edit_candidates(word=word)[:1]
        if len(r):
            return r[0]
        else:
            return word

    @check_type
    def edit_candidates(self, word: str):
        """
        Generate candidates given a word.

        Parameters
        ----------
        word: str

        Returns
        -------
        result: List[str]
        """
        return list(self._dictionary.suggest(word))


class Transformer(Spell):
    def __init__(self, model, corpus, sp_tokenizer):
        Spell.__init__(self, sp_tokenizer, corpus, add_norvig_method=False)
        self._model = model
        self._padding = tf.keras.preprocessing.sequence.pad_sequences

    def _correct(self, word, string, index, batch_size=20):
        possible_states = self.edit_candidates(word)
        replaced_masks = []
        for state in possible_states:
            mask = string[:]
            mask[index] = state
            replaced_masks.append(' '.join(mask))
        ids = [
            generate_ids(mask, self._model._tokenizer)
            for mask in replaced_masks
        ]
        tokens, input_ids, tokens_ids = list(zip(*ids))

        indices, ids = [], []
        for i in range(len(input_ids)):
            indices.extend([i] * len(input_ids[i]))
            ids.extend(input_ids[i])

        masked_padded = self._padding(ids, padding='post')
        input_masks = masked_padded.astype('bool').astype('int')
        preds = []
        for i in range(0, len(masked_padded), batch_size):
            index = min(i + batch_size, len(masked_padded))
            batch = masked_padded[i:index]
            batch_mask = input_masks[i:index]
            preds.append(self._model._log_vectorize(batch, batch_mask))

        preds = np.concatenate(preds, axis=0)
        indices = np.array(indices)
        scores = []
        for i in range(len(tokens)):
            filter_preds = preds[indices == i]
            total = np.sum(
                [filter_preds[k, k + 1, x] for k, x in enumerate(tokens_ids[i])]
            )
            scores.append(total)

        prob_scores = np.array(scores) / np.sum(scores)
        probs = list(zip(possible_states, prob_scores))
        probs.sort(key=lambda x: x[1])
        return probs[0][0]

    @check_type
    def correct(
        self, word: str, string: str, index: int = -1, batch_size: int = 20
    ):
        """
        Correct a word within a text, returning the corrected word.

        Parameters
        ----------
        word: str
        string: str
            Entire string, `word` must a word inside `string`.
        index: int, optional(default=-1)
            index of word in the string, if -1, will try to use `string.index(word)`.
        batch_size: int, optional(default=20)
            batch size to insert into model.

        Returns
        -------
        result: str
        """

        if batch_size < 1:
            raise ValueError('batch_size must be bigger than 0')
        string = string.split()
        if word not in string:
            raise ValueError('word not in string after split by spaces')
        if index < 0:
            index = string.index(word)

        if word in ENGLISH_WORDS:
            return word
        if word in MALAY_WORDS:
            return word
        if word in stopword_tatabahasa:
            return word

        if word in rules_normalizer:
            word = rules_normalizer[word]
        else:
            word = self._correct(word, string, index, batch_size=batch_size)
        return word

    @check_type
    def correct_text(self, text: str, batch_size: int = 20):
        """
        Correct all the words within a text, returning the corrected text.

        Parameters
        ----------
        text: str
        batch_size: int, optional(default=20)
            batch size to insert into model.

        Returns
        -------
        result: str
        """

        if batch_size < 1:
            raise ValueError('batch_size must be bigger than 0')

        text = re.sub('[^a-zA-Z]+', ' ', text)
        string = re.sub(r'[ ]+', ' ', text).strip()
        strings = []
        for no, word in enumerate(string.split()):
            if not word[0].isupper():
                word = case_of(word)(
                    self.correct(
                        word.lower(), string, no, batch_size=batch_size
                    )
                )
            strings.append(word)

        return ' '.join(strings)

    @check_type
    def correct_word(self, word: str, string: str, batch_size: int = 20):
        """
        Spell-correct word in match, and preserve proper upper/lower/title case.

        Parameters
        ----------
        word: str
        string: str
            Entire string, `word` must a word inside `string`.
        batch_size: int, optional(default=20)
            batch size to insert into model.

        Returns
        -------
        result: str
        """

        return case_of(word)(
            self.correct(word.lower(), string, batch_size=batch_size)
        )


@check_type
def probability(sentence_piece: bool = False, **kwargs):
    """
    Train a Probability Spell Corrector.

    Parameters
    ----------
    sentence_piece: bool, optional (default=False)
        if True, reduce possible augmentation states using sentence piece.

    Returns
    -------
    result: malaya.spell.Probability class
    """

    check_file(PATH_NGRAM[1], S3_PATH_NGRAM[1], **kwargs)

    tokenizer = None

    if sentence_piece:
        check_file(
            PATH_NGRAM['sentencepiece'],
            S3_PATH_NGRAM['sentencepiece'],
            **kwargs
        )

        vocab = PATH_NGRAM['sentencepiece']['vocab']
        vocab_model = PATH_NGRAM['sentencepiece']['model']
        tokenizer = SentencePieceTokenizer(vocab_file=vocab, spm_model_file=vocab_model)

    with open(PATH_NGRAM[1]['model']) as fopen:
        corpus = json.load(fopen)
    return Probability(corpus, tokenizer)


@check_type
def symspell(
    max_edit_distance_dictionary: int = 2,
    prefix_length: int = 7,
    term_index: int = 0,
    count_index: int = 1,
    top_k: int = 10,
    **kwargs
):
    """
    Load a symspell Spell Corrector for Malay.

    Returns
    -------
    result: malaya.spell.Symspell class
    """

    try:
        from symspellpy.symspellpy import SymSpell, Verbosity
    except BaseException:
        raise ModuleNotFoundError(
            'symspellpy not installed. Please install it and try again.'
        )

    check_file(PATH_NGRAM['symspell'], S3_PATH_NGRAM['symspell'], **kwargs)
    check_file(PATH_NGRAM[1], S3_PATH_NGRAM[1], **kwargs)

    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
    dictionary_path = PATH_NGRAM['symspell']['model']
    sym_spell.load_dictionary(dictionary_path, term_index, count_index)
    with open(PATH_NGRAM[1]['model']) as fopen:
        corpus = json.load(fopen)
    return Symspell(sym_spell, Verbosity.ALL, corpus, k=top_k)


@check_type
def jamspell(model: str = 'wiki', **kwargs):
    """
    Load a jamspell Spell Corrector for Malay.

    Parameters
    ----------
    model: str, optional (default='wiki+news')
        Supported models. Allowed values:

        * ``'wiki+news'`` - Wikipedia + News, 337MB.
        * ``'wiki'`` - Wikipedia, 148MB.
        * ``'news'`` - local news, 215MB.

    Returns
    -------
    result: malaya.spell.JamSpell class
    """

    try:
        import jamspell as jamspellpy
    except BaseException:
        raise ModuleNotFoundError(
            'jamspell not installed. Please install it and try again.'
        )

    model = model.lower()
    supported_models = ['wiki+news', 'wiki', 'news']
    if model not in supported_models:
        raise ValueError(
            f'model not supported, available models are {str(supported_models)}'
        )

    check_file(PATH_NGRAM['jamspell'][model], S3_PATH_NGRAM['jamspell'][model], **kwargs)
    try:
        corrector = jamspellpy.TSpellCorrector()
        corrector.LoadLangModel(PATH_NGRAM['jamspell'][model]['model'])
    except BaseException:
        raise Exception(
            f"failed to load jamspell model, please run `malaya.utils.delete_cache('preprocessing/jamspell/{model.replace('+', '-')}')`")
    return JamSpell(corrector=corrector)


@check_type
def spylls(model: str = 'libreoffice-pejam', **kwargs):
    """
    Load a spylls Spell Corrector for Malay.

    Parameters
    ----------
    model : str, optional (default='libreoffice-pejam')
        Model spelling correction supported. Allowed values:

        * ``'libreoffice-pejam'`` - from LibreOffice pEJAm, https://extensions.libreoffice.org/en/extensions/show/3868

    Returns
    -------
    result: malaya.spell.Spylls class
    """

    try:
        from spylls.hunspell import Dictionary
    except BaseException:
        raise ModuleNotFoundError(
            'spylls not installed. Please install it and try again.'
        )

    model = model.lower()
    supported_models = ['libreoffice-pejam']
    if model not in supported_models:
        raise ValueError(
            f'model not supported, available models are {str(supported_models)}'
        )

    check_file(PATH_NGRAM['spylls'][model], S3_PATH_NGRAM['spylls'][model], **kwargs)
    try:
        dictionary = Dictionary.from_zip(PATH_NGRAM['spylls'][model]['model'])
    except BaseException:
        raise Exception(
            f"failed to load spylls model, please run `malaya.utils.delete_cache('preprocessing/spylls/{model}')`")
    return Spylls(dictionary=dictionary)


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text='tested on 10k test set.'
    )


@check_type
def transformer(model: str = 'small-t5', quantized: bool = False, **kwargs):
    """
    Load a Transformer Spell Corrector.

    Parameters
    ----------
    model : str, optional (default='small-t5')
        Model architecture supported. Allowed values:

        * ``'small-t5'`` - T5 SMALL parameters.
        * ``'tiny-t5'`` - T5 TINY parameters.
        * ``'super-tiny-t5'`` - T5 SUPER TINY parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.t5.Spell class
    """
    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.spell.available_transformer()`.'
        )
    return t5_load.load(
        module='spelling-correction',
        model=model,
        model_class=T5_Spell,
        quantized=quantized,
        **kwargs,
    )


@check_type
def transformer_encoder(model, sentence_piece: bool = False, **kwargs):
    """
    Load a Transformer Encoder Spell Corrector. Right now only supported BERT and ALBERT.

    Parameters
    ----------
    sentence_piece: bool, optional (default=False)
        if True, reduce possible augmentation states using sentence piece.

    Returns
    -------
    result: malaya.spell.Transformer class
    """

    if not hasattr(model, '_log_vectorize'):
        raise ValueError('model must have `_log_vectorize` method')

    check_file(PATH_NGRAM[1], S3_PATH_NGRAM[1], **kwargs)

    tokenizer = None

    if sentence_piece:
        check_file(
            PATH_NGRAM['sentencepiece'],
            S3_PATH_NGRAM['sentencepiece'],
            **kwargs
        )

        vocab = PATH_NGRAM['sentencepiece']['vocab']
        vocab_model = PATH_NGRAM['sentencepiece']['model']
        tokenizer = SentencePieceTokenizer(vocab_file=vocab, spm_model_file=vocab_model)

    with open(PATH_NGRAM[1]['model']) as fopen:
        corpus = json.load(fopen)
    return Transformer(model, corpus, tokenizer)
