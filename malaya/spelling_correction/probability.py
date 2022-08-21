import json
from functools import partial
from collections import Counter
from malaya.text.function import case_of, is_english, is_malay
from malaya.text.rules import rules_normalizer
from malaya.text.bpe import SentencePieceTokenizer
from malaya.path import PATH_NGRAM, S3_PATH_NGRAM
from malaya.function import check_file
from malaya.spelling_correction.base import (
    _augment_vowel_prob_sp,
    _augment_vowel_prob,
    _augment_vowel,
)
from malaya.text.tatabahasa import (
    alphabet,
    consonants,
    vowels,
    permulaan,
    hujung,
    stopword_tatabahasa,
)
from herpetologist import check_type
from typing import List


class Spell:
    def __init__(self, sp_tokenizer, corpus, add_norvig_method=True):
        self._sp_tokenizer = sp_tokenizer
        if self._sp_tokenizer is not None:
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
        ttt = {i for i in ttt if len(i) > 3 and not is_english(i)}
        ttt = self.known([word]) | ttt
        if not len(ttt):
            ttt = {word}
        ttt = list(ttt)
        if word[-1] in vowels:
            ttt = [w for w in ttt if w[-1] == word[-1] or (w[-1] in 'e' and word[-1] in 'a')]
        return ttt

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
        Spell-correct word in re.match, and preserve proper upper, lower, title case.
        """

        word = match.group()
        if word[0].isupper():
            return word
        return case_of(word)(self.correct(word.lower()))

    @check_type
    def correct_word(self, word: str):
        """
        Spell-correct word, and preserve proper upper, lower and title case.

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

        if is_english(word):
            return word
        if self._corpus.get(word, 0) > 5000:
            return word
        if is_malay(word):
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


class ProbabilityLM(Probability):
    """
    The SpellCorrector extends the functionality of the Peter Norvig's with Language Model.
    spell-corrector in http://norvig.com/spell-correct.html
    And improve it using some algorithms from Normalization of noisy texts in Malaysian online reviews,
    https://www.researchgate.net/publication/287050449_Normalization_of_noisy_texts_in_Malaysian_online_reviews
    Added custom vowels augmentation
    """

    def __init__(self, language_model, corpus, sp_tokenizer=None):
        Spell.__init__(self, sp_tokenizer, corpus)
        self._language_model = language_model

    @check_type
    def correct(
        self,
        word: str,
        string: List[str],
        index: int = -1,
        lookback: int = 3,
        lookforward: int = 3,
    ):
        """
        Correct a word within a text, returning the corrected word.

        Parameters
        ----------
        word: str
        string: str
            Entire string, `word` must a word inside `string`.
        index: int, optional (default=-1)
            index of word in the string, if -1, will try to use `string.index(word)`.
        lookback: int, optional (default=3)
            N left hand side words.
        lookforward: int, optional (default=3)
            N right hand side words.

        Returns
        -------
        result: str
        """

        if word not in string:
            raise ValueError('word is not inside the string')
        if index < 0:
            index = string.index(word)
        else:
            if string[index] != word:
                raise ValueError('index of the splitted string is not equal to the word')

        if is_english(word):
            return word
        if is_malay(word):
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
        string = re.sub(r'[ ]+', ' ', text).strip()
        splitted = string.split()
        strings = []
        for no, word in enumerate(splitted):
            if not word.isupper():
                p = partial(self.correct_word, string=splitted, index=no, batch_size=batch_size)
                word = re.sub('[a-zA-Z]+', p, word)
            strings.append(word)

        return ' '.join(strings)

    @check_type
    def correct_word(self, word: str):
        """
        Spell-correct word in match, and preserve proper upper, lower and title case.

        Parameters
        ----------
        word: str

        Returns
        -------
        result: str
        """

        return case_of(word)(self.correct(word.lower()))


@check_type
def load(language_model=None, sentence_piece: bool = False, **kwargs):
    """
    Load a Probability Spell Corrector.

    Parameters
    ----------
    language_model: Callable, optional (default=None)
        If not None, must an instance of kenlm.Model.
    sentence_piece: bool, optional (default=False)
        if True, reduce possible augmentation states using sentence piece.

    Returns
    -------
    result: model
        List of model classes:

        * if passed `language_model` will return `malaya.spelling_correction.probability.ProbabilityLM`.
        * else will return `malaya.spelling_correction.probability.Probability`.
    """

    tokenizer = None
    if sentence_piece:
        path = check_file(
            PATH_NGRAM['sentencepiece'],
            S3_PATH_NGRAM['sentencepiece'],
            **kwargs
        )

        vocab = path['vocab']
        vocab_model = path['model']
        tokenizer = SentencePieceTokenizer(vocab_file=vocab, spm_model_file=vocab_model)

    path = check_file(PATH_NGRAM[1], S3_PATH_NGRAM[1], **kwargs)

    with open(path['model']) as fopen:
        corpus = json.load(fopen)

    if language_model is not None:
        try:
            import kenlm
        except BaseException:
            raise ModuleNotFoundError(
                'kenlm not installed. Please install it by `pip install pypi-kenlm` and try again.'
            )

        if not isinstance(language_model, kenlm.Model):
            raise ValueError('`language_model` must an instance of `kenlm.Model`.')
        return ProbabilityLM(language_model, corpus, tokenizer)
    else:
        return Probability(corpus, tokenizer)
