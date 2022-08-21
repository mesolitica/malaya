import json
from malaya.text.function import case_of, is_english, is_malay
from malaya.path import PATH_NGRAM, S3_PATH_NGRAM
from malaya.function import check_file
from malaya.spelling_correction.base import (
    _augment_vowel_alternate,
)
from herpetologist import check_type
from typing import List


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

    def edit_candidates(self, word, get_score=False):
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
            if len(k) > 3 and not is_english(k)
        }
        ttt[word] = ttt.get(word, 0) + 10
        if not len(ttt):
            ttt = {word: 10}
        if get_score:
            return ttt
        else:
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
            else:
                candidates1 = self.edit_candidates(word, get_score=True)
                candidates2 = self.edit_candidates(cp_word, get_score=True)
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


@check_type
def load(
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
    result: malaya.spelling_correction.symspell.Symspell class
    """

    try:
        from symspellpy.symspellpy import SymSpell, Verbosity
    except BaseException:
        raise ModuleNotFoundError(
            'symspellpy not installed. Please install it and try again.'
        )

    path = check_file(PATH_NGRAM['symspell'], S3_PATH_NGRAM['symspell'], **kwargs)
    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
    sym_spell.load_dictionary(path['model'], term_index, count_index)

    path = check_file(PATH_NGRAM[1], S3_PATH_NGRAM[1], **kwargs)
    with open(path['model']) as fopen:
        corpus = json.load(fopen)
    return Symspell(sym_spell, Verbosity.ALL, corpus, k=top_k)
