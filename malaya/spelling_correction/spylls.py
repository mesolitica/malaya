from malaya.spelling_correction.probability import Spell
from herpetologist import check_type
from typing import List


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
