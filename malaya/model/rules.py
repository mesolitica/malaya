from malaya.text.normalization import _is_number_regex
from malaya.text.function import is_emoji, PUNCTUATION
from typing import List


class LanguageDict:
    def __init__(self, **kwargs):
        try:
            import enchant

        except BaseException:
            raise ModuleNotFoundError(
                'pyenchant not installed. Please install it by `pip3 install pyenchant` and try again.'
            )

        try:
            self.d = enchant.Dict('en_US')
            self.d.check('Hello')
        except BaseException:
            raise ModuleNotFoundError(
                'cannot load `en_US` enchant dictionary. Please install it from https://pyenchant.github.io/pyenchant/install.html and try again.'
            )

    def predict(self, words: List[str]):
        """
        Predict [EN, MS, NOT_LANG] on word level. 
        This method assumed the string already tokenized.

        Parameters
        ----------
        words: List[str]

        Returns
        -------
        result: List[str]
        """

        results = []
        for word in words:
            if is_emoji(word):
                results.append('NOT_LANG')
            elif _is_number_regex(word.replace(',', '').replace('.', '')):
                results.append('NOT_LANG')
            elif word in PUNCTUATION:
                results.append('NOT_LANG')
            elif self.d.check(word):
                results.append('EN')
            else:
                results.append('MS')

        return results
