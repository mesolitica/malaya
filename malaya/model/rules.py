from malaya.text.normalization import _is_number_regex
from malaya.text.function import (
    is_emoji,
    check_ratio_numbers,
    check_ratio_punct,
    is_malay,
    PUNCTUATION,
)
from typing import List


class LanguageDict:
    def __init__(self, model_fasttext, **kwargs):
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

        self._model_fasttext = model_fasttext

    def predict(self, words: List[str], acceptable_ms_label: List[str] = ['ms', 'id']):
        """
        Predict [EN, MS, OTHERS, CAPITAL, NOT_LANG] on word level. 
        This method assumed the string already tokenized.

        Parameters
        ----------
        words: List[str]
        acceptable_ms_label: List[str], optional (default = ['ms', 'id'])
            accept labels from fast-text to assume a word is `MS`.

        Returns
        -------
        result: List[str]
        """

        results, others, indices = [], [], []
        for no, word in enumerate(words):
            if is_emoji(word):
                results.append('NOT_LANG')
            elif word.isupper():
                results.append('CAPITAL')
            elif _is_number_regex(word.replace(',', '').replace('.', '')):
                results.append('NOT_LANG')
            elif word in PUNCTUATION:
                results.append('NOT_LANG')
            elif check_ratio_numbers(word) > 0.6666:
                results.append('NOT_LANG')
            elif check_ratio_punct(word) > 0.66666:
                results.append('NOT_LANG')
            elif self.d.check(word):
                results.append('EN')
            elif is_malay(word.lower()):
                results.append('MS')
            else:
                results.append('REPLACE_ME')
                others.append(word)
                indices.append(no)

        labels = self._model_fasttext.predict(others)[0]
        labels = [l[0].replace('__label__', '') for l in labels]
        for no in range(len(labels)):
            if labels[no] in acceptable_ms_label:
                results[indices[no]] = 'MS'
            elif labels[no] in ['en']:
                results[indices[no]] = 'EN'
            else:
                results[indices[no]] = 'OTHERS'

        return results
