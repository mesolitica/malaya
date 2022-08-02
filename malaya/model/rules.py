from malaya.text.normalization import _is_number_regex
from malaya.text.function import (
    is_emoji,
    check_ratio_numbers,
    check_ratio_punct,
    is_malay,
    PUNCTUATION,
)
from typing import List
import logging

logger = logging.getLogger(__name__)


class LanguageDict:
    def __init__(self, model, **kwargs):

        enchant_available = True
        try:
            import enchant
        except BaseException:
            logger.warning(
                'pyenchant not installed. Please install it by `pip3 install pyenchant` and try again. For now, pyenchant will be disabled.')
            enchant_available = False

        try:
            self.d = enchant.Dict('en_US')
            self.d.check('Hello')
        except BaseException:
            logger.warning(
                'cannot load `en_US` enchant dictionary. Please install it from https://pyenchant.github.io/pyenchant/install.html and try again. For now, pyenchant will be disabled.')
            enchant_available = False

        self._enchant_available = enchant_available
        self._model = model

    def predict(
        self,
        words: List[str],
        acceptable_ms_label: List[str] = ['malay', 'ind'],
        acceptable_en_label: List[str] = ['eng', 'manglish'],
        use_is_malay: bool = True,
    ):
        """
        Predict [EN, MS, OTHERS, CAPITAL, NOT_LANG] on word level. 
        This method assumed the string already tokenized.

        Parameters
        ----------
        words: List[str]
        acceptable_ms_label: List[str], optional (default = ['malay', 'ind'])
            accept labels from language detection model to assume a word is `MS`.
        acceptable_en_label: List[str], optional (default = ['eng', 'manglish'])
            accept labels from language detection model to assume a word is `EN`.
        use_is_malay: bool, optional (default=True)
            if True`, will predict MS word using `malaya.text.function.is_malay`, 
            else use language detection model.

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
            elif self._enchant_available and self.d.check(word):
                results.append('EN')
            elif use_is_malay and is_malay(word.lower()):
                results.append('MS')
            else:
                results.append('REPLACE_ME')
                others.append(word)
                indices.append(no)

        labels = self._model.predict(others)
        for no in range(len(labels)):
            if labels[no] in acceptable_ms_label:
                results[indices[no]] = 'MS'
            elif labels[no] in acceptable_en_label:
                results[indices[no]] = 'EN'
            else:
                results[indices[no]] = 'OTHERS'

        return results
