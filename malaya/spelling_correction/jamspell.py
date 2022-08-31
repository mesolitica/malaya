import re
from functools import partial
from malaya.text.function import case_of, check_ratio_upper_lower
from malaya.dictionary import is_english, is_malay
from malaya.text.tatabahasa import stopword_tatabahasa
from malaya.text.rules import rules_normalizer
from malaya.path import PATH_NGRAM, S3_PATH_NGRAM
from malaya.function import check_file, describe_availability
from herpetologist import check_type
from typing import List

"""
Before you able to use this spelling correction, you need to install https://github.com/bakwc/JamSpell,

For mac,

```bash
wget http://prdownloads.sourceforge.net/swig/swig-3.0.12.tar.gz
tar -zxf swig-3.0.12.tar.gz
./swig-3.0.12/configure && make && make install
pip3 install jamspell
```

For debian / ubuntu,

```bash
apt install swig3
pip3 install jamspell
```
"""


class JamSpell:
    def __init__(self, corrector):
        self._corrector = corrector

    @check_type
    def correct(self, word: str, string: List[str], index: int = -1):
        """
        Correct a word within a text, returning the corrected word.

        Parameters
        ----------
        word: str
        string: List[str]
            Tokenized string, `word` must a word inside `string`.
        index: int, optional(default=-1)
            index of word in the string, if -1, will try to use `string.index(word)`.

        Returns
        -------
        result: str
        """

        if is_english(word):
            return word
        if is_malay(word):
            return word
        if word in stopword_tatabahasa:
            return word

        if word in rules_normalizer:
            word = rules_normalizer[word]
        else:
            candidates = self.edit_candidates(word=word, string=string, index=index)
            word = candidates[0]
        return word

    def correct_word(
        self,
        word,
        string: List[str],
        index: int = -1,
    ):
        """
        Spell-correct word in re.match, and preserve proper upper, lower, title case.
        """

        if len(word) < 2:
            return word
        return case_of(word)(self.correct(word.lower(), string=string, index=index))

    def correct_match(
        self,
        match,
        string: List[str],
        index: int = -1,
    ):
        """
        Spell-correct word in re.match, and preserve proper upper, lower, title case.
        """
        return self.correct_word(match.group(), string=string, index=index)

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
        for no, word in enumerate(splitted):
            if not word.isupper() and check_ratio_upper_lower(word) < 0.5:
                p = partial(
                    self.correct_match,
                    string=splitted,
                    index=no,
                )
                word = re.sub('[a-zA-Z]+', p, word)
            splitted[no] = word

        return ' '.join(splitted)

    def edit_candidates(self, word: str, string: List[str], index: int = -1):
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
        if index < 0:
            index = string.index(word)
        else:
            if word.lower() not in string[index].lower():
                raise ValueError('word is not a subset or equal to index of the splitted string')

        return self._corrector.GetCandidates(string, index)


_availability = {
    'wiki+news': {
        'Size (MB)': 337,
    },
    'wiki': {
        'Size (MB)': 148,
    },
    'news': {
        'Size (MB)': 215,
    }
}


def available_model():
    """
    List available jamspell models.
    """

    return describe_availability(_availability)


def load(model: str = 'wiki', **kwargs):
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
    if model not in _availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.spelling_correction.jamspell.available_model()`.'
        )

    path = check_file(PATH_NGRAM['jamspell'][model], S3_PATH_NGRAM['jamspell'][model], **kwargs)
    try:
        corrector = jamspellpy.TSpellCorrector()
        corrector.LoadLangModel(path['model'])
    except BaseException:
        raise Exception('failed to load jamspell model, please try clear cache or rerun again.')
    return JamSpell(corrector=corrector)
