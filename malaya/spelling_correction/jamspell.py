from malaya.path import PATH_NGRAM, S3_PATH_NGRAM
from malaya.function import check_file, describe_availability
from herpetologist import check_type
from typing import List


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
        if word not in string:
            raise ValueError('word is not inside the string')
        if index < 0:
            index = string.index(word)
        else:
            if string[index] != word:
                raise ValueError('index of the splitted string is not equal to the word')

        return self._corrector.GetCandidates(' '.join(string), index)


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
