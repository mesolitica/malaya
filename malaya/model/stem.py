import re
from malaya.text.tatabahasa import permulaan, hujung
from malaya.dictionary import is_english
from malaya.text.function import PUNCTUATION, case_of, is_emoji
from malaya.text.regex import _expressions, _money, _date
from malaya.preprocessing import Tokenizer


class Base:
    _tokenizer = Tokenizer().tokenize

    def stem(self, string: str, **kwargs):
        result = []
        tokenized = self._tokenizer(string)
        for no, word in enumerate(tokenized):
            if word in PUNCTUATION:
                result.append(word)
            elif (
                re.findall(_money, word.lower())
                or re.findall(_date, word.lower())
                or re.findall(_expressions['email'], word.lower())
                or re.findall(_expressions['url'], word.lower())
                or re.findall(_expressions['hashtag'], word.lower())
                or re.findall(_expressions['phone'], word.lower())
                or re.findall(_expressions['money'], word.lower())
                or re.findall(_expressions['date'], word.lower())
                or re.findall(_expressions['time'], word.lower())
                or re.findall(_expressions['ic'], word.lower())
                or re.findall(_expressions['user'], word.lower())
                or is_emoji(word.lower())
                or is_english(word.lower())
            ):
                result.append(word)
            else:
                result.append(case_of(word)(self.stem_word(word, **kwargs)))
        return ' '.join(result)

    def predict(self, string):
        return self.stem(string)


class Naive(Base):

    def stem_word(self, word, **kwargs):
        """
        Stem a word using Regex pattern.

        Parameters
        ----------
        string : str

        Returns
        -------
        result: str
        """
        word_temp = word
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

        if not len(word):
            word = word_temp
        return word

    def stem(self, string: str):
        """
        Stem a string using Regex pattern.

        Parameters
        ----------
        string : str

        Returns
        -------
        result: str
        """

        return super().stem(string)


class Sastrawi(Base):
    def __init__(self, factory):
        self.sastrawi_stemmer = factory.create_stemmer()

    def stem_word(self, word: str, **kwargs):
        """
        Stem a word using Sastrawi, this also include lemmatization.

        Parameters
        ----------
        string : str

        Returns
        -------
        result: str
        """
        return self.sastrawi_stemmer.stem(word)

    def stem(self, string: str):
        """
        Stem a string using Sastrawi, this also include lemmatization.

        Parameters
        ----------
        string : str

        Returns
        -------
        result: str
        """

        return super().stem(string)
