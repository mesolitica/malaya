import re
from unidecode import unidecode
from malaya.text.tatabahasa import permulaan, hujung
from malaya.text.rules import rules_normalizer
from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.text.function import PUNCTUATION, case_of
from malaya.text.regex import _expressions, _money, _date
from malaya.model.abstract import Abstract
from malaya.supervised.t2t import load_lstm_yttm
from malaya.preprocessing import Tokenizer
from malaya.path import STEMMER_VOCAB
from herpetologist import check_type


def _classification_textcleaning_stemmer(string, stemmer):
    string = re.sub(
        'http\\S+|www.\\S+',
        '',
        ' '.join(
            [i for i in string.split() if i.find('#') < 0 and i.find('@') < 0]
        ),
    )
    string = unidecode(string).replace('.', ' . ').replace(',', ' , ')
    string = re.sub('[^A-Za-z ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string.lower()).strip()
    string = [rules_normalizer.get(w, w) for w in string.split()]
    string = [(stemmer.stem(word), word) for word in string]
    return ' '.join([word[0] for word in string if len(word[0]) > 1])


class Base:
    _tokenizer = Tokenizer().tokenize

    @check_type
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
            ):
                result.append(word)
            else:
                result.append(case_of(word)(self.stem_word(word, **kwargs)))
        return ' '.join(result)

    def predict(self, string):
        return self.stem(string)


class Sastrawi(Base):
    def __init__(self, factory):
        self.sastrawi_stemmer = factory.create_stemmer()

    @check_type
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

    @check_type
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
        return word

    @check_type
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


class DeepStemmer(Abstract, Base):
    def __init__(
        self, input_nodes, output_nodes, sess, bpe, **kwargs,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self._bpe = bpe

    @check_type
    def greedy_decoder(self, string: str):
        """
        Stem a string, this also include lemmatization using greedy decoder.

        Parameters
        ----------
        string : str

        Returns
        -------
        result: str
        """

        return self.stem(string, beam_search=False)

    @check_type
    def beam_decoder(self, string: str):
        """
        Stem a string, this also include lemmatization using beam decoder.

        Parameters
        ----------
        string : str

        Returns
        -------
        result: str
        """

        return self.stem(string, beam_search=True)

    def predict(self, string: str, beam_search: bool = False):
        """
        Stem a string, this also include lemmatization.

        Parameters
        ----------
        string : str
        beam_search : bool, (optional=False)
            If True, use beam search decoder, else use greedy decoder.

        Returns
        -------
        result: str
        """

        return self.stem(string=string, beam_search=beam_search)

    def stem_word(self, word, beam_search=False, **kwargs):
        """
        Stem a word, this also include lemmatization.

        Parameters
        ----------
        string : str

        Returns
        -------
        result: str
        """

        batch = self._bpe.bpe.encode([word], output_type=self._bpe.mode)
        batch = [i + [1] for i in batch]

        if beam_search:
            output = 'beam'
        else:
            output = 'greedy'

        r = self._execute(
            inputs=[batch],
            input_labels=['Placeholder'],
            output_labels=[output],
        )
        output = r[output].tolist()[0]
        predicted = (
            self._bpe.bpe.decode(output)[0]
            .replace('<EOS>', '')
            .replace('<PAD>', '')
        )
        return predicted

    @check_type
    def stem(self, string: str, beam_search: bool = False):
        """
        Stem a string, this also include lemmatization.

        Parameters
        ----------
        string : str
        beam_search : bool, (optional=False)
            If True, use beam search decoder, else use greedy decoder.

        Returns
        -------
        result: str
        """

        return super().stem(string, beam_search=beam_search)


@check_type
def naive():
    """
    Load stemming model using startswith and endswith naively using regex patterns.

    Returns
    -------
    result : malaya.stem.Naive class
    """

    return Naive()


@check_type
def sastrawi():
    """
    Load stemming model using Sastrawi, this also include lemmatization.

    Returns
    -------
    result: malaya.stem.Sastrawi class
    """

    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    except BaseException:
        raise ModuleNotFoundError(
            'PySastrawi not installed. Please install it by `pip install PySastrawi` and try again.'
        )
    return Sastrawi(StemmerFactory())


@check_type
def deep_model(model: str = 'base', quantized: bool = False, **kwargs):
    """
    Load LSTM + Bahdanau Attention stemming model,
    256 filter size, 2 layers, BPE level (YouTokenToMe 20k vocab size).
    This model also include lemmatization.
    Original size 41.6MB, quantized size 10.6MB .

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'base'`` - trained on default dataset.
        * ``'noisy'`` - trained on default and augmentation dataset.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.stem.DeepStemmer class
    """

    return load_lstm_yttm(
        module='stem',
        vocab=STEMMER_VOCAB,
        model_class=DeepStemmer,
        quantized=quantized,
        model=model,
        **kwargs,
    )
