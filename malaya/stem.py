import re
from unidecode import unidecode
from malaya.text.tatabahasa import permulaan, hujung
from malaya.text.rules import rules_normalizer
from malaya.dictionary import is_english
from malaya.text.function import PUNCTUATION, case_of, is_emoji
from malaya.text.regex import _expressions, _money, _date
from malaya.model.abstract import Abstract
from malaya.supervised.t2t import load_lstm_yttm
from malaya.preprocessing import Tokenizer
from malaya.function import describe_availability
from malaya.path import STEMMER_VOCAB
from herpetologist import check_type
import logging

logger = logging.getLogger(__name__)


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
                or is_emoji(word.lower())
                or is_english(word.lower())
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


_availability = {
    'base': {
        'Size (MB)': 13.6,
        'Quantized Size (MB)': 3.64,
        'CER': 0.02143794,
        'WER': 0.04399622,
    },
    'noisy': {
        'Size (MB)': 28.5,
        'Quantized Size (MB)': 7.3,
        'CER': 0.02138838,
        'WER': 0.04952738,
    },
}


def available_deep_model():
    """
    List available stemmer deep models.
    """
    logger.info('trained on 90% dataset, tested on another 10% test set, dataset at https://github.com/huseinzol05/malay-dataset/tree/master/normalization/stemmer')
    logger.info('`base` tested on non-noisy dataset, while `noisy` tested on noisy dataset.')

    return describe_availability(_availability)


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
    Load LSTM + Bahdanau Attention stemming model, BPE level (YouTokenToMe 1000 vocab size).
    This model also include lemmatization.

    Parameters
    ----------
    model: str, optional (default='base')
        Check available models at `malaya.stem.available_deep_model()`.
    quantized: bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.stem.DeepStemmer class
    """

    model = model.lower()
    if model not in _availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.stem.available_deep_model()`.'
        )

    return load_lstm_yttm(
        module='stem-v2',
        vocab=STEMMER_VOCAB,
        model_class=DeepStemmer,
        quantized=quantized,
        model=model,
        **kwargs,
    )
