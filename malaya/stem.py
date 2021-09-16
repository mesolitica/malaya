import re
import json
import os
from unidecode import unidecode
from malaya.text.tatabahasa import permulaan, hujung
from malaya.text.rules import rules_normalizer
from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.text.function import pad_sentence_batch, case_of
from malaya.text.regex import _expressions, _money, _date
from malaya.model.abstract import Abstract
from malaya.preprocessing import Tokenizer
from malaya.text.bpe import YTTMEncoder
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


class Sastrawi:
    def __init__(self, factory):
        self.sastrawi_stemmer = factory.create_stemmer()

    @check_type
    def stem(self, string: str):
        return self.sastrawi_stemmer.stem(string)


class Naive:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def stem_word(self, word):
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
        result = []
        tokenized = self._tokenizer(string)
        for no, word in enumerate(tokenized):
            if word in '~@#$%^&*()_+{}|[:"\'];<>,.?/-':
                result.append(word)
            elif (
                re.findall(_money, word.lower())
                or re.findall(_date, word.lower())
                or re.findall(_expressions['time'], word.lower())
                or re.findall(_expressions['hashtag'], word.lower())
                or re.findall(_expressions['url'], word.lower())
                or re.findall(_expressions['user'], word.lower())
            ):
                result.append(word)
            else:
                result.append(self.stem_word(word))
        return ' '.join(result)


class DeepStemmer(Abstract):
    def __init__(
        self, input_nodes, output_nodes, sess, bpe, tokenizer
    ):

        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self._bpe = bpe
        self._tokenizer = tokenizer

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
        tokenized = self._tokenizer(string)
        result, batch, actual, mapping = [], [], [], {}
        for no, word in enumerate(tokenized):
            if word in '~@#$%^&*()_+{}|[:"\'];<>,.?/-':
                result.append(word)
            elif (
                re.findall(_money, word.lower())
                or re.findall(_date, word.lower())
                or re.findall(_expressions['time'], word.lower())
                or re.findall(_expressions['hashtag'], word.lower())
                or re.findall(_expressions['url'], word.lower())
                or re.findall(_expressions['user'], word.lower())
            ):
                result.append(word)
            else:
                mapping[len(batch)] = no
                result.append('REPLACE-ME')
                actual.append(word)
                batch.append(word.lower())

        if len(batch):

            batch = self._bpe.bpe.encode(batch, output_type=self._bpe.mode)

            batch = [i + [1] for i in batch]
            batch = pad_sentence_batch(batch, 0)[0]

            if beam_search:
                output = 'beam'
            else:
                output = 'greedy'

            r = self._execute(
                inputs=[batch],
                input_labels=['Placeholder'],
                output_labels=[output],
            )
            output = r[output].tolist()

            for no, o in enumerate(output):
                predicted = list(dict.fromkeys(o))
                predicted = (
                    self._bpe.bpe.decode(predicted)[0]
                    .replace('<EOS>', '')
                    .replace('<PAD>', '')
                )
                predicted = case_of(actual[no])(predicted)
                result[mapping[no]] = predicted

        return ' '.join(result)


@check_type
def naive():
    """
    Load stemming model using startswith and endswith naively using regex patterns.

    Returns
    -------
    result : malaya.stem.Naive class
    """
    tokenizer = Tokenizer().tokenize

    return Naive(tokenizer=tokenizer)


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
def deep_model(quantized: bool = False, **kwargs):
    """
    Load LSTM + Bahdanau Attention stemming model, this also include lemmatization.
    Original size 41.6MB, quantized size 10.6MB .

    Parameters
    ----------
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.stem.DeepStemmer class
    """

    path = check_file(
        file='lstm-bahdanau',
        module='stem',
        keys={'model': 'model.pb', 'vocab': STEMMER_VOCAB},
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)
    inputs = ['Placeholder']
    outputs = []
    bpe = YTTMEncoder(vocab_file=path['vocab'], id_mode=True)
    input_nodes, output_nodes = nodes_session(
        g,
        inputs,
        outputs,
        extra={
            'greedy': 'import/decode_1/greedy:0',
            'beam': 'import/decode_2/beam:0',
        },
    )

    tokenizer = Tokenizer().tokenize

    return DeepStemmer(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        bpe=bpe,
        tokenizer=tokenizer,
    )
