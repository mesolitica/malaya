import re
import json
import os
from unidecode import unidecode
from malaya.text.tatabahasa import permulaan, hujung
from malaya.text.rules import rules_normalizer
from malaya.function import load_graph, check_file, generate_session
from malaya.text.function import pad_sentence_batch, case_of
from malaya.text.bpe import load_yttm
from malaya.text.regex import _expressions, _money, _date
from malaya.path import PATH_STEM, S3_PATH_STEM
from herpetologist import check_type

factory = None
sastrawi_stemmer = None


def _load_sastrawi():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    global factory, sastrawi_stemmer
    factory = StemmerFactory()
    sastrawi_stemmer = factory.create_stemmer()


def _classification_textcleaning_stemmer(string):
    string = re.sub(
        'http\S+|www.\S+',
        '',
        ' '.join(
            [i for i in string.split() if i.find('#') < 0 and i.find('@') < 0]
        ),
    )
    string = unidecode(string).replace('.', ' . ').replace(',', ' , ')
    string = re.sub('[^A-Za-z ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string.lower()).strip()
    string = [rules_normalizer.get(w, w) for w in string.split()]
    string = [(naive(word), word) for word in string]
    return ' '.join([word[0] for word in string if len(word[0]) > 1])


class DEEP_STEMMER:
    def __init__(self, X, greedy, beam, sess, bpe, subword_mode, tokenizer):

        self._X = X
        self._greedy = greedy
        self._beam = beam
        self._sess = sess
        self._bpe = bpe
        self._subword_mode = subword_mode
        self._tokenizer = tokenizer

    @check_type
    def stem(self, string: str, beam_search: bool = True):
        """
        Stem a string, this also include lemmatization.

        Parameters
        ----------
        string : str
        beam_search : bool, (optional=True)
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

            batch = self._bpe.encode(batch, output_type = self._subword_mode)

            batch = [i + [1] for i in batch]
            batch = pad_sentence_batch(batch, 0)[0]

            if beam_search:
                output = self._beam
            else:
                output = self._greedy

            output = self._sess.run(output, feed_dict = {self._X: batch})
            output = output.tolist()

            for no, o in enumerate(output):
                predicted = list(dict.fromkeys(o))
                predicted = self._bpe.decode(predicted)[0].replace('<EOS>', '')
                predicted = case_of(actual[no])(predicted)
                result[mapping[no]] = predicted

        return ' '.join(result)


@check_type
def naive(word: str):
    """
    Stem a string using startswith and endswith.

    Parameters
    ----------
    string : str

    Returns
    -------
    result : str
        stemmed string.
    """
    hujung_result = [v for k, v in hujung.items() if word.endswith(k)]
    if len(hujung_result):
        hujung_result = max(hujung_result, key = len)
        if len(hujung_result):
            word = word[: -len(hujung_result)]
    permulaan_result = [v for k, v in permulaan.items() if word.startswith(k)]
    if len(permulaan_result):
        permulaan_result = max(permulaan_result, key = len)
        if len(permulaan_result):
            word = word[len(permulaan_result) :]
    return word


@check_type
def sastrawi(string: str):
    """
    Stem a string using Sastrawi, this also include lemmatization.

    Parameters
    ----------
    string : str

    Returns
    -------
    result: str
        stemmed string.
    """
    if sastrawi_stemmer is None:
        _load_sastrawi()
    return sastrawi_stemmer.stem(string)


def deep_model(**kwargs):
    """
    Load LSTM + Bahdanau Attention stemming model, this also include lemmatization.

    Returns
    -------
    result: malaya.stem.DEEP_STEMMER class
    """
    from malaya.preprocessing import _tokenizer

    check_file(PATH_STEM['deep'], S3_PATH_STEM['deep'], **kwargs)
    g = load_graph(PATH_STEM['deep']['model'], **kwargs)

    bpe, subword_mode = load_yttm(PATH_STEM['deep']['bpe'], id_mode = True)

    return DEEP_STEMMER(
        g.get_tensor_by_name('import/Placeholder:0'),
        g.get_tensor_by_name('import/decode_1/greedy:0'),
        g.get_tensor_by_name('import/decode_2/beam:0'),
        generate_session(graph = g, **kwargs),
        bpe,
        subword_mode,
        _tokenizer,
    )
