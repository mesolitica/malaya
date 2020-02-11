import re
import json
from unidecode import unidecode
from .texts._tatabahasa import permulaan, hujung, rules_normalizer
from ._utils._utils import (
    load_graph,
    check_file,
    check_available,
    generate_session,
)
from .texts._text_functions import (
    pad_sentence_batch,
    stemmer_str_idx,
    classification_textcleaning,
)
from ._utils._paths import PATH_STEM, S3_PATH_STEM
from . import home
from herpetologist import check_type

factory = None
sastrawi_stemmer = None
PAD = 0
GO = 1
EOS = 2
UNK = 3


def _load_sastrawi():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    global factory, sastrawi_stemmer
    factory = StemmerFactory()
    sastrawi_stemmer = factory.create_stemmer()


def _classification_textcleaning_stemmer(string, attention = False):
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
    if attention:
        return (
            ' '.join([word[0] for word in string if len(word[0]) > 1]),
            ' '.join([word[1] for word in string if len(word[0]) > 1]),
        )
    else:
        return ' '.join([word[0] for word in string if len(word[0]) > 1])


class _DEEP_STEMMER:
    def __init__(self, x, logits, sess, dicts):
        self._sess = sess
        self._x = x
        self._logits = logits
        self._dicts = dicts
        self._dicts['rev_dictionary_to'] = {
            int(k): v for k, v in self._dicts['rev_dictionary_to'].items()
        }

    @check_type
    def stem(self, string: str):
        """
        Stem a string.

        Parameters
        ----------
        string : str

        Returns
        -------
        string: stemmed string
        """
        token_strings = classification_textcleaning(string, True).split()
        idx = stemmer_str_idx(token_strings, self._dicts['dictionary_from'])
        predicted = self._sess.run(
            self._logits, feed_dict = {self._x: pad_sentence_batch(idx, PAD)[0]}
        )
        results = []
        for word in predicted:
            results.append(
                ''.join(
                    [
                        self._dicts['rev_dictionary_to'][c]
                        for c in word
                        if c not in [GO, PAD, EOS, UNK]
                    ]
                )
            )
        return ' '.join(results)


@check_type
def naive(word: str):
    """
    Stem a string using startswith and endswith.

    Parameters
    ----------
    string : str

    Returns
    -------
    string: stemmed string
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


def available_deep_model():
    """
    List available deep learning stemming models.
    """
    return ['lstm', 'bahdanau', 'luong']


@check_type
def sastrawi(string: str):
    """
    Stem a string using Sastrawi.

    Parameters
    ----------
    string : str

    Returns
    -------
    string: stemmed string.
    """
    if sastrawi_stemmer is None:
        _load_sastrawi()
    return sastrawi_stemmer.stem(string)


@check_type
def deep_model(model: str = 'bahdanau', validate: bool = True):
    """
    Load seq2seq stemmer deep learning model.

    Returns
    -------
    DEEP_STEMMER: malaya.stemmer._DEEP_STEMMER class
    """
    if validate:
        check_file(PATH_STEM[model], S3_PATH_STEM[model])
    else:
        if not check_available(PATH_STEM[model]):
            raise Exception(
                'stem/%s is not available, please `validate = True`' % (model)
            )
    try:
        with open(PATH_STEM[model]['setting'], 'r') as fopen:
            dic_stemmer = json.load(fopen)
        g = load_graph(PATH_STEM[model]['model'])
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('stem/%s') and try again"
            % (model)
        )
    return _DEEP_STEMMER(
        g.get_tensor_by_name('import/Placeholder:0'),
        g.get_tensor_by_name('import/logits:0'),
        generate_session(graph = g),
        dic_stemmer,
    )
