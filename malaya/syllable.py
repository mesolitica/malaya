import re
import logging
from malaya.supervised.t2t import load_lstm_yttm
from malaya.model.abstract import Abstract
from malaya.text.function import PUNCTUATION, case_of, is_emoji
from malaya.text.regex import _expressions, _money, _date
from malaya.path import SYLLABLE_VOCAB
from malaya.function import describe_availability
from herpetologist import check_type

logger = logging.getLogger(__name__)


def replace_same_length(l, r):
    l = l.replace('-', '')
    if len(l) != len(r.replace('.', '')):
        return False, r

    index = {}
    i = 0
    for no, c in enumerate(r):
        if c not in '.O':
            index[no] = i
            i += 1
    index = {v: k for k, v in index.items()}

    r = list(r)

    for no, c in enumerate(l):
        if c != r[index[no]]:
            r[index[no]] = c

    return True, ''.join(r)


class Base:
    @check_type
    def tokenize(self, string: str, **kwargs):
        word = string
        if (
            word in PUNCTUATION
            or re.findall(_money, word.lower())
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
        ):
            return string
        return self.tokenize_word(word, **kwargs)


class Tokenizer(Base):
    def __init__(self):
        """
        originally from https://github.com/fahadh4ilyas/syllable_splitter/blob/master/SyllableSplitter.py
        - improved `cuaca` double vocal `ua` based on https://en.wikipedia.org/wiki/Comparison_of_Indonesian_and_Standard_Malay#Syllabification
        - improved `rans` double consonant `ns` based on https://www.semanticscholar.org/paper/Syllabification-algorithm-based-on-syllable-rules-Musa-Kadir/a819f255f066ae0fd7a30b3534de41da37d04ea1
        - improved `au` and `ai` double vocal.
        """
        self.consonant = ['b', 'c', 'd', 'f', 'g', 'h', 'j',
                          'k', 'l', 'm', 'n', 'p', 'q', 'r',
                          's', 't', 'v', 'w', 'x', 'y', 'z',
                          'ng', 'ny', 'sy', 'ch', 'dh', 'gh',
                          'kh', 'ph', 'sh', 'th']

        self.double_consonant = ['ll', 'ks', 'rs', 'rt', 'ns']

        self.double_vocal = ['ua', 'au', 'ai', 'io']

        self.vocal = ['a', 'e', 'i', 'o', 'u']

    def split_letters(self, string):
        letters = []
        arrange = []

        while string != '':
            letter = string[:2]
            logger.info(f'letter: {letter}, string: {string}')

            if letter.lower() in self.double_consonant:

                if string[2:] != '' and string[2].lower() in self.vocal:
                    letters += [letter[0]]
                    arrange += ['c']
                    string = string[1:]

                else:
                    letters += [letter]
                    arrange += ['c']
                    string = string[2:]

            elif letter.lower() in self.double_vocal:
                letters += [letter]
                arrange += ['v']
                string = string[2:]

            elif letter.lower() in self.consonant:
                letters += [letter]
                arrange += ['c']
                string = string[2:]

            elif letter.lower() in self.vocal:
                letters += [letter]
                arrange += ['v']
                string = string[2:]

            else:
                letter = string[0]

                if letter.lower() in self.consonant:
                    letters += [letter]
                    arrange += ['c']
                    string = string[1:]

                elif letter.lower() in self.vocal:
                    letters += [letter]
                    arrange += ['v']
                    string = string[1:]

                else:
                    letters += [letter]
                    arrange += ['s']
                    string = string[1:]

        return letters, ''.join(arrange)

    def split_syllables_from_letters(self, letters, arrange):
        consonant_index = re.search('vc{2,}', arrange)
        while consonant_index:
            i = consonant_index.start()+1
            letters = letters[:i+1]+['|']+letters[i+1:]
            arrange = arrange[:i+1]+'|'+arrange[i+1:]
            consonant_index = re.search('vc{2,}', arrange)

        vocal_index = re.search(r'v{2,}', arrange)
        while vocal_index:
            i = vocal_index.start()
            letters = letters[:i+1]+['|']+letters[i+1:]
            arrange = arrange[:i+1]+'|'+arrange[i+1:]
            vocal_index = re.search(r'v{2,}', arrange)

        vcv_index = re.search(r'vcv', arrange)
        while vcv_index:
            i = vcv_index.start()
            letters = letters[:i+1]+['|']+letters[i+1:]
            arrange = arrange[:i+1]+'|'+arrange[i+1:]
            vcv_index = re.search(r'vcv', arrange)

        sep_index = re.search(r'[cvs]s', arrange)
        while sep_index:
            i = sep_index.start()
            letters = letters[:i+1]+['|']+letters[i+1:]
            arrange = arrange[:i+1]+'|'+arrange[i+1:]
            sep_index = re.search(r'[cvs]s', arrange)

        sep_index = re.search(r's[cvs]', arrange)
        while sep_index:
            i = sep_index.start()
            letters = letters[:i+1]+['|']+letters[i+1:]
            arrange = arrange[:i+1]+'|'+arrange[i+1:]
            sep_index = re.search(r's[cvs]', arrange)

        return ''.join(letters).split('|')

    def tokenize_word(self, string: str):

        letters, arrange = self.split_letters(string)
        return self.split_syllables_from_letters(letters, arrange)

    @check_type
    def tokenize(self, string: str):
        """
        Tokenize string into multiple strings using syllable patterns.
        Example from https://www.semanticscholar.org/paper/Syllabification-algorithm-based-on-syllable-rules-Musa-Kadir/a819f255f066ae0fd7a30b3534de41da37d04ea1/figure/0,
        'cuaca' -> ['cua', 'ca']
        'insurans' -> ['in', 'su', 'rans']
        'praktikal' -> ['prak', 'ti', 'kal']
        'strategi' -> ['stra', 'te', 'gi']
        'ayam' -> ['a', 'yam']
        'anda' -> ['an', 'da']
        'hantu' -> ['han', 'tu']

        Parameters
        ----------
        string : str

        Returns
        -------
        result: List[str]
        """
        return super().tokenize(string)


class DeepSyllable(Abstract, Base):
    def __init__(
        self, input_nodes, output_nodes, sess, bpe, **kwargs,
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self._bpe = bpe

    def tokenize_word(self, string: str, beam_search: bool = False, **kwargs):

        batch = self._bpe.bpe.encode([string], output_type=self._bpe.mode)
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

        r_ = replace_same_length(string, predicted)
        if r_[0]:
            predicted = r_[1]
        return predicted.split('.')

    @check_type
    def tokenize(self, string, beam_search: bool = False):
        """
        Tokenize string into multiple strings using deep learning.

        Parameters
        ----------
        string : str
        beam_search : bool, (optional=False)
            If True, use beam search decoder, else use greedy decoder.

        Returns
        -------
        result: List[str]
        """

        return super().tokenize(string, beam_search=beam_search)


_availability = {
    'base': {
        'Size (MB)': 11.7,
        'Quantized Size (MB)': 3.1,
        'CER': 0.00776934,
        'WER': 0.04303278,
    },
}


def available_deep_model():
    """
    List available syllable tokenizer deep models.
    """
    logger.info('trained on 95% dataset, tested on another 5% test set, dataset at https://github.com/huseinzol05/malay-dataset/tree/master/tokenization/syllable')

    return describe_availability(_availability)


def rules(**kwargs):
    """
    Load rules based syllable tokenizer.
    originally from https://github.com/fahadh4ilyas/syllable_splitter/blob/master/SyllableSplitter.py
    - improved `cuaca` double vocal `ua` based on https://en.wikipedia.org/wiki/Comparison_of_Indonesian_and_Standard_Malay#Syllabification
    - improved `rans` double consonant `ns` based on https://www.semanticscholar.org/paper/Syllabification-algorithm-based-on-syllable-rules-Musa-Kadir/a819f255f066ae0fd7a30b3534de41da37d04ea1
    - improved `au` and `ai` double vocal.

    Returns
    -------
    result: malaya.syllable.Tokenizer class
    """
    return Tokenizer()


@check_type
def deep_model(model: str = 'base', quantized: bool = False, **kwargs):
    """
    Load LSTM + Bahdanau Attention syllable tokenizer model, BPE level (YouTokenToMe 300 vocab size).

    Parameters
    ----------
    model: str, optional (default='base')
        Check available models at `malaya.syllable.available_deep_model()`.
    quantized: bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.syllable.DeepSyllable class
    """

    model = model.lower()
    if model not in _availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.syllable.available_deep_model()`.'
        )

    return load_lstm_yttm(
        module='syllable',
        vocab=SYLLABLE_VOCAB,
        model_class=DeepSyllable,
        quantized=quantized,
        model=model,
        **kwargs,
    )
