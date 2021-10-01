from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sentencepiece as spm
import unicodedata
import six
import logging
import collections
import tensorflow as tf
import regex as re
from functools import lru_cache
from malaya.text.function import transformer_textcleaning

SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_P = 0
SEG_ID_Q = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4

special_symbols = {
    '<unk>': 0,
    '<s>': 1,
    '</s>': 2,
    '<cls>': 3,
    '<sep>': 4,
    '<pad>': 5,
    '<mask>': 6,
    '<eod>': 7,
    '<eop>': 8,
}

UNK_ID = special_symbols['<unk>']
CLS_ID = special_symbols['<cls>']
SEP_ID = special_symbols['<sep>']
MASK_ID = special_symbols['<mask>']
EOD_ID = special_symbols['<eod>']

SPIECE_UNDERLINE = '▁'

MAXLEN = 508
SPECIAL_TOKENS = {
    'bert': {'pad': '[PAD]', 'cls': '[CLS]', 'sep': '[SEP]'},
    'xlnet': {'pad': '<pad>', 'cls': '<cls>', 'sep': '<sep>'},
}

BERT_TOKEN_MAPPING = {
    '-LRB-': '(',
    '-RRB-': ')',
    '-LCB-': '{',
    '-RCB-': '}',
    '-LSB-': '[',
    '-RSB-': ']',
    '``': '"',
    "''": '"',
    '`': "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    '\u2013': '--',  # en dash
    '\u2014': '--',  # em dash
}

PTB_TOKEN_ESCAPE = {
    '(': '-LRB-',
    ')': '-RRB-',
    '{': '-LCB-',
    '}': '-RCB-',
    '[': '-LSB-',
    ']': '-RSB-',
}


class SentencePieceTokenizer:
    def __init__(self, vocab_file, spm_model_file, **kwargs):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(spm_model_file)

        with open(vocab_file) as fopen:
            v = fopen.read().split('\n')[:-1]
        v = [i.split('\t') for i in v]
        self.vocab = {i[0]: i[1] for i in v}

    def tokenize(self, string):
        return encode_sentencepiece(
            self.sp_model, string, return_unicode=False, sample=False
        )

    def convert_tokens_to_ids(self, tokens):
        return [
            self.sp_model.PieceToId(printable_text(token)) for token in tokens
        ]

    def convert_ids_to_tokens(self, ids):
        return [self.sp_model.IdToPiece(id_) for id_ in ids]


class SentencePieceEncoder:
    def __init__(self, vocab_file, **kwargs):
        sp = spm.SentencePieceProcessor()
        sp.Load(vocab_file)

        self.sp = sp
        self.vocab_size = sp.GetPieceSize() + 100

    def encode(self, s):
        return self.sp.EncodeAsIds(s)

    def decode(self, ids, strip_extraneous=False):
        return self.sp.DecodeIds(list(ids))


class SentencePieceBatchEncoder:
    def __init__(self, vocab_file, **kwargs):
        sp = spm.SentencePieceProcessor()
        sp.Load(vocab_file)

        self.sp = sp
        self.vocab_size = sp.GetPieceSize() + 100

    def encode(self, s):
        s = [self.sp.EncodeAsIds(i) + [1] for i in s]
        return s

    def decode(self, ids, strip_extraneous=False):
        return [self.sp.DecodeIds(list(i)) for i in ids]


class YTTMEncoder:
    def __init__(self, vocab_file, id_mode=False, **kwargs):
        try:
            import youtokentome as yttm
        except BaseException:
            raise ModuleNotFoundError(
                'youtokentome not installed. Please install it by `pip install youtokentome` and try again.'
            )
        if id_mode:
            type = yttm.OutputType.ID
        else:
            type = yttm.OutputType.SUBWORD

        self.bpe = yttm.BPE(model=vocab_file)
        self.vocab_size = len(self.bpe.vocab())
        self.mode = type

    def encode(self, s):
        s = self.bpe.encode(s, output_type=self.mode)
        s = [i + [1] for i in s]
        return s

    def decode(self, ids, strip_extraneous=False):
        ids = [[k for k in i if k > 1] for i in ids]
        return self.bpe.decode(list(ids))


class WordPieceTokenizer(object):
    def __init__(self, vocab_file, do_lower_case=False, **kwargs):
        self.vocab = self.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = InternalWordPieceTokenizer(vocab=self.vocab)

    def load_vocab(self, vocab_file):
        vocab = collections.OrderedDict()
        index = 0
        with tf.compat.v1.gfile.GFile(vocab_file, 'r') as reader:
            while True:
                token = convert_to_unicode(reader.readline())
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return self.convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return self.convert_by_vocab(self.inv_vocab, ids)

    def convert_by_vocab(self, vocab, items):
        output = []
        for item in items:
            output.append(vocab[item])
        return output

    def encode(self, s):
        return self.convert_tokens_to_ids(self.tokenize(s))

    def decode(self, ids):
        tokens = self.convert_ids_to_tokens(ids)
        new_tokens = []
        n_tokens = len(tokens)
        i = 0
        while i < n_tokens:
            current_token = tokens[i]
            if current_token.startswith('##'):
                previous_token = new_tokens.pop()
                merged_token = previous_token
                while current_token.startswith('##'):
                    merged_token = merged_token + current_token.replace('##', '')
                    i = i + 1
                    current_token = tokens[i]
                new_tokens.append(merged_token)

            else:
                new_tokens.append(current_token)
                i = i + 1

        words = [
            i
            for i in new_tokens
            if i not in ['[CLS]', '[SEP]', '[PAD]']
        ]
        return ' '.join(words)


class BasicTokenizer(object):
    def __init__(self, do_lower_case=True, **kwargs):
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(' '.join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    def _run_split_on_punc(self, text):
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return [''.join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_chinese_char(self, cp):
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):
            return True

        return False

    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)


class InternalWordPieceTokenizer(object):
    def __init__(
        self, vocab, unk_token='[UNK]', max_input_chars_per_word=200
    ):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord('!'), ord('~') + 1))
        + list(range(ord('¡'), ord('¬') + 1))
        + list(range(ord('®'), ord('ÿ') + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class GPT2Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for
        # capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf'))
            )
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except BaseException:
                    new_word.extend(word[i:])
                    break

                if (
                    word[i] == first
                    and i < len(word) - 1
                    and word[i + 1] == second
                ):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(
                self.encoder[bpe_token]
                for bpe_token in self.bpe(token).split(' ')
            )
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors=self.errors
        )
        return text


def _is_whitespace(char):
    if char == ' ' or char == '\t' or char == '\n' or char == '\r':
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False


def _is_control(char):
    if char == '\t' or char == '\n' or char == '\r':
        return False
    cat = unicodedata.category(char)
    if cat in ('Cc', 'Cf'):
        return True
    return False


def _is_punctuation(char):
    cp = ord(char)
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False


def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def convert_to_unicode(text):
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode('utf-8', 'ignore')
        else:
            raise ValueError('Unsupported string type: %s' % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode('utf-8', 'ignore')
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError('Unsupported string type: %s' % (type(text)))
    else:
        raise ValueError('Not running on Python2 or Python 3?')


def padding_sequence(seq, maxlen=None, padding='post', pad_int=0):
    if not maxlen:
        maxlen = max([len(i) for i in seq])
    padded_seqs = []
    for s in seq:
        if padding == 'post':
            padded_seqs.append(s + [pad_int] * (maxlen - len(s)))
        if padding == 'pre':
            padded_seqs.append([pad_int] * (maxlen - len(s)) + s)
    return padded_seqs


def bert_tokenization(tokenizer, texts):
    input_ids, input_masks, segment_ids, s_tokens = [], [], [], []
    for text in texts:
        text = transformer_textcleaning(text)
        tokens_a = tokenizer.tokenize(text)[:MAXLEN]
        logging.debug(tokens_a)
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_id = [0] * len(tokens)
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_id)

        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        s_tokens.append(tokens)

    maxlen = max([len(i) for i in input_ids])
    input_ids = padding_sequence(input_ids, maxlen)
    input_masks = padding_sequence(input_masks, maxlen)
    segment_ids = padding_sequence(segment_ids, maxlen)

    return input_ids, input_masks, segment_ids, s_tokens


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def bert_tokenization_siamese(tokenizer, left, right):
    input_ids, input_masks, segment_ids, s_tokens = [], [], [], []
    a, b = [], []
    for i in range(len(left)):
        tokens_a = tokenizer.tokenize(transformer_textcleaning(left[i]))
        logging.debug(tokens_a)
        tokens_b = tokenizer.tokenize(transformer_textcleaning(right[i]))
        logging.debug(tokens_b)
        a.append(tokens_a)
        b.append(tokens_b)

    for i in range(len(left)):
        tokens_a = a[i]
        tokens_b = b[i]

        tokens = []
        segment_id = []
        tokens.append('[CLS]')
        segment_id.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_id.append(0)

        tokens.append('[SEP]')
        s_tokens.append(tokens[:])
        segment_id.append(0)
        for token in tokens_b:
            tokens.append(token)
            segment_id.append(1)
        tokens.append('[SEP]')
        segment_id.append(1)
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_id)

        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)

    maxlen = max([len(i) for i in input_ids])
    input_ids = padding_sequence(input_ids, maxlen)
    input_masks = padding_sequence(input_masks, maxlen)
    segment_ids = padding_sequence(segment_ids, maxlen)

    return input_ids, input_masks, segment_ids, s_tokens


def printable_text(text):
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode('utf-8', 'ignore')
        else:
            raise ValueError('Unsupported string type: %s' % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode('utf-8')
        else:
            raise ValueError('Unsupported string type: %s' % (type(text)))
    else:
        raise ValueError('Not running on Python2 or Python 3?')


def print_(*args):
    new_args = []
    for arg in args:
        if isinstance(arg, list):
            s = [printable_text(i) for i in arg]
            s = ' '.join(s)
            new_args.append(s)
        else:
            new_args.append(printable_text(arg))
    print(*new_args)


def preprocess_text(
    inputs, lower=False, remove_space=True, keep_accents=False
):
    if remove_space:
        outputs = ' '.join(inputs.strip().split())
    else:
        outputs = inputs
    outputs = outputs.replace('``', '"').replace("''", '"')

    if six.PY2 and isinstance(outputs, str):
        outputs = outputs.decode('utf-8')

    if not keep_accents:
        outputs = unicodedata.normalize('NFKD', outputs)
        outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def encode_sentencepiece(sp_model, text, return_unicode=True, sample=False):
    if six.PY2 and isinstance(text, unicode):
        text = text.encode('utf-8')

    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    for piece in pieces:
        if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                piece[:-1].replace(SPIECE_UNDERLINE, '')
            )
            if (
                piece[0] != SPIECE_UNDERLINE
                and cur_pieces[0][0] == SPIECE_UNDERLINE
            ):
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    if six.PY2 and return_unicode:
        ret_pieces = []
        for piece in new_pieces:
            if isinstance(piece, str):
                piece = piece.decode('utf-8')
            ret_pieces.append(piece)
        new_pieces = ret_pieces

    return new_pieces


def encode_sentencepiece_ids(tokenizer, text, sample=False):
    pieces = encode_sentencepiece(
        tokenizer.sp_model, text, return_unicode=False, sample=sample
    )
    logging.debug(pieces)
    ids = [tokenizer.sp_model.PieceToId(piece) for piece in pieces]
    return ids


def tokenize_xlnet_fn(text, tokenizer, sample=False):
    text = preprocess_text(text, lower=False)
    return encode_sentencepiece_ids(tokenizer, text)


def xlnet_tokenization_siamese(tokenizer, left, right):
    input_ids, input_mask, all_seg_ids, s_tokens = [], [], [], []
    for i in range(len(left)):
        tokens = tokenize_xlnet_fn(transformer_textcleaning(left[i]), tokenizer)
        tokens_right = tokenize_xlnet_fn(
            transformer_textcleaning(right[i]), tokenizer
        )
        segment_ids = [SEG_ID_A] * len(tokens)
        tokens.append(SEP_ID)
        s_tokens.append([tokenizer.sp_model.IdToPiece(i) for i in tokens])
        segment_ids.append(SEG_ID_A)

        tokens.extend(tokens_right)
        segment_ids.extend([SEG_ID_B] * len(tokens_right))
        tokens.append(SEP_ID)
        segment_ids.append(SEG_ID_B)

        tokens.append(CLS_ID)
        segment_ids.append(SEG_ID_CLS)

        cur_input_ids = tokens
        cur_input_mask = [0] * len(cur_input_ids)
        logging.debug(tokens)
        assert len(tokens) == len(cur_input_mask)
        assert len(tokens) == len(segment_ids)

        input_ids.append(tokens)
        input_mask.append(cur_input_mask)
        all_seg_ids.append(segment_ids)

    maxlen = max([len(i) for i in input_ids])
    input_ids = padding_sequence(input_ids, maxlen)
    input_mask = padding_sequence(input_mask, maxlen, pad_int=1)
    all_seg_ids = padding_sequence(all_seg_ids, maxlen, pad_int=4)
    return input_ids, input_mask, all_seg_ids, s_tokens


def xlnet_tokenization(tokenizer, texts, space_after_punct=False):
    input_ids, input_masks, segment_ids, s_tokens = [], [], [], []
    for text in texts:
        text = transformer_textcleaning(text, space_after_punct=space_after_punct)
        tokens_a = tokenize_xlnet_fn(text, tokenizer)[:MAXLEN]
        tokens = []
        segment_id = []
        for token in tokens_a:
            tokens.append(token)
            segment_id.append(SEG_ID_A)

        tokens.append(SEP_ID)
        segment_id.append(SEG_ID_A)
        tokens.append(CLS_ID)
        segment_id.append(SEG_ID_CLS)

        input_id = tokens
        input_mask = [0] * len(input_id)
        logging.debug(tokens)

        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        s_tokens.append([tokenizer.sp_model.IdToPiece(i) for i in tokens])

    maxlen = max([len(i) for i in input_ids])
    input_ids = padding_sequence(input_ids, maxlen)
    input_masks = padding_sequence(input_masks, maxlen, pad_int=1)
    segment_ids = padding_sequence(segment_ids, maxlen, pad_int=SEG_ID_PAD)

    return input_ids, input_masks, segment_ids, s_tokens


def xlnet_tokenization_token(tokenizer, tok, texts):
    input_ids, input_masks, segment_ids, s_tokens = [], [], [], []
    for text in texts:
        tokens = []
        for no, orig_token in enumerate(tok(text)):
            tokens_a = tokenize_xlnet_fn(orig_token, tokenizer)
            tokens.extend(tokens_a)
        tokens.extend([SEP_ID, CLS_ID])
        segment = [SEG_ID_A] * (len(tokens) - 1) + [SEG_ID_CLS]
        input_id = tokens
        input_mask = [0] * len(input_id)
        logging.debug(tokens)
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment)
        s_tokens.append([tokenizer.sp_model.IdToPiece(i) for i in tokens])

    maxlen = max([len(i) for i in input_ids])
    input_ids = padding_sequence(input_ids, maxlen)
    input_masks = padding_sequence(input_masks, maxlen, pad_int=1)
    segment_ids = padding_sequence(segment_ids, maxlen, pad_int=SEG_ID_PAD)

    return input_ids, input_masks, segment_ids, s_tokens


def merge_wordpiece_tokens(paired_tokens, weighted=True):
    new_paired_tokens = []
    n_tokens = len(paired_tokens)

    i = 0

    while i < n_tokens:
        current_token, current_weight = paired_tokens[i]
        if current_token.startswith('##'):
            previous_token, previous_weight = new_paired_tokens.pop()
            merged_token = previous_token
            merged_weight = [previous_weight]
            while current_token.startswith('##'):
                merged_token = merged_token + current_token.replace('##', '')
                merged_weight.append(current_weight)
                i = i + 1
                current_token, current_weight = paired_tokens[i]
            merged_weight = np.mean(merged_weight)
            new_paired_tokens.append((merged_token, merged_weight))

        else:
            new_paired_tokens.append((current_token, current_weight))
            i = i + 1

    words = [
        i[0]
        for i in new_paired_tokens
        if i[0] not in ['[CLS]', '[SEP]', '[PAD]']
    ]
    weights = [
        i[1]
        for i in new_paired_tokens
        if i[0] not in ['[CLS]', '[SEP]', '[PAD]']
    ]
    if weighted:
        weights = np.array(weights)
        weights = weights / np.sum(weights)
    return list(zip(words, weights))


def parse_bert_tagging(left, tokenizer, space_after_punct=False):
    left = transformer_textcleaning(left, space_after_punct=space_after_punct)
    bert_tokens = ['[CLS]'] + tokenizer.tokenize(left) + ['[SEP]']
    input_mask = [1] * len(bert_tokens)
    logging.debug(bert_tokens)
    return tokenizer.convert_tokens_to_ids(bert_tokens), input_mask, bert_tokens


def parse_bert_token_tagging(left, tok, tokenizer):
    bert_tokens = ['[CLS]']
    for no, orig_token in enumerate(tok(left)):
        t = tokenizer.tokenize(orig_token)
        bert_tokens.extend(t)
    bert_tokens.append('[SEP]')
    input_mask = [1] * len(bert_tokens)
    logging.debug(bert_tokens)
    return tokenizer.convert_tokens_to_ids(bert_tokens), input_mask, bert_tokens


def merge_sentencepiece_tokens(
    paired_tokens, weighted=True, vectorize=False, model='bert'
):
    new_paired_tokens = []
    n_tokens = len(paired_tokens)
    rejected = list(SPECIAL_TOKENS[model].values())

    i = 0

    while i < n_tokens:

        current_token, current_weight = paired_tokens[i]
        if isinstance(current_token, bytes):
            current_token = current_token.decode()
        if not current_token.startswith('▁') and current_token not in rejected:
            previous_token, previous_weight = new_paired_tokens.pop()
            merged_token = previous_token
            merged_weight = [previous_weight]
            while (
                not current_token.startswith('▁')
                and current_token not in rejected
            ):
                merged_token = merged_token + current_token.replace('▁', '')
                merged_weight.append(current_weight)
                i = i + 1
                current_token, current_weight = paired_tokens[i]
            if vectorize:
                merged_weight = np.mean(merged_weight, axis=0)
            else:
                merged_weight = np.mean(merged_weight)
            new_paired_tokens.append((merged_token, merged_weight))

        else:
            new_paired_tokens.append((current_token, current_weight))
            i = i + 1

    words = [
        i[0].replace('▁', '') for i in new_paired_tokens if i[0] not in rejected
    ]
    weights = [i[1] for i in new_paired_tokens if i[0] not in rejected]
    if weighted:
        weights = np.array(weights)
        weights = weights / np.sum(weights)
    return list(zip(words, weights))


def merge_sentencepiece_tokens_tagging(x, y, model='bert'):
    new_paired_tokens = []
    n_tokens = len(x)
    rejected = list(SPECIAL_TOKENS[model].values())

    i = 0

    while i < n_tokens:

        current_token, current_label = x[i], y[i]

        if isinstance(current_token, bytes):
            current_token = current_token.decode()
        if not current_token.startswith('▁') and current_token not in rejected:
            previous_token, previous_label = new_paired_tokens.pop()
            merged_token = previous_token
            merged_label = [previous_label]
            while (
                not current_token.startswith('▁')
                and current_token not in rejected
            ):
                merged_token = merged_token + current_token.replace('▁', '')
                merged_label.append(current_label)
                i = i + 1
                current_token, current_label = x[i], y[i]
            merged_label = merged_label[0]
            new_paired_tokens.append((merged_token, merged_label))

        else:
            new_paired_tokens.append((current_token, current_label))
            i = i + 1

    words = [
        i[0].replace('▁', '') for i in new_paired_tokens if i[0] not in rejected
    ]
    labels = [i[1] for i in new_paired_tokens if i[0] not in rejected]
    return words, labels


def constituency_bert(tokenizer, sentences):
    all_input_ids, all_word_end_mask, all_tokens = [], [], []

    subword_max_len = 0
    for snum, sentence in enumerate(sentences):
        tokens = []
        word_end_mask = []

        tokens.append('[CLS]')
        word_end_mask.append(1)

        cleaned_words = []
        for word in sentence:
            word = BERT_TOKEN_MAPPING.get(word, word)
            if word == "n't" and cleaned_words:
                cleaned_words[-1] = cleaned_words[-1] + 'n'
                word = "'t"
            cleaned_words.append(word)

        for word in cleaned_words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = ['[UNK]']
            for _ in range(len(word_tokens)):
                word_end_mask.append(0)
            word_end_mask[-1] = 1
            tokens.extend(word_tokens)
        tokens.append('[SEP]')
        word_end_mask.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        logging.debug(tokens)
        all_input_ids.append(input_ids)
        all_word_end_mask.append(word_end_mask)
        all_tokens.append(tokens)

    return all_input_ids, all_word_end_mask, all_tokens


def constituency_xlnet(tokenizer, sentences):
    all_input_ids, all_word_end_mask, all_tokens = [], [], []

    subword_max_len = 0
    for snum, sentence in enumerate(sentences):
        tokens = []
        word_end_mask = []

        cleaned_words = []
        for word in sentence:
            word = BERT_TOKEN_MAPPING.get(word, word)
            if word == "n't" and cleaned_words:
                cleaned_words[-1] = cleaned_words[-1] + 'n'
                word = "'t"
            cleaned_words.append(word)

        for word in cleaned_words:
            word_tokens = encode_sentencepiece(
                tokenizer.sp_model, word, return_unicode=False, sample=False
            )
            if not word_tokens:
                word_tokens = ['<unk>']
            for _ in range(len(word_tokens)):
                word_end_mask.append(0)
            word_end_mask[-1] = 1
            tokens.extend(word_tokens)

        tokens.append('<sep>')
        word_end_mask.append(1)
        tokens.append('<cls>')
        logging.debug(tokens)
        word_end_mask.append(1)

        input_ids = [tokenizer.sp_model.PieceToId(i) for i in tokens]
        all_input_ids.append(input_ids)
        all_word_end_mask.append(word_end_mask)
        all_tokens.append(tokens)

    return all_input_ids, all_word_end_mask, all_tokens
