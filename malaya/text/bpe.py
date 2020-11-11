from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from malaya.text.function import transformer_textcleaning

import numpy as np
import sentencepiece as spm
import unicodedata
import six

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
    def __init__(self, v, sp_model):
        self.vocab = v
        self.sp_model = sp_model

    def tokenize(self, string):
        return encode_pieces(
            self.sp_model, string, return_unicode = False, sample = False
        )

    def convert_tokens_to_ids(self, tokens):
        return [self.sp_model.PieceToId(piece) for piece in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.sp_model.IdToPiece(i) for i in ids]


class SentencePieceEncoder:
    def __init__(self, vocab):
        sp = spm.SentencePieceProcessor()
        sp.Load(vocab)

        self.sp = sp
        self.vocab_size = sp.GetPieceSize() + 100

    def encode(self, s):
        return self.sp.EncodeAsIds(s)

    def decode(self, ids, strip_extraneous = False):
        return self.sp.DecodeIds(list(ids))


class YTTMEncoder:
    def __init__(self, bpe, mode):
        self.bpe = bpe
        self.vocab_size = len(self.bpe.vocab())
        self.mode = mode

    def encode(self, s):
        s = self.bpe.encode(s, output_type = self.mode)
        s = [i + [1] for i in s]
        return s

    def decode(self, ids, strip_extraneous = False):
        ids = [[k for k in i if k > 1] for i in ids]
        return self.bpe.decode(list(ids))


def padding_sequence(seq, maxlen = None, padding = 'post', pad_int = 0):
    if not maxlen:
        maxlen = max([len(i) for i in seq])
    padded_seqs = []
    for s in seq:
        if padding == 'post':
            padded_seqs.append(s + [pad_int] * (maxlen - len(s)))
        if padding == 'pre':
            padded_seqs.append([pad_int] * (maxlen - len(s)) + s)
    return padded_seqs


def bert_tokenization(tokenizer, texts, cleaning = transformer_textcleaning):
    input_ids, input_masks, segment_ids, s_tokens = [], [], [], []
    for text in texts:
        if cleaning:
            text = cleaning(text)
        tokens_a = tokenizer.tokenize(text)[:MAXLEN]
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
        tokens_b = tokenizer.tokenize(transformer_textcleaning(right[i]))
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


SEG_ID_A = 0
SEG_ID_B = 1
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


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
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
    inputs, lower = False, remove_space = True, keep_accents = False
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


def encode_pieces(sp_model, text, return_unicode = True, sample = False):
    # return_unicode is used only for py2

    # note(zhiliny): in some systems, sentencepiece only accepts str for py2
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

    # note(zhiliny): convert back to unicode for py2
    if six.PY2 and return_unicode:
        ret_pieces = []
        for piece in new_pieces:
            if isinstance(piece, str):
                piece = piece.decode('utf-8')
            ret_pieces.append(piece)
        new_pieces = ret_pieces

    return new_pieces


def encode_ids(sp_model, text, sample = False):
    pieces = encode_pieces(
        sp_model, text, return_unicode = False, sample = sample
    )
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return ids


def tokenize_fn(text, sp_model):
    text = preprocess_text(text, lower = False)
    return encode_ids(sp_model, text)


def xlnet_tokenization_siamese(tokenizer, left, right):
    input_ids, input_mask, all_seg_ids, s_tokens = [], [], [], []
    for i in range(len(left)):
        tokens = tokenize_fn(transformer_textcleaning(left[i]), tokenizer)
        tokens_right = tokenize_fn(
            transformer_textcleaning(right[i]), tokenizer
        )
        segment_ids = [SEG_ID_A] * len(tokens)
        tokens.append(SEP_ID)
        s_tokens.append([tokenizer.IdToPiece(i) for i in tokens])
        segment_ids.append(SEG_ID_A)

        tokens.extend(tokens_right)
        segment_ids.extend([SEG_ID_B] * len(tokens_right))
        tokens.append(SEP_ID)
        segment_ids.append(SEG_ID_B)

        tokens.append(CLS_ID)
        segment_ids.append(SEG_ID_CLS)

        cur_input_ids = tokens
        cur_input_mask = [0] * len(cur_input_ids)
        assert len(tokens) == len(cur_input_mask)
        assert len(tokens) == len(segment_ids)
        input_ids.append(tokens)
        input_mask.append(cur_input_mask)
        all_seg_ids.append(segment_ids)

    maxlen = max([len(i) for i in input_ids])
    input_ids = padding_sequence(input_ids, maxlen)
    input_mask = padding_sequence(input_mask, maxlen, pad_int = 1)
    all_seg_ids = padding_sequence(all_seg_ids, maxlen, pad_int = 4)
    return input_ids, input_mask, all_seg_ids, s_tokens


def xlnet_tokenization(tokenizer, texts):
    input_ids, input_masks, segment_ids, s_tokens = [], [], [], []
    for text in texts:
        text = transformer_textcleaning(text)
        tokens_a = tokenize_fn(text, tokenizer)[:MAXLEN]
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

        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        s_tokens.append([tokenizer.IdToPiece(i) for i in tokens])

    maxlen = max([len(i) for i in input_ids])
    input_ids = padding_sequence(input_ids, maxlen)
    input_masks = padding_sequence(input_masks, maxlen, pad_int = 1)
    segment_ids = padding_sequence(segment_ids, maxlen, pad_int = SEG_ID_PAD)

    return input_ids, input_masks, segment_ids, s_tokens


def merge_wordpiece_tokens(paired_tokens, weighted = True):
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


def parse_bert_tagging(left, tokenizer):
    left = transformer_textcleaning(left)
    bert_tokens = ['[CLS]'] + tokenizer.tokenize(left) + ['[SEP]']
    input_mask = [1] * len(bert_tokens)
    return tokenizer.convert_tokens_to_ids(bert_tokens), input_mask, bert_tokens


def merge_sentencepiece_tokens(
    paired_tokens, weighted = True, vectorize = False, model = 'bert'
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
                merged_weight = np.mean(merged_weight, axis = 0)
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


def merge_sentencepiece_tokens_tagging(x, y, model = 'bert'):
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


def sentencepiece_tokenizer_xlnet(path_tokenizer):
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(path_tokenizer)
    return sp_model


def sentencepiece_tokenizer_bert(path_tokenizer, path_vocab):

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(path_tokenizer)

    with open(path_vocab) as fopen:
        v = fopen.read().split('\n')[:-1]
    v = [i.split('\t') for i in v]
    v = {i[0]: i[1] for i in v}
    tokenizer = SentencePieceTokenizer(v, sp_model)
    return tokenizer


def load_yttm(path, id_mode = False):
    try:
        import youtokentome as yttm
    except:
        raise ModuleNotFoundError(
            'youtokentome not installed. Please install it by `pip install youtokentome` and try again.'
        )
    try:
        if id_mode:
            type = yttm.OutputType.ID
        else:
            type = yttm.OutputType.SUBWORD
        return yttm.BPE(model = path), type
    except:
        path = path.split('Malaya/')[1]
        path = '/'.join(path.split('/')[:-1])
        raise Exception(
            f"model corrupted due to some reasons, please run malaya.clear_cache('{path}') and try again"
        )


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
            word_tokens = encode_pieces(
                tokenizer, word, return_unicode = False, sample = False
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
        word_end_mask.append(1)

        input_ids = [tokenizer.PieceToId(i) for i in tokens]
        all_input_ids.append(input_ids)
        all_word_end_mask.append(word_end_mask)
        all_tokens.append(tokens)

    return all_input_ids, all_word_end_mask, all_tokens
