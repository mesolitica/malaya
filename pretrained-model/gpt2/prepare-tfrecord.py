import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/husein/t5/prepare/mesolitica-tpu.json'

import itertools
from glob import glob
from multiprocessing import Pool
from google.cloud import storage
import tensorflow as tf
import json
import regex as re
from functools import lru_cache
import tensorflow as tf
import gpt_2_simple
from tqdm import tqdm
import collections
import numpy as np


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


class Encoder:
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


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values))
    )
    return feature


def write_tfrecord(s, file):
    r = tf.python_io.TFRecordWriter(file)
    for i in tqdm(range(len(s))):
        features = collections.OrderedDict()
        features['input_ids'] = create_int_feature(s[i])
        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features)
        )
        r.write(tf_example.SerializeToString())


def chunks(l, n):
    count = 0
    for i in range(0, len(l), n):
        yield l[i: i + n], count
        count += 1


def multiprocessing(strings, function, cores=16):
    df_split = chunks(strings, len(strings) // cores)
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()


with open('encoder.json', 'r') as f:
    en = json.load(f)
with open('vocab.bpe', 'r', encoding="utf-8") as f:
    bpe_data = f.read()

bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
enc_malay = Encoder(
    encoder=en,
    bpe_merges=bpe_merges,
)

length = 1024
combine = 50000
files = glob('dumping-*.txt')
files = [i for i in files if 'twitter' not in i and 'instagram' not in i and 'combined' not in i]
files.extend(glob('dialogpt/*.txt'))
files.extend(glob('the-pile/*.txt'))


def get_multiline(file):
    with open(file) as fopen:
        data = fopen.read().split('\n')

    results, result = [], []
    for i in data:
        if len(i) and i[-1] != '.':
            i = i + '.'
        if not len(i) and len(result):
            results.append(result)
            result = []
        else:
            if len(i):
                result.append(i)

    if len(result):
        results.append(result)

    return results


global_count = 0


def loop(files):
    client = storage.Client()
    bucket = client.bucket('mesolitica-tpu-general')
    files, index = files

    token_chunks = []
    raw_text = ''
    output_file = f'{index}-{global_count}.tfrecord'

    for file in tqdm(files):
        with open(file, 'r', encoding='utf8', errors='ignore') as fp:
            raw_text += fp.read()

        if len(raw_text) >= combine:
            tokens = enc_malay.encode(raw_text)
            token_chunks.append(tokens)
            raw_text = ''
        else:
            raw_text += '<|endoftext|>'

    s = []
    for l in range(len(token_chunks)):
        for i in range(0, len(token_chunks[l]), length):
            index = min(i + length, len(token_chunks[l]))
            x = token_chunks[l][i: index]
            if len(x) != length:
                x = token_chunks[l][index - length: index]
            s.append(x)

    write_tfrecord(s, output_file)
    blob = bucket.blob(f'gpt2-data/{output_file}')
    blob.upload_from_filename(output_file)
    os.system(f'rm {output_file}')


batch_size = 12
max_core = 6
for i in range(0, len(files), batch_size):
    b = files[i: i + batch_size]
    multiprocessing(b, loop, min(len(b), max_core))
    global_count += 1
