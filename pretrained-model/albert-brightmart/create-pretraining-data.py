import tensorflow as tf
from create_pretraining_data import *
import json
import re
import sentencepiece as spm
from prepro_utils import preprocess_text, encode_ids, encode_pieces

sp_model = spm.SentencePieceProcessor()
sp_model.Load('sp10m.cased.v8.model')

with open('sp10m.cased.v8.vocab') as fopen:
    v = fopen.read().split('\n')[:-1]
v = [i.split('\t') for i in v]
v = {i[0]: i[1] for i in v}


class Tokenizer:
    def __init__(self, v):
        self.vocab = v
        pass

    def tokenize(self, string):
        return encode_pieces(
            sp_model, string, return_unicode = False, sample = False
        )

    def convert_tokens_to_ids(self, tokens):
        return [sp_model.PieceToId(piece) for piece in tokens]

    def convert_ids_to_tokens(self, ids):
        return [sp_model.IdToPiece(i) for i in ids]


tokenizer = Tokenizer(v)
files = '../dumping-all.txt'

input_files = []
for input_pattern in files.split(','):
    input_files.extend(tf.gfile.Glob(input_pattern))

tf.logging.info('*** Reading from input files ***')
for input_file in input_files:
    tf.logging.info('  %s', input_file)

import random

max_seq_length = 512
dupe_factor = 5
max_predictions_per_seq = 51
masked_lm_prob = 0.15
short_seq_prob = 0.1
rng = random.Random(12345)
instances = create_training_instances(
    input_files,
    tokenizer,
    max_seq_length,
    dupe_factor,
    short_seq_prob,
    masked_lm_prob,
    max_predictions_per_seq,
    rng,
)

output_files = 'albert-brightmart.tfrecord'.split(',')
tf.logging.info('*** Writing to output files ***')
for output_file in output_files:
    tf.logging.info('  %s', output_file)

write_instance_to_example_files(
    instances, tokenizer, max_seq_length, max_predictions_per_seq, output_files
)
