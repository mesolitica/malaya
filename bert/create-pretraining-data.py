import tensorflow as tf
from create_pretraining_data import *
import json
import re
import malaya
import itertools
from unidecode import unidecode

_tokenizer = malaya.preprocessing._SocialTokenizer().tokenize
rules_normalizer = malaya.texts._tatabahasa.rules_normalizer
rejected = ['wkwk', 'http', 'https', 'lolol', 'hahaha']

def is_number_regex(s):
    if re.match("^\d+?\.\d+?$", s) is None:
        return s.isdigit()
    return True

def detect_money(word):
    if word[:2] == 'rm' and is_number_regex(word[2:]):
        return True
    else:
        return False

def preprocessing(string):
    string = ''.join(''.join(s)[:2] for _, s in itertools.groupby(unidecode(string)))
    tokenized = _tokenizer(string)
    tokenized = [malaya.stem.naive(w) for w in tokenized]
    tokenized = [w.lower() for w in tokenized if len(w) > 1]
    tokenized = [w for w in tokenized if all([r not in w for r in rejected])]
    tokenized = [rules_normalizer.get(w, w) for w in tokenized]
    tokenized = ['<NUM>' if is_number_regex(w) else w for w in tokenized]
    tokenized = ['<MONEY>' if detect_money(w) else w for w in tokenized]
    return tokenized

with open('dictionary.json') as fopen:
    d = json.load(fopen)
dictionary = d['dictionary']
rev_dictionary = d['reverse_dictionary']

class Tokenizer:
    def __init__(self, vocab, rev_dictionary):
        self.vocab = vocab
        self.inv_vocab = rev_dictionary
    
    def tokenize(self, string):
        return preprocessing(string)
    
    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(t, 1) for t in tokens]
    
    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab[i] for i in ids]
    
tokenizer = Tokenizer(dictionary, rev_dictionary)
files = 'test.txt'

input_files = []
for input_pattern in files.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))
    
tf.logging.info("*** Reading from input files ***")
for input_file in input_files:
    tf.logging.info("  %s", input_file)
    
import random

max_seq_length = 128
dupe_factor = 5
max_predictions_per_seq=20
masked_lm_prob=0.15
short_seq_prob=0.1
rng = random.Random(12345)
instances = create_training_instances(
      input_files, tokenizer, max_seq_length, dupe_factor,
      short_seq_prob, masked_lm_prob, max_predictions_per_seq,
      rng)

output_files = 'tests_output.tfrecord'.split(",")
tf.logging.info("*** Writing to output files ***")
for output_file in output_files:
    tf.logging.info("  %s", output_file)
    
write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                max_predictions_per_seq, output_files)
