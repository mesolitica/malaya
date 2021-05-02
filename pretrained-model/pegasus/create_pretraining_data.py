import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
from numpy.random import default_rng
import random
import collections
import re
import tensorflow as tf
from tqdm import tqdm

max_seq_length_encoder = 512
max_seq_length_decoder = 256
masked_lm_prob = 0
max_predictions_per_seq = 0
do_whole_word_mask = True
EOS_ID = 1

MaskedLmInstance = collections.namedtuple(
    'MaskedLmInstance', ['index', 'label']
)


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, tokens_y, masked_lm_positions, masked_lm_labels):
        self.tokens = tokens
        self.tokens_y = tokens_y
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels


def sliding(strings, n = 5):
    results = []
    for i in range(len(strings) - n):
        results.append(strings[i : i + n])
    return results


def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i : i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    assert len(sentences) > 0
    assert n > 0

    words = sum(sentences, [])
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {'f': f1_score, 'p': precision, 'r': recall}


def _rouge_clean(s):
    return re.sub(r'[^a-zA-Z0-9 ]', '', s)


def get_rouges(strings, n = 1):
    rouges = []
    for i in range(len(strings)):
        abstract = strings[i]
        doc_sent_list = [strings[k] for k in range(len(strings)) if k != i]
        sents = _rouge_clean(' '.join(doc_sent_list)).split()
        abstract = _rouge_clean(abstract).split()
        evaluated_1grams = _get_word_ngrams(n, [sents])
        reference_1grams = _get_word_ngrams(n, [abstract])
        rouges.append(cal_rouge(evaluated_1grams, reference_1grams)['f'])
    return rouges


# Principal Select top-m scored sentences according to importance.
# As a proxy for importance we compute ROUGE1-F1 (Lin, 2004) between the sentence and the rest of the document
def get_rouge(strings, top_k = 1, minlen = 4):
    rouges = get_rouges(strings)
    s = np.argsort(rouges)[::-1]
    s = [i for i in s if len(strings[i].split()) >= minlen]
    return s[:top_k]


# Random Uniformly select m sentences at random.
def get_random(strings, rng, top_k = 1):
    return rng.choice(len(strings), size = top_k, replace = False)


# Lead Select the first m sentences.
def get_lead(strings, top_k = 1):
    return [i for i in range(top_k)]


def combine(l):
    r = []
    for s in l:
        if s[-1] != '.':
            if s in ['[MASK]', '[MASK2]']:
                e = ' .'
            else:
                e = '.'
            s = s + e
        r.append(s)
    return ' '.join(r)


def is_number_regex(s):
    if re.match('^\d+?\.\d+?$', s) is None:
        return s.isdigit()
    return True


def reject(token):
    t = token.replace('##', '')
    if is_number_regex(t):
        return True
    if t.startswith('RM'):
        return True
    if token in '!{<>}:;.,"\'':
        return True
    return False


def create_masked_lm_predictions(tokens, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == '[CLS]' or token == '[SEP]' or token == '[MASK2]':
            continue
        if reject(token):
            continue
        if (
            do_whole_word_mask
            and len(cand_indexes) >= 1
            and token.startswith('##')
        ):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(
        max_predictions_per_seq,
        max(1, int(round(len(tokens) * masked_lm_prob))),
    )

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = '[MASK]'
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[
                        np.random.randint(0, len(vocab_words) - 1)
                    ]

            output_tokens[index] = masked_token

            masked_lms.append(
                MaskedLmInstance(index = index, label = tokens[index])
            )
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key = lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def get_feature(x, y, tokenizer, vocab_words, rng, dedup_factor = 5, **kwargs):
    tokens = tokenizer.tokenize(x)
    if len(tokens) > (max_seq_length_encoder - 2):
        tokens = tokens[: max_seq_length_encoder - 2]

    if '[MASK2]' not in tokens:
        return []

    tokens = tokens

    tokens_y = []
    for y_ in y:
        tokens_y.extend(tokenizer.tokenize(y_))
    if len(tokens_y) > (max_seq_length_decoder - 1):
        tokens_y = tokens_y[: max_seq_length_decoder - 1]

    tokens_y = tokenizer.convert_tokens_to_ids(tokens_y)
    tokens_y = tokens_y + [EOS_ID]
    results = []
    for i in range(dedup_factor):
        output_tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
            tokens, vocab_words, rng, **kwargs
        )
        output_tokens = tokenizer.convert_tokens_to_ids(output_tokens)
        masked_lm_labels = tokenizer.convert_tokens_to_ids(masked_lm_labels)
        t = TrainingInstance(
            output_tokens, tokens_y, masked_lm_positions, masked_lm_labels
        )
        results.append(t)
    return results


def group_doc(data):
    results, result = [], []
    for i in data:
        if not len(i) and len(result):
            results.append(result)
            result = []
        else:
            result.append(i)

    if len(result):
        results.append(result)
    return results


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list = tf.train.Int64List(value = list(values))
    )
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(
        float_list = tf.train.FloatList(value = list(values))
    )
    return feature


def write_instance_to_example_file(instances, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (inst_index, instance) in enumerate(instances):
        input_ids = list(instance.tokens)
        target_ids = list(instance.tokens_y)
        while len(input_ids) < max_seq_length_encoder:
            input_ids.append(0)
        while len(target_ids) < max_seq_length_decoder:
            target_ids.append(0)
        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = list(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)
        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        features = collections.OrderedDict()
        features['input_ids'] = create_int_feature(input_ids)
        features['target_ids'] = create_int_feature(target_ids)
        features['masked_lm_positions'] = create_int_feature(
            masked_lm_positions
        )
        features['masked_lm_ids'] = create_int_feature(masked_lm_ids)
        features['masked_lm_weights'] = create_float_feature(masked_lm_weights)
        tf_example = tf.train.Example(
            features = tf.train.Features(feature = features)
        )
        writer.write(tf_example.SerializeToString())

    tf.logging.info('Wrote %d total instances', inst_index)


def process_documents(
    file,
    output_file,
    tokenizer,
    min_slide = 7,
    max_slide = 13,
    min_sentence = 1,
    max_sentence = 3,
    dedup_mask = 1,
    use_rouge = True,
):
    with open(file) as fopen:
        data = fopen.read().split('\n')
    rng = default_rng()
    vocab_words = list(tokenizer.vocab.keys())
    grouped = group_doc(data)
    results = []
    for r in tqdm(grouped):
        for s in range(min_slide, max_slide, 1):
            slided = sliding(r, s)
            for i in range(len(slided)):
                try:
                    strings = slided[i]
                    if use_rouge:
                        rouge_ = get_rouge(strings,random.randint(min_sentence, max_sentence))
                    else:
                        rouge_ = get_random(strings, rng)

                    y = []
                    for index in rouge_:
                        y.append(strings[index])
                        strings[index] = '[MASK2]'

                    x = combine(strings)
                    result = get_feature(
                        x,
                        y,
                        tokenizer,
                        vocab_words,
                        rng,
                        dedup_factor = dedup_mask,
                    )
                    results.extend(result)
                except Exception as e:
                    # print(e)
                    pass

    write_instance_to_example_file(results, output_file)
