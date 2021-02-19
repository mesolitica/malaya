import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import tensorflow as tf
import malaya.train as train
from malaya.train.model.bigbird import modeling, utils
from itertools import cycle
import random
import linecache

bert_config = {
    'attention_probs_dropout_prob': 0.1,
    'hidden_act': 'gelu',
    'hidden_dropout_prob': 0.1,
    'hidden_size': 512,
    'initializer_range': 0.02,
    'intermediate_size': 2048,
    'max_position_embeddings': 2048,
    'max_encoder_length': 1024,
    'max_decoder_length': 1024,
    'num_attention_heads': 8,
    'num_hidden_layers': 6,
    'type_vocab_size': 2,
    'scope': 'bert',
    'use_bias': True,
    'rescale_embedding': False,
    'vocab_model_file': None,
    'attention_type': 'block_sparse',
    'block_size': 16,
    'num_rand_blocks': 3,
    'vocab_size': 32000,
    'couple_encoder_decoder': False,
    'beam_size': 1,
    'alpha': 0.0,
    'label_smoothing': 0.1,
    'norm_type': 'postnorm',
}

learning_rate_constant = 2.0
learning_rate_warmup_steps = 100000.0
total_steps = 500000


def learning_rate_schedule(step_num):
    step_num = tf.cast(step_num, tf.float32)
    lr = learning_rate_constant
    lr *= tf.minimum(1.0, step_num / learning_rate_warmup_steps)
    lr *= tf.rsqrt(tf.maximum(step_num, learning_rate_warmup_steps))
    lr *= bert_config['hidden_size'] ** -0.5
    return lr


import sentencepiece as spm

vocab = 'sp10m.cased.translation.model'
sp = spm.SentencePieceProcessor()
sp.Load(vocab)


class Encoder:
    def __init__(self, sp):
        self.sp = sp

    def encode(self, s):
        return self.sp.EncodeAsIds(s) + [1]

    def decode(self, ids, strip_extraneous = False):
        return self.sp.DecodeIds(list(ids))


encoder = Encoder(sp)

# !wget https://f000.backblazeb2.com/file/malay-dataset/train-en-ms.tar.gz
# !wget https://f000.backblazeb2.com/file/malay-dataset/train-ms-en-long-text.tar.gz
# !tar -zxf train-ms-en-long-text.tar.gz
# !mkdir english
# !tar -zxf train-en-ms.tar.gz -C english

files = [
    ('train-long-text/right.txt', 'train-long-text/left.txt'),
    ('english/train-en/left.txt', 'english/train-en/right.txt'),
]

lengths = {}
for f in files:
    lengths[f[0]] = sum(1 for line in open(f[0]))

file_cycle = cycle(files)


def generate_random():
    while True:
        left, right = next(file_cycle)

        while True:
            index = random.randint(0, lengths[left] - 1)
            line_left = linecache.getline(left, index)
            line_right = linecache.getline(right, index)
            line_left = line_left.strip()
            line_right = line_right.strip()
            if len(line_left) and len(line_right):
                break

        x = encoder.encode(line_left)
        y = encoder.encode(line_right)
        if (
            len(x) > bert_config['max_encoder_length']
            or len(y) > bert_config['max_decoder_length']
        ):
            continue
        yield {'x': x, 'y': y}


def generate():
    while True:
        left, right = next(file_cycle)

        fopen_left = open(left)
        fopen_right = open(right)

        while True:
            line_left = fopen_left.readline()
            line_right = fopen_right.readline()

            if not line_left or not line_right:
                break

            line_left = line_left.strip()
            line_right = line_right.strip()

            x = encoder.encode(line_left)
            y = encoder.encode(line_right)

            if (
                len(x) > bert_config['max_encoder_length']
                or len(y) > bert_config['max_decoder_length']
            ):
                continue

            yield {'x': x, 'y': y}

        fopen_left.close()
        fopen_right.close()


# g = generate()
# next(g)


def get_dataset(batch_size = 8):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate_random,
            {'x': tf.int32, 'y': tf.int32},
            output_shapes = {
                'x': tf.TensorShape([None]),
                'y': tf.TensorShape([None]),
            },
        )
        dataset = dataset.shuffle(batch_size)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'x': tf.TensorShape([bert_config['max_encoder_length']]),
                'y': tf.TensorShape([bert_config['max_decoder_length']]),
            },
            padding_values = {
                'x': tf.constant(0, dtype = tf.int32),
                'y': tf.constant(0, dtype = tf.int32),
            },
        )
        return dataset

    return get


# dataset = get_dataset()()
# dataset = dataset.make_one_shot_iterator().get_next()
# sess = tf.Session()
# sess.run(dataset)


def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
    with tf.name_scope('loss'):

        if labels is not None:
            with tf.name_scope('smoothing_cross_entropy'):
                confidence = 1.0 - smoothing
                vocab_float = tf.cast(vocab_size - 1, tf.float32)
                low_confidence = (1.0 - confidence) / vocab_float
                soft_targets = tf.one_hot(
                    labels,
                    depth = vocab_size,
                    on_value = confidence,
                    off_value = low_confidence,
                )
                xentropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits = logits, labels = soft_targets
                )

                normalizing_constant = -(
                    confidence * tf.math.log(confidence)
                    + vocab_float
                    * low_confidence
                    * tf.math.log(low_confidence + 1e-20)
                )
                xentropy -= normalizing_constant

            weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
            loss = tf.reduce_sum(xentropy * weights) / tf.reduce_sum(weights)

        else:
            loss = tf.constant(0.0)

        return loss


def model_fn(features, labels, mode, params):
    inputs = features['x']
    targets = features['y']
    model = modeling.TransformerModel(bert_config)
    (llh, logits, pred_ids), _ = model(
        inputs, target_ids = targets, training = True
    )
    total_loss = padded_cross_entropy_loss(
        logits,
        targets,
        bert_config['label_smoothing'],
        bert_config['vocab_size'],
    )
    tf.identity(total_loss, 'total_loss')
    tf.summary.scalar('total_loss', total_loss)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        lr = learning_rate_schedule(global_step)

        tf.summary.scalar('learning_rate', lr)
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        train_op = optimizer.minimize(total_loss, global_step = global_step)
        estimator_spec = tf.estimator.EstimatorSpec(
            mode = mode, loss = total_loss, train_op = train_op
        )
    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL, loss = total_loss
        )

    return estimator_spec


train_hooks = [tf.train.LoggingTensorHook(['total_loss'], every_n_iter = 100)]
train_dataset = get_dataset()

save_directory = 'bigbird-small-en-ms'

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = save_directory,
    num_gpus = 2,
    log_step = 100,
    save_checkpoint_step = 5000,
    max_steps = total_steps,
    train_hooks = train_hooks,
    eval_step = 0,
)
