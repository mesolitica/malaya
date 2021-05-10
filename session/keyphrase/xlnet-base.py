import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import xlnet
import numpy as np
import model_utils
import random
import json
import collections
import re
import sentencepiece as spm
from sklearn.utils import shuffle
from prepro_utils import preprocess_text, encode_ids
from malaya.text.function import transformer_textcleaning as cleaning
from tensorflow.python.estimator.run_config import RunConfig

with open('/home/husein/alxlnet/topics.json') as fopen:
    topics = set(json.load(fopen).keys())

list_topics = list(topics)


sp_model = spm.SentencePieceProcessor()
sp_model.Load('sp10m.cased.v9.model')


def tokenize_fn(text):
    text = preprocess_text(text, lower = False)
    return encode_ids(sp_model, text)


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

VOCAB_SIZE = 32000
UNK_ID = special_symbols['<unk>']
CLS_ID = special_symbols['<cls>']
SEP_ID = special_symbols['<sep>']
MASK_ID = special_symbols['<mask>']
EOD_ID = special_symbols['<eod>']


def F(left_train):
    tokens_a = tokenize_fn(left_train)
    segment_id = [SEG_ID_A] * len(tokens_a)
    tokens_a.append(SEP_ID)
    tokens_a.append(CLS_ID)
    segment_id.append(SEG_ID_A)
    segment_id.append(SEG_ID_CLS)
    input_mask = [0] * len(tokens_a)
    return tokens_a, segment_id, input_mask


def XY(data):

    if len(set(data[1]) & topics) and random.random() > 0.2:
        t = random.choice(data[1])
        label = 1
    else:
        s = set(data[1]) | set()
        t = random.choice(list(topics - s))
        label = 0
    X = F(cleaning(data[0]))
    Y = F(t)

    return X, Y, label


def generate():
    with open('/home/husein/alxlnet/trainset-keyphrase.json') as fopen:
        data = json.load(fopen)
    while True:
        data = shuffle(data)
        for i in range(len(data)):
            X, Y, label = XY(data[i])
            yield {
                'X': X[0],
                'segment': X[1],
                'mask': X[2],
                'X_b': Y[0],
                'segment_b': Y[1],
                'mask_b': Y[2],
                'label': [label],
            }


def get_dataset(
    batch_size = 60, shuffle_size = 20, thread_count = 24, maxlen_feature = 1800
):
    def get():
        dataset = tf.data.Dataset.from_generator(
            generate,
            {
                'X': tf.int32,
                'segment': tf.int32,
                'mask': tf.int32,
                'X_b': tf.int32,
                'segment_b': tf.int32,
                'mask_b': tf.int32,
                'label': tf.int32,
            },
            output_shapes = {
                'X': tf.TensorShape([None]),
                'segment': tf.TensorShape([None]),
                'mask': tf.TensorShape([None]),
                'X_b': tf.TensorShape([None]),
                'segment_b': tf.TensorShape([None]),
                'mask_b': tf.TensorShape([None]),
                'label': tf.TensorShape([None]),
            },
        )
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes = {
                'X': tf.TensorShape([None]),
                'segment': tf.TensorShape([None]),
                'mask': tf.TensorShape([None]),
                'X_b': tf.TensorShape([None]),
                'segment_b': tf.TensorShape([None]),
                'mask_b': tf.TensorShape([None]),
                'label': tf.TensorShape([None]),
            },
            padding_values = {
                'X': tf.constant(0, dtype = tf.int32),
                'segment': tf.constant(1, dtype = tf.int32),
                'mask': tf.constant(4, dtype = tf.int32),
                'X_b': tf.constant(0, dtype = tf.int32),
                'segment_b': tf.constant(1, dtype = tf.int32),
                'mask_b': tf.constant(4, dtype = tf.int32),
                'label': tf.constant(0, dtype = tf.int32),
            },
        )
        return dataset

    return get


class Parameter:
    def __init__(
        self,
        decay_method,
        warmup_steps,
        weight_decay,
        adam_epsilon,
        num_core_per_host,
        lr_layer_decay_rate,
        use_tpu,
        learning_rate,
        train_steps,
        min_lr_ratio,
        clip,
        **kwargs
    ):
        self.decay_method = decay_method
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.num_core_per_host = num_core_per_host
        self.lr_layer_decay_rate = lr_layer_decay_rate
        self.use_tpu = use_tpu
        self.learning_rate = learning_rate
        self.train_steps = train_steps
        self.min_lr_ratio = min_lr_ratio
        self.clip = clip


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match('^(.*):\\d+$', name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if 'xlnet/' + name not in name_to_variable:
            continue
        assignment_map[name] = name_to_variable['xlnet/' + name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ':0'] = 1

    return (assignment_map, initialized_variable_names)


num_train_steps = 300000
warmup_proportion = 0.1
num_warmup_steps = int(num_train_steps * warmup_proportion)
initial_learning_rate = 2e-5


def model_fn(features, labels, mode, params):
    kwargs = dict(
        is_training = True,
        use_tpu = False,
        use_bfloat16 = False,
        dropout = 0.1,
        dropatt = 0.1,
        init = 'normal',
        init_range = 0.1,
        init_std = 0.05,
        clamp_len = -1,
    )

    xlnet_parameters = xlnet.RunConfig(**kwargs)
    xlnet_config = xlnet.XLNetConfig(
        json_path = 'xlnet-base-29-03-2020/config.json'
    )
    training_parameters = dict(
        decay_method = 'poly',
        train_steps = num_train_steps,
        learning_rate = initial_learning_rate,
        warmup_steps = num_warmup_steps,
        min_lr_ratio = 0.0,
        weight_decay = 0.00,
        adam_epsilon = 1e-8,
        num_core_per_host = 1,
        lr_layer_decay_rate = 1,
        use_tpu = False,
        use_bfloat16 = False,
        dropout = 0.1,
        dropatt = 0.1,
        init = 'normal',
        init_range = 0.1,
        init_std = 0.05,
        clip = 1.0,
        clamp_len = -1,
    )
    training_parameters = Parameter(**training_parameters)

    X = features['X']
    segment_ids = features['segment']
    input_masks = tf.cast(features['mask'], tf.float32)

    X_b = features['X_b']
    segment_ids_b = features['segment_b']
    input_masks_b = tf.cast(features['mask_b'], tf.float32)

    Y = features['label'][:, 0]

    with tf.compat.v1.variable_scope('xlnet', reuse = False):
        xlnet_model = xlnet.XLNetModel(
            xlnet_config = xlnet_config,
            run_config = xlnet_parameters,
            input_ids = tf.transpose(X, [1, 0]),
            seg_ids = tf.transpose(segment_ids, [1, 0]),
            input_mask = tf.transpose(input_masks, [1, 0]),
        )

        summary = xlnet_model.get_pooled_out('last', True)

    with tf.compat.v1.variable_scope('xlnet', reuse = True):
        xlnet_model = xlnet.XLNetModel(
            xlnet_config = xlnet_config,
            run_config = xlnet_parameters,
            input_ids = tf.transpose(X_b, [1, 0]),
            seg_ids = tf.transpose(segment_ids_b, [1, 0]),
            input_mask = tf.transpose(input_masks_b, [1, 0]),
        )
        summary_b = xlnet_model.get_pooled_out('last', True)

    vectors_concat = [summary, summary_b, tf.abs(summary - summary_b)]
    vectors_concat = tf.concat(vectors_concat, axis = 1)
    logits = tf.layers.dense(vectors_concat, 2)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = logits, labels = Y
        )
    )
    tf.identity(loss, 'train_loss')

    accuracy = tf.metrics.accuracy(
        labels = Y, predictions = tf.argmax(logits, axis = 1)
    )
    tf.identity(accuracy[1], name = 'train_accuracy')

    tvars = tf.trainable_variables()
    init_checkpoint = 'xlnet-base-29-03-2020/model.ckpt-300000'
    assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(
        tvars, init_checkpoint
    )
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op, learning_rate, _ = model_utils.get_train_op(
            training_parameters, loss
        )
        tf.summary.scalar('learning_rate', learning_rate)
        estimator_spec = tf.estimator.EstimatorSpec(
            mode = mode, loss = loss, train_op = train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL,
            loss = loss,
            eval_metric_ops = {'accuracy': accuracy},
        )

    return estimator_spec


def run_training(
    train_fn,
    model_fn,
    model_dir: str,
    gpu_mem_fraction: float = 0.96,
    log_step: int = 100,
    summary_step: int = 100,
    save_checkpoint_step: int = 1000,
    max_steps: int = 10000,
    eval_step: int = 10,
    eval_throttle: int = 120,
    train_batch_size: int = 128,
    train_hooks = None,
    eval_fn = None,
):
    tf.logging.set_verbosity(tf.logging.INFO)
    dist_strategy = None

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction = gpu_mem_fraction
    )
    config = tf.ConfigProto(
        allow_soft_placement = True, gpu_options = gpu_options
    )
    run_config = RunConfig(
        train_distribute = dist_strategy,
        eval_distribute = dist_strategy,
        log_step_count_steps = log_step,
        model_dir = model_dir,
        save_checkpoints_steps = save_checkpoint_step,
        save_summary_steps = summary_step,
        session_config = config,
    )

    estimator = tf.estimator.Estimator(
        model_fn = model_fn, params = {}, config = run_config
    )

    if eval_fn:
        train_spec = tf.estimator.TrainSpec(
            input_fn = train_fn, max_steps = max_steps, hooks = train_hooks
        )

        eval_spec = tf.estimator.EvalSpec(
            input_fn = eval_fn, steps = eval_step, throttle_secs = eval_throttle
        )
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    else:
        estimator.train(
            input_fn = train_fn, max_steps = max_steps, hooks = train_hooks
        )


train_hooks = [
    tf.train.LoggingTensorHook(
        ['train_accuracy', 'train_loss'], every_n_iter = 1
    )
]

train_dataset = get_dataset()

save_directory = 'xlnet-base-keyphrase'

run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = save_directory,
    log_step = 1,
    save_checkpoint_step = 10000,
    max_steps = num_train_steps,
    train_hooks = train_hooks,
)
