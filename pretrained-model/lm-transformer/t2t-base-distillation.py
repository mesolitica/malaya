import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ[
    'GOOGLE_APPLICATION_CREDENTIALS'
] = '/home/husein/t5/prepare/mesolitica-tpu.json'

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry
from tensor2tensor import problems
from tensor2tensor import models
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import metrics
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import adafactor
from tensorflow.python.distribute.cross_device_ops import (
    AllReduceCrossDeviceOps,
)
from tensorflow.python.estimator.run_config import RunConfig
import tensorflow as tf
import logging
from tqdm import tqdm
from glob import glob
import sentencepiece as spm
import collections
import re

logger = logging.getLogger()
tf.logging.set_verbosity(tf.logging.DEBUG)

vocab = 'sp10m.cased.t5.model'
sp = spm.SentencePieceProcessor()
sp.Load(vocab)


class Encoder:
    def __init__(self, sp):
        self.sp = sp
        self.vocab_size = sp.GetPieceSize() + 100

    def encode(self, s):
        return self.sp.EncodeAsIds(s)

    def decode(self, ids, strip_extraneous = False):
        return self.sp.DecodeIds(list(ids))


encoder = Encoder(sp)


@registry.register_problem
class Seq2Seq(text_problems.Text2TextProblem):
    @property
    def approx_vocab_size(self):
        return 32100

    @property
    def is_generate_per_split(self):
        return False

    def feature_encoders(self, data_dir):
        encoder = Encoder(sp)
        return {'inputs': encoder, 'targets': encoder}


PROBLEM = 'seq2_seq'
t2t_problem = problems.problem(PROBLEM)


class Model:
    def __init__(
        self, X, Y, HPARAMS = 'transformer_base', DATA_DIR = 't2t/data'
    ):

        self.X = X
        self.Y = Y

        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype = tf.int32)
        maxlen_decode = tf.reduce_max(self.X_seq_len)

        x = tf.expand_dims(tf.expand_dims(self.X, -1), -1)
        y = tf.expand_dims(tf.expand_dims(self.Y, -1), -1)

        features = {
            'inputs': x,
            'targets': y,
            'target_space_id': tf.constant(1, dtype = tf.int32),
        }
        self.features = features

        Modes = tf.estimator.ModeKeys
        hparams = trainer_lib.create_hparams(
            HPARAMS, data_dir = DATA_DIR, problem_name = PROBLEM
        )
        hparams.filter_size = 3072
        hparams.hidden_size = 768
        hparams.num_heads = 12
        hparams.num_hidden_layers = 8
        hparams.vocab_divisor = 128
        hparams.label_smoothing = 0.0
        hparams.shared_embedding_and_softmax_weights = False
        hparams.dropout = 0.1
        hparams.max_length = 1024
        hparams.multiproblem_mixing_schedule = 'pretrain'

        hparams.optimizer = 'Adafactor'
        hparams.learning_rate_warmup_steps = 10000
        hparams.learning_rate_schedule = 'rsqrt_decay'

        translate_model = registry.model('transformer')(hparams, Modes.TRAIN)
        self.translate_model = translate_model
        logits, _ = translate_model(features)
        self.logits = logits


class StudentModel:
    def __init__(
        self, X, Y, HPARAMS = 'transformer_base', DATA_DIR = 't2t/data'
    ):

        with tf.compat.v1.variable_scope('student') as vs:

            self.X = X
            self.Y = Y

            self.X_seq_len = tf.count_nonzero(self.X, 1, dtype = tf.int32)
            maxlen_decode = tf.reduce_max(self.X_seq_len)

            x = tf.expand_dims(tf.expand_dims(self.X, -1), -1)
            y = tf.expand_dims(tf.expand_dims(self.Y, -1), -1)

            features = {
                'inputs': x,
                'targets': y,
                'target_space_id': tf.constant(1, dtype = tf.int32),
            }
            self.features = features

            Modes = tf.estimator.ModeKeys
            hparams = trainer_lib.create_hparams(
                HPARAMS, data_dir = DATA_DIR, problem_name = PROBLEM
            )
            hparams.filter_size = 1080
            hparams.hidden_size = 312
            hparams.num_heads = 12
            hparams.num_hidden_layers = 4
            hparams.vocab_divisor = 128
            hparams.label_smoothing = 0.0
            hparams.shared_embedding_and_softmax_weights = False
            hparams.dropout = 0.0
            hparams.max_length = 1024
            hparams.multiproblem_mixing_schedule = 'pretrain'

            hparams.optimizer = 'Adafactor'
            hparams.learning_rate_warmup_steps = 10000
            hparams.learning_rate_schedule = 'rsqrt_decay'

            translate_model = registry.model('transformer')(
                hparams, Modes.TRAIN
            )
            self.translate_model = translate_model
            logits, _ = translate_model(features)
            self.logits = logits


def input_fn_builder(
    input_files,
    max_seq_length_encoder,
    max_seq_length_decoder,
    is_training,
    num_cpu_threads = 4,
):

    data_fields = {
        'inputs': tf.VarLenFeature(tf.int64),
        'targets': tf.VarLenFeature(tf.int64),
    }
    data_len = {
        'inputs': max_seq_length_encoder,
        'targets': max_seq_length_decoder,
    }

    def parse(serialized_example):

        features = tf.parse_single_example(
            serialized_example, features = data_fields
        )
        for k in features.keys():
            features[k] = features[k].values
            features[k] = tf.pad(
                features[k], [[0, data_len[k] - tf.shape(features[k])[0]]]
            )
            features[k].set_shape((data_len[k]))

        return features

    def input_fn(batch_size = 6):

        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size = len(input_files))
            cycle_length = min(num_cpu_threads, len(input_files))
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy = is_training,
                    cycle_length = cycle_length,
                )
            )
            d = d.shuffle(buffer_size = 100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            d = d.repeat()
        d = d.map(parse, num_parallel_calls = 32)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, data_fields),
                batch_size = batch_size,
                num_parallel_batches = num_cpu_threads,
                drop_remainder = True,
            )
        )
        return d

    return input_fn


def _decode_record(example, name_to_features):
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


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
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ':0'] = 1

    tf.logging.info('**** Trainable Variables ****')
    for var in tvars:
        init_string = ''
        if var.name in initialized_variable_names:
            init_string = ', *INIT_FROM_CKPT*'
        tf.logging.info(
            '  name = %s, shape = %s%s', var.name, var.shape, init_string
        )

    return (assignment_map, initialized_variable_names)


def padded_cross_entropy_loss(
    logits, labels, smoothing = 0.0, vocab_size = 32128
):
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
            return tf.reduce_sum(xentropy * weights), weights

        else:
            loss = tf.constant(0.0)

        return loss


init_checkpoint = 'base/model.ckpt-500000'
task_balance = 0.5
distill_temperature = 1.0
total_steps = 500000
num_warmup_steps = 10000
learning_rate_decay_steps = 0.1


def model_fn(features, labels, mode, params):
    X = features['inputs']
    Y = features['targets']

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    model = Model(X, Y)
    student = StudentModel(X, Y)

    student_logits = student.logits[:, :, 0, 0]
    student_task_xent, weights = padded_cross_entropy_loss(
        student_logits, student.Y
    )

    teacher_targets = tf.nn.softmax(
        model.logits[:, :, 0, 0] / distill_temperature
    )
    student_distill_xent = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels = tf.stop_gradient(teacher_targets),
        logits = student_logits / distill_temperature,
    )
    student_distill_xent = tf.reduce_sum(student_distill_xent * weights)
    student_distill_xent *= distill_temperature ** 2

    phase_loss = task_balance * student_task_xent
    phase_loss += (1 - task_balance) * student_distill_xent

    loss = phase_loss / tf.reduce_sum(weights)
    task_loss = student_task_xent / tf.reduce_sum(weights)
    distill_loss = student_distill_xent / tf.reduce_sum(weights)

    tf.identity(loss, 'total_loss')
    tf.identity(task_loss, 'task_loss')
    tf.identity(distill_loss, 'distill_loss')

    tf.summary.scalar('total_loss', loss)
    tf.summary.scalar('task_loss', task_loss)
    tf.summary.scalar('distill_loss', distill_loss)

    tvars = [v for v in tf.trainable_variables() if 'student/' not in v.name]

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
        (
            assignment_map,
            initialized_variable_names,
        ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        lr = 1e-2 * tf.rsqrt(
            tf.maximum(tf.to_float(global_step), num_warmup_steps)
        )
        optimizer = adafactor.AdafactorOptimizer(
            learning_rate = lr, beta1 = 0.0
        )
        train_op = optimizer.minimize(loss, global_step = global_step)
        estimator_spec = tf.estimator.EstimatorSpec(
            mode = mode, loss = loss, train_op = train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        estimator_spec = tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL, loss = loss
        )

    return estimator_spec


def run_training(
    train_fn,
    model_fn,
    model_dir: str,
    num_gpus: int = 1,
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

    if num_gpus > 1 and not use_tpu:
        dist_strategy = tf.contrib.distribute.MirroredStrategy(
            num_gpus = num_gpus,
            auto_shard_dataset = True,
            cross_device_ops = AllReduceCrossDeviceOps(
                'nccl', num_packs = num_gpus
            ),
        )
    else:
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
        ['total_loss', 'task_loss', 'distill_loss'], every_n_iter = 1
    )
]

input_files = tf.gfile.Glob('gs://mesolitica-tpu-general/t2t/data/seq2*')
train_dataset = input_fn_builder(
    input_files = input_files,
    max_seq_length_encoder = 1024,
    max_seq_length_decoder = 1024,
    is_training = True,
)

save_directory = 't2t-base-distillation'

run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = save_directory,
    num_gpus = 1,
    log_step = 1,
    save_checkpoint_step = 3000,
    max_steps = total_steps,
    train_hooks = train_hooks,
    eval_step = 0,
)
