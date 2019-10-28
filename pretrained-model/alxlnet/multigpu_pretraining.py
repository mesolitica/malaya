"""Pretraining on GPUs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import math
import json
import time
import numpy as np

from absl import flags
import absl.logging as _logging

import tensorflow as tf

import data_utils
import model_utils
from tensorflow.python.distribute.cross_device_ops import (
    AllReduceCrossDeviceOps,
)
import custom_function_builder
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.estimator import Estimator

flags.DEFINE_integer('num_hosts', default = 1, help = 'Number of hosts')
flags.DEFINE_integer(
    'num_core_per_host', default = 1, help = 'Number of cores per host'
)
flags.DEFINE_bool(
    'use_tpu', default = False, help = 'Whether to use TPUs for training.'
)

flags.DEFINE_integer(
    'num_gpu_cores',
    2,
    'Only used if `use_gpu` is True. Total number of GPU cores to use.',
)

# Experiment (data/checkpoint/directory) config
flags.DEFINE_integer(
    'num_passes', default = 1, help = 'Number of passed used for training.'
)
flags.DEFINE_string(
    'record_info_dir',
    default = None,
    help = 'Path to local directory containing `record_info-lm.json`.',
)
flags.DEFINE_string('model_dir', default = None, help = 'Estimator model_dir.')
flags.DEFINE_string(
    'init_checkpoint',
    default = None,
    help = 'checkpoint path for initializing the model.',
)

# Optimization config
flags.DEFINE_float(
    'learning_rate', default = 1e-4, help = 'Maximum learning rate.'
)
flags.DEFINE_float('clip', default = 1.0, help = 'Gradient clipping value.')
# for cosine decay
flags.DEFINE_float(
    'min_lr_ratio', default = 0.001, help = 'Minimum ratio learning rate.'
)
flags.DEFINE_integer(
    'warmup_steps', default = 0, help = 'Number of steps for linear lr warmup.'
)
flags.DEFINE_float('adam_epsilon', default = 1e-8, help = 'Adam epsilon')
flags.DEFINE_string('decay_method', default = 'poly', help = 'poly or cos')
flags.DEFINE_float('weight_decay', default = 0.0, help = 'weight decay')

# Training config
flags.DEFINE_integer(
    'train_batch_size', default = 16, help = 'Size of train batch.'
)
flags.DEFINE_integer(
    'train_steps', default = 100000, help = 'Total number of training steps.'
)
flags.DEFINE_integer(
    'iterations', default = 1000, help = 'Number of iterations per repeat loop.'
)
flags.DEFINE_integer(
    'save_steps',
    default = None,
    help = 'number of steps for model checkpointing.',
)

# Data config
flags.DEFINE_integer(
    'seq_len', default = 0, help = 'Sequence length for pretraining.'
)
flags.DEFINE_integer(
    'reuse_len',
    default = 0,
    help = 'How many tokens to be reused in the next batch. '
    'Could be half of seq_len',
)
flags.DEFINE_bool(
    'bi_data',
    default = True,
    help = 'Use bidirectional data streams, i.e., forward & backward.',
)
flags.DEFINE_integer(
    'mask_alpha', default = 6, help = 'How many tokens to form a group.'
)
flags.DEFINE_integer(
    'mask_beta',
    default = 1,
    help = 'How many tokens to mask within each group.',
)
flags.DEFINE_integer(
    'num_predict',
    default = None,
    help = 'Number of tokens to predict in partial prediction.',
)
flags.DEFINE_integer('perm_size', default = None, help = 'perm size.')
flags.DEFINE_bool('uncased', False, help = 'Use uncased inputs or not.')
flags.DEFINE_integer('n_token', 32000, help = 'Vocab size')

# Model config
flags.DEFINE_integer('mem_len', default = 0, help = 'Number of steps to cache')
flags.DEFINE_bool(
    'same_length', default = False, help = 'Same length attention'
)
flags.DEFINE_integer('clamp_len', default = -1, help = 'Clamp length')

flags.DEFINE_integer('n_layer', default = 6, help = 'Number of layers.')
flags.DEFINE_integer('d_model', default = 32, help = 'Dimension of the model.')
flags.DEFINE_integer(
    'd_embed', default = 32, help = 'Dimension of the embeddings.'
)
flags.DEFINE_integer('n_head', default = 4, help = 'Number of attention heads.')
flags.DEFINE_integer(
    'd_head', default = 8, help = 'Dimension of each attention head.'
)
flags.DEFINE_integer(
    'd_inner',
    default = 32,
    help = 'Dimension of inner hidden size in positionwise feed-forward.',
)
flags.DEFINE_float('dropout', default = 0.0, help = 'Dropout rate.')
flags.DEFINE_float('dropatt', default = 0.0, help = 'Attention dropout rate.')
flags.DEFINE_bool(
    'untie_r', default = False, help = 'Untie r_w_bias and r_r_bias'
)
flags.DEFINE_string(
    'summary_type',
    default = 'last',
    help = 'Method used to summarize a sequence into a compact vector.',
)
flags.DEFINE_string(
    'ff_activation',
    default = 'relu',
    help = 'Activation type used in position-wise feed-forward.',
)
flags.DEFINE_bool('use_bfloat16', False, help = 'Whether to use bfloat16.')

# Parameter initialization
flags.DEFINE_enum(
    'init',
    default = 'normal',
    enum_values = ['normal', 'uniform'],
    help = 'Initialization method.',
)
flags.DEFINE_float(
    'init_std', default = 0.02, help = 'Initialization std when init is normal.'
)
flags.DEFINE_float(
    'init_range',
    default = 0.1,
    help = 'Initialization std when init is uniform.',
)


FLAGS = flags.FLAGS


def per_device_batch_size(batch_size, num_gpus):
    """For multi-gpu, batch-size must be a multiple of the number of GPUs.
  Note that this should eventually be handled by DistributionStrategies
  directly. Multi-GPU support is currently experimental, however,
  so doing the work here until that feature is in place.
  Args:
    batch_size: Global batch size to be divided among devices. This should be
      equal to num_gpus times the single-GPU batch_size for multi-gpu training.
    num_gpus: How many GPUs are used with DistributionStrategies.
  Returns:
    Batch size per device.
  Raises:
    ValueError: if batch_size is not divisible by number of devices
  """
    if num_gpus <= 1:
        return batch_size

    remainder = batch_size % num_gpus
    if remainder:
        err = (
            'When running with multiple GPUs, batch size '
            'must be a multiple of the number of available GPUs. Found {} '
            'GPUs with a batch size of {}; try --batch_size={} instead.'
        ).format(num_gpus, batch_size, batch_size - remainder)
        raise ValueError(err)
    return int(batch_size / num_gpus)


def get_model_fn():
    """doc."""

    def model_fn(features, labels, mode, params):
        """doc."""
        #### Training or Evaluation
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        assert is_training

        #### Retrieve `mems` from `params["cache"]`
        mems = {}
        idx = 0
        if FLAGS.mem_len > 0:
            mems['mems'] = params['cache']

        #### Get loss from inputs
        total_loss, total_accuracy, new_mems, monitor_dict = custom_function_builder.get_loss(
            FLAGS, features, labels, mems, is_training
        )

        #### Turn `new_mems` into `new_cache`
        new_cache = []
        if FLAGS.mem_len > 0:
            new_cache += new_mems['mems']

        #### Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        #### Configuring the optimizer
        train_op, learning_rate, gnorm = model_utils.get_train_op(
            FLAGS, total_loss
        )
        monitor_dict['lr'] = learning_rate
        monitor_dict['gnorm'] = gnorm

        #### Customized initial checkpoint
        scaffold_fn = model_utils.init_from_checkpoint(
            FLAGS, global_vars = True
        )

        output_spec = tf.estimator.EstimatorSpec(
            mode = mode,
            loss = total_loss,
            train_op = train_op,
            scaffold = scaffold_fn,
        )

        return output_spec

    return model_fn


def get_input_fn(split, batch_size):
    """doc."""
    assert split == 'train'

    input_fn, record_info_dict = data_utils.get_input_fn(
        tfrecord_dir = FLAGS.record_info_dir,
        split = split,
        bsz_per_host = batch_size,
        seq_len = FLAGS.seq_len,
        reuse_len = FLAGS.reuse_len,
        bi_data = FLAGS.bi_data,
        num_hosts = 1,
        num_core_per_host = 1,
        perm_size = FLAGS.perm_size,
        mask_alpha = FLAGS.mask_alpha,
        mask_beta = FLAGS.mask_beta,
        uncased = FLAGS.uncased,
        num_passes = FLAGS.num_passes,
        use_bfloat16 = FLAGS.use_bfloat16,
        num_predict = FLAGS.num_predict,
    )

    return input_fn, record_info_dict


def get_cache_fn(mem_len, batch_size):
    """doc."""
    tf_float = tf.bfloat16 if FLAGS.use_bfloat16 else tf.float32

    def cache_fn():
        mems = []
        if FLAGS.mem_len > 0:
            for _ in range(FLAGS.n_layer):
                zeros = tf.zeros(
                    [mem_len, batch_size, FLAGS.d_model], dtype = tf_float
                )
                mems.append(zeros)

        return mems

    if mem_len > 0:
        return cache_fn
    else:
        return None


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    # Get corpus info
    FLAGS.n_token = data_utils.VOCAB_SIZE
    tf.logging.info('n_token {}'.format(FLAGS.n_token))

    if not tf.gfile.Exists(FLAGS.model_dir):
        tf.gfile.MakeDirs(FLAGS.model_dir)

    bsz_per_core = per_device_batch_size(
        FLAGS.train_batch_size, FLAGS.num_gpu_cores
    )
    tf.logging.info('size of batch {}'.format(bsz_per_core))

    train_input_fn, train_record_info_dict = get_input_fn('train', bsz_per_core)
    tf.logging.info(
        'num of batches {}'.format(train_record_info_dict['num_batch'])
    )
    train_cache_fn = get_cache_fn(FLAGS.mem_len, bsz_per_core)
    tf.logging.info(train_cache_fn)

    tf.logging.info('Use normal RunConfig')
    tf.logging.info(FLAGS.num_gpu_cores)
    dist_strategy = tf.contrib.distribute.MirroredStrategy(
        num_gpus = FLAGS.num_gpu_cores,
        auto_shard_dataset = True,
        cross_device_ops = AllReduceCrossDeviceOps(
            'nccl', num_packs = FLAGS.num_gpu_cores
        ),
        # cross_device_ops=AllReduceCrossDeviceOps('hierarchical_copy'),
    )
    log_every_n_steps = 10
    run_config = RunConfig(
        train_distribute = dist_strategy,
        eval_distribute = dist_strategy,
        log_step_count_steps = log_every_n_steps,
        model_dir = FLAGS.model_dir,
        save_checkpoints_steps = FLAGS.save_steps,
        save_summary_steps = None,
    )
    model_fn = get_model_fn()
    tf.logging.info('Use normal Estimator')
    estimator = Estimator(
        model_fn = model_fn,
        params = {'batch_size': bsz_per_core, 'cache': None},
        config = run_config,
    )

    tf.logging.info('***** Running training *****')
    tf.logging.info('  Batch size = %d', FLAGS.train_batch_size)
    estimator.train(input_fn = train_input_fn, max_steps = FLAGS.train_steps)


if __name__ == '__main__':
    tf.app.run()
