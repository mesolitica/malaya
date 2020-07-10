from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import multilanguagebert_transformer
import custom_optimization
from tensorflow.python.distribute.cross_device_ops import (
    AllReduceCrossDeviceOps,
)
import tensorflow as tf
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.estimator import Estimator

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file',
    'multilanguagebert-train-*.tfrecord',
    'Input TF example files (can be a glob or comma separated).',
)

flags.DEFINE_string(
    'test_file',
    'multilanguagebert-test-*.tfrecord',
    'Input TF example files (can be a glob or comma separated).',
)
flags.DEFINE_string(
    'init_checkpoint',
    '../multi_cased_L-12_H-768_A-12/bert_model.ckpt',
    'Initial checkpoint (usually from a pre-trained BERT model).',
)
flags.DEFINE_integer(
    'max_seq_length',
    256,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded. Must match data generation.',
)
flags.DEFINE_integer(
    'train_batch_size', 18 * 3, 'Total batch size for training.'
)

flags.DEFINE_integer('eval_batch_size', 8, 'Total batch size for eval.')

flags.DEFINE_float('learning_rate', 5e-5, 'The initial learning rate for Adam.')

flags.DEFINE_integer('num_train_steps', 500000, 'Number of training steps.')

flags.DEFINE_integer('num_warmup_steps', 10000, 'Number of warmup steps.')

flags.DEFINE_integer(
    'save_checkpoints_steps', 25000, 'How often to save the model checkpoint.'
)
flags.DEFINE_bool('use_gpu', True, 'Whether to use GPU.')
flags.DEFINE_integer(
    'num_gpu_cores',
    3,
    'Only used if `use_gpu` is True. Total number of GPU cores to use.',
)
flags.DEFINE_bool('do_train', True, 'Whether to run training.')

flags.DEFINE_bool('do_eval', True, 'Whether to run eval on the dev set.')

flags.DEFINE_string(
    'output_dir',
    'multibert-transformer',
    'The output directory where the model checkpoints will be written.',
)
flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU or GPU/CPU.')
tf.flags.DEFINE_string(
    'tpu_name',
    None,
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.',
)

tf.flags.DEFINE_string(
    'tpu_zone',
    None,
    '[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.',
)

tf.flags.DEFINE_string(
    'gcp_project',
    None,
    '[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.',
)

tf.flags.DEFINE_string('master', None, '[Optional] TensorFlow master URL.')

flags.DEFINE_integer(
    'num_tpu_cores',
    8,
    'Only used if `use_tpu` is True. Total number of TPU cores to use.',
)

flags.DEFINE_integer(
    'iterations_per_loop',
    1000,
    'How many steps to make in each estimator call.',
)


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


import collections
import re


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
        if 'bert/' + name in name_to_variable:
            assignment_map[name] = name_to_variable['bert/' + name]
            initialized_variable_names[name] = 1
            initialized_variable_names[name + ':0'] = 1
        elif name in name_to_variable:
            assignment_map[name] = name_to_variable[name]
            initialized_variable_names[name] = 1
            initialized_variable_names[name + ':0'] = 1

    return (assignment_map, initialized_variable_names)


def model_fn_builder(
    init_checkpoint,
    learning_rate,
    num_train_steps,
    num_warmup_steps,
    use_tpu,
    use_one_hot_embeddings,
):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(
        features, labels, mode, params
    ):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info('*** Features ***')
        for name in sorted(features.keys()):
            tf.logging.info(
                '  name = %s, shape = %s' % (name, features[name].shape)
            )

        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']
        y = features['y']

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        model = multilanguagebert_transformer.Model(
            is_training = is_training,
            input_ids = input_ids,
            input_mask = input_mask,
            token_type_ids = segment_ids,
            Y = y,
        )
        o = model.get_sequence_output()
        Y_seq_len = tf.count_nonzero(y, 1, dtype = tf.int32)
        masks = tf.sequence_mask(Y_seq_len, tf.shape(y)[1], dtype = tf.float32)
        total_loss = tf.contrib.seq2seq.sequence_loss(
            logits = o, targets = y, weights = masks
        )
        y_t = tf.argmax(o, axis = 2)
        y_t = tf.cast(y_t, tf.int32)
        prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(y, masks)
        correct_pred = tf.equal(prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        total_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (
                assignment_map,
                initialized_variable_names,
            ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            print(initialized_variable_names)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(
                        init_checkpoint, assignment_map
                    )
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info('**** Trainable Variables ****')
        for var in tvars:
            init_string = ''
            if var.name in initialized_variable_names:
                init_string = ', *INIT_FROM_CKPT*'
            tf.logging.info(
                '  name = %s, shape = %s%s', var.name, var.shape, init_string
            )

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            if FLAGS.use_gpu and int(FLAGS.num_gpu_cores) >= 2:
                train_op = custom_optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps
                )
            else:
                train_op = optimization.create_optimizer(
                    total_loss,
                    learning_rate,
                    num_train_steps,
                    num_warmup_steps,
                    use_tpu,
                )
            if FLAGS.use_gpu and int(FLAGS.num_gpu_cores) >= 2:
                output_spec = tf.estimator.EstimatorSpec(
                    mode = mode,
                    loss = total_loss,
                    train_op = train_op,
                    scaffold = scaffold_fn,
                )
            else:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode = mode,
                    loss = total_loss,
                    train_op = train_op,
                    scaffold_fn = scaffold_fn,
                )
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(loss, accuracy):
                return {'total_loss': loss, 'total_accuracy': accuracy}

            eval_metrics = (metric_fn, [total_loss, total_accuracy])
            if FLAGS.use_gpu and int(FLAGS.num_gpu_cores) >= 2:
                output_spec = tf.estimator.EstimatorSpec(
                    mode = mode,
                    loss = total_loss,
                    eval_metrics = eval_metrics,
                    scaffold = scaffold_fn,
                )
            else:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode = mode,
                    loss = total_loss,
                    eval_metrics = eval_metrics,
                    scaffold_fn = scaffold_fn,
                )
        else:
            raise ValueError(
                'Only TRAIN and EVAL modes are supported: %s' % (mode)
            )

        return output_spec

    return model_fn


def input_fn_builder(
    input_files, max_seq_length, is_training, batch_size, num_cpu_threads = 4
):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""

        name_to_features = {
            'input_ids': tf.FixedLenFeature([max_seq_length], tf.int64),
            'input_mask': tf.FixedLenFeature([max_seq_length], tf.int64),
            'segment_ids': tf.FixedLenFeature([max_seq_length], tf.int64),
            'y': tf.FixedLenFeature([max_seq_length], tf.int64),
        }

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
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            d = d.repeat()
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size = batch_size,
                num_parallel_batches = num_cpu_threads,
                drop_remainder = True,
            )
        )
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError(
            'At least one of `do_train` or `do_eval` must be True.'
        )

    tf.gfile.MakeDirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(','):
        input_files.extend(tf.gfile.Glob(input_pattern))

    test_files = []
    for input_pattern in FLAGS.test_file.split(','):
        test_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info('*** Input Files ***')
    for input_file in input_files:
        tf.logging.info('  %s' % input_file)

    tf.logging.info('*** Test Files ***')
    for input_file in test_files:
        tf.logging.info('  %s' % input_file)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone = FLAGS.tpu_zone, project = FLAGS.gcp_project
        )

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    if FLAGS.use_gpu and int(FLAGS.num_gpu_cores) >= 2:
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
            model_dir = FLAGS.output_dir,
            save_checkpoints_steps = FLAGS.save_checkpoints_steps,
            save_summary_steps = None,
        )

    else:
        run_config = tf.contrib.tpu.RunConfig(
            cluster = tpu_cluster_resolver,
            master = FLAGS.master,
            model_dir = FLAGS.output_dir,
            save_checkpoints_steps = FLAGS.save_checkpoints_steps,
            tpu_config = tf.contrib.tpu.TPUConfig(
                iterations_per_loop = FLAGS.iterations_per_loop,
                num_shards = FLAGS.num_tpu_cores,
                per_host_input_for_training = is_per_host,
            ),
        )

    model_fn = model_fn_builder(
        init_checkpoint = FLAGS.init_checkpoint,
        learning_rate = FLAGS.learning_rate,
        num_train_steps = FLAGS.num_train_steps,
        num_warmup_steps = FLAGS.num_warmup_steps,
        use_tpu = FLAGS.use_tpu,
        use_one_hot_embeddings = FLAGS.use_tpu,
    )

    if FLAGS.use_gpu and int(FLAGS.num_gpu_cores) >= 2:
        tf.logging.info('Use normal Estimator')
        estimator = Estimator(
            model_fn = model_fn, params = {}, config = run_config
        )

    else:
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu = FLAGS.use_tpu,
            model_fn = model_fn,
            config = run_config,
            train_batch_size = FLAGS.train_batch_size,
            eval_batch_size = FLAGS.eval_batch_size,
        )

    if FLAGS.do_train:
        tf.logging.info('***** Running training *****')
        tf.logging.info('  Batch size = %d', FLAGS.train_batch_size)

        if FLAGS.use_gpu and int(FLAGS.num_gpu_cores) >= 2:
            train_input_fn = input_fn_builder(
                input_files = input_files,
                max_seq_length = FLAGS.max_seq_length,
                is_training = True,
                batch_size = per_device_batch_size(
                    FLAGS.train_batch_size, FLAGS.num_gpu_cores
                ),
            )
        else:
            train_input_fn = input_fn_builder(
                input_files = input_files,
                max_seq_length = FLAGS.max_seq_length,
                is_training = True,
            )
        estimator.train(
            input_fn = train_input_fn, max_steps = FLAGS.num_train_steps
        )

    if FLAGS.do_eval:
        tf.logging.info('***** Running evaluation *****')
        tf.logging.info('  Batch size = %d', FLAGS.eval_batch_size)

        if FLAGS.use_gpu and int(FLAGS.num_gpu_cores) >= 2:
            train_input_fn = input_fn_builder_gpu(
                input_files = test_files,
                max_seq_length = FLAGS.max_seq_length,
                is_training = False,
                batch_size = FLAGS.eval_batch_size,
            )
        else:
            eval_input_fn = input_fn_builder(
                input_files = test_files,
                max_seq_length = FLAGS.max_seq_length,
                is_training = False,
            )

        result = estimator.evaluate(
            input_fn = eval_input_fn, steps = FLAGS.max_eval_steps
        )

        output_eval_file = os.path.join(FLAGS.output_dir, 'eval_results.txt')
        with tf.gfile.GFile(output_eval_file, 'w') as writer:
            tf.logging.info('***** Eval results *****')
            for key in sorted(result.keys()):
                tf.logging.info('  %s = %s', key, str(result[key]))
                writer.write('%s = %s\n' % (key, str(result[key])))


if __name__ == '__main__':
    tf.app.run()
