import tensorflow as tf
from tensor2tensor.utils import adafactor
from pegasus import transformer
from tensorflow.contrib import layers as contrib_layers
import re
import collections
import six

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file',
    None,
    'Input TF example files (can be a glob or comma separated).',
)

flags.DEFINE_string(
    'output_dir',
    None,
    'The output directory where the model checkpoints will be written.',
)

flags.DEFINE_integer(
    'max_seq_length_encoder',
    512,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded. Must match data generation.',
)

flags.DEFINE_integer(
    'max_seq_length_decoder',
    128,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded. Must match data generation.',
)

flags.DEFINE_integer(
    'max_predictions_per_seq',
    30,
    'Maximum number of masked LM predictions per sequence. '
    'Must match data generation.',
)

flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')

flags.DEFINE_float(
    'learning_rate', 0.001, 'The initial learning rate for Adafactor.'
)


flags.DEFINE_integer('num_train_steps', 100000, 'Number of training steps.')

flags.DEFINE_integer('num_warmup_steps', 10000, 'Number of warmup steps.')

flags.DEFINE_integer(
    'save_checkpoints_steps', 10000, 'How often to save the model checkpoint.'
)

flags.DEFINE_integer(
    'iterations_per_loop', 100, 'How many steps to make in each estimator call.'
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

flags.DEFINE_bool('do_train', True, 'Whether to run training.')

flags.DEFINE_string(
    'init_checkpoint',
    None,
    'Initial checkpoint (usually from a pre-trained PEGASUS model).',
)

vocab_size = 32000
hidden_size = 512
filter_size = 3072
num_encoder_layers = 6
num_decoder_layers = 6
num_heads = 8
label_smoothing = 0.0
dropout = 0.1


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

    return (assignment_map, initialized_variable_names)


def input_fn_builder(
    input_files,
    max_seq_length_encoder,
    max_seq_length_decoder,
    max_predictions_per_seq,
    is_training,
    num_cpu_threads = 4,
):
    def input_fn(params):
        batch_size = params['batch_size']

        name_to_features = {
            'input_ids': tf.FixedLenFeature([max_seq_length_encoder], tf.int64),
            'target_ids': tf.FixedLenFeature(
                [max_seq_length_decoder], tf.int64
            ),
            'masked_lm_positions': tf.FixedLenFeature(
                [max_predictions_per_seq], tf.int64
            ),
            'masked_lm_ids': tf.FixedLenFeature(
                [max_predictions_per_seq], tf.int64
            ),
            'masked_lm_weights': tf.FixedLenFeature(
                [max_predictions_per_seq], tf.float32
            ),
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


def assert_rank(tensor, expected_rank, name = None):
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            'For the tensor `%s` in scope `%s`, the actual rank '
            '`%d` (shape = %s) is not equal to the expected rank `%s`'
            % (
                name,
                scope_name,
                actual_rank,
                str(tensor.shape),
                str(expected_rank),
            )
        )


def get_shape_list(tensor, expected_rank = None, name = None):
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = get_shape_list(sequence_tensor, expected_rank = 3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype = tf.int32) * seq_length, [-1, 1]
    )
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(
        sequence_tensor, [batch_size * seq_length, width]
    )
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def get_masked_lm_output(
    input_tensor, output_weights, positions, label_ids, label_weights
):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope('cls/predictions'):
        with tf.variable_scope('transform'):
            input_tensor = tf.layers.dense(
                input_tensor,
                units = hidden_size,
                activation = tf.nn.relu,
                kernel_initializer = tf.random_normal_initializer(
                    stddev = hidden_size ** -0.5, dtype = tf.float32
                ),
            )
            input_tensor = contrib_layers.layer_norm(input_tensor)

        output_bias = tf.get_variable(
            'output_bias',
            shape = [vocab_size],
            initializer = tf.zeros_initializer(),
        )
        logits = tf.matmul(input_tensor, output_weights, transpose_b = True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis = -1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth = vocab_size, dtype = tf.float32
        )

        per_example_loss = -tf.reduce_sum(
            log_probs * one_hot_labels, axis = [-1]
        )
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def model_fn_builder(
    init_checkpoint, learning_rate, num_train_steps, num_warmup_steps, use_tpu
):
    def model_fn(features, labels, mode, params):
        tf.logging.info('*** Features ***')
        for name in sorted(features.keys()):
            tf.logging.info(
                '  name = %s, shape = %s' % (name, features[name].shape)
            )

        input_ids = features['input_ids']
        target_ids = features['target_ids']
        masked_lm_positions = features['masked_lm_positions']
        masked_lm_ids = features['masked_lm_ids']
        masked_lm_weights = features['masked_lm_weights']

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        model = transformer.TransformerEncoderDecoderModel(
            vocab_size,
            hidden_size,
            filter_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            label_smoothing,
            dropout,
        )

        loss, outputs = model(
            {'inputs': input_ids, 'targets': target_ids}, training = is_training
        )

        (
            masked_lm_loss,
            masked_lm_example_loss,
            masked_lm_log_probs,
        ) = get_masked_lm_output(
            model._context['memory'],
            model._embedding_layer.weights_VxD,
            masked_lm_positions,
            masked_lm_ids,
            masked_lm_weights,
        )

        total_loss = masked_lm_loss + loss

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (
                assignment_map,
                initialized_variable_names,
            ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
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

            init_lr = learning_rate
            global_step = tf.train.get_global_step()
            lr = (
                init_lr
                / 0.01
                * tf.rsqrt(tf.maximum(tf.to_float(global_step), 10000))
            )

            if num_warmup_steps:
                global_steps_int = tf.cast(global_step, tf.int32)
                warmup_steps_int = tf.constant(
                    num_warmup_steps, dtype = tf.int32
                )

                global_steps_float = tf.cast(global_steps_int, tf.float32)
                warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

                warmup_percent_done = global_steps_float / warmup_steps_float
                warmup_learning_rate = init_lr * warmup_percent_done

                is_warmup = tf.cast(
                    global_steps_int < warmup_steps_int, tf.float32
                )
                lr = (1.0 - is_warmup) * lr + is_warmup * warmup_learning_rate

            optimizer = adafactor.AdafactorOptimizer(
                learning_rate = lr,
                decay_rate = adafactor.adafactor_decay_rate_pow(0.8),
                beta1 = 0.0,
            )
            if use_tpu:
                optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

            train_op = optimizer.minimize(loss, global_step = global_step)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode = mode,
                loss = total_loss,
                train_op = train_op,
                scaffold_fn = scaffold_fn,
            )
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(
                masked_lm_example_loss,
                masked_lm_log_probs,
                masked_lm_ids,
                masked_lm_weights,
            ):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.reshape(
                    masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]]
                )
                masked_lm_predictions = tf.argmax(
                    masked_lm_log_probs, axis = -1, output_type = tf.int32
                )
                masked_lm_example_loss = tf.reshape(
                    masked_lm_example_loss, [-1]
                )
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tf.metrics.accuracy(
                    labels = masked_lm_ids,
                    predictions = masked_lm_predictions,
                    weights = masked_lm_weights,
                )
                masked_lm_mean_loss = tf.metrics.mean(
                    values = masked_lm_example_loss, weights = masked_lm_weights
                )

                return {
                    'masked_lm_accuracy': masked_lm_accuracy,
                    'masked_lm_loss': masked_lm_mean_loss,
                }

            eval_metrics = (
                metric_fn,
                [
                    masked_lm_example_loss,
                    masked_lm_log_probs,
                    masked_lm_ids,
                    masked_lm_weights,
                ],
            )
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

    tf.logging.info('*** Input Files ***')
    for input_file in input_files:
        tf.logging.info('  %s' % input_file)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone = FLAGS.tpu_zone, project = FLAGS.gcp_project
        )

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
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
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu = FLAGS.use_tpu,
        model_fn = model_fn,
        config = run_config,
        train_batch_size = FLAGS.train_batch_size,
        eval_batch_size = FLAGS.train_batch_size,
    )

    if FLAGS.do_train:
        tf.logging.info('***** Running training *****')
        tf.logging.info('  Batch size = %d', FLAGS.train_batch_size)
        train_input_fn = input_fn_builder(
            input_files = input_files,
            max_seq_length_encoder = FLAGS.max_seq_length_encoder,
            max_seq_length_decoder = FLAGS.max_seq_length_decoder,
            max_predictions_per_seq = FLAGS.max_predictions_per_seq,
            is_training = True,
        )
        estimator.train(
            input_fn = train_input_fn, max_steps = FLAGS.num_train_steps
        )


if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('output_dir')
    tf.app.run()
