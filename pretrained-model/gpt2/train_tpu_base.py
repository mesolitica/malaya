import json
from gpt_2_simple.src import model
import optimization
import collections
import re
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 256, 'batch_size')
flags.DEFINE_integer('eval_batch_size', 8, 'eval_batch_size')
flags.DEFINE_integer('num_train_steps', 100000, 'num_train_steps')
flags.DEFINE_integer('summary_steps', 100, 'summary_steps')
flags.DEFINE_integer('num_warmup_steps', 10, 'num_warmup_steps')
flags.DEFINE_float('learning_rate', 1e-4, 'learning_rate')
flags.DEFINE_integer('save_checkpoints_steps', 1000, 'save_checkpoints_steps')
flags.DEFINE_integer('max_seq_length', 1024, 'max_seq_length')
flags.DEFINE_integer('max_eval_steps', 100, 'Maximum number of eval steps.')

flags.DEFINE_integer(
    'iterations_per_loop',
    1000,
    'How many steps to make in each estimator call.',
)

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

flags.DEFINE_string('init_checkpoint', None, 'Initial checkpoint')

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
flags.DEFINE_integer(
    'num_tpu_cores',
    8,
    'Only used if `use_tpu` is True. Total number of TPU cores to use.',
)

tf.flags.DEFINE_string('master', None, '[Optional] TensorFlow master URL.')

flags.DEFINE_bool('do_train', False, 'Whether to run training.')

flags.DEFINE_bool('do_eval', False, 'Whether to run eval on the dev set.')

flags.DEFINE_bool('use_tpu', True, 'Whether to use TPU or GPU/CPU.')


# https://storage.googleapis.com/gpt-2/models/117M/hparams.json
hparams = model.default_hparams()
with open('base-hparams.json') as f:
    hparams.override_from_dict(json.load(f))


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
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


def model_fn_builder(
    init_checkpoint, learning_rate, num_train_steps, num_warmup_steps
):
    def model_fn(features, labels, mode, params):
        tf.logging.info('*** Features ***')
        for name in sorted(features.keys()):
            tf.logging.info(
                '  name = %s, shape = %s' % (name, features[name].shape)
            )

        input_ids = features['input_ids']

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        output = model.model(hparams = hparams, X = input_ids)
        loss = tf.reduce_mean(
            input_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = input_ids[:, 1:], logits = output['logits'][:, :-1]
            )
        )

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        (
            assignment_map,
            initialized_variable_names,
        ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)

        def tpu_scaffold():
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
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
            train_op = optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, True
            )

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode = mode,
                loss = loss,
                train_op = train_op,
                scaffold_fn = scaffold_fn,
            )
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(loss, input_ids, output):
                next_sentence_predictions = tf.argmax(
                    next_sentence_log_probs, axis = -1, output_type = tf.int32
                )
                next_sentence_labels = tf.reshape(input_ids, [-1])
                next_sentence_accuracy = tf.metrics.accuracy(
                    labels = next_sentence_labels,
                    predictions = next_sentence_predictions,
                )
                next_sentence_mean_loss = tf.metrics.mean(values = loss)

                return {
                    'next_sentence_accuracy': next_sentence_accuracy,
                    'next_sentence_loss': next_sentence_mean_loss,
                }

            eval_metrics = (metric_fn, [loss, input_ids, output])
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
    input_files, max_seq_length, is_training, num_cpu_threads = 4
):
    def input_fn(params):
        batch_size = params['batch_size']
        name_to_features = {
            'input_ids': tf.FixedLenFeature([max_seq_length], tf.int64)
        }
        if is_training:
            d = tf.data.TFRecordDataset(input_files)
            d = d.repeat(1000)
            # d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            # d = d.repeat()
            # d = d.shuffle(buffer_size = len(input_files))
            # cycle_length = min(num_cpu_threads, len(input_files))
            # d = d.apply(
            #     tf.contrib.data.parallel_interleave(
            #         tf.data.TFRecordDataset,
            #         sloppy = is_training,
            #         cycle_length = cycle_length,
            #     )
            # )
            # d = d.shuffle(buffer_size = 100)

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

    tf.gfile.MakeDirs(FLAGS.output_dir)
    input_files = []
    for input_pattern in FLAGS.input_file.split(','):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info('*** Input Files ***')
    for input_file in input_files:
        tf.logging.info('  %s' % input_file)

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
        save_summary_steps = FLAGS.summary_steps,
    )
    model_fn = model_fn_builder(
        init_checkpoint = FLAGS.init_checkpoint,
        learning_rate = FLAGS.learning_rate,
        num_train_steps = FLAGS.num_train_steps,
        num_warmup_steps = FLAGS.num_warmup_steps,
    )
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu = FLAGS.use_tpu,
        model_fn = model_fn,
        config = run_config,
        train_batch_size = FLAGS.batch_size,
        eval_batch_size = FLAGS.eval_batch_size,
    )
    if FLAGS.do_train:
        tf.logging.info('***** Running training *****')
        tf.logging.info('  Batch size = %d', FLAGS.batch_size)
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
        eval_input_fn = input_fn_builder(
            input_files = input_files,
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
