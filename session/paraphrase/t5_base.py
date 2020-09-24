import tensorflow as tf
import tensorflow_datasets as tfds
import t5
import functools

vocab = 'gs://mesolitica-tpu-general/t5-data/sp10m.cased.t5.model'
tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
    'node-2', zone = 'europe-west4-a', project = 'mesolitica-tpu'
)
TPU_ADDRESS = tpu.get_master()
TPU_TOPOLOGY = '2x2'
print(TPU_ADDRESS)


def cnn_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/t5-data/bahasa-paraphrase-0.tsv',
            'gs://mesolitica-tpu-general/t5-data/bahasa-paraphrase-1.tsv',
            'gs://mesolitica-tpu-general/t5-data/bahasa-paraphrase-2.tsv',
            'gs://mesolitica-tpu-general/t5-data/bahasa-paraphrase-3.tsv',
        ]
    )

    ds = ds.map(
        functools.partial(
            tf.io.decode_csv,
            record_defaults = ['', ''],
            field_delim = '\t',
            use_quote_delim = False,
        ),
        num_parallel_calls = tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(lambda *ex: dict(zip(['question', 'answer'], ex)))
    return ds


def cnn_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['parafrasa: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls = tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('cnn_dataset')
t5.data.TaskRegistry.add(
    'cnn_dataset',
    dataset_fn = cnn_dataset,
    splits = ['train'],
    text_preprocessor = [cnn_preprocessor],
    sentencepiece_model_path = vocab,
    metric_fns = [t5.evaluation.metrics.accuracy],
)

t5.data.MixtureRegistry.remove('generator_bahasa')
t5.data.MixtureRegistry.add(
    'generator_bahasa', ['cnn_dataset'], default_rate = 1.0
)


def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    model_parallelism, train_batch_size, keep_checkpoint_max = {
        'small': (1, 256, 16),
        'base': (2, 128, 8),
        'large': (8, 64, 4),
        '3B': (8, 16, 1),
        '11B': (8, 16, 1),
    }['base']

    BASE_DIR = 'gs://mesolitica-tpu-general/t5-base-paraphrase-v1'
    model = t5.models.MtfModel(
        model_dir = BASE_DIR,
        tpu = TPU_ADDRESS,
        tpu_topology = TPU_TOPOLOGY,
        model_parallelism = model_parallelism,
        batch_size = train_batch_size,
        sequence_length = {'inputs': 1024, 'targets': 1024},
        learning_rate_schedule = 0.003,
        save_checkpoints_steps = 5000,
        keep_checkpoint_max = 5,
        iterations_per_loop = 100,
    )

    FINETUNE_STEPS = 50000
    MODEL_DIR = 'gs://mesolitica-tpu-general/old-t5-base'

    model.finetune(
        mixture_or_task_name = 'generator_bahasa',
        pretrained_model_dir = MODEL_DIR,
        finetune_steps = FINETUNE_STEPS,
    )


if __name__ == '__main__':
    tf.app.run()
