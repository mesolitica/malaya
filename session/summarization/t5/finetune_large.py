import tensorflow as tf
import tensorflow_datasets as tfds
import t5
import functools

vocab = 'gs://mesolitica-tpu-general/t5-data/sp10m.cased.t5.model'
tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
    'node-10', zone = 'europe-west4-a', project = 'mesolitica-tpu'
)
TPU_ADDRESS = tpu.get_master()
TPU_TOPOLOGY = '2x2'
print(TPU_ADDRESS)


def cnn_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/t5-data/cnn-summarization-0.tsv',
            'gs://mesolitica-tpu-general/t5-data/cnn-summarization-1.tsv',
            'gs://mesolitica-tpu-general/t5-data/cnn-summarization-2.tsv',
            'gs://mesolitica-tpu-general/t5-data/cnn-summarization-3.tsv',
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
            'inputs': tf.strings.join(['ringkasan: ', ex['question']]),
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


def multinews_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/t5-data/multinews-summarization-0.tsv',
            'gs://mesolitica-tpu-general/t5-data/multinews-summarization-1.tsv',
            'gs://mesolitica-tpu-general/t5-data/multinews-summarization-2.tsv',
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


def multinews_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['ringkasan: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls = tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('multinews_dataset')
t5.data.TaskRegistry.add(
    'multinews_dataset',
    dataset_fn = multinews_dataset,
    splits = ['train'],
    text_preprocessor = [multinews_preprocessor],
    sentencepiece_model_path = vocab,
    metric_fns = [t5.evaluation.metrics.accuracy],
)


def gigawords_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/t5-data/gigawords-summarization-0.tsv',
            'gs://mesolitica-tpu-general/t5-data/gigawords-summarization-1.tsv',
            'gs://mesolitica-tpu-general/t5-data/gigawords-summarization-2.tsv',
            'gs://mesolitica-tpu-general/t5-data/gigawords-summarization-3.tsv',
            'gs://mesolitica-tpu-general/t5-data/gigawords-summarization-4.tsv',
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


def gigawords_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['perenggan: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls = tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('gigawords_dataset')
t5.data.TaskRegistry.add(
    'gigawords_dataset',
    dataset_fn = gigawords_dataset,
    splits = ['train'],
    text_preprocessor = [gigawords_preprocessor],
    sentencepiece_model_path = vocab,
    metric_fns = [t5.evaluation.metrics.accuracy],
)


def news_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        ['gs://mesolitica-tpu-general/t5-data-cleaned/newstitle.tsv']
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


def news_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['tajuk: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls = tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('news_dataset')
t5.data.TaskRegistry.add(
    'news_dataset',
    dataset_fn = news_dataset,
    splits = ['train'],
    text_preprocessor = [news_preprocessor],
    sentencepiece_model_path = vocab,
    metric_fns = [t5.evaluation.metrics.accuracy],
)

t5.data.MixtureRegistry.remove('summarization_bahasa')
t5.data.MixtureRegistry.add(
    'summarization_bahasa',
    ['cnn_dataset', 'multinews_dataset', 'gigawords_dataset', 'news_dataset'],
    default_rate = 1.0,
)


def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    model_parallelism, train_batch_size, keep_checkpoint_max = {
        'small': (1, 256, 16),
        'base': (2, 128, 8),
        'large': (8, 64, 4),
        '3B': (8, 16, 1),
        '11B': (8, 16, 1),
    }['large']

    BASE_DIR = 'gs://mesolitica-tpu-general/t5-large-summary'
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
    MODEL_DIR = 'gs://mesolitica-tpu-general/t5-large-v2'

    model.finetune(
        mixture_or_task_name = 'summarization_bahasa',
        pretrained_model_dir = MODEL_DIR,
        finetune_steps = FINETUNE_STEPS,
    )


if __name__ == '__main__':
    tf.app.run()
