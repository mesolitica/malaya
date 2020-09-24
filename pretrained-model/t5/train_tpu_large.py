import tensorflow as tf
import tensorflow_datasets as tfds
from t5.data import preprocessors as prep
import functools
import t5
import gin


vocab = 'gs://mesolitica-tpu-general/t5-data/sp10m.cased.t5.model'
tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
    'node-4', zone = 'europe-west4-a', project = 'mesolitica-tpu'
)
TPU_ADDRESS = tpu.get_master()
TPU_TOPOLOGY = '2x2'
print(TPU_ADDRESS)


def dumping_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/t5-data/cleaned-news.txt.tsv',
            'gs://mesolitica-tpu-general/t5-data/dumping-parliament.txt.tsv',
            'gs://mesolitica-tpu-general/t5-data/filtered-dumping-academia.txt.tsv',
            'gs://mesolitica-tpu-general/t5-data/filtered-dumping-cleaned-common-crawl.txt.tsv',
            'gs://mesolitica-tpu-general/t5-data/filtered-dumping-wiki.txt.tsv',
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
    ds = ds.map(lambda *ex: dict(zip(['title', 'text'], ex)))
    return ds


t5.data.TaskRegistry.remove('dumping_dataset')
t5.data.TaskRegistry.add(
    'dumping_dataset',
    dataset_fn = dumping_dataset,
    splits = ['train'],
    text_preprocessor = functools.partial(
        t5.data.preprocessors.rekey,
        key_map = {'inputs': None, 'targets': 'text'},
    ),
    token_preprocessor = t5.data.preprocessors.unsupervised,
    sentencepiece_model_path = vocab,
    metric_fns = [],
)


def question_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(['gs://mesolitica-tpu-general/t5-data/qa.tsv'])

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


def question_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['soalan: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls = tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('question_dataset')
t5.data.TaskRegistry.add(
    'question_dataset',
    dataset_fn = question_dataset,
    splits = ['train'],
    text_preprocessor = [question_preprocessor],
    sentencepiece_model_path = vocab,
    metric_fns = [t5.evaluation.metrics.accuracy],
)


def pair_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/t5-data/cleaned-news.txt-pair.tsv',
            'gs://mesolitica-tpu-general/t5-data/dumping-parliament.txt-pair.tsv',
            'gs://mesolitica-tpu-general/t5-data/filtered-dumping-academia.txt-pair.tsv',
            'gs://mesolitica-tpu-general/t5-data/filtered-dumping-cleaned-common-crawl.txt-pair.tsv',
            'gs://mesolitica-tpu-general/t5-data/filtered-dumping-wiki.txt-pair.tsv',
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
    ds = ds.map(lambda *ex: dict(zip(['text'], ex)))
    return ds


t5.data.TaskRegistry.remove('pair_dataset')
t5.data.TaskRegistry.add(
    'pair_dataset',
    dataset_fn = pair_dataset,
    splits = ['train'],
    text_preprocessor = [prep.next_sentence_prediction],
    sentencepiece_model_path = vocab,
    metric_fns = [t5.evaluation.metrics.accuracy],
)


def news_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        ['gs://mesolitica-tpu-general/t5-data/newstitle.tsv']
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


def stemming_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        ['gs://mesolitica-tpu-general/t5-data/stemming.tsv']
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


def stemming_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['punca: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls = tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('stemming_dataset')
t5.data.TaskRegistry.add(
    'stemming_dataset',
    dataset_fn = stemming_dataset,
    splits = ['train'],
    text_preprocessor = [stemming_preprocessor],
    sentencepiece_model_path = vocab,
    metric_fns = [t5.evaluation.metrics.accuracy],
)


def synonym_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        ['gs://mesolitica-tpu-general/t5-data/synonyms.tsv']
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


def synonym_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['sinonim: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls = tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('synonym_dataset')
t5.data.TaskRegistry.add(
    'synonym_dataset',
    dataset_fn = synonym_dataset,
    splits = ['train'],
    text_preprocessor = [synonym_preprocessor],
    sentencepiece_model_path = vocab,
    metric_fns = [t5.evaluation.metrics.accuracy],
)


def similarity_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/t5-data/quora.tsv',
            'gs://mesolitica-tpu-general/t5-data/snli.tsv',
            'gs://mesolitica-tpu-general/t5-data/mnli.tsv',
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


def similarity_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {'inputs': ex['question'], 'targets': ex['answer']}

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls = tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('similarity_dataset')
t5.data.TaskRegistry.add(
    'similarity_dataset',
    dataset_fn = similarity_dataset,
    splits = ['train'],
    text_preprocessor = [similarity_preprocessor],
    sentencepiece_model_path = vocab,
    metric_fns = [t5.evaluation.metrics.accuracy],
)

t5.data.MixtureRegistry.remove('trivia_all_bahasa')
t5.data.MixtureRegistry.add(
    'trivia_all_bahasa',
    [
        'dumping_dataset',
        'question_dataset',
        'pair_dataset',
        'news_dataset',
        'stemming_dataset',
        'synonym_dataset',
        'similarity_dataset',
    ],
    default_rate = 1.0,
)


def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    gin.parse_config_file(
        'gs://mesolitica-tpu-general/t5-data/pretrained_models_large_operative_config.gin'
    )

    MODEL_SIZE = 'large'
    model_parallelism, train_batch_size, keep_checkpoint_max = {
        'small': (1, 256, 16),
        'base': (2, 128, 8),
        'large': (8, 64, 4),
        '3B': (8, 16, 1),
        '11B': (8, 16, 1),
    }[MODEL_SIZE]

    model = t5.models.MtfModel(
        model_dir = 'gs://mesolitica-tpu-general/t5-large-v2/',
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

    model.train(mixture_or_task_name = 'trivia_all_bahasa', steps = 1000000)


if __name__ == '__main__':
    tf.app.run()
