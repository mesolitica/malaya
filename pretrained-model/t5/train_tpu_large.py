import tensorflow as tf
import tensorflow_datasets as tfds
from t5.data import preprocessors as prep
import functools
import t5
import gin


vocab = 'gs://mesolitica-tpu-general/t5-data-v2/sp10m.cased.ms-en.model'
tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
    'node-5', zone='europe-west4-a', project='mesolitica-tpu'
)
TPU_ADDRESS = tpu.get_master()
TPU_TOPOLOGY = '2x2'
print(TPU_ADDRESS)


def dumping_dataset(split, shuffle_files=False):
    del shuffle_files
    files = [
        'gs://mesolitica-tpu-general/t5-data-v2/dumping-news.txt.tsv',
        'gs://mesolitica-tpu-general/t5-data-v2/dumping-parliament.txt.tsv',
        'gs://mesolitica-tpu-general/t5-data-v2/filtered-dumping-academia.txt.tsv',
        'gs://mesolitica-tpu-general/t5-data-v2/filtered-dumping-wiki.txt.tsv'
    ]
    files.extend(tf.io.gfile.glob('gs://mesolitica-tpu-general/t5-data-v2/00.jsonl-*.translated.txt.tsv'))
    ds = tf.data.TextLineDataset(files)

    ds = ds.map(
        functools.partial(
            tf.io.decode_csv,
            record_defaults=['', ''],
            field_delim='\t',
            use_quote_delim=False,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(lambda *ex: dict(zip(['title', 'text'], ex)))
    return ds


t5.data.TaskRegistry.remove('dumping_dataset')
t5.data.TaskRegistry.add(
    'dumping_dataset',
    dataset_fn=dumping_dataset,
    splits=['train'],
    text_preprocessor=functools.partial(
        t5.data.preprocessors.rekey,
        key_map={'inputs': None, 'targets': 'text'},
    ),
    token_preprocessor=t5.data.preprocessors.unsupervised,
    sentencepiece_model_path=vocab,
    metric_fns=[],
)


def question_dataset(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/t5-data-v2/qa.tsv',
        ]
    )

    ds = ds.map(
        functools.partial(
            tf.io.decode_csv,
            record_defaults=['', ''],
            field_delim='\t',
            use_quote_delim=False,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
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
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('question_dataset')
t5.data.TaskRegistry.add(
    'question_dataset',
    dataset_fn=question_dataset,
    splits=['train'],
    text_preprocessor=[question_preprocessor],
    sentencepiece_model_path=vocab,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
)


def pair_dataset(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(tf.io.gfile.glob('gs://mesolitica-tpu-general/t5-data-v2/*pair.tsv'))

    ds = ds.map(
        functools.partial(
            tf.io.decode_csv,
            record_defaults=['', ''],
            field_delim='\t',
            use_quote_delim=False,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(lambda *ex: dict(zip(['text'], ex)))
    return ds


t5.data.TaskRegistry.remove('pair_dataset')
t5.data.TaskRegistry.add(
    'pair_dataset',
    dataset_fn=pair_dataset,
    splits=['train'],
    text_preprocessor=[prep.next_sentence_prediction],
    sentencepiece_model_path=vocab,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
)


def news_dataset(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/t5-data-v2/newstitle.tsv'
        ]
    )

    ds = ds.map(
        functools.partial(
            tf.io.decode_csv,
            record_defaults=['', ''],
            field_delim='\t',
            use_quote_delim=False,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
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
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('news_dataset')
t5.data.TaskRegistry.add(
    'news_dataset',
    dataset_fn=news_dataset,
    splits=['train'],
    text_preprocessor=[news_preprocessor],
    sentencepiece_model_path=vocab,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
)


def summarization_dataset(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/t5-data-v2/summarization.tsv'
        ]
    )

    ds = ds.map(
        functools.partial(
            tf.io.decode_csv,
            record_defaults=['', ''],
            field_delim='\t',
            use_quote_delim=False,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(lambda *ex: dict(zip(['question', 'answer'], ex)))
    return ds


def summarization_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['ringkasan: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('summarization_dataset')
t5.data.TaskRegistry.add(
    'summarization_dataset',
    dataset_fn=summarization_dataset,
    splits=['train'],
    text_preprocessor=[summarization_preprocessor],
    sentencepiece_model_path=vocab,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
)


def similarity_dataset(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/t5-data-v2/snli.tsv',
            'gs://mesolitica-tpu-general/t5-data-v2/mnli.tsv'
        ]
    )

    ds = ds.map(
        functools.partial(
            tf.io.decode_csv,
            record_defaults=['', ''],
            field_delim='\t',
            use_quote_delim=False,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(lambda *ex: dict(zip(['question', 'answer'], ex)))
    return ds


def similarity_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': ex['question'],
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('similarity_dataset')
t5.data.TaskRegistry.add(
    'similarity_dataset',
    dataset_fn=similarity_dataset,
    splits=['train'],
    text_preprocessor=[similarity_preprocessor],
    sentencepiece_model_path=vocab,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
)


def en_ms_dataset(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/t5-data-v2/en-ms.tsv'
        ]
    )

    ds = ds.map(
        functools.partial(
            tf.io.decode_csv,
            record_defaults=['', ''],
            field_delim='\t',
            use_quote_delim=False,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(lambda *ex: dict(zip(['question', 'answer'], ex)))
    return ds


def en_ms_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['terjemah Inggeris ke Melayu: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('en_ms_dataset')
t5.data.TaskRegistry.add(
    'en_ms_dataset',
    dataset_fn=en_ms_dataset,
    splits=['train'],
    text_preprocessor=[en_ms_preprocessor],
    sentencepiece_model_path=vocab,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
)


def ms_en_dataset(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/t5-data-v2/ms-en.tsv'
        ]
    )

    ds = ds.map(
        functools.partial(
            tf.io.decode_csv,
            record_defaults=['', ''],
            field_delim='\t',
            use_quote_delim=False,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(lambda *ex: dict(zip(['question', 'answer'], ex)))
    return ds


def ms_en_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['terjemah Melayu ke Inggeris: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('ms_en_dataset')
t5.data.TaskRegistry.add(
    'ms_en_dataset',
    dataset_fn=ms_en_dataset,
    splits=['train'],
    text_preprocessor=[ms_en_preprocessor],
    sentencepiece_model_path=vocab,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
)


def knowledge_graph_dataset(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/t5-data-v2/knowledge-graph.tsv'
        ]
    )

    ds = ds.map(
        functools.partial(
            tf.io.decode_csv,
            record_defaults=['', ''],
            field_delim='\t',
            use_quote_delim=False,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(lambda *ex: dict(zip(['question', 'answer'], ex)))
    return ds


def knowledge_graph_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['grafik pengetahuan: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('knowledge_graph_dataset')
t5.data.TaskRegistry.add(
    'knowledge_graph_dataset',
    dataset_fn=knowledge_graph_dataset,
    splits=['train'],
    text_preprocessor=[knowledge_graph_preprocessor],
    sentencepiece_model_path=vocab,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
)


def paraphrase_dataset(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/t5-data-v2/paraphrase.tsv'
        ]
    )

    ds = ds.map(
        functools.partial(
            tf.io.decode_csv,
            record_defaults=['', ''],
            field_delim='\t',
            use_quote_delim=False,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(lambda *ex: dict(zip(['question', 'answer'], ex)))
    return ds


def paraphrase_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['parafrasa: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('paraphrase_dataset')
t5.data.TaskRegistry.add(
    'paraphrase_dataset',
    dataset_fn=paraphrase_dataset,
    splits=['train'],
    text_preprocessor=[paraphrase_preprocessor],
    sentencepiece_model_path=vocab,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
)

t5.data.MixtureRegistry.remove('trivia_all_bahasa')
t5.data.MixtureRegistry.add(
    'trivia_all_bahasa',
    [
        'dumping_dataset',
        'question_dataset',
        'pair_dataset',
        'news_dataset',
        'summarization_dataset',
        'similarity_dataset',
        'en_ms_dataset',
        'ms_en_dataset',
        'knowledge_graph_dataset',
        'paraphrase_dataset'
    ],
    default_rate=1.0,
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
        model_dir='gs://mesolitica-tpu-general/t5-large-v2/',
        tpu=TPU_ADDRESS,
        tpu_topology=TPU_TOPOLOGY,
        model_parallelism=model_parallelism,
        batch_size=train_batch_size,
        sequence_length={'inputs': 512, 'targets': 512},
        learning_rate_schedule=0.001,
        save_checkpoints_steps=5000,
        keep_checkpoint_max=5,
        iterations_per_loop=100,
    )

    model.train(mixture_or_task_name='trivia_all_bahasa', steps=1000000)


if __name__ == '__main__':
    tf.app.run()
