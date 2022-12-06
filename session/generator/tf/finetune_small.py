import tensorflow as tf
import tensorflow_datasets as tfds
import t5
import functools

"""
gcloud compute tpus create node-1 --zone=europe-west4-a --accelerator-type='v3-8' --version='1.15.3'
gcloud compute tpus create node-1 --zone=us-central1-f --accelerator-type='v2-8' --version='1.15.3'
"""

vocab = 'gs://mesolitica-tpu-general/sp10m.cased.t5.model'
tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
    'node-4', zone='us-central1-f', project='mesolitica-tpu'
)
TPU_ADDRESS = tpu.get_master()
TPU_TOPOLOGY = '2x2'
print(TPU_ADDRESS)


def artikel_dataset(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/generator/generator-artikel.tsv'
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


def artikel_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['artikel: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('artikel_dataset')
t5.data.TaskRegistry.add(
    'artikel_dataset',
    dataset_fn=artikel_dataset,
    splits=['train'],
    text_preprocessor=[artikel_preprocessor],
    sentencepiece_model_path=vocab,
    metric_fns=[t5.evaluation.metrics.accuracy],
)


def karangan_dataset(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/generator/generator-karangan.tsv'
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


def karangan_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['karangan: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('karangan_dataset')
t5.data.TaskRegistry.add(
    'karangan_dataset',
    dataset_fn=karangan_dataset,
    splits=['train'],
    text_preprocessor=[karangan_preprocessor],
    sentencepiece_model_path=vocab,
    metric_fns=[t5.evaluation.metrics.accuracy],
)


def penerangan_produk_dataset(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/generator/generator-penerangan-produk.tsv'
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


def penerangan_produk_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['penerangan-produk: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('penerangan_produk_dataset')
t5.data.TaskRegistry.add(
    'penerangan_produk_dataset',
    dataset_fn=penerangan_produk_dataset,
    splits=['train'],
    text_preprocessor=[penerangan_produk_preprocessor],
    sentencepiece_model_path=vocab,
    metric_fns=[t5.evaluation.metrics.accuracy],
)


def surat_khabar_dataset(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/generator/generator-surat-khabar.tsv'
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


def surat_khabar_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['surat-khabar: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('surat_khabar_dataset')
t5.data.TaskRegistry.add(
    'surat_khabar_dataset',
    dataset_fn=surat_khabar_dataset,
    splits=['train'],
    text_preprocessor=[surat_khabar_preprocessor],
    sentencepiece_model_path=vocab,
    metric_fns=[t5.evaluation.metrics.accuracy],
)


def tajuk_surat_khabar_dataset(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/generator/generator-tajuk-surat-khabar.tsv'
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


def tajuk_surat_khabar_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['tajuk-surat-khabar: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('tajuk_surat_khabar_dataset')
t5.data.TaskRegistry.add(
    'tajuk_surat_khabar_dataset',
    dataset_fn=tajuk_surat_khabar_dataset,
    splits=['train'],
    text_preprocessor=[tajuk_surat_khabar_preprocessor],
    sentencepiece_model_path=vocab,
    metric_fns=[t5.evaluation.metrics.accuracy],
)

t5.data.MixtureRegistry.remove('generator_bahasa')
t5.data.MixtureRegistry.add(
    'generator_bahasa',
    ['artikel_dataset',
     'karangan_dataset',
     'penerangan_produk_dataset',
     'surat_khabar_dataset',
     'tajuk_surat_khabar_dataset'],
    default_rate=1.0,
)


def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    model_parallelism, train_batch_size, keep_checkpoint_max = {
        'small': (1, 256, 16),
        'base': (2, 128, 8),
        'large': (8, 64, 4),
        '3B': (8, 16, 1),
        '11B': (8, 16, 1),
    }['small']

    BASE_DIR = 'gs://mesolitica-tpu-general/t5-small-generator'
    model = t5.models.MtfModel(
        model_dir=BASE_DIR,
        tpu=TPU_ADDRESS,
        tpu_topology=TPU_TOPOLOGY,
        model_parallelism=model_parallelism,
        batch_size=train_batch_size,
        sequence_length={'inputs': 1024, 'targets': 1024},
        learning_rate_schedule=0.003,
        save_checkpoints_steps=5000,
        keep_checkpoint_max=5,
        iterations_per_loop=100,
    )

    FINETUNE_STEPS = 500000
    MODEL_DIR = 'gs://mesolitica-tpu-general/t5-small-v2'

    model.finetune(
        mixture_or_task_name='generator_bahasa',
        pretrained_model_dir=MODEL_DIR,
        finetune_steps=FINETUNE_STEPS,
    )


if __name__ == '__main__':
    tf.app.run()
