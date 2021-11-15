import tensorflow as tf
import tensorflow_datasets as tfds
import t5
import functools

vocab = 'gs://mesolitica-tpu-general/t5-data-v2/sp10m.cased.ms-en.model'
tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
    'node-2', zone='us-central1-f', project='mesolitica-tpu'
)
TPU_ADDRESS = tpu.get_master()
TPU_TOPOLOGY = '2x2'
print(TPU_ADDRESS)


def kesalahan_tatabahasa_dataset(split, shuffle_files=False):
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


def kesalahan_tatabahasa_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['kesalahan tatabahasa: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('kesalahan_tatabahasa_dataset')
t5.data.TaskRegistry.add(
    'kesalahan_tatabahasa_dataset',
    dataset_fn=kesalahan_tatabahasa_dataset,
    splits=['train'],
    text_preprocessor=[kesalahan_tatabahasa_preprocessor],
    sentencepiece_model_path=vocab,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
)

t5.data.MixtureRegistry.remove('kesalahan_tatabahasa_bahasa')
t5.data.MixtureRegistry.add(
    'kesalahan_tatabahasa_bahasa',
    ['kesalahan_tatabahasa_dataset'],
    default_rate=1.0,
)


def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    model_parallelism, train_batch_size, keep_checkpoint_max = 1, 256, 16

    BASE_DIR = 'gs://mesolitica-tpu-general/t5-small-kesalahan-tatabahasa'
    model = t5.models.MtfModel(
        model_dir=BASE_DIR,
        tpu=TPU_ADDRESS,
        tpu_topology=TPU_TOPOLOGY,
        model_parallelism=model_parallelism,
        batch_size=train_batch_size,
        sequence_length={'inputs': 512, 'targets': 512},
        learning_rate_schedule=0.0005,
        save_checkpoints_steps=10000,
        keep_checkpoint_max=5,
        iterations_per_loop=100,
    )

    FINETUNE_STEPS = 50000
    MODEL_DIR = 'gs://mesolitica-tpu-general/t5-small-v2'

    model.finetune(
        mixture_or_task_name='kesalahan_tatabahasa_bahasa',
        pretrained_model_dir=MODEL_DIR,
        finetune_steps=FINETUNE_STEPS,
    )


if __name__ == '__main__':
    tf.app.run()
