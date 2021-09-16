import tensorflow as tf
import tensorflow_datasets as tfds
import t5
import functools

vocab = 'gs://mesolitica-tpu-general/t5-data-v2/sp10m.cased.ms-en.model'
tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
    'node-10', zone='us-central1-f', project='mesolitica-tpu'
)
TPU_ADDRESS = tpu.get_master()
TPU_TOPOLOGY = '2x2'
print(TPU_ADDRESS)


def spelling_dataset(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'gs://mesolitica-tpu-general/t5-data-v2/spelling-correction-news.tsv',
            'gs://mesolitica-tpu-general/t5-data-v2/spelling-correction-wiki.tsv'
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


def spelling_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['betulkan ejaan: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('spelling_dataset')
t5.data.TaskRegistry.add(
    'spelling_dataset',
    dataset_fn=spelling_dataset,
    splits=['train'],
    text_preprocessor=[spelling_preprocessor],
    sentencepiece_model_path=vocab,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
)

t5.data.MixtureRegistry.remove('spelling_bahasa')
t5.data.MixtureRegistry.add(
    'spelling_bahasa',
    ['spelling_dataset'],
    default_rate=1.0,
)


def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    model_parallelism, train_batch_size, keep_checkpoint_max = 1, 1024, 16

    BASE_DIR = 'gs://mesolitica-tpu-general/t5-tiny-spelling'
    model = t5.models.MtfModel(
        model_dir=BASE_DIR,
        tpu=TPU_ADDRESS,
        tpu_topology=TPU_TOPOLOGY,
        model_parallelism=model_parallelism,
        batch_size=train_batch_size,
        sequence_length={'inputs': 256, 'targets': 256},
        learning_rate_schedule=0.0005,
        save_checkpoints_steps=10000,
        keep_checkpoint_max=5,
        iterations_per_loop=100,
    )

    FINETUNE_STEPS = 50000
    MODEL_DIR = 'gs://mesolitica-tpu-general/t5-tiny-v2'

    model.finetune(
        mixture_or_task_name='spelling_bahasa',
        pretrained_model_dir=MODEL_DIR,
        finetune_steps=FINETUNE_STEPS,
    )


if __name__ == '__main__':
    tf.app.run()
