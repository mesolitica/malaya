import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import tensorflow_datasets as tfds
import t5
import functools
from t5 import models
from mesh_tensorflow.transformer import learning_rate_schedules
from glob import glob
import seqio

# !wget https://f000.backblazeb2.com/file/malaya-model/pretrained/t5-super-tiny-social-media-2021-11-15.tar.gz
# !tar -zxf t5-super-tiny-social-media-2021-11-15.tar.gz
# !rm t5-super-tiny-social-media-2021-11-15.tar.gz
# !wget https://f000.backblazeb2.com/file/malaya-model/bpe/sp10m.cased.ms-en.model
vocab = 'sp10m.cased.ms-en.model'

DEFAULT_SPM_PATH = vocab
DEFAULT_EXTRA_IDS = 100


def get_default_vocabulary():
    return seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS)


def translation_dataset(split, shuffle_files=False):
    del shuffle_files
    ds = tf.data.TextLineDataset(glob('t5-noisy-ms-en/*.tsv'))

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


def translation_preprocessor(ds):
    def to_inputs_and_targets(ex):
        return {
            'inputs': tf.strings.join(['terjemah Melayu ke Inggeris: ', ex['question']]),
            'targets': ex['answer'],
        }

    return ds.map(
        to_inputs_and_targets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


t5.data.TaskRegistry.remove('translation_dataset')

t5.data.TaskRegistry.add(
    'translation_dataset',
    dataset_fn=translation_dataset,
    splits=['train'],
    text_preprocessor=[translation_preprocessor],
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
    output_features=seqio.Feature(get_default_vocabulary())
)
t5.data.MixtureRegistry.remove('translation_bahasa')
t5.data.MixtureRegistry.add(
    'translation_bahasa',
    ['translation_dataset'],
    default_rate=1.0,
)

model_parallelism, train_batch_size, keep_checkpoint_max = 1, 52, 5
BASE_DIR = 't5-super-tiny-noisy-ms-en'
model = t5.models.MtfModel(
    model_dir=BASE_DIR,
    tpu=None,
    tpu_topology=None,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    mesh_shape='model:1,batch:1',
    mesh_devices=['gpu:0'],
    sequence_length={'inputs': 256, 'targets': 256},
    learning_rate_schedule=learning_rate_schedules.constant_learning_rate,
    save_checkpoints_steps=10000,
    keep_checkpoint_max=5,
    iterations_per_loop=100,
)

FINETUNE_STEPS = 1000000
MODEL_DIR = 't5-super-tiny-social-media'
model.finetune(
    mixture_or_task_name='translation_dataset',
    pretrained_model_dir=MODEL_DIR,
    finetune_steps=FINETUNE_STEPS,
)
