import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mesolitica-tpu.json'

import tensorflow as tf
import tensorflow_datasets as tfds
from t5.data import preprocessors as prep
import functools
import t5
import gin
import sentencepiece as spm
from glob import glob
import os
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

gin.parse_config_file('pretrained_models_base_operative_config.gin')
vocab = 'sp10m.cased.t5.model'
sp = spm.SentencePieceProcessor()
sp.Load(vocab)


def dumping_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(
        [
            'cleaned-news.txt.tsv',
            'dumping-parliament.txt.tsv',
            'filtered-dumping-academia.txt.tsv',
            'filtered-dumping-cleaned-common-crawl.txt.tsv',
            'filtered-dumping-wiki.txt.tsv',
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

from tqdm import tqdm


@registry.register_problem
class Seq2Seq(text_problems.Text2TextProblem):
    @property
    def approx_vocab_size(self):
        return 32100

    @property
    def is_generate_per_split(self):
        return False

    @property
    def dataset_splits(self):
        return [{'split': problem.DatasetSplit.TRAIN, 'shards': 50}]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        for k in range(10):
            nq_task = t5.data.TaskRegistry.get('dumping_dataset')
            ds = nq_task.get_dataset(
                split = 'qa.tsv',
                sequence_length = {'inputs': 768, 'targets': 768},
            )

            for ex in tqdm(tfds.as_numpy(ds)):
                yield ex

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):

        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        for sample in generator:
            sample['inputs'] = sample['inputs'].tolist()
            sample['targets'] = sample['targets'].tolist()
            yield sample


DATA_DIR = os.path.expanduser('t2t-dumping/data')
TMP_DIR = os.path.expanduser('t2t-dumping/tmp')

tf.gfile.MakeDirs(DATA_DIR)
tf.gfile.MakeDirs(TMP_DIR)

from tensor2tensor.utils import registry
from tensor2tensor import problems

PROBLEM = 'seq2_seq'
t2t_problem = problems.problem(PROBLEM)
t2t_problem.generate_data(DATA_DIR, TMP_DIR)
