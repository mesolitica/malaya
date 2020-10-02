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


def synonym_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(['synonyms.tsv'])

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

# nq_task = t5.data.TaskRegistry.get("synonym_dataset")
# ds = nq_task.get_dataset(split='qa.tsv', sequence_length={"inputs": 1024, "targets": 1024})


def stemming_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(['stemming.tsv'])

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

# nq_task = t5.data.TaskRegistry.get("stemming_dataset")
# ds = nq_task.get_dataset(split='qa.tsv', sequence_length={"inputs": 1024, "targets": 1024})


def pair_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(glob('*pair.tsv'))

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

# nq_task = t5.data.TaskRegistry.get("pair_dataset")
# ds = nq_task.get_dataset(split='qa.tsv', sequence_length={"inputs": 1024, "targets": 1024})


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

# nq_task = t5.data.TaskRegistry.get("dumping_dataset")
# ds = nq_task.get_dataset(split='qa.tsv', sequence_length={"inputs": 1024, "targets": 1024})


def news_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(['newstitle.tsv'])

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

# nq_task = t5.data.TaskRegistry.get("news_dataset")
# ds = nq_task.get_dataset(split='qa.tsv', sequence_length={"inputs": 1024, "targets": 1024})


def question_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(['qa.tsv'])

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

# nq_task = t5.data.TaskRegistry.get("question_dataset")
# ds = nq_task.get_dataset(split='qa.tsv', sequence_length={"inputs": 1024, "targets": 1024})


def similarity_dataset(split, shuffle_files = False):
    del shuffle_files
    ds = tf.data.TextLineDataset(['quora.tsv', 'mnli.tsv', 'snli.tsv'])

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

# nq_task = t5.data.TaskRegistry.get("similarity_dataset")
# ds = nq_task.get_dataset(split=file, sequence_length={"inputs": 1024, "targets": 1024})

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
        return [
            {'split': problem.DatasetSplit.TRAIN, 'shards': 200},
            {'split': problem.DatasetSplit.EVAL, 'shards': 1},
        ]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        datasets = [
            'synonym_dataset',
            'stemming_dataset',
            'pair_dataset',
            'question_dataset',
            'similarity_dataset',
            'news_dataset',
        ]

        for dataset in datasets:
            print(dataset)

            nq_task = t5.data.TaskRegistry.get(dataset)
            ds = nq_task.get_dataset(
                split = 'qa.tsv',
                sequence_length = {'inputs': 768, 'targets': 768},
            )

            for ex in tqdm(tfds.as_numpy(ds)):
                yield ex

        for k in range(5):
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


DATA_DIR = os.path.expanduser('t2t/data')
TMP_DIR = os.path.expanduser('t2t/tmp')

tf.gfile.MakeDirs(DATA_DIR)
tf.gfile.MakeDirs(TMP_DIR)

from tensor2tensor.utils import registry
from tensor2tensor import problems

PROBLEM = 'seq2_seq'
t2t_problem = problems.problem(PROBLEM)
t2t_problem.generate_data(DATA_DIR, TMP_DIR)
