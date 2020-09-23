import json
import youtokentome as yttm

files = [
    'results-news.json',
    'results-wiki.json',
    'results-news-single.json',
    'results-wiki-single.json',
]

bpe = yttm.BPE(model = 'truecase.yttm')


class Encoder:
    def __init__(self, bpe):
        self.bpe = bpe
        self.vocab_size = len(self.bpe.vocab())

    def encode(self, s):
        s = self.bpe.encode(s, output_type = yttm.OutputType.ID)
        s = [i + [1] for i in s]
        return s

    def decode(self, ids, strip_extraneous = False):
        return self.bpe.decode(list(ids))[0]


encoder = Encoder(bpe)

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
from tqdm import tqdm


@registry.register_problem
class TrueCase(text_problems.Text2TextProblem):
    @property
    def approx_vocab_size(self):
        return 32000

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
        return [
            {'split': problem.DatasetSplit.TRAIN, 'shards': 200},
            {'split': problem.DatasetSplit.EVAL, 'shards': 1},
        ]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):

        for file in files:
            with open(file) as fopen:
                data = json.load(fopen)

            for i in tqdm(range(len(data))):
                i, o = encoder.encode([data[i][0], data[i][1]])
                yield {'inputs': i, 'targets': o}

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):

        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        for sample in generator:
            yield sample


import os
import tensorflow as tf

os.system('rm -rf t2t-true-case/data')
DATA_DIR = os.path.expanduser('t2t-true-case/data')
TMP_DIR = os.path.expanduser('t2t-true-case/tmp')

tf.gfile.MakeDirs(DATA_DIR)
tf.gfile.MakeDirs(TMP_DIR)

from tensor2tensor.utils import registry
from tensor2tensor import problems

PROBLEM = 'true_case'
t2t_problem = problems.problem(PROBLEM)
t2t_problem.generate_data(DATA_DIR, TMP_DIR)
