import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry
from tensor2tensor import problems
import tensorflow as tf
import os
import logging
import sentencepiece as spm

vocab = 'truecase.model'
sp = spm.SentencePieceProcessor()
sp.Load(vocab)

logger = logging.getLogger()
tf.logging.set_verbosity(tf.logging.DEBUG)


class Encoder:
    def __init__(self, sp):
        self.sp = sp
        self.vocab_size = sp.GetPieceSize()

    def encode(self, s):
        return self.sp.EncodeAsIds(s) + [1]

    def decode(self, ids, strip_extraneous = False):
        return self.sp.DecodeIds(list(ids))


@registry.register_problem
class TrueCase(text_problems.Text2TextProblem):
    @property
    def approx_vocab_size(self):
        return 32000

    @property
    def is_generate_per_split(self):
        return False

    @property
    def dataset_splits(self):
        return [
            {'split': problem.DatasetSplit.TRAIN, 'shards': 500},
            {'split': problem.DatasetSplit.EVAL, 'shards': 1},
        ]

    def feature_encoders(self, data_dir):
        encoder = Encoder(sp)
        return {'inputs': encoder, 'targets': encoder}


os.system('mkdir t2t-true-case/train-base')
DATA_DIR = os.path.expanduser('t2t-true-case/data')
TMP_DIR = os.path.expanduser('t2t-true-case/tmp')
TRAIN_DIR = os.path.expanduser('t2t-true-case/train-base')

PROBLEM = 'true_case'
t2t_problem = problems.problem(PROBLEM)

train_steps = 500000
eval_steps = 20
batch_size = 4096 * 2
save_checkpoints_steps = 25000
ALPHA = 0.1
schedule = 'continuous_train_and_eval'
MODEL = 'transformer'
HPARAMS = 'transformer_base'

from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment
from tensor2tensor.utils.trainer_lib import create_hparams
from tensor2tensor.utils import registry
from tensor2tensor import models
from tensor2tensor import problems

hparams = create_hparams(HPARAMS)
hparams.batch_size = batch_size
hparams.learning_rate = ALPHA
hparams.max_length = 512
hparams.dropout = 0.2

RUN_CONFIG = create_run_config(
    model_dir = TRAIN_DIR,
    model_name = MODEL,
    save_checkpoints_steps = save_checkpoints_steps,
    num_gpus = 1,
)

tensorflow_exp_fn = create_experiment(
    run_config = RUN_CONFIG,
    hparams = hparams,
    model_name = MODEL,
    problem_name = PROBLEM,
    data_dir = DATA_DIR,
    train_steps = train_steps,
    eval_steps = eval_steps,
    # use_xla=True # For acceleration
)

tensorflow_exp_fn.train_and_evaluate()
