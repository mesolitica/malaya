import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry
from tensor2tensor import problems
from malaya.text.t2t import text_encoder
import tensorflow as tf
import os
import logging
import sentencepiece as spm

encoder = text_encoder.SubwordTextEncoder('ms-en.subwords')

logger = logging.getLogger()
tf.logging.set_verbosity(tf.logging.DEBUG)


class Encoder:
    def __init__(self, encoder):
        self.encoder = encoder
        self.vocab_size = encoder.vocab_size

    def encode(self, s):
        s = [self.encoder.encode(s_) for s_ in s]
        s = [i + [1] for i in s]
        return s

    def decode(self, ids, strip_extraneous=False):
        return self.encoder.decode(ids)


@registry.register_problem
class Translation(text_problems.Text2TextProblem):
    @property
    def approx_vocab_size(self):
        return encoder.vocab_size

    @property
    def is_generate_per_split(self):
        return False

    @property
    def dataset_splits(self):
        return [
            {'split': problem.DatasetSplit.TRAIN, 'shards': 100},
            {'split': problem.DatasetSplit.EVAL, 'shards': 1},
        ]

    def feature_encoders(self, data_dir):
        s_encoder = Encoder(encoder)
        return {'inputs': s_encoder, 'targets': s_encoder}


os.system('mkdir t2t-noisy-ms-en/small')
DATA_DIR = os.path.expanduser('t2t-noisy-ms-en/data')
TMP_DIR = os.path.expanduser('t2t-noisy-ms-en/tmp')
TRAIN_DIR = os.path.expanduser('t2t-noisy-ms-en/small')

PROBLEM = 'translation'
t2t_problem = problems.problem(PROBLEM)

train_steps = 1000000
eval_steps = 20
batch_size = 4096 * 4
save_checkpoints_steps = 25000
ALPHA = 0.1
schedule = 'continuous_train_and_eval'
MODEL = 'transformer'
HPARAMS = 'transformer_small'

from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment
from tensor2tensor.utils.trainer_lib import create_hparams
from tensor2tensor.utils import registry
from tensor2tensor import models
from tensor2tensor import problems

hparams = create_hparams(HPARAMS)
hparams.batch_size = batch_size
hparams.learning_rate = ALPHA
hparams.max_length = 256

RUN_CONFIG = create_run_config(
    model_dir=TRAIN_DIR,
    model_name=MODEL,
    save_checkpoints_steps=save_checkpoints_steps,
    num_gpus=1,
)

tensorflow_exp_fn = create_experiment(
    run_config=RUN_CONFIG,
    hparams=hparams,
    model_name=MODEL,
    problem_name=PROBLEM,
    data_dir=DATA_DIR,
    train_steps=train_steps,
    eval_steps=eval_steps,
    # use_xla=True # For acceleration
)

tensorflow_exp_fn.train_and_evaluate()
