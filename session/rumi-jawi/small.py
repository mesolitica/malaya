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
import youtokentome as yttm

bpe = yttm.BPE(model='rumi-jawi.yttm')

logger = logging.getLogger()
tf.logging.set_verbosity(tf.logging.DEBUG)


class Encoder:
    def __init__(self, bpe):
        self.bpe = bpe
        self.vocab_size = len(self.bpe.vocab())

    def encode(self, s):
        s = self.bpe.encode(s, output_type=yttm.OutputType.ID)
        s = [i + [1] for i in s]
        return s

    def decode(self, ids, strip_extraneous=False):
        return self.bpe.decode(list(ids))[0]


@registry.register_problem
class Jawi(text_problems.Text2TextProblem):
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

    def feature_encoders(self, data_dir):
        s_encoder = Encoder(bpe)
        return {'inputs': s_encoder, 'targets': s_encoder}


os.system('mkdir t2t-rumi-jawi/small')
DATA_DIR = os.path.expanduser('t2t-rumi-jawi/data')
TMP_DIR = os.path.expanduser('t2t-rumi-jawi/tmp')
TRAIN_DIR = os.path.expanduser('t2t-rumi-jawi/small')

PROBLEM = 'jawi'
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
