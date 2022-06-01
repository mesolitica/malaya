import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry
from tensor2tensor import problems
import tensorflow as tf
import os
import logging

left_dict = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    ' ': 4,
    '!': 5,
    '"': 6,
    "'": 7,
    '(': 8,
    ')': 9,
    '+': 10,
    ',': 11,
    '-': 12,
    '.': 13,
    '0': 14,
    '1': 15,
    '2': 16,
    '3': 17,
    '4': 18,
    '5': 19,
    '6': 20,
    '7': 21,
    '8': 22,
    '9': 23,
    ':': 24,
    ';': 25,
    '?': 26,
    'A': 27,
    'B': 28,
    'C': 29,
    'D': 30,
    'E': 31,
    'F': 32,
    'G': 33,
    'H': 34,
    'I': 35,
    'J': 36,
    'K': 37,
    'L': 38,
    'M': 39,
    'N': 40,
    'O': 41,
    'P': 42,
    'Q': 43,
    'R': 44,
    'S': 45,
    'T': 46,
    'U': 47,
    'V': 48,
    'W': 49,
    'X': 50,
    'Y': 51,
    'Z': 52,
    'a': 53,
    'b': 54,
    'c': 55,
    'd': 56,
    'e': 57,
    'f': 58,
    'g': 59,
    'h': 60,
    'i': 61,
    'j': 62,
    'k': 63,
    'l': 64,
    'm': 65,
    'n': 66,
    'o': 67,
    'p': 68,
    'q': 69,
    'r': 70,
    's': 71,
    't': 72,
    'u': 73,
    'v': 74,
    'w': 75,
    'x': 76,
    'y': 77,
    'z': 78,
    '،': 79,
    '؟': 80,
    'ء': 81,
    'آ': 82,
    'أ': 83,
    'ؤ': 84,
    'إ': 85,
    'ئ': 86,
    'ا': 87,
    'ب': 88,
    'ة': 89,
    'ت': 90,
    'ث': 91,
    'ج': 92,
    'ح': 93,
    'خ': 94,
    'د': 95,
    'ذ': 96,
    'ر': 97,
    'ز': 98,
    'س': 99,
    'ش': 100,
    'ص': 101,
    'ض': 102,
    'ط': 103,
    'ظ': 104,
    'ع': 105,
    'غ': 106,
    'ف': 107,
    'ق': 108,
    'ك': 109,
    'ل': 110,
    'م': 111,
    'ن': 112,
    'ه': 113,
    'و': 114,
    'ى': 115,
    'ي': 116,
    'ّ': 117,
    'ٓ': 118,
    '٠': 119,
    '١': 120,
    '٢': 121,
    '٣': 122,
    '٤': 123,
    '٥': 124,
    '٦': 125,
    '٧': 126,
    '٨': 127,
    '٩': 128,
    'چ': 129,
    'ڠ': 130,
    'ڤ': 131,
    'ڬ': 132,
    'ڽ': 133,
    'ۏ': 134,
    '﴾': 135,
    '﴿': 136
}
rev_left_dict = {v: k for k, v in left_dict.items()}

logger = logging.getLogger()
tf.logging.set_verbosity(tf.logging.DEBUG)


class Encoder:
    def __init__(self, dict):
        self.dict = dict
        self.vocab_size = len(self.dict)

    def encode(self, s):
        s = [left_dict[c] for c in s] + [1]
        return s

    def decode(self, ids):
        return ''.join([rev_left_dict[i] for i in ids if i > 3])


@registry.register_problem
class Jawi(text_problems.Text2TextProblem):
    @property
    def approx_vocab_size(self):
        return 32000

    @property
    def is_generate_per_split(self):
        return False

    @property
    def dataset_splits(self):
        return [
            {'split': problem.DatasetSplit.TRAIN, 'shards': 20},
            {'split': problem.DatasetSplit.EVAL, 'shards': 1},
        ]

    def feature_encoders(self, data_dir):
        encoder = Encoder(left_dict)
        return {'inputs': encoder, 'targets': encoder}


os.system('mkdir t2t-rumi-jawi/train-small')
DATA_DIR = os.path.expanduser('t2t-rumi-jawi/data')
TMP_DIR = os.path.expanduser('t2t-rumi-jawi/tmp')
TRAIN_DIR = os.path.expanduser('t2t-rumi-jawi/train-small')

PROBLEM = 'jawi'
t2t_problem = problems.problem(PROBLEM)

train_steps = 500000
eval_steps = 10
batch_size = 768
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
hparams.max_length = 128

RUN_CONFIG = create_run_config(
    model_dir=TRAIN_DIR,
    model_name=MODEL,
    save_checkpoints_steps=save_checkpoints_steps,
    num_gpus=0,
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
