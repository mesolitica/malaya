import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.layers import common_attention
from tensor2tensor.utils import registry
from tensor2tensor import problems
import tensorflow as tf
import os
import logging
import sentencepiece as spm
import transformer_tag
from tensor2tensor.layers import modalities

vocab = 'sp10m.cased.t5.model'
sp = spm.SentencePieceProcessor()
sp.Load(vocab)

logger = logging.getLogger()
tf.logging.set_verbosity(tf.logging.DEBUG)


class Encoder:
    def __init__(self, sp):
        self.sp = sp
        self.vocab_size = sp.GetPieceSize() + 100

    def encode(self, s):
        return self.sp.EncodeAsIds(s)

    def decode(self, ids, strip_extraneous = False):
        return self.sp.DecodeIds(list(ids))


d = [
    {'class': 0, 'Description': 'PAD', 'salah': '', 'betul': ''},
    {
        'class': 1,
        'Description': 'kesambungan subwords',
        'salah': '',
        'betul': '',
    },
    {'class': 2, 'Description': 'tiada kesalahan', 'salah': '', 'betul': ''},
    {
        'class': 3,
        'Description': 'kesalahan frasa nama, Perkara yang diterangkan mesti mendahului "penerang"',
        'salah': 'Cili sos',
        'betul': 'sos cili',
    },
    {
        'class': 4,
        'Description': 'kesalahan kata jamak',
        'salah': 'mereka-mereka',
        'betul': 'mereka',
    },
    {
        'class': 5,
        'Description': 'kesalahan kata penguat',
        'salah': 'sangat tinggi sekali',
        'betul': 'sangat tinggi',
    },
    {
        'class': 6,
        'Description': 'kata adjektif dan imbuhan "ter" tanpa penguat.',
        'salah': 'Sani mendapat markah yang tertinggi sekali.',
        'betul': 'Sani mendapat markah yang tertinggi.',
    },
    {
        'class': 7,
        'Description': 'kesalahan kata hubung',
        'salah': 'Sally sedang membaca bila saya tiba di rumahnya.',
        'betul': 'Sally sedang membaca apabila saya tiba di rumahnya.',
    },
    {
        'class': 8,
        'Description': 'kesalahan kata bilangan',
        'salah': 'Beribu peniaga tidak membayar cukai pendapatan.',
        'betul': 'Beribu-ribu peniaga tidak membayar cukai pendapatan',
    },
    {
        'class': 9,
        'Description': 'kesalahan kata sendi',
        'salah': 'Umar telah berpindah daripada sekolah ini bulan lalu.',
        'betul': 'Umar telah berpindah dari sekolah ini bulan lalu.',
    },
    {
        'class': 10,
        'Description': 'kesalahan penjodoh bilangan',
        'salah': 'Setiap orang pelajar',
        'betul': 'Setiap pelajar.',
    },
    {
        'class': 11,
        'Description': 'kesalahan kata ganti diri',
        'salah': 'Pencuri itu telah ditangkap. Beliau dibawa ke balai polis.',
        'betul': 'Pencuri itu telah ditangkap. Dia dibawa ke balai polis.',
    },
    {
        'class': 12,
        'Description': 'kesalahan ayat pasif',
        'salah': 'Cerpen itu telah dikarang oleh saya.',
        'betul': 'Cerpen itu telah saya karang.',
    },
    {
        'class': 13,
        'Description': 'kesalahan kata tanya',
        'salah': 'Kamu berasal dari manakah ?',
        'betul': 'Kamu berasal dari mana ?',
    },
    {
        'class': 14,
        'Description': 'kesalahan tanda baca',
        'salah': 'Kamu berasal dari manakah .',
        'betul': 'Kamu berasal dari mana ?',
    },
    {
        'class': 15,
        'Description': 'kesalahan kata kerja tak transitif',
        'salah': 'Dia kata kepada saya',
        'betul': 'Dia berkata kepada saya',
    },
    {
        'class': 16,
        'Description': 'kesalahan kata kerja transitif',
        'salah': 'Dia suka baca buku',
        'betul': 'Dia suka membaca buku',
    },
    {
        'class': 17,
        'Description': 'penggunaan kata yang tidak tepat',
        'salah': 'Tembuk Besar negeri Cina dibina oleh Shih Huang Ti.',
        'betul': 'Tembok Besar negeri Cina dibina oleh Shih Huang Ti',
    },
]


class Tatabahasa:
    def __init__(self, d):
        self.d = d
        self.kesalahan = {i['Description']: no for no, i in enumerate(self.d)}
        self.reverse_kesalahan = {v: k for k, v in self.kesalahan.items()}
        self.vocab_size = len(self.d)

    def encode(self, s):
        return [self.kesalahan[i] for i in s]

    def decode(self, ids, strip_extraneous = False):
        return [self.reverse_kesalahan[i] for i in ids]


@registry.register_problem
class Grammar(text_problems.Text2TextProblem):
    """grammatical error correction."""

    def feature_encoders(self, data_dir):
        encoder = Encoder(sp)
        t = Tatabahasa(d)
        return {'inputs': encoder, 'targets': encoder, 'targets_error_tag': t}

    def hparams(self, defaults, model_hparams):
        super(Grammar, self).hparams(defaults, model_hparams)
        if 'use_error_tags' not in model_hparams:
            model_hparams.add_hparam('use_error_tags', True)
        if 'middle_prediction' not in model_hparams:
            model_hparams.add_hparam('middle_prediction', False)
        if 'middle_prediction_layer_factor' not in model_hparams:
            model_hparams.add_hparam('middle_prediction_layer_factor', 2)
        if 'ffn_in_prediction_cascade' not in model_hparams:
            model_hparams.add_hparam('ffn_in_prediction_cascade', 1)
        if 'error_tag_embed_size' not in model_hparams:
            model_hparams.add_hparam('error_tag_embed_size', 12)
        if model_hparams.use_error_tags:
            defaults.modality[
                'targets_error_tag'
            ] = modalities.ModalityType.SYMBOL
            error_tag_vocab_size = self._encoders[
                'targets_error_tag'
            ].vocab_size
            defaults.vocab_size['targets_error_tag'] = error_tag_vocab_size

    def example_reading_spec(self):
        data_fields, _ = super(Grammar, self).example_reading_spec()
        data_fields['targets_error_tag'] = tf.VarLenFeature(tf.int64)
        return data_fields, None

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


DATA_DIR = os.path.expanduser('t2t-tatabahasa/data')
TMP_DIR = os.path.expanduser('t2t-tatabahasa/tmp')
TRAIN_DIR = os.path.expanduser('t2t-tatabahasa/train-base')

PROBLEM = 'grammar'
t2t_problem = problems.problem(PROBLEM)

train_steps = 200000
eval_steps = 20
batch_size = 1024 * 4
save_checkpoints_steps = 10000
ALPHA = 0.0005
schedule = 'continuous_train_and_eval'
MODEL = 'transformer_tag'
HPARAMS = 'transformer_base'

from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment
from tensor2tensor.utils.trainer_lib import create_hparams
from tensor2tensor.utils import registry
from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.utils import trainer_lib


hparams = create_hparams(HPARAMS)
hparams.batch_size = batch_size
hparams.learning_rate = ALPHA
hparams.train_batch_size = batch_size

hparams.filter_size = 3072
hparams.hidden_size = 768
hparams.num_heads = 12
hparams.num_hidden_layers = 8
hparams.vocab_divisor = 128
hparams.dropout = 0.1
hparams.max_length = 256

# LM
hparams.label_smoothing = 0.0
hparams.shared_embedding_and_softmax_weights = False
hparams.eval_drop_long_sequences = True
hparams.max_length = 256
hparams.multiproblem_mixing_schedule = 'pretrain'

# tpu
hparams.symbol_modality_num_shards = 1
hparams.attention_dropout_broadcast_dims = '0,1'
hparams.relu_dropout_broadcast_dims = '1'
hparams.layer_prepostprocess_dropout_broadcast_dims = '1'

hparams.optimizer = 'Adafactor'
hparams.learning_rate_warmup_steps = 10000
hparams.learning_rate_schedule = 'rsqrt_decay'
hparams.warm_start_from_second = 'base-tatabahasa/model.ckpt'

print(hparams)

RUN_CONFIG = create_run_config(
    model_dir = TRAIN_DIR,
    model_name = MODEL,
    save_checkpoints_steps = save_checkpoints_steps,
    num_gpus = 3,
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

tensorflow_exp_fn.train()
