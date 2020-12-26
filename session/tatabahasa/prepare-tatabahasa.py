import pickle
import sentencepiece as spm
import json
from glob import glob
import os
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
from tensor2tensor.layers import modalities
import tensorflow as tf

vocab = 'sp10m.cased.t5.model'
sp = spm.SentencePieceProcessor()
sp.Load(vocab)


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
        'Description': 'kesalahan kata kerja tak transitif',
        'salah': 'Dia kata kepada saya',
        'betul': 'Dia berkata kepada saya',
    },
    {
        'class': 17,
        'Description': 'kesalahan kata kerja transitif',
        'salah': 'Dia suka baca buku',
        'betul': 'Dia suka membaca buku',
    },
    {
        'class': 18,
        'Description': 'penggunaan kata yang tidak tepat',
        'salah': 'Tembuk Besar negeri Cina dibina oleh Shih Huang Ti.',
        'betul': 'Tembok Besar negeri Cina dibina oleh Shih Huang Ti',
    },
    {
        'class': 19,
        'Description': 'kesalahan frasa kerja tak transitif',
        'salah': 'berdasarkan pada keterangan ini',
        'betul': 'berdasarkan keterangan ini',
    },
    {
        'class': 20,
        'Description': 'kesalahan frasa kerja transitif',
        'salah': 'Dia membeli banyak buah',
        'betul': 'Dia banyak membeli buah',
    },
    {
        'class': 21,
        'Description': 'kesalahan frasa kerja pasif',
        'salah': 'Surat itu saga akan balas',
        'betul': 'Surat itu akan saga balas',
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


def get_xy(row, encoder):
    x, y, tag, before, after = [], [], [], [], []

    for i in range(len(row[0])):
        t = encoder.encode(row[0][i][0])
        x.extend(t)
        t = encoder.encode(row[1][0][i][0])
        y.extend(t)
        tag.extend([row[1][0][i][1]] * len(t))
        before.extend([row[1][1][i] + 1] * len(t))
        after.extend([row[1][2][i] + 1] * len(t))

    # EOS
    x.append(1)
    y.append(1)
    tag.append(0)
    before.append(row[1][1][-1] + 1)
    after.append(row[1][2][-1] + 1)

    return x, y, tag, before, after


@modalities.is_pointwise
def pointer_top(body_output, targets, model_hparams, vocab_size):
    """Like identity_top() with is_pointwise annotation."""
    del targets, model_hparams, vocab_size  # unused arg
    return body_output


def pointer_bottom(x, model_hparams, vocab_size):
    """Like identity_bottom() without converting to float."""
    del model_hparams, vocab_size  # unused arg
    return x


@registry.register_problem
class Seq2editsGec(text_problems.Text2TextProblem):
    """Seq2Edits for grammatical error correction."""

    def feature_encoders(self, data_dir):
        encoder = Encoder(sp)
        t = Tatabahasa()
        return {'inputs': encoder, 'targets': encoder, 'targets_error_tag': t}

    def hparams(self, defaults, model_hparams):
        super(Seq2editsGec, self).hparams(defaults, model_hparams)

        for pointer_feat in ['targets_start_token', 'targets_end_token']:
            defaults.modality[pointer_feat] = modalities.ModalityType.IDENTITY
            defaults.vocab_size[pointer_feat] = None
            model_hparams.bottom[pointer_feat] = pointer_bottom
            model_hparams.top[pointer_feat] = pointer_top
        # Whether to use tags.
        if 'use_error_tags' not in model_hparams:
            model_hparams.add_hparam('use_error_tags', True)
        # If true, span and tag prediction is in the middle of the decoder layer
        # stack. Otherwise, they are at the end of the decoder layer stack.
        if 'middle_prediction' not in model_hparams:
            model_hparams.add_hparam('middle_prediction', True)
        # If middle_prediction=True, divide num_decoder_layers by this to get the
        # number of layers before and after the middle prediction.
        if 'middle_prediction_layer_factor' not in model_hparams:
            model_hparams.add_hparam('middle_prediction_layer_factor', 2)
        # Number of feedforward layers between prediction layers in the cascade.
        if 'ffn_in_prediction_cascade' not in model_hparams:
            model_hparams.add_hparam('ffn_in_prediction_cascade', 1)
        # Embedding size for error tags.
        if 'error_tag_embed_size' not in model_hparams:
            model_hparams.add_hparam('error_tag_embed_size', 6)
        if model_hparams.use_error_tags:
            defaults.modality[
                'targets_error_tag'
            ] = modalities.ModalityType.SYMBOL
            error_tag_vocab_size = self._encoders[
                'targets_error_tag'
            ].vocab_size
            defaults.vocab_size['targets_error_tag'] = error_tag_vocab_size
            model_hparams.top['targets_error_tag'] = pointer_top

    def example_reading_spec(self):
        data_fields, _ = super(Seq2editsGec, self).example_reading_spec()
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

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        with open('../pure-text/dataset-tatabahasa.pkl', 'rb') as fopen:
            data = pickle.load(fopen)

        encoder = Encoder(sp)
        for row in tqdm(data):
            x, y, tag, before, after = get_xy(row, encoder)
            yield {'inputs': x, 'targets': y, 'targets_error_tag': tag}

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):

        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        for sample in generator:
            yield sample
