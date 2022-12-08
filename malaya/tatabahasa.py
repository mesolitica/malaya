from malaya.supervised import t5 as t5_load
from malaya.supervised import transformer as load_transformer
from malaya.model.tf import Tatabahasa
from malaya.model.t5 import Tatabahasa as T5_Tatabahasa
from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
from herpetologist import check_type
import logging
import warnings

logger = logging.getLogger(__name__)

_transformer_availability = {
    'small': {
        'Size (MB)': 397,
        'Quantized Size (MB)': 100,
        'exactly-match': 0.7891067538,
        'f1': 0.96789741,
        'exactly-match-tags': 0.88366013,
        'f1-tags': 0.983178559,
        'Suggested length': 256,
    },
    'base': {
        'Size (MB)': 875,
        'Quantized Size (MB)': 220,
        'exactly-match': 0.880656788,
        'f1': 0.978962508,
        'exactly-match-tags': 0.9144973968,
        'f1-tags': 0.9839226957,
        'Suggested length': 256,
    },
}

_huggingface_availability = {
    'mesolitica/finetune-tatabahasa-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'exactly-match': 0.7665198237,
        'f1': 0.970908229,
        'exactly-match-tags': 0.87404885863,
        'f1-tags': 0.9878723587,
        'Suggested length': 256,
    },
    'mesolitica/finetune-tatabahasa-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'exactly-match': 0.8145774929,
        'f1': 0.97812780,
        'exactly-match-tags': 0.89447336,
        'f1-tags': 0.990597377,
        'Suggested length': 256,
    },
    'mesolitica/finetune-tatabahasa-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'exactly-match': 0.760913095,
        'f1': 0.970136249,
        'exactly-match-tags': 0.86583900,
        'f1-tags': 0.9868999035,
        'Suggested length': 256,
    },
}


def describe_tagging():
    """
    Describe kesalahan tatabahasa supported.
    Full description at https://tatabahasabm.tripod.com/tata/salahtata.htm
    """
    d = [{'class': 0,
          'Description': 'PAD',
          'salah': '',
          'betul': ''},
         {'class': 1,
          'Description': 'kesambungan subwords',
          'salah': '',
          'betul': '',
          },
         {'class': 2,
         'Description': 'tiada kesalahan',
          'salah': '',
          'betul': '',
          },
         {'class': 3,
         'Description': 'kesalahan frasa nama, Perkara yang diterangkan mesti mendahului "penerang"',
          'salah': 'Cili sos',
          'betul': 'sos cili',
          },
         {'class': 4,
         'Description': 'kesalahan kata jamak',
          'salah': 'mereka-mereka',
          'betul': 'mereka',
          },
         {'class': 5,
         'Description': 'kesalahan kata penguat',
          'salah': 'sangat tinggi sekali',
          'betul': 'sangat tinggi',
          },
         {'class': 6,
         'Description': 'kata adjektif dan imbuhan "ter" tanpa penguat.',
          'salah': 'Sani mendapat markah yang tertinggi sekali.',
          'betul': 'Sani mendapat markah yang tertinggi.',
          },
         {'class': 7,
         'Description': 'kesalahan kata hubung',
          'salah': 'Sally sedang membaca bila saya tiba di rumahnya.',
          'betul': 'Sally sedang membaca apabila saya tiba di rumahnya.',
          },
         {'class': 8,
         'Description': 'kesalahan kata bilangan',
          'salah': 'Beribu peniaga tidak membayar cukai pendapatan.',
          'betul': 'Beribu-ribu peniaga tidak membayar cukai pendapatan',
          },
         {'class': 9,
         'Description': 'kesalahan kata sendi',
          'salah': 'Umar telah berpindah daripada sekolah ini bulan lalu.',
          'betul': 'Umar telah berpindah dari sekolah ini bulan lalu.',
          },
         {'class': 10,
         'Description': 'kesalahan penjodoh bilangan',
          'salah': 'Setiap orang pelajar',
          'betul': 'Setiap pelajar.',
          },
         {'class': 11,
         'Description': 'kesalahan kata ganti diri',
          'salah': 'Pencuri itu telah ditangkap. Beliau dibawa ke balai polis.',
          'betul': 'Pencuri itu telah ditangkap. Dia dibawa ke balai polis.',
          },
         {'class': 12,
         'Description': 'kesalahan ayat pasif',
          'salah': 'Cerpen itu telah dikarang oleh saya.',
          'betul': 'Cerpen itu telah saya karang.',
          },
         {'class': 13,
         'Description': 'kesalahan kata tanya',
          'salah': 'Kamu berasal dari manakah ?',
          'betul': 'Kamu berasal dari mana ?',
          },
         {'class': 14,
         'Description': 'kesalahan tanda baca',
          'salah': 'Kamu berasal dari manakah .',
          'betul': 'Kamu berasal dari mana ?',
          },
         {'class': 15,
         'Description': 'kesalahan kata kerja tak transitif',
          'salah': 'Dia kata kepada saya',
          'betul': 'Dia berkata kepada saya',
          },
         {'class': 16,
         'Description': 'kesalahan kata kerja transitif',
          'salah': 'Dia suka baca buku',
          'betul': 'Dia suka membaca buku',
          },
         {'class': 17,
         'Description': 'penggunaan kata yang tidak tepat',
          'salah': 'Tembuk Besar negeri Cina dibina oleh Shih Huang Ti.',
          'betul': 'Tembok Besar negeri Cina dibina oleh Shih Huang Ti',
          },
         ]

    return describe_availability(d, transpose=False)


def _describe():
    logger.info('tested on 5k generated dataset at https://f000.backblazeb2.com/file/malay-dataset/tatabahasa/test-set-tatabahasa.pkl')


def available_transformer():
    """
    List available transformer tagging models.
    """
    warnings.warn(
        '`malaya.tatabahasa.available_transformer` is deprecated, use `malaya.tatabahasa.available_huggingface` instead',
        DeprecationWarning)
    _describe()

    return describe_availability(_transformer_availability)


def available_huggingface():
    """
    List available huggingface models.
    """
    _describe()

    return describe_availability(_huggingface_availability)


@check_type
def transformer(model: str = 'base', quantized: bool = False, **kwargs):
    """
    Load Malaya transformer encoder-decoder + tagging model to correct a `kesalahan tatabahasa` text.

    Parameters
    ----------
    model: str, optional (default='base')
        Check available models at `malaya.tatabahasa.available_transformer()`.
    quantized: bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.tf.Tatabahasa class
    """

    warnings.warn(
        '`malaya.tatabahasa.transformer` is deprecated, use `malaya.tatabahasa.huggingface` instead',
        DeprecationWarning)

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.tatabahasa.available_transformer()`.'
        )

    return load_transformer.load_tatabahasa(
        module='kesalahan-tatabahasa',
        model=model,
        model_class=Tatabahasa,
        quantized=quantized,
        **kwargs
    )


@check_type
def huggingface(
    model: str = 'mesolitica/finetune-tatabahasa-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to fix kesalahan tatabahasa.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-tatabahasa-t5-small-standard-bahasa-cased')
        Check available models at `malaya.tatabahasa.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Tatabahasa
    """
    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.tatabahasa.available_huggingface()`.'
        )
    return load_huggingface.load_tatabahasa(model=model, initial_text='kesalahan tatabahasa:', **kwargs)
