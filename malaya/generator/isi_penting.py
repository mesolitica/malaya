from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import IsiPentingGenerator

available_huggingface = {
    'mesolitica/finetune-isi-penting-generator-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'ROUGE-1': 0.24620333,
        'ROUGE-2': 0.05896076,
        'ROUGE-L': 0.15158954,
        'Suggested length': 1024,
    },
    'mesolitica/finetune-isi-penting-generator-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'ROUGE-1': 0.24620333,
        'ROUGE-2': 0.05896076,
        'ROUGE-L': 0.15158954,
        'Suggested length': 1024,
    },
}

info = """
tested on semisupervised summarization on unseen AstroAwani 20 news, https://github.com/huseinzol05/malay-dataset/tree/master/summarization/semisupervised-astroawani
each news compared ROUGE with 5 different generated texts.
"""


def huggingface(
    model: str = 'mesolitica/finetune-isi-penting-generator-t5-base-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to generate text based on isi penting.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-isi-penting-generator-t5-base-standard-bahasa-cased')
        Check available models at `malaya.generator.isi_penting.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.IsiPentingGenerator
    """
    if model not in available_huggingface and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.generator.isi_penting.available_huggingface`.'
        )
    return load(
        model=model,
        class_model=IsiPentingGenerator,
        available_huggingface=available_huggingface,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
