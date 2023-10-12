from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Similarity

label = ['not similar', 'similar']

available_huggingface = {
    'mesolitica/finetune-mnli-t5-super-tiny-standard-bahasa-cased': {
        'Size (MB)': 50.7,
        'macro precision': 0.74562,
        'macro recall': 0.74574,
        'macro f1-score': 0.74501,
    },
    'mesolitica/finetune-mnli-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'macro precision': 0.76584,
        'macro recall': 0.76565,
        'macro f1-score': 0.76542,
    },
    'mesolitica/finetune-mnli-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'macro precision': 0.78067,
        'macro recall': 0.78063,
        'macro f1-score': 0.78010,
    },
    'mesolitica/finetune-mnli-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'macro precision': 0.78903,
        'macro recall': 0.79064,
        'macro f1-score': 0.78918,
    },
}


info = """
tested on matched dev set translated MNLI, https://huggingface.co/datasets/mesolitica/translated-MNLI
""".strip()


def huggingface(
    model: str = 'mesolitica/finetune-mnli-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to calculate semantic similarity between 2 sentences.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-mnli-t5-small-standard-bahasa-cased')
        Check available models at `malaya.similarity.semantic.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Similarity
    """

    if model not in available_huggingface and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.similarity.semantic.available_huggingface`.'
        )

    return load(
        model=model,
        class_model=Similarity,
        available_huggingface=available_huggingface,
        path=__name__,
        **kwargs,
    )
