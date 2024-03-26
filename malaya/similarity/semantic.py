from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Similarity

label = ['not similar', 'similar']

available_huggingface = {
    'mesolitica/finetune-mnli-nanot5-small': {
        'Size (MB)': 148,
        'macro precision': 0.87125,
        'macro recall': 0.87131,
        'macro f1-score': 0.87127,
    },
    'mesolitica/finetune-mnli-nanot5-base': {
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
    model: str = 'mesolitica/finetune-mnli-nanot5-small',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to calculate semantic similarity between 2 sentences.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-mnli-nanot5-small')
        Check available models at `malaya.similarity.semantic.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Similarity
    """

    return load(
        model=model,
        class_model=Similarity,
        available_huggingface=available_huggingface,
        path=__name__,
        **kwargs,
    )
