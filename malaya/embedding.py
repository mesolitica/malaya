from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Embedding

openai_metrics = {
    'b.cari.com.my': {
        'positive score': 0.8729225971201091,
        'negative score': 0.27480777421889363,
        'top1': 0.31621790857858484,
        'top3': 0.6242955541640576,
        'top5': 0.6944270507201001,
        'top10': 0.7623669380087664,
    },
    'c.cari.com.my': {
        'positive score': 0.8173745331635356,
        'negative score': 0.3100609159718768,
        'top1': 0.08380430943129637,
        'top3': 0.21388202048746027,
        'top5': 0.27861179795125396,
        'top10': 0.3589720946661957,
    },
    'malay-news': {
        'positive score': 0.8448714707337686,
        'negative score': 0.2741472719191583,
        'top1': 0.1386895659334196,
        'top3': 0.2952593812492648,
        'top5': 0.3745441712739678,
        'top10': 0.4754734737089754,
    },
    'twitter': {
        'positive score': 0.8928321128367129,
        'negative score': 0.26488808270585834,
        'top1': 0.22942090082094518,
        'top3': 0.4919014865764367,
        'top5': 0.5930774351009541,
        'top10': 0.7248724206789439,
    },
}

available_huggingface = {
    'mesolitica/mistral-embedding-191m-8k-contrastive': {
        'Size (MB)': 334,
        'embedding size': 768,
        'Suggested length': 8192,
    },
    'mesolitica/mistral-embedding-349m-8k-contrastive': {
        'Size (MB)': 633,
        'embedding size': 768,
        'Suggested length': 8192,
    },
    'mesolitica/embedding-malaysian-mistral-64M-32k': {
        'Size (MB)': 96.5,
        'embedding size': 768,
        'Suggested length': 20480,
    }
}

info = """
Entire Malaysian embedding benchmark at https://huggingface.co/spaces/mesolitica/malaysian-embedding-leaderboard
"""


def huggingface(
    model: str = 'mesolitica/embedding-malaysian-mistral-64M-32k',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model for embedding task.

    Parameters
    ----------
    model: str, optional (default='mesolitica/embedding-malaysian-mistral-64M-32k')
        Check available models at `malaya.embedding.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Embedding
    """
    return load(
        model=model,
        class_model=Embedding,
        available_huggingface=available_huggingface,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
