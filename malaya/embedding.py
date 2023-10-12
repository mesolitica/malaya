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
    'mesolitica/nanot5-base-embedding': {
        'Size (MB)': 2170,
        'embedding size': 1024,
        'Suggested length': 1024,
        'b.cari.com.my': {
            'positive score': 0.8928321128367129,
            'negative score': 0.26488808270585834,
            'top1': 0.22942090082094518,
            'top3': 0.4919014865764367,
            'top5': 0.5930774351009541,
            'top10': 0.7248724206789439,
        },
        'c.cari.com.my': {
            'positive score': 0.8928321128367129,
            'negative score': 0.26488808270585834,
            'top1': 0.22942090082094518,
            'top3': 0.4919014865764367,
            'top5': 0.5930774351009541,
            'top10': 0.7248724206789439,
        },
        'malay-news': {
            'positive score': 0.8928321128367129,
            'negative score': 0.26488808270585834,
            'top1': 0.22942090082094518,
            'top3': 0.4919014865764367,
            'top5': 0.5930774351009541,
            'top10': 0.7248724206789439,
        },
        'twitter': {
            'positive score': 0.8928321128367129,
            'negative score': 0.26488808270585834,
            'top1': 0.22942090082094518,
            'top3': 0.4919014865764367,
            'top5': 0.5930774351009541,
            'top10': 0.7248724206789439,
        },
    },
    'mesolitica/llama2-embedding-600m-8k': {
        'Size (MB)': 2170,
        'embedding size': 1536,
        'Suggested length': 32768,
        'b.cari.com.my': {
            'positive score': 0.79568475,
            'negative score': 0.6981619672232329,
            'top1': 0.3168440826549781,
            'top3': 0.6881653099561679,
            'top5': 0.7789605510331872,
            'top10': 0.8453350031308704,
        },
        'c.cari.com.my': {
            'positive score': 0.71944785,
            'negative score': 0.7663808533028701,
            'top1': 0.08327446132108796,
            'top3': 0.18730130695867184,
            'top5': 0.23975626986930412,
            'top10': 0.3140233133168492,
        },
        'malay-news': {
            'positive score': 0.71082395,
            'negative score': 0.7160432709481884,
            'top1': 0.14268909540054112,
            'top3': 0.27584990001176335,
            'top5': 0.3640748147276791,
            'top10': 0.47112104458299026,
        },
        'twitter': {
            'positive score': 0.8202477,
            'negative score': 0.7034184992996264,
            'top1': 0.23496782782338585,
            'top3': 0.5200798757488352,
            'top5': 0.6416685156423342,
            'top10': 0.785888617705791,
        },
    },
    'mesolitica/llama2-embedding-1b-8k': {
        'Size (MB)': 3790,
        'embedding size': 1536,
        'Suggested length': 32768,
        'b.cari.com.my': {
            'positive score': 0.82174283,
            'negative score': 0.7068604469821633,
            'top1': 0.32623669380087666,
            'top3': 0.6947401377582968,
            'top5': 0.7902316844082655,
            'top10': 0.8603631809643081,
        },
        'c.cari.com.my': {
            'positive score': 0.74685395,
            'negative score': 0.7647963317331168,
            'top1': 0.08689509007417874,
            'top3': 0.19966442953020133,
            'top5': 0.26086188625927237,
            'top10': 0.3430766513599435,
        },
        'malay-news': {
            'positive score': 0.7159956,
            'negative score': 0.776366058746266,
            'top1': 0.14610045876955652,
            'top3': 0.285260557581461,
            'top5': 0.3640748147276791,
            'top10': 0.4775908716621574,
        },
        'twitter': {
            'positive score': 0.8326124,
            'negative score': 0.748791925041112,
            'top1': 0.23053028622143332,
            'top3': 0.5342800088750832,
            'top5': 0.6498779676059463,
            'top10': 0.7925449301087197,
        },
    },
}

info = """
score only based on positive pairs. https://huggingface.co/datasets/mesolitica/embedding-pair-mining
"""


def huggingface(
    model: str = 'mesolitica/nanot5-small-embedding',
    **kwargs,
):
    """
    Load HuggingFace model to translate.

    Parameters
    ----------
    model: str, optional (default='mesolitica/nanot5-small-embedding')
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
