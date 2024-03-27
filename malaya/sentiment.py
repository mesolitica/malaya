from malaya.supervised import classification
from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Classification
from malaya.path import PATH_SENTIMENT, S3_PATH_SENTIMENT

label = ['negative', 'neutral', 'positive']

available_huggingface = {
    'mesolitica/sentiment-analysis-nanot5-tiny-malaysian-cased': {
        'Size (MB)': 93,
        'macro precision': 0.67768,
        'macro recall': 0.68266,
        'macro f1-score': 0.67997,
    },
    'mesolitica/sentiment-analysis-nanot5-small-malaysian-cased': {
        'Size (MB)': 167,
        'macro precision': 0.67602,
        'macro recall': 0.67120,
        'macro f1-score': 0.67339,
    }
}

info = """
Trained on https://huggingface.co/datasets/mesolitica/chatgpt-explain-sentiment
Split 90% to train, 10% to test.
""".strip()


def multinomial(**kwargs):
    """
    Load multinomial sentiment model.

    Returns
    -------
    result : malaya.model.ml.Bayes class
    """
    return classification.multinomial(
        path=PATH_SENTIMENT,
        s3_path=S3_PATH_SENTIMENT,
        module='sentiment',
        label=label,
        **kwargs
    )


def huggingface(
    model: str = 'mesolitica/sentiment-analysis-nanot5-small-malaysian-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to classify sentiment.

    Parameters
    ----------
    model: str, optional (default='mesolitica/sentiment-analysis-nanot5-small-malaysian-cased')
        Check available models at `malaya.sentiment.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Classification
    """

    return load(
        model=model,
        class_model=Classification,
        available_huggingface=available_huggingface,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
