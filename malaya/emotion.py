from malaya.supervised import classification
from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Classification
from malaya.path import PATH_EMOTION, S3_PATH_EMOTION

label = ['anger', 'fear', 'happy', 'love', 'sadness', 'surprise']

available_huggingface = {
    'mesolitica/emotion-analysis-nanot5-tiny-malaysian-cased': {
        'Size (MB)': 93,
        'macro precision': 0.96107,
        'macro recall': 0.96270,
        'macro f1-score': 0.96182,
    },
    'mesolitica/emotion-analysis-nanot5-small-malaysian-cased': {
        'Size (MB)': 167,
        'macro precision': 0.96814,
        'macro recall': 0.97004,
        'macro f1-score': 0.96905,
    }
}

info = """
Trained on https://github.com/huseinzol05/malaysian-dataset/tree/master/corpus/emotion
Split 80% to train, 20% to test.
""".strip()


def multinomial(**kwargs):
    """
    Load multinomial emotion model.

    Returns
    -------
    result: malaya.model.ml.MulticlassBayes class
    """
    return classification.multinomial(
        path=PATH_EMOTION,
        s3_path=S3_PATH_EMOTION,
        module='emotion',
        label=label,
        **kwargs
    )


def huggingface(
    model: str = 'mesolitica/emotion-analysis-nanot5-small-malaysian-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to classify emotion.

    Parameters
    ----------
    model: str, optional (default='mesolitica/emotion-analysis-nanot5-small-malaysian-cased')
        Check available models at `malaya.emotion.available_huggingface`.
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
