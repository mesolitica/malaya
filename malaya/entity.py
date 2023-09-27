from malaya.text.entity import EntityRegex
from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Tagging

label = {
    'OTHER': 0,
    'law': 1,
    'location': 2,
    'organization': 3,
    'person': 4,
    'quantity': 5,
    'time': 6,
    'event': 7,
}

available_huggingface = {
    'mesolitica/ner-nanot5-small-malaysian-cased': {
        'Size (MB)': 167,
        'overall_precision': 0.858053849787435,
        'overall_recall': 0.8780876879199497,
        'overall_f1': 0.8679551807344053,
        'overall_accuracy': 0.9828597446341846
    },
    'mesolitica/ner-nanot5-base-malaysian-cased': {
        'Size (MB)': 167,
        'overall_precision': 0.9583735288442987,
        'overall_recall': 0.9604582588098806,
        'overall_f1': 0.9594147613414135,
        'overall_accuracy': 0.9942963731787561
    },
}

describe = [
    {'Tag': 'OTHER', 'Description': 'other'},
    {
        'Tag': 'law',
        'Description': 'law, regulation, related law documents, documents, etc',
    },
    {'Tag': 'location', 'Description': 'location, place'},
    {
        'Tag': 'organization',
        'Description': 'organization, company, government, facilities, etc',
    },
    {
        'Tag': 'person',
        'Description': 'person, group of people, believes, unique arts (eg; food, drink), etc',
    },
    {'Tag': 'quantity', 'Description': 'numbers, quantity'},
    {'Tag': 'time', 'Description': 'date, day, time, etc'},
    {'Tag': 'event', 'Description': 'unique event happened, etc'},
]


def huggingface(
    model: str = 'mesolitica/ner-nanot5-small-malaysian-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to Entity Recognition.

    Parameters
    ----------
    model: str, optional (default='mesolitica/ner-analysis-nanot5-small-malaysian-cased')
        Check available models at `malaya.entity.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Tagging
    """

    return load(
        model=model,
        class_model=Classification,
        available_huggingface=available_huggingface,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )


def general_entity(model=None):
    """
    Load Regex based general entities tagging along with another supervised entity tagging model.

    Parameters
    ----------
    model: object
        model must have `predict` method.
        Make sure the `predict` method returned [(string, label), (string, label)].

    Returns
    -------
    result: malaya.text.entity.EntityRegex class
    """
    if not hasattr(model, 'predict') and model is not None:
        raise ValueError('model must have `predict` method')
    return EntityRegex(model=model)
