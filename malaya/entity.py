from malaya.text.entity import EntityRegex

label = {
    'PAD': 0,
    'X': 1,
    'OTHER': 2,
    'law': 3,
    'location': 4,
    'organization': 5,
    'person': 6,
    'quantity': 7,
    'time': 8,
    'event': 9,
}

available_huggingface = {
    'bert': {
        'Size (MB)': 425.4,
        'Quantized Size (MB)': 111,
        'macro precision': 0.99291,
        'macro recall': 0.97864,
        'macro f1-score': 0.98537,
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
    model: str = 'mesolitica/ner-nanot5-base-malaysian-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to Entity Recognition.

    Parameters
    ----------
    model: str, optional (default='mesolitica/ner-analysis-nanot5-base-malaysian-cased')
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
