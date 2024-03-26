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
    'mesolitica/ner-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 84.7,
        'law': {
            'precision': 0.9642625081221572,
            'recall': 0.9598965071151359,
            'f1': 0.9620745542949757,
            'number': 1546
        },
        'person': {
            'precision': 0.9673319980661648,
            'recall': 0.971424608128728,
            'f1': 0.9693739834584906,
            'number': 14418
        },
        'time': {
            'precision': 0.9796992481203007,
            'recall': 0.983148893360161,
            'f1': 0.9814210394175245,
            'number': 3976
        },
        'location': {
            'precision': 0.966455899689208,
            'recall': 0.9753406878650227,
            'f1': 0.970877967379017,
            'number': 9246
        },
        'organization': {
            'precision': 0.9308265342319971,
            'recall': 0.9475204622051036,
            'f1': 0.9390993140471219,
            'number': 8308
        },
        'quantity': {
            'precision': 0.9824689554419284,
            'recall': 0.9853479853479854,
            'f1': 0.9839063643013899,
            'number': 2730
        },
        'event': {
            'precision': 0.8535980148883374,
            'recall': 0.8973913043478261,
            'f1': 0.8749470114455278,
            'number': 1150
        },
        'overall_precision': 0.9585080133195985,
        'overall_recall': 0.9670566055977183,
        'overall_f1': 0.9627633336140621,
        'overall_accuracy': 0.9951433495221682
    },
    'mesolitica/ner-t5-small-standard-bahasa-cased': {
        'Size (MB)': 141,
        'law': {
            'precision': 0.9320327249842668,
            'recall': 0.9579560155239327,
            'f1': 0.9448165869218501,
            'number': 1546
        },
        'person': {
            'precision': 0.9745341614906833,
            'recall': 0.9794007490636704,
            'f1': 0.976961394769614,
            'number': 14418
        },
        'time': {
            'precision': 0.9583539910758553,
            'recall': 0.9723340040241448,
            'f1': 0.9652933832709114,
            'number': 3976
        },
        'location': {
            'precision': 0.9709677419354839,
            'recall': 0.9766385463984426,
            'f1': 0.9737948883856357,
            'number': 9246
        },
        'organization': {
            'precision': 0.9493625210488333,
            'recall': 0.9500481463649495,
            'f1': 0.9497052099627,
            'number': 8308
        },
        'quantity': {
            'precision': 0.9823008849557522,
            'recall': 0.9758241758241758,
            'f1': 0.9790518191841234,
            'number': 2730
        },
        'event': {
            'precision': 0.8669991687448046,
            'recall': 0.9069565217391304,
            'f1': 0.88652783680408,
            'number': 1150
        },
        'overall_precision': 0.9629220498535133,
        'overall_recall': 0.9691593754531832,
        'overall_f1': 0.9660306446949986,
        'overall_accuracy': 0.9953954840983863
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
    model: str = 'mesolitica/ner-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to Entity Recognition.

    Parameters
    ----------
    model: str, optional (default='mesolitica/ner-t5-small-standard-bahasa-cased')
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
        class_model=Tagging,
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
