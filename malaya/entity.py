from ._utils import _tag_class
from ._utils._paths import PATH_ENTITIES, S3_PATH_ENTITIES


def available_deep_model():
    """
    List available deep learning entities models, ['concat', 'bahdanau', 'luong']
    """
    return ['concat', 'bahdanau', 'luong']


def available_bert_model():
    """
    List available bert entities models, ['multilanguage', 'base', 'small']
    """
    return ['multilanguage', 'base', 'small']


def deep_model(model = 'bahdanau', validate = True):
    """
    Load deep learning NER model.

    Parameters
    ----------
    model : str, optional (default='bahdanau')
        Model architecture supported. Allowed values:

        * ``'concat'`` - Concating character and word embedded for BiLSTM.
        * ``'bahdanau'`` - Concating character and word embedded including Bahdanau Attention for BiLSTM.
        * ``'luong'`` - Concating character and word embedded including Luong Attention for BiLSTM.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    TAGGING: malaya._models._tensorflow_model.TAGGING class
    """

    if not isinstance(model, str):
        raise ValueError('model must be a string')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')

    model = model.lower()
    if model not in available_deep_model():
        raise Exception(
            'model not supported, please check supported models from malaya.entity.available_deep_model()'
        )

    return _tag_class.deep_model(
        PATH_ENTITIES,
        S3_PATH_ENTITIES,
        'entity',
        model = model,
        validate = validate,
    )


def bert(model = 'base', validate = True):

    """
    Load BERT NER model.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'multilanguage'`` - bert multilanguage released by Google, trained on NER.
        * ``'base'`` - base bert-bahasa released by Malaya, trained on NER.
        * ``'small'`` - small bert-bahasa released by Malaya, trained on NER.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    TAGGING_BERT: malaya._models._tensorflow_model.TAGGING_BERT class
    """

    if not isinstance(model, str):
        raise ValueError('model must be a string')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')

    model = model.lower()
    if model not in available_bert_model():
        raise Exception(
            'model not supported, please check supported models from malaya.entity.available_bert_model()'
        )

    return _tag_class.bert(
        PATH_ENTITIES,
        S3_PATH_ENTITIES,
        'entity',
        model = model,
        validate = validate,
    )
