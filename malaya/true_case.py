from malaya.model.tf import TRUE_CASE
from malaya.path import PATH_TRUE_CASE, S3_PATH_TRUE_CASE
from malaya.supervised import transformer as load_transformer
from herpetologist import check_type

_transformer_availability = {
    'small': {'Size (MB)': 42.7, 'Sequence Accuracy': 0.142},
    'base': {'Size (MB)': 234, 'Sequence Accuracy': 0.696},
}


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(_transformer_availability)


@check_type
def transformer(model: str = 'base', **kwargs):
    model = model.lower()
    if model not in _transformer_availability:
        raise Exception(
            'model not supported, please check supported models from malaya.true_case.available_transformer()'
        )

    return load_transformer.load(
        PATH_TRUE_CAS, S3_PATH_TRUE_CASE, model, 'yttm', TRUE_CASE
    )
