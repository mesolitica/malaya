import inspect

additional_parameters = {
    'from_lang': 'from lang',
    'to_lang': 'to lang',
}


def load(
    model,
    class_model,
    available_huggingface,
    force_check: bool = True,
    path: str = __name__,
    **kwargs,
):
    if model not in available_huggingface and force_check:
        raise ValueError(
            f'model not supported, please check supported models from `{path}.available_huggingface`.'
        )

    args = inspect.getfullargspec(class_model)
    for k, v in additional_parameters.items():
        if k in args.args:
            kwargs[k] = available_huggingface[model].get(v)

    return class_model(
        model=model,
        **kwargs,
    )
