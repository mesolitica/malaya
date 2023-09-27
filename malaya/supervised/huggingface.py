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

    return class_model(
        model=model,
        from_lang=available_huggingface[model].get('from lang'),
        to_lang=available_huggingface[model].get('to lang'),
        **kwargs,
    )
