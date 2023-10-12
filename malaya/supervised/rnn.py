from malaya_boilerplate.huggingface import download_files


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

    s3_file = {'model': 'model.pt'}
    path = download_files(model, s3_file, **kwargs)

    return class_model(
        model=model,
        hidden_size=available_huggingface[model]['hidden size'],
        pth=path['model'],
        **kwargs,
    )
