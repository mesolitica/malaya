import tensorflow as tf
from malaya_boilerplate import frozen_graph


def load_automodel(model, model_class, huggingface_class=None, **kwargs):
    from malaya_boilerplate.utils import check_tf2_huggingface
    check_tf2_huggingface()

    try:
        from transformers import TFAutoModel, AutoTokenizer
    except BaseException:
        raise ModuleNotFoundError(
            'transformers not installed. Please install it by `pip3 install transformers` and try again.'
        )

    tokenizer = AutoTokenizer.from_pretrained(model)
    device = frozen_graph.get_device(**kwargs)
    with tf.device(device):
        if huggingface_class is None:
            huggingface_class = TFAutoModel
        model = huggingface_class.from_pretrained(model)
    return model_class(model=model, tokenizer=tokenizer, **kwargs)
