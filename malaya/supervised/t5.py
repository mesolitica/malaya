from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.text.bpe import SentencePieceBatchEncoder
from malaya.path import MS_EN_BPE_MODEL, T2T_BPE_MODEL

VOCAB_MODEL = {'generator': T2T_BPE_MODEL}


def load(module, model, model_class, quantized=False, **kwargs):

    try:
        import tensorflow_text
    except BaseException:
        raise ModuleNotFoundError(
            'tensorflow-text not installed. Please install it by `pip install tensorflow-text` and try again. Also, make sure tensorflow-text version same as tensorflow version.'
        )

    path = check_file(
        file=model,
        module=module,
        keys={'model': 'model.pb', 'vocab': VOCAB_MODEL.get(module, MS_EN_BPE_MODEL)},
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], t5_graph=True, **kwargs)
    tokenizer = SentencePieceBatchEncoder(vocab_file=path['vocab'])
    inputs = ['inputs']
    outputs = []
    input_nodes, output_nodes = nodes_session(
        g, inputs, outputs, extra={'decode': 'import/SelectV2_3:0'}
    )

    return model_class(
        input_nodes=input_nodes, output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        tokenizer=tokenizer,
    )
