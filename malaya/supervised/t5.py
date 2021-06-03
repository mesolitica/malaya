from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)


def load(module, model, model_class, quantized=False, **kwargs):

    try:
        import tensorflow_text
    except BaseException:
        raise ModuleNotFoundError(
            'tensorflow-text not installed. Please install it by `pip install tensorflow-text==1.15.1` and try again. Also, make sure tensorflow-text version same as tensorflow version.'
        )

    path = check_file(
        file=model,
        module=module,
        keys={'model': 'model.pb'},
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)
    sess = generate_session(graph=g, **kwargs)
    inputs = ['inputs']
    outputs = []
    extra = 'import/SentenceTokenizer_1/SentenceTokenizer/SentencepieceDetokenizeOp:0'
    input_nodes, output_nodes = nodes_session(
        g, inputs, outputs, extra={'decode': extra}
    )

    return model_class(
        input_nodes=input_nodes, output_nodes=output_nodes, sess=sess
    )
