from malaya.function import check_file, load_graph, generate_session


def load(module, model, model_class, quantized = False, **kwargs):

    try:
        import tensorflow_text
        import tf_sentencepiece
    except:
        raise ModuleNotFoundError(
            'tensorflow-text and tf-sentencepiece not installed. Please install it by `pip install tensorflow-text==1.15.0 tf-sentencepiece==0.1.86` and try again. Also, make sure tensorflow-text version same as tensorflow version.'
        )

    path = check_file(
        path = model,
        module = module,
        keys = {'model': 'model.pb'},
        quantized = quantized,
        **kwargs,
    )
    g = load_graph(path[model]['model'], **kwargs)
    X = g.get_tensor_by_name('import/inputs:0')
    decode = g.get_tensor_by_name(
        'import/SentenceTokenizer_1/SentenceTokenizer/SentencepieceDetokenizeOp:0'
    )
    sess = generate_session(graph = g, **kwargs)
    pred = None

    return model_class(X = X, decode = decode, sess = sess, pred = pred)
