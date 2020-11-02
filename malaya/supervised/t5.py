from malaya.function import check_file, load_graph, generate_session
import tensorflow as tf
import os


def load(
    path,
    s3_path,
    model,
    model_class,
    compressed = True,
    quantized = False,
    **kwargs
):

    try:
        import tensorflow_text
        import tf_sentencepiece
    except:
        raise ModuleNotFoundError(
            'tensorflow-text and tf-sentencepiece not installed. Please install it by `pip install tensorflow-text==1.15.0 tf-sentencepiece==0.1.86` and try again. Also, make sure tensorflow-text version same as tensorflow version.'
        )

    if compressed and not quantized:
        path = path['t5-compressed']
        s3_path = s3_path['t5-compressed']
        check_file(path[model]['model'], s3_path[model], **kwargs)

        if not os.path.exists(path[model]['directory'] + 'saved_model.pb'):
            import tarfile

            with tarfile.open(path[model]['model']['model']) as tar:
                tar.extractall(path = path[model]['path'])

        X = None
        decode = None
        sess = generate_session(graph = None, **kwargs)
        meta_graph_def = tf.compat.v1.saved_model.load(
            sess, ['serve'], path[model]['directory']
        )
        signature_def = meta_graph_def.signature_def['serving_default']
        pred = lambda x: sess.run(
            fetches = signature_def.outputs['outputs'].name,
            feed_dict = {signature_def.inputs['input'].name: x},
        )

    else:
        path = path['t5']
        s3_path = s3_path['t5']
        check_file(
            path[model],
            s3_path[model],
            quantized = quantized,
            optimized = True,
            **kwargs
        )
        if quantized:
            model_path = 'quantized'
        else:
            model_path = 'model'
        g = load_graph(path[model][model_path], **kwargs)
        X = g.get_tensor_by_name('import/inputs:0')
        decode = g.get_tensor_by_name(
            'import/SentenceTokenizer_1/SentenceTokenizer/SentencepieceDetokenizeOp:0'
        )
        sess = generate_session(graph = g, **kwargs)
        pred = None

    return model_class(X = X, decode = decode, sess = sess, pred = pred)
