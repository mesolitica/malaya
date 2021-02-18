from malaya.function import check_file, load_graph, generate_session


def load(module, model, model_class, maxlen, quantized = False, **kwargs):
    path = check_file(
        file = model,
        module = module,
        keys = {'model': 'model.pb', 'vocab': T2T_BPE_MODEL},
        quantized = quantized,
        **kwargs,
    )
