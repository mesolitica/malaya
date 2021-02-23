from malaya.function import check_file, load_graph, generate_session
from malaya.text.bpe import SentencePieceEncoder
from malaya.path import TRANSLATION_BPE_MODEL, T2T_BPE_MODEL


def load(module, model, model_class, maxlen, quantized = False, **kwargs):
    path = check_file(
        file = model,
        module = module,
        keys = {'model': 'model.pb', 'vocab': TRANSLATION_BPE_MODEL},
        quantized = quantized,
        **kwargs,
    )

    g = load_graph(path['model'], **kwargs)
    X = g.get_tensor_by_name('import/Placeholder:0')
    logits = g.get_tensor_by_name('import/logits:0')
    encoder = SentencePieceEncoder(path['vocab'])

    return model_class(
        X = X,
        greedy = logits,
        sess = generate_session(graph = g, **kwargs),
        encoder = encoder,
        maxlen = maxlen,
    )


def load_lm(module, model, model_class, maxlen, quantized = False, **kwargs):
    path = check_file(
        file = model,
        module = module,
        keys = {'model': 'model.pb', 'vocab': T2T_BPE_MODEL},
        quantized = quantized,
        **kwargs,
    )

    g = load_graph(path['model'], **kwargs)
    X = g.get_tensor_by_name('import/Placeholder:0')
    top_p = g.get_tensor_by_name('import/top_p:0')
    temperature = g.get_tensor_by_name('import/temperature:0')
    logits = g.get_tensor_by_name('import/logits:0')
    tokenizer = SentencePieceEncoder(path['vocab'])

    return model_class(
        X = X,
        top_p = top_p,
        temperature = temperature,
        greedy = logits,
        sess = generate_session(graph = g, **kwargs),
        tokenizer = tokenizer,
        maxlen = maxlen,
    )
