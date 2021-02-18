from malaya.function import check_file, load_graph, generate_session
from malaya.text.bpe import SentencePieceEncoder, YTTMEncoder, load_yttm
from malaya.text.t2t import text_encoder
from malaya.path import T2T_BPE_MODEL, LM_VOCAB


def load_lm(module, model, model_class, quantized = False, **kwargs):
    path = check_file(
        file = model,
        module = module,
        keys = {'model': 'model.pb', 'vocab': T2T_BPE_MODEL},
        quantized = quantized,
        **kwargs,
    )

    g = load_graph(path['model'], **kwargs)
    X = g.get_tensor_by_name('import/Placeholder:0')
    top_p = g.get_tensor_by_name('import/Placeholder_2:0')
    greedy = g.get_tensor_by_name('import/greedy:0')
    beam = g.get_tensor_by_name('import/beam:0')
    nucleus = g.get_tensor_by_name('import/nucleus:0')

    tokenizer = SentencePieceEncoder(path['vocab'])

    return model_class(
        X = X,
        top_p = top_p,
        greedy = greedy,
        beam = beam,
        nucleus = nucleus,
        sess = generate_session(graph = g, **kwargs),
        tokenizer = tokenizer,
    )


def load(module, model, encoder, model_class, quantized = False, **kwargs):

    path = check_file(
        file = model,
        module = module,
        keys = {'model': 'model.pb', 'vocab': LM_VOCAB[module]},
        quantized = quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    if encoder == 'subword':
        encoder = text_encoder.SubwordTextEncoder(path['vocab'])

    if encoder == 'yttm':
        bpe, subword_mode = load_yttm(path['vocab'], True)
        encoder = YTTMEncoder(bpe, subword_mode)

    return model_class(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        greedy = g.get_tensor_by_name('import/greedy:0'),
        beam = g.get_tensor_by_name('import/beam:0'),
        sess = generate_session(graph = g, **kwargs),
        encoder = encoder,
    )


def load_tatabahasa(module, model, model_class, quantized = False, **kwargs):
    path = check_file(
        file = model,
        module = module,
        keys = {'model': 'model.pb', 'vocab': T2T_BPE_MODEL},
        quantized = quantized,
        **kwargs,
    )

    g = load_graph(path['model'], **kwargs)
    tokenizer = SentencePieceEncoder(path['vocab'])

    return model_class(
        X = g.get_tensor_by_name('import/x_placeholder:0'),
        greedy = g.get_tensor_by_name('import/greedy:0'),
        tag_greedy = g.get_tensor_by_name('import/tag_greedy:0'),
        sess = generate_session(graph = g, **kwargs),
        tokenizer = tokenizer,
    )
