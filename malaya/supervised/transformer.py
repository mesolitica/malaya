from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.text.bpe import (
    SentencePieceEncoder,
    SentencePieceBatchEncoder,
    YTTMEncoder,
    load_yttm,
)
from malaya.text.t2t import text_encoder
from malaya.path import T2T_BPE_MODEL, LM_VOCAB


def load_lm(module, model, model_class, quantized=False, **kwargs):
    path = check_file(
        file=model,
        module=module,
        keys={'model': 'model.pb', 'vocab': T2T_BPE_MODEL},
        quantized=quantized,
        **kwargs,
    )

    g = load_graph(path['model'], **kwargs)
    X = g.get_tensor_by_name('import/Placeholder:0')
    top_p = g.get_tensor_by_name('import/Placeholder_2:0')
    greedy = g.get_tensor_by_name('import/greedy:0')
    beam = g.get_tensor_by_name('import/beam:0')
    nucleus = g.get_tensor_by_name('import/nucleus:0')

    tokenizer = SentencePieceEncoder(path['vocab'])

    inputs = ['Placeholder', 'Placeholder_2']
    outputs = ['greedy', 'beam', 'nucleus']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return model_class(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        tokenizer=tokenizer,
    )


def load(module, model, encoder, model_class, quantized=False, **kwargs):

    path = check_file(
        file=model,
        module=module,
        keys={'model': 'model.pb', 'vocab': LM_VOCAB[module]},
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    if encoder == 'subword':
        encoder = text_encoder.SubwordTextEncoder(path['vocab'])

    if encoder == 'yttm':
        bpe, subword_mode = load_yttm(path['vocab'], True)
        encoder = YTTMEncoder(bpe, subword_mode)

    if encoder == 'sentencepiece':
        encoder = SentencePieceBatchEncoder(path['vocab'])

    inputs = ['Placeholder']
    outputs = ['greedy', 'beam']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return model_class(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        encoder=encoder,
    )


def load_tatabahasa(module, model, model_class, quantized=False, **kwargs):
    path = check_file(
        file=model,
        module=module,
        keys={'model': 'model.pb', 'vocab': T2T_BPE_MODEL},
        quantized=quantized,
        **kwargs,
    )

    g = load_graph(path['model'], **kwargs)
    tokenizer = SentencePieceEncoder(path['vocab'])

    inputs = ['x_placeholder']
    outputs = ['greedy', 'tag_greedy']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return model_class(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        tokenizer=tokenizer,
    )
