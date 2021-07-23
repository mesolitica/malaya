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
)
from malaya.text.t2t import text_encoder
from malaya.path import T2T_BPE_MODEL, LM_VOCAB

ENCODER_MODEL = {
    'subword': text_encoder.SubwordTextEncoder,
    'sentencepiece': SentencePieceBatchEncoder,
    'yttm': YTTMEncoder,
}


def load_lm(module, model, model_class, quantized=False, **kwargs):
    path = check_file(
        file=model,
        module=module,
        keys={'model': 'model.pb', 'vocab': T2T_BPE_MODEL},
        quantized=quantized,
        **kwargs,
    )

    g = load_graph(path['model'], **kwargs)
    tokenizer = SentencePieceEncoder(vocab_file=path['vocab'])

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
    encoder = ENCODER_MODEL[encoder](vocab_file=path['vocab'], id_mode=True)

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
    tokenizer = SentencePieceEncoder(vocab_file=path['vocab'])

    inputs = ['x_placeholder']
    outputs = ['greedy', 'tag_greedy']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return model_class(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        tokenizer=tokenizer,
    )
