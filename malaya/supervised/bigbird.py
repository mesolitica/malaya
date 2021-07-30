from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.text.bpe import SentencePieceEncoder, WordPieceTokenizer
from malaya.path import TRANSLATION_BPE_MODEL, PEGASUS_BPE_MODEL


def load(module, model, model_class, maxlen, quantized=False, **kwargs):
    path = check_file(
        file=model,
        module=module,
        keys={'model': 'model.pb', 'vocab': TRANSLATION_BPE_MODEL},
        quantized=quantized,
        **kwargs,
    )

    g = load_graph(path['model'], **kwargs)
    inputs = ['Placeholder']
    outputs = ['logits']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    encoder = SentencePieceEncoder(vocab_file=path['vocab'])

    return model_class(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        encoder=encoder,
        maxlen=maxlen,
    )


def load_pegasus(module, model, model_class, maxlen, quantized=False, **kwargs):
    path = check_file(
        file=model,
        module=module,
        keys={'model': 'model.pb', 'vocab': PEGASUS_BPE_MODEL},
        quantized=quantized,
        **kwargs,
    )

    g = load_graph(path['model'], **kwargs)

    inputs = ['Placeholder', 'top_p', 'temperature']
    outputs = ['logits']
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)
    tokenizer = WordPieceTokenizer(vocab_file=path['vocab'])

    return model_class(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        tokenizer=tokenizer,
        maxlen=maxlen,
    )
