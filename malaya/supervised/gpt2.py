from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.model.tf import GPT2
from malaya.text.bpe import GPT2Encoder
from malaya.path import GPT2_ENCODER, GPT2_VOCAB
import json


def load(model, quantized=False, **kwargs):
    path = check_file(
        file=model,
        module='gpt2',
        keys={'model': 'model.pb', 'encoder': GPT2_ENCODER, 'vocab': GPT2_VOCAB},
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    with open(path['encoder']) as f:
        en = json.load(f)
    with open(path['vocab'], encoding='utf-8') as f:
        bpe_data = f.read()

    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    encoder = GPT2Encoder(
        encoder=en,
        bpe_merges=bpe_merges,
    )
    inputs = ['X', 'temp', 'top_k', 'top_p', 'maxlen', 'n_samples']
    outputs = ['output']
    input_nodes, output_nodes = nodes_session(
        g, inputs, outputs
    )

    return GPT2(
        input_nodes=input_nodes, output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        encoder=encoder,
    )
