from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.model.tf import Seq2SeqLSTM, Seq2SeqLSTM_Split
from malaya.text.bpe import YTTMEncoder


def load_lstm(module, left_dict, right_dict, cleaning, split=False, quantized=False, **kwargs):
    path = check_file(
        file='lstm-bahdanau',
        module=module,
        keys={'model': 'model.pb'},
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)
    inputs = ['Placeholder']
    outputs = []
    input_nodes, output_nodes = nodes_session(
        g,
        inputs,
        outputs,
        extra={
            'greedy': 'import/decode_1/greedy:0',
            'beam': 'import/decode_2/beam:0',
        },
    )
    if split:
        model_class = Seq2SeqLSTM_Split
    else:
        model_class = Seq2SeqLSTM

    return model_class(
        input_nodes=input_nodes, output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        left_dict=left_dict,
        right_dict=right_dict,
        cleaning=cleaning,
    )


def load_lstm_yttm(module, vocab, model_class, quantized=False, tokenizer=None, **kwargs):
    path = check_file(
        file='lstm-bahdanau',
        module=module,
        keys={'model': 'model.pb', 'vocab': vocab},
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)
    inputs = ['Placeholder']
    outputs = []
    bpe = YTTMEncoder(vocab_file=path['vocab'], id_mode=True)
    input_nodes, output_nodes = nodes_session(
        g,
        inputs,
        outputs,
        extra={
            'greedy': 'import/decode_1/greedy:0',
            'beam': 'import/decode_2/beam:0',
        },
    )

    return model_class(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        bpe=bpe,
        tokenizer=tokenizer,
    )
