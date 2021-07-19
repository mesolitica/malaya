from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.text.bpe import SentencePieceTokenizer
from malaya.text.regex import _expressions
from malaya.model.bert import TaggingBERT
from malaya.model.xlnet import TaggingXLNET
from malaya.path import MODEL_VOCAB, MODEL_BPE, TAGGING_SETTING
import json
import re


def transformer(module, model='xlnet', quantized=False, tok=None, **kwargs):
    path = check_file(
        file=model,
        module=module,
        keys={
            'model': 'model.pb',
            'vocab': MODEL_VOCAB[model],
            'tokenizer': MODEL_BPE[model],
            'setting': TAGGING_SETTING[module],
        },
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    with open(path['setting']) as fopen:
        nodes = json.load(fopen)

    if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
        inputs = ['Placeholder', 'Placeholder_1']
        vectorizer = {'vectorizer': 'import/dense/BiasAdd:0'}
        selected_model = TaggingBERT

    if model in ['xlnet', 'alxlnet']:
        inputs = ['Placeholder', 'Placeholder_1', 'Placeholder_2']
        vectorizer = {'vectorizer': 'import/transpose_3:0'}
        selected_model = TaggingXLNET

    outputs = ['logits']
    tokenizer = SentencePieceTokenizer(vocab_file=path['vocab'], spm_model_file=path['tokenizer'])
    input_nodes, output_nodes = nodes_session(
        g, inputs, outputs, extra=vectorizer
    )

    return selected_model(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        tokenizer=tokenizer,
        settings=nodes,
        tok=tok
    )


def transformer_ontonotes5(
    module, model='xlnet', quantized=False, **kwargs
):

    hypen = r'\w+(?:-\w+)+'
    hypen_left = r'\w+(?: -\w+)+'
    hypen_right = r'\w+(?:- \w+)+'
    hypen_both = r'\w+(?: - \w+)+'

    pipeline = [
        hypen,
        hypen_left,
        hypen_right,
        hypen_both,
        _expressions['percent'],
        _expressions['money'],
        _expressions['time'],
        _expressions['date'],
        _expressions['repeat_puncts'],
        _expressions['number'],
        _expressions['word'],
    ]
    pipeline.append('(?:\\S)')
    compiled = re.compile(r'({})'.format('|'.join(pipeline)))

    def tok(string):
        tokens = compiled.findall(string)
        return [t[0] for t in tokens]

    return transformer(module=module, model=model, quantized=quantized, tok=tok, **kwargs)
