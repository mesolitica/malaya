import json
import re
from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.text.bpe import (
    sentencepiece_tokenizer_bert,
    sentencepiece_tokenizer_xlnet,
    AlbertTokenizer,
)
from malaya.model.bert import TaggingBERT
from malaya.model.xlnet import TaggingXLNET
from malaya.text.regex import _expressions
from malaya.path import MODEL_VOCAB, MODEL_BPE, TAGGING_SETTING


def transformer(class_name, model='xlnet', quantized=False, **kwargs):
    path = check_file(
        file=model,
        module=class_name,
        keys={
            'model': 'model.pb',
            'vocab': MODEL_VOCAB[model],
            'tokenizer': MODEL_BPE[model],
            'setting': TAGGING_SETTING[class_name],
        },
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    try:
        with open(path['setting']) as fopen:
            nodes = json.load(fopen)
    except BaseException:
        raise Exception(
            f"model corrupted due to some reasons, please run `malaya.clear_cache('{class_name}/{model}/{size}')` and try again"
        )

    if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
        if model in ['bert', 'tiny-bert']:
            tokenizer = sentencepiece_tokenizer_bert(
                path['tokenizer'], path['vocab']
            )

        if model in ['albert', 'tiny-albert']:
            tokenizer = AlbertTokenizer(
                vocab_file=path['vocab'], spm_model_file=path['tokenizer']
            )

        inputs = ['Placeholder', 'Placeholder_1']
        vectorizer = {'vectorizer': 'import/dense/BiasAdd:0'}
        Model = TaggingBERT

    if model in ['xlnet', 'alxlnet']:
        tokenizer = sentencepiece_tokenizer_xlnet(path['tokenizer'])
        inputs = ['Placeholder', 'Placeholder_1', 'Placeholder_2']
        vectorizer = {'vectorizer': 'import/transpose_3:0'}
        Model = TaggingXLNET

    outputs = ['logits']
    input_nodes, output_nodes = nodes_session(
        g, inputs, outputs, extra=vectorizer
    )

    return Model(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        tokenizer=tokenizer,
        settings=nodes,
    )


def transformer_ontonotes5(
    class_name, model='xlnet', quantized=False, **kwargs
):
    path = check_file(
        file=model,
        module=class_name,
        keys={
            'model': 'model.pb',
            'vocab': MODEL_VOCAB[model],
            'tokenizer': MODEL_BPE[model],
            'setting': TAGGING_SETTING[class_name],
        },
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

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

    try:
        with open(path['setting']) as fopen:
            nodes = json.load(fopen)
    except BaseException:
        raise Exception(
            f"model corrupted due to some reasons, please run `malaya.clear_cache('{class_name}/{model}/{size}')` and try again"
        )

    if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
        if model in ['bert', 'tiny-bert']:
            tokenizer = sentencepiece_tokenizer_bert(
                path['tokenizer'], path['vocab']
            )

        if model in ['albert', 'tiny-albert']:
            tokenizer = AlbertTokenizer(
                vocab_file=path['vocab'], spm_model_file=path['tokenizer']
            )

        inputs = ['Placeholder', 'Placeholder_1']
        vectorizer = {'vectorizer': 'import/dense/BiasAdd:0'}
        Model = TaggingBERT

    if model in ['xlnet', 'alxlnet']:
        tokenizer = sentencepiece_tokenizer_xlnet(path['tokenizer'])
        inputs = ['Placeholder', 'Placeholder_1', 'Placeholder_2']
        vectorizer = {'vectorizer': 'import/transpose_3:0'}
        Model = TaggingXLNET

    outputs = ['logits']
    input_nodes, output_nodes = nodes_session(
        g, inputs, outputs, extra=vectorizer
    )

    return Model(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        tokenizer=tokenizer,
        settings=nodes,
        tok=tok,
    )
