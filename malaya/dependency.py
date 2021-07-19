from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.text.bpe import SentencePieceTokenizer
from malaya.function.parse_dependency import DependencyGraph
from malaya.model.xlnet import DependencyXLNET
from malaya.model.bert import DependencyBERT
from malaya.path import MODEL_VOCAB, MODEL_BPE
from herpetologist import check_type

label = {
    'PAD': 0,
    'X': 1,
    'nsubj': 2,
    'cop': 3,
    'det': 4,
    'root': 5,
    'nsubj:pass': 6,
    'acl': 7,
    'case': 8,
    'obl': 9,
    'flat': 10,
    'punct': 11,
    'appos': 12,
    'amod': 13,
    'compound': 14,
    'advmod': 15,
    'cc': 16,
    'obj': 17,
    'conj': 18,
    'mark': 19,
    'advcl': 20,
    'nmod': 21,
    'nummod': 22,
    'dep': 23,
    'xcomp': 24,
    'ccomp': 25,
    'parataxis': 26,
    'compound:plur': 27,
    'fixed': 28,
    'aux': 29,
    'csubj': 30,
    'iobj': 31,
    'csubj:pass': 32,
}

_transformer_availability_v1 = {
    'bert': {
        'Size (MB)': 426,
        'Quantized Size (MB)': 112.0,
        'Arc Accuracy': 0.855,
        'Types Accuracy': 0.848,
        'Root Accuracy': 0.920,
    },
    'tiny-bert': {
        'Size (MB)': 59.5,
        'Quantized Size (MB)': 15.7,
        'Arc Accuracy': 0.718,
        'Types Accuracy': 0.694,
        'Root Accuracy': 0.886,
    },
    'albert': {
        'Size (MB)': 50,
        'Quantized Size (MB)': 13.2,
        'Arc Accuracy': 0.811,
        'Types Accuracy': 0.793,
        'Root Accuracy': 0.879,
    },
    'tiny-albert': {
        'Size (MB)': 24.8,
        'Quantized Size (MB)': 6.6,
        'Arc Accuracy': 0.708,
        'Types Accuracy': 0.673,
        'Root Accuracy': 0.817,
    },
    'xlnet': {
        'Size (MB)': 450.2,
        'Quantized Size (MB)': 119.0,
        'Arc Accuracy': 0.931,
        'Types Accuracy': 0.925,
        'Root Accuracy': 0.947,
    },
    'alxlnet': {
        'Size (MB)': 50,
        'Quantized Size (MB)': 14.3,
        'Arc Accuracy': 0.894,
        'Types Accuracy': 0.886,
        'Root Accuracy': 0.942,
    },
}

_transformer_availability_v2 = {
    'bert': {
        'Size (MB)': 455,
        'Quantized Size (MB)': 114.0,
        'Arc Accuracy': 0.82045,
        'Types Accuracy': 0.79970,
        'Root Accuracy': 0.98936,
    },
    'tiny-bert': {
        'Size (MB)': 69.7,
        'Quantized Size (MB)': 17.5,
        'Arc Accuracy': 0.795252,
        'Types Accuracy': 0.72470,
        'Root Accuracy': 0.98939,
    },
    'albert': {
        'Size (MB)': 60.8,
        'Quantized Size (MB)': 15.3,
        'Arc Accuracy': 0.821895,
        'Types Accuracy': 0.79752,
        'Root Accuracy': 1.0,
    },
    'tiny-albert': {
        'Size (MB)': 33.4,
        'Quantized Size (MB)': 8.51,
        'Arc Accuracy': 0.7865,
        'Types Accuracy': 0.7587,
        'Root Accuracy': 1.0,
    },
    'xlnet': {
        'Size (MB)': 480.2,
        'Quantized Size (MB)': 121.0,
        'Arc Accuracy': 0.84811,
        'Types Accuracy': 0.82741,
        'Root Accuracy': 0.92101,
    },
    'alxlnet': {
        'Size (MB)': 61.2,
        'Quantized Size (MB)': 16.4,
        'Arc Accuracy': 0.84929,
        'Types Accuracy': 0.8281,
        'Root Accuracy': 0.92099,
    },
}

_transformer_availability = {'v1': _transformer_availability_v1, 'v2': _transformer_availability_v2}


def _validate_version(version):
    version = version.lower()
    if version not in _transformer_availability:
        raise ValueError('version not supported, only supported `v1` or `v2`.')
    return version


def describe():
    """
    Describe Dependency supported.
    """

    d = [
        {'Tag': 'acl', 'Description': 'clausal modifier of noun'},
        {'Tag': 'advcl', 'Description': 'adverbial clause modifier'},
        {'Tag': 'advmod', 'Description': 'adverbial modifier'},
        {'Tag': 'amod', 'Description': 'adjectival modifier'},
        {'Tag': 'appos', 'Description': 'appositional modifier'},
        {'Tag': 'aux', 'Description': 'auxiliary'},
        {'Tag': 'case', 'Description': 'case marking'},
        {'Tag': 'ccomp', 'Description': 'clausal complement'},
        {'Tag': 'compound', 'Description': 'compound'},
        {'Tag': 'compound:plur', 'Description': 'plural compound'},
        {'Tag': 'conj', 'Description': 'conjunct'},
        {'Tag': 'cop', 'Description': 'cop'},
        {'Tag': 'csubj', 'Description': 'clausal subject'},
        {'Tag': 'dep', 'Description': 'dependent'},
        {'Tag': 'det', 'Description': 'determiner'},
        {'Tag': 'fixed', 'Description': 'multi-word expression'},
        {'Tag': 'flat', 'Description': 'name'},
        {'Tag': 'iobj', 'Description': 'indirect object'},
        {'Tag': 'mark', 'Description': 'marker'},
        {'Tag': 'nmod', 'Description': 'nominal modifier'},
        {'Tag': 'nsubj', 'Description': 'nominal subject'},
        {'Tag': 'obj', 'Description': 'direct object'},
        {'Tag': 'parataxis', 'Description': 'parataxis'},
        {'Tag': 'root', 'Description': 'root'},
        {'Tag': 'xcomp', 'Description': 'open clausal complement'},
    ]

    from malaya.function import describe_availability

    return describe_availability(
        d,
        transpose=False,
        text='you can read more from https://universaldependencies.org/treebanks/id_pud/index.html',
    )


def dependency_graph(tagging, indexing):
    """
    Return helper object for dependency parser results. Only accept tagging and indexing outputs from dependency models.
    """
    result = []
    for i in range(len(tagging)):
        result.append(
            '%d\t%s\t_\t_\t_\t_\t%d\t%s\t_\t_'
            % (i + 1, tagging[i][0], int(indexing[i][1]), tagging[i][1])
        )
    return DependencyGraph('\n'.join(result), top_relation_label='root')


def available_transformer(version: str = 'v2'):
    """
    List available transformer dependency parsing models.

    Parameters
    ----------
    version : str, optional (default='v2')
        Version supported. Allowed values:

        * ``'v1'`` - version 1, maintain for knowledge graph.
        * ``'v2'`` - Trained on bigger dataset, better version.

    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability[_validate_version(version)], text='tested on 20% test set.'
    )


@check_type
def transformer(version: str = 'v2', model: str = 'xlnet', quantized: bool = False, **kwargs):
    """
    Load Transformer Dependency Parsing model, transfer learning Transformer + biaffine attention.

    Parameters
    ----------
    version : str, optional (default='v2')
        Version supported. Allowed values:

        * ``'v1'`` - version 1, maintain for knowledge graph.
        * ``'v2'`` - Trained on bigger dataset, better version.

    model : str, optional (default='xlnet')
        Model architecture supported. Allowed values:

        * ``'bert'`` - Google BERT BASE parameters.
        * ``'tiny-bert'`` - Google BERT TINY parameters.
        * ``'albert'`` - Google ALBERT BASE parameters.
        * ``'tiny-albert'`` - Google ALBERT TINY parameters.
        * ``'xlnet'`` - Google XLNET BASE parameters.
        * ``'alxlnet'`` - Malaya ALXLNET BASE parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:

        * if `bert` in model, will return `malaya.model.bert.DependencyBERT`.
        * if `xlnet` in model, will return `malaya.model.xlnet.DependencyXLNET`.
    """

    version = _validate_version(version)
    model = model.lower()
    if model not in _transformer_availability[version]:
        raise ValueError(
            "model not supported, please check supported models from `malaya.dependency.available_transformer(version='{version}')`."
        )

    module = 'dependency'
    minus = 1
    if version != 'v1':
        module = f'{module}-{version}'
        minus = 2

    path = check_file(
        file=model,
        module=module,
        keys={
            'model': 'model.pb',
            'vocab': MODEL_VOCAB[model],
            'tokenizer': MODEL_BPE[model],
        },
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    if model in ['bert', 'tiny-bert', 'albert', 'tiny-albert']:
        inputs = ['Placeholder']
        vectorizer = {'vectorizer': 'import/dense/BiasAdd:0'}

        selected_model = DependencyBERT

    if model in ['xlnet', 'alxlnet']:
        inputs = ['Placeholder', 'Placeholder_1', 'Placeholder_2']
        vectorizer = {'vectorizer': 'import/transpose_3:0'}

        selected_model = DependencyXLNET

    outputs = ['logits', 'heads_seq']
    tokenizer = SentencePieceTokenizer(vocab_file=path['vocab'], spm_model_file=path['tokenizer'])
    input_nodes, output_nodes = nodes_session(
        g, inputs, outputs, extra=vectorizer
    )

    return selected_model(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        tokenizer=tokenizer,
        settings=label,
        minus=minus
    )
