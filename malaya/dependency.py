from malaya.function import check_file, load_graph, generate_session
from malaya.text.bpe import (
    sentencepiece_tokenizer_bert,
    sentencepiece_tokenizer_xlnet,
)
from malaya.function.parse_dependency import DependencyGraph
from malaya.path import PATH_DEPENDENCY, S3_PATH_DEPENDENCY

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

_transformer_availability = {
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
        {'Tag': 'advmod', 'Description': 'adverbial modifier'},
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
        transpose = False,
        text = 'you can read more from https://universaldependencies.org/treebanks/id_pud/index.html',
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
    return DependencyGraph('\n'.join(result), top_relation_label = 'root')


def available_transformer():
    """
    List available transformer dependency parsing models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text = 'tested on 20% test set.'
    )


@check_type
def transformer(model: str = 'xlnet', quantized: bool = False, **kwargs):
    """
    Load Transformer Dependency Parsing model, transfer learning Transformer + biaffine attention.

    Parameters
    ----------
    model : str, optional (default='bert')
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
    result : Transformer class
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.dependency.available_transformer()`.'
        )

    check_file(
        PATH_DEPENDENCY[model],
        S3_PATH_DEPENDENCY[model],
        quantized = quantized,
        **kwargs
    )

    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'
    g = load_graph(PATH_DEPENDENCY[model][model_path], **kwargs)

    if model in ['bert', 'tiny-bert', 'albert', 'tiny-albert']:
        from malaya.model.bert import DEPENDENCY_BERT

        tokenizer = sentencepiece_tokenizer_bert(
            PATH_DEPENDENCY[model]['tokenizer'], PATH_DEPENDENCY[model]['vocab']
        )

        return DEPENDENCY_BERT(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = None,
            input_masks = None,
            logits = g.get_tensor_by_name('import/logits:0'),
            vectorizer = g.get_tensor_by_name('import/dense/BiasAdd:0'),
            sess = generate_session(graph = g, **kwargs),
            tokenizer = tokenizer,
            settings = label,
            heads_seq = g.get_tensor_by_name('import/heads_seq:0'),
        )

    if model in ['xlnet', 'alxlnet']:
        from malaya.model.xlnet import DEPENDENCY_XLNET

        tokenizer = sentencepiece_tokenizer_xlnet(
            PATH_DEPENDENCY[model]['tokenizer']
        )

        return DEPENDENCY_XLNET(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = g.get_tensor_by_name('import/Placeholder_1:0'),
            input_masks = g.get_tensor_by_name('import/Placeholder_2:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            vectorizer = g.get_tensor_by_name('import/transpose_3:0'),
            sess = generate_session(graph = g, **kwargs),
            tokenizer = tokenizer,
            settings = label,
            heads_seq = g.get_tensor_by_name('import/heads_seq:0'),
        )
