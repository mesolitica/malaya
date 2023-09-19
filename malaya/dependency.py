from malaya.function.parse_dependency import DependencyGraph
from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
import logging
import warnings

logger = logging.getLogger(__name__)

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

_huggingface_availability = {
    'mesolitica/finetune-dependency-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 143,
        'Arc Accuracy': 0.8506069089930276,
        'Types Accuracy': 0.7831641780774206,
        'Root Accuracy': 0.8723021582733813,
    },
    'mesolitica/finetune-dependency-t5-small-standard-bahasa-cased': {
        'Size (MB)': 247,
        'Arc Accuracy': 0.8494045261191319,
        'Types Accuracy': 0.783103051811978,
        'Root Accuracy': 0.8669064748201439,
    },
    'mesolitica/finetune-dependency-t5-base-standard-bahasa-cased': {
        'Size (MB)': 898,
        'Arc Accuracy': 0.8528921010932324,
        'Types Accuracy': 0.7840908663367674,
        'Root Accuracy': 0.8597122302158273,
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


def available_huggingface():
    """
    List available huggingface models.
    """

    logger.info(
        'tested on test set at https://github.com/huseinzol05/malay-dataset/tree/master/parsing/dependency')
    return describe_availability(_huggingface_availability)


def huggingface(
    model: str = 'mesolitica/finetune-dependency-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to dependency parsing.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-dependency-t5-small-standard-bahasa-cased')
        Check available models at `malaya.dependency.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Dependency
    """
    logger.warning(
        '`malaya.dependency.huggingface` trained on indonesian dataset and augmented dataset, not an actual malay dataset.')

    return load_huggingface.load_dependency(model=model, **kwargs)
