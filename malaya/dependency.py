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


def describe():
    """
    Describe Dependency supported.
    """
    print('acl - clausal modifier of noun')
    print('advcl - adverbial clause modifier')
    print('advmod - adverbial modifier')
    print('amod - adjectival modifier')
    print('appos - appositional modifier')
    print('aux - auxiliary')
    print('case - case marking')
    print('ccomp - clausal complement')
    print('compound - compound')
    print('compound:plur - plural compound')
    print('conj - conjunct')
    print('cop - cop')
    print('csubj - clausal subject')
    print('dep - dependent')
    print('det - determiner')
    print('fixed - multi-word expression')
    print('flat - name')
    print('iobj - indirect object')
    print('mark - marker')
    print('nmod - nominal modifier')
    print('nsubj - nominal subject')
    print('obj - direct object')
    print('parataxis - parataxis')
    print('root - root')
    print('xcomp - open clausal complement')
    print(
        'you can read more from https://universaldependencies.org/treebanks/id_pud/index.html'
    )


def dependency_graph(tagging, indexing):
    """
    Return helper object for dependency parser results. Only accept tagging and indexing outputs from dependency models
    """
    result = []
    for i in range(len(tagging)):
        result.append(
            '%d\t%s\t_\t_\t_\t_\t%d\t%s\t_\t_'
            % (i + 1, tagging[i][0], int(indexing[i][1]), tagging[i][1])
        )
    return DependencyGraph('\n'.join(result), top_relation_label = 'root')


_availability = {
    'bert': [
        '426.0 MB',
        'arc accuracy: 0.855',
        'types accuracy: 0.848',
        'root accuracy: 0.920',
    ],
    'tiny-bert': [
        '59.5 MB',
        'arc accuracy: 0.718',
        'types accuracy: 0.694',
        'root accuracy: 0.886',
    ],
    'albert': [
        '50.0 MB',
        'arc accuracy: 0.811',
        'types accuracy: 0.793',
        'root accuracy: 0.879',
    ],
    'tiny-albert': [
        '24.8 MB',
        'arc accuracy: 0.708',
        'types accuracy: 0.673',
        'root accuracy: 0.817',
    ],
    'xlnet': [
        '450.2 MB',
        'arc accuracy: 0.931',
        'types accuracy: 0.925',
        'root accuracy: 0.947',
    ],
    'alxlnet': [
        '50.0 MB',
        'arc accuracy: 0.894',
        'types accuracy: 0.886',
        'root accuracy: 0.942',
    ],
}


def available_transformer():
    """
    List available transformer dependency parsing models.
    """
    return _availability


@check_type
def transformer(model: str = 'xlnet', **kwargs):
    """
    Load Transformer Entity Tagging model, transfer learning Transformer + biaffine attention.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'bert'`` - BERT architecture from google.
        * ``'tiny-bert'`` - BERT architecture from google with smaller parameters.
        * ``'albert'`` - ALBERT architecture from google.
        * ``'tiny-albert'`` - ALBERT architecture from google with smaller parameters.
        * ``'xlnet'`` - XLNET architecture from google.
        * ``'alxlnet'`` - XLNET architecture from google + Malaya.

    Returns
    -------
    result : Transformer class
    """

    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya.dependency.available_transformer()'
        )

    check_file(PATH_DEPENDENCY[model], S3_PATH_DEPENDENCY[model], **kwargs)
    g = load_graph(PATH_DEPENDENCY[model]['model'])

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
            sess = generate_session(graph = g, **kwargs),
            tokenizer = tokenizer,
            settings = label,
            heads_seq = g.get_tensor_by_name('import/heads_seq:0'),
        )
