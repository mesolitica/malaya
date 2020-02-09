from ._utils._utils import (
    check_file,
    load_graph,
    generate_session,
    check_available,
    sentencepiece_tokenizer_bert,
    sentencepiece_tokenizer_xlnet,
)
from ._utils._parse_dependency import DependencyGraph
from ._utils._paths import PATH_DEPEND, S3_PATH_DEPEND

from herpetologist import check_type

_dependency_tags = {
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


_availability = {'bert': ['base'], 'xlnet': ['base'], 'albert': ['base']}


def available_transformer_model():
    """
    List available transformer dependency parsing models.
    """
    return _availability


@check_type
def transformer(
    model: str = 'xlnet', size: str = 'base', validate: bool = True
):
    """
    Load Transformer Entity Tagging model, transfer learning Transformer + biaffine attention.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'bert'`` - BERT architecture from google.
        * ``'xlnet'`` - XLNET architecture from google.
        * ``'albert'`` - ALBERT architecture from google.
    size : str, optional (default='base')
        Model size supported. Allowed values:

        * ``'base'`` - BASE size.
        * ``'small'`` - SMALL size.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    MODEL : Transformer class
    """

    model = model.lower()
    size = size.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya.dependency.available_transformer_model()'
        )
    if size not in _availability[model]:
        raise Exception(
            'size not supported, please check supported models from malaya.dependency.available_transformer_model()'
        )

    if validate:
        check_file(PATH_DEPEND[model][size], S3_PATH_DEPEND[model][size])
    else:
        if not check_available(PATH_DEPEND[model][size]):
            raise Exception(
                'dependency/%s/%s is not available, please `validate = True`'
                % (model, size)
            )

    try:
        g = load_graph(PATH_DEPEND[model][size]['model'])
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('dependency/%s/%s') and try again"
            % (model, size)
        )

    if model in ['bert', 'albert']:
        from ._models._bert_model import DEPENDENCY_BERT

        tokenizer, cls, sep = sentencepiece_tokenizer_bert(
            PATH_DEPEND[model][size]['tokenizer'],
            PATH_DEPEND[model][size]['vocab'],
        )

        return DEPENDENCY_BERT(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = None,
            input_masks = None,
            logits = g.get_tensor_by_name('import/logits:0'),
            sess = generate_session(graph = g),
            tokenizer = tokenizer,
            cls = cls,
            sep = sep,
            settings = _dependency_tags,
            heads_seq = g.get_tensor_by_name('import/heads_seq:0'),
        )

    if model in ['xlnet']:
        from ._models._xlnet_model import DEPENDENCY_XLNET

        tokenizer = sentencepiece_tokenizer_xlnet(
            PATH_DEPEND[model][size]['tokenizer']
        )

        return DEPENDENCY_XLNET(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = g.get_tensor_by_name('import/Placeholder_1:0'),
            input_masks = g.get_tensor_by_name('import/Placeholder_2:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            sess = generate_session(graph = g),
            tokenizer = tokenizer,
            settings = _dependency_tags,
            heads_seq = g.get_tensor_by_name('import/heads_seq:0'),
        )
