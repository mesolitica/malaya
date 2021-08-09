from malaya.supervised import t5 as load_t5
from malaya.function.parse_dependency import DependencyGraph
from malaya.model.t5 import KnowledgeGraph
from herpetologist import check_type
from typing import List, Tuple

_transformer_availability = {
    't5': {
        'Size (MB)': 1250,
        'Quantized Size (MB)': 481,
        'BLEU': 0.919301,
        'Suggested length': 256,
    },
    'small-t5': {
        'Size (MB)': 355.6,
        'Quantized Size (MB)': 195,
        'BLEU': 0.91023386,
        'Suggested length': 512,
    },
    'tiny-t5': {
        'Size (MB)': 208,
        'Quantized Size (MB)': 103,
        'BLEU': 0.93337446,
        'Suggested length': 256,
    },
}


def _combined(r):
    results, last = [], []
    for i in r:
        if type(i) == tuple:
            last.append(i)
        else:
            for no, k in enumerate(last):
                if k[1] == i[0][1]:
                    results.append(last[:no] + i)
                    break
    results.append(last)
    return results


def _get_unique(lists):
    s = set()
    result = []
    for l in lists:
        str_s = str(l)
        if str_s not in s:
            result.append(l)
            s.add(str_s)
    return result


def _get_longest(lists):
    r = []
    for l in lists:
        if len(l) > len(r):
            r = l
    return r


def _postprocess(r, labels=['subject', 'relation', 'object']):
    if all([l not in r for l in labels]):
        return

    for l in labels:
        if len(r[l]) == 0:
            return

        r[l] = ' '.join([i[0] for i in r[l]])

    return r


@check_type
def parse_from_dependency(tagging: List[Tuple[str, str]],
                          indexing: List[Tuple[str, str]],
                          subjects: List[List[str]] = [['flat', 'subj', 'nsubj', 'csubj']],
                          relations: List[List[str]] = [['acl', 'xcomp', 'ccomp', 'obj', 'conj', 'advcl'], ['obj']],
                          objects: List[List[str]] = [['obj', 'compound', 'flat', 'nmod', 'obl']],
                          get_networkx: bool = True):
    """
    Generate knowledge graphs from dependency parsing, we suggest use dependency parsing v1.

    Parameters
    ----------
    tagging: List[Tuple(str, str)]
        `tagging` result from dependency model.
    indexing: List[Tuple(str, str)]
        `indexing` result from dependency model.
    subjects: List[List[str]], optional
        List of dependency labels for subjects.
    relations: List[List[str]], optional
        List of dependency labels for relations.
    objects: List[List[str]], optional
        List of dependency labels for objects.
    get_networkx: bool, optional (default=True)
            If True, will generate networkx.MultiDiGraph.

    Returns
    -------
    result: Dict[result, G]
    """

    if get_networkx:
        try:
            import pandas as pd
            import networkx as nx
        except BaseException:
            logging.warning(
                'pandas and networkx not installed. Please install it by `pip install pandas networkx` and try again. Will skip to generate networkx.MultiDiGraph'
            )
            get_networkx = False

    result = []
    for i in range(len(tagging)):
        result.append(
            '%d\t%s\t_\t_\t_\t_\t%d\t%s\t_\t_'
            % (i + 1, tagging[i][0], int(indexing[i][1]), tagging[i][1])
        )

    d_object = DependencyGraph('\n'.join(result), top_relation_label='root')
    results = []
    for i in range(1, len(indexing), 1):
        if d_object.nodes[i]['rel'] == 'root':
            subjects_, relations_ = [], []
            for s in subjects:
                s_ = d_object.traverse_children(i, s, initial_label=[d_object.nodes[i]['rel']])
                s_ = _combined(s_)
                s_ = [c[1:] for c in s_]
                subjects_.extend(s_)
            for s in relations:
                s_ = d_object.traverse_children(i, s, initial_label=[d_object.nodes[i]['rel']])
                s_ = _combined(s_)
                relations_.extend(s_)
            subjects_ = _get_unique(subjects_)
            subject = _get_longest(subjects_)
            relations_ = _get_unique(relations_)

            for relation in relations_:
                objects_ = []
                k = relation[-1][1]
                for s in objects:
                    s_ = d_object.traverse_children(k, s, initial_label=[d_object.nodes[k]['rel']])
                    s_ = _combined(s_)
                    objects_.extend(s_)
                objects_ = _get_unique(objects_)
                obj = _get_longest(objects_)
                if obj[0][0] == relation[-1][0] and len(obj) == 1:
                    results.append({'subject': subject, 'relation': relation[:-1], 'object': relation[-1:]})
                else:
                    if obj[0][0] == relation[-1][0]:
                        obj = obj[1:]
                    results.append({'subject': subject, 'relation': relation, 'object': obj})

    post_results = []
    for r in results:
        r = _postprocess(r)
        if r:
            post_results.append(r)

    r = {'result': post_results}

    if get_networkx:
        df = pd.DataFrame(post_results)
        G = nx.from_pandas_edgelist(
            df,
            source='subject',
            target='object',
            edge_attr='relation',
            create_using=nx.MultiDiGraph(),
        )
        r['G'] = G

    return r


def available_transformer():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text='tested on KELM test set.'
    )


@check_type
def transformer(model: str = 'small-t5', quantized: bool = False, **kwargs):
    """
    Load transformer to generate knowledge graphs in triples format from texts,
    MS text -> EN triples format.

    Parameters
    ----------
    model : str, optional (default='small-t5')
        Model architecture supported. Allowed values:

        * ``'t5'`` - T5 BASE parameters.
        * ``'small-t5'`` - T5 SMALL parameters.
        * ``'tiny-t5'`` - T5 TINY parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.t5.KnowledgeGraph class
    """
    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.knowledge_graph.available_transformer()`.'
        )

    return load_t5.load(
        module='knowledge-graph-triplet',
        model=model,
        model_class=KnowledgeGraph,
        quantized=quantized,
        **kwargs
    )
