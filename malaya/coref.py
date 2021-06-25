from sklearn.metrics.pairwise import cosine_similarity
from malaya.function.parse_dependency import DependencyGraph
from malaya.text.function import split_nya as _split_nya, PUNCTUATION
from malaya.stack import voting_stack
from malaya.cluster import cluster_words
from malaya.model.bert import DependencyBERT
from malaya.model.xlnet import DependencyXLNET
from herpetologist import check_type
import numpy as np
from typing import List, Callable

# Kakak mempunyai kucing. Dia menyayanginya. Dia -> Kakak, nya -> kucing
# Husein Zolkepli suka makan ayam. Dia pun suka makan daging. Dia -> Husein Zolkepli


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
    results_ = []
    for r in results:
        r = [i[0] for i in r]
        results_.append(' '.join(r))
    return results_


@check_type
def parse_from_dependency(models, string: str,
                          references: List[str] = ['dia', 'itu', 'ini', 'saya', 'awak', 'kamu', 'kita', 'kami', 'mereka'],
                          rejected_references: List[str] = ['saya', 'awak', 'kamu', 'kita', 'kami', 'mereka', 'nya'],
                          acceptable_subjects: List[str] = ['flat', 'subj', 'nsubj', 'csubj', 'obj'],
                          acceptable_nested_subjects: List[str] = ['compound', 'flat'],
                          split_nya: bool = True,
                          aggregate: Callable = np.mean,
                          top_k: int = 20):
    """
    Apply Coreference Resolution using stacks of dependency models.

    Parameters
    ----------
    models: list
        list of dependency models, must has `vectorize` method.
    string: str
    references: List[str], optional (default=['dia', 'itu', 'ini', 'saya', 'awak', 'kamu', 'kita', 'kami', 'mereka'])
        list of references.
    rejected_references: List[str], optional (default=['saya', 'awak', 'kamu', 'kita', 'kami', 'mereka'])
        list of rejected references during populating subjects.
    acceptable_subjects:List[str], optional
        List of dependency labels for subjects.
    acceptable_nested_subjects: List[str], optional
        List of dependency labels for nested subjects, eg, syarikat (obl) facebook (compound).
    split_nya: bool, optional (default=True)
        split `nya`, eg, `disifatkannya` -> `disifatkan`, `nya`.
    aggregate: Callable, optional (default=numpy.mean)
        Aggregate function to aggregate list of vectors from `model.vectorize`.
    top_k: int, optional (default=20)
        only accept near top_k to assume a coherence.

    Returns
    -------
    result: Dict[text, coref]
        {'text': ['Husein','Zolkepli','suka','makan','ayam','.','Dia','pun','suka','makan','daging','.'],
        'coref': {6: {'index': [0, 1], 'text': ['Husein', 'Zolkepli']}}}
    """
    if not isinstance(models, list):
        raise ValueError('models must be a list')

    for m in range(len(models)):
        if type(models[m]) not in [DependencyBERT, DependencyXLNET]:
            raise ValueError('model must one of [malaya.model.bert.DependencyBERT, malaya.model.xlnet.DependencyXLNET]')

    if split_nya:
        string = _split_nya(string)
        references = references + ['nya']

    tagging, indexing = voting_stack(models, string)

    result = []
    for i in range(len(tagging)):
        result.append(
            '%d\t%s\t_\t_\t_\t_\t%d\t%s\t_\t_'
            % (i + 1, tagging[i][0], int(indexing[i][1]), tagging[i][1])
        )

    d_object = DependencyGraph('\n'.join(result), top_relation_label='root')

    rs = []
    for i in range(len(indexing)):
        for s in acceptable_subjects:
            if d_object.nodes[i]['rel'] == s:
                r = []
                for n_s in acceptable_nested_subjects:
                    s_ = d_object.traverse_children(i, [n_s], initial_label=[s])
                    s_ = _combined(s_)
                    r.extend(s_)
                r = [i for i in r if i.lower() not in references and not i.lower() in rejected_references]
                rs.extend(r)
    rs = cluster_words(rs, lowercase=True)

    vs, X = [], None
    for m in range(len(models)):
        v = models[m].vectorize(string)
        X = [i[0] for i in v]
        y = [i[1] for i in v]
        vs.append(y)
    V = aggregate(vs, axis=0)

    indices, word_indices = {}, []
    for no, row in enumerate(rs):
        ind = []
        for word in row.split():
            indices[word] = indices.get(word, no)
            ind.append(X.index(word))
        word_indices.append(ind)

    index_word = []
    for key in indices:
        index_word.append(X.index(key))

    index_references = []
    for i in range(len(X)):
        if X[i].lower() in references:
            index_references.append(i)

    similarities = cosine_similarity(V)

    results = {}

    for r in index_references:
        r_ = [r, r - 1]
        i_ = -1

        # subject verb object . subject, we want to reject words before punct
        while X[r + i_] in PUNCTUATION:
            i_ -= 1
            r_.append(r + i_)

        index_word_ = [i for i in index_word if i < r]
        sorted_indices = similarities[r].argsort()[-top_k:][::-1]
        sorted_indices = sorted_indices[np.isin(sorted_indices, index_word_) & ~
                                        np.isin(sorted_indices, r_)]
        if len(sorted_indices):
            s = rs[indices[X[sorted_indices[0]]]]
            index = word_indices[indices[X[sorted_indices[0]]]]
            results[r] = {'index': index, 'text': s.split()}

    return {'text': X, 'coref': results}
