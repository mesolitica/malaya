import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

from itertools import product
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import euclidean
import pulp


def _tokens_to_fracdict(tokens):
    cntdict = defaultdict(lambda: 0)
    for token in tokens:
        cntdict[token] += 1
    totalcnt = sum(cntdict.values())
    return {token: float(cnt) / totalcnt for token, cnt in cntdict.items()}


def _word_mover(left_token, right_token, vectorizer):
    all_tokens = list(set(left_token + right_token))
    wordvecs = {
        token: vectorizer.get_vector_by_name(token) for token in all_tokens
    }
    left_bucket = _tokens_to_fracdict(left_token)
    right_bucket = _tokens_to_fracdict(right_token)

    T = pulp.LpVariable.dicts(
        'T_matrix', list(product(all_tokens, all_tokens)), lowBound = 0
    )
    prob = pulp.LpProblem('WMD', sense = pulp.LpMinimize)
    prob += pulp.lpSum(
        [
            T[token1, token2] * euclidean(wordvecs[token1], wordvecs[token2])
            for token1, token2 in product(all_tokens, all_tokens)
        ]
    )
    for token2 in right_bucket:
        prob += (
            pulp.lpSum([T[token1, token2] for token1 in left_bucket])
            == right_bucket[token2]
        )
    for token1 in left_bucket:
        prob += (
            pulp.lpSum([T[token1, token2] for token2 in right_bucket])
            == left_bucket[token1]
        )
    prob.solve()
    return prob


def distance(left_token, right_token, vectorizer):
    """
    calculate word mover distance between left hand-side sentence and right hand-side sentence

    Parameters
    ----------
    left_token : list
        Eg, ['saya','suka','makan','ayam']
    right_token : list
        Eg, ['saya','suka','makan','ikan']
    vectorizer : object
        fast-text or word2vec interface object

    Returns
    -------
    distance: float
    """
    assert isinstance(left_token, list), 'left_token must be a list'
    assert isinstance(right_token, list), 'right_token must be a list'
    assert hasattr(
        vectorizer, 'get_vector_by_name'
    ), 'vectorizer must has `get_vector_by_name` method'
    prob = _word_mover(left_token, right_token, vectorizer)
    return pulp.value(prob.objective)
