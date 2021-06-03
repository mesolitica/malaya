import numpy as np
import inspect
import itertools
from sklearn import preprocessing
from malaya.text.lexicon import (
    _sentiment_lexicon as sentiment,
    _emotion_lexicon as emotion,
)
from herpetologist import check_type
from typing import List, Dict, Tuple, Callable


class SelectedEmbedding:
    def __init__(self, wordvector, words, normalization=False):
        vs = {w: wordvector.get_vector_by_name(w) for w in words}
        self.iw = words
        self.wi = {w: i for i, w in enumerate(self.iw)}
        self.m = np.vstack(vs[w] for w in self.iw)
        if normalization:
            preprocessing.normalize(self.m, copy=False)


def teleport_set(words, seeds):
    return [i for i, w in enumerate(words) if w in seeds]


def weighted_teleport_set(words, seed_weights):
    return np.array(
        [seed_weights[word] if word in seed_weights else 0.0 for word in words]
    )


def run_iterative(M, r, update_seeds, max_iter=50, epsilon=1e-6):
    for i in range(max_iter):
        last_r = np.array(r)
        r = np.dot(M, r)
        update_seeds(r)
        if np.abs(r - last_r).sum() < epsilon:
            break
    return r


def logged_loop(iterable, silent=False):
    if not silent:
        from tqdm import tqdm

        loop = tqdm(iterable)
    else:
        loop = range(iterable)
    for i in loop:
        yield i


def similarity_matrix(
    embeddings, arccos=True, similarity_power=100, nn=25
):
    def make_knn(vec, nn=nn):
        vec[vec < vec[np.argsort(vec)[-nn]]] = 0
        return vec

    L = embeddings.m.dot(embeddings.m.T)
    if arccos:
        L = np.arccos(np.clip(-L, -1, 1)) / np.pi
    else:
        L += 1
    np.fill_diagonal(L, 0)
    L = np.apply_along_axis(make_knn, 1, L)
    return L ** similarity_power


def transition_matrix(
    embeddings, arccos=True, similarity_power=100, nn=25
):

    L = similarity_matrix(
        embeddings,
        arccos=arccos,
        similarity_power=similarity_power,
        nn=nn,
    )
    Dinv = np.diag(
        [
            1.0 / np.sqrt(L[i].sum()) if L[i].sum() > 0 else 0
            for i in range(L.shape[0])
        ]
    )
    return Dinv.dot(L).dot(Dinv)


def _populate(
    lexicon,
    wordvector,
    pool_size=10,
    soft=False,
    silent=False,
    normalization=True,
):

    pool_words, seeds = [], {}

    lexicon = lexicon.copy()
    for k in lexicon.keys():
        lexicons = []
        for i in range(len(lexicon[k])):
            lexicon[k][i] = lexicon[k][i].lower()
            if not soft and lexicon[k][i] in wordvector._dictionary:
                lexicons.append(lexicon[k][i])

        if not soft:
            lexicon[k] = lexicons

        lexicon[k] = list(set(lexicon[k]))
        seeds[k] = {word: 1.0 for word in lexicon[k]}
        pool_words.extend(lexicon[k])

    if not silent:
        print('populating nearest words from wordvector')

    batch_parameters = list(
        inspect.signature(wordvector.batch_n_closest).parameters.keys()
    )
    if 'soft' in batch_parameters:
        results = wordvector.batch_n_closest(
            pool_words, num_closest=pool_size, soft=soft
        )
    else:
        results = wordvector.batch_n_closest(
            pool_words, num_closest=pool_size
        )

    results = list(itertools.chain(*results))

    if not silent:
        print('populating vectors from populated nearest words')

    embeddings = SelectedEmbedding(
        wordvector, results, normalization=normalization
    )

    return results, seeds, embeddings


def random_walk(
    lexicon: Dict[str, List[str]],
    wordvector,
    pool_size: int = 10,
    top_n: int = 20,
    similarity_power: float = 10.0,
    beta: float = 0.9,
    arccos: bool = True,
    normalization: bool = True,
    soft: bool = False,
    silent: bool = False,
):
    """
    Induce lexicon by using random walk technique, use in paper, https://arxiv.org/pdf/1606.02820.pdf

    Parameters
    ----------

    lexicon: Dict[str : List[str]]
        curated lexicon from expert domain, {'label1': [str], 'label2': [str]}.
    wordvector: object
        wordvector interface object.
    pool_size: int, optional (default=10)
        pick top-pool size from each lexicons.
    top_n: int, optional (default=20)
        top_n for each vectors will multiple with `similarity_power`.
    similarity_power: float, optional (default=10.0)
        extra score for `top_n`, less will generate less bias induced but high chance unbalanced outcome.
    beta: float, optional (default=0.9)
        penalty score, towards to 1.0 means less penalty. 0 < beta < 1.
    arccos: bool, optional (default=True)
        covariance distribution for embedded.dot(embedded.T). If false, covariance + 1.
    normalization: bool, optional (default=True)
        normalize word vectors using L2 norm. L2 is good to penalize skewed vectors.
    soft: bool, optional (default=False)
        if True, a word not in the dictionary will be replaced with nearest jarowrinkler ratio.
        if False, it will throw an exception if a word not in the dictionary.
    silent: bool, optional (default=False)
        if True, will not print any logs.

    Returns
    -------
    result: tuple(labels[argmax(scores), axis = 1], scores, labels)
    """

    if not hasattr(wordvector, 'batch_n_closest'):
        raise ValueError('wordvector must have `batch_n_closest` method')
    if not hasattr(wordvector, '_dictionary'):
        raise ValueError('wordvector must have `_dictionary` attribute')
    if not (beta > 0 and beta < 1):
        raise ValueError('beta must be bigger than 0 and less than 1')

    results, seeds, embeddings = _populate(
        lexicon=lexicon,
        wordvector=wordvector,
        pool_size=pool_size,
        soft=soft,
        silent=silent,
        normalization=normalization,
    )

    words = embeddings.iw
    M = transition_matrix(
        embeddings,
        arccos=arccos,
        similarity_power=similarity_power,
        nn=top_n,
    )
    keys = list(seeds.keys())
    stacks = []

    if not silent:
        print('random walking from populated vectors \n')

    def run_random_walk(M, teleport, beta):
        def update_seeds(r):
            r += (1 - beta) * teleport / np.sum(teleport)

        return run_iterative(
            M * beta, np.ones(M.shape[1]) / M.shape[1], update_seeds
        )

    for k in keys:
        stacks.append(
            np.expand_dims(
                run_random_walk(
                    M, weighted_teleport_set(results, seeds[k]), beta
                ),
                -1,
            )
        )

    combined = np.concatenate(stacks, axis=1)
    argmax = np.argmax(combined, axis=1)
    return {w: keys[argmax[i]] for i, w in enumerate(results)}, combined, keys


def propagate_probabilistic(
    lexicon: Dict[str, List[str]],
    wordvector,
    pool_size: int = 10,
    top_n: int = 20,
    similarity_power: float = 10.0,
    arccos: bool = True,
    normalization: bool = True,
    soft: bool = False,
    silent: bool = False,
):
    """
    Learns polarity scores via standard label propagation from lexicon sets.

    Parameters
    ----------

    lexicon: Dict[str, List[str]]
        curated lexicon from expert domain, {'label1': [str], 'label2': [str]}.
    wordvector: object
        wordvector interface object.
    pool_size: int, optional (default=10)
        pick top-pool size from each lexicons.
    top_n: int, optional (default=20)
        top_n for each vectors will multiple with `similarity_power`.
    similarity_power: float, optional (default=10.0)
        extra score for `top_n`, less will generate less bias induced but high chance unbalanced outcome.
    arccos: bool, optional (default=True)
        covariance distribution for embedded.dot(embedded.T). If false, covariance + 1.
    normalization: bool, optional (default=True)
        normalize word vectors using L2 norm. L2 is good to penalize skewed vectors.
    soft: bool, optional (default=False)
        if True, a word not in the dictionary will be replaced with nearest jarowrinkler ratio.
        if False, it will throw an exception if a word not in the dictionary.
    silent: bool, optional (default=False)
        if True, will not print any logs.

    Returns
    -------
    result: tuple(labels[argmax(scores), axis = 1], scores, labels)
    """

    if not hasattr(wordvector, 'batch_n_closest'):
        raise ValueError('wordvector must have `batch_n_closest` method')
    if not hasattr(wordvector, '_dictionary'):
        raise ValueError('wordvector must have `_dictionary` attribute')

    results, seeds, embeddings = _populate(
        lexicon=lexicon,
        wordvector=wordvector,
        pool_size=pool_size,
        soft=soft,
        silent=silent,
        normalization=normalization,
    )

    words = embeddings.iw
    M = transition_matrix(
        embeddings,
        arccos=arccos,
        similarity_power=similarity_power,
        nn=top_n,
    )
    keys = list(seeds.keys())
    stacks = []

    if not silent:
        print('propagating probabilistic from populated vectors \n')

    for k in keys:
        stacks.append(teleport_set(words, seeds[k]))

    def update_seeds(r):
        for no, s in enumerate(stacks):
            c = np.zeros((len(stacks)))
            c[no] = 1.0
            r[s] = c
        r /= np.sum(r, axis=1)[:, np.newaxis]

    r = run_iterative(
        M, np.random.random((M.shape[0], len(keys))), update_seeds
    )
    argmax = np.argmax(r, axis=1)
    return {w: keys[argmax[i]] for i, w in enumerate(results)}, r, keys


def propagate_graph(
    lexicon: Dict[str, List[str]],
    wordvector,
    pool_size: int = 10,
    top_n: int = 20,
    similarity_power: float = 10.0,
    normalization: bool = True,
    soft: bool = False,
    silent: bool = False,
):
    """
    Graph propagation method dapted from Velikovich, Leonid, et al. "The viability of web-derived polarity lexicons." http://www.aclweb.org/anthology/N10-1119

    Parameters
    ----------

    lexicon: Dict[str, List[str]]
        curated lexicon from expert domain, {'label1': [str], 'label2': [str]}.
    wordvector: object
        wordvector interface object.
    pool_size: int, optional (default=10)
        pick top-pool size from each lexicons.
    top_n: int, optional (default=20)
        top_n for each vectors will multiple with `similarity_power`.
    similarity_power: float, optional (default=10.0)
        extra score for `top_n`, less will generate less bias induced but high chance unbalanced outcome.
    normalization: bool, optional (default=True)
        normalize word vectors using L2 norm. L2 is good to penalize skewed vectors.
    soft: bool, optional (default=False)
        if True, a word not in the dictionary will be replaced with nearest jarowrinkler ratio.
        if False, it will throw an exception if a word not in the dictionary.
    silent: bool, optional (default=False)
        if True, will not print any logs.

    Returns
    -------
    result: tuple(labels[argmax(scores), axis = 1], scores, labels)
    """

    if not hasattr(wordvector, 'batch_n_closest'):
        raise ValueError('wordvector must have `batch_n_closest` method')
    if not hasattr(wordvector, '_dictionary'):
        raise ValueError('wordvector must have `_dictionary` attribute')

    results, seeds, embeddings = _populate(
        lexicon=lexicon,
        wordvector=wordvector,
        pool_size=pool_size,
        soft=soft,
        silent=silent,
        normalization=normalization,
    )

    words = embeddings.iw
    M = transition_matrix(
        embeddings,
        arccos=True,
        similarity_power=similarity_power,
        nn=top_n,
    )
    M = (M + M.T) / 2
    keys = list(seeds.keys())

    from scipy.sparse import csr_matrix

    csr_M = csr_matrix(M)

    if not silent:
        print('propagate graph from populated nearest words')

    def run_graph_propagate(seeds, alpha_mat, trans_mat, T=3):
        def get_rel_edges(ind_set):
            rel_edges = set([])
            for node in ind_set:
                rel_edges = rel_edges.union(
                    [(node, other) for other in trans_mat[node, :].nonzero()[1]]
                )
            return rel_edges

        for seed in seeds:
            F = set([seed])
            for t in range(T):
                for edge in get_rel_edges(F):
                    alpha_mat[seed, edge[1]] = max(
                        alpha_mat[seed, edge[1]],
                        alpha_mat[seed, edge[0]] * trans_mat[edge[0], edge[1]],
                    )
                    F.add(edge[1])
        return alpha_mat

    stacks = []
    for k in keys:
        p = run_graph_propagate(
            [embeddings.wi[seed] for seed in seeds[k]], M.copy(), csr_M
        )
        p = p + p.T
        stacks.append(p)

    index = embeddings.wi
    for w in logged_loop(index, silent=silent):
        for no, k in enumerate(keys):
            if w not in seeds[k]:
                seeds[k][w] = sum(
                    stacks[no][index[w], index[seed]] for seed in seeds[k]
                )

    stacks = []
    for k in keys:
        stacks.append(np.expand_dims([seeds[k][w] for w in results], axis=1))

    combined = np.concatenate(stacks, axis=1)
    argmax = np.argmax(combined, axis=1)
    return {w: keys[argmax[i]] for i, w in enumerate(results)}, combined, keys
