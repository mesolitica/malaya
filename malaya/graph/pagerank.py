from malaya.graph import fast_pagerank as load_fast_pagerank
from scipy import sparse
import logging

logger = logging.getLogger(__name__)


def pagerank(array, fast_pagerank=True, retry=5, **kwargs):

    fail = True

    if fast_pagerank:
        G = sparse.csr_matrix(array)
        r = load_fast_pagerank.pagerank(G, **kwargs)
        scores = {i: r[i] for i in range(len(r))}
        fail = False

    else:
        try:
            import networkx as nx
        except BaseException:
            from networkx import nx

        nx_graph = nx.from_numpy_array(array)
        for _ in range(retry):
            try:
                scores = nx.pagerank(nx_graph, max_iter=10000)
                fail = False
                break
            except Exception as e:
                logger.warning(e)

    if fail:
        raise Exception(
            'pagerank not able to converge, increase `retry` and rerun may able to solve it.'
        )

    return scores
