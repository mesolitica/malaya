from malaya.function import get_device
from scipy import sparse
from networkx import nx
import numpy as np
import logging


def pagerank(array, fast_pagerank=True, retry=5, **kwargs):
    device = get_device(**kwargs)
    cpu = False
    fail = True
    if 'GPU' in device:
        try:
            import cugraph

            cpu = True
        except Exception as e:
            msg = (
                'cugraph not installed. Please install it from https://github.com/rapidsai/cugraph. \n'
                'Will calculate pagerank using networkx CPU version.')
            logging.warning(msg)
            cpu = True

    else:
        cpu = True

    if cpu:
        if fast_pagerank:
            from scipy import sparse
            from malaya.graph import fast_pagerank

            G = sparse.csr_matrix(array)
            r = fast_pagerank.pagerank(G)
            scores = {i: r[i] for i in range(len(r))}
            fail = False

        else:
            nx_graph = nx.from_numpy_array(array)
            for _ in range(retry):
                try:
                    scores = nx.pagerank(nx_graph, max_iter=10000)
                    fail = False
                    break
                except Exception as e:
                    logging.warning(e)

    if fail:
        raise Exception(
            'pagerank not able to converge, rerun may able to solve it.'
        )

    return scores
