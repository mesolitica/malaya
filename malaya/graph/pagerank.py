from malaya import gpu_available
from networkx import nx
import numpy as np
import logging


def pagerank(array, retry = 5):
    cpu = False
    fail = True
    if gpu_available():
        try:
            import cugraph

            cpu = True
        except Exception as e:
            msg = (
                'cugraph not installed. Please install it from https://github.com/rapidsai/cugraph. \n\n'
                'Will calculate pagerank using networkx CPU version.'
            )
            logging.warning(msg)
            cpu = True

    else:
        cpu = True

    if cpu:
        nx_graph = nx.from_numpy_array(array)
        for _ in range(retry):
            try:
                scores = nx.pagerank(nx_graph, max_iter = 10000)
                fail = False
                break
            except Exception as e:
                logging.warning(e)

    if fail:
        raise Exception(
            'pagerank not able to converge, rerun may able to solve it.'
        )

    return scores
