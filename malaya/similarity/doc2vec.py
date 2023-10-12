import numpy as np
from malaya.preprocessing import Tokenizer
from malaya.text.jarowinkler import JaroWinkler
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)
from typing import List, Tuple, Callable

similarity_functions = {
    'cosine': cosine_similarity,
    'euclidean': euclidean_distances,
    'manhattan': manhattan_distances,
}


class VectorizerSimilarity:
    def __init__(self, vectorizer):
        self._vectorizer = vectorizer

    def _predict(
        self,
        left_strings: List[str],
        right_strings: List[str],
        similarity: str = 'cosine',
    ):

        if len(left_strings) != len(right_strings):
            raise ValueError(
                'length list of left strings must be same with length list of right strings'
            )
        identical = left_strings == right_strings

        similarity = similarity.lower()
        similarity_function = similarity_functions.get(similarity)
        if similarity_function is None:
            raise ValueError(
                "similarity only supports 'cosine', 'euclidean', and 'manhattan'"
            )

        left_vectors = self._vectorizer.vectorize(left_strings)

        if identical:
            similar = similarity_function(left_vectors, left_vectors)
        else:
            right_vectors = self._vectorizer.vectorize(right_strings)
            similar = similarity_function(left_vectors, right_vectors)

        if similarity == 'cosine':
            return (similar + 1) / 2
        else:
            return 1 / (similar + 1)

    def predict_proba(
        self,
        left_strings: List[str],
        right_strings: List[str],
        similarity: str = 'cosine',
    ):
        """
        calculate similarity for two different batch of texts.

        Parameters
        ----------
        left_strings : list of str
        right_strings : list of str
        similarity : str, optional (default='mean')
            similarity supported. Allowed values:

            * ``'cosine'`` - cosine similarity.
            * ``'euclidean'`` - euclidean similarity.
            * ``'manhattan'`` - manhattan similarity.

        Returns
        -------
        result: List[float]
        """
        return self._predict(
            left_strings, right_strings, similarity=similarity
        ).diagonal()

    def heatmap(
        self,
        strings: List[str],
        similarity: str = 'cosine',
        visualize: bool = True,
        annotate: bool = True,
        figsize: Tuple[int, int] = (7, 7),
    ):
        """
        plot a heatmap based on output from bert similarity.

        Parameters
        ----------
        strings : list of str
            list of strings.
        similarity : str, optional (default='mean')
            similarity supported. Allowed values:

            * ``'cosine'`` - cosine similarity.
            * ``'euclidean'`` - euclidean similarity.
            * ``'manhattan'`` - manhattan similarity.
        visualize : bool
            if True, it will render plt.show, else return data.
        figsize : tuple, (default=(7, 7))
            figure size for plot.

        Returns
        -------
        result: list
            list of results
        """

        results = self._predict(strings, strings, similarity=similarity)
        if not visualize:
            return results

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set()
        except BaseException:
            raise ModuleNotFoundError(
                'matplotlib and seaborn not installed. Please install it and try again.'
            )

        plt.figure(figsize=figsize)
        g = sns.heatmap(
            results,
            cmap='Blues',
            xticklabels=strings,
            yticklabels=strings,
            annot=annotate,
        )
        plt.show()


def vectorizer(v):
    """
    Doc2vec interface for text similarity using Encoder model.

    Parameters
    ----------
    v: object
        encoder interface object, BERT, XLNET.
        should have `vectorize` method.

    Returns
    -------
    result: malaya.similarity.doc2vec.VectorizerSimilarity
    """

    if not hasattr(v, 'vectorize'):
        raise ValueError('vectorizer must have `vectorize` method')
    return VectorizerSimilarity(v)
