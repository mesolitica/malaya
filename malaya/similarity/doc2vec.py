import numpy as np
from malaya.preprocessing import Tokenizer
from malaya.text.jarowinkler import JaroWinkler
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)
from herpetologist import check_type
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

    @check_type
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

    @check_type
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


class Doc2VecSimilarity:
    def __init__(self, wordvector):
        self.wordvector = wordvector
        self._jarowinkler = JaroWinkler()
        self._tokenizer = Tokenizer().tokenize

    @check_type
    def _predict(
        self,
        left_strings: List[str],
        right_strings: List[str],
        aggregation: Callable = np.mean,
        similarity: str = 'cosine',
        soft: bool = True,
    ):

        if len(left_strings) != len(right_strings):
            raise ValueError(
                'length list of left strings must be same with length list of right strings'
            )
        identical = left_strings == right_strings

        similarity = similarity.lower()
        if similarity == 'cosine':
            similarity_function = cosine_similarity
        elif similarity == 'euclidean':
            similarity_function = euclidean_distances
        elif similarity == 'manhattan':
            similarity_function = manhattan_distances
        else:
            raise ValueError(
                "similarity only supports 'cosine', 'euclidean', and 'manhattan'"
            )

        left_vectors, right_vectors = [], []
        for i in range(len(left_strings)):
            left_string = left_strings[i]
            right_string = right_strings[i]
            left_tokenized = self._tokenizer(left_string)
            if not len(left_tokenized):
                raise ValueError('insert not empty left string')
            right_tokenized = self._tokenizer(right_string)
            if not len(right_tokenized):
                raise ValueError('insert not empty right string')

            in_vector = []
            for token in left_tokenized:
                if token in self.wordvector._dictionary:
                    v = self.wordvector.get_vector_by_name(token)
                else:
                    if not soft:
                        v = np.zeros((self.wordvector._embed_matrix.shape[1]))
                    else:
                        arr = np.array(
                            [
                                self.wordvector._jarowinkler.similarity(
                                    token, k
                                )
                                for k in self.wordvector.words
                            ]
                        )
                        idx = (-arr).argsort()[0]
                        v = self.wordvector.get_vector_by_name(
                            self.wordvector.words[idx]
                        )
                in_vector.append(v)
            left_vectors.append(aggregation(in_vector, axis=0))

            if not identical:
                in_vector = []
                for token in right_tokenized:
                    if token in self.wordvector._dictionary:
                        v = self.wordvector.get_vector_by_name(token)
                    else:
                        if not soft:
                            v = np.zeros(
                                (self.wordvector._embed_matrix.shape[1])
                            )
                        else:
                            arr = np.array(
                                [
                                    self.wordvector._jarowinkler.similarity(
                                        token, k
                                    )
                                    for k in self.wordvector.words
                                ]
                            )
                            idx = (-arr).argsort()[0]
                            v = self.wordvector.get_vector_by_name(
                                self.wordvector.words[idx]
                            )
                    in_vector.append(v)

                right_vectors.append(aggregation(in_vector, axis=0))

        if identical:
            similar = similarity_function(left_vectors, left_vectors)
        else:
            similar = similarity_function(left_vectors, right_vectors)

        if similarity == 'cosine':
            return (similar + 1) / 2
        else:
            return 1 / (similar + 1)

    @check_type
    def predict_proba(
        self,
        left_strings: List[str],
        right_strings: List[str],
        aggregation: Callable = np.mean,
        similarity: str = 'cosine',
        soft: bool = False,
    ):
        """
        calculate similarity for two different batch of texts.

        Parameters
        ----------
        left_strings : list of str
        right_strings : list of str
        aggregation : Callable, optional (default=numpy.mean)
        similarity : str, optional (default='mean')
            similarity supported. Allowed values:

            * ``'cosine'`` - cosine similarity.
            * ``'euclidean'`` - euclidean similarity.
            * ``'manhattan'`` - manhattan similarity.
        soft: bool, optional (default=False)
            word not inside word vector will replace with nearest word if True, else, will skip.

        Returns
        -------
        result: List[float]
        """

        return self._predict(
            left_strings,
            right_strings,
            aggregation=aggregation,
            similarity=similarity,
            soft=soft,
        ).diagonal()

    @check_type
    def heatmap(
        self,
        strings: List[str],
        aggregation: Callable = np.mean,
        similarity: str = 'cosine',
        soft: bool = False,
        visualize: bool = True,
        annotate: bool = True,
        figsize: Tuple[int, int] = (7, 7),
    ):
        """
        plot a heatmap based on output from bert similarity.

        Parameters
        ----------
        strings : list of str
            list of strings
        aggregation : Callable, optional (default=numpy.mean)
        similarity : str, optional (default='mean')
            similarity supported. Allowed values:

            * ``'cosine'`` - cosine similarity.
            * ``'euclidean'`` - euclidean similarity.
            * ``'manhattan'`` - manhattan similarity.
        soft: bool, optional (default=True)
            word not inside word vector will replace with nearest word if True, else, will skip.
        visualize : bool
            if True, it will render plt.show, else return data.
        figsize : tuple, (default=(7, 7))
            figure size for plot.

        Returns
        -------
        result: list
            list of results.
        """

        results = self._predict(
            strings,
            strings,
            aggregation=aggregation,
            similarity=similarity,
            soft=soft,
        )
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


def wordvector(wv):
    """
    Doc2vec interface for text similarity using Word Vector.

    Parameters
    ----------
    wv: object
        malaya.wordvector.WordVector object.
        should have `get_vector_by_name` method.

    Returns
    -------
    result: malaya.similarity.doc2vec.Doc2VecSimilarity
    """

    if not hasattr(wv, 'get_vector_by_name'):
        raise ValueError('wordvector must have `get_vector_by_name` method')
    return Doc2VecSimilarity(wv)


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
