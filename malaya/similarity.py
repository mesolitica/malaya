import re
import os
import random
import numpy as np
from fuzzywuzzy import fuzz
import json
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)
from ._utils._utils import (
    check_file,
    load_graph,
    check_available,
    generate_session,
)
from .preprocessing import _tokenizer
from ._utils._paths import PATH_SIMILARITY, S3_PATH_SIMILARITY
from ._models._bert_model import SIAMESE_BERT


class _VECTORIZER_SIMILARITY:
    def __init__(self, vectorizer):
        self._vectorizer = vectorizer

    def _predict(self, left_strings, right_strings, similarity = 'cosine'):
        if not isinstance(left_strings, list):
            raise ValueError('left_strings must be a list')
        if not isinstance(left_strings[0], str):
            raise ValueError('left_strings must be list of strings')
        if not isinstance(right_strings, list):
            raise ValueError('right_strings must be a list')
        if not isinstance(right_strings[0], str):
            raise ValueError('right_strings must be list of strings')

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

    def predict(self, left_string, right_string, similarity = 'cosine'):
        """
        calculate similarity for two different texts.

        Parameters
        ----------
        left_string : str
        right_string : str
        similarity : str, optional (default='mean')
            similarity supported. Allowed values:

            * ``'cosine'`` - cosine similarity.
            * ``'euclidean'`` - euclidean similarity.
            * ``'manhattan'`` - manhattan similarity.

        Returns
        -------
        float: float
        """
        if not isinstance(left_string, str):
            raise ValueError('left_string must be a string')
        if not isinstance(right_string, str):
            raise ValueError('right_string must be a string')
        return self._predict(
            [left_string], [right_string], similarity = similarity
        )[0, 0]

    def predict_batch(self, left_strings, right_strings, similarity = 'cosine'):
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
        list: list of float
        """
        return self._predict(
            left_strings, right_strings, similarity = similarity
        ).diagonal()

    def tree_plot(
        self,
        strings,
        similarity = 'cosine',
        visualize = True,
        figsize = (7, 7),
        annotate = True,
    ):
        """
        plot a tree plot based on output from bert similarity.

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
        list_dictionaries: list of results
        """
        if not isinstance(visualize, bool):
            raise ValueError('visualize must be a boolean')
        if not isinstance(figsize, tuple):
            raise ValueError('figsize must be a tuple')
        if not isinstance(annotate, bool):
            raise ValueError('annotate must be a boolean')

        results = self._predict(strings, strings, similarity = similarity)
        if not visualize:
            return results

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set()
        except:
            raise Exception(
                'matplotlib and seaborn not installed. Please install it and try again.'
            )

        plt.figure(figsize = figsize)
        g = sns.clustermap(
            results,
            cmap = 'Blues',
            xticklabels = strings,
            yticklabels = strings,
            annot = annotate,
        )
        plt.show()


class _DOC2VEC_SIMILARITY:
    def __init__(self, vectorizer):
        self._vectorizer = vectorizer

    def _predict(
        self,
        left_strings,
        right_strings,
        aggregation = 'mean',
        similarity = 'cosine',
        soft = True,
    ):

        if not isinstance(left_strings, list):
            raise ValueError('left_strings must be a list')
        if not isinstance(left_strings[0], str):
            raise ValueError('left_strings must be list of strings')
        if not isinstance(right_strings, list):
            raise ValueError('right_strings must be a list')
        if not isinstance(right_strings[0], str):
            raise ValueError('right_strings must be list of strings')
        if not isinstance(aggregation, str):
            raise ValueError('aggregation must be a string')
        if not isinstance(similarity, str):
            raise ValueError('similarity must be a string')
        if not isinstance(soft, bool):
            raise ValueError('soft must be a boolean')

        if len(left_strings) != len(right_strings):
            raise ValueError(
                'length list of left strings must be same with length list of right strings'
            )
        identical = left_strings == right_strings

        aggregation = aggregation.lower()
        if aggregation == 'mean':
            aggregation_function = np.mean
        elif aggregation == 'min':
            aggregation_function = np.min
        elif aggregation == 'max':
            aggregation_function = np.max
        elif aggregation == 'sum':
            aggregation_function = np.sum
        elif aggregation == 'sqrt':
            aggregation_function = np.sqrt
        else:
            raise ValueError(
                "aggregation only supports 'mean', 'min', 'max', 'sum' and 'sqrt'"
            )

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
            left_tokenized = _tokenizer(left_string)
            if not len(left_tokenized):
                raise ValueError('insert not empty left string')
            right_tokenized = _tokenizer(right_string)
            if not len(right_tokenized):
                raise ValueError('insert not empty right string')

            in_vector = []
            for token in left_tokenized:
                try:
                    in_vector.append(self._vectorizer.get_vector_by_name(token))
                except:
                    if not soft:
                        pass
                    else:
                        arr = np.array(
                            [
                                fuzz.ratio(token, k)
                                for k in self._vectorizer.words
                            ]
                        )
                        idx = (-arr).argsort()[0]
                        in_vector.append(
                            self._vectorizer.get_vector_by_name(
                                self._vectorizer.words[idx]
                            )
                        )
            left_vectors.append(aggregation_function(in_vector, axis = 0))

            if not identical:
                in_vector = []
                for token in right_tokenized:
                    try:
                        in_vector.append(
                            self._vectorizer.get_vector_by_name(token)
                        )
                    except:
                        if not soft:
                            pass
                        else:
                            arr = np.array(
                                [
                                    fuzz.ratio(token, k)
                                    for k in self._vectorizer.words
                                ]
                            )
                            idx = (-arr).argsort()[0]
                            in_vector.append(
                                self._vectorizer.get_vector_by_name(
                                    self._vectorizer.words[idx]
                                )
                            )

                right_vectors.append(aggregation_function(in_vector, axis = 0))

        if identical:
            similar = similarity_function(left_vectors, left_vectors)
        else:
            similar = similarity_function(left_vectors, right_vectors)

        if similarity == 'cosine':
            return (similar + 1) / 2
        else:
            return 1 / (similar + 1)

    def predict(
        self,
        left_string,
        right_string,
        aggregation = 'mean',
        similarity = 'cosine',
        soft = True,
    ):
        """
        calculate similarity for two different texts.

        Parameters
        ----------
        left_string : str
        right_string : str
        aggregation : str, optional (default='mean')
            Aggregation supported. Allowed values:

            * ``'mean'`` - mean.
            * ``'min'`` - min.
            * ``'max'`` - max.
            * ``'sum'`` - sum.
            * ``'sqrt'`` - square root.
        similarity : str, optional (default='mean')
            similarity supported. Allowed values:

            * ``'cosine'`` - cosine similarity.
            * ``'euclidean'`` - euclidean similarity.
            * ``'manhattan'`` - manhattan similarity.
        soft: bool, optional (default=True)
            word not inside word vector will replace with nearest word if True, else, will skip.

        Returns
        -------
        float: float
        """
        if not isinstance(left_string, str):
            raise ValueError('left_string must be a string')
        if not isinstance(right_string, str):
            raise ValueError('right_string must be a string')

        return self._predict(
            [left_string],
            [right_string],
            aggregation = aggregation,
            similarity = similarity,
            soft = soft,
        )[0, 0]

    def predict_batch(
        self,
        left_strings,
        right_strings,
        aggregation = 'mean',
        similarity = 'cosine',
        soft = True,
    ):
        """
        calculate similarity for two different batch of texts.

        Parameters
        ----------
        left_strings : list of str
        right_strings : list of str
        aggregation : str, optional (default='mean')
            Aggregation supported. Allowed values:

            * ``'mean'`` - mean.
            * ``'min'`` - min.
            * ``'max'`` - max.
            * ``'sum'`` - sum.
            * ``'sqrt'`` - square root.
        similarity : str, optional (default='mean')
            similarity supported. Allowed values:

            * ``'cosine'`` - cosine similarity.
            * ``'euclidean'`` - euclidean similarity.
            * ``'manhattan'`` - manhattan similarity.
        soft: bool, optional (default=True)
            word not inside word vector will replace with nearest word if True, else, will skip.

        Returns
        -------
        list: list of float
        """

        return self._predict(
            left_strings,
            right_strings,
            aggregation = aggregation,
            similarity = similarity,
            soft = soft,
        ).diagonal()

    def tree_plot(
        self,
        strings,
        aggregation = 'mean',
        similarity = 'cosine',
        soft = True,
        visualize = True,
        figsize = (7, 7),
        annotate = True,
    ):
        """
        plot a tree plot based on output from bert similarity.

        Parameters
        ----------
        strings : list of str
            list of strings
        aggregation : str, optional (default='mean')
            Aggregation supported. Allowed values:

            * ``'mean'`` - mean.
            * ``'min'`` - min.
            * ``'max'`` - max.
            * ``'sum'`` - sum.
            * ``'sqrt'`` - square root.
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
        list_dictionaries: list of results
        """
        if not isinstance(visualize, bool):
            raise ValueError('visualize must be a boolean')
        if not isinstance(figsize, tuple):
            raise ValueError('figsize must be a tuple')
        if not isinstance(annotate, bool):
            raise ValueError('annotate must be a boolean')

        results = self._predict(
            strings,
            strings,
            aggregation = aggregation,
            similarity = similarity,
            soft = soft,
        )
        if not visualize:
            return results

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set()
        except:
            raise Exception(
                'matplotlib and seaborn not installed. Please install it and try again.'
            )

        plt.figure(figsize = figsize)
        g = sns.clustermap(
            results,
            cmap = 'Blues',
            xticklabels = strings,
            yticklabels = strings,
            annot = annotate,
        )
        plt.show()


def doc2vec(vectorizer):
    """
    Doc2vec interface for text similarity.

    Parameters
    ----------
    vectorizer : object
        word vector interface object, fast-text, word2vec, elmo.

    Returns
    -------
    _DOC2VEC_SIMILARITY: malaya.similarity._DOC2VEC_SIMILARITY
    """

    if not hasattr(vectorizer, 'get_vector_by_name'):
        raise ValueError('vectorizer must has `get_vector_by_name` method')
    return _DOC2VEC_SIMILARITY(vectorizer)


def encoder(vectorizer):
    """
    Encoder interface for text similarity.

    Parameters
    ----------
    vectorizer : object
        encoder interface object, BERT, skip-thought, XLNET.

    Returns
    -------
    _DOC2VEC_SIMILARITY: malaya.similarity._DOC2VEC_SIMILARITY
    """

    if not hasattr(vectorizer, 'vectorize'):
        raise ValueError('vectorizer must has `vectorize` method')
    return _VECTORIZER_SIMILARITY(vectorizer)


def available_bert_model():
    """
    List available bert models.
    """
    return ['multilanguage', 'base', 'small']


def bert(model = 'base', validate = True):
    """
    Load BERT similarity model.

    Parameters
    ----------
    model : str, optional (default='base')
        Model architecture supported. Allowed values:

        * ``'multilanguage'`` - bert multilanguage released by Google, trained on cross-entropy similarity.
        * ``'base'`` - base bert-bahasa released by Malaya, trained on cross-entropy similarity.
        * ``'small'`` - small bert-bahasa released by Malaya, trained on cross-entropy similarity.
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    SIMILARITY_BERT : malaya._models._tensorflow_model.SIAMESE_BERT class
    """
    if not isinstance(model, str):
        raise ValueError('model must be a string')
    if not isinstance(validate, bool):
        raise ValueError('validate must be a boolean')

    model = model.lower()
    if model not in available_bert_model():
        raise Exception(
            'model not supported, please check supported models from malaya.similarity.available_bert_model()'
        )

    if validate:
        check_file(PATH_SIMILARITY[model], S3_PATH_SIMILARITY[model])
    else:
        if not check_available(PATH_SIMILARITY[model]):
            raise Exception(
                'similarity/%s is not available, please `validate = True`'
                % (model)
            )

    if model == 'multilanguage':
        from bert import tokenization

        tokenizer = tokenization.FullTokenizer(
            vocab_file = PATH_SIMILARITY[model]['vocab'], do_lower_case = False
        )
        cls = '[CLS]'
        sep = '[SEP]'
    else:

        import sentencepiece as spm
        from .texts._text_functions import SentencePieceTokenizer

        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(PATH_SIMILARITY[model]['tokenizer'])

        with open(PATH_SIMILARITY[model]['vocab']) as fopen:
            v = fopen.read().split('\n')[:-1]
        v = [i.split('\t') for i in v]
        v = {i[0]: i[1] for i in v}
        tokenizer = SentencePieceTokenizer(v, sp_model)
        cls = '<cls>'
        sep = '<sep>'
    try:
        g = load_graph(PATH_SIMILARITY[model]['model'])
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('similarity/%s') and try again"
            % (model)
        )

    return SIAMESE_BERT(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        segment_ids = g.get_tensor_by_name('import/Placeholder_1:0'),
        input_masks = g.get_tensor_by_name('import/Placeholder_2:0'),
        logits = g.get_tensor_by_name('import/logits:0'),
        sess = generate_session(graph = g),
        tokenizer = tokenizer,
        cls = cls,
        sep = sep,
        label = ['not similar', 'similar'],
    )
