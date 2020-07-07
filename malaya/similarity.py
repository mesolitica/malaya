import numpy as np
from malaya.text.jarowinkler import JaroWinkler
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)
from malaya.function import check_file, load_graph, generate_session
from malaya.text.bpe import (
    sentencepiece_tokenizer_bert,
    sentencepiece_tokenizer_xlnet,
)
from malaya.preprocessing import _tokenizer
from malaya.model.bert import SIAMESE_BERT
from malaya.model.xlnet import SIAMESE_XLNET
from malaya.path import PATH_SIMILARITY, S3_PATH_SIMILARITY
from herpetologist import check_type
from typing import List, Tuple, Callable


class VECTORIZER_SIMILARITY:
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
            left_strings, right_strings, similarity = similarity
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
        g = sns.heatmap(
            results,
            cmap = 'Blues',
            xticklabels = strings,
            yticklabels = strings,
            annot = annotate,
        )
        plt.show()


class DOC2VEC_SIMILARITY:
    def __init__(self, vectorizer):
        self._vectorizer = vectorizer
        self._jarowinkler = JaroWinkler()

    @check_type
    def _predict(
        self,
        left_strings: List[str],
        right_strings: List[str],
        aggregation: str = 'mean',
        similarity: str = 'cosine',
        soft: bool = True,
    ):

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
                                self._jarowinkler.similarity(token, k)
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
                                    self._jarowinkler.similarity(token, k)
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

    @check_type
    def predict_proba(
        self,
        left_strings: List[str],
        right_strings: List[str],
        aggregation: str = 'mean',
        similarity: str = 'cosine',
        soft: bool = True,
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
        result: List[float]
        """

        return self._predict(
            left_strings,
            right_strings,
            aggregation = aggregation,
            similarity = similarity,
            soft = soft,
        ).diagonal()

    @check_type
    def heatmap(
        self,
        strings: List[str],
        aggregation: str = 'mean',
        similarity: str = 'cosine',
        soft: bool = True,
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
        result: list
            list of results.
        """

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
        g = sns.heatmap(
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
    result: malaya.similarity.DOC2VEC_SIMILARITY
    """

    if not hasattr(vectorizer, 'get_vector_by_name'):
        raise ValueError('vectorizer must has `get_vector_by_name` method')
    return DOC2VEC_SIMILARITY(vectorizer)


def encoder(vectorizer):
    """
    Encoder interface for text similarity.

    Parameters
    ----------
    vectorizer : object
        encoder interface object, BERT, skip-thought, XLNET.

    Returns
    -------
    result: malaya.similarity.VECTORIZER_SIMILARITY
    """

    if not hasattr(vectorizer, 'vectorize'):
        raise ValueError('vectorizer must has `vectorize` method')
    return VECTORIZER_SIMILARITY(vectorizer)


_availability = {
    'bert': ['423.4 MB', 'accuracy: 0.885'],
    'tiny-bert': ['56.6 MB', 'accuracy: 0.873'],
    'albert': ['46.3 MB', 'accuracy: 0.873'],
    'tiny-albert': ['21.9 MB', 'accuracy: 0.824'],
    'xlnet': ['448.7 MB', 'accuracy: 0.784'],
    'alxlnet': ['49.0 MB', 'accuracy: 0.888'],
}


def available_transformer():
    """
    List available transformer similarity models.
    """
    return _availability


def _transformer(model, bert_class, xlnet_class, **kwargs):
    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya.similarity.available_transformer()'
        )

    check_file(PATH_SIMILARITY[model], S3_PATH_SIMILARITY[model], **kwargs)
    g = load_graph(PATH_SIMILARITY[model]['model'])

    path = PATH_SIMILARITY

    if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
        if model in ['bert', 'tiny-bert']:
            from malaya.transformers.bert import (
                _extract_attention_weights_import,
            )
            from malaya.transformers.bert import bert_num_layers

            tokenizer = sentencepiece_tokenizer_bert(
                path[model]['tokenizer'], path[model]['vocab']
            )

        if model in ['albert', 'tiny-albert']:
            from malaya.transformers.albert import (
                _extract_attention_weights_import,
            )
            from malaya.transformers.albert import bert_num_layers
            from albert import tokenization

            tokenizer = tokenization.FullTokenizer(
                vocab_file = path[model]['vocab'],
                do_lower_case = False,
                spm_model_file = path[model]['tokenizer'],
            )

        return bert_class(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = g.get_tensor_by_name('import/Placeholder_1:0'),
            input_masks = g.get_tensor_by_name('import/Placeholder_2:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            sess = generate_session(graph = g, **kwargs),
            tokenizer = tokenizer,
            label = ['not similar', 'similar'],
        )

    if model in ['xlnet', 'alxlnet']:
        if model in ['xlnet']:
            from malaya.transformers.xlnet import (
                _extract_attention_weights_import,
            )
        if model in ['alxlnet']:
            from malaya.transformers.alxlnet import (
                _extract_attention_weights_import,
            )

        tokenizer = sentencepiece_tokenizer_xlnet(path[model]['tokenizer'])

        return xlnet_class(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = g.get_tensor_by_name('import/Placeholder_1:0'),
            input_masks = g.get_tensor_by_name('import/Placeholder_2:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            sess = generate_session(graph = g, **kwargs),
            tokenizer = tokenizer,
            label = ['not similar', 'similar'],
        )


@check_type
def transformer(model: str = 'bert', **kwargs):
    """
    Load Transformer similarity model.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'bert'`` - BERT architecture from google.
        * ``'tiny-bert'`` - BERT architecture from google with smaller parameters.
        * ``'albert'`` - ALBERT architecture from google.
        * ``'tiny-albert'`` - ALBERT architecture from google with smaller parameters.
        * ``'xlnet'`` - XLNET architecture from google.
        * ``'alxlnet'`` - XLNET architecture from google + Malaya.

    Returns
    -------
    result : malaya.model.bert.SIAMESE_BERT class
    """

    return _transformer(
        model = model,
        bert_class = SIAMESE_BERT,
        xlnet_class = SIAMESE_XLNET,
        **kwargs
    )
