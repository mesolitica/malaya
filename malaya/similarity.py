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
            raise ModuleNotFoundError(
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
            raise ModuleNotFoundError(
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


_transformer_availability = {
    'bert': {'Size (MB)': 423.4, 'Quantized Size (MB)': 111, 'Accuracy': 0.885},
    'tiny-bert': {
        'Size (MB)': 56.6,
        'Quantized Size (MB)': 15,
        'Accuracy': 0.873,
    },
    'albert': {
        'Size (MB)': 48.3,
        'Quantized Size (MB)': 12.8,
        'Accuracy': 0.873,
    },
    'tiny-albert': {
        'Size (MB)': 21.9,
        'Quantized Size (MB)': 6,
        'Accuracy': 0.824,
    },
    'xlnet': {
        'Size (MB)': 448.7,
        'Quantized Size (MB)': 119,
        'Accuracy': 0.784,
    },
    'alxlnet': {
        'Size (MB)': 49.0,
        'Quantized Size (MB)': 13.9,
        'Accuracy': 0.888,
    },
}

_vectorizer_mapping = {
    'bert': 'import/bert/encoder/layer_11/output/LayerNorm/batchnorm/add_1:0',
    'tiny-bert': 'import/bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1:0',
    'albert': 'import/bert/encoder/transformer/group_0_11/layer_11/inner_group_0/LayerNorm_1/batchnorm/add_1:0',
    'tiny-albert': 'import/bert/encoder/transformer/group_0_3/layer_3/inner_group_0/LayerNorm_1/batchnorm/add_1:0',
    'xlnet': 'import/model/transformer/layer_11/ff/LayerNorm/batchnorm/add_1:0',
    'alxlnet': 'import/model/transformer/layer_shared_11/ff/LayerNorm/batchnorm/add_1:0',
}


def available_transformer():
    """
    List available transformer similarity models.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text = 'tested on 20% test set.'
    )


def _transformer(
    model, bert_class, xlnet_class, quantized = False, siamese = False, **kwargs
):
    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.similarity.available_transformer()`.'
        )

    check_file(
        PATH_SIMILARITY[model],
        S3_PATH_SIMILARITY[model],
        quantized = quantized,
        **kwargs
    )
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'
    g = load_graph(PATH_SIMILARITY[model][model_path], **kwargs)

    path = PATH_SIMILARITY

    if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
        if model in ['bert', 'tiny-bert']:
            tokenizer = sentencepiece_tokenizer_bert(
                path[model]['tokenizer'], path[model]['vocab']
            )

        if model in ['albert', 'tiny-albert']:
            tokenizer = tokenization.FullTokenizer(
                vocab_file = path[model]['vocab'],
                do_lower_case = False,
                spm_model_file = path[model]['tokenizer'],
            )

        selected_class = bert_class
        if siamese:
            selected_node = 'import/bert/pooler/dense/BiasAdd:0'

    if model in ['xlnet', 'alxlnet']:

        tokenizer = sentencepiece_tokenizer_xlnet(path[model]['tokenizer'])
        selected_class = xlnet_class
        if siamese:
            selected_node = 'import/model_1/sequnece_summary/summary/BiasAdd:0'

    if not siamese:
        selected_node = _vectorizer_mapping[model]

    return selected_class(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        segment_ids = g.get_tensor_by_name('import/Placeholder_1:0'),
        input_masks = g.get_tensor_by_name('import/Placeholder_2:0'),
        logits = g.get_tensor_by_name('import/logits:0'),
        vectorizer = g.get_tensor_by_name(selected_node),
        sess = generate_session(graph = g, **kwargs),
        tokenizer = tokenizer,
        label = ['not similar', 'similar'],
    )


@check_type
def transformer(model: str = 'bert', quantized: bool = False, **kwargs):
    """
    Load Transformer similarity model.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'bert'`` - Google BERT BASE parameters.
        * ``'tiny-bert'`` - Google BERT TINY parameters.
        * ``'albert'`` - Google ALBERT BASE parameters.
        * ``'tiny-albert'`` - Google ALBERT TINY parameters.
        * ``'xlnet'`` - Google XLNET BASE parameters.
        * ``'alxlnet'`` - Malaya ALXLNET BASE parameters.
    
    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model. 
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result : malaya.model.bert.SIAMESE_BERT class
    """

    return _transformer(
        model = model,
        bert_class = SIAMESE_BERT,
        xlnet_class = SIAMESE_XLNET,
        quantized = quantized,
        siamese = True,
        **kwargs
    )
