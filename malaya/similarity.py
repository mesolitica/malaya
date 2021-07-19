import numpy as np
from malaya.text.jarowinkler import JaroWinkler
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)
from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.text.bpe import SentencePieceTokenizer
from malaya.preprocessing import Tokenizer
from malaya.model.bert import SiameseBERT
from malaya.model.xlnet import SiameseXLNET
from malaya.path import MODEL_VOCAB, MODEL_BPE
from herpetologist import check_type
from typing import List, Tuple, Callable


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


def doc2vec_wordvector(wordvector):
    """
    Doc2vec interface for text similarity using Word Vector.

    Parameters
    ----------
    wordvector : object
        malaya.wordvector.WordVector object.
        should have `get_vector_by_name` method.

    Returns
    -------
    result: malaya.similarity.Doc2VecSimilarity
    """

    if not hasattr(wordvector, 'get_vector_by_name'):
        raise ValueError('wordvector must have `get_vector_by_name` method')
    return Doc2VecSimilarity(wordvector)


def doc2vec_vectorizer(vectorizer):
    """
    Doc2vec interface for text similarity using Encoder model.

    Parameters
    ----------
    vectorizer : object
        encoder interface object, BERT, XLNET.
        should have `vectorize` method.

    Returns
    -------
    result: malaya.similarity.VectorizerSimilarity
    """

    if not hasattr(vectorizer, 'vectorize'):
        raise ValueError('vectorizer must have `vectorize` method')
    return VectorizerSimilarity(vectorizer)


_transformer_availability = {
    'bert': {
        'Size (MB)': 423.4,
        'Quantized Size (MB)': 111,
        'macro precision': 0.88315,
        'macro recall': 0.88656,
        'macro f1-score': 0.88405,
    },
    'tiny-bert': {
        'Size (MB)': 56.6,
        'Quantized Size (MB)': 15,
        'macro precision': 0.87210,
        'macro recall': 0.87546,
        'macro f1-score': 0.87292,
    },
    'albert': {
        'Size (MB)': 48.3,
        'Quantized Size (MB)': 12.8,
        'macro precision': 0.87164,
        'macro recall': 0.87146,
        'macro f1-score': 0.87155,
    },
    'tiny-albert': {
        'Size (MB)': 21.9,
        'Quantized Size (MB)': 6,
        'macro precision': 0.82234,
        'macro recall': 0.82383,
        'macro f1-score': 0.82295,
    },
    'xlnet': {
        'Size (MB)': 448.7,
        'Quantized Size (MB)': 119,
        'macro precision': 0.80866,
        'macro recall': 0.76775,
        'macro f1-score': 0.77112,
    },
    'alxlnet': {
        'Size (MB)': 49.0,
        'Quantized Size (MB)': 13.9,
        'macro precision': 0.88756,
        'macro recall': 0.88700,
        'macro f1-score': 0.88727,
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
        _transformer_availability, text='tested on 20% test set.'
    )


def _transformer(
    model, bert_model, xlnet_model, quantized=False, siamese=False, **kwargs
):
    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.similarity.available_transformer()`.'
        )

    path = check_file(
        file=model,
        module='similarity',
        keys={
            'model': 'model.pb',
            'vocab': MODEL_VOCAB[model],
            'tokenizer': MODEL_BPE[model],
        },
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
        selected_model = bert_model
        if siamese:
            selected_node = 'import/bert/pooler/dense/BiasAdd:0'

    if model in ['xlnet', 'alxlnet']:
        selected_model = xlnet_model
        if siamese:
            selected_node = 'import/model_1/sequnece_summary/summary/BiasAdd:0'

    if not siamese:
        selected_node = _vectorizer_mapping[model]

    inputs = ['Placeholder', 'Placeholder_1', 'Placeholder_2']
    outputs = ['logits']
    tokenizer = SentencePieceTokenizer(vocab_file=path['vocab'], spm_model_file=path['tokenizer'])
    input_nodes, output_nodes = nodes_session(
        g, inputs, outputs, extra={'vectorizer': selected_node}
    )

    return selected_model(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        tokenizer=tokenizer,
        label=['not similar', 'similar'],
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
    result: model
        List of model classes:

        * if `bert` in model, will return `malaya.model.bert.SiameseBERT`.
        * if `xlnet` in model, will return `malaya.model.xlnet.SiameseXLNET`.
    """

    return _transformer(
        model=model,
        bert_model=SiameseBERT,
        xlnet_model=SiameseXLNET,
        quantized=quantized,
        siamese=True,
        **kwargs,
    )
