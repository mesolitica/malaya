import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from malaya.text.function import (
    simple_textcleaning,
    get_stopwords,
    print_topics_modeling,
)
from malaya.function import validator
from herpetologist import check_type
from typing import List, Tuple


class Topic:
    def __init__(
        self, features, comp, corpus, transformed, vectorizer, vectors
    ):
        self.features = features
        self.comp = comp
        self.corpus = corpus
        self.transformed = transformed
        self.vectorizer = vectorizer
        self._vectors = vectors

    @check_type
    def visualize_topics(self, notebook_mode: bool = False, mds: str = 'pcoa'):
        """
        Print important topics based on decomposition.

        Parameters
        ----------
        mds : str, optional (default='pcoa')
            2D Decomposition. Allowed values:

            * ``'pcoa'`` - Dimension reduction via Jensen-Shannon Divergence & Principal Coordinate Analysis (aka Classical Multidimensional Scaling)
            * ``'mmds'`` - Dimension reduction via Multidimensional scaling
            * ``'tsne'`` - Dimension reduction via t-distributed stochastic neighbor embedding
        """

        if not isinstance(self.comp, LatentDirichletAllocation):
            raise ValueError('only support LatentDirichletAllocation model.')

        import pyLDAvis
        import pyLDAvis.sklearn

        if notebook_mode:
            pyLDAvis.enable_notebook()

        prepared_vis_data = pyLDAvis.sklearn.prepare(
            self.comp, self._vectors, self.vectorizer, mds=mds
        )
        if notebook_mode:
            return prepared_vis_data
        else:
            pyLDAvis.show(prepared_vis_data)

    @check_type
    def top_topics(
        self, len_topic: int, top_n: int = 10, return_df: bool = True
    ):
        """
        Print important topics based on decomposition.

        Parameters
        ----------
        len_topic: int
            size of topics.
        top_n: int, optional (default=10)
            top n of each topic.
        return_df: bool, optional (default=True)
            return as pandas.DataFrame, else JSON.
        """
        return print_topics_modeling(
            len_topic,
            feature_names=np.array(self.features),
            sorting=np.argsort(self.comp.components_)[:, ::-1],
            n_words=top_n,
            return_df=return_df,
        )

    @check_type
    def get_topics(self, len_topic: int):
        """
        Return important topics based on decomposition.

        Parameters
        ----------
        len_topic: int

        Returns
        -------
        result: List[str]
        """
        if not isinstance(len_topic, int):
            raise ValueError('len_topic must be an integer')
        results = []
        for no, topic in enumerate(self.comp.components_):
            results.append(
                (
                    no,
                    ' '.join(
                        [
                            self.features[i]
                            for i in topic.argsort()[: -len_topic - 1: -1]
                        ]
                    ),
                )
            )
        return results

    @check_type
    def get_sentences(self, len_sentence: int, k: int = 0):
        """
        Return important sentences related to selected column based on decomposition.

        Parameters
        ----------
        len_sentence: int
        k: int, (default=0)
            index of decomposition matrix.

        Returns
        -------
        result: List[str]
        """
        if not (k < self.transformed.shape[1] and k >= 0):
            raise ValueError('k should be between 0 and n_topics')
        reverse_sorted = np.argsort(self.transformed[:, k])[::-1]
        return [self.corpus[i] for i in reverse_sorted[:len_sentence]]


@check_type
def fit(
    corpus: List[str],
    model,
    vectorizer,
    n_topics: int,
    cleaning=simple_textcleaning,
    stopwords=get_stopwords,
    **kwargs,
):
    """
    Train a SKlearn model to do topic modelling based on corpus given.

    Parameters
    ----------
    corpus: list
    model : object
        Should have `fit_transform` method. Commonly:

        * ``sklearn.decomposition.TruncatedSVD`` - LSA algorithm.
        * ``sklearn.decomposition.LatentDirichletAllocation`` - LDA algorithm.
        * ``sklearn.decomposition.NMF`` - NMF algorithm.
    vectorizer : object
        Should have `fit_transform` method. Commonly:

        * ``sklearn.feature_extraction.text.TfidfVectorizer`` - TFIDF algorithm.
        * ``sklearn.feature_extraction.text.CountVectorizer`` - Bag-of-Word algorithm.
        * ``malaya.text.vectorizer.SkipGramCountVectorizer`` - Skip Gram Bag-of-Word algorithm.
        * ``malaya.text.vectorizer.SkipGramTfidfVectorizer`` - Skip Gram TFIDF algorithm.
    n_topics: int, (default=10)
        size of decomposition column.
    cleaning: function, (default=malaya.text.function.simple_textcleaning)
        function to clean the corpus.
    stopwords: List[str], (default=malaya.texts.function.get_stopwords)
        A callable that returned a List[str], or a List[str], or a Tuple[str]

    Returns
    -------
    result: malaya.topic_model.decomposition.Topic class
    """
    stopwords = validator.validate_stopwords(stopwords)
    stopwords = list(stopwords)
    validator.validate_function(cleaning, 'cleaning')
    if not hasattr(vectorizer, 'fit_transform'):
        raise ValueError('vectorizer must have `fit_transform` method')

    if len(corpus) < n_topics:
        raise ValueError(
            'length corpus must be bigger than or equal to n_topics'
        )

    if cleaning is not None:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])

    tf = vectorizer.fit_transform(corpus)
    tf_features = vectorizer.get_feature_names()
    compose = model(n_topics).fit(tf)
    return Topic(
        tf_features, compose, corpus, compose.transform(tf), vectorizer, tf
    )
