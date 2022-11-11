import numpy as np
from malaya.text.function import (
    simple_textcleaning,
    get_stopwords,
    print_topics_modeling,
)
from sklearn.cluster import KMeans
from malaya.text.ngram import ngrams as ngrams_generator
from malaya.function import validator
from herpetologist import check_type
from typing import List, Tuple
import warnings


def generate_ngram(seq, ngram=(1, 3)):
    g = []
    for i in range(ngram[0], ngram[-1] + 1):
        g.extend(list(ngrams_generator(seq, i)))
    return g


class AttentionTopic:
    def __init__(self, features, components):
        self._features = features
        self._components = components

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
            feature_names=np.array(self._features),
            sorting=np.argsort(self._components)[:, ::-1],
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
            size of topics.

        Returns
        -------
        result: List[str]
        """
        results = []
        for no, topic in enumerate(self._components):
            results.append(
                (
                    no,
                    ' '.join(
                        [
                            self._features[i]
                            for i in topic.argsort()[: -len_topic - 1: -1]
                        ]
                    ),
                )
            )
        return results


@check_type
def attention(
    corpus: List[str],
    n_topics: int,
    vectorizer,
    cleaning=simple_textcleaning,
    stopwords=get_stopwords,
    ngram: Tuple[int, int] = (1, 3),
    batch_size: int = 10,
):
    """
    Use attention from malaya.transformer model to do topic modelling based on corpus given.

    Parameters
    ----------
    corpus: list
    n_topics: int, (default=10)
        size of decomposition column.
    vectorizer: object
    cleaning: function, (default=malaya.text.function.simple_textcleaning)
        function to clean the corpus.
    stopwords: List[str], (default=malaya.texts.function.get_stopwords)
        A callable that returned a List[str], or a List[str], or a Tuple[str]
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus.
    batch_size: int, (default=10)
        size of strings for each vectorization and attention.

    Returns
    -------
    result: malaya.topic_model.transformer.AttentionTopic class
    """

    stopwords = validator.validate_stopwords(stopwords)

    if not hasattr(vectorizer, 'attention') and not hasattr(
        vectorizer, 'vectorize'
    ):
        raise ValueError(
            'vectorizer must have `attention` and `vectorize` methods'
        )

    validator.validate_function(cleaning, 'cleaning')

    if len(corpus) < n_topics:
        raise ValueError(
            'length corpus must be bigger than or equal to n_topics'
        )

    if cleaning is not None:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])

    rows, attentions = [], []
    for i in range(0, len(corpus), batch_size):
        index = min(i + batch_size, len(corpus))
        rows.append(vectorizer.vectorize(corpus[i:index]))
        attentions.extend(vectorizer.attention(corpus[i:index]))

    concat = np.concatenate(rows, axis=0)
    kmeans = KMeans(n_clusters=n_topics, random_state=0).fit(concat)
    labels = kmeans.labels_

    overall, filtered_a = [], []
    for a in attentions:
        f = [i for i in a if i[0] not in stopwords]
        overall.extend(f)
        filtered_a.append(f)

    o_ngram = generate_ngram(overall, ngram)
    features = []
    for i in o_ngram:
        features.append(' '.join([w[0] for w in i]))
    features = list(set(features))

    components = np.zeros((n_topics, len(features)))
    for no, i in enumerate(labels):
        f = generate_ngram(filtered_a[no], ngram)
        for w in f:
            word = ' '.join([r[0] for r in w])
            score = np.mean([r[1] for r in w])
            if word in features:
                components[i, features.index(word)] += score

    return AttentionTopic(features, components)
