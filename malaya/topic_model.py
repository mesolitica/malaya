import numpy as np
from sklearn.utils import shuffle
from sklearn.decomposition import LatentDirichletAllocation
from malaya.model.lda2vec import LDA2Vec
from malaya.text.function import (
    simple_textcleaning,
    get_stopwords,
    classification_textcleaning,
    print_topics_modelling,
    build_dataset,
)
from malaya.function import validator
from malaya.text.vectorizer import skipgrams
from malaya.text.ngram import ngrams as ngrams_generator
from herpetologist import check_type
from typing import List, Tuple, Callable


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def _softmax_2d(x):
    y = x - x.max(axis=1, keepdims=True)
    np.exp(y, out=y)
    y /= y.sum(axis=1, keepdims=True)
    return y


def _prob_words(context, vocab, temperature=1.0):
    dot = np.dot(vocab, context)
    prob = _softmax(dot / temperature)
    return prob


def _prepare_topics(
    weights,
    factors,
    word_vectors,
    vocab,
    temperature=1.0,
    doc_lengths=None,
    term_frequency=None,
    normalize=False,
):
    topic_to_word = []
    msg = 'Vocabulary size did not match size of word vectors'
    if not len(vocab) == word_vectors.shape[0]:
        raise ValueError(msg)
    if normalize:
        word_vectors /= np.linalg.norm(word_vectors, axis=1)[:, None]
    for factor_vector in factors:
        factor_to_word = _prob_words(
            factor_vector, word_vectors, temperature=temperature
        )
        topic_to_word.append(np.ravel(factor_to_word))
    topic_to_word = np.array(topic_to_word)
    msg = 'Not all rows in topic_to_word sum to 1'
    if not np.allclose(np.sum(topic_to_word, axis=1), 1):
        raise ValueError(msg)
    doc_to_topic = _softmax_2d(weights)
    msg = 'Not all rows in doc_to_topic sum to 1'
    if not np.allclose(np.sum(doc_to_topic, axis=1), 1):
        raise ValueError(msg)
    data = {
        'topic_term_dists': topic_to_word,
        'doc_topic_dists': doc_to_topic,
        'doc_lengths': doc_lengths,
        'vocab': vocab,
        'term_frequency': term_frequency,
    }
    return data


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
        return print_topics_modelling(
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


class DeepTopic:
    def __init__(
        self,
        model,
        dictionary,
        reversed_dictionary,
        freqs,
        len_idx_text_clean,
        corpus,
    ):
        self._model = model
        self._dictionary = dictionary
        self._reversed_dictionary = reversed_dictionary
        self._doc_embed = model.sess.run(model.doc_embedding)
        self._topic_embed = model.sess.run(model.topic_embedding)
        self._word_embed = model.sess.run(model.word_embedding)
        self._components = self._topic_embed.dot(self._word_embed.T)
        self._freqs = freqs
        self._doc_len = len_idx_text_clean
        self._corpus = corpus
        self._features = []
        for i in range(len(self._dictionary)):
            self._features.append(self._reversed_dictionary[i])

    @check_type
    def visualize_topics(self, notebook_mode: int = False, mds: str = 'pcoa'):
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
        try:
            import pyLDAvis
            import pyLDAvis.sklearn
        except BaseException:
            raise ModuleNotFoundError(
                'pyldavis not installed. Please install it by `pip install pyldavis` and try again.'
            )

        if notebook_mode:
            pyLDAvis.enable_notebook()

        vis_data = _prepare_topics(
            self._doc_embed,
            self._topic_embed,
            self._word_embed,
            np.array(self._features),
            doc_lengths=self._doc_len,
            term_frequency=self._freqs,
            normalize=True,
        )
        prepared_vis_data = pyLDAvis.prepare(**vis_data)
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
        return print_topics_modelling(
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
        if not (k < self._doc_embed.shape[1] and k >= 0):
            raise ValueError('k should be between 0 and n_topics')
        reverse_sorted = np.argsort(self._doc_embed[:, k])[::-1]
        return [self._corpus[i] for i in reverse_sorted[:len_sentence]]


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
        return print_topics_modelling(
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


def available_vectorizer():
    """
    List available vectorizer topic modeling.
    """
    from malaya.function import describe_availability

    return describe_availability(_vectorizer_availability)


@check_type
def sklearn(
    corpus: List[str],
    model,
    vectorizer,
    n_topics: int,
    cleaning=simple_textcleaning,
    stopwords=get_stopwords,
    **kwargs,
):
    """
    Train a SKlearn model to do topic modelling based on corpus / list of strings given.

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
    result: malaya.topic_modelling.Topic class
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

    if cleaning:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])

    tf = vectorizer.fit_transform(corpus)
    tf_features = vectorizer.get_feature_names()
    compose = model(n_topics).fit(tf)
    return Topic(
        tf_features, compose, corpus, compose.transform(tf), vectorizer, tf
    )


@check_type
def lda2vec(
    corpus: List[str],
    vectorizer,
    n_topics: int = 10,
    cleaning=simple_textcleaning,
    stopwords=get_stopwords,
    window_size: int = 2,
    embedding_size: int = 128,
    epoch: int = 10,
    switch_loss: int = 1000,
    **kwargs,
):
    """
    Train a LDA2Vec model to do topic modelling based on corpus / list of strings given.

    Parameters
    ----------
    corpus: list
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
    embedding_size: int, (default=128)
        embedding size of lda2vec tensors.
    epoch: int, (default=10)
        training iteration, how many loop need to train.
    switch_loss: int, (default=3)
        baseline to switch from document based loss to document + word based loss.

    Returns
    -------
    result: malaya.topic_modelling.DeepTopic class
    """
    validator.validate_function(cleaning, 'cleaning')
    stopwords = validator.validate_stopwords(stopwords)
    stopwords = list(stopwords)

    tf_vectorizer = vectorizer

    if cleaning:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    text_clean = []
    for text in corpus:
        text_clean.append(
            ' '.join([word for word in text.split() if word not in stopwords])
        )
    tf_vectorizer.fit(text_clean)
    idx_text_clean, len_idx_text_clean = [], []
    transformed_text_clean = tf_vectorizer.transform(text_clean)
    for text in transformed_text_clean:
        splitted = text.nonzero()[1]
        idx_text_clean.append(splitted)
        len_idx_text_clean.append(len(splitted))
    dictionary = {
        i: no for no, i in enumerate(tf_vectorizer.get_feature_names())
    }
    reversed_dictionary = {
        no: i for no, i in enumerate(tf_vectorizer.get_feature_names())
    }
    freqs = transformed_text_clean.toarray().sum(axis=0).tolist()
    doc_ids = np.arange(len(idx_text_clean))
    num_unique_documents = doc_ids.max()
    pivot_words, target_words, doc_ids = [], [], []
    for i, t in enumerate(idx_text_clean):
        pairs, _ = skipgrams(
            t,
            vocabulary_size=len(dictionary),
            window_size=window_size,
            shuffle=True,
            negative_samples=0,
        )
        for pair in pairs:
            temp_data = pair
            pivot_words.append(temp_data[0])
            target_words.append(temp_data[1])
            doc_ids.append(i)
    pivot_words, target_words, doc_ids = shuffle(
        pivot_words, target_words, doc_ids, random_state=10
    )
    num_unique_documents = len(idx_text_clean)

    model = LDA2Vec(
        num_unique_documents,
        len(dictionary),
        n_topics,
        freqs,
        embedding_size=embedding_size,
        **kwargs,
    )
    model.train(
        pivot_words, target_words, doc_ids, epoch, switch_loss=switch_loss
    )
    return DeepTopic(
        model,
        dictionary,
        reversed_dictionary,
        freqs,
        len_idx_text_clean,
        text_clean,
    )


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
    Use attention from transformer model to do topic modelling based on corpus / list of strings given.

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
    result: malaya.topic_modelling.AttentionTopic class
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

    from sklearn.cluster import KMeans

    if cleaning:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])

    def generate_ngram(seq, ngram=(1, 3)):
        g = []
        for i in range(ngram[0], ngram[-1] + 1):
            g.extend(list(ngrams_generator(seq, i)))
        return g

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
