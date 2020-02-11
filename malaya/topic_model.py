import numpy as np
import collections
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from .stem import sastrawi
from ._models._lda2vec import LDA2VEC
from .texts._text_functions import (
    simple_textcleaning,
    STOPWORDS,
    classification_textcleaning,
    print_topics_modelling,
    build_dataset,
)
from .texts.vectorizer import skipgrams, SkipGramVectorizer
from .generator import ngrams as ngrams_generator
from herpetologist import check_type
from typing import List, Tuple, Callable


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def _softmax_2d(x):
    y = x - x.max(axis = 1, keepdims = True)
    np.exp(y, out = y)
    y /= y.sum(axis = 1, keepdims = True)
    return y


def _prob_words(context, vocab, temperature = 1.0):
    dot = np.dot(vocab, context)
    prob = _softmax(dot / temperature)
    return prob


def _prepare_topics(
    weights,
    factors,
    word_vectors,
    vocab,
    temperature = 1.0,
    doc_lengths = None,
    term_frequency = None,
    normalize = False,
):
    topic_to_word = []
    msg = 'Vocabulary size did not match size of word vectors'
    if not len(vocab) == word_vectors.shape[0]:
        raise ValueError(msg)
    if normalize:
        word_vectors /= np.linalg.norm(word_vectors, axis = 1)[:, None]
    for factor_vector in factors:
        factor_to_word = _prob_words(
            factor_vector, word_vectors, temperature = temperature
        )
        topic_to_word.append(np.ravel(factor_to_word))
    topic_to_word = np.array(topic_to_word)
    msg = 'Not all rows in topic_to_word sum to 1'
    if not np.allclose(np.sum(topic_to_word, axis = 1), 1):
        raise ValueError(msg)
    doc_to_topic = _softmax_2d(weights)
    msg = 'Not all rows in doc_to_topic sum to 1'
    if not np.allclose(np.sum(doc_to_topic, axis = 1), 1):
        raise ValueError(msg)
    data = {
        'topic_term_dists': topic_to_word,
        'doc_topic_dists': doc_to_topic,
        'doc_lengths': doc_lengths,
        'vocab': vocab,
        'term_frequency': term_frequency,
    }
    return data


class _ATTENTION_TOPIC:
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
        """
        return print_topics_modelling(
            len_topic,
            feature_names = np.array(self._features),
            sorting = np.argsort(self._components)[:, ::-1],
            n_words = top_n,
            return_df = return_df,
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
        results: list of strings
        """
        results = []
        for no, topic in enumerate(self._components):
            results.append(
                (
                    no,
                    ' '.join(
                        [
                            self._features[i]
                            for i in topic.argsort()[: -len_topic - 1 : -1]
                        ]
                    ),
                )
            )
        return results


class _DEEP_TOPIC:
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
        except:
            raise Exception(
                'pyldavis not installed. Please install it and try again.'
            )

        if notebook_mode:
            pyLDAvis.enable_notebook()

        vis_data = _prepare_topics(
            self._doc_embed,
            self._topic_embed,
            self._word_embed,
            np.array(self._features),
            doc_lengths = self._doc_len,
            term_frequency = self._freqs,
            normalize = True,
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
        """
        return print_topics_modelling(
            len_topic,
            feature_names = np.array(self._features),
            sorting = np.argsort(self._components)[:, ::-1],
            n_words = top_n,
            return_df = return_df,
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
        results: list of strings
        """
        results = []
        for no, topic in enumerate(self._components):
            results.append(
                (
                    no,
                    ' '.join(
                        [
                            self._features[i]
                            for i in topic.argsort()[: -len_topic - 1 : -1]
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
        results: list of strings
        """
        if not (k < self._doc_embed.shape[1] and k >= 0):
            raise ValueError('k should be between 0 and n_topics')
        reverse_sorted = np.argsort(self._doc_embed[:, k])[::-1]
        return [self._corpus[i] for i in reverse_sorted[:len_sentence]]


class _TOPIC:
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
            raise ValueError('only support lda_topic_modelling()')

        import pyLDAvis
        import pyLDAvis.sklearn

        if notebook_mode:
            pyLDAvis.enable_notebook()

        prepared_vis_data = pyLDAvis.sklearn.prepare(
            self.comp, self._vectors, self.vectorizer, mds = mds
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
        """
        return print_topics_modelling(
            len_topic,
            feature_names = np.array(self.features),
            sorting = np.argsort(self.comp.components_)[:, ::-1],
            n_words = top_n,
            return_df = return_df,
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
        results: list of strings
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
                            for i in topic.argsort()[: -len_topic - 1 : -1]
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
        results: list of strings
        """
        if not (k < self.transformed.shape[1] and k >= 0):
            raise ValueError('k should be between 0 and n_topics')
        reverse_sorted = np.argsort(self.transformed[:, k])[::-1]
        return [self.corpus[i] for i in reverse_sorted[:len_sentence]]


@check_type
def _base_topic_modelling(
    corpus: List[str],
    n_topics: int,
    decomposition,
    max_df: float = 0.95,
    min_df: int = 2,
    ngram: Tuple[int, int] = (1, 3),
    vectorizer: str = 'bow',
    stemming = sastrawi,
    cleaning: Callable = simple_textcleaning,
    stop_words: List[str] = None,
    **kwargs,
):
    if not isinstance(stemming, collections.Callable) and stemming is not None:
        raise ValueError('stemming must be a callable type or None')
    vectorizer = vectorizer.lower()
    if not vectorizer in ['tfidf', 'bow', 'skip-gram']:
        raise ValueError("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")
    if min_df < 1:
        raise ValueError('min_df must be bigger than 0')
    if not (max_df <= 1 and max_df > 0):
        raise ValueError(
            'max_df must be bigger than 0, less than or equal to 1'
        )
    if len(corpus) < n_topics:
        raise ValueError(
            'length corpus must be bigger than or equal to n_topics'
        )

    if cleaning:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)):
            corpus[i] = stemming(corpus[i])
    if vectorizer == 'tfidf':
        Vectorizer = TfidfVectorizer
    elif vectorizer == 'bow':
        Vectorizer = CountVectorizer
    elif vectorizer == 'skip-gram':
        Vectorizer = SkipGramVectorizer
    else:
        raise Exception("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")
    tf_vectorizer = Vectorizer(
        max_df = max_df,
        min_df = min_df,
        ngram_range = ngram,
        stop_words = stop_words,
        **kwargs,
    )
    tf = tf_vectorizer.fit_transform(corpus)
    tf_features = tf_vectorizer.get_feature_names()
    compose = decomposition(n_topics).fit(tf)
    return _TOPIC(
        tf_features,
        compose,
        [classification_textcleaning(c) for c in corpus],
        compose.transform(tf),
        tf_vectorizer,
        tf,
    )


def lda(
    corpus,
    n_topics = 10,
    max_df = 0.95,
    min_df = 2,
    ngram = (1, 3),
    stemming = sastrawi,
    vectorizer = 'bow',
    cleaning = simple_textcleaning,
    stop_words = None,
    **kwargs,
):
    """
    Train a LDA model to do topic modelling based on corpus / list of strings given.

    Parameters
    ----------
    corpus: list
    n_topics: int, (default=10)
        size of decomposition column.
    max_df: float, (default=0.95)
        maximum of a word selected based on document frequency.
    min_df: int, (default=2)
        minimum of a word selected on based on document frequency.
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus.
    stemming: function, (default=sastrawi)
        function to stem the corpus.
    vectorizer: str, (default='bow')
        vectorizer technique. Allowed values:

        * ``'bow'`` - Bag of Word.
        * ``'tfidf'`` - Term frequency inverse Document Frequency.
        * ``'skip-gram'`` - Bag of Word with skipping certain n-grams.
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus.
    stop_words: list, (default=None)
        list of stop words to remove. If None, default is malaya.texts._text_functions.STOPWORDS

    Returns
    -------
    _TOPIC: malaya.topic_modelling._TOPIC class
    """
    if stop_words is None:
        stop_words = STOPWORDS
    return _base_topic_modelling(
        corpus,
        n_topics,
        LatentDirichletAllocation,
        max_df = max_df,
        min_df = min_df,
        ngram = ngram,
        vectorizer = vectorizer,
        stemming = stemming,
        cleaning = cleaning,
        stop_words = stop_words,
        **kwargs,
    )


def nmf(
    corpus,
    n_topics = 10,
    max_df = 0.95,
    min_df = 2,
    ngram = (1, 3),
    stemming = sastrawi,
    vectorizer = 'bow',
    cleaning = simple_textcleaning,
    stop_words = None,
    **kwargs,
):
    """
    Train a NMF model to do topic modelling based on corpus / list of strings given.

    Parameters
    ----------
    corpus: list
    n_topics: int, (default=10)
        size of decomposition column.
    max_df: float, (default=0.95)
        maximum of a word selected based on document frequency.
    min_df: int, (default=2)
        minimum of a word selected on based on document frequency.
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus.
    stemming: function, (default=sastrawi)
        function to stem the corpus.
    vectorizer: str, (default='bow')
        vectorizer technique. Allowed values:

        * ``'bow'`` - Bag of Word.
        * ``'tfidf'`` - Term frequency inverse Document Frequency.
        * ``'skip-gram'`` - Bag of Word with skipping certain n-grams.
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus.
    stop_words: list, (default=None)
        list of stop words to remove. If None, default is malaya.texts._text_functions.STOPWORDS

    Returns
    -------
    _TOPIC: malaya.topic_modelling._TOPIC class
    """
    if stop_words is None:
        stop_words = STOPWORDS
    return _base_topic_modelling(
        corpus,
        n_topics,
        NMF,
        max_df = max_df,
        min_df = min_df,
        ngram = ngram,
        vectorizer = vectorizer,
        stemming = stemming,
        cleaning = cleaning,
        stop_words = stop_words,
        **kwargs,
    )


def lsa(
    corpus,
    n_topics,
    max_df = 0.95,
    min_df = 2,
    ngram = (1, 3),
    vectorizer = 'bow',
    stemming = sastrawi,
    cleaning = simple_textcleaning,
    stop_words = None,
    **kwargs,
):
    """
    Train a LSA model to do topic modelling based on corpus / list of strings given.

    Parameters
    ----------
    corpus: list
    n_topics: int, (default=10)
        size of decomposition column.
    max_df: float, (default=0.95)
        maximum of a word selected based on document frequency.
    min_df: int, (default=2)
        minimum of a word selected on based on document frequency.
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus.
    vectorizer: str, (default='bow')
        vectorizer technique. Allowed values:

        * ``'bow'`` - Bag of Word.
        * ``'tfidf'`` - Term frequency inverse Document Frequency.
        * ``'skip-gram'`` - Bag of Word with skipping certain n-grams.
    stemming: function, (default=sastrawi)
        function to stem the corpus.
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus.
    stop_words: list, (default=None)
        list of stop words to remove. If None, default is malaya.texts._text_functions.STOPWORDS

    Returns
    -------
    _TOPIC: malaya.topic_modelling._TOPIC class
    """
    if stop_words is None:
        stop_words = STOPWORDS
    return _base_topic_modelling(
        corpus,
        n_topics,
        TruncatedSVD,
        max_df = max_df,
        min_df = min_df,
        ngram = ngram,
        vectorizer = vectorizer,
        stemming = stemming,
        cleaning = cleaning,
        stop_words = stop_words,
        **kwargs,
    )


@check_type
def lda2vec(
    corpus: List[str],
    n_topics: int,
    stemming = sastrawi,
    max_df: float = 0.95,
    min_df: int = 2,
    ngram: Tuple[int, int] = (1, 3),
    cleaning: Callable = simple_textcleaning,
    vectorizer: str = 'bow',
    stop_words: List[str] = None,
    window_size: int = 2,
    embedding_size: int = 128,
    epoch: int = 10,
    switch_loss: int = 3,
    skip: int = 5,
    **kwargs,
):
    """
    Train a LDA2Vec model to do topic modelling based on corpus / list of strings given.

    Parameters
    ----------
    corpus: list
    n_topics: int, (default=10)
        size of decomposition column.
    stemming: function, (default=sastrawi)
        function to stem the corpus.
    max_df: float, (default=0.95)
        maximum of a word selected based on document frequency.
    min_df: int, (default=2)
        minimum of a word selected on based on document frequency.
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus.
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus.
    stop_words: list, (default=None)
        list of stop words to remove. If None, default is malaya.texts._text_functions.STOPWORDS
    embedding_size: int, (default=128)
        embedding size of lda2vec tensors.
    training_iteration: int, (default=10)
        training iteration, how many loop need to train.
    switch_loss: int, (default=3)
        baseline to switch from document based loss to document + word based loss.
    vectorizer: str, (default='bow')
        vectorizer technique. Allowed values:

        * ``'bow'`` - Bag of Word.
        * ``'tfidf'`` - Term frequency inverse Document Frequency.
        * ``'skip-gram'`` - Bag of Word with skipping certain n-grams.
    skip: int, (default=5)
        skip value if vectorizer = 'skip-gram'

    Returns
    -------
    _DEEP_TOPIC: malaya.topic_modelling._DEEP_TOPIC class
    """
    if not isinstance(stemming, collections.Callable) and stemming is not None:
        raise ValueError('stemming must be a callable type or None')

    vectorizer = vectorizer.lower()
    if not vectorizer in ['tfidf', 'bow', 'skip-gram']:
        raise ValueError("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")

    if min_df < 1:
        raise ValueError('min_df must be bigger than 0')
    if not (max_df <= 1 and max_df > 0):
        raise ValueError(
            'max_df must be bigger than 0, less than or equal to 1'
        )

    if vectorizer == 'tfidf':
        Vectorizer = TfidfVectorizer
    elif vectorizer == 'bow':
        Vectorizer = CountVectorizer
    elif vectorizer == 'skip-gram':
        Vectorizer = SkipGramVectorizer
    else:
        raise Exception("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")
    tf_vectorizer = Vectorizer(
        ngram_range = ngram,
        min_df = min_df,
        max_df = max_df,
        stop_words = stop_words,
    )
    if stop_words is None:
        stop_words = STOPWORDS

    if cleaning:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)):
            corpus[i] = stemming(corpus[i])
    text_clean = []
    for text in corpus:
        text_clean.append(
            ' '.join([word for word in text.split() if word not in stop_words])
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
    freqs = transformed_text_clean.toarray().sum(axis = 0).tolist()
    doc_ids = np.arange(len(idx_text_clean))
    num_unique_documents = doc_ids.max()
    pivot_words, target_words, doc_ids = [], [], []
    for i, t in enumerate(idx_text_clean):
        pairs, _ = skipgrams(
            t,
            vocabulary_size = len(dictionary),
            window_size = window_size,
            shuffle = True,
            negative_samples = 0,
        )
        for pair in pairs:
            temp_data = pair
            pivot_words.append(temp_data[0])
            target_words.append(temp_data[1])
            doc_ids.append(i)
    pivot_words, target_words, doc_ids = shuffle(
        pivot_words, target_words, doc_ids, random_state = 10
    )
    num_unique_documents = len(idx_text_clean)

    model = LDA2VEC(
        num_unique_documents,
        len(dictionary),
        n_topics,
        freqs,
        embedding_size = embedding_size,
        **kwargs,
    )
    model.train(
        pivot_words, target_words, doc_ids, epoch, switch_loss = switch_loss
    )
    return _DEEP_TOPIC(
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
    stemming = sastrawi,
    cleaning: Callable = simple_textcleaning,
    stop_words: List[str] = None,
    ngram: Tuple[int, int] = (1, 3),
    batch_size: int = 10,
):

    """
    Use attention from vectorizer model to do topic modelling based on corpus / list of strings given.

    Parameters
    ----------
    corpus: list
    n_topics: int, (default=10)
        size of decomposition column.
    vectorizer: object
    stemming: function, (default=sastrawi)
        function to stem the corpus.
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus.
    stop_words: list, (default=None)
        list of stop words to remove. If None, default is malaya.texts._text_functions.STOPWORDS
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus.
    batch_size: int, (default=10)
        size of strings for each vectorization and attention.

    Returns
    -------
    _ATTENTION_TOPIC: malaya.topic_modelling._ATTENTION_TOPIC class
    """

    if not hasattr(vectorizer, 'attention') and not hasattr(
        vectorizer, 'vectorize'
    ):
        raise ValueError(
            'vectorizer must has `attention` and `vectorize` methods'
        )
    if not isinstance(stemming, collections.Callable) and stemming is not None:
        raise ValueError('stemming must be a callable type or None')
    if len(corpus) < n_topics:
        raise ValueError(
            'length corpus must be bigger than or equal to n_topics'
        )

    from sklearn.cluster import KMeans

    if stop_words is None:
        stop_words = STOPWORDS

    if cleaning:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)):
            corpus[i] = stemming(corpus[i])

    def generate_ngram(seq, ngram = (1, 3)):
        g = []
        for i in range(ngram[0], ngram[-1] + 1):
            g.extend(list(ngrams_generator(seq, i)))
        return g

    rows, attentions = [], []
    for i in range(0, len(corpus), batch_size):
        index = min(i + batch_size, len(corpus))
        rows.append(vectorizer.vectorize(corpus[i:index]))
        attentions.extend(vectorizer.attention(corpus[i:index]))

    concat = np.concatenate(rows, axis = 0)
    kmeans = KMeans(n_clusters = n_topics, random_state = 0).fit(concat)
    labels = kmeans.labels_

    overall, filtered_a = [], []
    for a in attentions:
        f = [i for i in a if i[0] not in stop_words]
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

    return _ATTENTION_TOPIC(features, components)
