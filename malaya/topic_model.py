import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

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

    def visualize_topics(self, notebook_mode = False, mds = 'pcoa'):
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
        if not isinstance(mds, str):
            raise ValueError('mds must be a string')
        if not isinstance(notebook_mode, bool):
            raise ValueError('notebook_mode must be a boolean')
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

    def top_topics(self, len_topic, top_n = 10, return_df = True):
        """
        Print important topics based on decomposition.

        Parameters
        ----------
        len_topic: int
        """
        if not isinstance(len_topic, int):
            raise ValueError('len_topic must be an integer')
        if not isinstance(top_n, int):
            raise ValueError('top_n must be an integer')
        if not isinstance(return_df, bool):
            raise ValueError('return_df must be a boolean')
        return print_topics_modelling(
            len_topic,
            feature_names = np.array(self._features),
            sorting = np.argsort(self._components)[:, ::-1],
            n_words = top_n,
            return_df = return_df,
        )

    def get_topics(self, len_topic):
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

    def get_sentences(self, len_sentence, k = 0):
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
        if not isinstance(len_sentence, int):
            raise ValueError('len_sentence must be an integer')
        if not isinstance(k, int):
            raise ValueError('k must be an integer')
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

    def visualize_topics(self, notebook_mode = False, mds = 'pcoa'):
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
        if not isinstance(mds, str):
            raise ValueError('mds must be a string')
        if not isinstance(notebook_mode, bool):
            raise ValueError('notebook_mode must be a boolean')
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

    def top_topics(self, len_topic, top_n = 10, return_df = True):
        """
        Print important topics based on decomposition.

        Parameters
        ----------
        len_topic: int
        """
        if not isinstance(len_topic, int):
            raise ValueError('len_topic must be an integer')
        if not isinstance(return_df, bool):
            raise ValueError('return_df must be a boolean')
        return print_topics_modelling(
            len_topic,
            feature_names = np.array(self.features),
            sorting = np.argsort(self.comp.components_)[:, ::-1],
            n_words = top_n,
            return_df = return_df,
        )

    def get_topics(self, len_topic):
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

    def get_sentences(self, len_sentence, k = 0):
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
        if not isinstance(len_sentence, int):
            raise ValueError('len_sentence must be an integer')
        if not isinstance(k, int):
            raise ValueError('k must be an integer')
        if not (k < self.transformed.shape[1] and k >= 0):
            raise ValueError('k should be between 0 and n_topics')
        reverse_sorted = np.argsort(self.transformed[:, k])[::-1]
        return [self.corpus[i] for i in reverse_sorted[:len_sentence]]


def _base_topic_modelling(
    corpus,
    n_topics,
    decomposition,
    max_df = 0.95,
    min_df = 2,
    ngram = (1, 3),
    vectorizer = 'bow',
    stemming = True,
    cleaning = simple_textcleaning,
    stop_words = None,
    **kwargs,
):
    if not isinstance(corpus, list):
        raise ValueError('corpus must be a list')
    if not isinstance(corpus[0], str):
        raise ValueError('corpus must be list of strings')
    if not isinstance(n_topics, int):
        raise ValueError('n_topics must be an integer')
    if not isinstance(vectorizer, str):
        raise ValueError('vectorizer must be a string')
    if not isinstance(stemming, bool):
        raise ValueError('bool must be a boolean')
    vectorizer = vectorizer.lower()
    if not vectorizer in ['tfidf', 'bow', 'skip-gram']:
        raise ValueError("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")
    if not isinstance(ngram, tuple):
        raise ValueError('ngram must be a tuple')
    if not len(ngram) == 2:
        raise ValueError('ngram size must equal to 2')
    if not isinstance(min_df, int):
        raise ValueError('min_df must be an integer')
    if not (isinstance(max_df, int) or isinstance(max_df, float)):
        raise ValueError('max_df must be an integer or a float')
    if min_df < 1:
        raise ValueError('min_df must be bigger than 0')
    if not (max_df <= 1 and max_df > 0):
        raise ValueError(
            'max_df must be bigger than 0, less than or equal to 1'
        )

    if cleaning is not None:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)):
            corpus[i] = sastrawi(corpus[i])
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
    stemming = True,
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
    stemming: bool, (default=True)
        If True, sastrawi_stemmer will apply.
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
    stemming = True,
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
    stemming: bool, (default=True)
        If True, sastrawi_stemmer will apply.
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
    stemming = True,
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
    stemming: bool, (default=True)
        If True, sastrawi_stemmer will apply.
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


def lda2vec(
    corpus,
    n_topics,
    stemming = True,
    max_df = 0.95,
    min_df = 2,
    ngram = (1, 3),
    cleaning = simple_textcleaning,
    vectorizer = 'bow',
    stop_words = None,
    window_size = 2,
    embedding_size = 128,
    epoch = 10,
    switch_loss = 3,
    skip = 5,
    **kwargs,
):
    """
    Train a LDA2Vec model to do topic modelling based on corpus / list of strings given.

    Parameters
    ----------
    corpus: list
    n_topics: int, (default=10)
        size of decomposition column.
    stemming: bool, (default=True)
        If True, sastrawi_stemmer will apply.
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
    if not isinstance(corpus, list):
        raise ValueError('corpus must be a list')
    if not isinstance(corpus[0], str):
        raise ValueError('corpus must be list of strings')
    if not isinstance(n_topics, int):
        raise ValueError('n_topics must be an integer')
    if not isinstance(vectorizer, str):
        raise ValueError('vectorizer must be a string')
    if not isinstance(stemming, bool):
        raise ValueError('bool must be a boolean')
    vectorizer = vectorizer.lower()
    if not vectorizer in ['tfidf', 'bow', 'skip-gram']:
        raise ValueError("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")
    if not isinstance(ngram, tuple):
        raise ValueError('ngram must be a tuple')
    if not len(ngram) == 2:
        raise ValueError('ngram size must equal to 2')
    if not isinstance(min_df, int):
        raise ValueError('min_df must be an integer')
    if not (isinstance(max_df, int) or isinstance(max_df, float)):
        raise ValueError('max_df must be an integer or a float')
    if min_df < 1:
        raise ValueError('min_df must be bigger than 0')
    if not (max_df <= 1 and max_df > 0):
        raise ValueError(
            'max_df must be bigger than 0, less than or equal to 1'
        )
    if not isinstance(embedding_size, int):
        raise ValueError('embedding_size must be an integer')
    if not isinstance(window_size, int):
        raise ValueError('window_size must be an integer')
    if not isinstance(epoch, int):
        raise ValueError('epoch must be an integer')
    if not isinstance(switch_loss, int):
        raise ValueError('switch_loss must be an integer')
    if not isinstance(skip, int):
        raise ValueError('skip must be an integer')

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

    if cleaning is not None:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)):
            corpus[i] = sastrawi(corpus[i])
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
