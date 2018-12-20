import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import numpy as np
import collections
import pandas as pd
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from .stemmer import sastrawi_stemmer
from .lda2vec import LDA2VEC
from .text_functions import (
    simple_textcleaning,
    STOPWORDS,
    classification_textcleaning,
    print_topics_modelling,
    skipgrams,
    build_dataset,
)


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def _softmax_2d(x):
    y = x - x.max(axis = 1, keepdims = True)
    np.exp(y, out = y)
    y /= y.sum(axis = 1, keepdims = True)
    return y


def prob_words(context, vocab, temperature = 1.0):
    """ This calculates a softmax over the vocabulary as a function
    of the dot product of context and word.
    """
    dot = np.dot(vocab, context)
    prob = _softmax(dot / temperature)
    return prob


def prepare_topics(
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
    assert len(vocab) == word_vectors.shape[0], msg
    if normalize:
        word_vectors /= np.linalg.norm(word_vectors, axis = 1)[:, None]
    for factor_vector in factors:
        factor_to_word = prob_words(
            factor_vector, word_vectors, temperature = temperature
        )
        topic_to_word.append(np.ravel(factor_to_word))
    topic_to_word = np.array(topic_to_word)
    msg = 'Not all rows in topic_to_word sum to 1'
    assert np.allclose(np.sum(topic_to_word, axis = 1), 1), msg
    doc_to_topic = _softmax_2d(weights)
    msg = 'Not all rows in doc_to_topic sum to 1'
    assert np.allclose(np.sum(doc_to_topic, axis = 1), 1), msg
    data = {
        'topic_term_dists': topic_to_word,
        'doc_topic_dists': doc_to_topic,
        'doc_lengths': doc_lengths,
        'vocab': vocab,
        'term_frequency': term_frequency,
    }
    return data


class DEEP_TOPIC:
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
        assert isinstance(mds, str), 'mds must be a string'
        assert isinstance(
            notebook_mode, bool
        ), 'notebook_mode must be a boolean'
        import pyLDAvis
        import pyLDAvis.sklearn

        if notebook_mode:
            pyLDAvis.enable_notebook()

        vis_data = prepare_topics(
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

    def print_topics(self, len_topic):
        """
        Print important topics based on decomposition.

        Parameters
        ----------
        len_topic: int
        """
        assert isinstance(len_topic, int), 'len_topic must be an integer'
        print_topics_modelling(
            range(len_topic),
            feature_names = np.array(self._features),
            sorting = np.argsort(self._components)[:, ::-1],
            topics_per_chunk = 5,
            n_words = 10,
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
        assert isinstance(len_topic, int), 'len_topic must be an integer'
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
        assert isinstance(len_sentence, int), 'len_sentence must be an integer'
        assert isinstance(k, int), 'k must be an integer'
        assert (
            k < self._doc_embed.shape[1] and k >= 0
        ), 'k should be between 0 and n_topics'
        reverse_sorted = np.argsort(self._doc_embed[:, k])[::-1]
        return [self._corpus[i] for i in reverse_sorted[:len_sentence]]


class TOPIC:
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
        assert isinstance(mds, str), 'mds must be a string'
        assert isinstance(
            notebook_mode, bool
        ), 'notebook_mode must be a boolean'
        assert isinstance(
            self.comp, LatentDirichletAllocation
        ), 'only support lda_topic_modelling()'
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

    def print_topics(self, len_topic):
        """
        Print important topics based on decomposition.

        Parameters
        ----------
        len_topic: int
        """
        assert isinstance(len_topic, int), 'len_topic must be an integer'
        print_topics_modelling(
            range(len_topic),
            feature_names = np.array(self.features),
            sorting = np.argsort(self.comp.components_)[:, ::-1],
            topics_per_chunk = 5,
            n_words = 10,
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
        assert isinstance(len_topic, int), 'len_topic must be an integer'
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
        assert isinstance(len_sentence, int), 'len_sentence must be an integer'
        assert isinstance(k, int), 'k must be an integer'
        assert (
            k < self.transformed.shape[1] and k >= 0
        ), 'k should be between 0 and n_topics'
        reverse_sorted = np.argsort(self.transformed[:, k])[::-1]
        return [self.corpus[i] for i in reverse_sorted[:len_sentence]]


def _base_topic_modelling(
    corpus,
    n_topics,
    decomposition,
    max_df = 0.95,
    min_df = 2,
    vectorizer = 'bow',
    stemming = True,
    cleaning = simple_textcleaning,
    stop_words = STOPWORDS,
    **kwargs
):
    assert isinstance(corpus, list) and isinstance(
        corpus[0], str
    ), 'input must be list of strings'
    assert isinstance(n_topics, int), 'n_topics must be an integer'
    assert isinstance(max_df, float), 'max_df must be a float'
    assert isinstance(min_df, int), 'min_df must be an integer'
    assert isinstance(vectorizer, str), 'vectorizer must be a string'
    vectorizer = vectorizer.lower()
    assert vectorizer in ['tfidf', 'bow'], "vectorizer must be 'tfidf' or 'bow'"
    assert (
        max_df < 1 and max_df > 0
    ), 'max_df must be bigger than 0, less than 1'
    assert min_df > 0, 'min_df must be bigger than 0'
    assert isinstance(stemming, bool), 'bool must be a boolean'
    if cleaning is not None:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)):
            corpus[i] = sastrawi_stemmer(corpus[i])
    if vectorizer == 'tfidf':
        Vectorizer = TfidfVectorizer
    elif vectorizer == 'bow':
        Vectorizer = CountVectorizer
    else:
        raise Exception(
            "vectorizer not support, vectorizer must be 'tfidf' or 'bow'"
        )
    tf_vectorizer = Vectorizer(
        max_df = max_df, min_df = min_df, stop_words = stop_words, **kwargs
    )
    tf = tf_vectorizer.fit_transform(corpus)
    tf_features = tf_vectorizer.get_feature_names()
    compose = decomposition(n_topics).fit(tf)
    return TOPIC(
        tf_features,
        compose,
        [classification_textcleaning(c) for c in corpus],
        compose.transform(tf),
        tf_vectorizer,
        tf,
    )


def _lda2vec_preprocessing(
    corpus,
    window_size = 2,
    stemming = True,
    cleaning = simple_textcleaning,
    stop_words = STOPWORDS,
):
    import pandas as pd

    assert isinstance(corpus, list) and isinstance(
        corpus[0], str
    ), 'input must be list of strings'
    assert isinstance(stemming, bool), 'bool must be a boolean'
    assert isinstance(window_size, int), 'window_size must be an integer'
    if cleaning is not None:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)):
            corpus[i] = sastrawi_stemmer(corpus[i])
    text_clean = []
    for text in corpus:
        text_clean.append(
            ' '.join([word for word in text.split() if word not in stop_words])
        )
    concat = (' '.join(text_clean)).split()
    unique_count = len(list(set(concat)))
    _, count, dictionary, reversed_dictionary = build_dataset(
        concat, unique_count, included_prefix = False
    )
    idx_text_clean, len_idx_text_clean = [], []
    for text in text_clean:
        splitted = [dictionary[word] for word in text.split()]
        idx_text_clean.append(splitted)
        len_idx_text_clean.append(len(splitted))
    word_counts = collections.Counter(concat)
    df_freqs = pd.DataFrame.from_records(word_counts, index = ['Freqs'])
    df_freqs = df_freqs.T
    df_freqs = df_freqs.sort_values(['Freqs'], ascending = False)
    freqs = df_freqs.values.flatten().tolist()[:unique_count]
    doc_ids = np.arange(len(idx_text_clean))
    num_unique_documents = doc_ids.max()
    skipgrams_data = []
    for i, t in enumerate(idx_text_clean):
        pairs, _ = skipgrams(
            t,
            vocabulary_size = unique_count,
            window_size = window_size,
            shuffle = True,
            negative_samples = 0,
        )
        for pair in pairs:
            temp_data = pair
            temp_data.append(doc_ids[i])
            skipgrams_data.append(temp_data)
    skipgrams_data_df = pd.DataFrame(skipgrams_data)
    pivot_words = skipgrams_data_df[0].values
    target_words = skipgrams_data_df[1].values
    doc_ids = skipgrams_data_df[2].values
    pivot_words, target_words, doc_ids = shuffle(
        pivot_words, target_words, doc_ids, random_state = 10
    )
    num_unique_documents = doc_ids.max() + 1
    return (
        pivot_words,
        target_words,
        doc_ids,
        dictionary,
        reversed_dictionary,
        num_unique_documents,
        freqs,
        len_idx_text_clean,
        corpus,
    )


def lda_topic_modelling(
    corpus,
    n_topics = 10,
    max_df = 0.95,
    min_df = 2,
    stemming = True,
    vectorizer = 'bow',
    cleaning = simple_textcleaning,
    stop_words = STOPWORDS,
    **kwargs
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
    stemming: bool, (default=True)
        If True, sastrawi_stemmer will apply
    vectorizer: str, (default='bow')
        vectorization technique for corpus, only support 'bow' and 'tfidf'.
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus
    stop_words: list, (default=STOPWORDS)
        list of stop words to remove

    Returns
    -------
    TOPIC: malaya.topic_modelling.TOPIC class
    """
    return _base_topic_modelling(
        corpus,
        n_topics,
        LatentDirichletAllocation,
        max_df = max_df,
        min_df = min_df,
        vectorizer = vectorizer,
        stemming = stemming,
        cleaning = cleaning,
        stop_words = stop_words,
        **kwargs
    )


def nmf_topic_modelling(
    corpus,
    n_topics = 10,
    max_df = 0.95,
    min_df = 2,
    stemming = True,
    vectorizer = 'bow',
    cleaning = simple_textcleaning,
    stop_words = STOPWORDS,
    **kwargs
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
    stemming: bool, (default=True)
        If True, sastrawi_stemmer will apply.
    vectorizer: str, (default='bow')
        vectorization technique for corpus, only support 'bow' and 'tfidf'.
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus.
    stop_words: list, (default=STOPWORDS)
        list of stop words to remove.

    Returns
    -------
    TOPIC: malaya.topic_modelling.TOPIC class
    """
    return _base_topic_modelling(
        corpus,
        n_topics,
        NMF,
        max_df = max_df,
        min_df = min_df,
        vectorizer = vectorizer,
        stemming = stemming,
        cleaning = cleaning,
        stop_words = stop_words,
        **kwargs
    )


def lsa_topic_modelling(
    corpus,
    n_topics,
    max_df = 0.95,
    min_df = 2,
    vectorizer = 'bow',
    stemming = True,
    cleaning = simple_textcleaning,
    stop_words = STOPWORDS,
    **kwargs
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
    vectorizer: str, (default='bow')
        vectorization technique for corpus, only support 'bow' and 'tfidf'.
    stemming: bool, (default=True)
        If True, sastrawi_stemmer will apply
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus
    stop_words: list, (default=STOPWORDS)
        list of stop words to remove

    Returns
    -------
    TOPIC: malaya.topic_modelling.TOPIC class
    """
    return _base_topic_modelling(
        corpus,
        n_topics,
        TruncatedSVD,
        max_df = max_df,
        min_df = min_df,
        vectorizer = vectorizer,
        stemming = stemming,
        cleaning = cleaning,
        stop_words = stop_words,
        **kwargs
    )


def lda2vec_topic_modelling(
    corpus,
    n_topics,
    stemming = True,
    cleaning = simple_textcleaning,
    stop_words = STOPWORDS,
    window_size = 2,
    embedding_size = 128,
    training_iteration = 10,
    switch_loss = 3,
    **kwargs
):
    """
    Train a LDA2Vec model to do topic modelling based on corpus / list of strings given.

    Parameters
    ----------
    corpus: list
    n_topics: int, (default=10)
        size of decomposition column.
    stemming: bool, (default=True)
        If True, sastrawi_stemmer will apply
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus
    stop_words: list, (default=STOPWORDS)
        list of stop words to remove
    embedding_size: int, (default=128)
        embedding size of lda2vec tensors
    training_iteration: int, (default=10)
        training iteration, how many loop need to train
    switch_loss: int, (default=3)
        baseline to switch from document based loss to document + word based loss

    Returns
    -------
    TOPIC: malaya.topic_modelling.DEEP_TOPIC class
    """
    pivot_words, target_words, doc_ids, dictionary, reversed_dictionary, num_unique_documents, freqs, len_idx_text_clean, cleaned_corpus = _lda2vec_preprocessing(
        corpus,
        window_size = window_size,
        stemming = stemming,
        cleaning = cleaning,
        stop_words = stop_words,
    )
    model = LDA2VEC(
        num_unique_documents,
        len(dictionary),
        n_topics,
        freqs,
        embedding_size = embedding_size,
        **kwargs
    )
    model.train(
        pivot_words,
        target_words,
        doc_ids,
        training_iteration,
        switch_loss = switch_loss,
    )
    return DEEP_TOPIC(
        model,
        dictionary,
        reversed_dictionary,
        freqs,
        len_idx_text_clean,
        cleaned_corpus,
    )
