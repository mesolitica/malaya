import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from unidecode import unidecode
from .stemmer import sastrawi_stemmer
import itertools
import numpy as np
import re
from .text_functions import (
    simple_textcleaning,
    STOPWORDS,
    classification_textcleaning,
    print_topics_modelling,
)
from .lda2vec import process_data, generate_lda, train_lda2vec


class DEEP_TOPIC:
    def __init__(self, idx2word, similarity, corpus, doc_weights):
        self.idx2word = idx2word
        self.similarity = similarity
        self.corpus = corpus
        self.doc_weights = doc_weights

    def print_topics(self, len_topic):
        assert isinstance(len_topic, int), 'len_topic must be an integer'
        print_topics_modelling(
            range(len_topic),
            feature_names = np.array(self.idx2word),
            sorting = np.argsort(self.similarity)[:, ::-1],
            topics_per_chunk = 5,
            n_words = 10,
        )

    def get_topics(self, len_topic):
        assert isinstance(len_topic, int), 'len_topic must be an integer'
        results = []
        for no, topic in enumerate(self.similarity):
            results.append(
                (
                    no,
                    ' '.join(
                        [
                            self.idx2word[i]
                            for i in topic.argsort()[: -len_topic - 1 : -1]
                        ]
                    ),
                )
            )
        return results

    def get_sentences(self, len_sentence, k = 0):
        assert isinstance(len_sentence, int), 'len_sentence must be an integer'
        assert isinstance(k, int), 'k must be an integer'
        reverse_sorted = np.argsort(self.doc_weights[:, k])[::-1]
        return [self.corpus[i] for i in reverse_sorted[:len_sentence]]


class TOPIC:
    def __init__(self, features, comp, corpus, transformed):
        self.features = features
        self.comp = comp
        self.corpus = corpus
        self.transformed = transformed

    def print_topics(self, len_topic):
        assert isinstance(len_topic, int), 'len_topic must be an integer'
        print_topics_modelling(
            range(len_topic),
            feature_names = np.array(self.features),
            sorting = np.argsort(self.comp.components_)[:, ::-1],
            topics_per_chunk = 5,
            n_words = 10,
        )

    def get_topics(self, len_topic):
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
        assert isinstance(len_sentence, int), 'len_sentence must be an integer'
        assert isinstance(k, int), 'k must be an integer'
        reverse_sorted = np.argsort(self.transformed[:, k])[::-1]
        return [self.corpus[i] for i in reverse_sorted[:len_sentence]]


def lda_topic_modelling(
    corpus,
    n_topics = 10,
    max_df = 0.95,
    min_df = 2,
    stemming = True,
    cleaning = simple_textcleaning,
    stop_words = STOPWORDS,
):
    assert isinstance(corpus, list) and isinstance(
        corpus[0], str
    ), 'input must be list of strings'
    assert isinstance(n_topics, int), 'n_topics must be an integer'
    assert isinstance(max_df, float), 'max_df must be a float'
    assert isinstance(min_df, int), 'min_df must be an integer'
    assert isinstance(stemming, bool), 'bool must be a boolean'
    if cleaning is not None:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)):
            corpus[i] = sastrawi_stemmer(corpus[i])
    tf_vectorizer = CountVectorizer(
        max_df = max_df, min_df = min_df, stop_words = stop_words
    )
    tf = tf_vectorizer.fit_transform(corpus)
    tf_features = tf_vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(
        n_topics = n_topics,
        max_iter = 5,
        learning_method = 'online',
        learning_offset = 50.0,
        random_state = 0,
    ).fit(tf)
    return TOPIC(
        tf_features,
        lda,
        [classification_textcleaning(c) for c in corpus],
        lda.transform(tf),
    )


def nmf_topic_modelling(
    corpus,
    n_topics = 10,
    max_df = 0.95,
    min_df = 2,
    stemming = True,
    cleaning = simple_textcleaning,
    stop_words = STOPWORDS,
):
    assert isinstance(corpus, list) and isinstance(
        corpus[0], str
    ), 'input must be list of strings'
    assert isinstance(n_topics, int), 'n_topics must be an integer'
    assert isinstance(max_df, float), 'max_df must be a float'
    assert isinstance(min_df, int), 'min_df must be an integer'
    assert isinstance(stemming, bool), 'bool must be a boolean'
    if cleaning is not None:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)):
            corpus[i] = sastrawi_stemmer(corpus[i])
    tfidf_vectorizer = TfidfVectorizer(
        max_df = max_df, min_df = min_df, stop_words = stop_words
    )
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    tfidf_features = tfidf_vectorizer.get_feature_names()
    nmf = NMF(
        n_components = n_topics,
        random_state = 1,
        alpha = 0.1,
        l1_ratio = 0.5,
        init = 'nndsvd',
    ).fit(tfidf)
    return TOPIC(
        tfidf_features,
        nmf,
        [classification_textcleaning(c) for c in corpus],
        nmf.transform(tfidf),
    )


def lsa_topic_modelling(
    corpus,
    n_topics,
    max_df = 0.95,
    min_df = 2,
    stemming = True,
    cleaning = simple_textcleaning,
    stop_words = STOPWORDS,
):
    assert isinstance(corpus, list) and isinstance(
        corpus[0], str
    ), 'input must be list of strings'
    assert isinstance(n_topics, int), 'n_topics must be an integer'
    assert isinstance(max_df, float), 'max_df must be a float'
    assert isinstance(min_df, int), 'min_df must be an integer'
    assert isinstance(stemming, bool), 'bool must be a boolean'
    if cleaning is not None:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)):
            corpus[i] = sastrawi_stemmer(corpus[i])
    tfidf_vectorizer = TfidfVectorizer(
        max_df = max_df, min_df = min_df, stop_words = stop_words
    )
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    tfidf_features = tfidf_vectorizer.get_feature_names()
    tfidf = Normalizer().fit_transform(tfidf)
    lsa = TruncatedSVD(n_topics).fit(tfidf)
    return TOPIC(
        tfidf_features,
        lsa,
        [classification_textcleaning(c) for c in corpus],
        lsa.transform(tfidf),
    )


def lda2vec_topic_modelling(
    corpus,
    n_topics,
    word_vectors,
    word2idx,
    idx2word,
    max_df = 0.95,
    min_df = 2,
    stemming = True,
    cleaning = simple_textcleaning,
    min_words = 5,
    epoch = 10,
    batch_size = 32,
):
    assert isinstance(corpus, list) and isinstance(
        corpus[0], str
    ), 'input must be list of strings'
    assert batch_size < len(corpus), 'batch size must smaller with corpus size'
    if cleaning is not None:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)):
            corpus[i] = sastrawi_stemmer(corpus[i])
    encoded_data, unigram_distribution = process_data(
        corpus, word2idx, min_words = min_words
    )
    doc_weights_init = generate_lda(
        corpus, n_topics, max_df = max_df, min_df = min_df
    )
    doc_weights, similarity = train_lda2vec(
        encoded_data,
        unigram_distribution,
        word_vectors,
        doc_weights_init,
        n_topics,
        batch_size = batch_size,
        epoch = epoch,
    )
    return DEEP_TOPIC(idx2word, similarity, corpus, doc_weights)
