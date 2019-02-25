import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import numpy as np
import re
import random
from scipy.linalg import svd
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.decomposition import NMF, LatentDirichletAllocation
from .texts._text_functions import (
    summary_textcleaning,
    classification_textcleaning,
    STOPWORDS,
    split_by_dot,
)
from .stem import sastrawi
from ._models import _skip_thought
from .cluster import cluster_words


def deep_model_news():
    """
    Load skip-thought summarization deep learning model trained on news dataset.

    Returns
    -------
    DEEP_SUMMARIZER: malaya.skip_thought.DEEP_SUMMARIZER class
    """
    return _skip_thought.news_load_model()


def deep_model_wiki():
    """
    Load residual network with Bahdanau Attention summarization deep learning model trained on wikipedia dataset.

    Returns
    -------
    DEEP_SUMMARIZER: malaya.skip_thought.DEEP_SUMMARIZER class
    """
    print(
        'WARNING: this model is using convolutional based, Tensorflow-GPU above 1.10 may got a problem. Please downgrade to Tensorflow-GPU v1.8 if got any cuDNN error.'
    )
    return _skip_thought.wiki_load_model()


def train_skip_thought(
    corpus,
    epoch = 5,
    batch_size = 16,
    embedding_size = 256,
    maxlen = 50,
    vocab_size = None,
    stride = 1,
):
    """
    Train a deep skip-thought network for summarization agent

    Parameters
    ----------
    epoch: int, (default=5)
        iteration numbers
    batch_size: int, (default=32)
        batch size for every feed, batch size must <= size of corpus
    embedding_size: int, (default=256)
        vector size representation for a word
    maxlen: int, (default=50)
        max length of a string to be train
    vocab_size: int, (default=None)
        max vocabulary size, None for no limit
    stride: int, (default=1)
        stride size, skipping value for sentences

    Returns
    -------
    DEEP_SUMMARIZER: malaya.skip_thought.DEEP_SUMMARIZER class
    """
    assert isinstance(epoch, int), 'epoch must be an integer'
    assert isinstance(batch_size, int), 'batch_size must be an integer'
    assert isinstance(embedding_size, int), 'embedding_size must be an integer'
    assert isinstance(maxlen, int), 'maxlen must be an integer'
    assert vocab_size is None or isinstance(
        vocab_size, int
    ), 'vocab_size must be a None or an integer'
    assert (
        isinstance(corpus, list) and isinstance(corpus[0], str)
    ) or isinstance(corpus, str), 'input must be list of strings or a string'
    if isinstance(corpus, str):
        corpus = split_by_dot(corpus)
    else:
        corpus = ' '.join(corpus)
        corpus = re.findall('(?=\S)[^.\n]+(?<=\S)', corpus)
    corpus = [summary_textcleaning(i) for i in corpus]
    t_range = int((len(corpus) - 3) / stride + 1)
    left, middle, right = [], [], []
    for i in range(t_range):
        slices = corpus[i * stride : i * stride + 3]
        left.append(slices[0])
        middle.append(slices[1])
        right.append(slices[2])
    assert batch_size < len(left), 'batch size must smaller with corpus size'
    left, middle, right = shuffle(left, middle, right)
    sess, model, dictionary, _ = _skip_thought.train_model(
        middle,
        left,
        right,
        epoch = epoch,
        batch_size = batch_size,
        embedding_size = embedding_size,
        maxlen = maxlen,
        vocab_size = vocab_size,
    )
    return _skip_thought.DEEP_SUMMARIZER(
        sess,
        model.INPUT,
        model.get_thought,
        model.attention,
        dictionary,
        maxlen,
        model = model,
    )


def lsa(
    corpus,
    maintain_original = False,
    ngram = (1, 3),
    min_df = 2,
    top_k = 3,
    important_words = 3,
    return_cluster = True,
    **kwargs
):
    """
    summarize a list of strings using LSA.

    Parameters
    ----------
    corpus: list
    maintain_original: bool, (default=False)
        If False, will apply malaya.text_functions.classification_textcleaning
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus
    min_df: int, (default=2)
        minimum document frequency for a word
    top_k: int, (default=3)
        number of summarized strings
    important_words: int, (default=3)
        number of important words
    return_cluster: bool, (default=True)
        if True, will cluster important_words to similar texts

    Returns
    -------
    dictionary: result
    """
    assert isinstance(corpus, list) and isinstance(
        corpus[0], str
    ), 'input must be list of strings'
    assert isinstance(
        maintain_original, bool
    ), 'maintain_original must be a boolean'
    assert isinstance(top_k, int), 'top_k must be an integer'
    assert isinstance(
        important_words, int
    ), 'important_words must be an integer'
    assert isinstance(return_cluster, bool), 'return_cluster must be a boolean'
    if not isinstance(ngram, tuple):
        raise ValueError('ngram must be a tuple')
    if not len(ngram) == 2:
        raise ValueError('ngram size must equal to 2')

    corpus = [summary_textcleaning(i) for i in corpus]
    corpus = ' '.join(corpus)
    splitted_fullstop = re.findall('(?=\S)[^.\n]+(?<=\S)', corpus)
    splitted_fullstop = [
        classification_textcleaning(i) if not maintain_original else i
        for i in splitted_fullstop
        if len(i)
    ]
    stemmed = [sastrawi(i) for i in splitted_fullstop]
    tfidf = TfidfVectorizer(
        ngram_range = ngram, min_df = min_df, stop_words = STOPWORDS, **kwargs
    ).fit(stemmed)
    U, S, Vt = svd(tfidf.transform(stemmed).todense().T, full_matrices = False)
    summary = [
        (splitted_fullstop[i], np.linalg.norm(np.dot(np.diag(S), Vt[:, b]), 2))
        for i in range(len(splitted_fullstop))
        for b in range(len(Vt))
    ]
    summary = sorted(summary, key = itemgetter(1))
    summary = dict(
        (v[0], v) for v in sorted(summary, key = lambda summary: summary[1])
    ).values()
    summarized = '. '.join([a for a, b in summary][len(summary) - (top_k) :])
    indices = np.argsort(tfidf.idf_)[::-1]
    features = tfidf.get_feature_names()
    top_words = [features[i] for i in indices[:important_words]]
    if return_cluster:
        return {
            'summary': summarized,
            'top-words': top_words,
            'cluster-top-words': cluster_words(top_words),
        }
    return {'summary': summarized, 'top-words': top_words}


def nmf(
    corpus,
    maintain_original = False,
    ngram = (1, 3),
    min_df = 2,
    top_k = 3,
    important_words = 3,
    return_cluster = True,
    **kwargs
):
    """
    summarize a list of strings using NMF.

    Parameters
    ----------
    corpus: list
    maintain_original: bool, (default=False)
        If False, will apply malaya.text_functions.classification_textcleaning
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus
    top_k: int, (default=3)
        number of summarized strings
    important_words: int, (default=3)
        number of important words
    min_df: int, (default=2)
        minimum document frequency for a word
    return_cluster: bool, (default=True)
        if True, will cluster important_words to similar texts

    Returns
    -------
    dictionary: result
    """
    assert isinstance(corpus, list) and isinstance(
        corpus[0], str
    ), 'input must be list of strings'
    assert isinstance(
        maintain_original, bool
    ), 'maintain_original must be a boolean'
    assert isinstance(top_k, int), 'top_k must be an integer'
    assert isinstance(
        important_words, int
    ), 'important_words must be an integer'
    assert isinstance(return_cluster, bool), 'return_cluster must be a boolean'
    if not isinstance(ngram, tuple):
        raise ValueError('ngram must be a tuple')
    if not len(ngram) == 2:
        raise ValueError('ngram size must equal to 2')

    corpus = [summary_textcleaning(i) for i in corpus]
    corpus = ' '.join(corpus)
    splitted_fullstop = re.findall('(?=\S)[^.\n]+(?<=\S)', corpus)
    splitted_fullstop = [
        classification_textcleaning(i) if not maintain_original else i
        for i in splitted_fullstop
        if len(i)
    ]
    stemmed = [sastrawi(i) for i in splitted_fullstop]
    tfidf = TfidfVectorizer(
        ngram_range = ngram, min_df = min_df, stop_words = STOPWORDS, **kwargs
    ).fit(stemmed)
    densed_tfidf = tfidf.transform(stemmed).todense()
    nmf = NMF(len(splitted_fullstop)).fit(densed_tfidf)
    vectors = nmf.transform(densed_tfidf)
    components = nmf.components_.mean(axis = 1)
    summary = [
        (
            splitted_fullstop[i],
            np.linalg.norm(np.dot(np.diag(components), vectors[:, b]), 2),
        )
        for i in range(len(splitted_fullstop))
        for b in range(len(vectors))
    ]
    summary = sorted(summary, key = itemgetter(1))
    summary = dict(
        (v[0], v) for v in sorted(summary, key = lambda summary: summary[1])
    ).values()
    summarized = '. '.join([a for a, b in summary][len(summary) - (top_k) :])
    indices = np.argsort(tfidf.idf_)[::-1]
    features = tfidf.get_feature_names()
    top_words = [features[i] for i in indices[:important_words]]
    if return_cluster:
        return {
            'summary': summarized,
            'top-words': top_words,
            'cluster-top-words': cluster_words(top_words),
        }
    return {'summary': summarized, 'top-words': top_words}


def lda(
    corpus,
    maintain_original = False,
    ngram = (1, 3),
    min_df = 2,
    top_k = 3,
    important_words = 3,
    return_cluster = True,
    **kwargs
):
    """
    summarize a list of strings using LDA.

    Parameters
    ----------
    corpus: list
    maintain_original: bool, (default=False)
        If False, will apply malaya.text_functions.classification_textcleaning
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus
    min_df: int, (default=2)
        minimum document frequency for a word
    top_k: int, (default=3)
        number of summarized strings
    important_words: int, (default=3)
        number of important words
    return_cluster: bool, (default=True)
        if True, will cluster important_words to similar texts

    Returns
    -------
    dictionary: result
    """
    assert isinstance(corpus, list) and isinstance(
        corpus[0], str
    ), 'input must be list of strings'
    assert isinstance(
        maintain_original, bool
    ), 'maintain_original must be a boolean'
    assert isinstance(top_k, int), 'top_k must be an integer'
    assert isinstance(
        important_words, int
    ), 'important_words must be an integer'
    assert isinstance(return_cluster, bool), 'return_cluster must be a boolean'
    if not isinstance(ngram, tuple):
        raise ValueError('ngram must be a tuple')
    if not len(ngram) == 2:
        raise ValueError('ngram size must equal to 2')
    if not isinstance(min_df, int) or isinstance(min_df, float):
        raise ValueError('min_df must be an integer or a float')

    corpus = [summary_textcleaning(i) for i in corpus]
    corpus = ' '.join(corpus)
    splitted_fullstop = re.findall('(?=\S)[^.\n]+(?<=\S)', corpus)
    splitted_fullstop = [
        classification_textcleaning(i) if not maintain_original else i
        for i in splitted_fullstop
        if len(i)
    ]
    stemmed = [sastrawi(i) for i in splitted_fullstop]
    tfidf = TfidfVectorizer(
        ngram_range = ngram, min_df = min_df, stop_words = STOPWORDS, **kwargs
    ).fit(stemmed)
    densed_tfidf = tfidf.transform(stemmed).todense()
    lda = LatentDirichletAllocation(len(splitted_fullstop)).fit(densed_tfidf)
    vectors = lda.transform(densed_tfidf)
    components = lda.components_.mean(axis = 1)
    summary = [
        (
            splitted_fullstop[i],
            np.linalg.norm(np.dot(np.diag(components), vectors[:, b]), 2),
        )
        for i in range(len(splitted_fullstop))
        for b in range(len(vectors))
    ]
    summary = sorted(summary, key = itemgetter(1))
    summary = dict(
        (v[0], v) for v in sorted(summary, key = lambda summary: summary[1])
    ).values()
    summarized = '. '.join([a for a, b in summary][len(summary) - (top_k) :])
    indices = np.argsort(tfidf.idf_)[::-1]
    features = tfidf.get_feature_names()
    top_words = [features[i] for i in indices[:important_words]]
    if return_cluster:
        return {
            'summary': summarized,
            'top-words': top_words,
            'cluster-top-words': cluster_words(top_words),
        }
    return {'summary': summarized, 'top-words': top_words}
