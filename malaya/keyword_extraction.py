import re
import operator
import networkx as nx
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from malaya.text import rake as rake_function
from malaya.text.function import (
    simple_textcleaning,
    transformer_textcleaning,
    get_stopwords,
)
from malaya.function import validator
from malaya.graph.pagerank import pagerank
from typing import Callable, Tuple, List
from herpetologist import check_type


def _auto_ngram(string, stopwords):
    splitted = rake_function.split_sentences(string)
    stop_word_regex_list = []
    for word in stopwords:
        word_regex = r'\b' + word + r'(?![\w-])'
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile(
        '|'.join(stop_word_regex_list), re.IGNORECASE
    )
    phrase_list = rake_function.generate_candidate_keywords(
        splitted, stop_word_pattern
    )
    vocab = defaultdict(int)
    for t in phrase_list:
        vocab[t] += 1
    return vocab


def _base(string, vectorizer, **kwargs):
    s = vectorizer.fit([string])
    vocab = defaultdict(int)
    tokens = s.build_analyzer()(string)
    for t in tokens:
        vocab[t] += 1
    return vocab


@check_type
def rake(
    string: str,
    model = None,
    vectorizer = None,
    top_k: int = 5,
    atleast: int = 1,
    stopwords = get_stopwords,
    **kwargs
):
    """
    Extract keywords using Rake algorithm.

    Parameters
    ----------
    string: str
    model: Object, optional (default=None)
        Transformer model or any model has `attention` method.
    vectorizer: Object, optional (default=None)
        Prefer `sklearn.feature_extraction.text.CountVectorizer` or,
        `malaya.text.vectorizer.SkipGramCountVectorizer`.
        If None, will generate ngram automatically based on `stopwords`.
    top_k: int, optional (default=5)
        return top-k results.
    ngram: tuple, optional (default=(1,1))
        n-grams size.
    atleast: int, optional (default=1)
        at least count appeared in the string to accept as candidate.
    stopwords: List[str], (default=malaya.texts.function.get_stopwords)
        A callable that returned a List[str], or a List[str], or a Tuple[str]
        For automatic Ngram generator.

    Returns
    -------
    result: Tuple[float, str]
    """
    stopwords = validator.validate_stopwords(stopwords)

    if model is not None:
        if not hasattr(model, 'attention'):
            raise ValueError('model must have `attention` method')
    if top_k < 1:
        raise ValueError('top_k must bigger than 0')
    if atleast < 1:
        raise ValueError('atleast must bigger than 0')
    if not vectorizer:
        auto_ngram = True
    else:
        auto_ngram = False
        if not hasattr(vectorizer, 'fit'):
            raise ValueError('vectorizer must have `fit` method')
    if auto_ngram and not len(stopwords):
        raise ValueError('insert stopwords if auto_ngram')

    if model:
        string = transformer_textcleaning(string)
        attention = model.attention([string])[0]
        d = defaultdict(float)
        for k, v in attention:
            d[k] += v

    else:
        d = None

    if auto_ngram:
        vocab = _auto_ngram(string, stopwords)
    else:
        vocab = _base(string, vectorizer = vectorizer, **kwargs)
    phrase_list = list(vocab.keys())
    scores = rake_function.calculate_word_scores(phrase_list, attentions = d)
    keywordcandidates = rake_function.generate_candidate_keyword_scores(
        phrase_list, scores
    )

    sortedKeywords = sorted(
        keywordcandidates.items(), key = operator.itemgetter(1), reverse = True
    )

    total = sum([i[1] for i in sortedKeywords])

    ranked_sentences = [
        (i[1] / total, i[0]) for i in sortedKeywords if vocab[i[0]] >= atleast
    ]
    return ranked_sentences[:top_k]


@check_type
def textrank(
    string: str,
    model = None,
    vectorizer = None,
    top_k: int = 5,
    atleast: int = 1,
    stopwords = get_stopwords,
    **kwargs
):
    """
    Extract keywords using Textrank algorithm.

    Parameters
    ----------
    string: str
    model: Object, optional (default='None')
        model has `fit_transform` or `vectorize` method.
    vectorizer: Object, optional (default=None)
        Prefer `sklearn.feature_extraction.text.CountVectorizer` or, 
        `malaya.text.vectorizer.SkipGramCountVectorizer`.
        If None, will generate ngram automatically based on `stopwords`.
    top_k: int, optional (default=5)
        return top-k results.
    atleast: int, optional (default=1)
        at least count appeared in the string to accept as candidate.
    stopwords: List[str], (default=malaya.texts.function.get_stopwords)
        A callable that returned a List[str], or a List[str], or a Tuple[str]

    Returns
    -------
    result: Tuple[float, str]
    """
    stopwords = validator.validate_stopwords(stopwords)

    if not hasattr(model, 'fit_transform') and not hasattr(model, 'vectorize'):
        raise ValueError(
            'model must have `fit_transform` or `vectorize` method'
        )

    if top_k < 1:
        raise ValueError('top_k must bigger than 0')
    if atleast < 1:
        raise ValueError('atleast must bigger than 0')
    if not vectorizer:
        auto_ngram = True
    else:
        auto_ngram = False
        if not hasattr(vectorizer, 'fit'):
            raise ValueError('vectorizer must have `fit` method')
    if auto_ngram and not len(stopwords):
        raise ValueError('insert stopwords if auto_ngram')

    if auto_ngram:
        vocab = _auto_ngram(string, stopwords)
    else:
        vocab = _base(string, vectorizer = vectorizer, **kwargs)

    if hasattr(model, 'fit_transform'):
        vectors = model.fit_transform(list(vocab.keys()))
    if hasattr(model, 'vectorize'):
        vectors = model.vectorize(list(vocab.keys()))
    similar = cosine_similarity(vectors, vectors)
    similar[similar >= 0.99999] = 0
    scores = pagerank(similar)
    total = sum(scores)
    ranked_sentences = sorted(
        [
            (scores[i] / total, s)
            for i, s in enumerate(vocab.keys())
            if vocab[s] >= atleast
        ],
        reverse = True,
    )

    return ranked_sentences[:top_k]


@check_type
def attention(
    string: str,
    model,
    vectorizer = None,
    top_k: int = 5,
    atleast: int = 1,
    stopwords = get_stopwords,
    **kwargs
):
    """
    Extract keywords using Attention mechanism.

    Parameters
    ----------
    string: str
    model: Object
        Transformer model or any model has `attention` method.
    vectorizer: Object, optional (default=None)
        Prefer `sklearn.feature_extraction.text.CountVectorizer` or, 
        `malaya.text.vectorizer.SkipGramCountVectorizer`.
        If None, will generate ngram automatically based on `stopwords`.
    top_k: int, optional (default=5)
        return top-k results.
    atleast: int, optional (default=1)
        at least count appeared in the string to accept as candidate.
    stopwords: List[str], (default=malaya.texts.function.get_stopwords)
        A callable that returned a List[str], or a List[str], or a Tuple[str]

    Returns
    -------
    result: Tuple[float, str]
    """

    stopwords = validator.validate_stopwords(stopwords)

    if not hasattr(model, 'attention'):
        raise ValueError('model must have `attention` method')
    if top_k < 1:
        raise ValueError('top_k must bigger than 0')
    if atleast < 1:
        raise ValueError('atleast must bigger than 0')
    if not vectorizer:
        auto_ngram = True
    else:
        auto_ngram = False
        if not hasattr(vectorizer, 'fit'):
            raise ValueError('vectorizer must have `fit` method')
    if auto_ngram and not len(stopwords):
        raise ValueError('insert stopwords if auto_ngram')

    string = transformer_textcleaning(string)

    if auto_ngram:
        vocab = _auto_ngram(string, stopwords)
    else:
        vocab = _base(string, vectorizer = vectorizer, **kwargs)

    attention = model.attention([string])[0]
    d = defaultdict(float)
    for k, v in attention:
        d[k] += v

    scores = []
    for k in vocab.keys():
        scores.append(sum([d.get(w, 0) for w in k.split()]))

    total = sum(scores)

    ranked_sentences = sorted(
        [
            (scores[i] / total, s)
            for i, s in enumerate(vocab.keys())
            if vocab[s] >= atleast
        ],
        reverse = True,
    )
    return ranked_sentences[:top_k]


def similarity_transformer(
    string,
    model,
    vectorizer = None,
    top_k: int = 5,
    atleast: int = 1,
    stopwords = get_stopwords,
    **kwargs
):
    stopwords = validator.validate_stopwords(stopwords)
    if not hasattr(model, '_tree_plot'):
        raise ValueError('model must have `_tree_plot` method')
    if top_k < 1:
        raise ValueError('top_k must bigger than 0')
    if atleast < 1:
        raise ValueError('atleast must bigger than 0')
    if ngram_method not in methods:
        raise ValueError("ngram_method must be in ['bow', 'skip-gram']")

    if auto_ngram:
        vocab = _auto_ngram(string, stopwords)
    else:
        vocab = _base(
            string,
            ngram_method = ngram_method,
            ngram = ngram,
            stopwords = stopwords,
            **kwargs
        )

    similar = model._tree_plot(list(vocab.keys()))
    similar[similar >= 0.99999] = 0
    scores = pagerank(similar)
    ranked_sentences = sorted(
        [
            (scores[i], s)
            for i, s in enumerate(vocab.keys())
            if vocab[s] >= atleast
        ],
        reverse = True,
    )
    return ranked_sentences[:top_k]
