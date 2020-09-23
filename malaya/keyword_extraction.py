import re
import operator
import networkx as nx
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from malaya.text import rake as rake_function
from sklearn.feature_extraction.text import CountVectorizer
from malaya.text.vectorizer import SkipGramVectorizer
from malaya.text.function import (
    simple_textcleaning,
    transformer_textcleaning,
    STOPWORDS,
)
from malaya.graph.pagerank import pagerank
from typing import Callable, Tuple, List
from herpetologist import check_type

methods = {'bow': CountVectorizer, 'skipgram': SkipGramVectorizer}


def _auto_ngram(string, stop_words):
    splitted = rake_function.split_sentences(string)
    stop_word_regex_list = []
    for word in stop_words:
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


def _base(string, ngram_method, ngram, stop_words, **kwargs):
    s = methods[ngram_method](
        ngram_range = ngram,
        token_pattern = r'[\S]+',
        stop_words = stop_words,
        lowercase = False,
        **kwargs
    ).fit([string])
    vocab = defaultdict(int)
    tokens = s.build_analyzer()(string)
    for t in tokens:
        vocab[t] += 1
    return vocab


@check_type
def rake(
    string: str,
    model = None,
    top_k: int = 5,
    auto_ngram: bool = True,
    ngram_method: str = 'bow',
    ngram: Tuple[int, int] = (1, 1),
    atleast: int = 1,
    stop_words: List[str] = STOPWORDS,
    **kwargs
):
    """
    Extract keywords using Rake algorithm.

    Parameters
    ----------
    string: str
    model: Object, optional (default='None')
        Transformer model or any model has `attention` method.
    top_k: int, optional (default=5)
        return top-k results.
    auto_ngram: bool, optional (default=True)
        If True, will generate keyword candidates using N suitable ngram. Else use `ngram_method`.
    ngram_method: str, optional (default='bow')
        Only usable if `auto_ngram` is False. supported ngram generator:

        * ``'bow'`` - bag-of-word.
        * ``'skipgram'`` - bag-of-word with skip technique.
    ngram: tuple, optional (default=(1,1))
        n-grams size.
    atleast: int, optional (default=1)
        at least count appeared in the string to accept as candidate.
    stop_words: list, (default=malaya.text.function.STOPWORDS)
        list of stop words to remove. 

    Returns
    -------
    result: Tuple[float, str]
    """

    if model is not None:
        if not hasattr(model, 'attention'):
            raise ValueError('model must has or `attention` method')
    if top_k < 1:
        raise ValueError('top_k must bigger than 0')
    if atleast < 1:
        raise ValueError('atleast must bigger than 0')
    if ngram_method not in methods:
        raise ValueError("ngram_method must be in ['bow', 'skip-gram']")
    if auto_ngram and not len(stop_words):
        raise ValueError('insert stop_words if auto_ngram')

    if model:
        string = transformer_textcleaning(string)
        attention = model.attention([string])[0]
        d = defaultdict(float)
        for k, v in attention:
            d[k] += v

    else:
        d = None

    if auto_ngram:
        vocab = _auto_ngram(string, stop_words)
    else:
        vocab = _base(
            string,
            ngram_method = ngram_method,
            ngram = ngram,
            stop_words = stop_words,
            **kwargs
        )
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
    vectorizer,
    top_k: int = 5,
    auto_ngram: bool = True,
    ngram_method: str = 'bow',
    ngram: Tuple[int, int] = (1, 1),
    atleast: int = 1,
    stop_words: List[str] = STOPWORDS,
    **kwargs
):
    """
    Extract keywords using Textrank algorithm.

    Parameters
    ----------
    string: str
    vectorizer: Object, optional (default='None')
        model has `fit_transform` or `vectorize` method.
    top_k: int, optional (default=5)
        return top-k results.
    auto_ngram: bool, optional (default=True)
        If True, will generate keyword candidates using N suitable ngram. Else use `ngram_method`.
    ngram_method: str, optional (default='bow')
        Only usable if `auto_ngram` is False. supported ngram generator:

        * ``'bow'`` - bag-of-word.
        * ``'skipgram'`` - bag-of-word with skip technique.
    ngram: tuple, optional (default=(1,1))
        n-grams size.
    atleast: int, optional (default=1)
        at least count appeared in the string to accept as candidate.
    stop_words: list, (default=malaya.text.function.STOPWORDS)
        list of stop words to remove. 

    Returns
    -------
    result: Tuple[float, str]
    """

    if not hasattr(vectorizer, 'fit_transform') and not hasattr(
        vectorizer, 'vectorize'
    ):
        raise ValueError(
            'vectorizer must has `fit_transform` or `vectorize` method'
        )
    if top_k < 1:
        raise ValueError('top_k must bigger than 0')
    if atleast < 1:
        raise ValueError('atleast must bigger than 0')
    if ngram_method not in methods:
        raise ValueError("ngram_method must be in ['bow', 'skip-gram']")
    if auto_ngram and not len(stop_words):
        raise ValueError('insert stop_words if auto_ngram')

    if auto_ngram:
        vocab = _auto_ngram(string, stop_words)
    else:
        vocab = _base(
            string,
            ngram_method = ngram_method,
            ngram = ngram,
            stop_words = stop_words,
            **kwargs
        )

    if hasattr(vectorizer, 'fit_transform'):
        vectors = vectorizer.fit_transform(list(vocab.keys()))
    if hasattr(vectorizer, 'vectorize'):
        vectors = vectorizer.vectorize(list(vocab.keys()))
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
    top_k: int = 5,
    auto_ngram: bool = True,
    ngram_method: str = 'bow',
    ngram: Tuple[int, int] = (1, 1),
    atleast: int = 1,
    stop_words: List[str] = STOPWORDS,
    **kwargs
):
    """
    Extract keywords using Attention mechanism.

    Parameters
    ----------
    string: str
    model: Object, optional (default='None')
        Transformer model or any model has `attention` method.
    top_k: int, optional (default=5)
        return top-k results.
    auto_ngram: bool, optional (default=True)
        If True, will generate keyword candidates using N suitable ngram. Else use `ngram_method`.
    ngram_method: str, optional (default='bow')
        Only usable if `auto_ngram` is False. supported ngram generator:

        * ``'bow'`` - bag-of-word.
        * ``'skipgram'`` - bag-of-word with skip technique.
    ngram: tuple, optional (default=(1,1))
        n-grams size.
    atleast: int, optional (default=1)
        at least count appeared in the string to accept as candidate.
    stop_words: list, (default=malaya.text.function.STOPWORDS)
        list of stop words to remove. 

    Returns
    -------
    result: Tuple[float, str]
    """

    if not hasattr(model, 'attention'):
        raise ValueError('model must has or `attention` method')
    if top_k < 1:
        raise ValueError('top_k must bigger than 0')
    if atleast < 1:
        raise ValueError('atleast must bigger than 0')
    if ngram_method not in methods:
        raise ValueError("ngram_method must be in ['bow', 'skip-gram']")
    if auto_ngram and not len(stop_words):
        raise ValueError('insert stop_words if auto_ngram')

    string = transformer_textcleaning(string)

    if auto_ngram:
        vocab = _auto_ngram(string, stop_words)
    else:
        vocab = _base(
            string,
            ngram_method = ngram_method,
            ngram = ngram,
            stop_words = stop_words,
            **kwargs
        )

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
    top_k: int = 5,
    ngram_method: str = 'bow',
    ngram: Tuple[int, int] = (1, 1),
    atleast: int = 1,
    stop_words: List[str] = STOPWORDS,
    **kwargs
):
    if not hasattr(model, '_tree_plot'):
        raise ValueError('model must has or `_tree_plot` method')
    if top_k < 1:
        raise ValueError('top_k must bigger than 0')
    if atleast < 1:
        raise ValueError('atleast must bigger than 0')
    if ngram_method not in methods:
        raise ValueError("ngram_method must be in ['bow', 'skip-gram']")

    if auto_ngram:
        vocab = _auto_ngram(string, stop_words)
    else:
        vocab = _base(
            string,
            ngram_method = ngram_method,
            ngram = ngram,
            stop_words = stop_words,
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
