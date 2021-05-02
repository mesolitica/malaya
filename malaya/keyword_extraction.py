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
    **kwargs,
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
    **kwargs,
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
    **kwargs,
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


@check_type
def similarity_transformer(
    string: str,
    model,
    vectorizer = None,
    top_k: int = 5,
    atleast: int = 1,
    use_maxsum: bool = False,
    use_mmr: bool = False,
    diversity: float = 0.5,
    nr_candidates: int = 20,
    stopwords = get_stopwords,
    **kwargs,
):
    """
    Extract keywords using Sentence embedding VS keyword embedding similarity.
    https://github.com/MaartenGr/KeyBERT/blob/master/keybert/model.py

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
    use_maxsum: bool, optional (default=False) 
        Whether to use Max Sum Similarity.
    use_mmr: bool, optional (default=False) 
        Whether to use MMR.
    diversity: float, optional (default=0.5)
        The diversity of results between 0 and 1 if use_mmr is True.
    nr_candidates: int, optional (default=20) 
        The number of candidates to consider if use_maxsum is set to True.
    stopwords: List[str], (default=malaya.texts.function.get_stopwords)
        A callable that returned a List[str], or a List[str], or a Tuple[str]

    Returns
    -------
    result: Tuple[float, str]
    """
    stopwords = validator.validate_stopwords(stopwords)

    if not hasattr(model, 'vectorize'):
        raise ValueError('model must have `vectorize` method')
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

    if nr_candidates < top_k:
        raise Exception('nr_candidates must bigger than top_k')

    string = transformer_textcleaning(string)

    if auto_ngram:
        vocab = _auto_ngram(string, stopwords)
    else:
        vocab = _base(string, vectorizer = vectorizer, **kwargs)

    words = list(vocab.keys())
    vectors_keywords = model.vectorize(words)
    vectors_string = model.vectorize([string])

    if use_mmr:
        # https://github.com/MaartenGr/KeyBERT/blob/master/keybert/mmr.py

        word_doc_similarity = cosine_similarity(
            vectors_keywords, vectors_string
        )
        word_similarity = cosine_similarity(vectors_keywords)
        keywords_idx = [np.argmax(word_doc_similarity)]
        candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]
        for _ in range(top_n - 1):
            candidate_similarities = word_doc_similarity[candidates_idx, :]
            target_similarities = np.max(
                word_similarity[candidates_idx][:, keywords_idx], axis = 1
            )

            mmr = (
                1 - diversity
            ) * candidate_similarities - diversity * target_similarities.reshape(
                -1, 1
            )
            mmr_idx = candidates_idx[np.argmax(mmr)]

            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)
        ranked_sentences = [
            (word_doc_similarity.reshape(1, -1)[0][idx], words[idx])
            for idx in keywords_idx
        ]

    elif use_maxsum:
        # https://github.com/MaartenGr/KeyBERT/blob/master/keybert/maxsum.py

        distances = cosine_similarity(vectors_string, vectors_keywords)
        distances_words = cosine_similarity(vectors_keywords, vectors_keywords)
        words_idx = list(distances.argsort()[0][-nr_candidates:])
        words_vals = [words[index] for index in words_idx]
        candidates = distances_words[np.ix_(words_idx, words_idx)]
        min_sim = 100_000
        candidate = None
        for combination in itertools.combinations(range(len(words_idx)), top_n):
            sim = sum(
                [
                    candidates[i][j]
                    for i in combination
                    for j in combination
                    if i != j
                ]
            )
            if sim < min_sim:
                candidate = combination
                min_sim = sim

        ranked_sentences = [
            (distances[0][idx], words_vals[idx]) for idx in candidate
        ]

    else:
        distances = cosine_similarity(vectors_string, vectors_keywords)
        ranked_sentences = [
            (distances[0][index], words[index])
            for index in distances.argsort()[0]
        ][::-1]

    ranked_sentences = [i for i in ranked_sentences if vocab[i[1]] >= atleast]
    return ranked_sentences[:top_k]
