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
from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.text.bpe import SentencePieceTokenizer
from malaya.model.bert import KeyphraseBERT
from malaya.model.xlnet import KeyphraseXLNET
from malaya.path import MODEL_VOCAB, MODEL_BPE
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
    model=None,
    vectorizer=None,
    top_k: int = 5,
    atleast: int = 1,
    stopwords=get_stopwords,
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
        vocab = _base(string, vectorizer=vectorizer, **kwargs)
    phrase_list = list(vocab.keys())
    scores = rake_function.calculate_word_scores(phrase_list, attentions=d)
    keywordcandidates = rake_function.generate_candidate_keyword_scores(
        phrase_list, scores
    )

    sortedKeywords = sorted(
        keywordcandidates.items(), key=operator.itemgetter(1), reverse=True
    )

    total = sum([i[1] for i in sortedKeywords])

    ranked_sentences = [
        (i[1] / total, i[0]) for i in sortedKeywords if vocab[i[0]] >= atleast
    ]
    return ranked_sentences[:top_k]


@check_type
def textrank(
    string: str,
    model=None,
    vectorizer=None,
    top_k: int = 5,
    atleast: int = 1,
    stopwords=get_stopwords,
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
        vocab = _base(string, vectorizer=vectorizer, **kwargs)

    if hasattr(model, 'fit_transform'):
        vectors = model.fit_transform(list(vocab.keys()))
    if hasattr(model, 'vectorize'):
        vectors = model.vectorize(list(vocab.keys()))
    similar = cosine_similarity(vectors, vectors)
    similar[similar >= 0.99999] = 0
    scores = pagerank(similar, **kwargs)
    total = sum(scores)
    ranked_sentences = sorted(
        [
            (scores[i] / total, s)
            for i, s in enumerate(vocab.keys())
            if vocab[s] >= atleast
        ],
        reverse=True,
    )

    return ranked_sentences[:top_k]


@check_type
def attention(
    string: str,
    model,
    vectorizer=None,
    top_k: int = 5,
    atleast: int = 1,
    stopwords=get_stopwords,
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
        vocab = _base(string, vectorizer=vectorizer, **kwargs)

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
        reverse=True,
    )
    return ranked_sentences[:top_k]


@check_type
def similarity(
    string: str,
    model,
    vectorizer=None,
    top_k: int = 5,
    atleast: int = 1,
    stopwords=get_stopwords,
    **kwargs,
):
    """
    Extract keywords using Sentence embedding VS keyword embedding similarity.

    Parameters
    ----------
    string: str
    model: Object
        Transformer model or any model has `vectorize` method.
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
        raise ValueError('nr_candidates must bigger than top_k')

    string = transformer_textcleaning(string)

    if auto_ngram:
        vocab = _auto_ngram(string, stopwords)
    else:
        vocab = _base(string, vectorizer=vectorizer, **kwargs)

    words = list(vocab.keys())
    vectors_keywords = model.vectorize(words)
    vectors_string = model.vectorize([string])

    distances = cosine_similarity(vectors_string, vectors_keywords)
    ranked_sentences = [
        (distances[0][index], words[index]) for index in distances.argsort()[0]
    ][::-1]

    ranked_sentences = [i for i in ranked_sentences if vocab[i[1]] >= atleast]
    return ranked_sentences[:top_k]


_transformer_availability = {
    'bert': {
        'Size (MB)': 443,
        'Quantized Size (MB)': 112,
        'macro precision': 0.99403,
        'macro recall': 0.99568,
        'macro f1-score': 0.99485,
    },
    'tiny-bert': {
        'Size (MB)': 59.5,
        'Quantized Size (MB)': 15.1,
        'macro precision': 0.99494,
        'macro recall': 0.99707,
        'macro f1-score': 0.99600,
    },
    'alxlnet': {
        'Size (MB)': 53,
        'Quantized Size (MB)': 14,
        'macro precision': 0.98170,
        'macro recall': 0.99182,
        'macro f1-score': 0.98663,
    },
    'xlnet': {
        'Size (MB)': 472,
        'Quantized Size (MB)': 120,
        'macro precision': 0.99667,
        'macro recall': 0.99819,
        'macro f1-score': 0.99742,
    },
}


def available_transformer():
    """
    List available transformer keyword similarity model.
    """
    from malaya.function import describe_availability

    return describe_availability(
        _transformer_availability, text='tested on 20% test set.'
    )


@check_type
def transformer(model: str = 'bert', quantized: bool = False, **kwargs):
    """
    Load Transformer keyword similarity model.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'bert'`` - Google BERT BASE parameters.
        * ``'tiny-bert'`` - Google BERT TINY parameters.
        * ``'xlnet'`` - Google XLNET BASE parameters.
        * ``'alxlnet'`` - Malaya ALXLNET BASE parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:

        * if `bert` in model, will return `malaya.model.bert.KeyphraseBERT`.
        * if `xlnet` in model, will return `malaya.model.xlnet.KeyphraseXLNET`.
    """

    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.keyword_extraction.available_transformer()`.'
        )

    path = check_file(
        file=model,
        module='keyword-extraction',
        keys={
            'model': 'model.pb',
            'vocab': MODEL_VOCAB[model],
            'tokenizer': MODEL_BPE[model],
        },
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)
    outputs = ['logits']

    if model in ['bert', 'tiny-bert']:
        inputs = [
            'Placeholder',
            'Placeholder_1',
            'Placeholder_2',
            'Placeholder_3',
        ]
        outputs.append('bert/summary')
        selected_class = KeyphraseBERT

    if model in ['xlnet', 'alxlnet']:

        inputs = [
            'Placeholder',
            'Placeholder_1',
            'Placeholder_2',
            'Placeholder_3',
            'Placeholder_4',
            'Placeholder_5',
        ]
        outputs.append('xlnet/summary')
        selected_class = KeyphraseXLNET

    tokenizer = SentencePieceTokenizer(vocab_file=path['vocab'], spm_model_file=path['tokenizer'])
    input_nodes, output_nodes = nodes_session(g, inputs, outputs)

    return selected_class(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        tokenizer=tokenizer,
        label=['not similar', 'similar'],
    )
