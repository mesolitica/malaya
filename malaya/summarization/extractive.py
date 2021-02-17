from malaya.model.extractive_summarization import SKLearn, Doc2Vec, Encoder
from herpetologist import check_type


def encoder(vectorizer):
    """
    Encoder interface for summarization.

    Parameters
    ----------
    vectorizer : object
        encoder interface object, eg, BERT, XLNET, ALBERT, ALXLNET.
        should have `vectorize` method.

    Returns
    -------
    result: malaya.model.extractive_summarization.Encoder
    """

    if not hasattr(vectorizer, 'vectorize'):
        raise ValueError('vectorizer must have `vectorize` method')
    if not hasattr(vectorizer, 'attention'):
        import logging

        logging.warning(
            'vectorizer model does not have `attention` method, `top-words` will not work'
        )

    return Encoder(vectorizer)


def doc2vec(wordvector):
    """
    Doc2Vec interface for summarization.

    Parameters
    ----------
    wordvector : object
        malaya.wordvector.WordVector object.
        should have `get_vector_by_name` method.

    Returns
    -------
    result: malaya.model.extractive_summarization.Doc2Vec
    """
    if not hasattr(wordvector, 'get_vector_by_name'):
        raise ValueError('wordvector must have `get_vector_by_name` method')
    return Doc2Vec(wordvector)


def sklearn(model, vectorizer):
    """
    sklearn interface for summarization.

    Parameters
    ----------
    model : object
        Should have `fit_transform` method. Commonly:

        * ``sklearn.decomposition.TruncatedSVD`` - LSA algorithm.
        * ``sklearn.decomposition.LatentDirichletAllocation`` - LDA algorithm.
    vectorizer : object
        Should have `fit_transform` method. Commonly:

        * ``sklearn.feature_extraction.text.TfidfVectorizer`` - TFIDF algorithm.
        * ``sklearn.feature_extraction.text.CountVectorizer`` - Bag-of-Word algorithm.
        * ``malaya.text.vectorizer.SkipGramCountVectorizer`` - Skip Gram Bag-of-Word algorithm.
        * ``malaya.text.vectorizer.SkipGramTfidfVectorizer`` - Skip Gram TFIDF algorithm.

    Returns
    -------
    result: malaya.model.extractive_summarization.SKLearn
    """
    if not hasattr(model, 'fit_transform'):
        raise ValueError('model must have `fit_transform` method')
    if not hasattr(vectorizer, 'fit_transform'):
        raise ValueError('vectorizer must have `fit_transform` method')
    return SKLearn(model, vectorizer)
