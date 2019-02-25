import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import numpy as np
from ._utils import _softmax_class
from sklearn import metrics, datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from .stem import _classification_textcleaning_stemmer
from .texts._text_functions import separate_dataset
from ._models._sklearn_model import USER_BAYES
from ._utils._paths import PATH_SENTIMENTS, S3_PATH_SENTIMENTS
from .texts.vectorizer import SkipGramVectorizer


def available_sparse_deep_model():
    """
    List available sparse deep learning sentiment analysis models.
    """
    return ['fast-text-char']


def available_deep_model():
    """
    List available deep learning sentiment analysis models.
    """
    return [
        'fast-text',
        'hierarchical',
        'bahdanau',
        'luong',
        'bidirectional',
        'bert',
        'entity-network',
    ]


def sparse_deep_model(model = 'fast-text-char', validate = True):
    """
    Load deep learning sentiment analysis model.

    Parameters
    ----------
    model : str, optional (default='luong')
        Model architecture supported. Allowed values:

        * ``'fast-text-char'`` - Fast-text architecture for character based n-grams, embedded and logits layers only
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    SPARSE_SOFTMAX: malaya._models._tensorflow_model.SPARSE_SOFTMAX class
    """
    return _softmax_class.sparse_deep_model(
        PATH_SENTIMENTS,
        S3_PATH_SENTIMENTS,
        'sentiment',
        ['negative', 'positive'],
        2,
        model = model,
        validate = validate,
    )


def deep_model(model = 'luong', validate = True):
    """
    Load deep learning sentiment analysis model.

    Parameters
    ----------
    model : str, optional (default='luong')
        Model architecture supported. Allowed values:

        * ``'fast-text'`` - Fast-text architecture, embedded and logits layers only
        * ``'hierarchical'`` - LSTM with hierarchical attention architecture
        * ``'bahdanau'`` - LSTM with bahdanau attention architecture
        * ``'bidirectional'`` - LSTM with Bidirectional RNN architecture
        * ``'luong'`` - LSTM with luong attention architecture
        * ``'bert'`` - Deep Bidirectional transformers architecture
        * ``'entity-network'`` - Recurrent Entity-Network architecture
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    SOFTMAX: malaya._models._tensorflow_model.SOFTMAX class
    """
    return _softmax_class.deep_model(
        PATH_SENTIMENTS,
        S3_PATH_SENTIMENTS,
        'sentiment',
        ['negative', 'positive'],
        model = model,
        validate = validate,
    )


def multinomial(validate = True):
    """
    Load multinomial sentiment model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    USER_BAYES : malaya._models._sklearn_model.USER_BAYES class
    """
    return _softmax_class.multinomial(
        PATH_SENTIMENTS,
        S3_PATH_SENTIMENTS,
        'sentiment',
        ['negative', 'positive'],
        validate = validate,
    )


def xgb(validate = True):
    """
    Load XGB sentiment model.

    Parameters
    ----------
    validate: bool, optional (default=True)
        if True, malaya will check model availability and download if not available.

    Returns
    -------
    USER_XGB : malaya._models._sklearn_model.USER_XGB class
    """
    return _softmax_class.xgb(
        PATH_SENTIMENTS,
        S3_PATH_SENTIMENTS,
        'sentiment',
        ['negative', 'positive'],
        validate = validate,
    )


def train_multinomial(corpus, vector = 'tfidf', split_size = 0.2, **kwargs):
    """
    Train a multinomial model using your own corpus / dataset.

    Parameters
    ----------
    corpus : list, str
        List of (string, label). Allowed tuple and list only.

    vector: str
        Vectorization techniques to feed into the model. Allowed values:

        * ``'tfidf'`` - TF-IDF
        * ``'bow'`` - bag-of-word
        * ``'skip-gram'`` - bag-of-word with skipping module

    split_size: float
        Split size to train and test. Should bigger than 0, less than 1.

    Returns
    -------
    USER_BAYES: malaya.sklearn_model.USER_BAYES class
    """
    if not (
        isinstance(corpus, str)
        or isinstance(corpus, list)
        or isinstance(corpus, tuple)
    ):
        raise ValueError(
            'corpus must be a string location or list of strings or tuple of strings'
        )
    if not isinstance(vector, str):
        raise ValueError('vector must be a string')
    if not isinstance(split_size, float):
        raise ValueError('split_size must be a float')
    if not (split_size > 0 and split_size < 1):
        raise ValueError('split_size must bigger than 0, less than 1')
    multinomial, labels, vectorize = None, None, None
    if isinstance(corpus, str):
        trainset = datasets.load_files(
            container_path = corpus, encoding = 'UTF-8'
        )
        trainset.data, trainset.target = separate_dataset(trainset)
        data, target = trainset.data, trainset.target
        labels = trainset.target_names
    if isinstance(corpus, list) or isinstance(corpus, tuple):
        if not len(corpus[0]) == 2:
            raise ValueError(
                'element of corpus must be list or tuple of (string, label)'
            )
        if not isinstance(corpus[0][0], str):
            raise ValueError('left hand side of element must be a string')
        corpus = np.array(corpus)
        data, target = corpus[:, 0].tolist(), corpus[:, 1].tolist()
        labels = np.unique(target).tolist()
        target = LabelEncoder().fit_transform(target)

    from sklearn.utils import shuffle

    data, target = shuffle(data, target)

    for i in range(len(data)):
        data[i] = _classification_textcleaning_stemmer(data[i])
    if 'tfidf' in vector.lower():
        vectorize = TfidfVectorizer(**kwargs).fit(data)
    elif 'bow' in vector.lower():
        vectorize = CountVectorizer(**kwargs).fit(data)
    elif 'skip-gram' in vector.lower():
        vectorize = SkipGramVectorizer(**kwargs).fit(data)
    else:
        raise Exception(
            "vectorizing techniques not supported, only support ['tf-idf', 'bow', 'skip-gram']"
        )
    vectors = vectorize.transform(data)
    multinomial = MultinomialNB()
    if split_size:
        train_X, test_X, train_Y, test_Y = train_test_split(
            vectors, target, test_size = split_size
        )
        multinomial.partial_fit(train_X, train_Y, classes = np.unique(target))
        predicted = multinomial.predict(test_X)
        print(
            metrics.classification_report(
                test_Y, predicted, target_names = labels
            )
        )
    else:
        multinomial.partial_fit(vectors, target, classes = np.unique(target))
        predicted = multinomial.predict(vectors)
        print(
            metrics.classification_report(
                target, predicted, target_names = labels
            )
        )
    return USER_BAYES(
        multinomial,
        labels,
        vectorize,
        cleaning = _classification_textcleaning_stemmer,
    )
