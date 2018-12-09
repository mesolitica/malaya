import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

from sklearn import metrics, datasets
import tensorflow as tf
import numpy as np
import os
import json
import pickle
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
from .text_functions import separate_dataset
from .stemmer import classification_textcleaning_stemmer
from .sklearn_model import USER_XGB, USER_BAYES
from .utils import download_file, load_graph
from .tensorflow_model import SENTIMENT
from .paths import PATH_SENTIMENTS, S3_PATH_SENTIMENTS
from .vectorizer import SkipGramVectorizer


def get_available_sentiment_models():
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


def deep_sentiment(model = 'luong'):
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

    Returns
    -------
    SENTIMENT: malaya.tensorflow_model.SENTIMENT class
    """
    assert isinstance(model, str), 'model must be a string'
    model = model.lower()
    if model == 'fast-text':
        if not os.path.isfile(PATH_SENTIMENTS['fast-text']['model']):
            print('downloading SENTIMENT frozen fast-text model')
            download_file(
                S3_PATH_SENTIMENTS['fast-text']['model'],
                PATH_SENTIMENTS['fast-text']['model'],
            )
        if not os.path.isfile(PATH_SENTIMENTS['fast-text']['setting']):
            print('downloading SENTIMENT fast-text dictionary')
            download_file(
                S3_PATH_SENTIMENTS['fast-text']['setting'],
                PATH_SENTIMENTS['fast-text']['setting'],
            )
        if not os.path.isfile(PATH_SENTIMENTS['fast-text']['pickle']):
            print('downloading SENTIMENT fast-text bigrams')
            download_file(
                S3_PATH_SENTIMENTS['fast-text']['pickle'],
                PATH_SENTIMENTS['fast-text']['pickle'],
            )
        with open(PATH_SENTIMENTS['fast-text']['setting'], 'r') as fopen:
            dictionary = json.load(fopen)['dictionary']
        with open(PATH_SENTIMENTS['fast-text']['pickle'], 'rb') as fopen:
            ngram = pickle.load(fopen)
        g = load_graph(PATH_SENTIMENTS['fast-text']['model'])
        return SENTIMENT(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            ngram = ngram,
        )
    elif model == 'hierarchical':
        if not os.path.isfile(PATH_SENTIMENTS['hierarchical']['model']):
            print('downloading SENTIMENT frozen hierarchical model')
            download_file(
                S3_PATH_SENTIMENTS['hierarchical']['model'],
                PATH_SENTIMENTS['hierarchical']['model'],
            )
        if not os.path.isfile(PATH_SENTIMENTS['hierarchical']['setting']):
            print('downloading SENTIMENT hierarchical dictionary')
            download_file(
                S3_PATH_SENTIMENTS['hierarchical']['setting'],
                PATH_SENTIMENTS['hierarchical']['setting'],
            )
        with open(PATH_SENTIMENTS['hierarchical']['setting'], 'r') as fopen:
            dictionary = json.load(fopen)['dictionary']
        g = load_graph(PATH_SENTIMENTS['hierarchical']['model'])
        return SENTIMENT(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            alphas = g.get_tensor_by_name('import/alphas:0'),
        )
    elif model == 'bahdanau':
        if not os.path.isfile(PATH_SENTIMENTS['bahdanau']['model']):
            print('downloading SENTIMENT frozen bahdanau model')
            download_file(
                S3_PATH_SENTIMENTS['bahdanau']['model'],
                PATH_SENTIMENTS['bahdanau']['model'],
            )
        if not os.path.isfile(PATH_SENTIMENTS['bahdanau']['setting']):
            print('downloading SENTIMENT bahdanau dictionary')
            download_file(
                S3_PATH_SENTIMENTS['bahdanau']['setting'],
                PATH_SENTIMENTS['bahdanau']['setting'],
            )
        with open(PATH_SENTIMENTS['bahdanau']['setting'], 'r') as fopen:
            dictionary = json.load(fopen)['dictionary']
        g = load_graph(PATH_SENTIMENTS['bahdanau']['model'])
        return SENTIMENT(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            alphas = g.get_tensor_by_name('import/alphas:0'),
        )
    elif model == 'luong':
        if not os.path.isfile(PATH_SENTIMENTS['luong']['model']):
            print('downloading SENTIMENT frozen luong model')
            download_file(
                S3_PATH_SENTIMENTS['luong']['model'],
                PATH_SENTIMENTS['luong']['model'],
            )
        if not os.path.isfile(PATH_SENTIMENTS['luong']['setting']):
            print('downloading SENTIMENT luong dictionary')
            download_file(
                S3_PATH_SENTIMENTS['luong']['setting'],
                PATH_SENTIMENTS['luong']['setting'],
            )
        with open(PATH_SENTIMENTS['luong']['setting'], 'r') as fopen:
            dictionary = json.load(fopen)
        g = load_graph(PATH_SENTIMENTS['luong']['model'])
        return SENTIMENT(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            alphas = g.get_tensor_by_name('import/alphas:0'),
        )
    elif model == 'bidirectional':
        if not os.path.isfile(PATH_SENTIMENTS['bidirectional']['model']):
            print('downloading SENTIMENT frozen bidirectional model')
            download_file(
                S3_PATH_SENTIMENTS['bidirectional']['model'],
                PATH_SENTIMENTS['bidirectional']['model'],
            )
        if not os.path.isfile(PATH_SENTIMENTS['bidirectional']['setting']):
            print('downloading SENTIMENT bidirectional dictionary')
            download_file(
                S3_PATH_SENTIMENTS['bidirectional']['setting'],
                PATH_SENTIMENTS['bidirectional']['setting'],
            )
        with open(PATH_SENTIMENTS['bidirectional']['setting'], 'r') as fopen:
            dictionary = json.load(fopen)
        g = load_graph(PATH_SENTIMENTS['bidirectional']['model'])
        return SENTIMENT(
            g.get_tensor_by_name('import/Placeholder:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
        )
    elif model == 'bert':
        if not os.path.isfile(PATH_SENTIMENTS['bert']['model']):
            print('downloading SENTIMENT frozen bert model')
            download_file(
                S3_PATH_SENTIMENTS['bert']['model'],
                PATH_SENTIMENTS['bert']['model'],
            )
        if not os.path.isfile(PATH_SENTIMENTS['bert']['setting']):
            print('downloading SENTIMENT bert dictionary')
            download_file(
                S3_PATH_SENTIMENTS['bert']['setting'],
                PATH_SENTIMENTS['bert']['setting'],
            )
        with open(PATH_SENTIMENTS['bert']['setting'], 'r') as fopen:
            dictionary = json.load(fopen)
        g = load_graph(PATH_SENTIMENTS['bert']['model'])
        return SENTIMENT(
            g.get_tensor_by_name('import/Placeholder_input_ids:0'),
            g.get_tensor_by_name('import/loss/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            input_mask = g.get_tensor_by_name(
                'import/Placeholder_input_mask:0'
            ),
            segment_ids = g.get_tensor_by_name(
                'import/Placeholder_segment_ids:0'
            ),
            is_training = g.get_tensor_by_name(
                'import/Placeholder_is_training:0'
            ),
        )
    elif model == 'entity-network':
        if not os.path.isfile(PATH_SENTIMENTS['entity-network']['model']):
            print('downloading SENTIMENT frozen entity-network model')
            download_file(
                S3_PATH_SENTIMENTS['entity-network']['model'],
                PATH_SENTIMENTS['entity-network']['model'],
            )
        if not os.path.isfile(PATH_SENTIMENTS['entity-network']['setting']):
            print('downloading SENTIMENT entity-network dictionary')
            download_file(
                S3_PATH_SENTIMENTS['entity-network']['setting'],
                PATH_SENTIMENTS['entity-network']['setting'],
            )
        with open(PATH_SENTIMENTS['entity-network']['setting'], 'r') as fopen:
            dictionary = json.load(fopen)
        g = load_graph(PATH_SENTIMENTS['entity-network']['model'])
        return SENTIMENT(
            g.get_tensor_by_name('import/Placeholder_question:0'),
            g.get_tensor_by_name('import/logits:0'),
            tf.InteractiveSession(graph = g),
            model,
            dictionary,
            dropout_keep_prob = g.get_tensor_by_name(
                'import/Placeholder_dropout_keep_prob:0'
            ),
            story = g.get_tensor_by_name('import/Placeholder_story:0'),
        )
    else:
        raise Exception(
            'model sentiment not supported, please check supported models from malaya.get_available_sentiment_models()'
        )


def bayes_sentiment(corpus, vector = 'tfidf', split_size = 0.2, **kwargs):
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
    assert (
        isinstance(corpus, str)
        or isinstance(corpus, list)
        or isinstance(corpus, tuple)
    ), 'corpus must be a string location or list of strings or tuple of strings'
    assert isinstance(vector, str), 'vector must be a string'
    assert isinstance(split_size, float), 'split_size must be a float'
    assert (
        split_size > 0 and split_size < 1
    ), 'split_size must bigger than 0, less than 1'
    multinomial, labels, vectorize = None, None, None
    if isinstance(corpus, str):
        trainset = datasets.load_files(
            container_path = corpus, encoding = 'UTF-8'
        )
        trainset.data, trainset.target = separate_dataset(trainset)
        data, target = trainset.data, trainset.target
        labels = trainset.target_names
    if isinstance(corpus, list) or isinstance(corpus, tuple):
        assert (
            len(corpus[0]) == 2
        ), 'element of corpus must be list or tuple of (string, label)'
        assert isinstance(
            corpus[0][0], str
        ), 'left hand side of element must be a string'
        corpus = np.array(corpus)
        data, target = corpus[:, 0].tolist(), corpus[:, 1].tolist()
        labels = np.unique(target).tolist()
        target = LabelEncoder().fit_transform(target)
    c = list(zip(data, target))
    random.shuffle(c)
    data, target = zip(*c)
    data, target = list(data), list(target)
    for i in range(len(data)):
        data[i] = classification_textcleaning_stemmer(data[i])
    if 'tfidf' in vector.lower():
        vectorize = TfidfVectorizer(**kwargs).fit(data)
        vectors = vectorize.transform(data)
    elif 'bow' in vector.lower():
        vectorize = CountVectorizer(**kwargs).fit(data)
        vectors = vectorize.transform(data)
    elif 'skip-gram' in vector.lower():
        vectorize = SkipGramVectorizer(**kwargs).fit(data)
        vectors = vectorize.transform(data)
    else:
        raise Exception(
            "vectorizing techniques not supported, only support ['tf-idf', 'bow', 'skip-gram']"
        )
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
        cleaning = classification_textcleaning_stemmer,
    )


def pretrained_bayes_sentiment():
    """
    Load multinomial sentiment model.

    Returns
    -------
    USER_BAYES : malaya.sklearn_model.USER_BAYES class
    """
    if not os.path.isfile(PATH_SENTIMENTS['multinomial']['model']):
        print('downloading SENTIMENT pickled multinomial model')
        download_file(
            S3_PATH_SENTIMENTS['multinomial']['model'],
            PATH_SENTIMENTS['multinomial']['model'],
        )
    if not os.path.isfile(PATH_SENTIMENTS['multinomial']['vector']):
        print('downloading SENTIMENT pickled multinomial tfidf vectorization')
        download_file(
            S3_PATH_SENTIMENTS['multinomial']['vector'],
            PATH_SENTIMENTS['multinomial']['vector'],
        )
    with open(PATH_SENTIMENTS['multinomial']['model'], 'rb') as fopen:
        multinomial = pickle.load(fopen)
    with open(PATH_SENTIMENTS['multinomial']['vector'], 'rb') as fopen:
        vectorize = pickle.load(fopen)
    return USER_BAYES(
        multinomial,
        ['negative', 'positive'],
        vectorize,
        cleaning = classification_textcleaning_stemmer,
    )


def pretrained_xgb_sentiment():
    """
    Load XGB sentiment model.

    Returns
    -------
    USER_BAYES : malaya.sklearn_model.USER_BAYES class
    """
    if not os.path.isfile(PATH_SENTIMENTS['xgb']['model']):
        print('downloading SENTIMENT pickled XGB model')
        download_file(
            S3_PATH_SENTIMENTS['xgb']['model'], PATH_SENTIMENTS['xgb']['model']
        )
    if not os.path.isfile(PATH_SENTIMENTS['xgb']['vector']):
        print('downloading SENTIMENT pickled XGB tfidf vectorization')
        download_file(
            S3_PATH_SENTIMENTS['xgb']['vector'],
            PATH_SENTIMENTS['xgb']['vector'],
        )
    with open(PATH_SENTIMENTS['xgb']['model'], 'rb') as fopen:
        xgb = pickle.load(fopen)
    with open(PATH_SENTIMENTS['xgb']['vector'], 'rb') as fopen:
        vectorize = pickle.load(fopen)
    return USER_XGB(
        xgb,
        ['negative', 'positive'],
        vectorize,
        cleaning = classification_textcleaning_stemmer,
    )
