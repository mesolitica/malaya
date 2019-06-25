from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from .texts.vectorizer import SkipGramVectorizer
from .stem import sastrawi
from .texts._text_functions import simple_textcleaning, split_by_dot, STOPWORDS
from scipy.cluster.hierarchy import ward, dendrogram
from fuzzywuzzy import fuzz
import numpy as np
import re
import random

_accepted_pos = [
    'ADJ',
    'ADP',
    'ADV',
    'AUX',
    'CCONJ',
    'DET',
    'NOUN',
    'NUM',
    'PART',
    'PRON',
    'PROPN',
    'SCONJ',
    'SYM',
    'VERB',
    'X',
]
_accepted_entities = [
    'OTHER',
    'law',
    'location',
    'organization',
    'person',
    'quantity',
    'time',
    'event',
]


def cluster_words(list_words):
    """
    cluster similar words based on structure, eg, ['mahathir mohamad', 'mahathir'] = ['mahathir mohamad']

    Parameters
    ----------
    list_words : list of str

    Returns
    -------
    string: list of clustered words
    """
    if not isinstance(list_words, list):
        raise ValueError('list_words must be a list')
    if not isinstance(list_words[0], str):
        raise ValueError('list_words must be a list of strings')

    dict_words = {}
    for word in list_words:
        found = False
        for key in dict_words.keys():
            if word in key or any(
                [word in inside for inside in dict_words[key]]
            ):
                dict_words[key].append(word)
                found = True
            if key in word:
                dict_words[key].append(word)
        if not found:
            dict_words[word] = [word]
    results = []
    for key, words in dict_words.items():
        results.append(max(list(set([key] + words)), key = len))
    return list(set(results))


def cluster_pos(result):
    """
    cluster similar POS.

    Parameters
    ----------
    result: list

    Returns
    -------
    result: list
    """
    if not isinstance(result, list):
        raise ValueError('result must be a list')
    if not isinstance(result[0], tuple):
        raise ValueError('result must be a list of tuple')
    if not all([i[1] in _accepted_pos for i in result]):
        raise ValueError(
            'elements of result must be a subset or equal of supported POS, please run malaya.describe_pos() to get supported POS'
        )

    output = {
        'ADJ': [],
        'ADP': [],
        'ADV': [],
        'ADX': [],
        'AUX': [],
        'CCONJ': [],
        'DET': [],
        'NOUN': [],
        'NUM': [],
        'PART': [],
        'PRON': [],
        'PROPN': [],
        'SCONJ': [],
        'SYM': [],
        'VERB': [],
        'X': [],
    }
    last_label, words = None, []
    for word, label in result:
        if last_label != label and last_label:
            joined = ' '.join(words)
            if joined not in output[last_label]:
                output[last_label].append(joined)
            words = []
            last_label = label
            words.append(word)

        else:
            if not last_label:
                last_label = label
            words.append(word)
    return output


def cluster_entities(result):
    """
    cluster similar Entities.

    Parameters
    ----------
    result: list

    Returns
    -------
    result: list
    """
    if not isinstance(result, list):
        raise ValueError('result must be a list')
    if not isinstance(result[0], tuple):
        raise ValueError('result must be a list of tuple')
    if not all([i[1] in _accepted_entities for i in result]):
        raise ValueError(
            'elements of result must be a subset or equal of supported POS, please run malaya.describe_pos() to get supported POS'
        )

    output = {
        'OTHER': [],
        'law': [],
        'location': [],
        'organization': [],
        'person': [],
        'quantity': [],
        'time': [],
        'event': [],
    }
    last_label, words = None, []
    for word, label in result:
        if last_label != label and last_label:
            joined = ' '.join(words)
            if joined not in output[last_label]:
                output[last_label].append(joined)
            words = []
            last_label = label
            words.append(word)

        else:
            if not last_label:
                last_label = label
            words.append(word)
    return output


def cluster_scatter(
    corpus,
    titles = None,
    colors = None,
    stemming = True,
    max_df = 0.95,
    min_df = 2,
    ngram = (1, 3),
    cleaning = simple_textcleaning,
    vectorizer = 'bow',
    stop_words = None,
    num_clusters = 5,
    clustering = KMeans,
    decomposition = MDS,
    figsize = (17, 9),
):
    """
    plot scatter plot on similar text clusters.

    Parameters
    ----------

    corpus: list
    titles: list
        list of titles, length must same with corpus.
    colors: list
        list of colors, length must same with num_clusters.
    num_clusters: int, (default=5)
        size of unsupervised clusters.
    stemming: bool, (default=True)
        If True, sastrawi_stemmer will apply.
    max_df: float, (default=0.95)
        maximum of a word selected based on document frequency.
    min_df: int, (default=2)
        minimum of a word selected on based on document frequency.
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus.
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus.
    stop_words: list, (default=None)
        list of stop words to remove. If None, default is malaya.texts._text_functions.STOPWORDS
    vectorizer: str, (default='bow')
        vectorizer technique. Allowed values:

        * ``'bow'`` - Bag of Word.
        * ``'tfidf'`` - Term frequency inverse Document Frequency.
        * ``'skip-gram'`` - Bag of Word with skipping certain n-grams.

    Returns
    -------
    dictionary: {
        'X': X,
        'Y': Y,
        'labels': clusters,
        'vector': transformed_text_clean,
        'titles': titles,
    }
    """
    if not isinstance(corpus, list):
        raise ValueError('corpus must be a list')
    if not isinstance(corpus[0], str):
        raise ValueError('corpus must be list of strings')
    if not isinstance(titles, list) and titles is not None:
        raise ValueError('titles must be a list or None')
    if not isinstance(colors, list) and colors is not None:
        raise ValueError('colors must be a list or None')
    if titles:
        if len(titles) != len(corpus):
            raise ValueError('length of titles must be same with corpus')
    if colors:
        if len(colors) != num_clusters:
            raise ValueError(
                'size of colors must be same with number of clusters'
            )
    if not isinstance(vectorizer, str):
        raise ValueError('vectorizer must be a string')
    if not isinstance(stemming, bool):
        raise ValueError('bool must be a boolean')
    vectorizer = vectorizer.lower()
    if not vectorizer in ['tfidf', 'bow', 'skip-gram']:
        raise ValueError("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")
    if not isinstance(ngram, tuple):
        raise ValueError('ngram must be a tuple')
    if not len(ngram) == 2:
        raise ValueError('ngram size must equal to 2')
    if not isinstance(min_df, int):
        raise ValueError('min_df must be an integer')
    if not isinstance(max_df, float):
        raise ValueError('max_df must be a float')
    if min_df < 1:
        raise ValueError('min_df must be bigger than 0')
    if not (max_df <= 1 and max_df > 0):
        raise ValueError(
            'max_df must be bigger than 0, less than or equal to 1'
        )
    if vectorizer == 'tfidf':
        Vectorizer = TfidfVectorizer
    elif vectorizer == 'bow':
        Vectorizer = CountVectorizer
    elif vectorizer == 'skip-gram':
        Vectorizer = SkipGramVectorizer
    else:
        raise Exception("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")
    if stop_words is None:
        stop_words = STOPWORDS

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set()
    except:
        raise Exception(
            'matplotlib and seaborn not installed. Please install it and try again.'
        )

    tf_vectorizer = Vectorizer(
        ngram_range = ngram,
        min_df = min_df,
        max_df = max_df,
        stop_words = stop_words,
    )
    if cleaning is not None:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)):
            corpus[i] = sastrawi(corpus[i])
    text_clean = []
    for text in corpus:
        text_clean.append(
            ' '.join([word for word in text.split() if word not in stop_words])
        )
    tf_vectorizer.fit(text_clean)
    transformed_text_clean = tf_vectorizer.transform(text_clean)
    features = tf_vectorizer.get_feature_names()
    km = clustering(n_clusters = num_clusters)
    dist = 1 - cosine_similarity(transformed_text_clean)
    km.fit(transformed_text_clean)
    clusters = km.labels_.tolist()
    if isinstance(decomposition, MDS):
        decomposed = decomposition(
            n_components = 2, dissimilarity = 'precomputed'
        )
    else:
        decomposed = decomposition(n_components = 2)
    pos = decomposed.fit_transform(dist)
    if not titles:
        titles = []
        for i in range(transformed_text_clean.shape[0]):
            indices = np.argsort(
                np.array(transformed_text_clean[i].todense())[0]
            )[::-1]
            titles.append(' '.join([features[i] for i in indices[: ngram[1]]]))
    if not colors:
        colors = sns.color_palette(n_colors = num_clusters)
    X, Y = pos[:, 0], pos[:, 1]
    plt.figure(figsize = figsize)
    for i in np.unique(clusters):
        plt.scatter(
            X[clusters == i],
            Y[clusters == i],
            color = colors[i],
            label = 'cluster %d' % (i),
        )
    for i in range(len(X)):
        plt.text(X[i], Y[i], titles[i], size = 8)
    plt.legend()
    plt.show()
    return {
        'X': X,
        'Y': Y,
        'labels': clusters,
        'vector': transformed_text_clean,
        'titles': titles,
    }


def cluster_dendogram(
    corpus,
    titles = None,
    stemming = True,
    max_df = 0.95,
    min_df = 2,
    ngram = (1, 3),
    cleaning = simple_textcleaning,
    vectorizer = 'bow',
    stop_words = None,
    random_samples = 0.3,
    figsize = (17, 9),
    **kwargs
):
    """
    plot hierarchical dendogram with similar texts.

    Parameters
    ----------

    corpus: list
    titles: list
        list of titles, length must same with corpus.
    stemming: bool, (default=True)
        If True, sastrawi_stemmer will apply.
    max_df: float, (default=0.95)
        maximum of a word selected based on document frequency.
    min_df: int, (default=2)
        minimum of a word selected on based on document frequency.
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus.
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus.
    stop_words: list, (default=None)
        list of stop words to remove. If None, default is malaya.texts._text_functions.STOPWORDS
    vectorizer: str, (default='bow')
        vectorizer technique. Allowed values:

        * ``'bow'`` - Bag of Word.
        * ``'tfidf'`` - Term frequency inverse Document Frequency.
        * ``'skip-gram'`` - Bag of Word with skipping certain n-grams.

    Returns
    -------
    dictionary: {'linkage_matrix': linkage_matrix, 'titles': titles}
    """
    if not isinstance(corpus, list):
        raise ValueError('corpus must be a list')
    if not isinstance(corpus[0], str):
        raise ValueError('corpus must be list of strings')
    if not isinstance(titles, list) and titles is not None:
        raise ValueError('titles must be a list or None')
    if titles:
        if len(titles) != len(corpus):
            raise ValueError('length of titles must be same with corpus')
    if not isinstance(vectorizer, str):
        raise ValueError('vectorizer must be a string')
    if not isinstance(stemming, bool):
        raise ValueError('bool must be a boolean')
    vectorizer = vectorizer.lower()
    if not vectorizer in ['tfidf', 'bow', 'skip-gram']:
        raise ValueError("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")
    if not isinstance(ngram, tuple):
        raise ValueError('ngram must be a tuple')
    if not len(ngram) == 2:
        raise ValueError('ngram size must equal to 2')
    if not isinstance(min_df, int):
        raise ValueError('min_df must be an integer')
    if not isinstance(max_df, float):
        raise ValueError('max_df must be a float')
    if min_df < 1:
        raise ValueError('min_df must be bigger than 0')
    if not (max_df <= 1 and max_df > 0):
        raise ValueError(
            'max_df must be bigger than 0, less than or equal to 1'
        )
    if vectorizer == 'tfidf':
        Vectorizer = TfidfVectorizer
    elif vectorizer == 'bow':
        Vectorizer = CountVectorizer
    elif vectorizer == 'skip-gram':
        Vectorizer = SkipGramVectorizer
    else:
        raise ValueError("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set()
    except:
        raise Exception(
            'matplotlib and seaborn not installed. Please install it and try again.'
        )
    if stop_words is None:
        stop_words = STOPWORDS

    tf_vectorizer = Vectorizer(
        ngram_range = ngram,
        min_df = min_df,
        max_df = max_df,
        stop_words = stop_words,
        **kwargs
    )
    corpus = random.sample(corpus, k = int(random_samples * len(corpus)))
    if cleaning is not None:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)):
            corpus[i] = sastrawi(corpus[i])
    text_clean = []
    for text in corpus:
        text_clean.append(
            ' '.join([word for word in text.split() if word not in stop_words])
        )
    tf_vectorizer.fit(text_clean)
    transformed_text_clean = tf_vectorizer.transform(text_clean)
    features = tf_vectorizer.get_feature_names()
    dist = 1 - cosine_similarity(transformed_text_clean)
    linkage_matrix = ward(dist)
    if not titles:
        titles = []
        for i in range(transformed_text_clean.shape[0]):
            indices = np.argsort(
                np.array(transformed_text_clean[i].todense())[0]
            )[::-1]
            titles.append(' '.join([features[i] for i in indices[: ngram[1]]]))
    plt.figure(figsize = figsize)
    ax = dendrogram(linkage_matrix, orientation = 'right', labels = titles)
    plt.tick_params(
        axis = 'x',
        which = 'both',
        bottom = 'off',
        top = 'off',
        labelbottom = 'off',
    )
    plt.tight_layout()
    plt.show()
    return {'linkage_matrix': linkage_matrix, 'titles': titles}


def cluster_graph(
    corpus,
    titles = None,
    colors = None,
    threshold = 0.3,
    stemming = True,
    max_df = 0.95,
    min_df = 2,
    ngram = (1, 3),
    cleaning = simple_textcleaning,
    vectorizer = 'bow',
    stop_words = None,
    num_clusters = 5,
    clustering = KMeans,
    figsize = (17, 9),
    with_labels = True,
    **kwargs
):
    """
    plot undirected graph with similar texts.

    Parameters
    ----------
    corpus: list
    titles: list
        list of titles, length must same with corpus.
    colors: list
        list of colors, length must same with num_clusters.
    threshold: float, (default=0.3)
        threshold to assume similarity for covariance matrix.
    num_clusters: int, (default=5)
        size of unsupervised clusters.
    stemming: bool, (default=True)
        If True, sastrawi_stemmer will apply.
    max_df: float, (default=0.95)
        maximum of a word selected based on document frequency.
    min_df: int, (default=2)
        minimum of a word selected on based on document frequency.
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus.
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus.
    stop_words: list, (default=None)
        list of stop words to remove. If None, default is malaya.texts._text_functions.STOPWORDS
    vectorizer: str, (default='bow')
        vectorizer technique. Allowed values:

        * ``'bow'`` - Bag of Word.
        * ``'tfidf'`` - Term frequency inverse Document Frequency.
        * ``'skip-gram'`` - Bag of Word with skipping certain n-grams.

    Returns
    -------
    dictionary: {
        'G': G,
        'pos': pos,
        'node_colors': node_colors,
        'node_labels': node_labels,
    }
    """
    if not isinstance(corpus, list):
        raise ValueError('corpus must be a list')
    if not isinstance(corpus[0], str):
        raise ValueError('corpus must be list of strings')
    if not isinstance(titles, list) and titles is not None:
        raise ValueError('titles must be a list or None')
    if not isinstance(colors, list) and colors is not None:
        raise ValueError('colors must be a list or None')
    if titles:
        if len(titles) != len(corpus):
            raise ValueError('length of titles must be same with corpus')
    if colors:
        if len(colors) != num_clusters:
            raise ValueError(
                'size of colors must be same with number of clusters'
            )
    if not isinstance(vectorizer, str):
        raise ValueError('vectorizer must be a string')
    if not isinstance(stemming, bool):
        raise ValueError('bool must be a boolean')
    vectorizer = vectorizer.lower()
    if not vectorizer in ['tfidf', 'bow', 'skip-gram']:
        raise ValueError("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")
    if not isinstance(ngram, tuple):
        raise ValueError('ngram must be a tuple')
    if not len(ngram) == 2:
        raise ValueError('ngram size must equal to 2')
    if not isinstance(min_df, int):
        raise ValueError('min_df must be an integer')
    if not isinstance(max_df, float):
        raise ValueError('max_df must be a float')
    if min_df < 1:
        raise ValueError('min_df must be bigger than 0')
    if not (max_df <= 1 and max_df > 0):
        raise ValueError(
            'max_df must be bigger than 0, less than or equal to 1'
        )
    if not isinstance(threshold, float):
        raise ValueError('threshold must be a float')
    if not (threshold <= 1 and threshold > 0):
        raise ValueError(
            'threshold must be bigger than 0, less than or equal to 1'
        )

    if vectorizer == 'tfidf':
        Vectorizer = TfidfVectorizer
    elif vectorizer == 'bow':
        Vectorizer = CountVectorizer
    elif vectorizer == 'skip-gram':
        Vectorizer = SkipGramVectorizer
    else:
        raise ValueError("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import networkx as nx
        import networkx.drawing.layout as nxlayout

        sns.set()
    except:
        raise Exception(
            'matplotlib, seaborn, networkx not installed. Please install it and try again.'
        )
    if stop_words is None:
        stop_words = STOPWORDS

    tf_vectorizer = Vectorizer(
        ngram_range = ngram,
        min_df = min_df,
        max_df = max_df,
        stop_words = stop_words,
        **kwargs
    )
    if cleaning is not None:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)):
            corpus[i] = sastrawi(corpus[i])
    text_clean = []
    for text in corpus:
        text_clean.append(
            ' '.join([word for word in text.split() if word not in stop_words])
        )
    tf_vectorizer.fit(text_clean)
    DxT = tf_vectorizer.transform(text_clean)
    DxD = np.dot(DxT, DxT.T)
    km = clustering(n_clusters = num_clusters)
    km.fit(DxT)
    clusters = km.labels_.tolist()
    features = tf_vectorizer.get_feature_names()
    if not titles:
        titles = []
        for i in range(DxT.shape[0]):
            indices = np.argsort(np.array(DxT[i].todense())[0])[::-1]
            titles.append(' '.join([features[i] for i in indices[: ngram[1]]]))
    if not colors:
        colors = sns.color_palette(n_colors = num_clusters)
    G = nx.Graph()
    for i in range(DxT.shape[0]):
        G.add_node(i, text = titles[i], label = clusters[i])
    dense_DxD = DxD.toarray()
    len_dense = len(dense_DxD)
    for i in range(len_dense):
        for j in range(len_dense):
            if j == i:
                continue
            if dense_DxD[i, j] >= threshold:
                weight = dense_DxD[i, j]
                G.add_edge(i, j, weight = weight)
    node_colors, node_labels = [], {}
    for node in G:
        node_colors.append(colors[G.node[node]['label']])
        node_labels[node] = G.node[node]['text']
    pos = nxlayout.fruchterman_reingold_layout(
        G, k = 1.5 / np.sqrt(len(G.nodes()))
    )
    plt.figure(figsize = figsize)
    if with_labels:
        nx.draw(G, node_color = node_colors, pos = pos, labels = node_labels)
    else:
        nx.draw(G, node_color = node_colors, pos = pos)
    return {
        'G': G,
        'pos': pos,
        'node_colors': node_colors,
        'node_labels': node_labels,
    }


def cluster_entity_linking(
    corpus,
    entity_model,
    topic_modeling_model,
    topic_decomposition = 2,
    topic_length = 10,
    threshold = 0.3,
    fuzzy_ratio = 70,
    accepted_entities = ['law', 'location', 'organization', 'person', 'event'],
    colors = None,
    max_df = 1.0,
    min_df = 1,
    ngram = (2, 3),
    stemming = True,
    cleaning = simple_textcleaning,
    vectorizer = 'bow',
    stop_words = None,
    figsize = (17, 9),
    **kwargs
):
    """
    plot undirected graph for Entities and topics relationship.

    Parameters
    ----------
    corpus: list or str
    titles: list
        list of titles, length must same with corpus.
    colors: list
        list of colors, length must same with num_clusters.
    threshold: float, (default=0.3)
        threshold to assume similarity for covariance matrix.
    topic_decomposition: int, (default=2)
        size of decomposition.
    topic_length: int, (default=10)
        size of topic models.
    fuzzy_ratio: int, (default=70)
        size of ratio for fuzzywuzzy.
    stemming: bool, (default=True)
        If True, sastrawi_stemmer will apply.
    max_df: float, (default=0.95)
        maximum of a word selected based on document frequency.
    min_df: int, (default=2)
        minimum of a word selected on based on document frequency.
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus.
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus.
    stop_words: list, (default=None)
        list of stop words to remove. If None, default is malaya.texts._text_functions.STOPWORDS
    vectorizer: str, (default='bow')
        vectorizer technique. Allowed values:

        * ``'bow'`` - Bag of Word.
        * ``'tfidf'`` - Term frequency inverse Document Frequency.
        * ``'skip-gram'`` - Bag of Word with skipping certain n-grams.

    Returns
    -------
    dictionary: {
        'G': G,
        'pos': pos,
        'node_colors': node_colors,
        'node_labels': node_labels,
    }
    """
    if not isinstance(corpus, list) and not isinstance(corpus, str):
        raise ValueError('corpus must be a list')
    if isinstance(corpus, list):
        if not isinstance(corpus[0], str):
            raise ValueError('corpus must be list of strings')
    if not isinstance(colors, list) and colors is not None:
        raise ValueError('colors must be a list or None')
    if not isinstance(vectorizer, str):
        raise ValueError('vectorizer must be a string')
    if not isinstance(stemming, bool):
        raise ValueError('bool must be a boolean')
    vectorizer = vectorizer.lower()
    if not vectorizer in ['tfidf', 'bow', 'skip-gram']:
        raise ValueError("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")
    if not isinstance(ngram, tuple):
        raise ValueError('ngram must be a tuple')
    if not len(ngram) == 2:
        raise ValueError('ngram size must equal to 2')
    if not isinstance(min_df, int):
        raise ValueError('min_df must be an integer')
    if not isinstance(topic_decomposition, int):
        raise ValueError('topic_decomposition must be an integer')
    if not isinstance(topic_length, int):
        raise ValueError('topic_length must be an integer')
    if not isinstance(fuzzy_ratio, int):
        raise ValueError('fuzzy_ratio must be an integer')
    if not isinstance(max_df, float):
        raise ValueError('max_df must be a float')
    if min_df < 1:
        raise ValueError('min_df must be bigger than 0')
    if not (max_df <= 1 and max_df > 0):
        raise ValueError(
            'max_df must be bigger than 0, less than or equal to 1'
        )
    if not (fuzzy_ratio > 0 and fuzzy_ratio <= 100):
        raise ValueError(
            'fuzzy_ratio must be bigger than 0, less than or equal to 100'
        )
    if not isinstance(threshold, float):
        raise ValueError('threshold must be a float')
    if not (threshold <= 1 and threshold > 0):
        raise ValueError(
            'threshold must be bigger than 0, less than or equal to 1'
        )
    if stop_words is None:
        stop_words = STOPWORDS

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import networkx as nx
        import networkx.drawing.layout as nxlayout

        sns.set()
    except:
        raise Exception(
            'matplotlib, seaborn, networkx not installed. Please install it and try again.'
        )

    if vectorizer == 'tfidf':
        Vectorizer = TfidfVectorizer
    elif vectorizer == 'bow':
        Vectorizer = CountVectorizer
    elif vectorizer == 'skip-gram':
        Vectorizer = SkipGramVectorizer
    else:
        raise ValueError("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")

    if isinstance(corpus, str):
        corpus = corpus.replace('\n', '.')
        corpus = split_by_dot(corpus)
    else:
        corpus = [c + '.' for c in corpus]
        corpus = ' '.join(corpus)
        corpus = re.findall('(?=\S)[^.\n]+(?<=\S)', corpus)
    corpus = [string for string in corpus if len(string) > 5]

    if not colors:
        colors = sns.color_palette(n_colors = len(accepted_entities) + 1)
    else:
        if len(colors) != (len(accepted_entities) + 1):
            raise ValueError(
                'len of colors must same as %d' % (len(accepted_entities) + 1)
            )

    topic_model = topic_modeling_model(
        corpus,
        topic_decomposition,
        stemming = stemming,
        vectorizer = vectorizer,
        ngram = ngram,
        max_df = max_df,
        min_df = min_df,
    )
    topics = []
    for no, topic in enumerate(topic_model.comp.components_):
        for i in topic.argsort()[: -topic_length - 1 : -1]:
            topics.append(topic_model.features[i])

    entities_cluster = {entity: [] for entity in accepted_entities}
    for string in corpus:
        entities_clustered = cluster_entities(entity_model.predict(string))
        for entity in accepted_entities:
            entities_cluster[entity].extend(entities_clustered[entity])
    for entity in accepted_entities:
        entities_cluster[entity] = cluster_words(
            list(set(entities_cluster[entity]))
        )

    topics = cluster_words(list(set(topics)))
    color_dict = {topic: colors[-1] for topic in topics}
    for no, entity in enumerate(accepted_entities):
        for e in entities_cluster[entity]:
            topics.append(e)
            color_dict[e] = colors[no]

    topics_corpus = []
    for topic in topics:
        nested_corpus = []
        for string in corpus:
            if (
                topic in string
                or fuzz.token_set_ratio(topic, string) >= fuzzy_ratio
            ):
                nested_corpus.append(string)
        topics_corpus.append(' '.join(nested_corpus))

    tf_vectorizer = Vectorizer(
        ngram_range = ngram,
        min_df = min_df,
        max_df = max_df,
        stop_words = stop_words,
        **kwargs
    )
    if cleaning is not None:
        for i in range(len(topics_corpus)):
            topics_corpus[i] = cleaning(topics_corpus[i])
    if stemming:
        for i in range(len(topics_corpus)):
            topics_corpus[i] = sastrawi(topics_corpus[i])

    tf_vectorizer.fit(topics_corpus)
    DxT = tf_vectorizer.transform(topics_corpus)
    DxD = np.dot(DxT, DxT.T)

    G = nx.Graph()
    for i in range(DxT.shape[0]):
        G.add_node(i, text = topics[i], label = topics[i])

    dense_DxD = DxD.toarray()
    len_dense = len(dense_DxD)
    for i in range(len_dense):
        for j in range(len_dense):
            if j == i:
                continue
            if dense_DxD[i, j] >= threshold:
                weight = dense_DxD[i, j]
                G.add_edge(i, j, weight = weight)
    node_colors, node_labels = [], {}
    for node in G:
        node_colors.append(color_dict[G.node[node]['label']])
        node_labels[node] = G.node[node]['text']
    pos = nxlayout.fruchterman_reingold_layout(
        G, k = 1.5 / np.sqrt(len(G.nodes()))
    )
    f = plt.figure(figsize = figsize)
    ax = f.add_subplot(1, 1, 1)
    for no, entity in enumerate(accepted_entities):
        ax.plot([0], [0], color = colors[no], label = entity)
    ax.plot([0], [0], color = colors[-1], label = 'topics')
    nx.draw(
        G, node_color = node_colors, pos = pos, labels = node_labels, ax = ax
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
    return {
        'G': G,
        'pos': pos,
        'node_colors': node_colors,
        'node_labels': node_labels,
    }
