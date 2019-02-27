from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from .texts.vectorizer import SkipGramVectorizer
from .stem import sastrawi
from .texts._text_functions import simple_textcleaning, STOPWORDS
from scipy.cluster.hierarchy import ward, dendrogram
import numpy as np
import random

_accepted_pos = [
    'ADJ',
    'ADP',
    'ADV',
    'ADX',
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
    cluster similar POS

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
    cluster similar Entities

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
    stop_words = STOPWORDS,
    num_clusters = 5,
    clustering = KMeans,
    decomposition = MDS,
    figsize = (17, 9),
):
    """
    corpus: list
    titles: list
        list of titles, length must same with corpus
    colors: list
        list of colors, length must same with num_clusters
    num_clusters: int, (default=5)
        size of unsupervised clusters.
    stemming: bool, (default=True)
        If True, sastrawi_stemmer will apply
    max_df: float, (default=0.95)
        maximum of a word selected based on document frequency.
    min_df: int, (default=2)
        minimum of a word selected on based on document frequency.
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus
    stop_words: list, (default=STOPWORDS)
        list of stop words to remove
    vectorizer: str, (default='bow')
        vectorizer technique. Allowed values:

        * ``'bow'`` - Bag of Word
        * ``'tfidf'`` - Term frequency inverse Document Frequency
        * ``'skip-gram'`` - Bag of Word with skipping certain n-grams
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
    if not (isinstance(max_df, int) or isinstance(max_df, float)):
        raise ValueError('max_df must be an integer or a float')
    if min_df < 1:
        raise ValueError('min_df must be bigger than 0')
    if not (max_df < 1 and max_df > 0):
        raise ValueError('max_df must be bigger than 0, less than 1')
    if vectorizer == 'tfidf':
        Vectorizer = TfidfVectorizer
    elif vectorizer == 'bow':
        Vectorizer = CountVectorizer
    elif vectorizer == 'skip-gram':
        Vectorizer = SkipGramVectorizer
    else:
        raise Exception("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")

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
    stop_words = STOPWORDS,
    random_samples = 0.3,
    figsize = (17, 9),
):
    """
    corpus: list
    titles: list
        list of titles, length must same with corpus
    stemming: bool, (default=True)
        If True, sastrawi_stemmer will apply
    max_df: float, (default=0.95)
        maximum of a word selected based on document frequency.
    min_df: int, (default=2)
        minimum of a word selected on based on document frequency.
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus
    stop_words: list, (default=STOPWORDS)
        list of stop words to remove
    vectorizer: str, (default='bow')
        vectorizer technique. Allowed values:

        * ``'bow'`` - Bag of Word
        * ``'tfidf'`` - Term frequency inverse Document Frequency
        * ``'skip-gram'`` - Bag of Word with skipping certain n-grams
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
    if not (isinstance(max_df, int) or isinstance(max_df, float)):
        raise ValueError('max_df must be an integer or a float')
    if min_df < 1:
        raise ValueError('min_df must be bigger than 0')
    if not (max_df < 1 and max_df > 0):
        raise ValueError('max_df must be bigger than 0, less than 1')
    if vectorizer == 'tfidf':
        Vectorizer = TfidfVectorizer
    elif vectorizer == 'bow':
        Vectorizer = CountVectorizer
    elif vectorizer == 'skip-gram':
        Vectorizer = SkipGramVectorizer
    else:
        raise Exception("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")
    if vectorizer == 'tfidf':
        Vectorizer = TfidfVectorizer
    elif vectorizer == 'bow':
        Vectorizer = CountVectorizer
    elif vectorizer == 'skip-gram':
        Vectorizer = SkipGramVectorizer
    else:
        raise Exception("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")

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
    stop_words = STOPWORDS,
    num_clusters = 5,
    clustering = KMeans,
    figsize = (17, 9),
    with_labels = True,
):
    """
    corpus: list
    titles: list
        list of titles, length must same with corpus
    colors: list
        list of colors, length must same with num_clusters
    num_clusters: int, (default=5)
        size of unsupervised clusters.
    stemming: bool, (default=True)
        If True, sastrawi_stemmer will apply
    max_df: float, (default=0.95)
        maximum of a word selected based on document frequency.
    min_df: int, (default=2)
        minimum of a word selected on based on document frequency.
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus
    stop_words: list, (default=STOPWORDS)
        list of stop words to remove
    vectorizer: str, (default='bow')
        vectorizer technique. Allowed values:

        * ``'bow'`` - Bag of Word
        * ``'tfidf'`` - Term frequency inverse Document Frequency
        * ``'skip-gram'`` - Bag of Word with skipping certain n-grams
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
    if not (isinstance(max_df, int) or isinstance(max_df, float)):
        raise ValueError('max_df must be an integer or a float')
    if min_df < 1:
        raise ValueError('min_df must be bigger than 0')
    if not (max_df < 1 and max_df > 0):
        raise ValueError('max_df must be bigger than 0, less than 1')
    if vectorizer == 'tfidf':
        Vectorizer = TfidfVectorizer
    elif vectorizer == 'bow':
        Vectorizer = CountVectorizer
    elif vectorizer == 'skip-gram':
        Vectorizer = SkipGramVectorizer
    else:
        raise Exception("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")
    if vectorizer == 'tfidf':
        Vectorizer = TfidfVectorizer
    elif vectorizer == 'bow':
        Vectorizer = CountVectorizer
    elif vectorizer == 'skip-gram':
        Vectorizer = SkipGramVectorizer
    else:
        raise Exception("vectorizer must be in  ['tfidf', 'bow', 'skip-gram']")

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
        for j in range(i + 1, len_dense):
            if dense_DxD[i, j] >= threshold:
                weight = dense_DxD[i, j]
                G.add_edge(i, j, weight = weight)
    for node, degree in list(dict(G.degree()).items()):
        if degree == 0:
            G.remove_node(node)
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
