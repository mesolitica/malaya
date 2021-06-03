from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from malaya.text.function import (
    simple_textcleaning,
    split_into_sentences,
    get_stopwords,
)
from malaya.function import validator
from herpetologist import check_type
from typing import List, Tuple, Callable

import numpy as np
import re
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
    'PUNCT',
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
    'X',
]


@check_type
def cluster_words(list_words: List[str], lowercase: bool = False):
    """
    cluster similar words based on structure, eg, ['mahathir mohamad', 'mahathir'] = ['mahathir mohamad'].
    big O = n^2

    Parameters
    ----------
    list_words : List[str]
    lowercase: bool, optional (default=True)
        if True, will group using lowercase but maintain the original form.

    Returns
    -------
    string: List[str]
    """

    dict_words = {}
    for word in list_words:
        found = False
        for key in dict_words.keys():
            if lowercase:
                check = [
                    word.lower() in inside.lower() for inside in dict_words[key]
                ]
            else:
                check = [word in inside for inside in dict_words[key]]
            if word in key or any(check):
                dict_words[key].append(word)
                found = True
            if key in word:
                dict_words[key].append(word)
        if not found:
            dict_words[word] = [word]
    results = []
    for key, words in dict_words.items():
        results.append(max(list(set([key] + words)), key=len))
    return list(set(results))


@check_type
def cluster_pos(result: List[Tuple[str, str]]):
    """
    cluster similar POS.

    Parameters
    ----------
    result: List[Tuple[str, str]]

    Returns
    -------
    result: Dict[str, List[str]]
    """

    if not all([i[1] in _accepted_pos for i in result]):
        raise ValueError(
            'elements of result must be a subset or equal of supported POS, please run `malaya.pos.describe()` to get supported POS.'
        )

    output = {p: [] for p in _accepted_pos}
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
    output[last_label].append(' '.join(words))
    return output


@check_type
def cluster_entities(result: List[Tuple[str, str]]):
    """
    cluster similar Entities.

    Parameters
    ----------
    result: List[Tuple[str, str]]

    Returns
    -------
    result: Dict[str, List[str]]
    """
    if not all([i[1] in _accepted_entities for i in result]):
        raise ValueError(
            'elements of result must be a subset or equal of supported Entities, please run `malaya.entity.describe` to get supported Entities.'
        )

    output = {e: [] for e in _accepted_entities}
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
    output[last_label].append(' '.join(words))
    return output


@check_type
def cluster_tagging(result: List[Tuple[str, str]]):
    """
    cluster any tagging results, as long the data passed `[(string, label), (string, label)]`.

    Parameters
    ----------
    result: List[Tuple[str, str]]

    Returns
    -------
    result: Dict[str, List[str]]
    """

    _, labels = list(zip(*result))

    output = {l: [] for l in labels}
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
    output[last_label].append(' '.join(words))
    return output


@check_type
def cluster_scatter(
    corpus: List[str],
    vectorizer,
    num_clusters: int = 5,
    titles: List[str] = None,
    colors: List[str] = None,
    stopwords=get_stopwords,
    cleaning=simple_textcleaning,
    clustering=KMeans,
    decomposition=MDS,
    ngram: Tuple[int, int] = (1, 3),
    figsize: Tuple[int, int] = (17, 9),
    batch_size: int = 20,
):
    """
    plot scatter plot on similar text clusters.

    Parameters
    ----------

    corpus: List[str]
    vectorizer: class
        vectorizer class.
    num_clusters: int, (default=5)
        size of unsupervised clusters.
    titles: List[str], (default=None)
        list of titles, length must same with corpus.
    colors: List[str], (default=None)
        list of colors, length must same with num_clusters.
    stopwords: List[str], (default=malaya.texts.function.get_stopwords)
        A callable that returned a List[str], or a List[str], or a Tuple[str]
    ngram: Tuple[int, int], (default=(1,3))
        n-grams size to train a corpus.
    cleaning: function, (default=malaya.texts.function.simple_textcleaning)
        function to clean the corpus.
    batch_size: int, (default=10)
        size of strings for each vectorization and attention. Only useful if use transformer vectorizer.

    Returns
    -------
    dictionary: {'X': X, 'Y': Y, 'labels': clusters, 'vector': transformed_text_clean, 'titles': titles}
    """

    if titles:
        if len(titles) != len(corpus):
            raise ValueError('length of titles must be same with corpus')
    if colors:
        if len(colors) != num_clusters:
            raise ValueError(
                'size of colors must be same with number of clusters'
            )

    validator.validate_object_methods(
        vectorizer, ['vectorize', 'fit'], 'vectorizer'
    )
    stopwords = validator.validate_stopwords(stopwords)
    validator.validate_function(cleaning, 'cleaning')

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set()
    except BaseException:
        raise ModuleNotFoundError(
            'matplotlib and seaborn not installed. Please install it and try again.'
        )

    if cleaning:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    text_clean = []
    for text in corpus:
        text_clean.append(
            ' '.join([word for word in text.split() if word not in stopwords])
        )

    if hasattr(vectorizer, 'fit'):
        vectorizer.fit(text_clean)
        transformed_text_clean = vectorizer.transform(text_clean)
        features = vectorizer.get_feature_names()
    else:
        transformed_text_clean, attentions = [], []
        for i in range(0, len(text_clean), batch_size):
            index = min(i + batch_size, len(text_clean))
            transformed_text_clean.append(
                vectorizer.vectorize(text_clean[i:index])
            )
            if hasattr(vectorizer, 'attention'):
                attentions.extend(vectorizer.attention(text_clean[i:index]))
            else:
                t = []
                for s in text_clean[i:index]:
                    t.append([(w, 1.0) for w in s.split()])
                attentions.extend(t)
        transformed_text_clean = np.concatenate(
            transformed_text_clean, axis=0
        )
    km = clustering(n_clusters=num_clusters)
    dist = 1 - cosine_similarity(transformed_text_clean)
    km.fit(transformed_text_clean)
    clusters = km.labels_.tolist()
    if isinstance(decomposition, MDS):
        decomposed = decomposition(
            n_components=2, dissimilarity='precomputed'
        )
    else:
        decomposed = decomposition(n_components=2)
    pos = decomposed.fit_transform(dist)
    if not titles:
        titles = []
        for i in range(transformed_text_clean.shape[0]):

            if hasattr(vectorizer, 'fit'):
                indices = np.argsort(
                    np.array(transformed_text_clean[i].todense())[0]
                )[::-1]
                titles.append(
                    ' '.join([features[i] for i in indices[: ngram[1]]])
                )
            else:
                attentions[i].sort(key=lambda x: x[1])
                titles.append(
                    ' '.join([i[0] for i in attentions[i][-ngram[1]:]])
                )

    if not colors:
        colors = sns.color_palette(n_colors=num_clusters)
    X, Y = pos[:, 0], pos[:, 1]
    plt.figure(figsize=figsize)
    for i in np.unique(clusters):
        plt.scatter(
            X[clusters == i],
            Y[clusters == i],
            color=colors[i],
            label='cluster %d' % (i),
        )
    for i in range(len(X)):
        plt.text(X[i], Y[i], titles[i], size=8)
    plt.legend()
    plt.show()
    return {
        'X': X,
        'Y': Y,
        'labels': clusters,
        'vector': transformed_text_clean,
        'titles': titles,
    }


@check_type
def cluster_dendogram(
    corpus: List[str],
    vectorizer,
    titles: List[str] = None,
    stopwords=get_stopwords,
    cleaning=simple_textcleaning,
    random_samples: float = 0.3,
    ngram: Tuple[int, int] = (1, 3),
    figsize: Tuple[int, int] = (17, 9),
    batch_size: int = 20,
):
    """
    plot hierarchical dendogram with similar texts.

    Parameters
    ----------

    corpus: List[str]
    vectorizer: class
        vectorizer class.
    num_clusters: int, (default=5)
        size of unsupervised clusters.
    titles: List[str], (default=None)
        list of titles, length must same with corpus.
    stopwords: List[str], (default=malaya.texts.function.get_stopwords)
        A callable that returned a List[str], or a List[str], or a Tuple[str]
    cleaning: function, (default=malaya.text.function.simple_textcleaning)
        function to clean the corpus.
    random_samples: float, (default=0.3)
        random samples from the corpus, 0.3 means 30%.
    ngram: Tuple[int, int], (default=(1,3))
        n-grams size to train a corpus.
    batch_size: int, (default=20)
        size of strings for each vectorization and attention. Only useful if use transformer vectorizer.

    Returns
    -------
    dictionary: {'linkage_matrix': linkage_matrix, 'titles': titles}
    """

    if titles:
        if len(titles) != len(corpus):
            raise ValueError('length of titles must be same with corpus')

    validator.validate_object_methods(
        vectorizer, ['vectorize', 'fit'], 'vectorizer'
    )
    stopwords = validator.validate_stopwords(stopwords)
    validator.validate_function(cleaning, 'cleaning')

    if not (random_samples < 1 and random_samples > 0):
        raise ValueError('random_samples must be between 0 and 1')

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.cluster.hierarchy import ward, dendrogram

        sns.set()
    except BaseException:
        raise ModuleNotFoundError(
            'matplotlib and seaborn not installed. Please install it and try again.'
        )

    corpus = random.sample(corpus, k=int(random_samples * len(corpus)))

    if cleaning is not None:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    text_clean = []
    for text in corpus:
        text_clean.append(
            ' '.join([word for word in text.split() if word not in stopwords])
        )

    if hasattr(vectorizer, 'fit'):
        vectorizer.fit(text_clean)
        transformed_text_clean = vectorizer.transform(text_clean)
        features = vectorizer.get_feature_names()
    else:
        transformed_text_clean, attentions = [], []
        for i in range(0, len(text_clean), batch_size):
            index = min(i + batch_size, len(text_clean))
            transformed_text_clean.append(
                vectorizer.vectorize(text_clean[i:index])
            )
            if hasattr(vectorizer, 'attention'):
                attentions.extend(vectorizer.attention(text_clean[i:index]))
            else:
                t = []
                for s in text_clean[i:index]:
                    t.append([(w, 1.0) for w in s.split()])
                attentions.extend(t)
        transformed_text_clean = np.concatenate(
            transformed_text_clean, axis=0
        )

    dist = 1 - cosine_similarity(transformed_text_clean)
    linkage_matrix = ward(dist)
    if not titles:
        titles = []
        for i in range(transformed_text_clean.shape[0]):

            if hasattr(vectorizer, 'fit'):
                indices = np.argsort(
                    np.array(transformed_text_clean[i].todense())[0]
                )[::-1]
                titles.append(
                    ' '.join([features[i] for i in indices[: ngram[1]]])
                )
            else:
                attentions[i].sort(key=lambda x: x[1])
                titles.append(
                    ' '.join([i[0] for i in attentions[i][-ngram[1]:]])
                )
    plt.figure(figsize=figsize)
    ax = dendrogram(linkage_matrix, orientation='right', labels=titles)
    plt.tick_params(
        axis='x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off',
    )
    plt.tight_layout()
    plt.show()
    return {'linkage_matrix': linkage_matrix, 'titles': titles}


@check_type
def cluster_graph(
    corpus: List[str],
    vectorizer,
    threshold: float = 0.9,
    num_clusters: int = 5,
    titles: List[str] = None,
    colors: List[str] = None,
    stopwords=get_stopwords,
    ngram: Tuple[int, int] = (1, 3),
    cleaning=simple_textcleaning,
    clustering=KMeans,
    figsize: Tuple[int, int] = (17, 9),
    with_labels: bool = True,
    batch_size: int = 20,
):
    """
    plot undirected graph with similar texts.

    Parameters
    ----------

    corpus: List[str]
    vectorizer: class
        vectorizer class.
    threshold: float, (default=0.9)
        0.9 means, 90% above absolute pearson correlation.
    num_clusters: int, (default=5)
        size of unsupervised clusters.
    titles: List[str], (default=True)
        list of titles, length must same with corpus.
    stopwords: List[str], (default=malaya.texts.function.get_stopwords)
        A callable that returned a List[str] or List[str] or Tuple[str].
    cleaning: function, (default=malaya.texts.function.simple_textcleaning)
        function to clean the corpus.
    ngram: Tuple[int, int], (default=(1,3))
        n-grams size to train a corpus.
    batch_size: int, (default=20)
        size of strings for each vectorization and attention. Only useful if use transformer vectorizer.

    Returns
    -------
    dictionary: {'G': G, 'pos': pos, 'node_colors': node_colors, 'node_labels': node_labels}
    """
    validator.validate_object_methods(
        vectorizer, ['vectorize', 'fit'], 'vectorizer'
    )
    stopwords = validator.validate_stopwords(stopwords)
    validator.validate_function(cleaning, 'cleaning')
    if titles:
        if len(titles) != len(corpus):
            raise ValueError('length of titles must be same with corpus')
    if colors:
        if len(colors) != num_clusters:
            raise ValueError(
                'size of colors must be same with number of clusters'
            )
    if not (threshold <= 1 and threshold > 0):
        raise ValueError(
            'threshold must be bigger than 0, less than or equal to 1'
        )

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import networkx as nx
        import networkx.drawing.layout as nxlayout
        import pandas as pd

        sns.set()
    except BaseException:
        raise ModuleNotFoundError(
            'matplotlib, seaborn, networkx not installed. Please install it and try again.'
        )

    if cleaning is not None:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    text_clean = []
    for text in corpus:
        text_clean.append(
            ' '.join([word for word in text.split() if word not in stopwords])
        )

    if hasattr(vectorizer, 'fit'):
        vectorizer.fit(text_clean)
        transformed_text_clean = vectorizer.transform(text_clean).todense()
        features = vectorizer.get_feature_names()
    else:
        transformed_text_clean, attentions = [], []
        for i in range(0, len(text_clean), batch_size):
            index = min(i + batch_size, len(text_clean))
            transformed_text_clean.append(
                vectorizer.vectorize(text_clean[i:index])
            )
            if hasattr(vectorizer, 'attention'):
                attentions.extend(vectorizer.attention(text_clean[i:index]))
            else:
                t = []
                for s in text_clean[i:index]:
                    t.append([(w, 1.0) for w in s.split()])
                attentions.extend(t)
        transformed_text_clean = np.concatenate(
            transformed_text_clean, axis=0
        )

    DxT = transformed_text_clean
    DxD = np.abs(pd.DataFrame(DxT.T).corr()).values
    km = clustering(n_clusters=num_clusters)
    km.fit(DxT)
    clusters = km.labels_.tolist()

    if not titles:
        titles = []
        for i in range(transformed_text_clean.shape[0]):

            if hasattr(vectorizer, 'fit'):
                indices = np.argsort(np.array(transformed_text_clean[i])[0])[
                    ::-1
                ]
                titles.append(
                    ' '.join([features[i] for i in indices[: ngram[1]]])
                )
            else:
                attentions[i].sort(key=lambda x: x[1])
                titles.append(
                    ' '.join([i[0] for i in attentions[i][-ngram[1]:]])
                )

    if not colors:
        colors = sns.color_palette(n_colors=num_clusters)
    G = nx.Graph()
    for i in range(DxT.shape[0]):
        G.add_node(i, text=titles[i], label=clusters[i])

    len_dense = len(DxD)
    for i in range(len_dense):
        for j in range(len_dense):
            if j == i:
                continue
            if DxD[i, j] >= threshold:
                weight = DxD[i, j]
                G.add_edge(i, j, weight=weight)
    node_colors, node_labels = [], {}
    for node in G:
        node_colors.append(colors[G.node[node]['label']])
        node_labels[node] = G.node[node]['text']
    pos = nxlayout.fruchterman_reingold_layout(
        G, k=1.5 / np.sqrt(len(G.nodes()))
    )
    plt.figure(figsize=figsize)
    if with_labels:
        nx.draw(G, node_color=node_colors, pos=pos, labels=node_labels)
    else:
        nx.draw(G, node_color=node_colors, pos=pos)

    return {
        'G': G,
        'pos': pos,
        'node_colors': node_colors,
        'node_labels': node_labels,
    }


def cluster_entity_linking(
    corpus: List[str],
    vectorizer,
    entity_model,
    topic_modeling_model,
    threshold: float = 0.3,
    topic_decomposition: int = 2,
    topic_length: int = 10,
    fuzzy_ratio: int = 70,
    accepted_entities: List[str] = [
        'law',
        'location',
        'organization',
        'person',
        'event',
    ],
    cleaning=simple_textcleaning,
    colors: List[str] = None,
    stopwords=get_stopwords,
    max_df: float = 1.0,
    min_df: int = 1,
    ngram: Tuple[int, int] = (2, 3),
    figsize: Tuple[int, int] = (17, 9),
    batch_size: int = 20,
):
    """
    plot undirected graph for Entities and topics relationship.

    Parameters
    ----------
    corpus: list or str
    vectorizer: class
    titles: list
        list of titles, length must same with corpus.
    colors: list
        list of colors, length must same with num_clusters.
    threshold: float, (default=0.3)
        0.3 means, 30% above absolute pearson correlation.
    topic_decomposition: int, (default=2)
        size of decomposition.
    topic_length: int, (default=10)
        size of topic models.
    fuzzy_ratio: int, (default=70)
        size of ratio for fuzzywuzzy.
    max_df: float, (default=0.95)
        maximum of a word selected based on document frequency.
    min_df: int, (default=2)
        minimum of a word selected on based on document frequency.
    ngram: tuple, (default=(1,3))
        n-grams size to train a corpus.
    cleaning: function, (default=simple_textcleaning)
        function to clean the corpus.
    stopwords: List[str], (default=malaya.texts.function.get_stopwords)
        A callable that returned a List[str] or List[str] or Tuple[str]

    Returns
    -------
    dictionary: {'G': G, 'pos': pos, 'node_colors': node_colors, 'node_labels': node_labels}
    """

    import inspect

    validator.validate_object_methods(
        vectorizer, ['vectorize', 'fit'], 'vectorizer'
    )
    stopwords = validator.validate_stopwords(stopwords)
    validator.validate_function(cleaning, 'cleaning')

    if 'max_df' not in inspect.getargspec(topic_modeling_model)[0]:
        raise ValueError('topic_modeling_model must have `max_df` parameter')

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

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import networkx as nx
        import networkx.drawing.layout as nxlayout
        import pandas as pd
        from fuzzywuzzy import fuzz

        sns.set()
    except BaseException:
        raise ModuleNotFoundError(
            'matplotlib, seaborn, networkx, fuzzywuzzy not installed. Please install it and try again.'
        )

    if isinstance(corpus, str):
        corpus = split_into_sentences(corpus)
    else:
        corpus = '. '.join(corpus)
        corpus = split_into_sentences(corpus)

    corpus = [string for string in corpus if len(string) > 5]

    if not colors:
        colors = sns.color_palette(n_colors=len(accepted_entities) + 1)
    else:
        if len(colors) != (len(accepted_entities) + 1):
            raise ValueError(
                'len of colors must same as %d' % (len(accepted_entities) + 1)
            )

    topic_model = topic_modeling_model(
        corpus,
        topic_decomposition,
        ngram=ngram,
        max_df=max_df,
        min_df=min_df,
    )
    topics = []
    for no, topic in enumerate(topic_model.comp.components_):
        for i in topic.argsort()[: -topic_length - 1: -1]:
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

    corpus = topics_corpus

    if cleaning:
        for i in range(len(corpus)):
            corpus[i] = cleaning(corpus[i])
    text_clean = []
    for text in corpus:
        text_clean.append(
            ' '.join([word for word in text.split() if word not in stopwords])
        )

    if hasattr(vectorizer, 'fit'):
        vectorizer.fit(text_clean)
        transformed_text_clean = vectorizer.transform(text_clean).todense()
        features = vectorizer.get_feature_names()
    else:
        transformed_text_clean, attentions = [], []
        for i in range(0, len(text_clean), batch_size):
            index = min(i + batch_size, len(text_clean))
            transformed_text_clean.append(
                vectorizer.vectorize(text_clean[i:index])
            )
            if hasattr(vectorizer, 'attention'):
                attentions.extend(vectorizer.attention(text_clean[i:index]))
            else:
                attentions.extend(text_clean[i:index])
        transformed_text_clean = np.concatenate(
            transformed_text_clean, axis=0
        )

    DxT = transformed_text_clean
    DxD = np.abs(pd.DataFrame(DxT.T).corr()).values

    G = nx.Graph()
    for i in range(DxT.shape[0]):
        G.add_node(i, text=topics[i], label=topics[i])

    len_dense = len(DxD)
    for i in range(len_dense):
        for j in range(len_dense):
            if j == i:
                continue
            if DxD[i, j] >= threshold:
                weight = DxD[i, j]
                G.add_edge(i, j, weight=weight)

    node_colors, node_labels = [], {}
    for node in G:
        node_colors.append(color_dict[G.node[node]['label']])
        node_labels[node] = G.node[node]['text']
    pos = nxlayout.fruchterman_reingold_layout(
        G, k=1.5 / np.sqrt(len(G.nodes()))
    )
    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(1, 1, 1)
    for no, entity in enumerate(accepted_entities):
        ax.plot([0], [0], color=colors[no], label=entity)
    ax.plot([0], [0], color=colors[-1], label='topics')
    nx.draw(
        G, node_color=node_colors, pos=pos, labels=node_labels, ax=ax
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
