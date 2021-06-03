import numpy as np
import json
import tensorflow as tf
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from malaya.text.jarowinkler import JaroWinkler
from malaya.function import check_file
from malaya.text.calculator import Calculator
from malaya.path import PATH_WORDVECTOR, S3_PATH_WORDVECTOR
from malaya.function import get_device, generate_session
from herpetologist import check_type
from typing import List, Tuple, Dict


def _load(j, npy):
    with open(j) as fopen:
        vocab = json.load(fopen)
    vector = np.load(npy)
    return vocab, vector


_wordvector_availability = {
    'wikipedia': {
        'Size (MB)': 781.7,
        'Vocab size': 763350,
        'lowercase': True,
        'Description': 'pretrained on Malay wikipedia word2vec size 256',
    },
    'socialmedia': {
        'Size (MB)': 1300,
        'Vocab size': 1294638,
        'lowercase': True,
        'Description': 'pretrained on cleaned Malay twitter and Malay instagram size 256',
    },
    'news': {
        'Size (MB)': 200.2,
        'Vocab size': 195466,
        'lowercase': True,
        'Description': 'pretrained on cleaned Malay news size 256',
    },
    'combine': {
        'Size (MB)': 1900,
        'Vocab size': 1903143,
        'lowercase': True,
        'Description': 'pretrained on cleaned Malay news + Malay social media + Malay wikipedia size 256',
    },
}


def available_wordvector():
    """
    List available transformer models.
    """
    from malaya.function import describe_availability

    return describe_availability(_wordvector_availability)


@check_type
def load(model: str = 'wikipedia', **kwargs):
    """
    Return malaya.wordvector.WordVector object.

    Parameters
    ----------
    model : str, optional (default='wikipedia')
        Model architecture supported. Allowed values:

        * ``'wikipedia'`` - pretrained on Malay wikipedia word2vec size 256.
        * ``'socialmedia'`` - pretrained on cleaned Malay twitter and Malay instagram size 256.
        * ``'news'`` - pretrained on cleaned Malay news size 256.
        * ``'combine'`` - pretrained on cleaned Malay news + Malay social media + Malay wikipedia size 256.

    Returns
    -------
    vocabulary: indices dictionary for `vector`.
    vector: np.array, 2D.
    """

    model = model.lower()
    if model not in _wordvector_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.wordvector.available_wordvector()`.'
        )

    check_file(PATH_WORDVECTOR[model], S3_PATH_WORDVECTOR[model], **kwargs)
    return _load(
        PATH_WORDVECTOR[model]['vocab'], PATH_WORDVECTOR[model]['model']
    )


class WordVector:
    @check_type
    def __init__(self, embed_matrix, dictionary: dict, **kwargs):
        """
        Parameters
        ----------
        embed_matrix: numpy array
        dictionary: dictionary
        """

        self._embed_matrix = embed_matrix
        self._dictionary = dictionary
        self._reverse_dictionary = {v: k for k, v in dictionary.items()}
        self.words = list(dictionary.keys())
        self._jarowinkler = JaroWinkler()
        device = get_device(**kwargs)
        _graph = tf.Graph()
        with _graph.as_default():
            with tf.device(device):
                self._embedding = tf.compat.v1.placeholder(
                    tf.float32, self._embed_matrix.shape
                )
                self._x = tf.compat.v1.placeholder(
                    tf.float32, [None, self._embed_matrix.shape[1]]
                )
                normed_embedding = tf.nn.l2_normalize(self._embedding, axis=1)
                normed_array = tf.nn.l2_normalize(self._x, axis=1)
                self._cosine_similarity = tf.matmul(
                    normed_array, tf.transpose(normed_embedding, [1, 0])
                )
                self._sess = generate_session(_graph, **kwargs)

    @check_type
    def get_vector_by_name(
        self, word: str, soft: bool = False, topn_soft: int = 5
    ):
        """
        get vector based on string.

        Parameters
        ----------
        word: str
        soft: bool, (default=True)
            if True, a word not in the dictionary will be replaced with nearest JaroWinkler ratio.
            if False, it will throw an exception if a word not in the dictionary.
        topn_soft: int, (default=5)
            if word not found in dictionary, will returned `topn_soft` size of similar size using jarowinkler.

        Returns
        -------
        vector: np.array, 1D
        """

        if word not in self._dictionary:
            arr = np.array(
                [self._jarowinkler.similarity(word, k) for k in self.words]
            )
            idx = (-arr).argsort()[:topn_soft]
            words = [self.words[i] for i in idx]
            if soft:
                return words
            else:
                strings = ', '.join(words)
                raise Exception(
                    'input not found in dictionary, here top-5 nearest words [%s]'
                    % (strings)
                )
        return self._embed_matrix[self._dictionary[word]]

    @check_type
    def tree_plot(
        self, labels, figsize: Tuple[int, int] = (7, 7), annotate: bool = True
    ):
        """
        plot a tree plot based on output from calculator / n_closest / analogy.

        Parameters
        ----------
        labels : list
            output from calculator / n_closest / analogy.
        visualize : bool
            if True, it will render plt.show, else return data.
        figsize : tuple, (default=(7, 7))
            figure size for plot.

        Returns
        -------
        embed: np.array, 2D.
        labelled: labels for X / Y axis.
        """

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set()
        except BaseException:
            raise ModuleNotFoundError(
                'matplotlib and seaborn not installed. Please install it and try again.'
            )

        if not isinstance(labels, list):
            raise ValueError('input must be a list')
        if not isinstance(figsize, tuple):
            raise ValueError('figsize must be a tuple')
        if not isinstance(annotate, bool):
            raise ValueError('annotate must be a boolean')

        idx = [
            self.words.index(e[0] if isinstance(e, list) else e) for e in labels
        ]
        embed = self._embed_matrix[idx]
        embed = embed.dot(embed.T)
        embed = (embed - embed.min()) / (embed.max() - embed.min())
        labelled = []
        for label in labels:
            label = (
                '%s, %.3f' % (label[0], label[1])
                if isinstance(label, list)
                else label
            )
            labelled.append(label)

        plt.figure(figsize=figsize)
        g = sns.clustermap(
            embed,
            cmap='Blues',
            xticklabels=labelled,
            yticklabels=labelled,
            annot=annotate,
        )
        plt.show()
        return embed, labelled

    @check_type
    def scatter_plot(
        self,
        labels,
        centre: str = None,
        figsize: Tuple[int, int] = (7, 7),
        plus_minus: int = 25,
        handoff: float = 5e-5,
    ):
        """
        plot a scatter plot based on output from calculator / n_closest / analogy.

        Parameters
        ----------
        labels : list
            output from calculator / n_closest / analogy
        centre : str, (default=None)
            centre label, if a str, it will annotate in a red color.
        figsize : tuple, (default=(7, 7))
            figure size for plot.

        Returns
        -------
        tsne: np.array, 2D.
        """

        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set()
        except BaseException:
            raise ModuleNotFoundError(
                'matplotlib and seaborn not installed. Please install it and try again.'
            )

        idx = [
            self.words.index(e[0] if isinstance(e, list) else e) for e in labels
        ]
        if centre:
            idx.append(self.words.index(centre))
        cp_idx = idx[:]
        for i in idx:
            cp_idx.extend(np.arange(i - plus_minus, i).tolist())
            cp_idx.extend(np.arange(i, i + plus_minus).tolist())
        tsne = TSNE(n_components=2, random_state=0).fit_transform(
            self._embed_matrix[cp_idx]
        )

        plt.figure(figsize=figsize)
        plt.scatter(tsne[:, 0], tsne[:, 1])
        for label, x, y in zip(
            labels, tsne[: len(labels), 0], tsne[: len(labels), 1]
        ):
            label = (
                '%s, %.3f' % (label[0], label[1])
                if isinstance(label, list)
                else label
            )
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
            )
        if centre:
            plt.annotate(
                centre,
                xy=(tsne[len(labels), 0], tsne[len(labels), 1]),
                xytext=(0, 0),
                textcoords='offset points',
                color='red',
            )
        plt.xlim(
            tsne[: len(idx), 0].min() + handoff,
            tsne[: len(idx), 0].max() + handoff,
        )
        plt.ylim(
            tsne[: len(idx), 1].min() + handoff,
            tsne[: len(idx), 1].max() + handoff,
        )
        plt.xticks([])
        plt.yticks([])
        plt.show()
        return tsne

    def _calculate(self, equation):
        tokens, temp = [], ''
        for char in equation:
            if char == ' ':
                continue
            if char not in '()*+-':
                temp += char
            else:
                if len(temp):
                    row = self._dictionary[
                        self.words[
                            np.argmax(
                                [
                                    self._jarowinkler.similarity(temp, k)
                                    for k in self.words
                                ]
                            )
                        ]
                    ]
                    tokens.append(
                        ','.join(
                            self._embed_matrix[row, :].astype('str').tolist()
                        )
                    )
                    temp = ''
                tokens.append(char)
        if len(temp):
            row = self._dictionary[
                self.words[
                    np.argmax(
                        [
                            self._jarowinkler.similarity(temp, k)
                            for k in self.words
                        ]
                    )
                ]
            ]
            tokens.append(
                ','.join(self._embed_matrix[row, :].astype('str').tolist())
            )
        return _Calculator(tokens).exp()

    def _batch_process(self, batch, num_closest=5, return_similarity=True):
        top_k = tf.nn.top_k(self._cosine_similarity, k=num_closest)
        results = self._sess.run(
            top_k,
            feed_dict={self._x: batch, self._embedding: self._embed_matrix},
        )
        indices = results.indices
        values = results.values
        words = []
        if not return_similarity:
            for result in indices:
                words.append([self._reverse_dictionary[i] for i in result])
        else:
            for no in range(len(results)):
                words.append(
                    [
                        (
                            self._reverse_dictionary[indices[no, i]],
                            values[no, i],
                        )
                        for i in range(len(indices[no]))
                    ]
                )
        return words

    @check_type
    def batch_calculator(
        self,
        equations: List[str],
        num_closest: int = 5,
        return_similarity: bool = False,
    ):
        """
        batch calculator parser for word2vec using tensorflow.

        Parameters
        ----------
        equations: list of str
            Eg, '[(mahathir + najib) - rosmah]'
        num_closest: int, (default=5)
            number of words closest to the result.

        Returns
        -------
        word_list: list of nearest words
        """
        batches = np.array([self._calculate(eq) for eq in equations])
        return self._batch_process(
            batches,
            num_closest=num_closest,
            return_similarity=return_similarity,
        )

    @check_type
    def calculator(
        self,
        equation: str,
        num_closest: int = 5,
        metric: str = 'cosine',
        return_similarity: bool = True,
    ):
        """
        calculator parser for word2vec.

        Parameters
        ----------
        equation: str
            Eg, '(mahathir + najib) - rosmah'
        num_closest: int, (default=5)
            number of words closest to the result.
        metric: str, (default='cosine')
            vector distance algorithm.
        return_similarity: bool, (default=True)
            if True, will return between 0-1 represents the distance.

        Returns
        -------
        word_list: list of nearest words
        """
        calculated = self._calculate(equation)
        if return_similarity:
            nn = NearestNeighbors(num_closest + 1, metric=metric).fit(
                self._embed_matrix
            )
            distances, idx = nn.kneighbors(calculated.reshape((1, -1)))
            word_list = []
            for i in range(1, idx.shape[1]):
                word_list.append(
                    [self._reverse_dictionary[idx[0, i]], 1 - distances[0, i]]
                )
            return word_list
        else:
            closest_indices = self.closest_row_indices(
                calculated, num_closest + 1, metric
            )
            word_list = []
            for i in closest_indices:
                word_list.append(self._reverse_dictionary[i])
            return word_list

    @check_type
    def batch_n_closest(
        self,
        words: List[str],
        num_closest: int = 5,
        return_similarity: bool = False,
        soft: bool = True,
    ):
        """
        find nearest words based on a batch of words using Tensorflow.

        Parameters
        ----------
        words: list
            Eg, ['najib','anwar']
        num_closest: int, (default=5)
            number of words closest to the result.
        return_similarity: bool, (default=True)
            if True, will return between 0-1 represents the distance.
        soft: bool, (default=True)
            if True, a word not in the dictionary will be replaced with nearest JaroWinkler ratio.
            if False, it will throw an exception if a word not in the dictionary.

        Returns
        -------
        word_list: list of nearest words
        """
        if soft:
            for i in range(len(words)):
                if words[i] not in self.words:
                    words[i] = self.words[
                        np.argmax(
                            [
                                self._jarowinkler.similarity(words[i], k)
                                for k in self.words
                            ]
                        )
                    ]
        else:
            for i in range(len(words)):
                if words[i] not in self.words:
                    raise Exception(
                        '%s not in dictionary, please use another word or set `soft` = True'
                        % (words[i])
                    )
        batches = np.array([self.get_vector_by_name(w) for w in words])
        return self._batch_process(
            batches,
            num_closest=num_closest,
            return_similarity=return_similarity,
        )

    @check_type
    def n_closest(
        self,
        word: str,
        num_closest: int = 5,
        metric: str = 'cosine',
        return_similarity: bool = True,
    ):
        """
        find nearest words based on a word.

        Parameters
        ----------
        word: str
            Eg, 'najib'
        num_closest: int, (default=5)
            number of words closest to the result.
        metric: str, (default='cosine')
            vector distance algorithm.
        return_similarity: bool, (default=True)
            if True, will return between 0-1 represents the distance.

        Returns
        -------
        word_list: list of nearest words
        """
        if return_similarity:
            nn = NearestNeighbors(num_closest + 1, metric=metric).fit(
                self._embed_matrix
            )
            distances, idx = nn.kneighbors(
                self.get_vector_by_name(word).reshape((1, -1))
            )
            word_list = []
            for i in range(1, idx.shape[1]):
                word_list.append(
                    [self._reverse_dictionary[idx[0, i]], 1 - distances[0, i]]
                )
            return word_list
        else:
            wv = self.get_vector_by_name(word)
            closest_indices = self.closest_row_indices(
                wv, num_closest + 1, metric
            )
            word_list = []
            for i in closest_indices:
                word_list.append(self._reverse_dictionary[i])
            if word in word_list:
                word_list.remove(word)
            return word_list

    def closest_row_indices(self, wv, num, metric):
        dist_array = np.ravel(
            cdist(self._embed_matrix, wv.reshape((1, -1)), metric=metric)
        )
        sorted_indices = np.argsort(dist_array)
        return sorted_indices[:num]

    @check_type
    def analogy(
        self, a: str, b: str, c: str, num: int = 1, metric: str = 'cosine'
    ):
        """
        analogy calculation, vb - va + vc.

        Parameters
        ----------
        a: str
        b: str
        c: str
        num: int, (default=1)
        metric: str, (default='cosine')
            vector distance algorithm.

        Returns
        -------
        word_list: list of nearest words.
        """
        if a not in self._dictionary:
            raise ValueError('a not in dictinary')
        if b not in self._dictionary:
            raise ValueError('b not in dictinary')
        if c not in self._dictionary:
            raise ValueError('c not in dictinary')
        va = self.get_vector_by_name(a)
        vb = self.get_vector_by_name(b)
        vc = self.get_vector_by_name(c)
        vd = vb - va + vc
        closest_indices = self.closest_row_indices(vd, num, metric)
        d_word_list = []
        for i in closest_indices:
            d_word_list.append(self._reverse_dictionary[i])
        return d_word_list

    @check_type
    def project_2d(self, start: int, end: int):
        """
        project word2vec into 2d dimension.

        Parameters
        ----------
        start: int
        end: int

        Returns
        -------
        embed_2d: TSNE decomposition
        word_list: words in between `start` and `end`.
        """
        tsne = TSNE(n_components=2)
        embed_2d = tsne.fit_transform(self._embed_matrix[start:end, :])
        word_list = []
        for i in range(start, end):
            word_list.append(self._reverse_dictionary[i])
        return embed_2d, word_list

    @check_type
    def network(
        self,
        word: str,
        num_closest: int = 8,
        depth: int = 4,
        min_distance: float = 0.5,
        iteration: int = 300,
        figsize: Tuple[int, int] = (15, 15),
        node_color: str = '#72bbd0',
        node_factor: int = 50,
    ):
        """
        plot a social network based on word given

        Parameters
        ----------
        word : str
            centre of social network.
        num_closest: int, (default=8)
            number of words closest to the node.
        depth: int, (default=4)
            depth of social network. More deeper more expensive to calculate, big^O(num_closest ** depth).
        min_distance: float, (default=0.5)
            minimum distance among nodes. Increase the value to increase the distance among nodes.
        iteration: int, (default=300)
            number of loops to train the social network to fit min_distace.
        figsize: tuple, (default=(15, 15))
            figure size for plot.
        node_color: str, (default='#72bbd0')
            color for nodes.
        node_factor: int, (default=10)
            size factor for depth nodes. Increase this value will increase nodes sizes based on depth.

        Returns
        -------
        G: networkx graph object
        """

        try:
            import pandas as pd
            import networkx as nx
            import matplotlib.pyplot as plt
        except BaseException:
            raise ModuleNotFoundError(
                'matplotlib, networkx and pandas not installed. Please install it and try again.'
            )

        def get_follower(
            centre,
            top,
            max_depth=depth,
            current_depth=0,
            accepted_list=[],
            data=[],
        ):
            if current_depth == max_depth:
                return data, accepted_list
            if centre in accepted_list:
                return data, accepted_list
            else:
                accepted_list.append(centre)

            closest = n_closest(
                centre, num_closest=num_closest, return_similarity=False
            )

            d = {
                'name': centre,
                'followers_ids': closest,
                'centre': top,
                'power': max_depth - current_depth,
            }

            data.append(d)

            cd = current_depth

            if cd + 1 < max_depth:
                for fid in closest:
                    data, accepted_list = get_follower(
                        fid,
                        fid,
                        max_depth=max_depth,
                        current_depth=cd + 1,
                        accepted_list=accepted_list,
                        data=data,
                    )

            return data, accepted_list

        data, accepted_list = get_follower(word, word)
        df = pd.DataFrame(data)
        id_unique = np.unique(df['centre']).tolist()

        followers = []
        for i in range(df.shape[0]):
            followers += df.followers_ids.iloc[i]
        followers = list(set(followers))

        followers = [i for i in followers if i in id_unique]
        true_followers = []
        for i in range(df.shape[0]):
            follows = df.followers_ids.iloc[i]
            follows = [k for k in follows if k in followers]
            true_followers.append(follows)

        df['true_followers'] = true_followers

        G = nx.Graph()

        for i in range(df.shape[0]):
            size = df.power.iloc[i] * node_factor
            G.add_node(df.centre.iloc[i], name=df.centre.iloc[i], size=size)

        for i in range(df.shape[0]):
            for k in df.true_followers.iloc[i]:
                G.add_edge(df.centre.iloc[i], k)

        sizes = [G.node[node]['size'] for node in G]
        names = [G.node[node]['name'] for node in G]

        labeldict = dict(zip(G.nodes(), names))
        plt.figure(figsize=figsize)
        plt.axis('equal')
        nx.draw(
            G,
            node_color=node_color,
            labels=labeldict,
            with_labels=1,
            node_size=sizes,
            pos=nx.spring_layout(G, k=min_distance, iterations=iteration),
        )
        plt.show()
        return G
