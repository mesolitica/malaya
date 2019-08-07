import pickle
import os
import numpy as np
from fuzzywuzzy import fuzz
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from . import home
from ._utils._utils import download_file, _Calculator
from ._models import _fasttext

_doc2num = _fasttext.doc2num


def load_wiki():
    """
    Return malaya pretrained wikipedia fast-text size 1024.

    Returns
    -------
    dictionary: dictionary of dictionary, reverse dictionary and vectors
    """
    if not os.path.isfile('%s/fasttext-wiki/word2vec.p' % (home)):
        print('downloading fasttext-wiki embedded')
        download_file(
            'v16/fasttext/fasttext-wiki-1024.p',
            '%s/fasttext-wiki/word2vec.p' % (home),
        )
    with open('%s/fasttext-wiki/word2vec.p' % (home), 'rb') as fopen:
        return pickle.load(fopen), (2, 3)


class fast_text:
    def __init__(self, embed_matrix, dictionary, ngrams):
        self._embed_matrix = embed_matrix
        self._dictionary = dictionary
        self._reverse_dictionary = {v: k for k, v in dictionary.items()}
        self.words = list(dictionary.keys())
        self.ngrams = ngrams

    def to_vector(self, word_list):

        """
        get vector based on list of strings.

        Parameters
        ----------
        word_list: list

        Returns
        -------
        vector: numpy
        """

        pools = []
        for word in word_list:
            word = filter(None, word.split())
            pools.append(''.join(['<%s>' % (w) for w in word]))
        word_array = _doc2num(pools, self._dictionary, ngrams = self.ngrams)
        outside_array = []
        for arr in word_array:
            outside_array.append(
                np.array([self._embed_matrix[i] for i in arr]).sum(axis = 0)
            )
        return np.array(outside_array)

    def get_vector_by_name(self, word):

        """
        get vector based on string.

        Parameters
        ----------
        word: str

        Returns
        -------
        vector: numpy
        """

        word = filter(None, word.split())
        word = ''.join(['<%s>' % (w) for w in word])
        word_array = _doc2num([word], self._dictionary, ngrams = self.ngrams)[0]
        return np.array([self._embed_matrix[i] for i in word_array]).sum(
            axis = 0
        )

    def tree_plot(
        self, labels, visualize = True, figsize = (7, 7), annotate = True
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
        list_dictionaries: list of results
        """

        idx = [e[0] if isinstance(e, list) else e for e in labels]
        embed = self.to_vector(idx)
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

        if not visualize:
            return embed, labelled, labelled

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set()
        except:
            raise Exception(
                'matplotlib and seaborn not installed. Please install it and try again.'
            )

        plt.figure(figsize = figsize)
        g = sns.clustermap(
            embed,
            cmap = 'Blues',
            xticklabels = labelled,
            yticklabels = labelled,
            annot = annotate,
        )
        plt.show()

    def scatter_plot(
        self,
        labels,
        centre = None,
        visualize = True,
        figsize = (7, 7),
        handoff = 5e-5,
    ):
        """
        plot a scatter plot based on output from calculator / n_closest / analogy.

        Parameters
        ----------
        labels : list
            output from calculator / n_closest / analogy.
        centre : str, (default=None)
            centre label, if a str, it will annotate in a red color.
        visualize : bool
            if True, it will render plt.show, else return data.
        figsize : tuple, (default=(7, 7))
            figure size for plot

        Returns
        -------
        list_dictionaries: list of results
        """
        if not isinstance(labels, list):
            raise ValueError('input must be a list')
        if not isinstance(visualize, bool):
            raise ValueError('visualize must be a boolean')
        if not isinstance(figsize, tuple):
            raise ValueError('figsize must be a tuple')

        idx = [e[0] if isinstance(e, list) else e for e in labels]
        if centre:
            if not isinstance(centre, str):
                raise ValueError('centre must be a string')
            idx.append(centre)
        embed_matrix = self.to_vector(idx)
        tsne = TSNE(n_components = 2, random_state = 0).fit_transform(
            embed_matrix
        )

        if not visualize:
            return tsne

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set()
        except:
            raise Exception(
                'matplotlib and seaborn not installed. Please install it and try again.'
            )

        plt.figure(figsize = figsize)
        plt.scatter(tsne[:, 0], tsne[:, 1])
        for label, x, y in zip(labels, tsne[:, 0], tsne[:, 1]):
            label = (
                '%s, %.3f' % (label[0], label[1])
                if isinstance(label, list)
                else label
            )
            plt.annotate(
                label,
                xy = (x, y),
                xytext = (0, 0),
                textcoords = 'offset points',
            )
        if centre:
            plt.annotate(
                centre,
                xy = (tsne[-1, 0], tsne[-1, 1]),
                xytext = (0, 0),
                textcoords = 'offset points',
                color = 'red',
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

    def calculator(
        self,
        equation,
        words_pool,
        num_closest = 5,
        metric = 'cosine',
        return_similarity = True,
    ):
        """
        calculator parser for word2vec.

        Parameters
        ----------
        equation: str
            Eg, '(mahathir + najib) - rosmah'
        words_pool: list
            Eg, ['makan','najib','minum','mahathir pm']
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
        if not isinstance(equation, str):
            raise ValueError('input must be a string')
        if not isinstance(num_closest, int):
            raise ValueError('num_closest must be an integer')
        if not isinstance(metric, str):
            raise ValueError('metric must be a string')
        if not isinstance(return_similarity, bool):
            raise ValueError('num_closest must be a boolean')
        if not isinstance(words_pool[0], str):
            raise ValueError('elements of words_pool must be a string')
        if not (len(words_pool) >= num_closest):
            raise ValueError(
                'length of words_pool must bigger or equal than num_closest'
            )
        tokens, temp = [], ''
        for char in equation:
            if char == ' ':
                continue
            if char not in '()*+-':
                temp += char
            else:
                if len(temp):
                    tokens.append(
                        ','.join(
                            self.to_vector([temp])[0].astype('str').tolist()
                        )
                    )
                    temp = ''
                tokens.append(char)
        if len(temp):
            tokens.append(
                ','.join(self.to_vector([temp])[0].astype('str').tolist())
            )

        embed_matrix = self.to_vector(words_pool)
        if return_similarity:
            nn = NearestNeighbors(num_closest, metric = metric).fit(
                embed_matrix
            )
            distances, idx = nn.kneighbors(
                _Calculator(tokens).exp().reshape((1, -1))
            )
            word_list = []
            for i in range(idx.shape[1]):
                word_list.append([words_pool[idx[0, i]], 1 - distances[0, i]])
            return word_list
        else:
            closest_indices = self.closest_row_indices(
                _Calculator(tokens).exp(), num_closest + 1, metric, embed_matrix
            )
            word_list = []
            for i in closest_indices:
                word_list.append(words_pool[i])
            return word_list

    def n_closest(
        self,
        word,
        words_pool,
        num_closest = 5,
        metric = 'cosine',
        return_similarity = True,
    ):
        """
        find nearest words based on a word.

        Parameters
        ----------
        word: str
            Eg, 'najib'
        words_pool: list
            Eg, ['makan','najib','minum','mahathir pm']
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
        if not isinstance(word, str):
            raise ValueError('input must be a string')
        if not isinstance(num_closest, int):
            raise ValueError('num_closest must be an integer')
        if not isinstance(metric, str):
            raise ValueError('metric must be a string')
        if not isinstance(words_pool, list):
            raise ValueError('words_pool must be a list')
        if not isinstance(words_pool[0], str):
            raise ValueError('elements of words_pool must be a string')
        if not (len(words_pool) >= num_closest):
            raise ValueError(
                'length of words_pool must bigger or equal than num_closest'
            )
        if not isinstance(return_similarity, bool):
            raise ValueError('num_closest must be a boolean')
        words_pool.append(word)
        embed_matrix = self.to_vector(words_pool)

        if return_similarity:
            nn = NearestNeighbors(num_closest + 1, metric = metric).fit(
                embed_matrix
            )
            distances, idx = nn.kneighbors(embed_matrix[-1].reshape((1, -1)))
            word_list = []
            for i in range(1, idx.shape[1]):
                word_list.append([words_pool[idx[0, i]], 1 - distances[0, i]])
            return word_list
        else:
            wv = embed_matrix[-1].reshape((1, -1))
            closest_indices = self.closest_row_indices(
                wv, num_closest + 1, metric, embed_matrix
            )
            word_list = []
            for i in closest_indices:
                word_list.append(words_pool[i])
            if word in word_list:
                word_list.remove(word)
            return word_list

    def closest_row_indices(self, wv, num, metric, embed_matrix):
        dist_array = np.ravel(
            cdist(embed_matrix, wv.reshape((1, -1)), metric = metric)
        )
        sorted_indices = np.argsort(dist_array)
        return sorted_indices[:num]

    def analogy(self, a, b, c, words_pool, num = 1, metric = 'cosine'):
        """
        analogy calculation, vb - va + vc

        Parameters
        ----------
        a: str
        b: str
        c: str
        words_pool: list
            Eg, ['makan','najib','minum','mahathir pm']
        num: int, (default=1)
        metric: str, (default='cosine')
            vector distance algorithm.

        Returns
        -------
        word_list: list of nearest words
        """
        if not isinstance(a, str):
            raise ValueError('a must be a string')
        if not isinstance(b, str):
            raise ValueError('b must be a string')
        if not isinstance(c, str):
            raise ValueError('c must be a string')
        if not isinstance(words_pool, list):
            raise ValueError('words_pool must be a list')
        if not isinstance(words_pool[0], str):
            raise ValueError('elements of words_pool must be a string')
        if not (len(words_pool) >= num):
            raise ValueError(
                'length of words_pool must bigger or equal than num_closest'
            )

        words_pool.extend([a, b, c])
        embed_matrix = self.to_vector(words_pool)
        va = embed_matrix[-3]
        vb = embed_matrix[-2]
        vc = embed_matrix[-1]
        vd = vb - va + vc
        closest_indices = self.closest_row_indices(
            vd, num, metric, embed_matrix
        )
        d_word_list = []
        for i in closest_indices:
            d_word_list.append(words_pool[i])
        return d_word_list

    def project_2d(self, words_pool):
        """
        project fast-text into 2d dimension.

        Parameters
        ----------
        words_pool: list
            Eg, ['makan','najib','minum','mahathir pm']

        Returns
        -------
        tsne decomposition: numpy
        """
        if not isinstance(words_pool, list):
            raise ValueError('words_pool must be a list')
        if not isinstance(words_pool[0], str):
            raise ValueError('elements of words_pool must be a string')

        tsne = TSNE(n_components = 2)
        embed_2d = tsne.fit_transform(embed_matrix = self.to_vector(words_pool))
        return embed_2d, words_pool
