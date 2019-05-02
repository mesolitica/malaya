import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import pickle
import os
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from ._utils._paths import PATH_ELMO, S3_PATH_ELMO
from ._utils._utils import (
    check_file,
    load_graph,
    check_available,
    generate_session,
)
from ._models import _elmo
from .texts._text_functions import simple_textcleaning


def load_wiki(size = 128, validate = True):
    """
    Return malaya pretrained wikipedia ELMO size N.

    Parameters
    ----------
    size: int, (default=128)
    validate: bool, (default=True)

    Returns
    -------
    dictionary: dictionary of dictionary, reverse dictionary and vectors
    """
    if not isinstance(size, int):
        raise ValueError('size must be an integer')
    if size not in [128, 256]:
        raise ValueError('size only support [128,256]')
    if validate:
        check_file(PATH_ELMO[size], S3_PATH_ELMO[size])
    else:
        if not check_available(PATH_ELMO[size]):
            raise Exception(
                'elmo-wiki is not available, please `validate = True`'
            )
    with open(PATH_ELMO[size]['setting'], 'rb') as fopen:
        setting = pickle.load(fopen)
    g = load_graph(PATH_ELMO[size]['model'])
    return ELMO(
        g.get_tensor_by_name('import/tokens_characters:0'),
        g.get_tensor_by_name('import/tokens_characters_reverse:0'),
        g.get_tensor_by_name('import/softmax_score:0'),
        generate_session(graph = g),
        setting['dictionary'],
        setting['char_maxlen'],
        setting['steps'],
        setting['softmax_weight'],
    )


class ELMO:
    def __init__(
        self,
        X,
        X_reverse,
        logits,
        sess,
        dictionary,
        maxlen,
        steps,
        embed_matrix,
    ):
        self._X = X
        self._X_reverse = X_reverse
        self._logits = logits
        self._sess = sess
        self._dictionary = dictionary
        self.words = list(dictionary.keys())
        self._reverse_dictionary = {v: k for k, v in dictionary.items()}
        self._maxlen = maxlen
        self._steps = steps
        self._unichars_vocab = _elmo.UnicodeCharsVocabulary(
            dictionary, self._reverse_dictionary, maxlen
        )
        self._embed_matrix = embed_matrix
        _graph = tf.Graph()
        with _graph.as_default():
            self._embedding = tf.placeholder(
                tf.float32, self._embed_matrix.shape
            )
            self._x = tf.placeholder(
                tf.float32, [None, self._embed_matrix.shape[1]]
            )
            normed_embedding = tf.nn.l2_normalize(self._embedding, axis = 1)
            normed_array = tf.nn.l2_normalize(self._x, axis = 1)
            self._cosine_similarity = tf.matmul(
                normed_array, tf.transpose(normed_embedding, [1, 0])
            )
            self._sess_normed = tf.InteractiveSession()

    def get_vector_by_name(
        self,
        string,
        mode_bidirectional = 'mean',
        mode_sequence = 'mean',
        text_cleaning = simple_textcleaning,
    ):
        """
        get vector based on string. If string not found in dictionary,
        the model will change into characters representation and fit into ELMO model.

        Parameters
        ----------
        string: str
        mode_bidirectional: str, (default='mean')
            combining bidirectional outputs, Allowed values:

            * ``'mean'`` - mean the results
            * ``'sum'`` - sum the results

        mode_sequence: str, (default='mean')
            combining sequences outputs, Allowed values:

            * ``'mean'`` - mean the results
            * ``'sum'`` - sum the results
            * ``'last'`` - take the last sequence

        Returns
        -------
        vector: numpy
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(mode_bidirectional, str):
            raise ValueError('mode_bidirectional must be a string')
        if not isinstance(mode_sequence, str):
            raise ValueError('mode_sequence must be a string')
        if string not in self._dictionary:
            mode_bidirectional = mode_bidirectional.lower()
            mode_sequence = mode_sequence.lower()
            if mode_bidirectional not in ['mean', 'sum']:
                raise ValueError(
                    "mode_bidirectional only supports ['mean','sum']"
                )
            if mode_sequence not in ['mean', 'sum', 'last']:
                raise ValueError(
                    "mode_sequence only supports ['mean','sum','last']"
                )
            if text_cleaning:
                string = text_cleaning(string)
            string = string.split()
            if len(string) >= self._steps:
                raise ValueError(
                    'input must have less than %d words' % (self._steps)
                )
            string_reverse = string[:]
            string_reverse.reverse()
            batch, batch_reverse = _elmo.generate_batch(
                self._unichars_vocab,
                ' '.join(string),
                ' '.join(string_reverse),
                self._steps,
            )
            results = self._sess.run(
                self._logits,
                feed_dict = {self._X: batch, self._X_reverse: batch_reverse},
            )
            if 'mean' in mode_bidirectional:
                results = results.mean(axis = 0)
            elif 'sum' in mode_bidirectional:
                results = results.sum(axis = 0)
            else:
                raise ValueError(
                    "mode_bidirectional only supports ['mean','sum']"
                )

            results = results[: len(string)]
            if 'mean' in mode_sequence:
                results = results.mean(axis = 0)
            elif 'sum' in mode_sequence:
                results = results.sum(axis = 0)
            elif 'last' in mode_sequence:
                results = results[-1]
            else:
                raise ValueError(
                    "mode_sequence only supports ['mean','sum','last']"
                )
            return results
        else:
            return self._embed_matrix[self._dictionary[string]]

    def tree_plot(
        self,
        labels,
        notebook_mode = False,
        figsize = (7, 7),
        annotate = True,
        figname = 'fig.png',
    ):
        """
        plot a tree plot based on output from calculator / n_closest / analogy.

        Parameters
        ----------
        labels : list
            output from calculator / n_closest / analogy.
        notebook_mode : bool
            if True, it will render plt.show, else plt.savefig.
        figsize : tuple, (default=(7, 7))
            figure size for plot.
        figname : str, (default='fig.png')

        Returns
        -------
        list_dictionaries: list of results
        """
        if not isinstance(labels, list):
            raise ValueError('input must be a list')
        if not isinstance(notebook_mode, bool):
            raise ValueError('notebook_mode must be a boolean')
        if not isinstance(figsize, tuple):
            raise ValueError('figsize must be a tuple')
        if not isinstance(annotate, bool):
            raise ValueError('annotate must be a boolean')
        if not isinstance(figname, str):
            raise ValueError('figname must be a string')
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set()
        except:
            raise Exception(
                'matplotlib and seaborn not installed. Please install it and try again.'
            )
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
        plt.figure(figsize = figsize)
        g = sns.clustermap(
            embed,
            cmap = 'Blues',
            xticklabels = labelled,
            yticklabels = labelled,
            annot = annotate,
        )
        if notebook_mode:
            plt.show()
        else:
            plt.savefig(figname, bbox_inches = 'tight')

    def scatter_plot(
        self,
        labels,
        centre = None,
        notebook_mode = False,
        figsize = (7, 7),
        figname = 'fig.png',
        plus_minus = 25,
        handoff = 5e-5,
        mode_bidirectional = 'mean',
        mode_sequence = 'mean',
    ):
        """
        plot a scatter plot based on output from calculator / n_closest / analogy.

        Parameters
        ----------
        labels : list
            output from calculator / n_closest / analogy
        centre : str, (default=None)
            centre label, if a str, it will annotate in a red color.
        notebook_mode : bool
            if True, it will render plt.show, else plt.savefig.
        figsize : tuple, (default=(7, 7))
            figure size for plot.
        figname : str, (default='fig.png')

        Returns
        -------
        list_dictionaries: list of results
        """
        if not isinstance(labels, list):
            raise ValueError('input must be a list')
        if not isinstance(notebook_mode, bool):
            raise ValueError('notebook_mode must be a boolean')
        if not isinstance(figsize, tuple):
            raise ValueError('figsize must be a tuple')
        if not isinstance(figname, str):
            raise ValueError('figname must be a string')
        if not isinstance(plus_minus, int):
            raise ValueError('plus_minus must be an integer')
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set()
        except:
            raise Exception(
                'matplotlib and seaborn not installed. Please install it and try again.'
            )
        idx = [
            self.words.index(e[0] if isinstance(e, list) else e) for e in labels
        ]
        not_in = False
        if centre:
            if not isinstance(centre, str):
                raise ValueError('centre must be a string')
            if centre in self._dictionary:
                not_in = True
                idx.append(self.words.index(centre))
        cp_idx = idx[:]
        for i in idx:
            cp_idx.extend(np.arange(i - plus_minus, i).tolist())
            cp_idx.extend(np.arange(i, i + plus_minus).tolist())
        embed = self._embed_matrix[cp_idx]
        if not_in:
            vector = self.get_vector_by_name(
                centre,
                mode_bidirectional = mode_bidirectional,
                mode_sequence = mode_sequence,
            ).reshape((1, -1))
            embed = np.concatenate([embed, vector], axis = 0)

        tsne = TSNE(n_components = 2, random_state = 0).fit_transform(embed)
        plt.figure(figsize = figsize)
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
                xy = (x, y),
                xytext = (0, 0),
                textcoords = 'offset points',
            )
        if centre:
            plt.annotate(
                centre,
                xy = (tsne[len(labels), 0], tsne[len(labels), 1]),
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
        if notebook_mode:
            plt.show()
        else:
            plt.savefig(figname, bbox_inches = 'tight')

    def _batch_process(self, batch, num_closest = 5, return_similarity = True):
        top_k = tf.nn.top_k(self._cosine_similarity, k = num_closest)
        results = self._sess_normed.run(
            top_k,
            feed_dict = {self._x: batch, self._embedding: self._embed_matrix},
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

    def batch_n_closest(
        self,
        words,
        num_closest = 5,
        return_similarity = False,
        mode_bidirectional = 'mean',
        mode_sequence = 'mean',
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

        Returns
        -------
        word_list: list of nearest words
        """
        if not isinstance(words, list):
            raise ValueError('input must be list of strings')
        if not isinstance(num_closest, int):
            raise ValueError('num_closest must be an integer')
        if not isinstance(return_similarity, bool):
            raise ValueError('return_similarity must be a boolean')
        batches = np.array(
            [
                self.get_vector_by_name(
                    w,
                    mode_bidirectional = mode_bidirectional,
                    mode_sequence = mode_sequence,
                )
                for w in words
            ]
        )
        return self._batch_process(
            batches,
            num_closest = num_closest,
            return_similarity = return_similarity,
        )

    def n_closest(
        self,
        word,
        num_closest = 5,
        metric = 'cosine',
        return_similarity = True,
        mode_bidirectional = 'mean',
        mode_sequence = 'mean',
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
        if not isinstance(word, str):
            raise ValueError('input must be a string')
        if not isinstance(num_closest, int):
            raise ValueError('num_closest must be an integer')
        if not isinstance(metric, str):
            raise ValueError('metric must be a string')
        if not isinstance(return_similarity, bool):
            raise ValueError('num_closest must be a boolean')
        if return_similarity:
            nn = NearestNeighbors(num_closest + 1, metric = metric).fit(
                self._embed_matrix
            )
            distances, idx = nn.kneighbors(
                self.get_vector_by_name(
                    word,
                    mode_bidirectional = mode_bidirectional,
                    mode_sequence = mode_sequence,
                ).reshape((1, -1))
            )
            word_list = []
            for i in range(1, idx.shape[1]):
                word_list.append(
                    [self._reverse_dictionary[idx[0, i]], 1 - distances[0, i]]
                )
            return word_list
        else:
            wv = self.get_vector_by_name(
                word,
                mode_bidirectional = mode_bidirectional,
                mode_sequence = mode_sequence,
            )
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
            cdist(self._embed_matrix, wv.reshape((1, -1)), metric = metric)
        )
        sorted_indices = np.argsort(dist_array)
        return sorted_indices[:num]

    def analogy(
        self,
        a,
        b,
        c,
        num = 1,
        metric = 'cosine',
        mode_bidirectional = 'mean',
        mode_sequence = 'mean',
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
        word_list: list of nearest words
        """
        if not isinstance(a, str):
            raise ValueError('a must be a string')
        if not isinstance(b, str):
            raise ValueError('b must be a string')
        if not isinstance(c, str):
            raise ValueError('c must be a string')
        va = self.get_vector_by_name(
            a,
            mode_bidirectional = mode_bidirectional,
            mode_sequence = mode_sequence,
        )
        vb = self.get_vector_by_name(
            b,
            mode_bidirectional = mode_bidirectional,
            mode_sequence = mode_sequence,
        )
        vc = self.get_vector_by_name(
            c,
            mode_bidirectional = mode_bidirectional,
            mode_sequence = mode_sequence,
        )
        vd = vb - va + vc
        closest_indices = self.closest_row_indices(vd, num, metric)
        d_word_list = []
        for i in closest_indices:
            d_word_list.append(self._reverse_dictionary[i])
        return d_word_list

    def project_2d(self, start, end):
        """
        project word2vec into 2d dimension.

        Parameters
        ----------
        start: int
        end: int

        Returns
        -------
        tsne decomposition: numpy
        """
        if not isinstance(start, int):
            raise ValueError('start must be an integer')
        if not isinstance(end, int):
            raise ValueError('end must be an integer')
        tsne = TSNE(n_components = 2)
        embed_2d = tsne.fit_transform(self._embed_matrix[start:end, :])
        word_list = []
        for i in range(start, end):
            word_list.append(self._reverse_dictionary[i])
        return embed_2d, word_list
