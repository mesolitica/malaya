from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import sys
import collections
import re
import numpy as np
import tensorflow as tf
from . import home
from .utils import download_file

def malaya_word2vec(size = 256):
    if size not in [32,64,128,256,512]:
        raise Exception('size word2vec not supported')
    if not os.path.isfile('%s/word2vec-%d.p'%(home,size)):
        print('downloading word2vec-%d embedded'%(size))
        download_file('http://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/word2vec-%d.p'%(size),'%s/word2vec-%d.p'%(home,size))
    with open('%s/word2vec-%d.p'%(home,size), 'rb') as fopen:
        return pickle.load(fopen)

class Word2Vec:
    def __init__(self,embed_matrix, dictionary):
        self._embed_matrix = embed_matrix
        self._dictionary = dictionary
        self._reverse_dictionary = {v: k for k, v in dictionary.items()}

    def get_vector_by_name(self, word):
        return np.ravel(self._embed_matrix[self._dictionary[word], :])

    def n_closest(self, word, num_closest=5, metric='cosine', return_similarity=True):
        if return_similarity:
            nn = NearestNeighbors(num_closest + 1,metric=metric).fit(self._embed_matrix)
            distances, idx = nn.kneighbors(self._embed_matrix[self._dictionary[word], :].reshape((1,-1)))
            word_list = []
            for i in range(1,idx.shape[1]):
                word_list.append([self._reverse_dictionary[idx[0,i]],1-distances[0,i]])
            return word_list
        else:
            wv = self.get_vector_by_name(word)
            closest_indices = self.closest_row_indices(wv, num_closest + 1, metric)
            word_list = []
            for i in closest_indices:
                word_list.append(self._reverse_dictionary[i])
            if word in word_list:
                word_list.remove(word)
            return word_list

    def closest_row_indices(self, wv, num, metric):
        dist_array = np.ravel(cdist(self._embed_matrix, wv.reshape((1, -1)),metric=metric))
        sorted_indices = np.argsort(dist_array)
        return sorted_indices[:num]

    def analogy(self, a, b, c, num=1, metric='cosine'):
        va = self.get_vector_by_name(a)
        vb = self.get_vector_by_name(b)
        vc = self.get_vector_by_name(c)
        vd = vb - va + vc
        closest_indices = self.closest_row_indices(vd, num, metric)
        d_word_list = []
        for i in closest_indices:
            d_word_list.append(self._reverse_dictionary[i])
        return d_word_list

    def project_2d(self, start, end):
        tsne = TSNE(n_components=2)
        embed_2d = tsne.fit_transform(self._embed_matrix[start:end, :])
        word_list = []
        for i in range(start, end):
            word_list.append(self._reverse_dictionary[i])
        return embed_2d, word_list
