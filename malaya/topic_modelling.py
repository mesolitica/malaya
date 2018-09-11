from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from .text_functions import simple_textcleaning
from unidecode import unidecode
import itertools
import numpy as np
import re

class TOPIC:
    def __init__(self,features,comp):
        self.features = features
        self.comp = comp
    def get_topics(self, len_topic):
        results = []
        for no, topic in enumerate(self.comp.components_):
            results.append((no, " ".join([self.features[i] for i in topic.argsort()[:-len_topic -1:-1]])))
        return results

def lda_topic_modelling(corpus,n_topics=10, max_df=0.95, min_df=2,cleaning=simple_textcleaning,stop_words='english'):
    assert (isinstance(strings, list) and isinstance(strings[0], str)), "input must be list of strings"
    if cleaning is not None:
        for i in range(len(corpus)): corpus[i] = cleaning(corpus[i])
    tf_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words)
    tf = tf_vectorizer.fit_transform(corpus)
    tf_features = tf_vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter = 5, learning_method = 'online', learning_offset=50., random_state=0).fit(tf)
    return TOPIC(tf_features,lda)

def nmf_topic_modelling(corpus,n_topics=10, max_df=0.95, min_df=2,cleaning=simple_textcleaning,stop_words='english'):
    assert (isinstance(strings, list) and isinstance(strings[0], str)), "input must be list of strings"
    if cleaning is not None:
        for i in range(len(corpus)): corpus[i] = cleaning(corpus[i])
    tfidf_vectorizer = TfidfVectorizer(max_df = max_df, min_df = min_df, stop_words = stop_words)
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    tfidf_features = tfidf_vectorizer.get_feature_names()
    nmf = NMF(n_components=n_topics, random_state = 1, alpha =.1, l1_ratio=.5, init = 'nndsvd').fit(tfidf)
    return TOPIC(tfidf_features,nmf)

def lsa_topic_modelling(corpus,n_topics, max_df=0.95, min_df=2,cleaning=simple_textcleaning,stop_words='english'):
    assert (isinstance(strings, list) and isinstance(strings[0], str)), "input must be list of strings"
    if cleaning is not None:
        for i in range(len(corpus)): corpus[i] = cleaning(corpus[i])
    tfidf_vectorizer = TfidfVectorizer(max_df = max_df, min_df = min_df, stop_words = stop_words)
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    tfidf_features = tfidf_vectorizer.get_feature_names()
    tfidf = Normalizer().fit_transform(tfidf)
    lsa = TruncatedSVD(n_topics).fit(tfidf)
    return TOPIC(tfidf_features,lsa)
