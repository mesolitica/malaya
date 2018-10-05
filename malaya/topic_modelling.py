from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from .text_functions import simple_textcleaning, STOPWORDS, classification_textcleaning, print_topics_modelling
from .stemmer import sastrawi_stemmer
from unidecode import unidecode
import itertools
import numpy as np
import re

class TOPIC:
    def __init__(self,features,comp,corpus,transformed):
        self.features = features
        self.comp = comp
        self.corpus = corpus
        self.transformed = transformed
    def print_topics(self, len_topic):
        print_topics_modelling(range(len_topic), feature_names = np.array(self.features),
        sorting = np.argsort(self.comp.components_)[:,::-1], topics_per_chunk=5, n_words=10)
    def get_topics(self, len_topic):
        results = []
        for no, topic in enumerate(self.comp.components_):
            results.append((no, " ".join([self.features[i] for i in topic.argsort()[:-len_topic -1:-1]])))
        return results
    def get_sentences(self, len_sentence, k=0):
        reverse_sorted = np.argsort(self.transformed[:,k])[::-1]
        return [self.corpus[i] for i in reverse_sorted[:len_sentence]]

def lda_topic_modelling(corpus,n_topics=10, max_df=0.95, min_df=2,
    stemming=True,cleaning=simple_textcleaning,stop_words=STOPWORDS):
    assert (isinstance(corpus, list) and isinstance(corpus[0], str)), "input must be list of strings"
    if cleaning is not None:
        for i in range(len(corpus)): corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)): corpus[i] = sastrawi_stemmer(corpus[i])
    tf_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words)
    tf = tf_vectorizer.fit_transform(corpus)
    tf_features = tf_vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter = 5, learning_method = 'online', learning_offset=50., random_state=0).fit(tf)
    return TOPIC(tf_features,lda,[classification_textcleaning(c) for c in corpus],lda.transform(tf))

def nmf_topic_modelling(corpus,n_topics=10, max_df=0.95, min_df=2,
    stemming=True,cleaning=simple_textcleaning,stop_words=STOPWORDS):
    assert (isinstance(corpus, list) and isinstance(corpus[0], str)), "input must be list of strings"
    if cleaning is not None:
        for i in range(len(corpus)): corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)): corpus[i] = sastrawi_stemmer(corpus[i])
    tfidf_vectorizer = TfidfVectorizer(max_df = max_df, min_df = min_df, stop_words = stop_words)
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    tfidf_features = tfidf_vectorizer.get_feature_names()
    nmf = NMF(n_components=n_topics, random_state = 1, alpha =.1, l1_ratio=.5, init = 'nndsvd').fit(tfidf)
    return TOPIC(tfidf_features,nmf,[classification_textcleaning(c) for c in corpus],nmf.transform(tfidf))

def lsa_topic_modelling(corpus,n_topics, max_df=0.95, min_df=2,
    stemming=True,cleaning=simple_textcleaning,stop_words=STOPWORDS):
    assert (isinstance(corpus, list) and isinstance(corpus[0], str)), "input must be list of strings"
    if cleaning is not None:
        for i in range(len(corpus)): corpus[i] = cleaning(corpus[i])
    if stemming:
        for i in range(len(corpus)): corpus[i] = sastrawi_stemmer(corpus[i])
    tfidf_vectorizer = TfidfVectorizer(max_df = max_df, min_df = min_df, stop_words = stop_words)
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    tfidf_features = tfidf_vectorizer.get_feature_names()
    tfidf = Normalizer().fit_transform(tfidf)
    lsa = TruncatedSVD(n_topics).fit(tfidf)
    return TOPIC(tfidf_features,lsa,[classification_textcleaning(c) for c in corpus],lsa.transform(tfidf))
