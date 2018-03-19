from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import numpy as np
import re
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import itertools

class TOPIC:
    lsa = None
    nmf = None
    tfidf_features = None
    tf_features = None
    lda = None

def clearstring(string):
    string = unidecode(string)
    string = re.sub('[^A-Za-z ]+', '', string)
    string = word_tokenize(string)
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = ' '.join(string).lower()
    return ''.join(''.join(s)[:2] for _, s in itertools.groupby(string))

def train_lda(corpus,n_topics=10, max_df=0.95, min_df=2,cleaning=clearstring,stop_words='english'):
    if cleaning is not None:
        for i in range(len(corpus)): corpus[i] = clearstring(corpus[i])
    tf_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words)
    tf = tf_vectorizer.fit_transform(corpus)
    TOPIC.tf_features = tf_vectorizer.get_feature_names()
    TOPIC.lda = LatentDirichletAllocation(n_topics=n_topics, max_iter = 5, learning_method = 'online', learning_offset=50., random_state=0).fit(tf)
    
def topic_lda(len_topic):
    if TOPIC.lda is None or TOPIC.tf_features is None:
        raise Exception('you need to train LDA first, train_lda')
    results = []
    for no, topic in enumerate(TOPIC.lda.components_):
        results.append((no, " ".join([TOPIC.tf_features[i] for i in topic.argsort()[:-len_topic -1:-1]])))
    return results

def train_nmf(corpus,n_topics=10, max_df=0.95, min_df=2,cleaning=None,stop_words='english'):
    if cleaning is not None:
        for i in range(len(corpus)): corpus[i] = clearstring(corpus[i])
    tfidf_vectorizer = TfidfVectorizer(max_df = max_df, min_df = min_df, stop_words = stop_words)
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    TOPIC.tfidf_features = tfidf_vectorizer.get_feature_names()
    TOPIC.nmf = NMF(n_components=n_topics, random_state = 1, alpha =.1, l1_ratio=.5, init = 'nndsvd').fit(tfidf)
    
def topic_nmf(len_topic):
    if TOPIC.lda is None or TOPIC.tfidf_features is None:
        raise Exception('you need to train NMF first, train_nmf')
    results = []
    for no, topic in enumerate(TOPIC.nmf.components_):
        results.append((no, " ".join([TOPIC.tfidf_features[i] for i in topic.argsort()[:-len_topic -1:-1]])))
    return results

def train_lsa(corpus,n_topics, max_df=0.95, min_df=2,cleaning=None,stop_words='english'):
    tfidf_vectorizer = TfidfVectorizer(max_df = max_df, min_df = min_df, stop_words = stop_words)
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    TOPIC.tfidf_features = tfidf_vectorizer.get_feature_names()
    tfidf = Normalizer().fit_transform(tfidf)
    TOPIC.lsa = TruncatedSVD(n_topics).fit(tfidf)
    
def topic_lsa(len_topic):
    if TOPIC.lsa is None or TOPIC.tfidf_features is None:
        raise Exception('you need to train LSA first, train_lsa')
    results = []
    for no, topic in enumerate(TOPIC.lsa.components_):
        results.append((no, " ".join([TOPIC.tfidf_features[i] for i in topic.argsort()[:-len_topic -1:-1]])))
    return results