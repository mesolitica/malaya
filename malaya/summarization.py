import numpy as np
from scipy.linalg import svd, inv
import re, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from operator import itemgetter
from .text_functions import cluster_words, summary_textcleaning, deep_sentiment_textcleaning, STOPWORDS

def summarize_lsa(corpus, maintain_original = False, top_k = 3, important_words = 3,
    return_cluster = True, **kwargs):
    assert (isinstance(corpus, list) and isinstance(corpus[0], str)), "input must be list of strings"
    corpus = [summary_textcleaning(i) for i in corpus]
    corpus = ' '.join(corpus)
    splitted_fullstop = re.findall('(?=\S)[^.\n]+(?<=\S)', corpus)
    splitted_fullstop = [deep_sentiment_textcleaning(i) if not maintain_original else i for i in splitted_fullstop if len(i)]
    tfidf = TfidfVectorizer(ngram_range=(1, 3),min_df=2,stop_words=STOPWORDS,**kwargs).fit(splitted_fullstop)
    U, S, Vt = svd(tfidf.transform(splitted_fullstop).todense().T, full_matrices =False)
    summary =[(splitted_fullstop[i],
    np.linalg.norm(np.dot(np.diag(S),Vt[:,b]),2)) for i in range(len(splitted_fullstop)) for b in range(len(Vt))]
    summary = sorted(summary, key=itemgetter(1))
    summary = dict((v[0],v) for v in sorted(summary, key=lambda summary: summary[1])).values()
    summarized = '. '.join([a for a, b in summary][len(summary)-(top_k):])
    indices = np.argsort(tfidf.idf_)[::-1]
    features = tfidf.get_feature_names()
    top_words = [features[i] for i in indices[:important_words]]
    if return_cluster:
        return {'summary':summarized, 'top-words': top_words, 'cluster-top-words':cluster_words(top_words)}
    return {'summary':summarized, 'top-words': top_words}

def summarize_nmf(corpus, maintain_original = False, top_k = 3, important_words = 3,
    return_cluster = True, **kwargs):
    assert (isinstance(corpus, list) and isinstance(corpus[0], str)), "input must be list of strings"
    corpus = [summary_textcleaning(i) for i in corpus]
    corpus = ' '.join(corpus)
    splitted_fullstop = re.findall('(?=\S)[^.\n]+(?<=\S)', corpus)
    splitted_fullstop = [deep_sentiment_textcleaning(i) if not maintain_original else i for i in splitted_fullstop if len(i)]
    tfidf = TfidfVectorizer(ngram_range=(1, 3),min_df=2,stop_words=STOPWORDS,**kwargs).fit(splitted_fullstop)
    densed_tfidf = tfidf.transform(splitted_fullstop).todense()
    nmf = NMF(len(splitted_fullstop)).fit(densed_tfidf)
    vectors = nmf.transform(densed_tfidf)
    components = nmf.components_.mean(axis=1)
    summary =[(splitted_fullstop[i],
    np.linalg.norm(np.dot(np.diag(components),vectors[:,b]),2)) for i in range(len(splitted_fullstop)) for b in range(len(vectors))]
    summary = sorted(summary, key=itemgetter(1))
    summary = dict((v[0],v) for v in sorted(summary, key=lambda summary: summary[1])).values()
    summarized = '. '.join([a for a, b in summary][len(summary)-(top_k):])
    indices = np.argsort(tfidf.idf_)[::-1]
    features = tfidf.get_feature_names()
    top_words = [features[i] for i in indices[:important_words]]
    if return_cluster:
        return {'summary':summarized, 'top-words': top_words, 'cluster-top-words':cluster_words(top_words)}
    return {'summary':summarized, 'top-words': top_words}

def summarize_lda(corpus, maintain_original = False, top_k = 3, important_words = 3,
    return_cluster = True, **kwargs):
    assert (isinstance(corpus, list) and isinstance(corpus[0], str)), "input must be list of strings"
    corpus = [summary_textcleaning(i) for i in corpus]
    corpus = ' '.join(corpus)
    splitted_fullstop = re.findall('(?=\S)[^.\n]+(?<=\S)', corpus)
    splitted_fullstop = [deep_sentiment_textcleaning(i) if not maintain_original else i for i in splitted_fullstop if len(i)]
    tfidf = TfidfVectorizer(ngram_range=(1, 3),min_df=2,stop_words=STOPWORDS,**kwargs).fit(splitted_fullstop)
    densed_tfidf = tfidf.transform(splitted_fullstop).todense()
    lda = LatentDirichletAllocation(len(splitted_fullstop)).fit(densed_tfidf)
    vectors = lda.transform(densed_tfidf)
    components = lda.components_.mean(axis=1)
    summary =[(splitted_fullstop[i],
    np.linalg.norm(np.dot(np.diag(components),vectors[:,b]),2)) for i in range(len(splitted_fullstop)) for b in range(len(vectors))]
    summary = sorted(summary, key=itemgetter(1))
    summary = dict((v[0],v) for v in sorted(summary, key=lambda summary: summary[1])).values()
    summarized = '. '.join([a for a, b in summary][len(summary)-(top_k):])
    indices = np.argsort(tfidf.idf_)[::-1]
    features = tfidf.get_feature_names()
    top_words = [features[i] for i in indices[:important_words]]
    if return_cluster:
        return {'summary':summarized, 'top-words': top_words, 'cluster-top-words':cluster_words(top_words)}
    return {'summary':summarized, 'top-words': top_words}
