import malaya
import pandas as pd

df = pd.read_csv('tests/02032018.csv',sep=';')
df = df.iloc[3:,1:]
df.columns = ['text','label']
corpus = df.text.tolist()

def test_lda():
    lda = malaya.lda_topic_modelling(corpus,10)
    assert len(lda.get_topics(10))

def test_lda_print():
    lda = malaya.lda_topic_modelling(corpus,10)
    lda.print_topics(5)

def test_lda_get_sentences():
    lda = malaya.lda_topic_modelling(corpus,10)
    assert len(lda.get_sentences(5))

def test_nmf():
    nmf = malaya.nmf_topic_modelling(corpus,10)
    assert len(nmf.get_topics(10))

def test_lsa():
    lsa = malaya.lsa_topic_modelling(corpus,10)
    assert len(lsa.get_topics(10))
