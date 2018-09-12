import malaya

def test_word2vec_n_closest():
    embedded = malaya.malaya_word2vec(256)
    word_vector = malaya.Word2Vec(embedded['nce_weights'], embedded['dictionary'])
    word = 'anwar'
    assert len(word_vector.n_closest(word=word, num_closest=8, metric='cosine')) > 0

def test_word2vec_analogy():
    embedded = malaya.malaya_word2vec(256)
    word_vector = malaya.Word2Vec(embedded['nce_weights'], embedded['dictionary'])
    assert len(word_vector.analogy('anwar', 'penjara', 'kerajaan', 5)) == 5
