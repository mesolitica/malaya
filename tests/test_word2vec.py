import malaya

def test_word2vec_n_closest():
    embedded = malaya.malaya_word2vec(256)
    word_vector = malaya.Word2Vec(embedded['nce_weights'], embedded['dictionary'])
    word = 'anwar'
    assert len(word_vector.n_closest(word=word, num_closest=8, metric='cosine'))

def test_word2vec_n_closest_without_similarity():
    embedded = malaya.malaya_word2vec(256)
    word_vector = malaya.Word2Vec(embedded['nce_weights'], embedded['dictionary'])
    word = 'anwar'
    assert len(word_vector.n_closest(word=word, num_closest=8, metric='cosine', return_similarity=False))

def test_word2vec_calculator_n_closest():
    embedded = malaya.malaya_word2vec(256)
    word_vector = malaya.Word2Vec(embedded['nce_weights'], embedded['dictionary'])
    assert len(word_vector.calculator('anwar + mahathir', num_closest=8, metric='cosine'))

def test_word2vec_calculator_minus():
    embedded = malaya.malaya_word2vec(256)
    word_vector = malaya.Word2Vec(embedded['nce_weights'], embedded['dictionary'])
    assert len(word_vector.calculator('anwar - mahathir', num_closest=8, metric='cosine'))

def test_word2vec_calculator_multiply():
    embedded = malaya.malaya_word2vec(256)
    word_vector = malaya.Word2Vec(embedded['nce_weights'], embedded['dictionary'])
    assert len(word_vector.calculator('anwar * mahathir', num_closest=8, metric='cosine'))

def test_word2vec_calculator_divide():
    embedded = malaya.malaya_word2vec(256)
    word_vector = malaya.Word2Vec(embedded['nce_weights'], embedded['dictionary'])
    assert len(word_vector.calculator('anwar / mahathir', num_closest=8, metric='cosine'))

def test_word2vec_calculator_bracket():
    embedded = malaya.malaya_word2vec(256)
    word_vector = malaya.Word2Vec(embedded['nce_weights'], embedded['dictionary'])
    assert len(word_vector.calculator('(anwar+hadi) * mahathir', num_closest=8, metric='cosine'))

def test_word2vec_calculator_n_closest_without_similarity():
    embedded = malaya.malaya_word2vec(256)
    word_vector = malaya.Word2Vec(embedded['nce_weights'], embedded['dictionary'])
    assert len(word_vector.calculator('anwar + mahathir', num_closest=8, metric='cosine', return_similarity=False))

def test_word2vec_analogy():
    embedded = malaya.malaya_word2vec(256)
    word_vector = malaya.Word2Vec(embedded['nce_weights'], embedded['dictionary'])
    assert len(word_vector.analogy('anwar', 'penjara', 'kerajaan', 5)) == 5

def test_word2vec_tsne():
    embedded = malaya.malaya_word2vec(32)
    word_vector = malaya.Word2Vec(embedded['nce_weights'], embedded['dictionary'])
    embed_2d, word_list = word_vector.project_2d(0, 100)
    assert embed_2d.shape[1] == 2
