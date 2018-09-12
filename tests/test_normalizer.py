import malaya

def test_basic_stemmer():
    assert malaya.basic_normalizer('x masing2') == 'tidak masing-masing'

def test_naive_stemmber():
    corpus_normalize = ['maka','serious','yeke','masing-masing']
    normalizer = malaya.naive_normalizer(corpus_normalize)
    assert normalizer.normalize('masing2') == 'masing-masing'
