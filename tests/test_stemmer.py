import malaya

def test_naive_stemmer():
    makanan = malaya.naive_stemmer('makanan')
    perjalanan = malaya.naive_stemmer('perjalanan')
    assert makanan == 'makan' and perjalanan == 'jalan'

def test_sastrawi_stemmer():
    assert malaya.sastrawi_stemmer('menarik') == 'tarik'

def test_deep_stemmer():
    stemmer = malaya.deep_stemmer()
    assert stemmer.stem('saya sangat sukakan awak') == 'saya sangat suka awak'

def test_deep_stemmer_unknown():
    stemmer = malaya.deep_stemmer()
    assert len(stemmer.stem('!!!!!()*&^%!^@%4'))
