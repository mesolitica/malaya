import malaya

def test_naive_stemmer():
    makanan = malaya.naive_stemmer('makanan')
    perjalanan = malaya.naive_stemmer('perjalanan')
    assert makanan == 'makan' and perjalanan == 'jalan'
