import malaya

def test_entities():
    results = malaya.multinomial_entities('KUALA LUMPUR')
    assert results[0][0] == 'KUALA'
