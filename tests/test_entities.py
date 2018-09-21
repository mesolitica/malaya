import malaya
string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan'
malaya.get_available_entities_models()

def test_multinomial_entities():
    results = malaya.multinomial_entities(string)
    assert len(results)

def test_xgb_entities():
    results = malaya.xgb_entities(string)
    assert len(results)

def test_deep_entities_char():
    model = malaya.deep_entities('char')
    assert len(model.predict(string))

def test_deep_entities_word():
    model = malaya.deep_entities('word')
    assert len(model.predict(string))

def test_deep_entities_concat():
    model = malaya.deep_entities('concat')
    assert len(model.predict(string))
