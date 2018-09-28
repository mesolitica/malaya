import malaya

def test_pos_entities_char():
    available_models = malaya.get_available_pos_entities_models()
    string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan'
    assert len(malaya.deep_pos_entities('char').predict(string))

def test_pos_entities_concat():
    available_models = malaya.get_available_pos_entities_models()
    string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan'
    assert len(malaya.deep_pos_entities('concat').predict(string))

def test_pos_entities_attention():
    available_models = malaya.get_available_pos_entities_models()
    string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan'
    assert len(malaya.deep_pos_entities('attention').predict(string))

def test_pos_entities_attention_num():
    string = '34892347 23479 2312 35436 234'
    assert len(malaya.deep_pos_entities('attention').predict(string))
