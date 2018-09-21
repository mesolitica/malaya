import malaya
string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook'

malaya.get_available_pos_models()

def test_pos():
    assert len(malaya.naive_pos(string))

def test_multinomial_pos():
    assert len(malaya.multinomial_pos(string))

def test_xgb_pos():
    assert len(malaya.xgb_pos(string))

def test_deep_pos_char():
    model = malaya.deep_pos('char')
    assert len(model.predict(string))

def test_deep_pos_word():
    model = malaya.deep_pos('word')
    assert len(model.predict(string))

def test_deep_pos_concat():
    model = malaya.deep_pos('concat')
    assert len(model.predict(string))
