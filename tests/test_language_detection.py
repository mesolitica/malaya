import malaya

def test_lang_labels():
    assert malaya.get_language_labels()[0] == 'OTHER'

def test_multinomial_lang_sentence():
    multinomial = malaya.multinomial_detect_languages()
    malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'
    assert multinomial.predict(malay_text) == 'MALAY'

def test_multinomial_lang_sentence_proba():
    multinomial = malaya.multinomial_detect_languages()
    malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'
    assert multinomial.predict(malay_text,get_proba=True)['MALAY'] > 0

def test_multinomial_lang_sentences():
    multinomial = malaya.multinomial_detect_languages()
    malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'
    assert multinomial.predict_batch([malay_text,malay_text])[0] == 'MALAY'

def test_multinomial_lang_sentences_proba():
    multinomial = malaya.multinomial_detect_languages()
    malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'
    assert multinomial.predict_batch([malay_text,malay_text],get_proba=True)[0]['MALAY'] > 0

def test_xgb_lang_sentence():
    xgb = malaya.xgb_detect_languages()
    malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'
    assert xgb.predict(malay_text) == 'MALAY'

def test_xgb_lang_sentence_proba():
    xgb = malaya.xgb_detect_languages()
    malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'
    assert xgb.predict(malay_text,get_proba=True)['MALAY'] > 0

def test_xgb_lang_sentences():
    xgb = malaya.xgb_detect_languages()
    malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'
    assert xgb.predict_batch([malay_text,malay_text])[0] == 'MALAY'

def test_xgb_lang_sentences_proba():
    xgb = malaya.xgb_detect_languages()
    malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'
    assert xgb.predict_batch([malay_text,malay_text],get_proba=True)[0]['MALAY'] > 0
