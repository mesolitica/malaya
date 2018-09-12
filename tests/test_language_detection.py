import malaya
from malaya.language_detection import MULTINOMIAL, BOW

def test_lang_labels():
    assert malaya.get_language_labels()[0] == 'OTHER'

def test_lang_sentence():
    malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'
    assert malaya.detect_language(malay_text) == 'MALAY'

def test_lang_sentence_proba():
    malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'
    assert malaya.detect_language(malay_text,get_proba=True)['MALAY'] > 0

def test_lang_sentences():
    global MULTINOMIAL, BOW
    MULTINOMIAL, BOW = None, None
    malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'
    assert malaya.detect_languages([malay_text,malay_text])[0] == 'MALAY'

def test_lang_sentences_proba():
    malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'
    assert malaya.detect_languages([malay_text,malay_text],get_proba=True)[0]['MALAY'] > 0
