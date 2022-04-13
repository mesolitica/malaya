import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

string1 = 'xjdi ke, y u xsuke makan HUSEIN kt situ tmpt, i hate it. pelikle, pada'
string2 = 'i mmg2 xske mknn HUSEIN kampng tmpat, i love them. pelikle saye'
string3 = 'perdana menteri ke11 sgt suka makn ayam, harganya cuma rm15.50'
string4 = 'pada 10/4, kementerian mengumumkan, 1/100'
string5 = 'Husein Zolkepli dapat tempat ke-12 lumba lari hari ni'
string6 = 'Husein Zolkepli (2011 - 2019) adalah ketua kampng di kedah sekolah King Edward ke-IV'
string7 = '2jam 30 minit aku tunggu kau, 60.1 kg kau ni, suhu harini 31.2c, aku dahaga minum 600ml'
string8 = 'online & desktop: regexr.com or download the desktop version for Mac'
string9 = 'belajaq unity di google.us.edi?34535/534534?dfg=g&fg unity'
string10 = 'Gambar ni membantu. Gambar tutorial >>. facebook. com/story. story_fbid=10206183032200965&id=1418962070'


def test_tokenizer():
    tokenizer = malaya.tokenizer.Tokenizer()
    tokenizer.tokenize(string1)
    tokenizer.tokenize(string2)
    tokenizer.tokenize(string3)
    tokenizer.tokenize(string4)
    tokenizer.tokenize(string5)


def test_sentence_tokenizer():
    s = """
    no.1 polis bertemu dengan suspek di ladang getah. polis tembak pui pui pui bertubi tubi
    """
    s_tokenizer = malaya.tokenizer.SentenceTokenizer()
    s_tokenizer.tokenize(s)
