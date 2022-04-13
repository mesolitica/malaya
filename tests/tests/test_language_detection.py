import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

chinese_text = '今天是６月１８号，也是Muiriel的生日！'
english_text = 'i totally love it man'
indon_text = 'menjabat saleh perombakan menjabat periode komisi energi fraksi partai pengurus partai periode periode partai terpilih periode menjabat komisi perdagangan investasi persatuan periode'
malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'
socialmedia_malay_text = 'nti aku tengok dulu tiket dari kl pukul berapa ada nahh'
socialmedia_indon_text = 'saking kangen papanya pas vc anakku nangis'
rojak_text = 'jadi aku tadi bikin ini gengs dan dijual haha salad only k dan haha drinks only k'
manglish_text = 'power lah even shopback come to edmw riao'


def test_fasttext():
    fast_text = malaya.language_detection.fasttext(quantized=False)
    fast_text.predict_proba(['suka makan ayam dan daging'])

    fast_text = malaya.language_detection.fasttext(quantized=True)
    fast_text.predict_proba(['suka makan ayam dan daging'])

    os.system('rm -f ~/.cache/huggingface/hub/*')


def test_deep():
    deep = malaya.language_detection.deep_model()
    quantized_deep = malaya.language_detection.deep_model(quantized=True)
    deep.predict_proba([indon_text])
    quantized_deep.predict_proba([indon_text])

    os.system('rm -f ~/.cache/huggingface/hub/*')
