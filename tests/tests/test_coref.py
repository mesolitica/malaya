import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

string = 'Husein Zolkepli suka makan ayam. Dia pun suka makan daging.'
string1 = 'Kakak mempunyai kucing. Dia menyayanginya.'

# https://www.malaysiakini.com/news/580044
string2 = 'Pengerusi PKR Terengganu Azan Ismail menyelar pemimpin PAS yang disifatkannya sebagai membisu mengenai gesaan mengadakan sidang Dewan Undangan Negeri (DUN) di negeri yang dipimpin parti mereka.'

# https://www.sinarharian.com.my/article/146270/EDISI/Tiada-isu-penjualan-vaksin-Covid-19-di-Kelantan
string3 = 'Kota Bharu - Polis Kelantan mengesahkan masih belum menerima sebarang laporan berkaitan isu penjualan vaksin tidak sah berlaku di negeri ini. Timbalan Ketua Polis Kelantan, Senior Asisten Komisioner Abdullah Mohammad Piah berkata, bagaimanapun pihaknya sedia menjalankan siasatan lanjut jika menerima laporan berkaitan perkara itu.'


def test_coref():
    model = malaya.dependency.transformer(model='albert')
    string = 'Husein Zolkepli suka makan ayam. Dia pun suka makan daging.'
    malaya.coref.parse_from_dependency([model], string)
    malaya.coref.parse_from_dependency([model], string2)
    malaya.coref.parse_from_dependency([model], string3)

    os.system('rm -f ~/.cache/huggingface/hub/*')
    del model
