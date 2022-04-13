import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

p_wikipedia = """
Najib razak telah dipilih untuk Parlimen Malaysia pada tahun 1976,
pada usia 23 tahun, menggantikan bapanya duduk di kerusi Pekan yang berpangkalan di Pahang.
Dari tahun 1982 hingga 1986 beliau menjadi Menteri Besar (Ketua Menteri) Pahang,
sebelum memasuki persekutuan Kabinet Tun Dr Mahathir Mohamad pada tahun 1986 sebagai Menteri Kebudayaan, Belia dan Sukan.
Beliau telah berkhidmat dalam pelbagai jawatan Kabinet sepanjang baki tahun 1980-an dan 1990-an, termasuk sebagai Menteri Pertahanan dan Menteri Pelajaran.
Beliau menjadi Timbalan Perdana Menteri pada 7 Januari 2004, berkhidmat di bawah Perdana Menteri Tun Dato' Seri Abdullah Ahmad Badawi,
sebelum menggantikan Badawi setahun selepas Barisan Nasional mengalami kerugian besar dalam pilihan raya 2008.
Di bawah kepimpinan beliau, Barisan Nasional memenangi pilihan raya 2013,
walaupun buat kali pertama dalam sejarah Malaysia pembangkang memenangi majoriti undi popular.
"""
q_wikipedia = ['Siapakah Menteri Besar Pahang', 'Apakah jawatan yang pernah dipegang oleh Najib Razak']


def test_squad():
    models = malaya.qa.available_transformer_squad()
    for m in models.index:
        print(m)
        model = malaya.qa.transformer_squad(model=m, gpu_limit=0.3)
        print(model.predict(p_wikipedia, q_wikipedia))
        os.system('rm -f ~/.cache/huggingface/hub/*')
        del model
