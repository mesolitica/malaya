import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'
string1 = 'memperkenalkan Husein, dia sangat comel, berumur 25 tahun, bangsa melayu, agama islam, tinggal di cyberjaya malaysia, bercakap bahasa melayu, semua membaca buku undang-undang kewangan, dengar laju Siti Nurhaliza - Seluruh Cinta sambil makan ayam goreng KFC'


def test_transformer():
    models = malaya.pos.available_transformer()
    for m in models.index:
        print(m)
        model = malaya.pos.transformer(model=m)
        print(model.predict(string))
        print(model.predict(string1))
        print(model.analyze(string))
        os.system('rm -f ~/.cache/huggingface/hub/*')
        del model
