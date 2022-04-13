import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

string1 = "Yang Berhormat Dato Sri Haji Mohammad Najib bin Tun Haji Abdul Razak ialah ahli politik Malaysia dan merupakan bekas Perdana Menteri Malaysia ke-6 yang mana beliau menjawat jawatan dari 3 April 2009 hingga 9 Mei 2018. Beliau juga pernah berkhidmat sebagai bekas Menteri Kewangan dan merupakan Ahli Parlimen Pekan Pahang"
string2 = "Pahang ialah negeri yang ketiga terbesar di Malaysia Terletak di lembangan Sungai Pahang yang amat luas negeri Pahang bersempadan dengan Kelantan di utara Perak Selangor serta Negeri Sembilan di barat Johor di selatan dan Terengganu dan Laut China Selatan di timur."


def test_transformer():
    models = malaya.knowledge_graph.available_transformer()
    for m in models.index:
        print(m)
        model = malaya.knowledge_graph.transformer(model=m, gpu_limit=0.3)
        model.greedy_decoder([string1, string2])
        os.system('rm -f ~/.cache/huggingface/hub/*')
        del model


def test_dependency():
    alxlnet = malaya.dependency.transformer(version='v1', model='alxlnet')
    s = 'Najib yang juga Ahli Parlimen Pekan memuji sikap Ahli Parlimen Langkawi itu yang mengaku bersalah selepas melanggar SOP kerana tidak mengambil suhu badan ketika masuk ke sebuah surau di Langkawi pada Sabtu lalu'
    tagging, indexing = malaya.stack.voting_stack([alxlnet, alxlnet, alxlnet], s)
    malaya.knowledge_graph.parse_from_dependency(tagging, indexing)

    os.system('rm -f ~/.cache/huggingface/hub/*')
    del alxlnet
