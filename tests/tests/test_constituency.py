import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

string = 'Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar sekiranya mengantuk ketika memandu.'


def test_transformer():
    models = malaya.constituency.available_transformer()
    for m in models.index:
        print(m)
        model = malaya.constituency.transformer(model=m, gpu_limit=0.3)
        print(model.parse_tree(string))
        os.system('rm -f ~/.cache/huggingface/hub/*')
        del model
