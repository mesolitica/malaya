import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

string = 'Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar sekiranya mengantuk ketika memandu.'


def test_transformer_v2():
    models = malaya.dependency.available_transformer()
    for m in models.index:
        print(m)
        model = malaya.dependency.transformer(model=m, gpu_limit=0.3)
        d_object, tagging, indexing = model.predict(string)
        os.system('rm -f ~/.cache/huggingface/hub/*')
        del model


def test_transformer_v1():
    models = malaya.dependency.available_transformer(version='v1')
    for m in models.index:
        print(m)
        model = malaya.dependency.transformer(version='v1', model=m, gpu_limit=0.3)
        d_object, tagging, indexing = model.predict(string)
        os.system('rm -f ~/.cache/huggingface/hub/*')
        del model
