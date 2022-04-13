import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

string = 'makan ayam'


def test_transformer():
    models = malaya.transformer.available_transformer()
    for m in models.index:
        model = malaya.transformer.load(model=m)
        model.vectorize([string])
        print(model.attention([string], method='last'))
        try:
            malaya.utils.delete_cache(f'{m}-model')
        except:
            pass
        os.system('rm -f ~/.cache/huggingface/hub/*')
        del model
