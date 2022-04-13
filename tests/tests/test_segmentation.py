import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

string1 = 'jom makan di us makanan di sana sedap'


def test_transformer():
    models = malaya.segmentation.available_transformer()
    for m in models.index:
        print(m)
        model = malaya.segmentation.transformer(model=m)
        print(model.greedy_decoder([string1]))
        os.system('rm -f ~/.cache/huggingface/hub/*')
        del model
