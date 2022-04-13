import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

string = "Beliau yang juga saksi pendakwaan kesembilan berkata, ia bagi mengelak daripada wujud isu digunakan terhadap Najib."


def test_transformer():
    models = malaya.paraphrase.available_transformer()
    for m in models.index:
        print(m)
        model = malaya.paraphrase.transformer(model=m)
        print(model.greedy_decoder([string]))
        os.system('rm -f ~/.cache/huggingface/hub/*')
        del model
