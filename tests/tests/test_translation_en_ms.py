import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

random_string1 = 'i am in medical school.'


def test_transformer():
    models = malaya.translation.en_ms.available_transformer()
    for m in models.index:
        print(m)
        model = malaya.translation.en_ms.transformer(model=m, gpu_limit=0.3)
        print(model.greedy_decoder([random_string1]))
        os.system('rm -f ~/.cache/huggingface/hub/*')
        del model
