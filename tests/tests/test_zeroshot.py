import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

string = 'gov macam bengong, kami nk pilihan raya, gov backdoor, sakai'
texts = ['kerajaan sebenarnya sangat prihatin dengan rakyat, bagi duit bantuan',
         'gov macam bengong, kami nk pilihan raya, gov backdoor, sakai',
         'tolong order foodpanda jab, lapar',
         'Hapuskan vernacular school first, only then we can talk about UiTM']
labels = ['makan', 'makanan', 'novel', 'buku', 'kerajaan', 'food delivery',
          'kerajaan jahat', 'kerajaan prihatin', 'bantuan rakyat']


def test_transformer():
    models = malaya.zero_shot.classification.available_transformer()
    for m in models.index:
        print(m)
        model = malaya.zero_shot.classification.transformer(model=m, gpu_limit=0.3)
        model.predict_proba([string], labels=['najib razak', 'mahathir', 'kerajaan', 'PRU', 'anarki'])
        model.vectorize(texts, labels, method='first')
        os.system('rm -f ~/.cache/huggingface/hub/*')
        del model
