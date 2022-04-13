import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

string1 = 'Pemuda mogok lapar desak kerajaan prihatin isu iklim'
string2 = 'Perbincangan isu pembalakan perlu babit kerajaan negeri'
string3 = 'kerajaan perlu kisah isu iklim, pemuda mogok lapar'
string4 = 'Kerajaan dicadang tubuh jawatankuasa khas tangani isu alam sekitar'

news1 = 'Tun Dr Mahathir Mohamad mengakui pembubaran Parlimen bagi membolehkan pilihan raya diadakan tidak sesuai dilaksanakan pada masa ini berikutan isu COVID-19'
tweet1 = 'DrM sembang pilihan raya tak boleh buat sebab COVID 19'


def test_transformer():
    models = malaya.similarity.available_transformer()
    for m in models.index:
        print(m)
        model = malaya.similarity.transformer(model=m, gpu_limit=0.3)
        print(model.predict_proba([string1, string2, news1, news1], [string3, string4, tweet1, string1]))
        os.system('rm -f ~/.cache/huggingface/hub/*')
        del model
