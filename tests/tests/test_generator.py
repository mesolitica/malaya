import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)


def test_isi_penting():
    isi_penting = ['Dr M perlu dikekalkan sebagai perdana menteri',
                   'Muhyiddin perlulah menolong Dr M',
                   'rakyat perlu menolong Muhyiddin']
    models = malaya.generator.available_isi_penting()
    for m in models.index:
        model = malaya.generator.isi_penting(model=m, quantized=True)
        model.greedy_decoder(isi_penting)
        os.system('rm -f ~/.cache/huggingface/hub/*')
        del model


def test_gpt():
    string = 'ceritanya sebegini, aku bangun pagi'
    models = malaya.generator.available_gpt2()
    for m in models.index:
        if '345M' in m:
            continue
        model = malaya.generator.gpt2(model=m)
        model.generate(string)
        os.system('rm -f ~/.cache/huggingface/hub/*')
        del model


def test_babble():
    string = 'ceritanya sebegini, aku bangun pagi'
    electra = malaya.transformer.load(model='electra')
    malaya.generator.babble(string, electra)
    malaya.utils.delete_cache('electra-model')
    os.system('rm -f ~/.cache/huggingface/hub/*')
    del electra
