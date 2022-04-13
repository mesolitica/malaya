import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)

string1 = 'krajaan patut bagi pencen awal skt kpd warga emas supaya emosi'


def test_prob():
    prob_corrector = malaya.spell.probability()
    prob_corrector.correct('sy')


def test_prob_sentencepiece():
    prob_corrector_sp = malaya.spell.probability(sentence_piece=True)
    prob_corrector_sp.edit_candidates('smbng')


def test_jamspell():
    model = malaya.spell.jamspell(model='wiki')
    model.correct('suke', 'saya suke makan iyom')
    model.edit_candidates('ayem', 'saya suke makan ayem')
    model.correct_text('saya suke makan ayom')


def test_spylls():
    model = malaya.spell.spylls()
    model.correct('sy')
    model.edit_candidates('mhthir')
    model.correct_text(string1)


def test_transformer_encoder():
    model = malaya.transformer.load(model='electra')
    transformer_corrector = malaya.spell.transformer_encoder(model, sentence_piece=True)
    transformer_corrector.correct_text(string1)
    malaya.utils.delete_cache('electra-model')
    os.system('rm -f ~/.cache/huggingface/hub/*')
    del model


def test_symspell():
    symspell_corrector = malaya.spell.symspell()
    symspell_corrector.edit_candidates('test')
    symspell_corrector.correct('bntng')
    symspell_corrector.correct_text(string1)


def test_transformer():
    models = malaya.spell.available_transformer()
    for m in models.index:
        print(m)
        model = malaya.spell.transformer(model=m)
        print(model.greedy_decoder([string1]))
        os.system('rm -f ~/.cache/huggingface/hub/*')
        del model
