from pathlib import Path
import os
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

home = str(Path.home())+'/Malaya'

try:
    if not os.path.exists(home):
        os.makedirs(home)
except:
    print('cannot make directory for cache, exiting.')
    sys.exit(1)

from .entities import multinomial_entities
from .language_detection import get_language_labels, detect_language, detect_languages
from .normalizer import naive_normalizer, basic_normalizer
from .num2word import to_cardinal,to_ordinal,to_ordinal_num,to_currency,to_year
from .pos_entities import deep_pos_entities, get_available_pos_entities_models
from .pos import naive_pos
from .sentiment import deep_sentiment, bayes_sentiment, pretrained_bayes_sentiment, get_available_sentiment_models
from .stemmer import naive_stemmer
from .topic_modelling import lda_topic_modelling, nmf_topic_modelling, lsa_topic_modelling
from .word2vec import malaya_word2vec, Word2Vec

def describe_pos_malaya():
    print('KT - Kata Tanya')
    print('KJ - Kata Kerja')
    print('KP - Kata Perintah')
    print('KPA - Kata Pangkal')
    print('KB - Kata Bantu')
    print('KPENGUAT - Kata Penguat')
    print('KPENEGAS - Kata Penegas')
    print('NAFI - Kata Nafi')
    print('KPEMERI - Kata Pemeri')
    print('KS - Kata Sendi')
    print('KPEMBENAR - Kata Pembenar')
    print('NAFI - Kata Nafi')
    print('NO - Numbers')
    print('SUKU - Suku Bilangan')
    print('PISAHAN - Kata Pisahan')
    print('KETERANGAN - Kata Keterangan')
    print('ARAH - Kata Arah')
    print('KH - Kata Hubung')
    print('GN - Ganti Nama')
    print('KA - Kata Adjektif')
    print('O - not related, out scope')

def describe_entities_malaya():
    print('PRN - person, group of people, believes, etc')
    print('LOC - location')
    print('NORP - Military, police, government, Parties, etc')
    print('ORG - Organization, company')
    print('LAW - related law document, etc')
    print('ART - art of work, special names, etc')
    print('EVENT - event happen, etc')
    print('FAC - facility, hospitals, clinics, etc')
    print('TIME - date, day, time, etc')
    print('O - not related, out scope')

def describe_entities():
    print('OTHER - not related, out of scope')
    print('law - documents, law related')
    print('location - location, place')
    print('organization - Organization, company, government, parties')
    print('person - person, group of people, believes, special names')
    print('quantity - countable')
    print('time - date, day, time')
