import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

from pathlib import Path
import os

home = str(Path.home()) + '/Malaya'
version = '0.9'
bump_version = '0.9.0.0'
version_path = home + '/version'


def delete_folder(home):
    for root, dirs, files in os.walk(home):
        for file in files:
            os.remove(os.path.join(root, file))


try:
    if not os.path.exists(home):
        os.makedirs(home)
except:
    print('cannot make directory for caching, exiting.')
    sys.exit(1)

if not os.path.isfile(version_path):
    print('not found any version, deleting previous version models..')
    delete_folder(home)
    with open(version_path, 'w') as fopen:
        fopen.write(version)
else:
    with open(version_path, 'r') as fopen:
        cached_version = fopen.read()
    if version not in cached_version:
        print('deleting previous version models..')
        delete_folder(home)
        with open(version_path, 'w') as fopen:
            fopen.write(version)


def delete_cache():
    """
    Remove cached data, this is will delete entire cache folder. Selected items to delete will be implement soon.
    """
    try:
        print('deleting cached models..')
        delete_folder(home)
        with open(version_path, 'w') as fopen:
            fopen.write(version)
        print('Done.')
    except:
        print(
            'failed to clear cached models. Please make sure %s is able to overwrite from Malaya'
            % (home)
        )


from .entities import crf_entities, deep_entities, get_available_entities_models
from .language_detection import (
    multinomial_detect_languages,
    xgb_detect_languages,
    get_language_labels,
    sgd_detect_languages,
)
from .normalizer import (
    spell_normalizer,
    fuzzy_normalizer,
    load_malay_dictionary,
    basic_normalizer,
    deep_normalizer,
)
from .num2word import (
    to_cardinal,
    to_ordinal,
    to_ordinal_num,
    to_currency,
    to_year,
)
from .pos_entities import deep_pos_entities, get_available_pos_entities_models
from .pos import naive_pos, crf_pos, deep_pos, get_available_pos_models
from .sentiment import (
    deep_sentiment,
    bayes_sentiment,
    pretrained_bayes_sentiment,
    get_available_sentiment_models,
    pretrained_xgb_sentiment,
)
from .spell import naive_speller
from .stemmer import naive_stemmer, sastrawi_stemmer, deep_stemmer
from .summarization import (
    summarize_lsa,
    summarize_nmf,
    summarize_lda,
    summarize_deep_learning,
)
from .text_functions import voting_stack
from .topic_modelling import (
    lda_topic_modelling,
    nmf_topic_modelling,
    lsa_topic_modelling,
    lda2vec_topic_modelling,
)
from .topics_influencers import (
    load_internal_data,
    fuzzy_get_influencers,
    fuzzy_get_topics,
    fuzzy_get_location,
    fast_get_topics,
    fast_get_influencers,
    deep_get_topics,
    deep_get_influencers,
    deep_siamese_get_topics,
    deep_siamese_get_influencers,
)
from .toxic import multinomial_detect_toxic, logistics_detect_toxic, deep_toxic
from .word2vec import malaya_word2vec, Word2Vec


def describe_pos_malaya():
    """
    Describe Malaya Part-Of-Speech supported (deprecated, use describe_pos() instead).
    """
    print(
        'This classes are deprecated, we prefer to use `malaya.describe_pos()`'
    )
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


def describe_pos():
    """
    Describe Part-Of-Speech supported.
    """
    print('ADJ - Adjective, kata sifat')
    print('ADP - Adposition')
    print('ADV - Adverb, kata keterangan')
    print('ADX - Auxiliary verb, kata kerja tambahan')
    print('CCONJ - Coordinating conjuction, kata hubung')
    print('DET - Determiner, kata penentu')
    print('NOUN - Noun, kata nama')
    print('NUM - Number, nombor')
    print('PART - Particle')
    print('PRON - Pronoun, kata ganti')
    print('PROPN - Proper noun, kata ganti nama khas')
    print('SCONJ - Subordinating conjunction')
    print('SYM - Symbol')
    print('VERB - Verb, kata kerja')
    print('X - Other')


def describe_entities_malaya():
    """
    Describe Malaya Entities supported (deprecated, use describe_entities() instead).
    """
    print(
        'This classes are deprecated, we prefer to use `malaya.describe_entities()`'
    )
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
    """
    Describe Entities supported.
    """
    print('OTHER - Other')
    print('law - law, regulation, related law documents, documents, etc')
    print('location - location, place')
    print('organization - organization, company, government, facilities, etc')
    print('person - person, group of people, believes, etc')
    print('quantity - numbers, quantity')
    print('time - date, day, time, etc')
    print('event - unique event happened, etc')
