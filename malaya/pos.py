import re
from malaya.text.tatabahasa import tatabahasa_dict, hujung, permulaan


label = [
    'ADJ',
    'ADP',
    'ADV',
    'ADX',
    'CCONJ',
    'DET',
    'NOUN',
    'NUM',
    'PART',
    'PRON',
    'PROPN',
    'SCONJ',
    'SYM',
    'VERB',
    'X',
]


describe = [
    {'Tag': 'ADJ', 'Description': 'Adjective, kata sifat'},
    {'Tag': 'ADP', 'Description': 'Adposition'},
    {'Tag': 'ADV', 'Description': 'Adverb, kata keterangan'},
    {'Tag': 'ADX', 'Description': 'Auxiliary verb, kata kerja tambahan'},
    {'Tag': 'CCONJ', 'Description': 'Coordinating conjuction, kata hubung'},
    {'Tag': 'DET', 'Description': 'Determiner, kata penentu'},
    {'Tag': 'NOUN', 'Description': ' Noun, kata nama'},
    {'Tag': 'NUM', 'Description': 'Number, nombor'},
    {'Tag': 'PART', 'Description': 'Particle'},
    {'Tag': 'PRON', 'Description': 'Pronoun, kata ganti'},
    {'Tag': 'PROPN', 'Description': 'Proper noun, kata ganti nama khas'},
    {'Tag': 'SCONJ', 'Description': 'Subordinating conjunction'},
    {'Tag': 'SYM', 'Description': 'Symbol'},
    {'Tag': 'VERB', 'Description': 'Verb, kata kerja'},
    {'Tag': 'X', 'Description': 'Other'},
]


def _naive_POS_word(word):
    for key, vals in tatabahasa_dict.items():
        if word in vals:
            return (key, word)
    try:
        if len(re.findall(r'^(.*?)(%s)$' % ('|'.join(hujung[:1])), i)[0]) > 1:
            return ('KJ', word)
    except BaseException:
        pass
    try:
        if (
            len(re.findall(r'^(.*?)(%s)' % ('|'.join(permulaan[:-4])), word)[0])
            > 1
        ):
            return ('KJ', word)
    except Exception as e:
        pass
    if len(word) > 2:
        return ('KN', word)
    else:
        return ('', word)
