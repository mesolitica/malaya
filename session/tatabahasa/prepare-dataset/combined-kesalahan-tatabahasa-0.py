import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from tqdm import tqdm
from collections import defaultdict
import random
import json
import copy
import re

import malaya
from malaya.text.regex import _expressions

model = malaya.dependency.transformer(model = 'xlnet', quantized = True)
pos = malaya.pos.transformer(model = 'xlnet', quantized = True)

tokenizer = malaya.preprocessing.TOKENIZER(date = False, time = False).tokenize
sastrawi = malaya.stem.sastrawi()

# !wget https://raw.githubusercontent.com/huseinzol05/Malay-Dataset/master/dictionary/synonym/synonym0.json
# !wget https://raw.githubusercontent.com/huseinzol05/Malay-Dataset/master/dictionary/synonym/synonym1.json
files = ['synonym0.json', 'synonym1.json']
synonyms = defaultdict(list)

for file in files:
    with open(file) as fopen:
        data = json.load(fopen)
    for i in data:
        if not len(i[1]):
            continue
        synonyms[i[0]].extend(i[1])
        for r in i[1]:
            synonyms[r].append(i[0])

for k, v in synonyms.items():
    synonyms[k] = list(set(v))


def reset_t(tokens):
    t = []
    for i in range(len(tokens)):
        t.append([tokens[i], 2])
    return t


def augment_3_0(t, row, selected = ['compound', 'flat']):
    text, tokens, tokens_lower, graph = row
    l = list(graph.nodes.items())
    for no, n in enumerate(l[1:]):
        n = n[1]
        if n['rel'] in selected and n['address'] - 1 == n['head']:
            if n['word'] == t[n['head'] - 1][0]:
                print('repeated word, continue')
                continue
            if n['word'][0].isupper() or t[n['head'] - 1][0][0].isupper():
                continue
            if (
                n['word'].lower() in set_combined_penjodoh_bilangan
                or t[n['head'] - 1][0].lower() in set_combined_penjodoh_bilangan
            ):
                continue

            c = t[n['head'] - 1].copy()
            c[1] = 3
            t[n['head'] - 1] = [t[n['address'] - 1][0], 3]
            t[n['address'] - 1] = c
            tokens[n['head'] - 1] = t[n['address'] - 1][0]
            tokens[n['address'] - 1] = c[0]
            tokens_lower[n['head'] - 1] = t[n['address'] - 1][0].lower()
            tokens_lower[n['address'] - 1] = c[0].lower()


# https://ms.wikipedia.org/wiki/Penjodoh_bilangan_bahasa_Melayu
penjodoh_bilangan = [
    'angkatan',
    'baris',
    'batang',
    'bentuk',
    'bidang',
    'biji',
    'bilah',
    'buah',
    'buku',
    'bungkus',
    'butir',
    'carik',
    'cebis',
    'cekak',
    'cubit',
    'cucuk',
    'das',
    'deret',
    'ekor',
    'gugus',
    'gelung',
    'gemal',
    'genggam',
    'gulung',
    'gumpal',
    'helai',
    'hidangan',
    'hiris',
    'ikat',
    'jambak',
    'jambangan',
    'jemput',
    'kaki',
    'kalung',
    'kandang',
    'kapur',
    'kawan',
    'kelompok',
    'kepal',
    'keping',
    'kepul',
    'kerat',
    'ketul',
    'kotak',
    'kuntum',
    'laras',
    'lembar',
    'lingkar',
    'longgok',
    'naskhah',
    'orang',
    'papan',
    'pasang',
    'pasukan',
    'patah',
    'pintu',
    'potong',
    'pucuk',
    'puntung',
    'rangkap',
    'rawan',
    'ruas',
    'rumpun',
    'sikat',
    'sisir',
    'suap',
    'tandan',
    'tangkai',
    'teguk',
    'timbun',
    'titik',
    'tongkol',
    'ulas',
    'untai',
    'urat',
    'utas',
]
hubung_list = [
    'agar',
    'apabila',
    'atau',
    'bahawa',
    'dan',
    'hingga',
    'jika',
    'jikalau',
    'kecuali',
    'kerana',
    'lalu',
    'manakala',
    'sambil',
    'serta',
    'semenjak',
    'sementara',
    'sungguhpun',
    'supaya',
    'walaupun',
    'tetapi',
    'berkenan',
    'berkenaan',
    'yang',
    'juga',
    'tersebut',
]
end_4 = ['nya']
reserved_4 = [
    'mereka',
    'pelajar',
    'rakyat',
    'penduduk',
    'umat',
    'kami',
    'semua',
    'kumpulan',
    'para',
]
start_4 = ['be', 'ber', 'ter', 'se']
sepenjodoh_bilangan = [f'se{w}' for w in penjodoh_bilangan]
set_sepenjodoh_bilangan = set(sepenjodoh_bilangan)
set_penjodoh_bilangan = set(penjodoh_bilangan)
set_reserved_4 = set(reserved_4)
set_combined_penjodoh_bilangan = set_sepenjodoh_bilangan | set_penjodoh_bilangan

# [penjodoh bilangan] [kata nama] -> [penjodoh bilangan] [kata nama - kata nama]
# dua buah kereta -> dua buah kereta-kereta
def augment_4_0(t, row):
    text, tokens, tokens_lower, penjodoh = row
    for word in penjodoh:
        try:
            i = tokens_lower.index(word) + 1
            if tokens_lower[i] in hubung_list:
                continue
            if tokens[i][0].isupper():
                continue
            if tokens[i].endswith('nya'):
                tokens[i] = tokens[i][:-3]
                ends = 'nya'
            else:
                ends = ''
            word = f'{tokens[i]}-{tokens[i]}{ends}'
            t[i][0] = word
            t[i][1] = 4
            tokens[i] = word
            tokens_lower[i] = word.lower()
        except Exception as e:
            print('augment_4_0', e)
            pass


# [kata nama - kata nama] -> [kata nama]
# ayam-ayam itu -> ayam itu
def augment_4_1(t, row):
    text, tokens, tokens_lower, penjodoh = row
    for no, word in enumerate(tokens):
        if re.findall(_expressions['hypen'], word.lower()):
            stemmed = sastrawi.stem(word)
            if stemmed != word.split('-')[0]:
                continue
            if word[0].isupper():
                continue
            word = word.split('-')[0]
            t[no][0] = word
            t[no][1] = 4
            tokens[no] = word
            tokens_lower[no] = word.lower()


def augment_4_2(t, row):
    text, tokens, tokens_lower, penjodoh = row
    for word in penjodoh:
        try:
            i = tokens_lower.index(word)
            if tokens[i].endswith('nya'):
                tokens[i] = tokens[i][:-3]
                ends = 'nya'
            else:
                ends = ''
            t[i][0] = f'{tokens[i]}-{tokens[i]}{ends}'
            t[i][1] = 4
            tokens[i] = word
            tokens_lower[i] = word.lower()
        except Exception as e:
            print('augment_4_2', e)
            pass


penguat_list = [
    'paling',
    'agak',
    'sungguh',
    'amat',
    'terlalu',
    'nian',
    'benar',
    'paling',
    'sangat',
]
end_penguat_list = ['sekali', 'sungguh', 'sangat']
set_penguat_list = set(penguat_list)


def augment_5_0(t, row):
    text, tokens, tokens_lower, penguat = row
    for word in penguat:
        try:
            i = tokens_lower.index(word) + 1
            if tokens[i][0].isupper():
                continue
            ends = random.choice(end_penguat_list)
            word = f'{tokens[i]} {ends}'
            t[i][0] = word
            t[i][1] = 5
            tokens[i] = word
            tokens_lower[i] = word.lower()
        except Exception as e:
            print('augmentation_5_0', e)
            pass


def check_start_ter(word):
    stemmed = sastrawi.stem(word)
    if (
        word.startswith('ter')
        and not stemmed.startswith('ter')
        and stemmed in word
    ):
        return True
    return False


def augment_6_0(t, row):
    text, tokens, tokens_lower, penguat = row
    for i in range(len(tokens)):
        if check_start_ter(tokens[i]):
            ends = random.choice(end_penguat_list)
            word = f'{tokens[i]} {ends}'
            t[i][0] = word
            t[i][1] = 6
            tokens[i] = word
            tokens_lower[i] = word.lower()


hubung_list = [
    'agar',
    'apabila',
    'atau',
    'bahawa',
    'dan',
    'hingga',
    'jika',
    'jikalau',
    'kecuali',
    'kerana',
    'lalu',
    'manakala',
    'sambil',
    'serta',
    'semenjak',
    'sementara',
    'sungguhpun',
    'supaya',
    'walaupun',
    'tetapi',
    'berkenan',
    'berkenaan',
]
set_hubung_list = set(hubung_list)


def augment_7_0(t, row):
    text, tokens, tokens_lower, hubung = row
    for word in hubung:
        i = tokens_lower.index(word)
        negate = list(set_hubung_list - {word})
        choice = random.choice(negate)
        t[i][0] = choice
        t[i][1] = 7
        tokens[i] = choice
        tokens_lower[i] = choice.lower()


start_8 = ['be', 'ber', 'ter', 'se']


def check_bilangan(word):
    if re.findall(_expressions['hypen'], word.lower()):
        stemmed = sastrawi.stem(word)
        splitted = word.split('-')
        for s in start_8:
            if (
                word.startswith(s)
                and f'{s}{stemmed}' == splitted[0]
                and stemmed == splitted[1]
            ):
                return True
    return False


def augment_8_0(t, row):
    text, tokens, tokens_lower = row
    for i in range(len(tokens)):
        if check_bilangan(tokens[i]):
            word = tokens[i].split('-')[0]
            t[i][0] = word
            t[i][1] = 8
            tokens[i] = word
            tokens_lower[i] = word.lower()


sendi_list = [
    'akan',
    'kepada',
    'terhadap',
    'bagi',
    'untuk',
    'dari',
    'daripada',
    'di',
    'dengan',
    'hingga',
    'sampai',
    'ke',
    'kepada',
    'oleh',
    'pada',
    'sejak',
    'seperti',
    'umpama',
    'bak',
    'tentang',
    'laksanabagai',
    'semenjak',
    'dalam',
    'antara',
]
set_sendi_list = set(sendi_list)


def augment_9_0(t, row):
    text, tokens, tokens_lower, sendi = row
    for word in sendi:
        i = tokens_lower.index(word)
        negate = list(set_sendi_list - {word})
        choice = random.choice(negate)
        t[i][0] = choice
        t[i][1] = 9
        tokens[i] = choice
        tokens_lower[i] = choice.lower()


def augment_10_0(t, row):
    text, tokens, tokens_lower, penjodoh = row
    for word in penjodoh:
        try:
            i = tokens_lower.index(word)
            negate = list(set_penjodoh_bilangan - {word})
            choice = random.choice(negate)
            t[i][0] = choice
            t[i][1] = 10
            tokens[i] = choice
            tokens_lower[i] = choice.lower()
        except Exception as e:
            print(e)
            pass


def augment_10_1(t, row):
    text, tokens, tokens_lower, penjodoh = row
    for word in penjodoh:
        try:
            i = tokens_lower.index(word)
            negate = list(set_sepenjodoh_bilangan - {word})
            choice = random.choice(negate)
            t[i][0] = choice
            t[i][1] = 10
            tokens[i] = choice
            tokens_lower[i] = choice.lower()
        except Exception as e:
            print(e)
            pass


gantinama_list = [
    'aku',
    'saya',
    'hamba',
    'patik',
    'beta',
    'kami',
    'kita',
    'anda',
    'awak',
    'engkau',
    'tuanku',
    'kalian',
    'kamu',
    'baginda',
    'beliau',
    'mereka',
    'ini',
    'itu',
    'sini',
    'situ',
    'sana',
    'kini',
    'dia',
    'kau',
]
set_gantinama_list = set(gantinama_list)


def augment_11_0(t, row):
    text, tokens, tokens_lower, nama = row
    for word in nama:
        i = tokens_lower.index(word)
        negate = list(set_gantinama_list - {word})
        choice = random.choice(negate)
        t[i][0] = choice
        t[i][1] = 11


def augment_12_0(t, row):
    text, tokens, tokens_lower, tagging = row
    for i in range(len(tokens) - 2):
        if (
            tagging[i] == 'ADV'
            and tagging[i + 1] in ['PRON', 'NOUN']
            and tagging[i + 2] in ['VERB', 'NOUN']
            and tokens_lower[i] in ['telah', 'mesti']
        ):
            v = f'di{tokens[i + 2]}'
            n = f'oleh {tokens[i + 1]}'
            t[i][1] = 12
            t[i + 1][0] = v
            t[i + 1][1] = 12
            t[i + 2][0] = n
            t[i + 2][1] = 12


def augment_12_1(t, row):
    text, tokens, tokens_lower, tagging = row
    for i in range(len(tokens) - 1):
        if (
            tagging[i] == 'PRON'
            and tagging[i + 1] == 'VERB'
            and sastrawi.stem(tokens[i + 1]) == tokens[i + 1]
        ):
            v = f'men{tokens[i + 1]}'
            if sastrawi.stem(v) == v:
                v = f'mem{tokens[i + 1]}'
            t[i][1] = 12
            t[i + 1][0] = v
            t[i + 1][1] = 12


def augment_12_2(t, row):
    text, tokens, tokens_lower, tagging = row
    for i in range(len(tokens) - 2):
        if (
            tagging[i] == 'VERB'
            and tagging[i + 1] in ['ADP']
            and tagging[i + 2] in ['PRON', 'NOUN']
            and tokens_lower[i + 1] in ['oleh']
        ):
            v = sastrawi.stem(tokens[i])
            t[i][0] = tokens[i + 2]
            t[i][1] = 12
            t[i + 1][0] = v
            t[i + 1][1] = 12
            t[i + 2][0] = ''
            t[i + 2][1] = 12


tanya_list = [
    'kenapa',
    'bila',
    'siapa',
    'mengapa',
    'apa',
    'bagaimana',
    'berapa',
    'mana',
]
kah_tanya_list = [f'{w}kah' for w in tanya_list]
combined = tanya_list + kah_tanya_list
set_combined = set(combined)


def augment_13_0(t, row):
    text, tokens, tokens_lower, tanya = row
    for word in tanya:
        i = tokens_lower.index(word)
        negate = list(set_combined - {word})
        choice = random.choice(negate)
        t[i][0] = choice
        t[i][1] = 13


punc = '.?!,;:'
set_punc = set(punc)


def augment_14_0(t, row):
    text, tokens, tokens_lower, p = row
    for word in p:
        i = tokens_lower.index(word)
        negate = list(set_punc - {word})
        choice = random.choice(negate)
        t[i][0] = choice
        t[i][1] = 14


start_15 = ['ber', 'ter', 'me', 'men']


def check_tak_transitif(word):
    stemmed = sastrawi.stem(word)
    for s in start_15:
        if word.startswith(s) and f'{s}{stemmed}' == word:
            return True
    return False


def augment_15_0(t, row):
    text, tokens, tokens_lower, tagging = row
    for i in range(len(tokens) - 1):
        if (
            tagging[i] == 'VERB'
            and tagging[i + 1] not in ['PRON', 'NOUN']
            and check_tak_transitif(tokens[i])
        ):
            t[i][0] = sastrawi.stem(tokens[i])
            t[i][1] = 15


start_end = {
    'me': 'kan',
    'mem': 'kan',
    'men': 'kan',
    'mem': '',
    'me': '',
    'men': '',
}


def check_transitif(word):
    stemmed = sastrawi.stem(word)
    for k, v in start_end.items():
        if (
            word.startswith(k)
            and word.endswith(v)
            and f'{k}{stemmed}{v}' == word
        ):
            return True
    return False


def augment_16_0(t, row):
    text, tokens, tokens_lower, tagging = row
    for i in range(len(tokens) - 2):
        if (
            tagging[i] in ['PRON', 'NOUN']
            and tagging[i + 1] == 'VERB'
            and tagging[i + 2] in ['PRON', 'NOUN']
            and check_transitif(tokens[i + 1])
        ):
            t[i][1] = 16
            t[i + 1][0] = sastrawi.stem(tokens[i + 1])
            t[i + 1][1] = 16
            t[i + 2][1] = 16


def augment_17_0(t, row):
    text, tokens, tokens_lower = row
    for i in range(len(tokens)):
        if tokens_lower[i] in synonyms and random.gauss(0.5, 0.14) > 0.8:
            w = random.choice(synonyms[tokens_lower[i]])
            t[i][0] = w
            t[i][1] = 17


with open('filtered-dumping-wiki.txt') as fopen:
    data = list(filter(None, fopen.read().split('\n')))

data = [i for i in data if len(i) >= 2]
data = data[0:100_000]

results = []
threshold = 0.6
for text in tqdm(data):
    try:
        tokens = tokenizer(text)
        t = reset_t(tokens)
        t_ = copy.deepcopy(t)
        tokens_lower = tokenizer(text.lower())
        tagging, indexing = malaya.stack.voting_stack(
            [model] * 3, ' '.join(tokens)
        )
        graph = malaya.dependency.dependency_graph(tagging, indexing)

        pos_tagging = malaya.stack.voting_stack([pos] * 3, ' '.join(tokens))
        pos_tagging = list(zip(*pos_tagging))[1]

        r = (t, tokens, tokens_lower, graph)
        if random.random() > threshold:
            augment_3_0(t_, r)

        set_tokens = set(tokens_lower)
        r_penjodoh_bilangan = set_tokens & set_penjodoh_bilangan
        r_sepenjodoh_bilangan = set_tokens & set_sepenjodoh_bilangan
        r_reserved = set_tokens & set_reserved_4
        r = (
            t,
            tokens,
            tokens_lower,
            r_penjodoh_bilangan | r_sepenjodoh_bilangan,
        )

        if random.gauss(0.5, 0.14) > threshold:
            augment_4_1(t_, r)
        if random.gauss(0.5, 0.14) > threshold:
            augment_4_0(t_, r)

        r = (t, tokens, tokens_lower, r_reserved)
        if random.gauss(0.5, 0.14) > threshold:
            augment_4_2(t_, r)

        set_tokens = set(tokens_lower)
        r_penguat_list = set_tokens & set_penguat_list
        r = (t, tokens, tokens_lower, r_penguat_list)
        if random.gauss(0.5, 0.14) > threshold:
            augment_5_0(t_, r)

        set_tokens = set(tokens_lower)
        r_penguat_list = set_tokens & set_penguat_list
        r = (t, tokens, tokens_lower, r_penguat_list)
        if random.gauss(0.5, 0.14) > threshold:
            augment_6_0(t_, r)

        set_tokens = set(tokens_lower)
        r_hubung_list = set_tokens & set_hubung_list
        r = (t, tokens, tokens_lower, r_hubung_list)
        if random.gauss(0.5, 0.14) > threshold:
            augment_7_0(t_, r)

        r = (t, tokens, tokens_lower)
        if random.gauss(0.5, 0.14) > threshold:
            augment_8_0(t_, r)

        set_tokens = set(tokens_lower)
        r_sendi_list = set_tokens & set_sendi_list
        r = (t, tokens, tokens_lower, r_sendi_list)
        if random.gauss(0.5, 0.14) > threshold:
            augment_9_0(t_, r)

        set_tokens = set(tokens_lower)
        r_penjodoh_bilangan = set_tokens & set_penjodoh_bilangan
        r_sepenjodoh_bilangan = set_tokens & set_sepenjodoh_bilangan
        r = (t, tokens, tokens_lower, r_penjodoh_bilangan)
        if random.gauss(0.5, 0.14) > threshold:
            augment_10_0(t_, r)
        r = (t, tokens, tokens_lower, r_sepenjodoh_bilangan)
        if random.gauss(0.5, 0.14) > threshold:
            augment_10_1(t_, r)

        set_tokens = set(tokens_lower)
        r_gantinama_list = set_tokens & set_gantinama_list
        r = (t, tokens, tokens_lower, r_gantinama_list)
        if random.gauss(0.5, 0.14) > threshold:
            augment_11_0(t_, r)

        set_tokens = set(tokens_lower)
        r = (t, tokens, tokens_lower, pos_tagging)
        if random.gauss(0.5, 0.14) > threshold:
            augment_12_0(t_, r)
        a = list(zip(*t_))[1]
        if 12 not in a:
            if random.gauss(0.5, 0.14) > threshold:
                augment_12_1(t_, r)
        a = list(zip(*t_))[1]
        if 12 not in a:
            if random.gauss(0.5, 0.14) > threshold:
                augment_12_2(t_, r)

        set_tokens = set(tokens_lower)
        r_set_combined = set_tokens & set_combined
        r = (t, tokens, tokens_lower, r_set_combined)
        if random.gauss(0.5, 0.14) > threshold:
            augment_13_0(t_, r)

        set_tokens = set(tokens_lower)
        r_set_punc = set_tokens & set_punc
        r = (t, tokens, tokens_lower, r_set_punc)
        if random.gauss(0.5, 0.14) > threshold:
            augment_14_0(t_, r)

        set_tokens = set(tokens_lower)
        r = (t, tokens, tokens_lower, pos_tagging)
        if random.gauss(0.5, 0.14) > threshold:
            augment_15_0(t_, r)

        r = (t, tokens, tokens_lower, pos_tagging)
        if random.gauss(0.5, 0.14) > threshold:
            augment_16_0(t_, r)

        r = (t, tokens, tokens_lower)
        augment_17_0(t_, r)

        results.append((t, t_))
    except:
        pass

import pickle

with open('dataset-tatabahasa-0.pkl', 'wb') as fopen:
    pickle.dump(results, fopen)
