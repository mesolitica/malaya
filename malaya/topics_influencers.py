import numpy as np
import pandas as pd
import re
from fuzzywuzzy import fuzz
import zipfile
import os
from .text_functions import STOPWORDS
from . import home
from .utils import download_file

zip_location = home+'/rules-based.zip'

if not os.path.isfile(zip_location):
    print('downloading ZIP rules-based')
    download_file("rules-based.zip", zip_location)
    with zipfile.ZipFile(zip_location, 'r') as zip_ref:
        zip_ref.extractall(home)

def apply_stopwords_calon(string):
    string = re.sub('[^A-Za-z ]+', '', string)
    return ' '.join([i for i in string.split() if i not in STOPWORDS and len(i) > 1])

df = pd.read_csv(home+'/rules-based/calon.csv')
namacalon = df.NamaCalon.str.lower().unique().tolist()
for i in range(len(namacalon)):
    namacalon[i] = apply_stopwords_calon(namacalon[i])

df = pd.read_csv(home+'/rules-based/negeri.csv')
negeri = df.negeri.str.lower().unique().tolist()
parlimen = df.parlimen.str.lower().unique().tolist()
dun = df.dun.str.lower().unique().tolist()[:-1]

with open(home+'/rules-based/person-normalized','r') as fopen:
    person = list(filter(None, fopen.read().split('\n')))

person_dict = {}
for i in range(len(person)):
    splitted = person[i].split(':')
    uniques = list(filter(None,(set([k.strip().lower() for k in splitted[1].split(', ')] + [splitted[0].lower()]))))
    person_dict[splitted[0]] = uniques

with open(home+'/rules-based/topic-normalized','r') as fopen:
    topic = list(filter(None, fopen.read().split('\n')))

topic_dict = {}
for i in range(len(topic)):
    splitted = topic[i].split(':')
    uniques = list(filter(None,(set([k.strip().lower() for k in splitted[1].split(', ')] + [splitted[0].lower()]))))
    topic_dict[splitted[0]] = uniques

with open(home+'/rules-based/short-normalized','r') as fopen:
    short = list(filter(None, fopen.read().split('\n')))

short_dict = {}
for i in range(len(short)):
    splitted = short[i].split(':')
    uniques = list(filter(None,(set([k.strip().lower() for k in splitted[1].split(', ')] + [splitted[0].lower()]))))
    short_dict[splitted[0]] = uniques

def get_influencers(string):
    assert (isinstance(string, str)), "input must be a string"
    string = string.lower()
    influencers = []
    for key, vals in person_dict.items():
        for v in vals:
            if string.find(v) >= 0:
                influencers.append(key)
                break
    for key, vals in short_dict.items():
        for v in vals:
            if v in string.split():
                influencers.append(key)
                break

    for index in np.where(np.array([fuzz.token_set_ratio(i, string) for i in namacalon]) >= 80)[0]:
        influencers.append(namacalon[index])
    return list(set(influencers))

def get_topics(string):
    assert (isinstance(string, str)), "input must be a string"
    string = string.lower()
    topics = []
    for key, vals in topic_dict.items():
        for v in vals:
            if string.find(v) >= 0:
                topics.append(key)
                break
    for key, vals in person_dict.items():
        for v in vals:
            if string.find(v) >= 0:
                topics.append(key)
                break
    for key, vals in short_dict.items():
        for v in vals:
            if v in string.split():
                topics.append(key)
                break

    topics += [i for i in negeri if i in string.split()]
    topics += [i for i in parlimen if i in string.split()]
    topics += [i for i in dun if i in string.split()]
    return list(set(topics))
