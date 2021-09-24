# Malaya Natural Language Toolkit
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/Malaya/blob/master/LICENSE

from malaya_boilerplate.utils import get_home

version = '4.7'
bump_version = '4.7'
__version__ = bump_version

package = 'malaya'
url = 'https://f000.backblazeb2.com/file/malaya-model/'
__home__, _ = get_home(package=package, package_version=version)


from . import augmentation
from . import cluster
from . import constituency
from . import coref
from . import dependency
from . import emotion
from . import entity
from . import generator
from . import keyword_extraction
from . import knowledge_graph
from . import language_detection
from . import lexicon
from . import normalize
from . import nsfw
from . import num2word
from . import paraphrase
from . import pos
from . import preprocessing
from . import qa
from . import relevancy
from . import segmentation
from . import sentiment
from . import similarity
from . import spell
from . import stack
from . import stem
from . import subjectivity
from . import tatabahasa
from . import summarization
from . import topic_model
from . import toxicity
from . import transformer
from . import true_case
from . import translation
from . import word2num
from . import wordvector
from . import zero_shot
from . import utils
