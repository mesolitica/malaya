# Malaya Natural Language Toolkit
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/Malaya/blob/master/LICENSE

from malaya_boilerplate.utils import get_home

version = '5.0'
bump_version = '5.1.2'
__version__ = bump_version

package = 'malaya'
url = 'https://f000.backblazeb2.com/file/malaya-model/'
__home__, _ = get_home(package=package, package_version=version)

from . import augmentation
from . import dictionary
from . import generator
from . import keyword
from . import normalizer
from . import qa
from . import similarity
from . import spelling_correction
from . import summarization
from . import topic_model
from . import translation
from . import utils
from . import zero_shot

from . import constituency
from . import dependency
from . import embedding
from . import emotion
from . import entity
from . import jawi
from . import knowledge_graph
from . import language_detection
from . import language_model
from . import normalize
from . import nsfw
from . import num2word
from . import paraphrase
from . import pos
from . import preprocessing
from . import reranker
from . import segmentation
from . import sentiment
from . import stack
from . import stem
from . import syllable
from . import tatabahasa
from . import tokenizer
from . import topic_model
from . import transformer
from . import true_case
from . import word2num
from . import wordvector
