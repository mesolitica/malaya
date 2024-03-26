# Malaya Natural Language Toolkit
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/Malaya/blob/master/LICENSE

from malaya_boilerplate import huggingface
from malaya import package, url
import os


def check_file(file, s3_file=None, **kwargs):
    return huggingface.check_file(file, package, url, s3_file=s3_file, **kwargs)
