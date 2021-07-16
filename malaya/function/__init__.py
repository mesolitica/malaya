# Malaya Natural Language Toolkit
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/Malaya/blob/master/LICENSE

from malaya_boilerplate.frozen_graph import (
    nodes_session,
    generate_session,
    get_device,
)
from malaya_boilerplate.utils import describe_availability
from malaya_boilerplate import backblaze
from malaya_boilerplate import frozen_graph
from malaya import package, url


def check_file(file, s3_file=None, **kwargs):
    return backblaze.check_file(file, package, url, s3_file=s3_file, **kwargs)


def load_graph(frozen_graph_filename, **kwargs):
    return frozen_graph.load_graph(package, frozen_graph_filename, **kwargs)
