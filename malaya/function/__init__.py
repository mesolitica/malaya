# Malaya Natural Language Toolkit
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/Malaya/blob/master/LICENSE

import tensorflow as tf
import inspect
import numpy as np
import requests
import os
from tqdm import tqdm
from pathlib import Path
from malaya import _delete_folder, gpu_available, __gpu__

try:
    from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
except:
    import warnings

    warnings.warn(
        'Cannot import beam_search_ops from tensorflow, some deep learning models may not able to load, make sure Tensorflow version is, 1.10 < version < 2.0'
    )


def download_file(url, filename):
    if 'http' in url:
        r = requests.get(url, stream = True)
    else:
        r = requests.get(
            'https://f000.backblazeb2.com/file/malaya-model/' + url,
            stream = True,
        )
    total_size = int(r.headers['content-length'])
    os.makedirs(os.path.dirname(filename), exist_ok = True)
    with open(filename, 'wb') as f:
        for data in tqdm(
            iterable = r.iter_content(chunk_size = 1_048_576),
            total = total_size / 1_048_576,
            unit = 'MB',
            unit_scale = True,
        ):
            f.write(data)


def generate_session(graph, **kwargs):
    if gpu_available():
        config = tf.ConfigProto()
        if 'gpu' in kwargs:
            config.allow_soft_placement = True

        if 'gpu_limit' in kwargs:
            try:
                gpu_limit = float(kwargs.get('gpu_limit', 0.999))
            except:
                raise ValueError('gpu_limit must be a float')
            if not 0 < gpu_limit < 1:
                raise ValueError('gpu_limit must 0 < gpu_limit < 1')

            config.gpu_options.per_process_gpu_memory_fraction = gpu_limit

        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config = config, graph = graph)

    else:
        sess = tf.InteractiveSession(graph = graph)
    return sess


def load_graph(frozen_graph_filename, **kwargs):
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        try:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        except Exception as e:
            path = frozen_graph_filename.split('Malaya/')[1]
            path = '/'.join(path.split('/')[:-1])
            raise Exception(
                f"{e}, file corrupted due to some reasons, please run malaya.clear_cache('{path}') and try again"
            )

    # https://github.com/onnx/tensorflow-onnx/issues/77#issuecomment-445066091
    # to fix import T5
    for node in graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
        elif node.op == 'Assign':
            node.op = 'Identity'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
            if 'validate_shape' in node.attr:
                del node.attr['validate_shape']
            if len(node.input) == 2:
                node.input[0] = node.input[1]
                del node.input[1]

    with tf.Graph().as_default() as graph:
        if gpu_available():
            if 'gpu' in kwargs:
                gpu = kwargs.get('gpu', 0)
                if not isinstance(gpu, int):
                    raise ValueError('gpu must an int')
                if not 0 <= gpu < len(__gpu__):
                    raise ValueError(f'gpu must 0 <= gpu < {len(__gpu__)}')
                gpu = str(gpu)
                with tf.device(f'/device:GPU:{gpu}'):
                    tf.import_graph_def(graph_def)
            else:
                tf.import_graph_def(graph_def)
        else:
            tf.import_graph_def(graph_def)
    return graph


def check_available(file):
    for key, item in file.items():
        if 'version' in key:
            continue
        if not os.path.isfile(item):
            return False
    return True


def check_file(file, s3_file, validate = True, **kwargs):
    if validate:
        base_location = os.path.dirname(file['model'])
        version = base_location + '/version'
        download = False
        if os.path.isfile(version):
            with open(version) as fopen:
                if not file['version'] in fopen.read():
                    print(f'Found old version of {base_location}, deleting..')
                    _delete_folder(base_location)
                    print('Done.')
                    download = True
                else:
                    for key, item in file.items():
                        if not os.path.exists(item):
                            download = True
                            break
        else:
            download = True

        if download:
            for key, item in file.items():
                if 'version' in key:
                    continue
                if not os.path.isfile(item):
                    print(f'downloading frozen {base_location} {key}')
                    download_file(s3_file[key], item)
            with open(version, 'w') as fopen:
                fopen.write(file['version'])
    else:
        if not check_available(file):
            path = '/'.join(file['model'].split('/')[:-1])
            raise Exception(
                f'{path} is not available, please `validate = True`'
            )


class DisplayablePath(object):
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent = None, is_last = False, criteria = None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria
        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(
            list(path for path in root.iterdir() if criteria(path)),
            key = lambda s: str(s).lower(),
        )
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(
                    path,
                    parent = displayable_root,
                    is_last = is_last,
                    criteria = criteria,
                )
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (
            self.display_filename_prefix_last
            if self.is_last
            else self.display_filename_prefix_middle
        )

        parts = ['{!s} {!s}'.format(_filename_prefix, self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(
                self.display_parent_prefix_middle
                if parent.is_last
                else self.display_parent_prefix_last
            )
            parent = parent.parent

        return ''.join(reversed(parts))


def add_neutral(x, alpha = 1e-2):
    x = x.copy()
    divide = 1 / x.shape[1]
    x_minus = np.maximum(x - divide, alpha * x)
    x_divide = x_minus / divide
    sum_axis = x_divide.sum(axis = 1, keepdims = True)
    return np.concatenate([x_divide, 1 - sum_axis], axis = 1)


def describe_availability(dict, transpose = True, text = ''):
    if len(text):
        import logging

        logging.basicConfig(level = logging.INFO)

        logging.info(text)
    try:
        import pandas as pd

        df = pd.DataFrame(dict)

        if transpose:
            return df.T
        else:
            return df
    except:
        return dict
