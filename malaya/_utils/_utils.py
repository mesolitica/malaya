import tensorflow as tf
import inspect
import sentencepiece as spm
import numpy as np
import requests
import os
from tqdm import tqdm
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from functools import wraps
from typing import Dict, List, Tuple
from pathlib import Path
from .. import _delete_folder


def recursive_check(v, t):
    if '__module__' in t.__dict__.keys():
        if t.__module__ != 'typing':
            return isinstance(v, t)
    else:
        return isinstance(v, t)

    args = t.__args__
    if args:
        origin = isinstance(v, t.__origin__)
        if 'typing.' in str(args[0]):
            return origin and recursive_check(v[0], args[0])
        else:
            if t.__origin__ == Dict and origin:
                key_type = args[0]
                value_type = args[1]
                return all([isinstance(k, key_type) for k in v.keys()]) and all(
                    [isinstance(k, value_type) for k in v.values()]
                )
            else:
                if not isinstance(v, (tuple, list, dict, set)):
                    return False
                if len(v) != len(args):
                    return False
                if len(args) == 1:
                    return origin and all([isinstance(p, args[0]) for p in v])
    else:
        return isinstance(v, t)


def check_type(func):
    fullspec = inspect.getfullargspec(func)
    parameters = fullspec.args
    annotations = fullspec.annotations

    @wraps(func)
    def check(*args, **kwargs):
        for v, p in zip(args, parameters):
            t = annotations.get(p)
            if t:
                if not recursive_check(v, t):
                    raise Exception(f'{v} must be a {t}')

        return func(*args)

    return check


def sentencepiece_tokenizer_xlnet(path_tokenizer):
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(path_tokenizer)
    return sp_model


def sentencepiece_tokenizer_bert(path_tokenizer, path_vocab):
    from ..texts._text_functions import SentencePieceTokenizer

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(path_tokenizer)

    with open(path_vocab) as fopen:
        v = fopen.read().split('\n')[:-1]
    v = [i.split('\t') for i in v]
    v = {i[0]: i[1] for i in v}
    tokenizer = SentencePieceTokenizer(v, sp_model)
    cls = '<cls>'
    sep = '<sep>'
    return tokenizer, cls, sep


def add_neutral(x, alpha = 1e-2):
    x = x.copy()
    divide = 1 / x.shape[1]
    x_minus = np.maximum(x - divide, alpha * x)
    x_divide = x_minus / divide
    sum_axis = x_divide.sum(axis = 1, keepdims = True)
    return np.concatenate([x_divide, 1 - sum_axis], axis = 1)


def download_file(url, filename):
    if 'http' in url:
        r = requests.get(url, stream = True)
    else:
        r = requests.get(
            'http://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/' + url,
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


def generate_session(graph):
    return tf.InteractiveSession(graph = graph)


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph


def check_available(file):
    for key, item in file.items():
        if 'version' in key:
            continue
        if not os.path.isfile(item):
            return False
    return True


def check_file(file, s3_file):
    base_location = os.path.dirname(file['model'])
    version = base_location + '/version'
    download = False
    if os.path.isfile(version):
        with open(version) as fopen:
            if not file['version'] in fopen.read():
                print('Found old version of %s, deleting..' % (base_location))
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
                print('downloading frozen %s %s' % (base_location, key))
                download_file(s3_file[key], item)
        with open(version, 'w') as fopen:
            fopen.write(file['version'])


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


class _Calculator:
    def __init__(self, tokens):
        self._tokens = tokens
        self._current = tokens[0]

    def exp(self):
        result = self.term()
        while self._current in ('+', '-'):
            if self._current == '+':
                self.next()
                result += self.term()
            if self._current == '-':
                self.next()
                result -= self.term()
        return result

    def factor(self):
        result = None
        if self._current[0].isdigit() or self._current[-1].isdigit():
            result = np.array([float(i) for i in self._current.split(',')])
            self.next()
        elif self._current is '(':
            self.next()
            result = self.exp()
            self.next()
        return result

    def next(self):
        self._tokens = self._tokens[1:]
        self._current = self._tokens[0] if len(self._tokens) > 0 else None

    def term(self):
        result = self.factor()
        while self._current in ('*', '/'):
            if self._current == '*':
                self.next()
                result *= self.term()
            if self._current == '/':
                self.next()
                result /= self.term()
        return result
