from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
import numpy as np
import requests
import os
from pathlib import Path
from .. import _delete_folder


def add_neutral(x, alpha = 1e-2):
    x = x.copy()
    divide = 1 / x.shape[1]
    x_minus = np.maximum(x - divide, alpha * x)
    x_divide = x_minus / divide
    sum_axis = x_divide.sum(axis = 1, keepdims = True)
    return np.concatenate([x_divide, 1 - sum_axis], axis = 1)


def download_file(url, filename):
    r = requests.get(
        'http://s3-ap-southeast-1.amazonaws.com/huseinhouse-storage/' + url,
        stream = True,
    )
    total_size = int(r.headers['content-length'])
    os.makedirs(os.path.dirname(filename), exist_ok = True)
    with open(filename, 'wb') as f:
        for data in tqdm(
            iterable = r.iter_content(chunk_size = 1048576),
            total = total_size / 1048576,
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
                    if not os.path.isfile(item):
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
