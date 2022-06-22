from inspect import isfunction
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers


# helper functions

# The three functions of rearrange, irearrange and repeat have been written
# due to the incompatibility of the einops library with tensorflow 2.x.

def rearrange(x, r=2):
    b = tf.shape(x)
    b1 = b[:-1]
    b2 = b[-1, None]
    b3 = tf.constant([r], dtype=tf.int32)
    b4 = tf.cast(b2/b3, dtype=tf.int32)
    b_ = tf.concat([b1, b4, b3], axis=0)

    return tf.reshape(x, b_)


def irearrange(x):
    c = tf.shape(x)
    c1 = c[:-2]
    c2 = tf.reduce_prod(c[-2:])[None]
    c_ = tf.concat([c1, c2], axis=0)

    return tf.reshape(x, c_)


def repeat(x, r):
    c = tf.ones_like(tf.shape(x), dtype=tf.int32)
    c1 = c[:-1]
    c2 = c[-1][None] * r
    c_ = tf.concat([c1, c2], axis=0)

    return tf.tile(x, c_)


def exists(val):
    return val is not None


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
               ), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: tf.broadcast_to(t[0], t[1]), zip(tensors, expandable_shapes)))
    return tf.concat(tensors, axis=dim)

# rotary embedding helper functions


def rotate_half(x):
    x = rearrange(x, r=2)
    x1, x2 = tf.unstack(x, axis=-1)
    x = tf.stack((-x2, x1), axis=-1)
    return irearrange(x)


def apply_rotary_emb(freqs, t, start_index=0):
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * tf.cos(freqs)) + (rotate_half(t) * tf.sin(freqs))
    return tf.concat((t_left, t, t_right), axis=-1)

# learned rotation helpers


def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = tf.einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = irearrange(rotations)

    rotations = repeat(rotations, r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)


# classes

class RotaryEmbedding(layers.Layer):
    def __init__(
        self,
        dim,
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False
    ):
        super(RotaryEmbedding, self).__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = tf.convert_to_tensor(1. / (theta ** (np.arange(0, dim, 2)[:(dim // 2)] / dim)), dtype=tf.float32)
        elif freqs_for == 'pixel':
            freqs = tf.convert_to_tensor(np.logspace(0., np.log(max_freq / 2) / np.log(2),
                                         dim // 2, base=2) * np.pi, dtype=tf.float32)
        elif freqs_for == 'constant':
            freqs = tf.ones(num_freqs, dtype=tf.float32)
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.cache = dict()

        if learned_freq:
            self.freqs = tf.Variable(freqs, trainable=True)
        else:
            #    self.register_buffer('freqs', freqs)
            self.freqs = freqs

    def call(self, t, cache_key=None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if isfunction(t):
            t = t()

        freqs = self.freqs

        freqs = tf.einsum('..., f -> ... f', tf.cast(t, dtype=freqs.dtype), freqs)
        freqs = repeat(freqs, r=2)

        if exists(cache_key):
            self.cache[cache_key] = freqs

        return freqs
