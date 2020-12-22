import tensorflow as tf
import math


def init_(t, dim = None):
    dim = dim if dim is not None else t.shape[-1]
    std = 1.0 / math.sqrt(dim)
    return nn.init.normal_(t, mean = 0, std = std)


class Model(tf.keras.Model):
    def __init__(
        self,
        dim,
        heads = 4,
        num_keys = 128,
        topk = 32,
        dim_head = 256,
        input_dropout = 0.0,
        query_dropout = 0.0,
        value_dropout = 0.0,
        **kwargs
    ):
        super().__init__(self, **kwargs)
        assert (
            dim % heads == 0
        ), 'dimension must be divisible by number of heads'
        self.topk = topk
        self.heads = heads
        self.num_keys = num_keys
        dim_query = dim_head * heads
        self.to_queries = tf.keras.layers.Dense(dim_query, use_bias = False)
        self.keys = tf.zeros(shape = (heads, num_keys, 2, dim_head // 2))
