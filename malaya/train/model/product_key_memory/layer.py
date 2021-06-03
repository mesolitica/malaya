import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer


class Scaling(Initializer):
    def __init__(self, seed=None, dtype=tf.float32):
        self.seed = seed
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        stdv = 1.0 / (shape[0] * shape[1])
        w = tf.random.uniform(
            shape,
            minval=-stdv,
            maxval=stdv,
            dtype=self.dtype,
            seed=self.seed,
        )
        std = tf.math.reduce_std(w)
        scale = (std / self.reference) ** 0.5
        w = w / scale
        return w
