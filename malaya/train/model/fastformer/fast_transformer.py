import tensorflow as tf
import numpy as np
from einops import rearrange
from einops.layers.tensorflow import Rearrange
from ..rotary_embedding_tensorflow import RotaryEmbedding

from .fast_attention import FastAttention


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, mult=4):
        super(FeedForward, self).__init__()
        self.dim = dim
        self.mult = mult
        self.d1 = tf.keras.layers.Dense(dim * mult, input_dim=dim)
        self.d2 = tf.keras.layers.Dense(dim, input_dim=dim * mult)

    def call(self, inputs, **kwargs):
        return self.d2(gelu(self.d1(inputs)))


class FastTransformer(tf.keras.Model):
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        heads=8,
        dim_head=64,
        ff_mult=4,
        absolute_pos_emb=False,
        mask=None,
    ):
        super(FastTransformer, self).__init__()

        self.token_emb = tf.keras.layers.Embedding(num_tokens, dim)
        self.mask = mask

        # positional embeddings
        if absolute_pos_emb:
            self.abs_pos_emb = tf.keras.layers.Embedding(max_seq_len, dim)
        else:
            self.abs_pos_emb = None

        layer_pos_emb = None
        if not absolute_pos_emb:
            assert (
                dim_head % 4
            ) == 0, (
                "dimension of the head must be divisible by 4 to use rotary embeddings"
            )
            layer_pos_emb = RotaryEmbedding(dim_head // 2)

        self.fast_tranformer_layers = []

        for _ in range(depth):
            attn = FastAttention(
                dim,
                dim_head=dim_head,
                heads=heads,
                pos_emb=layer_pos_emb,
                max_seq_len=max_seq_len,
                mask=self.mask,
            )
            ff = FeedForward(dim, mult=ff_mult)

            self.fast_tranformer_layers.append(PreNorm(dim, attn))
            self.fast_tranformer_layers.append(PreNorm(dim, ff))

        first_block = self.fast_tranformer_layers[0]
        for block in self.fast_tranformer_layers[1:]:
            block.fn.to_q_attn_logits = first_block.fn.to_q_attn_logits
            block.fn.to_k_attn_logits = first_block.fn.to_k_attn_logits

        self.to_pooler = tf.keras.Sequential(
            [
                # tf.keras.layers.LayerNormalization(axis=-1),
                tf.keras.layers.Dense(dim, input_dim=dim,
                                      activation=tf.tanh),
            ]
        )

    def call(self, x, **kwargs):
        n = tf.shape(x)[1]
        x = self.token_emb(x)

        if self.abs_pos_emb is not None:
            pos_emb = self.abs_pos_emb(tf.range(n))
            x = x + rearrange(pos_emb, "n d -> () n d")

        for current_layer in self.fast_tranformer_layers:
            x = current_layer(x) + x

        first_token_tensor = tf.squeeze(x[:, 0:1, :], axis=1)
        return x, self.to_pooler(first_token_tensor)
