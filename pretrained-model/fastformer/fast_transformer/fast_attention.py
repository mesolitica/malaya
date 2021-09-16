import tensorflow as tf
from einops import rearrange, reduce
from rotary_embedding_tensorflow import apply_rotary_emb


class FastAttention(tf.keras.layers.Layer):
    def __init__(
        self, dim, heads=8, dim_head=64, max_seq_len=None, pos_emb=None, mask=None
    ):
        super(FastAttention, self).__init__()

        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.mask = mask

        self.to_qkv = tf.keras.layers.Dense(
            inner_dim * 3, input_dim=dim, use_bias=False
        )

        if pos_emb is None and max_seq_len is None:
            raise Exception(
                "If you are using Rotary positional embeddings, max_seq_len must be passed in"
            )

        self.pos_emb = pos_emb
        self.max_seq_len = max_seq_len

        # reduce pairs of consecutive feature dimension before doing projection to attention logits
        if pos_emb is None:
            kv_attn_proj_divisor = 1
        else:
            kv_attn_proj_divisor = 2

        # project queries to query attention logits
        self.to_q_attn_logits = tf.keras.layers.Dense(
            1, input_dim=dim_head, use_bias=False
        )

        # project keys to key attention logits
        self.to_k_attn_logits = tf.keras.layers.Dense(
            1, input_dim=dim_head // kv_attn_proj_divisor, use_bias=False
        )

        self.to_r = tf.keras.layers.Dense(
            dim_head, input_dim=dim_head // kv_attn_proj_divisor
        )

        self.to_out = tf.keras.layers.Dense(dim, input_dim=inner_dim)

    def call(self, x, **kwargs):
        n = tf.shape(x)[1]
        h = self.heads

        use_rotary_emb = False
        if self.pos_emb is not None:
            use_rotary_emb = True

        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)

        queries, keys, values = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv
        )

        mask_value = -1e9
        self.mask = rearrange(self.mask, "b n -> b () n")

        # relative positional encoding is needed
        if use_rotary_emb:
            frequencies = self.pos_emb(
                tf.range(self.max_seq_len), cache_key=self.max_seq_len
            )
            frequencies = rearrange(frequencies[:n], "n d -> () () n d")
            query_aggr, keys_aggr, values_aggr = map(
                lambda t: apply_rotary_emb(frequencies, t), (queries, keys, values)
            )
        else:
            query_aggr, keys_aggr, values_aggr = queries, keys, values

        # query attention logits
        query_attn_logits = (
            rearrange(self.to_q_attn_logits(queries), "b h n () -> b h n") * self.scale
        )
        mask = tf.tile(self.mask, (1, tf.shape(query_attn_logits)[1], 1))
        fill = tf.fill(tf.shape(mask), mask_value)
        query_attn_logits = tf.where(mask, query_attn_logits, fill)
        print(query_attn_logits)
        query_attn = tf.nn.softmax(query_attn_logits)

        # global query token
        global_query = tf.einsum("b h n, b h n d -> b h d", query_attn, query_aggr)
        global_query = rearrange(global_query, "b h d -> b h () d")

        # bias keys with global query token
        keys = keys * global_query

        # if using rotary embeddings, do an inner product between adjacent pairs in the feature dimension
        if use_rotary_emb:
            keys = reduce(keys, "b h n (d r) -> b h n d", "sum", r=2)

        # key attention logits
        keys_attn_logits = (
            rearrange(self.to_k_attn_logits(keys), "b h n () -> b h n") * self.scale
        )
        keys_attn_logits = tf.where(mask, keys_attn_logits, fill)
        print(query_attn_logits)
        keys_attn = tf.nn.softmax(keys_attn_logits)

        # global key token
        global_keys = tf.einsum("b h n, b h n d -> b h d", keys_attn, keys_aggr)
        global_keys = rearrange(global_keys, "b h d -> b h () d")

        # bias the values
        u = values_aggr * global_keys

        if use_rotary_emb:
            u = reduce(u, "b h n (d r) -> b h n d", "sum", r=2)

        r = self.to_r(u)

        # add queries as a residual
        r = r + queries

        # combine heads
        r = rearrange(r, "b h n d -> b n (h d)")
        return self.to_out(r)
