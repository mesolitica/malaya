import tensorflow as tf
import numpy as np


def get_shape_list(tensor, expected_rank = None, name = None):
    """Returns a list of the shape of tensor, preferring static dimensions.
  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.
  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
    if name is None:
        name = tensor.name

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def create_initializer(initializer_range = 0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev = initializer_range)


def layer_norm(input_tensor, name = None):
    return tf.contrib.layers.layer_norm(
        inputs = input_tensor,
        begin_norm_axis = -1,
        begin_params_axis = -1,
        scope = name,
    )


def embedding_lookup(
    input_ids,
    vocab_size,
    embedding_size = 128,
    initializer_range = 0.02,
    word_embedding_name = 'word_embeddings',
    use_one_hot_embeddings = False,
):
    """Looks up words embeddings for id tensor.
  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.gather()`.
  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
    # This function assumes that the input is of shape [batch_size, seq_length,
    # num_inputs].
    #
    # If the input is a 2D tensor of shape [batch_size, seq_length], we
    # reshape to [batch_size, seq_length, 1].
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis = [-1])

    embedding_table = tf.get_variable(
        name = word_embedding_name,
        shape = [vocab_size, embedding_size],
        initializer = create_initializer(initializer_range),
    )

    flat_input_ids = tf.reshape(input_ids, [-1])
    if use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth = vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.gather(embedding_table, flat_input_ids)

    input_shape = get_shape_list(input_ids)

    output = tf.reshape(
        output, input_shape[0:-1] + [input_shape[-1] * embedding_size]
    )
    return (output, embedding_table)


def embedding_postprocessor(
    input_tensor,
    use_token_type = False,
    token_type_ids = None,
    token_type_vocab_size = 2,
    token_type_embedding_name = 'token_type_embeddings',
    use_position_embeddings = True,
    position_embedding_name = 'position_embeddings',
    initializer_range = 0.02,
    max_position_embeddings = 512,
):
    """Performs various post-processing on a word embedding tensor.
  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.
  Returns:
    float tensor with same shape as `input_tensor`.
  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
    input_shape = get_shape_list(input_tensor, expected_rank = 3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError(
                '`token_type_ids` must be specified if'
                '`use_token_type` is True.'
            )
        token_type_table = tf.get_variable(
            name = token_type_embedding_name,
            shape = [token_type_vocab_size, width],
            initializer = create_initializer(initializer_range),
        )
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(
            flat_token_type_ids, depth = token_type_vocab_size
        )
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(
            token_type_embeddings, [batch_size, seq_length, width]
        )
        output += token_type_embeddings

    if use_position_embeddings:
        assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(
                name = position_embedding_name,
                shape = [max_position_embeddings, width],
                initializer = create_initializer(initializer_range),
            )
            position_embeddings = tf.slice(
                full_position_embeddings, [0, 0], [seq_length, -1]
            )
            num_dims = len(output.shape.as_list())
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(
                position_embeddings, position_broadcast_shape
            )
            output += position_embeddings

    return output


def gelu(x):
    cdf = 0.5 * (
        1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044_715 * tf.pow(x, 3))))
    )
    return x * cdf


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, hidden_dim, dropout = 0.0, **kwargs):
        super(FeedForward, self).__init__(name = 'FeedForward', **kwargs)
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(hidden_dim, activation = gelu))
        self.net.add(tf.keras.layers.Dropout(dropout))
        self.net.add(tf.keras.layers.Dense(dim))
        self.net.add(tf.keras.layers.Dropout(dropout))

    def call(self, x, training = True):
        return self.net(x, training = training)


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, fn, **kwargs):
        super(PreNorm, self).__init__(name = 'PreNorm', **kwargs)
        self.norm = tf.keras.layers.LayerNormalization()
        self.fn = fn

    def call(self, x, training = True):
        return self.fn(self.norm(x), training = training)


class FNetBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FNetBlock, self).__init__(name = 'FNetBlock', **kwargs)

    def call(self, x):
        g = tf.map_fn(
            tf.signal.fft, tf.cast(tf.transpose(x, (2, 0, 1)), tf.complex64)
        )
        g = tf.transpose(g, (2, 1, 0))
        g = tf.map_fn(tf.signal.fft, g)
        g = tf.transpose(g, (1, 0, 2))
        g = tf.math.real(g)
        return g


class Model(tf.keras.Model):
    def __init__(
        self,
        dim,
        vocab_size,
        depth,
        mlp_dim,
        dropout = 0.0,
        dropout_embedding = 0.1,
        max_position_embeddings = 1024,
        **kwargs,
    ):
        super(Model, self).__init__(name = 'Model', **kwargs)
        self.dim = dim
        self.hidden_size = dim
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.attn, self.ff = [], []
        for _ in range(depth):
            self.attn.append(PreNorm(FNetBlock()))
            self.ff.append(
                PreNorm(FeedForward(dim, mlp_dim, dropout = dropout))
            )
        self.layernorm_dropout = tf.keras.Sequential()
        self.layernorm_dropout.add(tf.keras.layers.LayerNormalization())
        self.layernorm_dropout.add(tf.keras.layers.Dropout(dropout_embedding))

    def call(self, x, token_type_ids = None, training = True):
        if token_type_ids is None:
            token_type_ids = tf.zeros(
                shape = [tf.shape(x)[0], tf.shape(x)[1]], dtype = tf.int32
            )
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids = x,
            vocab_size = self.vocab_size,
            embedding_size = self.dim,
            initializer_range = 0.02,
            word_embedding_name = 'word_embeddings',
            use_one_hot_embeddings = False,
        )
        self.embedding_output = embedding_postprocessor(
            input_tensor = self.embedding_output,
            use_token_type = True,
            token_type_ids = token_type_ids,
            token_type_vocab_size = 2,
            token_type_embedding_name = 'token_type_embeddings',
            use_position_embeddings = True,
            position_embedding_name = 'position_embeddings',
            initializer_range = 0.02,
            max_position_embeddings = self.max_position_embeddings,
        )
        x = self.embedding_output
        for no, attn in enumerate(self.attn):
            x = attn(x, training = training) + x
            x = self.ff[no](x, training = training) + x

        with tf.variable_scope('pooler'):
            first_token_tensor = tf.squeeze(x[:, 0:1, :], axis = 1)
            self.pooled_output = tf.layers.dense(
                first_token_tensor,
                self.hidden_size,
                activation = tf.tanh,
                kernel_initializer = create_initializer(0.02),
            )
        return x


# x = tf.placeholder(tf.int32, (None, None))
# model = Model(768, 32000, 12, 768)
# o = model(x)
