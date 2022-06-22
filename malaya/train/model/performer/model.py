import tensorflow as tf
import numpy as np


def gelu(x):
    cdf = 0.5 * (
        1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044_715 * tf.pow(x, 3))))
    )
    return x * cdf


def get_shape_list(tensor, expected_rank=None, name=None):
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


def embedding_lookup(
    input_ids,
    vocab_size,
    embedding_size=128,
    initializer_range=0.02,
    word_embedding_name='word_embeddings',
    use_one_hot_embeddings=False,
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
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range),
    )

    flat_input_ids = tf.reshape(input_ids, [-1])
    if use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
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
    use_token_type=False,
    token_type_ids=None,
    token_type_vocab_size=2,
    token_type_embedding_name='token_type_embeddings',
    use_position_embeddings=True,
    position_embedding_name='position_embeddings',
    initializer_range=0.02,
    max_position_embeddings=512,
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
    input_shape = get_shape_list(input_tensor, expected_rank=3)
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
            name=token_type_embedding_name,
            shape=[token_type_vocab_size, width],
            initializer=create_initializer(initializer_range),
        )
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(
            flat_token_type_ids, depth=token_type_vocab_size
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
                name=position_embedding_name,
                shape=[max_position_embeddings, width],
                initializer=create_initializer(initializer_range),
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


class Forward(tf.keras.layers.Layer):
    def __init__(self, dim, mlp_dim, dropout, **kwargs):
        super(Forward, self).__init__(**kwargs)
        self.rate = dropout
        self.dense1 = tf.keras.layers.Dense(mlp_dim, activation=gelu)
        self.dense2 = tf.keras.layers.Dense(dim)
        self.dropout = tf.keras.layers.Dropout(self.rate)

    def call(self, inputs, training=True):
        X = self.dense1(inputs)
        X = self.dropout(X, training=training)
        X = self.dense2(X)
        X = self.dropout(X, training=training)
        return X


class FNetBlock(tf.keras.layers.Layer):
    def __init__(self, dim, mlp_dim, dropout=0.1, **kwargs):
        super(FNetBlock, self).__init__(name='FNetBlock', **kwargs)
        self.norm_fourier = tf.keras.layers.LayerNormalization()
        self.norm_ffn = tf.keras.layers.LayerNormalization()
        self.ffn = Forward(dim, mlp_dim, dropout=dropout)

    def call(self, inputs, training=True):
        X_complex = tf.cast(inputs, tf.complex64)
        X_fft = tf.math.real(tf.signal.fft2d(X_complex))
        X_norm1 = self.norm_fourier(X_fft + inputs, training=training)
        X_dense = self.ffn(X_norm1, training=training)
        X_norm2 = self.norm_ffn(X_dense + X_norm1, training=training)
        return X_norm2


class Model(tf.keras.Model):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        nlayer,
        head_size,
        intermediate_size,
        dropout=0.1,
        dropout_embedding=0.1,
        max_position_embeddings=512,
        **kwargs,
    ):
        super(Model, self).__init__(name='Model', **kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_embedding = dropout_embedding
        self.max_position_embeddings = max_position_embeddings
