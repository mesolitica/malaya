import numpy as np
import collections
import tensorflow as tf


def build_dataset(words, n_words, atleast = 1):
    count = [['PAD', 0], ['GO', 1], ['EOS', 2], ['UNK', 3]]
    counter = collections.Counter(words).most_common(n_words)
    counter = [i for i in counter if i[1] >= atleast]
    count.extend(counter)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    for word in words:
        index = dictionary.get(word, 3)
        data.append(index)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, dictionary, reversed_dictionary


class Vocabulary:
    def __init__(self, dictionary, rev_dictionary):
        self._dictionary = dictionary
        self._rev_dictionary = rev_dictionary

    @property
    def start_string(self):
        return self._dictionary['GO']

    @property
    def end_string(self):
        return self._dictionary['EOS']

    @property
    def unk(self):
        return self._dictionary['UNK']

    @property
    def size(self):
        return len(self._dictionary)

    def word_to_id(self, word):
        return self._dictionary.get(word, self.unk)

    def id_to_word(self, cur_id):
        return self._rev_dictionary.get(cur_id, self._rev_dictionary[3])

    def decode(self, cur_ids):
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence, reverse = False, split = True):

        if split:
            sentence = sentence.split()
        word_ids = [self.word_to_id(cur_word) for cur_word in sentence]

        if reverse:
            return np.array(
                [self.end_string] + word_ids + [self.start_string],
                dtype = np.int32,
            )
        else:
            return np.array(
                [self.start_string] + word_ids + [self.end_string],
                dtype = np.int32,
            )


class UnicodeCharsVocabulary(Vocabulary):
    def __init__(self, dictionary, rev_dictionary, max_word_length, **kwargs):
        super(UnicodeCharsVocabulary, self).__init__(
            dictionary, rev_dictionary, **kwargs
        )
        self._max_word_length = max_word_length
        self.bos_char = 256
        self.eos_char = 257
        self.bow_char = 258
        self.eow_char = 259
        self.pad_char = 260
        num_words = self.size

        self._word_char_ids = np.zeros(
            [num_words, max_word_length], dtype = np.int32
        )

        def _make_bos_eos(c):
            r = np.zeros([self._max_word_length], dtype = np.int32)
            r[:] = self.pad_char
            r[0] = self.bow_char
            r[1] = c
            r[2] = self.eow_char
            return r

        self.bos_chars = _make_bos_eos(self.bos_char)
        self.eos_chars = _make_bos_eos(self.eos_char)

        for i, word in enumerate(self._dictionary.keys()):
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

        self._word_char_ids[self.start_string] = self.bos_chars
        self._word_char_ids[self.end_string] = self.eos_chars

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype = np.int32)
        code[:] = self.pad_char
        word_encoded = word.encode('utf-8', 'ignore')[
            : (self.max_word_length - 2)
        ]
        code[0] = self.bow_char
        for k, chr_id in enumerate(word_encoded, start = 1):
            code[k] = chr_id

        code[len(word_encoded) + 1] = self.eow_char
        return code

    def word_to_char_ids(self, word):
        if word in self._dictionary:
            return self._word_char_ids[self._dictionary[word]]
        else:
            return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence, reverse = False, split = True):
        if split:
            sentence = sentence.split()
        chars_ids = [self.word_to_char_ids(cur_word) for cur_word in sentence]

        if reverse:
            return np.vstack([self.eos_chars] + chars_ids + [self.bos_chars])
        else:
            return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])


def _get_batch(generator, batch_size, num_steps, max_word_length):
    cur_stream = [None] * batch_size

    no_more_data = False
    while True:
        inputs = np.zeros([batch_size, num_steps], np.int32)
        if max_word_length is not None:
            char_inputs = np.zeros(
                [batch_size, num_steps, max_word_length], np.int32
            )
        else:
            char_inputs = None
        targets = np.zeros([batch_size, num_steps], np.int32)
        for i in range(batch_size):
            cur_pos = 0

            while cur_pos < num_steps:
                if cur_stream[i] is None or len(cur_stream[i][0]) <= 1:
                    try:
                        cur_stream[i] = list(next(generator))
                    except StopIteration:
                        no_more_data = True
                        break
                how_many = min(len(cur_stream[i][0]) - 1, num_steps - cur_pos)
                next_pos = cur_pos + how_many

                inputs[i, cur_pos:next_pos] = cur_stream[i][0][:how_many]
                if max_word_length is not None:
                    char_inputs[i, cur_pos:next_pos] = cur_stream[i][1][
                        :how_many
                    ]
                targets[i, cur_pos:next_pos] = cur_stream[i][0][
                    1 : how_many + 1
                ]

                cur_pos = next_pos

                cur_stream[i][0] = cur_stream[i][0][how_many:]
                if max_word_length is not None:
                    cur_stream[i][1] = cur_stream[i][1][how_many:]

        if no_more_data:
            break

        X = {
            'token_ids': inputs,
            'tokens_characters': char_inputs,
            'next_token_id': targets,
        }

        yield X


class LMDataset:
    def __init__(self, string, vocab, reverse = False):
        self._vocab = vocab
        self._string = string
        self._reverse = reverse
        self._use_char_inputs = hasattr(vocab, 'encode_chars')
        self._i = 0
        self._nids = len(self._string)

    def _load_string(self, string):
        if self._reverse:
            string = string.split()
            string.reverse()
            string = ' '.join(string)

        ids = self._vocab.encode(string, self._reverse)

        if self._use_char_inputs:
            chars_ids = self._vocab.encode_chars(string, self._reverse)
        else:
            chars_ids = None

        return list(zip([ids], [chars_ids]))[0]

    def get_sentence(self):
        while True:
            if self._i == self._nids:
                self._i = 0
            ret = self._load_string(self._string[self._i])
            self._i += 1
            yield ret

    @property
    def max_word_length(self):
        if self._use_char_inputs:
            return self._vocab.max_word_length
        else:
            return None

    def iter_batches(self, batch_size, num_steps):
        for X in _get_batch(
            self.get_sentence(), batch_size, num_steps, self.max_word_length
        ):
            yield X

    @property
    def vocab(self):
        return self._vocab


class BidirectionalLMDataset:
    def __init__(self, string, vocab):
        self._data_forward = LMDataset(string, vocab, reverse = False)
        self._data_reverse = LMDataset(string, vocab, reverse = True)

    def iter_batches(self, batch_size, num_steps):
        max_word_length = self._data_forward.max_word_length

        for X, Xr in zip(
            _get_batch(
                self._data_forward.get_sentence(),
                batch_size,
                num_steps,
                max_word_length,
            ),
            _get_batch(
                self._data_reverse.get_sentence(),
                batch_size,
                num_steps,
                max_word_length,
            ),
        ):

            for k, v in Xr.items():
                X[k + '_reverse'] = v

            yield X


def generate_batch(vocab, sentence, sentence_reverse, seq_len):
    batch = np.zeros((1, seq_len, vocab.max_word_length), dtype = np.int32)
    batch_reverse = np.zeros(
        (1, seq_len, vocab.max_word_length), dtype = np.int32
    )
    encoded = vocab.encode_chars(sentence)
    encoded_reverse = vocab.encode_chars(sentence, reverse = True)
    for no_w in range(encoded.shape[0]):
        batch[0, no_w] = encoded[no_w]
        batch_reverse[0, no_w] = encoded_reverse[no_w]
    return batch, batch_reverse


class ELMO:
    def __init__(self, options, is_training):
        self.options = options
        self.is_training = is_training
        self.bidirectional = options.get('bidirectional', False)

        self.char_inputs = 'char_cnn' in self.options

        self.share_embedding_softmax = options.get(
            'share_embedding_softmax', False
        )
        if self.char_inputs and self.share_embedding_softmax:
            raise ValueError(
                'Sharing softmax and embedding weights requires ' 'word input'
            )

        self.sample_softmax = options.get('sample_softmax', False)
        self._build()
        lr = options.get('learning_rate', 0.2)
        self.optimizer = tf.train.AdagradOptimizer(
            learning_rate = lr, initial_accumulator_value = 1.0
        ).minimize(self.total_loss)

    def _build_word_embeddings(self):
        n_tokens_vocab = self.options['n_tokens_vocab']
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        projection_dim = self.options['lstm']['projection_dim']
        self.token_ids = tf.placeholder(
            tf.int32, shape = (None, unroll_steps), name = 'token_ids'
        )
        self.batch_size = tf.shape(self.token_ids)[0]
        with tf.device('/cpu:0'):

            self.embedding_weights = tf.get_variable(
                'embedding',
                [n_tokens_vocab, projection_dim],
                dtype = tf.float32,
                initializer = tf.random_uniform_initializer(-1.0, 1.0),
            )
            self.embedding = tf.nn.embedding_lookup(
                self.embedding_weights, self.token_ids
            )

        if self.bidirectional:
            self.token_ids_reverse = tf.placeholder(
                tf.int32,
                shape = (None, unroll_steps),
                name = 'token_ids_reverse',
            )
            with tf.device('/cpu:0'):
                self.embedding_reverse = tf.nn.embedding_lookup(
                    self.embedding_weights, self.token_ids_reverse
                )

    def _build_word_char_embeddings(self):

        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']
        projection_dim = self.options['lstm']['projection_dim']

        cnn_options = self.options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        max_chars = cnn_options['max_characters_per_token']
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options['n_characters']

        if cnn_options['activation'] == 'tanh':
            activation = tf.nn.tanh
        elif cnn_options['activation'] == 'relu':
            activation = tf.nn.relu

        self.tokens_characters = tf.placeholder(
            tf.int32,
            shape = (None, unroll_steps, max_chars),
            name = 'tokens_characters',
        )
        self.batch_size = tf.shape(self.tokens_characters)[0]
        with tf.device('/cpu:0'):
            self.embedding_weights = tf.get_variable(
                'char_embed',
                [n_chars, char_embed_dim],
                dtype = tf.float32,
                initializer = tf.random_uniform_initializer(-1.0, 1.0),
            )
            self.char_embedding = tf.nn.embedding_lookup(
                self.embedding_weights, self.tokens_characters
            )

            if self.bidirectional:
                self.tokens_characters_reverse = tf.placeholder(
                    tf.int32,
                    shape = (None, unroll_steps, max_chars),
                    name = 'tokens_characters_reverse',
                )
                self.char_embedding_reverse = tf.nn.embedding_lookup(
                    self.embedding_weights, self.tokens_characters_reverse
                )

        def make_convolutions(inp, reuse):
            with tf.variable_scope('CNN', reuse = reuse) as scope:
                convolutions = []
                for i, (width, num) in enumerate(filters):
                    if cnn_options['activation'] == 'relu':
                        w_init = tf.random_uniform_initializer(
                            minval = -0.05, maxval = 0.05
                        )
                    elif cnn_options['activation'] == 'tanh':
                        w_init = tf.random_normal_initializer(
                            mean = 0.0,
                            stddev = np.sqrt(1.0 / (width * char_embed_dim)),
                        )
                    w = tf.get_variable(
                        'W_cnn_%s' % i,
                        [1, width, char_embed_dim, num],
                        initializer = w_init,
                        dtype = tf.float32,
                    )
                    b = tf.get_variable(
                        'b_cnn_%s' % i,
                        [num],
                        dtype = tf.float32,
                        initializer = tf.constant_initializer(0.0),
                    )
                    conv = (
                        tf.nn.conv2d(
                            inp, w, strides = [1, 1, 1, 1], padding = 'VALID'
                        )
                        + b
                    )
                    conv = tf.nn.max_pool(
                        conv,
                        [1, 1, max_chars - width + 1, 1],
                        [1, 1, 1, 1],
                        'VALID',
                    )
                    conv = activation(conv)
                    conv = tf.squeeze(conv, squeeze_dims = [2])

                    convolutions.append(conv)

            return tf.concat(convolutions, 2)

        reuse = tf.get_variable_scope().reuse
        embedding = make_convolutions(self.char_embedding, reuse)
        self.token_embedding_layers = [embedding]
        if self.bidirectional:
            embedding_reverse = make_convolutions(
                self.char_embedding_reverse, True
            )
        n_highway = cnn_options.get('n_highway')
        use_highway = n_highway is not None and n_highway > 0
        use_proj = n_filters != projection_dim

        if use_highway or use_proj:
            embedding = tf.reshape(embedding, [-1, n_filters])
            if self.bidirectional:
                embedding_reverse = tf.reshape(
                    embedding_reverse, [-1, n_filters]
                )

        if use_proj:
            assert n_filters > projection_dim
            with tf.variable_scope('CNN_proj') as scope:
                W_proj_cnn = tf.get_variable(
                    'W_proj',
                    [n_filters, projection_dim],
                    initializer = tf.random_normal_initializer(
                        mean = 0.0, stddev = np.sqrt(1.0 / n_filters)
                    ),
                    dtype = tf.float32,
                )
                b_proj_cnn = tf.get_variable(
                    'b_proj',
                    [projection_dim],
                    initializer = tf.constant_initializer(0.0),
                    dtype = tf.float32,
                )

        def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
            carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
            transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
            return carry_gate * transform_gate + (1.0 - carry_gate) * x

        if use_highway:
            highway_dim = n_filters

            for i in range(n_highway):
                with tf.variable_scope('CNN_high_%s' % i) as scope:
                    W_carry = tf.get_variable(
                        'W_carry',
                        [highway_dim, highway_dim],
                        initializer = tf.random_normal_initializer(
                            mean = 0.0, stddev = np.sqrt(1.0 / highway_dim)
                        ),
                        dtype = tf.float32,
                    )
                    b_carry = tf.get_variable(
                        'b_carry',
                        [highway_dim],
                        initializer = tf.constant_initializer(-2.0),
                        dtype = tf.float32,
                    )
                    W_transform = tf.get_variable(
                        'W_transform',
                        [highway_dim, highway_dim],
                        initializer = tf.random_normal_initializer(
                            mean = 0.0, stddev = np.sqrt(1.0 / highway_dim)
                        ),
                        dtype = tf.float32,
                    )
                    b_transform = tf.get_variable(
                        'b_transform',
                        [highway_dim],
                        initializer = tf.constant_initializer(0.0),
                        dtype = tf.float32,
                    )

                embedding = high(
                    embedding, W_carry, b_carry, W_transform, b_transform
                )
                if self.bidirectional:
                    embedding_reverse = high(
                        embedding_reverse,
                        W_carry,
                        b_carry,
                        W_transform,
                        b_transform,
                    )
                self.token_embedding_layers.append(
                    tf.reshape(
                        embedding, [self.batch_size, unroll_steps, highway_dim]
                    )
                )

        if use_proj:
            embedding = tf.matmul(embedding, W_proj_cnn) + b_proj_cnn
            if self.bidirectional:
                embedding_reverse = (
                    tf.matmul(embedding_reverse, W_proj_cnn) + b_proj_cnn
                )
            self.token_embedding_layers.append(
                tf.reshape(
                    embedding, [self.batch_size, unroll_steps, projection_dim]
                )
            )

        if use_highway or use_proj:
            shp = [self.batch_size, unroll_steps, projection_dim]
            embedding = tf.reshape(embedding, shp)
            if self.bidirectional:
                embedding_reverse = tf.reshape(embedding_reverse, shp)

        self.embedding = embedding
        if self.bidirectional:
            self.embedding_reverse = embedding_reverse

    def _build(self):
        n_tokens_vocab = self.options['n_tokens_vocab']
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        lstm_dim = self.options['lstm']['dim']
        projection_dim = self.options['lstm']['projection_dim']
        n_lstm_layers = self.options['lstm'].get('n_layers', 1)
        dropout = self.options['dropout']
        keep_prob = 1.0 - dropout

        if self.char_inputs:
            self._build_word_char_embeddings()
        else:
            self._build_word_embeddings()

        self.init_lstm_state = []
        self.final_lstm_state = []

        if self.bidirectional:
            lstm_inputs = [self.embedding, self.embedding_reverse]
        else:
            lstm_inputs = [self.embedding]

        cell_clip = self.options['lstm'].get('cell_clip')
        proj_clip = self.options['lstm'].get('proj_clip')

        use_skip_connections = self.options['lstm'].get('use_skip_connections')

        lstm_outputs = []
        for lstm_num, lstm_input in enumerate(lstm_inputs):
            lstm_cells = []
            for i in range(n_lstm_layers):
                lstm_cell = tf.nn.rnn_cell.LSTMCell(
                    lstm_dim,
                    num_proj = lstm_dim // 2,
                    cell_clip = cell_clip,
                    proj_clip = proj_clip,
                )

                if use_skip_connections:
                    if i == 0:
                        pass
                    else:
                        lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)

                if self.is_training:
                    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                        lstm_cell, input_keep_prob = keep_prob
                    )

                lstm_cells.append(lstm_cell)

            if n_lstm_layers > 1:
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
            else:
                lstm_cell = lstm_cells[0]

            with tf.control_dependencies([lstm_input]):
                self.init_lstm_state.append(
                    lstm_cell.zero_state(self.batch_size, tf.float32)
                )
                if self.bidirectional:
                    with tf.variable_scope('RNN_%s' % lstm_num):
                        _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                            lstm_cell,
                            tf.unstack(lstm_input, axis = 1),
                            initial_state = self.init_lstm_state[-1],
                        )
                else:
                    _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                        lstm_cell,
                        tf.unstack(lstm_input, axis = 1),
                        initial_state = self.init_lstm_state[-1],
                    )
                self.final_lstm_state.append(final_state)

            lstm_output_flat = tf.reshape(
                tf.stack(_lstm_output_unpacked, axis = 1), [-1, projection_dim]
            )
            tf.add_to_collection(
                'lstm_output_embeddings', _lstm_output_unpacked
            )

            lstm_outputs.append(lstm_output_flat)
        self._build_loss(lstm_outputs)

    def _build_loss(self, lstm_outputs):
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        n_tokens_vocab = self.options['n_tokens_vocab']

        def _get_next_token_placeholders(suffix):
            name = 'next_token_id' + suffix
            id_placeholder = tf.placeholder(
                tf.int32, shape = (None, unroll_steps), name = name
            )
            return id_placeholder

        self.next_token_id = _get_next_token_placeholders('')
        if self.bidirectional:
            self.next_token_id_reverse = _get_next_token_placeholders(
                '_reverse'
            )
        softmax_dim = self.options['lstm']['projection_dim']
        if self.share_embedding_softmax:
            self.softmax_W = self.embedding_weights

        with tf.variable_scope('softmax'), tf.device('/cpu:0'):
            softmax_init = tf.random_normal_initializer(
                0.0, 1.0 / np.sqrt(softmax_dim)
            )
            if not self.share_embedding_softmax:
                self.softmax_W = tf.get_variable(
                    'W',
                    [n_tokens_vocab, softmax_dim],
                    dtype = tf.float32,
                    initializer = softmax_init,
                )
            self.softmax_b = tf.get_variable(
                'b',
                [n_tokens_vocab],
                dtype = tf.float32,
                initializer = tf.constant_initializer(0.0),
            )

        self.individual_losses = []

        if self.bidirectional:
            next_ids = [self.next_token_id, self.next_token_id_reverse]
        else:
            next_ids = [self.next_token_id]

        self.output_scores = tf.identity(lstm_outputs, name = 'softmax_score')

        for id_placeholder, lstm_output_flat in zip(next_ids, lstm_outputs):
            next_token_id_flat = tf.reshape(id_placeholder, [-1, 1])
            with tf.control_dependencies([lstm_output_flat]):
                if self.is_training and self.sample_softmax:
                    losses = tf.nn.sampled_softmax_loss(
                        self.softmax_W,
                        self.softmax_b,
                        next_token_id_flat,
                        lstm_output_flat,
                        int(
                            self.options['n_negative_samples_batch']
                            * self.options['n_tokens_vocab']
                        ),
                        self.options['n_tokens_vocab'],
                        num_true = 1,
                    )

                else:
                    output_scores = (
                        tf.matmul(
                            lstm_output_flat, tf.transpose(self.softmax_W)
                        )
                        + self.softmax_b
                    )

                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits = self.output_scores,
                        labels = tf.squeeze(
                            next_token_id_flat, squeeze_dims = [1]
                        ),
                    )

            self.individual_losses.append(tf.reduce_mean(losses))

        if self.bidirectional:
            self.total_loss = 0.5 * (
                self.individual_losses[0] + self.individual_losses[1]
            )
        else:
            self.total_loss = self.individual_losses[0]


def freeze_graph(model_dir, output_node_names):

    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            'directory: %s' % model_dir
        )

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    absolute_model_dir = '/'.join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + '/elmo.pb'
    clear_devices = True
    with tf.Session(graph = tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(
            input_checkpoint + '.meta', clear_devices = clear_devices
        )
        saver.restore(sess, input_checkpoint)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_names.split(','),
        )
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('%d ops in the final graph.' % len(output_graph_def.node))


def train(model, options):
    a
