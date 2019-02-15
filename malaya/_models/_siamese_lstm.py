import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

import tensorflow as tf
from tqdm import tqdm
from ..texts._text_functions import str_idx, build_dataset


class Model:
    def __init__(
        self,
        size_layer = 128,
        num_layers = 1,
        embedded_size = 128,
        dict_size = 5000,
        learning_rate = 1e-3,
        output_size = 300,
        dropout = 0.8,
    ):
        def cells(size, reuse = False):
            cell = tf.nn.rnn_cell.LSTMCell(
                size, initializer = tf.orthogonal_initializer(), reuse = reuse
            )
            return tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob = dropout
            )

        def birnn(inputs, scope):
            with tf.variable_scope(scope):
                for n in range(num_layers):
                    (out_fw, out_bw), (
                        state_fw,
                        state_bw,
                    ) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw = cells(size_layer // 2),
                        cell_bw = cells(size_layer // 2),
                        inputs = inputs,
                        dtype = tf.float32,
                        scope = 'bidirectional_rnn_%d' % (n),
                    )
                    inputs = tf.concat((out_fw, out_bw), 2)
                return tf.layers.dense(inputs[:, -1], output_size)

        self.X_left = tf.placeholder(tf.int32, [None, None])
        self.X_right = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None])
        self.batch_size = tf.shape(self.X_left)[0]
        encoder_embeddings = tf.Variable(
            tf.random_uniform([dict_size, embedded_size], -1, 1)
        )
        embedded_left = tf.nn.embedding_lookup(encoder_embeddings, self.X_left)
        embedded_right = tf.nn.embedding_lookup(
            encoder_embeddings, self.X_right
        )

        def contrastive_loss(y, d):
            tmp = y * tf.square(d)
            tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
            return (
                tf.reduce_sum(tmp + tmp2)
                / tf.cast(self.batch_size, tf.float32)
                / 2
            )

        self.output_left = birnn(embedded_left, 'left')
        self.output_right = birnn(embedded_right, 'right')
        self.distance = tf.sqrt(
            tf.reduce_sum(
                tf.square(tf.subtract(self.output_left, self.output_right)),
                1,
                keepdims = True,
            )
        )
        self.distance = tf.div(
            self.distance,
            tf.add(
                tf.sqrt(
                    tf.reduce_sum(
                        tf.square(self.output_left), 1, keepdims = True
                    )
                ),
                tf.sqrt(
                    tf.reduce_sum(
                        tf.square(self.output_right), 1, keepdims = True
                    )
                ),
            ),
        )
        self.distance = tf.reshape(self.distance, [-1])
        self.cost = contrastive_loss(self.Y, self.distance)

        self.temp_sim = tf.subtract(
            tf.ones_like(self.distance), tf.rint(self.distance)
        )
        correct_predictions = tf.equal(self.temp_sim, self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate = learning_rate
        ).minimize(self.cost)


def train_model(
    train_X_left,
    train_X_right,
    train_Y,
    epoch = 10,
    batch_size = 16,
    embedding_size = 256,
    output_size = 300,
    maxlen = 100,
    dropout = 0.8,
    num_layers = 1,
    **kwargs
):
    concat = (' '.join(train_X_left + train_X_right)).split()
    vocabulary_size = len(list(set(concat)))
    _, _, dictionary, reversed_dictionary = build_dataset(
        concat, vocabulary_size
    )
    _graph = tf.Graph()
    with _graph.as_default():
        sess = tf.InteractiveSession()
        model = Model(
            size_layer = embedding_size,
            num_layers = num_layers,
            embedded_size = embedding_size,
            dict_size = len(dictionary),
            output_size = output_size,
            dropout = dropout,
        )
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())

    vectors_left = str_idx(train_X_left, dictionary, maxlen, UNK = 3)
    vectors_right = str_idx(train_X_right, dictionary, maxlen, UNK = 3)
    for e in range(epoch):
        pbar = tqdm(
            range(0, len(vectors_left), batch_size), desc = 'minibatch loop'
        )
        for i in pbar:
            batch_x_left = vectors_left[
                i : min(i + batch_size, len(vectors_left))
            ]
            batch_x_right = vectors_right[
                i : min(i + batch_size, len(vectors_left))
            ]
            batch_y = train_Y[i : min(i + batch_size, len(vectors_left))]
            acc, loss, _ = sess.run(
                [model.accuracy, model.cost, model.optimizer],
                feed_dict = {
                    model.X_left: batch_x_left,
                    model.X_right: batch_x_right,
                    model.Y: batch_y,
                },
            )
            pbar.set_postfix(cost = loss, accuracy = acc)
    return sess, model, dictionary, saver, dropout


def load_siamese(location, json):
    graph = tf.Graph()
    with graph.as_default():
        model = Model(
            size_layer = json['embedding_size'],
            num_layers = json['num_layers'],
            embedded_size = json['embedding_size'],
            dict_size = len(json['dictionary']),
            output_size = json['output_size'],
            dropout = json['dropout'],
        )
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(sess, location + '/model.ckpt')
    return sess, model, saver
