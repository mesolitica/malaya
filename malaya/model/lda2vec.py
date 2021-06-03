from malaya.function import get_device, generate_session
import tensorflow.compat.v1 as tf
import numpy as np


class LDA2Vec:
    def __init__(
        self,
        num_unique_documents,
        vocab_size,
        num_topics,
        freqs,
        embedding_size=128,
        num_sampled=40,
        learning_rate=1e-3,
        lmbda=150.0,
        alpha=None,
        power=0.75,
        batch_size=32,
        clip_gradients=5.0,
        **kwargs
    ):
        device = get_device(**kwargs)
        _graph = tf.Graph()

        with _graph.as_default():
            with tf.device(device):
                moving_avgs = tf.train.ExponentialMovingAverage(0.9)
                self.batch_size = batch_size
                self.freqs = freqs

                self.X = tf.placeholder(tf.int32, shape=[None])
                self.Y = tf.placeholder(tf.int64, shape=[None])
                self.DOC = tf.placeholder(tf.int32, shape=[None])
                self.switch_loss = tf.Variable(0, trainable=False)
                train_labels = tf.reshape(self.Y, [-1, 1])
                sampler = tf.nn.fixed_unigram_candidate_sampler(
                    train_labels,
                    num_true=1,
                    num_sampled=num_sampled,
                    unique=True,
                    range_max=vocab_size,
                    distortion=power,
                    unigrams=self.freqs,
                )

                self.word_embedding = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)
                )
                self.nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [vocab_size, embedding_size],
                        stddev=tf.sqrt(1 / embedding_size),
                    )
                )
                self.nce_biases = tf.Variable(tf.zeros([vocab_size]))
                scalar = 1 / np.sqrt(num_unique_documents + num_topics)
                self.doc_embedding = tf.Variable(
                    tf.random_normal(
                        [num_unique_documents, num_topics],
                        mean=0,
                        stddev=50 * scalar,
                    )
                )
                self.topic_embedding = tf.get_variable(
                    'topic_embedding',
                    shape=[num_topics, embedding_size],
                    dtype=tf.float32,
                    initializer=tf.orthogonal_initializer(gain=scalar),
                )
                pivot = tf.nn.embedding_lookup(self.word_embedding, self.X)
                proportions = tf.nn.embedding_lookup(
                    self.doc_embedding, self.DOC
                )
                doc = tf.matmul(proportions, self.topic_embedding)
                doc_context = doc
                word_context = pivot
                context = tf.add(word_context, doc_context)
                loss_word2vec = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=self.nce_weights,
                        biases=self.nce_biases,
                        labels=self.Y,
                        inputs=context,
                        num_sampled=num_sampled,
                        num_classes=vocab_size,
                        num_true=1,
                        sampled_values=sampler,
                    )
                )
                self.fraction = tf.Variable(
                    1, trainable=False, dtype=tf.float32
                )

                n_topics = self.doc_embedding.get_shape()[1].value
                log_proportions = tf.nn.log_softmax(self.doc_embedding)
                if alpha is None:
                    alpha = 1.0 / n_topics
                loss = (alpha - 1) * log_proportions
                prior = tf.reduce_sum(loss)

                loss_lda = lmbda * self.fraction * prior
                global_step = tf.Variable(
                    0, trainable=False, name='global_step'
                )
                self.cost = tf.cond(
                    global_step < self.switch_loss,
                    lambda: loss_word2vec,
                    lambda: loss_word2vec + loss_lda,
                )
                loss_avgs_op = moving_avgs.apply(
                    [loss_lda, loss_word2vec, self.cost]
                )
                with tf.control_dependencies([loss_avgs_op]):
                    optimizer = tf.train.AdamOptimizer(
                        learning_rate=learning_rate
                    )
                    gvs = optimizer.compute_gradients(self.cost)
                    capped_gvs = [
                        (
                            tf.clip_by_value(
                                grad, -clip_gradients, clip_gradients
                            ),
                            var,
                        )
                        for grad, var in gvs
                    ]
                    self.optimizer = optimizer.apply_gradients(capped_gvs)
                self.sess = generate_session(_graph, **kwargs)
                self.sess.run(tf.global_variables_initializer())

    def train(
        self, pivot_words, target_words, doc_ids, num_epochs, switch_loss=3
    ):
        from tqdm import tqdm

        temp_fraction = self.batch_size / len(pivot_words)
        self.sess.run(tf.assign(self.fraction, temp_fraction))
        self.sess.run(tf.assign(self.switch_loss, switch_loss))
        for e in range(num_epochs):
            pbar = tqdm(
                range(0, len(pivot_words), self.batch_size),
                desc='minibatch loop',
            )
            for i in pbar:
                batch_x = pivot_words[
                    i: min(i + self.batch_size, len(pivot_words))
                ]
                batch_y = target_words[
                    i: min(i + self.batch_size, len(pivot_words))
                ]
                batch_doc = doc_ids[
                    i: min(i + self.batch_size, len(pivot_words))
                ]
                _, cost = self.sess.run(
                    [self.optimizer, self.cost],
                    feed_dict={
                        self.X: batch_x,
                        self.Y: batch_y,
                        self.DOC: batch_doc,
                    },
                )
                pbar.set_postfix(cost=cost, epoch=e + 1)
