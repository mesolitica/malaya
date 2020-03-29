import tensorflow as tf


def top_k_logits(logits, k):
    if k == 0:
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k = k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype = logits.dtype) * -1e10,
            logits,
        )

    return tf.cond(tf.equal(k, 0), lambda: logits, lambda: _top_k())


def top_p_logits(logits, p):
    with tf.variable_scope('top_p_logits'):
        logits_sort = tf.sort(logits, direction = 'DESCENDING')
        probs_sort = tf.nn.softmax(logits_sort)
        probs_sums = tf.cumsum(probs_sort, axis = 1, exclusive = True)
        logits_masked = tf.where(
            probs_sums < p, logits_sort, tf.ones_like(logits_sort) * 1000
        )  # [batchsize, vocab]
        min_logits = tf.reduce_min(
            logits_masked, axis = 1, keepdims = True
        )  # [batchsize, 1]
        return tf.where(
            logits < min_logits,
            tf.ones_like(logits, dtype = logits.dtype) * -1e10,
            logits,
        )
