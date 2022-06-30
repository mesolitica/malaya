from transformers import TFBertForMaskedLM
import tensorflow as tf

PAD_ID = 0
CLS_ID = 101
SEP_ID = 102


def return_extended_attention_mask(attention_mask, dtype):
    if len(tf.shape(attention_mask)) == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif len(tf.shape(attention_mask)) == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for input_ids or attention_mask"
        )
    extended_attention_mask = tf.cast(extended_attention_mask, dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


class BertGuideHead(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(BertGuideHead, self).__init__(**kwargs)

    def transpose_for_scores(self, x):
        new_x_shape = tf.concat([tf.shape(x)[:-1], (1, tf.shape(x)[-1])], axis=0)
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, [0, 2, 1, 3])

    def call(self, hidden_states_src, hidden_states_tgt,
             inputs_src, inputs_tgt,
             guide=None,
             extraction='softmax', softmax_threshold=0.001,
             train_so=True, train_co=False,
             output_prob=False,):

        attention_mask_src = tf.cast(((inputs_src == PAD_ID) + (inputs_src == CLS_ID) +
                                     (inputs_src == SEP_ID)), tf.float32)
        attention_mask_tgt = tf.cast(((inputs_tgt == PAD_ID) + (inputs_tgt == CLS_ID) +
                                     (inputs_tgt == SEP_ID)), tf.float32)

        len_src = tf.reduce_sum(1-attention_mask_src, -1)
        len_tgt = tf.reduce_sum(1-attention_mask_tgt, -1)

        attention_mask_src = return_extended_attention_mask(1-attention_mask_src, hidden_states_src.dtype)
        attention_mask_tgt = return_extended_attention_mask(1-attention_mask_tgt, hidden_states_src.dtype)

        query_src = self.transpose_for_scores(hidden_states_src)
        query_tgt = self.transpose_for_scores(hidden_states_tgt)
        key_src = query_src
        key_tgt = query_tgt
        value_src = query_src
        value_tgt = query_tgt

        attention_scores = tf.matmul(query_src, tf.transpose(key_tgt, [0, 1, 3, 2]))
        attention_scores_src = attention_scores + attention_mask_tgt
        attention_scores_tgt = attention_scores + tf.transpose(attention_mask_src, [0, 1, 3, 2])

        attention_probs_src = tf.nn.softmax(attention_scores_src, axis=-1)
        attention_probs_tgt = tf.nn.softmax(attention_scores_tgt, axis=-2)

        if guide is None:
            align_matrix = tf.cast(attention_probs_src > threshold, tf.float32) * \
                tf.cast(attention_probs_tgt > threshold, tf.float32)
            if not output_prob:
                return align_matrix

            attention_probs_src = tf.nn.softmax(attention_scores_src, axis=-1) / \
                tf.sqrt(tf.reshape(len_tgt, [-1, 1, 1, 1]))
            attention_probs_tgt = tf.nn.softmax(attention_scores_tgt, axis=-1) / \
                tf.sqrt(tf.reshape(len_src, [-1, 1, 1, 1]))
            align_prob = (2*attention_probs_src*attention_probs_tgt)/(attention_probs_src+attention_probs_tgt+1e-9)
            return align_matrix, align_prob

        so_loss = 0
        if train_so:
            so_loss_src = tf.reshape(tf.reduce_sum(tf.reduce_sum(attention_probs_src*guide, axis=-1), axis=-1), [-1])
            so_loss_tgt = tf.reshape(tf.reduce_sum(tf.reduce_sum(attention_probs_tgt*guide, axis=-1), axis=-1), [-1])

            so_loss = so_loss_src / len_src + so_loss_tgt / len_tgt
            so_loss = -tf.reduce_mean(so_loss)

        co_loss = 0
        if train_co:
            min_len = tf.reduce_min(len_src, len_tgt)
            trace = tf.squeeze(tf.matmul(attention_probs_src, tf.transpose(attention_probs_tgt, [0, 1, 3, 2])), 1)
            trace = tf.einsum('bii->b', trace)
            co_loss = -tf.reduce_mean(trace / min_len)

        return so_loss + co_loss
