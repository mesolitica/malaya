# Bert has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model,
# by Alex Wang, Kyunghyun Cho, NeuralGen 2019
# https://colab.research.google.com/drive/1MxKZGtQ9SSBjTK5ArsZ5LKhkztzg52RV
# https://arxiv.org/abs/1902.04094

import tensorflow as tf
import tensorflow_probability as tfp
import math
import numpy as np


def topk_distributions(logits, top_k):
    with tf.Session().as_default():
        logits = tf.convert_to_tensor(logits)
        kth_vals, kth_idx = tf.nn.top_k(logits, k = top_k)
        dist = tfp.distributions.categorical.Categorical(logits = kth_vals)
        idx = tf.gather(
            kth_idx, tf.expand_dims(dist.sample(), -1), batch_dims = 1
        )
        idx = tf.squeeze(idx, axis = -1)
        return idx.eval()


def distributions(logits):
    with tf.Session().as_default():
        logits = tf.convert_to_tensor(logits)
        dist = tfp.distributions.categorical.Categorical(logits = logits)
        return dist.sample().eval()


def tokenize_batch(batch, tokenizer):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]


def get_init_text(
    seed_text, max_len, mask, sep, batch_size = 1, rand_init = False
):
    batch = [seed_text + [mask] * max_len + [sep] for _ in range(batch_size)]


def generate_step(
    logits,
    gen_idx,
    top_k = 0,
    temperature = 1.0,
    sample = False,
    return_list = True,
):
    logits = out[:, gen_idx]
    logits = logits / temperature
    if top_k > 0:
        idx = topk_distributions(logits, top_k)
    elif sample:
        idx = distributions(logits)
    else:
        idx = np.argmax(logits, axis = -1)
    return idx.tolist() if return_list else idx


def parallel_sequential_generation(
    seed_text,
    model,
    batch_size = 10,
    max_len = 15,
    top_k = 0,
    temperature = 1.0,
    max_iter = 300,
    burnin = 200,
):
    mask_id = model._tokenizer.vocab['[MASK]']
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)
    for ii in range(max_iter):
        kk = np.random.randint(0, max_len)
        for jj in range(batch_size):
            batch[jj][seed_len + kk] = mask_id
        batch = np.array(batch)
        masks = np.zeros(*batch.shape)
        out = model._sess.run(
            model._logits, feed_dict = {model.X: batch, model.MASK: masks}
        )
        topk = top_k if (ii >= burnin) else 0
        idxs = generate_step(
            out,
            gen_idx = seed_len + kk,
            top_k = topk,
            temperature = temperature,
            sample = (ii < burnin),
        )
        for jj in range(batch_size):
            batch[jj][seed_len + kk] = idxs[jj]
