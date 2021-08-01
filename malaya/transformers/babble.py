# Bert has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model,
# by Alex Wang, Kyunghyun Cho, NeuralGen 2019
# https://colab.research.google.com/drive/1MxKZGtQ9SSBjTK5ArsZ5LKhkztzg52RV
# https://arxiv.org/abs/1902.04094

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import numpy as np
import math
from malaya.text.bpe import merge_sentencepiece_tokens, merge_wordpiece_tokens

CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'


def topk_distributions(logits, top_k):
    with tf.InteractiveSession().as_default():
        logits = tf.convert_to_tensor(logits)
        kth_vals, kth_idx = tf.nn.top_k(logits, k=top_k)
        dist = tfp.distributions.categorical.Categorical(logits=kth_vals)
        idx = tf.gather(
            kth_idx, tf.expand_dims(dist.sample(), -1), batch_dims=1
        )
        idx = tf.squeeze(idx, axis=-1)
        return idx.eval()


def distributions(logits):
    with tf.InteractiveSession().as_default():
        logits = tf.convert_to_tensor(logits)
        dist = tfp.distributions.categorical.Categorical(logits=logits)
        return dist.sample().eval()


def generate_step(
    logits,
    gen_idx,
    top_k=0,
    temperature=1.0,
    sample=False,
    return_list=True,
):
    logits = logits[:, gen_idx]
    logits = logits / temperature
    if top_k > 0:
        idx = topk_distributions(logits, top_k)
    elif sample:
        idx = distributions(logits)
    else:
        idx = np.argmax(logits, axis=-1)
    return idx.tolist() if return_list else idx


def tokenize_batch(batch, tokenizer):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]


def untokenize_batch(batch, tokenizer):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]


def get_init_text(seed_text, max_len, tokenizer, batch_size=1):
    batch = [seed_text + [MASK] * max_len + [SEP] for _ in range(batch_size)]
    return tokenize_batch(batch, tokenizer)


def sequential_generation(
    seed_text,
    model,
    batch_size=5,
    max_len=15,
    leed_out_len=1,
    temperature=1.0,
    top_k=100,
    burnin=20,
):
    mask_id = model._tokenizer.vocab['[MASK]']
    sep_id = model._tokenizer.vocab['[SEP]']
    seed_text = model._tokenizer.tokenize(seed_text)
    seed_len = len(seed_text)
    batch = get_init_text(
        seed_text, max_len, model._tokenizer, batch_size=batch_size
    )

    for ii in range(max_len):
        inp = [sent[: seed_len + ii] + [sep_id] for sent in batch]
        batch = np.array(batch)
        masks = np.ones(batch.shape)
        segments = np.zeros(batch.shape)
        out = model._sess.run(
            model._logits,
            feed_dict={
                model.X: batch,
                model.MASK: masks,
                model.segment_ids: segments,
            },
        )
        topk = top_k if (ii >= burnin) else 0
        idxs = generate_step(
            out,
            gen_idx=seed_len + ii,
            top_k=topk,
            temperature=temperature,
            sample=(ii < burnin),
        )
        for jj in range(batch_size):
            batch[jj][seed_len + ii] = idxs[jj]

    results = untokenize_batch(batch.tolist(), model._tokenizer)
    if hasattr(model._tokenizer, 'sp_model'):
        merge_function = merge_sentencepiece_tokens
    else:
        merge_function = merge_wordpiece_tokens

    outputs = []

    for r in results:
        r = [(t, 0) for t in r]
        r = merge_function(r)
        r = [t[0] for t in r]
        outputs.append(' '.join(r))

    return outputs
