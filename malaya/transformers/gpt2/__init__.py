import tensorflow.compat.v1 as tf
import os
import json
from malaya.function import get_device, generate_session
from malaya.transformers.gpt2 import model as gpt2_model, encoder
from herpetologist import check_type


def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )

    return tf.cond(
        pred=tf.equal(k, 0),
        true_fn=lambda: logits,
        false_fn=lambda: _top_k(),
    )


def top_p_logits(logits, p):
    with tf.variable_scope('top_p_logits'):
        logits_sort = tf.sort(logits, direction='DESCENDING')
        probs_sort = tf.nn.softmax(logits_sort)
        probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
        logits_masked = tf.where(
            probs_sums < p, logits_sort, tf.ones_like(logits_sort) * 1000
        )
        min_logits = tf.reduce_min(
            input_tensor=logits_masked, axis=1, keepdims=True
        )
        return tf.where(
            logits < min_logits,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )


def sample_sequence(
    hparams,
    length,
    start_token=None,
    batch_size=None,
    context=None,
    temperature=1,
    top_k=0,
    top_p=0.0,
):
    if start_token is None:
        assert (
            context is not None
        ), 'Specify exactly one of start_token and context!'
    else:
        assert (
            context is None
        ), 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        lm_output = gpt2_model.model(
            hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE
        )

        logits = lm_output['logits'][:, :, : hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(
            gpt2_model.past_shape(hparams=hparams, batch_size=batch_size)
        )
        return {'logits': logits, 'presents': presents}

    with tf.name_scope('sample_sequence'):
        context_output = step(hparams, context[:, :-1])

        def body(past, prev, output):
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :] / tf.cast(
                temperature, tf.float32
            )
            if top_p > 0.0:
                logits = top_p_logits(logits, p=top_p)
            else:
                logits = top_k_logits(logits, k=top_k)
            samples = tf.random.categorical(
                logits, num_samples=1, dtype=tf.int32
            )
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
            ]

        def cond(*args):
            return True

        _, _, tokens = tf.while_loop(
            cond=cond,
            body=body,
            maximum_iterations=length,
            loop_vars=[context_output['presents'], context[:, -1], context],
            shape_invariants=[
                tf.TensorShape(
                    gpt2_model.past_shape(
                        hparams=hparams, batch_size=batch_size
                    )
                ),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tokens


class Model:
    def __init__(
        self, hparams, encoder, generate_length, temperature, top_k, **kwargs
    ):
        self._encoder = encoder
        device = get_device(**kwargs)
        self._graph = tf.Graph()
        with self._graph.as_default():
            with tf.device(device):
                self._X = tf.placeholder(tf.int32, [1, None])
                self._model = sample_sequence(
                    hparams=hparams,
                    length=generate_length,
                    context=self._X,
                    batch_size=1,
                    temperature=temperature,
                    top_k=top_k,
                )
                self._sess = generate_session(self._graph, **kwargs)
                self._sess.run(tf.global_variables_initializer())
                self._saver = tf.train.Saver(tf.trainable_variables())

    @check_type
    def generate(self, string: str):
        """
        generate a text given an initial string.

        Parameters
        ----------
        string : str

        Returns
        -------
        result: str
        """
        encoded = self._encoder.encode(string)
        out = self._sess.run(self._model, feed_dict={self._X: [encoded]})
        return self._encoder.decode(out[0])


@check_type
def load(
    model='345M',
    generate_length=100,
    temperature=1.0,
    top_k=40,
    **kwargs
):
    """
    Load gpt2 model.

    Parameters
    ----------
    model : str, optional (default='345M')
        Model architecture supported. Allowed values:

        * ``'117M'`` - GPT2 117M parameters.
        * ``'345M'`` - GPT2 345M parameters.

    generate_length : int, optional (default=256)
        length of sentence to generate.

    temperature : float, optional (default=1.0)
        temperature value, value should between 0 and 1.

    top_k : int, optional (default=40)
        top-k in nucleus sampling selection.

    Returns
    -------
    result: malaya.transformers.gpt2.Model class
    """

    from malaya.path import PATH_GPT2, S3_PATH_GPT2
    from malaya.function import check_file

    check_file(PATH_GPT2[model]['model'], S3_PATH_GPT2[model], **kwargs)

    if not os.path.exists(PATH_GPT2[model]['directory'] + 'model.ckpt'):
        import tarfile

        with tarfile.open(PATH_GPT2[model]['model']['model']) as tar:
            tar.extractall(path=PATH_GPT2[model]['path'])

    params = PATH_GPT2[model]['directory'] + 'hparams.json'
    merges = PATH_GPT2[model]['directory'] + 'bahasa-merges.txt'
    vocab = PATH_GPT2[model]['directory'] + 'bahasa-vocab.json'
    gpt2_checkpoint = PATH_GPT2[model]['directory'] + 'model.ckpt'

    hparams = gpt2_model.default_hparams()
    with open(params) as f:
        hparams.override_from_dict(json.load(f))

    with open(vocab, 'r') as f:
        en = json.load(f)
    with open(merges, 'r', encoding='utf-8') as f:
        bpe_data = f.read()

    bpe_merges = [
        tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]
    ]
    enc_malay = encoder.Encoder(encoder=en, bpe_merges=bpe_merges)

    model = Model(
        hparams, enc_malay, generate_length, temperature, top_k, **kwargs
    )
    model._saver.restore(model._sess, gpt2_checkpoint)
    return model
