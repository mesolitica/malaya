import tensorflow as tf


def top_k_logits(logits, k):
    if k == 0:
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )

    return tf.cond(tf.equal(k, 0), lambda: logits, lambda: _top_k())


def top_p_logits(logits, p):
    with tf.compat.v1.variable_scope('top_p_logits'):
        logits_sort = tf.sort(logits, direction='DESCENDING')
        probs_sort = tf.nn.softmax(logits_sort)
        probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
        logits_masked = tf.where(
            probs_sums < p, logits_sort, tf.ones_like(logits_sort) * 1000
        )  # [batchsize, vocab]
        min_logits = tf.reduce_min(
            logits_masked, axis=1, keepdims=True
        )  # [batchsize, 1]
        return tf.where(
            logits < min_logits,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )


def sample(translate_model, features):
    logits, losses = translate_model(features)
    logits = logits / translate_model.hparams.sampling_temp
    logits = top_p_logits(logits, translate_model.hparams.top_p)
    samples = tf.compat.v1.multinomial(
        logits,
        num_samples=translate_model.hparams.top_k,
        output_dtype=tf.int32,
    )
    return samples, logits, losses


def nucleus_sampling(translate_model, features, decode_length):
    """A slow greedy inference method.
    Quadratic time in decode_length.
    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": None
          "logits": `Tensor` of shape [batch_size, time, 1, 1, vocab_size].
          "losses": a dictionary: {loss-name (string): floating point `Scalar`}
      }
    """
    if not features:
        features = {}
    inputs_old = None
    if 'inputs' in features and len(features['inputs'].shape) < 4:
        inputs_old = features['inputs']
        features['inputs'] = tf.expand_dims(features['inputs'], 2)
    # Save the targets in a var and reassign it after the tf.while loop to avoid
    # having targets being in a 'while' frame. This ensures targets when used
    # in metric functions stays in the same frame as other vars.
    targets_old = features.get('targets', None)

    target_modality = translate_model._problem_hparams.modality['targets']

    def infer_step(recent_output, recent_logits, unused_loss):
        """Inference step."""
        if not tf.executing_eagerly():
            if translate_model._target_modality_is_real:
                dim = translate_model._problem_hparams.vocab_size['targets']
                if dim is not None and hasattr(
                    translate_model._hparams, 'vocab_divisor'
                ):
                    dim += (-dim) % translate_model._hparams.vocab_divisor
                recent_output.set_shape([None, None, None, dim])
            else:
                recent_output.set_shape([None, None, None, 1])
        padded = tf.pad(recent_output, [[0, 0], [0, 1], [0, 0], [0, 0]])
        features['targets'] = padded
        # This is inefficient in that it generates samples at all timesteps,
        # not just the last one, except if target_modality is pointwise.
        samples, logits, losses = sample(translate_model, features)
        # Concatenate the already-generated recent_output with last timestep
        # of the newly-generated samples.
        top = translate_model._hparams.top.get(
            'targets', modalities.get_top(target_modality)
        )
        if getattr(top, 'pointwise', False):
            cur_sample = samples[:, -1, :, :]
        else:
            cur_sample = samples[
                :, common_layers.shape_list(recent_output)[1], :, :
            ]
        if translate_model._target_modality_is_real:
            cur_sample = tf.expand_dims(cur_sample, axis=1)
            samples = tf.concat([recent_output, cur_sample], axis=1)
        else:
            cur_sample = tf.to_int64(tf.expand_dims(cur_sample, axis=1))
            samples = tf.concat([recent_output, cur_sample], axis=1)
            if not tf.executing_eagerly():
                samples.set_shape([None, None, None, 1])

        # Assuming we have one shard for logits.
        logits = tf.concat([recent_logits, logits[:, -1:]], 1)
        loss = sum([l for l in losses.values() if l is not None])
        return samples, logits, loss

    # Create an initial output tensor. This will be passed
    # to the infer_step, which adds one timestep at every iteration.
    if 'partial_targets' in features:
        initial_output = tf.to_int64(features['partial_targets'])
        while len(initial_output.get_shape().as_list()) < 4:
            initial_output = tf.expand_dims(initial_output, 2)
        batch_size = common_layers.shape_list(initial_output)[0]
    else:
        batch_size = common_layers.shape_list(features['inputs'])[0]
        if translate_model._target_modality_is_real:
            dim = translate_model._problem_hparams.vocab_size['targets']
            if dim is not None and hasattr(
                translate_model._hparams, 'vocab_divisor'
            ):
                dim += (-dim) % translate_model._hparams.vocab_divisor
            initial_output = tf.zeros(
                (batch_size, 0, 1, dim), dtype=tf.float32
            )
        else:
            initial_output = tf.zeros((batch_size, 0, 1, 1), dtype=tf.int64)
    # Hack: foldl complains when the output shape is less specified than the
    # input shape, so we confuse it about the input shape.
    initial_output = tf.slice(
        initial_output, [0, 0, 0, 0], common_layers.shape_list(initial_output)
    )
    target_modality = translate_model._problem_hparams.modality['targets']
    if (
        target_modality == modalities.ModalityType.CLASS_LABEL
        or translate_model._problem_hparams.get('regression_targets')
    ):
        decode_length = 1
    else:
        if 'partial_targets' in features:
            prefix_length = common_layers.shape_list(
                features['partial_targets']
            )[1]
        else:
            prefix_length = common_layers.shape_list(features['inputs'])[1]
        decode_length = prefix_length + decode_length

    # Initial values of result, logits and loss.
    result = initial_output
    vocab_size = translate_model._problem_hparams.vocab_size['targets']
    if vocab_size is not None and hasattr(
        translate_model._hparams, 'vocab_divisor'
    ):
        vocab_size += (-vocab_size) % translate_model._hparams.vocab_divisor
    if translate_model._target_modality_is_real:
        logits = tf.zeros((batch_size, 0, 1, vocab_size))
        logits_shape_inv = [None, None, None, None]
    else:
        # tensor of shape [batch_size, time, 1, 1, vocab_size]
        logits = tf.zeros((batch_size, 0, 1, 1, vocab_size))
        logits_shape_inv = [None, None, None, None, None]
    if not tf.executing_eagerly():
        logits.set_shape(logits_shape_inv)

    loss = 0.0

    def while_exit_cond(
        result, logits, loss
    ):  # pylint: disable=unused-argument
        """Exit the loop either if reach decode_length or EOS."""
        length = common_layers.shape_list(result)[1]

        not_overflow = length < decode_length

        if translate_model._problem_hparams.stop_at_eos:

            def fn_not_eos():
                return tf.not_equal(  # Check if the last predicted element is a EOS
                    tf.squeeze(result[:, -1, :, :]), text_encoder.EOS_ID
                )

            not_eos = tf.cond(
                # We only check for early stopping if there is at least 1 element (
                # otherwise not_eos will crash).
                tf.not_equal(length, 0),
                fn_not_eos,
                lambda: True,
            )

            return tf.cond(
                tf.equal(batch_size, 1),
                # If batch_size == 1, we check EOS for early stopping.
                lambda: tf.logical_and(not_overflow, not_eos),
                # Else, just wait for max length
                lambda: not_overflow,
            )
        return not_overflow

    result, logits, loss = tf.while_loop(
        while_exit_cond,
        infer_step,
        [result, logits, loss],
        shape_invariants=[
            tf.TensorShape([None, None, None, None]),
            tf.TensorShape(logits_shape_inv),
            tf.TensorShape([]),
        ],
        back_prop=False,
        parallel_iterations=1,
    )
    if inputs_old is not None:  # Restore to not confuse Estimator.
        features['inputs'] = inputs_old
    # Reassign targets back to the previous value.
    if targets_old is not None:
        features['targets'] = targets_old
    losses = {'training': loss}
    if 'partial_targets' in features:
        partial_target_length = common_layers.shape_list(
            features['partial_targets']
        )[1]
        result = tf.slice(
            result, [0, partial_target_length, 0, 0], [-1, -1, -1, -1]
        )
    return {
        'outputs': result,
        'scores': None,
        'logits': logits,
        'losses': losses,
    }
