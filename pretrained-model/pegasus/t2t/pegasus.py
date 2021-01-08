# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer model from "Attention Is All You Need".

The Transformer model consists of an encoder and a decoder. Both are stacks
of self-attention layers followed by feed-forward layers. This model yields
good results on a number of problems, especially in NLP and machine translation.

See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) for the full
description of the model and the results obtained with its early version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import librispeech
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.layers import transformer_layers
from tensor2tensor.layers import transformer_memory
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow.compat.v1 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import inplace_ops
from tensorflow.python.util import nest

# pylint: enable=g-direct-tensorflow-import

# Alias some commonly reused layers, here and elsewhere.
transformer_prepare_encoder = transformer_layers.transformer_prepare_encoder
transformer_encoder = transformer_layers.transformer_encoder
transformer_ffn_layer = transformer_layers.transformer_ffn_layer


def transformer_encode(
    encoder_function,
    inputs,
    target_space,
    hparams,
    attention_weights = None,
    features = None,
    losses = None,
    prepare_encoder_fn = None,
    **kwargs
):
    """Encode transformer inputs.

  Args:
    encoder_function: the encoder function
    inputs: Transformer inputs [batch_size, input_length, 1, hidden_dim] which
      will be flattened along the two spatial dimensions.
    target_space: scalar, target space ID.
    hparams: hyperparameters for model.
    attention_weights: weight to store attention to.
    features: optionally pass the entire features dictionary as well. This is
      needed now for "packed" datasets.
    losses: optional list onto which to append extra training losses
    prepare_encoder_fn: optional, alternative to transformer_prepare_encoder.
    **kwargs: additional arguments to pass to encoder_function

  Returns:
    Tuple of:
        encoder_output: Encoder representation.
            [batch_size, input_length, hidden_dim]
        encoder_decoder_attention_bias: Bias and mask weights for
            encoder-decoder attention. [batch_size, input_length]
  """
    inputs = common_layers.flatten4d3d(inputs)

    if not prepare_encoder_fn:
        prepare_encoder_fn = transformer_prepare_encoder
    encoder_input, self_attention_bias, encoder_decoder_attention_bias = prepare_encoder_fn(
        inputs, target_space, hparams, features = features
    )

    mlperf_log.transformer_print(
        key = mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
        value = hparams.layer_prepostprocess_dropout,
        hparams = hparams,
    )

    encoder_input = tf.nn.dropout(
        encoder_input, 1.0 - hparams.layer_prepostprocess_dropout
    )

    attn_bias_for_padding = None
    # Otherwise the encoder will just use encoder_self_attention_bias.
    if hparams.unidirectional_encoder:
        attn_bias_for_padding = encoder_decoder_attention_bias

    encoder_output = encoder_function(
        encoder_input,
        self_attention_bias,
        hparams,
        nonpadding = features_to_nonpadding(features, 'inputs'),
        save_weights_to = attention_weights,
        make_image_summary = not common_layers.is_xla_compiled(),
        losses = losses,
        attn_bias_for_padding = attn_bias_for_padding,
        **kwargs
    )

    return encoder_output, encoder_decoder_attention_bias


def transformer_decode(
    decoder_function,
    decoder_input,
    encoder_output,
    encoder_decoder_attention_bias,
    decoder_self_attention_bias,
    hparams,
    attention_weights = None,
    cache = None,
    decode_loop_step = None,
    nonpadding = None,
    losses = None,
    **kwargs
):
    """Decode Transformer outputs from encoder representation.

  Args:
    decoder_function: the decoder function
    decoder_input: inputs to bottom of the model. [batch_size, decoder_length,
      hidden_dim]
    encoder_output: Encoder representation. [batch_size, input_length,
      hidden_dim]
    encoder_decoder_attention_bias: Bias and mask weights for encoder-decoder
      attention. [batch_size, input_length]
    decoder_self_attention_bias: Bias and mask weights for decoder
      self-attention. [batch_size, decoder_length]
    hparams: hyperparameters for model.
    attention_weights: weight to store attention to.
    cache: dict, containing tensors which are the results of previous
      attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop. Only used
      for inference on TPU.
    nonpadding: optional Tensor with shape [batch_size, decoder_length]
    losses: optional list onto which to append extra training losses
    **kwargs: additional arguments to pass to decoder_function

  Returns:
    Final decoder representation. [batch_size, decoder_length, hidden_dim]
  """
    mlperf_log.transformer_print(
        key = mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
        value = hparams.layer_prepostprocess_dropout,
        hparams = hparams,
    )
    decoder_input = tf.nn.dropout(
        decoder_input, 1.0 - hparams.layer_prepostprocess_dropout
    )

    decoder_output = decoder_function(
        decoder_input,
        encoder_output,
        decoder_self_attention_bias,
        encoder_decoder_attention_bias,
        hparams,
        cache = cache,
        decode_loop_step = decode_loop_step,
        nonpadding = nonpadding,
        save_weights_to = attention_weights,
        losses = losses,
        **kwargs
    )

    if (
        common_layers.is_xla_compiled()
        and hparams.mode == tf.estimator.ModeKeys.TRAIN
    ):
        # TPU does not react kindly to extra dimensions.
        # TODO(noam): remove this once TPU is more forgiving of extra dims.
        return decoder_output
    else:
        # Expand since t2t expects 4d tensors.
        return tf.expand_dims(decoder_output, axis = 2)


@registry.register_model
class Pegasus(t2t_model.T2TModel):
    """Attention net.  See file docstring."""

    def __init__(self, *args, **kwargs):
        super(Pegasus, self).__init__(*args, **kwargs)
        self.attention_weights = {}  # For visualizing attention heads.
        self.recurrent_memory_by_layer = (
            None
        )  # Override to enable recurrent memory
        self._encoder_function = transformer_encoder
        self._decoder_function = transformer_decoder
        self._init_cache_fn = _init_transformer_cache
        self._prepare_encoder_fn = transformer_prepare_encoder
        self._prepare_decoder_fn = transformer_prepare_decoder

    def encode(
        self, inputs, target_space, hparams, features = None, losses = None
    ):
        """Encode transformer inputs, see transformer_encode."""
        return transformer_encode(
            self._encoder_function,
            inputs,
            target_space,
            hparams,
            attention_weights = self.attention_weights,
            features = features,
            losses = losses,
            prepare_encoder_fn = self._prepare_encoder_fn,
        )

    def decode(
        self,
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        cache = None,
        decode_loop_step = None,
        nonpadding = None,
        losses = None,
        **kwargs
    ):
        """Decode Transformer outputs, see transformer_decode."""
        return transformer_decode(
            self._decoder_function,
            decoder_input,
            encoder_output,
            encoder_decoder_attention_bias,
            decoder_self_attention_bias,
            hparams,
            attention_weights = self.attention_weights,
            cache = cache,
            decode_loop_step = decode_loop_step,
            nonpadding = nonpadding,
            losses = losses,
            **kwargs
        )

    def body(self, features):
        """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs. [batch_size, input_length, 1,
            hidden_dim].
          "targets": Target decoder outputs. [batch_size, decoder_length, 1,
            hidden_dim]
          "target_space_id": A scalar int from data_generators.problem.SpaceID.

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
        hparams = self._hparams

        losses = []

        if self.has_input:
            inputs = self._prepare_inputs_for_body(features)
            target_space = features['target_space_id']
            encoder_output, encoder_decoder_attention_bias = self.encode(
                inputs,
                target_space,
                hparams,
                features = features,
                losses = losses,
            )
        else:
            encoder_output, encoder_decoder_attention_bias = (None, None)

        targets = features['targets']
        targets_shape = common_layers.shape_list(targets)
        targets = common_layers.flatten4d3d(targets)
        decoder_input, decoder_self_attention_bias = self._prepare_decoder_fn(
            targets, hparams, features = features
        )

        target_modality = self._problem_hparams.modality['targets']
        target_vocab_size = self._problem_hparams.vocab_size['targets']

        if target_vocab_size is not None and hasattr(hparams, 'vocab_divisor'):
            target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor

        modality_name = hparams.name.get(
            'targets', modalities.get_name(target_modality)
        )(hparams, target_vocab_size)
        with tf.variable_scope(modality_name):
            top = hparams.top.get(
                'targets', modalities.get_top(target_modality)
            )
            encoder_output_ = top(
                encoder_output, None, hparams, target_vocab_size
            )

        # Not all subclasses of Transformer support keyword arguments related to
        # recurrent memory, so only pass these arguments if memory is enabled.
        decode_kwargs = {}
        if self.recurrent_memory_by_layer is not None:
            # TODO(kitaev): The chunk_number feature currently has the same shape as
            # "targets", but this is only for the purposes of sharing sharding code.
            # In fact every token within an example must have the same chunk number.
            chunk_number_each_token = tf.squeeze(
                features['chunk_number'], (-1, -2)
            )
            chunk_number_each_example = chunk_number_each_token[:, 0]
            # Uncomment the code below to verify that tokens within a batch share the
            # same chunk number:
            # with tf.control_dependencies([
            #     tf.assert_equal(chunk_number_each_token,
            #                     chunk_number_each_example[:, None])
            # ]):
            #   chunk_number_each_example = tf.identity(chunk_number_each_example)
            decode_kwargs = dict(
                recurrent_memory_by_layer = self.recurrent_memory_by_layer,
                chunk_number = chunk_number_each_example,
            )
        decoder_output = self.decode(
            decoder_input,
            encoder_output,
            encoder_decoder_attention_bias,
            decoder_self_attention_bias,
            hparams,
            nonpadding = features_to_nonpadding(features, 'targets'),
            losses = losses,
            **decode_kwargs
        )
        expected_attentions = features.get('expected_attentions')
        if expected_attentions is not None:
            attention_loss = common_attention.encoder_decoder_attention_loss(
                expected_attentions,
                self.attention_weights,
                hparams.expected_attention_loss_type,
                hparams.expected_attention_loss_multiplier,
            )
            return decoder_output, {'attention_loss': attention_loss}

        ret = tf.reshape(decoder_output, targets_shape)
        if losses:
            return ret, {'extra_loss': tf.add_n(losses)}
        else:
            return ret

    def _prepare_inputs_for_body(self, features):
        """Prepare inputs for body.

    Args:
      features: Map of string to model features. Should contain
          "inputs": Transformer inputs. [batch_size, input_length, 1,
            hidden_dim].

    Returns:
      Inputs which will be passed to the model. [batch_size, input_length, 1,
          hidden_dim]
    """
        return features['inputs']

    def _greedy_infer(self, features, decode_length, use_tpu = False):
        """Fast version of greedy decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      use_tpu: A bool. Whether to build the inference graph for TPU.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
        # For real-valued modalities use the slow decode path for now.
        if (
            self._target_modality_is_real
            or self._hparams.self_attention_type != 'dot_product'
        ):
            return super(Transformer, self)._greedy_infer(
                features, decode_length
            )
        with tf.variable_scope(self.name):
            if use_tpu:
                return self._fast_decode_tpu(features, decode_length)
            return self._fast_decode(features, decode_length)

    def _beam_decode(
        self,
        features,
        decode_length,
        beam_size,
        top_beams,
        alpha,
        use_tpu = False,
    ):
        """Beam search decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: A bool, whether to do beam decode on TPU.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
    """
        if self._hparams.self_attention_type not in [
            'dot_product',
            'dot_product_relative',
        ]:
            # Caching is not guaranteed to work with attention types other than
            # dot_product and dot_product_relative.
            return self._beam_decode_slow(
                features, decode_length, beam_size, top_beams, alpha, use_tpu
            )
        with tf.variable_scope(self.name):
            if use_tpu:
                return self._fast_decode_tpu(
                    features, decode_length, beam_size, top_beams, alpha
                )
            return self._fast_decode(
                features, decode_length, beam_size, top_beams, alpha
            )

    def _prepare_inputs_for_decode(self, features):
        """Prepare inputs for decoding.

    Args:
      features: A map of string to model features.

    Returns:
      Inputs after fixing shape and applying modality.
    """
        dp = self._data_parallelism
        hparams = self._hparams
        inputs = features['inputs']
        # TODO(llion): Clean up this reshaping logic.
        inputs = tf.expand_dims(inputs, axis = 1)
        if len(inputs.shape) < 5:
            inputs = tf.expand_dims(inputs, axis = 4)
        s = common_layers.shape_list(inputs)
        inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
        # _shard_features called to ensure that the variable names match
        inputs = self._shard_features({'inputs': inputs})['inputs']
        input_modality = self._problem_hparams.modality['inputs']
        input_vocab_size = self._problem_hparams.vocab_size['inputs']
        if input_vocab_size is not None and hasattr(hparams, 'vocab_divisor'):
            input_vocab_size += (-input_vocab_size) % hparams.vocab_divisor
        modality_name = hparams.name.get(
            'inputs', modalities.get_name(input_modality)
        )(hparams, input_vocab_size)
        with tf.variable_scope(modality_name):
            bottom = hparams.bottom.get(
                'inputs', modalities.get_bottom(input_modality)
            )
            inputs = dp(bottom, inputs, hparams, input_vocab_size)
        return inputs

    def _fast_decode_tpu(
        self, features, decode_length, beam_size = 1, top_beams = 1, alpha = 1.0
    ):
        """Fast decoding.

    Implements both greedy and beam search decoding on TPU, uses beam search
    iff beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: A map of string to model features.
      decode_length: An integer, how many additional timesteps to decode.
      beam_size: An integer, number of beams.
      top_beams: An integer, how many of the beams to return.
      alpha: A float that controls the length penalty. Larger the alpha,
        stronger the preference for longer translations.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }.

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
        if self._num_datashards != 1:
            raise NotImplementedError(
                'Fast decoding only supports a single shard.'
            )
        if 'targets_segmentation' in features:
            raise NotImplementedError(
                'Decoding not supported on packed datasets '
                ' If you want to decode from a dataset, use the non-packed version'
                ' of the dataset when decoding.'
            )
        dp = self._data_parallelism
        hparams = self._hparams
        target_modality = self._problem_hparams.modality['targets']
        target_vocab_size = self._problem_hparams.vocab_size['targets']
        if target_vocab_size is not None and hasattr(hparams, 'vocab_divisor'):
            target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor

        if self.has_input:
            inputs_shape = common_layers.shape_list(features['inputs'])
            if (
                target_modality == modalities.ModalityType.CLASS_LABEL
                or self._problem_hparams.get('regression_targets')
            ):
                decode_length = 1
            else:
                decode_length = inputs_shape[1] + features.get(
                    'decode_length', decode_length
                )
            batch_size = inputs_shape[0]
            inputs = self._prepare_inputs_for_decode(features)
            with tf.variable_scope('body'):
                encoder_output, encoder_decoder_attention_bias = dp(
                    self.encode,
                    inputs,
                    features['target_space_id'],
                    hparams,
                    features = features,
                )
            encoder_output = encoder_output[0]
            encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
            partial_targets = None
        else:
            # The problem has no inputs.
            encoder_output = None
            encoder_decoder_attention_bias = None

            # Prepare partial targets.
            # In either features["inputs"] or features["targets"].
            # We force the outputs to begin with these sequences.
            partial_targets = features.get('inputs')
            if partial_targets is None:
                partial_targets = features['targets']
            assert partial_targets is not None
            partial_targets = common_layers.expand_squeeze_to_nd(
                partial_targets, 2
            )
            partial_targets = tf.to_int64(partial_targets)
            partial_targets_shape = common_layers.shape_list(partial_targets)
            partial_targets_length = partial_targets_shape[1]
            decode_length = partial_targets_length + features.get(
                'decode_length', decode_length
            )
            batch_size = partial_targets_shape[0]

        if hparams.pos == 'timing':
            positional_encoding = common_attention.get_timing_signal_1d(
                decode_length + 1, hparams.hidden_size
            )
        elif hparams.pos == 'timing_from_features':
            positional_encoding = common_attention.add_timing_signals_from_features(
                tf.zeros([1, decode_length + 1, hparams.hidden_size]),
                features,
                hparams.position_features,
            )
        elif hparams.pos == 'emb':
            positional_encoding = common_attention.add_positional_embedding(
                tf.zeros([1, decode_length + 1, hparams.hidden_size]),
                hparams.max_length,
                'body/targets_positional_embedding',
                None,
            )
        else:
            positional_encoding = None

        def preprocess_targets(targets, i):
            """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: A tensor, inputs ids to the decoder. [batch_size, 1].
        i: An integer, Step number of the decoding loop.

      Returns:
        A tensor, processed targets [batch_size, 1, hidden_dim].
      """
            # _shard_features called to ensure that the variable names match
            targets = self._shard_features({'targets': targets})['targets']
            modality_name = hparams.name.get(
                'targets', modalities.get_name(target_modality)
            )(hparams, target_vocab_size)
            with tf.variable_scope(modality_name):
                bottom = hparams.bottom.get(
                    'targets', modalities.get_targets_bottom(target_modality)
                )
                targets = dp(bottom, targets, hparams, target_vocab_size)[0]
            targets = common_layers.flatten4d3d(targets)

            # GO embeddings are all zero, this is because transformer_prepare_decoder
            # Shifts the targets along by one for the input which pads with zeros.
            # If the modality already maps GO to the zero embeddings this is not
            # needed.
            targets = tf.cond(
                tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets
            )

            if positional_encoding is not None:
                positional_encoding_shape = positional_encoding.shape.as_list()
                targets += tf.slice(
                    positional_encoding,
                    [0, i, 0],
                    [
                        positional_encoding_shape[0],
                        1,
                        positional_encoding_shape[2],
                    ],
                )
            return targets

        decoder_self_attention_bias = common_attention.attention_bias_lower_triangle(
            decode_length
        )
        if hparams.proximity_bias:
            decoder_self_attention_bias += common_attention.attention_bias_proximal(
                decode_length
            )

        def symbols_to_logits_tpu_fn(ids, i, cache):
            """Go from ids to logits for next symbol on TPU.

      Args:
        ids: A tensor, symbol IDs.
        i: An integer, step number of the decoding loop. Only used for inference
          on TPU.
        cache: A dict, containing tensors which are the results of previous
          attentions, used for fast decoding.

      Returns:
        ret: A tensor, computed logits.
        cache: A dict, containing tensors which are the results of previous
            attentions, used for fast decoding.
      """
            ids = ids[:, -1:]
            targets = tf.expand_dims(tf.expand_dims(ids, axis = 2), axis = 3)
            targets = preprocess_targets(targets, i)

            bias_shape = decoder_self_attention_bias.shape.as_list()
            bias = tf.slice(
                decoder_self_attention_bias,
                [0, 0, i, 0],
                [bias_shape[0], bias_shape[1], 1, bias_shape[3]],
            )

            with tf.variable_scope('body'):
                body_outputs = dp(
                    self.decode,
                    targets,
                    cache.get('encoder_output'),
                    cache.get('encoder_decoder_attention_bias'),
                    bias,
                    hparams,
                    cache,
                    i,
                    nonpadding = features_to_nonpadding(features, 'targets'),
                )
            modality_name = hparams.name.get(
                'targets', modalities.get_name(target_modality)
            )(hparams, target_vocab_size)
            with tf.variable_scope(modality_name):
                top = hparams.top.get(
                    'targets', modalities.get_top(target_modality)
                )
                logits = dp(
                    top, body_outputs, None, hparams, target_vocab_size
                )[0]

            ret = tf.squeeze(logits, axis = [1, 2, 3])
            if partial_targets is not None:
                # If the position is within the given partial targets, we alter the
                # logits to always return those values.
                # A faster approach would be to process the partial targets in one
                # iteration in order to fill the corresponding parts of the cache.
                # This would require broader changes, though.
                vocab_size = tf.shape(ret)[1]

                def forced_logits():
                    return tf.one_hot(
                        tf.tile(
                            tf.slice(
                                partial_targets,
                                [0, i],
                                [partial_targets.shape.as_list()[0], 1],
                            ),
                            [beam_size],
                        ),
                        vocab_size,
                        0.0,
                        -1e9,
                    )

                ret = tf.cond(
                    tf.less(i, partial_targets_length),
                    forced_logits,
                    lambda: ret,
                )
            return ret, cache

        eos_id = self.get_decode_end_id() or beam_search.EOS_ID
        temperature = features.get(
            'sampling_temp', getattr(hparams, 'sampling_temp', 0.0)
        )
        top_k = features.get(
            'sampling_keep_top_k', getattr(hparams, 'sampling_keep_top_k', -1)
        )

        ret = fast_decode_tpu(
            encoder_output = encoder_output,
            encoder_decoder_attention_bias = encoder_decoder_attention_bias,
            symbols_to_logits_fn = symbols_to_logits_tpu_fn,
            hparams = hparams,
            decode_length = decode_length,
            vocab_size = target_vocab_size,
            init_cache_fn = self._init_cache_fn,
            beam_size = beam_size,
            top_beams = top_beams,
            alpha = alpha,
            batch_size = batch_size,
            force_decode_length = self._decode_hparams.force_decode_length,
            eos_id = eos_id,
            sampling_temperature = temperature,
            top_k = top_k,
        )
        if partial_targets is not None:
            if beam_size <= 1 or top_beams <= 1:
                ret['outputs'] = ret['outputs'][:, partial_targets_length:]
            else:
                ret['outputs'] = ret['outputs'][:, :, partial_targets_length:]
        return ret

    def get_decode_start_id(self):
        """Returns the id of the first decoder input symbol.

    The default case maps None to a vector of 0's for transformer. This method
    can be overridden to return a different id by a model wanting to use a
    different decoder start symbol. The id returned by this method is used to
    index the embedding matrix, and retrieve the vector that will be used as the
    first input to the decoder
    """
        return None

    def get_decode_end_id(self):
        """Returns the id of the output symbol that terminates decoding.

    This method can be overridden by a different model. The id returned by this
    method is used to check if the generation is complete during decoding.
    """
        return None

    def _fast_decode(
        self,
        features,
        decode_length,
        beam_size = 1,
        top_beams = 1,
        alpha = 1.0,
        preprocess_targets_method = None,
    ):
        """Fast decoding.

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      preprocess_targets_method: method used to preprocess targets. If None,
      uses method "preprocess_targets" defined inside this method.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
        if self._num_datashards != 1:
            raise NotImplementedError(
                'Fast decoding only supports a single shard.'
            )
        dp = self._data_parallelism
        hparams = self._hparams
        target_modality = self._problem_hparams.modality['targets']
        target_vocab_size = self._problem_hparams.vocab_size['targets']
        if target_vocab_size is not None and hasattr(hparams, 'vocab_divisor'):
            target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor
        if 'targets_segmentation' in features:
            raise NotImplementedError(
                'Decoding not supported on packed datasets '
                ' If you want to decode from a dataset, use the non-packed version'
                ' of the dataset when decoding.'
            )
        if self.has_input:
            inputs_shape = common_layers.shape_list(features['inputs'])
            if (
                target_modality == modalities.ModalityType.CLASS_LABEL
                or self._problem_hparams.get('regression_targets')
            ):
                decode_length = 1
            else:
                decode_length = inputs_shape[1] + features.get(
                    'decode_length', decode_length
                )
            batch_size = inputs_shape[0]
            inputs = self._prepare_inputs_for_decode(features)
            with tf.variable_scope('body'):
                encoder_output, encoder_decoder_attention_bias = dp(
                    self.encode,
                    inputs,
                    features['target_space_id'],
                    hparams,
                    features = features,
                )
            encoder_output = encoder_output[0]
            encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
            partial_targets = features.get('partial_targets')
        else:
            # The problem has no inputs.
            encoder_output = None
            encoder_decoder_attention_bias = None

            # Prepare partial targets.
            # In either features["inputs"] or features["targets"].
            # We force the outputs to begin with these sequences.
            partial_targets = features.get('inputs')
            if partial_targets is None:
                partial_targets = features['targets']
            assert partial_targets is not None

        if partial_targets is not None:
            partial_targets = common_layers.expand_squeeze_to_nd(
                partial_targets, 2
            )
            partial_targets = tf.to_int64(partial_targets)
            partial_targets_shape = common_layers.shape_list(partial_targets)
            partial_targets_length = partial_targets_shape[1]
            decode_length = partial_targets_length + features.get(
                'decode_length', decode_length
            )
            batch_size = partial_targets_shape[0]

        if hparams.pos == 'timing':
            positional_encoding = common_attention.get_timing_signal_1d(
                decode_length + 1, hparams.hidden_size
            )
        elif hparams.pos == 'timing_from_features':
            positional_encoding = common_attention.add_timing_signals_from_features(
                tf.zeros([1, decode_length, hparams.hidden_size]),
                features,
                hparams.position_features,
            )
        elif hparams.pos == 'emb':
            positional_encoding = common_attention.add_positional_embedding(
                tf.zeros([1, decode_length, hparams.hidden_size]),
                hparams.max_length,
                'body/targets_positional_embedding',
                None,
            )
        else:
            positional_encoding = None

        def preprocess_targets(targets, i):
            """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.

      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """
            # _shard_features called to ensure that the variable names match
            targets = self._shard_features({'targets': targets})['targets']
            modality_name = hparams.name.get(
                'targets', modalities.get_name(target_modality)
            )(hparams, target_vocab_size)
            with tf.variable_scope(modality_name):
                bottom = hparams.bottom.get(
                    'targets', modalities.get_targets_bottom(target_modality)
                )
                targets = dp(bottom, targets, hparams, target_vocab_size)[0]
            targets = common_layers.flatten4d3d(targets)

            # GO embeddings are all zero, this is because transformer_prepare_decoder
            # Shifts the targets along by one for the input which pads with zeros.
            # If the modality already maps GO to the zero embeddings this is not
            # needed.
            if not self.get_decode_start_id():
                targets = tf.cond(
                    tf.equal(i, 0),
                    lambda: tf.zeros_like(targets),
                    lambda: targets,
                )

            if positional_encoding is not None:
                targets += positional_encoding[:, i : i + 1]
            return targets

        decoder_self_attention_bias = common_attention.attention_bias_lower_triangle(
            decode_length
        )
        if hparams.proximity_bias:
            decoder_self_attention_bias += common_attention.attention_bias_proximal(
                decode_length
            )

        # Create tensors for encoder-decoder attention history
        att_cache = {'attention_history': {}}
        num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
        if encoder_output is not None:
            att_batch_size, enc_seq_length = common_layers.shape_list(
                encoder_output
            )[0:2]
            for layer in range(num_layers):
                att_cache['attention_history']['layer_%d' % layer] = tf.zeros(
                    [att_batch_size, hparams.num_heads, 0, enc_seq_length]
                )

        def update_decoder_attention_history(cache):
            """Save attention weights in cache, e.g., for vizualization."""
            for k in [
                x
                for x in self.attention_weights
                if 'decoder' in x and 'self' not in x and 'logits' not in x
            ]:
                idx = k.find('layer_')
                if idx < 0:
                    continue
                # Get layer number from the string name.
                layer_nbr = k[idx + 6 :]
                idx = 0
                while (
                    idx + 1 < len(layer_nbr) and layer_nbr[: idx + 1].isdigit()
                ):
                    idx += 1
                layer_nbr = 'layer_%d' % int(layer_nbr[:idx])
                if layer_nbr in cache['attention_history']:
                    cache['attention_history'][layer_nbr] = tf.concat(
                        [
                            cache['attention_history'][layer_nbr],
                            self.attention_weights[k],
                        ],
                        axis = 2,
                    )

        if not preprocess_targets_method:
            preprocess_targets_method = preprocess_targets

        def symbols_to_logits_fn(ids, i, cache):
            """Go from ids to logits for next symbol."""
            ids = ids[:, -1:]
            targets = tf.expand_dims(tf.expand_dims(ids, axis = 2), axis = 3)
            targets = preprocess_targets_method(targets, i)

            bias = decoder_self_attention_bias[:, :, i : i + 1, : i + 1]
            with tf.variable_scope('body'):
                body_outputs = dp(
                    self.decode,
                    targets,
                    cache.get('encoder_output'),
                    cache.get('encoder_decoder_attention_bias'),
                    bias,
                    hparams,
                    cache,
                    nonpadding = features_to_nonpadding(features, 'targets'),
                )

            update_decoder_attention_history(cache)

            modality_name = hparams.name.get(
                'targets', modalities.get_name(target_modality)
            )(hparams, target_vocab_size)
            with tf.variable_scope(modality_name):
                top = hparams.top.get(
                    'targets', modalities.get_top(target_modality)
                )
                logits = dp(
                    top, body_outputs, None, hparams, target_vocab_size
                )[0]

            ret = tf.squeeze(logits, axis = [1, 2, 3])
            if partial_targets is not None:
                # If the position is within the given partial targets, we alter the
                # logits to always return those values.
                # A faster approach would be to process the partial targets in one
                # iteration in order to fill the corresponding parts of the cache.
                # This would require broader changes, though.
                vocab_size = tf.shape(ret)[1]

                def forced_logits():
                    return tf.one_hot(
                        tf.tile(partial_targets[:, i], [beam_size]),
                        vocab_size,
                        0.0,
                        -1e9,
                    )

                ret = tf.cond(
                    tf.less(i, partial_targets_length),
                    forced_logits,
                    lambda: ret,
                )
            return ret, cache

        sos_id = self.get_decode_start_id() or 0
        eos_id = self.get_decode_end_id() or beam_search.EOS_ID
        temperature = features.get(
            'sampling_temp', getattr(hparams, 'sampling_temp', 0.0)
        )
        top_k = features.get(
            'sampling_keep_top_k', getattr(hparams, 'sampling_keep_top_k', -1)
        )

        ret = fast_decode(
            encoder_output = encoder_output,
            encoder_decoder_attention_bias = encoder_decoder_attention_bias,
            symbols_to_logits_fn = symbols_to_logits_fn,
            hparams = hparams,
            decode_length = decode_length,
            vocab_size = target_vocab_size,
            init_cache_fn = self._init_cache_fn,
            beam_size = beam_size,
            top_beams = top_beams,
            alpha = alpha,
            batch_size = batch_size,
            force_decode_length = self._decode_hparams.force_decode_length,
            sos_id = sos_id,
            eos_id = eos_id,
            sampling_temperature = temperature,
            top_k = top_k,
            cache = att_cache,
        )
        if partial_targets is not None:
            if beam_size <= 1 or top_beams <= 1:
                ret['outputs'] = ret['outputs'][:, partial_targets_length:]
            else:
                ret['outputs'] = ret['outputs'][:, :, partial_targets_length:]
        return ret


def _init_transformer_cache(
    cache,
    hparams,
    batch_size,
    attention_init_length,
    encoder_output,
    encoder_decoder_attention_bias,
    scope_prefix,
):
    """Create the initial cache for Transformer fast decoding."""
    key_channels = hparams.attention_key_channels or hparams.hidden_size
    value_channels = hparams.attention_value_channels or hparams.hidden_size
    num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
    vars_3d_num_heads = (
        hparams.num_heads if hparams.get('attention_variables_3d') else 0
    )

    if cache is None:
        cache = {}
    cache.update(
        {
            'layer_%d'
            % layer: {  # pylint: disable=g-complex-comprehension
                'k': common_attention.split_heads(
                    tf.zeros([batch_size, attention_init_length, key_channels]),
                    hparams.num_heads,
                ),
                'v': common_attention.split_heads(
                    tf.zeros(
                        [batch_size, attention_init_length, value_channels]
                    ),
                    hparams.num_heads,
                ),
            }
            for layer in range(num_layers)
        }
    )

    # If `ffn_layer` is in `["dense_relu_dense" or "conv_hidden_relu"]`, then the
    # cache key "f" won't be used, which means that the` shape of cache["f"]`
    # won't be changed to
    # `[beamsize*batch_size, decode_length, hparams.hidden_size]` and may cause
    # error when applying `nest.map reshape function` on it.
    if hparams.ffn_layer not in ['dense_relu_dense', 'conv_hidden_relu']:
        for layer in range(num_layers):
            cache['layer_%d' % layer]['f'] = tf.zeros(
                [batch_size, 0, hparams.hidden_size]
            )

    if encoder_output is not None:
        for layer in range(num_layers):
            layer_name = 'layer_%d' % layer
            with tf.variable_scope(
                '%sdecoder/%s/encdec_attention/multihead_attention'
                % (scope_prefix, layer_name)
            ):
                k_encdec = common_attention.compute_attention_component(
                    encoder_output,
                    key_channels,
                    name = 'k',
                    vars_3d_num_heads = vars_3d_num_heads,
                )
                k_encdec = common_attention.split_heads(
                    k_encdec, hparams.num_heads
                )
                v_encdec = common_attention.compute_attention_component(
                    encoder_output,
                    value_channels,
                    name = 'v',
                    vars_3d_num_heads = vars_3d_num_heads,
                )
                v_encdec = common_attention.split_heads(
                    v_encdec, hparams.num_heads
                )
            cache[layer_name]['k_encdec'] = k_encdec
            cache[layer_name]['v_encdec'] = v_encdec

        cache['encoder_output'] = encoder_output
        cache['encoder_decoder_attention_bias'] = encoder_decoder_attention_bias
    return cache


def fast_decode_tpu(
    encoder_output,
    encoder_decoder_attention_bias,
    symbols_to_logits_fn,
    hparams,
    decode_length,
    vocab_size,
    init_cache_fn = _init_transformer_cache,
    beam_size = 1,
    top_beams = 1,
    alpha = 1.0,
    sos_id = 0,
    eos_id = beam_search.EOS_ID,
    batch_size = None,
    force_decode_length = False,
    scope_prefix = 'body/',
    use_top_k_with_unique = True,
    sampling_temperature = 0.0,
    top_k = -1,
):
    """Given encoder output and a symbols to logits function, does fast decoding.

  Implements both greedy and beam search decoding for TPU, uses beam search iff
  beam_size > 1, otherwise beam search related arguments are ignored.

  Args:
    encoder_output: A tensor, output from encoder.
    encoder_decoder_attention_bias: A tensor, bias for use in encoder-decoder
      attention.
    symbols_to_logits_fn: Incremental decoding, function mapping triple `(ids,
      step, cache)` to symbol logits.
    hparams: Run hyperparameters.
    decode_length: An integer, how many additional timesteps to decode.
    vocab_size: Output vocabulary size.
    init_cache_fn: Function that returns the initial cache dict.
    beam_size: An integer, number of beams.
    top_beams: An integer, how many of the beams to return.
    alpha: A float that controls the length penalty. Larger the alpha, stronger
      the preference for longer translations.
    sos_id: Start-of-sequence symbol.
    eos_id: End-of-sequence symbol.
    batch_size: An integer, must be passed if there is no input.
    force_decode_length: A bool, whether to force the full decode length, or if
      False, stop when all beams hit eos_id.
    scope_prefix: str, prefix for decoder layer variable scopes.
    use_top_k_with_unique: bool, whether to use a fast (but decreased precision)
      top_k during beam search.
    sampling_temperature: scalar, temperature with which to sample.
    top_k: scalar, sample only top k.

  Returns:
    A dict of decoding results {
        "outputs": integer `Tensor` of decoded ids of shape
            [batch_size, <= decode_length] if top_beams == 1 or
            [batch_size, top_beams, <= decode_length] otherwise
        "scores": decoding log probs from the beam search,
            None if using greedy decoding (beam_size=1)
    }.

  Raises:
    NotImplementedError: If beam size > 1 with partial targets.
  """
    if encoder_output is not None:
        batch_size = common_layers.shape_list(encoder_output)[0]

    cache = init_cache_fn(
        None,
        hparams,
        batch_size,
        decode_length,
        encoder_output,
        encoder_decoder_attention_bias,
        scope_prefix,
    )

    mlperf_log.transformer_print(
        key = mlperf_log.MODEL_HP_SEQ_BEAM_SEARCH,
        value = {
            'vocab_size': vocab_size,
            'batch_size': batch_size,
            'beam_size': beam_size,
            'alpha': alpha,
            'max_decode_length': decode_length,
        },
        hparams = hparams,
    )
    if beam_size > 1:  # Beam Search
        initial_ids = sos_id * tf.ones([batch_size], dtype = tf.int32)
        decoded_ids, scores, _ = beam_search.beam_search(
            symbols_to_logits_fn,
            initial_ids,
            beam_size,
            decode_length,
            vocab_size,
            alpha,
            states = cache,
            eos_id = eos_id,
            stop_early = (top_beams == 1),
            use_tpu = True,
            use_top_k_with_unique = use_top_k_with_unique,
        )

        if top_beams == 1:
            decoded_ids = decoded_ids[:, 0, 1:]
            scores = scores[:, 0]
        else:
            decoded_ids = decoded_ids[:, :top_beams, 1:]
            scores = scores[:, :top_beams]
    else:  # Greedy

        def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
            """One step of greedy decoding."""
            logits, cache = symbols_to_logits_fn(next_id, i, cache)
            log_probs = common_layers.log_prob_from_logits(logits)
            temperature = sampling_temperature
            if hparams.sampling_method == 'random_per_example':
                next_id = common_layers.sample_temperature_per_example(
                    logits, temperature, top_k
                )
            else:
                if hparams.sampling_method == 'argmax':
                    temperature = 0.0
                next_id = common_layers.sample_with_temperature(
                    logits, temperature, top_k
                )

            log_prob_indices = tf.stack(
                [tf.range(tf.to_int64(batch_size)), next_id], axis = 1
            )
            log_prob += tf.gather_nd(log_probs, log_prob_indices) * (
                1 - tf.to_float(hit_eos)
            )
            # Note(thangluong): we purposely update hit_eos after aggregating log_prob
            # There is a subtle detail here that we want to include log_probs up to
            # (and inclusive of) the first eos generated, but not subsequent tokens.
            hit_eos |= tf.equal(next_id, eos_id)

            next_id = tf.expand_dims(next_id, axis = 1)
            decoded_ids = tf.transpose(decoded_ids)
            decoded_ids = inplace_ops.alias_inplace_update(
                decoded_ids, i, tf.squeeze(next_id, axis = 1)
            )
            decoded_ids = tf.transpose(decoded_ids)
            return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob

        def is_not_finished(i, hit_eos, *_):
            finished = i >= decode_length
            if not force_decode_length:
                finished |= tf.reduce_all(hit_eos)
            return tf.logical_not(finished)

        decoded_ids = tf.zeros([batch_size, decode_length], dtype = tf.int64)
        hit_eos = tf.fill([batch_size], False)
        next_id = sos_id * tf.ones([batch_size, 1], dtype = tf.int64)
        initial_log_prob = tf.zeros([batch_size], dtype = tf.float32)

        def compute_cache_shape_invariants(tensor):
            return tf.TensorShape(tensor.shape.as_list())

        _, _, _, decoded_ids, _, log_prob = tf.while_loop(
            is_not_finished,
            inner_loop,
            [
                tf.constant(0),
                hit_eos,
                next_id,
                decoded_ids,
                cache,
                initial_log_prob,
            ],
            shape_invariants = [
                tf.TensorShape([]),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, 1]),
                tf.TensorShape([batch_size, decode_length]),
                nest.map_structure(compute_cache_shape_invariants, cache),
                tf.TensorShape([batch_size]),
            ],
        )
        scores = log_prob

    return {'outputs': decoded_ids, 'scores': scores}


def fast_decode(
    encoder_output,
    encoder_decoder_attention_bias,
    symbols_to_logits_fn,
    hparams,
    decode_length,
    vocab_size,
    init_cache_fn = _init_transformer_cache,
    beam_size = 1,
    top_beams = 1,
    alpha = 1.0,
    sos_id = 0,
    eos_id = beam_search.EOS_ID,
    batch_size = None,
    force_decode_length = False,
    scope_prefix = 'body/',
    sampling_temperature = 0.0,
    top_k = -1,
    cache = None,
):
    """Given encoder output and a symbols to logits function, does fast decoding.

  Implements both greedy and beam search decoding, uses beam search iff
  beam_size > 1, otherwise beam search related arguments are ignored.

  Args:
    encoder_output: Output from encoder.
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
    symbols_to_logits_fn: Incremental decoding; function mapping triple `(ids,
      step, cache)` to symbol logits.
    hparams: run hyperparameters
    decode_length: an integer.  How many additional timesteps to decode.
    vocab_size: Output vocabulary size.
    init_cache_fn: Function that returns the initial cache dict.
    beam_size: number of beams.
    top_beams: an integer. How many of the beams to return.
    alpha: Float that controls the length penalty. larger the alpha, stronger
      the preference for longer translations.
    sos_id: End-of-sequence symbol in beam search.
    eos_id: End-of-sequence symbol in beam search.
    batch_size: an integer scalar - must be passed if there is no input
    force_decode_length: bool, whether to force the full decode length, or if
      False, stop when all beams hit eos_id.
    scope_prefix: str, prefix for decoder layer variable scopes.
    sampling_temperature: scalar, temperature with which to sample.
    top_k: scalar, sample only top k.
    cache: cache dictionary for additional predictions.

  Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if top_beams == 1 or
              [batch_size, top_beams, <= decode_length] otherwise
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
  """
    if encoder_output is not None:
        batch_size = common_layers.shape_list(encoder_output)[0]

    cache = init_cache_fn(
        cache = cache,
        hparams = hparams,
        batch_size = batch_size,
        attention_init_length = 0,
        encoder_output = encoder_output,
        encoder_decoder_attention_bias = encoder_decoder_attention_bias,
        scope_prefix = scope_prefix,
    )

    if beam_size > 1:  # Beam Search
        initial_ids = sos_id * tf.ones([batch_size], dtype = tf.int32)
        decoded_ids, scores, cache = beam_search.beam_search(
            symbols_to_logits_fn,
            initial_ids,
            beam_size,
            decode_length,
            vocab_size,
            alpha,
            states = cache,
            eos_id = eos_id,
            stop_early = (top_beams == 1),
        )

        if top_beams == 1:
            decoded_ids = decoded_ids[:, 0, 1:]
            scores = scores[:, 0]
        else:
            decoded_ids = decoded_ids[:, :top_beams, 1:]
            scores = scores[:, :top_beams]
    else:  # Greedy

        def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
            """One step of greedy decoding."""
            logits, cache = symbols_to_logits_fn(next_id, i, cache)
            log_probs = common_layers.log_prob_from_logits(logits)
            temperature = sampling_temperature
            if hparams.sampling_method == 'random_per_example':
                next_id = common_layers.sample_temperature_per_example(
                    logits, temperature, top_k
                )
            else:
                if hparams.sampling_method == 'argmax':
                    temperature = 0.0
                next_id = common_layers.sample_with_temperature(
                    logits, temperature, top_k
                )

            log_prob_indices = tf.stack(
                [tf.range(tf.to_int64(batch_size)), next_id], axis = 1
            )
            log_prob += tf.gather_nd(log_probs, log_prob_indices) * (
                1 - tf.to_float(hit_eos)
            )
            # Note(thangluong): we purposely update hit_eos after aggregating log_prob
            # There is a subtle detail here that we want to include log_probs up to
            # (and inclusive of) the first eos generated, but not subsequent tokens.
            hit_eos |= tf.equal(next_id, eos_id)

            next_id = tf.expand_dims(next_id, axis = 1)
            decoded_ids = tf.concat([decoded_ids, next_id], axis = 1)

            return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob

        def is_not_finished(i, hit_eos, *_):
            finished = i >= decode_length
            if not force_decode_length:
                finished |= tf.reduce_all(hit_eos)
            return tf.logical_not(finished)

        decoded_ids = tf.zeros([batch_size, 0], dtype = tf.int64)
        hit_eos = tf.fill([batch_size], False)
        next_id = sos_id * tf.ones([batch_size, 1], dtype = tf.int64)
        initial_log_prob = tf.zeros([batch_size], dtype = tf.float32)
        _, _, _, decoded_ids, cache, log_prob = tf.while_loop(
            is_not_finished,
            inner_loop,
            [
                tf.constant(0),
                hit_eos,
                next_id,
                decoded_ids,
                cache,
                initial_log_prob,
            ],
            shape_invariants = [
                tf.TensorShape([]),
                tf.TensorShape([None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                nest.map_structure(
                    beam_search.get_state_shape_invariants, cache
                ),
                tf.TensorShape([None]),
            ],
        )
        scores = log_prob

    return {'outputs': decoded_ids, 'scores': scores, 'cache': cache}


def features_to_nonpadding(features, inputs_or_targets = 'inputs'):
    key = inputs_or_targets + '_segmentation'
    if features and key in features:
        return tf.minimum(tf.to_float(features[key]), 1.0)
    return None


def transformer_prepare_decoder(targets, hparams, features = None, pad = None):
    """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well. This is
      needed now for "packed" datasets.
    pad: vector to use for padding when shifting targets right

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a bias tensor for use in decoder self-attention
  """
    if hparams.causal_decoder_self_attention:
        # Causal attention.
        if hparams.prepend_mode == 'prepend_inputs_full_attention':
            decoder_self_attention_bias = common_attention.attention_bias_prepend_inputs_full_attention(
                common_attention.embedding_to_padding(targets)
            )
        else:
            decoder_self_attention_bias = common_attention.attention_bias_lower_triangle(
                common_layers.shape_list(targets)[1]
            )
    else:
        # Full attention.
        decoder_padding = common_attention.embedding_to_padding(targets)
        decoder_self_attention_bias = common_attention.attention_bias_ignore_padding(
            decoder_padding
        )

    if features and 'targets_segmentation' in features:
        # "Packed" dataset - keep the examples from seeing each other.
        targets_segmentation = features['targets_segmentation']
        targets_position = features['targets_position']
        decoder_self_attention_bias += common_attention.attention_bias_same_segment(
            targets_segmentation, targets_segmentation
        )
    else:
        targets_position = None
    if hparams.proximity_bias:
        decoder_self_attention_bias += common_attention.attention_bias_proximal(
            common_layers.shape_list(targets)[1]
        )
    decoder_input = common_layers.shift_right_3d(targets, pad)
    if hparams.pos == 'timing':
        if targets_position is not None:
            decoder_input = common_attention.add_timing_signal_1d_given_position(
                decoder_input, targets_position
            )
        else:
            decoder_input = common_attention.add_timing_signal_1d(decoder_input)
    elif hparams.pos == 'timing_from_features':
        decoder_input = common_attention.add_timing_signals_from_features(
            decoder_input, features, hparams.position_features
        )
    elif hparams.pos == 'emb':
        decoder_input = common_attention.add_positional_embedding(
            decoder_input,
            hparams.max_length,
            'targets_positional_embedding',
            targets_position,
        )

    if hparams.activation_dtype == 'bfloat16':
        decoder_self_attention_bias = tf.cast(
            decoder_self_attention_bias, tf.bfloat16
        )
    return (decoder_input, decoder_self_attention_bias)


def transformer_self_attention_layer(
    decoder_input,
    decoder_self_attention_bias,
    layer_idx,
    hparams,
    encoder_output = None,
    encoder_decoder_attention_bias = None,
    cache = None,
    decode_loop_step = None,
    save_weights_to = None,
    make_image_summary = False,
    layer_collection = None,
    recurrent_memory_by_layer = None,
    chunk_number = None,
):
    """A single transformer self-attention layer."""
    x = decoder_input
    layer = layer_idx
    layer_name = 'layer_%d' % layer
    layer_cache = cache[layer_name] if cache is not None else None

    attention_dropout_broadcast_dims = common_layers.comma_separated_string_to_integer_list(
        getattr(hparams, 'attention_dropout_broadcast_dims', '')
    )

    if recurrent_memory_by_layer is not None:
        recurrent_memory = recurrent_memory_by_layer[layer_name]
    else:
        recurrent_memory = None

    if layer < hparams.get('num_area_layers', 0):
        max_area_width = hparams.get('max_area_width', 1)
        max_area_height = hparams.get('max_area_height', 1)
        memory_height = hparams.get('max_area_height', 1)
    else:
        max_area_width = 1
        max_area_height = 1
        memory_height = 1
    with tf.variable_scope(layer_name):
        with tf.variable_scope('self_attention'):
            y = common_attention.multihead_attention(
                common_layers.layer_preprocess(
                    x, hparams, layer_collection = layer_collection
                ),
                None,
                decoder_self_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                attention_type = hparams.self_attention_type,
                max_relative_position = hparams.max_relative_position,
                heads_share_relative_embedding = (
                    hparams.heads_share_relative_embedding
                ),
                add_relative_to_values = hparams.add_relative_to_values,
                save_weights_to = save_weights_to,
                cache = layer_cache,
                make_image_summary = make_image_summary,
                dropout_broadcast_dims = attention_dropout_broadcast_dims,
                max_length = hparams.get('max_length'),
                decode_loop_step = decode_loop_step,
                vars_3d = hparams.get('attention_variables_3d'),
                activation_dtype = hparams.get('activation_dtype', 'float32'),
                weight_dtype = hparams.get('weight_dtype', 'float32'),
                layer_collection = layer_collection,
                recurrent_memory = recurrent_memory,
                chunk_number = chunk_number,
                hard_attention_k = hparams.get('hard_attention_k', 0),
                gumbel_noise_weight = hparams.get('gumbel_noise_weight', 0.0),
                max_area_width = max_area_width,
                max_area_height = max_area_height,
                memory_height = memory_height,
                area_key_mode = hparams.get('area_key_mode', 'none'),
                area_value_mode = hparams.get('area_value_mode', 'none'),
                training = (
                    hparams.get('mode', tf.estimator.ModeKeys.TRAIN)
                    == tf.estimator.ModeKeys.TRAIN
                ),
            )
            x = common_layers.layer_postprocess(x, y, hparams)
        if encoder_output is not None:
            if not isinstance(encoder_output, (list,)):
                encoder_output = [encoder_output]
            with tf.variable_scope('encdec_attention'):
                for enc_output in encoder_output:
                    y = common_attention.multihead_attention(
                        common_layers.layer_preprocess(
                            x, hparams, layer_collection = layer_collection
                        ),
                        enc_output,
                        encoder_decoder_attention_bias,
                        hparams.attention_key_channels or hparams.hidden_size,
                        hparams.attention_value_channels or hparams.hidden_size,
                        hparams.hidden_size,
                        hparams.num_heads,
                        hparams.attention_dropout,
                        max_relative_position = hparams.max_relative_position,
                        heads_share_relative_embedding = (
                            hparams.heads_share_relative_embedding
                        ),
                        add_relative_to_values = hparams.add_relative_to_values,
                        save_weights_to = save_weights_to,
                        cache = layer_cache,
                        make_image_summary = make_image_summary,
                        dropout_broadcast_dims = attention_dropout_broadcast_dims,
                        max_length = hparams.get('max_length'),
                        vars_3d = hparams.get('attention_variables_3d'),
                        activation_dtype = hparams.get(
                            'activation_dtype', 'float32'
                        ),
                        weight_dtype = hparams.get('weight_dtype', 'float32'),
                        layer_collection = layer_collection,
                        hard_attention_k = hparams.get('hard_attention_k', 0),
                        gumbel_noise_weight = hparams.get(
                            'gumbel_noise_weight', 0.0
                        ),
                        max_area_width = max_area_width,
                        max_area_height = max_area_height,
                        memory_height = memory_height,
                        area_key_mode = hparams.get('area_key_mode', 'none'),
                        area_value_mode = hparams.get(
                            'area_value_mode', 'none'
                        ),
                        training = (
                            hparams.get('mode', tf.estimator.ModeKeys.TRAIN)
                            == tf.estimator.ModeKeys.TRAIN
                        ),
                    )
                    x = common_layers.layer_postprocess(x, y, hparams)
        return x, layer_cache


def transformer_decoder_layer(
    decoder_input,
    decoder_self_attention_bias,
    layer_idx,
    hparams,
    encoder_output = None,
    encoder_decoder_attention_bias = None,
    cache = None,
    decode_loop_step = None,
    nonpadding = None,
    save_weights_to = None,
    make_image_summary = False,
    losses = None,
    layer_collection = None,
    recurrent_memory_by_layer = None,
    chunk_number = None,
):
    """A single transformer decoder layer."""
    x, layer_cache = transformer_self_attention_layer(
        decoder_input = decoder_input,
        decoder_self_attention_bias = decoder_self_attention_bias,
        layer_idx = layer_idx,
        hparams = hparams,
        encoder_output = encoder_output,
        encoder_decoder_attention_bias = encoder_decoder_attention_bias,
        cache = cache,
        decode_loop_step = decode_loop_step,
        save_weights_to = save_weights_to,
        make_image_summary = make_image_summary,
        layer_collection = layer_collection,
        recurrent_memory_by_layer = recurrent_memory_by_layer,
        chunk_number = chunk_number,
    )

    layer = layer_idx
    layer_name = 'layer_%d' % layer
    with tf.variable_scope(layer_name):
        with tf.variable_scope('ffn'):
            y = transformer_ffn_layer(
                common_layers.layer_preprocess(
                    x, hparams, layer_collection = layer_collection
                ),
                hparams,
                conv_padding = 'LEFT',
                nonpadding_mask = nonpadding,
                losses = losses,
                cache = layer_cache,
                decode_loop_step = decode_loop_step,
                layer_collection = layer_collection,
            )
            x = common_layers.layer_postprocess(x, y, hparams)
            return x


def transformer_decoder(
    decoder_input,
    encoder_output,
    decoder_self_attention_bias,
    encoder_decoder_attention_bias,
    hparams,
    cache = None,
    decode_loop_step = None,
    name = 'decoder',
    nonpadding = None,
    save_weights_to = None,
    make_image_summary = True,
    losses = None,
    layer_collection = None,
    recurrent_memory_by_layer = None,
    chunk_number = None,
):
    """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention (see
      common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
      attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop. Only used
      for inference on TPU.
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used to mask out
      padding in convolutional layers.  We generally only need this mask for
      "packed" datasets, because for ordinary datasets, no padding is ever
      followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights for
      visualization; the weights tensor will be appended there under a string
      key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the KFAC
      optimizer. Default is None.
    recurrent_memory_by_layer: Optional dict, mapping layer names to instances
      of transformer_memory.RecurrentMemory. Default is None.
    chunk_number: an optional integer Tensor with shape [batch] used to operate
      the recurrent_memory.

  Returns:
    y: a Tensors
  """
    x = decoder_input

    mlperf_log.transformer_print(
        key = mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
        value = hparams.num_decoder_layers or hparams.num_hidden_layers,
        hparams = hparams,
    )
    mlperf_log.transformer_print(
        key = mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
        value = hparams.attention_dropout,
        hparams = hparams,
    )
    mlperf_log.transformer_print(
        key = mlperf_log.MODEL_HP_ATTENTION_DENSE,
        value = {
            'use_bias': 'false',
            'num_heads': hparams.num_heads,
            'hidden_size': hparams.hidden_size,
        },
        hparams = hparams,
    )

    with tf.variable_scope(name):
        for layer_idx in range(
            hparams.num_decoder_layers or hparams.num_hidden_layers
        ):
            x = transformer_decoder_layer(
                x,
                decoder_self_attention_bias,
                layer_idx,
                hparams,
                encoder_decoder_attention_bias = encoder_decoder_attention_bias,
                encoder_output = encoder_output,
                cache = cache,
                decode_loop_step = decode_loop_step,
                nonpadding = nonpadding,
                save_weights_to = save_weights_to,
                make_image_summary = make_image_summary,
                losses = losses,
                layer_collection = layer_collection,
                recurrent_memory_by_layer = recurrent_memory_by_layer,
                chunk_number = chunk_number,
            )

        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        mlperf_log.transformer_print(
            key = mlperf_log.MODEL_HP_NORM,
            value = {'hidden_size': hparams.hidden_size},
        )
        return common_layers.layer_preprocess(
            x, hparams, layer_collection = layer_collection
        )
