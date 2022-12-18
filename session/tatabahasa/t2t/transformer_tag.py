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

"""The TransformerTag model.

TransformerTag is an adaptation of the Transformer that predicts tags.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import transformer_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.layers import modalities
from tensor2tensor.utils import beam_search
from tensorflow.python.util import nest
import tensorflow.compat.v1 as tf


def maybe_flatten4d3d(x):
    xshape = common_layers.shape_list(x)
    return common_layers.flatten4d3d(x) if len(xshape) == 4 else x


def maybe_flatten3d2d(x):
    """Flatten if tensor has 3 dimensions, similar to maybe_flatten4d3d()."""
    xshape = common_layers.shape_list(x)
    if len(xshape) != 3:
        return x
    return tf.reshape(x, [xshape[0], xshape[1] * xshape[2]])


def maybe_flatten4d2d(x):
    return maybe_flatten3d2d(maybe_flatten4d3d(x))


def features_to_nonpadding(features, inputs_or_targets = 'inputs'):
    """See transformer.features_to_nonpadding."""
    key = inputs_or_targets + '_segmentation'
    if features and key in features:
        return tf.minimum(tf.to_float(features[key]), 1.0)
    return None


def gather_2d(params, indices):
    """2D version of tf.gather.

  This is a batched version of tf.gather(), i.e. it applies tf.gather() to
  each batch separately.
  Example:
    params = [[10, 11, 12, 13, 14],
              [20, 21, 22, 23, 24]]
    indices = [[0, 0, 1, 1, 1, 2],
               [1, 3, 0, 0, 2, 2]]
    result = [[10, 10, 11, 11, 11, 12],
              [21, 23, 20, 20, 22, 22]]
  This method is copied from
    https://github.com/fstahlberg/tensor2tensor-usr/blob/master/usr/utils.py
  which is published under Apache 2.

  Args:
    params: A [batch_size, n, ...] tensor with data
    indices: A [batch_size, num_indices] int32 tensor with indices into params.
      Entries must be smaller than n

  Returns:
    The result of tf.gather() on each entry of the batch.
  """
    batch_size = tf.shape(params)[0]
    num_indices = tf.shape(indices)[1]
    batch_indices = tf.tile(
        tf.expand_dims(tf.range(batch_size), 1), [1, num_indices]
    )
    # batch_indices is [[0,0,0,0,...],[1,1,1,1,...],...]
    gather_nd_indices = tf.stack([batch_indices, indices], axis = 2)
    return tf.gather_nd(params, gather_nd_indices)


@registry.register_model
class TransformerTag(t2t_model.T2TModel):
    """The Seq2Edits model. See file docstring."""

    def __init__(self, *args, **kwargs):
        super(TransformerTag, self).__init__(*args, **kwargs)
        self.attention_weights = {}  # For visualizing attention heads.
        self._encoder_function = transformer_layers.transformer_encoder
        self._decoder_function = transformer.transformer_decoder
        self._prepare_encoder_fn = (
            transformer_layers.transformer_prepare_encoder
        )
        self._prepare_decoder_fn = transformer.transformer_prepare_decoder
        self.loss_num = {}
        self.logits = {}
        self.loss_den = None

    def encode(
        self, inputs, target_space, hparams, features = None, losses = None
    ):
        """Encodes transformer inputs, see transformer.transformer_encode()."""
        return transformer.transformer_encode(
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
        """Decodes Transformer outputs, see transformer.transformer_decode()."""
        return transformer.transformer_decode(
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
        """Seq2Edits main model_fn.

    Args:
      features: Feature dictionary. Should contain the following fields:
          "inputs": [batch_size, input_length, 1, hidden_dim] float tensor with
            input token embeddings.
          "targets": [batch_size, target_length, 1, hidden_dim] float tensor
            with target token embeddings.
          "targets_error_tag": [batch_size, target_length, 1, hidden_dim] float
            tensor with target error tag embeddings.
          "target_space_id": A scalar int from data_generators.problem.SpaceID.

    Returns:
      Final decoder representation. Dictionary containing the following fields:
        "targets": [batch_size, target_length, hidden_dim] float tensor with
          decoder outputs
        "targets_error_tag": [batch_size, target_length, hidden_dim] float
          tensor with decoder outputs
    """
        hparams = self._hparams

        losses = []

        if self.has_input:
            target_space = features['target_space_id']
            encoder_output, encoder_decoder_attention_bias = self.encode(
                features['inputs'],
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

        nonpadding = features_to_nonpadding(features, 'targets')

        # Add edit ops layer to condition on start_token, end_token, and error_tag
        decoder_input = transformer_edit_ops_layer(
            decoder_input,
            hparams,
            encoder_output,
            features,
            nonpadding = nonpadding,
            losses = losses,
        )
        if hparams.middle_prediction:
            num_decoder_layers = (
                hparams.num_decoder_layers or hparams.num_hidden_layers
            )
            hparams.num_decoder_layers = int(
                num_decoder_layers / hparams.middle_prediction_layer_factor
            )

        decode_kwargs = {}
        decoder_output = self.decode(
            decoder_input,
            encoder_output,
            encoder_decoder_attention_bias,
            decoder_self_attention_bias,
            hparams,
            nonpadding = nonpadding,
            losses = losses,
            **decode_kwargs
        )

        loss_mask = common_layers.weights_nonzero(
            maybe_flatten4d2d(features['targets_raw'])
        )
        self.loss_den = tf.reduce_sum(loss_mask)
        decoder_output = self._prediction_cascade(
            hparams = hparams,
            features = features,
            losses = losses,
            loss_mask = loss_mask,
            nonpadding = nonpadding,
            encoder_decoder_attention_bias = encoder_decoder_attention_bias,
            encoder_output = encoder_output,
            decoder_output = decoder_output,
        )

        if hparams.middle_prediction:
            with tf.variable_scope('after_prediction'):
                decoder_output = self.decode(
                    decoder_input + decoder_output,
                    encoder_output,
                    encoder_decoder_attention_bias,
                    decoder_self_attention_bias,
                    hparams,
                    nonpadding = nonpadding,
                    losses = losses,
                    **decode_kwargs
                )

        ret = {'targets': tf.reshape(decoder_output, targets_shape)}
        ret.update(self.logits)
        if losses:
            return ret, {'extra_loss': tf.add_n(losses)}
        else:
            return ret

    def _prediction_cascade_predict(
        self,
        hparams,
        nonpadding,
        encoder_decoder_attention_bias,
        encoder_output,
        decoder_output,
    ):
        logits = {}
        if hparams.use_error_tags:
            (
                decoder_output,
                error_tag_logits,
            ) = transformer_error_tag_prediction_layer_predict(
                decoder_output, hparams
            )
            logits['targets_error_tag'] = error_tag_logits
            decoder_output = transformer_between_predictions_layer(
                decoder_output,
                hparams,
                name = 'post_error_tag',
                nonpadding = nonpadding,
            )

        return decoder_output, logits

    def _prediction_cascade(
        self,
        hparams,
        features,
        losses,
        loss_mask,
        nonpadding,
        encoder_decoder_attention_bias,
        encoder_output,
        decoder_output,
    ):
        if hparams.use_error_tags:
            (
                decoder_output,
                error_tag_logits,
                error_tag_loss,
            ) = transformer_error_tag_prediction_layer(
                decoder_output, hparams, features, loss_mask = loss_mask
            )
            self.logits['targets_error_tag'] = error_tag_logits
            self.loss_num['targets_error_tag'] = error_tag_loss
            decoder_output = transformer_between_predictions_layer(
                decoder_output,
                hparams,
                name = 'post_error_tag',
                nonpadding = nonpadding,
                losses = losses,
            )

        return decoder_output

    def _loss_single(self, logits, feature_name, feature, weights = None):
        """Prevents modality loss computation for targets_*."""
        if feature_name in ['targets_error_tag']:
            loss_num = self.loss_num[feature_name]
            loss_num *= self._problem_hparams.loss_multiplier
            loss_den = self.loss_den
        else:
            loss_num, loss_den = super(TransformerTag, self)._loss_single(
                logits, feature_name, feature, weights
            )
        tf.summary.scalar('loss/%s' % feature_name, loss_num / loss_den)
        return loss_num, loss_den

    def top(self, body_output, features):
        """Adds additional dimensions and then calls super class implementation."""
        exp_features = features
        for feat in body_output.keys():
            while len(body_output[feat].shape) < 4:
                logging.warning('Expanding body output %s...', feat)
                body_output[feat] = tf.expand_dims(body_output[feat], -2)
            if feat in exp_features:
                while len(exp_features[feat].shape) < 4:
                    exp_features[feat] = tf.expand_dims(exp_features[feat], -1)
                    logging.warning('Expanding feature %s...', feat)
        return super(TransformerTag, self).top(body_output, exp_features)

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
        inputs = tf.expand_dims(inputs, axis = 1)
        if len(inputs.shape) < 5:
            inputs = tf.expand_dims(inputs, axis = 4)
        s = common_layers.shape_list(inputs)
        inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
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

    def get_decode_start_id(self):
        return None

    def get_decode_end_id(self):
        return None

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
            return super(TransformerTag, self)._greedy_infer(
                features, decode_length
            )
        with tf.variable_scope(self.name):
            if use_tpu:
                return self._fast_decode_tpu(features, decode_length)
            return self._fast_decode(features, decode_length)

    def _fast_decode(
        self,
        features,
        decode_length,
        beam_size = 1,
        top_beams = 1,
        alpha = 1.0,
        preprocess_targets_method = None,
    ):
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

        target_tag_modality = self._problem_hparams.modality[
            'targets_error_tag'
        ]
        target_tag_vocab_size = self._problem_hparams.vocab_size[
            'targets_error_tag'
        ]
        if target_tag_vocab_size is not None and hasattr(
            hparams, 'vocab_divisor'
        ):
            target_tag_vocab_size += (
                -target_tag_vocab_size
            ) % hparams.vocab_divisor

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
            encoder_output = None
            encoder_decoder_attention_bias = None
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
            targets = self._shard_features({'targets': targets})['targets']
            modality_name = hparams.name.get(
                'targets', modalities.get_name(target_modality)
            )(hparams, target_vocab_size)
            with tf.variable_scope(modality_name + '/targets'):
                bottom = hparams.bottom.get(
                    'targets', modalities.get_targets_bottom(target_modality)
                )
                targets = dp(bottom, targets, hparams, target_vocab_size)[0]
            targets = common_layers.flatten4d3d(targets)

            if not self.get_decode_start_id():
                targets = tf.cond(
                    tf.equal(i, 0),
                    lambda: tf.zeros_like(targets),
                    lambda: targets,
                )

            if positional_encoding is not None:
                targets += positional_encoding[:, i : i + 1]
            return targets

        def preprocess_targets_tag_method(targets, i):
            targets = self._shard_features({'targets_error_tag': targets})[
                'targets_error_tag'
            ]
            modality_name = hparams.name.get(
                'targets_error_tag', modalities.get_name(target_tag_modality)
            )(hparams, target_tag_vocab_size)
            with tf.variable_scope(modality_name + '/targets_error_tag'):
                bottom = hparams.bottom.get(
                    'targets_error_tag',
                    modalities.get_targets_bottom(target_tag_modality),
                )
                targets = dp(bottom, targets, hparams, target_tag_vocab_size)[0]
            targets = common_layers.flatten4d3d(targets)
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

        def symbols_to_logits_fn(ids, ids_tag, i, cache):
            """Go from ids to logits for next symbol."""
            ids = ids[:, -1:]
            targets = tf.expand_dims(tf.expand_dims(ids, axis = 2), axis = 3)
            targets = preprocess_targets_method(targets, i)

            ids_tag = ids_tag[:, -1:]
            targets_tag = tf.expand_dims(
                tf.expand_dims(ids_tag, axis = 2), axis = 3
            )
            targets_tag = preprocess_targets_tag_method(targets_tag, i)

            bias = decoder_self_attention_bias[:, :, i : i + 1, : i + 1]

            with tf.variable_scope('body'):
                with tf.variable_scope('edit_ops_layer'):
                    with tf.variable_scope('ffn'):
                        x = targets
                        preproc = lambda z: common_layers.layer_preprocess(
                            z, hparams, layer_collection = None
                        )
                        layer_inputs = [
                            tf.concat(preproc(x), axis = 0),
                            tf.concat(preproc(targets_tag), axis = 0),
                        ]
                        y = transformer_layers.transformer_ffn_layer(
                            tf.concat(layer_inputs, axis = 2),
                            hparams,
                            conv_padding = 'LEFT',
                            nonpadding_mask = features_to_nonpadding(
                                features, 'targets'
                            ),
                            losses = None,
                            cache = cache,
                            decode_loop_step = None,
                            layer_collection = None,
                        )
                        targets = common_layers.layer_postprocess(x, y, hparams)

                if hparams.middle_prediction:
                    num_decoder_layers = (
                        hparams.num_decoder_layers or hparams.num_hidden_layers
                    )
                    hparams.num_decoder_layers = int(
                        num_decoder_layers
                        / hparams.middle_prediction_layer_factor
                    )

                body_outputs = dp(
                    self.decode,
                    targets,
                    cache.get('encoder_output'),
                    cache.get('encoder_decoder_attention_bias'),
                    bias,
                    hparams,
                    cache,
                    nonpadding = features_to_nonpadding(features, 'targets'),
                )[0]

                body_outputs, logits_tag = dp(
                    self._prediction_cascade_predict,
                    hparams,
                    features_to_nonpadding(features, 'targets'),
                    cache.get('encoder_decoder_attention_bias'),
                    cache.get('encoder_output'),
                    body_outputs,
                )
                logits_tag = logits_tag[0]['targets_error_tag']
                if hparams.middle_prediction:
                    with tf.variable_scope('after_prediction'):
                        body_outputs = dp(
                            self.decode,
                            targets + body_outputs[0],
                            cache.get('encoder_output'),
                            cache.get('encoder_decoder_attention_bias'),
                            bias,
                            hparams,
                            cache,
                            nonpadding = features_to_nonpadding(
                                features, 'targets'
                            ),
                        )

            update_decoder_attention_history(cache)

            modality_name = hparams.name.get(
                'targets', modalities.get_name(target_modality)
            )(hparams, target_vocab_size)
            with tf.variable_scope('targets/' + modality_name):
                top = hparams.top.get(
                    'targets', modalities.get_top(target_modality)
                )
                logits = dp(
                    top, body_outputs, None, hparams, target_vocab_size
                )[0]

            ret = tf.squeeze(logits, axis = [1, 2])
            if partial_targets is not None:
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
            logits_tag = tf.squeeze(logits_tag, axis = [1])
            return ret, logits_tag, cache

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
            init_cache_fn = _init_transformer_cache,
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


def _pointer_feedback(pointers, encoder_output, shift = True):
    """Feedback loop for pointer networks.

  Args:
    pointers: [batch_size, target_length] int tensor with pointers into the
      source sentence.
    encoder_output: [batch_size, input_length, hidden_size] tensor with encoder
      outputs.
    shift: Whether to shift the pointers to the right.

  Returns:
    A [batch_size, target_length, hidden_size] tensor with encoder outputs.
  """
    if shift:
        pointers = common_layers.shift_right_2d(pointers)
    return gather_2d(encoder_output, pointers)


def transformer_edit_ops_layer(
    decoder_input,
    hparams,
    encoder_output,
    features,
    cache = None,
    decode_loop_step = None,
    nonpadding = None,
    losses = None,
    layer_collection = None,
):
    """Layer that conditions on the error tag and start and end token pointers."""
    if isinstance(encoder_output, list):  # Select forward encoder
        encoder_output = encoder_output[0]
    with tf.variable_scope('edit_ops_layer'):
        with tf.variable_scope('ffn'):
            x = decoder_input
            # Shorthand for layer preprocessing
            # pylint: disable=g-long-lambda
            preproc = lambda z: common_layers.layer_preprocess(
                z, hparams, layer_collection = layer_collection
            )
            # pylint: enable=g-long-lambda
            layer_inputs = [preproc(x)]
            error_tags = common_layers.shift_right_3d(
                common_layers.flatten4d3d(features['targets_error_tag'])
            )
            layer_inputs.append(preproc(error_tags))
            y = transformer_layers.transformer_ffn_layer(
                tf.concat(layer_inputs, axis = 2),
                hparams,
                conv_padding = 'LEFT',
                nonpadding_mask = nonpadding,
                losses = losses,
                cache = cache,
                decode_loop_step = decode_loop_step,
                layer_collection = layer_collection,
            )
            x = common_layers.layer_postprocess(x, y, hparams)
            return x


def transformer_between_predictions_layer(
    x,
    hparams,
    name,
    cache = None,
    decode_loop_step = None,
    nonpadding = None,
    losses = None,
    layer_collection = None,
):
    """Stack between prediction layers."""
    with tf.variable_scope(name):
        for i in range(hparams.ffn_in_prediction_cascade):
            with tf.variable_scope('layer_%d' % i):
                y = transformer_layers.transformer_ffn_layer(
                    common_layers.layer_preprocess(
                        x, hparams, layer_collection = layer_collection
                    ),
                    hparams,
                    conv_padding = 'LEFT',
                    nonpadding_mask = nonpadding,
                    losses = losses,
                    cache = cache,
                    decode_loop_step = decode_loop_step,
                    layer_collection = layer_collection,
                )
                x = common_layers.layer_postprocess(x, y, hparams)
    return x


def get_error_tag_embedding_matrix():
    candidates = [
        var
        for var in tf.global_variables()
        if 'targets_error_tag' in var.op.name
    ]
    if len(candidates) != 1:
        raise ValueError(
            'Could not identify error tag embedding matrix! '
            'Matching variable names: %s' % candidates
        )
    embed_mat = candidates
    return embed_mat


def transformer_error_tag_prediction_layer(
    x, hparams, features, loss_mask, layer_collection = None
):
    """Layer that predicts the error tag."""
    with tf.variable_scope('error_tag_prediction'):
        x = maybe_flatten4d3d(x)
        vocab_size = hparams.problem.feature_info[
            'targets_error_tag'
        ].vocab_size
        labels = features['targets_error_tag_raw']
        with tf.variable_scope('projection'):
            bottleneck = common_layers.dense(
                x,
                hparams.error_tag_embed_size,
                layer_collection = layer_collection,
                name = 'bottleneck',
            )
            logits = common_layers.dense(
                bottleneck,
                vocab_size,
                use_bias = False,
                layer_collection = layer_collection,
                name = 'logits',
            )
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits = logits, labels = labels
            )
            loss = tf.reduce_sum(xent * loss_mask)
        with tf.variable_scope('embedding'):
            # embed_mat = get_error_tag_embedding_matrix()
            y = common_layers.layer_preprocess(
                common_layers.embedding(
                    labels,
                    vocab_size,
                    hparams.hidden_size,
                    embedding_var = None,
                ),
                hparams,
                layer_collection = layer_collection,
            )
            x = common_layers.layer_postprocess(x, y, hparams)
        return x, logits, loss


def transformer_error_tag_prediction_layer_predict(
    x, hparams, layer_collection = None
):
    with tf.variable_scope('error_tag_prediction'):
        x = maybe_flatten4d3d(x)
        vocab_size = hparams.problem.feature_info[
            'targets_error_tag'
        ].vocab_size
        with tf.variable_scope('projection'):
            bottleneck = common_layers.dense(
                x,
                hparams.error_tag_embed_size,
                layer_collection = layer_collection,
                name = 'bottleneck',
            )
            logits = common_layers.dense(
                bottleneck,
                vocab_size,
                use_bias = False,
                layer_collection = layer_collection,
                name = 'logits',
            )
        labels = tf.argmax(logits, axis = -1)
        with tf.variable_scope('embedding'):
            y = common_layers.layer_preprocess(
                common_layers.embedding(
                    labels,
                    vocab_size,
                    hparams.hidden_size,
                    embedding_var = None,
                ),
                hparams,
                layer_collection = layer_collection,
            )
            x = common_layers.layer_postprocess(x, y, hparams)
        return x, logits


def _init_transformer_cache(
    cache,
    hparams,
    batch_size,
    attention_init_length,
    encoder_output,
    encoder_decoder_attention_bias,
    scope_prefix,
):
    """Create the initial cache for TransformerTag fast decoding."""
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
            % layer: {
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
    else:

        def inner_loop(
            i,
            hit_eos,
            next_id,
            next_id_tag,
            decoded_ids,
            decoded_ids_tag,
            cache,
            log_prob,
        ):
            """One step of greedy decoding."""
            logits, logits_tag, cache = symbols_to_logits_fn(
                next_id, next_id_tag, i, cache
            )
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

            if hparams.sampling_method == 'random_per_example':
                next_id_tag = common_layers.sample_temperature_per_example(
                    logits_tag, temperature, top_k
                )
            else:
                if hparams.sampling_method == 'argmax':
                    temperature = 0.0
                next_id_tag = common_layers.sample_with_temperature(
                    logits_tag, temperature, top_k
                )

            log_prob_indices = tf.stack(
                [tf.range(tf.to_int64(batch_size)), next_id], axis = 1
            )
            log_prob += tf.gather_nd(log_probs, log_prob_indices) * (
                1 - tf.to_float(hit_eos)
            )
            hit_eos |= tf.equal(next_id, eos_id)

            next_id = tf.expand_dims(next_id, axis = 1)
            decoded_ids = tf.concat([decoded_ids, next_id], axis = 1)
            next_id_tag = tf.expand_dims(next_id_tag, axis = 1)
            decoded_ids_tag = tf.concat(
                [decoded_ids_tag, next_id_tag], axis = 1
            )

            return (
                i + 1,
                hit_eos,
                next_id,
                next_id_tag,
                decoded_ids,
                decoded_ids_tag,
                cache,
                log_prob,
            )

        def is_not_finished(i, hit_eos, *_):
            finished = i >= decode_length
            if not force_decode_length:
                finished |= tf.reduce_all(hit_eos)
            return tf.logical_not(finished)

        decoded_ids = tf.zeros([batch_size, 0], dtype = tf.int64)
        decoded_ids_tag = tf.zeros([batch_size, 0], dtype = tf.int64)
        hit_eos = tf.fill([batch_size], False)
        next_id = sos_id * tf.ones([batch_size, 1], dtype = tf.int64)
        next_id_tag = sos_id * tf.ones([batch_size, 1], dtype = tf.int64)
        initial_log_prob = tf.zeros([batch_size], dtype = tf.float32)

        _, _, _, _, decoded_ids, decoded_ids_tag, cache, log_prob = tf.while_loop(
            is_not_finished,
            inner_loop,
            [
                tf.constant(0),
                hit_eos,
                next_id,
                next_id_tag,
                decoded_ids,
                decoded_ids_tag,
                cache,
                initial_log_prob,
            ],
            shape_invariants = [
                tf.TensorShape([]),
                tf.TensorShape([None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                nest.map_structure(
                    beam_search.get_state_shape_invariants, cache
                ),
                tf.TensorShape([None]),
            ],
        )
        scores = log_prob

    return {
        'outputs': decoded_ids,
        'outputs_tag': decoded_ids_tag,
        'scores': scores,
        'cache': cache,
    }
