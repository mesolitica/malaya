"""
coding: utf-8

Based on:
https://raw.githubusercontent.com/dmlc/gluon-nlp/8a23a8bcb10a05cdf1360cb237e0d5306ae17183/scripts/bert/model/classification.py

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

https://github.com/awslabs/mlm-scoring/blob/master/LICENSE
"""

import numpy as np
import torch
from transformers import BertForMaskedLM, AlbertForMaskedLM, RobertaForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from torch.nn import CrossEntropyLoss
from malaya_boilerplate.torch_utils import to_tensor_cuda, to_numpy


class BertForMaskedLMOptimized(BertForMaskedLM):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        select_positions=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if select_positions is not None:
            sequence_output = sequence_output[[[i] for i in range(sequence_output.shape[0])], select_positions, :]

        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AlbertForMaskedLMOptimized(AlbertForMaskedLM):

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        select_positions=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with
            labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_outputs = outputs[0]

        if select_positions is not None:
            sequence_outputs = sequence_outputs[[[i] for i in range(sequence_outputs.shape[0])], select_positions, :]

        prediction_scores = self.predictions(sequence_outputs)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForMaskedLMOptimized(RobertaForMaskedLM):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        select_positions=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if select_positions is not None:
            sequence_output = sequence_output[[[i] for i in range(sequence_output.shape[0])], select_positions, :]

        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MLMScorer:
    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    def cuda(self):
        return self._model.cuda()

    def _ids_to_masked(self, token_ids):
        token_ids_masked_list = []

        mask_indices = []
        mask_indices = [[mask_pos] for mask_pos in range(len(token_ids))]

        # We don't mask the [CLS], [SEP] for now for PLL
        mask_indices = mask_indices[1:-1]

        mask_token_id = self._tokenizer._convert_token_to_id(self._tokenizer.mask_token)
        for mask_set in mask_indices:
            token_ids_masked = token_ids.copy()
            token_ids_masked[mask_set] = mask_token_id

            token_ids_masked_list.append((token_ids_masked, mask_set))

        return token_ids_masked_list

    def corpus_to_dataset(self, corpus):
        sents_expanded = []
        for sent_idx, sent in enumerate(corpus):
            ids_original = np.array(self._tokenizer.encode(sent, add_special_tokens=True))
            ids_masked = self._ids_to_masked(ids_original)
            sents_expanded += [(
                sent_idx,
                ids,
                len(ids_original),
                mask_set,
                ids_original[mask_set],
                1)
                for ids, mask_set in ids_masked]
        return sents_expanded

    def score(self, string):
        """
        score a string.

        Parameters
        ----------
        string: str

        Returns
        -------
        result: float
        """
        cuda = next(self._model.parameters()).is_cuda
        corpus = [string]
        dataset = self.corpus_to_dataset(corpus)
        sent_idxs = np.stack([d[0] for d in dataset], 0)
        token_ids = np.stack([d[1] for d in dataset], 0)
        valid_length = np.stack([d[2] for d in dataset], 0)
        masked_positions = np.stack([d[3] for d in dataset], 0)
        token_masked_ids = np.stack([d[4] for d in dataset], 0)

        with torch.no_grad():

            token_ids = torch.tensor(token_ids)
            valid_length = torch.tensor(valid_length)
            masked_positions = torch.tensor(masked_positions).reshape(-1, 1)
            token_masked_ids = torch.tensor(token_masked_ids).reshape(-1)

            token_ids = to_tensor_cuda(token_ids, cuda)
            valid_length = to_tensor_cuda(valid_length, cuda)
            masked_positions = to_tensor_cuda(masked_positions, cuda)
            token_masked_ids = to_tensor_cuda(token_masked_ids, cuda)

            split_size = token_ids.shape[0]
            alen = torch.arange(token_ids.shape[1], dtype=torch.long)
            alen = to_tensor_cuda(alen, cuda)
            mask = alen < valid_length[:, None]
            out = self._model(input_ids=token_ids, attention_mask=mask, select_positions=masked_positions)
            out = out[0].squeeze()
            if len(out.shape) == 1:
                out = out.unsqueeze(0)
            out = out.log_softmax(dim=-1)
            out = out[list(range(split_size)), token_masked_ids]
            return to_numpy(out).sum()
