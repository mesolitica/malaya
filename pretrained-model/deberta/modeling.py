import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from transformers import (
    DebertaV2Config,
    DebertaV2PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from transformers.activations import get_activation
from transformers.modeling_outputs import ModelOutput
from debertav2 import DebertaV2EmdModel, DebertaV2EmdForPreTraining


@dataclass
class DebertaV2ForReplacedTokenDetectionOutput(ModelOutput):
    # copied from ElectraForPreTrainingOutput
    """
    Output type of [`DebertaV2ForReplacedTokenDetection`].
    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss of the ELECTRA objective.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class DebertaV3ForPreTrainingOutput(ModelOutput):
    """
    Output type of [`DebertaV3ForPreTraining`].

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss of the ELECTRA objective.
        gen_loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Generator loss of the ELECTRA objective.
        disc_loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Discriminator loss of the ELECTRA objective.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    gen_loss: Optional[torch.FloatTensor] = None
    disc_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class LMMaskPredictionHead(nn.Module):
    """Replaced token prediction head"""

    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = get_activation(config.hidden_act)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor):
        # b x d
        ctx_states = hidden_states[:, 0, :]
        seq_states = self.LayerNorm(ctx_states.unsqueeze(-2) + hidden_states)
        seq_states = self.dense(seq_states)
        seq_states = self.transform_act_fn(seq_states)

        # b x max_len
        logits = self.classifier(seq_states).squeeze(-1)
        return logits


class DebertaV2ForReplacedTokenDetection(DebertaV2PreTrainedModel):
    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
        self.deberta = DebertaV2EmdModel(config)
        self.discriminator_predictions = LMMaskPredictionHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DebertaV2ForReplacedTokenDetectionOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        discriminator_hidden_states = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = (
                    attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                )
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[
                    active_loss
                ]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(
                    logits.view(-1, discriminator_sequence_output.shape[1]),
                    labels.float(),
                )

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return DebertaV2ForReplacedTokenDetectionOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class DebertaV3ForPreTraining(DebertaV2ForReplacedTokenDetection):
    def __init__(
            self,
            config_generator: DebertaV2Config,
            config_discriminator: DebertaV2Config,
            keep_disc_ebd_weight: bool = False):
        super().__init__(config_discriminator)
        self.generator = DebertaV2EmdForPreTraining(config_generator)
        self.has_position_biased_input = config_generator.position_biased_input
        self.discriminator = DebertaV2ForReplacedTokenDetection(config_discriminator)
        # Initialize weights and apply final processing
        self.post_init()
        self.weight_share(keep_disc_ebd_weight)

    @staticmethod
    def _set_param(module, param_name, value):
        if hasattr(module, param_name):
            delattr(module, param_name)
        module.register_buffer(param_name, value)

    def register_discriminator_fw_hook(self, *kwargs):
        def fw_hook(module, *inputs):

            g_w_ebd = self.generator.deberta.embeddings.word_embeddings
            d_w_ebd = self.discriminator.deberta.embeddings.word_embeddings
            self._set_param(
                d_w_ebd, "weight", g_w_ebd.weight.detach() + d_w_ebd._weight
            )

            if self.has_position_biased_input:
                g_p_ebd = self.generator.deberta.embeddings.position_embeddings
                d_p_ebd = self.discriminator.deberta.embeddings.position_embeddings
                self._set_param(
                    d_p_ebd, "weight", g_p_ebd.weight.detach() + d_p_ebd._weight
                )
            return None
        self.discriminator.register_forward_pre_hook(fw_hook)

    def weight_share(self, keep_disc_ebd_weight: bool = False):
        # Gradient-disentangled weight/embedding sharing
        word_bias = torch.zeros_like(
            self.discriminator.deberta.embeddings.word_embeddings.weight
        )
        word_bias = torch.nn.Parameter(word_bias)
        if self.has_position_biased_input:
            position_bias = torch.zeros_like(
                self.discriminator.deberta.embeddings.position_embeddings.weight
            )
            position_bias = torch.nn.Parameter(position_bias)
        if not keep_disc_ebd_weight:
            delattr(self.discriminator.deberta.embeddings.word_embeddings, "weight")
        self.discriminator.deberta.embeddings.word_embeddings.register_parameter(
            "_weight", word_bias
        )
        if self.has_position_biased_input:
            if not keep_disc_ebd_weight:
                delattr(self.discriminator.deberta.embeddings.position_embeddings, "weight")
            self.discriminator.deberta.embeddings.position_embeddings.register_parameter(
                "_weight", position_bias
            )
        self.register_discriminator_fw_hook()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        outputs_gen = self.generator(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # if rand is None:
        #     rand = random
        lm_logits = outputs_gen.logits
        loss_gen = outputs_gen.loss
        if torch.isnan(loss_gen):
            raise ValueError("output_gen is NaN")
        with torch.no_grad():
            input_ids_disc = input_ids.detach()  # .cpu()
            labels_disc = labels.detach()  # .cpu()
            mask_index = (labels_disc.view(-1) > 0).nonzero().view(-1)

            # gen_pred = torch.argmax(lm_logits, dim=1).detach().cpu().numpy()
            # topk_labels, top_p = self.topk_sampling(lm_logits, topk=1, temp=1)
            top_p = torch.nn.functional.softmax(
                lm_logits.detach().view(-1, lm_logits.size(-1)), dim=-1
            )
            topk_labels = torch.multinomial(top_p, 1)

            top_ids = torch.zeros_like(labels_disc.view(-1), dtype=int)
            top_ids.scatter_(index=mask_index, src=topk_labels.view(-1), dim=-1)
            top_ids = top_ids.view(labels_disc.size())
            new_ids = torch.where(labels_disc > 0, top_ids, input_ids_disc)
            gold_ids = torch.where(labels_disc > 0, labels_disc, input_ids_disc)
            labels_disc = (new_ids == gold_ids).long()
        outputs_disc = self.discriminator(
            new_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels_disc,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            loss_disc = outputs_disc[0]
            if torch.isnan(loss_disc):
                raise ValueError("loss_disc is NaN")
            total_loss = loss_gen + loss_disc
            return (
                ((total_loss, loss_gen.detach(), loss_disc.detach()) + outputs_disc[1:])
                if total_loss is not None
                else outputs_disc
            )

        loss_disc = outputs_disc.loss
        if torch.isnan(loss_disc):
            raise ValueError("loss_disc is NaN")
        total_loss = loss_gen + loss_disc
        return DebertaV3ForPreTrainingOutput(
            loss=total_loss,
            gen_loss=loss_gen.detach(),
            disc_loss=loss_disc.detach(),
            logits=outputs_disc.logits,
            hidden_states=outputs_disc.hidden_states,
            attentions=outputs_disc.attentions,
        )

    @classmethod
    def from_pretrained_together(
        cls,
        input_dir: str,
        config_generator: DebertaV2Config,
        config_discriminator: DebertaV2Config,
    ):
        model = cls(config_generator, config_discriminator, keep_disc_ebd_weight=True)
        model.load_state_dict(torch.load(os.path.join(input_dir, "pytorch_model.bin")))
        delattr(model.discriminator.deberta.embeddings.word_embeddings, "_weight")
        if model.has_position_biased_input:
            delattr(model.discriminator.deberta.embeddings.position_embeddings, "_weight")
        return model
