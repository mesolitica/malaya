from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.modeling_outputs import (
    Seq2SeqSequenceClassifierOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput
)
from transformers import (
    T5Model,
    T5Config,
    T5ForConditionalGeneration,
    T5EncoderModel
)
from transformers.file_utils import ModelOutput
from malaya.torch_model.dependency_modules import MLP, Biaffine
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch import nn, Tensor
from dataclasses import dataclass
from torch import nn
from typing import Dict
import torch


@dataclass
class EncoderOutput(ModelOutput):
    q_reps = None
    p_reps = None
    loss = None
    scores = None


class T5Tagging(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config, **kwargs)
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.dropout_rate,
        )
        self._init_weights(self.classification_head.dense)
        self._init_weights(self.classification_head.out_proj)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        labels_tag=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        last_layer = outputs.decoder_hidden_states[-1]
        logits = self.classification_head(last_layer)
        if labels_tag is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels_tag.view(-1))
            outputs.loss += loss

        return outputs


class T5Diaparser(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config, **kwargs)
        n_mlp_arc = 500
        n_mlp_rel = 100
        mlp_dropout = .33
        mlp_input_size = config.d_model
        n_rels = config.num_labels
        self.mlp_arc_d = MLP(n_in=mlp_input_size,
                             n_out=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_arc_h = MLP(n_in=mlp_input_size,
                             n_out=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=mlp_input_size,
                             n_out=n_mlp_rel,
                             dropout=mlp_dropout)
        self.mlp_rel_h = MLP(n_in=mlp_input_size,
                             n_out=n_mlp_rel,
                             dropout=mlp_dropout)

        self.arc_attn = Biaffine(n_in=n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel,
                                 n_out=n_rels,
                                 bias_x=True,
                                 bias_y=True)

        self._init_weights(self.mlp_arc_d.linear.weight)
        self._init_weights(self.mlp_arc_h.linear.weight)

        self._init_weights(self.mlp_rel_d.linear.weight)
        self._init_weights(self.mlp_rel_h.linear.weight)

        self._init_weights(self.arc_attn.weight)

        self._init_weights(self.rel_attn.weight)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        labels_arc=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        indices=None,
    ):
        decoder_input_ids = input_ids

        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        last_layer = outputs.decoder_hidden_states[-1]
        x = last_layer
        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)
        s_arc = self.arc_attn(arc_d, arc_h)
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        mask = ~input_ids.ne(0)
        batch_size, seq_len = input_ids.shape
        s_arc.masked_fill_(mask.unsqueeze(1), float('-inf'))
        if indices is not None:
            arange = torch.arange(seq_len)
            for k in range(len(s_arc)):
                d = torch.diag(s_arc.new(seq_len).fill_(0.0))
                d[arange, indices[k]] = float('-inf')
                d[0] = 0.0
                s_arc[k] += d

        outputs['s_arc'] = s_arc
        outputs['s_rel'] = s_rel

        if labels_arc is not None:
            criterion = CrossEntropyLoss()
            mask = attention_mask.bool()
            s_arc, arcs = s_arc[mask], labels_arc[mask]
            s_rel, rels = s_rel[mask], labels[mask]
            s_rel = s_rel[torch.arange(len(arcs)), arcs]

            arc_loss = criterion(s_arc, arcs)
            rel_loss = criterion(s_rel, rels)
            outputs['loss'] = arc_loss + rel_loss

        return outputs


class T5Embedding(T5EncoderModel):
    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config, **kwargs)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def sentence_embedding(self, hidden_state, mask):
        if self.config.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.config.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    def encode(self, features):
        if features is None:
            return None
        psg_out = super().forward(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        if self.config.normalized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def forward(self, query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        if self.training:
            if self.negatives_cross_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores / self.temperature
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))
            loss = self.compute_loss(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )


class T5ForSequenceClassification(T5EncoderModel):
    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config, **kwargs)

        self.num_labels = self.config.num_labels
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        s = torch.sum(outputs.last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
        d = attention_mask.sum(axis=1, keepdim=True).float()
        logits = self.classifier(self.dropout(s / d))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class T5ForTokenClassification(T5EncoderModel):
    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config, **kwargs)

        self.num_labels = self.config.num_labels
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        s = outputs.last_hidden_state * attention_mask.unsqueeze(-1).float()
        logits = self.classifier(self.dropout(s))

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


"""
tokenizer = AutoTokenizer.from_pretrained('mesolitica/nanot5-base-malaysian-cased')
config = T5Config.from_pretrained('mesolitica/nanot5-base-malaysian-cased')
config.problem_type = 'single_label_classification'
config.num_labels = 2
input_ids = tokenizer(['hello', 'a', 'awak gile ke'], padding = True, return_tensors = 'pt')
input_ids['labels'] = torch.Tensor([0,1,0]).long()
model = T5ForSequenceClassification(config)
model(**input_ids)

model = T5ForTokenClassification(config)
input_ids['labels'] = torch.Tensor([[0,1,0,0], [1,1,1,1], [0, 0,0,0]]).long()
model(**input_ids)
"""
