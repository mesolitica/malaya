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
from malaya.torch_model.constituency_modules import (
    Encoder,
    LayerNormalization,
    MultiLevelEmbedding
)
from malaya.function.constituency import chart_helper, trees_newline as trees
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch import nn, Tensor
from dataclasses import dataclass
from torch import nn
from typing import Dict
import numpy as np
import torch


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


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
        self.dense_layer = nn.Linear(self.config.hidden_size, 1024)

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
        output = self.dense_layer(psg_out.last_hidden_state)
        p_reps = self.sentence_embedding(output, features['attention_mask'])
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


class T5Constituency(T5EncoderModel):
    emb_types = []

    partitioned = True
    num_layers_position_only = 0
    num_layers = 8
    d_model = 1024
    num_heads = 8
    d_kv = 64
    d_ff = 2048
    d_label_hidden = 250
    d_tag_hidden = 250
    attention_dropout = 0.2
    embedding_dropout = 0.0
    relu_dropout = 0.1
    residual_dropout = 0.2
    timing_dropout = 0.0
    morpho_emb_dropout = 0.2
    sentence_max_len = 1024

    d_content = (d_model // 2) if partitioned else d_model
    d_positional = (d_model // 2) if partitioned else None

    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config, **kwargs)

        self.label_vocab = {eval(k): v for k, v in self.config.label_vocab.items()}
        self.label_vocab_rev = {v: k for k, v in self.label_vocab.items()}
        self.tag_vocab = self.config.tag_vocab
        self.tag_vocab_rev = {v: k for k, v in self.tag_vocab.items()}

        num_embeddings_map = {
            'tags': len(self.tag_vocab),
        }
        emb_dropouts_map = {
            'tags': 0.2,
        }

        self.f_label = nn.Sequential(
            nn.Linear(self.d_model, self.d_label_hidden),
            LayerNormalization(self.d_label_hidden),
            nn.ReLU(),
            nn.Linear(self.d_label_hidden, self.config.num_labels - 1),
        )
        self.f_tag = nn.Sequential(
            nn.Linear(self.d_model, self.d_tag_hidden),
            LayerNormalization(self.d_tag_hidden),
            nn.ReLU(),
            nn.Linear(self.d_tag_hidden, self.config.num_tags),
        )

        self.project_bert = nn.Linear(self.config.d_model, self.d_content, bias=False)

        self.embedding = MultiLevelEmbedding(
            [num_embeddings_map[emb_type] for emb_type in self.emb_types],
            self.d_model,
            d_positional=self.d_positional,
            dropout=self.embedding_dropout,
            timing_dropout=self.timing_dropout,
            emb_dropouts_list=[emb_dropouts_map[emb_type] for emb_type in self.emb_types],
            extra_content_dropout=self.morpho_emb_dropout,
            max_len=self.sentence_max_len,
        )

        self.encoder_encoder = Encoder(
            self.embedding,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_kv=self.d_kv,
            d_ff=self.d_ff,
            d_positional=self.d_positional,
            num_layers_position_only=self.num_layers_position_only,
            relu_dropout=self.relu_dropout,
            residual_dropout=self.residual_dropout,
            attention_dropout=self.attention_dropout,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sentences=None,
        batch_idxs=None,
        tag_idxs=None,
        gold_tag_idxs=None,
        all_word_start_mask=None,
        all_word_end_mask=None,
        golds=None,
    ):
        is_train = golds is not None
        if golds is None:
            golds = [None] * len(sentences)
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        features = outputs.last_hidden_state
        features_packed = features.masked_select(all_word_end_mask.to(
            torch.bool).unsqueeze(-1)).reshape(-1, features.shape[-1])
        extra_content_annotations = self.project_bert(features_packed)

        annotations, _ = self.encoder_encoder(
            [], batch_idxs, extra_content_annotations=extra_content_annotations)

        if self.partitioned:
            annotations = torch.cat([
                annotations[:, 0::2],
                annotations[:, 1::2],
            ], 1)
        tag_annotations = annotations
        fencepost_annotations = torch.cat([
            annotations[:-1, :self.d_model//2],
            annotations[1:, self.d_model//2:],
        ], 1)
        fencepost_annotations_start = fencepost_annotations
        fencepost_annotations_end = fencepost_annotations

        tag_logits = self.f_tag(tag_annotations)

        fp_startpoints = batch_idxs.boundaries_np[:-1]
        fp_endpoints = batch_idxs.boundaries_np[1:] - 1

        if is_train:
            tag_loss = self.config.tag_loss_scale * \
                nn.functional.cross_entropy(tag_logits, gold_tag_idxs, reduction='sum')

            pis = []
            pjs = []
            plabels = []
            paugment_total = 0.0
            num_p = 0
            gis = []
            gjs = []
            glabels = []
            with torch.no_grad():
                for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
                    p_i, p_j, p_label, p_augment, g_i, g_j, g_label = self.parse_from_annotations(
                        fencepost_annotations_start[start:end, :], fencepost_annotations_end[start:end, :], sentences[i], golds[i])
                    paugment_total += p_augment
                    num_p += p_i.shape[0]
                    pis.append(p_i + start)
                    pjs.append(p_j + start)
                    gis.append(g_i + start)
                    gjs.append(g_j + start)
                    plabels.append(p_label)
                    glabels.append(g_label)

            cells_i = torch.from_numpy(np.concatenate(pis + gis)).to(self.device)
            cells_j = torch.from_numpy(np.concatenate(pjs + gjs)).to(self.device)
            cells_label = torch.from_numpy(np.concatenate(plabels + glabels)).to(self.device)

            cells_label_scores = self.f_label(
                fencepost_annotations_end[cells_j] -
                fencepost_annotations_start[cells_i])
            cells_label_scores = torch.cat([
                cells_label_scores.new_zeros((cells_label_scores.size(0), 1)),
                cells_label_scores
            ], 1)
            cells_scores = torch.gather(cells_label_scores, 1, cells_label[:, None])
            loss = cells_scores[:num_p].sum() - cells_scores[num_p:].sum() + paugment_total
            return loss, tag_loss
        else:
            trees = []
            scores = []
            tag_idxs = torch.argmax(tag_logits, -1).cpu()
            per_sentence_tag_idxs = torch.split_with_sizes(
                tag_idxs, [len(sentence) + 2 for sentence in sentences])
            per_sentence_tags = [[self.tag_vocab_rev[int(idx)] for idx in idxs[1:-1]]
                                 for idxs in per_sentence_tag_idxs]

            for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
                sentence = sentences[i]
                if self.f_tag is not None:
                    sentence = list(zip(per_sentence_tags[i], [x[1] for x in sentence]))
                tree, score = self.parse_from_annotations(
                    fencepost_annotations_start[start:end, :], fencepost_annotations_end[start:end, :], sentence, golds[i])
                trees.append(tree)
                scores.append(score)
            return trees, scores

    def label_scores_from_annotations(self, fencepost_annotations_start, fencepost_annotations_end):
        span_features = (torch.unsqueeze(fencepost_annotations_end, 0)
                         - torch.unsqueeze(fencepost_annotations_start, 1))

        label_scores_chart = self.f_label(span_features)
        label_scores_chart = torch.cat([label_scores_chart.new_zeros(
            (label_scores_chart.size(0), label_scores_chart.size(1), 1)), label_scores_chart], 2)
        return label_scores_chart

    def parse_from_annotations(
        self,
        fencepost_annotations_start,
        fencepost_annotations_end,
        sentence,
        gold=None
    ):
        is_train = gold is not None
        label_scores_chart = self.label_scores_from_annotations(
            fencepost_annotations_start, fencepost_annotations_end)
        label_scores_chart_np = label_scores_chart.cpu().data.numpy()

        if is_train:
            decoder_args = dict(
                sentence_len=len(sentence),
                label_scores_chart=label_scores_chart_np,
                gold=gold,
                label_vocab=self.label_vocab,
                is_train=is_train
            )

            p_score, p_i, p_j, p_label, p_augment = chart_helper.decode(False, **decoder_args)
            g_score, g_i, g_j, g_label, g_augment = chart_helper.decode(True, **decoder_args)
            return p_i, p_j, p_label, p_augment, g_i, g_j, g_label
        else:
            return self.decode_from_chart(sentence, label_scores_chart_np)

    def decode_from_chart(self, sentence, chart_np, gold=None):
        decoder_args = dict(
            sentence_len=len(sentence),
            label_scores_chart=chart_np,
            gold=gold,
            label_vocab=self.label_vocab,
            is_train=False
        )

        force_gold = (gold is not None)

        score, p_i, p_j, p_label, _ = chart_helper.decode(force_gold, **decoder_args)
        last_splits = []
        idx = -1

        def make_tree():
            nonlocal idx
            idx += 1
            i, j, label_idx = p_i[idx], p_j[idx], p_label[idx]
            label = self.label_vocab_rev[label_idx]
            if (i + 1) >= j:
                tag, word = sentence[i]
                tree = trees.LeafParseNode(int(i), tag, word)
                if label:
                    tree = trees.InternalParseNode(label, [tree])
                return [tree]
            else:
                left_trees = make_tree()
                right_trees = make_tree()
                children = left_trees + right_trees
                if label:
                    return [trees.InternalParseNode(label, children)]
                else:
                    return children

        tree = make_tree()[0]
        return tree, score


class T5CrossEncoder(T5ForSequenceClassification):
    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config, **kwargs)

        self.register_buffer(
            'target_label',
            torch.zeros(self.config.per_device_train_batch_size, dtype=torch.long)
        )
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, batch):
        ranker_out = super().forward(**batch, return_dict=True)
        logits = ranker_out.logits
        if self.training:
            scores = logits.view(
                self.config.per_device_train_batch_size,
                self.config.train_group_size
            )
            loss = self.cross_entropy(scores, self.target_label)
            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return ranker_out
