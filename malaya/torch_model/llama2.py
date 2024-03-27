
from transformers import LlamaModel, LlamaConfig, LlamaTokenizer
from typing import Dict
from transformers.file_utils import ModelOutput
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch import nn, Tensor
from dataclasses import dataclass
from torch import nn
from typing import Dict
import torch
from transformers.file_utils import ModelOutput


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class LlamaModelEmbedding(LlamaModel):
    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.dense_layer = nn.Linear(self.config.hidden_size, 1536)

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
            scores = scores / self.config.temperature
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
