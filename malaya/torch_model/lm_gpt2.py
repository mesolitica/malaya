import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from malaya_boilerplate.torch_utils import to_tensor_cuda, to_numpy


class LM(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

    def load_tokenizer(self, use_fast_tokenizer=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.config._name_or_path, use_fast=use_fast_tokenizer, add_special_tokens=False
        )
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<|pad|>']})
        self.tokenizer.pad_token = '<|pad|>'
        self.resize_token_embeddings(len(self.tokenizer))

    def _add_special_tokens(self, text: str):
        return self.tokenizer.bos_token + text + self.tokenizer.eos_token

    def _tokens_log_prob_for_batch(self, text):

        outputs = []
        if len(text) == 0:
            return outputs

        cuda = next(self.parameters()).is_cuda

        text = list(map(self._add_special_tokens, text))
        encoding = self.tokenizer.batch_encode_plus(
            text, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            ids = to_tensor_cuda(encoding['input_ids'], cuda)
            attention_mask = to_tensor_cuda(encoding['attention_mask'], cuda)
            nopad_mask = ids != self.tokenizer.pad_token_id
            logits = self(ids, attention_mask=attention_mask)[0]

        for sent_index in range(len(text)):
            sent_nopad_mask = nopad_mask[sent_index]
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            sent_logits[:, self.tokenizer.pad_token_id] = float('-inf')
            sent_ids_scores = sent_logits.gather(1, sent_ids.unsqueeze(1)).squeeze(1)
            sent_log_probs = sent_ids_scores - sent_logits.logsumexp(1)

            sent_log_probs = sent_log_probs.type(torch.DoubleTensor)
            sent_ids = sent_ids.type(torch.LongTensor)

            output = (sent_log_probs, sent_ids)
            outputs.append(output)

        return outputs

    def score(self, string, log: bool = True, reduce: str = 'prod'):
        log_probs = self._tokens_log_prob_for_batch([string])[0][0]

        if reduce == 'prod':
            scored = log_probs.sum()
        elif reduce == 'mean':
            scored = log_probs.logsumexp(0) - math.log(tlen)
        elif reduce == 'gmean':
            scored = log_probs.mean(0)
        elif reduce == 'hmean':
            scored = log_probs.neg().logsumexp(0).neg() + math.log(tlen)
        else:
            raise ValueError(f'Unrecognized aggregate strategy: {reduce}')

        if not log:
            scored = scored.exp()

        return float(to_numpy(scored))
