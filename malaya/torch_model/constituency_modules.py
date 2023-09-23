import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

class BatchIndices:
    def __init__(self, batch_idxs_np):
        self.batch_idxs_np = batch_idxs_np
        self.batch_idxs_torch = torch.from_numpy(batch_idxs_np)
        self.batch_size = int(1 + np.max(batch_idxs_np))

        batch_idxs_np_extra = np.concatenate([[-1], batch_idxs_np, [-1]])
        self.boundaries_np = np.nonzero(batch_idxs_np_extra[1:] != batch_idxs_np_extra[:-1])[0]
        self.seq_lens_np = self.boundaries_np[1:] - self.boundaries_np[:-1]
        assert len(self.seq_lens_np) == self.batch_size
        self.max_len = int(np.max(self.boundaries_np[1:] - self.boundaries_np[:-1]))


class FeatureDropoutFunction(torch.autograd.function.InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, batch_idxs, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))

        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = input.new().resize_(batch_idxs.batch_size, input.size(1))
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            ctx.noise = ctx.noise[batch_idxs.batch_idxs_torch, :]
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(ctx.noise), None, None, None, None
        else:
            return grad_output, None, None, None, None


class FeatureDropout(nn.Module):
    """
    Feature-level dropout: takes an input of size len x num_features and drops
    each feature with probabibility p. A feature is dropped across the full
    portion of the input that corresponds to a single batch element.
    """

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input, batch_idxs):
        return FeatureDropoutFunction.apply(input, batch_idxs, self.p, self.training, self.inplace)


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3, affine=True):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
            self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(-1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        if self.affine:
            ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        # NOTE(nikita): the t2t code does the following instead, with eps=1e-6
        # However, I currently have no reason to believe that this difference in
        # implementation matters.
        # mu = torch.mean(z, keepdim=True, dim=-1)
        # variance = torch.mean((z - mu.expand_as(z))**2, keepdim=True, dim=-1)
        # ln_out = (z - mu.expand_as(z)) * torch.rsqrt(variance + self.eps).expand_as(z)
        # ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        # q: [batch, slot, feat]
        # k: [batch, slot, feat]
        # v: [batch, slot, feat]

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                'Attention mask shape {} mismatch ' \
                'with Attention logit tensor shape ' \
                '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        # Note that this makes the distribution not sum to 1. At some point it
        # may be worth researching whether this is the right way to apply
        # dropout to the attention.
        # Note that the t2t code also applies dropout in this manner
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module
    """

    def __init__(
            self,
            n_head,
            d_model,
            d_k,
            d_v,
            residual_dropout=0.1,
            attention_dropout=0.1,
            d_positional=None):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        if d_positional is None:
            self.partitioned = False
        else:
            self.partitioned = True

        if self.partitioned:
            self.d_content = d_model - d_positional
            self.d_positional = d_positional

            self.w_qs1 = nn.Parameter(torch.FloatTensor(n_head, self.d_content, d_k // 2))
            self.w_ks1 = nn.Parameter(torch.FloatTensor(n_head, self.d_content, d_k // 2))
            self.w_vs1 = nn.Parameter(torch.FloatTensor(n_head, self.d_content, d_v // 2))

            self.w_qs2 = nn.Parameter(torch.FloatTensor(n_head, self.d_positional, d_k // 2))
            self.w_ks2 = nn.Parameter(torch.FloatTensor(n_head, self.d_positional, d_k // 2))
            self.w_vs2 = nn.Parameter(torch.FloatTensor(n_head, self.d_positional, d_v // 2))

            init.xavier_normal_(self.w_qs1)
            init.xavier_normal_(self.w_ks1)
            init.xavier_normal_(self.w_vs1)

            init.xavier_normal_(self.w_qs2)
            init.xavier_normal_(self.w_ks2)
            init.xavier_normal_(self.w_vs2)
        else:
            self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
            self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
            self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

            init.xavier_normal_(self.w_qs)
            init.xavier_normal_(self.w_ks)
            init.xavier_normal_(self.w_vs)

        self.attention = ScaledDotProductAttention(d_model, attention_dropout=attention_dropout)
        self.layer_norm = LayerNormalization(d_model)

        if not self.partitioned:
            # The lack of a bias term here is consistent with the t2t code, though
            # in my experiments I have never observed this making a difference.
            self.proj = nn.Linear(n_head*d_v, d_model, bias=False)
        else:
            self.proj1 = nn.Linear(n_head*(d_v//2), self.d_content, bias=False)
            self.proj2 = nn.Linear(n_head*(d_v//2), self.d_positional, bias=False)

        self.residual_dropout = FeatureDropout(residual_dropout)

    def split_qkv_packed(self, inp, qk_inp=None):
        v_inp_repeated = inp.repeat(self.n_head, 1).view(
            self.n_head, -1, inp.size(-1))  # n_head x len_inp x d_model
        if qk_inp is None:
            qk_inp_repeated = v_inp_repeated
        else:
            qk_inp_repeated = qk_inp.repeat(self.n_head, 1).view(self.n_head, -1, qk_inp.size(-1))

        if not self.partitioned:
            q_s = torch.bmm(qk_inp_repeated, self.w_qs)  # n_head x len_inp x d_k
            k_s = torch.bmm(qk_inp_repeated, self.w_ks)  # n_head x len_inp x d_k
            v_s = torch.bmm(v_inp_repeated, self.w_vs)  # n_head x len_inp x d_v
        else:
            q_s = torch.cat([
                torch.bmm(qk_inp_repeated[:, :, :self.d_content], self.w_qs1),
                torch.bmm(qk_inp_repeated[:, :, self.d_content:], self.w_qs2),
            ], -1)
            k_s = torch.cat([
                torch.bmm(qk_inp_repeated[:, :, :self.d_content], self.w_ks1),
                torch.bmm(qk_inp_repeated[:, :, self.d_content:], self.w_ks2),
            ], -1)
            v_s = torch.cat([
                torch.bmm(v_inp_repeated[:, :, :self.d_content], self.w_vs1),
                torch.bmm(v_inp_repeated[:, :, self.d_content:], self.w_vs2),
            ], -1)
        return q_s, k_s, v_s

    def pad_and_rearrange(self, q_s, k_s, v_s, batch_idxs):
        # Input is padded representation: n_head x len_inp x d
        # Output is packed representation: (n_head * mb_size) x len_padded x d
        # (along with masks for the attention and output)
        n_head = self.n_head
        d_k, d_v = self.d_k, self.d_v

        len_padded = batch_idxs.max_len
        mb_size = batch_idxs.batch_size
        q_padded = q_s.new_zeros((n_head, mb_size, len_padded, d_k))
        k_padded = k_s.new_zeros((n_head, mb_size, len_padded, d_k))
        v_padded = v_s.new_zeros((n_head, mb_size, len_padded, d_v))
        invalid_mask = q_s.new_ones((mb_size, len_padded), dtype=torch.bool)

        for i, (start, end) in enumerate(
                zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
            q_padded[:, i, :end-start, :] = q_s[:, start:end, :]
            k_padded[:, i, :end-start, :] = k_s[:, start:end, :]
            v_padded[:, i, :end-start, :] = v_s[:, start:end, :]
            invalid_mask[i, :end-start].fill_(False)

        return(
            q_padded.view(-1, len_padded, d_k),
            k_padded.view(-1, len_padded, d_k),
            v_padded.view(-1, len_padded, d_v),
            invalid_mask.unsqueeze(1).expand(mb_size, len_padded, len_padded).repeat(n_head, 1, 1),
            (~invalid_mask).repeat(n_head, 1),
        )

    def combine_v(self, outputs):
        # Combine attention information from the different heads
        n_head = self.n_head
        outputs = outputs.view(n_head, -1, self.d_v)

        if not self.partitioned:
            # Switch from n_head x len_inp x d_v to len_inp x (n_head * d_v)
            outputs = torch.transpose(outputs, 0, 1).contiguous().view(-1, n_head * self.d_v)

            # Project back to residual size
            outputs = self.proj(outputs)
        else:
            d_v1 = self.d_v // 2
            outputs1 = outputs[:, :, :d_v1]
            outputs2 = outputs[:, :, d_v1:]
            outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(-1, n_head * d_v1)
            outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(-1, n_head * d_v1)
            outputs = torch.cat([
                self.proj1(outputs1),
                self.proj2(outputs2),
            ], -1)

        return outputs

    def forward(self, inp, batch_idxs, qk_inp=None):
        residual = inp

        # While still using a packed representation, project to obtain the
        # query/key/value for each head
        q_s, k_s, v_s = self.split_qkv_packed(inp, qk_inp=qk_inp)

        # Switch to padded representation, perform attention, then switch back
        q_padded, k_padded, v_padded, attn_mask, output_mask = self.pad_and_rearrange(
            q_s, k_s, v_s, batch_idxs)

        outputs_padded, attns_padded = self.attention(
            q_padded, k_padded, v_padded,
            attn_mask=attn_mask,
        )
        outputs = outputs_padded[output_mask]
        outputs = self.combine_v(outputs)

        outputs = self.residual_dropout(outputs, batch_idxs)

        return self.layer_norm(outputs + residual), attns_padded


class PositionwiseFeedForward(nn.Module):
    """
    A position-wise feed forward module.

    Projects to a higher-dimensional space before applying ReLU, then projects
    back.
    """

    def __init__(self, d_hid, d_ff, relu_dropout=0.1, residual_dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_hid, d_ff)
        self.w_2 = nn.Linear(d_ff, d_hid)

        self.layer_norm = LayerNormalization(d_hid)
        # The t2t code on github uses relu dropout, even though the transformer
        # paper describes residual dropout only. We implement relu dropout
        # because we always have the option to set it to zero.
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()

    def forward(self, x, batch_idxs):
        residual = x

        output = self.w_1(x)
        output = self.relu_dropout(self.relu(output), batch_idxs)
        output = self.w_2(output)

        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)


class PartitionedPositionwiseFeedForward(nn.Module):
    def __init__(self, d_hid, d_ff, d_positional, relu_dropout=0.1, residual_dropout=0.1):
        super().__init__()
        self.d_content = d_hid - d_positional
        self.w_1c = nn.Linear(self.d_content, d_ff//2)
        self.w_1p = nn.Linear(d_positional, d_ff//2)
        self.w_2c = nn.Linear(d_ff//2, self.d_content)
        self.w_2p = nn.Linear(d_ff//2, d_positional)
        self.layer_norm = LayerNormalization(d_hid)
        # The t2t code on github uses relu dropout, even though the transformer
        # paper describes residual dropout only. We implement relu dropout
        # because we always have the option to set it to zero.
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()

    def forward(self, x, batch_idxs):
        residual = x
        xc = x[:, :self.d_content]
        xp = x[:, self.d_content:]

        outputc = self.w_1c(xc)
        outputc = self.relu_dropout(self.relu(outputc), batch_idxs)
        outputc = self.w_2c(outputc)

        outputp = self.w_1p(xp)
        outputp = self.relu_dropout(self.relu(outputp), batch_idxs)
        outputp = self.w_2p(outputp)

        output = torch.cat([outputc, outputp], -1)

        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)


class MultiLevelEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings_list,
                 d_embedding,
                 d_positional=None,
                 max_len=300,
                 normalize=True,
                 dropout=0.1,
                 timing_dropout=0.0,
                 emb_dropouts_list=None,
                 extra_content_dropout=None,
                 **kwargs):
        super().__init__()

        self.d_embedding = d_embedding
        self.partitioned = d_positional is not None

        if self.partitioned:
            self.d_positional = d_positional
            self.d_content = self.d_embedding - self.d_positional
        else:
            self.d_positional = self.d_embedding
            self.d_content = self.d_embedding

        if emb_dropouts_list is None:
            emb_dropouts_list = [0.0] * len(num_embeddings_list)
        assert len(emb_dropouts_list) == len(num_embeddings_list)

        embs = []
        emb_dropouts = []
        for i, (num_embeddings, emb_dropout) in enumerate(
                zip(num_embeddings_list, emb_dropouts_list)):
            emb = nn.Embedding(num_embeddings, self.d_content, **kwargs)
            embs.append(emb)
            emb_dropout = FeatureDropout(emb_dropout)
            emb_dropouts.append(emb_dropout)
        self.embs = nn.ModuleList(embs)
        self.emb_dropouts = nn.ModuleList(emb_dropouts)

        if extra_content_dropout is not None:
            self.extra_content_dropout = FeatureDropout(extra_content_dropout)
        else:
            self.extra_content_dropout = None

        if normalize:
            self.layer_norm = LayerNormalization(d_embedding)
        else:
            self.layer_norm = lambda x: x

        self.dropout = FeatureDropout(dropout)
        self.timing_dropout = FeatureDropout(timing_dropout)

        # Learned embeddings
        self.position_table = nn.Parameter(torch.FloatTensor(max_len, self.d_positional))
        init.normal_(self.position_table)

    def forward(self, xs, batch_idxs, extra_content_annotations=None):
        content_annotations = [
            emb_dropout(emb(x), batch_idxs)
            for x, emb, emb_dropout in zip(xs, self.embs, self.emb_dropouts)
        ]
        content_annotations = sum(content_annotations)
        if extra_content_annotations is not None:
            if self.extra_content_dropout is not None:
                content_annotations += self.extra_content_dropout(
                    extra_content_annotations, batch_idxs)
            else:
                content_annotations += extra_content_annotations

        timing_signal = torch.cat([self.position_table[:seq_len, :]
                                  for seq_len in batch_idxs.seq_lens_np], dim=0)
        timing_signal = self.timing_dropout(timing_signal, batch_idxs)

        # Combine the content and timing signals
        if self.partitioned:
            annotations = torch.cat([content_annotations, timing_signal], 1)
        else:
            annotations = content_annotations + timing_signal

        # TODO(nikita): reconsider the use of layernorm here
        annotations = self.layer_norm(self.dropout(annotations, batch_idxs))

        return annotations, timing_signal, batch_idxs


class Encoder(nn.Module):
    def __init__(self, embedding,
                 num_layers=1, num_heads=2, d_kv=32, d_ff=1024,
                 d_positional=None,
                 num_layers_position_only=0,
                 relu_dropout=0.1, residual_dropout=0.1, attention_dropout=0.1):
        super().__init__()
        # Don't assume ownership of the embedding as a submodule.
        # TODO(nikita): what's the right thing to do here?
        self.embedding_container = [embedding]
        d_model = embedding.d_embedding

        d_k = d_v = d_kv

        self.stacks = []
        for i in range(num_layers):
            attn = MultiHeadAttention(
                num_heads,
                d_model,
                d_k,
                d_v,
                residual_dropout=residual_dropout,
                attention_dropout=attention_dropout,
                d_positional=d_positional)
            if d_positional is None:
                ff = PositionwiseFeedForward(
                    d_model,
                    d_ff,
                    relu_dropout=relu_dropout,
                    residual_dropout=residual_dropout)
            else:
                ff = PartitionedPositionwiseFeedForward(
                    d_model,
                    d_ff,
                    d_positional,
                    relu_dropout=relu_dropout,
                    residual_dropout=residual_dropout)

            self.add_module(f"attn_{i}", attn)
            self.add_module(f"ff_{i}", ff)
            self.stacks.append((attn, ff))

        self.num_layers_position_only = num_layers_position_only
        if self.num_layers_position_only > 0:
            assert d_positional is None, "num_layers_position_only and partitioned are incompatible"

    def forward(self, xs, batch_idxs, extra_content_annotations=None):
        emb = self.embedding_container[0]
        res, timing_signal, batch_idxs = emb(
            xs, batch_idxs, extra_content_annotations=extra_content_annotations)

        for i, (attn, ff) in enumerate(self.stacks):
            if i >= self.num_layers_position_only:
                res, current_attns = attn(res, batch_idxs)
            else:
                res, current_attns = attn(res, batch_idxs, qk_inp=timing_signal)
            res = ff(res, batch_idxs)

        return res, batch_idxs
