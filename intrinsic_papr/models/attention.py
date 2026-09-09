import math

import torch
import torch.nn as nn
from torch import autocast
from torch.nn.utils import weight_norm

from .mlp import MLP
from .activations import activation_func
from .encodings import PoseEnc


def get_transformer(
    args,
    seq_len,
    v_extra_dim=0,
    k_extra_dim=0,
    q_extra_dim=0,
    eps=1e-6,
    use_amp=False,
    amp_dtype=torch.float16,
    albedo_value_MLP_input_portion=0.5,
    albedo_value_MLP_output_portion=0.5,
):
    k_dim_map = {
        1: [3, 3, 3],
    }
    k_dim = k_dim_map[args.k_type]

    q_dim_map = {
        1: [3],
    }
    q_dim = q_dim_map[args.q_type]

    v_dim_map = {
        1: [3, 3],
    }
    v_dim = v_dim_map[args.v_type]

    return Transformer(
        d_k=k_dim,
        d_q=q_dim,
        d_v=v_dim,
        d_model=args.d_model,
        d_out=args.d_out,
        seq_len=seq_len,
        embed_args=args.embed,
        block_args=args.block,
        d_ko=k_extra_dim,
        d_qo=q_extra_dim,
        d_vo=v_extra_dim,
        eps=eps,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        use_double_value_MLP=args.use_double_value_MLP,
        albedo_value_MLP_input_portion=albedo_value_MLP_input_portion,
        albedo_value_MLP_output_portion=albedo_value_MLP_output_portion,
    )


class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def attention(query, key):
    """
    Compute scaled dot-product attention scores
    query: [batch_size, n_heads, query_len, d_kq] or [batch_size, query_len, d_kq]
    key:   [batch_size, n_heads, seq_len, d_kq] or [batch_size, seq_len, d_kq]
    """
    d_kq = query.size(-1)
    return torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_kq)


class FeedForward(nn.Module):
    "Implements FFN module."

    def __init__(
        self,
        d_input,
        d_output,
        d_ff,
        n_layer=2,
        act="relu",
        last_act="none",
        dropout=0.1,
        norm="layernorm",
        residual=True,
        eps=1e-6,
        skip_layers=[],
        half_layers=[],
    ):
        super(FeedForward, self).__init__()
        self.eps = eps
        self.d_input = d_input
        self.d_output = d_output
        if norm == "layernorm":
            self.innorm = LayerNorm(d_input, eps)
            self.outnorm = LayerNorm(d_output, eps)
        elif norm == "none":
            self.innorm = nn.Identity()
            self.outnorm = nn.Identity()
        else:
            raise ValueError("Invalid Transformer norm type")
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLP(
            d_input,
            n_layer,
            d_ff,
            d_output,
            act_type=act,
            last_act_type=last_act,
            skip_layers=skip_layers,
            half_layers=half_layers,
        )
        self.residual = residual

    def forward(self, x):
        if self.residual and x.shape[-1] == self.d_output:
            return self.outnorm(x + self.dropout(self.mlp(self.innorm(x))))
        else:
            return self.outnorm(self.dropout(self.mlp(self.innorm(x))))



def _feed_forward(d_input, d_output, args, eps):
    """A FeedForward whose shape comes from one embedding's config block.

    ``args`` is the config for the key, query, value or shared embedding; the
    input and output widths are passed in because they depend on what the
    embedding is fed.
    """
    return FeedForward(
        d_input,
        d_output,
        args.d_ff,
        args.n_ff_layer,
        args.ff_act,
        args.ff_last_act,
        args.dropout_ff,
        args.norm,
        args.residual_ff,
        eps,
        args.skip_layers,
        args.half_layers,
    )

class Embeddings(nn.Module):

    def __init__(
        self,
        d_k,
        d_q,
        d_v,
        d_model,
        seq_len,
        args,
        use_double_value_MLP,
        albedo_value_MLP_input_portion,
        albedo_value_MLP_output_portion,
        d_ko=0,
        d_qo=0,
        d_vo=0,
        eps=1e-6,
    ):
        super(Embeddings, self).__init__()
        self.d_k = d_k
        self.d_q = d_q
        self.d_v = d_v
        self.use_double_value_MLP = use_double_value_MLP
        self.seq_len = seq_len
        self.args = args
        self.d_model = d_model
        self.share_embed = args.share_embed
        self.d_ko = d_ko
        self.d_qo = d_qo
        self.d_vo = d_vo
        self.eps = eps
        self.albedo_value_MLP_input_portion = albedo_value_MLP_input_portion
        self.albedo_value_MLP_output_portion = albedo_value_MLP_output_portion

        self.posenc = PoseEnc(args.pe_factor, args.pe_mult_factor)

        if args.pe_type == "none":
            self.positional_emb = None
        else:
            raise ValueError(
                "Unknown positional embedding type: {}".format(args.pe_type)
            )

        if self.share_embed:
            assert d_k == d_q == d_v

        d_k = sum([d + d * 2 * args.k_L[i] for i, d in enumerate(d_k)]) + d_ko
        d_q = sum([d + d * 2 * args.q_L[i] for i, d in enumerate(d_q)]) + d_qo
        if self.use_double_value_MLP:
            d_v_shading = d_v_albedo = sum(
                [d + d * 2 * args.v_L[i] for i, d in enumerate(d_v)]
            )
        else:
            d_v = sum([d + d * 2 * args.v_L[i] for i, d in enumerate(d_v)]) + d_vo

        if self.use_double_value_MLP:
            d_vo_albedo = int(d_vo * self.albedo_value_MLP_input_portion)
            d_vo_shading = d_vo - d_vo_albedo
            self.dim_point_feat_MLP_1_shading = d_vo_shading
            self.dim_point_feat_MLP_2_albedo = d_vo_albedo
            d_v_shading = d_v_shading + d_vo_shading
            d_v_albedo = d_v_albedo + d_vo_albedo
            d_v_output_albedo = int(
                args.value.d_ff_out * self.albedo_value_MLP_output_portion
            )
            d_v_output_shading = args.value.d_ff_out - d_v_output_albedo
        else:
            d_vo_albedo = int(d_vo * self.albedo_value_MLP_input_portion)
            d_vo_shading = d_vo - d_vo_albedo
            self.dim_point_feat_MLP_1_shading = d_vo_shading
            self.dim_point_feat_MLP_2_albedo = d_vo_albedo

        if self.share_embed:
            self.embed = _feed_forward(d_k, args.d_ff_out, args, eps)
        else:
            self.embed_k = _feed_forward(d_k, args.key.d_ff_out, args.key, eps)
            self.embed_q = _feed_forward(d_q, args.query.d_ff_out, args.query, eps)
            if self.use_double_value_MLP:
                self.embed_v_1 = _feed_forward(
                    d_v_shading, d_v_output_shading, args.value, eps
                )
                self.embed_v_2_albedo = _feed_forward(
                    d_v_albedo, d_v_output_albedo, args.value, eps
                )
            else:
                self.embed_v = _feed_forward(d_v, args.value.d_ff_out, args.value, eps)

    def forward(
        self,
        k_features,
        q_features,
        v_features,
        k_other=None,
        q_other=None,
        v_other=None,
    ):
        """
        k_features: [(B, H, W, N, Dk_i)]
        q_features: [(B, H, W, 1, Dq_i)]
        v_features: [(B, H, W, N, Dv_i)]
        v_other is 64 dim points features -> we need to split it into two 32 dim features in
        order to feed it to the two MLPs
        """
        # we only apply positional encoding to the first feature, not the points features
        # so we don't need to split the points features here
        pe_k_features = [
            self.posenc(f, self.args.k_L[i]) for i, f in enumerate(k_features)
        ]
        pe_q_features = [
            self.posenc(f, self.args.q_L[i]) for i, f in enumerate(q_features)
        ]
        pe_v_features = [
            self.posenc(f, self.args.v_L[i]) for i, f in enumerate(v_features)
        ]

        if self.d_ko > 0:
            pe_k_features = pe_k_features + k_other
        if self.d_qo > 0:
            pe_q_features = pe_q_features + q_other
        if self.d_vo > 0:
            # we need to split two the two MLPs
            if self.use_double_value_MLP:
                v_other_1_shading = [
                    v[:, :, :, :, : self.dim_point_feat_MLP_1_shading]
                    for v in v_other
                ]
                v_other_2_albedo = [
                    v[:, :, :, :, self.dim_point_feat_MLP_1_shading :]
                    for v in v_other
                ]
                pe_v_features_shading = pe_v_features + v_other_1_shading
                pe_v_features_albedo = pe_v_features + v_other_2_albedo
            else:
                pe_v_features = pe_v_features + v_other

        k = torch.cat(pe_k_features, dim=-1).flatten(0, 2)
        q = torch.cat(pe_q_features, dim=-1).flatten(0, 2)
        if self.use_double_value_MLP:
            v_shading = torch.cat(pe_v_features_shading, dim=-1).flatten(0, 2)
            v_albedo = torch.cat(pe_v_features_albedo, dim=-1).flatten(0, 2)
        else:
            v = torch.cat(pe_v_features, dim=-1).flatten(0, 2)

        if self.share_embed:
            k = self.embed(k)
            q = self.embed(q)
            v = self.embed(v)
        else:
            k = self.embed_k(k)
            q = self.embed_q(q)
            if self.use_double_value_MLP:
                v_shading = self.embed_v_1(v_shading)
                v_albedo = self.embed_v_2_albedo(v_albedo)
                v = torch.cat([v_shading, v_albedo], dim=-1)
            else:
                v = self.embed_v(v)

        if self.positional_emb is not None:
            k = k + self.positional_emb
            q = q + self.positional_emb
            v = v + self.positional_emb

        return k, q, v


class TransformerBlock(nn.Module):
    def __init__(self, d_k, d_q, d_v, d_model, d_out, args, eps=1e-6):
        super(TransformerBlock, self).__init__()
        self.n_head = args.n_head
        self.d_model = d_model
        assert self.d_model % self.n_head == 0
        self.d_mid = self.d_model // self.n_head
        self.eps = eps
        self.args = args

        self.temperature = args.temperature
        self.dropout_attn = nn.Dropout(p=args.dropout_attn)
        self.residual_attn = args.residual_attn
        self.residual_ff = args.residual_ff

        self.w_k = nn.Linear(d_k, self.d_model)
        self.w_q = nn.Linear(d_q, self.d_model)
        self.w_v = nn.Linear(d_v, self.d_model)
        self.w_o = nn.Linear(self.d_model, args.d_ff)

        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)

        self.ff = FeedForward(
            args.d_ff,
            d_out,
            args.d_ff,
            args.n_ff_layer,
            args.ff_act,
            args.ff_last_act,
            args.dropout_ff,
            args.norm,
            self.residual_ff,
            eps,
        )
        # Weight norm is applied here and nowhere else in the model. The released
        # checkpoints were trained with it on the block feed-forward, so they store
        # transformer.blocks.*.ff.mlp.model.*.weight_g / weight_v and only load if
        # it stays on. Every other MLP is built without weight norm.
        for i, layer in enumerate(self.ff.mlp.model):
            if isinstance(layer, nn.Linear):
                self.ff.mlp.model[i] = weight_norm(layer, name="weight")

        self.score_act = activation_func(self.args.score_act)
        self.k_act = activation_func(self.args.k_act)
        self.q_act = activation_func(self.args.q_act)

    def forward(self, key, query, value, score_only=False):
        nbatches, nseqv, _ = value.shape
        _, nseqk, _ = key.shape
        _, nseqq, _ = query.shape
        assert nseqv == nseqk

        self.attn = None

        if self.args.transform_kq:
            key = self.w_k(key)
            query = self.w_q(query)

        key = self.k_act(key)
        query = self.q_act(query)

        key = key.view(nbatches, -1, self.n_head, self.d_mid).transpose(1, 2)
        query = query.view(nbatches, -1, self.n_head, self.d_mid).transpose(1, 2)

        # [nbatches, nhead, nseq, nseq]
        scores = attention(query, key)
        scores = self.score_act(scores)

        if score_only:
            return value, scores

        x = self.w_v(value)
        x = x.view(nbatches, -1, self.n_head, self.d_mid).transpose(1, 2)

        attn = (scores * self.temperature).softmax(dim=-1)
        attn = self.dropout_attn(attn)
        self.attn = attn[..., :nseqq, :nseqk]

        x = torch.matmul(attn, x)
        x = self.w_o(x.transpose(1, 2).contiguous().view(nbatches, -1, self.d_model))

        if self.residual_attn and x.shape == value.shape:
            x = x + value
        x = self.ff(x)

        return x, scores


class Transformer(nn.Module):

    def __init__(
        self,
        d_k,
        d_q,
        d_v,
        d_model,
        d_out,
        seq_len,
        embed_args,
        block_args,
        use_double_value_MLP,
        albedo_value_MLP_input_portion,
        albedo_value_MLP_output_portion,
        d_ko=0,
        d_qo=0,
        d_vo=0,
        eps=1e-6,
        use_amp=False,
        amp_dtype=torch.float16,
    ):
        super(Transformer, self).__init__()
        self.eps = eps
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.use_double_value_MLP = use_double_value_MLP

        # we need to specify the extra dimension for the value and take an extra value
        self.embed = Embeddings(
            d_k=d_k,
            d_q=d_q,
            d_v=d_v,
            d_model=d_model,
            seq_len=seq_len,
            args=embed_args,
            use_double_value_MLP=use_double_value_MLP,
            albedo_value_MLP_input_portion=albedo_value_MLP_input_portion,
            albedo_value_MLP_output_portion=albedo_value_MLP_output_portion,
                d_ko=d_ko,
            d_qo=d_qo,
            d_vo=d_vo,
            eps=eps,
        )

        blocks = []
        for i in range(block_args.n_block):
            if i == block_args.n_block - 1:
                if embed_args.share_embed:
                    blocks.append(
                        TransformerBlock(
                            embed_args.d_ff_out,
                            embed_args.d_ff_out,
                            embed_args.d_ff_out,
                            d_model,
                            d_out,
                            block_args,
                            eps,
                        )
                    )
                else:
                    blocks.append(
                        TransformerBlock(
                            embed_args.key.d_ff_out,
                            embed_args.query.d_ff_out,
                            embed_args.value.d_ff_out,
                            d_model,
                            d_out,
                            block_args,
                            eps,
                        )
                    )
            else:
                blocks.append(
                    TransformerBlock(
                        d_out, d_out, d_out, d_model, d_model, block_args, eps
                    )
                )
        self.blocks = nn.Sequential(*blocks)

    def forward(
        self,
        k_features,
        q_features,
        v_features,
        k_other=None,
        q_other=None,
        v_other=None,
    ):
        """
        k_features: [(H, W, N, Dk_i)]
        q_features: [(H, W, 1, Dq_i)] or [(H, W, N, Dq_i)]
        v_features: [(H, W, N, Dv_i)]
        """
        score_only = len(self.blocks) == 1

        with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            # we need to feed the half of v_other as another value to the transformer if we
            # want to use double MLP
            k, q, v = self.embed(
                k_features, q_features, v_features, k_other, q_other, v_other
            )
            # the final value is the concatenation of the two value MLP
            for i, block in enumerate(self.blocks):
                if i == 0:
                    x, scores = block(k, q, v, score_only=score_only)
                else:
                    x, scores = block(x, x, x, score_only=score_only)

            return k, q, v, x, scores

