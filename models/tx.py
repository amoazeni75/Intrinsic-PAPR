import math

import torch
import torch.nn as nn
from torch import autocast
from torch.nn.utils import weight_norm

from .mlp import MLP
from .utils import PoseEnc, activation_func


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
    albedo_value_MLP_feature_side="right",
    shared_components=None,
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
        albedo_value_MLP_feature_side=albedo_value_MLP_feature_side,
        shared_components=shared_components,
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


class InstanceNorm(nn.Module):
    "Construct a InstanceNorm module"

    def __init__(self, eps=1e-6):
        super(InstanceNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(0, keepdim=True)
        std = x.std(0, keepdim=True)
        return (x - mean) / (std + self.eps)


def attention(query, key, kernel_type):
    """
    Compute Attention Scores
    query: [batch_size, n_heads, query_len, d_kq] or [batch_size, query_len, d_kq]
    key:   [batch_size, n_heads, seq_len, d_kq] or [batch_size, seq_len, d_kq]
    """
    d_kq = query.size(-1)

    if kernel_type == "scaled-dot":
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_kq)
    else:
        raise ValueError("Unknown kernel type: {}".format(kernel_type))

    return scores


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
        act_a=1.0,
        act_b=1.0,
        act_trainable=False,
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
        elif norm == "instancenorm":
            self.innorm = InstanceNorm(eps)
            self.outnorm = InstanceNorm(eps)
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
            a=act_a,
            b=act_b,
            trainable=act_trainable,
            skip_layers=skip_layers,
            half_layers=half_layers,
        )
        self.residual = residual

    def forward(self, x):
        if self.residual and x.shape[-1] == self.d_output:
            return self.outnorm(x + self.dropout(self.mlp(self.innorm(x))))
        else:
            return self.outnorm(self.dropout(self.mlp(self.innorm(x))))


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
        albedo_value_MLP_feature_side,
        d_ko=0,
        d_qo=0,
        d_vo=0,
        eps=1e-6,
        shared_components=None,
    ):
        super(Embeddings, self).__init__()
        self.d_k = d_k
        self.d_q = d_q
        self.d_v = d_v
        # print with red color that we are using double value MLP
        if use_double_value_MLP:
            print("\033[91m" + "Using double value MLP" + "\033[0m")
        self.use_double_value_MLP = use_double_value_MLP
        self.seq_len = seq_len
        self.args = args
        self.embed_type = args.embed_type
        self.d_model = d_model
        self.share_embed = args.share_embed
        self.d_ko = d_ko
        self.d_qo = d_qo
        self.d_vo = d_vo
        self.eps = eps
        self.albedo_value_MLP_input_portion = albedo_value_MLP_input_portion
        self.albedo_value_MLP_output_portion = albedo_value_MLP_output_portion
        self.albedo_value_MLP_feature_side = albedo_value_MLP_feature_side

        self.posenc = PoseEnc(args.pe_factor, args.pe_mult_factor)

        if args.pe_type == "none":
            self.positional_emb = None
        else:
            raise ValueError(
                "Unknown positional embedding type: {}".format(args.pe_type)
            )

        if self.share_embed:
            assert d_k == d_q == d_v

        if self.embed_type == 1:
            # Positional Encoding with itself
            d_k = sum([d + d * 2 * args.k_L[i] for i, d in enumerate(d_k)]) + d_ko
            d_q = sum([d + d * 2 * args.q_L[i] for i, d in enumerate(d_q)]) + d_qo
            if self.use_double_value_MLP:
                d_v_shading = d_v_albedo = sum(
                    [d + d * 2 * args.v_L[i] for i, d in enumerate(d_v)]
                )
                print("#" * 80)
                print("\033[91m" + "fixed s,t dim: {}".format(d_v_shading) + "\033[0m")
                print("#" * 80)
            else:
                d_v = sum([d + d * 2 * args.v_L[i] for i, d in enumerate(d_v)]) + d_vo

        elif self.embed_type == 2:
            # Positional Encoding without itself
            d_k = sum([d * 2 * args.k_L[i] for i, d in enumerate(d_k)]) + d_ko
            d_q = sum([d * 2 * args.q_L[i] for i, d in enumerate(d_q)]) + d_qo
            if self.use_double_value_MLP:
                d_v_shading = d_v_albedo = sum(
                    [d * 2 * args.v_L[i] for i, d in enumerate(d_v)]
                )
                print("#" * 80)
                print("\033[91m" + "fixed s,t dim".format(d_v_shading) + "\033[0m")
                print("#" * 80)
            else:
                d_v = sum([d * 2 * args.v_L[i] for i, d in enumerate(d_v)]) + d_vo

        else:
            raise ValueError("Unknown embedding type: {}".format(self.embed_type))

        if self.use_double_value_MLP:
            d_vo_albedo = int(d_vo * self.albedo_value_MLP_input_portion)
            d_vo_shading = d_vo - d_vo_albedo
            self.dim_point_feat_MLP_1_shading = d_vo_shading
            self.dim_point_feat_MLP_2_albedo = d_vo_albedo
            d_v_shading = d_v_shading + d_vo_shading
            d_v_albedo = d_v_albedo + d_vo_albedo
            print("#" * 80)
            print(
                "\033[91m"
                + "Albedo value MLP (double MLP) inp portion: {}, Albedo value MLP (double MLP) out portion: {}".format(
                    self.albedo_value_MLP_input_portion,
                    self.albedo_value_MLP_output_portion,
                )
                + "\033[0m"
            )
            print(
                "\033[91m"
                + "input dim value MLP 1: {}, input dim value MLP 2(albedo): {}".format(
                    d_v_shading, d_v_albedo
                )
                + "\033[0m"
            )
            d_v_output_albedo = int(
                args.value.d_ff_out * self.albedo_value_MLP_output_portion
            )
            d_v_output_shading = args.value.d_ff_out - d_v_output_albedo
            print(
                "\033[91m"
                + "output dim value MLP 1: {}, output dim value MLP 2(albedo): {}".format(
                    d_v_output_shading, d_v_output_albedo
                )
                + "\033[0m"
            )
            print("#" * 80)
        else:
            d_vo_albedo = int(d_vo * self.albedo_value_MLP_input_portion)
            d_vo_shading = d_vo - d_vo_albedo
            self.dim_point_feat_MLP_1_shading = d_vo_shading
            self.dim_point_feat_MLP_2_albedo = d_vo_albedo

        if self.share_embed:
            self.embed = FeedForward(
                d_k,
                args.d_ff_out,
                args.d_ff,
                args.n_ff_layer,
                args.ff_act,
                args.ff_last_act,
                args.dropout_ff,
                args.norm,
                args.residual_ff,
                args.ff_act_a,
                args.ff_act_b,
                args.ff_act_trainable,
                eps,
                args.skip_layers,
                args.half_layers,
            )
        else:
            self.embed_k = FeedForward(
                d_k,
                args.key.d_ff_out,
                args.key.d_ff,
                args.key.n_ff_layer,
                args.key.ff_act,
                args.key.ff_last_act,
                args.key.dropout_ff,
                args.key.norm,
                args.key.residual_ff,
                args.key.ff_act_a,
                args.key.ff_act_b,
                args.key.ff_act_trainable,
                eps,
                args.key.skip_layers,
                args.key.half_layers,
            )
            self.embed_q = FeedForward(
                d_q,
                args.query.d_ff_out,
                args.query.d_ff,
                args.query.n_ff_layer,
                args.query.ff_act,
                args.query.ff_last_act,
                args.query.dropout_ff,
                args.query.norm,
                args.query.residual_ff,
                args.query.ff_act_a,
                args.query.ff_act_b,
                args.query.ff_act_trainable,
                eps,
                args.query.skip_layers,
                args.query.half_layers,
            )
            if self.use_double_value_MLP:
                if (
                    shared_components is not None
                    and "shading_value_MLP" in shared_components
                ):
                    self.embed_v_1 = shared_components["shading_value_MLP"]
                    print(
                        "\033[91m"
                        + "***************** inside Embeddings class, we are using shared MLP as the first value MLP (shading)"
                        + "\033[0m"
                    )
                else:
                    self.embed_v_1 = FeedForward(
                        d_v_shading,
                        d_v_output_shading,
                        args.value.d_ff,
                        args.value.n_ff_layer,
                        args.value.ff_act,
                        args.value.ff_last_act,
                        args.value.dropout_ff,
                        args.value.norm,
                        args.value.residual_ff,
                        args.value.ff_act_a,
                        args.value.ff_act_b,
                        args.value.ff_act_trainable,
                        eps,
                        args.value.skip_layers,
                        args.value.half_layers,
                    )
                # print with red color that we are using double value MLP
                print(
                    "\033[91m"
                    + "***************** inside Embeddings class, we want to use double MLP"
                    + "\033[0m"
                )
                if (
                    shared_components is not None
                    and "albedo_value_MLP" in shared_components
                ):
                    self.embed_v_2_albedo = shared_components["albedo_value_MLP"]
                    print(
                        "\033[91m"
                        + "***************** inside Embeddings class, we are using shared MLP as the second value MLP (albedo)"
                        + "\033[0m"
                    )
                else:
                    self.embed_v_2_albedo = FeedForward(
                        d_v_albedo,
                        d_v_output_albedo,
                        args.value.d_ff,
                        args.value.n_ff_layer,
                        args.value.ff_act,
                        args.value.ff_last_act,
                        args.value.dropout_ff,
                        args.value.norm,
                        args.value.residual_ff,
                        args.value.ff_act_a,
                        args.value.ff_act_b,
                        args.value.ff_act_trainable,
                        eps,
                        args.value.skip_layers,
                        args.value.half_layers,
                    )
            else:
                self.embed_v = FeedForward(
                    d_v,
                    args.value.d_ff_out,
                    args.value.d_ff,
                    args.value.n_ff_layer,
                    args.value.ff_act,
                    args.value.ff_last_act,
                    args.value.dropout_ff,
                    args.value.norm,
                    args.value.residual_ff,
                    args.value.ff_act_a,
                    args.value.ff_act_b,
                    args.value.ff_act_trainable,
                    eps,
                    args.value.skip_layers,
                    args.value.half_layers,
                )

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
        if self.embed_type == 1:
            pe_k_features = [
                self.posenc(f, self.args.k_L[i]) for i, f in enumerate(k_features)
            ]
            pe_q_features = [
                self.posenc(f, self.args.q_L[i]) for i, f in enumerate(q_features)
            ]
            pe_v_features = [
                self.posenc(f, self.args.v_L[i]) for i, f in enumerate(v_features)
            ]

        elif self.embed_type == 2:
            pe_k_features = [
                self.posenc(f, self.args.k_L[i], without_self=True)
                for i, f in enumerate(k_features)
            ]
            pe_q_features = [
                self.posenc(f, self.args.q_L[i], without_self=True)
                for i, f in enumerate(q_features)
            ]
            pe_v_features = [
                self.posenc(f, self.args.v_L[i], without_self=True)
                for i, f in enumerate(v_features)
            ]

        else:
            raise ValueError("Unknown embedding type: {}".format(self.embed_type))

        if self.d_ko > 0:
            pe_k_features = pe_k_features + k_other
        if self.d_qo > 0:
            pe_q_features = pe_q_features + q_other
        if self.d_vo > 0:
            # we need to split two the two MLPs
            if self.use_double_value_MLP:
                if self.albedo_value_MLP_feature_side == "right":
                    v_other_1_shading = [
                        v[:, :, :, :, : self.dim_point_feat_MLP_1_shading]
                        for v in v_other
                    ]
                    v_other_2_albedo = [
                        v[:, :, :, :, self.dim_point_feat_MLP_1_shading :]
                        for v in v_other
                    ]
                elif self.albedo_value_MLP_feature_side == "left":
                    v_other_1_shading = [
                        v[:, :, :, :, self.dim_point_feat_MLP_2_albedo :]
                        for v in v_other
                    ]
                    v_other_2_albedo = [
                        v[:, :, :, :, : self.dim_point_feat_MLP_2_albedo]
                        for v in v_other
                    ]
                else:
                    raise ValueError(
                        "Unknown albedo value MLP feature side: {}".format(
                            self.albedo_value_MLP_feature_side
                        )
                    )
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

    def move_shared_albedo_value_MLP_to_device(self, device):
        if self.embed_v_2_albedo is not None:
            self.embed_v_2_albedo = self.embed_v_2_albedo.to(device)

    def move_shared_shading_value_MLP_to_device(self, device):
        if self.embed_v_1 is not None:
            self.embed_v_1 = self.embed_v_1.to(device)

    def get_shared_albedo_value_MLP(self):
        return self.embed_v_2_albedo

    def get_shared_shading_value_MLP(self):
        return self.embed_v_1

    def set_shared_albedo_value_MLP(self, shared_value_MLP):
        # we delete the old shared MLP and replace it with the new one
        del self.embed_v_2_albedo
        self.embed_v_2_albedo = shared_value_MLP
        # print with red color that we are using double value MLP
        print(
            "\033[91m"
            + "***************** inside Embeddings class, we recieved and set a new shared MLP"
            + "\033[0m"
        )

    def set_shared_shading_value_MLP(self, shared_value_MLP):
        # we delete the old shared MLP and replace it with the new one
        del self.embed_v_1
        self.embed_v_1 = shared_value_MLP
        # print with red color that we are using double value MLP
        print(
            "\033[91m"
            + "***************** inside Embeddings class, we recieved and set a new shared MLP"
            + "\033[0m"
        )


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
            args.ff_act_a,
            args.ff_act_b,
            args.ff_act_trainable,
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
        scores = attention(query, key, "scaled-dot")
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
        albedo_value_MLP_feature_side,
        d_ko=0,
        d_qo=0,
        d_vo=0,
        eps=1e-6,
        use_amp=False,
        amp_dtype=torch.float16,
        shared_components=None,
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
            albedo_value_MLP_feature_side=albedo_value_MLP_feature_side,
            d_ko=d_ko,
            d_qo=d_qo,
            d_vo=d_vo,
            eps=eps,
            shared_components=shared_components,
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
        step=-1,
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

    def move_shared_albedo_value_MLP_to_device(self, device):
        self.embed.move_shared_albedo_value_MLP_to_device(device)

    def get_shared_albedo_value_MLP(self):
        return self.embed.get_shared_albedo_value_MLP()

    def set_shared_albedo_value_MLP(self, shared_value_MLP):
        self.embed.set_shared_albedo_value_MLP(shared_value_MLP)

    def move_shared_shading_value_MLP_to_device(self, device):
        self.embed.move_shared_shading_value_MLP_to_device(device)

    def get_shared_shading_value_MLP(self):
        return self.embed.get_shared_shading_value_MLP()

    def set_shared_shading_value_MLP(self, shared_value_MLP):
        self.embed.set_shared_shading_value_MLP(shared_value_MLP)
