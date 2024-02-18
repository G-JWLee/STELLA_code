import torch.nn as nn
import torch
from models import heads


class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=12, norm_layer=nn.LayerNorm, pretrain_path="",
                 module_name='avmatching_module', **kwargs):
        super().__init__()
        self.norm_v = norm_layer(dim)
        self.norm_a = norm_layer(dim)
        self.dim = dim

        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.to_v_q = nn.Linear(dim, all_head_dim, bias=False)
        self.to_v_k = nn.Linear(dim, all_head_dim, bias=False)
        self.to_v_v = nn.Linear(dim, all_head_dim, bias=False)
        self.to_a_q = nn.Linear(dim, all_head_dim, bias=False)
        self.to_a_k = nn.Linear(dim, all_head_dim, bias=False)
        self.to_a_v = nn.Linear(dim, all_head_dim, bias=False)

        self.matching_score = heads.MatchingHead(hidden_size=dim * 2)
        self.init_weights()

        if pretrain_path != "":
            state_dict = torch.load(pretrain_path, map_location="cpu")
            state_dict = {".".join(k.split(".")[2:]): v for k, v in state_dict.items() if
                          k.startswith(f'module.{module_name}')}
            assert len(state_dict) != 0
            self.load_state_dict(state_dict)
            print("load init avm module weight from", pretrain_path)

    def init_weights(self,):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_a, x_v):
        B, num_patches_v, _ = x_v.shape
        B, num_patches_a, _ = x_a.shape

        x_v = self.norm_v(x_v)
        x_a = self.norm_a(x_a)

        v_q = self.to_v_q(x_v).reshape(B, num_patches_v, self.num_heads, -1).permute(0,2,1,3)
        v_k = self.to_v_k(x_v).reshape(B, num_patches_v, self.num_heads, -1).permute(0,2,1,3)
        v_v = self.to_v_v(x_v).reshape(B, num_patches_v, self.num_heads, -1).permute(0,2,1,3)
        a_q = self.to_a_q(x_a).reshape(B, num_patches_a, self.num_heads, -1).permute(0,2,1,3)
        a_k = self.to_a_k(x_a).reshape(B, num_patches_a, self.num_heads, -1).permute(0,2,1,3)
        a_v = self.to_a_v(x_a).reshape(B, num_patches_a, self.num_heads, -1).permute(0,2,1,3)

        cross_attn_av = (a_q @ v_k.transpose(-2, -1)) * self.scale
        cross_attn_va = (v_q @ a_k.transpose(-2, -1)) * self.scale

        # Perform Softmax normalization on row-wise
        cross_attn_av = cross_attn_av.softmax(dim=-1)
        cross_attn_va = cross_attn_va.softmax(dim=-1)

        x_a = (cross_attn_av @ v_v).reshape(B, num_patches_a, -1)
        x_v = (cross_attn_va @ a_v).reshape(B, num_patches_v, -1)
        x_av = torch.cat([torch.mean(x_v, dim=1), torch.mean(x_a, dim=1)], dim=1)

        av_logits = self.matching_score(x_av)

        return av_logits

    def infer_attention(self, x_a, x_v, compute_av_positive, compute_va_positive, normalize=True, temperature=1.0):
        B, num_patches_v, D = x_v.shape
        B, num_patches_a, D = x_a.shape

        x_v = self.norm_v(x_v)
        x_a = self.norm_a(x_a)

        if compute_av_positive:
            a_q = self.to_a_q(x_a).reshape(B, num_patches_a, self.num_heads, -1).permute(0, 2, 1, 3)
            v_k = self.to_v_k(x_v).reshape(B, num_patches_v, self.num_heads, -1).permute(0, 2, 1, 3)
            cross_attn_av = (a_q @ v_k.transpose(-2, -1)) * self.scale
            if normalize:
                cross_attn_av = (cross_attn_av / temperature).softmax(dim=-1)
        else:
            a_q = v_k = cross_attn_av = None

        if compute_va_positive:
            v_q = self.to_v_q(x_v).reshape(B, num_patches_v, self.num_heads, -1).permute(0,2,1,3)
            a_k = self.to_a_k(x_a).reshape(B, num_patches_a, self.num_heads, -1).permute(0,2,1,3)
            cross_attn_va = (v_q @ a_k.transpose(-2, -1)) * self.scale
            if normalize:
                cross_attn_va = (cross_attn_va / temperature).softmax(dim=-1)
        else:
            v_q = a_k = cross_attn_va = None

        return cross_attn_av, cross_attn_va, v_q, v_k, a_q, a_k

