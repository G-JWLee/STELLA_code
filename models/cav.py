import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np

from timm.models.layers import drop_path, to_2tuple
from models import heads, objectives



def get_2d_sincos_pos_embed(embed_dim, grid_h_size, grid_w_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_w_size, grid_h_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



class CAV(nn.Module):

    def __init__(
            self, img_size=224, patch_size=16, audio_patch_size=[16, 16], hidden_size=768, num_classes=309,
            encoder_in_chans=3, encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
            decoder_embed_dim=512, decoder_hidden_size=512, decoder_depth=8, decoder_num_heads=8,
            mlp_ratio=4., norm_layer=nn.LayerNorm, loss_names=[], use_audio=False, mid_fusion_depth=10,
            frequency_size=128, audio_size=1024, num_frames=4,
            mask_ratio_v=0.8, mask_ratio_a=0.8, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            **kwargs):

        super().__init__()
        self.use_mae = "mae_audio" in loss_names or "mae_frame" in loss_names
        self.num_classes = num_classes # Number of classes for downstream classification task.
        self.encoder_depth = encoder_depth
        self.mid_fusion_depth = mid_fusion_depth
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.mask_comp_ratio_a = 0
        self.mask_comp_ratio_v = 0

        # Video-encoder, no cls token
        self.patch_embed_v = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            embed_dim=encoder_embed_dim)
        self.num_patches_v = self.patch_embed_v.num_patches * num_frames
        self.num_features = self.embed_dim = encoder_embed_dim
        self.num_frames = num_frames
        # Since time-wise positional embedding is all the same at initialization, make this learnable
        # In TVLT, they made this learnable either.
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.num_patches_v, encoder_embed_dim), requires_grad=True)
        self.inter_pos_embed_v = nn.Parameter(torch.zeros(1, self.num_patches_v, encoder_embed_dim), requires_grad=True)
        # In TVLT, CAV and such, they used modality embedding to specify input token modality
        self.modality_v = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        if not "vacls" in loss_names:
            # In CAV, since audio and video has different distribution, implemented different normalization layer
            self.norm_v = norm_layer(encoder_embed_dim)

        # Audio-encoder
        self.use_audio = use_audio
        self.frequency_size = frequency_size
        if use_audio:
            self.patch_embed_a = AudioPatchEmbed(
                patch_size=audio_patch_size,
                in_chans=1,
                embed_dim=encoder_embed_dim,
                frequency_size=frequency_size,
                audio_size=audio_size,
            )
            self.patch_hw = self.patch_embed_a.patch_hw
            self.num_patches_a = self.patch_embed_a.num_patches
            self.pos_embed_a = nn.Parameter(torch.zeros(1, self.num_patches_a, encoder_embed_dim), requires_grad=True)
            self.inter_pos_embed_a = nn.Parameter(torch.zeros(1, self.num_patches_a, encoder_embed_dim), requires_grad=True)
            self.freq_patch_size = self.frequency_size//audio_patch_size[0]
            self.modality_a = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
            if not "vacls" in loss_names:
                self.norm_a = norm_layer(encoder_embed_dim)

        shared = any(loss_name in ['vacls', 'mae_frame', 'mae_audio', 'vam', 'embedding'] for loss_name in loss_names)
        separate = any(loss_name in ['contrastive', 'retrieval'] for loss_name in loss_names)
        if shared:
            self.norm = norm_layer(encoder_embed_dim)

        # Decoder part
        if self.use_mae:
            self.mask_ratio_v = mask_ratio_v
            self.decoder_embed_dim = decoder_embed_dim
            # Encoder -> Decoder embedding dimension change
            self.decoder_embed = nn.Linear(
                encoder_embed_dim, decoder_embed_dim, bias=True)
            self.mask_token_v = nn.Parameter(
                torch.zeros(1, 1, decoder_embed_dim))
            # Fixed video positional embedding, like CAV
            self.decoder_pos_embed_v = nn.Parameter(
                torch.zeros(1, self.num_patches_v, decoder_embed_dim), requires_grad=True)
            self.decoder_modality_v = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.decoder_norm = norm_layer(decoder_embed_dim)
            if use_audio:
                self.mask_ratio_a = mask_ratio_a
                self.mask_token_a = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim))
                # Fixed audio positional embedding
                self.decoder_pos_embed_a = nn.Parameter(torch.zeros(
                    1, self.num_patches_a, decoder_embed_dim), requires_grad=True)
                self.decoder_modality_a = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.num_frames = num_frames

        # Encoder block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, encoder_depth)]

        self.blocks_v = nn.ModuleList([
            Block(
                dim=encoder_embed_dim, num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(mid_fusion_depth)])

        if use_audio:
            self.blocks_a = nn.ModuleList([
                Block(
                    dim=encoder_embed_dim, num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(mid_fusion_depth)])

        self.blocks_u = nn.ModuleList([
            Block(
                dim=encoder_embed_dim, num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                separate=separate, shared=shared)
            for i in range(mid_fusion_depth, encoder_depth)])

        # Decoder block, check other parameters if it is correct
        if self.use_mae:
            self.decoder_blocks = nn.ModuleList([
                Block(
                    dim=decoder_embed_dim, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(decoder_depth)])

        if "mae_audio" in loss_names:
            self.mae_score_audio = heads.MAEHead(
                decoder_hidden_size, audio_patch_size[0]*audio_patch_size[1])
            self.audio_patch_size = [audio_patch_size[0],audio_patch_size[1]]
            self.mae_score_audio.apply(objectives.init_weights)

        if "mae_frame" in loss_names:
            self.patch_size = [patch_size,patch_size]
            self.mae_score_video = heads.MAEHead(
                decoder_hidden_size, patch_size**2*3)
            self.mae_score_video.apply(objectives.init_weights)

        if "vacls" in loss_names:
            if use_audio:
                hidden_size_ = hidden_size * 2
            else:
                hidden_size_ = hidden_size
            self.vacls_classifier = heads.Classifier(hidden_size_, num_classes=num_classes)
            self.vacls_criterion = heads.OneHotCrossEntropyLoss()
            # self.vacls_criterion = nn.CrossEntropyLoss()
            self.vacls_classifier.apply(objectives.init_weights)

    def freeze_backbone(self):
        self.requires_grad_(requires_grad=False)
        if hasattr(self, 'vacls_classifier'):
            self.vacls_classifier.requires_grad_(requires_grad=True)

    def init_weights(self, ):
        # initialize (and freeze) pos_embed by sin-cos embedding, opt the cls token, add by myself
        pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        pos_embed_v = torch.from_numpy(pos_embed_v).float().unsqueeze(0)
        pos_embed_v = pos_embed_v.repeat(1, self.num_frames, 1)
        self.pos_embed_v.data.copy_(pos_embed_v)
        self.inter_pos_embed_v.data.copy_(pos_embed_v)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed_v.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        nn.init.normal_(self.modality_v, std=.02)

        if self.use_audio:
            pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], self.freq_patch_size,
                                                  int(self.patch_embed_a.num_patches // self.freq_patch_size), cls_token=False)
            pos_embed_a = torch.from_numpy(pos_embed_a).float().unsqueeze(0)
            self.pos_embed_a.data.copy_(pos_embed_a)
            self.inter_pos_embed_a.data.copy_(pos_embed_a)
            w = self.patch_embed_a.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.normal_(self.modality_a, std=.02)

        if self.use_mae:
            decoder_pos_embed_v = get_2d_sincos_pos_embed(self.decoder_pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
            decoder_pos_embed_v = torch.from_numpy(decoder_pos_embed_v).float().unsqueeze(0)
            decoder_pos_embed_v = decoder_pos_embed_v.repeat(1, self.num_frames, 1)
            self.decoder_pos_embed_v.data.copy_(decoder_pos_embed_v)
            nn.init.normal_(self.mask_token_v, std=.02)
            nn.init.normal_(self.decoder_modality_v, std=.02)

            if self.use_audio:
                decoder_pos_embed_a = get_2d_sincos_pos_embed(self.decoder_pos_embed_a.shape[-1], self.freq_patch_size,
                                                              int(self.patch_embed_a.num_patches // self.freq_patch_size), cls_token=False)
                decoder_pos_embed_a = torch.from_numpy(decoder_pos_embed_a).float().unsqueeze(0)
                self.decoder_pos_embed_a.data.copy_(decoder_pos_embed_a)
                nn.init.normal_(self.mask_token_a, std=.02)
                nn.init.normal_(self.decoder_modality_a, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed_v", "pos_embed_a", "inter_pos_embed_v", "inter_pos_embed_a", "modality_v", "modality_a", "decoder_modality_v", "decoder_modality_a",
                "mask_token_v", "mask_token_a", "decoder_pos_embed_v", "decoder_pos_embed_a",}

    def random_masking(self, x, mask_ratio=0.8):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape

        num_input_tokens = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is to remove

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :num_input_tokens]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is to keep, 1 is to remove
        input_mask = torch.ones([N, L], device=x.device)
        input_mask[:, :num_input_tokens] = 0

        target_mask = torch.ones([N, L], device=x.device)
        target_mask[:, num_input_tokens:] = 0
        # unshuffle to get the binary mask
        input_mask = torch.gather(input_mask, dim=1, index=ids_restore)
        target_mask = torch.gather(target_mask, dim=1, index=ids_restore)

        return x_masked, input_mask.bool(), target_mask.bool(), ids_restore, ids_keep

    def modality_guided_masking(self, cross_attn_ids_sorted, x, mask_ratio=0.8, comp_ratio=0.5, reconstruct_compressed=False):
        """
        Cross_attn_ids_sorted is already descending-order-sorted indices that indicate important indices with respect to other modality.
        Input x is already masked input.
        """
        N, L, D = x.shape
        if reconstruct_compressed:
            L = int(L / (1-comp_ratio))
            num_zeros = int(L * comp_ratio)
            zeros_data = torch.zeros(N, num_zeros, D, device=x.device)
            x = torch.cat([x, zeros_data], dim=1)
            cross_attn_ids_restore = torch.argsort(cross_attn_ids_sorted, dim=1)
            x = torch.gather(x, dim=1, index=cross_attn_ids_restore[:,:,None].repeat(1,1,D))

        num_compressed_tokens = int(L * (1 - comp_ratio))
        num_input_tokens = int(L * (1 - mask_ratio))
        ids_sorted = cross_attn_ids_sorted.clone()

        noise = torch.rand(N, num_compressed_tokens, device=x.device) # noise in [0, 1]
        noise_ids_shuffle = torch.argsort(noise, dim=1)

        ids_sorted[:,:num_compressed_tokens] = torch.gather(ids_sorted[:,:num_compressed_tokens], dim=1, index=noise_ids_shuffle)
        ids_restore = torch.argsort(ids_sorted, dim=1)

        # keep the first subset
        ids_keep = ids_sorted[:, :num_input_tokens]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        input_mask = torch.ones([N, L], device=x.device)
        input_mask[:, :num_input_tokens] = 0

        target_mask = torch.ones([N, L], device=x.device)
        target_mask[:, num_input_tokens:num_compressed_tokens] = 0

        input_mask = torch.gather(input_mask, dim=1, index=ids_restore)
        target_mask = torch.gather(target_mask, dim=1, index=ids_restore)

        return x_masked, input_mask.bool(), target_mask.bool(), ids_restore, ids_keep


    def cat_mask(self, mask_token, x, ids_restore):
        mask_tokens = mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        return x_

    def forward(self, audio=None, video=None, mask_audio=False, mask_visual=False, use_mae=False,
                compute_from_a_codes=False, compute_from_v_codes=False, ids_keep_a=None, ids_keep_v=None,
                compute_unimodal_embedding=False, compute_embedding=False, compute_joint_embedding=False,
                att_map_av_ids=None, att_map_va_ids=None):

        if not compute_from_a_codes:
            if audio is not None:
                if len(audio.shape) == 5 and att_map_va_ids is not None: # B x L_A x 1 x 16 x 16
                    B, T, C, H, W = audio.shape
                    x_a = self.patch_embed_a(audio.reshape(B*T, C, H, W))
                    x_a = x_a.reshape(B, T*x_a.size(1), x_a.size(-1))
                    x_a += self.modality_a
                    pos_embed_a = torch.gather(self.pos_embed_a.repeat(B,1,1), dim=1,
                                                 index=att_map_va_ids[:, :x_a.size(1)].unsqueeze(-1).repeat(1, 1, x_a.size(2)))
                    x_a += pos_embed_a
                    reconstruct_compressed_a = True
                else: # B x 1 x T x F
                    x_a = self.patch_embed_a(audio)
                    x_a += self.modality_a
                    x_a += self.pos_embed_a
                    reconstruct_compressed_a = False

                if mask_audio: # Using mask reconstruction loss.
                    if att_map_va_ids is not None:
                        x_a, input_mask_a, target_mask_a, ids_restore_a, ids_keep_a = self.modality_guided_masking(
                            att_map_va_ids, x_a, mask_ratio=self.mask_ratio_a, comp_ratio=self.mask_comp_ratio_a,
                            reconstruct_compressed=reconstruct_compressed_a,
                        )
                    else:
                        x_a, input_mask_a, target_mask_a, ids_restore_a, ids_keep_a = self.random_masking(
                            x_a, mask_ratio=self.mask_ratio_a)

                for b_idx, blk_a in enumerate(self.blocks_a):
                    x_a = blk_a(x_a)
        else:
            if audio is not None:
                x_a = audio.clone()

        if not compute_from_v_codes:

            if video is not None:
                B, T, C, H, W = video.shape
                x_v = self.patch_embed_v(video.reshape(B*T, C, H, W))
                x_v = x_v.reshape(B, T*x_v.size(1), x_v.size(-1))
                x_v += self.modality_v
                if x_v.size(1) != self.num_patches_v and att_map_av_ids is not None: # Compressed video, B x L_V' x 3 x 16 x 16
                    pos_embed_v = torch.gather(self.pos_embed_v.repeat(B, 1, 1), dim=1,
                                               index=att_map_av_ids[:, :x_v.size(1)].unsqueeze(-1).repeat(1, 1, x_v.size(2)))
                    x_v += pos_embed_v
                    reconstruct_compressed_v = True

                else:
                    x_v += self.pos_embed_v.repeat(B, 1, 1)[:,:x_v.size(1),:]
                    reconstruct_compressed_v = False

                if mask_visual:  # Using mask reconstruction loss.
                    if att_map_av_ids is not None:
                        x_v, input_mask_v, target_mask_v, ids_restore_v, ids_keep_v = self.modality_guided_masking(
                            att_map_av_ids, x_v, mask_ratio=self.mask_ratio_v, comp_ratio=self.mask_comp_ratio_v,
                            reconstruct_compressed=reconstruct_compressed_v,
                        )
                    else:
                        x_v, input_mask_v, target_mask_v, ids_restore_v, ids_keep_v = self.random_masking(
                            x_v, mask_ratio=self.mask_ratio_v)

                for b_idx, blk_v in enumerate(self.blocks_v):
                    x_v = blk_v(x_v)

        else:
            if video is not None:
                x_v = video.clone()

        # Compute video intermediate logits (cv_inter) and audio intermediate logits (ca_inter)
        if compute_unimodal_embedding:
            ca_inter = cv_inter = None
            if audio is not None:
                ca_inter = x_a.clone()
            if video is not None:
                cv_inter = x_v.clone()
            if (not compute_embedding) and (not compute_joint_embedding) and (not use_mae):
                return None, None, None, None, None, None, None, ca_inter, cv_inter
        else:
            ca_inter = cv_inter = None

        # Add intermediate positional embedding
        if audio is not None:
            B, L_A, D = x_a.shape
            inter_pos_embed_a = self.inter_pos_embed_a.repeat(B, 1, 1)
            if ids_keep_a is not None:
                inter_pos_embed_a = torch.gather(inter_pos_embed_a, dim=1, index=ids_keep_a.unsqueeze(-1).repeat(1,1,D))
            elif att_map_va_ids is not None:
                inter_pos_embed_a = torch.gather(inter_pos_embed_a, dim=1, index=att_map_va_ids[:,:L_A].unsqueeze(-1).repeat(1,1,D))

            x_a += inter_pos_embed_a

        if video is not None:
            B, L_V, D = x_v.shape
            inter_pos_embed_v = self.inter_pos_embed_v.repeat(B, 1, 1)
            if ids_keep_v is not None:
                inter_pos_embed_v = torch.gather(inter_pos_embed_v, dim=1, index=ids_keep_v.unsqueeze(-1).repeat(1,1,D))
            elif att_map_av_ids is not None:
                inter_pos_embed_v = torch.gather(inter_pos_embed_v, dim=1, index=att_map_av_ids[:,:L_V].unsqueeze(-1).repeat(1,1,D))

            x_v += inter_pos_embed_v


        # Compute video logits (cv) and audio logits (ca)
        if compute_embedding:
            ca = cv = None
            if audio is not None:
                ca = x_a
                for blk in self.blocks_u:
                    ca = blk(ca, 'a')
                ca = self.norm_a(ca)
            if video is not None:
                cv = x_v
                for blk in self.blocks_u:
                    cv = blk(cv, 'v')
                cv = self.norm_v(cv)
        else:
            ca = cv = None

        # Compute joint embedding (x) and audio video features
        if compute_joint_embedding or ((mask_audio or mask_visual) and use_mae):
            # Pass audio and video tokens to multimodal shared layers
            if audio is not None and video is not None:
                x = torch.cat([x_a, x_v], dim=1)
                for blk in self.blocks_u:
                    x = blk(x)
                audio_feats = x[:, :x_a.size(1)]
                video_feats = x[:, -x_v.size(1):]
                x = self.norm(x)

            elif audio is not None:
                x = x_a
                for blk in self.blocks_u:
                    x = blk(x)
                audio_feats = x
                video_feats = None
                x = self.norm(x)

            elif video is not None:
                x = x_v
                for blk in self.blocks_u:
                    x = blk(x)
                audio_feats = None
                video_feats = x
                x = self.norm(x)

        else:
            audio_feats = None
            video_feats = None
            x = None

        # Masked Autoencoding decoder
        if (mask_audio or mask_visual) and use_mae:
            # Transform encoder embedding size to decoder embedding size
            decoder_x = self.decoder_embed(x)

            # Unlike TVLT or VideoAudioMAE, run reconstruction together
            if audio is not None and not compute_from_a_codes:
                decoder_x_a = decoder_x[:, :x_a.size(1)]  # no cls token
                decoder_x_a = self.cat_mask(
                    self.mask_token_a, decoder_x_a, ids_restore_a)
                decoder_x_a += self.decoder_pos_embed_a
                # Filter unnecessary masks
                if reconstruct_compressed_a:
                    input_mask_a = torch.gather(input_mask_a, dim=1, index=att_map_va_ids)
                    target_mask_a = torch.gather(target_mask_a, dim=1, index=att_map_va_ids)
                    valid_mask_a = torch.logical_or(~input_mask_a, ~target_mask_a)
                    valid_input_mask_a = input_mask_a[valid_mask_a].reshape(decoder_x_a.shape[0], -1)
                    valid_target_mask_a = target_mask_a[valid_mask_a].reshape(decoder_x_a.shape[0], -1)

                    decoder_x_a = torch.gather(decoder_x_a, dim=1, index=att_map_va_ids[:,:,None].repeat(1,1,self.decoder_pos_embed_a.size(-1)))
                else:
                    valid_mask_a = torch.logical_or(~input_mask_a, ~target_mask_a)
                    valid_input_mask_a = input_mask_a[valid_mask_a].reshape(decoder_x_a.shape[0], -1)
                    valid_target_mask_a = target_mask_a

                decoder_x_a = decoder_x_a[valid_mask_a].reshape(decoder_x_a.shape[0], -1, decoder_x_a.shape[-1])
                decoder_x_a += self.decoder_modality_a

                # Reconstruct audio tokens separately
                for blk in self.decoder_blocks:
                    decoder_x_a = blk(decoder_x_a)
                decoder_x_a = self.decoder_norm(decoder_x_a)
            else:
                decoder_x_a = valid_input_mask_a = valid_target_mask_a = None

            if video is not None and not compute_from_v_codes:
                decoder_x_v = decoder_x[:, -x_v.size(1):]  # no cls token
                decoder_x_v = self.cat_mask(
                    self.mask_token_v, decoder_x_v, ids_restore_v)
                decoder_x_v += self.decoder_pos_embed_v
                # Filter unnecessary masks
                if reconstruct_compressed_v:
                    input_mask_v = torch.gather(input_mask_v, dim=1, index=att_map_av_ids)
                    target_mask_v = torch.gather(target_mask_v, dim=1, index=att_map_av_ids)
                    valid_mask_v = torch.logical_or(~input_mask_v, ~target_mask_v)
                    valid_input_mask_v = input_mask_v[valid_mask_v].reshape(decoder_x_v.shape[0], -1)
                    valid_target_mask_v = target_mask_v[valid_mask_v].reshape(decoder_x_v.shape[0], -1)

                    decoder_x_v = torch.gather(decoder_x_v, dim=1, index=att_map_av_ids[:,:,None].repeat(1,1,self.decoder_pos_embed_v.size(-1)))
                else:
                    valid_mask_v = torch.logical_or(~input_mask_v, ~target_mask_v)
                    valid_input_mask_v = input_mask_v[valid_mask_v].reshape(decoder_x_v.shape[0], -1)
                    valid_target_mask_v = target_mask_v

                decoder_x_v = decoder_x_v[valid_mask_v].reshape(decoder_x_v.shape[0], -1, decoder_x_v.shape[-1])
                decoder_x_v += self.decoder_modality_v

                # Reconstruct video tokens separately
                for blk in self.decoder_blocks:
                    decoder_x_v = blk(decoder_x_v)
                decoder_x_v = self.decoder_norm(decoder_x_v)
            else:
                decoder_x_v = valid_input_mask_v = valid_target_mask_a = None

            if audio is not None:
                cls_token = torch.cat([torch.mean(x[:, :x_a.size(1)], dim=1), torch.mean(x[:, -x_v.size(1):], dim=1)], dim=1)
            else:
                cls_token = torch.mean(x, dim=1)

            return cls_token, decoder_x_a, decoder_x_v, (valid_input_mask_a, valid_target_mask_a), (valid_input_mask_v, valid_target_mask_v), \
                ca, cv, ca_inter, cv_inter

        elif compute_joint_embedding:
            if audio is not None and video is not None:
                cls_token = torch.cat((torch.mean(x[:, :x_a.size(1)], dim=1), torch.mean(x[:, -x_v.size(1):], dim=1)), dim=1)
            else:
                cls_token = torch.mean(x, dim=1)
            return cls_token, audio_feats, video_feats, None, None, ca, cv, ca_inter, cv_inter

        else:
            return None, None, None, None, None, ca, cv, ca_inter, cv_inter


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, separate=False, shared=True):
        super().__init__()
        if shared:
            self.norm1 = norm_layer(dim)
        if separate:
            self.norm1_a = norm_layer(dim)
            self.norm1_v = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if shared:
            self.norm2 = norm_layer(dim)
        if separate:
            self.norm2_a = norm_layer(dim)
            self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None):
        if modality == None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == 'a':
            x = x + self.drop_path(self.attn(self.norm1_a(x)))
            x = x + self.drop_path(self.mlp(self.norm2_a(x)))
        elif modality == 'v':
            x = x + self.drop_path(self.attn(self.norm1_v(x)))
            x = x + self.drop_path(self.mlp(self.norm2_v(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class AudioPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, frequency_size=128, audio_size=400, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        num_patches = (frequency_size // patch_size[1]) * (audio_size // patch_size[0])
        self.patch_hw = (frequency_size // patch_size[1], audio_size // patch_size[0])
        self.frequency_size = frequency_size
        self.audio_size = audio_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x