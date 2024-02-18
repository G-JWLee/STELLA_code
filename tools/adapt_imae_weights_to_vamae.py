# -*- coding: utf-8 -*-
# @Time    : 3/13/23 11:17 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : adapt_vmae_weights.py

# adapt original single-modality vision mae weights for multi-modality cav-mae pretraining initialization, decoder is also included.
import os
import torch
import models
from collections import OrderedDict

base_path = '/base_path'
modal_specific_layer = 10 # total of 12 layers

# weights from https://github.com/facebookresearch/mae
mdl_weight = torch.load(os.path.join(base_path, 'STELLA_code/baseline_ckpt/mae_pretrain_vit_base_full.pth'))['model']
additional_weight = OrderedDict()

for key in mdl_weight.keys():
    if 'blocks' in key and 'decoder' not in key:
        block_id = int(key.split('.')[1])
        key_var_name = '.'.join(key.split('.')[2:])
        if block_id <= modal_specific_layer-1:
            additional_weight['blocks_a.' + key[7:]] = mdl_weight[key].detach().clone()
            additional_weight['blocks_v.' + key[7:]] = mdl_weight[key].detach().clone()
        else:
            additional_weight['blocks_u.' + str(block_id-modal_specific_layer) + '.' + key_var_name] = mdl_weight[key].detach().clone()

for block_id in range(modal_specific_layer, 12):
    additional_weight['blocks_u.' + str(block_id - modal_specific_layer) + '.norm1_a.weight'] = mdl_weight['blocks.' + str(block_id) + '.norm1.weight'].detach().clone()
    additional_weight['blocks_u.' + str(block_id - modal_specific_layer) + '.norm1_v.weight'] = mdl_weight['blocks.' + str(block_id) + '.norm1.weight'].detach().clone()
    additional_weight['blocks_u.' + str(block_id - modal_specific_layer) + '.norm1_a.bias'] = mdl_weight['blocks.' + str(block_id) + '.norm1.bias'].detach().clone()
    additional_weight['blocks_u.' + str(block_id - modal_specific_layer) + '.norm1_v.bias'] = mdl_weight['blocks.' + str(block_id) + '.norm1.bias'].detach().clone()

    additional_weight['blocks_u.' + str(block_id - modal_specific_layer) + '.norm2_a.weight'] = mdl_weight['blocks.' + str(block_id) + '.norm2.weight'].detach().clone()
    additional_weight['blocks_u.' + str(block_id - modal_specific_layer) + '.norm2_v.weight'] = mdl_weight['blocks.' + str(block_id) + '.norm2.weight'].detach().clone()
    additional_weight['blocks_u.' + str(block_id - modal_specific_layer) + '.norm2_a.bias'] = mdl_weight['blocks.' + str(block_id) + '.norm2.bias'].detach().clone()
    additional_weight['blocks_u.' + str(block_id - modal_specific_layer) + '.norm2_v.bias'] = mdl_weight['blocks.' + str(block_id) + '.norm2.bias'].detach().clone()

additional_weight['norm_a.weight'] = mdl_weight['norm.weight'].detach().clone()
additional_weight['norm_v.weight'] = mdl_weight['norm.weight'].detach().clone()
additional_weight['norm_a.bias'] = mdl_weight['norm.bias'].detach().clone()
additional_weight['norm_v.bias'] = mdl_weight['norm.bias'].detach().clone()

mae_mdl = models.CAV(mid_fusion_depth=modal_specific_layer, use_audio=True, loss_names=["mae_audio", "mae_frame", "contrastive"], audio_size=1024)

miss, unexpect = mae_mdl.load_state_dict(mdl_weight, strict=False)
miss_a, unexpect_a = mae_mdl.load_state_dict(additional_weight, strict=False)

# miss 06
mae_mdl.pos_embed_v = torch.nn.Parameter(mdl_weight['pos_embed'][:,1:,:].repeat(1,4,1).detach().clone())
mae_mdl.inter_pos_embed_v = torch.nn.Parameter(mdl_weight['pos_embed'][:,1:,:].repeat(1,4,1).detach().clone())

# miss 08
mae_mdl.decoder_pos_embed_v = torch.nn.Parameter(mdl_weight['decoder_pos_embed'][:,1:,:].repeat(1,4,1).detach().clone())

# miss 09-10
mae_mdl.patch_embed_a.proj.weight = torch.nn.Parameter(torch.sum(mdl_weight['patch_embed.proj.weight'], dim=1).unsqueeze(1).detach().clone())
mae_mdl.patch_embed_a.proj.bias = torch.nn.Parameter(mdl_weight['patch_embed.proj.bias'].detach().clone())

# miss 11-12
mae_mdl.patch_embed_v.proj.weight = torch.nn.Parameter(mdl_weight['patch_embed.proj.weight'].detach().clone())
mae_mdl.patch_embed_v.proj.bias = torch.nn.Parameter(mdl_weight['patch_embed.proj.bias'].detach().clone())

# miss 13-14
mae_mdl.mae_score_audio.decoder.weight = torch.nn.Parameter(mdl_weight['decoder_pred.weight'][:256].detach().clone())
mae_mdl.mae_score_audio.decoder.bias = torch.nn.Parameter(mdl_weight['decoder_pred.bias'][:256].detach().clone())

# miss 15-16
mae_mdl.mae_score_video.decoder.weight = torch.nn.Parameter(mdl_weight['decoder_pred.weight'].detach().clone())
mae_mdl.mae_score_video.decoder.bias = torch.nn.Parameter(mdl_weight['decoder_pred.bias'].detach().clone())

mae_mdl.mask_token_v = torch.nn.Parameter(mdl_weight['mask_token'].detach().clone())
mae_mdl.mask_token_a = torch.nn.Parameter(mdl_weight['mask_token'].detach().clone())

torch.save(mae_mdl.state_dict(), os.path.join(base_path, 'STELLA_code/baseline_ckpt/init_checkpoint.pth'))