from typing import List

import torch
import torch.nn as nn
import timm

from models import objectives


class Transformer(nn.Module):
    def __init__(self, backbone: nn.Module, loss_names: List, norm_pix_loss: bool = False,
                 mae_loss_weight: float = 1.0, contrast_loss_weight: float = 0.1, tau: float = 0.05,
                 load_local_path: str = "", init_classifier: bool = False, **kwargs):
        super().__init__()

        self.patch_size = 16
        self.audio_patch_size = [16, 16]
        self.current_tasks = loss_names
        self.norm_pix_loss = norm_pix_loss
        self.mae_loss_weight = mae_loss_weight
        self.contrast_loss_weight = contrast_loss_weight
        self.contrast_tau = tau

        self.transformer = backbone
        if hasattr(self.transformer, 'init_weights'):
            self.transformer.init_weights()

        self.apply(objectives.init_weights)

        # ===================== load model weight ======================

        if load_local_path != "":
            state_dict = torch.load(load_local_path, map_location="cpu")
            if "model" in state_dict.keys():
                state_dict = state_dict["model"]
            shift_name = False
            for k,v in state_dict.items():
                if k.startswith('module.backbone.transformer'):
                    shift_name = True
            if shift_name:
                state_dict = {".".join(k.split(".")[3:]): v for k, v in state_dict.items() if k.startswith('module.backbone.transformer')}
            assert len(state_dict) != 0

            miss, unexpected = self.transformer.load_state_dict(state_dict, strict=False)
            print('miss', miss)
            print('unexpected', unexpected)
            print(f"Loaded model weight from {load_local_path}")
            if init_classifier:
                self.transformer.vacls_classifier.apply(objectives.init_weights)


    def infer(
        self,
        batch,
        mask_audio=False,
        mask_visual=False,
        use_mae=False,
        compute_from_a_codes=False,
        compute_from_v_codes=False,
        compute_unimodal_embedding=False,
        compute_embedding=False,
        compute_joint_embedding=False,
    ):
        videokey = "video_data"
        audiokey = "audio_data"
        labelkey = "label_idx"
        atten_avkey = "att_map_av_ids"
        atten_vakey = "att_map_va_ids"

        use_audio = audiokey in list(batch.keys())
        use_video = videokey in list(batch.keys())
        has_label = labelkey in list(batch.keys())
        has_atten_av = atten_avkey in list(batch.keys())
        has_atten_va = atten_vakey in list(batch.keys())

        if use_audio:
            audio = batch[audiokey]
        else:
            audio = None

        if use_video:
            video = batch[videokey] 
        else:
            video = None

        if has_label:
            label_idx = batch[labelkey]
        else:
            label_idx = None

        if has_atten_av:
            att_map_av_ids = batch[atten_avkey]
        else:
            att_map_av_ids = None

        if has_atten_va:
            att_map_va_ids = batch[atten_vakey]
        else:
            att_map_va_ids = None


        cls_feats, audio_feats, video_feats, audio_masks, video_masks, latent_c_a, latent_c_v, inter_c_a, inter_c_v = \
            self.transformer(audio=audio, video=video, mask_audio=mask_audio, mask_visual=mask_visual, use_mae=use_mae,
                             compute_from_a_codes=compute_from_a_codes, compute_from_v_codes=compute_from_v_codes,
                             compute_unimodal_embedding=compute_unimodal_embedding, compute_embedding=compute_embedding, compute_joint_embedding=compute_joint_embedding,
                             att_map_av_ids=att_map_av_ids, att_map_va_ids=att_map_va_ids)

        ret = {
            "audio_feats": audio_feats,
            "video_feats": video_feats,
            "cls_feats": cls_feats,
            "video_masks": video_masks,
            "video": video,
            "audio_masks": audio_masks,
            "audio": audio,
            "label_idx": label_idx,
            "latent_c_a": latent_c_a,
            "latent_c_v": latent_c_v,
            "inter_c_a": inter_c_a,
            "inter_c_v": inter_c_v,
        }

        return ret

    def forward(self, batch, reduction='mean'):
        ret = dict()
        # When not doing task in this format
        if len(self.current_tasks) == 0:
            ret.update(self.transformer(batch))
            return ret

        # Output embedding
        if "embedding" in self.current_tasks or 'masked_audio' in batch or 'masked_visual' in batch or 'modality_token' in batch or 'joint_token' in batch:
            ret.update(objectives.compute_embedding(self, batch))

            return ret

        elif "mae_audio" in self.current_tasks and "mae_frame" in self.current_tasks:
            ret.update(objectives.compute_mae_joint_frames(self, batch, self.patch_size, self.audio_patch_size, self.norm_pix_loss,
                                                    mae_loss_weight=self.mae_loss_weight, cont_loss_weight=self.contrast_loss_weight, tau=self.contrast_tau,
                                                    contrastive="contrastive" in self.current_tasks, reduction=reduction))

        # Masked Patch Prediction
        elif "mae_audio" in self.current_tasks:
            ret.update(objectives.compute_mae_audio(self, batch, self.audio_patch_size, self.norm_pix_loss, self.mae_loss_weight, reduction=reduction))
            
        elif "mae_mae_frame" in self.current_tasks:
            ret.update(objectives.compute_mae_frames(self, batch, self.patch_size, self.norm_pix_loss, self.mae_loss_weight, reduction=reduction))

        # Video-Audio Classification
        elif "vacls" in self.current_tasks:
            ret.update(objectives.compute_vacls(self, batch))

        return ret