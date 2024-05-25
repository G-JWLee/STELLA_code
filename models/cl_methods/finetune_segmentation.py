import torch
import torch.nn as nn
from einops import rearrange
from models import CrossModalAttention_ave
from utils import autograd_hacks_act
import torch.nn.functional as F


class TPAVIModule(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='dot',
                 dimension=3, bn_layer=True):
        """
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(TPAVIModule, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        ## add align channel
        self.align_channel = nn.Linear(128, in_channels)
        self.norm_layer = nn.LayerNorm(in_channels)

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x, audio=None):
        """
        args:
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
            audio: (N, T, C)
        """

        audio_temp = 0
        batch_size, C = x.size(0), x.size(1)
        if audio is not None:
            # print('==> audio.shape', audio.shape)
            H, W = x.shape[-2], x.shape[-1]
            audio_temp = self.align_channel(audio)  # [bs, T, C]
            audio = audio_temp.permute(0, 2, 1)  # [bs, C, T]
            audio = audio.unsqueeze(-1).unsqueeze(-1)  # [bs, C, T, 1, 1]
            audio = audio.repeat(1, 1, 1, H, W)  # [bs, C, T, H, W]
        else:
            audio = x

        # (N, C, THW)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # [bs, C, THW]
        # print('g_x.shape', g_x.shape)
        # g_x = x.view(batch_size, C, -1)  # [bs, C, THW]
        g_x = g_x.permute(0, 2, 1)  # [bs, THW, C]

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = audio.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # [bs, C', THW]
            phi_x = self.phi(audio).view(batch_size, self.inter_channels, -1)  # [bs, C', THW]
            theta_x = theta_x.permute(0, 2, 1)  # [bs, THW, C']
            f = torch.matmul(theta_x, phi_x)  # [bs, THW, THW]

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(audio).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N  # [bs, THW, THW]

        y = torch.matmul(f_div_C, g_x)  # [bs, THW, C]

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()  # [bs, C, THW]
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # [bs, C', T, H, W]

        W_y = self.W_z(y)  # [bs, C, T, H, W]
        # residual connection
        z = W_y + x  # # [bs, C, T, H, W]

        # add LayerNorm
        z = z.permute(0, 2, 3, 4, 1)  # [bs, T, H, W, C]
        z = self.norm_layer(z)
        z = z.permute(0, 4, 1, 2, 3)  # [bs, C, T, H, W]

        return z, audio_temp


class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class ResidualConvUnit(nn.Module):
	"""Residual convolution module.
	"""

	def __init__(self, features):
		"""Init.
		Args:
			features (int): number of features
		"""
		super().__init__()

		self.conv1 = nn.Conv2d(
			features, features, kernel_size=3, stride=1, padding=1, bias=True
		)
		self.conv2 = nn.Conv2d(
			features, features, kernel_size=3, stride=1, padding=1, bias=True
		)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		"""Forward pass.
		Args:
			x (tensor): input
		Returns:
			tensor: output
		"""
		out = self.relu(x)
		out = self.conv1(out)
		out = self.relu(out)
		out = self.conv2(out)

		return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""
    def __init__(self, features, first=False):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.first = first
        if not first:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if not self.first:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class Interpolate(nn.Module):
	"""Interpolation module.
	"""

	def __init__(self, scale_factor, mode, align_corners=False):
		"""Init.
		Args:
			scale_factor (float): scaling
			mode (str): interpolation mode
		"""
		super(Interpolate, self).__init__()

		self.interp = nn.functional.interpolate
		self.scale_factor = scale_factor
		self.mode = mode
		self.align_corners = align_corners

	def forward(self, x):
		"""Forward pass.
		Args:
			x (tensor): input
		Returns:
			tensor: interpolated data
		"""

		x = self.interp(
			x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
		)

		return x

class Finetune_segmentation(nn.Module):
    def __init__(self, model: nn.Module, device, channel=256, **kwargs) -> None:
        super().__init__()
        self.backbone = model
        self.device = device

        self.backbone.transformer.freeze_backbone()

        # self.avmatching_module = CrossModalAttention_ave(
        #     dim=embedding_dim,
        #     pretrain_path=avm_pretrain_path,
        # )
        # self.avmatching_module.requires_grad_(requires_grad=False)

        self._req_penalty = False
        self._req_opt = False

        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)

        self.path2 = FeatureFusionBlock(channel, first=True)
        self.path1 = FeatureFusionBlock(channel, first=False)

        self.x1_linear = nn.Linear(3072,512)
        self.x2_linear = nn.Linear(3072,512)

        self.audio_linear = nn.Linear(768, 128)

        for i in [0,1]:
            setattr(self, f"tpavi_b{i+1}", TPAVIModule(in_channels=channel, mode='dot'))

        self.output_conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

        self.avs_modules = nn.ModuleDict({
            "conv2": self.conv2,
            "conv1": self.conv1,
            "path2": self.path2,
            "path1": self.path1,
            "x1_linear": self.x1_linear,
            "x2_linear": self.x2_linear,
            "audio_linear": self.audio_linear,
            "output_conv": self.output_conv,
        })

        # Give handler to get intermediate outputs.
        for layer in self.backbone.transformer.blocks_u:
            autograd_hacks_act.add_hooks(layer.mlp.fc2)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)


    def pre_reshape_for_tpavi(self, x):
        # x: [B*5, C, H, W]
        _, C, H, W = x.shape
        try:
            x = x.reshape(-1, 4, C, H, W)
        except:
            print("pre_reshape_for_tpavi: ", x.shape)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
        return x

    def post_reshape_for_tpavi(self, x):
        # x: [B, C, T, H, W]
        # return: [B*T, C, H, W]
        _, C, _, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.view(-1, C, H, W)
        return x

    def tpavi_va(self, x, audio, stage):
        tpavi_b = getattr(self, f'tpavi_b{stage + 1}')
        audio = audio.unsqueeze(dim=1).repeat(1, 4, 1)  # [B, T, 128]
        x = self.pre_reshape_for_tpavi(x)  # [B, C, T, H, W]
        x, a = tpavi_b(x, audio) # [B, C, T, H, W], [B, T, C]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x, a


    def forward(self, inputs):

        inputs['masked_visual'] = False
        inputs['masked_audio'] = False
        inputs['joint_token'] = True
        embed_output = self.backbone(inputs)
        audio_embeds = embed_output['embedding_a']

        multi_scale = []
        for layer in self.backbone.transformer.blocks_u:
            multi_scale.append(layer.mlp.fc2.activations[:,audio_embeds.shape[1]:,:])

        audio_embeds = audio_embeds.mean(dim=1)
        audio_feature = self.audio_linear(audio_embeds)

        x1 = multi_scale[0].reshape(multi_scale[0].size(0) * 4, 14, 14, -1)
        x2 = multi_scale[1].reshape(multi_scale[1].size(0) * 4, 14, 14, -1)

        x1 = self.x1_linear(x1)
        x2 = self.x2_linear(x2)

        x1 = F.interpolate(rearrange(x1, 'BF w h c -> BF c w h'), mode='bicubic', size=[56,56])
        x2 = F.interpolate(rearrange(x2, 'BF w h c -> BF c w h'), mode='bicubic', size=[28,28])

        conv1_feat = self.conv1(x1)  # BF x 256 x 56 x 56
        conv2_feat = self.conv2(x2)  # BF x 256 x 28 x 28

        feature_map_list = [conv1_feat, conv2_feat]
        a_fea_list = [None] * 2

        for i in [0, 1]:
            conv_feat = torch.zeros_like(feature_map_list[i]).cuda()
            conv_feat_va, a_fea = self.tpavi_va(feature_map_list[i], audio_feature, stage=i)
            conv_feat += conv_feat_va
            a_fea_list[i] = a_fea
            feature_map_list[i] = conv_feat

        conv2_feat = self.path2(feature_map_list[1]) # BF x 256 x 56 x 56
        conv21 = self.path1(conv2_feat, feature_map_list[0]) # BF x 256 x 112 x 112

        pred = self.output_conv(conv21)  # BF x 1 x 224 x 224

        loss_dict = {}
        if self.training:
            loss_dict['avs_loss'] = self.IouSemanticAwareLoss(pred, inputs['mask_data'])
        else:
            loss_dict['avs_loss'] = torch.zeros(1).to('cuda')
        loss_dict['avs_pred'] = pred.detach()

        return loss_dict


    def IouSemanticAwareLoss(self, pred_masks, first_gt_mask):
        """
        loss for single sound source segmentation

        Args:
        pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
        first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
        a_fea_list: feature list of audio features
        v_map_list: feature map list of the encoder or decoder output, each of shape: [bs*5, C, H, W]
        count_stages: additional constraint loss on which stages' visual-audio features
        """
        f1_iou_loss, avs_logits, avs_labels = self.F1_IoU_BCELoss(pred_masks, first_gt_mask)

        return f1_iou_loss

    def F1_IoU_BCELoss(self, pred_masks, first_gt_mask):
        """
        binary cross entropy loss (iou loss) of the first frame for single sound source segmentation

        Args:
        pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
        first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
        """
        assert len(pred_masks.shape) == 4
        # pred_masks = torch.sigmoid(pred_masks)  # [bs*5, 1, 224, 224]
        indices = torch.tensor(list(range(0, len(pred_masks), 4)))
        indices = indices.cuda()

        first_pred = torch.index_select(pred_masks, dim=0, index=indices)  # [bs, 1, 224, 224]
        assert first_pred.requires_grad == True, "Error when indexing predited masks"
        if len(first_gt_mask.shape) == 5:
            first_gt_mask = first_gt_mask.squeeze(1)  # [bs, 1, 224, 224]
        first_bce_loss = nn.BCEWithLogitsLoss()(first_pred, first_gt_mask)

        return first_bce_loss, first_pred, first_gt_mask


