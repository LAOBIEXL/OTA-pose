# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
import cv2
import os.path as osp
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer, constant_init, normal_init


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
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
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) # [3, batch, num_head, patch, channels]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_head_dim=None
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio),
                              padding=4 + 2 * (ratio // 2 - 1))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)  # [batch, 1280, 16, 12]
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2) # [64, 16*12, 1280] -> [batch, H*W, Channel]
        return x, (Hp, Wp)


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class BaseBackbone(nn.Module):
    """Base backbone.

    This class defines the basic functions of a backbone. Any backbone that
    inherits this class should at least define its own `forward` function.
    """

    def init_weights(self, pretrained=None, patch_padding='pad'):
        """Init backbone weights.

        Args:
            pretrained (str | None): If pretrained is a string, then it
                initializes backbone weights by loading the pretrained
                checkpoint. If pretrained is None, then it follows default
                initializer or customized initializer in subclasses.
        """
        if isinstance(pretrained, str):
            self.load_checkpoint(self, pretrained, strict=False, patch_padding=patch_padding)
        elif pretrained is None:
            # use default initializer or customized initializer in subclasses
            pass
        else:
            raise TypeError('pretrained must be a str or None.'
                            f' But received {type(pretrained)}.')

    def _load_checkpoint(self,filename, map_location=None):
        if not osp.isfile(filename):
            raise IOError(f'{filename} is not a checkpoint file')
        checkpoint = torch.load(filename, map_location=map_location)

        return checkpoint

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """
        pass

class ViT(BaseBackbone):

    def __init__(self,
                 img_size=(256, 192), patch_size=16, in_chans=3, num_classes=80, embed_dim=1280, depth=32,
                 num_heads=16, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.55, hybrid_backbone=None, norm_layer=None, use_checkpoint=False,
                 frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False, freeze_weight=False, flatten=True
                 ):
        # Protect mutable default arguments
        super(ViT, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.freeze_weight = freeze_weight
        self.flatten = flatten
        self.depth = depth

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        # since the pretraining model has class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            )
            for i in range(depth)])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

        if self.freeze_weight:
            for module in self.modules():
                parameters = module.parameters()
                for parameter in parameters:
                    parameter.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super().init_weights(pretrained, patch_padding=self.patch_padding)

        if pretrained is None:
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

            self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)  # x: [Batch, H*W, Channel]

        if self.pos_embed is not None:
            # fit for multiple GPU training
            # since the first element for pos embed (sin-cos manner) is zero, it will cause no difference
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1] # [64, 192, 1280]

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.last_norm(x)  # [B, L, N]
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        if self.flatten:
            return x
        else:
            return xp

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()

class ViTBase(BaseBackbone):

    def __init__(self,
                 img_size=(256, 192), patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.3, hybrid_backbone=None, norm_layer=None, use_checkpoint=False,
                 frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False, freeze_weight=False,
                 ):
        # Protect mutable default arguments
        super(ViTBase, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.freeze_weight = freeze_weight
        self.depth = depth

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        # since the pretraining model has class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            )
            for i in range(depth)])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

        if self.freeze_weight:
            for module in self.modules():
                parameters = module.parameters()
                for parameter in parameters:
                    parameter.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super().init_weights(pretrained, patch_padding=self.patch_padding)

        if pretrained is None:
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

            self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x) # x: [Batch, H*W, Channel]

        if self.pos_embed is not None:
            # fit for multiple GPU training
            # since the first element for pos embed (sin-cos manner) is zero, it will cause no difference
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1] # [64, 192, 1280]

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.last_norm(x)

        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        return xp

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()

class TopdownHeatmapBaseHead(nn.Module):

    def forward(self, **kwargs):
        """Forward function."""

    def inference_model(self, **kwargs):
        """Inference function."""

class TopdownHeatmapSimpleHead(TopdownHeatmapBaseHead):
    """Top-down heatmap simple head. paper ref: Bin Xiao et al. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: 0
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            Default: None.

            - 'resize_concat': Multiple feature maps will be resized to the
                same size as the first one and then concat together.
                Usually used in FCN head of HRNet.
            - 'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            - None: Only one select feature map is allowed.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 in_channels=1280,
                 out_channels=17,
                 num_deconv_layers=2,
                 num_deconv_filters=(256, 256),
                 num_deconv_kernels=(4, 4),
                 extra=dict(final_conv_kernel=1, ),
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                 loss_keypoint=None,
                 train_cfg=dict(),
                 test_cfg=dict(),
                 upsample=0, ):
        super().__init__()

        self.in_channels = in_channels
        self.upsample = upsample

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[
                -1] if num_deconv_layers > 0 else self.in_channels

            # layers = [nn.Conv2d(in_channels=conv_channels,
            #                     out_channels=out_channels,
            #                     kernel_size=kernel_size,
            #                     stride=1,
            #                     padding=padding)]
            layers = []
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels',
                                             [1] * num_conv_layers)

                for i in range(num_conv_layers):
                    layers.append(
                        build_conv_layer(
                            dict(type='Conv2d'),
                            in_channels=conv_channels,
                            out_channels=conv_channels,
                            kernel_size=num_conv_kernels[i],
                            stride=1,
                            padding=(num_conv_kernels[i] - 1) // 2))
                    layers.append(
                        build_norm_layer(dict(type='BN'), conv_channels)[1])
                    layers.append(nn.ReLU(inplace=True))

            layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=conv_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding))

            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)
        x = self.deconv_layers(x)
        final_heatmap = self.final_layer(x)
        return final_heatmap

    def inference_model(self, x, flip_pairs=None):
        """Inference function."""

        output = self.forward(x)

        output_heatmap = output.detach().cpu().numpy()

        return output_heatmap

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.

                - 'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                - 'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                - None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            if not isinstance(inputs, list):
                if self.upsample > 0:
                    inputs = F.interpolate(
                        input=F.relu(inputs),
                        scale_factor=self.upsample,
                        mode='bilinear',
                        align_corners=self.align_corners
                    )
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            # kernel, padding, output_padding = self._get_deconv_cfg(deconv_kernel = num_kernels[i])
            kernel, padding, output_padding = [4, 1, 0]

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def freeze_weight(self):
        for module in self.modules():
            parameters = module.parameters()
            for parameter in parameters:
                parameter.requires_grad = False

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)


# class TopdownHeatmapSimpleHeadVitBase(TopdownHeatmapBaseHead):
#     """Top-down heatmap simple head. paper ref: Bin Xiao et al. ``Simple
#     Baselines for Human Pose Estimation and Tracking``.
#
#     TopdownHeatmapSimpleHead is consisted of (>=0) number of deconv layers
#     and a simple conv2d layer.
#
#     Args:
#         in_channels (int): Number of input channels
#         out_channels (int): Number of output channels
#         num_deconv_layers (int): Number of deconv layers.
#             num_deconv_layers should >= 0. Note that 0 means
#             no deconv layers.
#         num_deconv_filters (list|tuple): Number of filters.
#             If num_deconv_layers > 0, the length of
#         num_deconv_kernels (list|tuple): Kernel sizes.
#         in_index (int|Sequence[int]): Input feature index. Default: 0
#         input_transform (str|None): Transformation type of input features.
#             Options: 'resize_concat', 'multiple_select', None.
#             Default: None.
#
#             - 'resize_concat': Multiple feature maps will be resized to the
#                 same size as the first one and then concat together.
#                 Usually used in FCN head of HRNet.
#             - 'multiple_select': Multiple feature maps will be bundle into
#                 a list and passed into decode head.
#             - None: Only one select feature map is allowed.
#         align_corners (bool): align_corners argument of F.interpolate.
#             Default: False.
#         loss_keypoint (dict): Config for keypoint loss. Default: None.
#     """
#
#     def __init__(self,
#                  in_channels=1280,
#                  out_channels=17,
#                  num_deconv_layers=2,
#                  num_deconv_filters=(256, 256),
#                  num_deconv_kernels=(4, 4),
#                  extra=dict(final_conv_kernel=1, ),
#                  in_index=0,
#                  input_transform=None,
#                  align_corners=False,
#                  loss_keypoint=None,
#                  train_cfg=dict(),
#                  test_cfg=dict(),
#                  upsample=0, ):
#         super().__init__()
#
#         self.in_channels = in_channels
#         self.upsample = upsample
#
#         self.train_cfg = {} if train_cfg is None else train_cfg
#         self.test_cfg = {} if test_cfg is None else test_cfg
#         self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')
#
#         self._init_inputs(in_channels, in_index, input_transform)
#         self.in_index = in_index
#         self.align_corners = align_corners
#
#         if extra is not None and not isinstance(extra, dict):
#             raise TypeError('extra should be dict or None.')
#
#         if num_deconv_layers > 0:
#             self.deconv_layers = self._make_deconv_layer(
#                 num_deconv_layers,
#                 num_deconv_filters,
#                 num_deconv_kernels,
#             )
#         elif num_deconv_layers == 0:
#             self.deconv_layers = nn.Identity()
#         else:
#             raise ValueError(
#                 f'num_deconv_layers ({num_deconv_layers}) should >= 0.')
#
#         identity_final_layer = False
#         if extra is not None and 'final_conv_kernel' in extra:
#             assert extra['final_conv_kernel'] in [0, 1, 3]
#             if extra['final_conv_kernel'] == 3:
#                 padding = 1
#             elif extra['final_conv_kernel'] == 1:
#                 padding = 0
#             else:
#                 # 0 for Identity mapping.
#                 identity_final_layer = True
#             kernel_size = extra['final_conv_kernel']
#         else:
#             kernel_size = 1
#             padding = 0
#
#         if identity_final_layer:
#             self.final_layer = nn.Identity()
#         else:
#             conv_channels = num_deconv_filters[
#                 -1] if num_deconv_layers > 0 else self.in_channels
#
#             # layers = [nn.Conv2d(in_channels=conv_channels,
#             #                     out_channels=out_channels,
#             #                     kernel_size=kernel_size,
#             #                     stride=1,
#             #                     padding=padding)]
#             layers = []
#             if extra is not None:
#                 num_conv_layers = extra.get('num_conv_layers', 0)
#                 num_conv_kernels = extra.get('num_conv_kernels',
#                                              [1] * num_conv_layers)
#
#                 for i in range(num_conv_layers):
#                     layers.append(
#                         build_conv_layer(
#                             dict(type='Conv2d'),
#                             in_channels=conv_channels,
#                             out_channels=conv_channels,
#                             kernel_size=num_conv_kernels[i],
#                             stride=1,
#                             padding=(num_conv_kernels[i] - 1) // 2))
#                     layers.append(
#                         build_norm_layer(dict(type='BN'), conv_channels)[1])
#                     layers.append(nn.ReLU(inplace=True))
#
#             layers.append(
#                 build_conv_layer(
#                     cfg=dict(type='Conv2d'),
#                     in_channels=conv_channels,
#                     out_channels=out_channels,
#                     kernel_size=kernel_size,
#                     stride=1,
#                     padding=padding))
#
#             if len(layers) > 1:
#                 self.final_layer = nn.Sequential(*layers)
#             else:
#                 self.final_layer = layers[0]
#
#     def forward(self, x):
#         """Forward function."""
#         x = self._transform_inputs(x)
#         x = self.deconv_layers(x)
#         final_heatmap = self.final_layer(x)
#         return x, final_heatmap
#
#     def inference_model(self, x, flip_pairs=None):
#         """Inference function."""
#
#         output = self.forward(x)
#
#         output_heatmap = output.detach().cpu().numpy()
#
#         return output_heatmap
#
#     def _init_inputs(self, in_channels, in_index, input_transform):
#         """Check and initialize input transforms.
#
#         The in_channels, in_index and input_transform must match.
#         Specifically, when input_transform is None, only single feature map
#         will be selected. So in_channels and in_index must be of type int.
#         When input_transform is not None, in_channels and in_index must be
#         list or tuple, with the same length.
#
#         Args:
#             in_channels (int|Sequence[int]): Input channels.
#             in_index (int|Sequence[int]): Input feature index.
#             input_transform (str|None): Transformation type of input features.
#                 Options: 'resize_concat', 'multiple_select', None.
#
#                 - 'resize_concat': Multiple feature maps will be resize to the
#                     same size as first one and than concat together.
#                     Usually used in FCN head of HRNet.
#                 - 'multiple_select': Multiple feature maps will be bundle into
#                     a list and passed into decode head.
#                 - None: Only one select feature map is allowed.
#         """
#
#         if input_transform is not None:
#             assert input_transform in ['resize_concat', 'multiple_select']
#         self.input_transform = input_transform
#         self.in_index = in_index
#         if input_transform is not None:
#             assert isinstance(in_channels, (list, tuple))
#             assert isinstance(in_index, (list, tuple))
#             assert len(in_channels) == len(in_index)
#             if input_transform == 'resize_concat':
#                 self.in_channels = sum(in_channels)
#             else:
#                 self.in_channels = in_channels
#         else:
#             assert isinstance(in_channels, int)
#             assert isinstance(in_index, int)
#             self.in_channels = in_channels
#
#     def _transform_inputs(self, inputs):
#         """Transform inputs for decoder.
#
#         Args:
#             inputs (list[Tensor] | Tensor): multi-level img features.
#
#         Returns:
#             Tensor: The transformed inputs
#         """
#         if not isinstance(inputs, list):
#             if not isinstance(inputs, list):
#                 if self.upsample > 0:
#                     inputs = F.interpolate(
#                         input=F.relu(inputs),
#                         scale_factor=self.upsample,
#                         mode='bilinear',
#                         align_corners=self.align_corners
#                     )
#             return inputs
#
#         if self.input_transform == 'resize_concat':
#             inputs = [inputs[i] for i in self.in_index]
#             upsampled_inputs = [
#                 F.interpolate(
#                     input=x,
#                     size=inputs[0].shape[2:],
#                     mode='bilinear',
#                     align_corners=self.align_corners) for x in inputs
#             ]
#             inputs = torch.cat(upsampled_inputs, dim=1)
#         elif self.input_transform == 'multiple_select':
#             inputs = [inputs[i] for i in self.in_index]
#         else:
#             inputs = inputs[self.in_index]
#
#         return inputs
#
#     def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
#         """Make deconv layers."""
#         if num_layers != len(num_filters):
#             error_msg = f'num_layers({num_layers}) ' \
#                         f'!= length of num_filters({len(num_filters)})'
#             raise ValueError(error_msg)
#         if num_layers != len(num_kernels):
#             error_msg = f'num_layers({num_layers}) ' \
#                         f'!= length of num_kernels({len(num_kernels)})'
#             raise ValueError(error_msg)
#
#         layers = []
#         for i in range(num_layers):
#             # kernel, padding, output_padding = self._get_deconv_cfg(deconv_kernel = num_kernels[i])
#             kernel, padding, output_padding = [4, 1, 0]
#
#             planes = num_filters[i]
#             layers.append(
#                 build_upsample_layer(
#                     dict(type='deconv'),
#                     in_channels=self.in_channels,
#                     out_channels=planes,
#                     kernel_size=kernel,
#                     stride=2,
#                     padding=padding,
#                     output_padding=output_padding,
#                     bias=False))
#             layers.append(nn.BatchNorm2d(planes))
#             layers.append(nn.ReLU(inplace=True))
#             self.in_channels = planes
#
#         return nn.Sequential(*layers)
#
#     def _get_deconv_cfg(deconv_kernel):
#         """Get configurations for deconv layers."""
#         if deconv_kernel == 4:
#             padding = 1
#             output_padding = 0
#         elif deconv_kernel == 3:
#             padding = 1
#             output_padding = 1
#         elif deconv_kernel == 2:
#             padding = 0
#             output_padding = 0
#         else:
#             raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')
#
#         return deconv_kernel, padding, output_padding
#
#     def freeze_weight(self):
#         for module in self.modules():
#             parameters = module.parameters()
#             for parameter in parameters:
#                 parameter.requires_grad = False
#
#     def init_weights(self):
#         """Initialize model weights."""
#         for _, m in self.deconv_layers.named_modules():
#             if isinstance(m, nn.ConvTranspose2d):
#                 normal_init(m, std=0.001)
#             elif isinstance(m, nn.BatchNorm2d):
#                 constant_init(m, 1)
#         for m in self.final_layer.modules():
#             if isinstance(m, nn.Conv2d):
#                 normal_init(m, std=0.001, bias=0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 constant_init(m, 1)



