#!/usr/bin/python
# -*- coding:utf8 -*-

# model
from .build import build_model, get_model_hyperparameter

# HRNet
from .backbones.hrnet import HRNet

from .backbones.vit import ViT

# DINOv3
try:
    from .backbones.dinov3_vit import DINOv3ViT
except Exception as e:
    # DINOv3 依赖/类型注解在部分环境下可能导入失败；不影响其它模型（如 ViT/HRNet）训练
    import logging
    logging.getLogger(__name__).warning(f"Skip importing DINOv3ViT due to error: {e}")

# DINOv2
# from .backbones.dinov2_vit import DINOv2ViT  # 文件不存在，暂时注释

# SimpleBaseline
from .backbones.simplebaseline import SimpleBaseline

from .sldpose import SLDPOSE
from .sldpose_mi import SLDPOSE_MI
from .sldpose_mi_V2 import SLDPOSE_MI_V2
from .sldpose_mi_V3 import SLDPOSE_MI_V3
from .sldpose_mi_V2_5_frame_pe import SLDPOSE_MI_V2_5_FRAME_PE
from .sldpose_mi_V3_pe import SLDPOSE_MI_V3_PE

from .multi_scale_represt_encoder.newpose import NEWPOSE_V1
from .multi_scale_represt_encoder.newpose_v2 import NEWPOSE_V2
from .multi_scale_represt_encoder.newpose_v3 import NEWPOSE_V3

from .layers.block_zoo import *
from .layers.rope import *
from .layers.spatiotemporal_attention import *
from .layers.corss_block import *


from .auxpose.AuxposeV6 import AuxposeV6

try:
    from .mambapose.mamba_pose import MambaPose
except Exception as e:
    import logging
    logging.getLogger(__name__).warning(f"Skip importing MambaPose due to error: {e}")

try:
    from .otapose.otapose import *
except Exception as e:
    import logging
    logging.getLogger(__name__).warning(f"Skip importing OTAPose due to error: {e}")

try:
    from .DiffusionPose.Diffusion import *
except Exception as e:
    import logging
    logging.getLogger(__name__).warning(f"Skip importing DiffusionPose due to error: {e}")

try:
    from .UGPose.Uncertainty_Guided_Pose import UGPOSE
except Exception as e:
    import logging
    logging.getLogger(__name__).warning(f"Skip importing UGPOSE due to error: {e}")