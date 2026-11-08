import time
import logging
import os
import math
import random
from functools import partial

from typing import List
from collections import namedtuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


from .fusion_vit import *


class FusionNet(nn.Module):
    def __init__(self, cfg, is_train):
        super().__init__()

        # self.fusion_feat_model = get_fusion_vit_net(cfg, is_train)

    def forward(self, x , heatmap = None ,  phase = 'train'):
        true_batchsize  = x.shape[0]

        # 将17个通道的特征图加起来得到(batchsize, 1, 96, 72)
        heatmap_summed = heatmap.sum(dim=1, keepdim=True)

        # 将这个通道复制到(batchsize, 48, 96, 72)
        heatmap_expanded = heatmap_summed.expand(-1, x.shape[1], -1, -1)


        x_masked = heatmap_expanded * x

        return x_masked

        out = torch.cat((x,x_masked), dim=1)


        # x = self.fusion_feat_model(x)


        return out




