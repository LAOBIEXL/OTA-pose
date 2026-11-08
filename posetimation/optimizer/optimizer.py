#!/usr/bin/python
# -*- coding:utf8 -*-
import torch.optim as optimizer_zoo


def build_optimizer(cfg, model):
    backbone_lr_scale = getattr(cfg.MODEL, 'BACKBONE_LR_SCALE', 1.0)

    # 如果 backbone 需要以更低的学习率微调，构建双参数组
    if backbone_lr_scale != 1.0 and hasattr(model, 'backbone'):
        backbone_param_ids = set(id(p) for p in model.backbone.parameters())
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        other_params = [p for p in model.parameters()
                        if id(p) not in backbone_param_ids and p.requires_grad]
        param_groups = [
            {'params': backbone_params, 'lr': cfg.TRAIN.LR * backbone_lr_scale},
            {'params': other_params,    'lr': cfg.TRAIN.LR},
        ]
    else:
        param_groups = list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optimizer_zoo.SGD(
            param_groups,
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )

    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optimizer_zoo.Adam(
            param_groups, lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'adamW':
        optimizer = optimizer_zoo.AdamW(
            param_groups, lr=cfg.TRAIN.LR
        )
    return optimizer

