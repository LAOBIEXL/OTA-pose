#!/usr/bin/python
# -*- coding:utf8 -*-

from utils.utils_registry import MODEL_REGISTRY
from utils.common import TRAIN_PHASE, TEST_PHASE, VAL_PHASE
import torch.nn as nn
import logging

# LoRA: 动态注入（冻结原权重，只训练 LoRA）
from posetimation.lora import inject_lora

logger = logging.getLogger(__name__)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def build_model(cfg, phase, **kwargs):
    """
        return : model Instance
    """
    model_name = cfg.MODEL.NAME
    model_Class = MODEL_REGISTRY.get(model_name)
    model_instance = model_Class.get_net(cfg=cfg, phase=phase, **kwargs)

    if phase == TRAIN_PHASE and cfg.MODEL.INIT_WEIGHTS:
        model_instance.init_weights()
        model_instance.train()

        total_trainable_params = count_trainable_parameters(model_instance)
        total_params = count_parameters(model_instance)
        logger.info(f'=>Total trainable parameters: {total_trainable_params}')
        logger.info(f'=>Total parameters:          {total_params}')

    # LoRA：在 init_weights 之后注入，冻结原权重只训 LoRA
    # 注意：验证/测试时也需要注入 LoRA，才能加载训练时保存的 LoRA checkpoint
    if getattr(cfg.MODEL, 'USE_LORA', False):
        r = getattr(cfg.MODEL, 'LORA_R', 8)
        alpha = getattr(cfg.MODEL, 'LORA_ALPHA', 16)
        target_modules = getattr(cfg.MODEL, 'LORA_TARGET_MODULES', ['qkv', 'proj'])
        if not isinstance(target_modules, (list, tuple)):
            target_modules = [target_modules]
        model_instance = inject_lora(model_instance, r, alpha, target_modules, freeze_other=True)
        if phase == TRAIN_PHASE:
            total_trainable_params = count_trainable_parameters(model_instance)
            logger.info(f'=>Total trainable parameters (after LoRA): {total_trainable_params}')
        else:
            logger.info(f'=>LoRA injected for {phase}: r={r}, alpha={alpha}, targets={target_modules}')

    if phase != TRAIN_PHASE:
        model_instance.eval()

    return model_instance


def get_model_hyperparameter(args, cfg, **kwargs):
    """
        return : model hyperparameter
    """
    model_name = cfg.MODEL.NAME
    hyper_parameters_setting = MODEL_REGISTRY.get(model_name).get_model_hyper_parameters(args, cfg)
    return hyper_parameters_setting

