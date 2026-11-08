"""
LoRA (Low-Rank Adaptation)

用法：在 `posetimation/zoo/build.py` 里对模型做动态注入（替换目标 Linear），并冻结原权重，只训练 LoRA 参数。
"""

import math
import logging

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    包装 nn.Linear：
      y = Wx + (alpha/r) * (x @ A.T @ B.T)

    - W 来自原 Linear（冻结）
    - A, B 为可训练低秩矩阵
    """

    def __init__(self, linear: nn.Linear, r: int, alpha: int):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(linear)}")
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.linear = linear
        self.r = int(r)
        self.alpha = int(alpha)
        self.scale = self.alpha / self.r

        in_f = linear.in_features
        out_f = linear.out_features

        # A: (r, in_f), B: (out_f, r)
        self.lora_A = nn.Parameter(torch.empty(self.r, in_f))
        self.lora_B = nn.Parameter(torch.empty(out_f, self.r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # 使初始 delta=0，训练更稳定

        # 冻结原线性层权重/偏置
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.linear(x)
        # delta = (x @ A.T) @ B.T
        delta = (x @ self.lora_A.T) @ self.lora_B.T
        return out + delta * self.scale


def inject_lora(model: nn.Module, r: int, alpha: int, target_modules, freeze_other: bool = True):
    """
    对模型中名称包含 target_modules 任一子串的 nn.Linear 替换为 LoRALinear。

    - target_modules: 例如 ['qkv','proj'] 或 ['qkv','proj','fc1','fc2']
    - freeze_other=True: 替换成功后，冻结除 LoRA 参数外的所有参数
    """
    logger = logging.getLogger(__name__)

    if not isinstance(target_modules, (list, tuple)):
        target_modules = [target_modules]

    to_replace = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue
        parts = name.split(".")
        parent = model.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else model
        child_name = parts[-1]
        to_replace.append((name, parent, child_name, module))

    for name, parent, child_name, linear in to_replace:
        setattr(parent, child_name, LoRALinear(linear, r=r, alpha=alpha))

    replaced = len(to_replace)
    if freeze_other and replaced > 0:
        for n, p in model.named_parameters():
            if ("lora_A" not in n) and ("lora_B" not in n):
                p.requires_grad = False

    logger.info(
        f"=> LoRA injected: r={r}, alpha={alpha}, targets={list(target_modules)}, "
        f"replaced {replaced} Linear(s), freeze_other={freeze_other}"
    )
    return model
