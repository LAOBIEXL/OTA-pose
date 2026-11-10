#!/usr/bin/python
# -*- coding:utf8 -*-
"""
OTA-Pose 的训练、验证和测试流程，以及可视化输出和指标统计。
"""
import time
import torch
import numpy as np
import os
import os.path as osp
import logging
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter
from .evaludate import accuracy, pck_accuracy, pck_accuracy_origin_image
from engine.core import CORE_FUNCTION_REGISTRY, BaseFunction, AverageMeter

# from engine.evaludate import accuracy
from engine.defaults import VAL_PHASE, TEST_PHASE, TRAIN_PHASE
from datasets.process import dark_get_final_preds
from datasets.transforms import reverse_transforms

from utils.utils_bbox import cs2box
from utils.utils_folder import create_folder
from utils.utils_image_tensor import tensor2im

from tabulate import tabulate


def cnt_grad(model):
    """
    计算模型所有参数梯度的总范数（Total Norm）

    用于监测训练过程中梯度的大小，帮助判断模型是否正常收敛或是否存在梯度消失/爆炸问题。

    Args:
        model: PyTorch模型对象

    Returns:
        float: 所有参数梯度范数的总和
    """
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm().item()

    return total_norm


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _fmt_count(value):
    return f"{int(value):,}"


def _count_unique_parameters(parameters):
    seen = set()
    total = 0
    trainable = 0
    for param in parameters:
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def _get_submodule(root, module_name):
    if hasattr(root, "get_submodule"):
        return root.get_submodule(module_name)

    module = root
    for item in module_name.split("."):
        module = getattr(module, item)
    return module


def _deploy_module_names(model_ref):
    run_single_lrb = bool(
        getattr(model_ref, "train_single_branch_lrb", False)
        or getattr(model_ref, "inference_single_branch_lrb", False)
    )

    module_names = ["backbone"]
    if not run_single_lrb:
        module_names.extend([
            "fine_grain_deconv",
            "fg_pruned_blocks",
            "fg_pruned_norm",
        ])

    module_names.extend([
        "pos_emb_list",
        "prune_recover_blocks",
        "multi_grain_norm",
    ])

    if not run_single_lrb:
        module_names.append("multi_grain_fusion")

    module_names.append("up_sample")
    if getattr(model_ref, "final_fusion_backbone", False):
        module_names.append("out_backbone_fusion")
    module_names.append("final_layer")

    return tuple(name for name in module_names if hasattr(model_ref, name))


def _count_unique_module_parameters(root, module_names):
    seen = set()
    total = 0
    trainable = 0
    included = []

    for module_name in module_names:
        try:
            module = _get_submodule(root, module_name)
        except AttributeError:
            continue

        included.append(module_name)
        for param in module.parameters():
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            n = param.numel()
            total += n
            if param.requires_grad:
                trainable += n

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "modules": tuple(included),
    }


def _log_parameter_summary(model, logger, stage):
    model_ref = _unwrap_model(model)
    train_params = _count_unique_parameters(model_ref.parameters())
    deploy_module_names = _deploy_module_names(model_ref)
    deploy_params = _count_unique_module_parameters(model_ref, deploy_module_names)

    # ---- Per-module breakdown (modules counted individually, so shared params may be double-counted) ----
    per_module_rows = []
    backbone_stats = None
    for module_name in deploy_module_names:
        stats = _count_unique_module_parameters(model_ref, [module_name])
        row = {
            "module": module_name,
            "total": stats["total"],
            "trainable": stats["trainable"],
            "frozen": stats["frozen"],
        }
        per_module_rows.append(row)
        if module_name == "backbone":
            backbone_stats = row

    logger.info("=> [%s] === Params Analysis ===", stage)
    logger.info(
        "=> [%s] Train/Full Model Params #: %s (%.3f M); trainable %s (%.3f M); frozen %s (%.3f M)",
        stage,
        _fmt_count(train_params["total"]),
        train_params["total"] / 1e6,
        _fmt_count(train_params["trainable"]),
        train_params["trainable"] / 1e6,
        _fmt_count(train_params["frozen"]),
        train_params["frozen"] / 1e6,
    )
    logger.info(
        "=> [%s] Val/Deploy Params #: %s (%.3f M); trainable %s (%.3f M); frozen %s (%.3f M)",
        stage,
        _fmt_count(deploy_params["total"]),
        deploy_params["total"] / 1e6,
        _fmt_count(deploy_params["trainable"]),
        deploy_params["trainable"] / 1e6,
        _fmt_count(deploy_params["frozen"]),
        deploy_params["frozen"] / 1e6,
    )

    if backbone_stats is not None:
        logger.info(
            "=> [%s] Backbone Params only  : %s (%.3f M); trainable %s (%.3f M); frozen %s (%.3f M)",
            stage,
            _fmt_count(backbone_stats["total"]),
            backbone_stats["total"] / 1e6,
            _fmt_count(backbone_stats["trainable"]),
            backbone_stats["trainable"] / 1e6,
            _fmt_count(backbone_stats["frozen"]),
            backbone_stats["frozen"] / 1e6,
        )

    logger.info(
        "=> [%s] Val/Deploy modules counted: %s",
        stage,
        ", ".join(deploy_params["modules"]),
    )

    logger.info("=> [%s] --- Per-Module Params Breakdown ---", stage)
    logger.info(
        "=> [%s] (each module counted independently; shared parameters may be counted more than once, "
        "so module sum is >= Val/Deploy total)",
        stage,
    )
    # Group modules into semantic buckets so the log is easy to scan.
    bucket_map = [
        ("backbone", lambda n: n == "backbone"),
        ("fine_branch", lambda n: n in ("fine_grain_deconv", "fg_pruned_blocks", "fg_pruned_norm")),
        ("coarse_branch", lambda n: n in ("pos_emb_list", "prune_recover_blocks")),
        ("multi_grain", lambda n: n in ("multi_grain_norm", "multi_grain_fusion")),
        ("output_head", lambda n: n in ("up_sample", "out_backbone_fusion", "final_layer")),
    ]
    shown = set()
    for bucket_label, predicate in bucket_map:
        bucket_rows = [r for r in per_module_rows if predicate(r["module"]) and r["module"] not in shown]
        if not bucket_rows:
            continue
        bucket_total = sum(r["total"] for r in bucket_rows)
        bucket_trainable = sum(r["trainable"] for r in bucket_rows)
        bucket_frozen = sum(r["frozen"] for r in bucket_rows)
        mod_names = "+".join(r["module"] for r in bucket_rows)
        logger.info(
            "=> [%s]   %-14s %s (%.3f M); tr %s (%.3f M); fr %s (%.3f M)  [%s]",
            stage,
            bucket_label + ":",
            _fmt_count(bucket_total),
            bucket_total / 1e6,
            _fmt_count(bucket_trainable),
            bucket_trainable / 1e6,
            _fmt_count(bucket_frozen),
            bucket_frozen / 1e6,
            mod_names,
        )
        for r in bucket_rows:
            shown.add(r["module"])
            logger.info(
                "=> [%s]       %-26s %s (%.3f M); tr %s (%.3f M); fr %s (%.3f M)",
                stage,
                r["module"] + ":",
                _fmt_count(r["total"]),
                r["total"] / 1e6,
                _fmt_count(r["trainable"]),
                r["trainable"] / 1e6,
                _fmt_count(r["frozen"]),
                r["frozen"] / 1e6,
            )
    # Any remaining modules that did not fall into a bucket.
    leftover = [r for r in per_module_rows if r["module"] not in shown]
    for r in leftover:
        logger.info(
            "=> [%s]   %-14s %s (%.3f M); tr %s (%.3f M); fr %s (%.3f M)  [other]",
            stage,
            r["module"] + ":",
            _fmt_count(r["total"]),
            r["total"] / 1e6,
            _fmt_count(r["trainable"]),
            r["trainable"] / 1e6,
            _fmt_count(r["frozen"]),
            r["frozen"] / 1e6,
        )
    logger.info("=> [%s] --- End Per-Module Breakdown ---", stage)

    logger.info(
        "=> [%s] Params note: unique nn.Parameter objects are counted; state_dict buffers such as BatchNorm running stats are excluded.",
        stage,
    )


def _test(model, logger=None, stage="Eval", cfg=None):
    """
    模型性能测试函数

    使用随机输入数据对模型进行性能评估，包括：
    - 模型FLOPS（使用torchprofile返回的MACs口径）
    - 推理时间测量
    - FPS（每秒帧率）计算

    该函数用于评估模型的计算效率，而非训练性能。

    Args:
        model: 待测试的PyTorch模型（需支持profile_macs）
    """
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return

    if logger is None:
        logger = logging.getLogger(__name__)

    def _cfg_get(node, name, default):
        return getattr(node, name, default) if node is not None else default

    def _profile(module, inputs, name):
        try:
            return profile_macs(module, inputs)
        except Exception as exc:
            logger.warning("=> [%s] FLOPs profile skipped for %s: %s", stage, name, exc)
            return 0

    def _fmt_flops(value):
        return f"{int(value):,}"

    model_ref = _unwrap_model(model)
    was_training = model_ref.training
    _log_parameter_summary(model_ref, logger, stage)
    frame_num = int(getattr(model_ref, "frame_num", 5))
    image_size = _cfg_get(getattr(cfg, "MODEL", None), "IMAGE_SIZE", [192, 256])
    h = int(image_size[1]) if len(image_size) > 1 else 256
    w = int(image_size[0]) if len(image_size) > 0 else 192
    c = frame_num * 3
    device = next(model_ref.parameters()).device
    channel_num = c // frame_num

    debug_cfg = getattr(cfg, "DEBUG", None)
    flops_batch_size = int(_cfg_get(debug_cfg, "PROFILE_FLOPS_BATCH_SIZE", 1))
    speed_batch_size = int(_cfg_get(debug_cfg, "PROFILE_BATCH_SIZE", 1))
    warmup_iters = int(_cfg_get(debug_cfg, "PROFILE_WARMUP", 20))
    num_iterations = int(_cfg_get(debug_cfg, "PROFILE_ITERS", 100))
    flops_batch_size = max(1, flops_batch_size)
    speed_batch_size = max(1, speed_batch_size)
    warmup_iters = max(0, warmup_iters)
    num_iterations = max(1, num_iterations)

    random_x = torch.randn((flops_batch_size, c, h, w), device=device)

    model_ref.eval()

    # 1. Full model FLOPs using the torchprofile MACs convention.
    full_flops = int(_profile(model_ref, random_x, "full_model"))
    full_flops_per_clip = full_flops / flops_batch_size
    run_single_lrb = bool(
        getattr(model_ref, "train_single_branch_lrb", False)
        or getattr(model_ref, "inference_single_branch_lrb", False)
    )
    logger.info(f"=> [{stage}] === FLOPs Analysis ===")
    logger.info(
        "=> [%s] Profile input: batch=%d, frames=%d, image=%dx%d",
        stage, flops_batch_size, frame_num, h, w
    )
    logger.info(
        "=> [%s] 1. Full Model FLOPs #: %s total, %s / clip",
        stage, _fmt_flops(full_flops), _fmt_flops(full_flops_per_clip)
    )
    logger.info(
        "=> [%s] Note: FLOPs are reported as torchprofile MACs.",
        stage
    )
    logger.info(
        "=> [%s] Note: token-selection overhead such as cdist/topk/sort may be excluded if torchprofile does not support the op.",
        stage
    )
    if run_single_lrb:
        logger.info(
            "=> [%s] Inference path: single LRB branch; fine-grain and multi-grain fusion are skipped in the real forward.",
            stage
        )

    # 2. Backbone FLOPs
    img_cat_batch = torch.cat(random_x.split(channel_num, dim=1), dim=0)
    backbone_flops = int(_profile(model_ref.backbone, img_cat_batch, "backbone"))
    logger.info(
        "=> [%s] 2. Backbone FLOPs #: %s total, %s / clip (%.1f%%)",
        stage, _fmt_flops(backbone_flops), _fmt_flops(backbone_flops / flops_batch_size),
        backbone_flops / max(1, full_flops) * 100
    )

    # Get intermediate feature shapes
    with torch.no_grad():
        backbone_out = model_ref.backbone(img_cat_batch)
        if isinstance(backbone_out, tuple):
            _, feats_backbone_out_cat_batch, _, feats_up2_backbone_out_cat_batch = backbone_out[:4]
        else:
            raise ValueError("OTAPose backbone must return heatmaps/features tuple for FLOPs breakdown.")

        b = flops_batch_size
        feats_backbone_out_list = list(feats_backbone_out_cat_batch.split(b, dim=0))
        key_frame_idx = frame_num // 2
        feats_backbone_out_keyframe = feats_backbone_out_list[key_frame_idx]
        _, _, c_fg = feats_backbone_out_keyframe.shape
        Hp, Wp = model_ref.map_size_up_1x[0], model_ref.map_size_up_1x[1]
        fine_deconv_in = feats_backbone_out_keyframe.reshape(b, Hp, Wp, c_fg).permute(0, 3, 1, 2).contiguous()
        fine_total_tokens = (Hp * 2) * (Wp * 2)
        coarse_input = torch.cat(feats_backbone_out_list, dim=1)

    # 3. Fine-grain branch FLOPs: key-frame deconv + transformer on kept fine tokens.
    fine_macs = 0
    fine_keep_tokens = 0
    if run_single_lrb:
        fine_flops = 0
        logger.info(
            "=> [%s] 3. Fine-grain Branch FLOPs #: skipped because single-LRB inference does not execute this branch",
            stage
        )
    else:
        fine_keep_tokens = int(getattr(model_ref, "token_keep_stats", {}).get("fine_keep_tokens", 0))
        fine_keep_tokens = max(1, fine_keep_tokens)
        if not getattr(model_ref, "no_prune", False):
            fine_keep_tokens = min(fine_keep_tokens, fine_total_tokens)
        else:
            fine_keep_tokens = fine_total_tokens

        if hasattr(model_ref, "fine_grain_deconv"):
            fine_macs += _profile(model_ref.fine_grain_deconv, fine_deconv_in, "fine_grain_deconv")
        if hasattr(model_ref, "fg_pruned_blocks"):
            fg_input = torch.randn((b, fine_keep_tokens, c_fg), device=device)
            for i, blk in enumerate(model_ref.fg_pruned_blocks):
                fine_macs += _profile(blk, fg_input, f"fg_pruned_blocks.{i}")
                with torch.no_grad():
                    fg_input = blk(fg_input)
            if hasattr(model_ref, "fg_pruned_norm"):
                fine_macs += _profile(model_ref.fg_pruned_norm, fg_input, "fg_pruned_norm")

        fine_flops = int(fine_macs)
        logger.info(
            "=> [%s] 3. Fine-grain Branch FLOPs #: %s total, %s / clip (%.1f%%); kept tokens=%d/%d",
            stage, _fmt_flops(fine_flops), _fmt_flops(fine_flops / flops_batch_size),
            fine_flops / max(1, full_flops) * 100, fine_keep_tokens, fine_total_tokens
        )

    # 4. Coarse-grain branch FLOPs: STE/TTE + prune/recover module with actual token path.
    coarse_macs = 0
    if hasattr(model_ref, "prune_recover_blocks"):
        coarse_macs = _profile(model_ref.prune_recover_blocks, coarse_input, "prune_recover_blocks")
    coarse_flops = int(coarse_macs)
    coarse_total_tokens = int(coarse_input.shape[1])
    if getattr(model_ref, "no_prune", False):
        coarse_keep_tokens = coarse_total_tokens
    else:
        prune_recover_blocks = getattr(model_ref, "prune_recover_blocks", None)
        coarse_keep_tokens = int(getattr(prune_recover_blocks, "prune_token_num", coarse_total_tokens))
    logger.info(
        "=> [%s] 4. Coarse-grain Branch FLOPs #: %s total, %s / clip (%.1f%%); kept tokens=%d/%d",
        stage, _fmt_flops(coarse_flops), _fmt_flops(coarse_flops / flops_batch_size),
        coarse_flops / max(1, full_flops) * 100, coarse_keep_tokens, coarse_total_tokens
    )

    # 5. Multi-grain fusion FLOPs: coarse current-frame tokens query fine kept tokens.
    cross_macs = 0
    if hasattr(model_ref, "multi_grain_fusion") and not run_single_lrb:
        mg_out_input = torch.randn((b, model_ref.token_num, c_fg), device=device)
        mg_feat_2x_input = torch.randn((b, fine_keep_tokens, c_fg), device=device)
        for i, blk in enumerate(model_ref.multi_grain_fusion):
            cross_macs += _profile(blk, (mg_out_input, mg_feat_2x_input), f"multi_grain_fusion.{i}")
            with torch.no_grad():
                mg_out_input = blk(mg_out_input, mg_feat_2x_input)
        if hasattr(model_ref, "multi_grain_norm"):
            cross_macs += _profile(model_ref.multi_grain_norm, mg_out_input, "multi_grain_norm")
    elif run_single_lrb:
        logger.info(f"=> [{stage}] 5. Multi-grain Fusion FLOPs #: skipped because single-LRB inference does not execute this branch")

    cross_flops = int(cross_macs)
    if cross_flops > 0:
        logger.info(
            "=> [%s] 5. Multi-grain Fusion FLOPs #: %s total, %s / clip (%.1f%%)",
            stage, _fmt_flops(cross_flops), _fmt_flops(cross_flops / flops_batch_size),
            cross_flops / max(1, full_flops) * 100
        )

    # 6. Output head FLOPs
    head_macs = 0
    if hasattr(model_ref, 'up_sample'):
        up_input = torch.randn((b, c_fg, Hp, Wp), device=device)
        head_macs += _profile(model_ref.up_sample, up_input, "up_sample")

        with torch.no_grad():
            up_out = model_ref.up_sample(up_input)

        if up_out.shape[1] != 128:
            up_out = torch.randn((b, 128, up_out.shape[2], up_out.shape[3]), device=device)

        if hasattr(model_ref, 'out_backbone_fusion') and getattr(model_ref, 'final_fusion_backbone', False):
            feat_up2 = torch.randn((b, 256, up_out.shape[2], up_out.shape[3]), device=device)
            concat_out = torch.cat((up_out, feat_up2), dim=1)
            head_macs += _profile(model_ref.out_backbone_fusion, concat_out, "out_backbone_fusion")
            final_in = torch.randn((b, 128, up_out.shape[2], up_out.shape[3]), device=device)
        else:
            final_in = up_out

        if hasattr(model_ref, 'final_layer'):
            head_macs += _profile(model_ref.final_layer, final_in, "final_layer")

    head_flops = int(head_macs)
    logger.info(
        "=> [%s] 6. Output Head FLOPs #: %s total, %s / clip (%.1f%%)",
        stage, _fmt_flops(head_flops), _fmt_flops(head_flops / flops_batch_size),
        head_flops / max(1, full_flops) * 100
    )

    component_sum = backbone_flops + fine_flops + coarse_flops + cross_flops + head_flops
    logger.info(f"=> [{stage}] ======================")
    logger.info(f"=> [{stage}] FLOPs Breakdown Summary:")
    logger.info("=> [%s]   Backbone:         %s", stage, _fmt_flops(backbone_flops))
    logger.info("=> [%s]   Fine-grain:       %s", stage, _fmt_flops(fine_flops))
    logger.info("=> [%s]   Coarse-grain:     %s", stage, _fmt_flops(coarse_flops))
    logger.info("=> [%s]   Multi-grain fuse: %s", stage, _fmt_flops(cross_flops))
    logger.info("=> [%s]   Output head:      %s", stage, _fmt_flops(head_flops))
    logger.info("=> [%s]   --------------------------", stage)
    logger.info("=> [%s]   Component Sum:    %s", stage, _fmt_flops(component_sum))
    logger.info("=> [%s]   Full Model:       %s", stage, _fmt_flops(full_flops))
    logger.info("=> [%s]   Diff:             %s", stage, _fmt_flops(full_flops - component_sum))
    logger.info(f"=> [{stage}] ======================")

    # FPS calculation
    speed_x = torch.randn((speed_batch_size, c, h, w), device=device)
    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = model_ref(speed_x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model_ref(speed_x)
        end_event.record()
        torch.cuda.synchronize(device)
        elapsed = start_event.elapsed_time(end_event) / 1000.0
    else:
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model_ref(speed_x)
        elapsed = time.time() - start_time

    clips_per_s = (num_iterations * speed_batch_size) / max(elapsed, 1e-12)
    input_frames_per_s = clips_per_s * frame_num
    logger.info(
        "=> [%s] Throughput: %.4f clips/s (batch=%d, warmup=%d, iters=%d)",
        stage, clips_per_s, speed_batch_size, warmup_iters, num_iterations
    )
    logger.info(
        "=> [%s] Input-frame throughput: %.4f frames/s; pose-output FPS: %.4f",
        stage, input_frames_per_s, clips_per_s
    )
    logger.info(
        "=> [%s] FPS: %.4f",
        stage, clips_per_s
    )

    backbone_speed_x = torch.randn((speed_batch_size * frame_num, channel_num, h, w), device=device)
    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = model_ref.backbone(backbone_speed_x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model_ref.backbone(backbone_speed_x)
        end_event.record()
        torch.cuda.synchronize(device)
        backbone_elapsed = start_event.elapsed_time(end_event) / 1000.0
    else:
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model_ref.backbone(backbone_speed_x)
        backbone_elapsed = time.time() - start_time

    backbone_images_per_s = (num_iterations * speed_batch_size * frame_num) / max(backbone_elapsed, 1e-12)
    backbone_clips_per_s = backbone_images_per_s / frame_num
    logger.info(
        "=> [%s] Backbone FPS: %.4f images/s",
        stage, backbone_images_per_s
    )
    logger.info(
        "=> [%s] Backbone clip FPS: %.4f %d-frame-clips/s",
        stage, backbone_clips_per_s, frame_num
    )
    logger.info(
        "=> [%s] Backbone throughput detail: %.4f images/s, %.4f %d-frame-clips/s (batch=%d clips -> backbone batch=%d images)",
        stage, backbone_images_per_s, backbone_clips_per_s, frame_num, speed_batch_size, speed_batch_size * frame_num
    )

    model_ref.train(was_training)


def _get_token_keep_stats(model):
    # 支持 DataParallel 和 DDP
    model_ref = model.module if hasattr(model, 'module') else model
    return getattr(model_ref, "token_keep_stats", None)


def _get_token_keep_history(model):
    # 支持 DataParallel 和 DDP
    model_ref = model.module if hasattr(model, 'module') else model
    return getattr(model_ref, "token_keep_history", None)


def _reset_token_keep_history(model):
    # 支持 DataParallel 和 DDP
    model_ref = model.module if hasattr(model, 'module') else model
    reset_fn = getattr(model_ref, "reset_token_keep_history", None)
    if callable(reset_fn):
        reset_fn()


def _build_token_keep_summary(history):
    if not history:
        return None

    fine_tokens = history.get("fine_keep_tokens", [])
    coarse_tokens = history.get("coarse_keep_tokens", [])
    fine_ratios = history.get("fine_keep_ratio", [])
    coarse_ratios = history.get("coarse_keep_ratio", [])

    if not fine_tokens or not coarse_tokens:
        return None

    summary = {
        "num_forwards": int(len(fine_tokens)),
        "fine_keep_tokens_min": int(min(fine_tokens)),
        "fine_keep_tokens_max": int(max(fine_tokens)),
        "fine_keep_tokens_avg": float(sum(fine_tokens) / len(fine_tokens)),
        "coarse_keep_tokens_min": int(min(coarse_tokens)),
        "coarse_keep_tokens_max": int(max(coarse_tokens)),
        "coarse_keep_tokens_avg": float(sum(coarse_tokens) / len(coarse_tokens)),
        "fine_keep_ratio_min": float(min(fine_ratios)) if fine_ratios else 0.0,
        "fine_keep_ratio_max": float(max(fine_ratios)) if fine_ratios else 0.0,
        "fine_keep_ratio_avg": float(sum(fine_ratios) / len(fine_ratios)) if fine_ratios else 0.0,
        "coarse_keep_ratio_min": float(min(coarse_ratios)) if coarse_ratios else 0.0,
        "coarse_keep_ratio_max": float(max(coarse_ratios)) if coarse_ratios else 0.0,
        "coarse_keep_ratio_avg": float(sum(coarse_ratios) / len(coarse_ratios)) if coarse_ratios else 0.0,
    }
    return summary


def _format_token_keep_stats(stats):
    if not stats:
        return ""
    return (
        'Prune Fine {}/{} ({:.1%}) '
        'Coarse {}/{} ({:.1%})'
    ).format(
        int(stats.get("fine_keep_tokens", 0)),
        int(stats.get("fine_total_tokens", 0)),
        float(stats.get("fine_keep_ratio", 0.0)),
        int(stats.get("coarse_keep_tokens", 0)),
        int(stats.get("coarse_total_tokens", 0)),
        float(stats.get("coarse_keep_ratio", 0.0)),
    )


def _parse_spatiotemporal_batch(batch):
    """Normalize 3-frame and 5-frame dataset batches to the 5-value core format."""
    if len(batch) == 5:
        input_x, ref_data_numpy_list, target_heatmaps, target_heatmaps_weight, meta = batch
    elif len(batch) == 6:
        input_x, input_prev, input_next, target_heatmaps, target_heatmaps_weight, meta = batch
        ref_data_numpy_list = torch.cat((input_prev, input_next), 1)
    else:
        raise ValueError(f"Unsupported spatiotemporal batch format with {len(batch)} values.")

    return input_x, ref_data_numpy_list, target_heatmaps, target_heatmaps_weight, meta


@CORE_FUNCTION_REGISTRY.register()
class Function_No_Supp_Targ_5_Frame_OT(BaseFunction):
    """
    5帧无支持目标最优传输训练评估类

    该类是DC-Pose的核心功能类，专门用于处理5帧输入的姿态估计任务。
    支持最优传输（Optimal Transport, OT）损失，用于衡量和优化帧间的特征匹配。
    继承自BaseFunction，提供标准的训练/验证/测试接口。

    主要特性：
    - 5帧输入处理：当前帧 + 4个参考帧
    - 最优传输损失支持：用于衡量热力图之间的传输成本
    - 混合精度支持：可选FP16训练
    - 梯度累积：支持大batch训练
    - 可视化支持：TensorBoard和图像输出
    """

    def __init__(self, cfg, **kwargs):
        """
        初始化训练评估类

        初始化各种训练状态变量和配置参数，建立与外部组件（如criterion、tensorboard）的连接。

        Args:
            cfg: 配置对象，包含所有训练和模型配置
            **kwargs: 可选关键字参数：
                - criterion: 损失函数计算器
                - tb_log_dir: TensorBoard日志目录
                - writer_dict: TensorBoard writer字典
                - PE_Name: 姿态估计名称，默认为"OTAPOSE"
        """
        super().__init__()
        self.cfg = cfg
        self.output_dir = cfg.OUTPUT_DIR

        if "criterion" in kwargs.keys():
            self.criterion = kwargs["criterion"]
        if "tb_log_dir" in kwargs.keys():
            self.tb_log_dir = kwargs["tb_log_dir"]
        if "writer_dict" in kwargs.keys():
            self.writer_dict = kwargs["writer_dict"]

        self.PE_Name = kwargs.get("PE_Name", "OTAPOSE")
        ##
        self.max_iter_num = 0  # 当前epoch的最大迭代次数
        self.dataloader_iter = None  # 数据加载器迭代器
        self.tb_writer = None  # TensorBoard写入器
        self.global_steps = 0  # 全局训练步数（跨epoch累加）
        self.DataSetName = str(self.cfg.DATASET.NAME).upper()  # 数据集名称（大写）
        self._logged_train_params = False

    def train(self, model, epoch, optimizer, dataloader, tb_writer_dict, **kwargs):
        """
        训练阶段主循环

        执行一个epoch的训练，包含以下核心步骤：
        1. 数据加载与预处理
        2. 前向传播（拼接当前帧与参考帧）
        3. 损失计算（包含热力图损失 + OT损失）
        4. 反向传播与参数更新
        5. 指标统计与日志记录
        6. TensorBoard可视化

        Args:
            model: 待训练的PyTorch模型
            epoch: 当前训练轮次
            optimizer: 优化器（如Adam、SGD等）
            dataloader: 训练数据加载器
            tb_writer_dict: TensorBoard写入器字典，包含writer和global_steps
            **kwargs: 其他可选参数
        """
        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]
        logger = logging.getLogger(__name__)
        _reset_token_keep_history(model)
        batch_time = AverageMeter()  # 批次处理时间统计
        data_time = AverageMeter()  # 数据加载时间统计
        losses = AverageMeter()  # 损失值统计
        acc = AverageMeter()  # 准确率统计
        grad = AverageMeter()  # 梯度范数统计
        # 切换到训练模式（启用dropout、batchNorm等）
        model.train()
        if not self._logged_train_params:
            _log_parameter_summary(model, logger, "Train")
            self._logged_train_params = True

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        end = time.time()

        for iter_step in range(self.max_iter_num):
            # 从数据加载器获取一个批次的数据
            # input_x: 当前帧图像, ref_data_numpy_list: 参考帧列表
            # target_heatmaps: 目标热力图, target_heatmaps_weight: 热力图权重
            # meta: 元数据（图像路径、中心点、尺度等）
            input_x, ref_data_numpy_list, target_heatmaps, target_heatmaps_weight, meta = _parse_spatiotemporal_batch(
                next(self.dataloader_iter)
            )

            self._before_train_iter(input_x)

            data_time.update(time.time() - end)

            # 将目标热力图和权重移至GPU
            target_heatmaps = target_heatmaps.cuda(non_blocking=True)
            target_heatmaps_weight = target_heatmaps_weight.cuda(non_blocking=True)

            # 直接连接当前帧和参考帧（参考frame.py的做法）
            # 通道维度拼接：input_x [B,3,H,W] + ref_data_numpy_list [B,12,H,W] -> [B,15,H,W]
            concat_input = torch.cat((input_x, ref_data_numpy_list), 1).cuda()

            # 处理模型输出，支持OT损失和单分支辅助损失
            # 模型输出格式：2元素或3元素或4元素 tuple
            model_output = model(concat_input)
            out_single_branch = None
            if isinstance(model_output, (list, tuple)) and len(model_output) == 4:
                outputs, htms_backbone, ot_loss, out_single_branch = model_output
            elif isinstance(model_output, (list, tuple)) and len(model_output) == 3:
                outputs, htms_backbone, ot_loss = model_output
            elif isinstance(model_output, (list, tuple)) and len(model_output) == 2:
                outputs, htms_backbone = model_output
                ot_loss = 0.0
            else:
                raise ValueError("Model output must be a tuple/list of 2, 3, or 4 elements.")

            # 处理多尺度输出情况，计算总损失
            if isinstance(outputs, (list, tuple)):
                pred_heatmaps = outputs[0]
                loss = self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
                # 累加所有尺度的损失
                for pred_heat in outputs[1:]:
                    loss += self.criterion(pred_heat, target_heatmaps, target_heatmaps_weight)
            else:
                pred_heatmaps = outputs
                loss = self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)

            # 添加单分支辅助损失（粗粒度分支不经 cross attention 直接走输出头产生的结果与 GT 的差）
            if out_single_branch is not None:
                cfg_extra = getattr(self.cfg.MODEL, "EXTRA", None)
                sb_weight = float(getattr(cfg_extra, "SINGLE_BRANCH_LOSS_WEIGHT", 0.1)) if cfg_extra is not None else 0.1
                loss = loss + sb_weight * self.criterion(out_single_branch, target_heatmaps, target_heatmaps_weight)

            # 添加OT损失（最优传输损失，用于衡量帧间特征匹配质量）
            if torch.is_tensor(ot_loss):
                loss += ot_loss
            elif ot_loss > 0:
                loss += ot_loss

            # 计算梯度并执行参数更新
            optimizer.zero_grad()
            loss.backward()
            grad.update(cnt_grad(model))
            optimizer.step()

            # 统计损失值
            losses.update(loss.item(), input_x.size(0))

            # 计算并统计准确率
            _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            # 统计批次处理时间
            batch_time.update(time.time() - end)
            end = time.time()

            # 定期打印训练日志
            if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= self.max_iter_num - 1:
                # 获取feat_max值（如果模型有这个属性）
                feat_max_val = 0.0
                if hasattr(model, 'module') and hasattr(model.module, 'last_feat_max'):
                    feat_max_val = model.module.last_feat_max.item() if torch.is_tensor(model.module.last_feat_max) else float(model.module.last_feat_max)
                elif hasattr(model, 'last_feat_max'):
                    feat_max_val = model.last_feat_max.item() if torch.is_tensor(model.last_feat_max) else float(model.last_feat_max)

                # 获取当前batch的token_keep信息
                token_keep_str = _format_token_keep_stats(_get_token_keep_stats(model))

                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'OT Loss {ot_loss_val:.5f}\t' \
                      'Feat Max {feat_max_val:.5f}\t' \
                      'Grad {grad.val:.3f} ({grad.avg:.3f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
                      '{token_keep_str}'.format(epoch, iter_step, self.max_iter_num,
                                                                        batch_time=batch_time,
                                                                        speed=input_x.size(0) / batch_time.val,
                                                                        data_time=data_time, grad=grad, loss=losses,
                                                                        ot_loss_val=ot_loss.item() if torch.is_tensor(ot_loss) else float(ot_loss),
                                                                        feat_max_val=feat_max_val,
                                                                        acc=acc,
                                                                        token_keep_str=token_keep_str)

                logger.info(msg)

            # TensorBoard记录
            if self.tb_writer is not None:
                self.tb_writer.add_scalar('train_loss', losses.val, self.global_steps)
                self.tb_writer.add_scalar('train_acc', acc.val, self.global_steps)
            self.global_steps += 1

        tb_writer_dict["global_steps"] = self.global_steps

        # 打印训练阶段token_keep汇总
        token_keep_summary = _build_token_keep_summary(_get_token_keep_history(model))
        if token_keep_summary is not None:
            logger.info(
                "=> TokenKeep Train Summary ({} forwards): "
                "Fine keep[min/max/avg]={}/{}/{:.2f}, Fine ratio[min/max/avg]={:.2%}/{:.2%}/{:.2%}; "
                "Coarse keep[min/max/avg]={}/{}/{:.2f}, Coarse ratio[min/max/avg]={:.2%}/{:.2%}/{:.2%}".format(
                    token_keep_summary["num_forwards"],
                    token_keep_summary["fine_keep_tokens_min"],
                    token_keep_summary["fine_keep_tokens_max"],
                    token_keep_summary["fine_keep_tokens_avg"],
                    token_keep_summary["fine_keep_ratio_min"],
                    token_keep_summary["fine_keep_ratio_max"],
                    token_keep_summary["fine_keep_ratio_avg"],
                    token_keep_summary["coarse_keep_tokens_min"],
                    token_keep_summary["coarse_keep_tokens_max"],
                    token_keep_summary["coarse_keep_tokens_avg"],
                    token_keep_summary["coarse_keep_ratio_min"],
                    token_keep_summary["coarse_keep_ratio_max"],
                    token_keep_summary["coarse_keep_ratio_avg"],
                )
            )

    def eval(self, model, dataloader, tb_writer_dict, **kwargs):
        """
        验证/测试阶段主循环

        执行模型评估，包含以下核心步骤：
        1. 数据加载与预处理
        2. 前向传播（推理模式，无梯度计算）
        3. 精度计算（主输出和backbone输出）
        4. 预测结果收集与保存
        5. 数据集评估（调用数据集自带的评估接口）
        6. 可视化输出（骨架、边界框等）

        Args:
            model: 待评估的PyTorch模型
            dataloader: 验证/测试数据加载器
            tb_writer_dict: TensorBoard写入器字典
            **kwargs: 其他可选参数：
                - phase: 阶段（VAL_PHASE或TEST_PHASE）
                - epoch: 当前轮次
                - max_validate_batches: 最大验证batch数（-1表示全部）
        """
        logger = logging.getLogger(__name__)

        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]

        spatial_capture_enabled = bool(os.environ.get("OTA_SPATIAL_EXPORT_DIR", "").strip())
        if not spatial_capture_enabled:
            _test(model, logger=logger, stage="Eval", cfg=self.cfg)
        else:
            logger.info("=> Spatial capture enabled: skipping synthetic FLOPs/speed benchmarks")

        _reset_token_keep_history(model)

        # 初始化各类统计量
        batch_time, data_time, losses, acc, acc_backbone = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        phase = kwargs.get("phase", VAL_PHASE)
        epoch = kwargs.get("epoch", "specified_model")
        # -1 表示跑完整验证集；>0 表示只跑前 N 个 batch（用于周期性快速验证）
        max_validate_batches = int(kwargs.get("max_validate_batches", -1))
        # 切换到评估模式（禁用dropout、使用eval模式的batchNorm）
        model.eval()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        dataset = dataloader.dataset
        spatial_capture = None
        if spatial_capture_enabled:
            from .spatial_retention_capture import SpatialCaptureSession
            spatial_capture = SpatialCaptureSession(
                model, self.cfg, dataloader, phase=phase,
                max_validate_batches=max_validate_batches, evaluation_source=__file__,
            )
            logger.info("=> Recording ISP spatial masks to %s", spatial_capture.writer.out)
        # 准备验证用的数据结构
        num_samples = len(dataset)
        all_preds = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=float)  # 主输出预测结果
        all_preds_backbone = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=float)  # backbone预测结果
        all_boxes = np.zeros((num_samples, 6))  # 主输出边界框
        all_boxes_backbone = np.zeros((num_samples, 6))  # backbone边界框
        image_path = []
        filenames = []
        filenames_map = {}  # 文件名到索引的映射
        filenames_counter = 0
        imgnums = []
        idx = 0  # 主输出结果索引
        idx_backbone = 0  # backbone结果索引
        acc_threshold = 0.7

        # 设置可视化输出目录
        result_output_dir, vis_output_dir = self.vis_setup(logger, phase, epoch)

        with torch.no_grad():
            end = time.time()
            num_batch = self.max_iter_num
            if max_validate_batches > 0:
                # 通过截断batch数缩短验证时长，降低DDP barrier等待超时风险
                num_batch = min(self.max_iter_num, max_validate_batches)
                logger.info("{}: limit eval batches to {}".format(phase, num_batch))

            for iter_step in range(num_batch):
                # 获取验证数据
                input_x, ref_data_numpy_list, target_heatmaps, target_heatmaps_weight, meta = _parse_spatiotemporal_batch(
                    next(self.dataloader_iter)
                )

                if phase == VAL_PHASE:
                    self._before_val_iter(input_x)

                data_time.update(time.time() - end)

                target_heatmaps = target_heatmaps.cuda(non_blocking=True)

                # 拼接当前帧与参考帧作为输入
                concat_input = torch.cat((input_x, ref_data_numpy_list), 1).cuda()

                # 处理模型输出，支持OT损失和单分支辅助损失
                if spatial_capture is None:
                    model_output = model(concat_input)
                else:
                    # One unflipped validation forward; never enable a different branch.
                    model_output = spatial_capture.model(
                        concat_input, collect_prune_vis=True, force_full_vis=False,
                    )
                    spatial_capture.add_batch(meta, input_x.size(0))
                if isinstance(model_output, (list, tuple)) and len(model_output) == 4:
                    outputs, htms_backbone, ot_loss, _ = model_output
                elif isinstance(model_output, (list, tuple)) and len(model_output) == 3:
                    outputs, htms_backbone, ot_loss = model_output
                elif isinstance(model_output, (list, tuple)) and len(model_output) == 2:
                    outputs, htms_backbone = model_output
                    ot_loss = 0.0
                else:
                    raise ValueError("Model output must be a tuple/list of 2, 3, or 4 elements.")

                # 验证阶段可视化：获取模型的vis_dict用于中间特征可视化
                if phase == VAL_PHASE:
                    if isinstance(model, torch.nn.DataParallel):
                        vis_dict = getattr(model.module, "vis_dict", None)
                    else:
                        vis_dict = getattr(model, "vis_dict", None)

                    if vis_dict:
                        self._running_val_iter(vis_dict=vis_dict, model_input=[input_x, ref_data_numpy_list])

                # 使用最终输出计算主要精度
                if isinstance(outputs, (list, tuple)):
                    pred_heatmaps = outputs[0]
                else:
                    pred_heatmaps = outputs

                if getattr(self.cfg.DEBUG, "VIS_HEATMAP", False):
                    ref_frames = list(ref_data_numpy_list.split(input_x.shape[1], dim=1))
                    left_image = ref_frames[0] if ref_frames else input_x
                    right_idx = max(1, len(ref_frames) // 2)
                    if right_idx >= len(ref_frames):
                        right_idx = len(ref_frames) - 1
                    right_image = ref_frames[right_idx] if ref_frames else input_x

                    model_ref = getattr(model, "module", model)
                    htms_dict = getattr(model_ref, "htms_dict", None)
                    if htms_dict:
                        from utils.vis import save_heatmaps
                        save_heatmaps(
                            htms_dict,
                            image=[input_x, left_image, right_image],
                            target=target_heatmaps,
                            batch_id=iter_step,
                            save_dir=vis_output_dir,
                            skip_joints={3, 4},
                            image_paths=meta["image"],
                        )

                # 计算主输出的准确率
                _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(),
                                              target_heatmaps.detach().cpu().numpy())

                acc.update(avg_acc, cnt)

                # 同时报告backbone输出精度作为参考
                _, avg_acc_backbone, cnt_backbone, _ = accuracy(htms_backbone.detach().cpu().numpy(),
                                                                target_heatmaps.detach().cpu().numpy())

                acc_backbone.update(avg_acc_backbone, cnt_backbone)

                # 统计批次处理时间
                batch_time.update(time.time() - end)
                end = time.time()

                # 定期打印验证日志
                if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= (num_batch - 1):
                    token_keep_str = _format_token_keep_stats(_get_token_keep_stats(model))
                    msg = '{}: [{}/{}]\t' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t   \
                          {acc_backbone.val:.3f} ({acc_backbone.avg:.3f})\t' \
                          '{token_keep_str}'.format(phase, iter_step, num_batch,
                                                                                  batch_time=batch_time,
                                                                                  data_time=data_time, acc=acc,
                                                                                  acc_backbone=acc_backbone,
                                                                                  token_keep_str=token_keep_str)
                    logger.info(msg)

                # 为评估阶段准备数据
                for ff in range(len(meta['image'])):
                    cur_nm = meta['image'][ff]
                    if not cur_nm in filenames_map:
                        filenames_map[cur_nm] = [filenames_counter]
                    else:
                        filenames_map[cur_nm].append(filenames_counter)
                    filenames_counter += 1

                center = meta['center'].numpy()
                scale = meta['scale'].numpy()
                score = meta['score'].numpy()
                num_images = input_x.size(0)

                # 从热力图解码出最终的关键点坐标
                preds, maxvals = dark_get_final_preds(pred_heatmaps.clone().cpu().numpy(), center, scale)
                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals  # 置信度分数
                all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(scale * 200, 1)  # 面积
                all_boxes[idx:idx + num_images, 5] = score
                image_path.extend(meta['image'])
                idx += num_images

                # backbone输出同样进行解码和保存
                preds_backbone, maxvals_backbone = dark_get_final_preds(htms_backbone.clone().cpu().numpy(), center, scale)
                all_preds_backbone[idx_backbone:idx_backbone + num_images, :, 0:2] = preds_backbone[:, :, 0:2]
                all_preds_backbone[idx_backbone:idx_backbone + num_images, :, 2:3] = maxvals_backbone
                all_boxes_backbone[idx_backbone:idx_backbone + num_images, 0:2] = center[:, 0:2]
                all_boxes_backbone[idx_backbone:idx_backbone + num_images, 2:4] = scale[:, 0:2]
                all_boxes_backbone[idx_backbone:idx_backbone + num_images, 4] = np.prod(scale * 200, 1)
                all_boxes_backbone[idx_backbone:idx_backbone + num_images, 5] = score
                idx_backbone += num_images

                # TensorBoard记录和可视化后处理
                self.global_steps += 1
                self._after_val_iter(meta["image"], preds, maxvals, vis_output_dir, center, scale)



        if spatial_capture is not None:
            spatial_summary = spatial_capture.finish()
            logger.info("=> Spatial capture complete: %s", spatial_summary)

        # 打印最终评估结果
        logger.info('########################################')
        logger.info('{}'.format(self.cfg.EXPERIMENT_NAME))

        # 调用数据集的评估接口计算PCK等指标
        name_values, perf_indicator = dataset.evaluate(self.cfg, all_preds, result_output_dir, all_boxes,
                                                       filenames_map, filenames, imgnums)

        backbone_result_output_dir = result_output_dir + "_backbone"
        name_values_backbone, perf_indicator_backbone = dataset.evaluate(self.cfg, all_preds_backbone,
                                                                         backbone_result_output_dir, all_boxes_backbone,
                                                                         filenames_map, filenames, imgnums)

        # 打印backbone和主模型的评估结果
        self._print_name_value(name_values_backbone, 'ViT_Backbone')

        model_name = self.cfg.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                self._print_name_value(name_value, model_name)
        else:
            self._print_name_value(name_values, model_name)

        tb_writer_dict["global_steps"] = self.global_steps

        # 打印验证阶段token_keep汇总
        token_keep_summary = _build_token_keep_summary(_get_token_keep_history(model))
        if token_keep_summary is not None:
            logger.info(
                "=> TokenKeep Val Summary ({} forwards): "
                "Fine keep[min/max/avg]={}/{}/{:.2f}, Fine ratio[min/max/avg]={:.2%}/{:.2%}/{:.2%}; "
                "Coarse keep[min/max/avg]={}/{}/{:.2f}, Coarse ratio[min/max/avg]={:.2%}/{:.2%}/{:.2%}".format(
                    token_keep_summary["num_forwards"],
                    token_keep_summary["fine_keep_tokens_min"],
                    token_keep_summary["fine_keep_tokens_max"],
                    token_keep_summary["fine_keep_tokens_avg"],
                    token_keep_summary["fine_keep_ratio_min"],
                    token_keep_summary["fine_keep_ratio_max"],
                    token_keep_summary["fine_keep_ratio_avg"],
                    token_keep_summary["coarse_keep_tokens_min"],
                    token_keep_summary["coarse_keep_tokens_max"],
                    token_keep_summary["coarse_keep_tokens_avg"],
                    token_keep_summary["coarse_keep_ratio_min"],
                    token_keep_summary["coarse_keep_ratio_max"],
                    token_keep_summary["coarse_keep_ratio_avg"],
                )
            )

    def _before_train_iter(self, batch_x):
        """
        训练迭代前的可视化回调

        在每个训练迭代开始前，将输入图像记录到TensorBoard。
        仅在cfg.DEBUG.VIS_TENSORBOARD开启时生效。

        Args:
            batch_x: 当前批次的输入图像 [B, C, H, W]
        """
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        show_image_num = min(6, len(batch_x))
        batch_x = batch_x[:show_image_num]
        label_name = "train_{}_x".format(self.global_steps)
        save_image = []
        for x in batch_x:
            x = tensor2im(x)  # 将tensor转换为可显示的图像格式
            save_image.append(x)
        save_image = np.stack(save_image, axis=0)
        self.tb_writer.add_images(label_name, save_image, global_step=self.global_steps, dataformats="NHWC")

    def _before_val_iter(self, batch_x):
        """
        验证迭代前的可视化回调

        在每个验证迭代开始前，将输入图像记录到TensorBoard。
        仅在cfg.DEBUG.VIS_TENSORBOARD开启时生效。

        Args:
            batch_x: 当前批次的输入图像 [B, C, H, W]
        """
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        show_image_num = min(6, len(batch_x))
        batch_x = batch_x[:show_image_num]
        label_name = "val_{}_x".format(self.global_steps)
        save_image = []
        for x in batch_x:
            x = tensor2im(x)
            save_image.append(x)
        save_image = np.stack(save_image, axis=0)
        self.tb_writer.add_images(label_name, save_image, global_step=self.global_steps, dataformats="NHWC")

    def _running_val_iter(self, **kwargs):
        """
        验证迭代运行中的可视化回调

        记录验证过程中模型内部的中间结果到TensorBoard，包括：
        - 三帧输入图像（前一帧、当前帧、后一帧）
        - 三帧的粗略热力图
        - 差异热力图（current-previous, current-next）
        - 支持热力图
        - 融合特征图
        - 变形（warped）热力图列表

        用于调试和理解模型的中间表示。

        Args:
            **kwargs: 包含vis_dict（模型中间可视化字典）和model_input的字典
        """
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        vis_dict = kwargs.get("vis_dict")
        #
        show_image_num = min(3, len(vis_dict["current_x"]))
        current_x = vis_dict["current_x"][0:show_image_num]  # [N,3,384,288]
        previous_x = vis_dict["previous_x"][0:show_image_num]  # [N,3,384,288]
        next_x = vis_dict["next_x"][0:show_image_num]  # [N,3,384,288]
        current_rough_heatmaps = vis_dict["current_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        previous_rough_heatmaps = vis_dict["previous_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        next_rough_heatmaps = vis_dict["next_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        diff_A = vis_dict["diff_A"][0:show_image_num]  # [N,17,96,72] current与previous的差异
        diff_B = vis_dict["diff_B"][0:show_image_num]  # [N,17,96,72] current与next的差异
        diff_heatmaps = vis_dict["diff_heatmaps"][0:show_image_num]  # [N,34,96,72] 拼接的差异
        support_heatmaps = vis_dict["support_heatmaps"][0:show_image_num]  # [N,17,96,72]
        prf_ptm_combine_featuremaps = vis_dict["prf_ptm_combine_featuremaps"][0:show_image_num]  # [N,96,96,72]
        warped_heatmaps_list = [warped_heatmaps[0:show_image_num] for warped_heatmaps in
                                vis_dict["warped_heatmaps_list"]]  # [N,17,96,72]
        output_heatmaps = vis_dict["output_heatmaps"][0:show_image_num]  # [N,17,96,72]

        # 可视化1: 三帧输入图像（前一帧、当前帧、后一帧）
        show_three_input_image = make_grid(reverse_transforms(torch.cat([previous_x, current_x, next_x], dim=0)),
                                           nrow=show_image_num)
        self.tb_writer.add_image('01_three_input_image', show_three_input_image, global_step=self.global_steps)

        # 可视化2: 三帧的粗略热力图（通道交错排列）
        three_rough_heatmaps_channels = []
        current_rough_heatmap_channels = current_rough_heatmaps.split(1, dim=1)
        previous_rough_heatmap_channels = previous_rough_heatmaps.split(1, dim=1)
        next_rough_heatmap_channels = next_rough_heatmaps.split(1, dim=1)
        num_channel = current_rough_heatmaps.shape[1]
        for i in range(num_channel):
            three_rough_heatmaps_channels.append(current_rough_heatmap_channels[i])
            three_rough_heatmaps_channels.append(previous_rough_heatmap_channels[i])
            three_rough_heatmaps_channels.append(next_rough_heatmap_channels[i])

        three_heatmaps_tensor = torch.clamp_min(torch.cat(three_rough_heatmaps_channels, dim=0), 0)
        three_heatmaps_image = make_grid(three_heatmaps_tensor, nrow=show_image_num)
        self.tb_writer.add_image('02_three_heatmaps_image', three_heatmaps_image, global_step=self.global_steps)

        # 可视化3: 两组差异热力图（diff_A和diff_B）
        two_diff_channels = []
        diff_A_channels = diff_A.split(1, dim=1)
        diff_B_channels = diff_B.split(1, dim=1)
        num_channel = current_rough_heatmaps.shape[1]
        for i in range(num_channel):
            two_diff_channels.append(diff_A_channels[i])
            two_diff_channels.append(diff_B_channels[i])

        two_diff_channels_tensor = torch.clamp_min(torch.cat(two_diff_channels, dim=0), 0)
        two_diff_image = make_grid(two_diff_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('03_two_diff_image', two_diff_image, global_step=self.global_steps)

        # 可视化4: 拼接的差异热力图
        diff_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(diff_heatmaps, 1, dim=1), dim=0), 0)
        diff_heatmaps_channels_image = make_grid(diff_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('04_diff_heatmaps_image', diff_heatmaps_channels_image, global_step=self.global_steps)

        # 可视化5: 支持热力图
        support_heatmaps_tensor = torch.clamp_min(torch.cat(torch.split(support_heatmaps, 1, dim=1), dim=0), 0)
        support_heatmaps_image = make_grid(support_heatmaps_tensor, nrow=show_image_num)
        self.tb_writer.add_image('05_support_heatmaps_image', support_heatmaps_image, global_step=self.global_steps)

        # 可视化6: 融合特征图
        prf_ptm_combine_featuremaps_tensor = torch.clamp_min(
            torch.cat(torch.split(prf_ptm_combine_featuremaps, 1, dim=1), dim=0), 0)
        prf_ptm_combine_featuremaps_image = make_grid(prf_ptm_combine_featuremaps_tensor, nrow=show_image_num)
        self.tb_writer.add_image('06_prf_ptm_combine_featuremaps_image', prf_ptm_combine_featuremaps_image,
                                 global_step=self.global_steps)

        # 可视化7: warping后的热力图列表（展示光流变形效果）
        if len(warped_heatmaps_list) > 0:
            warped_heatmaps_tensor = torch.clamp_min(
                torch.cat([warped_heatmaps[0:show_image_num] for warped_heatmaps in warped_heatmaps_list], dim=0), 0)
            warped_heatmaps_image = make_grid(warped_heatmaps_tensor, nrow=show_image_num)
            self.tb_writer.add_image('07_warped_heatmaps_image', warped_heatmaps_image, global_step=self.global_steps)

        # 可视化8: 最终输出热力图
        output_heatmaps_tensor = torch.clamp_min(torch.cat(torch.split(output_heatmaps, 1, dim=1), dim=0), 0)
        output_heatmaps_image = make_grid(output_heatmaps_tensor, nrow=show_image_num)
        self.tb_writer.add_image('08_output_heatmaps_image', output_heatmaps_image, global_step=self.global_steps)

    def _after_val_iter(self, image_paths, preds, maxvals, vis_output_dir, center, scale):
        """
        验证迭代后的可视化回调

        在每个验证迭代结束后，将预测结果可视化并保存到文件系统。
        可视化内容包括：
        - 关键点预测结果（骨架绘制）
        - 边界框
        - 置信度热力图

        Args:
            image_paths: 图像路径列表
            preds: 预测的关键点坐标 [N, num_joints, 2]
            maxvals: 关键点置信度 [N, num_joints, 1]
            vis_output_dir: 可视化输出目录
            center: 图像中心点坐标 [N, 2]
            scale: 图像缩放因子 [N, 2]
        """
        cfg = self.cfg

        coords = np.concatenate([preds, maxvals], axis=-1)
        bboxes = []
        for index in range(len(center)):
            xyxy_bbox = cs2box(center[index], scale[index], pattern="xyxy")
            bboxes.append(xyxy_bbox)

        if cfg.DEBUG.VIS_SKELETON or cfg.DEBUG.VIS_BBOX:
            from .vis_helper import draw_skeleton_in_origin_image
            draw_skeleton_in_origin_image(image_paths, coords, bboxes, vis_output_dir, vis_skeleton=cfg.DEBUG.VIS_SKELETON,
                                          vis_bbox=cfg.DEBUG.VIS_BBOX)

    def vis_setup(self, logger, phase, epoch):
        """
        设置可视化输出目录

        创建用于保存验证/测试阶段可视化结果的目录结构。

        Args:
            logger: 日志记录器
            phase: 当前阶段（VAL_PHASE或TEST_PHASE）
            epoch: 当前轮次

        Returns:
            tuple: (result_output_dir, vis_output_dir)
                - result_output_dir: 结果输出目录（用于保存评估结果）
                - vis_output_dir: 可视化输出目录（用于保存可视化图像）
        """
        if phase == TEST_PHASE:
            prefix_dir = "test"
        elif phase == TRAIN_PHASE:
            prefix_dir = "train"
        elif phase == VAL_PHASE:
            prefix_dir = "validate"
        else:
            prefix_dir = "inference"

        if isinstance(epoch, int):
            epoch = "model_{}".format(str(epoch))

        output_dir_base = osp.join(self.output_dir, epoch, prefix_dir,
                                   "use_gt_box" if self.cfg.VAL.USE_GT_BBOX else "use_precomputed_box")
        vis_output_dir = osp.join(output_dir_base, "vis")
        result_output_dir = osp.join(output_dir_base, "prediction_result")
        create_folder(vis_output_dir)
        create_folder(result_output_dir)
        logger.info("=> Vis Output Dir : {}".format(vis_output_dir))
        logger.info("=> Result Output Dir : {}".format(result_output_dir))

        if phase == VAL_PHASE:
            tensorboard_log_dir = osp.join(self.output_dir, epoch, prefix_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        if self.cfg.DEBUG.VIS_SKELETON:
            logger.info("=> VIS_SKELETON")
        if self.cfg.DEBUG.VIS_BBOX:
            logger.info("=> VIS_BBOX")
        if getattr(self.cfg.DEBUG, "VIS_HEATMAP", False):
            logger.info("=> VIS_HEATMAP")

        return result_output_dir, vis_output_dir

    def _print_name_value(self, name_value, full_arch_name):
        """
        打印评估结果表格

        将评估指标以表格形式打印到日志中，方便查看和比较不同模型的性能。

        Args:
            name_value: 评估指标字典或列表
            full_arch_name: 模型名称
        """
        if isinstance(name_value, list):
            name_value = {item['name']: item['value'] for item in name_value}

        logger = logging.getLogger(__name__)
        logger.info('==> Performance of {}'.format(full_arch_name))
        logger.info(tabulate([[k, '{:.3f}'.format(v)] for k, v in name_value.items()],
                             headers=['Metric', 'Value'], tablefmt='simple'))
