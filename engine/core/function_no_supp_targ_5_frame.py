#!/usr/bin/python
# -*- coding:utf8 -*-
import time
import torch
import numpy as np
import os.path as osp
import logging
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter
from .evaludate import accuracy, pck_accuracy, pck_accuracy_origin_image
from engine.core import CORE_FUNCTION_REGISTRY, BaseFunction, AverageMeter

# from engine.evaludate import accuracy
from engine.defaults import VAL_PHASE, TEST_PHASE, TRAIN_PHASE
from datasets.process import get_final_preds
from datasets.transforms import reverse_transforms

from utils.utils_bbox import cs2box
from utils.utils_folder import create_folder
from utils.utils_image_tensor import tensor2im

from tabulate import tabulate


def cnt_grad(model):
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


def _test(model, logger=None, stage="Eval"):
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return

    if logger is None:
        logger = logging.getLogger(__name__)

    model_ref = _unwrap_model(model)
    was_training = model_ref.training
    _log_parameter_summary(model_ref, logger, stage)
    frame_num = int(getattr(model_ref, "frame_num", 5))
    b, c, h, w = 1, frame_num * 3, 256, 192
    device = next(model_ref.parameters()).device
    random_x = torch.randn((b, c, h, w), device=device)
    channel_num = c // frame_num

    model_ref.eval()

    # 1. Full model MACs (captures dynamic pruning automatically)
    full_macs = int(profile_macs(model_ref, random_x))
    logger.info(f"=> [{stage}] === MACs Analysis ===")
    logger.info(f"=> [{stage}] 1. Full Model MACs #: {full_macs:,}")

    # 2. Backbone MACs
    img_cat_batch = torch.cat(random_x.split(channel_num, dim=1), dim=0)
    backbone_macs = int(profile_macs(model_ref.backbone, img_cat_batch))
    logger.info(f"=> [{stage}] 2. Backbone MACs #: {backbone_macs:,} ({backbone_macs/full_macs*100:.1f}%)")

    # Get intermediate feature shapes
    with torch.no_grad():
        backbone_out = model_ref.backbone(img_cat_batch)
        if isinstance(backbone_out, tuple):
            _, feats_backbone_out_cat_batch = backbone_out[:2]
        
        feats_backbone_out_curr = feats_backbone_out_cat_batch.split(b, dim=0)[0]
        _, _, c_fg = feats_backbone_out_curr.shape
        
        Hp, Wp = model_ref.map_size_up_1x[0], model_ref.map_size_up_1x[1]

    # 3. Output head MACs
    head_macs = 0
    if hasattr(model_ref, 'up_sample'):
        up_input = torch.randn((b, c_fg, Hp, Wp), device=device)
        head_macs += int(profile_macs(model_ref.up_sample, up_input))
        
        with torch.no_grad():
            up_out = model_ref.up_sample(up_input)
        
        if up_out.shape[1] != 128:
            up_out = torch.randn((b, 128, up_out.shape[2], up_out.shape[3]), device=device)
        
        if hasattr(model_ref, 'out_backbone_fusion') and getattr(model_ref, 'final_fusion_backbone', False):
            feat_up2 = torch.randn((b, 256, up_out.shape[2], up_out.shape[3]), device=device)
            concat_out = torch.cat((up_out, feat_up2), dim=1)
            head_macs += int(profile_macs(model_ref.out_backbone_fusion, concat_out))
            final_in = torch.randn((b, 128, up_out.shape[2], up_out.shape[3]), device=device)
        else:
            final_in = up_out
        
        if hasattr(model_ref, 'final_layer'):
            head_macs += int(profile_macs(model_ref.final_layer, final_in))
    
    logger.info(f"=> [{stage}] 4. Output Head MACs #: {head_macs:,} ({head_macs/full_macs*100:.1f}%)")

    # 4. Coarse-grain branch MACs = Full - Backbone - Head (automatically captures pruning!)
    coarse_macs = full_macs - backbone_macs - head_macs
    
    # Measure STE blocks - measure first layer then multiply by number of layers
    ste_macs = 0
    if hasattr(model_ref, 'prune_recover_blocks') and hasattr(model_ref.prune_recover_blocks, 'STEblocks'):
        ste_blocks = model_ref.prune_recover_blocks.STEblocks
        # STE input shape: (b*f, n, c) = (5, 192, 1024)
        # All STE layers receive the same input shape, so measure one and multiply
        ste_input = feats_backbone_out_cat_batch
        ste_macs_per_layer = int(profile_macs(ste_blocks[0], ste_input))
        ste_macs = int(ste_macs_per_layer * len(ste_blocks))
    
    # Measure TTE blocks - measure first layer then multiply by number of layers
    tte_macs = 0
    if hasattr(model_ref, 'prune_recover_blocks') and hasattr(model_ref.prune_recover_blocks, 'TTEblocks'):
        tte_blocks = model_ref.prune_recover_blocks.TTEblocks
        # First run STE to get proper TTE input shape: (b*n, f, c) = (192, 5, 1024)
        x = feats_backbone_out_cat_batch
        b_ste, n_ste, c_ste = x.shape
        f = frame_num
        b = b_ste // f
        n = n_ste
        
        if hasattr(model_ref.prune_recover_blocks, 'STEblocks'):
            ste_blocks = model_ref.prune_recover_blocks.STEblocks
            with torch.no_grad():
                # Run STE first layer
                x = ste_blocks[0](x)
            # Rearrange to TTE input shape: (b*f, n, c) -> (b*n, f, c)
            x = x.reshape(b, f, n, c_ste).transpose(1, 2).reshape(b * n, f, c_ste)
        
        # Now x has TTE input shape (b*n, f, c) = (192, 5, 1024)
        # All TTE layers receive the same input shape, so measure one and multiply
        tte_macs_per_layer = int(profile_macs(tte_blocks[0], x))
        tte_macs = int(tte_macs_per_layer * len(tte_blocks))
    
    # Middle blocks + recovery = Coarse - STE - TTE (includes pruning effects)
    middle_macs = coarse_macs - ste_macs - tte_macs
    
    logger.info(f"=> [{stage}] 3. Coarse-grain Branch MACs #: {coarse_macs:,} ({coarse_macs/full_macs*100:.1f}%)")
    if ste_macs > 0:
        logger.info(f"=> [{stage}]    - STE Blocks:          {ste_macs:,} ({ste_macs/full_macs*100:.1f}%)")
    if tte_macs > 0:
        logger.info(f"=> [{stage}]    - TTE Blocks:          {tte_macs:,} ({tte_macs/full_macs*100:.1f}%)")
    if middle_macs > 0:
        logger.info(f"=> [{stage}]    - Middle + Recovery:   {middle_macs:,} ({middle_macs/full_macs*100:.1f}%)")
        logger.info(f"=> [{stage}]    - Middle+Recovery = Coarse - STE - TTE: {middle_macs:,}")

    # FPS calculation
    for _ in range(10):
        with torch.no_grad():
            _ = model_ref(random_x)
    
    num_iterations = 100
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model_ref(random_x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end_time = time.time()
    
    fps = num_iterations / (end_time - start_time)
    logger.info(f"=> [{stage}] FPS: {fps:.4f}")

    model_ref.train(was_training)


def _get_token_keep_stats(model):
    model_ref = model.module if isinstance(model, torch.nn.DataParallel) else model
    return getattr(model_ref, "token_keep_stats", None)


def _get_token_keep_history(model):
    model_ref = model.module if isinstance(model, torch.nn.DataParallel) else model
    return getattr(model_ref, "token_keep_history", None)


def _reset_token_keep_history(model):
    model_ref = model.module if isinstance(model, torch.nn.DataParallel) else model
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
        ' | TokenKeep '
        'Fine {}/{} ({:.1%}) '
        'Coarse {}/{} ({:.1%})'
    ).format(
        int(stats.get("fine_keep_tokens", 0)),
        int(stats.get("fine_total_tokens", 0)),
        float(stats.get("fine_keep_ratio", 0.0)),
        int(stats.get("coarse_keep_tokens", 0)),
        int(stats.get("coarse_total_tokens", 0)),
        float(stats.get("coarse_keep_ratio", 0.0)),
    )

@CORE_FUNCTION_REGISTRY.register()
class Function_No_Supp_Targ_5_Frame(BaseFunction):

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.output_dir = cfg.OUTPUT_DIR

        if "criterion" in kwargs.keys():
            self.criterion = kwargs["criterion"]
        if "tb_log_dir" in kwargs.keys():
            self.tb_log_dir = kwargs["tb_log_dir"]
        if "writer_dict" in kwargs.keys():
            self.writer_dict = kwargs["writer_dict"]

        self.PE_Name = kwargs.get("PE_Name", "SLDPOSE")
        ##
        self.max_iter_num = 0
        self.dataloader_iter = None
        self.tb_writer = None
        self.global_steps = 0
        self.DataSetName = str(self.cfg.DATASET.NAME).upper()

    def train(self, model, epoch, optimizer, dataloader, tb_writer_dict, **kwargs):
        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]
        logger = logging.getLogger(__name__)
        _reset_token_keep_history(model)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        grad = AverageMeter()
        # switch to train mode
        model.train()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        end = time.time()

        _test(model, logger=logger, stage="Train")

        for iter_step in range(self.max_iter_num):
            input_x, ref_data_numpy_list, target_heatmaps, target_heatmaps_weight, meta = next(self.dataloader_iter)

            self._before_train_iter(input_x)

            data_time.update(time.time() - end)

            target_heatmaps = target_heatmaps.cuda(non_blocking=True)
            target_heatmaps_weight = target_heatmaps_weight.cuda(non_blocking=True)


            concat_input = torch.cat((input_x, ref_data_numpy_list), 1).cuda()

            outputs, htms_backbone = model(concat_input)

            if isinstance(outputs, list) or isinstance(outputs, tuple):
                pred_heatmaps = outputs[0]
                loss = self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
                for pred_heatmaps in outputs[1:]:
                    loss += self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
            else:
                pred_heatmaps = outputs
                loss = self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)

            # compute gradient and do update step
            optimizer.zero_grad()
            loss.backward()
            grad.update(cnt_grad(model))
            optimizer.step()

            # measure accuracy and record loss
            losses.update(loss.item(), input_x.size(0))

            _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= self.max_iter_num - 1:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'Grad {grad.val:.3f} ({grad.avg:.3f})\t' \
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(epoch, iter_step, self.max_iter_num,
                                                                        batch_time=batch_time,
                                                                        speed=input_x.size(0) / batch_time.val,
                                                                        data_time=data_time, grad=grad, loss=losses,
                                                                        acc=acc)

                logger.info(msg)

            # For Tensorboard
            if self.tb_writer is not None:
                self.tb_writer.add_scalar('train_loss', losses.val, self.global_steps)
                self.tb_writer.add_scalar('train_acc', acc.val, self.global_steps)
            self.global_steps += 1

        tb_writer_dict["global_steps"] = self.global_steps

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

        logger = logging.getLogger(__name__)
        _reset_token_keep_history(model)
        _test(model, logger=logger, stage="Eval")

        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]

        batch_time, data_time, losses, acc, acc_backbone = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        phase = kwargs.get("phase", VAL_PHASE)
        epoch = kwargs.get("epoch", "specified_model")
        # switch to evaluate mode
        model.eval()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        dataset = dataloader.dataset
        # prepare data fro validate
        num_samples = len(dataset)
        all_preds = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=float)
        all_preds_backbone = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=float)
        all_boxes = np.zeros((num_samples, 6))
        all_boxes_backbone = np.zeros((num_samples, 6))
        image_path = []
        filenames = []
        filenames_map = {}
        filenames_counter = 0
        imgnums = []
        idx = 0
        idx_backbone = 0
        acc_threshold = 0.7

        ###
        result_output_dir, vis_output_dir = self.vis_setup(logger, phase, epoch)
        ###

        with torch.no_grad():
            end = time.time()
            num_batch = len(dataloader)
            for iter_step in range(self.max_iter_num):
                input_x, ref_data_numpy_list, target_heatmaps, target_heatmaps_weight, meta  = next(self.dataloader_iter)

                if phase == VAL_PHASE:
                    self._before_val_iter(input_x)

                data_time.update(time.time() - end)

                target_heatmaps = target_heatmaps.cuda(non_blocking=True)

                # outputs = model(concat_input, margin)
                concat_input = torch.cat((input_x, ref_data_numpy_list), 1).cuda()


                outputs, htms_backbone = model(concat_input.cuda())

                if phase == VAL_PHASE:
                    if isinstance(model, torch.nn.DataParallel):
                        vis_dict = getattr(model.module, "vis_dict", None)
                    else:
                        vis_dict = getattr(model, "vis_dict", None)

                    if vis_dict:
                        self._running_val_iter(vis_dict=vis_dict, model_input=[input_x, ref_data_numpy_list])

                if isinstance(outputs, list) or isinstance(outputs, tuple):
                    pred_heatmaps = outputs[-1]
                else:
                    pred_heatmaps = outputs

                _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(),
                                              target_heatmaps.detach().cpu().numpy())

                acc.update(avg_acc, cnt)

                _, avg_acc_backbone, cnt_backbone, _ = accuracy(htms_backbone.detach().cpu().numpy(),
                                                                target_heatmaps.detach().cpu().numpy())

                acc_backbone.update(avg_acc_backbone, cnt_backbone)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= (num_batch - 1):
                    msg = '{}: [{}/{}]\t' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t   \
                          {acc_backbone.val:.3f} ({acc_backbone.avg:.3f})'.format(phase, iter_step, num_batch,
                                                                                  batch_time=batch_time,
                                                                                  data_time=data_time, acc=acc,
                                                                                  acc_backbone=acc_backbone)
                    logger.info(msg)

                #### for eval ####
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

                preds, maxvals = get_final_preds(pred_heatmaps.clone().cpu().numpy(), center, scale)
                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(scale * 200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                image_path.extend(meta['image'])
                idx += num_images

                preds_backbone, maxvals_backbone = get_final_preds(htms_backbone.clone().cpu().numpy(), center, scale)
                all_preds_backbone[idx_backbone:idx_backbone + num_images, :, 0:2] = preds_backbone[:, :, 0:2]
                all_preds_backbone[idx_backbone:idx_backbone + num_images, :, 2:3] = maxvals_backbone
                all_boxes_backbone[idx_backbone:idx_backbone + num_images, 0:2] = center[:, 0:2]
                all_boxes_backbone[idx_backbone:idx_backbone + num_images, 2:4] = scale[:, 0:2]
                all_boxes_backbone[idx_backbone:idx_backbone + num_images, 4] = np.prod(scale * 200, 1)
                all_boxes_backbone[idx_backbone:idx_backbone + num_images, 5] = score
                idx_backbone += num_images

                # tensorboard writ
                self.global_steps += 1
                #
                self._after_val_iter(meta["image"], preds, maxvals, vis_output_dir, center, scale)

        logger.info('########################################')
        logger.info('{}'.format(self.cfg.EXPERIMENT_NAME))

        name_values, perf_indicator = dataset.evaluate(self.cfg, all_preds, result_output_dir, all_boxes,
                                                       filenames_map, filenames, imgnums)

        name_values_backbone, perf_indicator_backbone = dataset.evaluate(self.cfg, all_preds_backbone,
                                                                         result_output_dir, all_boxes_backbone,
                                                                         filenames_map, filenames, imgnums)

        self._print_name_value(name_values_backbone, 'ViT_Backbone')

        model_name = self.cfg.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                self._print_name_value(name_value, model_name)
        else:
            self._print_name_value(name_values, model_name)

        token_keep_summary = _build_token_keep_summary(_get_token_keep_history(model))
        if token_keep_summary is not None:
            logger.info(
                "=> TokenKeep Eval Summary ({} forwards): "
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

        tb_writer_dict["global_steps"] = self.global_steps

    def _before_train_iter(self, batch_x):
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        show_image_num = min(6, len(batch_x))
        batch_x = batch_x[:show_image_num]
        label_name = "train_{}_x".format(self.global_steps)
        save_image = []
        for x in batch_x:
            x = tensor2im(x)
            save_image.append(x)
        save_image = np.stack(save_image, axis=0)
        self.tb_writer.add_images(label_name, save_image, global_step=self.global_steps, dataformats="NHWC")

    def _before_val_iter(self, batch_x):
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
        diff_A = vis_dict["diff_A"][0:show_image_num]  # [N,17,96,72]
        diff_B = vis_dict["diff_B"][0:show_image_num]  # [N,17,96,72]
        diff_heatmaps = vis_dict["diff_heatmaps"][0:show_image_num]  # [N,34,96,72]
        support_heatmaps = vis_dict["support_heatmaps"][0:show_image_num]  # [N,17,96,72]
        prf_ptm_combine_featuremaps = vis_dict["prf_ptm_combine_featuremaps"][0:show_image_num]  # [N,96,96,72]
        warped_heatmaps_list = [warped_heatmaps[0:show_image_num] for warped_heatmaps in
                                vis_dict["warped_heatmaps_list"]]  # [N,17,96,72]
        output_heatmaps = vis_dict["output_heatmaps"][0:show_image_num]  # [N,17,96,72]

        show_three_input_image = make_grid(reverse_transforms(torch.cat([previous_x, current_x, next_x], dim=0)),
                                           nrow=show_image_num)
        self.tb_writer.add_image('01_three_input_image', show_three_input_image, global_step=self.global_steps)

        # show 2.
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

        # show 3.
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

        # show4.
        diff_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(diff_heatmaps, 1, dim=1), dim=0), 0)
        diff_heatmaps_channels_image = make_grid(diff_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('04_diff_heatmaps_channels_image', diff_heatmaps_channels_image,
                                 global_step=self.global_steps)

        # show5.
        support_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(support_heatmaps, 1, dim=1), dim=0), 0)
        support_heatmaps_channels_image = make_grid(support_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('05_support_heatmaps_channels_image', support_heatmaps_channels_image,
                                 global_step=self.global_steps)

        # show6.
        prf_ptm_combine_featuremaps_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(prf_ptm_combine_featuremaps, 1, dim=1), dim=0),
            0)
        prf_ptm_combine_featuremaps_channels_image = make_grid(prf_ptm_combine_featuremaps_channels_tensor,
                                                               nrow=show_image_num)
        self.tb_writer.add_image('06_prf_ptm_combine_featuremaps_channels_image',
                                 prf_ptm_combine_featuremaps_channels_image, global_step=self.global_steps)

        # show7.
        warped_heatmaps_1_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(warped_heatmaps_list[0], 1, dim=1), dim=0), 0)
        warped_heatmaps_1_channels_image = make_grid(warped_heatmaps_1_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('07_warped_heatmaps_1_channels_image', warped_heatmaps_1_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_2_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(warped_heatmaps_list[1], 1, dim=1), dim=0), 0)
        warped_heatmaps_2_channels_image = make_grid(warped_heatmaps_2_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('08_warped_heatmaps_2_channels_image', warped_heatmaps_2_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_3_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(warped_heatmaps_list[2], 1, dim=1), dim=0), 0)
        warped_heatmaps_3_channels_image = make_grid(warped_heatmaps_3_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('09_warped_heatmaps_3_channels_image', warped_heatmaps_3_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_4_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(warped_heatmaps_list[3], 1, dim=1), dim=0), 0)
        warped_heatmaps_4_channels_image = make_grid(warped_heatmaps_4_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('10_warped_heatmaps_4_channels_image', warped_heatmaps_4_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_5_channels_tensor = torch.clamp_min(
            torch.cat(torch.split(warped_heatmaps_list[4], 1, dim=1), dim=0), 0)
        warped_heatmaps_5_channels_image = make_grid(warped_heatmaps_5_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('11_warped_heatmaps_5_channels_image', warped_heatmaps_5_channels_image,
                                 global_step=self.global_steps)

        # show8.
        output_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(output_heatmaps, 1, dim=1), dim=0), 0)
        output_heatmaps_channels_image = make_grid(output_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('12_output_heatmaps_channels_image', output_heatmaps_channels_image,
                                 global_step=self.global_steps)

    def _after_val_iter(self, image, preds_joints, preds_confidence, vis_output_dir, center, scale):
        cfg = self.cfg
        # prepare data
        coords = np.concatenate([preds_joints, preds_confidence], axis=-1)
        bboxes = []
        for index in range(len(center)):
            xyxy_bbox = cs2box(center[index], scale[index], pattern="xyxy")
            bboxes.append(xyxy_bbox)

        if cfg.DEBUG.VIS_SKELETON or cfg.DEBUG.VIS_BBOX:
            from .vis_helper import draw_skeleton_in_origin_image
            draw_skeleton_in_origin_image(image, coords, bboxes, vis_output_dir, vis_skeleton=cfg.DEBUG.VIS_SKELETON,
                                          vis_bbox=cfg.DEBUG.VIS_BBOX)

    def vis_setup(self, logger, phase, epoch):
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
        return result_output_dir, vis_output_dir
