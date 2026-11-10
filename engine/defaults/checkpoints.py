#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
import os
import os.path as osp
import logging
import tempfile
import pickle
from utils.utils_folder import list_immediate_childfile_paths


def get_latest_checkpoint(checkpoint_save_folder):
    # 找出 checkpoints 目录下最新的 pth 文件。
    checkpoint_saves_paths = list_immediate_childfile_paths(checkpoint_save_folder, ext="pth")
    if len(checkpoint_saves_paths) == 0:
        return None

    latest_checkpoint = checkpoint_saves_paths[0]
    # we define the format of checkpoint like "epoch_0_state.pth"
    latest_index = int(osp.basename(latest_checkpoint).split("_")[1])
    for checkpoint_save_path in checkpoint_saves_paths:
        checkpoint_save_file_name = osp.basename(checkpoint_save_path)
        now_index = int(checkpoint_save_file_name.split("_")[1])
        if now_index > latest_index:
            latest_checkpoint = checkpoint_save_path
            latest_index = now_index
    return latest_checkpoint


def get_all_checkpoints(checkpoint_save_folder):
    # 返回目录里所有 checkpoint 文件，给验证时批量评估用。
    checkpoint_saves_paths = list_immediate_childfile_paths(checkpoint_save_folder, ext="pth")
    if len(checkpoint_saves_paths) == 0:
        return None
    checkpoints_list = []
    # we define the format of checkpoint like "epoch_0_state.pth"
    for checkpoint_save_path in checkpoint_saves_paths:
        checkpoints_list.append(checkpoint_save_path)
    return checkpoints_list


def save_checkpoint(epoch, save_folder, model, optimizer, lr_scheduler=None, **kwargs):
    """Save model, optimizer, and scheduler state through a temporary file."""
    logger = logging.getLogger(__name__)
    # 统一把 checkpoint 保存成 epoch_x_state.pth。
    model_save_path = osp.join(save_folder, 'epoch_{}_state.pth'.format(epoch))
    checkpoint_dict = dict()
    # begin_epoch 表示下一次恢复时应该从哪一轮继续。
    checkpoint_dict['begin_epoch'] = epoch

    # Because nn.DataParallel
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/5
    model_state_dict = model.state_dict()
    if list(model_state_dict.keys())[0].startswith('module.'):
        # 如果模型外面包了一层 DataParallel/DDP，就去掉 module. 前缀。
        model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}

    # 保存模型权重。
    checkpoint_dict['state_dict'] = model_state_dict
    # 保存优化器状态，这样恢复训练时动量、学习率状态都能接上。
    checkpoint_dict['optimizer'] = optimizer.state_dict()
    # 保存学习率调度器状态，避免 resume 后调度进度错位。
    if lr_scheduler is not None:
        checkpoint_dict['lr_scheduler'] = lr_scheduler.state_dict()
    # 保存 tensorboard 全局步数。
    checkpoint_dict['tensorboard_global_steps'] = kwargs.get("global_steps", 0)

    # Atomic write: save to temp file, then rename
    # 先写到临时文件，避免保存一半中断导致文件损坏。
    temp_fd, temp_path = tempfile.mkstemp(suffix='.pth', dir=save_folder, text=False)
    try:
        os.close(temp_fd)  # Close temp file descriptor
        # 真正执行 torch.save。
        torch.save(checkpoint_dict, temp_path)
        # Atomic rename (works on Unix and Windows)
        if os.path.exists(model_save_path):
            os.remove(model_save_path)
        # 把临时文件改名成正式文件。
        os.rename(temp_path, model_save_path)
        logger.info(f"=> Checkpoint saved to {model_save_path}")
    except Exception as e:
        logger.error(f"=> Failed to save checkpoint: {e}")
        # Clean up temp file if save failed
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise


    return model_save_path


def resume(model, optimizer, checkpoint_file, lr_scheduler=None, **kwargs):
    """Resume training from checkpoint with graceful degradation on load failure."""
    logger = logging.getLogger(__name__)

    ext_dict = {}
    begin_epoch = 0

    try:
        # 读取 checkpoint。
        checkpoint = torch.load(checkpoint_file)
        # begin_epoch 加 1，表示从下一轮开始训练。
        begin_epoch = checkpoint['begin_epoch'] + 1
        state_dict = {}
        # 统一去掉 module. 前缀，兼容 DP/DDP 保存的权重。
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('module.'):
                new_key = k.replace('module.', '', 1)
            else:
                new_key = k
            state_dict[new_key] = v

        model.load_state_dict(state_dict, strict=True)

        # 恢复优化器状态。
        optimizer.load_state_dict(checkpoint['optimizer'])

        # 恢复学习率调度器状态；旧 checkpoint 没有该字段时做兼容对齐。
        if lr_scheduler is not None:
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            else:
                # 对旧 checkpoint：按照 begin_epoch 对齐到“上一个已完成 epoch”的学习率。
                aligned_epoch = max(0, begin_epoch - 1)
                if hasattr(lr_scheduler, 'milestones') and hasattr(lr_scheduler, 'gamma'):
                    decay_steps = sum(1 for m in lr_scheduler.milestones if m <= aligned_epoch)
                    decay_factor = lr_scheduler.gamma ** decay_steps
                    for i, base_lr in enumerate(lr_scheduler.base_lrs):
                        optimizer.param_groups[i]['lr'] = base_lr * decay_factor
                    lr_scheduler.last_epoch = aligned_epoch
                    lr_scheduler._last_lr = [pg['lr'] for pg in optimizer.param_groups]

        # 优化器里的张量状态也要搬回 GPU。
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        # 恢复 tensorboard 的全局步数。
        ext_dict["tensorboard_global_steps"] = checkpoint.get("tensorboard_global_steps", 0)
        logger.info(f"=> Successfully loaded checkpoint from {checkpoint_file}")

    except (EOFError, RuntimeError, pickle.UnpicklingError) as e:
        logger.warning(f"=> Failed to load checkpoint {checkpoint_file}: {type(e).__name__}: {e}")
        logger.warning("=> Will skip resume and start from scratch with begin_epoch=0")
        # Try to remove corrupted checkpoint
        try:
            if os.path.getsize(checkpoint_file) == 0:
                logger.info(f"=> Removing empty checkpoint file: {checkpoint_file}")
                os.remove(checkpoint_file)
            elif os.path.getsize(checkpoint_file) < 1024:  # Less than 1KB is suspicious
                logger.warning(f"=> Checkpoint file is very small ({os.path.getsize(checkpoint_file)} bytes), might be corrupted")
        except:
            pass
        # 如果恢复失败，就从头开始训练。
        begin_epoch = 0  # Start from epoch 0
        ext_dict["tensorboard_global_steps"] = 0

    return model, optimizer, begin_epoch, ext_dict
