#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
import torch.nn
import logging
import os
from datetime import timedelta
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .base import BaseExecutor
from .checkpoints import get_latest_checkpoint, save_checkpoint, resume
from datasets import build_train_loader, build_eval_loader
from posetimation.zoo import build_model
from posetimation.optimizer import build_lr_scheduler, build_optimizer
from posetimation.loss import build_loss
from engine.core import build_core_function
from engine.defaults import TRAIN_PHASE, VAL_PHASE
from engine.apis import set_random_seed


class DefaultTrainer(BaseExecutor):
    def exec(self):
        # 对外只暴露 exec()，实际训练逻辑都在 train() 里。
        self.train()

    def __init__(self, cfg, output_folders: dict, *args, **kwargs):
        super().__init__(cfg, output_folders, TRAIN_PHASE, **kwargs)
        logger = logging.getLogger(__name__)

        cfg = self.cfg
        # 这些参数由 runner 传进来，描述当前进程在 DDP 里的位置。
        self.distributed = kwargs.get("distributed", False)
        self.rank = kwargs.get("rank", 0)
        self.local_rank = kwargs.get("local_rank", 0)
        self.world_size = kwargs.get("world_size", 1)
        self.final_val_only = bool(
            kwargs.get("final_val_only", getattr(cfg.TRAIN, "FINAL_VAL_ONLY", True))
        )
        self.enable_periodic_val = bool(
            kwargs.get("periodic_val", getattr(cfg.TRAIN, "PERIODIC_VAL", True))
        ) and (not self.final_val_only)
        self.val_interval = max(
            1,
            int(kwargs.get("val_interval", getattr(cfg.TRAIN, "VAL_INTERVAL", 2))),
        )
        # 允许通过配置/命令行调大 DDP 通信超时，避免主卡长时间验证时其他 rank barrier 超时。
        self.ddp_timeout_minutes = max(
            1,
            int(kwargs.get("ddp_timeout_minutes", getattr(cfg.TRAIN, "DDP_TIMEOUT_MINUTES", 120))),
        )
        # 验证批次数上限：>0 时只跑前 N 个 batch，用于缩短周期性验证耗时。
        self.max_validate_batches = int(
            kwargs.get("max_validate_batches", getattr(cfg.VAL, "MAX_VALIDATE_BATCHES", -1))
        )

        # 如果是 DDP，就先初始化进程组，进程之间后面才能做通信和梯度同步。
        if self.distributed and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            # 显式设置进程组超时时间，覆盖 PyTorch 默认 30 分钟。
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(minutes=self.ddp_timeout_minutes),
            )
        # 每个进程固定使用自己的 GPU，避免多个进程抢同一张卡。
        if self.distributed and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

        # 只有主进程打印这类关键信息，避免日志重复。
        if self.rank == 0:
            logger.info("Set the random seed to {}".format(cfg.SEED))
            extra_cfg = getattr(cfg.MODEL, "EXTRA", None)
            matching_strategy = getattr(extra_cfg, "TOKEN_MATCHING_STRATEGY", None) if extra_cfg is not None else None
            if matching_strategy is None:
                matching_strategy = "ot" if bool(getattr(extra_cfg, "USE_OT_DISTILLATION", False)) else "none"
            matching_weight = float(getattr(
                extra_cfg,
                "MATCHING_LOSS_WEIGHT",
                getattr(extra_cfg, "OT_LOSS_WEIGHT", 0.0),
            )) if extra_cfg is not None else 0.0
            logger.info(
                "Token matching strategy: {}. MATCHING_LOSS_WEIGHT: {:.6f}".format(
                    matching_strategy, matching_weight
                )
            )
            if self.distributed:
                # 记录关键 DDP 防超时参数，便于复现实验与排障。
                logger.info(
                    "DDP timeout set to {} minutes; periodic val max batches: {}".format(
                        self.ddp_timeout_minutes, self.max_validate_batches
                    )
                )
        # 设置随机种子，保证可复现。
        set_random_seed(cfg.SEED)

        # 构建训练集 DataLoader；DDP 时会自动给每个进程分片。
        self.dataloader = build_train_loader(
            cfg,
            distributed=self.distributed,
            rank=self.rank,
            world_size=self.world_size,
        )
        self.val_dataloader = None
        if self.rank == 0 and self.cfg.MODEL.EVALUATE:
            self.val_dataloader = build_eval_loader(cfg, phase=VAL_PHASE)
        # 构建模型本体。
        self.model = build_model(cfg, phase='train')
        # 构建优化器。
        self.optimizer = build_optimizer(cfg, self.model)
        # 构建学习率调度器。
        self.lr_scheduler = build_lr_scheduler(cfg, self.optimizer)
        # 构建损失函数。
        self.loss_criterion = build_loss(cfg)

        self.begin_epoch = 0
        self.end_epoch = cfg.TRAIN.END_EPOCH

        # 把模型放到 GPU 上。
        if torch.cuda.is_available():
            if self.distributed:
                # DDP 模式下先移动到本进程绑定的 GPU。
                self.model = self.model.cuda(self.local_rank)
            else:
                # 非 DDP 时直接放到默认 GPU。
                self.model = self.model.cuda()
        self.GPUS = cfg.GPUS
        # core_function 里封装了训练/验证步骤。
        self.core_function = build_core_function(cfg, criterion=self.loss_criterion, **kwargs)

        # tensorboard 相关状态。
        self.tb_writer_dict = {"writer": None,
                               "global_steps": 0}

    def train(self):
        logger = logging.getLogger(__name__)
        # 如果存在可恢复的 checkpoint，就先恢复。
        self.model_resume()
        if self.distributed:
            # DDP 会把模型包装起来，负责跨进程梯度同步。
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                # 各次前向传播可启用不同分支，因此允许存在未使用的参数。
                find_unused_parameters=True,
            )
        elif len(self.GPUS) > 1:
            # 非 DDP 的多卡模式仍然保留旧的 DataParallel。
            self.model = torch.nn.DataParallel(self.model)
        for epoch in range(self.begin_epoch, self.end_epoch):
            # DDP 下每个 epoch 需要重新设置 sampler 的随机种子，保证每张卡切片不同。
            sampler = getattr(self.dataloader, "sampler", None)
            if self.distributed and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            
            # 清空上一个 epoch 留下的显存缓存，避免碎片化导致 OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # train
            if self.rank == 0:
                # 只在主进程打印 epoch 开始信息。
                logger.info('=> Start train epoch {}'.format(epoch))
                # 打印当前 epoch 将使用的学习率。
                lrs = [pg['lr'] for pg in self.optimizer.param_groups]
                lr_str = ', '.join([f'LR{i}: {lr:.2e}' for i, lr in enumerate(lrs)])
                logger.info(f'Epoch {epoch} current {lr_str}')


            self.core_function.train(model=self.model, epoch=epoch, optimizer=self.optimizer, dataloader=self.dataloader,
                                     tb_writer_dict=self.tb_writer_dict)
            
            # 清空训练过程中的中间激活值
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 学习率调度器每个 epoch 走一步。
            self.lr_scheduler.step()  # Don't pass epoch param to avoid deprecation warning

            # save model
            if self.rank == 0:
                # 只在主进程保存 checkpoint；每个 epoch 都保存一份。
                model_save_path = self.save_model(epoch)
                logger.info('=> Saved epoch {} model state to {}'.format(epoch, model_save_path))

            run_periodic_val = self.enable_periodic_val and (epoch >= 0) and ((epoch + 1) % self.val_interval == 0)
            if run_periodic_val:
                if self.distributed and dist.is_initialized():
                    dist.barrier()

                if self.rank == 0 and self.val_dataloader is not None:
                    logger.info('=> Start periodic validation at epoch {}'.format(epoch))
                    eval_model = self.model.module if isinstance(self.model, DDP) else self.model
                    # 把验证截断参数传给 core eval，避免验证过长导致其他 rank 等待超时。
                    self.core_function.eval(
                        model=eval_model,
                        dataloader=self.val_dataloader,
                        tb_writer_dict=self.tb_writer_dict,
                        epoch=epoch,
                        phase=VAL_PHASE,
                        cfg=self.cfg,
                        max_validate_batches=self.max_validate_batches,
                    )
                    
                    # 清空验证过程中的中间激活值
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if self.distributed and dist.is_initialized():
                    dist.barrier()

            # record learning_rate

        # 训练结束后仅做一次最终完整验证（不做中间周期验证）。
        if self.cfg.MODEL.EVALUATE:
            if self.distributed and dist.is_initialized():
                dist.barrier()

            if self.rank == 0 and self.val_dataloader is not None:
                logger.info('=> Start final validation after training')
                eval_model = self.model.module if isinstance(self.model, DDP) else self.model
                self.core_function.eval(
                    model=eval_model,
                    dataloader=self.val_dataloader,
                    tb_writer_dict=self.tb_writer_dict,
                    epoch=self.end_epoch - 1,
                    phase=VAL_PHASE,
                    cfg=self.cfg,
                    max_validate_batches=-1,
                )

            if self.distributed and dist.is_initialized():
                dist.barrier()

    def save_model(self, epoch):
        # 保存当前 epoch 的训练状态，包括模型和优化器。
        model_save_path = save_checkpoint(epoch, self.checkpoints_save_folder, self.model, self.optimizer,
                          self.lr_scheduler,
                                          global_steps=self.tb_writer_dict["global_steps"])
        return model_save_path

    def model_resume(self):
        logger = logging.getLogger(__name__)
        # 查找最近一次保存的 checkpoint。
        checkpoint_file = get_latest_checkpoint(self.checkpoints_save_folder)
        if checkpoint_file is not None:
            if self.rank == 0:
                # 主进程打印恢复信息。
                logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            # 把 checkpoint 重新载入模型和优化器。
            self.model, self.optimizer, self.begin_epoch, ext_dict = resume(
                self.model,
                self.optimizer,
                checkpoint_file,
                self.lr_scheduler,
                gpus=self.GPUS,
            )
            # 恢复 tensorboard 的全局步数。
            self.tb_writer_dict["global_steps"] = ext_dict["tensorboard_global_steps"]

        else:
            if self.rank == 0:
                # 没有 checkpoint 就从头开始。
                logger.warning("=> no checkpoint file available to resume")

    def __del__(self):
        super(DefaultTrainer, self).__del__()
