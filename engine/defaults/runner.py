#!/usr/bin/python
# -*- coding:utf8 -*-
import logging
import os
import os.path as osp
from .trainer import DefaultTrainer
from .evaluator import DefaultEvaluator
from utils.utils_folder import create_folder
from engine.defaults import TRAIN_PHASE, VAL_PHASE, TEST_PHASE
from posetimation.zoo import get_model_hyperparameter
from datasets import get_dataset_name
from collections import defaultdict


class DefaultRunner:
    def __init__(self, cfg, args, **kwargs):
        self.cfg = cfg
        self.args = args
        self.train = args.train
        self.val = args.val
        self.val_from_checkpoint_id = args.val_from_checkpoint
        self.test = args.test
        self.output_path_dict = defaultdict(str)
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1
        # 优先使用命令行设置的CUDA_VISIBLE_DEVICES，否则使用配置文件中的GPUS
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((map(str, cfg.GPUS)))
        # 从实际的环境变量中读取GPU设置，确保一致性
        gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        self.gpus = tuple(int(gpu_id.strip()) for gpu_id in gpu_ids if gpu_id.strip())
        self.setup_cfg()

    def setup_cfg(self):
        cfg = self.cfg
        hyper_parameters_setting = get_model_hyperparameter(self.args,self.cfg)
        dataset_name = get_dataset_name(self.cfg)
        cfg.defrost()

        cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, cfg.EXPERIMENT_NAME, dataset_name, hyper_parameters_setting)
        cfg.freeze()

        checkpoints_save_folder = osp.join(cfg.OUTPUT_DIR, "checkpoints")
        log_save_folder = osp.join(cfg.OUTPUT_DIR, "log")
        writer_save_folder = osp.join(cfg.OUTPUT_DIR, 'tensorboard')
        create_folder(checkpoints_save_folder)
        create_folder(writer_save_folder)
        create_folder(log_save_folder)

        self.output_path_dict["checkpoints_save_folder"] = checkpoints_save_folder
        self.output_path_dict["tb_save_folder"] = writer_save_folder  # tensorboard save
        self.output_path_dict["log_save_folder"] = log_save_folder

        self.cfg = cfg


    def launch(self):
        if self.train:
            trainer = DefaultTrainer(
                self.cfg,
                self.output_path_dict,
                PE_Name=self.args.PE_Name,
                rank=self.rank,
                local_rank=self.local_rank,
                world_size=self.world_size,
                distributed=self.distributed,
            )
            trainer.exec()

        if self.val and (not self.distributed or self.rank == 0):
            evaluator = DefaultEvaluator(self.cfg, self.output_path_dict, PE_Name=self.args.PE_Name,
                                         phase=VAL_PHASE, eval_from_checkpoint_id=self.val_from_checkpoint_id,
                                         rank=self.rank, local_rank=self.local_rank,
                                         world_size=self.world_size, distributed=self.distributed)
            evaluator.exec()

        if self.test and (not self.distributed or self.rank == 0):
            evaluator = DefaultEvaluator(self.cfg, self.output_path_dict, PE_Name=self.args.PE_Name,
                                         phase=TEST_PHASE,
                                         rank=self.rank, local_rank=self.local_rank,
                                         world_size=self.world_size, distributed=self.distributed)
            evaluator.exec()
