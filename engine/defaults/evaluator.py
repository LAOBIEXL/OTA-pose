#!/usr/bin/python
# -*- coding:utf8 -*-
import logging
import torch
import torch.nn
import os.path as osp
from tensorboardX import SummaryWriter

from .base import BaseExecutor
from datasets import build_eval_loader
from posetimation.zoo import build_model
from .checkpoints import get_all_checkpoints, get_latest_checkpoint
from engine.core import build_core_function
from engine.defaults import VAL_PHASE, TEST_PHASE
from engine.apis import set_random_seed


class DefaultEvaluator(BaseExecutor):

    def exec(self):
        self.eval()

    def __init__(self, cfg, output_folders: dict, phase=TEST_PHASE, **kwargs):
        super().__init__(cfg, output_folders, phase, **kwargs)

        cfg = self.cfg
        self.PE_Name = kwargs.get("PE_Name")
        self.phase = phase
        self.distributed = kwargs.get("distributed", False)
        self.rank = kwargs.get("rank", 0)
        self.local_rank = kwargs.get("local_rank", 0)
        
        # 检查 LoRA 配置
        logger = logging.getLogger(__name__)
        use_lora = getattr(cfg.MODEL, 'USE_LORA', False)
        if use_lora:
            logger.info(f"=> Building model with LoRA: r={cfg.MODEL.LORA_R}, alpha={cfg.MODEL.LORA_ALPHA}, targets={cfg.MODEL.LORA_TARGET_MODULES}")
        else:
            logger.info("=> Building model without LoRA")
        logger.info("Set the random seed to {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
        
        self.dataloader = build_eval_loader(cfg, phase)
        self.model = build_model(cfg, phase=phase)
        self.dataset = self.dataloader.dataset
        self.GPUS = cfg.GPUS

        self.output = cfg.OUTPUT_DIR

        self.eval_from_checkpoint_id = kwargs.get("eval_from_checkpoint_id", -1)
        self.evaluate_model_state_files = []
        self.list_evaluate_model_files(cfg, phase)
        self.core_function = build_core_function(cfg, phase=phase, **kwargs)
        # self.core_function = build_core_function(cfg, PE_name=kwargs.get("PE_Name", "DcPose"), phase=phase,**kwargs)

        self.tb_writer_dict = {"writer": SummaryWriter(self.tb_save_folder),
                               "global_steps": 0}

    def list_evaluate_model_files(self, cfg, phase):
        subCfgNode = cfg.VAL if phase == VAL_PHASE else cfg.TEST
        if subCfgNode.MODEL_FILE:
            if subCfgNode.MODEL_FILE[0] == '.':
                model_state_file = osp.abspath(osp.join(self.checkpoints_save_folder, subCfgNode.MODEL_FILE))
            else:
                model_state_file = osp.join(self.cfg.ROOT_DIR, subCfgNode.MODEL_FILE)

            # model_state_file = osp.abspath(osp.join(cfg.ROOT_DIR, subCfgNode.MODEL_FILE))
            self.evaluate_model_state_files.append(model_state_file)
        else:
            if self.eval_from_checkpoint_id == -1:
                model_state_file = get_latest_checkpoint(self.checkpoints_save_folder)
                self.evaluate_model_state_files.append(model_state_file)
            else:
                candidate_model_files = get_all_checkpoints(self.checkpoints_save_folder)
                for model_file in candidate_model_files:
                    model_file_epoch_num = int(osp.basename(model_file).split("_")[1])
                    if model_file_epoch_num >= self.eval_from_checkpoint_id:
                        self.evaluate_model_state_files.append(model_file)

    def eval(self):
        if len(self.evaluate_model_state_files) == 0:
            logger = logging.getLogger(__name__)
            logger.error("=> No model state file available for evaluation")
        else:
            for model_checkpoint_file in self.evaluate_model_state_files:
                model, epoch = self.model_load(model_checkpoint_file)
                self.core_function.eval(model=model, dataloader=self.dataloader, tb_writer_dict=self.tb_writer_dict, epoch=epoch,
                                        phase=self.phase,cfg = self.cfg)

    def model_load(self, checkpoints_file):
        logger = logging.getLogger(__name__)
        logger.info("=> loading checkpoints from {}".format(checkpoints_file))
        checkpoint_dict = torch.load(checkpoints_file)
        # epoch = checkpoint_dict['begin_epoch']
        epoch = checkpoint_dict.get("begin_epoch", "0")
        model = self.model

        if "state_dict" in checkpoint_dict:
            model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}

        else:
            model_state_dict = checkpoint_dict

        if self.PE_Name == 'MSRA':
            model_state_dict = {k.replace('rough_pose_estimation_net.', ''): v for k, v in model_state_dict.items()}
        
        # 检测 checkpoint 是否包含 LoRA 结构
        has_lora_in_checkpoint = any('lora_A' in k or 'lora_B' in k for k in model_state_dict.keys())
        # 检测当前模型是否包含 LoRA 结构
        model_keys = set(model.state_dict().keys())
        has_lora_in_model = any('lora_A' in k or 'lora_B' in k for k in model_keys)
        
        logger.info(f"=> Checkpoint has LoRA: {has_lora_in_checkpoint}, Model has LoRA: {has_lora_in_model}")
        logger.info(f"=> Checkpoint keys count: {len(model_state_dict)}, Model keys count: {len(model_keys)}")
        
        # 仅当 checkpoint 与模型 LoRA 结构不一致时使用 strict=False，否则必须严格匹配，避免静默部分加载
        if has_lora_in_checkpoint != has_lora_in_model:
            logger.warning("=> LoRA structure mismatch between checkpoint and model. Using strict=False to load.")
            logger.warning(f"=> This usually means the config MODEL.USE_LORA doesn't match between training and validation.")
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
            if missing_keys:
                logger.warning(f"=> Missing keys (will use default/pretrained): {len(missing_keys)} keys")
                missing_samples = [k for k in missing_keys if 'qkv' in k or 'proj' in k][:5]
                if missing_samples:
                    logger.warning(f"   Sample missing keys: {missing_samples}")
            if unexpected_keys:
                logger.warning(f"=> Unexpected keys (will be ignored): {len(unexpected_keys)} keys")
                unexpected_samples = [k for k in unexpected_keys if 'lora' in k.lower()][:5]
                if unexpected_samples:
                    logger.warning(f"   Sample unexpected keys: {unexpected_samples}")
        else:
            # 结构一致时用 strict=True，确保完整加载，避免 LoRA 改动引入的静默错误
            model.load_state_dict(model_state_dict, strict=True)
        if self.distributed:
            model = model.cuda(self.local_rank)
        elif len(self.GPUS) > 1:
            model = torch.nn.DataParallel(model.cuda())
        else:
            model = model.cuda()
        return model, epoch
