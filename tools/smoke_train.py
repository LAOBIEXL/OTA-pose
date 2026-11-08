#!/usr/bin/env python3
"""Run one real training batch with the configured model, dataset and loss."""
import argparse
import os
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import torch
from posetimation import get_cfg, update_config
from engine.defaults.trainer import DefaultTrainer

class OneBatch:
    def __init__(self, loader):
        self.loader = loader
    def __len__(self):
        return 1
    def __iter__(self):
        return iter(self.loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    args = parser.parse_args()
    args.cfg = str((root / args.cfg).resolve())
    args.rootDir = str(root)
    args.opts = []
    cfg = get_cfg(args)
    update_config(cfg, args)
    cfg.defrost()
    cfg.MODEL.EVALUATE = False
    cfg.TRAIN.BATCH_SIZE_PER_GPU = 1
    cfg.TRAIN.AUTO_RESUME = False
    cfg.WORKERS = 0
    cfg.GPUS = (0,)
    cfg.OUTPUT_DIR = str(root / 'output' / 'smoke_train')
    cfg.freeze()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is not available')
    out = Path(cfg.OUTPUT_DIR) / Path(args.cfg).parent.name
    for part in ('logs', 'checkpoints', 'tensorboard'):
        (out / part).mkdir(parents=True, exist_ok=True)
    folders = {'log_save_folder': str(out/'logs'), 'checkpoints_save_folder': str(out/'checkpoints'), 'tb_save_folder': str(out/'tensorboard')}
    print('CONFIG:', args.cfg, 'GPU:', torch.cuda.get_device_name(0), flush=True)
    trainer = DefaultTrainer(cfg, folders)
    state = {'writer': None, 'global_steps': 0}
    trainer.core_function.train(model=trainer.model, epoch=0, optimizer=trainer.optimizer,
                                dataloader=OneBatch(trainer.dataloader), tb_writer_dict=state)
    if state['global_steps'] != 1:
        raise RuntimeError('Training did not finish one step')
    print('SMOKE_TRAIN_SUCCESS steps=1', flush=True)

if __name__ == '__main__':
    main()
