#!/usr/bin/python
# -*- coding:utf8 -*-

import os
import sys
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from posetimation import get_cfg, update_config
from engine import default_parse_args, DefaultRunner


def setup(args):
    # 读取配置文件并合并命令行覆盖项。
    cfg = get_cfg(args)
    update_config(cfg, args)

    # cudnn related setting
    # 这里控制 cuDNN 的性能和确定性。
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED

    return cfg


@record
def main():
    # 解析命令行参数。
    args = default_parse_args()
    # 构建并更新配置。
    cfg = setup(args)
    # 创建统一的运行调度器。
    runner = DefaultRunner(cfg, args)
    try:
        # 启动训练/验证/测试。
        runner.launch()
    finally:
        # 如果当前进程组已经初始化，就在退出前销毁，避免资源残留。
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
