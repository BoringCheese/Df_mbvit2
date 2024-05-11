import os
import subprocess

# 定义参数的值
NNODES = 1
NODE_RANK = 0
MASTER_ADDR = "127.0.0.1"
GPUS = 1
PORT = 29158

# 构建命令行参数
command = [
    "python", "-m", "torch.distributed.launch",
    "--nnodes=" + str(NNODES),
    "--node_rank=" + str(NODE_RANK),
    "--master_addr=" + MASTER_ADDR,
    "--nproc_per_node=" + str(GPUS),
    "--master_port=" + str(PORT),
    "utils/train.py",
    "--config=local_configs.NYUDepthv2.DFormer_Tiny",
    "--gpus=" + str(GPUS)
]

# 执行命令
subprocess.run(command)
