#!/bin/bash

# Move into the folder where the python code lives
cd /home/yanivgra/DinoTraining/dino

# Default: tiny 1-GPU smoke test on multivariate UCR data
# Use this to quickly verify that training runs and the loss decreases.
python main_dino.py \
    --arch vit_tiny \
    --patch_size 8 \
    --data_path /home/yanivgra/DinoTraining/data/multivariate \
    --output_dir /home/yanivgra/DinoTraining/output_smoke_1gpu \
    --epochs 10 \
    --warmup_epochs 0 \
    --batch_size_per_gpu 16 \
    --saveckp_freq 0 \
    --lr 0.0005

# For full training on multiple GPUs, you can switch back to e.g.:
# python -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
#   --arch vit_base \
#   --patch_size 8 \
#   --data_path /home/yanivgra/DinoTraining/data/multivariate \
#   --output_dir /home/yanivgra/DinoTraining/output \
#   --epochs 100 \
#   --lr 0.0001