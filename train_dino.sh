#!/bin/bash

# Move into the folder where the python code lives
cd /home/yanivgra/DinoTraining/dino

# Run the training
# Note: I updated the path to the weights to be absolute so it still works
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
    --arch vit_base \
    --patch_size 8 \
    --data_path /home/yanivgra/DinoTraining/data/multivariate \
    --output_dir /home/yanivgra/DinoTraining/output \
    --epochs 100 \
    --lr 0.0001

# For tiny training-  data PR
# cd /home/yanivgra/DinoTraining/dino

# python -m torch.distributed.launch --nproc_per_node=1 main_dino.py \
#   --arch vit_tiny \
#   --patch_size 8 \
#   --data_path /home/yanivgra/DinoTraining/data/multivariate \
#   --output_dir /home/yanivgra/DinoTraining/output_smoke_1gpu \
#   --epochs 2 \
#   --warmup_epochs 0 \
#   --batch_size_per_gpu 16 \
#   --saveckp_freq 0 \
#   --lr 0.0005    