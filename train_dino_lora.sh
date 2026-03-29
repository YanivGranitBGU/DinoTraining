#!/bin/bash

# Tiny 1-GPU DINO run with LoRA adapters on multivariate UCR data.
# Use this as a quick smoke test to verify that training runs
# and the loss decreases while only LoRA + head parameters are trained.

# Move into the folder where the python code lives
cd /home/yanivgra/DinoTraining/dino

python main_dino.py \
  --arch vit_tiny \
  --patch_size 8 \
  --data_path /home/yanivgra/DinoTraining/data/multivariate \
  --output_dir /home/yanivgra/DinoTraining/output_smoke_1gpu_lora \
  --epochs 10 \
  --warmup_epochs 0 \
  --batch_size_per_gpu 16 \
  --saveckp_freq 0 \
  --lr 0.0005 \
  --use_lora true \
  --lora_rank 8 \
  --lora_alpha 16.0
