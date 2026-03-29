#!/bin/bash

# 1-GPU DINO run with LoRA adapters on multivariate UCR data,
# starting from a pretrained ViT-Base/8 DINO checkpoint.

# Move into the folder where the python code lives and ensure it's on PYTHONPATH
cd /home/yanivgra/DinoTraining/dino
export PYTHONPATH=/home/yanivgra/DinoTraining/dino:"$PYTHONPATH"

python main_dino.py \
  --arch vit_base \
  --patch_size 8 \
  --data_path /home/yanivgra/DinoTraining/data/multivariate \
  --output_dir /home/yanivgra/DinoTraining/output_smoke_1gpu_lora \
  --epochs 10 \
  --warmup_epochs 0 \
  --batch_size_per_gpu 8 \
  --saveckp_freq 0 \
  --lr 0.0005 \
  --pretrained_weights ../dino_vitbase8_pretrain_full_checkpoint.pth \
  --checkpoint_key teacher \
  --use_lora true \
  --lora_rank 8 \
  --lora_alpha 16.0
