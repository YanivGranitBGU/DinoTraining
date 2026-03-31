#!/bin/bash

# Slurm directives for running DINO+LoRA on 8 GPUs.
# Adjust partition/qos/time if needed for your cluster.
#SBATCH --job-name=dino_lora
#SBATCH --partition=rtx6000
#SBATCH --qos=azencot
#SBATCH --gpus=rtx_6000:8
#SBATCH --cpus-per-task=20
#SBATCH --time=7-00:00:00
#SBATCH -o /home/yanivgra/DinoTraining/slurm_logs/dino_lora_%j.out

# Move into the folder where the python code lives and ensure it's on PYTHONPATH
cd /home/yanivgra/DinoTraining/dino
export PYTHONPATH=/home/yanivgra/DinoTraining/dino:"$PYTHONPATH"

python -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
  --arch vit_base \
  --patch_size 8 \
  --data_path /home/yanivgra/DinoTraining/data/multivariate \
  --output_dir /home/yanivgra/DinoTraining/output_lora_8gpu \
  --epochs 10 \
  --warmup_epochs 1 \
  --batch_size_per_gpu 8 \
  --saveckp_freq 1 \
  --global_samples_per_epoch 100000 \
  --lr 0.0005 \
  --pretrained_weights ../dino_vitbase8_pretrain_full_checkpoint.pth \
  --checkpoint_key teacher \
  --use_lora true \
  --lora_rank 8 \
  --lora_alpha 16.0
