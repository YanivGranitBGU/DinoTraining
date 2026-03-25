#!/bin/bash

# Move into the folder where the python code lives
cd /home/yanivgra/DinoTraining/dino

# Run the training
# Note: I updated the path to the weights to be absolute so it still works
python -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
    --arch vit_base \
    --patch_size 8 \
    --pretrained_weights /home/yanivgra/DinoTraining/dino_vitbase8_pretrain_full_checkpoint.pth \
    --data_path /path/to/your/custom_dataset \
    --output_dir /home/yanivgra/DinoTraining/output \
    --epochs 100 \
    --lr 0.0001