#!/bin/bash

set -euo pipefail

# ============================================================
# UCR_CLSA evaluation launcher for DinoTraining
# Uses checkpoint from /home/yanivgra/DinoTraining/output_lora_8gpu
# ============================================================

PROJECT_ROOT="/home/yanivgra/DinoTraining"
DINO_DIR="$PROJECT_ROOT/dino"
DATA_ROOT="/home/yanivgra/Frequency-masked-Embedding-Inference/datasets_clsa/UCR_CLSA"
OUTPUT_DIR="$PROJECT_ROOT/output_lora_8gpu/eval_ucr_clsa"

# Default checkpoint is the latest training checkpoint, but you can override
# it to evaluate checkpoint0000.pth or any other saved checkpoint.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_ROOT/output_lora_8gpu/checkpoint.pth}"

# Defaults (override with environment variables if needed)
DATASET="${DATASET:-all}"                      # Example: ECG200 or all
USE_LORA="${USE_LORA:-true}"                   # true/false
LINEAR_FINETUNE="${LINEAR_FINETUNE:-true}"     # true=linear head, false=zero-shot
PRINT_LOGS="${PRINT_LOGS:-true}"               # true/false
COMPUTE_TSNE="${COMPUTE_TSNE:-false}"          # true/false
DEVICE="${DEVICE:-cuda:0}"                     # cuda:0 or cpu
CHECKPOINT_KEY="${CHECKPOINT_KEY:-teacher}"    # teacher/student
INCLUDE_EXCLUDED_DATASETS="${INCLUDE_EXCLUDED_DATASETS:-false}"  # true/false
MIN_TRAIN_SAMPLES_FOR_EVAL="${MIN_TRAIN_SAMPLES_FOR_EVAL:-30}"
MIN_TEST_SAMPLES_FOR_EVAL="${MIN_TEST_SAMPLES_FOR_EVAL:-30}"
MIN_SEQUENCE_LENGTH_FOR_EVAL="${MIN_SEQUENCE_LENGTH_FOR_EVAL:-30}"
SKIP_NAN_INF_DATASETS="${SKIP_NAN_INF_DATASETS:-true}"

# Optional eval hyperparameters
BATCH_SIZE="${BATCH_SIZE:-64}"
LINEAR_BATCH_SIZE="${LINEAR_BATCH_SIZE:-256}"
LINEAR_EPOCHS="${LINEAR_EPOCHS:-100}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"

mkdir -p "$OUTPUT_DIR"

echo "[eval] checkpoint: $CHECKPOINT_PATH"
echo "[eval] dataset:    $DATASET"
echo "[eval] use_lora:   $USE_LORA"
echo "[eval] mode:       $LINEAR_FINETUNE (true=linear_finetune, false=zero_shot)"
echo "[eval] include_excluded_datasets: $INCLUDE_EXCLUDED_DATASETS"
echo "[eval] min_train_samples_for_eval: $MIN_TRAIN_SAMPLES_FOR_EVAL"
echo "[eval] min_test_samples_for_eval:  $MIN_TEST_SAMPLES_FOR_EVAL"
echo "[eval] min_sequence_length_for_eval: $MIN_SEQUENCE_LENGTH_FOR_EVAL"
echo "[eval] skip_nan_inf_datasets: $SKIP_NAN_INF_DATASETS"
echo "[eval] output_dir: $OUTPUT_DIR"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "[error] checkpoint not found: $CHECKPOINT_PATH"
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[error] data root not found: $DATA_ROOT"
  exit 1
fi

cd "$DINO_DIR"

# Run inside the requested conda environment.
conda run -n dino_training python eval_ucr_clsa.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --checkpoint_key "$CHECKPOINT_KEY" \
  --use_lora "$USE_LORA" \
  --linear_finetune "$LINEAR_FINETUNE" \
  --print_logs "$PRINT_LOGS" \
  --compute_tsne "$COMPUTE_TSNE" \
  --include_excluded_datasets "$INCLUDE_EXCLUDED_DATASETS" \
  --skip_nan_inf_datasets "$SKIP_NAN_INF_DATASETS" \
  --min_train_samples_for_eval "$MIN_TRAIN_SAMPLES_FOR_EVAL" \
  --min_test_samples_for_eval "$MIN_TEST_SAMPLES_FOR_EVAL" \
  --min_sequence_length_for_eval "$MIN_SEQUENCE_LENGTH_FOR_EVAL" \
  --data_root "$DATA_ROOT" \
  --dataset "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --arch vit_base \
  --patch_size 8 \
  --n_last_blocks 1 \
  --avgpool_patchtokens true \
  --lora_rank 8 \
  --lora_alpha 16.0 \
  --batch_size "$BATCH_SIZE" \
  --linear_batch_size "$LINEAR_BATCH_SIZE" \
  --linear_epochs "$LINEAR_EPOCHS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --num_workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --device "$DEVICE"

echo "[done] Evaluation finished."
