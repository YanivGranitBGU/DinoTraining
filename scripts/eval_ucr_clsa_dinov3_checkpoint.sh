#!/bin/bash

set -euo pipefail

# ============================================================
# UCR_CLSA evaluation launcher for DINOv3
# ============================================================

PROJECT_ROOT="/home/yanivgra/DinoTraining"
DINOV3_DIR="$PROJECT_ROOT/dinov3"
DATA_ROOT="/home/yanivgra/Frequency-masked-Embedding-Inference/datasets_clsa"
OUTPUT_DIR="$PROJECT_ROOT/output_dinov3/eval_ucr_clsa_dinov3"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

# Defaults (override with environment variables)
DATASET="${DATASET:-all}"                      # Example: ECG200 or all
LINEAR_FINETUNE="${LINEAR_FINETUNE:-true}"     # true=linear head, false=zero-shot
PRINT_LOGS="${PRINT_LOGS:-true}"               # true/false
COMPUTE_TSNE="${COMPUTE_TSNE:-false}"          # true/false
DEVICE="${DEVICE:-cuda:0}"                     # cuda:0 or cpu
CHECKPOINT_KEY="${CHECKPOINT_KEY:-model}"      # model/state_dict key

INCLUDE_EXCLUDED_DATASETS="${INCLUDE_EXCLUDED_DATASETS:-false}"  # true/false
MIN_TRAIN_SAMPLES_FOR_EVAL="${MIN_TRAIN_SAMPLES_FOR_EVAL:-30}"
MIN_TEST_SAMPLES_FOR_EVAL="${MIN_TEST_SAMPLES_FOR_EVAL:-30}"
MIN_SEQUENCE_LENGTH_FOR_EVAL="${MIN_SEQUENCE_LENGTH_FOR_EVAL:-30}"
SKIP_NAN_INF_DATASETS="${SKIP_NAN_INF_DATASETS:-true}"

ARCH="${ARCH:-dinov3_vitb16}"                  # dinov3_vits16 / dinov3_vitb16 / dinov3_vitl16
AVGPOOL_PATCHTOKENS="${AVGPOOL_PATCHTOKENS:-true}"

# Optional eval hyperparameters
BATCH_SIZE="${BATCH_SIZE:-64}"
LINEAR_BATCH_SIZE="${LINEAR_BATCH_SIZE:-256}"
LINEAR_EPOCHS="${LINEAR_EPOCHS:-100}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"

mkdir -p "$OUTPUT_DIR"

echo "[eval] checkpoint: ${CHECKPOINT_PATH:-<none>}"
echo "[eval] dataset:    $DATASET"
echo "[eval] arch:       $ARCH"
echo "[eval] mode:       $LINEAR_FINETUNE (true=linear_finetune, false=zero_shot)"
echo "[eval] output_dir: $OUTPUT_DIR"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[error] data root not found: $DATA_ROOT"
  exit 1
fi

if [[ -n "$CHECKPOINT_PATH" && ! -f "$CHECKPOINT_PATH" ]]; then
  echo "[error] checkpoint not found: $CHECKPOINT_PATH"
  exit 1
fi

cd "$DINOV3_DIR"

conda run -n dino_training310 env PYTHONPATH=. python eval_ucr_clsa.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --checkpoint_key "$CHECKPOINT_KEY" \
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
  --arch "$ARCH" \
  --avgpool_patchtokens "$AVGPOOL_PATCHTOKENS" \
  --batch_size "$BATCH_SIZE" \
  --linear_batch_size "$LINEAR_BATCH_SIZE" \
  --linear_epochs "$LINEAR_EPOCHS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --num_workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --device "$DEVICE"

echo "[done] DINOv3 UCR evaluation finished."
