#!/bin/bash

set -euo pipefail

# ============================================================
# UCR_CLSA checkpoint sweep for DinoTraining
# Evaluates zero-shot with best label remapping on the requested checkpoints.
# ============================================================

PROJECT_ROOT="/home/yanivgra/DinoTraining"
DINO_DIR="$PROJECT_ROOT/dino"
DATA_ROOT="/home/yanivgra/Frequency-masked-Embedding-Inference/datasets_clsa/UCR_CLSA"
OUTPUT_ROOT="$PROJECT_ROOT/output_lora_8gpu_DelayEmbeddings/eval_ucr_clsa_zeroshot_bestperm"

# Eval defaults (override with environment variables if needed)
DATASET="${DATASET:-all}"
PRINT_LOGS="${PRINT_LOGS:-true}"
COMPUTE_TSNE="${COMPUTE_TSNE:-false}"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT_KEY="${CHECKPOINT_KEY:-teacher}"
INCLUDE_EXCLUDED_DATASETS="${INCLUDE_EXCLUDED_DATASETS:-false}"
MIN_TRAIN_SAMPLES_FOR_EVAL="${MIN_TRAIN_SAMPLES_FOR_EVAL:-30}"
MIN_TEST_SAMPLES_FOR_EVAL="${MIN_TEST_SAMPLES_FOR_EVAL:-30}"
MIN_SEQUENCE_LENGTH_FOR_EVAL="${MIN_SEQUENCE_LENGTH_FOR_EVAL:-30}"
SKIP_NAN_INF_DATASETS="${SKIP_NAN_INF_DATASETS:-true}"
SEARCH_LABEL_PERMUTATIONS="${SEARCH_LABEL_PERMUTATIONS:-true}"

# Model/eval params
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"

BASE_CKPT="$PROJECT_ROOT/dino_vitbase8_pretrain_full_checkpoint.pth"
CKPT_0002="$PROJECT_ROOT/output_lora_8gpu_DelayEmbeddings/checkpoints/checkpoint0002.pth"
CKPT_0004="$PROJECT_ROOT/output_lora_8gpu_DelayEmbeddings/checkpoints/checkpoint0004.pth"
CKPT_0007="$PROJECT_ROOT/output_lora_8gpu_DelayEmbeddings/checkpoints/checkpoint0007.pth"

mkdir -p "$OUTPUT_ROOT"

declare -a CHECKPOINT_PATHS=()
declare -a CHECKPOINT_TAGS=()
declare -a USE_LORA_FLAGS=()

# Pre-LoRA base model
CHECKPOINT_PATHS+=("$BASE_CKPT")
CHECKPOINT_TAGS+=("pre_lora")
USE_LORA_FLAGS+=("false")

# Early LoRA checkpoint after 2 epochs
CHECKPOINT_PATHS+=("$CKPT_0002")
CHECKPOINT_TAGS+=("lora_0002")
USE_LORA_FLAGS+=("true")

# Requested LoRA checkpoints
CHECKPOINT_PATHS+=("$CKPT_0004")
CHECKPOINT_TAGS+=("lora_0004")
USE_LORA_FLAGS+=("true")

CHECKPOINT_PATHS+=("$CKPT_0007")
CHECKPOINT_TAGS+=("lora_0007")
USE_LORA_FLAGS+=("true")

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[error] data root not found: $DATA_ROOT"
  exit 1
fi

cd "$DINO_DIR"

for i in "${!CHECKPOINT_PATHS[@]}"; do
  ckpt="${CHECKPOINT_PATHS[$i]}"
  tag="${CHECKPOINT_TAGS[$i]}"
  use_lora="${USE_LORA_FLAGS[$i]}"

  if [[ ! -f "$ckpt" ]]; then
    echo "[warn] skipping missing checkpoint: $ckpt"
    continue
  fi

  echo "============================================================"
  echo "[eval] checkpoint_tag: $tag"
  echo "[eval] checkpoint:     $ckpt"
  echo "[eval] use_lora:       $use_lora"
  echo "[eval] zero_shot:      true"
  echo "[eval] dataset:        $DATASET"
  echo "[eval] skip_nan_inf_datasets: $SKIP_NAN_INF_DATASETS"
  echo "============================================================"

  CHECKPOINT_OUTPUT_DIR="$OUTPUT_ROOT/$tag"
  mkdir -p "$CHECKPOINT_OUTPUT_DIR"

  conda run -n dino_training python eval_ucr_clsa.py \
    --checkpoint_path "$ckpt" \
    --checkpoint_key "$CHECKPOINT_KEY" \
    --checkpoint_tag "$tag" \
    --use_lora "$use_lora" \
    --linear_finetune false \
    --search_label_permutations "$SEARCH_LABEL_PERMUTATIONS" \
    --print_logs "$PRINT_LOGS" \
    --compute_tsne "$COMPUTE_TSNE" \
    --include_excluded_datasets "$INCLUDE_EXCLUDED_DATASETS" \
    --skip_nan_inf_datasets "$SKIP_NAN_INF_DATASETS" \
    --min_train_samples_for_eval "$MIN_TRAIN_SAMPLES_FOR_EVAL" \
    --min_test_samples_for_eval "$MIN_TEST_SAMPLES_FOR_EVAL" \
    --min_sequence_length_for_eval "$MIN_SEQUENCE_LENGTH_FOR_EVAL" \
    --data_root "$DATA_ROOT" \
    --dataset "$DATASET" \
    --output_dir "$CHECKPOINT_OUTPUT_DIR" \
    --arch vit_base \
    --patch_size 8 \
    --n_last_blocks 1 \
    --avgpool_patchtokens true \
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --device "$DEVICE"
done

echo "[done] Checkpoint sweep finished. Results are under: $OUTPUT_ROOT"
