#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=$(dirname "$SCRIPT_DIR")

# Qwen3.5-0.8B has 24 text decoder blocks. Freeze the complete model, then
# activate only the final two blocks and the scalar regression/action head.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
swift sft \
    --model Qwen/Qwen3.5-0.8B \
    --dataset "$EXAMPLE_DIR/data/train.jsonl" \
    --val_dataset "$EXAMPLE_DIR/data/validation.jsonl" \
    --task_type seq_cls \
    --num_labels 1 \
    --problem_type regression \
    --tuner_type full \
    --freeze_parameters_ratio 1 \
    --trainable_parameters model.language_model.layers.22 model.language_model.layers.23 score \
    --torch_dtype bfloat16 \
    --use_chat_template false \
    --max_length 64 \
    --packing false \
    --padding_free false \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --output_dir "$EXAMPLE_DIR/output"
