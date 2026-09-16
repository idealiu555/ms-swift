# Qwen3.5 continuous action head

This example replaces the causal language-model head with ms-swift's scalar
sequence-regression head and fine-tunes only the final two text decoder blocks.
The training target is an integer value, not a tokenized answer.

## 1. Generate data

The default command creates 9,000 training examples and 1,000 validation
examples. Operand pairs are unique up to commutativity.

```bash
python examples/train/seq_cls/qwen3_5_action_head/data/generate_dataset.py
```

## 2. Train

```bash
bash examples/train/seq_cls/qwen3_5_action_head/train/train.sh
```

With `task_type=seq_cls`, `num_labels=1`, and `problem_type=regression`, ms-swift
replaces `lm_head` with `Identity`, creates a four-layer SiLU MLP as `score`,
pools the final non-padding hidden state, and optimizes mean squared error. The
training script freezes every parameter before re-enabling decoder blocks 22 and
23 and `score`.

## 3. Infer

Use the path of a checkpoint emitted under the example's `output` directory:

```bash
python examples/train/seq_cls/qwen3_5_action_head/infer/infer.py \
    --checkpoint examples/train/seq_cls/qwen3_5_action_head/output/vx-xxx/checkpoint-xxx \
    --a 123 \
    --b -45
```

This is a forward regression pass; it does not call token generation. Accuracy
outside the training range `[-1000, 1000]` is not guaranteed.
