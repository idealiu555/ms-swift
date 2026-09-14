import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate an integer-addition regression dataset.')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--num-samples', type=int, default=10_000)
    parser.add_argument('--num-validation-samples', type=int, default=1_000)
    parser.add_argument('--min-value', type=int, default=-1_000)
    parser.add_argument('--max-value', type=int, default=1_000)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def build_examples(num_samples: int, min_value: int, max_value: int, seed: int) -> List[Tuple[int, int]]:
    if min_value > max_value:
        raise ValueError('min-value must not be greater than max-value.')
    # Canonicalize each pair because a + b and b + a are the same example.
    pair_count = max_value - min_value + 1
    max_unique_pairs = pair_count * (pair_count + 1) // 2
    if num_samples > max_unique_pairs:
        raise ValueError(f'num-samples must not exceed {max_unique_pairs} for the configured value range.')

    rng = random.Random(seed)
    pairs = set()
    while len(pairs) < num_samples:
        a = rng.randint(min_value, max_value)
        b = rng.randint(min_value, max_value)
        pairs.add((min(a, b), max(a, b)))

    examples = list(pairs)
    rng.shuffle(examples)
    return examples


def write_jsonl(path: Path, examples: List[Tuple[int, int]]) -> None:
    with path.open('w', encoding='utf-8') as file:
        for a, b in examples:
            record = {
                'messages': [{
                    'role': 'user',
                    'content': f'Calculate: {a} + {b}',
                }],
                'label': a + b,
            }
            file.write(json.dumps(record, ensure_ascii=False) + '\n')


def main() -> None:
    args = parse_args()
    if not 0 < args.num_validation_samples < args.num_samples:
        raise ValueError('num-validation-samples must be between 1 and num-samples - 1.')

    examples = build_examples(args.num_samples, args.min_value, args.max_value, args.seed)
    split_index = args.num_samples - args.num_validation_samples
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / 'train.jsonl', examples[:split_index])
    write_jsonl(args.output_dir / 'validation.jsonl', examples[split_index:])
    print(f'Wrote {split_index} training and {args.num_validation_samples} validation examples to {args.output_dir}')


if __name__ == '__main__':
    main()
