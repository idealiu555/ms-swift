import argparse

from swift import InferRequest, TransformersEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Qwen3.5 integer-addition action-head inference.')
    parser.add_argument('--checkpoint', required=True, help='Path to a full-training checkpoint.')
    parser.add_argument('--a', type=int, required=True)
    parser.add_argument('--b', type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    a, b = sorted((args.a, args.b))
    engine = TransformersEngine(
        args.checkpoint,
        task_type='seq_cls',
        num_labels=1,
    )
    request = InferRequest(messages=[{
        'role': 'user',
        'content': f'Calculate: {a} + {b}',
    }])
    response = engine.infer([request])[0]
    prediction = float(response.choices[0].message.content)
    print(f'prediction={prediction:.6f} expected={a + b}')


if __name__ == '__main__':
    main()
