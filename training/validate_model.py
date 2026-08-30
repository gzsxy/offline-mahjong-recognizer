#!/usr/bin/env python3
"""Print the tensor contract of a TFLite Mahjong detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "app/src/main/assets/mahjong_28cls_int8.tflite"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()
    input_tensor = interpreter.get_input_details()[0]
    output_tensor = interpreter.get_output_details()[0]

    print(f"model: {args.model}")
    print(
        "input: "
        f"shape={input_tensor['shape'].tolist()} "
        f"dtype={input_tensor['dtype'].__name__} "
        f"quantization={input_tensor['quantization']}"
    )
    print(
        "output: "
        f"shape={output_tensor['shape'].tolist()} "
        f"dtype={output_tensor['dtype'].__name__} "
        f"quantization={output_tensor['quantization']}"
    )

    shape = output_tensor["shape"].tolist()
    if len(shape) != 3 or shape[0] != 1 or shape[2] != 8400 or shape[1] < 5:
        raise SystemExit(f"unexpected output shape, expected [1, 4+classes, 8400]: {shape}")
    print(f"contract: OK (4 box channels + {shape[1] - 4} class channels)")


if __name__ == "__main__":
    main()
