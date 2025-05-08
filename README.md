# Quantization

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue?logo=python)](https://www.python.org/)


## 🧠 Philosophy

This repo is designed to provide complete transparency and experimental freedom in weight quantization workflows. It exposes every part of the quantization and compression process—perfect for debugging, benchmarking, or building custom deployment tools.

## 🚀 Features

This repository is a modular, test-driven framework for experimenting with weight quantization, linear layer replacement, and custom packing schemes in PyTorch. It includes full support for symmetric/asymmetric quantization, granular control over scaling strategies, and bit-level weight packing for memory compression.

## ✅ Quantization Modes & Granularity

- Modes: **Symmetric**, **Asymmetric**

- Granularity:
  - **PerTensor**
  - **PerDimension** (row/column)
  - **PerGroup** (e.g., 32-element groups)

## 🧩 Custom Linear Layer Replacement

- `QWQALinearLayer`: A quantized wrapper over `nn.Linear` supporting `int8` weights with `float32`/`bfloat16` activations.

- Dynamically replaces any `nn.Linear` module in a PyTorch model with quantized counterparts using: `replace_linear_layers_with_w8a16(model, Target.Linear(QWQALinearLayer), exclude_list)`

## 🧮 Bit-Level Weight Packing

- Packs 2D tensors into lower-bit formats using bitwise operations (e.g., pack 2-bit weights into `uint8`)

- Optimized for memory compression and alignment

- Includes corresponding unpack routines

## 📦 Modules

```
quantization/
├── linear_layer.py          # Quantized LinearLayer and model replacement logic
├── linear_quantizer.py      # Quantization logic: scale, zero-point, modes and granularity
├── weight_pack.py           # Bitwise tensor packing/unpacking routines
├── main.py                  # Integration test suite for quantization and replacement
├── test_weight_pack.py      # Unit tests for weight packing
├── test_linear_quantizer.py # Unit tests for linear quantization logic
└── README.md                # Project overview and documentation
```

## 🧪 Running Tests

`python3 main.py`

Or run individual unit test modules:

```bash
python3 -m unittest test_linear_quantizer.py
python3 -m unittest test_weight_pack.py
```

## 🔍 Example Usage

```python
import torch
from linear_layer import QWQALinearLayer

layer = QWQALinearLayer(16, 32)
input = torch.randn(4, 16)
output = layer(input)
```
