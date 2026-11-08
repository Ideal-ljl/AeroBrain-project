# Reproducing the software checks

This source archive contains the model definitions and checkpoint loader. Trained weights, datasets, training pipelines, simulator integrations, and benchmark results are not included. The verification below tests software behavior using random weights and synthetic inputs; it does not reproduce navigation success rates or onboard latency.

## Tested environment

The CPU checks were run on Linux with Python 3.10.20 and these installed packages:

| Package | Version |
| --- | --- |
| PyTorch | 2.6.0 |
| Transformers | 5.5.0 |
| safetensors | 0.7.0 |
| NumPy | 1.26.4 |
| Pillow | 11.3.0 |
| torchvision | 0.21.0 |
| pytest | 8.4.2 |

The installed PyTorch build supports CUDA 12.4, but the verification runs on CPU. Flash Attention and custom linear-attention kernels are not required for this check.

## Run the verification

From the repository root, in an environment with the dependencies installed:

```bash
pip install -e '.[test]'
python -m pytest -q tests/test_checkpoint.py
```

The test builds a small model with full and linear attention, a vision encoder, MAM, MSP, and an action Expert. It saves and reloads the full checkpoint, checks exact weight equality and the MAM chunk size, and compares deterministic action outputs with active memory, repeated cache reuse, and empty history. It uses temporary files and needs no network access or model download.

For a trained checkpoint, use `load_model.py` and the input contract in the README. The checkpoint must include the environment names and normalization parameters corresponding to the observations. A standalone source archive is not sufficient to run a trained navigation policy without these weights and inputs.
