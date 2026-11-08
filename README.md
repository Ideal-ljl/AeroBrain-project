# AeroBrain

AeroBrain combines a Qwen3.5 vision-language backbone, Motion-Adaptive Memory (MAM), a template motion prior (MSP), and an action Expert. The motion head predicts a motion class, path length, and yaw from the history token and current state. The Expert generates an action chunk by flow matching.

This repository contains model definitions and checkpoint loading. Trained weights, datasets, and evaluation pipelines are not included in this source archive.

## Installation

```bash
pip install -e .
```

## Loading a checkpoint

```bash
python load_model.py /path/to/checkpoint
```

The checkpoint directory must contain `config.json` and complete model weights. Sharded safetensors are supported. Pass `--no-processor` when the image/text processor is stored separately.

```python
import torch
from load_model import load_model

model, processor = load_model(
    "/path/to/checkpoint",
    device="cuda",
    dtype=torch.bfloat16,
    load_processor=False,
)
```

Templates, environment normalization, covariance matrices, and prediction-head weights are stored in the checkpoint. Loading fails if model weights are missing, unexpected, or incompatible in shape.

## Action inference

Prepare input tensors on the model device, then call `model.sample_actions`. Each sample must contain exactly one visible `history_token_id`. States and past actions must use the checkpoint's per-environment normalization.

```python
environment_names = model.motion_prior.spec["environments"]
environment_ids = model.motion_prior.environment_ids(
    [environment_names[0]], device=model.device
)
actions, source_noise = model.sample_actions(
    **inputs,
    motion_environment=environment_ids,
    denoise_steps=5,
)
```

`inputs` contains the processed instruction, current observation, and state. For MAM, provide action-history tensors, selected anchor images, event masks, and segment endpoints through the `mam_*` arguments. Segment endpoints are inclusive. Empty history produces no memory prefix.

Choose the environment matching the observation; the example uses the first configured environment. `sample_actions` returns `(actions, source_noise)`. Both tensors have shape `[batch, chunk_size, 4]` in normalized coordinates. The caller handles history management, image/text preprocessing, and action denormalization.

## Layout

- `load_model.py`: checkpoint and processor loading.
- `openfly_his/`: model configuration, vision-language backbone wrapper, action Expert, motion head, and template prior.
- `airbrain/memory/mam/`: event selection, action-segment encoding, and visual memory encoding.

## Verification

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the tested environment and a self-contained CPU checkpoint/inference test.

## Citation

Software author: Junli Liu. Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

The code is licensed under [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for copyright and attribution information.
