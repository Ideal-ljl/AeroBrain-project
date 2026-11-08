"""Load a complete, self-contained MSP + MAM checkpoint."""

from pathlib import Path
import argparse
import torch
from transformers import AutoProcessor
from openfly_his.configuration_openfly import OpenFLYModelConfig
from openfly_his.modeling_openfly import OpenFLYModel


def load_model(
    checkpoint,
    *,
    device="cuda",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    load_processor=True,
):
    """Return the evaluation model and optional image/text processor.

    checkpoint must contain config.json and full model safetensors (or shards).
    Set load_processor=False if processor assets live in a separate directory.
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(
            f"Missing checkpoint config: {checkpoint / 'config.json'}"
        )
    config = OpenFLYModelConfig.from_pretrained(checkpoint, local_files_only=True)
    model, info = OpenFLYModel.from_pretrained(
        checkpoint,
        config=config,
        dtype=dtype,
        attn_implementation=attn_implementation,
        local_files_only=True,
        output_loading_info=True,
    )
    problems = {
        key: info[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if info.get(key)
    }
    if problems:
        raise RuntimeError(
            f"Checkpoint is not a complete compatible MSP + MAM model: {problems}"
        )
    model = model.to(device=device).eval()
    model.requires_grad_(False)
    processor = (
        AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        if load_processor
        else None
    )
    return model, processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
    )
    parser.add_argument("--no-processor", action="store_true")
    args = parser.parse_args()
    model, processor = load_model(
        args.checkpoint,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation,
        load_processor=not args.no_processor,
    )
    print(
        f"Loaded MSP + MAM: {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    print(f"Motion environments: {model.motion_prior.spec['environments']}")
