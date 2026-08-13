#!/usr/bin/env python3
"""Export a training checkpoint as a Hugging Face-ready safetensors directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


DEFAULT_CONFIG = {
    "architecture": "JoggAvatar",
    "base_model": "Wan-AI/Wan2.2-TI2V-5B",
    "model_size": "5B",
    "weight_file": "diffusion_pytorch_model.safetensors",
    "i2v": True,
    "use_audio": True,
    "random_prefix_frames": True,
    "sp_size": 1,
    "model_config": {"in_dim": 97, "out_dim": 48, "audio_hidden_size": 32},
    "lora_target_modules": "q,k,v,o,ffn.0,ffn.2",
    "init_lora_weights": "kaiming",
    "lora_rank": 128,
    "lora_alpha": 64.0,
    "train_architecture": "lora",
}


def normalize_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a mapping.")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint state_dict must be a mapping.")

    tensors: dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            continue
        normalized_name = name.removeprefix("pipe.dit.")
        tensors[normalized_name] = value.detach().contiguous().cpu()
    if not tensors:
        raise ValueError("Checkpoint contains no tensors.")
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Lightning .ckpt or tensor state dict.")
    parser.add_argument("output_dir", help="Destination model directory.")
    parser.add_argument(
        "--config",
        help="Optional config.json to use instead of the bundled 5B defaults.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output files.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    weight_path = output_dir / "diffusion_pytorch_model.safetensors"
    config_path = output_dir / "config.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not args.force and (weight_path.exists() or config_path.exists()):
        raise FileExistsError(f"Output already exists in {output_dir}; pass --force")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    state_dict = normalize_state_dict(checkpoint)
    config = DEFAULT_CONFIG
    if args.config:
        with open(args.config, "r", encoding="utf-8") as file:
            config = json.load(file)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        state_dict,
        weight_path,
        metadata={
            "format": "pt",
            "architecture": "Jogg-Avatar-Wan2.2-TI2V-5B",
            "base_model": str(config.get("base_model", DEFAULT_CONFIG["base_model"])),
        },
    )
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
        file.write("\n")
    print(f"Exported {len(state_dict)} tensors to {output_dir}")


if __name__ == "__main__":
    main()

