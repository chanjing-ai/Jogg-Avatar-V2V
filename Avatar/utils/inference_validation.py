from __future__ import annotations

import json
import os
from pathlib import Path


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4"}


def split_model_paths(value: str) -> list[str]:
    return [path.strip() for path in value.split(",") if path.strip()]


def sidecar_paths(
    video_path: str,
    mouth_info_path: str | None = None,
    latent_path: str | None = None,
) -> tuple[str, str]:
    stem = os.path.splitext(video_path)[0]
    return (
        mouth_info_path or f"{stem}_mouth_info.json",
        latent_path or f"{stem}.tensors.vae2.2.pth",
    )


def validate_runtime_config(args) -> None:
    errors: list[str] = []
    required_files = {
        "text encoder": [args.text_encoder_path],
        "Wan2.2 VAE": [args.vae_path],
        "Jogg-Avatar checkpoint": [args.lora_ckpt_path],
        "Wan2.2 DiT": split_model_paths(args.dit_path),
    }
    for label, paths in required_files.items():
        for path in paths:
            if not path or not os.path.isfile(path):
                errors.append(f"Missing {label}: {path or '<not configured>'}")

    if args.use_audio and not os.path.isdir(args.wav2vec_path):
        errors.append(f"Missing Wav2Vec directory: {args.wav2vec_path}")
    if args.sp_size != args.world_size:
        errors.append(
            f"sp_size ({args.sp_size}) must equal torchrun world size ({args.world_size})"
        )
    if args.overlap_frame < 1 or args.overlap_frame % 4 != 1:
        errors.append("overlap_frame must equal 1 + 4*n")
    if args.overlap_frame >= 121:
        errors.append("overlap_frame must be smaller than 121")
    if args.num_steps < 1:
        errors.append("num_steps must be at least 1")
    if args.fps <= 0 or args.sample_rate <= 0:
        errors.append("fps and sample_rate must be positive")
    if errors:
        raise ValueError("Invalid inference configuration:\n- " + "\n- ".join(errors))


def validate_job(
    video_path: str,
    audio_path: str | None,
    use_audio: bool,
    mouth_info_path: str | None = None,
    latent_path: str | None = None,
) -> tuple[str, str]:
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"Source video not found: {video}")
    if video.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported source video extension: {video.suffix}")
    if use_audio and not audio_path:
        raise ValueError("Driving audio is required when use_audio is enabled")
    if audio_path and not Path(audio_path).is_file():
        raise FileNotFoundError(f"Driving audio not found: {audio_path}")

    mouth_info, latent = sidecar_paths(video_path, mouth_info_path, latent_path)
    if not os.path.isfile(latent):
        if not os.path.isfile(mouth_info):
            raise FileNotFoundError(
                f"Face-box JSON not found: {mouth_info}. Run extract_mouth_info.py first."
            )
        with open(mouth_info, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not data or not all("face_bbox" in item for item in data.values()):
            raise ValueError(f"Invalid face-box JSON: {mouth_info}")
    return mouth_info, latent

