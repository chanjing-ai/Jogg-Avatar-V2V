from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import librosa
import lightning as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor

from Avatar.models.audio_pack import AudioPack
from Avatar.models.model_manager import ModelManager
from Avatar.models.vae2_2 import Wan2_2_VAE
from Avatar.models.wav2vec import Wav2VecModel
from Avatar.utils.io_utils import load_state_dict
from Avatar.wan_video import WanVideoPipeline


DEFAULT_PROMPT = (
    "A realistic video of a face speaking directly to the camera. The camera "
    "remains steady and every facial detail is sharp and clearly visible."
)
FEATURE_SUFFIX = ".tensors.vae2.2.pth"
REQUIRED_FEATURE_KEYS = {
    "image_lat",
    "latents",
    "pre_latents",
    "last_latents",
    "prompt_emb",
    "audio_emb",
}


def _training_state_dict(
    state_dict: dict[str, torch.Tensor], model_keys: set[str]
) -> dict[str, torch.Tensor]:
    """Accept both raw Lightning keys and normalized release weight keys."""
    converted = {}
    for name, value in state_dict.items():
        if name in model_keys:
            converted[name] = value
        elif f"pipe.dit.{name}" in model_keys:
            converted[f"pipe.dit.{name}"] = value
        else:
            converted[name] = value
    return converted


def _video_path(dataset_path: Path, file_name: str) -> Path:
    path = Path(file_name).expanduser()
    return path if path.is_absolute() else dataset_path / path


def _sidecar_path(video_path: Path, suffix: str) -> Path:
    return video_path.with_name(f"{video_path.stem}{suffix}")


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    return value


class VideoFeatureDataset(torch.utils.data.Dataset):
    """Load paired video/audio samples for Wan2.2 VAE feature extraction."""

    def __init__(
        self,
        dataset_path: str,
        metadata_path: str,
        clip_frames: int = 121,
        height: int = 480,
        width: int = 480,
        overwrite: bool = False,
    ) -> None:
        self.dataset_path = Path(dataset_path).expanduser()
        metadata = pd.read_csv(metadata_path)
        if "file_name" not in metadata.columns:
            raise ValueError(f"{metadata_path} must contain a file_name column")

        self.samples: list[tuple[Path, str]] = []
        for _, row in metadata.iterrows():
            path = _video_path(self.dataset_path, str(row["file_name"]))
            feature_path = Path(f"{path}{FEATURE_SUFFIX}")
            if feature_path.exists() and not overwrite:
                continue
            prompt = str(row.get("text", DEFAULT_PROMPT))
            if not prompt or prompt.lower() == "nan":
                prompt = DEFAULT_PROMPT
            self.samples.append((path, prompt))

        self.clip_frames = clip_frames
        self.total_frames = clip_frames * 2
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return len(self.samples)

    def _load_face_video(self, video_path: Path) -> torch.Tensor:
        mouth_info_path = _sidecar_path(video_path, "_mouth_info.json")
        if not video_path.is_file():
            raise FileNotFoundError(f"video not found: {video_path}")
        if not mouth_info_path.is_file():
            raise FileNotFoundError(f"face metadata not found: {mouth_info_path}")

        with mouth_info_path.open("r", encoding="utf-8") as file:
            mouth_info = json.load(file)
        xmin, xmax, ymin, ymax = map(int, mouth_info["1"]["face_bbox"])

        reader = imageio.get_reader(str(video_path))
        try:
            frame_count = reader.count_frames()
            if frame_count < self.total_frames:
                raise ValueError(
                    f"{video_path} has {frame_count} frames; "
                    f"at least {self.total_frames} are required"
                )

            frames = []
            for frame_id in range(self.total_frames):
                frame = reader.get_data(frame_id)
                frame_height, frame_width = frame.shape[:2]
                left = max(0, min(xmin, frame_width - 1))
                right = max(left + 1, min(xmax, frame_width))
                top = max(0, min(ymin, frame_height - 1))
                bottom = max(top + 1, min(ymax, frame_height))
                crop = Image.fromarray(frame[top:bottom, left:right]).convert("RGB")
                crop = crop.resize((self.width, self.height), Image.Resampling.BICUBIC)
                array = np.asarray(crop, dtype=np.float32) / 127.5 - 1.0
                frames.append(torch.from_numpy(array).permute(2, 0, 1))
        finally:
            reader.close()

        return torch.stack(frames, dim=1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        video_path, prompt = self.samples[index]
        try:
            frames = self._load_face_video(video_path)
            return {"path": str(video_path), "prompt": prompt, "frames": frames}
        except (FileNotFoundError, KeyError, ValueError, OSError) as error:
            return {"path": "", "error": f"{video_path}: {error}"}


class FeaturePreprocessor(pl.LightningModule):
    def __init__(
        self,
        text_encoder_path: str,
        vae_path: str,
        wav2vec_path: str,
        clip_frames: int = 121,
        fps: int = 25,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([text_encoder_path])
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.vae = Wan2_2_VAE(vae_pth=vae_path, device="cpu")
        self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_path, local_files_only=True
        )
        self.audio_encoder = Wav2VecModel.from_pretrained(
            wav2vec_path,
            local_files_only=True,
            attn_implementation="eager",
        )
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.clip_frames = clip_frames
        self.fps = fps
        self.sample_rate = sample_rate

    def on_test_start(self) -> None:
        self.pipe.device = self.device
        self.vae.todevice(dtype=torch.bfloat16, device=self.device)

    def _encode_audio(self, audio_path: Path) -> torch.Tensor:
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        input_values = np.squeeze(
            self.wav_feature_extractor(
                audio, sampling_rate=self.sample_rate
            ).input_values
        )
        audio_frames = math.ceil(len(input_values) / self.sample_rate * self.fps)
        if audio_frames < self.clip_frames:
            raise ValueError(
                f"audio provides {audio_frames} frames; "
                f"at least {self.clip_frames} are required"
            )

        values = torch.from_numpy(input_values).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            states = self.audio_encoder(
                values, seq_len=audio_frames, output_hidden_states=True
            )
            embedding = states.last_hidden_state
            for hidden_state in states.hidden_states:
                embedding = torch.cat((embedding, hidden_state), dim=-1)
        return embedding[:, : self.clip_frames]

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        if not batch["path"][0]:
            self.print(f"Skipping sample: {batch['error'][0]}")
            return

        video_path = Path(batch["path"][0])
        audio_path = _sidecar_path(video_path, ".wav")
        if not audio_path.is_file():
            self.print(f"Skipping sample without audio: {audio_path}")
            return

        frames = batch["frames"].squeeze(0).to(
            device=self.device, dtype=torch.bfloat16
        )
        source = frames[:, : self.clip_frames]
        reference = frames[:, self.clip_frames :]
        try:
            audio_embedding = self._encode_audio(audio_path)
        except (OSError, ValueError) as error:
            self.print(f"Skipping {video_path}: {error}")
            return

        with torch.no_grad():
            features = {
                "image_lat": self.vae.encode([reference])[0],
                "latents": self.vae.encode([source])[0],
                "pre_latents": self.vae.encode([source[:, :13]])[0],
                "last_latents": self.vae.encode([source[:, -13:]])[0],
                "prompt_emb": self.pipe.encode_prompt(batch["prompt"][0]),
                "audio_emb": audio_embedding,
            }
        output_path = Path(f"{video_path}{FEATURE_SUFFIX}")
        torch.save(_to_cpu(features), output_path)
        self.print(f"Saved {output_path}")


class Wan22FeatureDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset_path: str, metadata_path: str, steps_per_epoch: int | None
    ) -> None:
        dataset_root = Path(dataset_path).expanduser()
        metadata = pd.read_csv(metadata_path)
        if "file_name" not in metadata.columns:
            raise ValueError(f"{metadata_path} must contain a file_name column")

        self.feature_paths = []
        for file_name in metadata["file_name"]:
            video_path = _video_path(dataset_root, str(file_name))
            feature_path = Path(f"{video_path}{FEATURE_SUFFIX}")
            if feature_path.is_file():
                self.feature_paths.append(feature_path)
        if not self.feature_paths:
            raise ValueError(f"no {FEATURE_SUFFIX} files found from {metadata_path}")
        self.epoch_size = steps_per_epoch or len(self.feature_paths)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.feature_paths[index % len(self.feature_paths)]
        data = torch.load(path, map_location="cpu", weights_only=True)
        missing = REQUIRED_FEATURE_KEYS.difference(data)
        if missing:
            raise ValueError(f"{path} is missing feature keys: {sorted(missing)}")
        data["pre_fix_frames_num"] = 13 if random.random() < 0.5 else -13
        return data


class Wan22LightningModule(pl.LightningModule):
    def __init__(
        self,
        dit_path: str,
        learning_rate: float = 1e-5,
        train_architecture: str = "lora",
        lora_rank: int = 128,
        lora_alpha: float = 64.0,
        lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2",
        init_lora_weights: str = "kaiming",
        pretrained_lora_path: str | None = None,
    ) -> None:
        super().__init__()
        model_manager = ModelManager(device="cpu")
        dit_paths = [path.strip() for path in dit_path.split(",") if path.strip()]
        model_manager.load_models(
            [dit_paths], torch_dtype=torch.bfloat16, device="cpu"
        )
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        dit = self.pipe.denoising_model()
        self.audio_proj = AudioPack(10752, (4, 1, 1), 32, layernorm=True)
        self.audio_cond_projs = nn.ModuleList(
            nn.Linear(32, dit.dim) for _ in range(len(dit.blocks) // 2 - 1)
        )
        self.patch_embedding = nn.Conv3d(
            dit.patch_embedding.in_channels,
            dit.dim,
            kernel_size=dit.patch_size,
            stride=dit.patch_size,
        )

        self.pipe.requires_grad_(False)
        if train_architecture == "lora":
            self._add_lora(
                dit,
                rank=lora_rank,
                alpha=lora_alpha,
                target_modules=lora_target_modules,
                init_method=init_lora_weights,
            )
        else:
            dit.requires_grad_(True)

        dit.train()
        self.audio_proj.train()
        self.audio_cond_projs.train()
        self.patch_embedding.train()
        self.learning_rate = learning_rate

        if pretrained_lora_path:
            state_dict = load_state_dict(pretrained_lora_path)
            state_dict = _training_state_dict(state_dict, set(self.state_dict()))
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(
                f"Loaded {pretrained_lora_path}: "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys"
            )

    @staticmethod
    def _add_lora(
        model: nn.Module,
        rank: int,
        alpha: float,
        target_modules: str,
        init_method: str,
    ) -> None:
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights=True if init_method == "kaiming" else init_method,
            target_modules=target_modules.split(","),
        )
        inject_adapter_in_model(config, model)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.float()

    @staticmethod
    def _prompt_context(batch: dict[str, Any]) -> torch.Tensor:
        context = batch["prompt_emb"]["context"]
        return context.squeeze(1) if context.ndim == 4 else context

    @staticmethod
    def _audio_embedding(batch: dict[str, Any]) -> torch.Tensor:
        audio = batch["audio_emb"]
        if audio.ndim == 4 and audio.shape[1] == 1:
            audio = audio.squeeze(1)
        if audio.ndim != 3:
            raise ValueError(f"unexpected audio embedding shape: {tuple(audio.shape)}")
        return audio

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        dtype = torch.bfloat16
        latents = batch["latents"].to(self.device, dtype)
        pre_latents = batch["pre_latents"].to(self.device, dtype)
        last_latents = batch["last_latents"].to(self.device, dtype)
        image_lat = batch["image_lat"].to(self.device, dtype)
        prompt_context = self._prompt_context(batch).to(self.device, dtype)
        audio = self._audio_embedding(batch).to(self.device, dtype)

        fixed_frames = int(batch["pre_fix_frames_num"][0].item())
        fixed_latents = (fixed_frames + 3) // 4
        if fixed_latents == -3:
            fixed_latents = -4

        mask = torch.zeros_like(image_lat[:, :1])
        if fixed_latents == 4:
            mask[:, :, fixed_latents:] = 1
        elif fixed_latents == -4:
            mask[:, :, :fixed_latents] = 1
        else:
            mask.fill_(1)
        image_condition = torch.cat([image_lat, mask], dim=1)

        audio = audio.permute(0, 2, 1)[:, :, :, None, None]
        audio = torch.cat([audio[:, :, :1].repeat(1, 1, 3, 1, 1), audio], dim=2)
        audio = self.audio_proj(audio)
        audio = torch.cat(
            [projection(audio) for projection in self.audio_cond_projs], dim=0
        )

        noise = torch.randn_like(latents)
        timestep_id = torch.randint(
            0, self.pipe.scheduler.num_train_timesteps, (latents.shape[0],)
        )
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(
            device=self.device, dtype=self.pipe.torch_dtype
        )
        noisy_source = self.pipe.scheduler.add_noise(latents, noise, timestep)
        noisy_latents = latents.clone()

        if fixed_latents == 4:
            noisy_latents[:, :, fixed_latents:, 3:-2, 2:-2] = noisy_source[
                :, :, fixed_latents:, 3:-2, 2:-2
            ]
            noisy_latents[:, :, :fixed_latents] = pre_latents
        elif fixed_latents == -4:
            noisy_latents[:, :, :fixed_latents, 3:-2, 2:-2] = noisy_source[
                :, :, :fixed_latents, 3:-2, 2:-2
            ]
            noisy_latents[:, :, fixed_latents:] = last_latents
        else:
            noisy_latents[:, :, :, 3:-2, 2:-2] = noisy_source[:, :, :, 3:-2, 2:-2]

        target = self.pipe.scheduler.training_target(latents, noise, timestep)
        if fixed_latents == 4:
            target = target[:, :, fixed_latents:, 3:-2, 2:-2]
        elif fixed_latents == -4:
            target = target[:, :, :fixed_latents, 3:-2, 2:-2]
        else:
            target = target[:, :, :, 3:-2, 2:-2]

        latent_height, latent_width = noisy_latents.shape[-2:]
        model_input = self.patch_embedding(
            torch.cat([noisy_latents, image_condition], dim=1)
        )
        prediction = self.pipe.denoising_model()(
            model_input,
            timestep=timestep,
            context=prompt_context,
            y=image_condition,
            audio_emb=audio,
            lat_h=latent_height,
            lat_w=latent_width,
        )
        if fixed_latents == 4:
            prediction = prediction[:, :, fixed_latents:, 3:-2, 2:-2]
        elif fixed_latents == -4:
            prediction = prediction[:, :, :fixed_latents, 3:-2, 2:-2]
        else:
            prediction = prediction[:, :, :, 3:-2, 2:-2]

        loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep).mean()
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        parameters = [
            {
                "params": filter(
                    lambda parameter: parameter.requires_grad,
                    self.pipe.denoising_model().parameters(),
                )
            },
            {"params": self.audio_proj.parameters()},
            {"params": self.audio_cond_projs.parameters()},
            {"params": self.patch_embedding.parameters()},
        ]
        return torch.optim.AdamW(parameters, lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        state_dict = self.state_dict()
        checkpoint.clear()
        checkpoint.update(
            {name: value for name, value in state_dict.items() if name in trainable_names}
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess data and train Jogg-Avatar on Wan2.2-TI2V-5B."
    )
    parser.add_argument("--task", required=True, choices=("data_process", "train"))
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument(
        "--metadata_path",
        help="CSV metadata path; defaults to <dataset_path>/all.csv.",
    )
    parser.add_argument("--output_path", default="./models/wan2.2-5b")
    parser.add_argument("--text_encoder_path")
    parser.add_argument("--vae_path")
    parser.add_argument("--wav2vec_path")
    parser.add_argument(
        "--dit_path", help="Comma-separated Wan2.2-TI2V-5B DiT shards."
    )
    parser.add_argument("--clip_frames", type=int, default=121)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=float, default=64.0)
    parser.add_argument(
        "--lora_target_modules", default="q,k,v,o,ffn.0,ffn.2"
    )
    parser.add_argument(
        "--init_lora_weights", choices=("gaussian", "kaiming"), default="kaiming"
    )
    parser.add_argument(
        "--train_architecture", choices=("lora", "full"), default="lora"
    )
    parser.add_argument("--pretrained_lora_path")
    parser.add_argument(
        "--training_strategy",
        choices=("auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"),
        default="auto",
    )
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _require_paths(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [f"--{name}" for name in names if not getattr(args, name)]
    if missing:
        raise ValueError(f"{args.task} requires {', '.join(missing)}")


def run_data_process(args: argparse.Namespace) -> None:
    _require_paths(args, ("text_encoder_path", "vae_path", "wav2vec_path"))
    dataset = VideoFeatureDataset(
        args.dataset_path,
        args.metadata_path,
        clip_frames=args.clip_frames,
        height=args.height,
        width=args.width,
        overwrite=args.overwrite,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )
    model = FeaturePreprocessor(
        args.text_encoder_path,
        args.vae_path,
        args.wav2vec_path,
        clip_frames=args.clip_frames,
        fps=args.fps,
        sample_rate=args.sample_rate,
    )
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, default_root_dir=args.output_path, logger=False
    )
    trainer.test(model, dataloader)


def run_train(args: argparse.Namespace) -> None:
    _require_paths(args, ("dit_path",))
    dataset = Wan22FeatureDataset(
        args.dataset_path, args.metadata_path, args.steps_per_epoch
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    model = Wan22LightningModule(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        pretrained_lora_path=args.pretrained_lora_path,
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision=args.precision,
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
    )
    trainer.fit(model, dataloader)


def main() -> None:
    args = build_parser().parse_args()
    args.metadata_path = args.metadata_path or os.path.join(
        args.dataset_path, "all.csv"
    )
    pl.seed_everything(args.seed, workers=True)
    if args.clip_frames != 121:
        raise ValueError("Wan2.2-5B training currently requires --clip_frames 121")
    if args.task == "data_process":
        run_data_process(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
