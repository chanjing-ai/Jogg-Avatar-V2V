# Jogg-Avatar V2V 5B

[English](README.md) | [简体中文](README_zh.md)

Jogg-Avatar V2V is an audio-driven avatar video generation model based on
[Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B). Given a source
video and a driving audio track, it preserves body, camera, and background
motion while regenerating the face region with synchronized speech.

This repository contains only the Wan2.2 5B V2V training, preprocessing, and
inference paths. The original Jogg-Avatar 14B I2V project remains separate.

## Requirements

- Linux with an NVIDIA GPU and CUDA 12.8-compatible driver
- `ffmpeg` available on `PATH`
- [`uv`](https://docs.astral.sh/uv/)
- Python 3.13, PyTorch 2.8.0, and CUDA 12.8 wheels are pinned by the project

```bash
git clone https://github.com/chanjing-ai/Jogg-Avatar-V2V.git
cd Jogg-Avatar-V2V

# Inference and SCRFD face detection
uv sync --extra inference

# Training and tests
uv sync --extra train --extra test
```

FlashAttention is optional but recommended for inference:

```bash
uv sync --extra build
uv pip install flash-attn==2.8.3 --no-build-isolation
```

## Models

Download the base model, audio encoder, and Jogg-Avatar weights:

```bash
mkdir -p models
uv run hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir models/Wan2.2-TI2V-5B
uv run hf download facebook/wav2vec2-base-960h \
  --local-dir models/wav2vec2-base-960h
uv run hf download cicada-ai/Jogg-Avatar-V2V \
  --local-dir models/Jogg-Avatar-Wan2.2-5B
```

The default config expects:

```text
models/
|-- Wan2.2-TI2V-5B/
|   |-- diffusion_pytorch_model-00001-of-00003.safetensors
|   |-- diffusion_pytorch_model-00002-of-00003.safetensors
|   |-- diffusion_pytorch_model-00003-of-00003.safetensors
|   |-- models_t5_umt5-xxl-enc-bf16.pth
|   `-- Wan2.2_VAE.pth
|-- Jogg-Avatar-Wan2.2-5B/
|   |-- config.json
|   `-- diffusion_pytorch_model.safetensors
|-- wav2vec2-base-960h/
`-- scrfd_crop_face/
    `-- scrfd_500m_bnkps.onnx
```

Set `JOGG_AVATAR_MODEL_DIR` to use another model root. The SCRFD ONNX detector
is used only to create face-box metadata and is not included in this repository.

## Inference

The source video must contain a clearly visible face and at least 121 frames.
First extract stabilized face boxes:

```bash
uv run python script/extract_mouth_info.py \
  --video_path /path/to/source.mp4 \
  --scrfd_model models/scrfd_crop_face/scrfd_500m_bnkps.onnx
```

Validate all inputs and model paths without loading the 5B model:

```bash
uv run python script/inference_wan2_2_v2v.py \
  --config configs/inference_5B.yaml \
  --prompt "A person speaking naturally to the camera" \
  --video_path /path/to/source.mp4 \
  --audio_path /path/to/driving.wav \
  --validate_only
```

Run single-GPU inference. The VAE cache is created automatically on the first
run and reused afterward:

```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --standalone --nproc_per_node=1 \
  script/inference_wan2_2_v2v.py \
  --config configs/inference_5B.yaml \
  --prompt "A person speaking naturally to the camera" \
  --video_path /path/to/source.mp4 \
  --audio_path /path/to/driving.wav \
  --output_dir demo_out/example
```

For batch inference, use `--input_file examples/infer_samples.txt`. Each
non-comment line has the format `prompt@@source_video@@driving_audio`. When the
audio is longer than the source video, source frames and reference latents use
bounce playback. Set `sp_size` equal to `torchrun --nproc_per_node` for multiple
GPUs. Memory and speed controls are documented in `configs/inference_5B.yaml`.

## Training

The metadata CSV requires `file_name` and may include `text`; see
`examples/dataset.csv`. Paths may be absolute or relative to `--dataset_path`.
Each sample requires:

```text
clips/example.mp4
clips/example.wav
clips/example_mouth_info.json
```

Training preprocessing reads the first 242 frames and creates
`example.mp4.tensors.vae2.2.pth`:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python examples/wanvideo/train_wan2_2_5b.py \
  --task data_process \
  --dataset_path /path/to/dataset \
  --metadata_path /path/to/dataset/all.csv \
  --text_encoder_path models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth \
  --vae_path models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --wav2vec_path models/wav2vec2-base-960h \
  --output_path outputs/preprocess
```

Train LoRA and audio-conditioning modules:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run python \
  examples/wanvideo/train_wan2_2_5b.py \
  --task train \
  --dataset_path /path/to/dataset \
  --metadata_path /path/to/dataset/all.csv \
  --dit_path models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors,models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors,models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors \
  --output_path outputs/train \
  --max_epochs 100 \
  --learning_rate 1e-4 \
  --lora_rank 128 \
  --lora_alpha 64
```

Checkpoints contain only trainable tensors. Export one to the Hugging Face
layout with:

```bash
uv run python script/export_checkpoint.py \
  outputs/train/lightning_logs/version_0/checkpoints/epoch=0-step=1000.ckpt \
  outputs/Jogg-Avatar-Wan2.2-5B
```

## Model Release

Create a Hugging Face model repository. Copy `huggingface/README.md` into the
exported model directory as `README.md`, then upload the complete directory:

```bash
cp huggingface/README.md outputs/Jogg-Avatar-Wan2.2-5B/README.md
uv run hf upload cicada-ai/Jogg-Avatar-V2V \
  outputs/Jogg-Avatar-Wan2.2-5B .
```

Do not commit checkpoints, private media, dataset manifests, absolute machine
paths, logs, tokens, or generated VAE caches to the Git repository. See
`SECURITY.md` for the release checklist.

## Acknowledgments

This project builds on [Wan](https://github.com/Wan-Video/Wan2.2) and was
informed by [OmniAvatar](https://github.com/Omni-Avatar/OmniAvatar) and
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).

## License

The code is released under the [Apache License 2.0](LICENSE). The licenses and
usage terms of Wan2.2, Wav2Vec2, SCRFD, and any training data also apply.
