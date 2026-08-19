# Jogg-Avatar V2V 5B

[English](README.md) | [简体中文](README_zh.md)

Jogg-Avatar V2V 是一个基于
[Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) 的音频驱动数字人
视频生成模型。[Jogg-Avatar V2V 权重](https://huggingface.co/cicada-ai/Jogg-Avatar-V2V)
已发布在 Hugging Face。输入源视频与驱动音频后，模型保留原视频中的身体、镜头和背景运动，
并重新生成与音频同步的人脸区域。

本仓库只包含 Wan2.2 5B V2V 的训练、预处理与推理代码，不包含原有 14B I2V 流程。

## 环境要求

- Linux、NVIDIA GPU，以及兼容 CUDA 12.8 的驱动
- `PATH` 中可用的 `ffmpeg`
- [`uv`](https://docs.astral.sh/uv/)
- 项目锁定 Python 3.13、PyTorch 2.8.0 和 CUDA 12.8 wheel

```bash
git clone https://github.com/chanjing-ai/Jogg-Avatar-V2V.git
cd Jogg-Avatar-V2V

uv sync --extra inference
uv sync --extra train --extra test
```

FlashAttention 是推荐的可选推理依赖：

```bash
uv sync --extra build
uv pip install flash-attn==2.8.3 --no-build-isolation
```

## 模型准备

```bash
mkdir -p models
uv run hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir models/Wan2.2-TI2V-5B
uv run hf download facebook/wav2vec2-base-960h \
  --local-dir models/wav2vec2-base-960h
uv run hf download cicada-ai/Jogg-Avatar-V2V \
  --local-dir models/Jogg-Avatar-Wan2.2-5B

# 从 InsightFace 官方模型发布页下载人脸检测模型
curl -L https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip \
  -o /tmp/buffalo_sc.zip
unzip -j /tmp/buffalo_sc.zip det_500m.onnx -d models/scrfd_crop_face
mv models/scrfd_crop_face/det_500m.onnx \
  models/scrfd_crop_face/scrfd_500m_bnkps.onnx
```

默认目录结构如下：

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

可通过环境变量 `JOGG_AVATAR_MODEL_DIR` 修改模型根目录。SCRFD ONNX 只用于生成人脸框
元数据，来源为上面链接的 InsightFace 官方模型发布包。

## 推理

源视频需要清晰可见的人脸，且至少包含 121 帧。首先提取稳定的人脸框：

```bash
uv run python script/extract_mouth_info.py \
  --video_path /path/to/source.mp4 \
  --scrfd_model models/scrfd_crop_face/scrfd_500m_bnkps.onnx
```

在加载 5B 模型前检查所有模型路径和输入：

```bash
uv run python script/inference_wan2_2_v2v.py \
  --config configs/inference_5B.yaml \
  --video_path /path/to/source.mp4 \
  --audio_path /path/to/driving.wav \
  --validate_only
```

单卡推理命令如下。首次运行会自动生成 VAE 缓存，之后直接复用：

```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --standalone --nproc_per_node=1 \
  script/inference_wan2_2_v2v.py \
  --config configs/inference_5B.yaml \
  --video_path /path/to/source.mp4 \
  --audio_path /path/to/driving.wav \
  --output_dir demo_out/example
```

未传入 `--prompt` 时，推理会自动使用与训练一致的固定提示词：
`A realistic video of a face speaking directly to the camera. The camera remains
steady and every facial detail is sharp and clearly visible.`

批量推理使用 `--input_file examples/infer_samples.txt`，每个有效行的格式为
`提示词@@源视频@@驱动音频`。当音频长于源视频时，源视频帧与 reference latent 都会使用
往返播放。多卡运行时，配置中的 `sp_size` 必须等于 `torchrun --nproc_per_node`。

## 训练

元数据 CSV 必须包含 `file_name`，可选 `text`，格式参考 `examples/dataset.csv`。
相对路径以 `--dataset_path` 为根目录。每个样本需要以下文件：

```text
clips/example.mp4
clips/example.wav
clips/example_mouth_info.json
```

训练预处理读取前 242 帧，并生成 `example.mp4.tensors.vae2.2.pth`：

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

训练 LoRA 与音频条件模块：

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

训练 checkpoint 只包含可训练张量。将它转换为 Hugging Face 发布格式：

```bash
uv run python script/export_checkpoint.py \
  outputs/train/lightning_logs/version_0/checkpoints/epoch=0-step=1000.ckpt \
  outputs/Jogg-Avatar-Wan2.2-5B
```

## 致谢

本项目基于 [Wan](https://github.com/Wan-Video/Wan2.2)，并参考了
[OmniAvatar](https://github.com/Omni-Avatar/OmniAvatar) 与
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)。

## 许可证

代码使用 [Apache License 2.0](LICENSE)。Wan2.2、Wav2Vec2、SCRFD 与训练数据各自的
许可证和使用条款同样适用。
