# 视频预处理：输入视频路径 + mouth_info_path，输出 VAE 特征 .tensors.vae2.2.pth
# 人脸框必须由 _mouth_info.json 提供

from __future__ import annotations

import os
import numpy as np
import torch
import imageio
import torchvision.transforms as TT
from PIL import Image
from einops import rearrange
from typing import Dict, List, Optional, Tuple, Union

from .face_detect import get_all_face_bboxes_required


# 与原 v2v 预处理一致：实际输出为 480x480
TARGET_H = 480
TARGET_W = 480
CHUNK_FRAMES = 121
CHUNK_STRIDE = 108  # 与 inference 中 times 的步长一致（L - fixed_frame）


def _safe_crop(frame: np.ndarray, xmin: int, xmax: int, ymin: int, ymax: int) -> np.ndarray:
    """裁剪帧，bbox 越界时用边缘像素补边再裁（支持负坐标和超出帧尺寸）。"""
    h, w = frame.shape[:2]
    pad_l = max(0, -xmin)
    pad_r = max(0, xmax - w)
    pad_t = max(0, -ymin)
    pad_b = max(0, ymax - h)
    if pad_l or pad_r or pad_t or pad_b:
        frame = np.pad(frame, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
        xmin += pad_l
        xmax += pad_l
        ymin += pad_t
        ymax += pad_t
    return frame[ymin:ymax, xmin:xmax, :]


def _match_size(image_sizes: List[Tuple[int, int]], h: int, w: int) -> Tuple[int, int]:
    ratio_, size_ = 9999, 9999
    select = None
    for (sh, sw) in image_sizes:
        ratio_tmp = abs(sh / sw - h / w)
        size_tmp = abs(max(sh, sw) - max(w, h))
        if ratio_tmp < ratio_ or (ratio_tmp == ratio_ and size_tmp < size_):
            ratio_, size_ = ratio_tmp, size_tmp
            select = (sh, sw)
    return select or (TARGET_H, TARGET_W)


def load_and_process_frames(
    video_path: str,
    bboxes: Union[Tuple[int, int, int, int], List[Tuple[int, int, int, int]]],
    height: int = TARGET_H,
    width: int = TARGET_W,
    image_sizes: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]]]:
    """
    读取视频全部帧，按 bbox 裁剪、resize 到 height x width、归一化到 [-1,1]。

    bboxes: 单个 (xmin, xmax, ymin, ymax) 或逐帧列表 [(xmin, xmax, ymin, ymax), ...]
    返回 (C, T, H, W) 和实际使用的逐帧 bbox 列表。
    """
    if image_sizes is None:
        image_sizes = [(height, width)]
    per_frame = isinstance(bboxes, list)
    transform = TT.Compose([TT.ToTensor()])

    reader = imageio.get_reader(video_path)
    frames = []
    used_bboxes: List[Tuple[int, int, int, int]] = []
    try:
        total = reader.count_frames()
        for i in range(total):
            frame_ori = reader.get_data(i)
            if per_frame:
                bb = bboxes[min(i, len(bboxes) - 1)]
            else:
                bb = bboxes
            xmin, xmax, ymin, ymax = bb
            crop = _safe_crop(frame_ori, xmin, xmax, ymin, ymax)
            H, W = crop.shape[0], crop.shape[1]
            pil = Image.fromarray(crop)
            t = transform(pil)
            _match_size(image_sizes, H, W)
            t = TT.Resize(size=[height, width])(t)
            t = t * 2.0 - 1.0
            frames.append(t)
            used_bboxes.append((xmin, xmax, ymin, ymax))
    finally:
        reader.close()

    if not frames:
        raise ValueError(f"Video has no frames: {video_path}")
    frames = torch.stack(frames, dim=0)
    frames = rearrange(frames, "T C H W -> C T H W")
    return frames, used_bboxes


def preprocess_video_to_vae_features(
    video_path: str,
    vae_path: str,
    mouth_info_path: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
    num_frames_chunk: int = CHUNK_FRAMES,
    height: int = TARGET_H,
    width: int = TARGET_W,
    overwrite: bool = False,
) -> str:
    """
    将输入视频预处理为 VAE 特征并保存为 .tensors.vae2.2.pth。
    人脸框必须由 mouth_info_path（_mouth_info.json）提供。

    - video_path: 输入视频路径（如 .mp4）
    - vae_path: Wan2.2 VAE 权重路径（Wan2.2_VAE.pth）
    - mouth_info_path: 必须，_mouth_info.json 路径（含 face_bbox）
    - output_path: 输出 .pth 路径，默认 video_path + ".tensors.vae2.2.pth"
    - device: 运行 VAE 的设备
    - num_frames_chunk: 每段帧数，默认 121
    - height, width: 人脸区域 resize 尺寸，默认 480

    返回保存的 .pth 路径。
    """
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + ".tensors.vae2.2.pth"
    if os.path.isfile(output_path) and not overwrite:
        return output_path

    from ..models.vae2_2 import Wan2_2_VAE

    all_bboxes = get_all_face_bboxes_required(video_path, mouth_info_path)

    frames_tensor, used_bboxes = load_and_process_frames(
        video_path, all_bboxes, height=height, width=width
    )
    # (C, T, H, W)，H=W=480
    T = frames_tensor.shape[1]
    if T < num_frames_chunk:
        raise ValueError(
            f"Video has {T} frames, need at least {num_frames_chunk} frames. "
            f"Please use a longer video: {video_path}"
        )

    vae = Wan2_2_VAE(vae_pth=vae_path, device=device)
    vae.model.eval()

    data = {}
    chunk_stride = CHUNK_STRIDE
    idx = 0
    start = 0
    chunk_starts: List[int] = []

    # Pad video so that stride-aligned chunks cover all frames.
    # Without padding the last chunk starts at T-121 which may not be
    # stride-aligned, causing a temporal offset vs inference chunks.
    last_stride_start = ((T - num_frames_chunk) // chunk_stride) * chunk_stride
    next_stride_start = last_stride_start + chunk_stride
    padded_T = T
    if next_stride_start + num_frames_chunk > T and next_stride_start < T:
        padded_T = next_stride_start + num_frames_chunk
        pad_count = padded_T - T
        last_frame = frames_tensor[:, -1:, :, :]
        frames_tensor = torch.cat(
            [frames_tensor, last_frame.expand(-1, pad_count, -1, -1)], dim=1
        )
        print(f"[preprocess] padded {T} -> {padded_T} frames (+{pad_count}) "
              f"to align VAE chunks with inference stride")

    while start + num_frames_chunk <= padded_T:
        chunk = frames_tensor[:, start : start + num_frames_chunk, :, :]
        chunk = chunk.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            lat_list = vae.encode([chunk])
        lat = lat_list[0].to(device="cpu", dtype=torch.bfloat16)
        data[str(idx)] = lat
        chunk_starts.append(start)
        idx += 1
        start += chunk_stride

    # ---- Reversed VAE chunks for bounce-backward playback ----
    # When audio > video, bounce plays the video backward. Reference latents
    # must also be in reverse temporal order so the model sees the correct
    # motion direction.
    rev_frames = frames_tensor[:, :T, :, :].flip(1)
    if padded_T > T:
        rev_pad = padded_T - T
        rev_last = rev_frames[:, -1:, :, :]
        rev_frames = torch.cat(
            [rev_frames, rev_last.expand(-1, rev_pad, -1, -1)], dim=1
        )
    rev_idx = 0
    rev_start = 0
    rev_chunk_starts: List[int] = []
    while rev_start + num_frames_chunk <= padded_T:
        chunk = rev_frames[:, rev_start : rev_start + num_frames_chunk, :, :]
        chunk = chunk.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            lat_list = vae.encode([chunk])
        lat = lat_list[0].to(device="cpu", dtype=torch.bfloat16)
        data[f"rev_{rev_idx}"] = lat
        rev_chunk_starts.append(rev_start)
        rev_idx += 1
        rev_start += chunk_stride
    del rev_frames
    data["rev_chunk_starts"] = torch.tensor(rev_chunk_starts, dtype=torch.long)
    print(f"[preprocess] encoded {len(rev_chunk_starts)} reversed chunks "
          f"at positions {rev_chunk_starts}")

    coodr = [
        torch.tensor([b[0] for b in used_bboxes], dtype=torch.long),
        torch.tensor([b[1] for b in used_bboxes], dtype=torch.long),
        torch.tensor([b[2] for b in used_bboxes], dtype=torch.long),
        torch.tensor([b[3] for b in used_bboxes], dtype=torch.long),
    ]
    data["chunk_starts"] = torch.tensor(chunk_starts, dtype=torch.long)
    data["coodr"] = coodr
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(data, output_path)
    return output_path
