import subprocess
import shutil
import torch, os
from safetensors import safe_open
from contextlib import contextmanager

import tempfile
import numpy as np
import imageio
import soundfile as sf
from einops import rearrange
import hashlib
import cv2
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@contextmanager
def init_weights_on_device(device = torch.device("meta"), include_buffers :bool = False):

    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)

def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict

def smart_load_weights(model, ckpt_state_dict):
    model_state_dict = model.state_dict()
    new_state_dict = {}
    # print(model.head.head.weight.shape)
    for name, param in model_state_dict.items():
        # if "head" in name:
        #     print(name)
        if name in ckpt_state_dict:
            ckpt_param = ckpt_state_dict[name]
            if param.shape == ckpt_param.shape:
                new_state_dict[name] = ckpt_param
            else:
                # 自动修剪维度以匹配
                if all(p >= c for p, c in zip(param.shape, ckpt_param.shape)):
                    print(f"[Truncate] {name}: ckpt {ckpt_param.shape} -> model {param.shape}")
                    # 创建新张量，拷贝旧数据
                    new_param = param.clone()
                    slices = tuple(slice(0, s) for s in ckpt_param.shape)
                    new_param[slices] = ckpt_param
                    new_state_dict[name] = new_param
                else:
                    print(f"[Skip] {name}: ckpt {ckpt_param.shape} is larger than model {param.shape}")

    # 更新 state_dict，只更新那些匹配的
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, assign=True, strict=False)
    return model, missing_keys, unexpected_keys

def save_wav(audio, audio_path):
    if isinstance(audio, torch.Tensor):
        audio = audio.float().detach().cpu().numpy()

    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)  # (1, samples)

    sf.write(audio_path, audio.T, 16000)

    return True


# 与预处理一致：人脸区域 480x480
FACE_H, FACE_W = 480, 480


def loop_video_index(index: int, length: int) -> int:
    """Bounce (ping-pong): 0,1,...,N-1,N-2,...,1,0,1,..."""
    if length <= 1:
        return 0
    cycle = 2 * (length - 1)
    pos = index % cycle
    if pos >= length:
        pos = cycle - pos
    return pos


def save_video_with_audio(
    video_batch: torch.Tensor,
    save_path: str,
    fps: float = 25,
    prompt=None,
    prompt_path=None,
    audio_path=None,
    prefix="result",
):
    """Save generated RGB frames and optionally mux a driving audio track."""
    os.makedirs(save_path, exist_ok=True)
    videos = [video_batch] if isinstance(video_batch, list) else list(video_batch)
    output_paths = []
    with tempfile.TemporaryDirectory() as tmp_path:
        for index, video in enumerate(videos):
            chunks = video if isinstance(video, list) else [video]
            suffix = f"_{index:03d}" if len(videos) > 1 else ""
            output_path = os.path.join(save_path, f"{prefix}{suffix}.mp4")
            silent_path = os.path.join(tmp_path, f"{prefix}{suffix}.mp4")
            with imageio.get_writer(silent_path, fps=fps) as writer:
                for chunk in chunks:
                    frames = chunk[0] if chunk.ndim == 5 else chunk
                    for frame in frames:
                        frame = rearrange(frame, "c h w -> h w c")
                        writer.append_data((frame.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
            if audio_path:
                subprocess.run(
                    [
                        "ffmpeg", "-i", silent_path, "-i", audio_path,
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                        "-c:a", "aac", "-shortest", output_path, "-y",
                    ],
                    check=True,
                )
            else:
                shutil.copy2(silent_path, output_path)
            output_paths.append(output_path)
            print(f"Saved result video to: {output_path}")
    if prompt is not None and prompt_path is not None:
        with open(prompt_path, "w") as file:
            file.write(prompt)
    return output_paths


def save_video_as_grid_and_mp4(video_batch: torch.Tensor, ori_video_path: str,coodr: list,save_path: str, fps: float = 5,prompt=None, prompt_path=None, audio=None, audio_path=None, prefix=None):
    os.makedirs(save_path, exist_ok=True)
    out_videos = []

    # with open(ori_json, "r") as file:
    #     data = json.load(file)
    #     x1, x2, y1, y2 = data["1"]["face_bbox"]
    per_frame_bbox = coodr[0].numel() > 1
    reader = imageio.get_reader(ori_video_path)
    first_frame = reader.get_data(0)
    try:
        frame_count = reader.count_frames()
    except Exception:
        meta = reader.get_meta_data()
        frame_count = meta.get("nframes") or float("inf")
    bbox_frame_count = int(coodr[0].numel()) if per_frame_bbox else None
    if frame_count == float("inf"):
        source_frame_count = bbox_frame_count or 1
    elif bbox_frame_count is None:
        source_frame_count = int(frame_count)
    else:
        source_frame_count = max(1, min(int(frame_count), bbox_frame_count))
    # 基础 mask（480x480），每帧按实际 bbox 尺寸 resize
    base_mask = np.zeros([FACE_H, FACE_W, 3], dtype=np.uint8)
    margin = 48
    base_mask[margin:-margin, margin:-margin, :] = 255
    base_mask = cv2.stackBlur(base_mask, (27, 27)).astype(np.float64) / 255

    def _get_bbox(frame_idx):
        if per_frame_bbox:
            idx = min(frame_idx, coodr[0].numel() - 1)
        else:
            idx = 0
        return (int(coodr[0][idx]), int(coodr[1][idx]),
                int(coodr[2][idx]), int(coodr[3][idx]))

    def _compose_frame(frame, face, x1, x2, y1, y2):
        """将生成的 face 贴回 frame 的 (x1:x2, y1:y2) 区域，处理越界。"""
        pad_l = max(0, -x1)
        pad_r = max(0, x2 - frame.shape[1])
        pad_t = max(0, -y1)
        pad_b = max(0, y2 - frame.shape[0])
        need_pad = pad_l > 0 or pad_r > 0 or pad_t > 0 or pad_b > 0

        if need_pad:
            frame = np.pad(frame, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
            ax1, ax2, ay1, ay2 = x1 + pad_l, x2 + pad_l, y1 + pad_t, y2 + pad_t
        else:
            ax1, ax2, ay1, ay2 = x1, x2, y1, y2

        ori_face = frame[ay1:ay2, ax1:ax2, :]
        face_resized = cv2.resize(face, (ax2 - ax1, ay2 - ay1))
        mask_resized = cv2.resize(base_mask, (ax2 - ax1, ay2 - ay1))
        blended = ori_face * (1 - mask_resized) + face_resized * mask_resized
        frame[ay1:ay2, ax1:ax2, :] = blended

        if need_pad:
            end_h = -pad_b if pad_b else None
            end_w = -pad_r if pad_r else None
            frame = frame[pad_t:end_h, pad_l:end_w, :]
        return frame

    def _iter_videos(batch):
        if isinstance(batch, list):
            return [batch]
        return list(batch)

    def _iter_faces(vid):
        if isinstance(vid, list):
            for chunk in vid:
                if chunk.ndim != 5 or chunk.shape[0] != 1:
                    raise ValueError("Chunked video inputs must have shape [1, T, C, H, W].")
                for face in chunk[0]:
                    yield face
            return
        for face in vid:
            yield face

    with tempfile.TemporaryDirectory() as tmp_path:
        for i, vid in enumerate(_iter_videos(video_batch)):
            idex = 1
            if prefix is not None:
                now_save_path = os.path.join(save_path, f"{prefix}_{i:03d}.mp4")
                tmp_save_path = os.path.join(tmp_path, f"{prefix}_{i:03d}.mp4")
            else:
                now_save_path = os.path.join(save_path, f"{i:03d}.mp4")
                tmp_save_path = os.path.join(tmp_path, f"{i:03d}.mp4")
            with imageio.get_writer(tmp_save_path, fps=fps) as writer:
                for j, face in enumerate(_iter_faces(vid)):
                    face = rearrange(face, "c h w -> h w c")
                    face = (255.0 * face).cpu().numpy().astype(np.uint8)
                    # Loop the source clip from the start once audio outlasts it.
                    frame_idx = loop_video_index(idex, source_frame_count)
                    frame = reader.get_data(frame_idx)
                    x1, x2, y1, y2 = _get_bbox(frame_idx)
                    idex = idex + 1

                    frame = _compose_frame(frame, face, x1, x2, y1, y2)
                    writer.append_data(frame)
            shutil.copy2(tmp_save_path, now_save_path)
            final_save_path = now_save_path
            if audio is not None or audio_path is not None:
                if audio is not None:
                    audio_path = os.path.join(tmp_path, f"{i:06d}.mp3")
                    save_wav(audio[i], audio_path)
                muxed_tmp_path = f"{tmp_save_path[:-4]}_wav.mp4"
                final_save_path = f"{now_save_path[:-4]}_wav.mp4"
                subprocess.run(
                    ["ffmpeg", "-i", tmp_save_path, "-i", audio_path, "-v", "quiet",
                     "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                     muxed_tmp_path, "-y"],
                    check=True,
                )
                shutil.copy2(muxed_tmp_path, final_save_path)
                os.remove(now_save_path)
            if prompt is not None and prompt_path is not None:
                with open(prompt_path, "w") as f:
                    f.write(prompt)
            print(f"Saved result video to: {final_save_path}")
            out_videos.append(final_save_path)
        # cap.release()
        reader.close()
    return out_videos

def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()

def split_state_dict_with_prefix(state_dict):
    keys = sorted([key for key in state_dict if isinstance(key, str)])
    prefix_dict = {}
    for key in  keys:
        prefix = key if "." not in key else key.split(".")[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(key)
    state_dicts = []
    for prefix, keys in prefix_dict.items():
        sub_state_dict = {key: state_dict[key] for key in keys}
        state_dicts.append(sub_state_dict)
    return state_dicts

def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str
