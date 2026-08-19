import gc
import math
import os
import random
import sys
from datetime import datetime
from functools import partial

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import librosa
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from Avatar.utils.args_config import parse_args

args = parse_args()
from Avatar.utils.io_utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model

from Avatar.models.model_manager import ModelManager
from Avatar.wan_video import WanVideoPipeline
from Avatar.utils.io_utils import save_video_as_grid_and_mp4
import torch.distributed as dist
import torchvision.transforms as TT
from transformers import Wav2Vec2FeatureExtractor
import torch.nn.functional as F
from Avatar.utils.audio_preprocess import add_silence_to_audio_ffmpeg
from Avatar.distributed.fsdp import shard_model
from Avatar.models.vae2_2 import Wan2_2_VAE
from Avatar.utils.video_preprocess import (
    CHUNK_FRAMES,
    CHUNK_STRIDE,
    preprocess_video_to_vae_features,
)
from Avatar.utils.inference_validation import validate_job, validate_runtime_config
from Avatar.utils.prompts import DEFAULT_PROMPT, resolve_prompt


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def read_from_file(p):
    with open(p, "r") as fin:
        for l in fin:
            yield l.strip()


def loop_video_index(index: int, length: int) -> int:
    """Bounce (ping-pong): 0,1,...,N-1,N-2,...,1,0,1,..."""
    if length <= 1:
        return 0
    cycle = 2 * (length - 1)
    pos = index % cycle
    if pos >= length:
        pos = cycle - pos
    return pos


def _is_bounce_backward(index: int, length: int) -> bool:
    """Return True when *index* falls in the backward half of a bounce cycle."""
    if length <= 1:
        return False
    cycle = 2 * (length - 1)
    return (index % cycle) >= length


def normalize_chunk_starts(chunk_starts, latent_keys):
    if chunk_starts is None:
        return [int(k) * CHUNK_STRIDE for k in latent_keys]
    if isinstance(chunk_starts, torch.Tensor):
        chunk_starts = chunk_starts.tolist()
    return [int(x) for x in chunk_starts]


def select_latent_chunk(source_frame_idx: int, chunk_starts: list[int], chunk_frames: int) -> int:
    covered_pos = None
    covered_start = -1
    for pos, start in enumerate(chunk_starts):
        if start <= source_frame_idx < start + chunk_frames and start >= covered_start:
            covered_pos = pos
            covered_start = start
    if covered_pos is not None:
        return covered_pos

    best_pos = 0
    best_score = None
    for pos, start in enumerate(chunk_starts):
        score = abs(start - source_frame_idx)
        if best_score is None or score < best_score:
            best_score = score
            best_pos = pos
    return best_pos


class WanInferencePipeline(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not torch.cuda.is_available():
            raise RuntimeError("Wan2.2 5B inference requires a CUDA GPU.")
        self.device = torch.device(f"cuda:{args.local_rank}")
        torch.cuda.set_device(self.device)
        # VAE for latent <-> frame encoding/decoding
        self.vae = Wan2_2_VAE(
            vae_pth=args.vae_path,
            device=self.device)
        if args.dtype == 'bf16':
            self.dtype = torch.bfloat16
        elif args.dtype == 'fp16':
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.pipe = self.load_model()
        if args.i2v:
            chained_trainsforms = []
            chained_trainsforms.append(TT.ToTensor())
            self.transform = TT.Compose(chained_trainsforms)
        if args.use_audio:
            from Avatar.models.wav2vec import Wav2VecModel
            self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                args.wav2vec_path, local_files_only=True
            )
            self.audio_encoder = Wav2VecModel.from_pretrained(args.wav2vec_path, local_files_only=True).to(device=self.device)
            self.audio_encoder.feature_extractor._freeze_parameters()

    def load_model(self):
        # Distributed init + model parallel
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl", init_method="env://", device_id=self.device
            )
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=args.sp_size,
            ring_degree=1,
            ulysses_degree=args.sp_size,
        )
        torch.cuda.set_device(self.device)

        ckpt_path = getattr(args, "lora_ckpt_path", None)
        if isinstance(ckpt_path, str):
            ckpt_path = ckpt_path.strip()
        if not ckpt_path:
            raise ValueError("lora_ckpt_path must point to the Jogg-Avatar 5B checkpoint.")

        if args.train_architecture == 'lora':
            args.pretrained_lora_path = pretrained_lora_path = ckpt_path
        else:
            resume_path = ckpt_path

        self.step = 0

        # Load base models (DiT + text encoder)
        model_manager = ModelManager(device="cpu", infer=True)
        dit_paths = [p.strip() for p in args.dit_path.split(",") if p.strip()]
        model_manager.load_models(
            [
                dit_paths,
                args.text_encoder_path.strip() if isinstance(args.text_encoder_path, str) else args.text_encoder_path,
                # args.vae_path
            ],
            torch_dtype=self.dtype,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
            device='cpu',
        )

        # Build inference pipeline
        pipe = WanVideoPipeline.from_model_manager(model_manager,
                                                   torch_dtype=self.dtype,
                                                   device=str(self.device),
                                                   use_usp=True if args.sp_size > 1 else False,
                                                   infer=True)
        if args.train_architecture == "lora":
            print(f'Use LoRA: lora rank: {args.lora_rank}, lora alpha: {args.lora_alpha}')
            self.add_lora_to_model(
                pipe.denoising_model(),
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_target_modules=args.lora_target_modules,
                init_lora_weights=args.init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            resume_paths = [resume_path] if isinstance(resume_path, str) else resume_path
            for path in resume_paths:
                missing_keys, unexpected_keys = pipe.denoising_model().load_state_dict(load_state_dict(path), strict=False)
                print(f"Loaded {path}: {len(missing_keys)} missing keys, {len(unexpected_keys)} unexpected keys")
        pipe.requires_grad_(False)
        pipe.eval()
        pipe.enable_vram_management(num_persistent_param_in_dit=args.num_persistent_param_in_dit)  # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.
        if args.use_fsdp:
            shard_fn = partial(shard_model, device_id=self.device)
            pipe.dit = shard_fn(pipe.dit)
        return pipe

    def smart_load_weights(self, model, ckpt_state_dict):
        model_state_dict = model.state_dict()
        new_state_dict = {}

        for name, param in model_state_dict.items():
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

                    elif all(m <= c for m, c in zip(param.shape, ckpt_param.shape)):
                        # 检查点权重较大 - 截断加载
                        # 创建切片来截取检查点权重的一部分
                        slices = tuple(slice(0, s) for s in param.shape)
                        partial_param = ckpt_param[slices].clone()
                        new_state_dict[name] = partial_param

                    else:
                        print("skip detect")
                        print(f"[Skip] {name}: ckpt {ckpt_param.shape} is larger than model {param.shape}")

        # 更新 state_dict，只更新那些匹配的
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, assign=True, strict=False)
        return model, missing_keys, unexpected_keys

    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            state_dict_new = {}
            for key in state_dict.keys():
                if 'pipe.dit.' in key:
                    # print(key)
                    key_new = key.split("pipe.dit.")[1]
                    state_dict_new[key_new] = state_dict[key]
                else:
                    # if "audio_proj" in key or "audio_cond_projs" in key or "patch" in key:
                    # print(key)
                    state_dict_new[key] = state_dict[key]
            missing_keys, unexpected_keys = model.load_state_dict(state_dict_new, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")

    def masks_like(tensor, zero=False, generator=None, p=0.2):
        assert isinstance(tensor, list)
        out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

        out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

        if zero:
            if generator is not None:
                for u, v in zip(out1, out2):
                    random_num = torch.rand(
                        1, generator=generator, device=generator.device).item()
                    if random_num < p:
                        u[:, 0] = torch.normal(
                            mean=-3.5,
                            std=0.5,
                            size=(1,),
                            device=u.device,
                            generator=generator).expand_as(u[:, 0]).exp()
                        v[:, 0] = torch.zeros_like(v[:, 0])
                    else:
                        u[:, 0] = u[:, 0]
                        v[:, 0] = v[:, 0]
            else:
                for u, v in zip(out1, out2):
                    u[:, 0] = torch.zeros_like(u[:, 0])
                    v[:, 0] = torch.zeros_like(v[:, 0])

        return out1, out2

    def forward(self, prompt,
                ori_video_path=None,
                audio_path=None,
                mouth_info_path=None,
                latent_path=None,
                seq_len=101,  # not used while audio_path is not None
                height=720,
                width=720,
                overlap_frame=None,
                num_steps=None,
                negative_prompt=None,
                guidance_scale=None,
                audio_scale=None):
        overlap_frame = overlap_frame if overlap_frame is not None else self.args.overlap_frame
        num_steps = num_steps if num_steps is not None else self.args.num_steps
        negative_prompt = negative_prompt if negative_prompt is not None else self.args.negative_prompt
        guidance_scale = guidance_scale if guidance_scale is not None else self.args.guidance_scale
        audio_scale = audio_scale if audio_scale is not None else self.args.audio_scale

        L = 121
        T = (L + 3) // 4  # latent frames

        if self.args.i2v:
            if self.args.random_prefix_frames:
                fixed_frame = overlap_frame
                assert fixed_frame % 4 == 1
            else:
                fixed_frame = 1
            prefix_lat_frame = (3 + fixed_frame) // 4
            first_fixed_frame = 1
        else:
            fixed_frame = 0
            prefix_lat_frame = 0
            first_fixed_frame = 0

        # Audio -> wav2vec embeddings. Keep the full embedding cache on CPU and
        # only move the active chunk to GPU to avoid long-video VRAM buildup.
        if audio_path is not None and self.args.use_audio:
            self.audio_encoder.to(self.device)
            audio, sr = librosa.load(audio_path, sr=self.args.sample_rate)
            input_values = np.squeeze(
                self.wav_feature_extractor(audio, sampling_rate=16000).input_values
            )
            input_values = torch.from_numpy(input_values).float()
            ori_audio_len = audio_len = math.ceil(len(input_values) / self.args.sample_rate * self.args.fps)
            input_values = input_values.unsqueeze(0)
            # padding audio
            if audio_len < L - first_fixed_frame:
                audio_len = audio_len + ((L - first_fixed_frame) - audio_len % (L - first_fixed_frame))
            elif (audio_len - (L - first_fixed_frame)) % (L - fixed_frame) != 0:
                audio_len = audio_len + ((L - fixed_frame) - (audio_len - (L - first_fixed_frame)) % (L - fixed_frame))
            input_values = F.pad(input_values, (0, audio_len * int(self.args.sample_rate / self.args.fps) - input_values.shape[1]), mode='constant', value=0)
            input_values = input_values.to(device=self.device)

            with torch.no_grad():
                hidden_states = self.audio_encoder(input_values, seq_len=audio_len, output_hidden_states=True)
                audio_state_chunks = [
                    hidden_states.last_hidden_state.squeeze(0).detach().to(device="cpu", dtype=torch.float32)
                ]
                for mid_hidden_states in hidden_states.hidden_states:
                    audio_state_chunks.append(
                        mid_hidden_states.squeeze(0).detach().to(device="cpu", dtype=torch.float32)
                    )
                audio_embeddings = torch.cat(audio_state_chunks, dim=-1)
            del hidden_states
            del audio_state_chunks
            del input_values
            self.audio_encoder.to("cpu")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            seq_len = audio_len
            audio_prefix = torch.zeros_like(audio_embeddings[:first_fixed_frame])
        else:
            audio_embeddings = None
            ori_audio_len = seq_len

        # loop
        times = (seq_len - L + first_fixed_frame) // (L - fixed_frame) + 1
        if times * (L - fixed_frame) + fixed_frame < seq_len:
            times += 1
        video_chunks = []
        video_frame_count = 0
        image_emb = {}
        img_lat = None

        # Load (or build) VAE latents for the source video
        default_stem = os.path.splitext(ori_video_path)[0]
        safetensor_path = latent_path or default_stem + ".tensors.vae2.2.pth"
        if not os.path.isfile(safetensor_path):
            mouth_info_path = mouth_info_path or default_stem + "_mouth_info.json"
            if dist.get_rank() == 0:
                preprocess_video_to_vae_features(
                    ori_video_path,
                    vae_path=args.vae_path,
                    mouth_info_path=mouth_info_path,
                    output_path=safetensor_path,
                    device=str(self.device),
                )
            dist.barrier()
        safetensor_data = torch.load(safetensor_path, weights_only=True, map_location="cpu")
        latent_keys = sorted([int(k) for k in safetensor_data.keys() if k.isdigit()])
        if not latent_keys:
            raise ValueError(f"No VAE latent chunks found in {safetensor_path}")
        coodr = safetensor_data["coodr"]
        source_frame_count = int(coodr[0].numel()) if coodr and coodr[0].numel() > 0 else 0
        chunk_starts = normalize_chunk_starts(safetensor_data.get("chunk_starts"), latent_keys)

        def _load_rev_chunks(sd):
            rkeys = sorted(
                [int(k.split("_")[1]) for k in sd.keys()
                 if k.startswith("rev_") and k != "rev_chunk_starts"]
            )
            rst = sd.get("rev_chunk_starts")
            if rst is not None:
                rcs = normalize_chunk_starts(rst, rkeys)
            else:
                rcs = []
            return rkeys, rcs

        rev_latent_keys, rev_chunk_starts = _load_rev_chunks(safetensor_data)

        def _expected_stride_starts(n_frames, chunk_sz, stride):
            starts = list(range(0, n_frames - chunk_sz + 1, stride))
            nxt = (starts[-1] + stride) if starts else 0
            if nxt < n_frames and nxt + chunk_sz > n_frames:
                starts.append(nxt)
            return starts

        expected_starts = _expected_stride_starts(source_frame_count, CHUNK_FRAMES, CHUNK_STRIDE)
        cache_incomplete = (
            source_frame_count >= CHUNK_FRAMES and (
                not chunk_starts or
                len(chunk_starts) != len(latent_keys) or
                not all(s in chunk_starts for s in expected_starts) or
                not rev_chunk_starts
            )
        )
        if cache_incomplete:
            mouth_info_path = mouth_info_path or default_stem + "_mouth_info.json"
            if dist.get_rank() == 0:
                preprocess_video_to_vae_features(
                    ori_video_path,
                    vae_path=args.vae_path,
                    mouth_info_path=mouth_info_path,
                    output_path=safetensor_path,
                    device=str(self.device),
                    overwrite=True,
                )
            dist.barrier()
            safetensor_data = torch.load(safetensor_path, weights_only=True, map_location="cpu")
            latent_keys = sorted([int(k) for k in safetensor_data.keys() if k.isdigit()])
            coodr = safetensor_data["coodr"]
            source_frame_count = int(coodr[0].numel()) if coodr and coodr[0].numel() > 0 else 0
            chunk_starts = normalize_chunk_starts(safetensor_data.get("chunk_starts"), latent_keys)
            rev_latent_keys, rev_chunk_starts = _load_rev_chunks(safetensor_data)
        # 段间重叠 fixed_frame 帧线性混合，减轻接缝（fixed_frame==0 时不混合）
        video_cpu_dtype = torch.float16
        mix_ratio_cpu = (
            torch.linspace(0, 1, steps=fixed_frame, dtype=video_cpu_dtype).view(1, -1, 1, 1, 1)
            if fixed_frame > 0 else None
        )
        # Initialize image embedding with first latent segment
        if args.i2v:
            image = None
            img_lat = safetensor_data["0"].unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)
            msk = torch.zeros_like(img_lat[:, :1])
            image_cat = img_lat
            msk[:, :, 1:, 3:-3, 3:-3] = 1  # 首帧不 mask，从第 1 帧开始 inpainting
            image_emb["y"] = torch.cat([image_cat, msk], dim=1)
        for t in range(times):
            print(f"[{t + 1}/{times}]")
            audio_emb = {}
            if t == 0:
                overlap = first_fixed_frame
            else:
                overlap = fixed_frame
                tdim = image_emb["y"].shape[2]
                h = image_emb["y"].shape[-2]
                w = image_emb["y"].shape[-1]
                msk = torch.zeros((1, 1, tdim, h, w), device=image_emb["y"].device, dtype=image_emb["y"].dtype)
                if h > 6 and w > 6:
                    msk[:, :, 4:, 3:-3, 3:-3] = 1
                else:
                    msk[:, :, 4:, :, :] = 1
                image_emb["y"][:, -1:, :, :, :] = msk
                source_frame_idx = loop_video_index(video_frame_count + 1, source_frame_count)
                backward = _is_bounce_backward(video_frame_count + 1, source_frame_count)

                if backward and rev_chunk_starts:
                    rev_frame = (source_frame_count - 1) - source_frame_idx
                    chunk_pos = select_latent_chunk(rev_frame, rev_chunk_starts, CHUNK_FRAMES)
                    ref_key = f"rev_{rev_latent_keys[chunk_pos]}"
                else:
                    chunk_pos = select_latent_chunk(source_frame_idx, chunk_starts, CHUNK_FRAMES)
                    ref_key = str(latent_keys[chunk_pos])
                try:
                    ref_latent = safetensor_data[ref_key]
                    image_emb["y"][:, :-1, :] = ref_latent.unsqueeze(0)
                except (KeyError, TypeError):
                    return None, None

            prefix_overlap = (3 + overlap) // 4
            if audio_embeddings is not None:
                if t == 0:
                    audio_tensor = audio_embeddings[
                        :min(L - overlap, audio_embeddings.shape[0])
                    ]
                else:
                    audio_start = L - first_fixed_frame + (t - 1) * (L - overlap)
                    audio_tensor = audio_embeddings[
                        audio_start: min(audio_start + L - overlap, audio_embeddings.shape[0])
                    ]

                audio_tensor = torch.cat([audio_prefix, audio_tensor], dim=0)
                audio_prefix = audio_tensor[-fixed_frame:].clone() if fixed_frame > 0 else audio_tensor[:0]
                audio_tensor = audio_tensor.unsqueeze(0).to(device=self.device, dtype=self.dtype, non_blocking=True)
                audio_emb["audio_emb"] = audio_tensor
            else:
                audio_prefix = None

            if img_lat is None:
                img_lat_pre = self.vae.encode([image.squeeze(0).to(dtype=self.dtype)])[0].to(device=self.device, dtype=torch.bfloat16)
                image_emb["y"][:, :-1, :prefix_overlap] = img_lat_pre
                img_lat = image_emb["y"][:, :-1, :]
            latents = self.pipe.log_video(img_lat, prompt, prefix_overlap, image_emb, audio_emb,
                                          negative_prompt, num_inference_steps=num_steps,
                                          cfg_scale=guidance_scale, audio_cfg_scale=audio_scale if audio_scale is not None else guidance_scale,
                                          return_latent=True,
                                          tea_cache_l1_thresh=args.tea_cache_l1_thresh, tea_cache_model_id="Wan2.2-TI2V-5B",
                                          feature_caching=getattr(args, "feature_caching", "Tea"))
            latents = latents.squeeze(0)
            self.pipe.load_models_to_device([])
            gc.collect()
            torch.cuda.empty_cache()
            frames = self.vae.decode([latents])[0]
            frames = (frames.unsqueeze(0).permute(0, 2, 1, 3, 4) + 1) / 2
            frames = frames.clamp_(0, 1)
            img_lat = None
            if fixed_frame > 0:
                image = (frames[:, -fixed_frame:] * 2 - 1).permute(0, 2, 1, 3, 4).contiguous()
            frames_cpu = frames.detach().to(device="cpu", dtype=video_cpu_dtype)

            if t == 0:
                new_chunk = frames_cpu[:, 1:]
            else:
                if fixed_frame > 0 and mix_ratio_cpu is not None and video_chunks:
                    blend_count = min(fixed_frame, video_chunks[-1].shape[1], frames_cpu.shape[1])
                    if blend_count > 0:
                        blend_ratio = mix_ratio_cpu[:, :blend_count]
                        video_chunks[-1][:, -blend_count:] = (
                            video_chunks[-1][:, -blend_count:] * (1 - blend_ratio) +
                            frames_cpu[:, :blend_count] * blend_ratio
                        )
                    new_chunk = frames_cpu[:, fixed_frame:]
                else:
                    new_chunk = frames_cpu[:, overlap:]

            if new_chunk.numel() > 0 and video_frame_count < ori_audio_len:
                remaining = ori_audio_len - video_frame_count
                if new_chunk.shape[1] > remaining:
                    new_chunk = new_chunk[:, :remaining]
                if new_chunk.shape[1] > 0:
                    video_chunks.append(new_chunk.contiguous())
                    video_frame_count += new_chunk.shape[1]

            if "audio_emb" in audio_emb:
                del audio_emb["audio_emb"]
            del latents
            del frames
            del frames_cpu
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if video_frame_count >= ori_audio_len:
                break

        return video_chunks, coodr


def _input_jobs():
    if args.prompt:
        lines = [args.prompt]
        input_name = "prompt"
    elif args.input_file:
        if not os.path.isfile(args.input_file):
            raise FileNotFoundError(f"Prompt file not found: {args.input_file}")
        lines = read_from_file(args.input_file)
        input_name = os.path.splitext(os.path.basename(args.input_file))[0]
    elif args.video_path:
        lines = [DEFAULT_PROMPT]
        input_name = "prompt"
    else:
        raise ValueError("Provide --video_path, --prompt, or --input_file.")

    jobs = []
    video_exts = (".mp4", ".avi", ".mov", ".mkv")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("@@")
        if len(fields) > 3:
            raise ValueError("Each input line must use prompt@@video@@audio format.")
        prompt = resolve_prompt(fields[0])
        video_path = args.video_path
        if not video_path and len(fields) >= 2 and fields[1].lower().endswith(video_exts):
            video_path = fields[1]
        video_path = video_path or getattr(args, "ori_video_path", None)
        audio_path = args.audio_path or (fields[2] if len(fields) == 3 else None)
        if not video_path:
            raise ValueError("A source video is required via --video_path or the input file.")
        mouth_info_path, latent_path = validate_job(
            video_path,
            audio_path,
            args.use_audio,
            args.mouth_info_path,
            args.latent_path,
        )
        jobs.append((prompt, video_path, audio_path, mouth_info_path, latent_path))
    if not jobs:
        raise ValueError("No inference jobs found.")
    return input_name, jobs


def main():
    set_seed(args.seed)
    validate_runtime_config(args)
    input_name, jobs = _input_jobs()
    if args.validate_only:
        print(f"Validation passed for {len(jobs)} inference job(s).")
        return
    exp_name = os.path.basename(args.exp_path)
    seq_len = args.seq_len
    date_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Text-to-video
    inferpipe = WanInferencePipeline(args)
    if args.sp_size > 1:
        date_name = inferpipe.pipe.sp_group.broadcast_object_list([date_name])
        date_name = date_name[0]
    if getattr(args, "output_dir", None):
        output_dir = args.output_dir
    else:
        output_dir = f'demo_out/{exp_name}/res_{input_name}_' \
                     f'seed{args.seed}_step{args.num_steps}_cfg{args.guidance_scale}_' \
                     f'ovlp{args.overlap_frame}_{args.max_tokens}_{args.fps}_{date_name}'
        if args.tea_cache_l1_thresh > 0:
            output_dir = f'{output_dir}_tea{args.tea_cache_l1_thresh}'
        fc = getattr(args, "feature_caching", "Tea")
        if (fc or "").lower() == "custom":
            output_dir = f'{output_dir}_custom'
        if args.audio_scale is not None:
            output_dir = f'{output_dir}_acfg{args.audio_scale}'
        if args.max_hw == 1280:
            output_dir = f'{output_dir}_720p'
    for idx, job in tqdm(enumerate(jobs), total=len(jobs)):
        text, ori_video_path, audio_path, mouth_info_path, latent_path = job
        audio_dir = output_dir + '/audio'
        os.makedirs(audio_dir, exist_ok=True)
        if args.silence_duration_s > 0 and audio_path:
            input_audio_path = os.path.join(audio_dir, f"audio_input_{idx:03d}.wav")
        else:
            input_audio_path = audio_path
        prompt_dir = output_dir + '/prompt'
        os.makedirs(prompt_dir, exist_ok=True)
        if dist.get_rank() == 0 and args.silence_duration_s > 0 and audio_path:
            add_silence_to_audio_ffmpeg(audio_path, input_audio_path, args.silence_duration_s)
        dist.barrier()
        video, coodr = inferpipe(
            prompt=text,
            ori_video_path=ori_video_path,
            audio_path=input_audio_path,
            mouth_info_path=mouth_info_path,
            latent_path=latent_path,
            seq_len=seq_len
        )
        if video is None:
            continue
        tmp2_audio_path = os.path.join(audio_dir, f"audio_out_{idx:03d}.wav")
        prompt_path = os.path.join(prompt_dir, f"prompt_{idx:03d}.txt")
        if dist.get_rank() == 0:
            if audio_path and args.use_audio:
                if args.silence_duration_s > 0:
                    add_silence_to_audio_ffmpeg(
                        audio_path, tmp2_audio_path, args.silence_duration_s
                    )
                else:
                    tmp2_audio_path = audio_path
            result_prefix = getattr(args, "result_prefix", None)
            if result_prefix is None:
                result_prefix = f'result_{idx:03d}'
            elif idx > 0:
                result_prefix = f'{result_prefix}_{idx:03d}'
            save_video_as_grid_and_mp4(video,
                                       ori_video_path,
                                       coodr,
                                       output_dir,
                                       args.fps,
                                       prompt=text,
                                       prompt_path=prompt_path,
                                       audio_path=tmp2_audio_path if (args.use_audio and audio_path) else None,
                                       prefix=result_prefix)
        dist.barrier()


class NoPrint:
    def write(self, x):
        pass

    def flush(self):
        pass


if __name__ == '__main__':
    if not args.debug:
        if args.local_rank != 0:  # 屏蔽除0外的输出
            sys.stdout = NoPrint()
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
