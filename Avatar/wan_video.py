import types
from .models.model_manager import ModelManager
from .models.wan_video_dit import WanModel
from .models.wan_video_text_encoder import WanTextEncoder
from .models.wan_video_vae import WanVideoVAE
from .schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from .prompters import WanPrompter
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from .vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from .models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from .models.wan_video_dit import RMSNorm
from .models.wan_video_vae import RMS_norm, CausalConv3d, Upsample


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae', 'image_encoder']
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False
        self.sp_size = 1


    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        # dtype = next(iter(self.vae.parameters())).dtype
        # enable_vram_management(
        #     self.vae,
        #     module_map = {
        #         torch.nn.Linear: AutoWrappedLinear,
        #         torch.nn.Conv2d: AutoWrappedModule,
        #         RMS_norm: AutoWrappedModule,
        #         CausalConv3d: AutoWrappedModule,
        #         Upsample: AutoWrappedModule,
        #         torch.nn.SiLU: AutoWrappedModule,
        #         torch.nn.Dropout: AutoWrappedModule,
        #     },
        #     module_config = dict(
        #         offload_dtype=dtype,
        #         offload_device="cpu",
        #         onload_dtype=dtype,
        #         onload_device=self.device,
        #         computation_dtype=self.torch_dtype,
        #         computation_device=self.device,
        #     ),
        # )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()


    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, use_usp=False, infer=False):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size, get_sp_group
            from Avatar.distributed.xdit_context_parallel import usp_attn_forward
            for block in pipe.dit.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            pipe.sp_size = get_sequence_parallel_world_size()
            pipe.use_unified_sequence_parallel = True
            pipe.sp_group = get_sp_group()
        return pipe


    def denoising_model(self):
        return self.dit


    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}


    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]

        vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device)[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}


    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames


    def prepare_extra_input(self, latents=None):
        return {}


    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents


    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames


    def prepare_unified_sequence_parallel(self):
        return {"use_unified_sequence_parallel": self.use_unified_sequence_parallel}


    @torch.no_grad()
    def log_video(
        self,
        lat,
        prompt,
        fixed_frame=0, # lat frames
        image_emb={},
        audio_emb={},
        negative_prompt="",
        cfg_scale=5.0,
        audio_cfg_scale=5.0,
        num_inference_steps=50,
        denoising_strength=1.0,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        feature_caching="Tea",
        progress_bar_cmd=tqdm,
        return_latent=False,
    ):
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # latents = lat.clone()
        # temp = lat.clone()
        latents = torch.randn_like(lat)

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Extra input
        extra_input = self.prepare_extra_input(latents)

        if self.sp_size > 1:
            latents = self.sp_group.broadcast(latents)

        # Feature caching: Tea = TeaCache, Custom = CustomCache (Tea + Taylor 补偿, 见 LightX2V cache_source.md)
        cache_cls = CustomCache if (feature_caching or "").lower() == "custom" else TeaCache
        make_cache = lambda: cache_cls(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None and tea_cache_l1_thresh > 0 else None
        tea_cache_posi = {"tea_cache": make_cache()}
        tea_cache_nega = {"tea_cache": make_cache()}
        for tc in (tea_cache_posi["tea_cache"], tea_cache_nega["tea_cache"]):
            if tc is not None:
                tc.reset()

        # Unified Sequence Parallel
        usp_kwargs = self.prepare_unified_sequence_parallel()
        # Denoise
        self.load_models_to_device(["dit"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps, disable=self.sp_size > 1 and torch.distributed.get_rank() != 0)):
            # fixed_frame: I2V overlap (pixel frames). 1 = keep first frame as ref; 13 = overlap 13 frames between chunks. Passed from inference script.
            # if fixed_frame > 0: # latents[:, :, :fixed_frame] = lat[:, :, :fixed_frame]
            # fixed_frame=0
            # if fixed_frame == 1:
            #    temp[:, :, :,3:-2,2:-2] = latents[:, :, :,3:-2,2:-2]
            #    latents = temp.clone()
            # else:
            #     temp[:, :, 4:,3:-2,2:-2] = latents[:, :, 4:,3:-2,2:-2]
            #     latents = temp.clone()
            # if fixed_frame == 0:
            #    temp[:, :, :,3:-2,2:-2] = latents[:, :, :,3:-2,2:-2]
            #    latents = temp.clone()
            # else:
            #     temp[:, :, :-4,3:-2,2:-2] = latents[:, :, :-4,3:-2,2:-2]
            #     latents = temp.clone()

            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference

            noise_pred_posi = self.dit(latents, timestep=timestep, **prompt_emb_posi, **image_emb, **audio_emb, **tea_cache_posi, **extra_input)
            if cfg_scale != 1.0:
                audio_emb_uc = {}
                for key in audio_emb.keys():
                    audio_emb_uc[key] = torch.zeros_like(audio_emb[key])
                if audio_cfg_scale == cfg_scale:
                    noise_pred_nega = self.dit(latents, timestep=timestep, **prompt_emb_nega, **image_emb, **audio_emb_uc, **tea_cache_nega, **extra_input)
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    tea_cache_nega_audio = {"tea_cache": make_cache()}
                    audio_noise_pred_nega = self.dit(latents, timestep=timestep, **prompt_emb_posi, **image_emb, **audio_emb_uc, **tea_cache_nega_audio, **extra_input)
                    text_noise_pred_nega = self.dit(latents, timestep=timestep, **prompt_emb_nega, **image_emb, **audio_emb_uc, **tea_cache_nega, **extra_input)
                    noise_pred = text_noise_pred_nega + cfg_scale * (audio_noise_pred_nega - text_noise_pred_nega) + audio_cfg_scale * (noise_pred_posi - audio_noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

        # if fixed_frame > 0: # new
            # latents[:, :, :fixed_frame] = lat[:, :, :fixed_frame]
        # if fixed_frame == 1:
        #     # lat[:, :, :,3:-2,2:-2] = latents[:, :, :,3:-2,2:-2]
        #     lat[:, :, :,3:-2,2:-2] = latents[:, :, :,3:-2,2:-2]
        # else:
        #     # lat[:, :, 4:,3:-2,2:-2] = latents[:, :, 4:,3:-2,2:-2]
        #     lat[:, :, 4:,3:-2,2:-2] = latents[:, :, 4:,3:-2,2:-2]
        # if fixed_frame == 0:
        #     lat[:, :, :,3:-2,2:-2] = latents[:, :, :,3:-2,2:-2]
        # else:
        #     lat[:, :, :-4,3:-2,2:-2] = latents[:, :, :-4,3:-2,2:-2]

        return latents
        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, **tiler_kwargs)
        recons = self.decode_video(lat, **tiler_kwargs)
        self.load_models_to_device([])
        frames = (frames.permute(0, 2, 1, 3, 4).float() + 1) / 2
        recons = (recons.permute(0, 2, 1, 3, 4).float() + 1) / 2
        if return_latent:
            return frames, recons, latents
        return frames, recons




class TeaCache:
    """Timestep Embedding Aware Cache. 当累计相对 L1 距离未超过阈值则复用缓存，达到阈值则完整计算并清零."""

    def __init__(self, num_inference_steps, rel_l1_thresh, model_id, use_residual_caching=True):
        self.num_inference_steps = num_inference_steps
        self.rel_l1_thresh = rel_l1_thresh
        self.model_id = model_id
        self.use_residual_caching = use_residual_caching  # False: 只存上一轮输出，省显存；True: 残差补偿
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
            "Wan2.2-TI2V-5B": [1.57472669e+05, -1.15702395e+05, 3.10761669e+04, -3.83116651e+03, 2.21608777e+02, -4.81179567e+00],
        }
        if model_id not in self.coefficients_dict:
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Choose from: {list(self.coefficients_dict)}")
        self.coefficients = self.coefficients_dict[model_id]
        # Wan2.2 6-term 多项式在 rel_change 较小时易震荡出负值，可直接用 rel_change 累加
        self.use_raw_rel_change = model_id == "Wan2.2-TI2V-5B"
        self.reset()

    def reset(self):
        """推理开始前显式调用，避免复用实例时 step 错位."""
        self.step = 0
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.previous_hidden_states = None

    def check(self, dit: WanModel, x, t_mod):
        """
        返回 True 表示使用缓存（跳过完整计算），False 表示需要完整计算.
        t_mod: 当前步的 Timestep Embedding（调制后的时间嵌入），与上一步对比算相对 L1 变化.
        """
        modulated_inp = t_mod.detach()
        if self.step == 0:
            should_calc = True
            self.accumulated_rel_l1_distance = 0.0
        else:
            # 相对 L1：diff/base，对整张量 .mean() 得到标量（与官方/LightX2V 一致）
            diff = (modulated_inp - self.previous_modulated_input).abs().mean()
            base = self.previous_modulated_input.abs().mean() + 1e-8
            rel_change = (diff / base).cpu().item()
            if self.use_raw_rel_change:
                delta = rel_change
            else:
                rescale_func = np.poly1d(self.coefficients)  # 系数从高阶到低阶
                delta = float(rescale_func(rel_change))
                delta = max(0.0, delta)  # 多项式在 rel_change 极小时可能震荡为负
            self.accumulated_rel_l1_distance += delta
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = modulated_inp.clone()
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc and self.use_residual_caching:
            self.previous_hidden_states = x.clone()
        return not should_calc  # True => 使用缓存(skip), False => 完整计算

    def store(self, hidden_states):
        """完整计算后调用：存残差（残差模式）或仅存输出（省显存模式）."""
        if self.use_residual_caching and self.previous_hidden_states is not None:
            self.previous_residual = hidden_states - self.previous_hidden_states
            self.previous_hidden_states = None
        else:
            self.previous_hidden_states = hidden_states.clone()

    def update(self, hidden_states):
        """命中缓存时调用：返回上一轮 block 输出（残差模式：input + residual；非残差：直接上一轮输出）."""
        if self.use_residual_caching and self.previous_residual is not None:
            return hidden_states + self.previous_residual
        if self.previous_hidden_states is not None:
            return self.previous_hidden_states.clone()
        return hidden_states


class CustomCache(TeaCache):
    """
    CustomCache = TeaCache 决策 + TaylorSeer 式泰勒补偿（见 LightX2V cache_source.md）。
    复用缓存时用一阶泰勒近似残差：residual ≈ cache[0] + cache[1] * step_diff，提升精度。
    """

    def __init__(self, num_inference_steps, rel_l1_thresh, model_id, **kwargs):
        super().__init__(num_inference_steps, rel_l1_thresh, model_id, use_residual_caching=True)
        self._last_full_step = None
        self._taylor_cache = {}  # {0: residual, 1: derivative}
        self._previous_residual = None

    def reset(self):
        super().reset()
        self._last_full_step = None
        self._taylor_cache = {}
        self._previous_residual = None

    def store(self, hidden_states):
        """完整计算后：存残差并可选存一阶导数，供泰勒补偿."""
        if self.previous_hidden_states is None:
            self.previous_hidden_states = None
            return
        current_residual = hidden_states - self.previous_hidden_states
        current_step = self.step - 1
        if self._previous_residual is not None and self._last_full_step is not None:
            step_diff_used = max(1, current_step - self._last_full_step)
            derivative = (current_residual - self._previous_residual.to(current_residual.device)) / step_diff_used
            self._taylor_cache = {0: current_residual.clone(), 1: derivative.clone()}
        else:
            self._taylor_cache = {0: current_residual.clone()}
        self._previous_residual = current_residual.clone()
        self._last_full_step = current_step
        self.previous_hidden_states = None

    def update(self, hidden_states):
        """命中缓存时：用一阶泰勒近似残差后加回."""
        if not self._taylor_cache:
            return hidden_states
        step_diff = max(0, (self.step - 1) - (self._last_full_step or 0))
        residual_0 = self._taylor_cache[0]
        if residual_0.device != hidden_states.device:
            residual_0 = residual_0.to(hidden_states.device)
        out = residual_0
        if 1 in self._taylor_cache:
            residual_1 = self._taylor_cache[1].to(hidden_states.device)
            out = out + residual_1 * step_diff
        return hidden_states + out
