import os
import gc
import sys
import math
import json
import queue
import logging
import threading
import subprocess
import torch
import torch.nn.functional as F
import numpy as np
import imageio_ffmpeg
from PIL import Image
from decord import VideoReader, cpu
from diffusers import WanVACETransformer3DModel, AutoencoderKLWan
from diffusers.hooks import apply_group_offloading
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, UMT5EncoderModel
import ftfy
import html
import re
import random
from fire import Fire

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from ttkthemes import ThemedTk

# ruff: noqa: E402
from core.ui.widgets import Tooltip
from core.ui.theme_manager import ThemeManager
from core.ui.dnd_support import init_dnd, register_dnd_entries, configure_dnd_styles
from core.common.gpu_utils import check_cuda_availability, release_cuda_memory
from core.common.video_io import get_video_stream_info
from core.common.file_organizer import move_files_to_finished, restore_finished_files as _restore_finished_files

logger = logging.getLogger(__name__)

GUI_VERSION = "26-07-22.0"
CONFIG_FILE = "config_inpaint_v2.json"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm")

# Hardcoded model locations (not exposed in the GUI).
PRE_TRAINED_PATH = "./weights/Wan2.1-VACE-14B-diffusers"  # base Wan model (tokenizer/text_encoder/vae)
TRANSFORMER_PATH = "./weights/StereoCrafter2"             # fine-tuned VACE transformer
FP8_STATE_FILE = "diffusion_pytorch_model_fp8.pt"         # inside <transformer>-FP8/ (see export_fp8_transformer.py)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

# Windows builds of PyTorch ship without flash-attention kernels and leave the
# cuDNN SDPA backend runtime-disabled, so SDPA silently dispatches to a much
# slower backend. cuDNN attention is ~2.5x faster at Wan's sequence lengths
# and is numerically exact.
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(True)
PROMPT = ""


class InferenceCancelled(Exception):
    """Raised internally when a caller-supplied ``should_stop()`` returns True."""


def _noop_progress(stage, current, total, message=""):
    """Default progress callback: do nothing."""


def _never_stop():
    """Default cancellation hook: never stop."""
    return False


def detect_dual_input(input_video_path):
    """Detect a dual-layout (1x2: mask | warped) video from its filename.

    Mirrors inpainting_gui.py: stems ending with ``_splatted2`` / ``_splatted2F``
    are dual; anything else is treated as quad (2x2: left / mask / warped)."""
    stem = os.path.splitext(os.path.basename(input_video_path))[0]
    return stem.endswith("_splatted2") or stem.endswith("_splatted2F")


def get_exact_fps(input_video_path, video_reader=None):
    """Return the source frame rate, preserving irregular/fractional rates exactly.

    Prefers ffprobe's ``r_frame_rate`` (an exact rational string such as
    ``"24000/1001"``) so the value can be handed to ffmpeg verbatim. Falls back to
    decord's average fps at full float precision (never truncated)."""
    info = None
    try:
        info = get_video_stream_info(input_video_path)
    except Exception:
        info = None

    if info:
        for key in ("r_frame_rate", "avg_frame_rate"):
            rate = info.get(key)
            if not rate:
                continue
            try:
                if "/" in rate:
                    num, den = rate.split("/")
                    if float(den) != 0 and float(num) > 0:
                        return rate  # exact rational string, e.g. "24000/1001"
                elif float(rate) > 0:
                    return rate
            except ValueError:
                continue

    if video_reader is not None:
        return repr(float(video_reader.get_avg_fps()))  # full precision, not int()
    return "30"


def write_video_exact(frames, output_path, fps, stream_info=None):
    """Encode ``frames`` ([F, H, W, 3], float 0-1 or uint8) to an mp4 at exactly ``fps``.

    ``fps`` may be a number or a rational string (e.g. ``"24000/1001"``) and is passed
    to ffmpeg verbatim on both input and output so the saved video matches the source
    frame rate exactly. Uses the ffmpeg bundled with imageio-ffmpeg."""
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = (np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"Expected frames of shape [F, H, W, 3], got {frames.shape}")

    n, h, w, _ = frames.shape
    rate = str(fps)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", rate, "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", rate, output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.stdin.write(frames.tobytes())
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    stderr = proc.stderr.read().decode("utf-8", "ignore") if proc.stderr else ""
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed ({ret}) writing {output_path}:\n{stderr}")
    return output_path


class FlowMatchScheduler():

    def __init__(self,):
        self.set_timesteps_fn = FlowMatchScheduler.set_timesteps_wan
        self.num_train_timesteps = 1000

    @staticmethod
    def set_timesteps_wan(num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 5 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
        
    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, **kwargs):
        self.sigmas, self.timesteps = self.set_timesteps_fn(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            **kwargs,
        )

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    

def encode_vae_mode(vae, x):
    dist = vae.encode(x).latent_dist
    return dist.mode() if hasattr(dist, "mode") else dist.mean

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def get_t5_prompt_embeds(
    prompt = None,
    num_videos_per_prompt = 1,
    max_sequence_length = 226,
    device = None,
    dtype = None,
    tokenizer = None,
    text_encoder = None,
):
    # device = device or self._execution_device
    # dtype = dtype or self.text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    prompt,
    negative_prompt = None,
    do_classifier_free_guidance = True,
    num_videos_per_prompt = 1,
    prompt_embeds = None,
    negative_prompt_embeds = None,
    max_sequence_length = 226,
    device = None,
    dtype = None,
    tokenizer = None,
    text_encoder = None,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
            Whether to use classifier free guidance or not.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        device: (`torch.device`, *optional*):
            torch device
        dtype: (`torch.dtype`, *optional*):
            torch dtype
    """
    # device = device or self._execution_device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt_embeds = get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

    if do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        negative_prompt_embeds = get_t5_prompt_embeds(
            prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

    return prompt_embeds, negative_prompt_embeds


def prepare_masks(
    mask: torch.Tensor,
    reference_images = None,
    # generator = None,
    transformer_patch_size = None,
    vae_scale_factor_temporal = None,
    vae_scale_factor_spatial = None,
) -> torch.Tensor:
    # if isinstance(generator, list):
    #     # TODO: support this
    #     raise ValueError("Passing a list of generators is not yet supported. This may be supported in the future.")

    if reference_images is None:
        # For each batch of video, we set no reference image (as one or more can be passed by user)
        reference_images = [[None] for _ in range(mask.shape[0])]
    else:
        if mask.shape[0] != len(reference_images):
            raise ValueError(
                f"Batch size of `mask` {mask.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
            )

    # if mask.shape[0] != 1:
    #     # TODO: support this
    #     raise ValueError(
    #         "Generating with more than one video is not yet supported. This may be supported in the future."
    #     )

    # transformer_patch_size = (
    #     self.transformer.config.patch_size[1]
    #     if self.transformer is not None
    #     else self.transformer_2.config.patch_size[1]
    # )

    mask_list = []
    for mask_, reference_images_batch in zip(mask, reference_images):
        num_channels, num_frames, height, width = mask_.shape
        new_num_frames = (num_frames + vae_scale_factor_temporal - 1) // vae_scale_factor_temporal
        new_height = height // (vae_scale_factor_spatial * transformer_patch_size) * transformer_patch_size
        new_width = width // (vae_scale_factor_spatial * transformer_patch_size) * transformer_patch_size
        mask_ = mask_[0, :, :, :]
        mask_ = mask_.view(
            num_frames, new_height, vae_scale_factor_spatial, new_width, vae_scale_factor_spatial
        )
        mask_ = mask_.permute(2, 4, 0, 1, 3).flatten(0, 1)  # [8x8, num_frames, new_height, new_width]
        mask_ = torch.nn.functional.interpolate(
            mask_.unsqueeze(0), size=(new_num_frames, new_height, new_width), mode="nearest-exact"
        ).squeeze(0)
        num_ref_images = len(reference_images_batch)
        if num_ref_images > 0:
            mask_padding = torch.zeros_like(mask_[:, :num_ref_images, :, :])
            mask_ = torch.cat([mask_padding, mask_], dim=1)
        mask_list.append(mask_)
    return torch.stack(mask_list)


def preprocess_conditions(
    video = None,
    mask = None,
    reference_images = None,
    batch_size: int = 1,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    dtype = None,
    device = None,
    video_processor = None,
    base = None,
):
    if video is not None:
        # base = self.vae_scale_factor_spatial * (
        #     self.transformer.config.patch_size[1]
        #     if self.transformer is not None
        #     else self.transformer_2.config.patch_size[1]
        # )
        video_height, video_width = video_processor.get_default_height_width(video[0])

        if video_height * video_width > height * width:
            scale = min(width / video_width, height / video_height)
            video_height, video_width = int(video_height * scale), int(video_width * scale)

        if video_height % base != 0 or video_width % base != 0:
            # logger.warning(
            #     f"Video height and width should be divisible by {base}, but got {video_height} and {video_width}. "
            # )
            video_height = (video_height // base) * base
            video_width = (video_width // base) * base

        assert video_height * video_width <= height * width

        video = video_processor.preprocess_video(video, video_height, video_width)
        image_size = (video_height, video_width)  # Use the height/width of video (with possible rescaling)
    else:
        video = torch.zeros(batch_size, 3, num_frames, height, width, dtype=dtype, device=device)
        image_size = (height, width)  # Use the height/width provider by user


    if mask is not None:
        mask = video_processor.preprocess_video(mask, image_size[0], image_size[1])
        mask = torch.clamp((mask + 1) / 2, min=0, max=1)
    else:
        mask = torch.ones_like(video)

    video = video.to(dtype=dtype, device=device)
    mask = mask.to(dtype=dtype, device=device)

    # Make a list of list of images where the outer list corresponds to video batch size and the inner list
    # corresponds to list of conditioning images per video
    if reference_images is None or isinstance(reference_images, Image.Image):
        reference_images = [[reference_images] for _ in range(video.shape[0])]
    elif isinstance(reference_images, (list, tuple)) and isinstance(next(iter(reference_images)), Image.Image):
        reference_images = [reference_images]
    elif (
        isinstance(reference_images, (list, tuple))
        and isinstance(next(iter(reference_images)), list)
        and isinstance(next(iter(reference_images[0])), Image.Image)
    ):
        reference_images = reference_images
    else:
        raise ValueError(
            "`reference_images` has to be of type `PIL.Image.Image` or `list` of `PIL.Image.Image`, or "
            "`list` of `list` of `PIL.Image.Image`, but is {type(reference_images)}"
        )

    if video.shape[0] != len(reference_images):
        raise ValueError(
            f"Batch size of `video` {video.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
        )

    ref_images_lengths = [len(reference_images_batch) for reference_images_batch in reference_images]
    if any(l != ref_images_lengths[0] for l in ref_images_lengths):
        raise ValueError(
            f"All batches of `reference_images` should have the same length, but got {ref_images_lengths}. Support for this "
            "may be added in the future."
        )

    reference_images_preprocessed = []
    for i, reference_images_batch in enumerate(reference_images):
        preprocessed_images = []
        for j, image in enumerate(reference_images_batch):
            if image is None:
                continue
            image = video_processor.preprocess(image, None, None)
            img_height, img_width = image.shape[-2:]
            scale = min(image_size[0] / img_height, image_size[1] / img_width)
            new_height, new_width = int(img_height * scale), int(img_width * scale)
            resized_image = torch.nn.functional.interpolate(
                image, size=(new_height, new_width), mode="bilinear", align_corners=False
            ).squeeze(0)  # [C, H, W]
            top = (image_size[0] - new_height) // 2
            left = (image_size[1] - new_width) // 2
            canvas = torch.ones(3, *image_size, device=device, dtype=dtype)
            canvas[:, top : top + new_height, left : left + new_width] = resized_image
            preprocessed_images.append(canvas)
        reference_images_preprocessed.append(preprocessed_images)

    return video, mask, reference_images_preprocessed


def prepare_video_latents(
    video: torch.Tensor,
    mask: torch.Tensor,
    reference_images = None,
    device = None,
    vae = None,
) -> torch.Tensor:
    # device = device or self._execution_device

    # if isinstance(generator, list):
    #     # TODO: support this
    #     raise ValueError("Passing a list of generators is not yet supported. This may be supported in the future.")

    if reference_images is None:
        # For each batch of video, we set no re
        # ference image (as one or more can be passed by user)
        reference_images = [[None] for _ in range(video.shape[0])]
    else:
        if video.shape[0] != len(reference_images):
            raise ValueError(
                f"Batch size of `video` {video.shape[0]} and length of `reference_images` {len(reference_images)} does not match."
            )

    # if video.shape[0] != 1:
    #     # TODO: support this
    #     raise ValueError(
    #         "Generating with more than one video is not yet supported. This may be supported in the future."
    #     )

    vae_dtype = vae.dtype
    video = video.to(dtype=vae_dtype)

    latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32).view(
        1, vae.config.z_dim, 1, 1, 1
    )

    if mask is None:
        # latents = retrieve_latents(vae.encode(video), generator, sample_mode="argmax").unbind(0)
        latents = encode_vae_mode(vae, video)
        latents = ((latents.float() - latents_mean) * latents_std).to(vae_dtype)
    else:
        mask = torch.where(mask > 0.5, 1.0, 0.0).to(dtype=vae_dtype)
        inactive = video * (1 - mask)
        reactive = video * mask
        # inactive = retrieve_latents(vae.encode(inactive), generator, sample_mode="argmax")
        inactive = encode_vae_mode(vae, inactive)
        # reactive = retrieve_latents(vae.encode(reactive), generator, sample_mode="argmax")
        reactive = encode_vae_mode(vae, reactive)
        inactive = ((inactive.float() - latents_mean) * latents_std).to(vae_dtype)
        reactive = ((reactive.float() - latents_mean) * latents_std).to(vae_dtype)
        latents = torch.cat([inactive, reactive], dim=1)


    latent_list = []
    for latent, reference_images_batch in zip(latents, reference_images):
        for reference_image in reference_images_batch:
            assert reference_image.ndim == 3
            reference_image = reference_image.to(dtype=vae_dtype)
            reference_image = reference_image[None, :, None, :, :]  # [1, C, 1, H, W]
            # reference_latent = retrieve_latents(vae.encode(reference_image), generator, sample_mode="argmax")
            reference_latent = vae.encode(reference_image).latent_dist.sample()
            reference_latent = ((reference_latent.float() - latents_mean) * latents_std).to(vae_dtype)
            reference_latent = reference_latent.squeeze(0)  # [C, 1, H, W]
            reference_latent = torch.cat([reference_latent, torch.zeros_like(reference_latent)], dim=0)
            latent = torch.cat([reference_latent.squeeze(0), latent], dim=1)
        latent_list.append(latent)

    return torch.stack(latent_list)


def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """水平方向融合 Latents，支持 [B, C, F, H, W]"""
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, 1, -1) / overlap_size).to(b.device, dtype=b.dtype)
    b[:, :, :, :, :overlap_size] = (1 - weight_b) * a[:, :, :, :, -overlap_size:] + weight_b * b[:, :, :, :, :overlap_size]
    return b


def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """垂直方向融合 Latents，支持 [B, C, F, H, W]"""
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1, 1) / overlap_size).to(b.device, dtype=b.dtype)
    b[:, :, :, :overlap_size, :] = (1 - weight_b) * a[:, :, :, -overlap_size:, :] + weight_b * b[:, :, :, :overlap_size, :]
    return b


def run_wan_pipeline(
    cond_frames, 
    mask_frames, 
    prompt_embeds, 
    transformer, 
    vae, 
    noise_scheduler, 
    videoprocessor,
    vae_scale_factor_spatial,
    vae_scale_factor_temporal,
    transformer_patch_size,
    should_stop=None,
):
    """封装单次 Wan 去噪 Pipeline 以供分块调用"""
    should_stop = should_stop or _never_stop
    # 此时进入的 cond_frames 是正确的 [B, C, F, H, W]
    height, width = cond_frames.shape[3], cond_frames.shape[4]
    num_frames = cond_frames.shape[2]

    with torch.no_grad():
        # VideoProcessor 强制要求输入格式为 [B, F, C, H, W]
        # 所以我们在这里做一次临时的维度翻转：[B, C, F, H, W] -> [B, F, C, H, W]
        cond_frames_vp = cond_frames.permute(0, 2, 1, 3, 4)
        mask_frames_vp = mask_frames.permute(0, 2, 1, 3, 4)
        
        condition_video, mask, reference_images = preprocess_conditions(
            video=cond_frames_vp,
            mask=mask_frames_vp,
            reference_images=None,
            batch_size=1,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=DTYPE,
            device=DEVICE,
            video_processor=videoprocessor,
            base=vae_scale_factor_spatial * transformer_patch_size,
        )
    
        conditioning_latents = prepare_video_latents(condition_video, mask, reference_images, DEVICE, vae)
        mask_for_transformer = prepare_masks(mask, reference_images, transformer_patch_size, vae_scale_factor_temporal, vae_scale_factor_spatial).to(DEVICE, dtype=DTYPE)
        control_hidden_states = torch.cat([conditioning_latents, mask_for_transformer], dim=1).to(DTYPE)

    c = transformer.config.in_channels
    f = (num_frames - 1) // vae_scale_factor_temporal + 1
    h = height // vae_scale_factor_spatial
    w = width // vae_scale_factor_spatial
    
    latents = torch.randn(1, c, f, h, w, device=DEVICE, dtype=DTYPE)

    for i, t in enumerate(noise_scheduler.timesteps):
        if should_stop():
            raise InferenceCancelled()
        timestep_tensor = t.unsqueeze(0).to(DEVICE, dtype=DTYPE)
        with torch.no_grad():
            model_pred = transformer(
                hidden_states=latents,
                timestep=timestep_tensor,
                encoder_hidden_states=prompt_embeds,
                control_hidden_states=control_hidden_states,
                return_dict=False,
            )[0]
        latents = noise_scheduler.step(model_pred, t, latents)
    
    return latents


def spatial_tiled_process(
    cond_frames, mask_frames, tile_num, tile_overlap, prompt_embeds, transformer, vae, noise_scheduler, videoprocessor,
    vae_scale_factor_spatial, vae_scale_factor_temporal, transformer_patch_size, should_stop=None
):
    """处理单段视频的空间分块"""
    should_stop = should_stop or _never_stop
    if tile_num == 1:
        return run_wan_pipeline(cond_frames, mask_frames, prompt_embeds, transformer, vae, noise_scheduler, videoprocessor, vae_scale_factor_spatial, vae_scale_factor_temporal, transformer_patch_size, should_stop=should_stop)

    height = cond_frames.shape[3]
    width = cond_frames.shape[4]
    
    # 确保切块大小能够被 VAE 和 Transformer Patch 的乘积（通常是 16）整除
    base = vae_scale_factor_spatial * transformer_patch_size
    tile_size = (
        int((height + tile_overlap * (tile_num - 1)) / tile_num) // base * base,
        int((width + tile_overlap * (tile_num - 1)) / tile_num) // base * base
    )
    tile_stride = (tile_size[0] - tile_overlap, tile_size[1] - tile_overlap)

    cols = []
    for i in range(tile_num):
        rows = []
        for j in range(tile_num):
            h_start = min(i * tile_stride[0], height - tile_size[0])
            w_start = min(j * tile_stride[1], width - tile_size[1])
            
            cond_tile = cond_frames[:, :, :, h_start : h_start + tile_size[0], w_start : w_start + tile_size[1]]
            mask_tile = mask_frames[:, :, :, h_start : h_start + tile_size[0], w_start : w_start + tile_size[1]]

            tile_latent = run_wan_pipeline(
                cond_tile, mask_tile, prompt_embeds, transformer, vae, noise_scheduler, videoprocessor,
                vae_scale_factor_spatial, vae_scale_factor_temporal, transformer_patch_size, should_stop=should_stop
            )
            rows.append(tile_latent)
        cols.append(rows)

    # 映射回 Latent 空间的 stride 和 overlap
    latent_stride = (tile_stride[0] // vae_scale_factor_spatial, tile_stride[1] // vae_scale_factor_spatial)
    latent_overlap = (tile_overlap // vae_scale_factor_spatial, tile_overlap // vae_scale_factor_spatial)

    # 融合 Latents
    results_cols = []
    for i, rows in enumerate(cols):
        results_rows = []
        for j, tile in enumerate(rows):
            if i > 0:
                tile = blend_v(cols[i - 1][j], tile, latent_overlap[0])
            if j > 0:
                tile = blend_h(rows[j - 1], tile, latent_overlap[1])
            results_rows.append(tile)
        results_cols.append(results_rows)

    pixels = []
    for i, rows in enumerate(results_cols):
        for j, tile in enumerate(rows):
            if i < len(results_cols) - 1:
                tile = tile[:, :, :, :latent_stride[0], :]
            if j < len(rows) - 1:
                tile = tile[:, :, :, :, :latent_stride[1]]
            rows[j] = tile
        pixels.append(torch.cat(rows, dim=4))
    
    return torch.cat(pixels, dim=3)


def plan_next_chunk(global_len, total_frames, frames_chunk, frames_overlap, vae_scale_factor_temporal):
    """Decide the next temporal chunk's start and (valid) length.

    Returns ``(start, length)`` where ``length`` is always a valid VAE temporal size
    (``length % vsft == 1``). For the final chunk the start is chosen so the chunk
    lands exactly on the last frame while keeping the overlap at or just below
    ``frames_overlap`` -- instead of the old behaviour that forced a full
    ``frames_chunk`` window and produced a huge overlap on the tail.
    """
    vsft = vae_scale_factor_temporal

    def round_down(n):
        return ((n - 1) // vsft) * vsft + 1

    if global_len == 0:
        return 0, round_down(min(frames_chunk, total_frames))

    desired_start = global_len - frames_overlap
    remaining = total_frames - desired_start  # frames from the desired start to the end

    if remaining > frames_chunk:
        # Middle chunk: step forward by exactly the requested overlap.
        return desired_start, round_down(frames_chunk)

    # Final chunk: anchor the end on the last frame so the output length matches the
    # input length. Pick the largest valid length that fits the clip (and frames_chunk)
    # and keeps overlap <= frames_overlap; rounding the length DOWN only ever shrinks
    # the overlap, never grows it.
    uncovered = total_frames - global_len
    max_len = min(frames_chunk, total_frames)
    length = round_down(min(uncovered + frames_overlap, max_len))
    start = max(0, total_frames - length)
    start = min(start, global_len)             # never produce a negative overlap
    length = round_down(total_frames - start)  # exact valid length for [start, total_frames]
    return start, length


class WanInpaintEngine:
    """Holds the loaded Wan VACE models and derived constants so they can be
    reused across multiple videos (load once, run many)."""

    def __init__(self, tokenizer, text_encoder, vae, transformer, device, dtype, offload_mode="none"):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.transformer = transformer
        self.device = device
        self.dtype = dtype
        self.offload_mode = offload_mode

        self.videoprocessor = VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial)
        self.transformer_patch_size = transformer.config.patch_size[1]
        self.vae_scale_factor_temporal = 2 ** sum(vae.temperal_downsample)
        self.vae_scale_factor_spatial = 2 ** len(vae.temperal_downsample)


def _load_fp8_transformer(fp8_dir):
    """Load a pre-quantized FP8 transformer exported by export_fp8_transformer.py.

    Meta-device init + ``assign`` load keeps peak RAM at ~the checkpoint size
    (~16 GiB) instead of materializing the ~28 GiB bf16 model first."""
    import torchao  # noqa: F401  - required to unpickle Float8Tensor weights
    from accelerate import init_empty_weights

    cfg = WanVACETransformer3DModel.load_config(fp8_dir)
    with init_empty_weights():
        tf = WanVACETransformer3DModel.from_config(cfg)
    # weights_only=False: the file contains torchao tensor subclasses and is a
    # locally generated, trusted artifact (see export_fp8_transformer.py).
    sd = torch.load(os.path.join(fp8_dir, FP8_STATE_FILE), map_location="cpu", weights_only=False)
    tf.load_state_dict(sd, assign=True)
    return tf


def load_models(pre_trained_path, transformer_path, device=DEVICE, dtype=DTYPE,
                offload_mode="none", use_sageattention=False,
                progress_cb=None):
    """Loads the tokenizer, text encoder, VAE and transformer once and returns a
    :class:`WanInpaintEngine`. Heavy and slow; call once and reuse the engine.

    ``offload_mode``:
      * ``"none"``        -> all models resident on ``device`` (fastest; needs the VRAM).
      * ``"transformer"`` -> only the 14B transformer is block-level CPU-offloaded;
        the text encoder + VAE stay resident on the GPU. Saves ~12 GB of system RAM
        vs ``"group"`` at the cost of ~12 GB more VRAM (needs a ~24 GB+ card).
      * ``"group"``       -> everything CPU-offloaded. The models stay on CPU and
        their weights are streamed to the GPU on demand, letting the 14B
        transformer run on far less VRAM (slower due to PCIe transfers; needs
        enough system RAM/pagefile to hold all weights).
      * ``"fp8"``         -> the transformer is loaded from the pre-quantized FP8
        checkpoint at ``<transformer_path>-FP8`` and kept fully GPU-resident
        (~16 GiB weights; needs a ~24 GB+ card). Text encoder + VAE are
        leaf-level CPU-offloaded. No PCIe streaming for the transformer ->
        fastest option on cards that fit it. Requires running
        export_fp8_transformer.py once (or copying its output folder).
    """
    progress_cb = progress_cb or _noop_progress
    offload_mode = str(offload_mode).strip().lower()

    fp8_resident = offload_mode == "fp8"
    if fp8_resident and device.type != "cuda":
        raise RuntimeError("FP8 resident mode requires a CUDA GPU.")
    if fp8_resident:
        total_gib = torch.cuda.get_device_properties(device).total_memory / 2**30
        if total_gib < 20:
            raise RuntimeError(
                f"FP8 resident mode needs ~20 GB VRAM; this GPU has {total_gib:.0f} GB. "
                "Use 'Transformer only' or 'Group offload' instead."
            )

    offload_transformer = offload_mode in ("group", "transformer")
    offload_others = offload_mode in ("group", "fp8")
    if offload_transformer and device.type != "cuda":
        logger.warning("CPU offload requested but CUDA is unavailable; loading without offload.")
        offload_transformer = offload_others = False
        offload_mode = "none"

    def place(model, offloaded):
        # Offloaded models keep their weights on CPU; the offload hooks stream them to GPU.
        return model if offloaded else model.to(device)

    progress_cb("load_models", 0, 4, "Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_path, subfolder="tokenizer")

    progress_cb("load_models", 1, 4, "Loading text encoder...")
    text_encoder = place(
        UMT5EncoderModel.from_pretrained(pre_trained_path, subfolder="text_encoder", torch_dtype=dtype),
        offload_others,
    )

    progress_cb("load_models", 2, 4, "Loading VAE...")
    vae = place(
        AutoencoderKLWan.from_pretrained(pre_trained_path, subfolder="vae", torch_dtype=dtype),
        offload_others,
    )

    progress_cb("load_models", 3, 4, "Loading transformer...")
    if fp8_resident:
        fp8_dir = transformer_path.rstrip("/\\") + "-FP8"
        logger.info("Loading pre-quantized FP8 transformer from %s (GPU-resident).", fp8_dir)
        transformer = _load_fp8_transformer(fp8_dir).to(device)
    else:
        transformer = place(
            WanVACETransformer3DModel.from_pretrained(transformer_path, torch_dtype=dtype),
            offload_transformer,
        )

    transformer.eval()
    vae.eval()
    text_encoder.eval()

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    if use_sageattention:
        try:
            transformer.set_attention_backend("sage")
            logger.info("SageAttention attention backend enabled for the Wan transformer.")
        except Exception as e:
            logger.warning("Could not enable SageAttention (%s); using PyTorch SDPA instead.", e)

    offload_device = torch.device("cpu")
    if offload_transformer:
        progress_cb("load_models", 4, 4, "Enabling CPU offload...")
        logger.info(
            "Enabling block-level CPU offload for the transformer (stream overlap on)%s.",
            "" if offload_others else "; text encoder + VAE stay on GPU",
        )
        # Transformer is the large one -> block-level streaming. Stream overlap
        # requires num_blocks_per_group=1 (diffusers clamps anything else to 1).
        # low_cpu_mem_usage is essential: without it, diffusers pins ALL weights
        # (page-locked RAM) upfront at load. Pinned memory cannot be backed by
        # the page file, so on RAM-constrained systems that instantly fails with
        # a raw "CUDA error: out of memory" even though the GPU is nearly empty.
        apply_group_offloading(
            transformer, onload_device=device, offload_device=offload_device,
            offload_type="block_level", num_blocks_per_group=1,
            use_stream=True, low_cpu_mem_usage=True,
        )
    if offload_others:
        # VAE + text encoder are smaller -> leaf-level offload.
        apply_group_offloading(
            vae, onload_device=device, offload_device=offload_device,
            offload_type="leaf_level", use_stream=True, low_cpu_mem_usage=True,
        )
        apply_group_offloading(
            text_encoder, onload_device=device, offload_device=offload_device,
            offload_type="leaf_level", use_stream=True, low_cpu_mem_usage=True,
        )

    progress_cb("load_models", 4, 4, "Models loaded.")
    logger.info("Wan VACE models loaded (%s, offload=%s).", dtype, offload_mode)
    return WanInpaintEngine(tokenizer, text_encoder, vae, transformer, device, dtype, offload_mode)


def run_inpaint(
    engine,
    input_video_path,
    save_dir,
    prompt="",
    frames_chunk=81,
    frames_overlap=10,
    tile_overlap=128,
    tile_num=2,
    inference_steps=10,
    seed=0,
    is_dual_input=None,
    progress_cb=None,
    should_stop=None,
):
    """Runs stereo inpainting on a single video and writes the outputs to ``save_dir``.

    Two input layouts are supported (matching inpainting_gui.py):

    * **Quad** (2x2 grid): top-left = left eye, bottom-left = mask,
      bottom-right = warped right. Outputs an SBS video ``[left | inpainted-right]``
      plus an anaglyph -> ``{"sbs": path, "anaglyph": path}``.
    * **Dual** (1x2 halves): left = mask, right = warped right. No left eye is
      available, so only the inpainted right eye is written -> ``{"right_eye": path}``.

    ``is_dual_input`` may be ``True``/``False`` to force a layout, or ``None`` to
    auto-detect from the filename (see :func:`detect_dual_input`).

    ``progress_cb(stage, current, total, message)`` is called for UI updates and
    ``should_stop()`` is polled between chunks and denoise steps so a caller can
    cancel; cancellation raises :class:`InferenceCancelled`.
    """
    progress_cb = progress_cb or _noop_progress
    should_stop = should_stop or _never_stop

    device = engine.device
    dtype = engine.dtype
    tokenizer = engine.tokenizer
    text_encoder = engine.text_encoder
    vae = engine.vae
    transformer = engine.transformer
    videoprocessor = engine.videoprocessor
    transformer_patch_size = engine.transformer_patch_size
    vae_scale_factor_temporal = engine.vae_scale_factor_temporal
    vae_scale_factor_spatial = engine.vae_scale_factor_spatial

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if is_dual_input is None:
        is_dual_input = detect_dual_input(input_video_path)

    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    video_name = base_name.replace("_splatting_results", "") + "_inpainting_results"
    logger.info("Input layout: %s", "dual (1x2)" if is_dual_input else "quad (2x2)")

    progress_cb("encode_prompt", 0, 1, "Encoding prompt...")
    logger.info("Encoding prompt...")
    with torch.no_grad():
        prompt_embeds, _ = encode_prompt(
            [prompt], do_classifier_free_guidance=False, max_sequence_length=226,
            device=device, dtype=dtype, tokenizer=tokenizer, text_encoder=text_encoder
        )

    progress_cb("load_video", 0, 1, "Loading video...")
    logger.info("Loading video: %s", input_video_path)
    video_reader = VideoReader(input_video_path, ctx=cpu(0))
    fps = get_exact_fps(input_video_path, video_reader)
    logger.info("Source FPS: %s", fps)
    total_frames = len(video_reader)
    frame_indices = list(range(total_frames))
    frames = video_reader.get_batch(frame_indices)


    # [t,h,w,c] -> [1,c,t,h,w]
    frames = torch.from_numpy(frames.asnumpy()).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0

    if is_dual_input:
        # Dual (1x2): left half = mask, right half = warped right eye. No left eye.
        width = frames.shape[4] // 2
        frames_left = None
        all_masks = frames[:, :, :, :, :width]
        all_frames = frames[:, :, :, :, width:]
    else:
        # Quad (2x2): TL = left eye, BL = mask, BR = warped right eye.
        height, width = frames.shape[3] // 2, frames.shape[4] // 2
        frames_left = frames[:, :, :, :height, :width]
        all_masks = frames[:, :, :, height:, :width]
        all_frames = frames[:, :, :, height:, width:]

    all_frames = all_frames * (1.0 - all_masks) + 0.5 * all_masks

    base = vae_scale_factor_spatial * transformer_patch_size
    h_orig, w_orig = all_frames.shape[3], all_frames.shape[4]

    # 1. 向上取整 (math.ceil)，倒推计算出能被 Tiling 完美拼接的“目标分辨率”
    min_tile_h = (h_orig + tile_overlap * (tile_num - 1)) / tile_num
    tile_size_h = math.ceil(min_tile_h / base) * base

    min_tile_w = (w_orig + tile_overlap * (tile_num - 1)) / tile_num
    tile_size_w = math.ceil(min_tile_w / base) * base

    tile_stride_h = tile_size_h - tile_overlap
    tile_stride_w = tile_size_w - tile_overlap

    target_h = tile_stride_h * (tile_num - 1) + tile_size_h
    target_w = tile_stride_w * (tile_num - 1) + tile_size_w

    # 2. 计算需要补充的边缘像素数
    pad_h = target_h - h_orig
    pad_w = target_w - w_orig

    if pad_h > 0 or pad_w > 0:
        logger.info(
            "Padding resolution from %dx%d to %dx%d to perfectly match Tiling output.",
            w_orig, h_orig, target_w, target_h,
        )

        # 1. 临时取出 Batch 并对调通道和帧数维度: [1, C, F, H, W] -> [F, C, H, W] (变成标准的 4D 图片格式)
        frames_4d = all_frames[0].permute(1, 0, 2, 3)

        # 2. 对 4D 张量进行边缘复制填充 (PyTorch 对此支持极其完美)
        frames_4d = F.pad(frames_4d, (0, pad_w, 0, pad_h), mode='replicate')

        # 3. 还原回 5D 视频张量: [F, C, H_new, W_new] -> [1, C, F, H_new, W_new]
        all_frames = frames_4d.permute(1, 0, 2, 3).unsqueeze(0)

        # Mask 填充 0 (constant 模式原生支持 5D，直接 pad 即可)
        all_masks = F.pad(all_masks, (0, pad_w, 0, pad_h), mode='constant', value=0)

    noise_scheduler = FlowMatchScheduler()
    noise_scheduler.set_timesteps(num_inference_steps=inference_steps, denoising_strength=1.0)

    generated_video_chunks = []

    logger.info("Starting Temporal Chunking inference (Total Frames: %d)...", total_frames)

    # 记录全局已经完美生成的有效帧数
    global_len = 0

    while global_len < total_frames:
        if should_stop():
            raise InferenceCancelled()
        cur_i, valid_chunk_size = plan_next_chunk(
            global_len, total_frames, frames_chunk, frames_overlap, vae_scale_factor_temporal
        )

        chunk_cond = all_frames[:, :, cur_i : cur_i + valid_chunk_size].clone()
        chunk_mask = all_masks[:, :, cur_i : cur_i + valid_chunk_size]

        actual_overlap = 0
        if global_len > 0:
            # 真实重叠长度 = 已生成的总进度 - 当前段的倒推起点
            # (例如：已生成 81 帧，当前段为了凑 81 帧从第 9 帧开始取，那么重叠帧数就是 81 - 9 = 72 帧！)
            actual_overlap = global_len - cur_i
            
            # 把已经生成的历史画面作为“绝对条件”覆盖到当前段的前面
            # 我们通过 torch.cat 临时拼一下历史结果以便提取
            temp_global_generated = torch.cat(generated_video_chunks, dim=2)
            chunk_cond[:, :, :actual_overlap] = temp_global_generated[:, :, cur_i : global_len]

        progress_cb(
            "inpaint", min(global_len, total_frames), total_frames,
            f"Processing chunk [{cur_i}:{cur_i + valid_chunk_size}] | Overlap: {actual_overlap} frames...",
        )
        logger.info(
            "Processing chunk [%d:%d] | Overlap context: %d frames...",
            cur_i, cur_i + valid_chunk_size, actual_overlap,
        )

        # --- 空间分块推理 ---
        chunk_latents = spatial_tiled_process(
            chunk_cond, chunk_mask, tile_num, tile_overlap, prompt_embeds, transformer, vae, noise_scheduler,
            videoprocessor, vae_scale_factor_spatial, vae_scale_factor_temporal, transformer_patch_size,
            should_stop=should_stop,
        )

        # --- 解码当前分段 ---
        with torch.no_grad():
            latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(1, vae.config.z_dim, 1, 1, 1)
            latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32).view(1, vae.config.z_dim, 1, 1, 1)
            chunk_latents = chunk_latents.float() * latents_std + latents_mean
            chunk_latents = chunk_latents.to(vae.dtype)
            video_chunk_tensor = vae.decode(chunk_latents, return_dict=False)[0]

            video_chunk_tensor = (video_chunk_tensor / 2 + 0.5).clamp(0, 1)

        # 保存并剔除重复片段
        if global_len == 0:
            generated_video_chunks.append(video_chunk_tensor)
            global_len += video_chunk_tensor.shape[2]
        else:
            # 严格剔除历史重叠帧
            new_frames = video_chunk_tensor[:, :, actual_overlap:]
            generated_video_chunks.append(new_frames)
            global_len += new_frames.shape[2]

        # 保护机制：如果因为取整或视频极短导致无法前进，跳出避免死循环
        if global_len > 0 and actual_overlap >= valid_chunk_size:
            logger.warning("Chunk progression stuck due to temporal scaling limits. Exiting loop.")
            break

    # 拼接所有时间分段
    final_video = torch.cat(generated_video_chunks, dim=2)

    if pad_h > 0 or pad_w > 0:
        logger.info("Removing padding, restoring resolution to %dx%d...", w_orig, h_orig)
        final_video = final_video[:, :, :, :h_orig, :w_orig]

    progress_cb("export", 0, 1, "Exporting final video...")
    logger.info("Exporting final video...")

    # 1. 取出 batch 0，形状变为 [C, F, H, W]
    video_tensor = final_video[0]

    # 2. 转换为 numpy 并调整维度顺序为 [F, H, W, C]
    video_np = video_tensor.permute(1, 2, 3, 0).cpu().float().numpy()

    if is_dual_input:
        # No left eye available -> output only the inpainted right eye.
        right_eye_path = os.path.join(save_dir, f"{video_name}_right_eye.mp4")
        write_video_exact(video_np, right_eye_path, fps)

        progress_cb("done", total_frames, total_frames, "Done.")
        logger.info("Saved:\n  Right eye: %s", right_eye_path)
        return {"right_eye": right_eye_path}

    video_left_np = frames_left[0].permute(1, 2, 3, 0).cpu().float().numpy()

    frames_sbs = np.concatenate([video_left_np, video_np], axis=2)
    frames_sbs_path = os.path.join(save_dir, f"{video_name}_sbs.mp4")
    write_video_exact(frames_sbs, frames_sbs_path, fps)


    video_left_np[:, :, :, 1] = 0
    video_left_np[:, :, :, 2] = 0
    video_np[:, :, :, 0] = 0

    vid_anaglyph = video_left_np + video_np
    vid_anaglyph_path = os.path.join(save_dir, f"{video_name}_anaglyph.mp4")
    write_video_exact(vid_anaglyph, vid_anaglyph_path, fps)

    progress_cb("done", total_frames, total_frames, "Done.")
    logger.info("Saved:\n  SBS:      %s\n  Anaglyph: %s", frames_sbs_path, vid_anaglyph_path)
    return {"sbs": frames_sbs_path, "anaglyph": vid_anaglyph_path}


def resolve_layout(mode):
    """Map a layout mode string to ``is_dual_input`` (None = auto-detect)."""
    return {"auto": None, "dual": True, "quad": False}[str(mode).strip().lower()]


def resolve_offload(mode):
    """Map a GUI offload label to a ``load_models`` offload_mode string."""
    mode = str(mode).strip().lower()
    if mode.startswith("group"):
        return "group"
    if mode.startswith("transformer"):
        return "transformer"
    if mode.startswith("fp8"):
        return "fp8"
    return "none"


def main(
    pre_trained_path,
    transformer_path,
    input_video_path,
    save_dir,
    frames_chunk=81,
    frames_overlap=10,
    tile_overlap=128,
    tile_num=2,
    inference_steps=10,
    seed=0,
    prompt=PROMPT,
    input_layout="auto",
    offload_mode="none",
    use_sageattention=True,
):
    """CLI entrypoint (via Fire). Loads the models then inpaints a single video.

    ``input_layout`` is one of ``auto`` (detect from filename), ``dual``, or ``quad``.
    ``offload_mode`` is ``none``, ``transformer`` (offload only the 14B transformer;
    text encoder + VAE stay on GPU — saves system RAM, needs ~24 GB+ VRAM),
    ``group`` (offload everything — lowest VRAM, highest system RAM use), or
    ``fp8`` (pre-quantized FP8 transformer fully GPU-resident — fastest on
    ~24 GB+ cards; requires the ``<transformer>-FP8`` folder from
    export_fp8_transformer.py).
    ``use_sageattention`` runs the transformer's attention through the quantized
    SageAttention kernel (much faster at Wan's sequence lengths, near-exact quality).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")

    def console_progress(stage, current, total, message=""):
        total = total or 1
        logger.info("[%-12s %3d%%] %s", stage, int(100 * current / total), message)

    engine = load_models(
        pre_trained_path, transformer_path,
        offload_mode=offload_mode,
        use_sageattention=use_sageattention,
        progress_cb=console_progress,
    )
    try:
        run_inpaint(
            engine,
            input_video_path,
            save_dir,
            prompt=prompt,
            frames_chunk=frames_chunk,
            frames_overlap=frames_overlap,
            tile_overlap=tile_overlap,
            tile_num=tile_num,
            inference_steps=inference_steps,
            seed=seed,
            is_dual_input=resolve_layout(input_layout),
            progress_cb=console_progress,
        )
    except InferenceCancelled:
        logger.warning("Inference cancelled.")


class TextHandler(logging.Handler):
    """Logging handler that forwards records onto a thread-safe queue.

    The GUI drains the queue on the Tk main thread (see ``_poll_queue``), so this
    handler is safe to call from the background inference worker without touching
    Tk from another thread.
    """

    def __init__(self, ui_queue):
        super().__init__()
        self.ui_queue = ui_queue

    def emit(self, record):
        try:
            self.ui_queue.put(("log", self.format(record)))
        except Exception:
            pass


class WanInpaintingGUI(ThemedTk):
    """Tkinter front-end for the Wan VACE stereo inpainting engine."""

    def __init__(self):
        super().__init__(theme="clam")
        self.title(f"StereoCrafter Wan Inpainting (v2) {GUI_VERSION}")

        self.app_config = self.load_config()
        self._is_startup = True

        # Drag-and-drop (degrades gracefully if tkinterdnd2 unavailable)
        self._dnd_enabled = init_dnd(self)

        # Worker / cancellation state
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.engine = None
        self._engine_paths = None  # (pre_trained_path, transformer_path) the engine was loaded with
        self._batch_prefix = ""
        self._ui_queue = queue.Queue()  # worker -> main-thread UI updates

        self._init_vars()

        self.style = ttk.Style()
        self.theme_manager = ThemeManager(self.dark_mode_var, self.app_config)

        self.create_widgets()
        self.update_idletasks()
        self._apply_theme(is_startup=True)
        self._set_saved_geometry()
        self._is_startup = False

        self._configure_logging()
        self.update_status("Ready")
        self.protocol("WM_DELETE_WINDOW", self.exit_application)
        self.after(100, self._poll_queue)

        if check_cuda_availability():
            logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
        else:
            logger.warning("CUDA not available — inference will run on CPU (very slow).")

    # ------------------------------------------------------------------ vars
    def _init_vars(self):
        cfg = self.app_config
        self.dark_mode_var = tk.BooleanVar(value=cfg.get("dark_mode_enabled", False))

        self.input_folder_var = tk.StringVar(value=cfg.get("input_folder", "./output_splatted"))
        self.output_folder_var = tk.StringVar(value=cfg.get("output_folder", "./completed_output"))

        self.frames_chunk_var = tk.StringVar(value=str(cfg.get("frames_chunk", 81)))
        self.frames_overlap_var = tk.StringVar(value=str(cfg.get("frames_overlap", 10)))
        self.tile_overlap_var = tk.StringVar(value=str(cfg.get("tile_overlap", 128)))
        self.tile_num_var = tk.StringVar(value=str(cfg.get("tile_num", 2)))
        self.inference_steps_var = tk.StringVar(value=str(cfg.get("inference_steps", 10)))
        self.seed_var = tk.StringVar(value=str(cfg.get("seed", 0)))
        self.offload_mode_var = tk.StringVar(value=cfg.get("offload_mode", "Group offload"))
        self.use_sageattention_var = tk.BooleanVar(value=cfg.get("use_sageattention", True))
        self.move_to_finished_var = tk.BooleanVar(value=cfg.get("move_to_finished", True))

        # Window geometry
        # Full "WxH+X+Y" string so size *and* on-screen position round-trip exactly.
        self.window_geometry = cfg.get("window_geometry", None)
        self.default_width = 620

        # Status / progress
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0.0)

    # --------------------------------------------------------------- widgets
    def create_widgets(self):
        self._create_menubar()

        pad = {"padx": 6, "pady": 3}

        # --- Paths ---
        paths = ttk.LabelFrame(self, text="Paths")
        paths.pack(fill="x", **pad)
        paths.columnconfigure(1, weight=1)
        self._dnd_entries = []

        # Input row: pick a single file OR a folder to batch-process.
        input_tip = ("A single video file (processed alone) or a folder of videos "
                     "(every supported video is batch-processed).")
        in_lbl = ttk.Label(paths, text="Input file/folder:")
        in_lbl.grid(row=0, column=0, sticky="w", padx=4, pady=2)
        in_entry = ttk.Entry(paths, textvariable=self.input_folder_var)
        in_entry.grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        in_btns = ttk.Frame(paths)
        in_btns.grid(row=0, column=2, padx=4, pady=2)
        ttk.Button(in_btns, text="File", width=6, command=self._browse_input_file).pack(side="left", padx=(0, 2))
        ttk.Button(in_btns, text="Folder", width=7, command=self._browse_input_folder).pack(side="left")
        Tooltip(in_lbl, input_tip)
        Tooltip(in_entry, input_tip)
        self._dnd_entries.append((in_entry, self.input_folder_var, False, VIDEO_EXTENSIONS))

        self._add_path_row(paths, 1, "Output folder:", self.output_folder_var,
                           lambda: self._browse_folder(self.output_folder_var),
                           "Where the inpainted videos are written.", folder_only=True)
        register_dnd_entries(self._dnd_entries, self._dnd_enabled)

        # --- Parameters ---
        params = ttk.LabelFrame(self, text="Parameters")
        params.pack(fill="x", **pad)
        for c in (1, 3):
            params.columnconfigure(c, weight=1)
        self._add_param_row(params, 0, 0, "Frames / chunk:", self.frames_chunk_var,
                            "Frames processed per temporal chunk (e.g. 81).")
        self._add_param_row(params, 0, 2, "Frames overlap:", self.frames_overlap_var,
                            "Overlapping frames between consecutive chunks for temporal consistency.")
        self._add_param_row(params, 1, 0, "Tile overlap (px):", self.tile_overlap_var,
                            "Pixel overlap between spatial tiles when tiling is enabled.")
        self._add_param_row(params, 1, 2, "Tiles / axis:", self.tile_num_var,
                            "Number of spatial tiles per axis. 1 disables tiling.")
        self._add_param_row(params, 2, 0, "Inference steps:", self.inference_steps_var,
                            "Number of denoising steps. More = slower, potentially higher quality.")
        self._add_param_row(params, 2, 2, "Seed:", self.seed_var,
                            "Random seed for reproducible results.")
        resume_check = ttk.Checkbutton(params, text="Resume", variable=self.move_to_finished_var,
                                       command=self.save_config)
        resume_check.grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        Tooltip(resume_check, ("After a video finishes successfully, move it into a 'finished' subfolder "
                               "of the input folder so re-running the batch skips it. "
                               "Use File > Restore Finished to move files back."))

        # --- VRAM / Performance ---
        vram = ttk.LabelFrame(self, text="VRAM / Performance")
        vram.pack(fill="x", **pad)
        offload_tip = ("CPU offloading to save VRAM.\n"
                       "None: all models stay on the GPU (fastest, needs the VRAM).\n"
                       "Transformer only: stream the 14B transformer CPU<->GPU on demand; "
                       "text encoder + VAE stay on the GPU. Saves ~12 GB of system RAM vs "
                       "Group offload but needs ~24 GB+ VRAM.\n"
                       "Group offload: stream everything CPU<->GPU on demand "
                       "(lowest VRAM, highest system RAM use, slower).\n"
                       "FP8 resident: load the pre-quantized FP8 transformer fully onto the "
                       "GPU (needs ~20 GB+ VRAM, e.g. RTX 4090/5090; no PCIe streaming = "
                       "fastest on such cards). Requires the StereoCrafter2-FP8 folder from "
                       "export_fp8_transformer.py.")
        off_lbl = ttk.Label(vram, text="Offload mode:")
        off_lbl.grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.offload_combo = ttk.Combobox(vram, textvariable=self.offload_mode_var,
                                          values=["None", "Transformer only", "Group offload", "FP8 resident"],
                                          state="readonly", width=16)
        self.offload_combo.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        for w in (off_lbl, self.offload_combo):
            Tooltip(w, offload_tip)

        sage_check = ttk.Checkbutton(vram, text="Use SageAttention", variable=self.use_sageattention_var)
        sage_check.grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        Tooltip(sage_check, ("Run the transformer's attention through the quantized SageAttention "
                             "kernel - much faster at Wan's long sequence lengths, near-exact "
                             "quality. Uncheck to A/B against standard SDPA attention. "
                             "Takes effect when models are (re)loaded at the next Start."))

        # --- Controls ---
        controls = ttk.Frame(self)
        controls.pack(fill="x", **pad)
        self.start_button = ttk.Button(controls, text="Start", command=self.start_processing)
        self.start_button.pack(side="left", padx=2)
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop_processing, state="disabled")
        self.stop_button.pack(side="left", padx=2)

        # --- Progress / status ---
        prog_frame = ttk.Frame(self)
        prog_frame.pack(fill="x", **pad)
        self.progress_bar = ttk.Progressbar(prog_frame, variable=self.progress_var, maximum=100.0)
        self.progress_bar.pack(fill="x", side="top", pady=(0, 2))
        self.status_label = ttk.Label(prog_frame, textvariable=self.status_var, anchor="w")
        self.status_label.pack(fill="x", side="top")

        # --- Log pane ---
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_widget = scrolledtext.ScrolledText(log_frame, height=10, state="disabled", wrap="word")
        self.log_widget.pack(fill="both", expand=True, padx=4, pady=4)

    def _create_menubar(self):
        self.menubar = tk.Menu(self)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label="Load settings...", command=self.load_settings)
        self.file_menu.add_command(label="Save settings...", command=self.save_settings)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Restore Finished", command=self.restore_finished_files)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.exit_application)
        self.menubar.add_cascade(label="File", menu=self.file_menu)

        self.options_menu = tk.Menu(self.menubar, tearoff=0)
        self.options_menu.add_checkbutton(
            label="Dark mode", variable=self.dark_mode_var, command=self.toggle_dark_mode
        )
        self.menubar.add_cascade(label="Options", menu=self.options_menu)

        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.help_menu.add_command(label="About", command=self._show_about)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)

        self.config(menu=self.menubar)

    def _add_path_row(self, parent, row, label, var, browse_cmd, tooltip,
                      folder_only=False, extensions=None):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", padx=4, pady=2)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        btn = ttk.Button(parent, text="Browse", command=browse_cmd, width=8)
        btn.grid(row=row, column=2, padx=4, pady=2)
        Tooltip(lbl, tooltip)
        Tooltip(entry, tooltip)
        self._dnd_entries.append((entry, var, folder_only, extensions))
        return entry

    def _add_param_row(self, parent, row, col, label, var, tooltip):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=col, sticky="w", padx=4, pady=2)
        entry = ttk.Entry(parent, textvariable=var, width=10)
        entry.grid(row=row, column=col + 1, sticky="ew", padx=4, pady=2)
        Tooltip(lbl, tooltip)
        Tooltip(entry, tooltip)
        return entry

    # --------------------------------------------------------------- browsing
    def _browse_folder(self, var):
        folder = filedialog.askdirectory(initialdir=var.get() or ".")
        if folder:
            var.set(folder)

    def _browse_input_file(self):
        current = self.input_folder_var.get()
        initial = current if os.path.isdir(current) else os.path.dirname(current)
        path = filedialog.askopenfilename(
            initialdir=initial or ".",
            title="Select a video file",
            filetypes=[("Video files", " ".join(f"*{e}" for e in VIDEO_EXTENSIONS)), ("All files", "*.*")],
        )
        if path:
            self.input_folder_var.set(path)

    def _browse_input_folder(self):
        current = self.input_folder_var.get()
        initial = current if os.path.isdir(current) else os.path.dirname(current)
        folder = filedialog.askdirectory(initialdir=initial or ".")
        if folder:
            self.input_folder_var.set(folder)

    # ------------------------------------------------------------------ theme
    def _apply_theme(self, is_startup=False):
        self.theme_manager.apply_theme_to_style(self.style, root_window=self)
        self.theme_manager.apply_theme_to_menus(
            [self.file_menu, self.options_menu, self.help_menu], self.menubar
        )
        colors = self.theme_manager.get_colors()
        # tk.Text widgets aren't ttk-themed; style them directly.
        log_widget = getattr(self, "log_widget", None)
        if log_widget is not None:
            try:
                log_widget.configure(bg=colors["entry_bg"], fg=colors["fg"], insertbackground=colors["fg"])
            except tk.TclError:
                pass
        configure_dnd_styles(self.style, self.dark_mode_var.get(), self._dnd_enabled)

    def toggle_dark_mode(self):
        self._apply_theme()

    def _show_about(self):
        messagebox.showinfo(
            "About",
            f"StereoCrafter Wan VACE Inpainting GUI\nVersion {GUI_VERSION}\n\n"
            "Stereo SBS video inpainting (Wan VACE pipeline).",
        )

    # ----------------------------------------------------------------- config
    def load_config(self):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse %s: %s", CONFIG_FILE, e)
            return {}

    def _get_current_config(self):
        return {
            "dark_mode_enabled": self.dark_mode_var.get(),
            "input_folder": self.input_folder_var.get(),
            "output_folder": self.output_folder_var.get(),
            "frames_chunk": self.frames_chunk_var.get(),
            "frames_overlap": self.frames_overlap_var.get(),
            "tile_overlap": self.tile_overlap_var.get(),
            "tile_num": self.tile_num_var.get(),
            "inference_steps": self.inference_steps_var.get(),
            "seed": self.seed_var.get(),
            "offload_mode": self.offload_mode_var.get(),
            "use_sageattention": self.use_sageattention_var.get(),
            "move_to_finished": self.move_to_finished_var.get(),
            "window_geometry": self.geometry(),
        }

    def save_config(self):
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self._get_current_config(), f, indent=4)
            logger.info("Configuration saved.")
        except Exception as e:
            logger.warning("Failed to save config: %s", e)

    def restore_finished_files(self):
        """Moves all files from the 'finished' folder back to the input folder."""
        if not messagebox.askyesno(
            "Restore Finished Files",
            "Are you sure you want to move all processed videos from the 'finished' "
            "folder back to the input folder?",
        ):
            return
        input_path = self.input_folder_var.get().strip()
        input_folder = input_path if os.path.isdir(input_path) else os.path.dirname(input_path)
        if not input_folder or not os.path.isdir(input_folder):
            messagebox.showerror("Restore Finished", "Input folder does not exist.")
            return
        restored_count, errors_count, _failed = _restore_finished_files([(input_folder, "finished")], logger=logger)
        messagebox.showinfo(
            "Restore Finished",
            f"Finished files restoration attempted.\n{restored_count} files moved.\n{errors_count} errors occurred.",
        )

    def load_settings(self):
        filename = filedialog.askopenfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json")], title="Load settings"
        )
        if not filename:
            return
        try:
            with open(filename, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for key, value in cfg.items():
                var = getattr(self, key + "_var", None)
                if isinstance(var, tk.BooleanVar):
                    var.set(bool(value))
                elif isinstance(var, tk.StringVar):
                    var.set(str(value))
            self._apply_theme()
            self.update_status(f"Loaded settings from {os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Load error", f"Failed to load settings:\n{e}")

    def save_settings(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json")], title="Save settings"
        )
        if not filename:
            return
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self._get_current_config(), f, indent=4)
            self.update_status(f"Saved settings to {os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save settings:\n{e}")

    def _set_saved_geometry(self):
        geom = self.window_geometry
        if geom and self._geometry_visible(geom):
            try:
                self.geometry(geom)
                return
            except tk.TclError:
                pass
        self.geometry(f"{self.default_width}x{self.winfo_reqheight()}")

    def _geometry_visible(self, geom):
        """True if the saved "WxH+X+Y" leaves a usable chunk of the title bar on screen.

        Guards against restoring onto a monitor that is no longer connected. Uses the
        virtual desktop bounds so multi-monitor (including negative coords) is honored."""
        match = re.match(r"^(\d+)x(\d+)([+-]\d+)([+-]\d+)$", geom.strip())
        if not match:
            return False
        w, h, x, y = (int(g) for g in match.groups())
        vx = self.winfo_vrootx()
        vy = self.winfo_vrooty()
        vw = self.winfo_vrootwidth() or self.winfo_screenwidth()
        vh = self.winfo_vrootheight() or self.winfo_screenheight()
        margin = 80  # keep at least this many px of the window grabbable
        if x + w < vx + margin or x > vx + vw - margin:
            return False
        if y + h < vy + margin or y > vy + vh - margin:
            return False
        return True

    # ----------------------------------------------------------------- logging
    def _configure_logging(self):
        handler = TextHandler(self._ui_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def _append_log(self, msg):
        try:
            self.log_widget.configure(state="normal")
            self.log_widget.insert("end", msg + "\n")
            self.log_widget.see("end")
            self.log_widget.configure(state="disabled")
        except tk.TclError:
            pass

    def _poll_queue(self):
        """Drain worker->UI events on the Tk main thread, then reschedule."""
        try:
            while True:
                item = self._ui_queue.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._append_log(item[1])
                elif kind == "progress":
                    _, current, total, message = item
                    denom = total or 1
                    self.progress_var.set(max(0.0, min(100.0, 100.0 * current / denom)))
                    self.status_var.set(f"{self._batch_prefix}  {message}".strip())
                elif kind == "done":
                    self._processing_done(item[1], item[2])
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_queue)

    def update_status(self, message):
        self.status_var.set(message)

    # -------------------------------------------------------------- processing
    def _gather_videos(self, input_path):
        """Resolve the input field to a sorted list of video files."""
        if os.path.isfile(input_path):
            return [input_path]
        if os.path.isdir(input_path):
            return [
                os.path.join(input_path, name)
                for name in sorted(os.listdir(input_path))
                if name.lower().endswith(VIDEO_EXTENSIONS)
            ]
        return []

    def _validate_and_gather(self):
        """Validate all fields. Returns (videos, params, output_dir, prompt) or None."""
        input_path = self.input_folder_var.get().strip()
        output_dir = self.output_folder_var.get().strip()

        if not os.path.isdir(PRE_TRAINED_PATH):
            messagebox.showerror("Missing models", f"Base model folder not found:\n{PRE_TRAINED_PATH}")
            return None
        if not os.path.isdir(TRANSFORMER_PATH):
            messagebox.showerror("Missing models", f"Transformer folder not found:\n{TRANSFORMER_PATH}")
            return None
        if resolve_offload(self.offload_mode_var.get()) == "fp8":
            fp8_dir = TRANSFORMER_PATH.rstrip("/\\") + "-FP8"
            if not os.path.isfile(os.path.join(fp8_dir, FP8_STATE_FILE)):
                messagebox.showerror(
                    "FP8 checkpoint missing",
                    f"FP8 resident mode needs the pre-quantized transformer at:\n{fp8_dir}\n\n"
                    "Run 'uv run python export_fp8_transformer.py' on a machine with ~64 GB RAM, "
                    "or copy the folder from a machine that has.",
                )
                return None
            if torch.cuda.is_available():
                total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
                if total_gib < 20:
                    messagebox.showerror(
                        "Not enough VRAM",
                        f"FP8 resident mode needs ~20 GB VRAM; this GPU has {total_gib:.0f} GB.\n"
                        "Use 'Transformer only' or 'Group offload' instead.",
                    )
                    return None
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Invalid path", "Input folder/file does not exist.")
            return None

        videos = self._gather_videos(input_path)
        if videos and os.path.isdir(input_path) and self.move_to_finished_var.get():
            finished_dir = os.path.join(input_path, "finished")
            if os.path.isdir(finished_dir):
                finished_names = set(os.listdir(finished_dir))
                remaining = [v for v in videos if os.path.basename(v) not in finished_names]
                if len(remaining) < len(videos):
                    logger.info(
                        "Resume mode: skipped %d already processed video(s) found in 'finished'.",
                        len(videos) - len(remaining),
                    )
                if not remaining:
                    messagebox.showinfo(
                        "All videos processed",
                        "All input videos were already processed (found in the 'finished' folder).\n"
                        "Use File > Restore Finished to re-process them.",
                    )
                    return None
                videos = remaining
        if not videos:
            messagebox.showerror(
                "No videos",
                "No video files found at the input path.\n"
                f"Supported extensions: {', '.join(VIDEO_EXTENSIONS)}",
            )
            return None

        if not output_dir:
            messagebox.showerror("Invalid path", "Please choose an output folder.")
            return None
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            messagebox.showerror("Invalid path", f"Cannot create output folder:\n{e}")
            return None

        # Numeric parameters
        specs = [
            ("frames_chunk", self.frames_chunk_var, 1),
            ("frames_overlap", self.frames_overlap_var, 0),
            ("tile_overlap", self.tile_overlap_var, 0),
            ("tile_num", self.tile_num_var, 1),
            ("inference_steps", self.inference_steps_var, 1),
            ("seed", self.seed_var, None),
        ]
        params = {}
        for name, var, minimum in specs:
            raw = var.get().strip()
            try:
                value = int(raw)
            except ValueError:
                messagebox.showerror("Invalid value", f"'{name}' must be an integer (got '{raw}').")
                return None
            if minimum is not None and value < minimum:
                messagebox.showerror("Invalid value", f"'{name}' must be >= {minimum} (got {value}).")
                return None
            params[name] = value

        if params["frames_overlap"] >= params["frames_chunk"]:
            messagebox.showerror("Invalid value", "'frames_overlap' must be smaller than 'frames_chunk'.")
            return None

        return videos, params, output_dir

    def start_processing(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
        validated = self._validate_and_gather()
        if validated is None:
            return
        videos, params, output_dir = validated
        prompt = PROMPT  # prompt input removed from the GUI; always use the default

        # Model paths are hardcoded (not exposed in the GUI).
        pre_trained = PRE_TRAINED_PATH
        transformer = TRANSFORMER_PATH

        # Input layout is always auto-detected per file (see detect_dual_input).
        is_dual_input = None
        offload_mode = resolve_offload(self.offload_mode_var.get())
        # Only batch (folder) inputs move to 'finished'; an explicitly chosen
        # single file is left in place.
        input_path = self.input_folder_var.get().strip()
        move_finished = self.move_to_finished_var.get() and os.path.isdir(input_path)
        if self.move_to_finished_var.get() and not move_finished:
            logger.info("Single-file input: processed file will be left in place (no move to 'finished').")

        self.stop_event.clear()
        self._set_running_state(True)
        self.progress_var.set(0.0)
        logger.info("Starting batch: %d video(s) -> %s", len(videos), output_dir)

        self.worker_thread = threading.Thread(
            target=self._run_batch,
            args=(videos, params, output_dir, prompt, pre_trained, transformer,
                  is_dual_input, offload_mode, self.use_sageattention_var.get(),
                  move_finished),
            daemon=True,
        )
        self.worker_thread.start()

    def _run_batch(self, videos, params, output_dir, prompt, pre_trained, transformer,
                   is_dual_input, offload_mode, use_sageattention, move_finished):
        """Worker thread: load models once, then inpaint each video.

        Must not touch Tk widgets/variables directly — all UI updates go through
        ``self._ui_queue`` (drained by ``_poll_queue`` on the main thread).
        """
        succeeded, failed = 0, 0
        try:
            engine_key = (pre_trained, transformer, offload_mode, use_sageattention)
            if self.engine is None or self._engine_paths != engine_key:
                # Drop any previous engine first so its VRAM is freed before reloading.
                self.engine = None
                release_cuda_memory()
                gc.collect()
                self._on_progress("load_models", 0, 1, "Loading models...")
                self.engine = load_models(
                    pre_trained, transformer,
                    offload_mode=offload_mode,
                    use_sageattention=use_sageattention,
                    progress_cb=self._on_progress,
                )
                self._engine_paths = engine_key

            total = len(videos)
            for idx, video in enumerate(videos, 1):
                if self.stop_event.is_set():
                    break
                self._batch_prefix = f"[{idx}/{total}] {os.path.basename(video)}"
                completed = False
                try:
                    run_inpaint(
                        self.engine, video, output_dir, prompt=prompt,
                        is_dual_input=is_dual_input,
                        progress_cb=self._on_progress, should_stop=self.stop_event.is_set,
                        **params,
                    )
                    succeeded += 1
                    completed = True
                except InferenceCancelled:
                    logger.warning("Cancelled during %s.", os.path.basename(video))
                    break
                except Exception as e:
                    failed += 1
                    logger.error("Failed on %s: %s", os.path.basename(video), e, exc_info=True)
                finally:
                    release_cuda_memory()
                    gc.collect()
                if completed and move_finished:
                    move_files_to_finished(
                        files_to_move=[(video, os.path.dirname(video))],
                        logger=logger, wait_before_move=0.5,
                    )
        except Exception as e:
            logger.error("Fatal error (model load?): %s", e, exc_info=True)
        finally:
            self._ui_queue.put(("done", succeeded, failed))

    def _on_progress(self, stage, current, total, message=""):
        """Thread-safe progress callback passed into the engine."""
        self._ui_queue.put(("progress", current, total, message))

    def _set_running_state(self, running):
        self.start_button.config(state="disabled" if running else "normal")
        self.stop_button.config(state="normal" if running else "disabled")

    def _processing_done(self, succeeded=0, failed=0):
        self._set_running_state(False)
        self._batch_prefix = ""
        if self.stop_event.is_set():
            self.update_status(f"Stopped. {succeeded} done, {failed} failed.")
        else:
            self.progress_var.set(100.0)
            self.update_status(f"Finished. {succeeded} done, {failed} failed.")
        logger.info("Batch complete: %d succeeded, %d failed.", succeeded, failed)

    def stop_processing(self):
        self.stop_event.set()
        self.update_status("Stopping (finishing current step)...")

    # ------------------------------------------------------------------- exit
    def exit_application(self):
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.stop_event.set()
        self.save_config()
        self.destroy()


def launch_gui():
    app = WanInpaintingGUI()
    app.mainloop()


if __name__ == "__main__":
    # GUI is the default entrypoint; pass --cli to use the Fire command line.
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        Fire(main)
    else:
        launch_gui()
