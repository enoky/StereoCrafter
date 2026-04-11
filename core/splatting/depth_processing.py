"""Depth map processing utilities for StereoCrafter.

Provides functions and classes for reading, processing, and analyzing
depth maps including 10-bit+ depth support via FFmpeg.
"""

import logging
import math
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Optional, Tuple, Any

from core.common.video_io import VideoReader, read_video_frames

logger = logging.getLogger(__name__)

# Constants for TV-range depth map normalization
DEPTH_VIS_TV10_BLACK_NORM = 64.0 / 1023.0
DEPTH_VIS_TV10_WHITE_NORM = 940.0 / 1023.0
DEPTH_VIS_APPLY_TV_RANGE_EXPANSION_10BIT = False


def custom_dilate(
    tensor: torch.Tensor,
    kernel_size_x: float,
    kernel_size_y: float,
    use_gpu: bool = False,
    max_content_value: float = 1.0,
) -> torch.Tensor:
    """Applies 16-bit fractional dilation or erosion to preserve 10-bit+ depth fidelity."""
    kx_raw = float(kernel_size_x)
    ky_raw = float(kernel_size_y)

    if abs(kx_raw) <= 1e-5 and abs(ky_raw) <= 1e-5:
        return tensor

    if (kx_raw > 0 and ky_raw < 0) or (kx_raw < 0 and ky_raw > 0):
        tensor = custom_dilate(tensor, kx_raw, 0, use_gpu, max_content_value)
        return custom_dilate(tensor, 0, ky_raw, use_gpu, max_content_value)

    is_erosion = kx_raw < 0 or ky_raw < 0
    kx_abs, ky_abs = abs(kx_raw), abs(ky_raw)

    def get_dilation_params(value):
        if value <= 1e-5:
            return 1, 1, 0.0
        elif value < 3.0:
            return 1, 3, (value / 3.0)
        else:
            base = 3 + 2 * int((value - 3) // 2)
            return base, base + 2, (value - base) / 2.0

    kx_low, kx_high, tx = get_dilation_params(kx_abs)
    ky_low, ky_high, ty = get_dilation_params(ky_abs)

    device = torch.device("cpu")
    tensor_cpu = tensor.to(device)
    processed_frames = []

    for t in range(tensor_cpu.shape[0]):
        frame_float = tensor_cpu[t].numpy()
        frame_2d_raw = frame_float[0] if frame_float.shape[0] == 1 else np.transpose(frame_float, (1, 2, 0))
        effective_max_value = max(max_content_value, 1e-5)

        src_img = np.ascontiguousarray(
            np.clip((frame_2d_raw / effective_max_value) * 65535, 0, 65535).astype(np.uint16)
        )

        def do_op(k_w, k_h, img):
            if k_w <= 1 and k_h <= 1:
                return img.astype(np.float32)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
            if is_erosion:
                return cv2.erode(img, kernel, iterations=1).astype(np.float32)
            return cv2.dilate(img, kernel, iterations=1).astype(np.float32)

        is_x_int, is_y_int = (tx <= 1e-4), (ty <= 1e-4)
        if is_x_int and is_y_int:
            final_float = do_op(kx_low, ky_low, src_img)
        elif not is_x_int and is_y_int:
            final_float = (1.0 - tx) * do_op(kx_low, ky_low, src_img) + tx * do_op(kx_high, ky_low, src_img)
        elif is_x_int and not is_y_int:
            final_float = (1.0 - ty) * do_op(kx_low, ky_low, src_img) + ty * do_op(kx_low, ky_high, src_img)
        else:
            r11, r12 = do_op(kx_low, ky_low, src_img), do_op(kx_low, ky_high, src_img)
            r21, r22 = do_op(kx_high, ky_low, src_img), do_op(kx_high, ky_high, src_img)
            final_float = (1.0 - tx) * ((1.0 - ty) * r11 + ty * r12) + tx * ((1.0 - ty) * r21 + ty * r22)

        processed_raw = (final_float / 65535.0) * effective_max_value
        processed_frames.append(torch.from_numpy(processed_raw).unsqueeze(0).float())

    return torch.stack(processed_frames).to(tensor.device)


def custom_dilate_left(
    tensor: torch.Tensor, kernel_size: float, use_gpu: bool = False, max_content_value: float = 1.0
) -> torch.Tensor:
    """Directional 16-bit fractional dilation to the LEFT."""
    k_raw = float(kernel_size)
    if abs(k_raw) <= 1e-5:
        return tensor

    is_erosion = k_raw < 0
    k_raw = abs(k_raw)

    def get_dilation_params(value: float):
        if value <= 1e-5:
            return 1, 1, 0.0
        elif value < 3.0:
            return 1, 3, (value / 3.0)
        else:
            base = 3 + 2 * int((value - 3) // 2)
            return base, base + 2, (value - base) / 2.0

    k_w_low, k_w_high, t = get_dilation_params(k_raw)
    k_low, k_high = int(k_w_low // 2), int(k_w_high // 2)

    if k_low <= 0 and k_high <= 0:
        return tensor

    effective_max_value = max(float(max_content_value), 1e-5)
    device = torch.device("cpu")
    tensor = tensor.to(device)

    def do_op(k_int: int, src_img: np.ndarray) -> np.ndarray:
        if k_int <= 0:
            return src_img.astype(np.float32)
        k_w = int(k_int) + 1
        kernel = np.ones((1, k_w), dtype=np.uint8)
        anchor = (0, 0)
        if is_erosion:
            return cv2.erode(src_img, kernel, anchor=anchor, iterations=1).astype(np.float32)
        return cv2.dilate(src_img, kernel, anchor=anchor, iterations=1).astype(np.float32)

    processed_frames = []
    for t_idx in range(tensor.shape[0]):
        frame_float = tensor[t_idx].cpu().numpy()
        frame_2d_raw = frame_float[0] if frame_float.shape[0] == 1 else np.transpose(frame_float, (1, 2, 0))
        frame_norm_2d = frame_2d_raw / effective_max_value
        frame_cv_uint16 = np.ascontiguousarray(np.clip(frame_norm_2d * 65535, 0, 65535).astype(np.uint16))
        src = frame_cv_uint16.astype(np.float32)

        if abs(t) <= 1e-4:
            out = do_op(k_low, src)
        else:
            out_low, out_high = do_op(k_low, src), do_op(k_high, src)
            out = (1.0 - t) * out_low + t * out_high

        out_u16 = np.ascontiguousarray(np.clip(out, 0, 65535).astype(np.uint16))
        out_float = (out_u16.astype(np.float32) / 65535.0) * effective_max_value
        processed_frames.append(torch.from_numpy(out_float).unsqueeze(0).float())

    return torch.stack(processed_frames).to(tensor.device)


def custom_blur(
    tensor: torch.Tensor, kernel_size_x: int, kernel_size_y: int, use_gpu: bool = True, max_content_value: float = 1.0
) -> torch.Tensor:
    """Applies 16-bit Gaussian blur."""
    k_x, k_y = int(kernel_size_x), int(kernel_size_y)
    if k_x <= 0 and k_y <= 0:
        return tensor

    k_x = k_x if k_x % 2 == 1 else k_x + 1
    k_y = k_y if k_y % 2 == 1 else k_y + 1

    device = torch.device("cpu")
    tensor = tensor.to(device)
    processed_frames = []

    for t in range(tensor.shape[0]):
        frame_float = tensor[t].cpu().numpy()
        frame_2d_raw = frame_float[0] if frame_float.shape[0] == 1 else np.transpose(frame_float, (1, 2, 0))
        effective_max_value = max(max_content_value, 1e-5)
        frame_norm_2d = frame_2d_raw / effective_max_value
        frame_cv_uint16 = np.ascontiguousarray(np.clip(frame_norm_2d * 65535, 0, 65535).astype(np.uint16))
        processed_cv_uint16 = cv2.GaussianBlur(frame_cv_uint16, (k_x, k_y), 0)
        processed_norm_float = processed_cv_uint16.astype(np.float32) / 65535.0
        processed_raw_float = processed_norm_float * effective_max_value
        processed_frames.append(torch.from_numpy(processed_raw_float).unsqueeze(0).float())

    return torch.stack(processed_frames).to(tensor.device)


def process_depth_batch(
    batch_depth_numpy_raw: np.ndarray,
    depth_gamma: float,
    depth_dilate_size_x: float,
    depth_dilate_size_y: float,
    depth_blur_size_x: float,
    depth_blur_size_y: float,
    max_raw_value: float,
    depth_dilate_left: float = 0.0,
    depth_blur_left: float = 0.0,
    depth_blur_left_mix: float = 0.5,
    skip_preprocessing: bool = False,
    debug_task_name: str = "Render",
) -> np.ndarray:
    """Unified depth processor for batch of depth maps."""
    # from dependency.stereocrafter_util import log_debug_args
    # log_debug_args(locals(), "process_depth_batch", "depth_processing")

    if batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 3:
        batch_depth_numpy = batch_depth_numpy_raw.mean(axis=-1)
    else:
        batch_depth_numpy = (
            batch_depth_numpy_raw.squeeze(-1) if batch_depth_numpy_raw.ndim == 4 else batch_depth_numpy_raw
        )

    batch_depth_float = batch_depth_numpy.astype(np.float32)

    # Standardize input for logging (4D)
    if batch_depth_numpy_raw.ndim == 3:
        batch_depth_log_input = batch_depth_numpy_raw[..., None]
    else:
        batch_depth_log_input = batch_depth_numpy_raw

    # from dependency.stereocrafter_util import dump_debug_tensor
    # dump_debug_tensor(batch_depth_log_input, f"{debug_task_name}_0_raw_depth", "depth_processing")

    if skip_preprocessing:
        return batch_depth_float

    current_width = (
        batch_depth_numpy_raw.shape[2] if batch_depth_numpy_raw.ndim == 4 else batch_depth_numpy_raw.shape[1]
    )
    res_scale = math.sqrt(current_width / 960.0)

    def map_val(v):
        f_v = float(v)
        if f_v > 30.0 and f_v <= 40.0:
            return -(f_v - 30.0)
        return f_v

    render_dilate_x = map_val(depth_dilate_size_x) * res_scale
    render_dilate_y = map_val(depth_dilate_size_y) * res_scale
    render_blur_x, render_blur_y = depth_blur_size_x * res_scale, depth_blur_size_y * res_scale
    render_dilate_left, render_blur_left = float(depth_dilate_left) * res_scale, float(depth_blur_left) * res_scale

    if (
        abs(render_dilate_left) > 1e-5
        or render_blur_left > 0
        or abs(render_dilate_x) > 1e-5
        or abs(render_dilate_y) > 1e-5
        or render_blur_x > 0
        or render_blur_y > 0
    ):
        device = torch.device("cpu")
        tensor_4d = torch.from_numpy(batch_depth_float).unsqueeze(1).to(device)

        if abs(render_dilate_left) > 1e-5:
            tensor_4d = custom_dilate_left(tensor_4d, float(render_dilate_left), False, max_raw_value)

        if render_blur_left > 0:
            effective_max_value = max(max_raw_value, 1e-5)
            EDGE_STEP_8BIT = 3.0
            step_thresh = effective_max_value * (EDGE_STEP_8BIT / 255.0)
            dx = tensor_4d[:, :, :, 1:] - tensor_4d[:, :, :, :-1]
            edge_core = dx > step_thresh
            edge_mask = torch.zeros_like(tensor_4d, dtype=torch.float32)
            edge_mask[:, :, :, 1:] = edge_core.float()

            k_blur = int(round(render_blur_left))
            k_blur = k_blur if k_blur % 2 == 1 else k_blur + 1
            band_half = max(1, int(math.ceil(k_blur / 4.0)))
            edge_band = (
                F.max_pool2d(edge_mask, kernel_size=(1, 2 * band_half + 1), stride=1, padding=(0, band_half)) > 0.5
            ).float()
            alpha = torch.clamp(custom_blur(edge_band, 7, 1, False, 1.0), 0.0, 1.0)

            mix_f = max(0.0, min(1.0, float(depth_blur_left_mix)))
            BLUR_LEFT_V_WEIGHT, BLUR_LEFT_H_WEIGHT = mix_f, 1.0 - mix_f

            blurred_h = custom_blur(tensor_4d, k_blur, 1, False, max_raw_value) if BLUR_LEFT_H_WEIGHT > 1e-6 else None
            blurred_v = custom_blur(tensor_4d, 1, k_blur, False, max_raw_value) if BLUR_LEFT_V_WEIGHT > 1e-6 else None

            if blurred_h is not None and blurred_v is not None:
                blurred = (blurred_h * BLUR_LEFT_H_WEIGHT + blurred_v * BLUR_LEFT_V_WEIGHT) / max(
                    BLUR_LEFT_H_WEIGHT + BLUR_LEFT_V_WEIGHT, 1e-6
                )
            elif blurred_h is not None:
                blurred = blurred_h
            elif blurred_v is not None:
                blurred = blurred_v
            else:
                blurred = tensor_4d

            tensor_4d = tensor_4d * (1.0 - alpha) + blurred * alpha

        if abs(render_dilate_x) > 1e-5 or abs(render_dilate_y) > 1e-5:
            tensor_4d = custom_dilate(tensor_4d, float(render_dilate_x), float(render_dilate_y), False, max_raw_value)
        if render_blur_x > 0 or render_blur_y > 0:
            tensor_4d = custom_blur(tensor_4d, float(render_blur_x), float(render_blur_y), False, max_raw_value)

        batch_depth_float = tensor_4d.squeeze(1).cpu().numpy()
        del tensor_4d
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # from dependency.stereocrafter_util import dump_debug_tensor
    # dump_debug_tensor(batch_depth_float[..., None], f"{debug_task_name}_2_filtered_depth", "depth_processing")

    return batch_depth_float


def normalize_and_gamma_depth(
    batch_depth_numpy_raw: np.ndarray,
    assume_raw_input: bool,
    global_depth_max: float,
    global_depth_min: float,
    max_expected_raw_value: float,
    zero_disparity_anchor_val: float,
    depth_gamma: float,
    debug_task_name: str = "Render",
) -> np.ndarray:
    """Normalizes and applies gamma to a batch of raw depth frames."""
    # from dependency.stereocrafter_util import log_debug_args
    # log_debug_args(locals(), "normalize_and_gamma_depth", "depth_processing")

    if batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 3:
        batch_depth_gray = batch_depth_numpy_raw.mean(axis=-1)
    elif batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 1:
        batch_depth_gray = batch_depth_numpy_raw.squeeze(-1)
    else:
        batch_depth_gray = batch_depth_numpy_raw

    batch_depth_float = batch_depth_gray.astype(np.float32)
    curr_max = float(batch_depth_float.max())

    if assume_raw_input:
        # Scale/Unification Mode
        if global_depth_max > 1.05:
            batch_depth_normalized = batch_depth_float / global_depth_max
            logger.debug(f"[DEPTH] Raw Mode: Normalized by global_depth_max={global_depth_max}")
        elif curr_max > 1.05:
            # CONTENT IS RAW. Let's decide on the divisor.
            if curr_max > max_expected_raw_value * 1.5:
                # The expected bit-depth was likely wrong (e.g. 10-bit reported but 16-bit delivered)
                if curr_max <= 255.0:
                    divisor = 255.0
                elif curr_max <= 1024.0:
                    divisor = 1023.0
                elif curr_max <= 4096.0:
                    divisor = 4095.0
                else:
                    divisor = 65535.0
                logger.debug(
                    f"[DEPTH] Raw Mode: Auto-sizing divisor to {divisor} (ContentMax={curr_max:.1f}, ExpMax={max_expected_raw_value:.1f})"
                )
            else:
                divisor = max(max_expected_raw_value, 1.0)

            batch_depth_normalized = batch_depth_float / divisor
            logger.debug(f"[DEPTH] Raw Mode: Normalized by divisor={divisor}")
        else:
            batch_depth_normalized = batch_depth_float
            logger.debug(f"[DEPTH] Raw Mode: Input already normalized (max={curr_max:.3f})")
    else:
        # Global Normalization Mode (AutoGain)
        depth_range = global_depth_max - global_depth_min
        if depth_range > 1e-5:
            batch_depth_normalized = (batch_depth_float - global_depth_min) / depth_range
            logger.debug(f"[DEPTH] Global Norm Mode: Range [{global_depth_min:.2f}, {global_depth_max:.2f}]")
        else:
            batch_depth_normalized = np.full_like(
                batch_depth_float, fill_value=zero_disparity_anchor_val, dtype=np.float32
            )
            logger.debug(f"[DEPTH] Global Norm Mode: Range too small, using anchor={zero_disparity_anchor_val}")

    batch_depth_normalized = np.clip(batch_depth_normalized, 0.0, 1.0)

    if round(float(depth_gamma), 2) != 1.0:
        batch_depth_normalized = 1.0 - np.power(1.0 - batch_depth_normalized, depth_gamma)
        batch_depth_normalized = np.clip(batch_depth_normalized, 0.0, 1.0)

    # --- VITALS LOGGING ---
    # vital_max = float(batch_depth_normalized.max())
    # vital_min = float(batch_depth_normalized.min())
    # print(f"VITALS [{debug_task_name}]: InputMax={curr_max:.1f}, DivMax={global_depth_max:.1f}, ExpMax={max_expected_raw_value:.1f}, OutputRange=[{vital_min:.4f}, {vital_max:.4f}]")

    # from dependency.stereocrafter_util import dump_debug_tensor
    # dump_debug_tensor(batch_depth_normalized, "1_normalized_gamma_depth", "depth_processing")

    return batch_depth_normalized


def compute_global_depth_stats(
    depth_map_reader: VideoReader, total_frames: int, chunk_size: int = 100
) -> Tuple[float, float]:
    """Compute the global min and max depth values from a depth video.

    Reads the depth video in chunks to compute the overall min and max
    pixel values. Assumes raw pixel values that need to be scaled
    (e.g., from 0-255 or 0-1023 range).

    Args:
        depth_map_reader: Active VideoReader for the depth map
        total_frames: Total number of frames in the depth video
        chunk_size: Number of frames to process per chunk

    Returns:
        Tuple of (global_min, global_max) as float values
    """
    logger.info(f"==> Starting global depth stats pre-pass for {total_frames} frames...")
    global_min, global_max = np.inf, -np.inf

    for i in range(0, total_frames, chunk_size):
        current_indices = list(range(i, min(i + chunk_size, total_frames)))
        if not current_indices:
            break

        chunk_numpy_raw = depth_map_reader.get_batch(current_indices).asnumpy()

        # Handle RGB vs Grayscale depth maps
        if chunk_numpy_raw.ndim == 4:
            if chunk_numpy_raw.shape[-1] == 3:  # RGB
                chunk_numpy = chunk_numpy_raw.mean(axis=-1)
            else:  # Grayscale with channel dim
                chunk_numpy = chunk_numpy_raw.squeeze(-1)
        else:
            chunk_numpy = chunk_numpy_raw

        chunk_min = chunk_numpy.min()
        chunk_max = chunk_numpy.max()

        if chunk_min < global_min:
            global_min = chunk_min
        if chunk_max > global_max:
            global_max = chunk_max
    logger.info(f"==> Global depth stats computed: min_raw={global_min:.3f}, max_raw={global_max:.3f}")
    return float(global_min), float(global_max)


def load_pre_rendered_depth(
    depth_map_path: str, process_length: int, target_height: int, target_width: int, match_resolution_to_target: bool
) -> Tuple[Any, int, int, int, Optional[dict]]:
    """Initialize a reader for depth map reading using consolidated video_io tools."""
    logger.debug(f"==> Initializing depth reader from: {depth_map_path}")
    if depth_map_path.lower().endswith(".npz"):
        raise NotImplementedError("NPZ depth map loading is not yet supported with disk chunking.")

    (
        depth_reader,
        _fps,
        _orig_h,
        _orig_w,
        actual_depth_height,
        actual_depth_width,
        depth_stream_info,
        total_depth_frames_to_process,
    ) = read_video_frames(
        video_path=depth_map_path,
        process_length=process_length,
        set_pre_res=match_resolution_to_target,
        pre_res_width=target_width,
        pre_res_height=target_height,
        is_depth=True,
    )
    return (depth_reader, total_depth_frames_to_process, actual_depth_height, actual_depth_width, depth_stream_info)
