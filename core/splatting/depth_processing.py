"""Depth map processing utilities for StereoCrafter.

Provides functions and classes for reading, processing, and analyzing
depth maps including 10-bit+ depth support via FFmpeg.
"""

import logging
import os
import re
import subprocess
import math
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from decord import VideoReader, cpu
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Constants for TV-range depth map normalization
DEPTH_VIS_TV10_BLACK_NORM = 64.0 / 1023.0
DEPTH_VIS_TV10_WHITE_NORM = 940.0 / 1023.0
DEPTH_VIS_APPLY_TV_RANGE_EXPANSION_10BIT = True


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
        frame_2d_raw = (
            frame_float[0]
            if frame_float.shape[0] == 1
            else np.transpose(frame_float, (1, 2, 0))
        )
        effective_max_value = max(max_content_value, 1e-5)

        src_img = np.ascontiguousarray(
            np.clip((frame_2d_raw / effective_max_value) * 65535, 0, 65535).astype(
                np.uint16
            )
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
            final_float = (1.0 - tx) * do_op(kx_low, ky_low, src_img) + tx * do_op(
                kx_high, ky_low, src_img
            )
        elif is_x_int and not is_y_int:
            final_float = (1.0 - ty) * do_op(kx_low, ky_low, src_img) + ty * do_op(
                kx_low, ky_high, src_img
            )
        else:
            r11, r12 = do_op(kx_low, ky_low, src_img), do_op(kx_low, ky_high, src_img)
            r21, r22 = do_op(kx_high, ky_low, src_img), do_op(kx_high, ky_high, src_img)
            final_float = (1.0 - tx) * ((1.0 - ty) * r11 + ty * r12) + tx * (
                (1.0 - ty) * r21 + ty * r22
            )

        processed_raw = (final_float / 65535.0) * effective_max_value
        processed_frames.append(torch.from_numpy(processed_raw).unsqueeze(0).float())

    return torch.stack(processed_frames).to(tensor.device)


def custom_dilate_left(
    tensor: torch.Tensor,
    kernel_size: float,
    use_gpu: bool = False,
    max_content_value: float = 1.0,
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
        frame_2d_raw = (
            frame_float[0]
            if frame_float.shape[0] == 1
            else np.transpose(frame_float, (1, 2, 0))
        )
        frame_norm_2d = frame_2d_raw / effective_max_value
        frame_cv_uint16 = np.ascontiguousarray(
            np.clip(frame_norm_2d * 65535, 0, 65535).astype(np.uint16)
        )
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
    tensor: torch.Tensor,
    kernel_size_x: int,
    kernel_size_y: int,
    use_gpu: bool = True,
    max_content_value: float = 1.0,
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
        frame_2d_raw = (
            frame_float[0]
            if frame_float.shape[0] == 1
            else np.transpose(frame_float, (1, 2, 0))
        )
        effective_max_value = max(max_content_value, 1e-5)
        frame_norm_2d = frame_2d_raw / effective_max_value
        frame_cv_uint16 = np.ascontiguousarray(
            np.clip(frame_norm_2d * 65535, 0, 65535).astype(np.uint16)
        )
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
) -> np.ndarray:
    """Unified depth processor for batch of depth maps."""
    if batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 3:
        batch_depth_numpy = batch_depth_numpy_raw.mean(axis=-1)
    else:
        batch_depth_numpy = (
            batch_depth_numpy_raw.squeeze(-1)
            if batch_depth_numpy_raw.ndim == 4
            else batch_depth_numpy_raw
        )

    batch_depth_numpy_float = batch_depth_numpy.astype(np.float32)

    if skip_preprocessing:
        return batch_depth_numpy_float

    current_width = (
        batch_depth_numpy_raw.shape[2]
        if batch_depth_numpy_raw.ndim == 4
        else batch_depth_numpy_raw.shape[1]
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
        tensor_4d = torch.from_numpy(batch_depth_numpy_float).unsqueeze(1).to(device)

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
            edge_band = (F.max_pool2d(edge_mask, kernel_size=(1, 2 * band_half + 1), stride=1, padding=(0, band_half)) > 0.5).float()
            alpha = torch.clamp(custom_blur(edge_band, 7, 1, False, 1.0), 0.0, 1.0)

            mix_f = max(0.0, min(1.0, float(depth_blur_left_mix)))
            BLUR_LEFT_V_WEIGHT, BLUR_LEFT_H_WEIGHT = mix_f, 1.0 - mix_f

            blurred_h = custom_blur(tensor_4d, k_blur, 1, False, max_raw_value) if BLUR_LEFT_H_WEIGHT > 1e-6 else None
            blurred_v = custom_blur(tensor_4d, 1, k_blur, False, max_raw_value) if BLUR_LEFT_V_WEIGHT > 1e-6 else None

            if blurred_h is not None and blurred_v is not None:
                blurred = (blurred_h * BLUR_LEFT_H_WEIGHT + blurred_v * BLUR_LEFT_V_WEIGHT) / max(BLUR_LEFT_H_WEIGHT + BLUR_LEFT_V_WEIGHT, 1e-6)
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
        
        batch_depth_numpy_float = tensor_4d.squeeze(1).cpu().numpy()
        del tensor_4d
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return batch_depth_numpy_float


def compute_global_depth_stats(
    depth_map_reader: VideoReader,
    total_frames: int,
    chunk_size: int = 100,
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
    logger.info(
        f"==> Starting global depth stats pre-pass for {total_frames} frames..."
    )
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

    logger.info(
        f"==> Global depth stats computed: "
        f"min_raw={global_min:.3f}, max_raw={global_max:.3f}"
    )
    return float(global_min), float(global_max)


def _infer_depth_bit_depth(depth_stream_info: Optional[dict]) -> int:
    """Infer the bit depth of a depth stream from ffprobe info.

    Attempts to determine the bit depth of a depth video stream based
    on pixel format and profile information from ffprobe.

    Args:
        depth_stream_info: Dictionary containing stream info from ffprobe

    Returns:
        Inferred bit depth (8, 10, 12, 16, etc.)
    """
    if not depth_stream_info:
        return 8
    pix_fmt = str(depth_stream_info.get("pix_fmt", "")).lower()
    profile = str(depth_stream_info.get("profile", "")).lower()

    # Common patterns: yuv420p10le, yuv444p12le, gray16le, etc.
    m = re.search(r"(?:p|gray)(\d+)", pix_fmt)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    if "main10" in profile or "10" in profile:
        return 10
    return 8


def _build_depth_vf(pix_fmt: str, out_w: int, out_h: int) -> str:
    """Build FFmpeg video filter for depth map extraction.

    Creates an FFmpeg filter string to extract the depth (luma) plane
    from a video and convert it to gray16le format.

    Args:
        pix_fmt: Pixel format of the source video
        out_w: Output width
        out_h: Output height

    Returns:
        FFmpeg filter string for depth extraction
    """
    pix_fmt = (pix_fmt or "").lower()
    # If it's already gray*, don't try to extract planes.
    if pix_fmt.startswith("gray"):
        return f"scale={out_w}:{out_h}:flags=bilinear,format=gray16le"
    # Otherwise, extract luma plane (Y) and convert to gray16.
    return f"extractplanes=y,scale={out_w}:{out_h}:flags=bilinear,format=gray16le"


class _NumpyBatch:
    """Minimal wrapper to match Decord's get_batch(...).asnumpy() API."""

    def __init__(self, arr: np.ndarray):
        """Initialize with a numpy array.

        Args:
            arr: Numpy array to wrap
        """
        self._arr = arr

    def asnumpy(self) -> np.ndarray:
        """Return the underlying numpy array.

        Returns:
            The wrapped numpy array
        """
        return self._arr


class FFmpegDepthPipeReader:
    """Sequential FFmpeg-backed depth reader preserving 10-bit+ values.

    Implements a small subset of Decord's VideoReader API used by the
    splatter. Optimized for sequential access (render path).

    Supported operations:
        - __len__()
        - seek(idx)
        - get_batch([idx0, idx1, ...]).asnumpy()

    Note:
        This reader is optimized for sequential access patterns
        typical in the render path.
    """

    def __init__(
        self,
        path: str,
        out_w: int,
        out_h: int,
        bit_depth: int,
        num_frames: int,
        pix_fmt: str = "",
    ):
        """Initialize the FFmpeg depth pipe reader.

        Args:
            path: Path to the depth video file
            out_w: Output width for decoded frames
            out_h: Output height for decoded frames
            bit_depth: Bit depth of the depth values
            num_frames: Number of frames to decode
            pix_fmt: Pixel format of the source video
        """
        self.path = path
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.bit_depth = int(bit_depth) if bit_depth else 16
        self._num_frames = int(num_frames) if num_frames is not None else 0
        self._pix_fmt = pix_fmt or ""
        self._proc: Optional[subprocess.Popen] = None
        self._next_index = 0
        self._frame_bytes = self.out_w * self.out_h * 2  # gray16le
        self._msb_shift: Optional[int] = None
        self._use_16_to_n_scale: bool = False
        self._start_process()

    def __len__(self) -> int:
        """Return the number of frames.

        Returns:
            Total number of frames in the video
        """
        return self._num_frames

    def _start_process(self):
        """Start the FFmpeg process for depth decoding."""
        self.close()
        vf = _build_depth_vf(self._pix_fmt, self.out_w, self.out_h)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            self.path,
            "-an",
            "-sn",
            "-dn",
            "-vframes",
            str(self._num_frames)
            if self._num_frames and self._num_frames > 0
            else "999999999",
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._next_index = 0
        self._msb_shift = None

    def close(self):
        """Close the FFmpeg process and cleanup resources."""
        try:
            if self._proc is not None:
                if self._proc.stdout:
                    try:
                        self._proc.stdout.close()
                    except Exception:
                        pass
                if self._proc.stderr:
                    try:
                        self._proc.stderr.close()
                    except Exception:
                        pass
                try:
                    self._proc.terminate()
                except Exception:
                    pass
        finally:
            self._proc = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()

    def seek(self, idx: int):
        """Seek to a specific frame index.

        Best-effort seek. Efficient only if seeking forward sequentially.

        Args:
            idx: Target frame index
        """
        idx = int(idx)
        if idx == self._next_index:
            return
        if idx < self._next_index:
            # Restart and fast-forward by discarding frames
            self._start_process()
        # Discard until we reach idx
        to_skip = idx - self._next_index
        if to_skip <= 0:
            self._next_index = idx
            return
        if self._proc is None or self._proc.stdout is None:
            self._start_process()
        discard_bytes = to_skip * self._frame_bytes
        _ = self._proc.stdout.read(discard_bytes)
        self._next_index = idx

    def _read_exact(self, nbytes: int) -> bytes:
        """Read exactly nbytes from the FFmpeg pipe.

        Args:
            nbytes: Number of bytes to read

        Returns:
            Bytes read from the pipe

        Raises:
            EOFError: If unexpected end of data
        """
        if self._proc is None or self._proc.stdout is None:
            self._start_process()
        buf = b""
        while len(buf) < nbytes:
            chunk = self._proc.stdout.read(nbytes - len(buf))
            if not chunk:
                break
            buf += chunk
        return buf

    def _maybe_apply_shift(self, arr_u16: np.ndarray) -> np.ndarray:
        """Normalize decoded gray16 values back into native bit-depth range.

        FFmpeg's format conversion to gray16le may either:
          1. Left-shift the original N-bit values into the MSBs, or
          2. Scale values to the full 16-bit range.

        This method detects the behavior and applies the appropriate
        normalization.

        Args:
            arr_u16: Array of uint16 depth values

        Returns:
            Normalized depth array
        """
        if self._msb_shift is None:
            expected_max = (
                (1 << self.bit_depth) - 1 if 0 < self.bit_depth < 16 else None
            )
            if expected_max is None:
                self._msb_shift = 0
                self._use_16_to_n_scale = False
            else:
                # Sample to avoid full-frame scans on large batches
                flat = arr_u16.reshape(-1)
                step = max(1, flat.size // 50000)
                sample = flat[::step]
                max_val = int(sample.max(initial=0))

                if max_val <= expected_max:
                    self._msb_shift = 0
                    self._use_16_to_n_scale = False
                else:
                    shift = 16 - self.bit_depth
                    if shift <= 0:
                        self._msb_shift = 0
                        self._use_16_to_n_scale = False
                    else:
                        low_mask = (1 << shift) - 1
                        # If low bits are zero, it's MSB-aligned
                        low_bits_max = int((sample & low_mask).max(initial=0))
                        if low_bits_max == 0:
                            self._msb_shift = shift
                            self._use_16_to_n_scale = False
                        else:
                            # Scaled to 16-bit; scale back down
                            self._msb_shift = 0
                            self._use_16_to_n_scale = True

        expected_max = (1 << self.bit_depth) - 1 if 0 < self.bit_depth < 16 else None
        if expected_max is None:
            return arr_u16

        if self._msb_shift and self._msb_shift > 0:
            return (arr_u16 >> self._msb_shift).astype(np.uint16)

        if self._use_16_to_n_scale:
            # Map 0..65535 -> 0..expected_max with rounding
            arr32 = arr_u16.astype(np.uint32)
            return ((arr32 * expected_max + 32767) // 65535).astype(np.uint16)

        return arr_u16

    def get_batch(self, indices):
        """Get a batch of frames by their indices.

        Args:
            indices: List of frame indices to retrieve

        Returns:
            _NumpyBatch containing the requested frames
        """
        indices = list(indices)
        if not indices:
            return _NumpyBatch(
                np.zeros((0, self.out_h, self.out_w, 1), dtype=np.uint16)
            )

        # Render path calls are contiguous and increasing
        first = int(indices[0])
        if first != self._next_index:
            self.seek(first)

        n = len(indices)
        expected = self._frame_bytes * n
        buf = self._read_exact(expected)
        if len(buf) != expected:
            raise EOFError(
                f"FFmpegDepthPipeReader: expected {expected} bytes, got {len(buf)}"
            )

        arr = np.frombuffer(buf, dtype=np.uint16).reshape(
            n, self.out_h, self.out_w, 1
        )
        arr = self._maybe_apply_shift(arr)
        self._next_index = first + n
        return _NumpyBatch(arr.copy())


class _ResizingDepthReader:
    """Wrapper reader that resizes 8-bit depth frames to target resolution.

    Uses OpenCV resizing to ensure parity between preview and render paths.
    """

    def __init__(self, inner_reader, out_w, out_h):
        """Initialize with inner reader and target dimensions.

        Args:
            inner_reader: Inner VideoReader or similar
            out_w: Target output width
            out_h: Target output height
        """
        self._inner = inner_reader
        self._out_w = int(out_w)
        self._out_h = int(out_h)

    def __len__(self) -> int:
        """Return total number of frames.

        Returns:
            Frame count
        """
        return len(self._inner)

    def seek(self, *args, **kwargs):
        """Proxy seek to inner reader."""
        return self._inner.seek(*args, **kwargs)

    def get_batch(self, indices):
        """Get and resize frames.

        Args:
            indices: List of frame indices

        Returns:
            _NumpyBatch containing resized frames
        """
        arr = self._inner.get_batch(indices).asnumpy()
        # arr is typically (N,H,W,C) from Decord
        in_h = int(arr.shape[1])
        in_w = int(arr.shape[2])
        if in_w == self._out_w and in_h == self._out_h:
            return _NumpyBatch(arr)

        interp = (
            cv2.INTER_LINEAR
            if (self._out_w > in_w or self._out_h > in_h)
            else cv2.INTER_AREA
        )

        if arr.ndim == 4:
            out = np.empty(
                (arr.shape[0], self._out_h, self._out_w, arr.shape[3]),
                dtype=arr.dtype,
            )
            for i in range(arr.shape[0]):
                out[i] = cv2.resize(
                    arr[i], (self._out_w, self._out_h), interpolation=interp
                )
        else:
            out = np.empty(
                (arr.shape[0], self._out_h, self._out_w), dtype=arr.dtype
            )
            for i in range(arr.shape[0]):
                out[i] = cv2.resize(
                    arr[i], (self._out_w, self._out_h), interpolation=interp
                )

        return _NumpyBatch(out)


def load_pre_rendered_depth(
    depth_map_path: str,
    process_length: int,
    target_height: int,
    target_width: int,
    match_resolution_to_target: bool,
) -> Tuple[Any, int, int, int, Optional[dict]]:
    """Initialize a reader for chunked depth map reading.

    Preserves 10-bit+ depth maps by using an FFmpeg-backed reader when
    needed. No normalization or autogain is applied here.

    Args:
        depth_map_path: Path to the depth map video file
        process_length: Number of frames to process (-1 for all)
        target_height: Target height for output frames
        target_width: Target width for output frames
        match_resolution_to_target: Whether to resize to target resolution

    Returns:
        Tuple of (depth_reader, total_depth_frames_to_process,
                  actual_depth_height, actual_depth_width, depth_stream_info)

    Raises:
        NotImplementedError: If NPZ format is encountered
        ValueError: If unsupported format is encountered
    """
    # Import here to avoid circular dependency
    from dependency.stereocrafter_util import get_video_stream_info

    logger.debug(f"==> Initializing depth reader from: {depth_map_path}")

    depth_stream_info = get_video_stream_info(depth_map_path)
    bit_depth = _infer_depth_bit_depth(depth_stream_info)
    pix_fmt = str((depth_stream_info or {}).get("pix_fmt", ""))

    logger.info(
        f"==> Depth map stream: pix_fmt='{pix_fmt}', "
        f"profile='{str((depth_stream_info or {}).get('profile', ''))}', "
        f"inferred_bit_depth={bit_depth}"
    )

    if depth_map_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        # Determine total frames available
        total_depth_frames_available = 0
        try:
            nb = None
            if isinstance(depth_stream_info, dict):
                nb = depth_stream_info.get("nb_frames")
                if nb in (None, "", "N/A"):
                    nb = depth_stream_info.get("num_frames") or depth_stream_info.get(
                        "nb_read_frames"
                    )
            if nb not in (None, "", "N/A"):
                total_depth_frames_available = int(float(nb))
            else:
                _tmp = VideoReader(depth_map_path, ctx=cpu(0))
                total_depth_frames_available = len(_tmp)
                del _tmp
        except Exception as e:
            logger.warning(
                f"Could not determine depth frame count for '{depth_map_path}': {e}"
            )

        total_depth_frames_to_process = total_depth_frames_available
        if process_length != -1 and process_length < total_depth_frames_available:
            total_depth_frames_to_process = process_length

        # Choose reader implementation
        if bit_depth > 8:
            depth_reader = FFmpegDepthPipeReader(
                depth_map_path,
                out_w=target_width,
                out_h=target_height,
                bit_depth=bit_depth,
                num_frames=total_depth_frames_available,
                pix_fmt=pix_fmt,
            )
        else:
            # 8-bit: decode at native res with Decord
            depth_reader = VideoReader(
                depth_map_path, ctx=cpu(0)
            )  # decode at native res

            first_depth_frame_shape = depth_reader.get_batch([0]).asnumpy().shape
            actual_depth_height, actual_depth_width = first_depth_frame_shape[1:3]

            if match_resolution_to_target and (
                actual_depth_width != target_width
                or actual_depth_height != target_height
            ):
                depth_reader = _ResizingDepthReader(
                    depth_reader, out_w=target_width, out_h=target_height
                )
                actual_depth_height, actual_depth_width = (
                    int(target_height),
                    int(target_width),
                )

        first_depth_frame_shape = depth_reader.get_batch([0]).asnumpy().shape
        actual_depth_height, actual_depth_width = first_depth_frame_shape[1:3]

        logger.debug(
            f"==> Depth reader ready. Final depth resolution: "
            f"{actual_depth_width}x{actual_depth_height}. "
            f"Frames available: {total_depth_frames_available}. "
            f"Frames to process: {total_depth_frames_to_process}. "
            f"bit_depth={bit_depth}, pix_fmt='{pix_fmt}'."
        )

        return (
            depth_reader,
            total_depth_frames_to_process,
            actual_depth_height,
            actual_depth_width,
            depth_stream_info,
        )

    elif depth_map_path.lower().endswith(".npz"):
        logger.error(
            "NPZ support is temporarily disabled with disk chunking refactor. "
            "Please convert NPZ to MP4 depth video."
        )
        raise NotImplementedError(
            "NPZ depth map loading is not yet supported with disk chunking."
        )
    else:
        raise ValueError(
            f"Unsupported depth map format: {os.path.basename(depth_map_path)}. "
            "Only MP4 are supported with disk chunking."
        )
