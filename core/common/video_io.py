"""Video I/O utilities for StereoCrafter.

Provides video reading and frame extraction utilities using decord
for efficient video loading.
"""

import os
import re
import json
import shutil
import threading
import time
import logging
import subprocess
from typing import Optional, Tuple, Any

import numpy as np
import torch
import cv2
from decord import VideoReader, cpu

from core.common.gpu_utils import CUDA_AVAILABLE
from core.common.encoding_utils import build_encoder_args, get_encoding_config_from_dict

logger = logging.getLogger(__name__)


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


class VideoIO:
    """Video input/output operations for video processing."""

    @staticmethod
    def read_video_info(video_path: str) -> Tuple[int, int, int, float]:
        """Read video information without loading frames.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple of (total_frames, height, width, fps)
        """
        logger.debug(f"==> Reading video info: {video_path}")
        reader = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(reader)
        first_frame = reader.get_batch([0]).asnumpy()
        height, width = first_frame.shape[1:3]
        fps = float(reader.get_avg_fps())

        logger.debug(f"==> Video info: {total_frames} frames, {width}x{height}, {fps} fps")

        return total_frames, height, width, fps

    @staticmethod
    def read_frame(reader: VideoReader, index: int) -> np.ndarray:
        """Read a single frame from a video reader.

        Args:
            reader: Active VideoReader instance
            index: Frame index to read

        Returns:
            Frame as numpy array [H, W, C]
        """
        return reader.get_batch([index]).asnumpy()[0]

    @staticmethod
    def read_frames_batch(reader: VideoReader, indices: list) -> np.ndarray:
        """Read multiple frames from a video reader.

        Args:
            reader: Active VideoReader instance
            indices: List of frame indices to read

        Returns:
            Frames as numpy array [N, H, W, C]
        """
        return reader.get_batch(indices).asnumpy()


class FFmpegRGBPipeReader:
    """
    Sequential RGB frame reader backed by an FFmpeg pipe (rawvideo).
    Designed for render-time usage where get_batch() is called with
    increasing frame indices (typically contiguous batches).
    """

    def __init__(
        self,
        video_path: str,
        width: int,
        height: int,
        fps: float,
        total_frames: int,
        in_range: str = "tv",
        in_matrix: str = "bt709",
    ):
        self.video_path = video_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps else 0.0
        self.total_frames = int(total_frames) if total_frames is not None else -1
        self.in_range = in_range
        self.in_matrix = in_matrix
        self._proc = None
        self._next_frame = 0
        self._frame_size = self.width * self.height * 3
        self._force_fallback = False  # set True if strict scale params fail

    def __len__(self):
        return self.total_frames if self.total_frames >= 0 else 0

    def get_avg_fps(self):
        return self.fps

    def _build_cmd(self, start_frame: int = 0):
        # Start from the requested frame index. For render, this is usually 0.
        # Use -ss time seek as a best-effort fast start for non-zero start_frame.
        args = ["ffmpeg", "-v", "error", "-nostdin"]
        if start_frame and self.fps:
            start_time = start_frame / self.fps
            args += ["-ss", f"{start_time:.6f}"]
        args += ["-i", self.video_path, "-an", "-sn", "-dn"]

        vf_strict = (
            f"scale={self.width}:{self.height}:flags=bicubic:"
            f"in_range={self.in_range}:out_range={self.in_range}:"
            f"in_color_matrix={self.in_matrix}:out_color_matrix={self.in_matrix},"
            "format=rgb24"
        )
        vf_fallback = f"scale={self.width}:{self.height}:flags=bicubic,format=rgb24"
        vf = vf_fallback if self._force_fallback else vf_strict
        args += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "0", "-"]
        return args

    def _ensure_process(self, start_frame: int):
        if self._proc is not None:
            return
        cmd = self._build_cmd(start_frame=start_frame)
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._next_frame = start_frame

    def _restart(self, start_frame: int):
        try:
            if self._proc is not None:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        finally:
            self._proc = None
        self._ensure_process(start_frame=start_frame)

    def get_batch(self, indices):
        if not indices:
            return _NumpyBatch(np.empty((0, self.height, self.width, 3), dtype=np.uint8))

        try:
            # Expect indices to be increasing most of the time; if not, restart.
            min_idx = int(min(indices))
            max_idx = int(max(indices))

            if self._proc is None:
                self._ensure_process(start_frame=min_idx)
            elif min_idx < self._next_frame:
                self._restart(start_frame=min_idx)

            # Discard frames until we reach min_idx
            while self._next_frame < min_idx:
                junk = self._proc.stdout.read(self._frame_size)
                if not junk or len(junk) < self._frame_size:
                    if not self._force_fallback:
                        self._force_fallback = True
                        self._restart(start_frame=min_idx)
                        return self.get_batch(indices)
                    raise EOFError("FFmpegRGBPipeReader reached EOF while skipping frames.")
                self._next_frame += 1

            # Read frames for requested indices
            out = np.empty((len(indices), self.height, self.width, 3), dtype=np.uint8)
            for j, idx in enumerate(indices):
                idx = int(idx)
                if idx < self._next_frame:
                    # non-monotonic request; restart and recurse (rare)
                    self._restart(start_frame=idx)
                    return self.get_batch(indices)

                # Skip gap frames if needed
                while self._next_frame < idx:
                    junk = self._proc.stdout.read(self._frame_size)
                    if not junk or len(junk) < self._frame_size:
                        if not self._force_fallback:
                            self._force_fallback = True
                            self._restart(start_frame=min_idx)
                            return self.get_batch(indices)
                        raise EOFError("FFmpegRGBPipeReader reached EOF while skipping gap frames.")
                    self._next_frame += 1

                raw = self._proc.stdout.read(self._frame_size)
                if not raw or len(raw) < self._frame_size:
                    if not self._force_fallback:
                        self._force_fallback = True
                        self._restart(start_frame=min_idx)
                        return self.get_batch(indices)
                    raise EOFError("FFmpegRGBPipeReader reached EOF while reading a frame.")
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
                out[j] = frame
                self._next_frame += 1

            return _NumpyBatch(out)

        except EOFError as e:
            if not self._force_fallback:
                # Some FFmpeg builds don't support scale=in_range/in_color_matrix.
                # Retry once with a simpler filter chain.
                self._force_fallback = True
                try:
                    self._restart(start_frame=int(min(indices)) if indices else 0)
                except Exception:
                    pass
                return self.get_batch(indices)
            raise


class FFmpegRGBSingleFrameReader:
    """
    Random-access RGB reader for preview usage.
    Each get_batch([idx]) spawns a small FFmpeg decode for that frame.
    Slower than Decord, but matches FFmpeg's YUV->RGB conversion.
    """

    def __init__(
        self,
        video_path: str,
        width: int,
        height: int,
        fps: float,
        total_frames: int,
        in_range: str = "tv",
        in_matrix: str = "bt709",
    ):
        self.video_path = video_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps else 0.0
        self.total_frames = int(total_frames) if total_frames is not None else -1
        self.in_range = in_range
        self.in_matrix = in_matrix
        self._frame_size = self.width * self.height * 3
        self._force_fallback = False  # set True if strict scale params fail

    def __len__(self):
        return self.total_frames if self.total_frames >= 0 else 0

    def get_avg_fps(self):
        return self.fps

    def get_batch(self, indices):
        if not indices:
            return _NumpyBatch(np.empty((0, self.height, self.width, 3), dtype=np.uint8))

        def _read_exact(proc, nbytes: int) -> bytes:
            buf = b""
            while len(buf) < nbytes:
                chunk = proc.stdout.read(nbytes - len(buf)) if proc.stdout else b""
                if not chunk:
                    break
                buf += chunk
            return buf

        def _decode_one(idx: int, vf: str) -> tuple[bytes, str, int]:
            cmd = [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-i",
                self.video_path,
                "-an",
                "-sn",
                "-dn",
                "-vf",
                vf,
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw = b""
            err = ""
            try:
                raw = _read_exact(proc, self._frame_size)
                if proc.stderr:
                    try:
                        err = proc.stderr.read().decode("utf-8", errors="ignore")
                    except Exception:
                        err = ""
            finally:
                try:
                    if proc.stdout:
                        proc.stdout.close()
                except Exception:
                    pass
                try:
                    if proc.stderr:
                        proc.stderr.close()
                except Exception:
                    pass
                try:
                    proc.wait()
                except Exception:
                    pass
            return raw, err, int(proc.returncode or 0)

        frames = np.empty((len(indices), self.height, self.width, 3), dtype=np.uint8)
        for j, idx in enumerate(indices):
            idx = int(idx)

            # Try the strict matrix/range path first; if unsupported by the user's ffmpeg build,
            # fall back to a plain scale->rgb24 path (still FFmpeg-based conversion, just less explicit).
            vf_strict = (
                f"select='eq(n\\,{idx})',"
                f"scale={self.width}:{self.height}:flags=bicubic:"
                f"in_range={self.in_range}:out_range={self.in_range}:"
                f"in_color_matrix={self.in_matrix}:out_color_matrix={self.in_matrix},"
                "format=rgb24"
            )
            vf_fallback = f"select='eq(n\\,{idx})',scale={self.width}:{self.height}:flags=bicubic,format=rgb24"

            raw, err, rc = _decode_one(idx, vf_strict)
            if not raw or len(raw) < self._frame_size:
                raw2, err2, rc2 = _decode_one(idx, vf_fallback)
                if raw2 and len(raw2) >= self._frame_size:
                    raw, err, rc = raw2, err2, rc2
                else:
                    msg = (err2 or err or "").strip()
                    if msg:
                        raise EOFError(f"FFmpegRGBSingleFrameReader failed to decode frame {idx}: {msg}")
                    raise EOFError(f"FFmpegRGBSingleFrameReader failed to decode frame {idx}.")

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
            frames[j] = frame

        return _NumpyBatch(frames)


def _infer_depth_bit_depth(depth_stream_info: Optional[dict]) -> int:
    """Infer the bit depth of a depth stream from ffprobe info."""
    if not depth_stream_info:
        return 8
    pix_fmt = str(depth_stream_info.get("pix_fmt", "")).lower()
    profile = str(depth_stream_info.get("profile", "")).lower()
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
    """Build FFmpeg video filter for depth map extraction."""
    pix_fmt = (pix_fmt or "").lower()
    if pix_fmt.startswith("gray"):
        return f"scale={out_w}:{out_h}:flags=bilinear,format=gray16le"
    return f"extractplanes=y,scale={out_w}:{out_h}:flags=bilinear,format=gray16le"


class FFmpegDepthPipeReader:
    """Sequential FFmpeg-backed depth reader preserving 10-bit+ values."""

    def __init__(self, path: str, out_w: int, out_h: int, bit_depth: int, num_frames: int, pix_fmt: str = ""):
        self.path = path
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.bit_depth = int(bit_depth) if bit_depth else 16
        try:
            if isinstance(num_frames, (int, float)) and not isinstance(num_frames, bool):
                self._num_frames = int(num_frames)
            elif isinstance(num_frames, str) and num_frames not in ("", "N/A"):
                self._num_frames = int(float(num_frames))
            else:
                self._num_frames = 0
        except (ValueError, TypeError):
            self._num_frames = 0
        self._pix_fmt = pix_fmt or ""
        self._proc: Optional[subprocess.Popen] = None
        self._next_index = 0
        self._frame_bytes = self.out_w * self.out_h * 2
        self._msb_shift: Optional[int] = None
        self._use_16_to_n_scale: bool = False
        self._start_process()

    def __len__(self) -> int:
        return self._num_frames

    def _start_process(self):
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
            str(self._num_frames) if isinstance(self._num_frames, int) and self._num_frames > 0 else "999999999",
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._next_index = 0
        self._msb_shift = None

    def close(self):
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
        self.close()

    def seek(self, idx: int):
        idx = int(idx)
        if idx == self._next_index:
            return
        if idx < self._next_index:
            self._start_process()
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
        if self._msb_shift is None:
            expected_max = (1 << self.bit_depth) - 1 if 0 < self.bit_depth < 16 else None
            if expected_max is None:
                self._msb_shift = 0
                self._use_16_to_n_scale = False
            else:
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
                        low_bits_max = int((sample & low_mask).max(initial=0))
                        if low_bits_max == 0:
                            self._msb_shift = shift
                            self._use_16_to_n_scale = False
                        else:
                            self._msb_shift = 0
                            self._use_16_to_n_scale = True
        expected_max = (1 << self.bit_depth) - 1 if 0 < self.bit_depth < 16 else None
        if expected_max is None:
            return arr_u16
        if self._msb_shift and self._msb_shift > 0:
            return (arr_u16 >> self._msb_shift).astype(np.uint16)
        if self._use_16_to_n_scale:
            arr32 = arr_u16.astype(np.uint32)
            return ((arr32 * expected_max + 32767) // 65535).astype(np.uint16)
        return arr_u16

    def get_batch(self, indices):
        indices = list(indices)
        if not indices:
            return _NumpyBatch(np.zeros((0, self.out_h, self.out_w, 1), dtype=np.uint16))
        first = int(indices[0])
        if first != self._next_index:
            self.seek(first)
        n = len(indices)
        expected = self._frame_bytes * n
        buf = self._read_exact(expected)
        if len(buf) != expected:
            raise EOFError(f"FFmpegDepthPipeReader: expected {expected} bytes, got {len(buf)}")
        arr = np.frombuffer(buf, dtype=np.uint16).reshape(n, self.out_h, self.out_w, 1)
        arr = self._maybe_apply_shift(arr)
        self._next_index = first + n
        return _NumpyBatch(arr.copy())


class _ResizingDepthReader:
    """Wrapper reader that resizes depth frames to target resolution."""

    def __init__(self, inner_reader, out_w, out_h):
        self._inner = inner_reader
        self._out_w = int(out_w)
        self._out_h = int(out_h)

    def __len__(self) -> int:
        return len(self._inner)

    def seek(self, *args, **kwargs):
        return self._inner.seek(*args, **kwargs)

    def get_batch(self, indices):
        _batch = self._inner.get_batch(indices)
        arr = _batch.asnumpy()
        in_h, in_w = arr.shape[1:3]
        if in_w == self._out_w and in_h == self._out_h:
            return _batch
        interp = cv2.INTER_LINEAR if (self._out_w > in_w or self._out_h > in_h) else cv2.INTER_AREA
        if arr.ndim == 4:
            out = np.empty((arr.shape[0], self._out_h, self._out_w, arr.shape[3]), dtype=arr.dtype)
            for i in range(arr.shape[0]):
                res = cv2.resize(arr[i], (self._out_w, self._out_h), interpolation=interp)
                if res.ndim == 2:
                    res = res[..., np.newaxis]
                out[i] = res
        else:
            out = np.empty((arr.shape[0], self._out_h, self._out_w), dtype=arr.dtype)
            for i in range(arr.shape[0]):
                out[i] = cv2.resize(arr[i], (self._out_w, self._out_h), interpolation=interp)
        return _NumpyBatch(out)


def read_video_frames(
    video_path: str,
    process_length: int,
    set_pre_res: bool,
    pre_res_width: int,
    pre_res_height: int,
    strict_ffmpeg_decode: bool = False,
    dataset: str = "open",
    is_depth: bool = False,
) -> Tuple[Any, float, int, int, int, int, Optional[dict], int]:
    """Initialize a VideoReader for chunked reading, with optional high-bit depth support."""
    try:
        process_length = int(process_length) if process_length not in (None, "", "N/A") else -1
    except (ValueError, TypeError):
        process_length = -1

    if dataset != "open":
        raise NotImplementedError(f"Dataset '{dataset}' not supported.")

    logger.info(f"==> Initializing VideoReader for: {video_path}")
    video_stream_info = get_video_stream_info(video_path)
    fps = 0.0
    original_height, original_width = 0, 0
    total_frames_available = 0

    # Use bit-depth awareness for depth maps
    bit_depth = 8
    if is_depth:
        bit_depth = _infer_depth_bit_depth(video_stream_info)

    # Determine dimensions and frame count
    try:
        # Use decord for metadata if ffprobe failed or was insufficient
        tmp = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = tmp.get_batch([0]).asnumpy().shape[1:3]
        total_frames_available = len(tmp)
        fps = float(tmp.get_avg_fps())
        del tmp
    except Exception:
        if video_stream_info:
            original_height = int(video_stream_info.get("height", 0))
            original_width = int(video_stream_info.get("width", 0))
            total_frames_available = int(video_stream_info.get("nb_frames", 0))
            fr_str = video_stream_info.get("r_frame_rate", "0/0")
            if "/" in fr_str:
                num, den = fr_str.split("/")
                fps = float(num) / float(den) if float(den) > 0 else 0.0

    if original_height == 0 or original_width == 0:
        raise ValueError(f"Could not determine video dimensions for {video_path}")

    h_reader = pre_res_height if set_pre_res and pre_res_height > 0 else original_height
    w_reader = pre_res_width if set_pre_res and pre_res_width > 0 else original_width

    # Choose reader implementation
    if is_depth and bit_depth > 8:
        logger.info(f"==> Using high-bit depth path (FFmpegDepthPipeReader) for {bit_depth}-bit source.")
        video_reader = FFmpegDepthPipeReader(
            video_path,
            out_w=w_reader,
            out_h=h_reader,
            bit_depth=bit_depth,
            num_frames=total_frames_available,
            pix_fmt=str(video_stream_info.get("pix_fmt", "")),
        )
    elif strict_ffmpeg_decode:
        in_range, in_matrix = "tv", "bt709"
        try:
            cr = str((video_stream_info or {}).get("color_range") or "").lower()
            cs = str((video_stream_info or {}).get("color_space") or "").lower()
            if "full" in cr or cr == "pc":
                in_range = "pc"
            if "2020" in cs:
                in_matrix = "bt2020"
            elif "601" in cs:
                in_matrix = "bt601"
        except Exception:
            pass
        video_reader = FFmpegRGBPipeReader(
            video_path=video_path,
            width=w_reader,
            height=h_reader,
            fps=fps,
            total_frames=total_frames_available,
            in_range=in_range,
            in_matrix=in_matrix,
        )
    else:
        video_reader = VideoReader(video_path, ctx=cpu(0), width=w_reader, height=h_reader)
        # Wrap in ResizingDepthReader if Decord didn't resize or if we need specific parity
        if is_depth:
            actual_h, actual_w = video_reader.get_batch([0]).asnumpy().shape[1:3]
            if actual_h != h_reader or actual_w != w_reader:
                video_reader = _ResizingDepthReader(video_reader, out_w=w_reader, out_h=h_reader)

    # Verify final dimensions
    first_shape = video_reader.get_batch([0]).asnumpy().shape
    actual_h, actual_w = first_shape[1:3]

    total_to_process = total_frames_available
    if total_frames_available > 0 and process_length != -1 and process_length < total_frames_available:
        total_to_process = process_length

    return (video_reader, fps, original_height, original_width, actual_h, actual_w, video_stream_info, total_to_process)


_FFPROBE_AVAIL: Optional[bool] = None
_INFO_CACHE: dict = {}


def get_video_stream_info(video_path: str) -> Optional[dict]:
    """Get video stream information using ffprobe.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing stream info (width, height, codec, etc.) or None if unavailable
    """
    global _FFPROBE_AVAIL, _INFO_CACHE
    if not video_path:
        return None
    if video_path in _INFO_CACHE:
        return _INFO_CACHE[video_path]

    if _FFPROBE_AVAIL is None:
        try:
            subprocess.run(["ffprobe", "-version"], check=True, capture_output=True)
            _FFPROBE_AVAIL = True
        except Exception:
            _FFPROBE_AVAIL = False
    if not _FFPROBE_AVAIL:
        return None

    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name,profile,pix_fmt,color_range,color_primaries,transfer_characteristics,color_space,r_frame_rate",
            "-show_entries",
            "side_data=mastering_display_metadata,max_content_light_level",
            "-of",
            "json",
            video_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        if "streams" in data and data["streams"]:
            info = {k: v for k, v in data["streams"][0].items() if v and v not in ("N/A", "und", "unknown")}
            _INFO_CACHE[video_path] = info
            return info
    except Exception:
        pass
    return None


def read_video_frames_decord(
    video_path: str,
    process_length: int = -1,
    target_fps: float = -1.0,
    set_res_width: Optional[int] = None,
    set_res_height: Optional[int] = None,
    decord_ctx=cpu(0),
) -> Tuple[np.ndarray, float, int, int, int, int, Optional[dict]]:
    """Read video frames using decord with optional resizing and fps conversion.

    Args:
        video_path: Path to the video file
        process_length: Number of frames to process (-1 for all)
        target_fps: Target fps (-1 for original)
        set_res_width: Target width (None for original)
        set_res_height: Target height (None for original)
        decord_ctx: Decord context for reading

    Returns:
        Tuple of (frames as float32 numpy array [T,H,W,C] normalized to 0-1,
                  fps, original_height, original_width, output_height, output_width,
                  stream_info)
    """
    info = get_video_stream_info(video_path)
    temp_reader = VideoReader(video_path, ctx=cpu(0))
    oh, ow = temp_reader.get_batch([0]).shape[1:3]
    del temp_reader
    dw, dh = (set_res_width, set_res_height) if set_res_width and set_res_height else (ow, oh)
    vid = VideoReader(video_path, ctx=decord_ctx, width=dw, height=dh)
    total = len(vid)
    fps = target_fps if target_fps > 0 else vid.get_avg_fps()
    stride = max(round(vid.get_avg_fps() / fps), 1)
    idxs = list(range(0, total, stride))
    if process_length != -1:
        idxs = idxs[:process_length]
    frames = vid.get_batch(idxs).asnumpy().astype("float32") / 255.0
    return frames, fps, oh, ow, frames.shape[1], frames.shape[2], info


def encode_frames_to_mp4(
    temp_png_dir: str,
    final_output_mp4_path: str,
    fps: float,
    total_output_frames: int,
    video_stream_info: Optional[dict],
    stop_event: Optional[threading.Event] = None,
    sidecar_json_data: Optional[dict] = None,
    user_output_crf: Optional[int] = None,
    output_sidecar_ext: str = ".json",
) -> bool:
    """Encode a directory of PNG frames to an MP4 video using FFmpeg.

    Args:
        temp_png_dir: Directory containing PNG frames named %05d.png
        final_output_mp4_path: Path to the output MP4 file
        fps: Output video frame rate
        total_output_frames: Total number of frames to encode
        video_stream_info: Source video stream info for matching color space
        stop_event: Optional threading event to cancel encoding
        sidecar_json_data: Optional data to save to a JSON sidecar file
        user_output_crf: Optional override for CRF (quality) setting
        output_sidecar_ext: Extension for the sidecar file

    Returns:
        True if encoding succeeded, False otherwise
    """
    if total_output_frames == 0:
        logger.warning(f"No frames to encode for {os.path.basename(final_output_mp4_path)}. Skipping.")
        if os.path.exists(temp_png_dir):
            shutil.rmtree(temp_png_dir)
        return False

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        os.path.join(temp_png_dir, "%05d.png"),
    ]

    # Determine codec, pixel format, and CRF
    output_codec, output_pix_fmt, crf, output_profile = "libx264", "yuv420p", "23", "main"

    if user_output_crf is not None:
        crf = str(user_output_crf)

    is_hdr = (
        video_stream_info
        and video_stream_info.get("color_primaries") == "bt2020"
        and video_stream_info.get("transfer_characteristics") in ("smpte2084", "arib-std-b67")
    )
    orig_pix = video_stream_info.get("pix_fmt", "") if video_stream_info else ""
    is_high_bit = "10" in orig_pix or "12" in orig_pix or "16" in orig_pix

    if is_hdr or (is_high_bit and video_stream_info.get("codec_name") in ("hevc", "prores", "dnxhd", "dnxhr")):
        output_codec = "hevc_nvenc" if CUDA_AVAILABLE else "libx265"
        output_pix_fmt = "yuv420p10le"
        output_profile = "main10"
    else:
        output_codec = "h264_nvenc" if CUDA_AVAILABLE else "libx264"

    # Check for color_tags_mode to force 10-bit if needed
    color_tags_mode = str(video_stream_info.get("color_tags_mode", "")).lower() if video_stream_info else ""
    if color_tags_mode in ("bt.2020 pq", "bt.2020 hlg", "bt.2020"):
        output_codec = "hevc_nvenc" if CUDA_AVAILABLE else "libx265"
        output_pix_fmt = "yuv420p10le"
        output_profile = "main10"

    # Build ffmpeg command
    ffmpeg_cmd.extend(["-c:v", output_codec])

    # Add CRF/QP
    if "nvenc" in output_codec:
        ffmpeg_cmd.extend(["-preset", "p4", "-qp", crf])
    else:
        ffmpeg_cmd.extend(["-crf", crf])

    ffmpeg_cmd.extend(["-pix_fmt", output_pix_fmt])
    if output_profile:
        ffmpeg_cmd.extend(["-profile:v", output_profile])

    if video_stream_info:
        for k, f in [
            ("color_primaries", "-color_primaries"),
            ("transfer_characteristics", "-color_trc"),
            ("color_space", "-colorspace"),
            ("color_range", "-color_range"),
        ]:
            if video_stream_info.get(k):
                ffmpeg_cmd.extend([f, video_stream_info[k]])

    if os.path.splitext(final_output_mp4_path)[1].lower() in (".mp4", ".mov", ".m4v"):
        ffmpeg_cmd.extend(["-movflags", "+write_colr"])

    ffmpeg_cmd.append(final_output_mp4_path)

    try:
        process = subprocess.Popen(
            ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8"
        )
        while process.poll() is None:
            if stop_event and stop_event.is_set():
                process.terminate()
                return False
            time.sleep(0.1)
        if process.returncode != 0:
            return False
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        return False
    finally:
        if os.path.exists(temp_png_dir):
            shutil.rmtree(temp_png_dir)

    if sidecar_json_data:
        path = f"{os.path.splitext(final_output_mp4_path)[0]}{output_sidecar_ext}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sidecar_json_data, f, indent=4)
    return True


def start_ffmpeg_pipe_process(
    content_width: int,
    content_height: int,
    final_output_mp4_path: str,
    fps: float,
    video_stream_info: Optional[dict] = None,
    output_format_str: str = "",
    user_output_crf: Optional[int] = None,
    pad_to_16_9: bool = False,
    debug_label: Optional[str] = None,
    encoding_options: Optional[dict] = None,
) -> Optional[subprocess.Popen]:
    """Start an FFmpeg process that reads raw video from a pipe.

    Args:
        content_width: Width of input frames
        content_height: Height of input frames
        final_output_mp4_path: Output path for the video
        fps: Frame rate
        video_stream_info: Source video stream info (may contain color_tags_mode)
        output_format_str: Optional format string
        user_output_crf: Optional CRF override
        pad_to_16_9: Whether to pad to 16:9 (currently unused)
        debug_label: Optional label for logging
        encoding_options: Optional dict with encoding settings:
            - codec: "Auto", "H.264", "H.265"
            - encoding_encoder: "Auto", "Force CPU"
            - encoding_quality: quality preset
            - encoding_tune: tune option
            - output_crf: CRF value
            - nvenc_lookahead_enabled, nvenc_lookahead, etc.

    Returns:
        The subprocess.Popen instance
    """
    from core.common.encoding_utils import build_encoder_args, get_encoding_config_from_dict

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{content_width}x{content_height}",
        "-pix_fmt",
        "bgr48le",
        "-r",
        str(fps),
        "-i",
        "-",
    ]

    # Determine if we need 10-bit output based on color tags
    color_tags_mode = str(video_stream_info.get("color_tags_mode", "")).lower() if video_stream_info else ""
    # Also check encoding_options for color_tags if not in video_stream_info
    if not color_tags_mode and encoding_options and "color_tags" in encoding_options:
        color_tags_mode = str(encoding_options["color_tags"]).lower()
    force_10bit = color_tags_mode in ("bt.2020 pq", "bt.2020 hlg", "bt.2020")

    # Build encoding arguments from encoding_options if provided
    if encoding_options:
        enc_config = get_encoding_config_from_dict(encoding_options)
        encoder_args = build_encoder_args(
            codec=enc_config.get("codec", "Auto"),
            encoder=enc_config.get("encoder", "Auto"),
            quality=enc_config.get("quality", "Medium"),
            tune=enc_config.get("tune", "None"),
            crf=int(enc_config.get("crf", 23)) if user_output_crf is None else user_output_crf,
            force_10bit=force_10bit,
            nvenc_options={
                "lookahead_enabled": enc_config.get("nvenc_lookahead_enabled", False),
                "lookahead": enc_config.get("nvenc_lookahead", 16),
                "spatial_aq": enc_config.get("nvenc_spatial_aq", False),
                "temporal_aq": enc_config.get("nvenc_temporal_aq", False),
                "aq_strength": enc_config.get("nvenc_aq_strength", 8),
            }
            if enc_config.get("nvenc_lookahead_enabled")
            else None,
        )
        codec = encoder_args["codec"]
        crf = str(encoder_args.get("crf", enc_config.get("crf", 23)))
        preset = encoder_args["preset"]
        tune_flag = encoder_args["tune"]
        pix = encoder_args["pix_fmt"]
        extra_args = encoder_args.get("extra_args", [])
    else:
        # Fallback to old behavior (simple auto-detection)
        codec, crf = "libx264", "23"
        if CUDA_AVAILABLE:
            codec = "h264_nvenc"
        if user_output_crf is not None:
            crf = str(user_output_crf)
        pix = "yuv420p10le" if force_10bit else "yuv420p"
        preset, tune_flag, extra_args = None, None, []

    cmd.extend(["-c:v", codec])

    # Add preset and tune
    if preset:
        if "nvenc" in codec:
            cmd.extend(["-preset", preset])
        else:
            cmd.extend(["-preset", preset])
    elif "nvenc" in codec:
        cmd.extend(["-preset", "p4"])

    # Add CRF/QP
    if extra_args:
        # Handle nvenc args which are in extra_args
        cmd.extend(extra_args)
    else:
        if "nvenc" in codec:
            cmd.extend(["-qp", crf])
        else:
            cmd.extend(["-crf", crf])

    cmd.extend(["-pix_fmt", pix])

    # Add color metadata if forcing 10-bit and we have source info
    if force_10bit and video_stream_info:
        for k, f in [
            ("color_primaries", "-color_primaries"),
            ("transfer_characteristics", "-color_trc"),
            ("color_space", "-colorspace"),
            ("color_range", "-color_range"),
        ]:
            if video_stream_info.get(k):
                cmd.extend([f, video_stream_info[k]])

    cmd.extend(["-movflags", "+write_colr", final_output_mp4_path])
    logger.info(f"Starting FFmpeg pipe: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def start_ffmpeg_pipe_process_dnxhr(
    content_width: int, content_height: int, final_output_mov_path: str, fps: float, dnxhr_profile: str = "HQX"
) -> Optional[subprocess.Popen]:
    """Start an FFmpeg process for high-quality DNxHR output via pipe.

    Args:
        content_width: Width of input frames
        content_height: Height of input frames
        final_output_mov_path: Output path for the MOV file
        fps: Frame rate
        dnxhr_profile: DNxHR profile (SQ, HQ, HQX, 444)

    Returns:
        The subprocess.Popen instance
    """
    prof = {"SQ": "dnxhr_sq", "HQ": "dnxhr_hq", "HQX": "dnxhr_hqx", "444": "dnxhr_444"}.get(
        dnxhr_profile.strip().upper()[:3], "dnxhr_hqx"
    )
    pix = "yuv444p10le" if prof == "dnxhr_444" else "yuv422p10le"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{content_width}x{content_height}",
        "-pix_fmt",
        "bgr48le",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "dnxhd",
        "-profile:v",
        prof,
        "-pix_fmt",
        pix,
        "-an",
        final_output_mov_path,
    ]
    logger.info(f"Starting DNxHR pipe: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def reverse_frames(frames: torch.Tensor) -> torch.Tensor:
    """Reverse the temporal order of video frames.

    Args:
        frames: Tensor of shape [T, C, H, W] or [T, H, W, C]

    Returns:
        Reversed frames tensor
    """
    return torch.flip(frames, dims=[0])


def reverse_frames_numpy(frames: np.ndarray) -> np.ndarray:
    """Reverse the temporal order of video frames (numpy).

    Args:
        frames: Numpy array of shape [T, H, W, C] or [T, H, W]

    Returns:
        Reversed frames array
    """
    return np.flip(frames, axis=0)
