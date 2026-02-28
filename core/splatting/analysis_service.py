"""Analysis Service for StereoCrafter.

Pure analysis pipelines that read depth video files, run math, and return
numbers. Zero GUI dependency — no Tkinter, no self.* GUI vars.

Extracted from splatting_gui.py (Item 2 of the refactor plan).
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Optional, Tuple

import numpy as np
from decord import VideoReader, cpu

from dependency.stereocrafter_util import get_video_stream_info
from core.splatting.depth_processing import (
    DEPTH_VIS_TV10_BLACK_NORM,
    DEPTH_VIS_TV10_WHITE_NORM,
    _infer_depth_bit_depth,
    load_pre_rendered_depth,
    process_depth_batch,
)

logger = logging.getLogger(__name__)


class AnalysisService:
    """Collection of pure analysis methods for depth / disparity estimation.

    All methods are ``@staticmethod`` or ``@classmethod`` — they take explicit
    arguments and return plain values.  The GUI keeps only thread creation,
    progress reporting and ``self.after()`` wiring.
    """

    # ------------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------------

    @staticmethod
    def make_dp_cache_key(depth_path: str, max_disp: float, gamma: float) -> str:
        """Return a stable cache key for Total(D+P) estimates.

        Convergence is intentionally **excluded** so that moving the
        convergence slider does not invalidate an already-computed estimate.
        Tolerates call sites that accidentally swap ``conv`` and ``max_disp``.

        Args:
            depth_path: Absolute path to the depth map video.
            max_disp: Maximum disparity value (usually > 1).
            gamma: Depth gamma exponent.

        Returns:
            Pipe-separated string suitable as a dict key.
        """
        try:
            a, b = float(max_disp), float(gamma)
            # Guard: if someone passes (conv, max_disp) swapped, detect & fix.
            if 0.0 <= a <= 1.0:
                # a looks like a convergence value — ignore it; b is max_disp
                a = b
                b = gamma
            return f"{depth_path}|{float(a):.4f}|{float(b):.4f}"
        except Exception:
            return str(depth_path)

    # ------------------------------------------------------------------
    # Global depth stats (GN pre-pass)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_clip_global_depth_stats(
        depth_map_path: str,
        stop_event,
        chunk_size: int = 100,
    ) -> Tuple[float, float]:
        """Compute global min/max depth values from a depth video by chunked scan.

        Used only for the preview's Global-Normalisation cache.  The caller
        owns the ``stop_event`` and may share it with other threads.

        Args:
            depth_map_path: Path to the depth map video file.
            stop_event: ``threading.Event`` checked between chunks.  Pass
                ``None`` to disable early-stop checks.
            chunk_size: Number of frames decoded per chunk.

        Returns:
            ``(global_min, global_max)`` as ``float``, or ``(0.0, 1.0)`` on
            error / early stop.
        """
        logger.info(
            f"==> Starting clip-local depth stats pre-pass for "
            f"{os.path.basename(depth_map_path)}..."
        )
        global_min: float = float("inf")
        global_max: float = float("-inf")

        try:
            temp_reader = VideoReader(depth_map_path, ctx=cpu(0))
            total_frames = len(temp_reader)

            if total_frames == 0:
                logger.error("Depth reader found 0 frames for global stats.")
                return 0.0, 1.0

            for i in range(0, total_frames, chunk_size):
                if stop_event is not None and stop_event.is_set():
                    logger.warning("Global stats scan stopped by user.")
                    return 0.0, 1.0

                current_indices = list(range(i, min(i + chunk_size, total_frames)))
                chunk_numpy_raw = temp_reader.get_batch(current_indices).asnumpy()

                # Handle RGB vs grayscale depth maps
                if chunk_numpy_raw.ndim == 4:
                    if chunk_numpy_raw.shape[-1] == 3:  # RGB
                        chunk_numpy = chunk_numpy_raw.mean(axis=-1)
                    else:  # Grayscale with channel dim
                        chunk_numpy = chunk_numpy_raw.squeeze(-1)
                else:
                    chunk_numpy = chunk_numpy_raw

                chunk_min = float(chunk_numpy.min())
                chunk_max = float(chunk_numpy.max())

                if chunk_min < global_min:
                    global_min = chunk_min
                if chunk_max > global_max:
                    global_max = chunk_max

            logger.info(
                f"==> Clip-local depth stats computed: "
                f"min_raw={global_min:.3f}, max_raw={global_max:.3f}"
            )
            return float(global_min), float(global_max)

        except Exception as e:
            logger.error(
                f"Error during clip-local depth stats scan for preview: {e}"
            )
            return 0.0, 1.0
        finally:
            gc.collect()

    # ------------------------------------------------------------------
    # D+P total estimator
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_dp_total_max(
        depth_path: str,
        convergence_point: float,
        max_disp: float,
        depth_gamma: float,
        params: Optional[dict] = None,
        sample_frames: int = 10,
        pixel_stride: int = 8,
        total_frames_override: Optional[int] = None,
        depth_native_size: Optional[Tuple[int, int]] = None,
        sidecar_folder: Optional[str] = None,
        clip_norm_cache: Optional[dict] = None,
        depth_blur_left_mix: float = 0.5,
    ) -> Optional[float]:
        """Estimate the maximum Total(D+P) for a clip by sampling frames.

        Matches preview/render math as closely as possible:

        * Reads depth in **RAW code values** (10-bit stays 0..1023, not
          RGB-expanded).
        * Runs the same depth pre-processing (dilate/blur) used by preview/render.
        * Applies the same normalisation policy and gamma curve:
          ``depth = clip(depth, 0..1);  depth = 1 - (1 - depth) ** gamma``

        Args:
            depth_path: Path to the depth map video file.
            convergence_point: Zero-disparity anchor (0..1).
            max_disp: Maximum disparity percentage.
            depth_gamma: Gamma exponent for the depth curve.
            params: Optional dict of pre-processing knobs (matching sidecar
                key names: ``depth_dilate_size_x``, ``depth_blur_size_x``,
                etc.).  Falls back to defaults when keys are absent.
            sample_frames: Number of evenly-spaced frames to sample.
            pixel_stride: Stride for the pixel sub-sample when computing D+P.
            total_frames_override: Skip reader probe when frame count is known.
            depth_native_size: ``(width, height)`` of the depth map — obtained
                from the active previewer when available.  Falls back to
                ffprobe.
            sidecar_folder: If provided, checks whether a sidecar file exists
                (which forces GN off).
            clip_norm_cache: Dict of ``{path: (min, max)}`` used when Global
                Normalisation is on.
            depth_blur_left_mix: Left-blur mix factor (from GUI var).

        Returns:
            Maximum Total(D+P) as a ``float``, or ``None`` on failure.
        """
        if not depth_path or not os.path.exists(depth_path):
            return None

        # -- Unpack pre-processing knobs from params dict ------------------
        p = params or {}

        def _pfloat(key: str, default: float) -> float:
            try:
                return float(p.get(key, default))
            except Exception:
                return float(default)

        depth_dilate_size_x = _pfloat("depth_dilate_size_x", 3.0)
        depth_dilate_size_y = _pfloat("depth_dilate_size_y", 3.0)
        depth_blur_size_x   = _pfloat("depth_blur_size_x",   5.0)
        depth_blur_size_y   = _pfloat("depth_blur_size_y",   5.0)
        depth_dilate_left   = _pfloat("depth_dilate_left",   0.0)
        depth_blur_left     = _pfloat("depth_blur_left",     0.0)

        try:
            depth_gamma = float(p.get("depth_gamma", depth_gamma))
        except Exception:
            depth_gamma = float(depth_gamma)

        # -- Frame count ---------------------------------------------------
        total_frames = 0
        if total_frames_override is not None:
            try:
                total_frames = int(total_frames_override)
            except Exception:
                total_frames = 0

        if total_frames <= 0:
            try:
                tmp = VideoReader(depth_path, ctx=cpu(0))
                total_frames = len(tmp)
                del tmp
            except Exception:
                total_frames = 0

        if total_frames <= 0:
            return None

        # -- Sample indices ------------------------------------------------
        sample_frames = int(max(1, sample_frames))
        if sample_frames >= total_frames:
            indices = list(range(total_frames))
        else:
            indices = [
                int(round(i * (total_frames - 1) / (sample_frames - 1)))
                for i in range(sample_frames)
            ]
        indices = sorted(set(max(0, min(total_frames - 1, i)) for i in indices))

        # -- Depth stream info / bit depth ---------------------------------
        depth_stream_info = None
        bit_depth = 8
        pix_fmt = ""
        try:
            depth_stream_info = get_video_stream_info(depth_path)
            bit_depth = _infer_depth_bit_depth(depth_stream_info)
            pix_fmt = str((depth_stream_info or {}).get("pix_fmt", ""))
        except Exception:
            depth_stream_info = None
            bit_depth = 8
            pix_fmt = ""

        # -- Output size ---------------------------------------------------
        out_w, out_h = 0, 0
        if depth_native_size:
            try:
                out_w, out_h = int(depth_native_size[0]), int(depth_native_size[1])
            except Exception:
                out_w, out_h = 0, 0

        if not out_w or not out_h:
            try:
                out_w = int((depth_stream_info or {}).get("width", 0) or 0)
                out_h = int((depth_stream_info or {}).get("height", 0) or 0)
            except Exception:
                out_w, out_h = 0, 0

        if not out_w or not out_h:
            return None

        # -- Depth reader --------------------------------------------------
        try:
            depth_reader, _, _, _, _ = load_pre_rendered_depth(
                depth_map_path=depth_path,
                process_length=-1,
                target_height=out_h,
                target_width=out_w,
                match_resolution_to_target=False,
            )
        except Exception:
            try:
                depth_reader = VideoReader(depth_path, ctx=cpu(0))
            except Exception:
                return None

        # -- GN sidecar check ----------------------------------------------
        enable_global_norm = bool(p.get("enable_global_norm", False))
        if enable_global_norm and sidecar_folder:
            try:
                depth_map_basename = os.path.splitext(os.path.basename(depth_path))[0]
                sidecar_ext = p.get("sidecar_ext", ".fssidecar")
                json_sidecar_path = os.path.join(
                    sidecar_folder, f"{depth_map_basename}{sidecar_ext}"
                )
                if os.path.exists(json_sidecar_path):
                    enable_global_norm = False
            except Exception:
                pass

        # -- GN global min/max from cache ----------------------------------
        global_min, global_max = 0.0, 1.0
        if enable_global_norm and clip_norm_cache:
            try:
                cached = clip_norm_cache.get(depth_path)
                if cached:
                    global_min = float(cached[0])
                    global_max = float(cached[1])
            except Exception:
                global_min, global_max = 0.0, 1.0

        # -- Main sampling loop --------------------------------------------
        max_total: Optional[float] = None

        for idx in indices:
            try:
                if hasattr(depth_reader, "seek"):
                    depth_reader.seek(int(idx))
                frame_np = depth_reader.get_batch([int(idx)]).asnumpy()

                # Ensure channel-last single-channel
                if frame_np.ndim == 3:
                    frame_np = frame_np[..., None]
                elif frame_np.ndim == 4 and frame_np.shape[-1] != 1:
                    frame_np = frame_np[..., :1]
                elif frame_np.ndim != 4:
                    continue

                frame_raw = frame_np.astype(np.float32, copy=False)

                max_raw_content_value = float(np.max(frame_raw))
                if max_raw_content_value < 1.0:
                    max_raw_content_value = 1.0

                # Pre-process (dilate/blur) — identical to preview pipeline
                try:
                    processed = process_depth_batch(
                        batch_depth_numpy_raw=frame_raw,
                        depth_gamma=depth_gamma,
                        depth_dilate_size_x=depth_dilate_size_x,
                        depth_dilate_size_y=depth_dilate_size_y,
                        depth_blur_size_x=depth_blur_size_x,
                        depth_blur_size_y=depth_blur_size_y,
                        max_raw_value=max_raw_content_value,
                        depth_dilate_left=depth_dilate_left,
                        depth_blur_left=depth_blur_left,
                        depth_blur_left_mix=depth_blur_left_mix,
                        skip_preprocessing=False,
                        debug_task_name="EstimateMaxTotal",
                    )
                except Exception:
                    processed = frame_raw  # fallback

                # Extract 2-D normalised depth array
                try:
                    if hasattr(processed, "ndim") and processed.ndim == 4:
                        depth_norm = processed[0, ..., 0]
                    elif hasattr(processed, "ndim") and processed.ndim == 3:
                        depth_norm = processed[0, ...]
                    else:
                        depth_norm = processed
                except Exception:
                    depth_norm = processed[0, ..., 0]

                try:
                    maxv = float(np.max(depth_norm))
                except Exception:
                    maxv = 1.0

                # If still in raw code space, scale using fixed bit-depth ranges
                if maxv > 1.5:
                    if maxv <= 256.0:
                        depth_norm = depth_norm / 255.0
                    elif maxv <= 1024.0:
                        depth_norm = depth_norm / 1023.0
                    elif maxv <= 4096.0:
                        depth_norm = depth_norm / 4095.0
                    elif maxv <= 65536.0:
                        depth_norm = depth_norm / 65535.0
                    else:
                        depth_norm = depth_norm / float(maxv)
                    depth_norm = np.clip(depth_norm, 0.0, 1.0)
                    if depth_gamma and abs(depth_gamma - 1.0) > 1e-6:
                        inv = np.clip(1.0 - depth_norm, 0.0, 1.0)
                        depth_norm = 1.0 - np.power(inv, float(depth_gamma))

                depth_norm = np.clip(depth_norm, 0.0, 1.0)

                # Pixel sub-sample (match preview stride)
                ds = depth_norm[
                    :: max(1, int(pixel_stride)),
                    :: max(1, int(pixel_stride)),
                ].astype(np.float32, copy=False)
                valid = (ds > 0.001) & (ds < 0.999)
                if not np.any(valid):
                    continue

                dmin = float(np.min(ds[valid]))
                dmax = float(np.max(ds[valid]))

                # TV-range compensation
                tv_disp_comp = 1.0
                if not enable_global_norm:
                    try:
                        if (
                            _infer_depth_bit_depth(depth_stream_info) > 8
                            and str(
                                (depth_stream_info or {}).get("color_range", "unknown")
                            ).lower()
                            == "tv"
                        ):
                            tv_disp_comp = 1.0 / (
                                DEPTH_VIS_TV10_WHITE_NORM - DEPTH_VIS_TV10_BLACK_NORM
                            )
                    except Exception:
                        tv_disp_comp = 1.0

                scale     = 2.0 * (float(max_disp) / 20.0) * tv_disp_comp
                min_pct   = (dmin - float(convergence_point)) * scale
                max_pct   = (dmax - float(convergence_point)) * scale

                depth_pct = abs(min_pct) if min_pct < 0 else 0.0
                pop_pct   = max_pct      if max_pct  > 0 else 0.0
                total     = float(depth_pct + pop_pct)

                if max_total is None or total > max_total:
                    max_total = total

            except Exception:
                continue

        # -- Cleanup -------------------------------------------------------
        try:
            if hasattr(depth_reader, "close"):
                depth_reader.close()
        except Exception:
            pass

        return max_total
