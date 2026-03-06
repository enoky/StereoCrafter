"""Proxy for core utilities for backward compatibility.

This module redirects calls to the newly modularized locations in core.common and core.ui.
It also hosts project-specific FFmpeg video IO logic.
"""

# Hardware/GPU Utils - only CUDA_AVAILABLE is used
from core.common.gpu_utils import CUDA_AVAILABLE

import os
import json
import shutil
import threading
from typing import Optional
import logging
import subprocess
import time

logger = logging.getLogger(__name__)


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

    output_codec, output_pix_fmt, default_cpu_crf, output_profile = "libx264", "yuv420p", "23", "main"
    x265_params, default_nvenc_cq = [], "23"

    if user_output_crf is not None:
        default_cpu_crf = str(user_output_crf)
        default_nvenc_cq = str(user_output_crf)

    is_hdr = (
        video_stream_info
        and video_stream_info.get("color_primaries") == "bt2020"
        and video_stream_info.get("transfer_characteristics") == "smpte2084"
    )
    orig_pix = video_stream_info.get("pix_fmt", "") if video_stream_info else ""
    is_high_bit = "10" in orig_pix or "12" in orig_pix or "16" in orig_pix

    if is_hdr or (is_high_bit and video_stream_info.get("codec_name") in ("hevc", "prores", "dnxhd", "dnxhr")):
        output_codec = "hevc_nvenc" if CUDA_AVAILABLE else "libx265"
        output_pix_fmt = "yuv420p10le"
        output_profile = "main10"
    else:
        output_codec = "h264_nvenc" if CUDA_AVAILABLE else "libx264"

    ffmpeg_cmd.extend(["-c:v", output_codec])
    if "nvenc" in output_codec:
        ffmpeg_cmd.extend(["-preset", "medium", "-cq", default_nvenc_cq])
    else:
        ffmpeg_cmd.extend(["-crf", default_cpu_crf])
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


def set_util_logger_level(level):
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)


def start_ffmpeg_pipe_process(
    content_width,
    content_height,
    final_output_mp4_path,
    fps,
    video_stream_info,
    output_format_str="",
    user_output_crf=None,
    pad_to_16_9=False,
    debug_label=None,
    encoding_options=None,
) -> Optional[subprocess.Popen]:
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
    codec, pix, crf = "libx264", "yuv420p", "23"
    if CUDA_AVAILABLE:
        codec = "h264_nvenc"
    if user_output_crf is not None:
        crf = str(user_output_crf)
    cmd.extend(["-c:v", codec])
    if "nvenc" in codec:
        cmd.extend(["-preset", "p4", "-qp", crf])
    else:
        cmd.extend(["-crf", crf])
    cmd.extend(["-pix_fmt", pix, "-movflags", "+write_colr", final_output_mp4_path])
    logger.info(f"Starting FFmpeg pipe: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def start_ffmpeg_pipe_process_dnxhr(content_width, content_height, final_output_mov_path, fps, dnxhr_profile="HQX"):
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
