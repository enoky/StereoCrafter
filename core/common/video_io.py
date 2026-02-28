"""Video I/O utilities for StereoCrafter.

Provides video reading and frame extraction utilities using decord
for efficient video loading.
"""

import logging
from typing import Any, Optional, Tuple, Union

import subprocess  # Needed for FFmpeg-based preview readers

import numpy as np
from decord import VideoReader, cpu

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
    def read_video_info(
        video_path: str,
    ) -> Tuple[int, int, int, float]:
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

        logger.debug(
            f"==> Video info: {total_frames} frames, {width}x{height}, {fps} fps"
        )

        return total_frames, height, width, fps

    @staticmethod
    def read_frame(
        reader: VideoReader,
        index: int,
    ) -> np.ndarray:
        """Read a single frame from a video reader.

        Args:
            reader: Active VideoReader instance
            index: Frame index to read

        Returns:
            Frame as numpy array [H, W, C]
        """
        return reader.get_batch([index]).asnumpy()[0]

    @staticmethod
    def read_frames_batch(
        reader: VideoReader,
        indices: list,
    ) -> np.ndarray:
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
    def __init__(self, video_path: str, width: int, height: int, fps: float, total_frames: int,
                 in_range: str = "tv", in_matrix: str = "bt709"):
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
        vf_fallback = (
            f"scale={self.width}:{self.height}:flags=bicubic,"
            "format=rgb24"
        )
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
    def __init__(self, video_path: str, width: int, height: int, fps: float, total_frames: int,
                 in_range: str = "tv", in_matrix: str = "bt709"):
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
                "ffmpeg", "-v", "error", "-nostdin",
                "-i", self.video_path, "-an", "-sn", "-dn",
                "-vf", vf,
                "-frames:v", "1",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
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
            vf_fallback = (
                f"select='eq(n\\,{idx})',"
                f"scale={self.width}:{self.height}:flags=bicubic,"
                "format=rgb24"
            )

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


def read_video_frames(
    video_path: str,
    process_length: int,
    set_pre_res: bool,
    pre_res_width: int,
    pre_res_height: int,
    strict_ffmpeg_decode: bool = False,
    dataset: str = "open",
) -> Tuple[VideoReader, float, int, int, int, int, Optional[dict], int]:
    """Initialize a VideoReader for chunked reading.

    Args:
        video_path: Path to the video file
        process_length: Number of frames to process (-1 for all)
        set_pre_res: Whether to set custom resolution
        pre_res_width: Target width if set_pre_res is True
        pre_res_height: Target height if set_pre_res is True
        dataset: Dataset type (only 'open' supported)

    Returns:
        Tuple of (video_reader, fps, original_height, original_width,
                  actual_processed_height, actual_processed_width,
                  video_stream_info, total_frames_to_process)

    Raises:
        NotImplementedError: If dataset is not 'open'
    """
    if dataset == "open":
        logger.debug(f"==> Initializing VideoReader for: {video_path}")
        vid_info_only = VideoReader(
            video_path, ctx=cpu(0)
        )  # Use separate reader for info
        original_height, original_width = vid_info_only.get_batch([0]).shape[1:3]
        total_frames_original = len(vid_info_only)
        logger.debug(
            f"==> Original video shape: {total_frames_original} frames, "
            f"{original_height}x{original_width} per frame"
        )

        height_for_reader = original_height
        width_for_reader = original_width

        if set_pre_res and pre_res_width > 0 and pre_res_height > 0:
            height_for_reader = pre_res_height
            width_for_reader = pre_res_width
            logger.debug(
                f"==> Pre-processing resolution set to: "
                f"{width_for_reader}x{height_for_reader}"
            )
        else:
            logger.debug(
                f"==> Using original video resolution for reading: "
                f"{width_for_reader}x{height_for_reader}"
            )

    else:
        raise NotImplementedError(f"Dataset '{dataset}' not supported.")

    # decord automatically resizes if width/height are passed to VideoReader
    video_reader = VideoReader(
        video_path, ctx=cpu(0), width=width_for_reader, height=height_for_reader
    )

    # Verify the actual shape after Decord processing, using the first frame
    first_frame_shape = video_reader.get_batch([0]).shape
    actual_processed_height, actual_processed_width = first_frame_shape[1:3]

    fps = float(video_reader.get_avg_fps())  # Use actual FPS from the reader

    total_frames_available = len(video_reader)
    total_frames_to_process = total_frames_available  # Use available frames directly
    if process_length != -1 and process_length < total_frames_available:
        total_frames_to_process = process_length

    logger.debug(
        f"==> VideoReader initialized. Final processing dimensions: "
        f"{actual_processed_width}x{actual_processed_height}. "
        f"Total frames for processing: {total_frames_to_process}"
    )

    # Import here to avoid circular dependency
    from dependency.stereocrafter_util import get_video_stream_info

    video_stream_info = get_video_stream_info(
        video_path
    )  # Get stream info for FFmpeg later

    # If strict FFmpeg decode is requested, swap in an FFmpeg-backed reader for frame fetch.
    # This keeps decode/colorspace conversion consistent across preview + renders for problem clips.
    if strict_ffmpeg_decode:
        try:
            in_range = "tv"
            in_matrix = "bt709"
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
                width=width_for_reader,
                height=height_for_reader,
                fps=float(fps),
                total_frames=total_frames_available,
                in_range=in_range,
                in_matrix=in_matrix,
            )
        except Exception as e:
            logger.warning(
                f"Strict FFmpeg decode requested, but FFmpeg reader init failed; falling back to Decord. ({e})"
            )

    return (
        video_reader,
        fps,
        original_height,
        original_width,
        actual_processed_height,
        actual_processed_width,
        video_stream_info,
        total_frames_to_process,
    )