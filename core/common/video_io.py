"""Video I/O utilities for StereoCrafter.

Provides video reading and frame extraction utilities using decord
for efficient video loading.
"""

import logging
from typing import Any, Optional, Tuple, Union

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


def read_video_frames(
    video_path: str,
    process_length: int,
    set_pre_res: bool,
    pre_res_width: int,
    pre_res_height: int,
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

    fps = video_reader.get_avg_fps()  # Use actual FPS from the reader

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
