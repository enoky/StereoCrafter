import logging
from typing import Optional, Dict, Any, Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class PreviewFrameBuffer:
    """
    Buffer for caching processed preview frames to enable fast playback.

    Stores processed frames in memory to avoid re-processing each frame
    during playback loop. The buffer is invalidated when processing
    parameters change.
    """

    def __init__(self, max_frames: int = 500, max_memory_mb: int = 2048):
        """
        Initialize the preview frame buffer.

        Args:
            max_frames: Maximum number of frames to cache
            max_memory_mb: Maximum memory to use for缓存 (approximate)
        """
        self._main_buffer: Dict[int, Image.Image] = {}
        self._display_buffer: Dict[int, Image.Image] = {}  # Pre-scaled display images
        self._sbs_buffer: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._params_signature: Optional[str] = None
        self._video_path: Optional[str] = None
        self._preview_size: Optional[str] = None  # Track preview size to invalidate display buffer
        self._max_frames = max_frames
        self._max_memory_mb = max_memory_mb

    def _generate_signature_from_params(self, params: Dict[str, Any], video_path: str) -> str:
        """
        Generate a signature hash from processing parameters.

        Args:
            params: Dictionary of processing parameters
            video_path: Path to current video (used to invalidate on video change)

        Returns:
            String signature that changes when parameters change
        """
        preview_size = params.get("preview_size", "100%")
        key_parts = [
            video_path or "",
            str(params.get("max_disp", "")),
            str(params.get("convergence_point", "")),
            str(params.get("depth_gamma", "")),
            str(params.get("depth_dilate_size_x", "")),
            str(params.get("depth_dilate_size_y", "")),
            str(params.get("depth_blur_size_x", "")),
            str(params.get("depth_blur_size_y", "")),
            str(params.get("depth_dilate_left", "")),
            str(params.get("depth_blur_left", "")),
            str(params.get("depth_blur_left_mix", "")),
            str(params.get("left_border_pct", "")),
            str(params.get("right_border_pct", "")),
            str(params.get("preview_source", "")),
            preview_size,
            str(params.get("enable_global_norm", "")),
        ]
        return "|".join(key_parts)

    def check_and_update_buffer(self, params: Dict[str, Any], video_path: str) -> bool:
        """
        Check if buffer needs to be cleared due to parameter changes.

        Args:
            params: Current processing parameters
            video_path: Path to current video

        Returns:
            True if buffer was cleared, False otherwise
        """
        current_sig = self._generate_signature_from_params(params, video_path)
        preview_size = params.get("preview_size", "100%")

        if (
            self._params_signature != current_sig
            or self._video_path != video_path
            or self._preview_size != preview_size
        ):
            self.clear()
            self._params_signature = current_sig
            self._video_path = video_path
            self._preview_size = preview_size
            logger.debug("Preview buffer cleared due to parameter or video change")
            return True
        return False

    def get_cached_frame(self, frame_idx: int) -> Optional[Image.Image]:
        """Get a cached processed frame."""
        return self._main_buffer.get(frame_idx)

    def cache_frame(self, frame_idx: int, frame: Image.Image) -> None:
        """Cache a processed frame."""
        if len(self._main_buffer) >= self._max_frames:
            oldest_idx = min(self._main_buffer.keys())
            del self._main_buffer[oldest_idx]
            # Also remove from display buffer
            if oldest_idx in self._display_buffer:
                del self._display_buffer[oldest_idx]
        self._main_buffer[frame_idx] = frame

    def get_cached_display_frame(self, frame_idx: int) -> Optional[Image.Image]:
        """Get a cached display-ready (scaled) frame."""
        return self._display_buffer.get(frame_idx)

    def cache_display_frame(self, frame_idx: int, display_frame: Image.Image) -> None:
        """Cache a display-ready (scaled) frame."""
        if len(self._display_buffer) >= self._max_frames:
            oldest_idx = min(self._display_buffer.keys())
            del self._display_buffer[oldest_idx]
        self._display_buffer[frame_idx] = display_frame

    def get_cached_sbs_frame(self, frame_idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get cached SBS frame data (left_np, right_np)."""
        return self._sbs_buffer.get(frame_idx)

    def cache_sbs_frame(self, frame_idx: int, left_np: np.ndarray, right_np: np.ndarray) -> None:
        """Cache SBS frame data."""
        if len(self._sbs_buffer) >= self._max_frames:
            oldest_idx = min(self._sbs_buffer.keys())
            del self._sbs_buffer[oldest_idx]
        self._sbs_buffer[frame_idx] = (left_np, right_np)

    def clear(self) -> None:
        """Clear all cached frames."""
        self._main_buffer.clear()
        self._display_buffer.clear()
        self._sbs_buffer.clear()
        logger.debug("Preview frame buffers cleared")

    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics for debugging."""
        return {
            "main_buffer_size": len(self._main_buffer),
            "display_buffer_size": len(self._display_buffer),
            "sbs_buffer_size": len(self._sbs_buffer),
        }
