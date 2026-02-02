"""Border scanning module for automatic border detection.

Provides methods for scanning depth map edges to determine
required border widths for zero-parallax plane adjustments.
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
from decord import VideoReader, cpu

from .depth_processing import (
    DEPTH_VIS_TV10_BLACK_NORM,
    DEPTH_VIS_TV10_WHITE_NORM,
    _infer_depth_bit_depth,
)
from dependency.stereocrafter_util import get_video_stream_info

logger = logging.getLogger(__name__)


class BorderScanner:
    """Handles automatic border detection from depth maps.

    Provides methods for scanning depth map edges to determine
    required border widths for zero-parallax plane adjustments.

    Args:
        gui_context: Optional reference to GUI for status updates
    """

    def __init__(self, gui_context=None):
        """Initialize border scanner.

        Args:
            gui_context: Optional reference to GUI for status updates
        """
        self.gui_context = gui_context
        self.logger = logging.getLogger(__name__)

    def scan_current_clip(
        self,
        depth_path: str,
        conv: float,
        max_disp: float,
        gamma: float = 1.0,
        force: bool = False,
        stop_event=None,
        status_callback=None,
    ) -> Optional[Tuple[float, float]]:
        """Scan current depth map for border requirements.

        Samples frames from the depth map and calculates the maximum border
        width needed on left and right edges based on convergence and disparity
        settings.

        Args:
            depth_path: Path to depth map video
            conv: Convergence plane value (0.0 to 1.0)
            max_disp: Maximum disparity in pixels
            gamma: Gamma correction value (default: 1.0)
            force: Force rescan even if cached (unused, for API compatibility)
            stop_event: Optional threading.Event for cancellation
            status_callback: Optional callback(status_text) for status updates

        Returns:
            Tuple of (left_border_pct, right_border_pct) if successful, None otherwise
        """
        if not depth_path or not os.path.exists(depth_path):
            self.logger.warning(f"Depth path not found: {depth_path}")
            return None

        try:
            vr = VideoReader(depth_path, ctx=cpu(0))
            total_frames = len(vr)
            if total_frames == 0:
                return None

            # Show scanning status
            if status_callback:
                status_callback(f"Scanning borders for {os.path.basename(depth_path)}...")

            return self._scan_depth_video(
                vr, total_frames, conv, max_disp, gamma, stop_event, status_callback
            )

        except Exception as e:
            self.logger.error(f"Border scan failed: {e}", exc_info=True)
            if status_callback:
                status_callback("Border scan failed.")
            return None

    def scan_depth_path(
        self,
        depth_map_path: str,
        conv: float,
        max_disp: float,
        gamma: float = 1.0,
        stop_event=None,
    ) -> Optional[Tuple[float, float]]:
        """Thread-safe helper for scanning a depth-map video for border requirements.

        Samples frames from the depth map and calculates the maximum border
        width needed on left and right edges based on convergence and disparity
        settings.

        Args:
            depth_map_path: Absolute path to depth map video file
            conv: Convergence plane value (0.0 to 1.0)
            max_disp: Maximum disparity in pixels
            gamma: Gamma correction value for depth (default: 1.0)
            stop_event: Optional threading.Event for cancellation

        Returns:
            Tuple of (left_border_pct, right_border_pct) if successful,
            None if scan failed or was cancelled
        """
        try:
            vr_depth = VideoReader(depth_map_path, ctx=cpu(0))
            total_frames = len(vr_depth)
            if total_frames <= 0:
                return None

            return self._scan_depth_video(
                vr_depth, total_frames, conv, max_disp, gamma, stop_event
            )

        except Exception as e:
            self.logger.error(f"Border scan failed: {e}", exc_info=True)
            return None

    def _scan_depth_video(
        self,
        video_reader: VideoReader,
        total_frames: int,
        conv: float,
        max_disp: float,
        gamma: float = 1.0,
        stop_event=None,
        status_callback=None,
    ) -> Optional[Tuple[float, float]]:
        """Internal method to perform the actual border scanning.

        Args:
            video_reader: Active VideoReader for the depth map
            total_frames: Total number of frames to scan
            conv: Convergence plane value
            max_disp: Maximum disparity
            gamma: Gamma correction value
            stop_event: Optional threading.Event for cancellation
            status_callback: Optional callback(status_text) for status updates

        Returns:
            Tuple of (left_border_pct, right_border_pct) if successful
        """
        step = 5
        max_L = 0.0
        max_R = 0.0

        # Get TV-range compensation factor
        tv_disp_comp = self._get_tv_compensation(video_reader)

        gamma_f = float(gamma) if gamma else 1.0

        for i in range(0, total_frames, step):
            # Check for cancellation
            if stop_event and stop_event.is_set():
                break

            try:
                frame_raw = video_reader[i].asnumpy()
            except Exception:
                continue

            # Convert to grayscale if RGB
            if frame_raw.ndim == 3:
                frame = frame_raw.mean(axis=2)
            else:
                frame = frame_raw

            # Sample 5px wide at each edge
            L_sample = frame[:, :5]
            R_sample = frame[:, -5:]

            # 99th percentile to ignore noise
            d_L = np.percentile(L_sample, 99) / 255.0
            d_R = np.percentile(R_sample, 99) / 255.0

            # Apply the same gamma curve used by the render path
            if gamma_f != 1.0:
                d_L = float(np.clip(d_L, 0.0, 1.0))
                d_R = float(np.clip(d_R, 0.0, 1.0))
                d_L = 1.0 - (1.0 - d_L) ** gamma_f
                d_R = 1.0 - (1.0 - d_R) ** gamma_f

            # Calculate border widths using the same formula as Auto Basic
            b_L = max(0.0, (d_L - conv) * 2.0 * (max_disp / 20.0) * tv_disp_comp)
            b_R = max(0.0, (d_R - conv) * 2.0 * (max_disp / 20.0) * tv_disp_comp)

            max_L = max(max_L, b_L)
            max_R = max(max_R, b_R)

        max_L = min(5.0, round(float(max_L), 3))
        max_R = min(5.0, round(float(max_R), 3))

        self.logger.info(
            f"Border scan complete: L={max_L}, R={max_R} (Conv={conv:.2f}, Disp={max_disp:.1f})"
        )

        if status_callback:
            status_callback(f"Scan complete: L={max_L}%, R={max_R}%")

        return max_L, max_R

    def _get_tv_compensation(self, video_reader: VideoReader) -> float:
        """Get TV-range compensation factor for depth maps.

        TV-range 10-bit depth maps preserve the 64-940 code window;
        this compensates so the disparity feels the same as full-range.

        Args:
            video_reader: VideoReader instance

        Returns:
            Compensation factor (1.0 for full-range, scaled for TV-range)
        """
        try:
            # Try to get the file path from the reader
            if hasattr(video_reader, '_file_path'):
                depth_path = video_reader._file_path
            else:
                # Fallback - can't determine compensation
                return 1.0

            _info = get_video_stream_info(depth_path)
            if _infer_depth_bit_depth(_info) > 8:
                color_range = str((_info or {}).get("color_range", "unknown")).lower()
                if color_range == "tv":
                    return 1.0 / (
                        DEPTH_VIS_TV10_WHITE_NORM - DEPTH_VIS_TV10_BLACK_NORM
                    )
        except Exception:
            pass

        return 1.0

    @staticmethod
    def calculate_basic_border(
        convergence: float, max_disp: float, tv_comp: float = 1.0
    ) -> float:
        """Calculate border width for Auto Basic mode.

        Args:
            convergence: Convergence plane value (0.0 to 1.0)
            max_disp: Maximum disparity in pixels
            tv_comp: TV-range compensation factor (default: 1.0)

        Returns:
            Border width as percentage (capped at 5.0%)
        """
        width = max(0.0, (1.0 - convergence) * 2.0 * (max_disp / 20.0) * tv_comp)
        return min(5.0, width)

    @staticmethod
    def calculate_border_from_depth(
        depth_value: float,
        conv: float,
        max_disp: float,
        gamma: float = 1.0,
        tv_comp: float = 1.0,
    ) -> float:
        """Calculate border width from a single depth value.

        Args:
            depth_value: Normalized depth value (0.0 to 1.0)
            conv: Convergence plane value
            max_disp: Maximum disparity
            gamma: Gamma correction value
            tv_comp: TV-range compensation factor

        Returns:
            Border width as percentage
        """
        # Apply gamma if needed
        if gamma != 1.0:
            depth_value = float(np.clip(depth_value, 0.0, 1.0))
            depth_value = 1.0 - (1.0 - depth_value) ** float(gamma)

        border = max(0.0, (depth_value - conv) * 2.0 * (max_disp / 20.0) * tv_comp)
        return min(5.0, border)

    @staticmethod
    def sync_sliders_to_auto_borders(
        left_border: float, right_border: float
    ) -> Tuple[float, float]:
        """Convert left/right borders to width/bias values.

        Args:
            left_border: Left border percentage
            right_border: Right border percentage

        Returns:
            Tuple of (width, bias) values for sliders
        """
        width = max(left_border, right_border)
        if width > 0:
            if left_border >= right_border:
                bias = (right_border / left_border) - 1.0
            else:
                bias = 1.0 - (left_border / right_border)
        else:
            bias = 0.0

        return round(width, 2), round(bias, 2)
