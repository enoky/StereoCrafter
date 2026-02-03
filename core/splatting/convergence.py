"""Auto-convergence estimation module using U2NETP neural network.

Provides convergence plane estimation using visual saliency analysis
of RGB and depth map pairs.
"""

import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu

logger = logging.getLogger(__name__)


class ConvergenceEstimatorWrapper:
    """Handles automatic convergence plane detection using U2NETP.

    Estimates the optimal zero-parallax plane based on visual saliency
    analysis of RGB and depth map pairs. Wraps the dependency.convergence_estimator
    neural network model.

    Args:
        model_path: Optional path to U2NETP model weights
        device: Optional torch device ('cuda' or 'cpu')
    """

    def __init__(
        self, model_path: Optional[str] = None, device: Optional[str] = None
    ):
        """Initialize convergence estimation model.

        Args:
            model_path: Path to model weights file
            device: Torch device for inference
        """
        self.logger = logging.getLogger(__name__)
        self._model_path = model_path
        self._device = device
        self._estimator = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the U2NETP model for convergence estimation."""
        try:
            # Lazy import to avoid circular dependency issues
            from dependency.convergence_estimator import (
                ConvergenceEstimator as NeuralEstimator,
            )

            self._estimator = NeuralEstimator(
                model_path=self._model_path, device=self._device
            )
            if self._estimator.model is None:
                self.logger.error(
                    "ConvergenceEstimator model failed to load."
                )
        except ImportError as e:
            self.logger.error(f"Could not import ConvergenceEstimator: {e}")
            self._estimator = None

    def is_model_loaded(self) -> bool:
        """Check if the neural network model is loaded and ready.

        Returns:
            True if model is loaded, False otherwise
        """
        return self._estimator is not None and self._estimator.model is not None

    def estimate_convergence(
        self,
        rgb_path: str,
        depth_path: str,
        process_length: int = -1,
        sample_stride: int = 6,
        gamma: float = 1.0,
        fallback_value: float = 0.5,
        stop_event: Optional[threading.Event] = None,
        scan_borders: bool = False,
    ) -> Tuple[float, float, Optional[float], Optional[float]]:
        """Estimate optimal convergence plane and optionally scan borders.

        Samples frames uniformly from the video, analyzes them using the
        U2NETP model to detect salient objects, and returns both average
        and peak convergence values. If scan_borders is True, also returns
        the maximum depth values found at the left and right 5px edges.

        Args:
            rgb_path: Path to RGB source video
            depth_path: Path to depth map video
            process_length: Number of frames to process (-1 for all)
            sample_stride: Stride between sampled frames (default: 6)
            gamma: Gamma correction for depth (default: 1.0)
            fallback_value: Value to return on failure (default: 0.5)
            stop_event: Optional threading.Event for cancellation
            scan_borders: Whether to also scan left/right edges for max depth

        Returns:
            Tuple of (average_convergence, peak_convergence, max_edge_l, max_edge_r)
        """
        if not self.is_model_loaded():
            self.logger.warning("Model not loaded, returning fallback values")
            return fallback_value, fallback_value, None, None

        try:
            # Initialize Readers
            vr_rgb = VideoReader(rgb_path, ctx=cpu(0))
            vr_depth = VideoReader(depth_path, ctx=cpu(0))

            len_rgb = len(vr_rgb)
            len_depth = len(vr_depth)

            # Sanity check
            if len_rgb == 0 or len_depth == 0:
                self.logger.warning("Empty video or depth map found.")
                return fallback_value, fallback_value, None, None

            total_frames = min(len_rgb, len_depth)

            # Respect process_length if set > 0
            if process_length > 0:
                total_frames = min(total_frames, process_length)

            # Sample frames
            indices = list(range(0, total_frames, sample_stride))

            # Ensure at least one frame is sampled
            if not indices:
                indices = [0]

            estimates = []
            max_edge_l = 0.0 if scan_borders else None
            max_edge_r = 0.0 if scan_borders else None

            self.logger.info(
                f"Auto-Converge{' + Border Scan' if scan_borders else ''}: Sampling {len(indices)} frames from {os.path.basename(rgb_path)}..."
            )

            for idx in indices:
                if stop_event and stop_event.is_set():
                    self.logger.info("Auto-Converge scan cancelled.")
                    break

                # Read RGB
                rgb_frame = vr_rgb[idx].asnumpy()  # H, W, 3 (uint8)
                # Read Depth
                depth_frame = vr_depth[idx].asnumpy()  # H, W, C or H, W

                # Depth: Handle various formats (Gray8, Gray16, RGB-encoding)
                if depth_frame.ndim == 3:
                    depth_mono = depth_frame.mean(axis=2)
                else:
                    depth_mono = depth_frame

                # --- Optional Border Scan (Lightweight) ---
                if scan_borders:
                    # Sample 5px wide at each edge
                    # We use numpy here as it's already a numpy array from decord
                    L_sample = depth_mono[:, :5]
                    R_sample = depth_mono[:, -5:]

                    # 99th percentile to ignore noise
                    d_L = float(np.percentile(L_sample, 99))
                    d_R = float(np.percentile(R_sample, 99))

                    # Normalize if uint8
                    if depth_mono.dtype == np.uint8 or d_L > 1.0 or d_R > 1.0:
                        d_L /= 255.0
                        d_R /= 255.0

                    max_edge_l = max(max_edge_l, d_L)
                    max_edge_r = max(max_edge_r, d_R)

                # Preprocess for Torch (NN inference)
                rgb_t = (
                    torch.from_numpy(rgb_frame).float().permute(2, 0, 1) / 255.0
                )

                depth_t = torch.from_numpy(depth_mono).float()
                # Normalize if not 0-1
                if depth_t.max() > 1.0:
                    depth_t = depth_t / 255.0

                # Clamp and apply gamma
                depth_t = torch.clamp(depth_t, 0.0, 1.0)
                gamma_f = float(gamma) if gamma else 1.0
                if gamma_f != 1.0:
                    depth_t = 1.0 - torch.pow((1.0 - depth_t), gamma_f)

                # Format: 1, C, H, W
                depth_t = depth_t.unsqueeze(0).unsqueeze(0)
                rgb_b = rgb_t.unsqueeze(0)

                # Predict
                res = self._estimator.predict(rgb_b, depth_t)
                estimates.extend(res)

            if not estimates:
                return fallback_value, fallback_value, max_edge_l, max_edge_r

            avg_val = sum(estimates) / len(estimates)
            # Using Max as 'Peak' estimate
            peak_val = max(estimates)

            self.logger.info(
                f"Auto-Converge Result: Avg={avg_val:.3f}, Peak={peak_val:.3f}"
            )
            if scan_borders:
                self.logger.info(f"Edge Depth Result: L={max_edge_l:.3f}, R={max_edge_r:.3f}")

            return avg_val, peak_val, max_edge_l, max_edge_r

        except Exception as e:
            self.logger.error(
                f"Auto convergence determination failed: {e}", exc_info=True
            )
            return fallback_value, fallback_value

    def calculate_hybrid_value(
        self, avg_value: float, peak_value: float
    ) -> float:
        """Calculate hybrid convergence value from average and peak.

        Args:
            avg_value: Average convergence value
            peak_value: Peak convergence value

        Returns:
            Hybrid value (average of avg and peak)
        """
        return (avg_value + peak_value) / 2.0

    def get_cached_value(
        self,
        mode: str,
        cache: Dict[str, float],
        fallback: float = 0.5,
    ) -> float:
        """Get convergence value from cache based on mode.

        Args:
            mode: Mode - 'Average', 'Peak', or 'Hybrid'
            cache: Dictionary with cached values
            fallback: Fallback value if mode not in cache

        Returns:
            Convergence value for the specified mode
        """
        if mode == "Average":
            return cache.get("Average", fallback)
        elif mode == "Peak":
            return cache.get("Peak", fallback)
        elif mode == "Hybrid":
            avg = cache.get("Average", fallback)
            peak = cache.get("Peak", fallback)
            return self.calculate_hybrid_value(avg, peak)
        else:
            return fallback
