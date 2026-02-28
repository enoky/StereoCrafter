"""Convergence and depth metric cache service.

Centralises all per-clip cache state that was previously scattered across
``SplatterGUI.__init__``.  The GUI creates a single ``ConvergenceCache``
instance and delegates cache reads / writes / invalidation here.

The class is deliberately GUI-agnostic: it stores plain Python values,
never touches Tkinter, and is therefore unit-testable.
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ConvergenceCache:
    """Owns per-clip convergence and depth-metric caches.

    Attributes
    ----------
    auto_conv : dict
        ``{"Average": float|None, "Peak": float|None}`` for the current clip.
    auto_conv_path : str | None
        Depth-map path that *auto_conv* is keyed to.
    dp_total_est : dict
        Estimated max Total(D+P) keyed by signature string.
    dp_total_true : dict
        Measured (render-time) max Total(D+P) keyed by signature string.
    clip_norm : dict
        ``{depth_path: (global_min, global_max)}`` — cached normalisation
        stats per depth-map file.
    dp_total_max_seen : float | None
        Running max of the most-recently computed D+P estimate (UI overlay).
    """

    def __init__(self) -> None:
        self.reset_all()

    # ------------------------------------------------------------------
    # Full reset
    # ------------------------------------------------------------------
    def reset_all(self) -> None:
        """Clear every cache bucket — typically on project/folder change."""
        self.auto_conv: Dict[str, Optional[float]] = {"Average": None, "Peak": None}
        self.auto_conv_path: Optional[str] = None
        self.dp_total_est: Dict[str, float] = {}
        self.dp_total_true: Dict[str, float] = {}
        self.clip_norm: Dict[str, Tuple[float, float]] = {}
        self.dp_total_max_seen: Optional[float] = None

    # ------------------------------------------------------------------
    # Auto-convergence helpers
    # ------------------------------------------------------------------
    def clear_auto_conv(self) -> None:
        """Invalidate auto-convergence values (e.g. on clip navigation)."""
        self.auto_conv = {"Average": None, "Peak": None}
        self.auto_conv_path = None

    def set_auto_conv(
        self,
        avg: Optional[float],
        peak: Optional[float],
        depth_path: Optional[str] = None,
    ) -> None:
        """Store auto-convergence results for both modes at once."""
        self.auto_conv["Average"] = avg
        self.auto_conv["Peak"] = peak
        if depth_path is not None:
            self.auto_conv_path = depth_path

    def is_auto_conv_stale(self, current_depth_path: Optional[str]) -> bool:
        """Return True when the cached values belong to a different clip."""
        return current_depth_path != self.auto_conv_path

    def has_auto_conv(self) -> bool:
        """Return True if at least one convergence mode is cached."""
        return (
            self.auto_conv.get("Average") is not None
            or self.auto_conv.get("Peak") is not None
        )

    # ------------------------------------------------------------------
    # D+P estimate helpers
    # ------------------------------------------------------------------
    def store_dp_est(self, sig: str, value: float) -> None:
        """Cache an estimated max Total(D+P) keyed by *sig*."""
        self.dp_total_est[sig] = value

    def get_dp_est(self, sig: str) -> Optional[float]:
        """Return cached estimate or *None*."""
        v = self.dp_total_est.get(sig)
        return float(v) if v is not None else None

    def store_dp_true(self, sig: str, value: float) -> None:
        """Cache a measured (render-time) max Total(D+P)."""
        self.dp_total_true[sig] = value

    def get_dp_true(self, sig: str) -> Optional[float]:
        """Return cached measured value or *None*."""
        v = self.dp_total_true.get(sig)
        return float(v) if v is not None else None

    # ------------------------------------------------------------------
    # Clip normalisation helpers
    # ------------------------------------------------------------------
    def get_clip_norm(self, depth_path: str) -> Optional[Tuple[float, float]]:
        """Return ``(global_min, global_max)`` or *None*."""
        return self.clip_norm.get(depth_path)

    def store_clip_norm(self, depth_path: str, global_min: float, global_max: float) -> None:
        self.clip_norm[depth_path] = (global_min, global_max)
