"""UI components and management for StereoCrafter.

This package contains modules related to the GUI presentation layer,
including theming, preview windows, and frame buffering for the UI.
"""

from .sbs_preview import SBSPreviewWindow
from .theme_manager import ThemeManager, DARK_COLORS, LIGHT_COLORS, get_theme_colors
from .preview_buffer import PreviewFrameBuffer

__all__ = [
    "SBSPreviewWindow",
    "ThemeManager",
    "DARK_COLORS",
    "LIGHT_COLORS",
    "get_theme_colors",
    "PreviewFrameBuffer",
]
