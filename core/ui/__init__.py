"""UI components and management for StereoCrafter.

This package contains modules related to the GUI presentation layer,
including theming, preview windows, and frame buffering for the UI.
"""

from .sbs_preview import SBSPreviewWindow
from .theme_manager import ThemeManager, DARK_COLORS, LIGHT_COLORS, get_theme_colors
from .preview_buffer import PreviewFrameBuffer
from .encoding_settings import EncodingSettingsDialog, create_encoding_dialog
from .dnd_support import init_dnd, register_dnd_entries, configure_dnd_styles

__all__ = [
    "SBSPreviewWindow",
    "ThemeManager",
    "DARK_COLORS",
    "LIGHT_COLORS",
    "get_theme_colors",
    "PreviewFrameBuffer",
    "EncodingSettingsDialog",
    "create_encoding_dialog",
    "init_dnd",
    "register_dnd_entries",
    "configure_dnd_styles",
]
