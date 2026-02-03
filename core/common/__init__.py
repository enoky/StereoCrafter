"""Shared utilities across all GUI applications.

This package contains modules that are common to multiple GUI applications
within the StereoCrafter project.
"""

from .video_io import VideoIO, read_video_frames

from .theme_manager import (
    ThemeManager,
    DARK_COLORS,
    LIGHT_COLORS,
    get_theme_colors,
)

__all__ = [
    'VideoIO',
    'read_video_frames',
    'ThemeManager',
    'DARK_COLORS',
    'LIGHT_COLORS',
    'get_theme_colors',
]
