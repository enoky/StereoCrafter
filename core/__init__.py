"""StereoCrafter core modules package.

This package contains modularized components for the StereoCrafter
2D-to-3D video conversion tool.
"""

from .common import (
    VideoIO,
    read_video_frames,
)

from .ui import (
    ThemeManager,
    DARK_COLORS,
    LIGHT_COLORS,
    PreviewFrameBuffer,
    SBSPreviewWindow,
)

# Specialized sub-packages like .splatting should be imported directly 
# from their respective modules to avoid unnecessary dependency loading
# and log messages in applications that don't use them (like Merging GUI).

__all__ = [
    'ThemeManager',
    'DARK_COLORS',
    'LIGHT_COLORS',
    'PreviewFrameBuffer',
    'SBSPreviewWindow',
    'VideoIO',
    'read_video_frames',
]
