"""StereoCrafter core modules package.

This package contains modularized components for the StereoCrafter
2D-to-3D video conversion tool.
"""

from .common import (
    ThemeManager,
    DARK_COLORS,
    LIGHT_COLORS,
    VideoIO,
    read_video_frames,
)

from .splatting import (
    ConfigManager,
    SPLATTER_DEFAULT_CONFIG,
    load_config,
    save_config,
    compute_global_depth_stats,
    load_pre_rendered_depth,
    FFmpegDepthPipeReader,
    ForwardWarpStereo,
    FusionSidecarGenerator,
)

__all__ = [
    'ConfigManager',
    'SPLATTER_DEFAULT_CONFIG',
    'load_config',
    'save_config',
    'ThemeManager',
    'DARK_COLORS',
    'LIGHT_COLORS',
    'VideoIO',
    'read_video_frames',
    'compute_global_depth_stats',
    'load_pre_rendered_depth',
    'FFmpegDepthPipeReader',
    'ForwardWarpStereo',
    'FusionSidecarGenerator',
]
