"""Splatting GUI core modules.

This package contains the modularized components of the Splatting GUI,
organized by functionality.
"""

from .batch_processing import (
    BatchProcessor,
    ProcessingTask,
    ProcessingSettings,
    BatchSetupResult,
)

from .border_scanning import BorderScanner

from .config_manager import (
    ConfigManager,
    SPLATTER_DEFAULT_CONFIG,
    load_config,
    save_config,
    load_settings_from_file,
    save_settings_to_file,
    get_current_config,
    reset_to_defaults,
)

from .convergence import ConvergenceEstimatorWrapper

from .depth_processing import (
    compute_global_depth_stats,
    load_pre_rendered_depth,
    FFmpegDepthPipeReader,
    DEPTH_VIS_TV10_BLACK_NORM,
    DEPTH_VIS_TV10_WHITE_NORM,
)

from .forward_warp import ForwardWarpStereo

from .fusion_export import FusionSidecarGenerator

from .preview_rendering import PreviewRenderer

