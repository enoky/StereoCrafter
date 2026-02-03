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

from .gui_widgets import (
    Tooltip,
    create_tooltip,
    create_labeled_entry,
    create_folder_selection_row,
    create_checkbox_group,
    create_labeled_slider,
    create_button_group,
    create_section_frame,
    create_dropdown,
    configure_grid_weights,
    get_common_tooltip,
    COMMON_TOOLTIPS,
)

from .main_gui import SplatterGUI

__all__ = [
    # Config Management
    'ConfigManager',
    'SPLATTER_DEFAULT_CONFIG',
    'load_config',
    'save_config',
    'load_settings_from_file',
    'save_settings_to_file',
    'get_current_config',
    'reset_to_defaults',
    # Border & Convergence
    'BorderScanner',
    'ConvergenceEstimatorWrapper',
    # Depth Processing
    'compute_global_depth_stats',
    'load_pre_rendered_depth',
    'FFmpegDepthPipeReader',
    'DEPTH_VIS_TV10_BLACK_NORM',
    'DEPTH_VIS_TV10_WHITE_NORM',
    # Core Processing
    'ForwardWarpStereo',
    'FusionSidecarGenerator',
    'PreviewRenderer',
    # Batch Processing
    'BatchProcessor',
    'ProcessingTask',
    'ProcessingSettings',
    'BatchSetupResult',
    # GUI Widgets
    'Tooltip',
    'create_tooltip',
    'create_labeled_entry',
    'create_folder_selection_row',
    'create_checkbox_group',
    'create_labeled_slider',
    'create_button_group',
    'create_section_frame',
    'create_dropdown',
    'configure_grid_weights',
    'get_common_tooltip',
    'COMMON_TOOLTIPS',
    # Main GUI
    'SplatterGUI',
]
