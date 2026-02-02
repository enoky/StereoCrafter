"""Main GUI module for StereoCrafter Splatting application.

This module provides the main SplatterGUI class which serves as a coordinator
delegating functionality to specialized modules in the core.splatting package.

The GUI follows a modular architecture where:
- ConfigManager handles configuration loading/saving
- ThemeManager handles dark/light mode switching
- BorderScanner handles automatic border detection
- ConvergenceEstimatorWrapper handles auto-convergence
- PreviewRenderer handles preview frame generation
- BatchProcessor handles batch video processing
- GUI widgets are created using helpers from gui_widgets

Example:
    from core.splatting.main_gui import SplatterGUI
    app = SplatterGUI()
    app.mainloop()
"""

# Standard library imports
import gc
import glob
import json
import logging
import math
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
from ttkthemes import ThemedTk

# tkinter imports
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkfont

# Local/core imports - Stage 9: Updated to use modular architecture
from core.common import ThemeManager
from core.splatting import (
    ConfigManager,
    load_config,
    save_config,
    SPLATTER_DEFAULT_CONFIG,
    BorderScanner,
    ConvergenceEstimatorWrapper,
    ForwardWarpStereo,
    FusionSidecarGenerator,
    PreviewRenderer,
    BatchProcessor,
    ProcessingTask,
    ProcessingSettings,
    BatchSetupResult,
    compute_global_depth_stats,
    load_pre_rendered_depth,
    FFmpegDepthPipeReader,
    DEPTH_VIS_TV10_BLACK_NORM,
    DEPTH_VIS_TV10_WHITE_NORM,
    # GUI Widget helpers
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

# Dependency imports
from dependency.stereocrafter_util import (
    logger,
    get_video_stream_info,
    draw_progress_bar,
    check_cuda_availability,
    release_cuda_memory,
    CUDA_AVAILABLE,
    set_util_logger_level,
    start_ffmpeg_pipe_process,
    custom_blur,
    custom_dilate,
    custom_dilate_left,
    create_single_slider_with_label_updater,
    create_dual_slider_layout,
    SidecarConfigManager,
    apply_dubois_anaglyph,
    apply_optimized_anaglyph,
)

from dependency.video_previewer import VideoPreviewer

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    # Fallback/stub for systems without moviepy
    class VideoFileClip:
        def __init__(self, *args, **kwargs):
            logging.warning("moviepy.editor not found. Frame counting disabled.")

        def close(self):
            pass

        @property
        def fps(self):
            return None

        @property
        def duration(self):
            return None


# Version information
GUI_VERSION = "26-01-30.0"


class SplatterGUI(ThemedTk):
    """Main GUI class for StereoCrafter Splatting application.
    
    This class serves as a coordinator, delegating functionality to specialized
    modules while maintaining GUI state and handling user interactions.
    
    The class initializes and coordinates:
    - ConfigManager: Configuration loading/saving
    - ThemeManager: Dark/light theme management
    - BorderScanner: Automatic border detection
    - ConvergenceEstimatorWrapper: Auto-convergence estimation
    - PreviewRenderer: Preview frame generation
    - BatchProcessor: Batch video processing
    
    Attributes:
        UI_PROCESS_COL_MIN: Minimum width for process settings column
        UI_DEPTH_COL_MIN: Minimum width for depth adjustment column
        APP_CONFIG_DEFAULTS: Default configuration values
        SIDECAR_KEY_MAP: Mapping between sidecar and internal config keys
    """

    # --- UI MINIMUM WIDTHS ---
    UI_PROCESS_COL_MIN = 330
    UI_DEPTH_COL_MIN = 520

    # --- GLOBAL CONFIGURATION DICTIONARY ---
    APP_CONFIG_DEFAULTS = {
        # File Extensions
        "SIDECAR_EXT": ".fssidecar",
        "OUTPUT_SIDECAR_EXT": ".spsidecar",
        "DEFAULT_CONFIG_FILENAME": "config_splat.splatcfg",
        # GUI/Processing Defaults (Used for reset/fallback)
        "MAX_DISP": "30.0",
        "CONV_POINT": "0.5",
        "PROC_LENGTH": "-1",
        "BATCH_SIZE_FULL": "10",
        "BATCH_SIZE_LOW": "15",
        "CRF_OUTPUT": "23",
        # Depth Processing Defaults
        "DEPTH_GAMMA": "1.0",
        "DEPTH_DILATE_SIZE_X": "3",
        "DEPTH_DILATE_SIZE_Y": "3",
        "DEPTH_BLUR_SIZE_X": "5",
        "DEPTH_BLUR_SIZE_Y": "5",
        "DEPTH_DILATE_LEFT": "0",
        "DEPTH_BLUR_LEFT": "0",
        "DEPTH_BLUR_LEFT_MIX": "0.5",
        "BORDER_WIDTH": "0.0",
        "BORDER_BIAS": "0.0",
        "BORDER_LEFT": "0.0",
        "BORDER_RIGHT": "0.0",
        "BORDER_MODE": "Off",
        "AUTO_BORDER_L": "0.0",
        "AUTO_BORDER_R": "0.0",
    }

    # Maps Sidecar JSON Key to the internal variable key
    SIDECAR_KEY_MAP = {
        "convergence_plane": "CONV_POINT",
        "max_disparity": "MAX_DISP",
        "gamma": "DEPTH_GAMMA",
        "depth_dilate_size_x": "DEPTH_DILATE_SIZE_X",
        "depth_dilate_size_y": "DEPTH_DILATE_SIZE_Y",
        "depth_blur_size_x": "DEPTH_BLUR_SIZE_X",
        "depth_blur_size_y": "DEPTH_BLUR_SIZE_Y",
        "depth_dilate_left": "DEPTH_DILATE_LEFT",
        "depth_blur_left": "DEPTH_BLUR_LEFT",
        "depth_blur_left_mix": "DEPTH_BLUR_LEFT_MIX",
        "frame_overlap": "FRAME_OVERLAP",
        "input_bias": "INPUT_BIAS",
        "selected_depth_map": "SELECTED_DEPTH_MAP",
        "left_border": "BORDER_LEFT",
        "right_border": "BORDER_RIGHT",
        "border_mode": "BORDER_MODE",
        "auto_border_L": "AUTO_BORDER_L",
        "auto_border_R": "AUTO_BORDER_R",
    }

    MOVE_TO_FINISHED_ENABLED = True

    def __init__(self):
        """Initialize the SplatterGUI application.
        
        Sets up the main window, initializes coordinator modules,
        creates tkinter variables, builds the UI, and starts the
        main event loop components.
        """
        super().__init__(theme="default")
        self.title(f"Stereocrafter Splatting (Batch) {GUI_VERSION}")

        # --- Stage 9: Initialize coordinator modules ---
        self.config_manager = ConfigManager(
            defaults=SPLATTER_DEFAULT_CONFIG,
            config_filename="config_splat.splatcfg"
        )
        self.theme_manager = None  # Initialized after dark_mode_var is created
        self.border_scanner = None  # Initialized after widgets are created
        self.convergence_estimator = ConvergenceEstimatorWrapper()
        self.preview_renderer = PreviewRenderer(cuda_available=CUDA_AVAILABLE)
        self.batch_processor = None  # Initialized when processing starts
        
        # Load configuration
        self.app_config = self.config_manager.load()
        self.help_texts = {}
        self._load_help_texts()
        
        # Sidecar manager (from dependency)
        self.sidecar_manager = SidecarConfigManager()
        
        # Caches and state
        self._dp_total_est_cache = {}
        self._dp_total_true_cache = {}
        self._dp_total_true_active_sig = None
        self._dp_total_true_active_val = None
        self._auto_pass_csv_cache = None
        self._auto_pass_csv_path = None
        self._auto_conv_cache = {"Average": None, "Peak": None}
        self._auto_conv_cached_path = None
        self._is_auto_conv_running = False
        self._preview_debounce_timer = None
        self.slider_label_updaters = []
        self.set_convergence_value_programmatically = None
        self._clip_norm_cache: Dict[str, Tuple[float, float]] = {}
        self._gn_warning_shown = False

        # Startup flag
        self._is_startup = True
        self.debug_mode_var = tk.BooleanVar(
            value=self.app_config.get("debug_mode_enabled", False)
        )
        self._debug_logging_enabled = False
        
        # Window geometry
        self.window_x = self.app_config.get("window_x", None)
        self.window_y = self.app_config.get("window_y", None)
        self.window_width = self.app_config.get("window_width", 620)
        self.window_height = self.app_config.get("window_height", 750)

        # --- Initialize all tkinter variables ---
        self._init_variables()
        
        # --- Stage 9: Initialize theme manager after dark_mode_var exists ---
        self.theme_manager = ThemeManager(
            dark_mode_var=self.dark_mode_var,
            config=self.app_config
        )

        # Processing control
        self.stop_event = threading.Event()
        self.progress_queue = queue.Queue()
        self.processing_thread = None

        # Create UI
        self._create_widgets()
        self._setup_keyboard_shortcuts()
        self.style = ttk.Style()

        # Apply theme and geometry
        self.update_idletasks()
        self._apply_theme(is_startup=True)
        self._set_saved_geometry()
        self._is_startup = False
        self._configure_logging()

        # Initialize widget states
        self.after(10, self.toggle_processing_settings_fields)
        self.after(10, self._toggle_sidecar_update_button_state)

        # Multi-map initialization
        if self.multi_map_var.get():
            self.after(15, self._on_multi_map_toggle)

        # Apply preview overlays
        self.after(20, self._apply_preview_overlay_toggles)

        # Start queue checker
        self.after(100, self.check_queue)

        # Bind closing protocol
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

        # Slider widgets list
        self.slider_widgets = []
        
        # --- Stage 9: Initialize border scanner after widgets are created ---
        self.border_scanner = BorderScanner(gui_context=self)

    def _init_variables(self):
        """Initialize all tkinter variables from configuration.
        
        Creates StringVar, BooleanVar, and other tkinter variables
        with values loaded from the configuration or defaults.
        """
        defaults = self.APP_CONFIG_DEFAULTS

        # Theme and display
        self.dark_mode_var = tk.BooleanVar(
            value=self.app_config.get("dark_mode_enabled", False)
        )

        # Input/Output paths
        self.input_source_clips_var = tk.StringVar(
            value=self.app_config.get("input_source_clips", "./input_source_clips")
        )
        self.input_depth_maps_var = tk.StringVar(
            value=self.app_config.get("input_depth_maps", "./input_depth_maps")
        )
        self.output_splatted_var = tk.StringVar(
            value=self.app_config.get("output_splatted", "./output_splatted")
        )

        # Multi-map mode
        self.multi_map_var = tk.BooleanVar(
            value=bool(self.app_config.get("multi_map_enabled", False))
        )
        self.selected_depth_map_var = tk.StringVar(value="")
        self.depth_map_subfolders = []
        self.depth_map_radio_buttons = []
        self.depth_map_radio_dict = {}
        self._current_video_sidecar_map = None
        self._suppress_sidecar_map_update = False
        self._last_loaded_source_video = None
        
        # Trace depth map folder changes
        self.input_depth_maps_var.trace_add(
            "write", lambda *args: self._on_depth_map_folder_changed()
        )

        # Processing settings
        self.max_disp_var = tk.StringVar(
            value=self.app_config.get("max_disp", defaults["MAX_DISP"])
        )
        self.process_length_var = tk.StringVar(
            value=self.app_config.get("process_length", defaults["PROC_LENGTH"])
        )
        self.process_from_var = tk.StringVar(value="")
        self.process_to_var = tk.StringVar(value="")
        self.batch_size_var = tk.StringVar(
            value=self.app_config.get("batch_size", defaults["BATCH_SIZE_FULL"])
        )

        # Output settings
        self.dual_output_var = tk.BooleanVar(
            value=self.app_config.get("dual_output", False)
        )
        self.enable_global_norm_var = tk.BooleanVar(
            value=self.app_config.get("enable_global_norm", False)
        )
        self.enable_full_res_var = tk.BooleanVar(
            value=self.app_config.get("enable_full_resolution", True)
        )
        self.enable_low_res_var = tk.BooleanVar(
            value=self.app_config.get("enable_low_resolution", True)
        )
        self.pre_res_width_var = tk.StringVar(
            value=self.app_config.get("pre_res_width", "1024")
        )
        self.pre_res_height_var = tk.StringVar(
            value=self.app_config.get("pre_res_height", "512")
        )
        self.low_res_batch_size_var = tk.StringVar(
            value=self.app_config.get("low_res_batch_size", defaults["BATCH_SIZE_LOW"])
        )
        self.zero_disparity_anchor_var = tk.StringVar(
            value=self.app_config.get("convergence_point", defaults["CONV_POINT"])
        )
        self.output_crf_var = tk.StringVar(
            value=self.app_config.get("output_crf", defaults["CRF_OUTPUT"])
        )
        
        # Separate CRF values for Full vs Low output
        _legacy_crf = self.app_config.get("output_crf", defaults["CRF_OUTPUT"])
        self.output_crf_full_var = tk.StringVar(
            value=self.app_config.get("output_crf_full", _legacy_crf)
        )
        self.output_crf_low_var = tk.StringVar(
            value=self.app_config.get("output_crf_low", _legacy_crf)
        )
        
        # Output color metadata tags
        self.color_tags_mode_var = tk.StringVar(
            value=self.app_config.get("color_tags_mode", "Auto")
        )

        # Dev Tools
        self.skip_lowres_preproc_var = tk.BooleanVar(
            value=bool(self.app_config.get("skip_lowres_preproc", False))
        )
        self.track_dp_total_true_on_render_var = tk.BooleanVar(value=False)

        self.move_to_finished_var = tk.BooleanVar(
            value=self.app_config.get("move_to_finished", True)
        )

        # Preview overlays
        self.crosshair_enabled_var = tk.BooleanVar(
            value=bool(self.app_config.get("crosshair_enabled", False))
        )
        self.crosshair_white_var = tk.BooleanVar(
            value=bool(self.app_config.get("crosshair_white", False))
        )
        self.crosshair_multi_var = tk.BooleanVar(
            value=bool(self.app_config.get("crosshair_multi", False))
        )
        self.depth_pop_enabled_var = tk.BooleanVar(
            value=bool(self.app_config.get("depth_pop_enabled", False))
        )

        self.auto_convergence_mode_var = tk.StringVar(
            value=self.app_config.get("auto_convergence_mode", "Off")
        )

        # Depth pre-processing
        self.depth_gamma_var = tk.StringVar(
            value=self.app_config.get("depth_gamma", defaults["DEPTH_GAMMA"])
        )
        
        # Handle backward compatibility for dilate size
        _ddx = self.app_config.get(
            "depth_dilate_size_x", defaults["DEPTH_DILATE_SIZE_X"]
        )
        _ddy = self.app_config.get(
            "depth_dilate_size_y", defaults["DEPTH_DILATE_SIZE_Y"]
        )
        try:
            _ddx_f = float(_ddx)
            if 30.0 < _ddx_f <= 40.0:
                _ddx = -(_ddx_f - 30.0)
        except Exception:
            pass
        try:
            _ddy_f = float(_ddy)
            if 30.0 < _ddy_f <= 40.0:
                _ddy = -(_ddy_f - 30.0)
        except Exception:
            pass
        
        self.depth_dilate_size_x_var = tk.StringVar(value=str(_ddx))
        self.depth_dilate_size_y_var = tk.StringVar(value=str(_ddy))
        self.depth_blur_size_x_var = tk.StringVar(
            value=self.app_config.get("depth_blur_size_x", defaults["DEPTH_BLUR_SIZE_X"])
        )
        self.depth_blur_size_y_var = tk.StringVar(
            value=self.app_config.get("depth_blur_size_y", defaults["DEPTH_BLUR_SIZE_Y"])
        )
        self.depth_dilate_left_var = tk.StringVar(
            value=self.app_config.get("depth_dilate_left", defaults["DEPTH_DILATE_LEFT"])
        )
        self.depth_blur_left_var = tk.StringVar(
            value=self.app_config.get("depth_blur_left", defaults["DEPTH_BLUR_LEFT"])
        )
        self.depth_blur_left_mix_var = tk.StringVar(
            value=self.app_config.get("depth_blur_left_mix", defaults["DEPTH_BLUR_LEFT_MIX"])
        )

        # Sidecar control toggles
        self.enable_sidecar_gamma_var = tk.BooleanVar(
            value=self.app_config.get("enable_sidecar_gamma", True)
        )
        self.enable_sidecar_blur_dilate_var = tk.BooleanVar(
            value=self.app_config.get("enable_sidecar_blur_dilate", True)
        )
        self.update_slider_from_sidecar_var = tk.BooleanVar(
            value=self.app_config.get("update_slider_from_sidecar", True)
        )
        self.auto_save_sidecar_var = tk.BooleanVar(
            value=self.app_config.get("auto_save_sidecar", False)
        )

        # Border settings
        self.border_width_var = tk.StringVar(
            value=self.app_config.get("border_width", defaults["BORDER_WIDTH"])
        )
        self.border_bias_var = tk.StringVar(
            value=self.app_config.get("border_bias", defaults["BORDER_BIAS"])
        )
        self.border_mode_var = tk.StringVar(
            value=self.app_config.get("border_mode", defaults["BORDER_MODE"])
        )
        self.auto_border_L_var = tk.StringVar(
            value=self.app_config.get("auto_border_L", defaults["AUTO_BORDER_L"])
        )
        self.auto_border_R_var = tk.StringVar(
            value=self.app_config.get("auto_border_R", defaults["AUTO_BORDER_R"])
        )

        # Add traces for automatic border calculation
        self.zero_disparity_anchor_var.trace_add(
            "write", self._on_convergence_or_disparity_changed
        )
        self.max_disp_var.trace_add("write", self._on_convergence_or_disparity_changed)
        self.border_mode_var.trace_add("write", self._on_border_mode_change)

        # Previewer settings
        self.preview_source_var = tk.StringVar(
            value=self.app_config.get("preview_source", "Splat Result")
        )
        self.preview_size_var = tk.StringVar(
            value=self.app_config.get("preview_size", "75%")
        )

        # Current processing information display
        self.processing_filename_var = tk.StringVar(value="N/A")
        self.processing_resolution_var = tk.StringVar(value="N/A")
        self.processing_frames_var = tk.StringVar(value="N/A")
        self.processing_disparity_var = tk.StringVar(value="N/A")
        self.processing_convergence_var = tk.StringVar(value="N/A")
        self.processing_task_name_var = tk.StringVar(value="N/A")
        self.processing_gamma_var = tk.StringVar(value="N/A")
        self.processing_map_var = tk.StringVar(value="N/A")

        # Widget management
        self.widgets_to_disable = []

    def _create_widgets(self):
        """Create all GUI widgets and layout.
        
        This method builds the complete user interface including:
        - Menu bar
        - Input/Output folder selection
        - Processing settings
        - Depth adjustment controls
        - Border settings
        - Preview panel
        - Control buttons
        - Status bar
        
        Uses helper functions from core.splatting.gui_widgets where applicable.
        """
        # --- Menu Bar ---
        self._create_menu_bar()
        
        # --- Main Container Frame ---
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Input/Output Folders Frame ---
        self._create_folder_selection_frame()

        # --- Process Resolution and Settings Frame ---
        self._create_processing_settings_frame()

        # --- Depth Adjustment Frame ---
        self._create_depth_adjustment_frame()

        # --- Preview Frame ---
        self._create_preview_frame()

        # --- Control Buttons Frame ---
        self._create_control_buttons_frame()

        # --- Current Processing Information Frame ---
        self._create_processing_info_frame()

        # --- Status Bar ---
        self._create_status_bar()

    def _create_menu_bar(self):
        """Create the menu bar with File, Settings, Tools, and Help menus."""
        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)

        # File Menu
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(
            label="Load Settings", command=self.load_settings
        )
        self.file_menu.add_command(
            label="Save Settings", command=self.save_settings
        )
        self.file_menu.add_separator()
        self.file_menu.add_command(
            label="Exit", command=self.exit_app
        )

        # Settings Menu
        self.settings_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Settings", menu=self.settings_menu)
        self.settings_menu.add_checkbutton(
            label="Dark Mode",
            variable=self.dark_mode_var,
            command=self._apply_theme
        )
        self.settings_menu.add_separator()
        self.settings_menu.add_command(
            label="Reset to Defaults", command=self.reset_to_defaults
        )

        # Tools Menu
        self.tools_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Tools", menu=self.tools_menu)
        self.tools_menu.add_command(
            label="Fusion Export to Sidecar",
            command=self._on_fusion_export_menu
        )
        self.tools_menu.add_command(
            label="Custom Fusion Sidecar Export",
            command=self._on_custom_fusion_export_menu
        )

        # Help Menu
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(
            label="About", command=self._show_about
        )

    def _create_folder_selection_frame(self):
        """Create the Input/Output folder selection section."""
        folder_frame = ttk.LabelFrame(
            self.main_container, text="Input / Output Folders", padding=5
        )
        folder_frame.pack(fill=tk.X, pady=(0, 5))

        # Source clips folder
        ttk.Label(folder_frame, text="Source Clips:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(folder_frame, textvariable=self.input_source_clips_var).grid(
            row=0, column=1, sticky=tk.EW, padx=5
        )
        ttk.Button(
            folder_frame, text="Browse...",
            command=lambda: self._browse_folder(self.input_source_clips_var)
        ).grid(row=0, column=2)

        # Depth maps folder
        ttk.Label(folder_frame, text="Depth Maps:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(folder_frame, textvariable=self.input_depth_maps_var).grid(
            row=1, column=1, sticky=tk.EW, padx=5
        )
        ttk.Button(
            folder_frame, text="Browse...",
            command=lambda: self._browse_folder(self.input_depth_maps_var)
        ).grid(row=1, column=2)

        # Output folder
        ttk.Label(folder_frame, text="Output:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(folder_frame, textvariable=self.output_splatted_var).grid(
            row=2, column=1, sticky=tk.EW, padx=5
        )
        ttk.Button(
            folder_frame, text="Browse...",
            command=lambda: self._browse_folder(self.output_splatted_var)
        ).grid(row=2, column=2)

        # Multi-map checkbox
        ttk.Checkbutton(
            folder_frame, text="Multi-Map Mode",
            variable=self.multi_map_var,
            command=self._on_multi_map_toggle
        ).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))

        folder_frame.columnconfigure(1, weight=1)

    def _create_processing_settings_frame(self):
        """Create the processing settings section."""
        process_frame = ttk.LabelFrame(
            self.main_container, text="Processing Settings", padding=5
        )
        process_frame.pack(fill=tk.X, pady=(0, 5))

        # Resolution checkboxes
        res_frame = ttk.Frame(process_frame)
        res_frame.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        ttk.Checkbutton(
            res_frame, text="Full Resolution",
            variable=self.enable_full_res_var,
            command=self.toggle_processing_settings_fields
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Checkbutton(
            res_frame, text="Low Resolution",
            variable=self.enable_low_res_var,
            command=self.toggle_processing_settings_fields
        ).pack(side=tk.LEFT)

        # Max Disparity
        ttk.Label(process_frame, text="Max Disparity:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        ttk.Entry(process_frame, textvariable=self.max_disp_var, width=10).grid(
            row=1, column=1, sticky=tk.W, padx=5
        )

        # Process Length
        ttk.Label(process_frame, text="Process Length:").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        ttk.Entry(process_frame, textvariable=self.process_length_var, width=10).grid(
            row=2, column=1, sticky=tk.W, padx=5
        )

        # Batch Size
        ttk.Label(process_frame, text="Batch Size:").grid(
            row=3, column=0, sticky=tk.W, pady=2
        )
        ttk.Entry(process_frame, textvariable=self.batch_size_var, width=10).grid(
            row=3, column=1, sticky=tk.W, padx=5
        )

        process_frame.columnconfigure(1, weight=1)

    def _create_depth_adjustment_frame(self):
        """Create the depth adjustment controls section."""
        depth_frame = ttk.LabelFrame(
            self.main_container, text="Depth Adjustment", padding=5
        )
        depth_frame.pack(fill=tk.X, pady=(0, 5))

        # Convergence (Zero Disparity Anchor)
        ttk.Label(depth_frame, text="Convergence:").grid(row=0, column=0, sticky=tk.W)
        conv_scale = ttk.Scale(
            depth_frame, from_=0.0, to=1.0,
            variable=self.zero_disparity_anchor_var,
            orient=tk.HORIZONTAL
        )
        conv_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        ttk.Entry(depth_frame, textvariable=self.zero_disparity_anchor_var, width=8).grid(
            row=0, column=2
        )

        # Gamma
        ttk.Label(depth_frame, text="Gamma:").grid(row=1, column=0, sticky=tk.W, pady=2)
        gamma_scale = ttk.Scale(
            depth_frame, from_=0.1, to=3.0,
            variable=self.depth_gamma_var,
            orient=tk.HORIZONTAL
        )
        gamma_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
        ttk.Entry(depth_frame, textvariable=self.depth_gamma_var, width=8).grid(
            row=1, column=2
        )

        depth_frame.columnconfigure(1, weight=1)

    def _create_preview_frame(self):
        """Create the video preview section."""
        preview_frame = ttk.LabelFrame(
            self.main_container, text="Preview", padding=5
        )
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Preview source dropdown
        ttk.Label(preview_frame, text="Source:").grid(row=0, column=0, sticky=tk.W)
        preview_combo = ttk.Combobox(
            preview_frame,
            textvariable=self.preview_source_var,
            values=["Splat Result", "Depth Map", "Anaglyph 3D"],
            state="readonly"
        )
        preview_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        preview_combo.bind("<<ComboboxSelected>>", lambda e: self._on_preview_source_changed())

        # Create previewer
        self.previewer = VideoPreviewer(
            preview_frame,
            processing_callback=self._preview_processing_callback,
            find_sources_callback=self._find_preview_sources,
            get_params_callback=self._get_current_settings_dict,
            preview_size_var=self.preview_size_var,
        )
        self.previewer.grid(row=1, column=0, columnspan=2, sticky=tk.NSEW, pady=5)

        preview_frame.rowconfigure(1, weight=1)
        preview_frame.columnconfigure(1, weight=1)

    def _create_control_buttons_frame(self):
        """Create the control buttons section."""
        control_frame = ttk.Frame(self.main_container)
        control_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(
            control_frame, text="Start Processing",
            command=self.start_processing
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            control_frame, text="Stop",
            command=self.stop_processing
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_frame, text="Preview",
            command=self._update_preview
        ).pack(side=tk.RIGHT)

    def _create_processing_info_frame(self):
        """Create the current processing information display section."""
        self.info_frame = ttk.LabelFrame(
            self.main_container, text="Current Processing Information", padding=5
        )
        self.info_frame.pack(fill=tk.X, pady=(0, 5))

        # Create info labels
        info_items = [
            ("Filename:", self.processing_filename_var),
            ("Resolution:", self.processing_resolution_var),
            ("Frames:", self.processing_frames_var),
            ("Disparity:", self.processing_disparity_var),
            ("Convergence:", self.processing_convergence_var),
            ("Task:", self.processing_task_name_var),
        ]

        self.info_labels = []
        for i, (label_text, var) in enumerate(info_items):
            ttk.Label(self.info_frame, text=label_text).grid(
                row=i, column=0, sticky=tk.W
            )
            label = ttk.Label(self.info_frame, textvariable=var)
            label.grid(row=i, column=1, sticky=tk.W, padx=5)
            self.info_labels.append(label)

        self.info_frame.columnconfigure(1, weight=1)

    def _create_status_bar(self):
        """Create the status bar at the bottom."""
        self.status_label = ttk.Label(
            self, text="Ready", relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_keyboard_shortcuts(self):
        """Set up keyboard shortcuts for the application."""
        self.bind("<Control-o>", lambda e: self.load_settings())
        self.bind("<Control-s>", lambda e: self.save_settings())
        self.bind("<F5>", lambda e: self.start_processing())
        self.bind("<Escape>", lambda e: self.stop_processing())

    def _apply_theme(self, is_startup: bool = False):
        """Apply the selected theme (dark or light) to the GUI.
        
        Stage 9: Delegates to ThemeManager for theme application.
        
        Args:
            is_startup: Whether this is being called during initial startup
        """
        # Use ThemeManager to apply theme
        self.theme_manager.apply_theme_to_style(self.style, self)
        self.theme_manager.apply_theme_to_menus(
            [self.file_menu, self.settings_menu, self.tools_menu, self.help_menu],
            self.menubar
        )
        self.theme_manager.apply_theme_to_labels(self.info_labels)
        
        # Apply to previewer canvas
        if hasattr(self, "previewer") and hasattr(self.previewer, "preview_canvas"):
            self.theme_manager.apply_theme_to_canvas(self.previewer.preview_canvas)

        # Update idletasks for accurate sizing
        self.update_idletasks()

    def _on_fusion_export_menu(self):
        """Handle Fusion Export to Sidecar menu command."""
        generator = FusionSidecarGenerator(self, self.sidecar_manager)
        generator.generate_sidecars(filedialog, messagebox)

    def _on_custom_fusion_export_menu(self):
        """Handle Custom Fusion Sidecar Export menu command."""
        generator = FusionSidecarGenerator(self, self.sidecar_manager)
        generator.generate_custom_sidecars(filedialog, messagebox)

    def _load_config(self):
        """Load configuration from file (backward compatibility)."""
        # Stage 9: Uses ConfigManager
        self.app_config = self.config_manager.load()

    def _load_help_texts(self):
        """Load help texts from JSON file."""
        help_file = "help_texts.json"
        if os.path.exists(help_file):
            try:
                with open(help_file, "r") as f:
                    self.help_texts = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load help texts: {e}")

    def _set_saved_geometry(self):
        """Set the window geometry from saved configuration."""
        if self.window_x is not None and self.window_y is not None:
            self.geometry(f"{self.window_width}x{self.window_height}+{self.window_x}+{self.window_y}")
        else:
            self.geometry(f"{self.window_width}x{self.window_height}")

    def _configure_logging(self):
        """Configure logging based on debug mode."""
        if self.debug_mode_var.get():
            set_util_logger_level(logging.DEBUG)
            self._debug_logging_enabled = True
        else:
            set_util_logger_level(logging.INFO)
            self._debug_logging_enabled = False

    def _browse_folder(self, var):
        """Open a folder dialog and update a StringVar."""
        current_path = var.get()
        if os.path.isdir(current_path):
            initial_dir = current_path
        elif os.path.exists(current_path):
            initial_dir = os.path.dirname(current_path)
        else:
            initial_dir = None

        folder = filedialog.askdirectory(initialdir=initial_dir)
        if folder:
            var.set(folder)

    def _browse_file(self, var, filetypes_list):
        """Open a file dialog and update a StringVar."""
        current_path = var.get()
        if os.path.exists(current_path):
            initial_dir = (
                os.path.dirname(current_path)
                if os.path.isfile(current_path)
                else current_path
            )
        else:
            initial_dir = None

        file_path = filedialog.askopenfilename(
            initialdir=initial_dir, filetypes=filetypes_list
        )
        if file_path:
            var.set(file_path)

    def _safe_float(self, var, default=0.0):
        """Safely convert StringVar/BooleanVar to float."""
        try:
            val = var.get()
            if isinstance(val, bool):
                return float(val)
            return float(val)
        except (ValueError, TypeError, tk.TclError):
            return default

    # --- Placeholder methods for preview functionality ---
    def _find_preview_sources(self):
        """Find preview sources for the previewer."""
        source_path = self.input_source_clips_var.get()
        depth_path = self.input_depth_maps_var.get()
        multi_map = self.multi_map_var.get()
        
        return self.preview_renderer.find_preview_sources(
            source_path, depth_path, multi_map
        )

    def _preview_processing_callback(self, source_frame, depth_frame, settings, mode):
        """Process frames for preview."""
        return self.preview_renderer.render_preview_frame(
            source_frame, depth_frame, settings, mode
        )

    def _on_preview_source_changed(self):
        """Handle preview source change."""
        self._update_preview()

    def _update_preview(self):
        """Update the preview display."""
        if hasattr(self, "previewer"):
            self.previewer.show_current_frame()

    # --- Placeholder methods for border/convergence functionality ---
    def _on_convergence_or_disparity_changed(self, *args):
        """Handle changes to convergence or disparity values."""
        if self.border_mode_var.get() == "Auto Basic":
            self._calculate_auto_borders()

    def _on_border_mode_change(self, *args):
        """Handle border mode changes."""
        mode = self.border_mode_var.get()
        if mode == "Auto Basic":
            self._calculate_auto_borders()
        elif mode == "Auto Adv.":
            self._scan_borders_for_current_clip()

    def _calculate_auto_borders(self):
        """Calculate automatic borders in Auto Basic mode."""
        try:
            conv = float(self.zero_disparity_anchor_var.get())
            max_disp = float(self.max_disp_var.get())
            gamma = float(self.depth_gamma_var.get())
            
            # Use default values for depth edges in basic mode
            left_border, right_border = self.border_scanner.calculate_basic_border(
                conv, max_disp, gamma
            )
            
            self.auto_border_L_var.set(str(left_border))
            self.auto_border_R_var.set(str(right_border))
            self._sync_sliders_to_auto_borders()
        except Exception as e:
            logger.error(f"Auto border calculation failed: {e}")

    def _scan_borders_for_current_clip(self):
        """Scan borders for the currently loaded clip."""
        # Get current depth path (would need to be determined from current source)
        depth_path = self._get_current_depth_path()
        if not depth_path:
            return
        
        try:
            conv = float(self.zero_disparity_anchor_var.get())
            max_disp = float(self.max_disp_var.get())
            gamma = float(self.depth_gamma_var.get())
            
            result = self.border_scanner.scan_current_clip(
                depth_path=depth_path,
                conv=conv,
                max_disp=max_disp,
                gamma=gamma,
                stop_event=self.stop_event,
                status_callback=lambda msg: self.status_label.config(text=msg)
            )
            
            if result:
                left_border, right_border = result
                self.auto_border_L_var.set(str(left_border))
                self.auto_border_R_var.set(str(right_border))
                self._sync_sliders_to_auto_borders()
        except Exception as e:
            logger.error(f"Border scan failed: {e}")

    def _get_current_depth_path(self):
        """Get the depth path for the currently selected source."""
        # This would need to be implemented based on how sources are tracked
        # For now, return the depth folder path
        return self.input_depth_maps_var.get()

    def _sync_sliders_to_auto_borders(self):
        """Synchronize border sliders to auto-calculated values."""
        try:
            left = float(self.auto_border_L_var.get())
            right = float(self.auto_border_R_var.get())
            
            # Convert to width/bias representation
            width = (left + right) / 2.0
            bias = (right - left) / 2.0
            
            self.border_width_var.set(str(width))
            self.border_bias_var.set(str(bias))
        except Exception as e:
            logger.error(f"Failed to sync border sliders: {e}")

    # --- Placeholder methods for multi-map functionality ---
    def _on_multi_map_toggle(self):
        """Handle Multi-Map mode toggle."""
        if self.multi_map_var.get():
            self._scan_depth_map_folders()
        else:
            self._clear_depth_map_radio_buttons()

    def _on_depth_map_folder_changed(self):
        """Handle depth map folder path changes."""
        if self.multi_map_var.get():
            self._scan_depth_map_folders()

    def _scan_depth_map_folders(self):
        """Scan for depth map subfolders in Multi-Map mode."""
        depth_path = self.input_depth_maps_var.get()
        if not os.path.isdir(depth_path):
            return
        
        # Find subfolders
        self.depth_map_subfolders = []
        for entry in os.listdir(depth_path):
            full_path = os.path.join(depth_path, entry)
            if os.path.isdir(full_path) and entry.lower() != "sidecars":
                self.depth_map_subfolders.append(entry)
        
        self._update_depth_map_radio_buttons()

    def _update_depth_map_radio_buttons(self):
        """Update the depth map radio button UI."""
        # This would create radio buttons for each subfolder
        pass

    def _clear_depth_map_radio_buttons(self):
        """Clear the depth map radio buttons."""
        self.depth_map_subfolders = []
        self.depth_map_radio_buttons = []
        self.depth_map_radio_dict = {}

    # --- Placeholder methods for overlay functionality ---
    def _apply_preview_overlay_toggles(self):
        """Apply preview overlay settings (crosshair, depth pop)."""
        # This would apply the overlay settings to the preview
        pass

    def toggle_processing_settings_fields(self):
        """Toggle processing settings fields based on resolution selections."""
        # This would enable/disable fields based on which resolutions are selected
        pass

    def _toggle_sidecar_update_button_state(self):
        """Toggle the sidecar update button state."""
        # This would update the UI state for sidecar controls
        pass

    def _get_current_settings_dict(self):
        """Get current processing settings as a dictionary.
        
        Returns a dictionary of all current GUI settings for use by
        the VideoPreviewer and other components.
        
        Returns:
            Dictionary containing current processing parameters
        """
        return {
            "max_disp": float(self.max_disp_var.get()),
            "convergence_point": float(self.zero_disparity_anchor_var.get()),
            "depth_gamma": float(self.depth_gamma_var.get()),
            "depth_dilate_size_x": float(self.depth_dilate_size_x_var.get()),
            "depth_dilate_size_y": float(self.depth_dilate_size_y_var.get()),
            "depth_blur_size_x": float(self.depth_blur_size_x_var.get()),
            "depth_blur_size_y": float(self.depth_blur_size_y_var.get()),
            "depth_dilate_left": float(self.depth_dilate_left_var.get()),
            "depth_blur_left": float(self.depth_blur_left_var.get()),
            "border_mode": self.border_mode_var.get(),
            "border_width": float(self.border_width_var.get()),
            "border_bias": float(self.border_bias_var.get()),
            "auto_border_L": float(self.auto_border_L_var.get()),
            "auto_border_R": float(self.auto_border_R_var.get()),
            "enable_global_norm": self.enable_global_norm_var.get(),
            "dual_output": self.dual_output_var.get(),
            "process_length": int(self.process_length_var.get()),
        }

    # --- Processing methods ---
    def start_processing(self):
        """Start the batch processing workflow.
        
        Stage 9: Uses BatchProcessor for the processing workflow.
        """
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Processing", "Processing is already running.")
            return

        # Validate settings
        validation_result = self._validate_processing_settings()
        if not validation_result.is_valid:
            messagebox.showerror("Validation Error", validation_result.error_message)
            return

        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            progress_queue=self.progress_queue,
            stop_event=self.stop_event
        )

        # Reset stop event
        self.stop_event.clear()

        # Start processing in a thread
        self.processing_thread = threading.Thread(
            target=self._processing_worker,
            args=(validation_result.tasks, validation_result.settings)
        )
        self.processing_thread.start()

        self.status_label.config(text="Processing started...")

    def _validate_processing_settings(self):
        """Validate processing settings and prepare tasks.
        
        Returns:
            BatchSetupResult with validation status and prepared tasks
        """
        # This would validate all settings and prepare the task list
        # For now, return a placeholder result
        return BatchSetupResult(
            is_valid=True,
            tasks=[],
            settings=ProcessingSettings(
                max_disp=float(self.max_disp_var.get()),
                convergence=float(self.zero_disparity_anchor_var.get()),
                gamma=float(self.depth_gamma_var.get()),
            ),
            error_message=""
        )

    def _processing_worker(self, tasks, settings):
        """Worker thread for batch processing.
        
        Args:
            tasks: List of ProcessingTask objects
            settings: ProcessingSettings object
        """
        try:
            output_dir = self.output_splatted_var.get()
            self.batch_processor.run_batch_process(
                tasks=tasks,
                settings=settings,
                output_dir=output_dir,
                progress_callback=self._on_progress_update,
                completion_callback=self._on_processing_complete
            )
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            self.progress_queue.put(("error", str(e)))

    def _on_progress_update(self, progress_info):
        """Handle progress updates from the batch processor."""
        self.progress_queue.put(("progress", progress_info))

    def _on_processing_complete(self, success, message):
        """Handle processing completion."""
        self.progress_queue.put(("complete", success, message))

    def stop_processing(self):
        """Stop the current processing operation."""
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_event.set()
            self.status_label.config(text="Stopping...")

    def check_queue(self):
        """Check the progress queue for updates."""
        try:
            while True:
                msg = self.progress_queue.get_nowait()
                msg_type = msg[0]
                
                if msg_type == "progress":
                    # Update progress display
                    progress_info = msg[1]
                    self.status_label.config(text=f"Processing: {progress_info}")
                
                elif msg_type == "error":
                    messagebox.showerror("Error", msg[1])
                    self.status_label.config(text="Error occurred")
                
                elif msg_type == "complete":
                    success = msg[1]
                    message = msg[2]
                    if success:
                        self.status_label.config(text=f"Complete: {message}")
                    else:
                        self.status_label.config(text=f"Failed: {message}")
                
                self.progress_queue.task_done()
        except queue.Empty:
            pass
        
        # Schedule next check
        self.after(100, self.check_queue)

    # --- Settings management ---
    def load_settings(self):
        """Load settings from a user-selected file."""
        filename = filedialog.askopenfilename(
            defaultextension=".splatcfg",
            filetypes=[("Splat Config", "*.splatcfg"), ("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not filename:
            return

        loaded = self.config_manager.load_settings_from_file(filename)
        if loaded:
            # Sync to tkinter variables
            for key, value in loaded.items():
                var_name = f"{key}_var"
                if hasattr(self, var_name):
                    var = getattr(self, var_name)
                    try:
                        var.set(value)
                    except Exception as e:
                        logger.warning(f"Failed to set {key}: {e}")
            
            self.status_label.config(text=f"Settings loaded from {os.path.basename(filename)}")

    def save_settings(self):
        """Save settings to a user-selected file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".splatcfg",
            filetypes=[("Splat Config", "*.splatcfg"), ("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not filename:
            return

        # Get current config from tkinter variables
        current_config = self._get_current_config_from_vars()
        
        if self.config_manager.save_settings_to_file(current_config, filename):
            self.status_label.config(text=f"Settings saved to {os.path.basename(filename)}")

    def _get_current_config_from_vars(self):
        """Extract current configuration from tkinter variables."""
        config = {}
        
        # Map of tkinter variable names to config keys
        var_map = {
            "input_source_clips_var": "input_source_clips",
            "input_depth_maps_var": "input_depth_maps",
            "output_splatted_var": "output_splatted",
            "multi_map_var": "multi_map_enabled",
            "max_disp_var": "max_disp",
            "process_length_var": "process_length",
            "batch_size_var": "batch_size",
            "dual_output_var": "dual_output",
            "enable_global_norm_var": "enable_global_norm",
            "enable_full_res_var": "enable_full_resolution",
            "enable_low_res_var": "enable_low_resolution",
            "pre_res_width_var": "pre_res_width",
            "pre_res_height_var": "pre_res_height",
            "low_res_batch_size_var": "low_res_batch_size",
            "zero_disparity_anchor_var": "convergence_point",
            "output_crf_var": "output_crf",
            "output_crf_full_var": "output_crf_full",
            "output_crf_low_var": "output_crf_low",
            "color_tags_mode_var": "color_tags_mode",
            "dark_mode_var": "dark_mode_enabled",
            "move_to_finished_var": "move_to_finished",
            "crosshair_enabled_var": "crosshair_enabled",
            "crosshair_white_var": "crosshair_white",
            "crosshair_multi_var": "crosshair_multi",
            "depth_pop_enabled_var": "depth_pop_enabled",
            "auto_convergence_mode_var": "auto_convergence_mode",
            "depth_gamma_var": "depth_gamma",
            "depth_dilate_size_x_var": "depth_dilate_size_x",
            "depth_dilate_size_y_var": "depth_dilate_size_y",
            "depth_blur_size_x_var": "depth_blur_size_x",
            "depth_blur_size_y_var": "depth_blur_size_y",
            "depth_dilate_left_var": "depth_dilate_left",
            "depth_blur_left_var": "depth_blur_left",
            "depth_blur_left_mix_var": "depth_blur_left_mix",
            "enable_sidecar_gamma_var": "enable_sidecar_gamma",
            "enable_sidecar_blur_dilate_var": "enable_sidecar_blur_dilate",
            "update_slider_from_sidecar_var": "update_slider_from_sidecar",
            "auto_save_sidecar_var": "auto_save_sidecar",
            "border_width_var": "border_width",
            "border_bias_var": "border_bias",
            "border_mode_var": "border_mode",
            "auto_border_L_var": "auto_border_L",
            "auto_border_R_var": "auto_border_R",
            "preview_source_var": "preview_source",
            "preview_size_var": "preview_size",
            "debug_mode_var": "debug_mode_enabled",
        }
        
        for var_name, config_key in var_map.items():
            if hasattr(self, var_name):
                var = getattr(self, var_name)
                try:
                    config[config_key] = var.get()
                except Exception:
                    pass
        
        # Add window geometry
        config["window_width"] = self.winfo_width()
        config["window_height"] = self.winfo_height()
        config["window_x"] = self.winfo_x()
        config["window_y"] = self.winfo_y()
        
        return config

    def reset_to_defaults(self):
        """Reset all settings to their default values."""
        defaults = self.APP_CONFIG_DEFAULTS
        
        # Reset variables to defaults
        self.max_disp_var.set(defaults["MAX_DISP"])
        self.zero_disparity_anchor_var.set(defaults["CONV_POINT"])
        self.process_length_var.set(defaults["PROC_LENGTH"])
        self.batch_size_var.set(defaults["BATCH_SIZE_FULL"])
        self.depth_gamma_var.set(defaults["DEPTH_GAMMA"])
        self.border_mode_var.set(defaults["BORDER_MODE"])
        self.border_width_var.set(defaults["BORDER_WIDTH"])
        self.border_bias_var.set(defaults["BORDER_BIAS"])
        
        self.status_label.config(text="Settings reset to defaults")

    # --- Utility methods ---
    def _show_about(self):
        """Show the About dialog."""
        messagebox.showinfo(
            "About",
            f"StereoCrafter Splatting GUI\nVersion: {GUI_VERSION}\n\n"
            "A diffusion-based 2D-to-3D video conversion tool."
        )

    def exit_app(self):
        """Exit the application, saving configuration."""
        # Save configuration
        current_config = self._get_current_config_from_vars()
        self.config_manager.config = current_config
        self.config_manager.save()
        
        # Stop any running processing
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_event.set()
            self.processing_thread.join(timeout=2.0)
        
        # Destroy the window
        self.destroy()


# Make the class available at module level
__all__ = ['SplatterGUI']
