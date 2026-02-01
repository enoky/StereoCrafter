import gc
import os
import re
import cv2
import glob
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import write_video
from decord import VideoReader, cpu
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import tkinter.font as tkfont
from ttkthemes import ThemedTk
import json
import threading
import queue
import subprocess
import time
import logging
from typing import Optional, Tuple, Any
from PIL import Image
import math  # <--- ADD THIS

# --- Depth Map Visualization Levels ---
# These affect ONLY depth-map visualization (Preview 'Depth Map' and Map Test images),
# not the depth values used for splatting.
DEPTH_VIS_APPLY_TV_RANGE_EXPANSION_10BIT = True
DEPTH_VIS_TV10_BLACK_NORM = 64.0 / 1023.0
DEPTH_VIS_TV10_WHITE_NORM = 940.0 / 1023.0

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


# Import custom modules
CUDA_AVAILABLE = False  # start state, will check automaticly later

# --- MODIFIED IMPORT ---
from dependency.stereocrafter_util import (
    Tooltip,
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

try:
    from Forward_Warp import forward_warp

    logger.info("CUDA Forward Warp is available.")
except:
    from dependency.forward_warp_pytorch import forward_warp

    logger.info("Forward Warp Pytorch is active.")
from dependency.video_previewer import VideoPreviewer

GUI_VERSION = "26-01-23.0"


class FusionSidecarGenerator:
    """Handles parsing Fusion Export files, matching them to depth maps,
    and generating/saving FSSIDECAR files using carry-forward logic."""

    FUSION_PARAMETER_CONFIG = {
        # Key: {Label, Type, Default, FusionKey(fsexport), SidecarKey(fssidecar), Decimals}
        "convergence": {
            "label": "Convergence Plane",
            "type": float,
            "default": 0.5,
            "fusion_key": "Convergence",
            "sidecar_key": "convergence_plane",
            "decimals": 3,
        },
        "max_disparity": {
            "label": "Max Disparity",
            "type": float,
            "default": 35.0,
            "fusion_key": "MaxDisparity",
            "sidecar_key": "max_disparity",
            "decimals": 1,
        },
        "gamma": {
            "label": "Gamma Correction",
            "type": float,
            "default": 1.0,
            "fusion_key": "FrontGamma",
            "sidecar_key": "gamma",
            "decimals": 2,
        },
        # These keys exist in the sidecar manager but are usually set in the source tool
        # We include them here for completeness if Fusion ever exported them
        "frame_overlap": {
            "label": "Frame Overlap",
            "type": float,
            "default": 3,
            "fusion_key": "Overlap",
            "sidecar_key": "frame_overlap",
            "decimals": 0,
        },
        "input_bias": {
            "label": "Input Bias",
            "type": float,
            "default": 0.0,
            "fusion_key": "Bias",
            "sidecar_key": "input_bias",
            "decimals": 2,
        },
        "left_border": {
            "label": "Left Border",
            "type": float,
            "default": 0.0,
            "fusion_key": "LeftBorder",
            "sidecar_key": "left_border",
            "decimals": 3,
        },
        "right_border": {
            "label": "Right Border",
            "type": float,
            "default": 0.0,
            "fusion_key": "RightBorder",
            "sidecar_key": "right_border",
            "decimals": 3,
        },
        "manual_border": {
            "label": "Border Mode",
            "type": str,
            "default": "Off",
            "fusion_key": "BorderMode",
            "sidecar_key": "border_mode",
            "decimals": 0,
        },
        "auto_border_l": {
            "label": "Auto Border L",
            "type": float,
            "default": 0.0,
            "fusion_key": "AutoBorderL",
            "sidecar_key": "auto_border_L",
            "decimals": 3,
        },
        "auto_border_r": {
            "label": "Auto Border R",
            "type": float,
            "default": 0.0,
            "fusion_key": "AutoBorderR",
            "sidecar_key": "auto_border_R",
            "decimals": 3,
        },
    }

    def __init__(self, master_gui, sidecar_manager):
        self.master_gui = master_gui
        self.sidecar_manager = sidecar_manager
        self.logger = logging.getLogger(__name__)

    def _get_video_frame_count(self, file_path):
        """Safely gets the frame count of a video file using moviepy."""
        try:
            clip = VideoFileClip(file_path)
            fps = clip.fps
            duration = clip.duration
            if fps is None or duration is None:
                # If moviepy failed to get reliable info, fall back
                fps = 24
                if duration is None:
                    return 0

            frames = math.ceil(duration * fps)
            clip.close()
            return frames
        except Exception as e:
            self.logger.warning(
                f"Error getting frame count for {os.path.basename(file_path)}: {e}"
            )
            return 0

    def _load_and_validate_fsexport(self, file_path):
        """Loads, parses, and validates marker data from a Fusion Export file."""
        try:
            with open(file_path, "r") as f:
                export_data = json.load(f)
        except json.JSONDecodeError as e:
            messagebox.showerror(
                "File Error",
                f"Failed to parse JSON in {os.path.basename(file_path)}: {e}",
            )
            return None
        except Exception as e:
            messagebox.showerror(
                "File Error", f"Failed to read {os.path.basename(file_path)}: {e}"
            )
            return None

        markers = export_data.get("markers", [])
        if not markers:
            messagebox.showwarning(
                "Data Warning", "No 'markers' found in the export file."
            )
            return None

        # Sort markers by frame number (critical for carry-forward logic)
        markers.sort(key=lambda m: m["frame"])
        self.logger.info(
            f"Loaded {len(markers)} markers from {os.path.basename(file_path)}."
        )
        return markers

    def _scan_target_videos(self, folder):
        """Scans the target folder for video files and computes their frame counts."""
        video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
        found_files_paths = []
        for ext in video_extensions:
            found_files_paths.extend(glob.glob(os.path.join(folder, ext)))
        sorted_files_paths = sorted(found_files_paths)

        if not sorted_files_paths:
            messagebox.showwarning(
                "No Files", f"No video depth map files found in: {folder}"
            )
            return None

        target_video_data = []
        cumulative_frames = 0

        for full_path in sorted_files_paths:
            total_frames = self._get_video_frame_count(full_path)

            if total_frames == 0:
                self.logger.warning(
                    f"Skipping {os.path.basename(full_path)} due to zero frame count."
                )
                continue

            target_video_data.append(
                {
                    "full_path": full_path,
                    "basename": os.path.basename(full_path),
                    "total_frames": total_frames,
                    "timeline_start_frame": cumulative_frames,
                    "timeline_end_frame": cumulative_frames + total_frames - 1,
                }
            )
            cumulative_frames += total_frames

        self.logger.info(
            f"Scanned {len(target_video_data)} video files. Total timeline frames: {cumulative_frames}."
        )
        return target_video_data

    def generate_sidecars(self):
        """Main entry point for the Fusion Export to Sidecar generation workflow."""

        # 1. Select Fusion Export File
        export_file_path = filedialog.askopenfilename(
            defaultextension=".fsexport",
            filetypes=[
                ("Fusion Export Files", "*.fsexport.txt;*.fsexport"),
                ("All Files", "*.*"),
            ],
            title="Select Fusion Export (.fsexport) File",
        )
        if not export_file_path:
            self.master_gui.status_label.config(
                text="Fusion export selection cancelled."
            )
            return

        markers = self._load_and_validate_fsexport(export_file_path)
        if markers is None:
            self.master_gui.status_label.config(text="Fusion export loading failed.")
            return

        # 2. Select Target Depth Map Folder
        target_folder = filedialog.askdirectory(title="Select Target Depth Map Folder")
        if not target_folder:
            self.master_gui.status_label.config(
                text="Depth map folder selection cancelled."
            )
            return

        target_videos = self._scan_target_videos(target_folder)
        if target_videos is None or not target_videos:
            self.master_gui.status_label.config(text="No valid depth map videos found.")
            return

        # 3. Apply Parameters (Carry-Forward Logic)
        applied_count = 0

        # Initialize last known values with the config defaults
        last_param_vals = {}
        for key, config in self.FUSION_PARAMETER_CONFIG.items():
            last_param_vals[key] = config["default"]

        for file_data in target_videos:
            file_start_frame = file_data["timeline_start_frame"]

            # Find the most relevant marker (latest marker frame <= file_start_frame)
            relevant_marker = None
            for marker in markers:
                if marker["frame"] <= file_start_frame:
                    relevant_marker = marker
                else:
                    break

            current_param_vals = last_param_vals.copy()

            if relevant_marker and relevant_marker.get("values"):
                marker_values = relevant_marker["values"]
                updated_from_marker = False

                for key, config in self.FUSION_PARAMETER_CONFIG.items():
                    fusion_key = config["fusion_key"]
                    default_val = config["default"]

                    if fusion_key in marker_values:
                        # Attempt to cast the value from the marker to the expected type
                        val = marker_values.get(fusion_key, default_val)
                        try:
                            current_param_vals[key] = config["type"](val)
                            updated_from_marker = True
                        except (ValueError, TypeError):
                            self.logger.warning(
                                f"Marker value for '{fusion_key}' is invalid ({val}). Using previous/default value."
                            )

                if updated_from_marker:
                    applied_count += 1

            # 4. Save Sidecar JSON
            sidecar_data = {}
            for key, config in self.FUSION_PARAMETER_CONFIG.items():
                value = current_param_vals[key]
                if config["type"] is bool:
                    sidecar_data[config["sidecar_key"]] = bool(value)
                else:
                    # Round to configured decimals for clean sidecar output
                    sidecar_data[config["sidecar_key"]] = round(
                        float(value), config["decimals"]
                    )

            base_name_without_ext = os.path.splitext(file_data["full_path"])[0]
            json_filename = (
                base_name_without_ext + ".fssidecar"
            )  # Target sidecar extension

            if not self.sidecar_manager.save_sidecar_data(json_filename, sidecar_data):
                self.logger.error(
                    f"Failed to save sidecar for {file_data['basename']}."
                )

            # Update last values for carry-forward to the next file
            last_param_vals = current_param_vals.copy()

        # 5. Final Status
        if applied_count == 0:
            self.master_gui.status_label.config(
                text="Finished: No parameters were applied from the export file."
            )
        else:
            self.master_gui.status_label.config(
                text=f"Finished: Applied markers to {applied_count} files, generated {len(target_videos)} FSSIDECARs."
            )
        messagebox.showinfo(
            "Sidecar Generation Complete",
            f"Successfully processed {os.path.basename(export_file_path)} and generated {len(target_videos)} FSSIDECAR files.",
        )

    def generate_custom_sidecars(self):
        """Generates sidecars with a custom name without requiring existing video files."""

        # 1. Select Fusion Export File
        export_file_path = filedialog.askopenfilename(
            defaultextension=".fsexport",
            filetypes=[
                ("Fusion Export Files", "*.fsexport.txt;*.fsexport"),
                ("All Files", "*.*"),
            ],
            title="Select Fusion Export (.fsexport) File",
        )
        if not export_file_path:
            self.master_gui.status_label.config(
                text="Fusion export selection cancelled."
            )
            return

        markers = self._load_and_validate_fsexport(export_file_path)
        if markers is None:
            self.master_gui.status_label.config(text="Fusion export loading failed.")
            return

        # 2. Select Output Sidecar Path and Name
        custom_save_path = filedialog.asksaveasfilename(
            defaultextension=".fssidecar",
            filetypes=[("Sidecar Files", "*.fssidecar")],
            title="Save Sidecar As",
            initialfile=os.path.splitext(os.path.basename(export_file_path))[0],
        )
        if not custom_save_path:
            self.master_gui.status_label.config(text="Custom sidecar export cancelled.")
            return

        # 3. Process markers
        applied_count = 0
        last_param_vals = {
            key: config["default"]
            for key, config in self.FUSION_PARAMETER_CONFIG.items()
        }

        for i, marker in enumerate(markers):
            current_param_vals = last_param_vals.copy()
            if marker.get("values"):
                marker_values = marker["values"]
                updated_from_marker = False
                for key, config in self.FUSION_PARAMETER_CONFIG.items():
                    fusion_key = config["fusion_key"]
                    if fusion_key in marker_values:
                        try:
                            current_param_vals[key] = config["type"](
                                marker_values[fusion_key]
                            )
                            updated_from_marker = True
                        except (ValueError, TypeError):
                            pass
                if updated_from_marker:
                    applied_count += 1

            # Prepare sidecar data
            sidecar_data = {}
            for key, config in self.FUSION_PARAMETER_CONFIG.items():
                value = current_param_vals[key]
                if config["type"] is bool:
                    sidecar_data[config["sidecar_key"]] = bool(value)
                else:
                    sidecar_data[config["sidecar_key"]] = round(
                        float(value), config["decimals"]
                    )

            # Determine filename
            if len(markers) == 1:
                target_filename = custom_save_path
            else:
                # Append index if multiple markers (5-digit zero-padded)
                base, ext = os.path.splitext(custom_save_path)
                target_filename = f"{base}_{i + 1:04d}{ext}"

            if not self.sidecar_manager.save_sidecar_data(
                target_filename, sidecar_data
            ):
                self.logger.error(f"Failed to save custom sidecar: {target_filename}")

            last_param_vals = current_param_vals.copy()

        # 4. Final Status
        self.master_gui.status_label.config(
            text=f"Finished: Generated {len(markers)} custom FSSIDECARs."
        )
        messagebox.showinfo(
            "Custom Export Complete",
            f"Successfully generated {len(markers)} custom FSSIDECAR files.",
        )


class ForwardWarpStereo(nn.Module):
    """
    PyTorch module for forward warping an image based on a disparity map.
    """

    def __init__(self, eps=1e-6, occlu_map=False):
        super(ForwardWarpStereo, self).__init__()
        self.eps = eps
        self.occlu_map = occlu_map
        self.fw = forward_warp()

    def forward(self, im, disp):
        im = im.contiguous()
        disp = disp.contiguous()
        weights_map = disp - disp.min()
        weights_map = (1.414) ** weights_map
        flow = -disp.squeeze(1)
        dummy_flow = torch.zeros_like(flow, requires_grad=False)
        flow = torch.stack((flow, dummy_flow), dim=-1)
        res_accum = self.fw(im * weights_map, flow)
        mask = self.fw(weights_map, flow)
        mask.clamp_(min=self.eps)
        res = res_accum / mask
        if not self.occlu_map:
            return res
        else:
            ones = torch.ones_like(disp, requires_grad=False)
            occlu_map = self.fw(ones, flow)
            occlu_map.clamp_(0.0, 1.0)
            occlu_map = 1.0 - occlu_map
            return res, occlu_map


class SplatterGUI(ThemedTk):
    # --- UI MINIMUM WIDTHS (tweak these numbers) ---
    # These are used as grid column *minimums* for the left (settings) and middle (sliders) columns.
    # Tkinter's grid doesn't support true max-width caps, so the previous max-width clamp code was removed.
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
    # ---------------------------------------
    # Maps Sidecar JSON Key to the internal variable key (used in APP_CONFIG_DEFAULTS)
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
    # ---------------------------------------

    def __init__(self):
        super().__init__(theme="default")
        self.title(f"Stereocrafter Splatting (Batch) {GUI_VERSION}")

        self.app_config = {}
        self.help_texts = {}
        self.sidecar_manager = SidecarConfigManager()

        # --- NEW CACHE AND STATE ---
        self._auto_conv_cache = {"Average": None, "Peak": None}
        self._auto_conv_cached_path = None
        self._is_auto_conv_running = False
        self._preview_debounce_timer = None
        self.slider_label_updaters = []
        self.set_convergence_value_programmatically = None
        self._clip_norm_cache: Dict[str, Tuple[float, float]] = {}
        self._gn_warning_shown: bool = False

        self._load_config()
        self._load_help_texts()

        self._is_startup = True  # NEW: for theme/geometry handling
        self.debug_mode_var = tk.BooleanVar(
            value=self.app_config.get("debug_mode_enabled", False)
        )
        self._debug_logging_enabled = False  # start in INFO mode
        # NEW: Window size and position variables
        self.window_x = self.app_config.get("window_x", None)
        self.window_y = self.app_config.get("window_y", None)
        self.window_width = self.app_config.get("window_width", 620)
        self.window_height = self.app_config.get("window_height", 750)

        # --- Variables with defaults ---
        defaults = self.APP_CONFIG_DEFAULTS  # Convenience variable

        self.dark_mode_var = tk.BooleanVar(
            value=self.app_config.get("dark_mode_enabled", False)
        )
        self.input_source_clips_var = tk.StringVar(
            value=self.app_config.get("input_source_clips", "./input_source_clips")
        )
        self.input_depth_maps_var = tk.StringVar(
            value=self.app_config.get("input_depth_maps", "./input_depth_maps")
        )
        self.multi_map_var = tk.BooleanVar(
            value=bool(self.app_config.get("multi_map_enabled", False))
        )
        self.selected_depth_map_var = tk.StringVar(value="")
        self.depth_map_subfolders = []  # List of valid subfolders
        self.depth_map_radio_buttons = []  # keep list for UI management
        self.depth_map_radio_dict = {}  # NEW: map text->widget
        self._current_video_sidecar_map = None  # Track sidecar's selected map
        self._suppress_sidecar_map_update = (
            False  # Prevent overwriting manual selections
        )
        self._last_loaded_source_video = (
            None  # Track source video for NEW video detection
        )
        self.input_depth_maps_var.trace_add(
            "write", lambda *args: self._on_depth_map_folder_changed()
        )
        self.output_splatted_var = tk.StringVar(
            value=self.app_config.get("output_splatted", "./output_splatted")
        )

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
        # Separate CRF values for Full vs Low output (fallback to legacy "output_crf")
        _legacy_crf = self.app_config.get("output_crf", defaults["CRF_OUTPUT"])
        self.output_crf_full_var = tk.StringVar(
            value=self.app_config.get("output_crf_full", _legacy_crf)
        )
        self.output_crf_low_var = tk.StringVar(
            value=self.app_config.get("output_crf_low", _legacy_crf)
        )
        # Output color metadata tags (metadata-only; does not affect splat math)
        self.color_tags_mode_var = tk.StringVar(
            value=self.app_config.get("color_tags_mode", "Auto")
        )

        # Dev Tools (not saved to config)
        self.skip_lowres_preproc_var = tk.BooleanVar(
            value=bool(self.app_config.get("skip_lowres_preproc", False))
        )

        self.move_to_finished_var = tk.BooleanVar(
            value=self.app_config.get("move_to_finished", True)
        )

        # Crosshair overlay for convergence checking (preview-only; not exported)
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

        # --- Depth Pre-processing Variables ---
        self.depth_gamma_var = tk.StringVar(
            value=self.app_config.get("depth_gamma", defaults["DEPTH_GAMMA"])
        )
        _ddx = self.app_config.get(
            "depth_dilate_size_x", defaults["DEPTH_DILATE_SIZE_X"]
        )
        _ddy = self.app_config.get(
            "depth_dilate_size_y", defaults["DEPTH_DILATE_SIZE_Y"]
        )
        # Backward compatibility: older configs used 30..40 to represent erosion (-0..-10)
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
            value=self.app_config.get(
                "depth_blur_size_x", defaults["DEPTH_BLUR_SIZE_X"]
            )
        )
        self.depth_blur_size_y_var = tk.StringVar(
            value=self.app_config.get(
                "depth_blur_size_y", defaults["DEPTH_BLUR_SIZE_Y"]
            )
        )
        self.depth_dilate_left_var = tk.StringVar(
            value=self.app_config.get(
                "depth_dilate_left", defaults["DEPTH_DILATE_LEFT"]
            )
        )
        self.depth_blur_left_var = tk.StringVar(
            value=self.app_config.get("depth_blur_left", defaults["DEPTH_BLUR_LEFT"])
        )
        self.depth_blur_left_mix_var = tk.StringVar(
            value=self.app_config.get(
                "depth_blur_left_mix", defaults["DEPTH_BLUR_LEFT_MIX"]
            )
        )
        # --- NEW: Sidecar Control Toggle Variables ---
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
        # Add traces for automatic border calculation (Auto Basic mode)
        self.zero_disparity_anchor_var.trace_add(
            "write", self._on_convergence_or_disparity_changed
        )
        self.max_disp_var.trace_add("write", self._on_convergence_or_disparity_changed)
        self.border_mode_var.trace_add("write", self._on_border_mode_change)

        # --- NEW: Previewer Variables ---
        self.preview_source_var = tk.StringVar(
            value=self.app_config.get("preview_source", "Splat Result")
        )
        self.preview_size_var = tk.StringVar(
            value=self.app_config.get("preview_size", "75%")
        )

        # --- Variables for "Current Processing Information" display ---
        self.processing_filename_var = tk.StringVar(value="N/A")
        self.processing_resolution_var = tk.StringVar(value="N/A")
        self.processing_frames_var = tk.StringVar(value="N/A")
        self.processing_disparity_var = tk.StringVar(value="N/A")
        self.processing_convergence_var = tk.StringVar(value="N/A")
        self.processing_task_name_var = tk.StringVar(value="N/A")
        self.processing_gamma_var = tk.StringVar(value="N/A")
        self.processing_map_var = tk.StringVar(value="N/A")

        self.slider_label_updaters = []

        self.widgets_to_disable = []

        # --- Processing control variables ---
        self.stop_event = threading.Event()
        self.progress_queue = queue.Queue()
        self.processing_thread = None

        self._create_widgets()
        self._setup_keyboard_shortcuts()
        self.style = ttk.Style()

        self.update_idletasks()  # Ensure widgets are rendered for correct reqheight
        self._apply_theme(is_startup=True)  # Pass is_startup=True here
        self._set_saved_geometry()  # NEW: Call to set initial geometry
        self._is_startup = (
            False  # Set to false after initial startup geometry is handled
        )
        self._configure_logging()  # Ensure this call is still present

        self.after(10, self.toggle_processing_settings_fields)  # Set initial state
        self.after(10, self._toggle_sidecar_update_button_state)

        # If Multi-Map is enabled at startup (from config), we must populate the map selector UI once.
        # (Initial BooleanVar state does not trigger the checkbox command.)
        if self.multi_map_var.get():
            self.after(15, self._on_multi_map_toggle)

        # Apply preview-only overlay toggles (Crosshair / D/P) at startup.
        # (Initial BooleanVar state does not trigger the checkbox command.)
        self.after(20, self._apply_preview_overlay_toggles)

        self.after(100, self.check_queue)  # Start checking progress queue

        # Bind closing protocol
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

        # --- NEW: Add slider release binding for preview updates ---
        # We will add this to the sliders in _create_widgets
        self.slider_widgets = []

    def _adjust_window_height_for_content(self):
        """Adjusts the window height to fit the current content, preserving user-set width."""
        if self._is_startup:  # Don't adjust during initial setup
            return

        current_actual_width = self.winfo_width()
        if current_actual_width <= 1:  # Fallback for very first call
            current_actual_width = self.window_width

        # --- NEW: More accurate height calculation ---
        # --- FIX: Calculate base_height by summing widgets *other* than the previewer ---
        # This is more stable than subtracting a potentially out-of-sync canvas height.
        base_height = 0
        for widget in self.winfo_children():
            if widget is not self.previewer:
                # --- FIX: Correctly handle tuple and int for pady ---
                try:
                    pady_value = widget.pack_info().get("pady", 0)
                    total_pady = 0
                    if isinstance(pady_value, int):
                        total_pady = pady_value * 2
                    elif isinstance(pady_value, (tuple, list)):
                        total_pady = sum(pady_value)
                    base_height += widget.winfo_reqheight() + total_pady
                except tk.TclError:
                    # This widget (e.g., the menubar) is not packed, so it has no pady.
                    base_height += widget.winfo_reqheight()
        # --- END FIX ---

        # Get the actual height of the displayed preview image, if it exists
        preview_image_height = 0
        if (
            hasattr(self.previewer, "preview_image_tk")
            and self.previewer.preview_image_tk
        ):
            preview_image_height = self.previewer.preview_image_tk.height()

        # Add a small buffer for padding/borders
        padding = 10

        # The new total height is the base UI height + the actual image height + padding
        new_height = base_height + preview_image_height + padding
        # --- END NEW ---

        self.geometry(f"{current_actual_width}x{new_height}")
        logger.debug(
            f"Content resize applied geometry: {current_actual_width}x{new_height}"
        )
        self.window_width = current_actual_width  # Update stored width

    def _apply_theme(self, is_startup: bool = False):
        """Applies the selected theme (dark or light) to the GUI."""
        # 1. Define color palettes
        dark_colors = {
            "bg": "#2b2b2b",
            "fg": "white",
            "entry_bg": "#3c3c3c",
            "menu_bg": "#3c3c3c",
            "menu_fg": "white",
            "active_bg": "#555555",
            "active_fg": "white",
            "theme": "black",
        }
        light_colors = {
            "bg": "#d9d9d9",
            "fg": "black",
            "entry_bg": "#ffffff",
            "menu_bg": "#f0f0f0",
            "menu_fg": "black",
            "active_bg": "#dddddd",
            "active_fg": "black",
            "theme": "default",
        }

        # 2. Select the current palette and theme
        if self.dark_mode_var.get():
            colors = dark_colors
        else:
            colors = light_colors

        self.style.theme_use(colors["theme"])
        self.configure(bg=colors["bg"])

        # 3. Apply styles to ttk widgets
        self.style.configure("TFrame", background=colors["bg"], foreground=colors["fg"])
        self.style.configure(
            "TLabelframe", background=colors["bg"], foreground=colors["fg"]
        )
        self.style.configure(
            "TLabelframe.Label", background=colors["bg"], foreground=colors["fg"]
        )
        self.style.configure("TLabel", background=colors["bg"], foreground=colors["fg"])
        self.style.configure(
            "TCheckbutton", background=colors["bg"], foreground=colors["fg"]
        )
        self.style.map(
            "TCheckbutton",
            foreground=[("active", colors["fg"])],
            background=[("active", colors["bg"])],
        )

        # 4. Configure Entry and Combobox widgets using style.map for robust background override
        self.style.map(
            "TEntry",
            fieldbackground=[("", colors["entry_bg"])],
            foreground=[("", colors["fg"])],
        )
        self.style.configure("TEntry", insertcolor=colors["fg"])
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", colors["entry_bg"])],
            foreground=[("readonly", colors["fg"])],
            selectbackground=[("readonly", colors["entry_bg"])],
            selectforeground=[("readonly", colors["fg"])],
        )

        # Manually set the background for the previewer's canvas widget
        if hasattr(self, "previewer") and hasattr(self.previewer, "preview_canvas"):
            self.previewer.preview_canvas.config(bg=colors["bg"], highlightthickness=0)

        # 5. Manually configure non-ttk widgets (Menu, tk.Label)
        if hasattr(self, "menubar"):
            for menu in [self.menubar, self.file_menu, self.help_menu]:
                menu.config(
                    bg=colors["menu_bg"],
                    fg=colors["menu_fg"],
                    activebackground=colors["active_bg"],
                    activeforeground=colors["active_fg"],
                )
        if hasattr(self, "info_frame"):
            for label in self.info_labels:
                label.config(bg=colors["bg"], fg=colors["fg"])

        # 6. Handle window geometry adjustment (only after startup)
        self.update_idletasks()  # Ensure all theme changes are rendered for accurate reqheight

        # --- Apply geometry only if not during startup (NEW conditional block) ---
        # if not is_startup:
        #     self._adjust_window_height_for_content()

    def _auto_converge_worker(
        self, depth_map_path, process_length, batch_size, fallback_value, mode
    ):
        """Worker thread for running the Auto-Convergence calculation."""

        # Run the existing auto-convergence logic (no mode parameter needed now)
        new_anchor_avg, new_anchor_peak = self._determine_auto_convergence(
            depth_map_path,
            process_length,
            batch_size,
            fallback_value,
        )

        # Use self.after to safely update the GUI from the worker thread
        self.after(
            0,
            lambda: self._complete_auto_converge_update(
                new_anchor_avg,
                new_anchor_peak,
                fallback_value,
                mode,  # Still pass the current mode to know which value to select immediately
            ),
        )

    def _auto_save_current_sidecar(self):
        """
        Saves the current GUI values to the sidecar file without user interaction.
        Only runs if self.auto_save_sidecar_var is True.
        """
        if not self.auto_save_sidecar_var.get():
            return

        self._save_current_sidecar_data(is_auto_save=True)

    def _browse_folder(self, var):
        """Opens a folder dialog and updates a StringVar."""
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
        """Opens a file dialog and updates a StringVar."""
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

    def _compute_clip_global_depth_stats(
        self, depth_map_path: str, chunk_size: int = 100
    ) -> Tuple[float, float]:
        """
        [NEW HELPER] Computes the global min and max depth values from a depth video
        by reading it in chunks. Used only for the preview's GN cache.
        """
        logger.info(
            f"==> Starting clip-local depth stats pre-pass for {os.path.basename(depth_map_path)}..."
        )
        global_min, global_max = np.inf, -np.inf

        try:
            temp_reader = VideoReader(depth_map_path, ctx=cpu(0))
            total_frames = len(temp_reader)

            if total_frames == 0:
                logger.error("Depth reader found 0 frames for global stats.")
                return 0.0, 1.0  # Fallback

            for i in range(0, total_frames, chunk_size):
                if self.stop_event.is_set():
                    logger.warning("Global stats scan stopped by user.")
                    return 0.0, 1.0

                current_indices = list(range(i, min(i + chunk_size, total_frames)))
                chunk_numpy_raw = temp_reader.get_batch(current_indices).asnumpy()

                # Handle RGB vs Grayscale depth maps
                if chunk_numpy_raw.ndim == 4:
                    if chunk_numpy_raw.shape[-1] == 3:  # RGB
                        chunk_numpy = chunk_numpy_raw.mean(axis=-1)
                    else:  # Grayscale with channel dim
                        chunk_numpy = chunk_numpy_raw.squeeze(-1)
                else:
                    chunk_numpy = chunk_numpy_raw

                chunk_min = chunk_numpy.min()
                chunk_max = chunk_numpy.max()

                if chunk_min < global_min:
                    global_min = chunk_min
                if chunk_max > global_max:
                    global_max = chunk_max

                # Skip progress bar for speed, use console log if needed

            logger.info(
                f"==> Clip-local depth stats computed: min_raw={global_min:.3f}, max_raw={global_max:.3f}"
            )

            # Cache the result before returning
            self._clip_norm_cache[depth_map_path] = (
                float(global_min),
                float(global_max),
            )

            return float(global_min), float(global_max)

        except Exception as e:
            logger.error(f"Error during clip-local depth stats scan for preview: {e}")
            return 0.0, 1.0  # Fallback
        finally:
            gc.collect()

    def _on_multi_map_toggle(self):
        """Called when Multi-Map checkbox is toggled."""
        if self.multi_map_var.get():
            # Multi-Map enabled - scan for subfolders
            self._scan_depth_map_folders()
        else:
            # Multi-Map disabled - clear radio buttons
            self._clear_depth_map_radio_buttons()
            self.selected_depth_map_var.set("")

    def _on_depth_map_folder_changed(self):
        """Called when the Input Depth Maps folder path changes."""
        if self.multi_map_var.get():
            # Re-scan if Multi-Map is enabled
            self._scan_depth_map_folders()

    def _on_convergence_or_disparity_changed(self, *args):
        """Web of traces: Update border width when convergence or disparity changes, if in Auto Basic mode."""
        if self.border_mode_var.get() != "Auto Basic":
            return

        try:
            # width = (1.0 - convergence) * 2.0 * (max_disp / 20.0)
            c_val = self.zero_disparity_anchor_var.get()
            d_val = self.max_disp_var.get()
            if not c_val or not d_val:
                return
            c = float(c_val)
            d = float(d_val)

            width = max(0.0, (1.0 - c) * 2.0 * (d / 20.0))
            width = min(5.0, width)

            self.border_width_var.set(f"{width:.2f}")
            self.border_bias_var.set("0.0")

            if (
                hasattr(self, "set_border_width_programmatically")
                and self.set_border_width_programmatically
            ):
                self.set_border_width_programmatically(width)
            if (
                hasattr(self, "set_border_bias_programmatically")
                and self.set_border_bias_programmatically
            ):
                self.set_border_bias_programmatically(0.0)
        except (ValueError, TypeError):
            pass

    def _sync_sliders_to_auto_borders(self, l_val=None, r_val=None):
        """Updates Manual Width/Bias sliders to match the given or current Auto Border values."""
        if l_val is None:
            l_val = self._safe_float(self.auto_border_L_var)
        if r_val is None:
            r_val = self._safe_float(self.auto_border_R_var)

        w = max(l_val, r_val)
        if w > 0:
            if l_val >= r_val:
                b = (r_val / l_val) - 1.0
            else:
                b = 1.0 - (l_val / r_val)
        else:
            b = 0.0

        self.border_width_var.set(f"{w:.2f}")
        self.border_bias_var.set(f"{b:.2f}")
        if hasattr(self, "set_border_width_programmatically"):
            self.set_border_width_programmatically(w)
        if hasattr(self, "set_border_bias_programmatically"):
            self.set_border_bias_programmatically(b)

    def _on_border_mode_change(self, *args):
        """Called when the 'Border' mode pulldown is changed."""
        mode = self.border_mode_var.get()
        state = "normal" if mode == "Manual" else "disabled"

        # Enable/Disable sliders
        if hasattr(self, "border_sliders_row_frame"):
            for subframe in self.border_sliders_row_frame.winfo_children():
                for child in subframe.winfo_children():
                    if isinstance(child, (tk.Scale, ttk.Scale, ttk.Entry, ttk.Label)):
                        try:
                            # Labels don't have state, but some themes allow it or we skip
                            if hasattr(child, "configure"):
                                child.configure(state=state)
                        except Exception:
                            pass

        if mode == "Auto Basic":
            self._on_convergence_or_disparity_changed()
        elif mode == "Auto Adv.":
            # If we have a clip, check if we already have scan data
            result = self._get_current_sidecar_paths_and_data()
            if result:
                json_sidecar_path, _, data = result
                l_val = data.get("auto_border_L")
                r_val = data.get("auto_border_R")

                if l_val is not None and r_val is not None:
                    # We have data (even if 0.0), just load it
                    self.auto_border_L_var.set(str(l_val))
                    self.auto_border_R_var.set(str(r_val))
                    self._sync_sliders_to_auto_borders(l_val, r_val)
                else:
                    # No data, perform scan
                    scan_result = self._scan_borders_for_current_clip()
                    if scan_result:
                        l_val, r_val = scan_result
                        self._sync_sliders_to_auto_borders(l_val, r_val)
                        self._save_current_sidecar_data(is_auto_save=True)
            else:
                scan_result = self._scan_borders_for_current_clip()
                if scan_result:
                    l_val, r_val = scan_result
                    self._sync_sliders_to_auto_borders(l_val, r_val)
                    self._save_current_sidecar_data(is_auto_save=True)
        elif mode == "Off":
            self.border_width_var.set("0.0")
            self.border_bias_var.set("0.0")
            if hasattr(self, "set_border_width_programmatically"):
                self.set_border_width_programmatically(0.0)
            if hasattr(self, "set_border_bias_programmatically"):
                self.set_border_bias_programmatically(0.0)

        # Trigger preview update
        if getattr(self, "previewer", None):
            self.previewer.update_preview()

    def _on_border_rescan_click(self):
        """Handler for the Rescan button."""
        mode = self.border_mode_var.get()

        # 1. Perform scan
        scan_result = self._scan_borders_for_current_clip(force=True)
        newL, newR = (0.0, 0.0)
        if scan_result:
            newL, newR = scan_result

        # 2. Implement state transition logic
        if mode == "Off":
            # Switch to Manual and set sliders to match scan
            self._sync_sliders_to_auto_borders(newL, newR)
            self.border_mode_var.set("Manual")

        elif mode == "Auto Basic":
            # Switch to Auto Adv.
            self.border_mode_var.set("Auto Adv.")
            self._sync_sliders_to_auto_borders(newL, newR)
        else:
            # We are already in Auto Adv. (or Manual), still update sliders to reflect fresh scan
            self._sync_sliders_to_auto_borders(newL, newR)

        # Force a sidecar save with new auto values IMMEDIATELY
        # Pass values explicitly to ensure they are saved even if mode isn't Auto Adv yet
        self._save_current_sidecar_data(
            is_auto_save=True, force_auto_L=newL, force_auto_R=newR
        )

        if getattr(self, "previewer", None):
            self.previewer.update_preview()

    def _scan_borders_for_current_clip(self, force=False):
        """Advanced border scanning: Samples edges of depth map."""
        result = self._get_current_sidecar_paths_and_data()
        if not result:
            return

        json_sidecar_path, depth_path, _ = result
        if not depth_path or not os.path.exists(depth_path):
            return

        try:
            vr = VideoReader(depth_path, ctx=cpu(0))
            total_frames = len(vr)
            if total_frames == 0:
                return

            # Show scanning status
            old_status = self.status_label.cget("text")
            self.status_label.config(
                text=f"Scanning borders for {os.path.basename(depth_path)}..."
            )
            self.update_idletasks()

            step = 5
            max_L = 0.0
            max_R = 0.0

            conv = self._safe_float(self.zero_disparity_anchor_var, 0.5)
            max_disp = self._safe_float(self.max_disp_var, 20.0)

            for i in range(0, total_frames, step):
                if self.stop_event.is_set():
                    break

                frame_raw = vr[i].asnumpy()
                if frame_raw.ndim == 3:
                    # Simple average for grayscale
                    frame = frame_raw.mean(axis=2)
                else:
                    frame = frame_raw

                # Sample 5px wide at each edge
                L_sample = frame[:, :5]
                R_sample = frame[:, -5:]

                # 99th percentile to ignore noise
                d_L = np.percentile(L_sample, 99) / 255.0
                d_R = np.percentile(R_sample, 99) / 255.0

                # Scaling matches Auto Basic but localized to depth
                b_L = max(0.0, (d_L - conv) * 2.0 * (max_disp / 20.0))
                b_R = max(0.0, (d_R - conv) * 2.0 * (max_disp / 20.0))

                max_L = max(max_L, b_L)
                max_R = max(max_R, b_R)

            max_L = min(5.0, round(float(max_L), 3))
            max_R = min(5.0, round(float(max_R), 3))

            self.auto_border_L_var.set(str(max_L))
            self.auto_border_R_var.set(str(max_R))

            logger.info(
                f"Border scan complete: L={max_L}, R={max_R} (Conv={conv:.2f}, Disp={max_disp:.1f})"
            )

            self.status_label.config(text=f"Scan complete: L={max_L}%, R={max_R}%")
            # Wait a bit so user can see the result before status might be clobbered
            self.after(2000, lambda: self.status_label.config(text=old_status))

            return max_L, max_R

        except Exception as e:
            logger.error(f"Border scan failed: {e}")
            self.status_label.config(text="Border scan failed.")
            return None

    def _scan_depth_map_folders(self):
        """Scans the Input Depth Maps folder for subfolders containing *_depth.mp4 files."""
        base_folder = self.input_depth_maps_var.get()

        # Clear existing radio buttons
        self._clear_depth_map_radio_buttons()
        self.depth_map_subfolders = []

        if not os.path.isdir(base_folder):
            return

        # Find all subfolders that contain depth map files
        try:
            for item in sorted(os.listdir(base_folder)):
                subfolder_path = os.path.join(base_folder, item)
                if os.path.isdir(subfolder_path):
                    # Check if this subfolder contains *_depth.mp4 files
                    depth_files = glob.glob(os.path.join(subfolder_path, "*_depth.mp4"))
                    if depth_files:
                        self.depth_map_subfolders.append(item)
        except Exception as e:
            logger.error(f"Error scanning depth map subfolders: {e}")
            return

        if self.depth_map_subfolders:
            # Select the first one by default (alphabetically first)
            self.selected_depth_map_var.set(self.depth_map_subfolders[0])
            # Create radio buttons in the previewer
            self._create_depth_map_radio_buttons()
            # Trigger preview update
            self.on_slider_release(None)
        else:
            logger.warning("No valid depth map subfolders found")
            self.selected_depth_map_var.set("")

    def _update_map_selector_highlight(self):
        """Visually emphasize the currently selected map radio button (no layout change)."""
        try:
            sel = (self.selected_depth_map_var.get() or "").strip()
            # Only applies when radio buttons exist
            rb_dict = getattr(self, "depth_map_radio_dict", {}) or {}
            for name, rb in rb_dict.items():
                if not rb:
                    continue
                try:
                    rb.configure(
                        style="MapSelSelected.TRadiobutton"
                        if name == sel
                        else "MapSel.TRadiobutton"
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def _clear_depth_map_radio_buttons(self):
        """Removes all depth map radio buttons from the GUI."""
        for widget in self.depth_map_radio_buttons:
            widget.destroy()
        self.depth_map_radio_buttons = []

    def _create_depth_map_radio_buttons(self):
        """Creates radio buttons for each valid depth map subfolder."""
        logger.info(
            f"Creating radio buttons, current selected_depth_map_var = {self.selected_depth_map_var.get()}"
        )
        self._clear_depth_map_radio_buttons()

        if not hasattr(self, "previewer") or self.previewer is None:
            return

        # Get the preview button frame from the previewer
        # The radio buttons should be added to the same frame as preview_size_combo
        preview_button_frame = self.previewer.preview_size_combo.master

        # --- Map selector styles (selected vs unselected) ---
        if not getattr(self, "_map_selector_styles_inited", False):
            try:
                style = ttk.Style()
                base_font = tkfont.nametofont("TkDefaultFont")
                bold_font = tkfont.Font(
                    root=preview_button_frame,
                    family=base_font.cget("family"),
                    size=base_font.cget("size"),
                    weight="bold",
                )
                style.configure("MapSel.TRadiobutton", font=base_font)
                style.configure("MapSelSelected.TRadiobutton", font=bold_font)
            except Exception:
                # Fail quietly; styles are optional
                pass
            self._map_selector_styles_inited = True

        for subfolder_name in self.depth_map_subfolders:
            rb = ttk.Radiobutton(
                preview_button_frame,
                text=subfolder_name,
                variable=self.selected_depth_map_var,
                value=subfolder_name,
                style="MapSel.TRadiobutton",
                command=self._on_map_selection_changed,
            )
            rb.pack(side="left", padx=5)
            self.depth_map_radio_buttons.append(rb)
            self.depth_map_radio_dict[subfolder_name] = rb

        # Apply visual emphasis to the current selection
        self._update_map_selector_highlight()

    def _on_map_selection_changed(self, from_sidecar=False):
        """
        Called when the user changes the depth map selection (radio buttons),
        or when a sidecar restores a map (from_sidecar=True).

        In Multi-Map mode this now ONLY updates the CURRENT video’s depth map
        path instead of iterating over every video.
        """
        _sel = self.selected_depth_map_var.get()
        _last = getattr(self, "_last_map_change_log", None)
        if _sel != _last:
            logger.info(
                f"Depth map selection changed -> {_sel} (from_sidecar={from_sidecar})"
            )
            self._last_map_change_log = _sel
        if not from_sidecar:
            # User clicked a radio button – suppress sidecar overwrites
            self._suppress_sidecar_map_update = True

        # Keep UI + info panel in sync with the currently selected map (no heavy work)
        try:
            self._update_map_selector_highlight()
        except Exception:
            pass
        try:
            self._update_processing_info_for_preview_clip()
        except Exception:
            pass

        # Compute the folder for the newly selected map
        new_depth_folder = self._get_effective_depth_map_folder()

        # If there is no previewer / no videos, nothing to do
        if not hasattr(self, "previewer") or self.previewer is None:
            return

        current_index = getattr(self.previewer, "current_video_index", None)
        if current_index is None:
            return
        if current_index < 0 or current_index >= len(self.previewer.video_list):
            return

        # Work only on the CURRENT video entry
        video_entry = self.previewer.video_list[current_index]
        source_video = video_entry.get("source_video", "")
        if not source_video:
            return

        video_name = os.path.splitext(os.path.basename(source_video))[0]
        depth_mp4 = os.path.join(new_depth_folder, f"{video_name}_depth.mp4")
        depth_npz = os.path.join(new_depth_folder, f"{video_name}_depth.npz")

        depth_path = None
        if os.path.exists(depth_mp4):
            depth_path = depth_mp4
        elif os.path.exists(depth_npz):
            depth_path = depth_npz

        # Update the current entry only
        video_entry["depth_map"] = depth_path

        # Only log for the current video, and only if it’s missing
        if depth_path is None:
            logger.info(
                f"Depth map for current video {video_name} not found in "
                f"{os.path.basename(new_depth_folder)}"
            )

        # Refresh previewer so the current video immediately reflects the new map
        try:
            self.previewer.replace_source_path_for_current_video(
                "depth_map", depth_path or ""
            )
        except Exception as e:
            logger.exception(f"Error refreshing preview after map switch: {e}")

        # Keep the processing queue entry (if present) in sync for this one video
        if hasattr(self, "resolution_output_list") and 0 <= current_index < len(
            self.resolution_output_list
        ):
            self.resolution_output_list[current_index].depth_map = depth_path

        # Post-switch: update info panel + visual emphasis
        try:
            self._update_processing_info_for_preview_clip()
        except Exception:
            pass
        try:
            self._update_map_selector_highlight()
        except Exception:
            pass

    def _get_effective_depth_map_folder(self, base_folder=None):
        """Returns the effective depth map folder based on Multi-Map settings.

        Args:
            base_folder: Optional override for base folder (used during processing)

        Returns:
            str: The folder path to use for depth maps
        """
        if base_folder is None:
            base_folder = self.input_depth_maps_var.get()

        # If the user has selected a single depth MAP FILE, treat its directory as the folder.
        if base_folder and os.path.isfile(base_folder):
            base_folder = os.path.dirname(base_folder)

        if self.multi_map_var.get() and self.selected_depth_map_var.get().strip():
            # Multi-Map is enabled and a subfolder is selected
            return os.path.join(base_folder, self.selected_depth_map_var.get().strip())
        else:
            # Normal mode - use the base folder directly
            return base_folder

    def _get_sidecar_base_folder(self):
        """Returns the folder where sidecars should be stored.

        When Multi-Map is enabled, sidecars are stored in a 'sidecars' subfolder.
        When Multi-Map is disabled, sidecars are stored alongside depth maps.

        Returns:
            str: The folder path for sidecar storage
        """
        if self.multi_map_var.get():
            # Multi-Map mode: store sidecars in 'sidecars' subfolder
            base_folder = self.input_depth_maps_var.get()
            sidecar_folder = os.path.join(base_folder, "sidecars")
            # Create the sidecars folder if it doesn't exist
            os.makedirs(sidecar_folder, exist_ok=True)
            return sidecar_folder
        else:
            # Normal mode: store sidecars with depth maps
            return self._get_effective_depth_map_folder()

    def _get_sidecar_selected_map_for_video(self, video_path):
        """
        Returns the Multi-Map subfolder name for a given video based on its sidecar,
        or None if there is no sidecar / no selected_depth_map entry.
        """
        try:
            # Derive expected sidecar name from *video name* (matches your depth sidecars)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            sidecar_ext = self.APP_CONFIG_DEFAULTS.get("SIDECAR_EXT", ".fssidecar")

            # In Multi-Map mode, sidecars live in <InputDepthMaps>/sidecars
            sidecar_folder = self._get_sidecar_base_folder()
            sidecar_path = os.path.join(
                sidecar_folder, f"{video_name}_depth{sidecar_ext}"
            )

            if not os.path.exists(sidecar_path):
                return None

            sidecar_config = self.sidecar_manager.load_sidecar_data(sidecar_path) or {}
            selected_map_val = sidecar_config.get("selected_depth_map", "")
            if selected_map_val:
                return selected_map_val

        except Exception as e:
            logger.error(f"Error reading sidecar map for {video_path}: {e}")

        return None

    def check_queue(self):
        """Periodically checks the progress queue for updates to the GUI."""
        try:
            while True:
                message = self.progress_queue.get_nowait()
                if message == "finished":
                    self.status_label.config(text="Processing finished")
                    self.start_button.config(state="normal")
                    self.start_single_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.progress_var.set(0)
                    # --- NEW: Enable all inputs at finish ---
                    self._set_input_state("normal")
                    logger.info(f"==> All process completed.")
                    break

                elif message[0] == "total":
                    total_tasks = message[1]
                    self.progress_bar.config(maximum=total_tasks)
                    self.progress_var.set(0)
                    self.status_label.config(
                        text=f"Processing 0 of {total_tasks} tasks"
                    )
                elif message[0] == "processed":
                    processed_tasks = message[1]
                    total_tasks = self.progress_bar["maximum"]
                    self.progress_var.set(processed_tasks)
                    self.status_label.config(
                        text=f"Processed tasks: {processed_tasks}/{total_tasks} (overall)"
                    )
                elif message[0] == "status":
                    self.status_label.config(
                        text=f"Overall: {self.progress_var.get()}/{self.progress_bar['maximum']} - {message[1].split(':', 1)[-1].strip()}"
                    )
                elif message[0] == "update_info":
                    info_data = message[1]
                    if "filename" in info_data:
                        self.processing_filename_var.set(info_data["filename"])
                    if "resolution" in info_data:
                        self.processing_resolution_var.set(info_data["resolution"])
                    if "frames" in info_data:
                        self.processing_frames_var.set(str(info_data["frames"]))
                    if "disparity" in info_data:
                        self.processing_disparity_var.set(info_data["disparity"])
                    if "convergence" in info_data:
                        self.processing_convergence_var.set(info_data["convergence"])
                    if "gamma" in info_data:
                        self.processing_gamma_var.set(info_data["gamma"])
                    if "map" in info_data:
                        self.processing_map_var.set(info_data["map"])
                    if "task_name" in info_data:
                        self.processing_task_name_var.set(info_data["task_name"])

        except queue.Empty:
            pass
        self.after(100, self.check_queue)

    def clear_processing_info(self):
        """Resets all 'Current Processing Information' labels to default 'N/A'."""
        self.processing_filename_var.set("N/A")
        self.processing_resolution_var.set("N/A")
        self.processing_frames_var.set("N/A")
        self.processing_disparity_var.set("N/A")
        self.processing_convergence_var.set("N/A")
        self.processing_gamma_var.set("N/A")
        self.processing_task_name_var.set("N/A")
        self.processing_map_var.set("N/A")

    def _complete_auto_converge_update(
        self,
        new_anchor_avg: float,
        new_anchor_peak: float,
        fallback_value: float,
        mode: str,
    ):
        """
        Safely updates the GUI and preview after Auto-Convergence worker is done.

        Now receives both calculated values.
        """
        # Re-enable inputs
        self._is_auto_conv_running = False
        self.btn_auto_converge_preview.config(state="normal")
        self.start_button.config(state="normal")
        self.start_single_button.config(state="normal")
        self.auto_convergence_combo.config(state="readonly")

        if self.stop_event.is_set():
            self.status_label.config(text="Auto-Converge pre-pass was stopped.")
            self.stop_event.clear()
            return

        # Check if EITHER calculation yielded a result different from the fallback
        if new_anchor_avg != fallback_value or new_anchor_peak != fallback_value:
            # 1. Cache BOTH results
            self._auto_conv_cache["Average"] = new_anchor_avg
            self._auto_conv_cache["Peak"] = new_anchor_peak

            # CRITICAL: Store the path of the file that was just scanned
            current_index = self.previewer.current_video_index
            if 0 <= current_index < len(self.previewer.video_list):
                depth_map_path = self.previewer.video_list[current_index].get(
                    "depth_map"
                )
                self._auto_conv_cached_path = depth_map_path

            # 2. Determine which value to apply immediately (based on the current 'mode' selection)
            anchor_to_apply = new_anchor_avg if mode == "Average" else new_anchor_peak

            # 3. Update the Tkinter variable and refresh the slider/label

            is_setter_successful = False
            if self.set_convergence_value_programmatically:
                try:
                    # Pass the numeric value. The setter handles setting var and updating the label.
                    self.set_convergence_value_programmatically(anchor_to_apply)
                    is_setter_successful = True
                except Exception as e:
                    logger.error(f"Error calling convergence setter: {e}")

            # Fallback if setter failed (should not happen if fixed)
            if not is_setter_successful:
                self.zero_disparity_anchor_var.set(f"{anchor_to_apply:.2f}")

            self.status_label.config(
                text=f"Auto-Converge: Avg Cached at {new_anchor_avg:.2f}, Peak Cached at {new_anchor_peak:.2f}. Applied: {mode} ({anchor_to_apply:.2f})"
            )

            # 4. Immediately trigger a preview update to show the change
            self.on_slider_release(None)

        else:
            # Calculation failed (both returned fallback)
            self.status_label.config(
                text=f"Auto-Converge: Failed to find a valid anchor. Value remains {fallback_value:.2f}"
            )
            messagebox.showwarning(
                "Auto-Converge Preview",
                f"Failed to find a valid anchor point in any mode. No changes were made.",
            )
            # If it was triggered by the combo box, reset the combo box to "Off"
            if self.auto_convergence_mode_var.get() == mode:
                self.auto_convergence_combo.set("Off")

    def _configure_logging(self):
        """Sets the logging level for the stereocrafter_util logger based on debug_mode_var."""
        if self.debug_mode_var.get():
            level = logging.DEBUG
            # Also set the root logger if it hasn't been configured to debug, to catch other messages
            if logging.root.level > logging.DEBUG:
                logging.root.setLevel(logging.DEBUG)
        else:
            level = logging.INFO
            # Reset root logger if it was temporarily set to debug by this GUI
            if logging.root.level == logging.DEBUG:  # Check if this GUI set it
                logging.root.setLevel(logging.INFO)  # Reset to a less verbose default

        # Make sure 'set_util_logger_level' is imported and available.
        # It's already in dependency/stereocrafter_util, ensure it's imported at the top.
        # Add 'import logging' at the top of splatting_gui.py if not already present.
        set_util_logger_level(level)  # Call the function from stereocrafter_util.py
        logger.info(f"Logging level set to {logging.getLevelName(level)}.")

    def _create_hover_tooltip(self, widget, key):
        """Creates a tooltip for a given widget based on a key from help_texts."""
        if key in self.help_texts:
            Tooltip(widget, self.help_texts[key])

    def _create_widgets(self):
        """Initializes and places all GUI widgets."""

        current_row = 0

        # --- Menu Bar ---
        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)

        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)

        # Add new commands to the File menu
        self.file_menu.add_command(
            label="Load Settings from File...", command=self.load_settings
        )
        self.file_menu.add_command(
            label="Save Settings", command=self._save_current_settings_and_notify
        )
        self.file_menu.add_command(
            label="Save Settings to File...", command=self.save_settings
        )
        self.file_menu.add_separator()  # Separator for organization

        self.file_menu.add_command(
            label="Load Fusion Export (.fsexport)...",
            command=self.run_fusion_sidecar_generator,
        )
        self.file_menu.add_command(
            label="FSExport to custom sidecar...",
            command=self.run_fusion_sidecar_generator_custom,
        )
        self.file_menu.add_separator()

        self.file_menu.add_checkbutton(
            label="Dark Mode", variable=self.dark_mode_var, command=self._apply_theme
        )
        self.file_menu.add_separator()

        # Update Slider from Sidecar Toggle (Existing)
        self.file_menu.add_checkbutton(
            label="Update Slider from Sidecar",
            variable=self.update_slider_from_sidecar_var,
        )

        # --- Auto Save Sidecar Toggle ---
        self.file_menu.add_checkbutton(
            label="Auto Save Sidecar on Next", variable=self.auto_save_sidecar_var
        )
        self.file_menu.add_separator()

        self.file_menu.add_command(
            label="Reset to Default", command=self.reset_to_defaults
        )
        self.file_menu.add_command(
            label="Restore Finished", command=self.restore_finished_files
        )

        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.debug_logging_var = tk.BooleanVar(value=self._debug_logging_enabled)
        self.help_menu.add_checkbutton(
            label="Debug Logging",
            variable=self.debug_logging_var,
            command=self._toggle_debug_logging,
        )
        self.help_menu.add_command(label="User Guide", command=self.show_user_guide)
        self.help_menu.add_separator()

        # Add "About" submenu (after "Debug Logging")
        self.help_menu.add_command(
            label="About Stereocrafter Splatting", command=self.show_about
        )
        self.menubar.add_cascade(label="Help", menu=self.help_menu)

        # --- Folder selection frame ---
        self.folder_frame = ttk.LabelFrame(self, text="Input/Output Folders")
        self.folder_frame.pack(pady=2, padx=10, fill="x")
        self.folder_frame.grid_columnconfigure(1, weight=1)

        # Settings Container (NEW)
        self.settings_container_frame = ttk.Frame(
            self
        )  # <-- ADD self. to settings_container_frame
        self.settings_container_frame.pack(pady=2, padx=10, fill="x")

        # Input Source Clips Row
        self.lbl_source_clips = ttk.Label(self.folder_frame, text="Input Source Clips:")
        self.lbl_source_clips.grid(
            row=current_row, column=0, sticky="e", padx=5, pady=0
        )
        self.entry_source_clips = ttk.Entry(
            self.folder_frame, textvariable=self.input_source_clips_var
        )
        self.entry_source_clips.grid(
            row=current_row, column=1, padx=5, pady=0, sticky="ew"
        )
        self.btn_browse_source_clips_folder = ttk.Button(
            self.folder_frame,
            text="Browse Folder",
            command=lambda: self._browse_folder(self.input_source_clips_var),
        )
        self.btn_browse_source_clips_folder.grid(
            row=current_row, column=2, padx=2, pady=0
        )
        self.btn_select_source_clips_file = ttk.Button(
            self.folder_frame,
            text="Select File",
            command=lambda: self._browse_file(
                self.input_source_clips_var,
                [("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
            ),
        )
        self.btn_select_source_clips_file.grid(
            row=current_row, column=3, padx=2, pady=0
        )
        self._create_hover_tooltip(self.lbl_source_clips, "input_source_clips")
        self._create_hover_tooltip(self.entry_source_clips, "input_source_clips")
        self._create_hover_tooltip(
            self.btn_browse_source_clips_folder, "input_source_clips_folder"
        )
        self._create_hover_tooltip(
            self.btn_select_source_clips_file, "input_source_clips_file"
        )
        current_row += 1

        # Input Depth Maps Row
        self.lbl_input_depth_maps = ttk.Label(
            self.folder_frame, text="Input Depth Maps:"
        )
        self.lbl_input_depth_maps.grid(
            row=current_row, column=0, sticky="e", padx=5, pady=0
        )
        self.entry_input_depth_maps = ttk.Entry(
            self.folder_frame, textvariable=self.input_depth_maps_var
        )
        self.entry_input_depth_maps.grid(
            row=current_row, column=1, padx=5, pady=0, sticky="ew"
        )
        self.btn_browse_input_depth_maps_folder = ttk.Button(
            self.folder_frame,
            text="Browse Folder",
            command=lambda: self._browse_folder(self.input_depth_maps_var),
        )
        self.btn_browse_input_depth_maps_folder.grid(
            row=current_row, column=2, padx=2, pady=0
        )
        self.btn_select_input_depth_maps_file = ttk.Button(
            self.folder_frame,
            text="Select File",
            command=lambda: self._browse_file(
                self.input_depth_maps_var,
                [("Depth Files", "*.mp4 *.npz"), ("All files", "*.*")],
            ),
        )
        self.btn_select_input_depth_maps_file.grid(
            row=current_row, column=3, padx=2, pady=0
        )
        self._create_hover_tooltip(self.lbl_input_depth_maps, "input_depth_maps")
        self._create_hover_tooltip(self.entry_input_depth_maps, "input_depth_maps")
        self._create_hover_tooltip(
            self.btn_browse_input_depth_maps_folder, "input_depth_maps_folder"
        )
        self._create_hover_tooltip(
            self.btn_select_input_depth_maps_file, "input_depth_maps_file"
        )
        current_row += 1

        # Output Splatted Row
        self.lbl_output_splatted = ttk.Label(self.folder_frame, text="Output Splatted:")
        self.lbl_output_splatted.grid(
            row=current_row, column=0, sticky="e", padx=5, pady=0
        )
        self.entry_output_splatted = ttk.Entry(
            self.folder_frame, textvariable=self.output_splatted_var
        )
        self.entry_output_splatted.grid(
            row=current_row, column=1, padx=5, pady=0, sticky="ew"
        )
        self.btn_browse_output_splatted = ttk.Button(
            self.folder_frame,
            text="Browse Folder",
            command=lambda: self._browse_folder(self.output_splatted_var),
        )
        self.btn_browse_output_splatted.grid(row=current_row, column=2, padx=5, pady=0)
        self.chk_multi_map = ttk.Checkbutton(
            self.folder_frame,
            text="Multi-Map",
            variable=self.multi_map_var,
            command=self._on_multi_map_toggle,
        )
        self.chk_multi_map.grid(row=current_row, column=3, padx=5, pady=0)
        self._create_hover_tooltip(self.lbl_output_splatted, "output_splatted")
        self._create_hover_tooltip(self.entry_output_splatted, "output_splatted")
        self._create_hover_tooltip(self.chk_multi_map, "multi_map")
        self._create_hover_tooltip(self.btn_browse_output_splatted, "output_splatted")
        # Reset current_row for next frame
        current_row = 0

        # --- NEW: PREVIEW FRAME ---
        self.previewer = VideoPreviewer(
            self,
            processing_callback=self._preview_processing_callback,
            find_sources_callback=self._find_preview_sources_callback,
            get_params_callback=self.get_current_preview_settings,
            preview_size_var=self.preview_size_var,  # Pass the preview size variable
            resize_callback=self._adjust_window_height_for_content,  # Pass the resize callback
            update_clip_callback=self._update_clip_state_and_text,
            on_clip_navigate_callback=self._auto_save_current_sidecar,
            help_data=self.help_texts,
        )
        self.previewer.pack(fill="both", expand=True, padx=10, pady=1)
        self.previewer.preview_source_combo.configure(
            textvariable=self.preview_source_var
        )

        # Set the preview options ONCE at startup
        self.previewer.preview_source_combo["values"] = [
            "Splat Result",
            "Splat Result(Low)",
            "Occlusion Mask",
            "Occlusion Mask(Low)",
            "Original (Left Eye)",
            "Depth Map",
            "Depth Map (Color)",
            "Anaglyph 3D",
            "Dubois Anaglyph",
            "Optimized Anaglyph",
            "Wigglegram",
        ]
        if not self.preview_source_var.get():
            self.preview_source_var.set("Splat Result")

        # --- NEW: MAIN LAYOUT CONTAINER (Holds Settings Left and Info Right) ---
        self.main_layout_frame = ttk.Frame(self)
        self.main_layout_frame.pack(pady=2, padx=10, fill="x")
        self.main_layout_frame.grid_columnconfigure(0, weight=1)  # Left settings column
        self.main_layout_frame.grid_columnconfigure(
            1, weight=0
        )  # Right info column (fixed width)

        # --- LEFT COLUMN: Settings Stack Frame ---
        self.settings_stack_frame = ttk.Frame(self.main_layout_frame)
        self.settings_stack_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # --- Settings Container Frame (to hold two side-by-side frames) ---
        self.settings_container_frame = ttk.Frame(self.settings_stack_frame)
        self.settings_container_frame.pack(
            pady=(0, 2), fill="x"
        )  # Pack it inside the stack frame
        # Spacer columns on BOTH SIDES of the depth/sliders column:
        #   - col 0: left/process settings (fixed)
        #   - col 1: spacer (expands)
        #   - col 2: depth/sliders (expands more than spacers)
        #   - col 3: spacer (expands)
        #
        # This keeps the slider column visually centered within the available space
        # (while the process/settings column stays pinned left).
        self.settings_container_frame.grid_columnconfigure(
            0, weight=0, minsize=int(self.UI_PROCESS_COL_MIN)
        )
        self.settings_container_frame.grid_columnconfigure(1, weight=1, minsize=0)
        self.settings_container_frame.grid_columnconfigure(
            2, weight=2, minsize=int(self.UI_DEPTH_COL_MIN)
        )
        self.settings_container_frame.grid_columnconfigure(3, weight=1, minsize=0)

        self.settings_container_spacer_left = ttk.Frame(self.settings_container_frame)
        self.settings_container_spacer_left.grid(row=0, column=1, sticky="nsew")

        self.settings_container_spacer_right = ttk.Frame(self.settings_container_frame)
        self.settings_container_spacer_right.grid(row=0, column=3, sticky="nsew")

        # ===================================================================
        # LEFT SIDE: Process Resolution and Settings Frame
        # ===================================================================

        # This container holds both the resolution settings (top) and the splatting/output settings (bottom)
        self.process_settings_container = ttk.Frame(self.settings_container_frame)
        self.process_settings_container.grid(
            row=0, column=0, padx=(5, 0), sticky="nsew"
        )
        self.process_settings_container.grid_columnconfigure(0, weight=1)

        # --- 1. Process Resolution Frame (Top Left) ---
        self.preprocessing_frame = ttk.LabelFrame(
            self.process_settings_container, text="Process Resolution"
        )
        self.preprocessing_frame.grid(
            row=0, column=0, padx=(0, 5), sticky="nsew"
        )  # <-- Grid 0,0 in process_settings_container
        self.preprocessing_frame.grid_columnconfigure(
            1, weight=0
        )  # Allow Entry to expand

        current_row = 0

        # --- Enable Full/Low Resolution + Tests (Compact 3x3 Grid) ---
        # Layout:
        #   Row 0: [Enable Full Res]  [Batch Size]  [Dual Output Only]
        #   Row 1: [Enable Low Res ]  [Batch Size]  [Splat Test]
        #   Row 2: [Width]           [Height]      [Map Test]
        # Keep vertical padding minimal so preview area can expand.

        # Ensure 3 primary columns exist
        self.preprocessing_frame.grid_columnconfigure(0, weight=0)
        self.preprocessing_frame.grid_columnconfigure(1, weight=0)
        self.preprocessing_frame.grid_columnconfigure(2, weight=0)

        # --- Row 0: Full res ---
        self.enable_full_res_checkbox = ttk.Checkbutton(
            self.preprocessing_frame,
            text="Enable Full Res",
            variable=self.enable_full_res_var,
            command=self.toggle_processing_settings_fields,
            width=15,
        )
        self.enable_full_res_checkbox.grid(row=0, column=0, sticky="w", padx=5, pady=0)
        self._create_hover_tooltip(self.enable_full_res_checkbox, "enable_full_res")

        self.full_batch_frame = ttk.Frame(self.preprocessing_frame)
        self.full_batch_frame.grid(row=0, column=1, sticky="w", padx=5, pady=0)
        self.lbl_full_res_batch_size = ttk.Label(
            self.full_batch_frame, text="Batch Size:"
        )
        self.lbl_full_res_batch_size.pack(side="left", padx=(0, 2))
        self.entry_full_res_batch_size = ttk.Entry(
            self.full_batch_frame, textvariable=self.batch_size_var, width=5
        )
        self.entry_full_res_batch_size.pack(side="left")
        self._create_hover_tooltip(self.lbl_full_res_batch_size, "full_res_batch_size")
        self._create_hover_tooltip(
            self.entry_full_res_batch_size, "full_res_batch_size"
        )

        self.dual_output_checkbox = ttk.Checkbutton(
            self.preprocessing_frame,
            text="Dual Output Only",
            variable=self.dual_output_var,
        )
        self.dual_output_checkbox.grid(row=0, column=2, sticky="w", padx=5, pady=0)
        self._create_hover_tooltip(self.dual_output_checkbox, "dual_output")

        # --- Row 1: Low res ---
        self.enable_low_res_checkbox = ttk.Checkbutton(
            self.preprocessing_frame,
            text="Enable Low Res",
            variable=self.enable_low_res_var,
            command=self.toggle_processing_settings_fields,
            width=15,
        )
        self.enable_low_res_checkbox.grid(row=1, column=0, sticky="w", padx=5, pady=0)
        self._create_hover_tooltip(self.enable_low_res_checkbox, "enable_low_res")

        self.low_batch_frame = ttk.Frame(self.preprocessing_frame)
        self.low_batch_frame.grid(row=1, column=1, sticky="w", padx=5, pady=0)
        self.lbl_low_res_batch_size = ttk.Label(
            self.low_batch_frame, text="Batch Size:"
        )
        self.lbl_low_res_batch_size.pack(side="left", padx=(0, 2))
        self.entry_low_res_batch_size = ttk.Entry(
            self.low_batch_frame, textvariable=self.low_res_batch_size_var, width=5
        )
        self.entry_low_res_batch_size.pack(side="left")
        self._create_hover_tooltip(self.lbl_low_res_batch_size, "low_res_batch_size")
        self._create_hover_tooltip(self.entry_low_res_batch_size, "low_res_batch_size")

        # Tests (mutually exclusive)
        self.splat_test_var = tk.BooleanVar(value=False)
        self.map_test_var = tk.BooleanVar(value=False)

        # --- Row 2: Width/Height + Map Test ---
        self.width_frame = ttk.Frame(self.preprocessing_frame)
        self.width_frame.grid(row=2, column=0, sticky="w", padx=5, pady=0)
        self.pre_res_width_label = ttk.Label(self.width_frame, text="Width:")
        self.pre_res_width_label.pack(side="left", padx=(0, 2))
        self.pre_res_width_entry = ttk.Entry(
            self.width_frame, textvariable=self.pre_res_width_var, width=8
        )
        self.pre_res_width_entry.pack(side="left")
        self._create_hover_tooltip(self.pre_res_width_label, "low_res_width")
        self._create_hover_tooltip(self.pre_res_width_entry, "low_res_width")

        self.height_frame = ttk.Frame(self.preprocessing_frame)
        self.height_frame.grid(row=2, column=1, sticky="w", padx=5, pady=0)
        self.pre_res_height_label = ttk.Label(self.height_frame, text="Height:")
        self.pre_res_height_label.pack(side="left", padx=(0, 2))
        self.pre_res_height_entry = ttk.Entry(
            self.height_frame, textvariable=self.pre_res_height_var, width=8
        )
        self.pre_res_height_entry.pack(side="left")
        self._create_hover_tooltip(self.pre_res_height_label, "low_res_height")
        self._create_hover_tooltip(self.pre_res_height_entry, "low_res_height")
        # --- 2. Splatting & Output Settings Frame (Bottom Left) ---
        # Compact 2x2 layout:
        #   Row 0: Process Length | Auto-Convergence
        #   Row 1: Output CRF Full | Output CRF Low

        self.output_settings_frame = ttk.LabelFrame(
            self.process_settings_container, text="Splatting & Output Settings"
        )
        self.output_settings_frame.grid(
            row=1, column=0, padx=(0, 5), sticky="ew", pady=(2, 0)
        )
        self.output_settings_frame.grid_columnconfigure(0, weight=0)
        self.output_settings_frame.grid_columnconfigure(1, weight=0)

        # Row 0, Col 0: Process Length
        self.proc_len_frame = ttk.Frame(self.output_settings_frame)
        self.proc_len_frame.grid(row=0, column=0, sticky="w", padx=5, pady=0)
        self.lbl_process_length = ttk.Label(self.proc_len_frame, text="Process Length:")
        self.lbl_process_length.pack(side="left", padx=(0, 3))
        self.entry_process_length = ttk.Entry(
            self.proc_len_frame, textvariable=self.process_length_var, width=6
        )
        self.entry_process_length.pack(side="left")
        self._create_hover_tooltip(self.lbl_process_length, "process_length")
        self._create_hover_tooltip(self.entry_process_length, "process_length")

        # Row 0, Col 1: Auto-Convergence
        self.auto_conv_frame = ttk.Frame(self.output_settings_frame)
        self.auto_conv_frame.grid(row=0, column=1, sticky="w", padx=5, pady=0)
        self.lbl_auto_convergence = ttk.Label(
            self.auto_conv_frame, text="Auto-Convergence:"
        )
        self.lbl_auto_convergence.pack(side="left", padx=(0, 3))
        self.auto_convergence_combo = ttk.Combobox(
            self.auto_conv_frame,
            textvariable=self.auto_convergence_mode_var,
            values=["Off", "Average", "Peak"],
            state="readonly",
            width=10,
        )
        self.auto_convergence_combo.pack(side="left")
        self._create_hover_tooltip(self.lbl_auto_convergence, "auto_convergence_toggle")
        self._create_hover_tooltip(
            self.auto_convergence_combo, "auto_convergence_toggle"
        )
        self.auto_convergence_combo.bind(
            "<<ComboboxSelected>>", self.on_auto_convergence_mode_select
        )

        # Row 1, Col 0: Output CRF Full
        self.crf_full_frame = ttk.Frame(self.output_settings_frame)
        self.crf_full_frame.grid(row=1, column=0, sticky="w", padx=5, pady=0)
        self.lbl_output_crf_full = ttk.Label(
            self.crf_full_frame, text="Output CRF Full:"
        )
        self.lbl_output_crf_full.pack(side="left", padx=(0, 3))
        self.entry_output_crf_full = ttk.Entry(
            self.crf_full_frame, textvariable=self.output_crf_full_var, width=3
        )
        self.entry_output_crf_full.pack(side="left")
        self._create_hover_tooltip(self.lbl_output_crf_full, "output_crf")
        self._create_hover_tooltip(self.entry_output_crf_full, "output_crf")

        # Row 1, Col 1: Output CRF Low
        self.crf_low_frame = ttk.Frame(self.output_settings_frame)
        self.crf_low_frame.grid(row=1, column=1, sticky="w", padx=5, pady=0)
        self.lbl_output_crf_low = ttk.Label(self.crf_low_frame, text="Output CRF Low:")
        self.lbl_output_crf_low.pack(side="left", padx=(0, 3))
        self.entry_output_crf_low = ttk.Entry(
            self.crf_low_frame, textvariable=self.output_crf_low_var, width=3
        )
        self.entry_output_crf_low.pack(side="left")
        self._create_hover_tooltip(self.lbl_output_crf_low, "output_crf")
        self._create_hover_tooltip(self.entry_output_crf_low, "output_crf")

        # Row 2, Col 0: Output color tag mode (metadata-only; written into output file headers)
        self.color_tags_frame = ttk.Frame(self.output_settings_frame)
        self.color_tags_frame.grid(row=2, column=0, sticky="w", padx=5, pady=0)
        self.lbl_color_tags_mode = ttk.Label(self.color_tags_frame, text="Color Tags:")
        self.lbl_color_tags_mode.pack(side="left", padx=(0, 3))
        self.combo_color_tags_mode = ttk.Combobox(
            self.color_tags_frame,
            textvariable=self.color_tags_mode_var,
            values=["Off", "Auto", "BT.709", "BT.2020"],
            state="readonly",
            width=10,
        )
        self.combo_color_tags_mode.pack(side="left")
        self._create_hover_tooltip(self.lbl_color_tags_mode, "color_tags_mode")
        self._create_hover_tooltip(self.combo_color_tags_mode, "color_tags_mode")

        # Row 2, Col 1: Border Mode Pulldown + Rescan Button
        self.border_mode_frame = ttk.Frame(self.output_settings_frame)
        self.border_mode_frame.grid(row=2, column=1, sticky="w", padx=5, pady=0)
        self.lbl_border_mode = ttk.Label(self.border_mode_frame, text="Border:")
        self.lbl_border_mode.pack(side="left", padx=(0, 3))
        self.combo_border_mode = ttk.Combobox(
            self.border_mode_frame,
            textvariable=self.border_mode_var,
            values=["Manual", "Auto Basic", "Auto Adv.", "Off"],
            state="readonly",
            width=10,
        )
        self.combo_border_mode.pack(side="left")

        # Rescan Button (similar to loop button)
        self.btn_border_rescan = ttk.Button(
            self.border_mode_frame,
            text="⟳",  # Unicode rescan-like symbol
            width=2,
            command=self._on_border_rescan_click,
        )
        self.btn_border_rescan.pack(side="left", padx=(3, 0))

        self._create_hover_tooltip(self.lbl_border_mode, "border_mode")
        self._create_hover_tooltip(self.combo_border_mode, "border_mode")
        self._create_hover_tooltip(self.btn_border_rescan, "border_rescan")

        # Track these for disabling during processing
        self.widgets_to_disable.extend(
            [self.combo_color_tags_mode, self.combo_border_mode, self.btn_border_rescan]
        )

        current_row = 0  # Reset for next frame
        # ===================================================================
        # RIGHT SIDE: Depth Map Pre-processing Frame
        # ===================================================================
        self.depth_settings_container = ttk.Frame(self.settings_container_frame)
        self.depth_settings_container.grid(row=0, column=2, padx=(5, 0), sticky="nsew")
        self.depth_settings_container.grid_columnconfigure(0, weight=1)

        # --- Hi-Res Depth Pre-processing Frame (Top-Right) ---
        current_depth_row = 0  # Use a new counter for this container
        self.depth_prep_frame = ttk.LabelFrame(
            self.depth_settings_container, text="Depth Map Pre-processing"
        )
        self.depth_prep_frame.grid(
            row=current_depth_row, column=0, sticky="ew"
        )  # Use grid here for placement inside container
        self.depth_prep_frame.grid_columnconfigure(1, weight=1)

        # Slider Implementation for dilate and blur
        # --- MODIFIED: Dilation Slider with Expanded Erosion Range (Slider goes to 40) ---
        row_inner = 0
        _, _, _ = create_dual_slider_layout(
            self,
            self.depth_prep_frame,
            "Dilate X:",
            "Y:",
            self.depth_dilate_size_x_var,
            self.depth_dilate_size_y_var,
            -10.0,
            30.0,
            row_inner,
            decimals=1,
            is_integer=False,
            tooltip_key_x="depth_dilate_size_x",
            tooltip_key_y="depth_dilate_size_y",
            trough_increment=0.5,
            display_next_odd_integer=False,
            default_x=0.0,
            default_y=0.0,
        )
        row_inner += 1
        _, _, _ = create_dual_slider_layout(
            self,
            self.depth_prep_frame,
            "   Blur X:",
            "Y:",
            self.depth_blur_size_x_var,
            self.depth_blur_size_y_var,
            0,
            35,
            row_inner,
            decimals=0,
            is_integer=True,
            tooltip_key_x="depth_blur_size_x",
            tooltip_key_y="depth_blur_size_y",
            trough_increment=1.0,
            default_x=0.0,
            default_y=0.0,
        )

        row_inner += 1

        # Dilate Left (0.5 steps) + Blur Left (integer steps)
        _, _, (_, left_blur_frame) = create_dual_slider_layout(
            self,
            self.depth_prep_frame,
            "Dilate Left:",
            "Blur Left:",
            self.depth_dilate_left_var,
            self.depth_blur_left_var,
            0.0,
            20.0,
            row_inner,
            decimals=1,
            decimals_y=0,
            tooltip_key_x="depth_dilate_left",
            tooltip_key_y="depth_blur_left",
            trough_increment=0.5,
            default_x=0.0,
            default_y=0.0,
        )

        # Blur Left H↔V balance (0.0 = all horizontal, 1.0 = all vertical; 0.5 = 50/50)
        try:
            mix_values = [f"{i / 10:.1f}" for i in range(0, 11)]
        except Exception:
            mix_values = [
                "0.0",
                "0.1",
                "0.2",
                "0.3",
                "0.4",
                "0.5",
                "0.6",
                "0.7",
                "0.8",
                "0.9",
                "1.0",
            ]

        # Place a small selector immediately after the Blur Left slider (keep it compact so the row doesn't grow).
        left_blur_mix_combo = ttk.Combobox(
            left_blur_frame,
            textvariable=self.depth_blur_left_mix_var,
            values=mix_values,
            width=4,
            state="readonly",
        )
        # Most slider layouts use columns 0-2; put this in the next free column.
        left_blur_mix_combo.grid(row=0, column=3, sticky="w", padx=(4, 0))
        self._create_hover_tooltip(left_blur_mix_combo, "depth_blur_left_mix")
        # Trigger an immediate preview refresh when the mix selector changes
        left_blur_mix_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self.previewer.update_preview()
            if getattr(self, "previewer", None)
            else None,
        )

        # --- NEW: Depth Pre-processing (All) Frame (Bottom-Right) ---
        current_depth_row += 1
        self.depth_all_settings_frame = ttk.LabelFrame(
            self.depth_settings_container, text="Stereo Projection"
        )
        self.depth_all_settings_frame.grid(
            row=current_depth_row, column=0, sticky="ew", pady=(2, 0)
        )  # Pack it below Hi-Res frame
        self.depth_all_settings_frame.grid_columnconfigure(0, weight=0)
        self.depth_all_settings_frame.grid_columnconfigure(1, weight=1)
        self.depth_all_settings_frame.grid_columnconfigure(2, weight=0)

        all_settings_row = 0

        # Gamma + Disparity (same row)
        gamma_disp_row_frame = ttk.Frame(self.depth_all_settings_frame)
        gamma_disp_row_frame.grid(
            row=all_settings_row, column=0, columnspan=3, sticky="ew"
        )
        gamma_disp_row_frame.grid_columnconfigure(0, weight=1)
        gamma_disp_row_frame.grid_columnconfigure(1, weight=1)

        gamma_subframe = ttk.Frame(gamma_disp_row_frame)
        gamma_subframe.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        gamma_subframe.grid_columnconfigure(1, weight=1)

        disp_subframe = ttk.Frame(gamma_disp_row_frame)
        disp_subframe.grid(row=0, column=1, sticky="ew")
        disp_subframe.grid_columnconfigure(1, weight=1)

        create_single_slider_with_label_updater(
            self,
            gamma_subframe,
            "Gamma:",
            self.depth_gamma_var,
            0.1,
            3.0,
            0,
            decimals=2,
            tooltip_key="depth_gamma",
            trough_increment=0.05,
            step_size=0.05,
            default_value=1.0,
        )

        self.set_disparity_value_programmatically = (
            create_single_slider_with_label_updater(
                self,
                disp_subframe,
                "Disparity:",
                self.max_disp_var,
                0.0,
                100.0,
                0,
                decimals=0,
                tooltip_key="max_disp",
                step_size=1.0,
                default_value=30.0,
            )
        )
        all_settings_row += 1

        # Convergence Point Slider
        setter_func_conv = create_single_slider_with_label_updater(
            self,
            self.depth_all_settings_frame,
            "Convergence:",
            self.zero_disparity_anchor_var,
            0.0,
            2.0,
            all_settings_row,
            decimals=2,
            tooltip_key="convergence_point",
            step_size=0.01,
            default_value=0.5,
        )
        self.set_convergence_value_programmatically = setter_func_conv
        all_settings_row += 1

        # Border Width & Bias Sliders
        # The improved create_dual_slider_layout returns the frame, setters, and subframes.
        (
            self.border_sliders_row_frame,
            (
                self.set_border_width_programmatically,
                self.set_border_bias_programmatically,
            ),
            _,
        ) = create_dual_slider_layout(
            self,
            self.depth_all_settings_frame,
            "Border Width:",
            "Bias:",
            self.border_width_var,
            self.border_bias_var,
            0.0,
            5.0,
            all_settings_row,
            decimals=2,
            tooltip_key_x="border_width",
            tooltip_key_y="border_bias",
            trough_increment=0.1,
            step_size_x=0.01,
            step_size_y=0.01,
            default_x=0.0,
            default_y=0.0,
            from_y=-1.0,
            to_y=1.0,
        )
        # Store the frame containing the sliders for the manual toggle logic
        # Since create_dual_slider_layout now returns the frame directly,
        # the lookup loop is no longer needed.
        # for child in self.depth_all_settings_frame.winfo_children():
        #     info = child.grid_info()
        #     if info.get("row") == str(all_settings_row):
        #         self.border_sliders_row_frame = child
        #         break

        all_settings_row += 1
        # --- Global Normalization + Resume (packed in a sub-frame so slider columns stay aligned) ---
        checkbox_row = ttk.Frame(self.depth_all_settings_frame)
        checkbox_row.grid(
            row=all_settings_row, column=0, columnspan=3, sticky="w", padx=5, pady=0
        )

        self.global_norm_checkbox = ttk.Checkbutton(
            checkbox_row,
            text="Enable Global Normalization",
            variable=self.enable_global_norm_var,
            command=self.on_slider_release,
        )
        self.global_norm_checkbox.pack(side="left")
        self._create_hover_tooltip(
            self.global_norm_checkbox, "enable_global_normalization"
        )

        self.move_to_finished_checkbox = ttk.Checkbutton(
            checkbox_row, text="Resume", variable=self.move_to_finished_var
        )
        self.move_to_finished_checkbox.pack(side="left", padx=(10, 0))
        self._create_hover_tooltip(
            self.move_to_finished_checkbox, "move_to_finished_folder"
        )
        # Crosshair overlay controls (preview only)
        self.crosshair_checkbox = ttk.Checkbutton(
            checkbox_row,
            text="Crosshair",
            variable=self.crosshair_enabled_var,
            takefocus=False,
            command=lambda: (
                self.previewer.set_crosshair_settings(
                    self.crosshair_enabled_var.get(),
                    self.crosshair_white_var.get(),
                    self.crosshair_multi_var.get(),
                ),
                getattr(self.previewer, "preview_canvas", self.previewer).focus_set(),
            )
            if getattr(self, "previewer", None)
            else None,
        )
        self.crosshair_checkbox.pack(side="left", padx=(24, 0))
        self._create_hover_tooltip(self.crosshair_checkbox, "crosshair_enabled")

        self.crosshair_white_checkbox = ttk.Checkbutton(
            checkbox_row,
            text="White",
            variable=self.crosshair_white_var,
            takefocus=False,
            command=lambda: (
                self.previewer.set_crosshair_settings(
                    self.crosshair_enabled_var.get(),
                    self.crosshair_white_var.get(),
                    self.crosshair_multi_var.get(),
                ),
                getattr(self.previewer, "preview_canvas", self.previewer).focus_set(),
            )
            if getattr(self, "previewer", None)
            else None,
        )
        self.crosshair_white_checkbox.pack(side="left", padx=(6, 0))
        self._create_hover_tooltip(self.crosshair_white_checkbox, "crosshair_white")

        self.crosshair_multi_checkbox = ttk.Checkbutton(
            checkbox_row,
            text="Multi",
            variable=self.crosshair_multi_var,
            takefocus=False,
            command=lambda: (
                self.previewer.set_crosshair_settings(
                    self.crosshair_enabled_var.get(),
                    self.crosshair_white_var.get(),
                    self.crosshair_multi_var.get(),
                ),
                getattr(self.previewer, "preview_canvas", self.previewer).focus_set(),
            )
            if getattr(self, "previewer", None)
            else None,
        )
        self.crosshair_multi_checkbox.pack(side="left", padx=(6, 0))
        self._create_hover_tooltip(self.crosshair_multi_checkbox, "crosshair_multi")
        self.depth_pop_checkbox = ttk.Checkbutton(
            checkbox_row,
            text="D/P",
            variable=self.depth_pop_enabled_var,
            takefocus=False,
            command=lambda: (
                getattr(self.previewer, "set_depth_pop_enabled", lambda *_: None)(
                    self.depth_pop_enabled_var.get()
                ),
                getattr(self.previewer, "preview_canvas", self.previewer).focus_set(),
            )
            if getattr(self, "previewer", None)
            else None,
        )
        self.depth_pop_checkbox.pack(side="left", padx=(24, 0))
        self._create_hover_tooltip(self.depth_pop_checkbox, "depth_pop_readout")

        all_settings_row += 1

        current_row = 0  # Reset for next frame
        # ===================================================================
        # --- RIGHT COLUMN: Current Processing Information frame ---
        # ===================================================================
        # --- RIGHT COLUMN STACK (Current Processing Information + Dev Tools) ---
        self.right_column_stack = ttk.Frame(self.main_layout_frame)
        self.right_column_stack.grid(row=0, column=1, sticky="nsew", padx=(0, 0))
        self.right_column_stack.grid_columnconfigure(0, weight=1, minsize=0)
        self.right_column_stack.grid_rowconfigure(0, weight=0)
        self.right_column_stack.grid_rowconfigure(1, weight=0)

        # --- Current Processing Information frame ---
        self.info_frame = ttk.LabelFrame(
            self.right_column_stack, text="Current Processing Information"
        )
        self.info_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 2))

        # Two-column pairs: [Label, Value] [Label, Value]
        self.info_frame.grid_columnconfigure(0, weight=0)
        self.info_frame.grid_columnconfigure(1, weight=1, minsize=0)
        self.info_frame.grid_columnconfigure(2, weight=0)
        self.info_frame.grid_columnconfigure(3, weight=1, minsize=0)

        self.info_labels = []  # List to hold the tk.Label widgets for easy iteration
        LABEL_VALUE_WIDTH = 18
        info_row = 0

        # Row 0: Filename (span across)
        lbl_filename_static = tk.Label(self.info_frame, text="Filename:")
        lbl_filename_static.grid(row=info_row, column=0, sticky="e", padx=5, pady=1)
        lbl_filename_value = tk.Label(
            self.info_frame,
            textvariable=self.processing_filename_var,
            anchor="w",
            width=LABEL_VALUE_WIDTH,
        )
        lbl_filename_value.grid(
            row=info_row, column=1, columnspan=3, sticky="ew", padx=5, pady=1
        )
        self.info_labels.extend([lbl_filename_static, lbl_filename_value])
        info_row += 1

        # Row 1: Task (span across)
        lbl_task_static = tk.Label(self.info_frame, text="Task:")
        lbl_task_static.grid(row=info_row, column=0, sticky="e", padx=5, pady=1)
        lbl_task_value = tk.Label(
            self.info_frame,
            textvariable=self.processing_task_name_var,
            anchor="w",
            width=LABEL_VALUE_WIDTH,
        )
        lbl_task_value.grid(
            row=info_row, column=1, columnspan=3, sticky="ew", padx=5, pady=1
        )
        self.info_labels.extend([lbl_task_static, lbl_task_value])
        info_row += 1

        # Row 2: Resolution | Disparity
        lbl_resolution_static = tk.Label(self.info_frame, text="Resolution:")
        lbl_resolution_static.grid(row=info_row, column=0, sticky="e", padx=5, pady=1)
        lbl_resolution_value = tk.Label(
            self.info_frame,
            textvariable=self.processing_resolution_var,
            anchor="w",
            width=LABEL_VALUE_WIDTH,
        )
        lbl_resolution_value.grid(row=info_row, column=1, sticky="ew", padx=5, pady=1)

        lbl_disparity_static = tk.Label(self.info_frame, text="Disparity:")
        lbl_disparity_static.grid(row=info_row, column=2, sticky="e", padx=5, pady=1)
        lbl_disparity_value = tk.Label(
            self.info_frame,
            textvariable=self.processing_disparity_var,
            anchor="w",
            width=LABEL_VALUE_WIDTH,
        )
        lbl_disparity_value.grid(row=info_row, column=3, sticky="ew", padx=5, pady=1)

        self.info_labels.extend(
            [
                lbl_resolution_static,
                lbl_resolution_value,
                lbl_disparity_static,
                lbl_disparity_value,
            ]
        )
        info_row += 1

        # Row 3: Frames | Converge
        lbl_frames_static = tk.Label(self.info_frame, text="Frames:")
        lbl_frames_static.grid(row=info_row, column=0, sticky="e", padx=5, pady=1)
        lbl_frames_value = tk.Label(
            self.info_frame,
            textvariable=self.processing_frames_var,
            anchor="w",
            width=LABEL_VALUE_WIDTH,
        )
        lbl_frames_value.grid(row=info_row, column=1, sticky="ew", padx=5, pady=1)

        lbl_converge_static = tk.Label(self.info_frame, text="Converge:")
        lbl_converge_static.grid(row=info_row, column=2, sticky="e", padx=5, pady=1)
        lbl_converge_value = tk.Label(
            self.info_frame,
            textvariable=self.processing_convergence_var,
            anchor="w",
            width=LABEL_VALUE_WIDTH,
        )
        lbl_converge_value.grid(row=info_row, column=3, sticky="ew", padx=5, pady=1)

        self.info_labels.extend(
            [
                lbl_frames_static,
                lbl_frames_value,
                lbl_converge_static,
                lbl_converge_value,
            ]
        )
        info_row += 1

        # Row 4: Gamma | Map
        lbl_gamma_static = tk.Label(self.info_frame, text="Gamma:")
        lbl_gamma_static.grid(row=info_row, column=0, sticky="e", padx=5, pady=1)
        lbl_gamma_value = tk.Label(
            self.info_frame,
            textvariable=self.processing_gamma_var,
            anchor="w",
            width=LABEL_VALUE_WIDTH,
        )
        lbl_gamma_value.grid(row=info_row, column=1, sticky="ew", padx=5, pady=1)

        lbl_map_static = tk.Label(self.info_frame, text="Map:")
        lbl_map_static.grid(row=info_row, column=2, sticky="e", padx=5, pady=1)
        lbl_map_value = tk.Label(
            self.info_frame,
            textvariable=self.processing_map_var,
            anchor="w",
            width=LABEL_VALUE_WIDTH,
        )
        lbl_map_value.grid(row=info_row, column=3, sticky="ew", padx=5, pady=1)

        self.info_labels.extend(
            [lbl_gamma_static, lbl_gamma_value, lbl_map_static, lbl_map_value]
        )

        # --- Dev Tools (right column) ---
        self.dev_tools_frame = ttk.LabelFrame(self.right_column_stack, text="Dev Tools")
        self.dev_tools_frame.grid(row=1, column=0, sticky="ew", padx=0, pady=(0, 0))
        self.dev_tools_frame.grid_columnconfigure(0, weight=0)
        self.dev_tools_frame.grid_columnconfigure(1, weight=0)
        self.dev_tools_frame.grid_columnconfigure(2, weight=0)
        self.dev_tools_frame.grid_columnconfigure(3, weight=1)

        self.chk_skip_lowres_preproc = ttk.Checkbutton(
            self.dev_tools_frame,
            text="Skip Low-Res Pre-proc",
            variable=self.skip_lowres_preproc_var,
        )
        self.chk_skip_lowres_preproc.grid(row=0, column=0, sticky="w", padx=5, pady=0)
        self._create_hover_tooltip(self.chk_skip_lowres_preproc, "skip_lowres_preproc")

        self.chk_map_test = ttk.Checkbutton(
            self.dev_tools_frame,
            text="Map Test",
            variable=self.map_test_var,
            command=lambda: self.splat_test_var.set(False)
            if self.map_test_var.get()
            else None,
        )
        self.chk_map_test.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=0)
        self._create_hover_tooltip(self.chk_map_test, "map_test")

        self.chk_splat_test = ttk.Checkbutton(
            self.dev_tools_frame,
            text="Splat Test",
            variable=self.splat_test_var,
            command=lambda: self.map_test_var.set(False)
            if self.splat_test_var.get()
            else None,
        )
        self.chk_splat_test.grid(row=0, column=2, sticky="w", padx=(10, 0), pady=0)
        self._create_hover_tooltip(self.chk_splat_test, "splat_test")

        progress_frame = ttk.LabelFrame(self, text="Progress")
        progress_frame.pack(pady=2, padx=10, fill="x")
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(fill="x", expand=True, padx=5, pady=2)
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack(padx=5, pady=2)

        # --- Button frame ---
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=2)

        # --- Single Process Button ---
        self.start_single_button = ttk.Button(
            button_frame, text="SINGLE", command=self.start_single_processing
        )
        self.start_single_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.start_single_button, "start_single_button")

        # --- Start Process Button ---
        self.start_button = ttk.Button(
            button_frame, text="START", command=self.start_processing
        )
        self.start_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.start_button, "start_button")

        # --- From/To Process Range ---
        ttk.Label(button_frame, text="From:").pack(side="left", padx=(15, 2))
        self.entry_process_from = ttk.Entry(
            button_frame, textvariable=self.process_from_var, width=6
        )
        self.entry_process_from.pack(side="left", padx=2)
        self._create_hover_tooltip(self.entry_process_from, "process_from")

        ttk.Label(button_frame, text="To:").pack(side="left", padx=(5, 2))
        self.entry_process_to = ttk.Entry(
            button_frame, textvariable=self.process_to_var, width=6
        )
        self.entry_process_to.pack(side="left", padx=2)
        self._create_hover_tooltip(self.entry_process_to, "process_to")

        # --- Stop Process Button ---
        self.stop_button = ttk.Button(
            button_frame, text="STOP", command=self.stop_processing, state="disabled"
        )
        self.stop_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.stop_button, "stop_button")

        # --- Preview Auto-Converge Button ---
        self.btn_auto_converge_preview = ttk.Button(
            button_frame,
            text="Preview Auto-Converge",
            command=self.run_preview_auto_converge,
        )
        self.btn_auto_converge_preview.pack(side="left", padx=5)
        self._create_hover_tooltip(
            self.btn_auto_converge_preview, "preview_auto_converge"
        )

        # --- Update Sidecar Button ---
        self.update_sidecar_button = ttk.Button(
            button_frame, text="Update Sidecar", command=self.update_sidecar_file
        )
        self.update_sidecar_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.update_sidecar_button, "update_sidecar_button")

    def _setup_keyboard_shortcuts(self):
        """Sets up keyboard shortcuts for quick adjustments.

        Shortcuts only work when NOT in a text entry field:
        - 7/9: Previous/Next depth map (Multi-Map)
        - 4/6: Decrease/Increase Max Disparity
        - 1/3: Decrease/Increase Convergence
        - 2: Cycle Border Mode
        """
        self.bind("<KeyPress>", self._handle_keypress)

    def _handle_keypress(self, event):
        """Handles keyboard shortcuts, but only when not in a text entry."""
        # Check if focus is in an Entry or Text widget
        focused_widget = self.focus_get()
        if isinstance(
            focused_widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Combobox, ttk.Spinbox)
        ):
            # User is typing in a text field - don't intercept
            return

        # ENTER: Update sidecar (confirmation dialog will appear as usual)
        if event.keysym in ("Return", "KP_Enter"):
            try:
                self.update_sidecar_file()
            except Exception as e:
                logger.error(f"Enter-key sidecar update failed: {e}", exc_info=True)
            return "break"

        # Map shortcuts
        if event.char == "7":
            self._cycle_depth_map(-1)  # Previous map
        elif event.char == "9":
            self._cycle_depth_map(1)  # Next map
        elif event.char == "4":
            self._adjust_disparity(-1)  # Decrease disparity
        elif event.char == "6":
            self._adjust_disparity(1)  # Increase disparity
        elif event.char == "1":
            self._adjust_convergence(-0.01)  # Decrease convergence
        elif event.char == "3":
            self._adjust_convergence(0.01)  # Increase convergence
        elif event.char == "2":
            self._cycle_border_mode()  # Cycle border mode

    def _cycle_depth_map(self, direction):
        """Cycles through depth map subfolders.

        Args:
            direction: -1 for previous, 1 for next
        """
        if not self.multi_map_var.get():
            return  # Multi-Map not enabled

        if not self.depth_map_subfolders:
            return  # No subfolders

        current_value = self.selected_depth_map_var.get()
        try:
            current_index = self.depth_map_subfolders.index(current_value)
        except ValueError:
            current_index = 0

        # Calculate new index with wrapping
        new_index = (current_index + direction) % len(self.depth_map_subfolders)
        new_value = self.depth_map_subfolders[new_index]

        # Update the selection
        self.selected_depth_map_var.set(new_value)

        # Trigger the map change
        self._on_map_selection_changed()

    def _cycle_border_mode(self):
        """Cycles through border modes: Manual -> Auto Basic -> Auto Adv. -> Off."""
        modes = ["Manual", "Auto Basic", "Auto Adv.", "Off"]
        current = self.border_mode_var.get()
        try:
            idx = modes.index(current)
        except ValueError:
            idx = modes.index("Off")

        new_idx = (idx + 1) % len(modes)
        self.border_mode_var.set(modes[new_idx])

    def _adjust_disparity(self, direction):
        """Adjusts Max Disparity value.

        Args:
            direction: -1 to decrease, 1 to increase
        """
        try:
            current = float(self.max_disp_var.get())
            new_value = max(0, min(100, current + direction))  # Clamp 0-100

            # Use the proper setter function which updates both slider AND label
            if (
                hasattr(self, "set_disparity_value_programmatically")
                and self.set_disparity_value_programmatically
            ):
                self.set_disparity_value_programmatically(new_value)
            else:
                self.max_disp_var.set(f"{new_value:.1f}")

            # Trigger preview update
            self.on_slider_release(None)
        except ValueError:
            pass  # Invalid current value

    def _adjust_convergence(self, delta):
        """Adjusts Convergence Plane value.

        Args:
            delta: Amount to change (e.g., 0.01 or -0.01)
        """
        try:
            current = float(self.zero_disparity_anchor_var.get())
            new_value = max(0.0, min(1.0, current + delta))  # Clamp between 0 and 1

            # Use the proper setter function which updates both slider AND label
            if self.set_convergence_value_programmatically:
                self.set_convergence_value_programmatically(new_value)
            else:
                self.zero_disparity_anchor_var.set(f"{new_value:.2f}")

            # Trigger preview update
            self.on_slider_release(None)
        except ValueError:
            pass  # Invalid current value

    def depthSplatting(
        self: "SplatterGUI",
        input_video_reader: VideoReader,
        depth_map_reader: VideoReader,
        total_frames_to_process: int,
        processed_fps: float,
        output_video_path_base: str,
        target_output_height: int,
        target_output_width: int,
        max_disp: float,
        process_length: int,
        batch_size: int,
        dual_output: bool,
        zero_disparity_anchor_val: float,
        video_stream_info: Optional[dict],
        input_bias: Optional[float],
        assume_raw_input: bool,
        global_depth_min: float,
        global_depth_max: float,
        depth_stream_info: Optional[dict],
        user_output_crf: Optional[int] = None,
        is_low_res_task: bool = False,
        depth_gamma: float = 1.0,
        depth_dilate_size_x: float = 0.0,
        depth_dilate_size_y: float = 0.0,
        depth_blur_size_x: float = 0.0,
        depth_blur_size_y: float = 0.0,
        depth_dilate_left: float = 0.0,
        depth_blur_left: float = 0.0,
    ):
        logger.debug("==> Initializing ForwardWarpStereo module")
        stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

        num_frames = total_frames_to_process
        height, width = target_output_height, target_output_width
        os.makedirs(os.path.dirname(output_video_path_base), exist_ok=True)

        # --- Determine output grid dimensions and final path ---
        grid_height, grid_width = (
            (height, width * 2) if dual_output else (height * 2, width * 2)
        )
        suffix = "_splatted2" if dual_output else "_splatted4"
        res_suffix = f"_{width}"
        final_output_video_path = (
            f"{os.path.splitext(output_video_path_base)[0]}{res_suffix}{suffix}.mp4"
        )

        # --- DEBUG: Log ffprobe-derived color metadata for this pass (Hi/Lo parity checks) ---
        task_name = "LowRes" if is_low_res_task else "HiRes"
        try:
            logger.info(
                f"[COLOR_META][{task_name}] input ffprobe: "
                f"pix_fmt={video_stream_info.get('pix_fmt') if video_stream_info else None}, "
                f"range={video_stream_info.get('color_range') if video_stream_info else None}, "
                f"primaries={video_stream_info.get('color_primaries') if video_stream_info else None}, "
                f"trc={video_stream_info.get('transfer_characteristics') if video_stream_info else None}, "
                f"matrix={video_stream_info.get('color_space') if video_stream_info else None}"
            )
        except Exception:
            pass
        # --- END DEBUG ---

        # --- Start FFmpeg pipe process ---
        is_test_mode = bool(
            (getattr(self, "map_test_var", None) and self.map_test_var.get())
            or (getattr(self, "splat_test_var", None) and self.splat_test_var.get())
        )
        ffmpeg_process = None
        if is_test_mode:
            # In test mode we don't want to create/overwrite any output video files.
            logger.debug(
                "Test mode active: skipping FFmpeg encoding (no output video will be written)."
            )
        else:
            # --- Start FFmpeg pipe process ---
            # Apply output color tag mode (metadata-only; does not change splat math)
            encode_stream_info = (
                dict(video_stream_info) if isinstance(video_stream_info, dict) else {}
            )
            try:
                _ct_mode = (
                    self.color_tags_mode_var.get()
                    if hasattr(self, "color_tags_mode_var")
                    else "Auto"
                )
            except Exception:
                _ct_mode = "Auto"

            def _ct_defaults(d: dict) -> dict:
                d.setdefault("color_primaries", "bt709")
                d.setdefault("transfer_characteristics", "bt709")
                d.setdefault("color_space", "bt709")
                # Only metadata; used for tagging if missing. Preserve source if present.
                d.setdefault("color_range", "tv")
                return d

            if _ct_mode == "Auto":
                encode_stream_info = _ct_defaults(encode_stream_info)
            elif _ct_mode == "BT.709 L":
                encode_stream_info.update(
                    {
                        "color_primaries": "bt709",
                        "transfer_characteristics": "bt709",
                        "color_space": "bt709",
                        "color_range": "tv",
                    }
                )
            elif _ct_mode == "BT.709 F":
                encode_stream_info.update(
                    {
                        "color_primaries": "bt709",
                        "transfer_characteristics": "bt709",
                        "color_space": "bt709",
                        "color_range": "pc",
                    }
                )
            elif _ct_mode == "BT.2020 PQ":
                encode_stream_info.update(
                    {
                        "color_primaries": "bt2020",
                        "transfer_characteristics": "smpte2084",
                        "color_space": "bt2020nc",
                        "color_range": "tv",
                    }
                )
            elif _ct_mode == "BT.2020 HLG":
                encode_stream_info.update(
                    {
                        "color_primaries": "bt2020",
                        "transfer_characteristics": "arib-std-b67",
                        "color_space": "bt2020nc",
                        "color_range": "tv",
                    }
                )
            else:
                # Unknown/legacy value; keep current behavior
                encode_stream_info = _ct_defaults(encode_stream_info)

            ffmpeg_process = start_ffmpeg_pipe_process(
                content_width=grid_width,
                content_height=grid_height,
                final_output_mp4_path=final_output_video_path,
                fps=processed_fps,
                video_stream_info=encode_stream_info,
                user_output_crf=user_output_crf,
                output_format_str="splatted_grid",  # Pass a placeholder for the new argument
                debug_label=task_name,
            )
            if ffmpeg_process is None:
                logger.error("Failed to start FFmpeg pipe. Aborting splatting task.")
                return False

        # --- DEBUG: Capture and compare encoding flags between HiRes and LowRes passes ---
        try:
            flags = getattr(ffmpeg_process, "sc_encode_flags", None)
            if flags:
                if not hasattr(self, "_sc_color_encode_flags"):
                    self._sc_color_encode_flags = {}
                subset_keys = [
                    "enc_codec",
                    "enc_pix_fmt",
                    "enc_profile",
                    "enc_color_primaries",
                    "enc_color_trc",
                    "enc_colorspace",
                    "quality_mode",
                    "quality_value",
                ]
                subset = {k: flags.get(k) for k in subset_keys}
                self._sc_color_encode_flags[task_name] = subset

                other_name = "HiRes" if task_name == "LowRes" else "LowRes"
                if other_name in self._sc_color_encode_flags:
                    other = self._sc_color_encode_flags[other_name]
                    diffs = {
                        k: (other.get(k), subset.get(k))
                        for k in subset_keys
                        if other.get(k) != subset.get(k)
                    }
                    if diffs:
                        logger.warning(
                            f"[COLOR_META] Encoding flags differ ({other_name} vs {task_name}): {diffs}"
                        )
                    else:
                        logger.info(
                            f"[COLOR_META] Encoding flags match between {other_name} and {task_name}."
                        )
        except Exception:
            pass
        # --- END DEBUG ---

        # --- Determine max_expected_raw_value for consistent Gamma ---
        max_expected_raw_value = 1.0
        depth_pix_fmt = depth_stream_info.get("pix_fmt") if depth_stream_info else None
        depth_profile = depth_stream_info.get("profile") if depth_stream_info else None
        is_source_10bit = False
        if depth_pix_fmt:
            if (
                "10" in depth_pix_fmt
                or "gray10" in depth_pix_fmt
                or "12" in depth_pix_fmt
                or (depth_profile and "main10" in depth_profile)
            ):
                is_source_10bit = True
        if is_source_10bit:
            max_expected_raw_value = 1023.0
        elif depth_pix_fmt and (
            "8" in depth_pix_fmt or depth_pix_fmt in ["yuv420p", "yuv422p", "yuv444p"]
        ):
            max_expected_raw_value = 255.0
        elif isinstance(depth_pix_fmt, str) and "float" in depth_pix_fmt:
            max_expected_raw_value = 1.0
        logger.debug(
            f"Determined max_expected_raw_value: {max_expected_raw_value:.1f} (Source: {depth_pix_fmt}/{depth_profile})"
        )

        frame_count = 0
        encoding_successful = True  # Assume success unless an error occurs

        try:
            # Map/Splat Test: render the *same frame* the preview is showing.
            test_target_frame_idx = None
            try:
                if (
                    getattr(self, "map_test_var", None) and self.map_test_var.get()
                ) or (
                    getattr(self, "splat_test_var", None) and self.splat_test_var.get()
                ):
                    if (
                        hasattr(self, "previewer")
                        and getattr(self.previewer, "last_loaded_frame_index", None)
                        is not None
                    ):
                        test_target_frame_idx = int(
                            self.previewer.last_loaded_frame_index
                        )
                        test_target_frame_idx = max(
                            0, min(int(num_frames) - 1, test_target_frame_idx)
                        )
            except Exception:
                test_target_frame_idx = None

            frame_index_iter = (
                [test_target_frame_idx]
                if test_target_frame_idx is not None
                else range(0, num_frames, batch_size)
            )
            for i in frame_index_iter:
                t_start_batch = time.perf_counter()  # <--- TIMER START: Total Batch
                if self.stop_event.is_set() or (
                    ffmpeg_process is not None and ffmpeg_process.poll() is not None
                ):
                    if ffmpeg_process is not None and ffmpeg_process.poll() is not None:
                        logger.error(
                            "FFmpeg process terminated unexpectedly. Stopping frame processing."
                        )
                    else:
                        logger.warning(
                            "Stop event received. Terminating FFmpeg process."
                        )
                    encoding_successful = False
                    break

                # --- TIMER 1: Video/Depth I/O (Disk/Decode/Resize) ---
                t_start_io = time.perf_counter()

                current_frame_indices = (
                    [i]
                    if test_target_frame_idx is not None
                    else list(range(i, min(i + batch_size, num_frames)))
                )
                if not current_frame_indices:
                    break

                batch_frames_numpy = input_video_reader.get_batch(
                    current_frame_indices
                ).asnumpy()
                # This often resolves issues where Decord/FFmpeg loses the internal stream position
                try:
                    # Seek to the first frame of the current batch
                    depth_map_reader.seek(current_frame_indices[0])
                    # Then read the full batch from that position
                    batch_depth_numpy_raw = depth_map_reader.get_batch(
                        current_frame_indices
                    ).asnumpy()
                except Exception as e:
                    logger.error(
                        f"Error seeking/reading depth map batch starting at index {i}: {e}. Falling back to a potentially blank read."
                    )
                    batch_depth_numpy_raw = depth_map_reader.get_batch(
                        current_frame_indices
                    ).asnumpy()
                t_end_io = time.perf_counter()

                file_frame_idx = current_frame_indices[0]
                task_name = "LowRes" if is_low_res_task else "HiRes"

                if batch_depth_numpy_raw.min() == batch_depth_numpy_raw.max() == 0:
                    logger.warning(
                        f"Depth map batch starting at index {i} is entirely blank/zero after read. **Seeking failed to resolve.**"
                    )

                if batch_depth_numpy_raw.min() == batch_depth_numpy_raw.max():
                    logger.warning(
                        f"Depth map batch starting at index {i} is entirely uniform/flat after read. Min/Max: {batch_depth_numpy_raw.min():.2f}"
                    )

                # Use the FIRST frame index for the file name (e.g., 00000.png)
                file_frame_idx = current_frame_indices[0]

                # self._save_debug_numpy(batch_depth_numpy_raw, "01_RAW_INPUT", i, file_frame_idx, task_name)
                # --- TIMER 2: CPU Pre-processing (Dilate, Blur, Grayscale, Gamma, Min/Max Calc) ---
                # --- 8-bit resize parity (Map Test Preview vs Render) ---
                # For 8-bit depth videos, Decord's decode-time scaling can differ by ~1-2 code values vs OpenCV.
                # To match the Preview path, decode at native res (see load_pre_rendered_depth) and resize here
                # BEFORE preprocessing using the same OpenCV interpolation policy.
                _bd = (
                    _infer_depth_bit_depth(depth_stream_info)
                    if depth_stream_info
                    else 8
                )
                if _bd <= 8:
                    # Determine pre-processing resolution (matches preview/render ordering).
                    _target_w = int(batch_frames_numpy.shape[2])
                    _target_h = int(batch_frames_numpy.shape[1])
                    if is_low_res_task:
                        _skip_lr = bool(
                            getattr(self, "skip_lowres_preproc_var", None) is not None
                            and self.skip_lowres_preproc_var.get()
                        )
                        if not _skip_lr:
                            # For Low-Res tasks we want to apply preprocessing at the incoming depth-reader resolution
                            # (typically clip-sized when match_resolution_to_target is enabled), then downscale after preprocessing.
                            _target_w = int(batch_depth_numpy_raw.shape[2])
                            _target_h = int(batch_depth_numpy_raw.shape[1])

                    # Reduce to a single channel (depth maps are grayscale-encoded).
                    if batch_depth_numpy_raw.ndim == 4:
                        if batch_depth_numpy_raw.shape[-1] == 3:
                            _depth_1c = batch_depth_numpy_raw[..., 0]
                        elif batch_depth_numpy_raw.shape[-1] == 1:
                            _depth_1c = batch_depth_numpy_raw[..., 0]
                        else:
                            _depth_1c = batch_depth_numpy_raw.squeeze(-1)
                    else:
                        _depth_1c = batch_depth_numpy_raw

                    if (_depth_1c.shape[1] != _target_h) or (
                        _depth_1c.shape[2] != _target_w
                    ):
                        _interp = (
                            cv2.INTER_LINEAR
                            if (
                                _target_w > _depth_1c.shape[2]
                                or _target_h > _depth_1c.shape[1]
                            )
                            else cv2.INTER_AREA
                        )
                        _resized = np.empty(
                            (_depth_1c.shape[0], _target_h, _target_w),
                            dtype=_depth_1c.dtype,
                        )
                        for _di in range(_depth_1c.shape[0]):
                            _resized[_di] = cv2.resize(
                                _depth_1c[_di],
                                (_target_w, _target_h),
                                interpolation=_interp,
                            )
                        batch_depth_numpy_raw = _resized[..., None]
                    else:
                        batch_depth_numpy_raw = _depth_1c[..., None]

                t_start_preproc = time.perf_counter()

                batch_depth_numpy = self._process_depth_batch(
                    batch_depth_numpy_raw=batch_depth_numpy_raw,
                    depth_stream_info=depth_stream_info,
                    depth_gamma=depth_gamma,
                    depth_dilate_size_x=depth_dilate_size_x,
                    depth_dilate_size_y=depth_dilate_size_y,
                    depth_blur_size_x=depth_blur_size_x,
                    depth_blur_size_y=depth_blur_size_y,
                    depth_dilate_left=depth_dilate_left,
                    depth_blur_left=depth_blur_left,
                    is_low_res_task=is_low_res_task,
                    max_raw_value=max_expected_raw_value,
                    global_depth_min=global_depth_min,
                    global_depth_max=global_depth_max,
                    debug_batch_index=i,
                    debug_frame_index=file_frame_idx,
                    debug_task_name=task_name,
                )

                # If the depth batch resolution differs from the source frames,
                # resize AFTER preprocessing so dilation/erosion/blur parity is preserved.
                if (batch_depth_numpy.shape[1] != batch_frames_numpy.shape[1]) or (
                    batch_depth_numpy.shape[2] != batch_frames_numpy.shape[2]
                ):
                    target_h = batch_frames_numpy.shape[1]
                    target_w = batch_frames_numpy.shape[2]
                    # INTER_AREA seems best for downscaling depth maps (minimizes aliasing/jaggies-cv2.INTER_LANCZOS4 works too).
                    if (
                        target_w < batch_depth_numpy.shape[2]
                        and target_h < batch_depth_numpy.shape[1]
                    ):
                        interp = cv2.INTER_AREA
                    else:
                        interp = cv2.INTER_LINEAR
                    resized_depth = np.empty(
                        (batch_depth_numpy.shape[0], target_h, target_w),
                        dtype=batch_depth_numpy.dtype,
                    )
                    for _di in range(batch_depth_numpy.shape[0]):
                        resized_depth[_di] = cv2.resize(
                            batch_depth_numpy[_di],
                            (target_w, target_h),
                            interpolation=interp,
                        )
                    batch_depth_numpy = resized_depth

                # self._save_debug_numpy(batch_depth_numpy, "02_PROCESSED_PRE_NORM", i, file_frame_idx, task_name)

                batch_frames_float = batch_frames_numpy.astype("float32") / 255.0

                # 1. Normalize based on mode
                if assume_raw_input:
                    if global_depth_max > 1.0:
                        batch_depth_normalized = batch_depth_numpy / global_depth_max
                    else:
                        batch_depth_normalized = batch_depth_numpy.copy()
                else:
                    depth_range = global_depth_max - global_depth_min
                    if depth_range > 1e-5:
                        batch_depth_normalized = (
                            batch_depth_numpy - global_depth_min
                        ) / depth_range
                    else:
                        batch_depth_normalized = np.full_like(
                            batch_depth_numpy,
                            fill_value=zero_disparity_anchor_val,
                            dtype=np.float32,
                        )
                        logger.warning(
                            f"Normalization collapsed to zero range ({global_depth_min:.4f} - {global_depth_max:.4f})."
                        )

                # 2. Unified Inverted Gamma (Matches Preview Math exactly)
                if round(float(depth_gamma), 2) != 1.0:
                    # Math: 1.0 - (1.0 - depth)^gamma
                    batch_depth_normalized = 1.0 - np.power(
                        1.0 - np.clip(batch_depth_normalized, 0, 1), depth_gamma
                    )
                    # (Normal) Depth gamma applied in normalized space; no warning log.

                batch_depth_normalized = np.clip(batch_depth_normalized, 0, 1)

                # --- START OF DIAGNOSTIC SYNC BLOCK ---
                # Keeps depth visualization at 16-bit to prevent banding in the Render output
                batch_depth_vis_list = []
                for d_frame in batch_depth_normalized:
                    # We keep this as float32 for the assembly to prevent 8-bit quantization
                    vis_gray_3ch = np.stack([d_frame] * 3, axis=-1)
                    batch_depth_vis_list.append(vis_gray_3ch.astype("float32"))

                batch_depth_vis = np.stack(batch_depth_vis_list, axis=0)
                # --- END OF DIAGNOSTIC SYNC BLOCK ---

                t_end_preproc = time.perf_counter()
                # --- END TIMER 2 ---

                # --- TIMER 3: HtoD Transfer (CPU to GPU) ---
                t_start_transfer_HtoD = time.perf_counter()

                left_video_tensor = (
                    torch.from_numpy(batch_frames_numpy)
                    .permute(0, 3, 1, 2)
                    .float()
                    .cuda()
                    / 255.0
                )
                disp_map_tensor = (
                    torch.from_numpy(batch_depth_normalized).unsqueeze(1).float().cuda()
                )
                disp_map_tensor = (disp_map_tensor - zero_disparity_anchor_val) * 2.0
                disp_map_tensor = disp_map_tensor * max_disp

                torch.cuda.synchronize()  # Force synchronization before compute
                t_end_transfer_HtoD = time.perf_counter()
                # --- END TIMER 3 ---

                # --- TIMER 4: GPU Compute (Core Splatting) ---
                t_start_compute = time.perf_counter()

                with torch.no_grad():
                    right_video_tensor_raw, occlusion_mask_tensor = stereo_projector(
                        left_video_tensor, disp_map_tensor
                    )
                    right_video_tensor = right_video_tensor_raw

                right_video_numpy = right_video_tensor.cpu().permute(0, 2, 3, 1).numpy()
                occlusion_mask_numpy = (
                    occlusion_mask_tensor.cpu()
                    .permute(0, 2, 3, 1)
                    .numpy()
                    .repeat(3, axis=-1)
                )

                t_end_transfer_DtoH = time.perf_counter()
                # --- END TIMER 5 ---

                # --- TIMER 6: FFmpeg Write (Blocking I/O) ---
                t_start_write = time.perf_counter()

                # --- START OF DIAGNOSTIC TEST ASSEMBLY ---
                test_generated_this_task = False
                if self.map_test_var.get() or self.splat_test_var.get():
                    for j in range(len(batch_frames_numpy)):
                        # Capture logic
                        clean_base = (
                            os.path.splitext(output_video_path_base)[0]
                            .replace("_splatted2", "")
                            .replace("_splatted4", "")
                        )
                        base_img_path = f"{clean_base}_{width}"

                        # Emergency pump to ensure the 100% scale update_preview finished
                        for _ in range(25):
                            self.update_idletasks()
                            self.update()
                            if self.previewer.pil_image_for_preview is not None:
                                break
                            time.sleep(0.1)

                        if self.previewer.pil_image_for_preview is None:
                            logger.error(
                                "Diagnostic Capture Failed: Preview image is missing."
                            )
                            return False

                        preview_pil = self.previewer.pil_image_for_preview.convert(
                            "RGB"
                        )
                        # Resize preview to match RENDER resolution for SBS parity
                        preview_pil = preview_pil.resize(
                            (width, height), Image.Resampling.LANCZOS
                        )
                        preview_side = np.array(preview_pil).astype(np.float32) / 255.0

                        if self.map_test_var.get():
                            # Render depth visualization (apply the same visualization-only TV->full expansion for 10-bit, when tagged 'tv')
                            render_map_vis = batch_depth_vis[j]
                            try:
                                if DEPTH_VIS_APPLY_TV_RANGE_EXPANSION_10BIT:
                                    bd = (
                                        int(_infer_depth_bit_depth(depth_stream_info))
                                        if depth_stream_info
                                        else 8
                                    )
                                    rng = str(
                                        (depth_stream_info or {}).get("color_range")
                                        or (depth_stream_info or {}).get("range")
                                        or ""
                                    ).lower()
                                    if bd >= 10 and rng == "tv":
                                        render_map_vis = (
                                            render_map_vis - DEPTH_VIS_TV10_BLACK_NORM
                                        ) / (
                                            DEPTH_VIS_TV10_WHITE_NORM
                                            - DEPTH_VIS_TV10_BLACK_NORM
                                        )
                            except Exception:
                                pass

                            # Save labeled PREVIEW / RENDER single images
                            try:
                                prev_u8 = (np.clip(preview_side, 0, 1) * 255).astype(
                                    np.uint8
                                )
                                prev_bgr = cv2.cvtColor(prev_u8, cv2.COLOR_RGB2BGR)
                                cv2.putText(
                                    prev_bgr,
                                    "PREVIEW",
                                    (10, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2,
                                    (0, 0, 255),
                                    2,
                                )
                                cv2.imwrite(
                                    f"{base_img_path}_map_test_preview.png", prev_bgr
                                )

                                rend_u8 = (np.clip(render_map_vis, 0, 1) * 255).astype(
                                    np.uint8
                                )
                                rend_bgr = cv2.cvtColor(rend_u8, cv2.COLOR_RGB2BGR)
                                cv2.putText(
                                    rend_bgr,
                                    "RENDER",
                                    (10, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2,
                                    (0, 0, 255),
                                    2,
                                )
                                cv2.imwrite(
                                    f"{base_img_path}_map_test_render.png", rend_bgr
                                )
                            except Exception:
                                pass

                            # SBS (kept for convenience)
                            sbs_map = np.concatenate(
                                [preview_side, render_map_vis], axis=1
                            )
                            sbs_map_uint8 = (np.clip(sbs_map, 0, 1) * 255).astype(
                                np.uint8
                            )
                            sbs_map_bgr = cv2.cvtColor(sbs_map_uint8, cv2.COLOR_RGB2BGR)
                            cv2.putText(
                                sbs_map_bgr,
                                "PREVIEW",
                                (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0, 0, 255),
                                2,
                            )
                            cv2.putText(
                                sbs_map_bgr,
                                "RENDER",
                                (width + 10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0, 0, 255),
                                2,
                            )
                            cv2.imwrite(f"{base_img_path}_map_test.png", sbs_map_bgr)
                            logger.info(
                                f"Saved Map Test: {os.path.basename(base_img_path)}_map_test.png"
                            )

                        if self.splat_test_var.get():
                            # Save labeled PREVIEW / RENDER single images
                            try:
                                prev_u8 = (np.clip(preview_side, 0, 1) * 255).astype(
                                    np.uint8
                                )
                                prev_bgr = cv2.cvtColor(prev_u8, cv2.COLOR_RGB2BGR)
                                cv2.putText(
                                    prev_bgr,
                                    "PREVIEW",
                                    (10, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2,
                                    (0, 0, 255),
                                    2,
                                )
                                cv2.imwrite(
                                    f"{base_img_path}_splat_test_preview.png", prev_bgr
                                )

                                rend_u8 = (
                                    np.clip(right_video_numpy[j], 0, 1) * 255
                                ).astype(np.uint8)
                                rend_bgr = cv2.cvtColor(rend_u8, cv2.COLOR_RGB2BGR)
                                cv2.putText(
                                    rend_bgr,
                                    "RENDER",
                                    (10, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2,
                                    (0, 0, 255),
                                    2,
                                )
                                cv2.imwrite(
                                    f"{base_img_path}_splat_test_render.png", rend_bgr
                                )
                            except Exception:
                                pass

                            # SBS (kept for convenience)
                            sbs_splat = np.concatenate(
                                [preview_side, right_video_numpy[j]], axis=1
                            )
                            sbs_splat_uint8 = (np.clip(sbs_splat, 0, 1) * 255).astype(
                                np.uint8
                            )
                            sbs_splat_bgr = cv2.cvtColor(
                                sbs_splat_uint8, cv2.COLOR_RGB2BGR
                            )
                            cv2.putText(
                                sbs_splat_bgr,
                                "PREVIEW",
                                (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0, 0, 255),
                                2,
                            )
                            cv2.putText(
                                sbs_splat_bgr,
                                "RENDER",
                                (width + 10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0, 0, 255),
                                2,
                            )
                            cv2.imwrite(
                                f"{base_img_path}_splat_test.png", sbs_splat_bgr
                            )
                            logger.info(
                                f"Saved Splat Test: {os.path.basename(base_img_path)}_splat_test.png"
                            )

                        test_generated_this_task = True
                        break  # Only process one frame for images

                    if test_generated_this_task:
                        return True  # Exit early: NO VIDEO encoding occurs

                # --- STANDARD VIDEO ENCODING (Only reached if NO tests enabled) ---
                for j in range(len(batch_frames_numpy)):
                    if dual_output:
                        video_grid = np.concatenate(
                            [occlusion_mask_numpy[j], right_video_numpy[j]], axis=1
                        )
                    else:
                        video_grid_top = np.concatenate(
                            [batch_frames_float[j], batch_depth_vis[j]], axis=1
                        )
                        video_grid_bottom = np.concatenate(
                            [occlusion_mask_numpy[j], right_video_numpy[j]], axis=1
                        )
                        video_grid = np.concatenate(
                            [video_grid_top, video_grid_bottom], axis=0
                        )

                    video_grid_uint16 = (
                        np.clip(video_grid, 0.0, 1.0) * 65535.0
                    ).astype(np.uint16)
                    ffmpeg_process.stdin.write(
                        cv2.cvtColor(video_grid_uint16, cv2.COLOR_RGB2BGR).tobytes()
                    )
                    frame_count += 1
                # --- END OF FRAME ASSEMBLY ---

                t_end_write = time.perf_counter()
                # --- END TIMER 6 ---

                del (
                    left_video_tensor,
                    disp_map_tensor,
                    right_video_tensor,
                    occlusion_mask_tensor,
                )
                torch.cuda.empty_cache()
                draw_progress_bar(frame_count, num_frames, prefix=f"  Encoding:")

                t_end_batch = time.perf_counter()  # <--- TIMER END: Total Batch

                # --- LOG RESULTS: Conditionally log at DEBUG level ---
                if logger.isEnabledFor(logging.DEBUG):
                    batch_size_actual = len(current_frame_indices)
                    task_tag = "LowRes" if is_low_res_task else "HiRes"

                    io_time = t_end_io - t_start_io
                    preproc_time = t_end_preproc - t_start_preproc
                    htod_time = t_end_transfer_HtoD - t_start_transfer_HtoD
                    compute_time = t_end_compute - t_start_compute
                    dtoh_time = t_end_transfer_DtoH - t_start_transfer_DtoH
                    write_time = t_end_write - t_start_write
                    total_batch_time = t_end_batch - t_start_batch

                    logger.info(
                        f"[{task_tag} Batch {i // batch_size_actual + 1}] Frames={batch_size_actual} Total={total_batch_time * 1000:.0f}ms | "
                        f"IO={io_time * 1000:.0f}ms | CPU_Proc={preproc_time * 1000:.0f}ms | HtoD={htod_time * 1000:.0f}ms | "
                        f"GPU_Comp={compute_time * 1000:.0f}ms | DtoH={dtoh_time * 1000:.0f}ms | FFmpeg_Write={write_time * 1000:.0f}ms"
                    )
                # --- END LOG RESULTS ---

        except (IOError, BrokenPipeError) as e:
            logger.error(f"FFmpeg pipe error: {e}. Encoding may have failed.")
            encoding_successful = False
        finally:
            del stereo_projector
            torch.cuda.empty_cache()
            gc.collect()

            # --- Finalize FFmpeg process ---
            if ffmpeg_process is not None:
                if ffmpeg_process.stdin:
                    try:
                        ffmpeg_process.stdin.close()  # Close the pipe to signal end of input
                    finally:
                        ffmpeg_process.stdin = None   # <-- IMPORTANT: prevent communicate() flushing closed stdin

                # Wait for the process to finish and get output
                stdout, stderr = ffmpeg_process.communicate(timeout=120)

                if self.stop_event.is_set():
                    ffmpeg_process.terminate()
                    logger.warning(
                        f"FFmpeg encoding stopped by user for {os.path.basename(final_output_video_path)}."
                    )
                    encoding_successful = False
                elif ffmpeg_process.returncode != 0:
                    logger.error(
                        f"FFmpeg encoding failed for {os.path.basename(final_output_video_path)} (return code {ffmpeg_process.returncode}):\n{stderr.decode()}"
                    )
                    encoding_successful = False
                else:
                    logger.info(
                        f"Successfully encoded video to {final_output_video_path}"
                    )
                    logger.debug(f"FFmpeg stderr log:\n{stderr.decode()}")
            else:
                # Test mode: no encoding step was started, so nothing to finalize.
                pass

        if not encoding_successful:
            return False

        # --- Check for Low-Res Task BEFORE writing sidecar ---
        if is_low_res_task:
            # --- Write sidecar JSON after successful encoding ---
            output_sidecar_data = {}

            # Check and include frame_overlap and input_bias
            has_non_zero_setting = False

            if input_bias is not None and input_bias != 0.0:
                output_sidecar_data["input_bias"] = input_bias
                has_non_zero_setting = True

            # Use the combined condition: non-zero setting AND is low-res
            if has_non_zero_setting:
                sidecar_ext = self.APP_CONFIG_DEFAULTS.get(
                    "OUTPUT_SIDECAR_EXT", ".spsidecar"
                )
                output_sidecar_path = (
                    f"{os.path.splitext(final_output_video_path)[0]}{sidecar_ext}"
                )
                try:
                    with open(output_sidecar_path, "w", encoding="utf-8") as f:
                        json.dump(output_sidecar_data, f, indent=4)
                    logger.info(f"Created output sidecar file: {output_sidecar_path}")
                except Exception as e:
                    logger.error(
                        f"Error creating output sidecar file '{output_sidecar_path}': {e}"
                    )
            else:
                logger.debug(
                    "Skipping output sidecar creation: frame_overlap and input_bias are zero."
                )
        else:
            logger.debug(
                "Skipping output sidecar creation: High-resolution output does not require spsidecar."
            )

        return True

    def _determine_auto_convergence(
        self,
        depth_map_path: str,
        total_frames_to_process: int,
        batch_size: int,
        fallback_value: float,
    ) -> Tuple[float, float]:
        """
        Calculates the Auto Convergence points for the entire video (Average and Peak)
        in a single pass.

        Args:
            fallback_value (float): The current GUI/Sidecar value to return if auto-convergence fails.

        Returns:
            Tuple[float, float]: (new_anchor_avg: 0.0-1.0, new_anchor_peak: 0.0-1.0).
                                 Returns (fallback_value, fallback_value) if the process fails.
        """
        logger.info(
            "==> Starting Auto-Convergence pre-pass to determine global average and peak depth."
        )

        # --- Constants for Auto-Convergence Logic ---
        BLUR_KERNEL_SIZE = 9
        CENTER_CROP_PERCENT = 0.75
        MIN_VALID_PIXELS = 5
        # The offset is only applied at the end for the 'Average' mode.
        INTERNAL_ANCHOR_OFFSET = 0.1
        # -------------------------------------------

        all_valid_frame_values = []
        fallback_tuple = (fallback_value, fallback_value)  # Value to return on failure

        try:
            # 1. Initialize Decord Reader (No target height/width needed, raw is fine)
            depth_reader = VideoReader(depth_map_path, ctx=cpu(0))
            if len(depth_reader) == 0:
                logger.error(
                    "Depth map reader has no frames. Cannot calculate Auto-Convergence."
                )
                return fallback_tuple
        except Exception as e:
            logger.error(
                f"Error initializing depth map reader for Auto-Convergence: {e}"
            )
            return fallback_tuple

        # 2. Iterate and Collect Data

        video_length = len(depth_reader)
        if total_frames_to_process <= 0 or total_frames_to_process > video_length:
            num_frames = video_length
        else:
            num_frames = total_frames_to_process

        logger.debug(
            f"  AutoConv determined actual frames to process: {num_frames} (from input length {total_frames_to_process})."
        )

        for i in range(0, num_frames, batch_size):
            if self.stop_event.is_set():
                logger.warning("Auto-Convergence pre-pass stopped by user.")
                return fallback_tuple

            current_frame_indices = list(range(i, min(i + batch_size, num_frames)))
            if not current_frame_indices:
                break

            # CRITICAL FIX: Ensure seeking/reading works
            try:
                depth_reader.seek(current_frame_indices[0])
                batch_depth_numpy_raw = depth_reader.get_batch(
                    current_frame_indices
                ).asnumpy()
            except Exception as e:
                logger.error(
                    f"Error seeking/reading depth map batch starting at index {i}: {e}. Skipping batch."
                )
                continue

            # Process depth frames (Grayscale, Float conversion)
            if batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 3:
                batch_depth_numpy = batch_depth_numpy_raw.mean(axis=-1)
            elif (
                batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 1
            ):
                batch_depth_numpy = batch_depth_numpy_raw.squeeze(-1)
            else:
                batch_depth_numpy = batch_depth_numpy_raw

            batch_depth_float = batch_depth_numpy.astype(np.float32)

            # Get chunk min/max for normalization (using the chunk's range)
            min_val = batch_depth_float.min()
            max_val = batch_depth_float.max()

            if max_val - min_val > 1e-5:
                batch_depth_normalized = (batch_depth_float - min_val) / (
                    max_val - min_val
                )
            else:
                batch_depth_normalized = np.full_like(
                    batch_depth_float, fill_value=0.5, dtype=np.float32
                )

            # Frame-by-Frame Processing (Blur & Crop)
            for j, frame in enumerate(batch_depth_normalized):
                current_frame_idx = current_frame_indices[j]
                H, W = frame.shape

                # a) Blur
                frame_blurred = cv2.GaussianBlur(
                    frame, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0
                )

                # b) Center Crop (75% of H and W)
                margin_h = int(H * (1 - CENTER_CROP_PERCENT) / 2)
                margin_w = int(W * (1 - CENTER_CROP_PERCENT) / 2)

                cropped_frame = frame_blurred[
                    margin_h : H - margin_h, margin_w : W - margin_w
                ]

                # c) Average (Exclude true black/white pixels (0.0 or 1.0) which may be background/edges)
                valid_pixels = cropped_frame[
                    (cropped_frame > 0.001) & (cropped_frame < 0.999)
                ]

                if valid_pixels.size > MIN_VALID_PIXELS:
                    # Append the mean of the valid pixels for this frame
                    all_valid_frame_values.append(valid_pixels.mean())
                else:
                    # FALLBACK FOR THE FRAME: Use the mean of the WHOLE cropped, blurred frame
                    all_valid_frame_values.append(cropped_frame.mean())
                    logger.warning(
                        f"  [AutoConv Frame {current_frame_idx:03d}] SKIPPED: Valid pixel count ({valid_pixels.size}) below threshold ({MIN_VALID_PIXELS}). Forcing mean from full cropped frame."
                    )

            draw_progress_bar(
                i + len(current_frame_indices),
                num_frames,
                prefix="  Auto-Conv Pre-Pass:",
            )

        # 3. Final Temporal Calculations
        if all_valid_frame_values:
            valid_values_np = np.array(all_valid_frame_values)

            # Calculate final RAW values (Temporal Mean and Temporal Max)
            raw_anchor_avg = np.mean(valid_values_np)
            raw_anchor_peak = np.max(valid_values_np)

            # Apply Offset only for Average mode
            final_anchor_avg_offset = raw_anchor_avg + INTERNAL_ANCHOR_OFFSET

            # Clamp to the valid range [0.0, 1.0]
            final_anchor_avg = np.clip(final_anchor_avg_offset, 0.0, 1.0)
            final_anchor_peak = np.clip(raw_anchor_peak, 0.0, 1.0)

            logger.info(
                f"\n==> Auto-Convergence Calculated: Avg={raw_anchor_avg:.4f} + Offset ({INTERNAL_ANCHOR_OFFSET:.2f}) = Final Avg {final_anchor_avg:.4f}"
            )
            logger.info(
                f"==> Auto-Convergence Calculated: Peak={raw_anchor_peak:.4f} = Final Peak {final_anchor_peak:.4f}"
            )

            # Return both calculated values
            return float(final_anchor_avg), float(final_anchor_peak)
        else:
            logger.warning(
                "\n==> Auto-Convergence failed: No valid frames found. Using fallback value."
            )
            return fallback_tuple

    def exit_app(self):
        """Handles application exit, including stopping the processing thread."""
        self._save_config()
        self.stop_event.set()
        if self.processing_thread and self.processing_thread.is_alive():
            logger.info("==> Waiting for processing thread to finish...")
            # --- NEW: Cleanup previewer resources ---
            if hasattr(self, "previewer"):
                self.previewer.cleanup()
            self.processing_thread.join(timeout=5.0)
            if self.processing_thread.is_alive():
                logger.debug("==> Thread did not terminate gracefully within timeout.")
        release_cuda_memory()
        self.destroy()

    def _find_preview_sources_callback(self) -> list:
        """
        Callback for VideoPreviewer. Scans for matching source video and depth map pairs.
        Handles both folder (batch) and file (single) input modes.
        """
        source_path = self.input_source_clips_var.get()
        depth_raw_path = self.input_depth_maps_var.get()

        if not source_path or not depth_raw_path:
            logger.warning("Preview Scan Failed: Source or depth path is empty.")
            return []

        # ------------------------------------------------------------
        # 1) SINGLE-FILE MODE (both are actual files)
        # ------------------------------------------------------------
        is_source_file = os.path.isfile(source_path)
        is_depth_file = os.path.isfile(depth_raw_path)

        if is_source_file and is_depth_file:
            logger.debug(
                f"Preview Scan: Single file mode detected. "
                f"Source: {source_path}, Depth: {depth_raw_path}"
            )
            return [
                {
                    "source_video": source_path,
                    "depth_map": depth_raw_path,
                }
            ]

        # ------------------------------------------------------------
        # 2) FOLDER / BATCH MODE
        # ------------------------------------------------------------
        if not os.path.isdir(source_path) or not os.path.isdir(depth_raw_path):
            logger.error(
                "Preview Scan Failed: Inputs must either be two files or two valid directories."
            )
            return []

        source_folder = source_path
        base_depth_folder = depth_raw_path

        # Collect all source videos
        video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
        source_videos = []
        for ext in video_extensions:
            source_videos.extend(glob.glob(os.path.join(source_folder, ext)))

        if not source_videos:
            logger.warning(f"No source videos found in folder: {source_folder}")
            return []

        video_source_list = []

        # ------------------------------------------------------------
        # 2A) MULTI-MAP PREVIEW: search all map subfolders
        # ------------------------------------------------------------
        if self.multi_map_var.get():
            depth_candidate_folders = []

            # Treat each subdirectory (except 'sidecars') as a map folder
            try:
                for entry in os.listdir(base_depth_folder):
                    full_sub = os.path.join(base_depth_folder, entry)
                    if os.path.isdir(full_sub) and entry.lower() != "sidecars":
                        depth_candidate_folders.append(full_sub)
            except FileNotFoundError:
                logger.error(
                    f"Preview Scan Failed: Depth folder not found: {base_depth_folder}"
                )
                return []

            if not depth_candidate_folders:
                logger.warning(
                    f"Preview Scan: No map subfolders found in Multi-Map base folder: {base_depth_folder}"
                )

            for video_path in sorted(source_videos):
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                matched = False

                for dpath in depth_candidate_folders:
                    mp4 = os.path.join(dpath, f"{base_name}_depth.mp4")
                    npz = os.path.join(dpath, f"{base_name}_depth.npz")

                    if os.path.exists(mp4):
                        video_source_list.append(
                            {
                                "source_video": video_path,
                                "depth_map": mp4,
                            }
                        )
                        matched = True
                        break
                    elif os.path.exists(npz):
                        video_source_list.append(
                            {
                                "source_video": video_path,
                                "depth_map": npz,
                            }
                        )
                        matched = True
                        break

                if not matched:
                    logger.debug(
                        f"Preview Scan: No depth map found in any map folder for '{base_name}'."
                    )

        # ------------------------------------------------------------
        # 2B) NORMAL MODE PREVIEW: single depth folder
        # ------------------------------------------------------------
        else:
            depth_folder = base_depth_folder

            for video_path in sorted(source_videos):
                base_name = os.path.splitext(os.path.basename(video_path))[0]

                candidates = [
                    os.path.join(depth_folder, f"{base_name}_depth.mp4"),
                    os.path.join(depth_folder, f"{base_name}_depth.npz"),
                    os.path.join(depth_folder, f"{base_name}.mp4"),
                    os.path.join(depth_folder, f"{base_name}.npz"),
                ]

                matching_depth_path = None
                for dp in candidates:
                    if os.path.exists(dp):
                        matching_depth_path = dp
                        break

                if matching_depth_path:
                    logger.debug(f"Preview Scan: Found pair for '{base_name}'.")
                    video_source_list.append(
                        {
                            "source_video": video_path,
                            "depth_map": matching_depth_path,
                        }
                    )

        if not video_source_list:
            logger.warning("Preview Scan: No matching source/depth pairs found.")
        else:
            logger.info(
                f"Preview Scan: Found {len(video_source_list)} matching source/depth pairs."
            )

        # Ensure preview-only overlay toggles are applied on Refresh/Reload as well.
        self._apply_preview_overlay_toggles()
        return video_source_list

    def _get_current_config(self):
        """Collects all current GUI variable values into a single dictionary."""
        config = {
            # Folder Configurations
            "input_source_clips": self.input_source_clips_var.get(),
            "input_depth_maps": self.input_depth_maps_var.get(),
            "output_splatted": self.output_splatted_var.get(),
            "dark_mode_enabled": self.dark_mode_var.get(),
            "window_width": self.winfo_width(),
            "window_height": self.winfo_height(),
            "window_x": self.winfo_x(),
            "window_y": self.winfo_y(),
            "update_slider_from_sidecar": self.update_slider_from_sidecar_var.get(),
            "auto_save_sidecar": self.auto_save_sidecar_var.get(),
            "multi_map_enabled": self.multi_map_var.get(),
            "skip_lowres_preproc": self.skip_lowres_preproc_var.get(),
            "crosshair_enabled": self.crosshair_enabled_var.get(),
            "crosshair_white": self.crosshair_white_var.get(),
            "crosshair_multi": self.crosshair_multi_var.get(),
            "depth_pop_enabled": self.depth_pop_enabled_var.get(),
            "debug_logging_enabled": self.debug_logging_var.get(),
            "loop_playback": bool(
                getattr(
                    getattr(self, "previewer", None),
                    "loop_playback_var",
                    tk.BooleanVar(value=False),
                ).get()
            ),
            "enable_full_resolution": self.enable_full_res_var.get(),
            "batch_size": self.batch_size_var.get(),
            "enable_low_resolution": self.enable_low_res_var.get(),
            "pre_res_width": self.pre_res_width_var.get(),
            "pre_res_height": self.pre_res_height_var.get(),
            "low_res_batch_size": self.low_res_batch_size_var.get(),
            "depth_dilate_size_x": self.depth_dilate_size_x_var.get(),
            "depth_dilate_size_y": self.depth_dilate_size_y_var.get(),
            "depth_blur_size_x": self.depth_blur_size_x_var.get(),
            "depth_blur_size_y": self.depth_blur_size_y_var.get(),
            "depth_dilate_left": self.depth_dilate_left_var.get(),
            "depth_blur_left": self.depth_blur_left_var.get(),
            "depth_blur_left_mix": self.depth_blur_left_mix_var.get(),
            "preview_size": self.preview_size_var.get(),
            "preview_source": self.preview_source_var.get(),
            "process_length": self.process_length_var.get(),
            "output_crf": self.output_crf_full_var.get(),  # legacy
            "output_crf_full": self.output_crf_full_var.get(),
            "output_crf_low": self.output_crf_low_var.get(),
            "color_tags_mode": getattr(
                self, "color_tags_mode_var", tk.StringVar(value="Auto")
            ).get(),
            "dual_output": self.dual_output_var.get(),
            "auto_convergence_mode": self.auto_convergence_mode_var.get(),
            "depth_gamma": self.depth_gamma_var.get(),
            "max_disp": self.max_disp_var.get(),
            "convergence_point": self.zero_disparity_anchor_var.get(),
            "enable_global_norm": self.enable_global_norm_var.get(),  # Renamed
            "move_to_finished": self.move_to_finished_var.get(),
        }
        # Avoid persisting test-forced preview settings
        try:
            test_active = bool(self.splat_test_var.get()) or bool(
                self.map_test_var.get()
            )
        except Exception:
            test_active = False
        if test_active:
            config["preview_source"] = self.app_config.get(
                "preview_source", "Splat Result"
            )
            config["preview_size"] = self.app_config.get("preview_size", "75%")

        return config

    def get_current_preview_settings(self) -> dict:
        """Gathers settings from the GUI needed for the preview callback."""
        try:
            settings = {
                "max_disp": float(self.max_disp_var.get()),
                "convergence_point": float(self.zero_disparity_anchor_var.get()),
                "depth_gamma": float(self.depth_gamma_var.get()),
                "depth_dilate_size_x": self._safe_float(self.depth_dilate_size_x_var),
                "depth_dilate_size_y": self._safe_float(self.depth_dilate_size_y_var),
                "depth_blur_size_x": self._safe_float(self.depth_blur_size_x_var),
                "depth_blur_size_y": self._safe_float(self.depth_blur_size_y_var),
                "depth_dilate_left": self._safe_float(self.depth_dilate_left_var),
                "depth_blur_left": self._safe_float(self.depth_blur_left_var),
                "depth_blur_left_mix": self._safe_float(self.depth_blur_left_mix_var),
                "preview_size": self.preview_size_var.get(),
                "preview_source": self.preview_source_var.get(),
                "enable_global_norm": self.enable_global_norm_var.get(),
            }

            # Resolve Border Percentages based on Mode
            mode = self.border_mode_var.get()
            l_pct, r_pct = 0.0, 0.0

            if mode == "Auto Basic":
                # uses Trace-updated border_width_var
                w = self._safe_float(self.border_width_var)
                l_pct = w
                r_pct = w
            elif mode == "Auto Adv.":
                l_pct = self._safe_float(self.auto_border_L_var)
                r_pct = self._safe_float(self.auto_border_R_var)
            elif mode == "Manual":
                w = self._safe_float(self.border_width_var)
                b = self._safe_float(self.border_bias_var)
                if b <= 0:
                    l_pct = w
                    r_pct = w * (1.0 + b)
                else:
                    r_pct = w
                    l_pct = w * (1.0 - b)
            # Off mode stays 0.0, 0.0

            settings["left_border_pct"] = l_pct
            settings["right_border_pct"] = r_pct

            return settings

        except (ValueError, tk.TclError) as e:
            logger.error(f"Invalid preview setting value: {e}")
            return None

    def _get_current_sidecar_paths_and_data(self) -> Optional[Tuple[str, str, dict]]:
        """Helper to get current file path, sidecar path, and existing data (merged with defaults)."""
        if (
            not hasattr(self, "previewer")
            or not self.previewer.video_list
            or self.previewer.current_video_index == -1
        ):
            return None

        current_index = self.previewer.current_video_index
        depth_map_path = self.previewer.video_list[current_index].get("depth_map")

        if not depth_map_path:
            return None

        depth_map_basename = os.path.splitext(os.path.basename(depth_map_path))[0]
        sidecar_ext = self.APP_CONFIG_DEFAULTS["SIDECAR_EXT"]
        # Use base folder for sidecars when Multi-Map is enabled
        sidecar_folder = self._get_sidecar_base_folder()
        json_sidecar_path = os.path.join(
            sidecar_folder, f"{depth_map_basename}{sidecar_ext}"
        )

        # Load existing data (merged with defaults) to preserve non-GUI parameters like overlap/bias
        current_data = self.sidecar_manager.load_sidecar_data(json_sidecar_path)

        return json_sidecar_path, depth_map_path, current_data

    def _get_defined_tasks(self, settings):
        """Helper to return a list of processing tasks based on GUI settings."""
        processing_tasks = []
        if settings["enable_full_resolution"]:
            processing_tasks.append(
                {
                    "name": "Full-Resolution",
                    "output_subdir": "hires",
                    "set_pre_res": False,
                    "target_width": -1,
                    "target_height": -1,
                    "batch_size": settings["full_res_batch_size"],
                    "is_low_res": False,
                }
            )
        if settings["enable_low_resolution"]:
            processing_tasks.append(
                {
                    "name": "Low-Resolution",
                    "output_subdir": "lowres",
                    "set_pre_res": True,
                    "target_width": settings["low_res_width"],
                    "target_height": settings["low_res_height"],
                    "batch_size": settings["low_res_batch_size"],
                    "is_low_res": True,
                }
            )
        return processing_tasks

    def _get_video_specific_settings(
        self,
        video_path,
        input_depth_maps_path_setting,
        default_zero_disparity_anchor,
        gui_max_disp,
        is_single_file_mode,
    ):
        """
        Determine the actual depth map path and video-specific settings.

        Behavior in Multi-Map mode:
          * If a sidecar exists for this video and contains 'selected_depth_map',
            that subfolder is used for the depth map lookup.
          * Otherwise, we fall back to the map selected in the GUI when Start was pressed.
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        base_name = video_name

        # ------------------------------------------------------------------
        # 1) Locate sidecar for this video (if any)
        # ------------------------------------------------------------------
        sidecar_ext = self.APP_CONFIG_DEFAULTS["SIDECAR_EXT"]
        sidecar_folder = self._get_sidecar_base_folder()
        json_sidecar_path = os.path.join(
            sidecar_folder, f"{video_name}_depth{sidecar_ext}"
        )

        merged_config = None
        sidecar_exists = False
        selected_map_for_video = None

        if os.path.exists(json_sidecar_path):
            try:
                merged_config = (
                    self.sidecar_manager.load_sidecar_data(json_sidecar_path) or {}
                )
                sidecar_exists = True
            except Exception as e:
                logger.error(f"Failed to load sidecar for {video_name}: {e}")
                merged_config = None

            if isinstance(merged_config, dict):
                selected_map_for_video = merged_config.get("selected_depth_map") or None

        # ------------------------------------------------------------------
        # 2) GUI defaults used when sidecar is missing or incomplete
        # ------------------------------------------------------------------
        gui_config = {
            "convergence_plane": float(default_zero_disparity_anchor),
            "max_disparity": float(gui_max_disp),
            "gamma": float(
                self.depth_gamma_var.get() or self.APP_CONFIG_DEFAULTS["DEPTH_GAMMA"]
            ),
        }

        # ------------------------------------------------------------------
        # 3) Resolve per-video depth map path
        # ------------------------------------------------------------------

        base_folder = input_depth_maps_path_setting
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        actual_depth_map_path = None

        # --- Single-file mode: depth path setting is the actual file ---
        if is_single_file_mode:
            # Here input_depth_maps_path_setting is expected to be the
            # depth map *file* path, not a directory.
            if os.path.isfile(base_folder):
                actual_depth_map_path = base_folder
                logger.info(
                    f"Single-file mode: using depth map file '{actual_depth_map_path}'"
                )
                # Info panel: in Multi-Map mode, still show the clip's sidecar-selected map (if any).
                _mm_map_info = "Direct file"
                if self.multi_map_var.get() and selected_map_for_video:
                    _mm_map_info = f"{selected_map_for_video} (Sidecar)"
                    logger.info(
                        f"[MM] USING sidecar map '{selected_map_for_video}' for '{video_name}'"
                    )
                self.progress_queue.put(("update_info", {"map": _mm_map_info}))
            else:
                return {
                    "error": (
                        f"Single-file mode: depth map file '{base_folder}' does not exist."
                    )
                }

        # --- Batch / folder mode ---
        else:
            #
            # MULTI-MAP MODE
            #
            if self.multi_map_var.get():
                # 1) First try sidecar’s selected map for this video
                sidecar_map = self._get_sidecar_selected_map_for_video(video_path)

                if sidecar_map:
                    candidate_dir = os.path.join(base_folder, sidecar_map)
                    c_mp4 = os.path.join(candidate_dir, f"{video_name}_depth.mp4")
                    c_npz = os.path.join(candidate_dir, f"{video_name}_depth.npz")

                    if os.path.exists(c_mp4):
                        actual_depth_map_path = c_mp4
                    elif os.path.exists(c_npz):
                        actual_depth_map_path = c_npz

                    if actual_depth_map_path:
                        logger.info(
                            f"[MM] USING sidecar map '{sidecar_map}' for '{video_name}'"
                        )
                        # Show map name PLUS source (Sidecar)
                        self.progress_queue.put(
                            ("update_info", {"map": f"{sidecar_map} (Sidecar)"})
                        )
                    else:
                        logger.warning(
                            f"[MM] sidecar map '{sidecar_map}' has no depth file for '{video_name}'"
                        )

                # 2) If sidecar FAILED, fall back to GUI-selected map
                if not actual_depth_map_path:
                    gui_map = self.selected_depth_map_var.get().strip()
                    if gui_map:
                        candidate_dir = os.path.join(base_folder, gui_map)
                        c_mp4 = os.path.join(candidate_dir, f"{video_name}_depth.mp4")
                        c_npz = os.path.join(candidate_dir, f"{video_name}_depth.npz")

                        if os.path.exists(c_mp4):
                            actual_depth_map_path = c_mp4
                        elif os.path.exists(c_npz):
                            actual_depth_map_path = c_npz

                        if actual_depth_map_path:
                            logger.info(
                                f"[MM] USING GUI map '{gui_map}' for '{video_name}'"
                            )
                            # Show map name PLUS source (GUI/Default)
                            self.progress_queue.put(
                                ("update_info", {"map": f"{gui_map} (GUI/Default)"})
                            )

                # 3) Absolute hard fallback: look in base folder
                if not actual_depth_map_path:
                    c_mp4 = os.path.join(base_folder, f"{video_name}_depth.mp4")
                    c_npz = os.path.join(base_folder, f"{video_name}_depth.npz")
                    if os.path.exists(c_mp4):
                        actual_depth_map_path = c_mp4
                    elif os.path.exists(c_npz):
                        actual_depth_map_path = c_npz

                if not actual_depth_map_path:
                    return {
                        "error": f"No depth map for '{video_name}' in ANY multimap source"
                    }

            #
            # NORMAL (non-multi-map) MODE
            #
            else:
                # Here base_folder is expected to be a directory containing all depth maps.
                c_mp4 = os.path.join(base_folder, f"{video_name}_depth.mp4")
                c_npz = os.path.join(base_folder, f"{video_name}_depth.npz")

                if os.path.exists(c_mp4):
                    actual_depth_map_path = c_mp4
                elif os.path.exists(c_npz):
                    actual_depth_map_path = c_npz
                else:
                    return {"error": f"No depth for '{video_name}' in '{base_folder}'"}

        actual_depth_map_path = os.path.normpath(actual_depth_map_path)

        # ------------------------------------------------------------
        # Multi-Map: resolve map folder from sidecar per-video
        # ------------------------------------------------------------
        depth_map_path = None

        if self.multi_map_var.get():
            # new helper we already added earlier
            selected_map = self._get_sidecar_selected_map_for_video(video_path)

            if selected_map:
                candidate_folder = os.path.join(
                    self.input_depth_maps_var.get(), selected_map
                )
                candidate_mp4 = os.path.join(candidate_folder, f"{base_name}_depth.mp4")
                candidate_npz = os.path.join(candidate_folder, f"{base_name}_depth.npz")

                if os.path.exists(candidate_mp4):
                    depth_map_path = candidate_mp4
                elif os.path.exists(candidate_npz):
                    depth_map_path = candidate_npz

        # ------------------------------------------------------------------
        # 4) Build merged settings (sidecar values with GUI defaults)
        # ------------------------------------------------------------------
        if not merged_config or not isinstance(merged_config, dict):
            merged_config = {
                "convergence_plane": gui_config["convergence_plane"],
                "max_disparity": gui_config["max_disparity"],
                "gamma": gui_config["gamma"],
                "input_bias": 0.0,
            }

        # Determine map source label for Multi-Map status display
        if self.multi_map_var.get():
            map_source = "Sidecar" if sidecar_exists else "GUI/Default"
        else:
            map_source = "N/A"

        # --- NEW: Determine Global Normalization Policy ---
        enable_global_norm_policy = self.enable_global_norm_var.get()
        if sidecar_exists:
            # Policy: If a sidecar exists, GN is DISABLED (manual mode)
            enable_global_norm_policy = False
            logger.debug(f"GN Policy: Sidecar exists for {video_name}. GN forced OFF.")

        # Determine the source for GN info
        gn_source = (
            "Sidecar"
            if sidecar_exists
            else ("GUI/ON" if enable_global_norm_policy else "GUI/OFF")
        )

        settings = {
            "actual_depth_map_path": actual_depth_map_path,
            "convergence_plane": merged_config.get(
                "convergence_plane", gui_config["convergence_plane"]
            ),
            "max_disparity_percentage": merged_config.get(
                "max_disparity", gui_config["max_disparity"]
            ),
            "input_bias": merged_config.get("input_bias"),
            "depth_gamma": merged_config.get("gamma", gui_config["gamma"]),
            # GUI-derived depth pre-processing settings
            "depth_dilate_size_x": float(self.depth_dilate_size_x_var.get()),
            "depth_dilate_size_y": float(self.depth_dilate_size_y_var.get()),
            "depth_blur_size_x": int(float(self.depth_blur_size_x_var.get())),
            "depth_blur_size_y": int(float(self.depth_blur_size_y_var.get())),
            # Left-edge depth pre-processing defaults (must behave like other GUI settings)
            "depth_dilate_left": float(self.depth_dilate_left_var.get()),
            "depth_blur_left": int(round(float(self.depth_blur_left_var.get()))),
            "depth_blur_left_mix": float(self.depth_blur_left_mix_var.get()),
            # Tracking / info sources
            "sidecar_found": sidecar_exists,
            "anchor_source": "Sidecar" if sidecar_exists else "GUI/Default",
            "max_disp_source": "Sidecar" if sidecar_exists else "GUI/Default",
            "gamma_source": "Sidecar" if sidecar_exists else "GUI/Default",
            "map_source": map_source,
            "enable_global_norm": enable_global_norm_policy,
            "gn_source": gn_source,
        }

        # If a sidecar exists and Sidecar Blur/Dilate is enabled, use its depth pre-processing values.
        # (Render should respect sidecars for both Single and Batch processing.)
        if (
            sidecar_exists
            and getattr(self, "enable_sidecar_blur_dilate_var", None)
            and self.enable_sidecar_blur_dilate_var.get()
        ):
            try:
                settings["depth_dilate_size_x"] = float(
                    merged_config.get(
                        "depth_dilate_size_x", settings["depth_dilate_size_x"]
                    )
                )
                settings["depth_dilate_size_y"] = float(
                    merged_config.get(
                        "depth_dilate_size_y", settings["depth_dilate_size_y"]
                    )
                )
                settings["depth_blur_size_x"] = int(
                    float(
                        merged_config.get(
                            "depth_blur_size_x", settings["depth_blur_size_x"]
                        )
                    )
                )
                settings["depth_blur_size_y"] = int(
                    float(
                        merged_config.get(
                            "depth_blur_size_y", settings["depth_blur_size_y"]
                        )
                    )
                )
                settings["depth_dilate_left"] = float(
                    merged_config.get(
                        "depth_dilate_left", settings["depth_dilate_left"]
                    )
                )
                settings["depth_blur_left"] = int(
                    float(
                        merged_config.get(
                            "depth_blur_left", settings["depth_blur_left"]
                        )
                    )
                )
                settings["depth_blur_left_mix"] = float(
                    merged_config.get(
                        "depth_blur_left_mix", settings["depth_blur_left_mix"]
                    )
                )
            except Exception:
                pass

        # If no sidecar file exists at all, enforce GUI values explicitly
        if not sidecar_exists:
            settings["convergence_plane"] = gui_config["convergence_plane"]
            settings["max_disparity_percentage"] = gui_config["max_disparity"]
            settings["depth_gamma"] = gui_config["gamma"]

        return settings

    def _initialize_video_and_depth_readers(
        self,
        video_path,
        actual_depth_map_path,
        process_length,
        task_settings,
        match_depth_res,
    ):
        """
        Initializes VideoReader objects for source video and depth map,
        and returns their metadata.
        Returns: (video_reader, depth_reader, processed_fps, original_vid_h, original_vid_w, current_processed_height, current_processed_width,
                  video_stream_info, total_frames_input, total_frames_depth, actual_depth_height, actual_depth_width,
                  depth_stream_info)
        """
        video_reader_input = None
        processed_fps = 0.0
        original_vid_h, original_vid_w = 0, 0
        current_processed_height, current_processed_width = 0, 0
        video_stream_info = None
        total_frames_input = 0

        depth_reader_input = None
        total_frames_depth = 0
        actual_depth_height, actual_depth_width = 0, 0
        depth_stream_info = None  # Initialize to None

        try:
            # 1. Initialize input video reader
            (
                video_reader_input,
                processed_fps,
                original_vid_h,
                original_vid_w,
                current_processed_height,
                current_processed_width,
                video_stream_info,
                total_frames_input,
            ) = read_video_frames(
                video_path,
                process_length,
                set_pre_res=task_settings["set_pre_res"],
                pre_res_width=task_settings["target_width"],
                pre_res_height=task_settings["target_height"],
            )
        except Exception as e:
            logger.error(
                f"==> Error initializing input video reader for {os.path.basename(video_path)} {task_settings['name']} pass: {e}. Skipping this pass."
            )
            return (
                None,
                None,
                0.0,
                0,
                0,
                0,
                0,
                None,
                0,
                0,
                0,
                0,
                None,
            )  # Return None for depth_stream_info
            # Determine map source for Multi-Map
            map_display = "N/A"
            if self.multi_map_var.get():
                if self._current_video_sidecar_map:
                    map_display = f"Sidecar > {self._current_video_sidecar_map}"
                elif self.selected_depth_map_var.get():
                    map_display = f"Default > {self.selected_depth_map_var.get()}"

        self.progress_queue.put(
            (
                "update_info",
                {
                    "resolution": f"{current_processed_width}x{current_processed_height}",
                    "frames": total_frames_input,
                },
            )
        )

        try:
            # 2. Initialize depth maps reader and capture depth_stream_info
            # For low-res rendering we need depth preprocessing parity with preview/full-res:
            # preprocess at the original clip resolution, then downscale to the low-res splat size.
            depth_target_h = current_processed_height
            depth_target_w = current_processed_width
            depth_match = match_depth_res
            if task_settings.get("is_low_res"):
                if (
                    getattr(self, "skip_lowres_preproc_var", None) is not None
                    and self.skip_lowres_preproc_var.get()
                ):
                    # Dev Tools: load depth directly at low-res and skip parity pre-proc path.
                    depth_target_h = current_processed_height
                    depth_target_w = current_processed_width
                    depth_match = match_depth_res
                else:
                    # Default: load depth at clip resolution so low-res pre-processing matches full-res preview/render.
                    depth_target_h = original_vid_h
                    depth_target_w = original_vid_w
                    depth_match = True

            (
                depth_reader_input,
                total_frames_depth,
                actual_depth_height,
                actual_depth_width,
                depth_stream_info,
            ) = load_pre_rendered_depth(
                actual_depth_map_path,
                process_length=process_length,
                target_height=depth_target_h,
                target_width=depth_target_w,
                match_resolution_to_target=depth_match,
            )
        except Exception as e:
            logger.error(
                f"==> Error initializing depth map reader for {os.path.basename(video_path)} {task_settings['name']} pass: {e}. Skipping this pass."
            )
            if video_reader_input:
                del video_reader_input
            return (
                None,
                None,
                0.0,
                0,
                0,
                0,
                0,
                None,
                0,
                0,
                0,
                0,
                None,
            )  # Return None for depth_stream_info

        # CRITICAL CHECK: Ensure input video and depth map have consistent frame counts
        if total_frames_input != total_frames_depth:
            logger.error(
                f"==> Frame count mismatch for {os.path.basename(video_path)} {task_settings['name']} pass: Input video has {total_frames_input} frames, Depth map has {total_frames_depth} frames. Skipping."
            )
            if video_reader_input:
                del video_reader_input
            if depth_reader_input:
                del depth_reader_input
            return (
                None,
                None,
                0.0,
                0,
                0,
                0,
                0,
                None,
                0,
                0,
                0,
                0,
                None,
            )  # Return None for depth_stream_info

        return (
            video_reader_input,
            depth_reader_input,
            processed_fps,
            original_vid_h,
            original_vid_w,
            current_processed_height,
            current_processed_width,
            video_stream_info,
            total_frames_input,
            total_frames_depth,
            actual_depth_height,
            actual_depth_width,
            depth_stream_info,
        )

    def _load_config(self):
        """Loads configuration from default config file."""
        config_filename = self.APP_CONFIG_DEFAULTS["DEFAULT_CONFIG_FILENAME"]
        # --- MODIFIED: Use the new dictionary constant ---
        if os.path.exists(config_filename):
            try:
                with open(config_filename, "r") as f:
                    self.app_config = json.load(f)

                # --- BACKWARD COMPATIBILITY FIX: Handle the old 'enable_autogain' key ---
                # Old meaning: True = Raw Input / Disable Normalization (GN OFF)
                # New meaning: True = Enable Global Normalization (GN ON)
                if "enable_autogain" in self.app_config:
                    old_value = self.app_config.pop("enable_autogain")  # Remove old key
                    # New value is the inverse of the old value
                    self.app_config["enable_global_norm"] = not bool(old_value)
                # --- END FIX ---
            except Exception as e:
                logger.error(f"Failed to load config file: {e}. Using defaults.")
                self.app_config = {}

    def _load_help_texts(self):
        """Loads help texts from a JSON file."""
        try:
            with open(
                os.path.join("dependency", "splatter_help.json"), "r", encoding="utf-8"
            ) as f:
                self.help_texts = json.load(f)
        except FileNotFoundError:
            logger.error(
                "Error: splatter_help.json not found. Tooltips will not be available."
            )
            self.help_texts = {}
        except json.JSONDecodeError:
            logger.error(
                "Error: Could not decode splatter_help.json. Check file format."
            )
            self.help_texts = {}

    def load_settings(self):
        """Loads settings from a user-selected JSON file."""
        filename = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Load Settings from File",
        )
        if not filename:
            return

        try:
            with open(filename, "r") as f:
                loaded_config = json.load(f)
            # Apply loaded config values to the variables
            for (
                config_key,
                config_value,
            ) in loaded_config.items():  # Iterate over loaded keys
                # --- NEW MAPPING LOGIC ---
                # Construct the expected name of the Tkinter variable
                tk_var_attr_name = config_key + "_var"

                if hasattr(self, tk_var_attr_name):
                    tk_var_object = getattr(self, tk_var_attr_name)

                    if isinstance(tk_var_object, tk.BooleanVar):
                        # Ensure value is converted to a proper boolean/int before setting BooleanVar
                        tk_var_object.set(bool(config_value))
                    elif isinstance(tk_var_object, tk.StringVar):
                        # Set StringVar directly
                        tk_var_object.set(str(config_value))

            # Apply loaded config values to the variables
            for key, var in self.__dict__.items():
                if key.endswith("_var") and key in loaded_config:
                    # Logic to safely set values:
                    # For tk.StringVar, set()
                    # For tk.BooleanVar, use set() with the bool/int value
                    if isinstance(var, tk.BooleanVar):
                        var.set(bool(loaded_config[key]))
                    elif isinstance(var, tk.StringVar):
                        var.set(str(loaded_config[key]))

            self._apply_theme()  # Re-apply theme in case dark mode setting was loaded
            self.toggle_processing_settings_fields()  # Update state of dependent fields
            messagebox.showinfo(
                "Settings Loaded",
                f"Successfully loaded settings from:\n{os.path.basename(filename)}",
            )
            self.status_label.config(text="Settings loaded.")

        except Exception as e:
            messagebox.showerror(
                "Load Error",
                f"Failed to load settings from {os.path.basename(filename)}:\n{e}",
            )
            self.status_label.config(text="Settings load failed.")

    def _move_processed_files(
        self,
        video_path,
        actual_depth_map_path,
        finished_source_folder,
        finished_depth_folder,
    ):
        """Moves source video, depth map, and its sidecar file to 'finished' folders."""
        max_retries = 5
        retry_delay_sec = 0.5  # Wait half a second between retries

        # Move source video
        if finished_source_folder:
            dest_path_src = os.path.join(
                finished_source_folder, os.path.basename(video_path)
            )
            for attempt in range(max_retries):
                try:
                    if os.path.exists(dest_path_src):
                        logger.warning(
                            f"File '{os.path.basename(video_path)}' already exists in '{finished_source_folder}'. Overwriting."
                        )
                        os.remove(dest_path_src)
                    shutil.move(video_path, finished_source_folder)
                    logger.debug(
                        f"==> Moved processed video '{os.path.basename(video_path)}' to: {finished_source_folder}"
                    )
                    break
                except PermissionError as e:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries}: PermissionError (file in use) when moving '{os.path.basename(video_path)}'. Retrying in {retry_delay_sec}s..."
                    )
                    time.sleep(retry_delay_sec)
                except Exception as e:
                    logger.error(
                        f"==> Failed to move source video '{os.path.basename(video_path)}' to '{finished_source_folder}': {e}",
                        exc_info=True,
                    )
                    break
            else:
                logger.error(
                    f"==> Failed to move source video '{os.path.basename(video_path)}' after {max_retries} attempts due to PermissionError."
                )
        else:
            logger.warning(
                f"==> Cannot move source video '{os.path.basename(video_path)}': 'finished_source_folder' is not set (not in batch mode)."
            )

        # Move depth map and its sidecar file
        if actual_depth_map_path and finished_depth_folder:
            dest_path_depth = os.path.join(
                finished_depth_folder, os.path.basename(actual_depth_map_path)
            )
            # --- Retry for Depth Map ---
            for attempt in range(max_retries):
                try:
                    if os.path.exists(dest_path_depth):
                        logger.warning(
                            f"File '{os.path.basename(actual_depth_map_path)}' already exists in '{finished_depth_folder}'. Overwriting."
                        )
                        os.remove(dest_path_depth)
                    shutil.move(actual_depth_map_path, finished_depth_folder)
                    logger.debug(
                        f"==> Moved depth map '{os.path.basename(actual_depth_map_path)}' to: {finished_depth_folder}"
                    )
                    break
                except PermissionError as e:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries}: PermissionError (file in use) when moving depth map '{os.path.basename(actual_depth_map_path)}'. Retrying in {retry_delay_sec}s..."
                    )
                    time.sleep(retry_delay_sec)
                except Exception as e:
                    logger.error(
                        f"==> Failed to move depth map '{os.path.basename(actual_depth_map_path)}' to '{finished_depth_folder}': {e}",
                        exc_info=True,
                    )
                    break
            else:
                logger.error(
                    f"==> Failed to move depth map '{os.path.basename(actual_depth_map_path)}' after {max_retries} attempts due to PermissionError."
                )

            # --- Retry for Sidecar file (if it exists) ---
            depth_map_dirname = os.path.dirname(actual_depth_map_path)
            depth_map_basename_without_ext = os.path.splitext(
                os.path.basename(actual_depth_map_path)
            )[0]
            input_sidecar_ext = self.APP_CONFIG_DEFAULTS.get(
                "SIDECAR_EXT", ".fssidecar"
            )  # Fallback to .fssidecar

            json_sidecar_path_to_move = os.path.join(
                depth_map_dirname,
                f"{depth_map_basename_without_ext}{input_sidecar_ext}",
            )
            dest_path_json = os.path.join(
                finished_depth_folder,
                f"{depth_map_basename_without_ext}{input_sidecar_ext}",
            )

            if os.path.exists(json_sidecar_path_to_move):
                for attempt in range(max_retries):
                    try:
                        if os.path.exists(dest_path_json):
                            logger.warning(
                                f"Sidecar file '{os.path.basename(json_sidecar_path_to_move)}' already exists in '{finished_depth_folder}'. Overwriting."
                            )
                            os.remove(dest_path_json)
                        shutil.move(json_sidecar_path_to_move, finished_depth_folder)
                        logger.debug(
                            f"==> Moved sidecar file '{os.path.basename(json_sidecar_path_to_move)}' to: {finished_depth_folder}"
                        )
                        break
                    except PermissionError as e:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries}: PermissionError (file in use) when moving file '{os.path.basename(json_sidecar_path_to_move)}'. Retrying in {retry_delay_sec}s..."
                        )
                        time.sleep(retry_delay_sec)
                    except Exception as e:
                        logger.error(
                            f"==> Failed to move sidecar file '{os.path.basename(json_sidecar_path_to_move)}' to '{finished_depth_folder}': {e}",
                            exc_info=True,
                        )
                        break
                else:
                    logger.error(
                        f"==> Failed to move sidecar file '{os.path.basename(json_sidecar_path_to_move)}' after {max_retries} attempts due to PermissionError."
                    )
            else:
                logger.debug(
                    f"==> No sidecar file '{json_sidecar_path_to_move}' found to move."
                )
        elif actual_depth_map_path:
            logger.info(
                f"==> Cannot move depth map '{os.path.basename(actual_depth_map_path)}': 'finished_depth_folder' is not set (not in batch mode)."
            )

    def on_auto_convergence_mode_select(self, event):
        """
        Handles selection in the Auto-Convergence combo box.
        If a mode is selected, it checks the cache and runs the calculation if needed.
        """
        mode = self.auto_convergence_mode_var.get()

        if mode == "Off":
            # self._auto_conv_cache = {"Average": None, "Peak": None} # Clear cache on Off
            return

        if self._is_auto_conv_running:
            logger.warning("Auto-Converge calculation is already running. Please wait.")
            return

        if self._auto_conv_cache[mode] is not None:
            # Value is cached, apply it immediately
            cached_value = self._auto_conv_cache[mode]

            # 1. Set the Tkinter variable to the cached value (needed for the setter)
            self.zero_disparity_anchor_var.set(f"{cached_value:.2f}")

            # 2. Call the programmatic setter to update the slider position and its label
            if self.set_convergence_value_programmatically:
                try:
                    self.set_convergence_value_programmatically(cached_value)
                except Exception as e:
                    logger.error(f"Error calling convergence setter on cache hit: {e}")

            # 3. Update status label
            self.status_label.config(
                text=f"Auto-Converge ({mode}): Loaded cached value {cached_value:.2f}"
            )

            # 4. Refresh preview
            self.on_slider_release(None)

            return

        # Cache miss, run the calculation (using the existing run_preview_auto_converge logic)
        self.run_preview_auto_converge(force_run=True)

    def on_slider_release(self, event=None):
        """Called when a slider is released. Updates the preview with DEBOUNCING."""
        # 1. Stop any current wigglegram animation immediately for responsiveness
        if hasattr(self, "previewer"):
            self.previewer._stop_wigglegram_animation()

        # 2. Cancel any pending update timer (this is the "debounce" logic)
        if self._preview_debounce_timer is not None:
            self.after_cancel(self._preview_debounce_timer)
            self._preview_debounce_timer = None

        # 3. Start a new timer.
        # 350ms is a good "norm" for responsiveness vs. stability.
        # If you click 10 times quickly, this only fires after the 10th click.
        self._preview_debounce_timer = self.after(
            350, self._perform_delayed_preview_update
        )

    def _perform_delayed_preview_update(self):
        """Actually triggers the heavy preview processing once the delay expires."""
        self._preview_debounce_timer = None  # Clear timer reference

        if hasattr(self, "previewer") and self.previewer.source_readers:
            # Trigger the standard preview update
            self.previewer.update_preview()

            # IMPORTANT: Do not refresh the "Current Processing Information" panel here.
            # This function is called by the slider debounce and would cause that panel to
            # flash on every slider move. Clip/state + info should update only when the clip changes.

    def _process_depth_batch(
        self,
        batch_depth_numpy_raw: np.ndarray,
        depth_stream_info: Optional[dict],
        depth_gamma: float,
        depth_dilate_size_x: float,
        depth_dilate_size_y: float,
        depth_blur_size_x: float,
        depth_blur_size_y: float,
        is_low_res_task: bool,
        max_raw_value: float,
        global_depth_min: float,
        global_depth_max: float,
        depth_dilate_left: float = 0.0,
        depth_blur_left: float = 0.0,
        debug_batch_index: int = 0,
        debug_frame_index: int = 0,
        debug_task_name: str = "PreProcess",
    ) -> np.ndarray:
        """
        Unified depth processor. Pre-processes filters in float space.
        Gamma is now unified to occur in normalized space.
        """
        if batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 3:
            batch_depth_numpy = batch_depth_numpy_raw.mean(axis=-1)
        else:
            batch_depth_numpy = (
                batch_depth_numpy_raw.squeeze(-1)
                if batch_depth_numpy_raw.ndim == 4
                else batch_depth_numpy_raw
            )

        batch_depth_numpy_float = batch_depth_numpy.astype(np.float32)

        # Dev Tools: allow skipping ALL low-res preprocessing (gamma/dilate/blur)
        if (
            is_low_res_task
            and getattr(self, "skip_lowres_preproc_var", None) is not None
            and self.skip_lowres_preproc_var.get()
        ):
            return batch_depth_numpy_float

        # Apply Filters BEFORE Gamma (Standard pipeline)
        current_width = (
            batch_depth_numpy_raw.shape[2]
            if batch_depth_numpy_raw.ndim == 4
            else batch_depth_numpy_raw.shape[1]
        )
        res_scale = math.sqrt(current_width / 960.0)

        def map_val(v):
            f_v = float(v)
            # Backward compatibility: older configs stored erosion as 30..40 => -0..-10
            if f_v > 30.0 and f_v <= 40.0:
                return -(f_v - 30.0)
            return f_v

        render_dilate_x = map_val(depth_dilate_size_x) * res_scale
        render_dilate_y = map_val(depth_dilate_size_y) * res_scale
        render_blur_x = depth_blur_size_x * res_scale
        render_blur_y = depth_blur_size_y * res_scale
        render_dilate_left = float(depth_dilate_left) * res_scale
        render_blur_left = float(depth_blur_left) * res_scale

        if (
            abs(render_dilate_left) > 1e-5
            or render_blur_left > 0
            or abs(render_dilate_x) > 1e-5
            or abs(render_dilate_y) > 1e-5
            or render_blur_x > 0
            or render_blur_y > 0
        ):
            device = torch.device("cpu")
            tensor_4d = (
                torch.from_numpy(batch_depth_numpy_float).unsqueeze(1).to(device)
            )
            # Left-only pre-step (directional): applied before normal X/Y dilate/blur to preserve parity

            # Dilate Left (directional) - optional
            if abs(render_dilate_left) > 1e-5:
                tensor_before = tensor_4d
                tensor_4d = custom_dilate_left(
                    tensor_before, float(render_dilate_left), False, max_raw_value
                )

            if render_blur_left > 0:
                # Blur Left: blur *only* along strong left edges (dark->bright when moving left->right).
                # This avoids blurring smooth gradients that typically don't create warp/splat jaggies.
                effective_max_value = max(max_raw_value, 1e-5)
                EDGE_STEP_8BIT = (
                    3.0  # raise to blur fewer edges; lower to blur more edges
                )
                step_thresh = effective_max_value * (EDGE_STEP_8BIT / 255.0)

                dx = tensor_4d[:, :, :, 1:] - tensor_4d[:, :, :, :-1]
                edge_core = dx > step_thresh

                edge_mask = torch.zeros_like(tensor_4d, dtype=torch.float32)
                edge_mask[:, :, :, 1:] = edge_core.float()

                # Expand into a small band around the edge (both sides) so it feels like a normal blur (no hard cut-off).
                k_blur = int(round(render_blur_left))
                if k_blur <= 0:
                    k_blur = 1
                if k_blur % 2 == 0:
                    k_blur += 1

                # Keep the band relatively tight around the detected edge so we don't soften large interior regions.
                band_half = max(1, int(math.ceil(k_blur / 4.0)))
                edge_band = (
                    F.max_pool2d(
                        edge_mask,
                        kernel_size=(1, 2 * band_half + 1),
                        stride=1,
                        padding=(0, band_half),
                    )
                    > 0.5
                ).float()

                # Feather the band so the blend ramps on/off smoothly.
                alpha = custom_blur(edge_band, 7, 1, False, 1.0)
                alpha = torch.clamp(alpha, 0.0, 1.0)

                # Two-pass blur for Blur Left:
                # - Horizontal-only blur helps anti-alias along X (like your regular Blur X behavior),
                # - Vertical-only blur helps smooth stair-steps along the edge.
                # We blend horizontal/vertical Blur Left based on a compact UI selector:
                #   0.0 = all horizontal, 1.0 = all vertical, 0.5 = 50/50.
                try:
                    mix_f = float(self.depth_blur_left_mix_var.get())
                except Exception:
                    mix_f = 0.5
                mix_f = max(0.0, min(1.0, mix_f))

                BLUR_LEFT_V_WEIGHT = mix_f
                BLUR_LEFT_H_WEIGHT = 1.0 - mix_f

                blurred_h = None
                blurred_v = None
                if BLUR_LEFT_H_WEIGHT > 1e-6:
                    blurred_h = custom_blur(tensor_4d, k_blur, 1, False, max_raw_value)
                if BLUR_LEFT_V_WEIGHT > 1e-6:
                    blurred_v = custom_blur(tensor_4d, 1, k_blur, False, max_raw_value)

                if blurred_h is not None and blurred_v is not None:
                    wsum = BLUR_LEFT_H_WEIGHT + BLUR_LEFT_V_WEIGHT
                    blurred = (
                        blurred_h * BLUR_LEFT_H_WEIGHT + blurred_v * BLUR_LEFT_V_WEIGHT
                    ) / max(wsum, 1e-6)
                elif blurred_h is not None:
                    blurred = blurred_h
                elif blurred_v is not None:
                    blurred = blurred_v
                else:
                    blurred = tensor_4d

                tensor_4d = tensor_4d * (1.0 - alpha) + blurred * alpha
            if abs(render_dilate_x) > 1e-5 or abs(render_dilate_y) > 1e-5:
                tensor_4d = custom_dilate(
                    tensor_4d,
                    float(render_dilate_x),
                    float(render_dilate_y),
                    False,
                    max_raw_value,
                )
            if render_blur_x > 0 or render_blur_y > 0:
                tensor_4d = custom_blur(
                    tensor_4d,
                    float(render_blur_x),
                    float(render_blur_y),
                    False,
                    max_raw_value,
                )
            batch_depth_numpy_float = tensor_4d.squeeze(1).cpu().numpy()
            release_cuda_memory()

        return batch_depth_numpy_float

    def _process_single_video_tasks(
        self,
        video_path,
        settings,
        initial_overall_task_counter,
        is_single_file_mode,
        finished_source_folder=None,
        finished_depth_folder=None,
    ):
        """
        Handles the full processing lifecycle (sidecar, auto-conv, task loop, move-to-finished)
        for a single video and its depth map.

        Returns: (tasks_processed_count: int, any_task_completed_successfully: bool)
        """
        # Initialize task-local variables (some of these were local in the old _run_batch_process loop)
        current_depth_dilate_size_x = 0
        current_depth_dilate_size_y = 0
        current_depth_blur_size_x = 0
        current_depth_blur_size_y = 0
        current_depth_dilate_left = 0.0
        current_depth_blur_left = 0.0
        current_depth_blur_left_mix = 0.0

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        logger.info(f"==> Processing Video: {video_name}")
        self.progress_queue.put(("update_info", {"filename": video_name}))

        # Keep a local counter for tasks processed in this function
        local_task_counter = initial_overall_task_counter

        video_specific_settings = self._get_video_specific_settings(
            video_path,
            settings["input_depth_maps"],
            settings["zero_disparity_anchor"],
            settings["max_disp"],
            is_single_file_mode,
        )

        processing_tasks = self._get_defined_tasks(settings)
        expected_task_count = len(processing_tasks)
        processed_tasks_count = 0
        any_task_completed_successfully_for_this_video = False

        if video_specific_settings.get("error"):
            logger.error(
                f"Error getting video specific settings for {video_name}: {video_specific_settings['error']}. Skipping."
            )
            # Skip the expected task count in the progress bar
            local_task_counter += expected_task_count
            self.progress_queue.put(("processed", local_task_counter))
            return expected_task_count, False

        actual_depth_map_path = video_specific_settings["actual_depth_map_path"]
        current_zero_disparity_anchor = video_specific_settings["convergence_plane"]
        current_max_disparity_percentage = video_specific_settings[
            "max_disparity_percentage"
        ]
        current_input_bias = video_specific_settings["input_bias"]
        anchor_source = video_specific_settings["anchor_source"]
        max_disp_source = video_specific_settings["max_disp_source"]
        gamma_source = video_specific_settings["gamma_source"]
        map_source = video_specific_settings.get("map_source", "N/A")
        current_depth_gamma = video_specific_settings["depth_gamma"]
        current_depth_dilate_size_x = video_specific_settings["depth_dilate_size_x"]
        current_depth_dilate_size_y = video_specific_settings["depth_dilate_size_y"]
        current_depth_blur_size_x = video_specific_settings["depth_blur_size_x"]
        current_depth_blur_size_y = video_specific_settings["depth_blur_size_y"]
        current_depth_dilate_left = float(
            video_specific_settings.get(
                "depth_dilate_left", settings.get("depth_dilate_left", 0.0)
            )
        )
        current_depth_blur_left = float(
            video_specific_settings.get(
                "depth_blur_left", settings.get("depth_blur_left", 0.0)
            )
        )
        current_depth_blur_left_mix = float(
            video_specific_settings.get(
                "depth_blur_left_mix",
                getattr(self, "depth_blur_left_mix_var", None).get()
                if getattr(self, "depth_blur_left_mix_var", None)
                else 0.0,
            )
        )

        if not processing_tasks:
            logger.debug(
                f"==> No processing tasks configured for {video_name}. Skipping."
            )
            return 0, False

        # --- Auto-Convergence Logic (BEFORE initializing readers) ---
        auto_conv_mode = settings["auto_convergence_mode"]

        # --- NEW: Global Normalization Policy variables ---
        enable_global_norm_policy = video_specific_settings["enable_global_norm"]
        gn_source = video_specific_settings["gn_source"]

        if (
            video_specific_settings["sidecar_found"]
            and self.enable_global_norm_var.get()
        ):
            # Policy: Sidecar exists AND GUI toggle is ON. Policy forces GN OFF.
            if not self._gn_warning_shown:
                messagebox.showwarning(
                    "GN Policy Warning",
                    f"Sidecar found for '{video_name}'.\n"
                    f"Global Normalization is DISABLED for this clip, overriding the GUI setting.\n"
                    f"Further warnings will be logged to console only.",
                )
                self._gn_warning_shown = (
                    True  # Set flag to log to console only next time
                )
            else:
                logger.warning(
                    f"GN Policy: Sidecar found for {video_name}. GN forced OFF (console log only)."
                )

        # --- NEW LOGIC: Sidecar overrides Auto-Convergence ---
        if anchor_source == "Sidecar" and auto_conv_mode != "Off":
            logger.info(
                f"Sidecar found for {video_name}. Convergence Point locked to Sidecar value ({current_zero_disparity_anchor:.4f}). Auto-Convergence SKIPPED."
            )
            auto_conv_mode = "Off"

        if auto_conv_mode != "Off":
            logger.info(
                f"Auto-Convergence is ENABLED (Mode: {auto_conv_mode}). Running pre-pass..."
            )

            try:
                anchor_float = float(current_zero_disparity_anchor)
            except (ValueError, TypeError):
                logger.error(
                    f"Invalid convergence anchor value found: {current_zero_disparity_anchor}. Defaulting to 0.5."
                )
                anchor_float = 0.5

            new_anchor_avg, new_anchor_peak = self._determine_auto_convergence(
                actual_depth_map_path,
                settings["process_length"],
                settings["full_res_batch_size"],
                anchor_float,
            )
            new_anchor_val = (
                new_anchor_avg if auto_conv_mode == "Average" else new_anchor_peak
            )

            if new_anchor_val != current_zero_disparity_anchor:
                current_zero_disparity_anchor = new_anchor_val
                anchor_source = "Auto"

            logger.info(
                f"Using Convergence Point: {current_zero_disparity_anchor:.4f} (Source: {anchor_source})"
            )

        for task in processing_tasks:
            if self.stop_event.is_set():
                logger.info(
                    f"==> Stopping {task['name']} processing for {video_name} due to user request"
                )
                # Increment the global counter for all remaining, skipped tasks
                remaining_tasks_to_increment = (
                    expected_task_count - processed_tasks_count
                )
                local_task_counter += remaining_tasks_to_increment
                self.progress_queue.put(("processed", local_task_counter))
                return (
                    expected_task_count,
                    any_task_completed_successfully_for_this_video,
                )

            logger.debug(f"\n==> Starting {task['name']} pass for {video_name}")
            self.progress_queue.put(
                ("status", f"Processing {task['name']} for {video_name}")
            )

            # Decide what to show in the Map field
            if self.multi_map_var.get():
                # Multi-Map mode
                if actual_depth_map_path and map_source not in ("", "N/A"):
                    map_folder = os.path.basename(
                        os.path.dirname(actual_depth_map_path)
                    ).strip()
                    map_label = f"{map_folder} ({map_source})"
                else:
                    map_label = "N/A"
            else:
                # Normal mode
                map_label = "Direct file" if is_single_file_mode else "Direct folder"

            self.progress_queue.put(
                (
                    "update_info",
                    {
                        "task_name": task["name"],
                        "convergence": f"{current_zero_disparity_anchor:.2f} ({anchor_source})",
                        "disparity": f"{current_max_disparity_percentage:.1f}% ({max_disp_source})",
                        "gamma": f"{current_depth_gamma:.2f} ({gamma_source})",
                        "map": map_label,
                    },
                )
            )

            (
                video_reader_input,
                depth_reader_input,
                processed_fps,
                original_vid_h,
                original_vid_w,
                current_processed_height,
                current_processed_width,
                video_stream_info,
                total_frames_input,
                total_frames_depth,
                actual_depth_height,
                actual_depth_width,
                depth_stream_info,
            ) = self._initialize_video_and_depth_readers(
                video_path,
                actual_depth_map_path,
                settings["process_length"],
                task,
                settings["match_depth_res"],
            )

            # Explicitly check for None for critical components before proceeding
            if (
                video_reader_input is None
                or depth_reader_input is None
                or video_stream_info is None
            ):
                logger.error(
                    f"Skipping {task['name']} pass for {video_name} due to reader initialization error, frame count mismatch, or missing stream info."
                )
                local_task_counter += 1
                processed_tasks_count += 1
                self.progress_queue.put(("processed", local_task_counter))
                release_cuda_memory()
                continue

            # --- MODIFIED: Use the policy to determine the mode ---
            assume_raw_input_mode = (
                not enable_global_norm_policy
            )  # If GN is OFF, assume RAW Input
            global_depth_min = 0.0
            global_depth_max = 1.0

            # --- UNCONDITIONAL Max Content Value Scan for RAW/Normalization Modes ---
            max_content_value = 1.0
            raw_depth_reader_temp = None
            try:
                depth_info_tmp = get_video_stream_info(actual_depth_map_path)

                bit_depth_tmp = _infer_depth_bit_depth(depth_info_tmp)

                pix_fmt_tmp = str((depth_info_tmp or {}).get("pix_fmt", ""))

                # Use source/processing dimensions for the content scan (avoids NameError and keeps preview/render parity)
                scan_w = original_vid_w or current_processed_width or actual_depth_width
                scan_h = (
                    original_vid_h or current_processed_height or actual_depth_height
                )

                if bit_depth_tmp > 8:
                    raw_depth_reader_temp = FFmpegDepthPipeReader(
                        actual_depth_map_path,
                        out_w=scan_w,
                        out_h=scan_h,
                        bit_depth=bit_depth_tmp,
                        num_frames=total_frames_depth,
                        pix_fmt=pix_fmt_tmp,
                    )

                else:
                    raw_depth_reader_temp = VideoReader(
                        actual_depth_map_path, ctx=cpu(0), width=scan_w, height=scan_h
                    )

                if len(raw_depth_reader_temp) > 0:
                    _, max_content_value = compute_global_depth_stats(
                        depth_map_reader=raw_depth_reader_temp,
                        total_frames=total_frames_depth,
                        chunk_size=task["batch_size"],
                    )
                    logger.debug(f"Max content depth scanned: {max_content_value:.3f}.")
                else:
                    logger.error("RAW depth reader has no frames for content scan.")
            except Exception as e:
                logger.error(f"Failed to scan max content depth: {e}")
            finally:
                if raw_depth_reader_temp:
                    if hasattr(raw_depth_reader_temp, "close"):
                        try:
                            raw_depth_reader_temp.close()
                        except Exception:
                            pass
                    del raw_depth_reader_temp
                    gc.collect()
            # --- END UNCONDITIONAL SCAN ---

            if not assume_raw_input_mode:
                logger.info(
                    "==> Global Depth Normalization selected. Starting global depth stats pre-pass with RAW reader."
                )

                raw_depth_reader_temp = None
                try:
                    depth_info_tmp = get_video_stream_info(actual_depth_map_path)

                    bit_depth_tmp = _infer_depth_bit_depth(depth_info_tmp)

                    pix_fmt_tmp = str((depth_info_tmp or {}).get("pix_fmt", ""))

                    if bit_depth_tmp > 8:
                        raw_depth_reader_temp = FFmpegDepthPipeReader(
                            actual_depth_map_path,
                            out_w=actual_depth_width,
                            out_h=actual_depth_height,
                            bit_depth=bit_depth_tmp,
                            num_frames=total_frames_depth,
                            pix_fmt=pix_fmt_tmp,
                        )

                    else:
                        raw_depth_reader_temp = VideoReader(
                            actual_depth_map_path,
                            ctx=cpu(0),
                            width=actual_depth_width,
                            height=actual_depth_height,
                        )

                    if len(raw_depth_reader_temp) > 0:
                        global_depth_min, global_depth_max = compute_global_depth_stats(
                            depth_map_reader=raw_depth_reader_temp,
                            total_frames=total_frames_depth,
                            chunk_size=task["batch_size"],
                        )
                        logger.debug(
                            "Successfully computed global stats from RAW reader."
                        )
                    else:
                        logger.error("RAW depth reader has no frames.")
                except Exception as e:
                    logger.error(
                        f"Failed to initialize/read RAW depth reader for global stats: {e}"
                    )
                    global_depth_min = 0.0
                    global_depth_max = 1.0
                finally:
                    if raw_depth_reader_temp:
                        if hasattr(raw_depth_reader_temp, "close"):
                            try:
                                raw_depth_reader_temp.close()
                            except Exception:
                                pass
                        del raw_depth_reader_temp
                        gc.collect()
            else:
                logger.debug(
                    "==> No Normalization (Assume Raw 0-1 Input) selected. Skipping global stats pre-pass."
                )

                # --- RAW INPUT MODE SCALING ---
                final_scaling_factor = 1.0

                if max_content_value <= 256.0 and max_content_value > 1.0:
                    final_scaling_factor = 255.0
                    logger.debug(
                        f"Content Max {max_content_value:.2f} <= 8-bit. SCALING BY 255.0."
                    )
                elif max_content_value > 256.0 and max_content_value <= 1024.0:
                    final_scaling_factor = 1023.0
                    logger.debug(
                        f"Content Max {max_content_value:.2f} (9-10bit). SCALING BY 1023.0 (expected 10-bit max) for preview parity."
                    )
                else:
                    final_scaling_factor = 1023.0
                    logger.warning(
                        f"Max content value is too high/low ({max_content_value:.2f}). Using fallback 1023.0."
                    )

                global_depth_max = final_scaling_factor
                global_depth_min = 0.0

                logger.debug(
                    f"Raw Input Final Scaling Factor set to: {global_depth_max:.3f}"
                )

            if not (
                actual_depth_height == current_processed_height
                and actual_depth_width == current_processed_width
            ):
                # Low-res parity path intentionally loads depth at clip resolution so dilation/blur happens pre-resize.
                # The depth frames are resized to match the (possibly low-res) video frames *after* preprocessing inside depthSplatting().
                if task.get("is_low_res"):
                    skip_lr_preproc = bool(
                        getattr(self, "skip_lowres_preproc_var", None) is not None
                        and self.skip_lowres_preproc_var.get()
                    )
                    if (not skip_lr_preproc) and (
                        actual_depth_width == original_vid_w
                        and actual_depth_height == original_vid_h
                    ):
                        logger.info(
                            f"==> Low-Resolution pass: Upscaling depthmap to clip resolution ({actual_depth_width}x{actual_depth_height}) "
                            f"before preprocessing for parity; it will be resized to ({current_processed_width}x{current_processed_height}) after preprocessing."
                        )
                    else:
                        _bd_warn = (
                            _infer_depth_bit_depth(depth_stream_info)
                            if depth_stream_info
                            else 8
                        )
                        if _bd_warn > 8:
                            logger.warning(
                                f"==> Warning: Depth map reader output resolution ({actual_depth_width}x{actual_depth_height}) does not match processed "
                                f"video resolution ({current_processed_width}x{current_processed_height}) for {task['name']} pass. "
                                f"This indicates an issue with `load_pre_rendered_depth`'s `width`/`height` parameters. Processing may proceed but results might be misaligned."
                            )
                else:
                    _bd_warn = (
                        _infer_depth_bit_depth(depth_stream_info)
                        if depth_stream_info
                        else 8
                    )
                    if _bd_warn > 8:
                        logger.warning(
                            f"==> Warning: Depth map reader output resolution ({actual_depth_width}x{actual_depth_height}) does not match processed "
                            f"video resolution ({current_processed_width}x{current_processed_height}) for {task['name']} pass. "
                            f"This indicates an issue with `load_pre_rendered_depth`'s `width`/`height` parameters. Processing may proceed but results might be misaligned."
                        )

            actual_percentage_for_calculation = current_max_disparity_percentage / 20.0
            actual_max_disp_pixels = (
                actual_percentage_for_calculation / 100.0
            ) * current_processed_width
            logger.debug(
                f"==> Max Disparity Input: {current_max_disparity_percentage:.1f}% -> Calculated Max Disparity for splatting ({task['name']}): {actual_max_disp_pixels:.2f} pixels"
            )

            self.progress_queue.put(
                (
                    "update_info",
                    {
                        "disparity": f"{current_max_disparity_percentage:.1f}% ({actual_max_disp_pixels:.2f} pixels)"
                    },
                )
            )

            current_output_subdir = os.path.join(
                settings["output_splatted"], task["output_subdir"]
            )
            os.makedirs(current_output_subdir, exist_ok=True)
            output_video_path_base = os.path.join(
                current_output_subdir, f"{video_name}.mp4"
            )

            completed_splatting_task = self.depthSplatting(
                input_video_reader=video_reader_input,
                depth_map_reader=depth_reader_input,
                total_frames_to_process=total_frames_input,
                processed_fps=processed_fps,
                output_video_path_base=output_video_path_base,
                target_output_height=current_processed_height,
                target_output_width=current_processed_width,
                max_disp=actual_max_disp_pixels,
                process_length=settings["process_length"],
                batch_size=task["batch_size"],
                dual_output=settings["dual_output"],
                zero_disparity_anchor_val=current_zero_disparity_anchor,
                video_stream_info=video_stream_info,
                input_bias=current_input_bias,
                assume_raw_input=assume_raw_input_mode,
                global_depth_min=global_depth_min,
                global_depth_max=global_depth_max,
                depth_stream_info=depth_stream_info,
                user_output_crf=(
                    settings.get("output_crf_low")
                    if task["is_low_res"]
                    else settings.get("output_crf_full", settings.get("output_crf"))
                ),
                is_low_res_task=task["is_low_res"],
                depth_gamma=current_depth_gamma,
                depth_dilate_size_x=(
                    0.0
                    if (
                        task["is_low_res"]
                        and getattr(self, "skip_lowres_preproc_var", None) is not None
                        and self.skip_lowres_preproc_var.get()
                    )
                    else current_depth_dilate_size_x
                ),
                depth_dilate_size_y=(
                    0.0
                    if (
                        task["is_low_res"]
                        and getattr(self, "skip_lowres_preproc_var", None) is not None
                        and self.skip_lowres_preproc_var.get()
                    )
                    else current_depth_dilate_size_y
                ),
                depth_blur_size_x=(
                    0.0
                    if (
                        task["is_low_res"]
                        and getattr(self, "skip_lowres_preproc_var", None) is not None
                        and self.skip_lowres_preproc_var.get()
                    )
                    else current_depth_blur_size_x
                ),
                depth_blur_size_y=(
                    0.0
                    if (
                        task["is_low_res"]
                        and getattr(self, "skip_lowres_preproc_var", None) is not None
                        and self.skip_lowres_preproc_var.get()
                    )
                    else current_depth_blur_size_y
                ),
                depth_dilate_left=(
                    0.0
                    if (
                        task["is_low_res"]
                        and getattr(self, "skip_lowres_preproc_var", None) is not None
                        and self.skip_lowres_preproc_var.get()
                    )
                    else float(current_depth_dilate_left)
                ),
                depth_blur_left=(
                    0.0
                    if (
                        task["is_low_res"]
                        and getattr(self, "skip_lowres_preproc_var", None) is not None
                        and self.skip_lowres_preproc_var.get()
                    )
                    else float(current_depth_blur_left)
                ),
            )

            if self.stop_event.is_set():
                logger.info(
                    f"==> Stopping {task['name']} pass for {video_name} due to user request"
                )
                break

            if completed_splatting_task:
                logger.debug(
                    f"==> Splatted {task['name']} video saved for {video_name}."
                )
                any_task_completed_successfully_for_this_video = True

                if video_reader_input is not None:
                    del video_reader_input
                if depth_reader_input is not None:
                    del depth_reader_input
                torch.cuda.empty_cache()
                gc.collect()
                logger.debug(
                    "Explicitly deleted VideoReader objects and forced garbage collection to release file handles."
                )
            else:
                logger.info(
                    f"==> Splatting task '{task['name']}' for '{video_name}' was skipped or failed. Files will NOT be moved."
                )
                if video_reader_input:
                    del video_reader_input
                if depth_reader_input:
                    del depth_reader_input
                torch.cuda.empty_cache()
                gc.collect()

            local_task_counter += 1
            processed_tasks_count += 1
            self.progress_queue.put(("processed", local_task_counter))
            logger.debug(f"==> Completed {task['name']} pass for {video_name}.")

        # After all tasks for the current video are processed or stopped
        if self.stop_event.is_set():
            return expected_task_count, any_task_completed_successfully_for_this_video

        # Move to finished logic
        move_enabled = settings[
            "move_to_finished"
        ]  # Use the setting from the dictionary

        if is_single_file_mode:
            # CRITICAL FIX: Get the finished folders directly from settings (set in start_single_processing)
            single_finished_src = settings.get("single_finished_source_folder")
            single_finished_depth = settings.get("single_finished_depth_folder")

            # --- Check move_enabled setting ---
            if (
                single_finished_src
                and single_finished_depth
                and move_enabled
                and any_task_completed_successfully_for_this_video
            ):
                self._move_processed_files(
                    video_path,
                    actual_depth_map_path,
                    single_finished_src,
                    single_finished_depth,
                )
            else:
                logger.debug(
                    f"Single file move skipped. Enabled={move_enabled}, Success={any_task_completed_successfully_for_this_video}, PathsValid={bool(single_finished_src)}"
                )

        elif (
            any_task_completed_successfully_for_this_video
            and finished_source_folder
            and finished_depth_folder
            and move_enabled
        ):
            # Batch mode move (uses the arguments passed from _run_batch_process)
            self._move_processed_files(
                video_path,
                actual_depth_map_path,
                finished_source_folder,
                finished_depth_folder,
            )

        # Return the number of tasks actually processed for the global counter update
        return processed_tasks_count, any_task_completed_successfully_for_this_video

    def _preview_processing_callback(
        self, source_frames: dict, params: dict
    ) -> Optional[Image.Image]:
        """
        Callback for VideoPreviewer. Performs splatting on a single frame for preview.
        """
        # NOTE: Do not clear the 'Current Processing Information' panel on every preview render.
        # Clearing here caused filename/task info to briefly appear and then reset to N/A each time the preview updates.

        if not globals()["CUDA_AVAILABLE"]:
            logger.error("Preview processing requires a CUDA-enabled GPU.")
            return None

        logger.debug("--- Starting Preview Processing Callback ---")

        left_eye_tensor = source_frames.get("source_video")
        depth_tensor_raw = source_frames.get("depth_map")

        if left_eye_tensor is None or depth_tensor_raw is None:
            logger.error("Preview failed: Missing source video or depth map tensor.")
            return None

        # --- Get latest settings and Preview Mode ---
        params = self.get_current_preview_settings()
        if not params:
            logger.error("Preview failed: Could not get current preview settings.")
            return None

        preview_source = self.preview_source_var.get()
        is_low_res_preview = preview_source in [
            "Splat Result(Low)",
            "Occlusion Mask(Low)",
        ]

        # Determine the target resolution for the preview tensor
        W_orig = left_eye_tensor.shape[3]
        H_orig = left_eye_tensor.shape[2]

        # ----------------------------------------------------------------------
        # NEW SIDECAR LOGIC FOR PREVIEW
        # ----------------------------------------------------------------------
        depth_map_path = None
        if 0 <= self.previewer.current_video_index < len(self.previewer.video_list):
            current_source_dict = self.previewer.video_list[
                self.previewer.current_video_index
            ]
            depth_map_path = current_source_dict.get("depth_map")

        gui_config = {
            "convergence_plane": float(self.zero_disparity_anchor_var.get()),
            "max_disparity": float(self.max_disp_var.get()),
            "gamma": float(self.depth_gamma_var.get()),
        }

        merged_config = gui_config.copy()

        # Set final parameters from the merged config
        params["convergence_point"] = merged_config["convergence_plane"]
        params["max_disp"] = merged_config["max_disparity"]
        params["depth_gamma"] = merged_config["gamma"]

        # ----------------------------------------------------------------------
        # END NEW SIDECAR LOGIC FOR PREVIEW
        # ----------------------------------------------------------------------

        W_target, H_target = W_orig, H_orig

        if is_low_res_preview:
            try:
                W_target_requested = int(self.pre_res_width_var.get())

                if W_target_requested <= 0:
                    W_target_requested = W_orig  # Fallback

                # 1. Calculate aspect-ratio-correct height based on the requested width
                aspect_ratio = W_orig / H_orig
                H_target_calculated = int(round(W_target_requested / aspect_ratio))

                # 2. Ensure both W and H are divisible by 2 for codec compatibility
                W_target = (
                    W_target_requested
                    if W_target_requested % 2 == 0
                    else W_target_requested + 1
                )
                H_target = (
                    H_target_calculated
                    if H_target_calculated % 2 == 0
                    else H_target_calculated + 1
                )

                # 3. Handle potential extreme fallbacks
                if W_target <= 0 or H_target <= 0:
                    W_target, H_target = W_orig, H_orig
                    logger.warning(
                        "Low-Res preview: Calculated dimensions invalid, falling back to original."
                    )
                else:
                    logger.debug(
                        f"Low-Res preview: AR corrected target {W_target}x{H_target}. (Original W: {W_orig}, H: {H_orig})"
                    )

                # Resize Left Eye to aspect-ratio-correct low-res target for consistency
                left_eye_tensor_resized = F.interpolate(
                    left_eye_tensor.cuda(),
                    size=(H_target, W_target),
                    mode="bilinear",
                    align_corners=False,
                )
            except Exception as e:
                logger.error(
                    f"Low-Res preview failed during AR calculation/resize: {e}. Falling back to original res.",
                    exc_info=True,
                )
                W_target, H_target = W_orig, H_orig
                left_eye_tensor_resized = left_eye_tensor.cuda()
        else:
            left_eye_tensor_resized = left_eye_tensor.cuda()  # Use original res

        logger.debug(f"Preview Params: {params}")
        logger.debug(
            f"Target Resolution: {W_target}x{H_target} (Low-Res: {is_low_res_preview})"
        )

        # --- Process Depth Frame ---
        depth_numpy_raw = depth_tensor_raw.squeeze(0).permute(1, 2, 0).cpu().numpy()
        logger.debug(
            f"Raw depth numpy shape: {depth_numpy_raw.shape}, range: [{depth_numpy_raw.min():.2f}, {depth_numpy_raw.max():.2f}]"
        )

        # Ensure depth pre-processing happens at the same resolution as the *render* pipeline.
        # - Full-res preview: pre-proc at clip resolution (W_orig/H_orig).
        # - Low-res preview: pre-proc at clip resolution, then downscale to low-res AFTER pre-proc.
        #   This matches the low-res render ordering and avoids the "extra thick" low-res preview look.
        W_preproc, H_preproc = (W_orig, H_orig)
        if depth_numpy_raw.ndim == 2:
            depth_numpy_raw = depth_numpy_raw[:, :, None]
        if (
            depth_numpy_raw.shape[0] != H_preproc
            or depth_numpy_raw.shape[1] != W_preproc
        ):
            try:
                had_singleton_channel = (
                    depth_numpy_raw.ndim == 3 and depth_numpy_raw.shape[2] == 1
                )
                interp = (
                    cv2.INTER_LINEAR
                    if (
                        W_preproc > depth_numpy_raw.shape[1]
                        or H_preproc > depth_numpy_raw.shape[0]
                    )
                    else cv2.INTER_AREA
                )
                depth_numpy_raw = cv2.resize(
                    depth_numpy_raw, (W_preproc, H_preproc), interpolation=interp
                )
                if had_singleton_channel and depth_numpy_raw.ndim == 2:
                    depth_numpy_raw = depth_numpy_raw[:, :, None]
                logger.debug(
                    f"Preview depth resized for pre-proc: {depth_numpy_raw.shape}"
                )
            except Exception as e:
                logger.error(
                    f"Preview depth resize (pre-proc) failed: {e}. Continuing with raw depth resolution.",
                    exc_info=True,
                )

        # 1. DETERMINE MAX CONTENT VALUE FOR THE FRAME (for AutoGain scaling)
        # We need the max *raw* value of the depth frame content
        max_raw_content_value = depth_numpy_raw.max()
        if max_raw_content_value < 1.0:
            max_raw_content_value = 1.0  # Fallback for already 0-1 normalized content

        # --- NEW: Get Global Normalization Policy for Preview (Sidecar check) ---
        enable_global_norm = params.get("enable_global_norm", False)

        # Policy Check: Sidecar existence forces GN OFF
        sidecar_exists = False
        if depth_map_path:
            sidecar_folder = self._get_sidecar_base_folder()
            depth_map_basename = os.path.splitext(os.path.basename(depth_map_path))[0]
            sidecar_ext = self.APP_CONFIG_DEFAULTS["SIDECAR_EXT"]
            json_sidecar_path = os.path.join(
                sidecar_folder, f"{depth_map_basename}{sidecar_ext}"
            )
            sidecar_exists = os.path.exists(json_sidecar_path)

        if sidecar_exists:
            # Policy: If sidecar exists, GN is forced OFF
            enable_global_norm = False

        # --- NEW: Determine Global Min/Max from cache if GN is ON ---
        global_min, global_max = 0.0, 1.0

        if enable_global_norm and depth_map_path:
            if depth_map_path not in self._clip_norm_cache:
                # --- CACHE MISS: Run the slow scan synchronously ---
                logger.info(
                    f"Preview GN: Cache miss for {os.path.basename(depth_map_path)}. Running clip-local scan..."
                )
                global_min, global_max = self._compute_clip_global_depth_stats(
                    depth_map_path
                )
            else:
                # --- CACHE HIT: Use cached values ---
                global_min, global_max = self._clip_norm_cache[depth_map_path]
                logger.debug(
                    f"Preview GN: Cache hit for {os.path.basename(depth_map_path)}. Min/Max: {global_min:.3f}/{global_max:.3f}"
                )

        # --- END NEW CACHE LOGIC ---

        # Determine the scaling factor (Only relevant for MANUAL/RAW mode)
        final_scaling_factor = 1.0

        if not enable_global_norm:  # MANUAL/RAW INPUT MODE
            if max_raw_content_value <= 256.0 and max_raw_content_value > 1.0:
                final_scaling_factor = 255.0
            elif max_raw_content_value > 256.0 and max_raw_content_value <= 1024.0:
                final_scaling_factor = max_raw_content_value
            elif max_raw_content_value > 1024.0:
                final_scaling_factor = 65535.0
            else:
                final_scaling_factor = 1.0
        else:  # GLOBAL NORMALIZATION MODE
            # Use the global max from the cache/scan as the "max value" for scaling (only to correctly apply pre-processing if needed)
            final_scaling_factor = max(global_max, 1e-5)

        logger.debug(
            f"Preview: GN={enable_global_norm}. Final Scaling Factor for Pre-Proc: {final_scaling_factor:.3f}"
        )

        depth_numpy_processed = self._process_depth_batch(
            batch_depth_numpy_raw=np.expand_dims(depth_numpy_raw, axis=0),
            depth_stream_info=None,
            depth_gamma=params["depth_gamma"],
            depth_dilate_size_x=params["depth_dilate_size_x"],
            depth_dilate_size_y=params["depth_dilate_size_y"],
            depth_blur_size_x=params["depth_blur_size_x"],
            depth_blur_size_y=params["depth_blur_size_y"],
            depth_dilate_left=params.get("depth_dilate_left", 0.0),
            depth_blur_left=params.get("depth_blur_left", 0.0),
            is_low_res_task=is_low_res_preview,
            max_raw_value=final_scaling_factor,
            global_depth_min=0.0,
            global_depth_max=1.0,
            debug_task_name="Preview",
        )
        logger.debug(
            f"Processed depth numpy shape: {depth_numpy_processed.shape}, range: [{depth_numpy_processed.min():.2f}, {depth_numpy_processed.max():.2f}]"
        )

        # Low-Res preview: downscale the *processed* depth AFTER dilation/blur (matches render ordering).
        if is_low_res_preview and (
            depth_numpy_processed.shape[1] != H_target
            or depth_numpy_processed.shape[2] != W_target
        ):
            try:
                _d = depth_numpy_processed.squeeze(0)
                _interp = (
                    cv2.INTER_AREA
                    if (W_target < _d.shape[1] and H_target < _d.shape[0])
                    else cv2.INTER_LINEAR
                )
                _d = cv2.resize(_d, (W_target, H_target), interpolation=_interp)
                depth_numpy_processed = np.expand_dims(_d.astype(np.float32), axis=0)
                logger.debug(
                    f"Low-Res preview: resized processed depth to target {W_target}x{H_target}."
                )
            except Exception as e:
                logger.error(
                    f"Low-Res preview post-preproc resize failed: {e}. Continuing with processed depth resolution.",
                    exc_info=True,
                )

        # 2. Normalize based on the 'enable_global_norm' policy
        depth_normalized = depth_numpy_processed.squeeze(0)

        # Match render behavior:
        # - In GN mode: normalize using cached/scanned min/max
        # - In Manual/RAW mode: scale into 0-1 using the detected scaling factor
        if enable_global_norm:
            min_val, max_val = global_min, global_max
            depth_range = max_val - min_val
            if depth_range > 1e-5:
                depth_normalized = (depth_normalized - min_val) / depth_range
            else:
                depth_normalized = np.zeros_like(depth_normalized)
        else:
            # In manual mode, preview frames may arrive as uint8/uint16-like ranges.
            # Scale to 0..1 (render path uses the clip's max content value for this).
            if final_scaling_factor and final_scaling_factor > 1.0 + 1e-5:
                depth_normalized = depth_normalized / float(final_scaling_factor)

        depth_normalized = np.clip(depth_normalized, 0, 1)

        # 3. Apply the SAME gamma math as the render path (so preview == output)
        depth_gamma_val = float(params.get("depth_gamma", 1.0))
        if round(depth_gamma_val, 2) != 1.0:
            # Math: 1.0 - (1.0 - depth)^gamma
            depth_normalized = 1.0 - np.power(
                1.0 - np.clip(depth_normalized, 0, 1), depth_gamma_val
            )
            depth_normalized = np.clip(depth_normalized, 0, 1)

        logger.debug(
            f"Final normalized depth shape: {depth_normalized.shape}, range: [{depth_normalized.min():.2f}, {depth_normalized.max():.2f}]"
        )

        # --- Perform Splatting ---
        stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()
        # Ensure depth map is resized to the target resolution (low-res or original)
        disp_map_tensor = (
            torch.from_numpy(depth_normalized).unsqueeze(0).unsqueeze(0).float().cuda()
        )

        # Resize Disparity Map to match the (potentially resized) Left Eye
        if H_target != disp_map_tensor.shape[2] or W_target != disp_map_tensor.shape[3]:
            logger.debug(f"Resizing depth map to match target {W_target}x{H_target}.")
            disp_map_tensor = F.interpolate(
                disp_map_tensor,
                size=(H_target, W_target),
                mode="bilinear",
                align_corners=False,
            )

        disp_map_tensor = (disp_map_tensor - params["convergence_point"]) * 2.0

        # Calculate disparity in pixels based on the TARGET width (W_target)
        actual_max_disp_pixels = (params["max_disp"] / 20.0 / 100.0) * W_target
        disp_map_tensor = disp_map_tensor * actual_max_disp_pixels

        # Preview-only depth/pop separation metrics (percent of screen width)
        # Uses a lightweight sampled scan of the current normalized depth map.
        try:
            if getattr(self, "previewer", None) is not None and hasattr(
                self.previewer, "set_depth_pop_metrics"
            ):
                show_metrics = bool(
                    self.depth_pop_enabled_var.get()
                ) and preview_source in (
                    "Anaglyph 3D",
                    "Dubois Anaglyph",
                    "Optimized Anaglyph",
                )
                if show_metrics:
                    _stride = 8  # sample stride for speed (1/64 pixels)
                    _sample = depth_normalized[::_stride, ::_stride]
                    _valid = (_sample > 0.001) & (_sample < 0.999)
                    if _valid.any():
                        _ds = _sample[_valid]
                        # Disparity percent of screen width (W cancels out):
                        # disp_pct = (depth - conv) * 2 * (max_disp / 20)
                        _disp_pct = (
                            (_ds - params["convergence_point"])
                            * 2.0
                            * (params["max_disp"] / 20.0)
                        )
                        _min_pct = float(_disp_pct.min())
                        _max_pct = float(_disp_pct.max())
                        _depth_pct = max(0.0, -_min_pct)  # screen-behind
                        _pop_pct = float(
                            _max_pct
                        )  # screen-out (can be negative if all behind)
                    else:
                        _depth_pct, _pop_pct = 0.0, 0.0
                    self.previewer.set_depth_pop_metrics(_depth_pct, _pop_pct)
                else:
                    self.previewer.set_depth_pop_metrics(None, None)
        except Exception:
            pass

        with torch.no_grad():
            # Use the potentially resized Left Eye
            right_eye_tensor_raw, occlusion_mask = stereo_projector(
                left_eye_tensor_resized, disp_map_tensor
            )

            # Apply low-res specific post-processing
            right_eye_tensor = right_eye_tensor_raw

        # --- Apply black borders for Anaglyph and Wigglegram ---
        if preview_source in [
            "Anaglyph 3D",
            "Dubois Anaglyph",
            "Optimized Anaglyph",
            "Wigglegram",
        ]:
            l_pct = params.get("left_border_pct", 0.0)
            r_pct = params.get("right_border_pct", 0.0)

            l_px = int(round(l_pct * W_target / 100.0))
            r_px = int(round(r_pct * W_target / 100.0))

            if l_px > 0:
                # Left eye: Opaque black border on the left side
                # left_eye_tensor_resized is (1, 3, H, W)
                left_eye_tensor_resized[:, :, :, :l_px] = 0.0
            if r_px > 0:
                # Right eye: Opaque black border on the right side
                # right_eye_tensor is (1, 3, H, W)
                right_eye_tensor[:, :, :, -r_px:] = 0.0

        if preview_source == "Splat Result" or preview_source == "Splat Result(Low)":
            final_tensor = right_eye_tensor.cpu()
        elif (
            preview_source == "Occlusion Mask"
            or preview_source == "Occlusion Mask(Low)"
        ):
            final_tensor = occlusion_mask.repeat(1, 3, 1, 1).cpu()

        elif preview_source == "Depth Map":
            # Direct grayscale view (visualization-only TV-range expansion for 10-bit depth, when tagged 'tv')
            depth_vis = depth_normalized
            try:
                if (
                    DEPTH_VIS_APPLY_TV_RANGE_EXPANSION_10BIT
                    and getattr(self, "previewer", None) is not None
                ):
                    bd = int(getattr(self.previewer, "_depth_bit_depth", 8) or 8)
                    dpath = getattr(self.previewer, "_depth_path", None)
                    if bd >= 10 and isinstance(dpath, str) and dpath:
                        if not hasattr(self, "_depth_color_range_cache"):
                            self._depth_color_range_cache = {}
                        rng = self._depth_color_range_cache.get(dpath)
                        if rng is None:
                            info = get_video_stream_info(dpath)
                            rng = str(
                                (info or {}).get("color_range")
                                or (info or {}).get("range")
                                or ""
                            ).lower()
                            self._depth_color_range_cache[dpath] = rng
                        if rng == "tv":
                            depth_vis = (depth_vis - DEPTH_VIS_TV10_BLACK_NORM) / (
                                DEPTH_VIS_TV10_WHITE_NORM - DEPTH_VIS_TV10_BLACK_NORM
                            )
            except Exception:
                pass
            depth_vis_uint8 = (np.clip(depth_vis, 0, 1) * 255).astype(np.uint8)
            depth_vis_3ch = np.stack([depth_vis_uint8] * 3, axis=-1)
            final_tensor = (
                torch.from_numpy(depth_vis_3ch).permute(2, 0, 1).unsqueeze(0).float()
                / 255.0
            )

        elif preview_source == "Depth Map (Color)":
            # Diagnostic color view (visualization-only TV-range expansion for 10-bit depth, when tagged 'tv')
            depth_vis = depth_normalized
            try:
                if (
                    DEPTH_VIS_APPLY_TV_RANGE_EXPANSION_10BIT
                    and getattr(self, "previewer", None) is not None
                ):
                    bd = int(getattr(self.previewer, "_depth_bit_depth", 8) or 8)
                    dpath = getattr(self.previewer, "_depth_path", None)
                    if bd >= 10 and isinstance(dpath, str) and dpath:
                        if not hasattr(self, "_depth_color_range_cache"):
                            self._depth_color_range_cache = {}
                        rng = self._depth_color_range_cache.get(dpath)
                        if rng is None:
                            info = get_video_stream_info(dpath)
                            rng = str(
                                (info or {}).get("color_range")
                                or (info or {}).get("range")
                                or ""
                            ).lower()
                            self._depth_color_range_cache[dpath] = rng
                        if rng == "tv":
                            depth_vis = (depth_vis - DEPTH_VIS_TV10_BLACK_NORM) / (
                                DEPTH_VIS_TV10_WHITE_NORM - DEPTH_VIS_TV10_BLACK_NORM
                            )
            except Exception:
                pass
            depth_vis_uint8 = (np.clip(depth_vis, 0, 1) * 255).astype(np.uint8)
            vis_color = cv2.applyColorMap(depth_vis_uint8, cv2.COLORMAP_VIRIDIS)
            vis_rgb = cv2.cvtColor(vis_color, cv2.COLOR_BGR2RGB)
            final_tensor = (
                torch.from_numpy(vis_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            )

        elif preview_source == "Original (Left Eye)":
            # Use the resized or original left eye depending on the low-res flag
            final_tensor = left_eye_tensor_resized.cpu()
        elif preview_source == "Anaglyph 3D":
            left_np_anaglyph = (
                left_eye_tensor_resized.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            ).astype(np.uint8)
            right_np_anaglyph = (
                right_eye_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            ).astype(np.uint8)
            left_gray_np = cv2.cvtColor(left_np_anaglyph, cv2.COLOR_RGB2GRAY)
            anaglyph_np = right_np_anaglyph.copy()
            anaglyph_np[:, :, 0] = left_gray_np
            final_tensor = (
                torch.from_numpy(anaglyph_np).permute(2, 0, 1).float() / 255.0
            ).unsqueeze(0)
        elif preview_source == "Dubois Anaglyph":
            left_np_anaglyph = (
                left_eye_tensor_resized.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            ).astype(np.uint8)
            right_np_anaglyph = (
                right_eye_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            ).astype(np.uint8)
            anaglyph_np = apply_dubois_anaglyph(left_np_anaglyph, right_np_anaglyph)
            final_tensor = (
                torch.from_numpy(anaglyph_np).permute(2, 0, 1).float() / 255.0
            ).unsqueeze(0)
        elif preview_source == "Optimized Anaglyph":
            left_np_anaglyph = (
                left_eye_tensor_resized.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            ).astype(np.uint8)
            right_np_anaglyph = (
                right_eye_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
            ).astype(np.uint8)
            anaglyph_np = apply_optimized_anaglyph(left_np_anaglyph, right_np_anaglyph)
            final_tensor = (
                torch.from_numpy(anaglyph_np).permute(2, 0, 1).float() / 255.0
            ).unsqueeze(0)
        elif preview_source == "Wigglegram":
            # Pass the resized left eye and the splatted right eye
            self.previewer._start_wigglegram_animation(
                left_eye_tensor_resized.cpu(), right_eye_tensor.cpu()
            )
            return None
        else:
            final_tensor = right_eye_tensor.cpu()

        pil_img = Image.fromarray(
            (final_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )

        del stereo_projector, disp_map_tensor, right_eye_tensor_raw, occlusion_mask
        release_cuda_memory()
        logger.debug("--- Finished Preview Processing Callback ---")
        return pil_img

    def reset_to_defaults(self):
        """Resets all GUI parameters to their default hardcoded values."""
        if not messagebox.askyesno(
            "Reset Settings",
            "Are you sure you want to reset all settings to their default values?",
        ):
            return

        self.input_source_clips_var.set("./input_source_clips")
        self.input_depth_maps_var.set("./input_depth_maps")
        self.output_splatted_var.set("./output_splatted")
        self.max_disp_var.set("20.0")
        self.process_length_var.set("-1")
        self.enable_full_res_var.set(True)
        self.batch_size_var.set("10")
        self.enable_low_res_var.set(False)
        self.pre_res_width_var.set("1920")
        self.pre_res_height_var.set("1080")
        self.low_res_batch_size_var.set("50")
        self.dual_output_var.set(False)
        self.enable_global_norm_var.set(False)
        self.zero_disparity_anchor_var.set("0.5")
        self.enable_sidecar_gamma_var.set(True)
        self.enable_sidecar_blur_dilate_var.set(True)

        self.border_manual_var.set(False)
        self.border_width_var.set("0.0")
        self.border_bias_var.set("0.0")

        self.output_crf_var.set("23")
        self.output_crf_full_var.set("23")
        self.output_crf_low_var.set("23")
        self.move_to_finished_var.set(True)

        # Ensure UI/Toggles match the new reset states
        self.toggle_processing_settings_fields()
        self._on_border_manual_toggle()
        self.on_slider_release()
        self._save_config()
        self.clear_processing_info()
        self.status_label.config(text="Settings reset to defaults.")

    def restore_finished_files(self):
        """Moves all files from 'finished' folders back to their original input folders."""
        if not messagebox.askyesno(
            "Restore Finished Files",
            "Are you sure you want to move all files from 'finished' folders back to their input directories?",
        ):
            return

        source_clip_dir = self.input_source_clips_var.get()
        depth_map_dir = self.input_depth_maps_var.get()

        is_source_dir = os.path.isdir(source_clip_dir)
        is_depth_dir = os.path.isdir(depth_map_dir)

        if not (is_source_dir and is_depth_dir):
            messagebox.showerror(
                "Restore Error",
                "Restore 'finished' operation is only applicable when Input Source Clips and Input Depth Maps are set to directories (batch mode). Please ensure current settings reflect this.",
            )
            self.status_label.config(text="Restore finished: Not in batch mode.")
            return

        finished_source_folder = os.path.join(source_clip_dir, "finished")
        finished_depth_folder = os.path.join(depth_map_dir, "finished")

        restored_count = 0
        errors_count = 0

        if os.path.isdir(finished_source_folder):
            logger.info(f"==> Restoring source clips from: {finished_source_folder}")
            for filename in os.listdir(finished_source_folder):
                src_path = os.path.join(finished_source_folder, filename)
                dest_path = os.path.join(source_clip_dir, filename)
                if os.path.isfile(src_path):
                    try:
                        shutil.move(src_path, dest_path)
                        restored_count += 1
                        logger.debug(f"Moved '{filename}' to '{source_clip_dir}'")
                    except Exception as e:
                        errors_count += 1
                        logger.error(f"Error moving source clip '{filename}': {e}")
        else:
            logger.info(
                f"==> Finished source folder not found: {finished_source_folder}"
            )

        if os.path.isdir(finished_depth_folder):
            logger.info(
                f"==> Restoring depth maps and sidecars from: {finished_depth_folder}"
            )
            for filename in os.listdir(finished_depth_folder):
                src_path = os.path.join(finished_depth_folder, filename)
                dest_path = os.path.join(depth_map_dir, filename)
                if os.path.isfile(src_path):
                    try:
                        shutil.move(src_path, dest_path)
                        restored_count += 1
                        logger.debug(f"Moved '{filename}' to '{depth_map_dir}'")
                    except Exception as e:
                        errors_count += 1
                        logger.error(
                            f"Error moving depth map/sidecar '{filename}': {e}"
                        )
        else:
            logger.info(f"==> Finished depth folder not found: {finished_depth_folder}")

        if restored_count > 0 or errors_count > 0:
            self.clear_processing_info()
            self.status_label.config(
                text=f"Restore complete: {restored_count} files moved, {errors_count} errors."
            )
            messagebox.showinfo(
                "Restore Complete",
                f"Finished files restoration attempted.\n{restored_count} files moved.\n{errors_count} errors occurred.",
            )
        else:
            self.clear_processing_info()
            self.status_label.config(text="No files found to restore.")
            messagebox.showinfo(
                "Restore Complete", "No files found in 'finished' folders to restore."
            )

    def _round_slider_variable_value(self, tk_var: tk.Variable, decimals: int):
        """Rounds the float/string value of a tk.Variable and sets it back."""
        try:
            current_value = float(tk_var.get())
            rounded_value = round(current_value, decimals)
            if current_value != rounded_value:
                tk_var.set(rounded_value)
                logger.debug(
                    f"Rounded {current_value} to {rounded_value} (decimals={decimals})"
                )
        except ValueError:
            pass

    def _run_batch_process(self, settings):
        """
        Batch processing entry point.

        In multi-file mode:
          - 'From' and 'To' are treated as 1-based indices into the *GUI list*
            (self.previewer.video_list), i.e. the same numbers you see when
            jumping between clips.
        In single-file mode:
          - The From/To fields are ignored and the single video is processed.
        """
        try:
            # --- 1. Basic setup (folder paths, discovered videos, etc.) ---
            setup_result = self._setup_batch_processing(settings)
            if "error" in setup_result:
                logger.error(setup_result["error"])
                messagebox.showerror("Batch Processing Error", setup_result["error"])
                return

            input_videos = setup_result["input_videos"]
            is_single_file_mode = setup_result["is_single_file_mode"]
            finished_source_folder = setup_result["finished_source_folder"]
            finished_depth_folder = setup_result["finished_depth_folder"]

            if not input_videos:
                logger.error("No input videos found for processing.")
                messagebox.showerror(
                    "Processing Error", "No input videos found for processing."
                )
                return

            # --- 2. Apply From/To range on the *preview list* when available ---
            # In single-file mode, we always process the one file and ignore From/To.
            if (
                not is_single_file_mode
                and hasattr(self, "previewer")
                and getattr(self.previewer, "video_list", None)
            ):
                # The previewer list is what you see in the GUI (1/XXXX, 2/XXXX, ...).
                available_entries = self.previewer.video_list
                total_videos = len(available_entries)

                # Defaults: full range
                start_index_0 = 0  # 0-based
                end_index_0 = total_videos  # exclusive

                # Parse "From" (1-based in UI)
                from_str = self.process_from_var.get().strip()
                if from_str:
                    try:
                        from_val = int(from_str)
                        if from_val > 0:
                            # convert to 0-based, clamp to bounds
                            start_index_0 = max(0, min(total_videos, from_val - 1))
                    except ValueError:
                        logger.warning(f"Invalid 'From' value '{from_str}', ignoring.")

                # Parse "To" (1-based in UI, inclusive)
                to_str = self.process_to_var.get().strip()
                if to_str:
                    try:
                        to_val = int(to_str)
                        if to_val > 0:
                            # convert to exclusive end index, clamp, and ensure at least 1 video
                            end_index_0 = max(
                                start_index_0 + 1, min(total_videos, to_val)
                            )
                    except ValueError:
                        logger.warning(f"Invalid 'To' value '{to_str}', ignoring.")

                # Log the range in GUI-style indices if we're not using the full list
                if start_index_0 > 0 or end_index_0 < total_videos:
                    logger.info(
                        f"Processing range: videos {start_index_0 + 1} to {end_index_0} "
                        f"(out of {total_videos} total)"
                    )

                # Slice the preview list and build the actual video path list
                selected_entries = available_entries[start_index_0:end_index_0]
                sliced_videos = [
                    entry.get("source_video")
                    for entry in selected_entries
                    if entry.get("source_video")
                ]

                input_videos = sliced_videos

            else:
                # Multi-file mode with no previewer/video_list: treat From/To as simple
                # 1-based indices over the discovered input_videos list (old behavior).
                # In *single-file* mode, we intentionally ignore From/To and leave
                # input_videos unchanged so the current preview clip always runs.
                if not is_single_file_mode:
                    total_videos = len(input_videos)
                    start_index_0 = 0
                    end_index_0 = total_videos

                    from_str = self.process_from_var.get().strip()
                    if from_str:
                        try:
                            from_val = int(from_str)
                            if from_val > 0:
                                start_index_0 = max(0, min(total_videos, from_val - 1))
                        except ValueError:
                            logger.warning(
                                f"Invalid 'From' value '{from_str}', ignoring."
                            )

                    to_str = self.process_to_var.get().strip()
                    if to_str:
                        try:
                            to_val = int(to_str)
                            if to_val > 0:
                                end_index_0 = max(
                                    start_index_0 + 1, min(total_videos, to_val)
                                )
                        except ValueError:
                            logger.warning(f"Invalid 'To' value '{to_str}', ignoring.")

                    if start_index_0 > 0 or end_index_0 < total_videos:
                        logger.info(
                            f"Processing range: videos {start_index_0 + 1} to {end_index_0} "
                            f"(out of {total_videos} total)"
                        )
                    input_videos = input_videos[start_index_0:end_index_0]
                # else: single-file mode -> From/To boxes are ignored on purpose

            # After applying the range, make sure we still have something to do
            if not input_videos:
                logger.error(
                    "No input videos left to process after applying From/To range."
                )
                messagebox.showerror(
                    "Processing Error",
                    "No input videos left to process after applying the From/To range.",
                )
                return

            # --- 3. Determine total tasks for the progress bar ---
            processing_tasks = self._get_defined_tasks(settings)
            if not processing_tasks:
                logger.error(
                    "No processing tasks defined. Please enable at least one output resolution."
                )
                messagebox.showerror(
                    "Processing Error",
                    "No processing tasks defined. Please enable at least one output resolution.",
                )
                return

            tasks_per_video = len(processing_tasks)
            total_tasks = len(input_videos) * tasks_per_video
            logger.info(
                f"Total tasks to process: {total_tasks} "
                f"({len(input_videos)} videos × {tasks_per_video} tasks each)"
            )
            self.progress_queue.put(("total", total_tasks))

            overall_task_counter = 0

            # --- 4. Main processing loop ---
            for idx, video_path in enumerate(input_videos):
                if self.stop_event.is_set():
                    logger.info("==> Stopping processing due to user request")
                    break

                # Delegates all per-video work to the helper
                tasks_processed, _ = self._process_single_video_tasks(
                    video_path=video_path,
                    settings=settings,
                    initial_overall_task_counter=overall_task_counter,
                    is_single_file_mode=is_single_file_mode,
                    finished_source_folder=finished_source_folder,
                    finished_depth_folder=finished_depth_folder,
                )

                overall_task_counter += tasks_processed

        except Exception as e:
            logger.error(
                f"An unexpected error occurred during batch processing: {e}",
                exc_info=True,
            )
            self.progress_queue.put(("status", f"Error: {e}"))
            error_message = str(e)
            self.after(
                0,
                lambda msg=error_message: messagebox.showerror(
                    "Processing Error",
                    f"An unexpected error occurred during batch processing: {msg}",
                ),
            )
        finally:
            release_cuda_memory()
            self.progress_queue.put("finished")
            self.after(0, self.clear_processing_info)

    def run_fusion_sidecar_generator(self):
        """Initializes and runs the FusionSidecarGenerator tool."""

        # Use an external thread to prevent the GUI from freezing during the file scan
        def worker():
            self.status_label.config(
                text="Starting Fusion Export Sidecar Generation..."
            )
            generator = FusionSidecarGenerator(self, self.sidecar_manager)
            generator.generate_sidecars()

        threading.Thread(target=worker, daemon=True).start()

    def run_fusion_sidecar_generator_custom(self):
        """Initializes and runs the FusionSidecarGenerator tool for custom sidecar export."""

        def worker():
            self.status_label.config(
                text="Starting Custom Fusion Export Sidecar Generation..."
            )
            generator = FusionSidecarGenerator(self, self.sidecar_manager)
            generator.generate_custom_sidecars()

        threading.Thread(target=worker, daemon=True).start()

    def run_preview_auto_converge(self, force_run=False):
        """
        Starts the Auto-Convergence pre-pass on the current preview clip in a thread,
        and updates the convergence slider/preview upon completion.
        'force_run=True' is used when triggered by the combo box, as validation is needed.
        """
        if not hasattr(self, "previewer") or not self.previewer.source_readers:
            if force_run:
                messagebox.showwarning(
                    "Auto-Converge Preview",
                    "Please load a video in the Previewer first.",
                )
                self.auto_convergence_combo.set("Off")  # Reset combo on fail
            return

        current_index = self.previewer.current_video_index
        if current_index == -1:
            if force_run:
                messagebox.showwarning(
                    "Auto-Converge Preview",
                    "No video is currently selected for processing.",
                )
                self.auto_convergence_combo.set("Off")  # Reset combo on fail
            return

        mode = self.auto_convergence_mode_var.get()
        if mode == "Off":
            if (
                force_run
            ):  # This should be caught by the cache check, but as a safeguard
                return
            messagebox.showwarning(
                "Auto-Converge Preview",
                "Auto-Convergence Mode must be set to 'Average' or 'Peak'.",
            )
            return

        current_source_dict = self.previewer.video_list[current_index]
        single_video_path = current_source_dict.get("source_video")
        single_depth_path = current_source_dict.get("depth_map")

        # --- NEW: Check if calculation is already done for a different mode/path ---
        is_path_mismatch = single_depth_path != self._auto_conv_cached_path
        is_cache_complete = (self._auto_conv_cache["Average"] is not None) or (
            self._auto_conv_cache["Peak"] is not None
        )

        # If running from the combo box (force_run=True) AND the cache is incomplete
        # BUT the path has changed, we must clear the cache and run.
        if force_run and is_path_mismatch and is_cache_complete:
            logger.info("New video detected. Clearing Auto-Converge cache.")
            self._auto_conv_cache = {"Average": None, "Peak": None}
            self._auto_conv_cached_path = None

        if not single_video_path or not single_depth_path:
            messagebox.showerror(
                "Auto-Converge Preview Error",
                "Could not get both video and depth map paths from previewer.",
            )
            if force_run:
                self.auto_convergence_combo.set("Off")
            return

        try:
            current_anchor = float(self.zero_disparity_anchor_var.get())
            process_length = int(self.process_length_var.get())
            batch_size = int(self.batch_size_var.get())
        except ValueError as e:
            messagebox.showerror(
                "Auto-Converge Preview Error",
                f"Invalid input for slider or process length: {e}",
            )
            if force_run:
                self.auto_convergence_combo.set("Off")
            return

        # Set running flag and disable inputs
        self._is_auto_conv_running = True
        self.btn_auto_converge_preview.config(state="disabled")
        self.start_button.config(state="disabled")
        self.start_single_button.config(state="disabled")
        self.auto_convergence_combo.config(state="disabled")  # Disable combo during run

        self.status_label.config(
            text=f"Auto-Convergence pre-pass started ({mode} mode)..."
        )

        # Start the calculation in a new thread
        worker_args = (
            single_depth_path,
            process_length,
            batch_size,
            current_anchor,
            mode,
        )
        self.auto_converge_thread = threading.Thread(
            target=self._auto_converge_worker, args=worker_args
        )
        self.auto_converge_thread.start()

    def _save_current_settings_and_notify(self):
        """Saves current GUI settings to default config file and notifies the user."""
        config_filename = self.APP_CONFIG_DEFAULTS["DEFAULT_CONFIG_FILENAME"]
        try:
            self._save_config()
            # --- MODIFIED: Use the new dictionary constant in messages ---
            self.status_label.config(text=f"Settings saved to {config_filename}.")
            messagebox.showinfo(
                "Settings Saved",
                f"Current settings successfully saved to {config_filename}.",
            )
            # --- END MODIFIED ---
        except Exception as e:
            self.status_label.config(text="Settings save failed.")
            # --- MODIFIED: Use the new dictionary constant in messages ---
            messagebox.showerror(
                "Save Error", f"Failed to save settings to {config_filename}:\n{e}"
            )

    def _save_config(self):
        """Saves current GUI settings to the default file."""
        config = self._get_current_config()
        config_filename = self.APP_CONFIG_DEFAULTS["DEFAULT_CONFIG_FILENAME"]
        with open(config_filename, "w") as f:
            json.dump(config, f, indent=4)

    def _save_current_sidecar_data(
        self,
        is_auto_save: bool = False,
        force_auto_L: Optional[float] = None,
        force_auto_R: Optional[float] = None,
    ) -> bool:
        """
        Core method to prepare data and save the sidecar file.

        Args:
            is_auto_save (bool): If True, logs are DEBUG/INFO, otherwise ERROR.
            force_auto_L (float): Explicit left auto-border value to save.
            force_auto_R (float): Explicit right auto-border value to save.

        Returns:
            bool: True on success, False on failure.
        """
        result = self._get_current_sidecar_paths_and_data()
        if result is None:
            if not is_auto_save:
                messagebox.showwarning(
                    "Sidecar Save", "Please load a video in the Previewer first."
                )
            return False

        json_sidecar_path, depth_map_path, current_data = result

        # 1. Get current GUI values (the data to override/save)
        try:
            gui_save_data = {
                "convergence_plane": self._safe_float(
                    self.zero_disparity_anchor_var, 0.5
                ),
                "max_disparity": self._safe_float(self.max_disp_var, 20.0),
                "gamma": self._safe_float(self.depth_gamma_var, 1.0),
                "depth_dilate_size_x": self._safe_float(self.depth_dilate_size_x_var),
                "depth_dilate_size_y": self._safe_float(self.depth_dilate_size_y_var),
                "depth_blur_size_x": self._safe_float(self.depth_blur_size_x_var),
                "depth_blur_size_y": self._safe_float(self.depth_blur_size_y_var),
                "depth_dilate_left": self._safe_float(self.depth_dilate_left_var),
                "depth_blur_left": int(
                    round(self._safe_float(self.depth_blur_left_var))
                ),
                "depth_blur_left_mix": self._safe_float(
                    self.depth_blur_left_mix_var, 0.5
                ),
                "selected_depth_map": self.selected_depth_map_var.get(),
            }

            # Convert Border Width/Bias to Left/Right for storage
            w = self._safe_float(self.border_width_var)
            b = self._safe_float(self.border_bias_var)
            if b <= 0:
                left_b = w
                right_b = w * (1.0 + b)
            else:
                right_b = w
                left_b = w * (1.0 - b)

            # Auto Adv values: only save if we are in that mode OR if they were forced by a scan
            final_auto_L = (
                force_auto_L
                if force_auto_L is not None
                else self._safe_float(self.auto_border_L_var)
            )
            final_auto_R = (
                force_auto_R
                if force_auto_R is not None
                else self._safe_float(self.auto_border_R_var)
            )

            # If not in Auto Adv and not forced, we might want to keep the sidecar's existing values
            # instead of overwriting with GUI 0.0s. This prevents accidental clearing.
            mode = self.border_mode_var.get()
            if mode != "Auto Adv." and force_auto_L is None:
                final_auto_L = current_data.get("auto_border_L", 0.0)
                final_auto_R = current_data.get("auto_border_R", 0.0)

            gui_save_data.update(
                {
                    "left_border": round(left_b, 3),
                    "right_border": round(right_b, 3),
                    "border_mode": mode,
                    "auto_border_L": round(final_auto_L, 3),
                    "auto_border_R": round(final_auto_R, 3),
                }
            )
        except ValueError:
            logger.error("Sidecar Save: Invalid input value in GUI. Skipping save.")
            if not is_auto_save:
                messagebox.showerror(
                    "Sidecar Error", "Invalid input value in GUI. Skipping save."
                )
            return False

        # 2. Merge GUI values into current data (preserving overlap/bias)
        current_data.update(gui_save_data)

        # 3. Write the updated data back to the file using the manager
        if self.sidecar_manager.save_sidecar_data(json_sidecar_path, current_data):
            action = (
                "Auto-Saved"
                if is_auto_save
                else ("Updated" if os.path.exists(json_sidecar_path) else "Created")
            )

            logger.info(f"{action} sidecar: {os.path.basename(json_sidecar_path)}")
            self.status_label.config(text=f"{action} sidecar.")

            # Update button text in case a file was just created
            self._update_sidecar_button_text()

            return True
        else:
            logger.error(
                f"Sidecar Save: Failed to write sidecar file '{os.path.basename(json_sidecar_path)}'."
            )
            if not is_auto_save:
                messagebox.showerror(
                    "Sidecar Error",
                    f"Failed to write sidecar file '{os.path.basename(json_sidecar_path)}'. Check logs.",
                )
            return False

    def _save_debug_image(
        self,
        data: np.ndarray,
        filename_tag: str,
        batch_index: int,
        frame_index: int,
        task_name: str,
    ):
        """Saves a normalized (0-1) NumPy array as a grayscale PNG to a debug folder."""
        if not self._debug_logging_enabled:
            return

        debug_dir = os.path.join(
            os.path.dirname(self.input_source_clips_var.get()),
            "splat_debug",
            task_name,
            "images",
        )
        os.makedirs(debug_dir, exist_ok=True)

        # Create a filename that includes frame index, batch index, and tag
        filename = os.path.join(
            debug_dir, f"{frame_index:05d}_B{batch_index:02d}_{filename_tag}.png"
        )

        try:
            # 1. Normalize data to 0-255 uint8 range for PIL
            # If data is BxHxW, take the first frame (index 0)
            if data.ndim == 3:
                frame_np = data[0]
            elif data.ndim == 4:
                frame_np = data[0].squeeze()  # Assuming Bx1xHxW or similar
            else:
                frame_np = data  # Assume HxW

            # 2. Ensure data is float 0-1 (if not already) and clip
            if frame_np.dtype != np.float32:
                # Assume raw values (e.g., 0-255) and normalize for visualization
                frame_np = (
                    frame_np.astype(np.float32) / frame_np.max()
                    if frame_np.max() > 0
                    else frame_np
                )

            frame_uint8 = (np.clip(frame_np, 0.0, 1.0) * 255).astype(np.uint8)

            # 3. Save as Grayscale PNG
            img = Image.fromarray(frame_uint8, mode="L")
            img.save(filename)

            logger.debug(
                f"Saved debug image {filename_tag} (Shape: {frame_uint8.shape}) to {os.path.basename(debug_dir)}"
            )
        except Exception as e:
            logger.error(f"Failed to save debug image {filename_tag}: {e}")

    def _save_debug_numpy(
        self,
        data: np.ndarray,
        filename_tag: str,
        batch_index: int,
        frame_index: int,
        task_name: str,
    ):
        """Saves a NumPy array to a debug folder if debug logging is enabled."""
        if not self._debug_logging_enabled:
            return

        output_path = self.output_splatted_var.get()
        debug_root = os.path.join(os.path.dirname(output_path), "splat_debug")

        # 1. Save NPZ (Existing Logic)
        debug_dir_npz = os.path.join(debug_root, task_name)
        os.makedirs(debug_dir_npz, exist_ok=True)
        filename_npz = os.path.join(
            debug_dir_npz, f"{frame_index:05d}_B{batch_index:02d}_{filename_tag}.npz"
        )
        logger.debug(f"Save path {filename_tag}")

        try:
            np.savez_compressed(filename_npz, data=data)
            logger.debug(
                f"Saved debug array {filename_tag} (Shape: {data.shape}) to {os.path.basename(debug_dir_npz)}"
            )
        except Exception as e:
            logger.error(f"Failed to save debug array {filename_tag}: {e}")

        # 2. Save PNG Image (New Logic)
        self._save_debug_image(data, filename_tag, batch_index, frame_index, task_name)

    def save_settings(self):
        """Saves current GUI settings to a user-selected JSON file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Settings to File",
        )
        if not filename:
            return

        try:
            config_to_save = self._get_current_config()
            with open(filename, "w") as f:
                json.dump(config_to_save, f, indent=4)

            messagebox.showinfo(
                "Settings Saved",
                f"Successfully saved settings to:\n{os.path.basename(filename)}",
            )
            self.status_label.config(text="Settings saved.")

        except Exception as e:
            messagebox.showerror(
                "Save Error",
                f"Failed to save settings to {os.path.basename(filename)}:\n{e}",
            )
            self.status_label.config(text="Settings save failed.")

    def _set_input_state(self, state):
        """Sets the state of all input widgets to 'normal' or 'disabled'."""

        # Helper to set the state of all children in a frame
        def set_frame_children_state(frame, state, exclude_frames=False):
            """Recursively sets the state of all configurable widgets within a frame."""
            for child in frame.winfo_children():
                child_type = child.winfo_class()

                # Check if the child is a Frame/LabelFrame that we need to recurse into
                if (
                    isinstance(child, (ttk.Frame, tk.Frame, ttk.LabelFrame))
                    and not exclude_frames
                ):
                    set_frame_children_state(child, state, exclude_frames)

                # Check for widgets that accept the 'state' configuration
                if child_type in ("TEntry", "TButton", "TCheckbutton", "TCombobox"):
                    try:
                        # Use a keyword argument to pass the state
                        child.config(state=state)
                    except tk.TclError:
                        # Some buttons/labels might throw an error if they don't support 'state' directly,
                        # but Entries, Buttons, and Checkbuttons should be fine.
                        pass

                # Special handling for labels whose colors might need adjusting if they are linked to entry/button states
                # (Not needed for simple ttk styles, but left for reference)

        # --- 1. Top-level Frames ---

        # Folder Frame (Input/Output Paths)
        set_frame_children_state(self.folder_frame, state)

        # Output Settings Frame (Max Disp, CRF, etc.)
        set_frame_children_state(self.output_settings_frame, state)

        # --- 2. Depth/Resolution Frames (Containers) ---

        # Process Resolution Frame (Left Side)
        set_frame_children_state(self.preprocessing_frame, state)

        # Depth Map Pre-processing Container (Right Side)
        set_frame_children_state(self.depth_settings_container, state)

        # --- CRITICAL FIX: Explicitly re-enable slider widgets if state is 'normal' ---
        if state == "normal" and hasattr(self, "widgets_to_disable"):
            for widget in self.widgets_to_disable:
                # ttk.Scale can use 'normal' or 'disabled'
                widget.config(state="normal")

        if hasattr(self, "update_sidecar_button"):
            if state == "disabled":
                self.update_sidecar_button.config(state="disabled")
            else:  # state == 'normal'
                # When batch is done, re-apply the sidecar override logic immediately
                self._toggle_sidecar_update_button_state()

        # 3. Re-apply the specific field enable/disable logic
        # This is CRITICAL. If we set state='normal' for everything,
        # toggle_processing_settings_fields will correctly re-disable the Low Res W/H fields
        # if the "Enable Low Resolution" checkbox is unchecked.
        if hasattr(self, "previewer"):
            self.previewer.set_ui_processing_state(state == "disabled")

        if state == "normal":
            self.toggle_processing_settings_fields()

    def _set_saved_geometry(self: "SplatterGUI"):
        """Applies the saved window width and position, with dynamic height."""
        # Ensure the window is visible and all widgets are laid out for accurate height calculation
        self.update_idletasks()

        # 1. Use the saved/default width and height, with fallbacks
        current_width = self.window_width
        saved_height = self.window_height

        # Recalculate height only if we are using the fallback default, otherwise respect saved size
        if saved_height == 750:
            calculated_height = self.winfo_reqheight()
            if calculated_height < 100:
                calculated_height = 750
            current_height = calculated_height
        else:
            current_height = saved_height

        # Fallback if saved width is invalid or too small
        if current_width < 200:  # Minimum sensible width
            current_width = 620  # Use default width

        # 2. Construct the geometry string
        geometry_string = f"{current_width}x{current_height}"
        if self.window_x is not None and self.window_y is not None:
            geometry_string += f"+{self.window_x}+{self.window_y}"
        else:
            # If no saved position, let Tkinter center it initially or place it at default
            pass  # No position appended, Tkinter will handle default placement

        # 3. Apply the geometry
        self.geometry(geometry_string)
        logger.debug(f"Applied saved geometry: {geometry_string}")

        # Store the actual width that was applied (which is current_width) for save_config
        self.window_width = current_width  # Update instance variable for save_config

    def _setup_batch_processing(self, settings):
        """
        Handles input path validation, mode determination (single file vs batch),
        and creates necessary 'finished' folders.

        Returns a dict:

            {
                "input_videos": [...],
                "is_single_file_mode": bool,
                "finished_source_folder": str or None,
                "finished_depth_folder": str or None,
            }

        or, on error:

            { "error": "message" }
        """
        input_source_clips_path = settings["input_source_clips"]
        input_depth_maps_path = settings["input_depth_maps"]
        output_splatted = settings["output_splatted"]

        is_source_file = os.path.isfile(input_source_clips_path)
        is_source_dir = os.path.isdir(input_source_clips_path)
        is_depth_file = os.path.isfile(input_depth_maps_path)
        is_depth_dir = os.path.isdir(input_depth_maps_path)

        input_videos = []
        finished_source_folder = None
        finished_depth_folder = None
        is_single_file_mode = False

        if is_source_file and is_depth_file:
            # Single-file mode
            is_single_file_mode = True
            logger.debug(
                "==> Running in single file mode. Files will not be moved to "
                "'finished' folders (unless specifically enabled in Single Process mode)."
            )
            input_videos.append(input_source_clips_path)
            os.makedirs(output_splatted, exist_ok=True)

        elif is_source_dir and is_depth_dir:
            # Batch (folder) mode
            logger.debug("==> Running in batch (folder) mode.")

            if settings["move_to_finished"]:
                finished_source_folder = os.path.join(
                    input_source_clips_path, "finished"
                )
                finished_depth_folder = os.path.join(input_depth_maps_path, "finished")
                os.makedirs(finished_source_folder, exist_ok=True)
                os.makedirs(finished_depth_folder, exist_ok=True)
                logger.debug("Finished folders enabled for batch mode.")
            else:
                logger.debug(
                    "Finished folders DISABLED by user setting. "
                    "Files will remain in input folders."
                )

            os.makedirs(output_splatted, exist_ok=True)

            video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
            for ext in video_extensions:
                input_videos.extend(
                    glob.glob(os.path.join(input_source_clips_path, ext))
                )
            input_videos = sorted(input_videos)

        else:
            msg = (
                "==> Error: Input Source Clips and Input Depth Maps must both be "
                "either files or directories. Skipping processing."
            )
            logger.error(msg)
            return {"error": msg}

        if not input_videos:
            msg = f"No video files found in {input_source_clips_path}"
            logger.error(msg)
            return {"error": msg}

        return {
            "input_videos": input_videos,
            "is_single_file_mode": is_single_file_mode,
            "finished_source_folder": finished_source_folder,
            "finished_depth_folder": finished_depth_folder,
        }

    def show_about(self):
        """Displays the 'About' message box."""
        message = (
            f"Stereocrafter Splatting (Batch) - {GUI_VERSION}\n"
            "A tool for generating right-eye stereo views from source video and depth maps.\n"
            "Based on Decord, PyTorch, and OpenCV.\n"
            "\n(C) 2024 Some Rights Reserved"
        )
        tk.messagebox.showinfo("About Stereocrafter Splatting", message)

    def show_user_guide(self):
        """Reads and displays the user guide from a markdown file in a new window."""
        # Use a path relative to the script's directory for better reliability
        guide_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "assets", "merger_gui_guide.md"
        )
        try:
            with open(guide_path, "r", encoding="utf-8") as f:
                guide_content = f.read()
        except FileNotFoundError:
            messagebox.showerror(
                "File Not Found",
                f"The user guide file could not be found at:\n{guide_path}",
            )
            return
        except Exception as e:
            messagebox.showerror(
                "Error", f"An error occurred while reading the user guide:\n{e}"
            )
            return

        # Determine colors based on current theme
        if self.dark_mode_var.get():
            bg_color, fg_color = "#2b2b2b", "white"
        else:
            # Use a standard light bg for text that's slightly different from the main window
            bg_color, fg_color = "#fdfdfd", "black"

        # Create a new Toplevel window
        guide_window = tk.Toplevel(self)
        guide_window.title("SplatterGUI - User Guide")  # Corrected title
        guide_window.geometry("600x700")
        guide_window.transient(self)  # Keep it on top of the main window
        guide_window.grab_set()  # Modal behavior
        guide_window.configure(bg=bg_color)

        text_frame = ttk.Frame(guide_window, padding="10")
        text_frame.configure(style="TFrame")  # Ensure it follows the theme
        text_frame.pack(expand=True, fill="both")

        # Apply theme colors to the Text widget
        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            relief="flat",
            borderwidth=0,
            padx=5,
            pady=1,
            font=("Segoe UI", 9),
            bg=bg_color,
            fg=fg_color,
            insertbackground=fg_color,
        )
        text_widget.insert(tk.END, guide_content)
        text_widget.config(state=tk.DISABLED)  # Make it read-only

        scrollbar = ttk.Scrollbar(
            text_frame, orient=tk.VERTICAL, command=text_widget.yview
        )
        text_widget["yscrollcommand"] = scrollbar.set

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(side=tk.LEFT, expand=True, fill="both")

        button_frame = ttk.Frame(guide_window, padding=(0, 0, 0, 10))
        button_frame.pack()
        ok_button = ttk.Button(button_frame, text="Close", command=guide_window.destroy)
        ok_button.pack(pady=2)

    def start_processing(self):
        """Starts the video processing in a separate thread."""
        self.stop_event.clear()
        self.start_button.config(state="disabled")
        self.start_single_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Starting processing...")
        # --- Disable all inputs at start ---
        self._set_input_state("disabled")

        # --- CRITICAL FIX: Explicitly disable slider widgets ---
        if hasattr(self, "widgets_to_disable"):
            for widget in self.widgets_to_disable:
                widget.config(state="disabled")

        # --- NEW: Disable previewer widgets ---
        if hasattr(self, "previewer"):
            self.previewer.set_ui_processing_state(True)
            self.previewer.cleanup()  # Release any loaded preview videos

        # Input validation for all fields
        try:
            max_disp_val = float(self.max_disp_var.get())
            if max_disp_val <= 0:
                raise ValueError("Max Disparity must be positive.")

            anchor_val = float(self.zero_disparity_anchor_var.get())
            if not (0.0 <= anchor_val <= 1.0):
                raise ValueError("Zero Disparity Anchor must be between 0.0 and 1.0.")

            if self.enable_full_res_var.get():
                full_res_batch_size_val = int(self.batch_size_var.get())
                if full_res_batch_size_val <= 0:
                    raise ValueError("Full Resolution Batch Size must be positive.")

            if self.enable_low_res_var.get():
                pre_res_w = int(self.pre_res_width_var.get())
                pre_res_h = int(self.pre_res_height_var.get())
                if pre_res_w <= 0 or pre_res_h <= 0:
                    raise ValueError(
                        "Low-Resolution Width and Height must be positive."
                    )
                low_res_batch_size_val = int(self.low_res_batch_size_var.get())
                if low_res_batch_size_val <= 0:
                    raise ValueError("Low-Resolution Batch Size must be positive.")

            if not (self.enable_full_res_var.get() or self.enable_low_res_var.get()):
                raise ValueError(
                    "At least one resolution (Full or Low) must be enabled to start processing."
                )

            # --- NEW: Depth Pre-processing Validation ---
            depth_gamma_val = float(self.depth_gamma_var.get())
            if depth_gamma_val <= 0:
                raise ValueError("Depth Gamma must be positive.")

            # Validate Dilate X/Y
            depth_dilate_size_x_val = float(self.depth_dilate_size_x_var.get())
            depth_dilate_size_y_val = float(self.depth_dilate_size_y_var.get())
            if (
                depth_dilate_size_x_val < -10.0
                or depth_dilate_size_x_val > 30.0
                or depth_dilate_size_y_val < -10.0
                or depth_dilate_size_y_val > 30.0
            ):
                raise ValueError("Depth Dilate Sizes (X/Y) must be between -10 and 30.")

            # Validate Blur X/Y
            depth_blur_size_x_val = int(float(self.depth_blur_size_x_var.get()))
            depth_blur_size_y_val = int(float(self.depth_blur_size_y_var.get()))
            if depth_blur_size_x_val < 0 or depth_blur_size_y_val < 0:
                raise ValueError("Depth Blur Sizes (X/Y) must be non-negative.")
            # Validate Dilate/Blur Left
            depth_dilate_left_val = float(self.depth_dilate_left_var.get())
            depth_blur_left_val = int(float(self.depth_blur_left_var.get()))
            if depth_dilate_left_val < 0.0 or depth_dilate_left_val > 20.0:
                raise ValueError("Dilate Left must be between 0 and 20.")
            if depth_blur_left_val < 0 or depth_blur_left_val > 20:
                raise ValueError("Blur Left must be between 0 and 20.")

        except ValueError as e:
            self.status_label.config(text=f"Error: {e}")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            return

        settings = {
            "input_source_clips": self.input_source_clips_var.get(),
            "input_depth_maps": self.input_depth_maps_var.get(),
            "output_splatted": self.output_splatted_var.get(),
            "max_disp": float(self.max_disp_var.get()),
            "process_length": int(self.process_length_var.get()),
            "enable_full_resolution": self.enable_full_res_var.get(),
            "full_res_batch_size": int(self.batch_size_var.get()),
            "enable_low_resolution": self.enable_low_res_var.get(),
            "low_res_width": int(self.pre_res_width_var.get()),
            "low_res_height": int(self.pre_res_height_var.get()),
            "low_res_batch_size": int(self.low_res_batch_size_var.get()),
            "dual_output": self.dual_output_var.get(),
            "zero_disparity_anchor": float(self.zero_disparity_anchor_var.get()),
            "enable_global_norm": self.enable_global_norm_var.get(),  # Renamed
            "match_depth_res": True,
            "move_to_finished": self.move_to_finished_var.get(),
            "output_crf": int(self.output_crf_full_var.get()),  # legacy
            "output_crf_full": int(self.output_crf_full_var.get()),
            "output_crf_low": int(self.output_crf_low_var.get()),
            # --- Depth Pre-processing & Auto-Convergence Settings ---
            "depth_gamma": depth_gamma_val,
            "depth_dilate_size_x": depth_dilate_size_x_val,
            "depth_dilate_size_y": depth_dilate_size_y_val,
            "depth_blur_size_x": depth_blur_size_x_val,
            "depth_blur_size_y": depth_blur_size_y_val,
            "depth_dilate_left": float(self.depth_dilate_left_var.get()),
            "depth_blur_left": int(round(float(self.depth_blur_left_var.get()))),
            "auto_convergence_mode": self.auto_convergence_mode_var.get(),
            "enable_sidecar_gamma": self.enable_sidecar_gamma_var.get(),
            "enable_sidecar_blur_dilate": self.enable_sidecar_blur_dilate_var.get(),
        }
        self.processing_thread = threading.Thread(
            target=self._run_batch_process, args=(settings,)
        )
        self.processing_thread.start()
        self.check_queue()

    def start_single_processing(self):
        """
        Starts processing for the single video currently loaded in the previewer.
        It runs the batch logic in single-file mode.
        """

        # --- CRITICAL FIX: Explicitly disable slider widgets ---
        if hasattr(self, "widgets_to_disable"):
            for widget in self.widgets_to_disable:
                widget.config(state="disabled")
        # --- END CRITICAL FIX ---

        if not hasattr(self, "previewer") or not self.previewer.source_readers:
            messagebox.showwarning(
                "Process Single Clip", "Please load a video in the Previewer first."
            )
            return

        current_index = self.previewer.current_video_index
        if current_index == -1:
            messagebox.showwarning(
                "Process Single Clip", "No video is currently selected for processing."
            )
            return

        # 1. Get the current single file paths
        current_source_dict = self.previewer.video_list[current_index]
        single_video_path = current_source_dict.get("source_video")
        single_depth_path = current_source_dict.get("depth_map")
        # In Multi-Map mode, ensure Single/Test processing uses the clip's sidecar-selected map file.
        if self.multi_map_var.get() and single_video_path:
            try:
                _mm_sidecar_map = self._get_sidecar_selected_map_for_video(
                    single_video_path
                )
                if _mm_sidecar_map:
                    _mm_base = self.input_depth_maps_var.get()
                    _mm_video_name = os.path.splitext(
                        os.path.basename(single_video_path)
                    )[0]
                    _mm_folder = os.path.join(_mm_base, _mm_sidecar_map)
                    _mm_mp4 = os.path.join(_mm_folder, f"{_mm_video_name}_depth.mp4")
                    _mm_npz = os.path.join(_mm_folder, f"{_mm_video_name}_depth.npz")
                    if os.path.exists(_mm_mp4):
                        single_depth_path = _mm_mp4
                    elif os.path.exists(_mm_npz):
                        single_depth_path = _mm_npz
            except Exception as _e:
                logger.debug(
                    f"[MM] Failed resolving sidecar-selected map for Single/Test: {_e}"
                )

        if not single_video_path or not single_depth_path:
            messagebox.showerror(
                "Process Single Clip Error",
                "Could not get both video and depth map paths from previewer.",
            )
            return

        # 2. Perform validation checks (copied from start_processing)
        try:
            # Full Resolution/Low Resolution checks
            if not (self.enable_full_res_var.get() or self.enable_low_res_var.get()):
                raise ValueError(
                    "At least one resolution (Full or Low) must be enabled to start processing."
                )

            # Simplified validation for speed/simplicity (relying on start_processing for full checks)
            float(self.max_disp_var.get())

        except ValueError as e:
            self.status_label.config(text=f"Error: {e}")
            messagebox.showerror("Validation Error", str(e))
            return

        # --- MODIFIED: Delayed cleanup for diagnostic parity ---
        is_test_active = self.enable_full_res_var.get() or self.enable_low_res_var.get()

        # --- MODIFIED: Auto-Switch Preview for Tests ---
        if self.map_test_var.get() or self.splat_test_var.get():
            # If a test is active, sync GUI controls from sidecar once before capture/run (keeps parity without freezing live slider changes).
            try:
                self.update_gui_from_sidecar(single_depth_path)
                # Sidecar restore may switch the map for this clip; re-read the effective depth path.
                _cur_entry = self.previewer.video_list[current_index]
                single_depth_path = _cur_entry.get("depth_map") or single_depth_path
            except Exception:
                pass

            # Auto-switch preview mode for tests:
            # - Map Test: always show Depth Map
            # - Splat Test: if Full is enabled (even with Low), show Full splat preview; if only Low is enabled, show Low splat preview
            if self.map_test_var.get():
                target_mode = "Depth Map"
            else:
                full_enabled = (
                    bool(getattr(self, "enable_full_res_var", None).get())
                    if getattr(self, "enable_full_res_var", None)
                    else False
                )
                low_enabled = (
                    bool(getattr(self, "enable_low_res_var", None).get())
                    if getattr(self, "enable_low_res_var", None)
                    else False
                )
                if (
                    full_enabled
                    or (full_enabled and low_enabled)
                    or (not low_enabled and full_enabled)
                ):
                    target_mode = "Splat Result"
                elif low_enabled and not full_enabled:
                    target_mode = "Splat Result(Low)"
                else:
                    # Fallback (shouldn't happen in normal usage)
                    target_mode = "Splat Result"
            self.preview_source_var.set(target_mode)
            self.preview_size_var.set("100%")  # Ensure 1:1 scale for capture parity
            logger.info(
                f"Test Active: Auto-switching preview to {target_mode} at 100% scale."
            )
            self.previewer.update_preview()
            # Give the previewer a moment to actually render the new mode
            self.update()
        else:
            # Standard cleanup only if NOT in test mode
            if hasattr(self, "previewer"):
                self.previewer.cleanup()
        # --- END AUTO-SWITCH ---

        # 3. Compile settings dictionary
        # We explicitly set the input paths to the single files, which forces batch logic
        # to execute in single-file mode (checking os.path.isfile).

        # --- NEW: Determine Finished Folders for Single Process (only if enabled) ---
        single_finished_source_folder = None
        single_finished_depth_folder = None

        # --- Check the new GUI variable ---
        if self.move_to_finished_var.get():
            # We assume the finished folder is in the same directory as the original input file/depth map
            single_finished_source_folder = os.path.join(
                os.path.dirname(single_video_path), "finished"
            )
            single_finished_depth_folder = os.path.join(
                os.path.dirname(single_depth_path), "finished"
            )
            os.makedirs(single_finished_source_folder, exist_ok=True)
            os.makedirs(single_finished_depth_folder, exist_ok=True)
            logger.debug(
                f"Single Process: Finished folders set to: {single_finished_source_folder}"
            )

        settings = {
            # --- OVERRIDDEN INPUTS FOR SINGLE MODE ---
            "input_source_clips": single_video_path,
            "input_depth_maps": single_depth_path,
            "output_splatted": self.output_splatted_var.get(),  # Use the batch output folder
            # --- END OVERRIDE ---
            "max_disp": float(self.max_disp_var.get()),
            "process_length": int(self.process_length_var.get()),
            "enable_full_resolution": self.enable_full_res_var.get(),
            "full_res_batch_size": int(self.batch_size_var.get()),
            "enable_low_resolution": self.enable_low_res_var.get(),
            "low_res_width": int(self.pre_res_width_var.get()),
            "low_res_height": int(self.pre_res_height_var.get()),
            "low_res_batch_size": int(self.low_res_batch_size_var.get()),
            "dual_output": self.dual_output_var.get(),
            "zero_disparity_anchor": float(self.zero_disparity_anchor_var.get()),
            "enable_global_norm": self.enable_global_norm_var.get(),  # Renamed
            "match_depth_res": True,
            "output_crf": int(self.output_crf_full_var.get()),  # legacy
            "output_crf_full": int(self.output_crf_full_var.get()),
            "output_crf_low": int(self.output_crf_low_var.get()),
            # --- Depth Pre-processing Settings ---
            "depth_gamma": float(self.depth_gamma_var.get()),
            "depth_dilate_size_x": int(float(self.depth_dilate_size_x_var.get())),
            "depth_dilate_size_y": int(float(self.depth_dilate_size_y_var.get())),
            "depth_blur_size_x": int(float(self.depth_blur_size_x_var.get())),
            "depth_blur_size_y": int(float(self.depth_blur_size_y_var.get())),
            "depth_dilate_left": float(self.depth_dilate_left_var.get()),
            "depth_blur_left": int(round(float(self.depth_blur_left_var.get()))),
            "depth_blur_left_mix": float(self.depth_blur_left_mix_var.get()),
            "auto_convergence_mode": self.auto_convergence_mode_var.get(),
            "enable_sidecar_gamma": self.enable_sidecar_gamma_var.get(),
            "enable_sidecar_blur_dilate": self.enable_sidecar_blur_dilate_var.get(),
            "single_finished_source_folder": single_finished_source_folder,
            "single_finished_depth_folder": single_finished_depth_folder,
            "move_to_finished": self.move_to_finished_var.get(),
        }

        # 4. Start the processing thread
        self.stop_event.clear()
        self.start_button.config(state="disabled")
        self.start_single_button.config(state="disabled")  # Disable single button too
        self.stop_button.config(state="normal")
        self.status_label.config(
            text=f"Starting single-clip processing for: {os.path.basename(single_video_path)}"
        )
        self._set_input_state("disabled")  # Disable all inputs

        self.processing_thread = threading.Thread(
            target=self._run_batch_process, args=(settings,)
        )
        self.processing_thread.start()
        self.check_queue()

    def stop_processing(self):
        """Sets the stop event to gracefully halt processing."""
        self.stop_event.set()
        self.status_label.config(text="Stopping...")
        self.stop_button.config(state="disabled")
        self.start_single_button.config(state="normal")
        # --- Re-enable previewer widgets on stop ---
        if hasattr(self, "previewer"):
            self.previewer.set_ui_processing_state(False)

    def _toggle_debug_logging(self):
        """Toggles debug logging and updates shared logger."""
        self._debug_logging_enabled = (
            self.debug_logging_var.get()
        )  # Get checkbutton state

        if self._debug_logging_enabled:
            new_level = logging.DEBUG
            level_str = "DEBUG"
        else:
            new_level = logging.INFO
            level_str = "INFO"

        # Call the utility function to change the root logger level
        set_util_logger_level(new_level)

        logger.info(f"Setting application logging level to: {level_str}")

    def toggle_processing_settings_fields(self):
        """Enables/disables resolution input fields and the START button based on checkbox states."""
        # Full Resolution controls
        if self.enable_full_res_var.get():
            self.entry_full_res_batch_size.config(state="normal")
            self.lbl_full_res_batch_size.config(state="normal")
        else:
            self.entry_full_res_batch_size.config(state="disabled")
            self.lbl_full_res_batch_size.config(state="disabled")

        # Low Resolution controls
        if self.enable_low_res_var.get():
            self.pre_res_width_label.config(state="normal")
            self.pre_res_width_entry.config(state="normal")
            self.pre_res_height_label.config(state="normal")
            self.pre_res_height_entry.config(state="normal")
            self.lbl_low_res_batch_size.config(state="normal")
            self.entry_low_res_batch_size.config(state="normal")
        else:
            self.pre_res_width_label.config(state="disabled")
            self.pre_res_width_entry.config(state="disabled")
            self.pre_res_height_label.config(state="disabled")
            self.pre_res_height_entry.config(state="disabled")
            self.lbl_low_res_batch_size.config(state="disabled")
            self.entry_low_res_batch_size.config(state="disabled")

        # START button enable/disable logic: Must have at least one resolution enabled
        if self.enable_full_res_var.get() or self.enable_low_res_var.get():
            self.start_button.config(state="normal")
        else:
            self.start_button.config(state="disabled")

    def _apply_preview_overlay_toggles(self):
        """Apply preview-only overlay toggles to the previewer (Crosshair + D/P)."""
        if not getattr(self, "previewer", None):
            return

        try:
            self.previewer.set_crosshair_settings(
                self.crosshair_enabled_var.get(),
                self.crosshair_white_var.get(),
                self.crosshair_multi_var.get(),
            )
        except Exception:
            pass

        try:
            getattr(self.previewer, "set_depth_pop_enabled", lambda *_: None)(
                self.depth_pop_enabled_var.get()
            )
        except Exception:
            pass

    def _toggle_sidecar_update_button_state(self):
        """
        Controls the Update Sidecar button state based on the Override Sidecar checkbox.
        """

        # Check if batch processing is currently active (easiest way is to check the stop button's state)
        is_batch_processing_active = self.stop_button.cget("state") == "normal"

        # Check if a video is currently loaded in the previewer
        is_video_loaded = (
            hasattr(self, "previewer") and self.previewer.current_video_index != -1
        )

        # If batch is active, the button MUST be disabled, regardless of override state.
        if is_batch_processing_active:
            self.update_sidecar_button.config(state="disabled")
            return

        # If a video is loaded and batch is NOT active, ENABLE the button.
        if is_video_loaded:
            self.update_sidecar_button.config(state="normal")
        else:
            self.update_sidecar_button.config(state="disabled")

    def _update_clip_state_and_text(self):
        """Combines state and text updates for the Sidecar button, run after a new video loads."""

        # 1. Update the button text (Create vs Update)
        if hasattr(self, "_update_sidecar_button_text"):
            self._update_sidecar_button_text()

        # 2. Update the button state (Normal vs Disabled by Override)
        if hasattr(self, "_toggle_sidecar_update_button_state"):
            self._toggle_sidecar_update_button_state()

        # 3. Update the info panel to reflect the currently loaded preview clip
        self._update_processing_info_for_preview_clip()

    def _update_processing_info_for_preview_clip(self):
        """Updates the 'Current Processing Information' panel to reflect the currently loaded preview clip."""
        try:
            if not getattr(self, "previewer", None):
                return

            idx = getattr(self.previewer, "current_video_index", -1)
            video_list = getattr(self.previewer, "video_list", []) or []
            if not (0 <= idx < len(video_list)):
                return

            source_dict = video_list[idx] if isinstance(video_list[idx], dict) else {}

            # Filename (the main thing you care about in practice)
            src_path = source_dict.get("source_video")
            if src_path:
                new_name = os.path.basename(src_path)
                if self.processing_filename_var.get() != new_name:
                    self.processing_filename_var.set(new_name)
            # Map display during preview:
            # - Normal mode: keep blank (map is already implied by the UI and filename prefix).
            # - Multi-Map mode: show the currently selected map label (same text as the selector),
            #   because the radio dot can be hard to see (e.g., with 3D glasses).
            try:
                if self.multi_map_var.get():
                    sel_map = (self.selected_depth_map_var.get() or "").strip()
                    if sel_map and self.processing_map_var.get() != sel_map:
                        self.processing_map_var.set(sel_map)
                else:
                    if self.processing_map_var.get() != "":
                        self.processing_map_var.set("")
            except Exception:
                # Fail closed: don't spam logs; leave whatever is currently displayed.
                pass

            # Task name: keep stable during preview
            new_task = "Preview"
            if self.processing_task_name_var.get() != new_task:
                self.processing_task_name_var.set(new_task)

            # Frames display: keep stable during preview; show total frames only
            try:
                to_val = self.previewer.frame_scrubber.cget("to")
                total_frames = int(float(to_val)) + 1 if to_val is not None else None

                if total_frames is not None:
                    new_frames = f"{total_frames}"
                    if self.processing_frames_var.get() != new_frames:
                        self.processing_frames_var.set(new_frames)
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Preview info update skipped due to error: {e}")

    def update_gui_from_sidecar(self, depth_map_path: str):
        """
        Reads the sidecar config for the given depth map path and updates the
        Convergence, Max Disparity, and Gamma sliders.
        """
        # Clear suppression flag when opening a NEW video
        # (Allow sidecar to load for the first time on new video)
        # Get current source video to track video changes (not depth map changes)
        current_source_video = None
        if (
            hasattr(self, "previewer")
            and self.previewer
            and 0 <= self.previewer.current_video_index < len(self.previewer.video_list)
        ):
            current_source_video = self.previewer.video_list[
                self.previewer.current_video_index
            ].get("source_video")

        # Clear suppression flag when opening a NEW video (not when changing maps)
        if current_source_video and current_source_video != getattr(
            self, "_last_loaded_source_video", None
        ):
            self._suppress_sidecar_map_update = False
            self._last_loaded_source_video = (
                current_source_video  # Track source video, not depth map
            )
        if not self.update_slider_from_sidecar_var.get():
            logger.debug(
                "update_gui_from_sidecar: Feature is toggled OFF. Skipping update."
            )
            return

        if not depth_map_path:
            return

        # 1. Determine sidecar path
        depth_map_basename = os.path.splitext(os.path.basename(depth_map_path))[0]
        sidecar_ext = self.APP_CONFIG_DEFAULTS["SIDECAR_EXT"]
        # Use base folder for sidecars when Multi-Map is enabled
        sidecar_folder = self._get_sidecar_base_folder()
        json_sidecar_path = os.path.join(
            sidecar_folder, f"{depth_map_basename}{sidecar_ext}"
        )
        logger.info(f"Looking for sidecar at: {json_sidecar_path}")

        if not os.path.exists(json_sidecar_path):
            logger.debug(
                f"update_gui_from_sidecar: No sidecar found at {json_sidecar_path}. Calling _on_map_selection_changed to sync preview."
            )
            # FIXED: When no sidecar, update previewer with currently-selected map
            self._on_map_selection_changed(from_sidecar=False)
            return

        # 2. Load merged config (Sidecar values merged with defaults)
        # We use merge to ensure we get a complete dictionary even if keys are missing
        sidecar_config = self.sidecar_manager.load_sidecar_data(json_sidecar_path)

        logger.debug(
            f"Updating sliders from sidecar: {os.path.basename(json_sidecar_path)}"
        )

        # 3. Update Sliders Programmatically (Requires programmatic setter/updater)

        # Convergence
        conv_val = sidecar_config.get(
            "convergence_plane", self.zero_disparity_anchor_var.get()
        )
        self.zero_disparity_anchor_var.set(conv_val)
        if self.set_convergence_value_programmatically:
            self.set_convergence_value_programmatically(conv_val)

        # Max Disparity (Simple set)
        disp_val = sidecar_config.get("max_disparity", self.max_disp_var.get())
        self.max_disp_var.set(disp_val)

        # Gamma (Simple set)
        gamma_val = sidecar_config.get("gamma", self.depth_gamma_var.get())
        self.depth_gamma_var.set(gamma_val)

        # Dilate X
        dilate_x_val = sidecar_config.get(
            "depth_dilate_size_x", self.depth_dilate_size_x_var.get()
        )
        self.depth_dilate_size_x_var.set(dilate_x_val)

        # Dilate Y
        dilate_y_val = sidecar_config.get(
            "depth_dilate_size_y", self.depth_dilate_size_y_var.get()
        )
        self.depth_dilate_size_y_var.set(dilate_y_val)

        # Blur X
        blur_x_val = sidecar_config.get(
            "depth_blur_size_x", self.depth_blur_size_x_var.get()
        )
        self.depth_blur_size_x_var.set(blur_x_val)
        # Blur Y
        blur_y_val = sidecar_config.get(
            "depth_blur_size_y", self.depth_blur_size_y_var.get()
        )
        self.depth_blur_size_y_var.set(blur_y_val)

        # Dilate Left
        dilate_left_val = sidecar_config.get(
            "depth_dilate_left", self.depth_dilate_left_var.get()
        )
        self.depth_dilate_left_var.set(dilate_left_val)

        # Blur Left (stored as integer steps; accept older float values)
        blur_left_val = sidecar_config.get(
            "depth_blur_left", self.depth_blur_left_var.get()
        )
        try:
            self.depth_blur_left_var.set(int(round(float(blur_left_val))))
        except Exception:
            self.depth_blur_left_var.set(0)

        # Blur Left H↔V balance (defaults to 0.5 for older sidecars)
        mix_val = sidecar_config.get(
            "depth_blur_left_mix", self.depth_blur_left_mix_var.get()
        )
        try:
            mix_f = float(mix_val)
            if mix_f < 0.0:
                mix_f = 0.0
            if mix_f > 1.0:
                mix_f = 1.0
            # store as one decimal string to match UI selector
            self.depth_blur_left_mix_var.set(f"{mix_f:.1f}")
        except Exception:
            self.depth_blur_left_mix_var.set("0.5")

        # Selected Depth Map (for Multi-Map mode)
        if self.multi_map_var.get():
            selected_map_val = sidecar_config.get("selected_depth_map", "")
            if selected_map_val:
                # Only log when we actually switch maps (reduces redundant messages).
                _current_sel = self.selected_depth_map_var.get()
                if selected_map_val != _current_sel:
                    logger.info(
                        f"Restoring depth map selection from sidecar: {selected_map_val}"
                    )
                    self.selected_depth_map_var.set(selected_map_val)

                # Ensure the current preview entry uses the sidecar-selected map (even if unchanged).
                try:
                    self._on_map_selection_changed(from_sidecar=True)
                except Exception as e:
                    logger.error(
                        f"Failed to apply sidecar map '{selected_map_val}': {e}"
                    )

                # Do not allow sidecar to override manual click after this point
                self._suppress_sidecar_map_update = True

        # --- Border Settings ---
        self.auto_border_L_var.set(str(sidecar_config.get("auto_border_L", 0.0)))
        self.auto_border_R_var.set(str(sidecar_config.get("auto_border_R", 0.0)))

        mode = sidecar_config.get("border_mode")
        if mode is None:
            # Migration from older sidecars
            if "manual_border" in sidecar_config:
                is_manual = sidecar_config.get("manual_border", False)
                mode = "Manual" if is_manual else "Auto Basic"
            else:
                # Fresh clip or non-configured sidecar: keep current mode
                mode = self.border_mode_var.get()

        self.border_mode_var.set(mode)

        left_b = sidecar_config.get("left_border", 0.0)
        right_b = sidecar_config.get("right_border", 0.0)

        # Convert back to width/bias for the sliders
        w = max(left_b, right_b)
        if w > 0:
            if left_b > right_b:
                b = (right_b / left_b) - 1.0
            elif right_b > left_b:
                b = 1.0 - (left_b / right_b)
            else:
                b = 0.0
        else:
            b = 0.0

        self.border_width_var.set(f"{w:.2f}")
        self.border_bias_var.set(f"{b:.2f}")

        if self.set_border_width_programmatically:
            self.set_border_width_programmatically(w)
        if self.set_border_bias_programmatically:
            self.set_border_bias_programmatically(b)

        # Ensure UI state matches restored mode
        self._on_border_mode_change()

        # --- FIX: Refresh slider labels after restoring sidecar values ---
        if hasattr(self, "slider_label_updaters"):
            for updater in self.slider_label_updaters:
                updater()

        # --- Fix: resync processing queue depth map paths after refresh ---
        if hasattr(self.previewer, "video_list") and hasattr(
            self, "resolution_output_list"
        ):
            for i, video_entry in enumerate(self.previewer.video_list):
                if i < len(self.resolution_output_list):
                    self.resolution_output_list[i].depth_map = video_entry.get(
                        "depth_map", None
                    )
        # 4. Refresh preview to show the new values
        self.on_slider_release(None)

    def _update_sidecar_button_text(self):
        """Checks if a sidecar exists for the current preview video and updates the button text."""
        is_sidecar_present = False

        if 0 <= self.previewer.current_video_index < len(self.previewer.video_list):
            current_source_dict = self.previewer.video_list[
                self.previewer.current_video_index
            ]
            depth_map_path = current_source_dict.get("depth_map")

            if depth_map_path:
                depth_map_basename = os.path.splitext(os.path.basename(depth_map_path))[
                    0
                ]
                sidecar_ext = self.APP_CONFIG_DEFAULTS["SIDECAR_EXT"]
                sidecar_folder = (
                    self._get_sidecar_base_folder()
                )  # Use proper folder for multi-map mode
                json_sidecar_path = os.path.join(
                    sidecar_folder, f"{depth_map_basename}{sidecar_ext}"
                )
                is_sidecar_present = os.path.exists(json_sidecar_path)

        button_text = "Update Sidecar" if is_sidecar_present else "Create Sidecar"
        self.update_sidecar_button.config(text=button_text)

    def update_sidecar_file(self):
        """
        Saves the current GUI values to the sidecar file after checking for user confirmation.
        """
        # 1. Get current sidecar path and data (needed for overwrite check)
        result = self._get_current_sidecar_paths_and_data()
        if result is None:
            messagebox.showwarning(
                "Sidecar Action", "Please load a video in the Previewer first."
            )
            return

        json_sidecar_path, _, _ = result
        is_sidecar_present = os.path.exists(json_sidecar_path)

        # 2. Conditional Confirmation Dialog
        if is_sidecar_present:
            title = "Overwrite Sidecar File?"
            message = (
                f"This will overwrite parameters (Convergence, Disparity, Gamma, Borders) "
                f"in the existing sidecar file:\n\n{os.path.basename(json_sidecar_path)}\n\n"
                f"Do you want to continue?"
            )
            if not messagebox.askyesno(title, message):
                self.status_label.config(text="Sidecar update cancelled.")
                return

        # 3. Call the core saving function
        if self._save_current_sidecar_data(is_auto_save=False):
            # Immediately refresh the preview to show the *effect* of the newly saved sidecar
            self.on_slider_release(None)


def compute_global_depth_stats(
    depth_map_reader: VideoReader, total_frames: int, chunk_size: int = 100
) -> Tuple[float, float]:
    """
    Computes the global min and max depth values from a depth video by reading it in chunks.
    Assumes raw pixel values that need to be scaled (e.g., from 0-255 or 0-1023 range).
    """
    logger.info(
        f"==> Starting global depth stats pre-pass for {total_frames} frames..."
    )
    global_min, global_max = np.inf, -np.inf

    for i in range(0, total_frames, chunk_size):
        current_indices = list(range(i, min(i + chunk_size, total_frames)))
        if not current_indices:
            break

        chunk_numpy_raw = depth_map_reader.get_batch(current_indices).asnumpy()

        # Handle RGB vs Grayscale depth maps
        if chunk_numpy_raw.ndim == 4:
            if chunk_numpy_raw.shape[-1] == 3:  # RGB
                chunk_numpy = chunk_numpy_raw.mean(axis=-1)
            else:  # Grayscale with channel dim
                chunk_numpy = chunk_numpy_raw.squeeze(-1)
        else:
            chunk_numpy = chunk_numpy_raw

        chunk_min = chunk_numpy.min()
        chunk_max = chunk_numpy.max()

        if chunk_min < global_min:
            global_min = chunk_min
        if chunk_max > global_max:
            global_max = chunk_max

        # draw_progress_bar(i + len(current_indices), total_frames, prefix="  Depth Stats:", suffix="Complete")

    logger.info(
        f"==> Global depth stats computed: min_raw={global_min:.3f}, max_raw={global_max:.3f}"
    )
    return float(global_min), float(global_max)


def read_video_frames(
    video_path: str,
    process_length: int,
    set_pre_res: bool,
    pre_res_width: int,
    pre_res_height: int,
    dataset: str = "open",
) -> Tuple[VideoReader, float, int, int, int, int, Optional[dict], int]:
    """
    Initializes a VideoReader for chunked reading.
    Returns: (video_reader, fps, original_height, original_width, actual_processed_height, actual_processed_width, video_stream_info, total_frames_to_process)
    """
    if dataset == "open":
        logger.debug(f"==> Initializing VideoReader for: {video_path}")
        vid_info_only = VideoReader(
            video_path, ctx=cpu(0)
        )  # Use separate reader for info
        original_height, original_width = vid_info_only.get_batch([0]).shape[1:3]
        total_frames_original = len(vid_info_only)
        logger.debug(
            f"==> Original video shape: {total_frames_original} frames, {original_height}x{original_width} per frame"
        )

        height_for_reader = original_height
        width_for_reader = original_width

        if set_pre_res and pre_res_width > 0 and pre_res_height > 0:
            height_for_reader = pre_res_height
            width_for_reader = pre_res_width
            logger.debug(
                f"==> Pre-processing resolution set to: {width_for_reader}x{height_for_reader}"
            )
        else:
            logger.debug(
                f"==> Using original video resolution for reading: {width_for_reader}x{height_for_reader}"
            )

    else:
        raise NotImplementedError(f"Dataset '{dataset}' not supported.")

    # decord automatically resizes if width/height are passed to VideoReader
    video_reader = VideoReader(
        video_path, ctx=cpu(0), width=width_for_reader, height=height_for_reader
    )

    # Verify the actual shape after Decord processing, using the first frame
    first_frame_shape = video_reader.get_batch([0]).shape
    actual_processed_height, actual_processed_width = first_frame_shape[1:3]

    fps = video_reader.get_avg_fps()  # Use actual FPS from the reader

    total_frames_available = len(video_reader)
    total_frames_to_process = total_frames_available  # Use available frames directly
    if process_length != -1 and process_length < total_frames_available:
        total_frames_to_process = process_length

    logger.debug(
        f"==> VideoReader initialized. Final processing dimensions: {actual_processed_width}x{actual_processed_height}. Total frames for processing: {total_frames_to_process}"
    )

    video_stream_info = get_video_stream_info(
        video_path
    )  # Get stream info for FFmpeg later

    return (
        video_reader,
        fps,
        original_height,
        original_width,
        actual_processed_height,
        actual_processed_width,
        video_stream_info,
        total_frames_to_process,
    )


# -----------------------------
# 10-bit+ depth decode helpers
# -----------------------------
class _NumpyBatch:
    """Minimal wrapper to match Decord's get_batch(...).asnumpy() API."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def asnumpy(self) -> np.ndarray:
        return self._arr


def _infer_depth_bit_depth(depth_stream_info: Optional[dict]) -> int:
    """Best-effort bit-depth inference from ffprobe info."""
    if not depth_stream_info:
        return 8
    pix_fmt = str(depth_stream_info.get("pix_fmt", "")).lower()
    profile = str(depth_stream_info.get("profile", "")).lower()

    # Common patterns: yuv420p10le, yuv444p12le, gray16le, etc.
    m = re.search(r"(?:p|gray)(\d+)", pix_fmt)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    if "main10" in profile or "10" in profile:
        return 10
    return 8


def _build_depth_vf(pix_fmt: str, out_w: int, out_h: int) -> str:
    pix_fmt = (pix_fmt or "").lower()
    # If it's already gray*, don't try to extract planes.
    if pix_fmt.startswith("gray"):
        return f"scale={out_w}:{out_h}:flags=bilinear,format=gray16le"
    # Otherwise, extract luma plane (Y) and convert to gray16.
    return f"extractplanes=y,scale={out_w}:{out_h}:flags=bilinear,format=gray16le"


class FFmpegDepthPipeReader:
    """
    Sequential ffmpeg-backed depth reader that preserves 10-bit+ values.
    Implements a small subset of Decord's VideoReader API used by the splatter:
      - __len__()
      - seek(idx)
      - get_batch([idx0, idx1, ...]).asnumpy()
    NOTE: This reader is optimized for sequential access (render path).
    """

    def __init__(
        self,
        path: str,
        out_w: int,
        out_h: int,
        bit_depth: int,
        num_frames: int,
        pix_fmt: str = "",
    ):
        self.path = path
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.bit_depth = int(bit_depth) if bit_depth else 16
        self._num_frames = int(num_frames) if num_frames is not None else 0
        self._pix_fmt = pix_fmt or ""
        self._proc: Optional[subprocess.Popen] = None
        self._next_index = 0
        self._frame_bytes = self.out_w * self.out_h * 2  # gray16le
        self._msb_shift: Optional[int] = None
        self._use_16_to_n_scale: bool = False
        self._start_process()

    def __len__(self) -> int:
        return self._num_frames

    def _start_process(self):
        self.close()
        vf = _build_depth_vf(self._pix_fmt, self.out_w, self.out_h)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            self.path,
            "-an",
            "-sn",
            "-dn",
            "-vframes",
            str(self._num_frames)
            if self._num_frames and self._num_frames > 0
            else "999999999",
            "-vf",
            vf,
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._next_index = 0
        self._msb_shift = None

    def close(self):
        try:
            if self._proc is not None:
                if self._proc.stdout:
                    try:
                        self._proc.stdout.close()
                    except Exception:
                        pass
                if self._proc.stderr:
                    try:
                        self._proc.stderr.close()
                    except Exception:
                        pass
                try:
                    self._proc.terminate()
                except Exception:
                    pass
        finally:
            self._proc = None

    def __del__(self):
        self.close()

    def seek(self, idx: int):
        """Best-effort seek. Efficient only if seeking forward sequentially."""
        idx = int(idx)
        if idx == self._next_index:
            return
        if idx < self._next_index:
            # Restart and fast-forward by discarding frames (rare in render path).
            self._start_process()
        # Discard until we reach idx
        to_skip = idx - self._next_index
        if to_skip <= 0:
            self._next_index = idx
            return
        if self._proc is None or self._proc.stdout is None:
            self._start_process()
        discard_bytes = to_skip * self._frame_bytes
        _ = self._proc.stdout.read(discard_bytes)
        self._next_index = idx

    def _read_exact(self, nbytes: int) -> bytes:
        if self._proc is None or self._proc.stdout is None:
            self._start_process()
        buf = b""
        while len(buf) < nbytes:
            chunk = self._proc.stdout.read(nbytes - len(buf))
            if not chunk:
                break
            buf += chunk
        return buf

    def _maybe_apply_shift(self, arr_u16: np.ndarray) -> np.ndarray:
        """Normalize decoded gray16 values back into the native bit-depth range (e.g., 0..1023 for 10-bit).

        ffmpeg's format conversion to gray16le may either:
          1) left-shift the original N-bit values into the MSBs (exact multiples of 2^(16-N)), or
          2) scale values to the full 16-bit range (0..65535).

        We detect the likely behavior once and then apply either a right-shift or a scale-down so the
        output values land back in 0..(2^bit_depth-1).
        """
        if self._msb_shift is None:
            expected_max = (
                (1 << self.bit_depth) - 1 if 0 < self.bit_depth < 16 else None
            )
            if expected_max is None:
                self._msb_shift = 0
                self._use_16_to_n_scale = False
            else:
                # Sample to avoid full-frame scans on large batches
                flat = arr_u16.reshape(-1)
                step = max(1, flat.size // 50000)
                sample = flat[::step]
                max_val = int(sample.max(initial=0))

                if max_val <= expected_max:
                    self._msb_shift = 0
                    self._use_16_to_n_scale = False
                else:
                    shift = 16 - self.bit_depth
                    if shift <= 0:
                        self._msb_shift = 0
                        self._use_16_to_n_scale = False
                    else:
                        low_mask = (1 << shift) - 1
                        # If the sample's low bits are consistently zero, it's very likely MSB-aligned (pure left shift).
                        low_bits_max = int((sample & low_mask).max(initial=0))
                        if low_bits_max == 0:
                            self._msb_shift = shift
                            self._use_16_to_n_scale = False
                        else:
                            # Otherwise assume scaled-to-16bit; scale back down into expected range.
                            self._msb_shift = 0
                            self._use_16_to_n_scale = True

        expected_max = (1 << self.bit_depth) - 1 if 0 < self.bit_depth < 16 else None
        if expected_max is None:
            return arr_u16

        if self._msb_shift and self._msb_shift > 0:
            return (arr_u16 >> self._msb_shift).astype(np.uint16)

        if self._use_16_to_n_scale:
            # Map 0..65535 -> 0..expected_max with rounding.
            arr32 = arr_u16.astype(np.uint32)
            return ((arr32 * expected_max + 32767) // 65535).astype(np.uint16)

        return arr_u16

    def get_batch(self, indices):
        indices = list(indices)
        if not indices:
            return _NumpyBatch(
                np.zeros((0, self.out_h, self.out_w, 1), dtype=np.uint16)
            )

        # Render path calls are contiguous and increasing. Enforce best-effort alignment.
        first = int(indices[0])
        if first != self._next_index:
            self.seek(first)

        n = len(indices)
        expected = self._frame_bytes * n
        buf = self._read_exact(expected)
        if len(buf) != expected:
            raise EOFError(
                f"FFmpegDepthPipeReader: expected {expected} bytes, got {len(buf)}"
            )

        arr = np.frombuffer(buf, dtype=np.uint16).reshape(n, self.out_h, self.out_w, 1)
        arr = self._maybe_apply_shift(arr)
        self._next_index = first + n
        return _NumpyBatch(arr.copy())


def load_pre_rendered_depth(
    depth_map_path: str,
    process_length: int,
    target_height: int,
    target_width: int,
    match_resolution_to_target: bool,
) -> Tuple[Any, int, int, int, Optional[dict]]:
    """
    Initializes a reader for chunked depth map reading.
    Preserves 10-bit+ depth maps by using an ffmpeg-backed reader when needed.
    No normalization or autogain is applied here.
    Returns:
        (depth_reader, total_depth_frames_to_process, actual_depth_height, actual_depth_width, depth_stream_info)
    """
    logger.debug(f"==> Initializing depth reader from: {depth_map_path}")

    depth_stream_info = get_video_stream_info(depth_map_path)
    bit_depth = _infer_depth_bit_depth(depth_stream_info)
    pix_fmt = str((depth_stream_info or {}).get("pix_fmt", ""))

    logger.info(
        f"==> Depth map stream: pix_fmt='{pix_fmt}', profile='{str((depth_stream_info or {}).get('profile', ''))}', inferred_bit_depth={bit_depth}"
    )

    if depth_map_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        # Determine total frames available (for mismatch checks) without relying on the ffmpeg pipe.
        total_depth_frames_available = 0
        try:
            _tmp = VideoReader(depth_map_path, ctx=cpu(0))
            total_depth_frames_available = len(_tmp)
            del _tmp
        except Exception as e:
            logger.warning(
                f"Could not determine depth frame count via Decord for '{depth_map_path}': {e}"
            )

        total_depth_frames_to_process = total_depth_frames_available
        if process_length != -1 and process_length < total_depth_frames_available:
            total_depth_frames_to_process = process_length

        # Choose reader implementation
        if bit_depth > 8:
            depth_reader = FFmpegDepthPipeReader(
                depth_map_path,
                out_w=target_width,
                out_h=target_height,
                bit_depth=bit_depth,
                num_frames=total_depth_frames_available,
                pix_fmt=pix_fmt,
            )
        else:
            # 8-bit: decode at native res with Decord, then (optionally) resize with OpenCV for parity.
            depth_reader = VideoReader(
                depth_map_path, ctx=cpu(0)
            )  # decode at native res; resize with OpenCV later for parity

            # --- NEW: When match_resolution_to_target=True, wrap the reader so it *outputs* frames at (target_width,target_height)
            # using OpenCV resizing (NOT Decord scaling). This keeps 8-bit preview/render parity and also enables the low-res
            # pipeline to preprocess at clip resolution (depth_target_w/h are set to original clip res for low-res tasks).
            class _NumpyBatch:
                def __init__(self, arr):
                    self._arr = arr

                def asnumpy(self):
                    return self._arr

            class _ResizingDepthReader:
                def __init__(self, inner_reader, out_w, out_h):
                    self._inner = inner_reader
                    self._out_w = int(out_w)
                    self._out_h = int(out_h)

                def __len__(self):
                    return len(self._inner)

                def seek(self, *args, **kwargs):
                    return self._inner.seek(*args, **kwargs)

                def get_batch(self, indices):
                    arr = self._inner.get_batch(indices).asnumpy()
                    # arr is typically (N,H,W,C) from Decord
                    in_h = int(arr.shape[1])
                    in_w = int(arr.shape[2])
                    if in_w == self._out_w and in_h == self._out_h:
                        return _NumpyBatch(arr)

                    interp = (
                        cv2.INTER_LINEAR
                        if (self._out_w > in_w or self._out_h > in_h)
                        else cv2.INTER_AREA
                    )

                    if arr.ndim == 4:
                        out = np.empty(
                            (arr.shape[0], self._out_h, self._out_w, arr.shape[3]),
                            dtype=arr.dtype,
                        )
                        for i in range(arr.shape[0]):
                            out[i] = cv2.resize(
                                arr[i], (self._out_w, self._out_h), interpolation=interp
                            )
                    else:
                        out = np.empty(
                            (arr.shape[0], self._out_h, self._out_w), dtype=arr.dtype
                        )
                        for i in range(arr.shape[0]):
                            out[i] = cv2.resize(
                                arr[i], (self._out_w, self._out_h), interpolation=interp
                            )

                    return _NumpyBatch(out)

            first_depth_frame_shape = depth_reader.get_batch([0]).asnumpy().shape
            actual_depth_height, actual_depth_width = first_depth_frame_shape[1:3]

            if match_resolution_to_target and (
                actual_depth_width != target_width
                or actual_depth_height != target_height
            ):
                depth_reader = _ResizingDepthReader(
                    depth_reader, out_w=target_width, out_h=target_height
                )
                actual_depth_height, actual_depth_width = (
                    int(target_height),
                    int(target_width),
                )

        first_depth_frame_shape = depth_reader.get_batch([0]).asnumpy().shape
        actual_depth_height, actual_depth_width = first_depth_frame_shape[1:3]

        logger.debug(
            f"==> Depth reader ready. Final depth resolution: {actual_depth_width}x{actual_depth_height}. "
            f"Frames available: {total_depth_frames_available}. Frames to process: {total_depth_frames_to_process}. "
            f"bit_depth={bit_depth}, pix_fmt='{pix_fmt}'."
        )

        return (
            depth_reader,
            total_depth_frames_to_process,
            actual_depth_height,
            actual_depth_width,
            depth_stream_info,
        )

    elif depth_map_path.lower().endswith(".npz"):
        logger.error(
            "NPZ support is temporarily disabled with disk chunking refactor. Please convert NPZ to MP4 depth video."
        )
        raise NotImplementedError(
            "NPZ depth map loading is not yet supported with disk chunking."
        )
    else:
        raise ValueError(
            f"Unsupported depth map format: {os.path.basename(depth_map_path)}. Only MP4 are supported with disk chunking."
        )


if __name__ == "__main__":
    CUDA_AVAILABLE = check_cuda_availability()  # Sets the global flag

    app = SplatterGUI()
    app.mainloop()
