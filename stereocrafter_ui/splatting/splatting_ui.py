import gc
import os
import cv2
import glob
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import write_video
from decord import VideoReader, cpu
import gradio as gr
import json
import threading
import queue
import subprocess
import time
import logging
from typing import Optional, Tuple, Optional, Dict, Any, List
from PIL import Image
import math
from gui.config import APP_CONFIG_DEFAULTS, GUI_VERSION
from gui.sidecar import FusionSidecarGenerator
from gui.warp import ForwardWarpStereo
from gui.utils import VideoFileClip

# Import custom modules
CUDA_AVAILABLE = False # start state, will check automaticly later
        
# --- MODIFIED IMPORT ---
from core.common.cli_utils import draw_progress_bar, set_logger_level
from core.common.gpu_utils import check_cuda_availability, release_cuda_memory, CUDA_AVAILABLE
from core.common.video_io import start_ffmpeg_pipe_process, get_video_stream_info
from core.common.image_processing import custom_blur, custom_dilate, custom_dilate_left, apply_dubois_anaglyph, apply_optimized_anaglyph
from core.common.sidecar_manager import SidecarConfigManager
from core.ui.widgets import Tooltip, create_single_slider_with_label_updater, create_dual_slider_layout
from core.ui.video_previewer import VideoPreviewer

logger = logging.getLogger(__name__)

try:
    from Forward_Warp import forward_warp
    logger.info("CUDA Forward Warp is available.")
except:
    from dependency.forward_warp_pytorch import forward_warp
    logger.info("Forward Warp Pytorch is active.")

# Import core modules for advanced features
try:
    from core.splatting import (
        ForwardWarpStereo as CoreForwardWarpStereo,
        ConvergenceEstimatorWrapper,
        BorderScanner,
        BatchProcessor,
        ProcessingSettings,
        ProcessingTask,
        BatchSetupResult,
    )
    from core.splatting.depth_processing import (
        DEPTH_VIS_TV10_BLACK_NORM,
        DEPTH_VIS_TV10_WHITE_NORM,
        _infer_depth_bit_depth,
    )
    from core.splatting.config_manager import ConfigManager
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False
    logger.warning("Core modules not available. Some advanced features will be disabled.")


class SplatterWebUI:
    # --- GLOBAL CONFIGURATION DICTIONARY ---
    APP_CONFIG_DEFAULTS = APP_CONFIG_DEFAULTS
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
        self.app_config = {}
        self.help_texts = {}
        self.sidecar_manager = SidecarConfigManager()
        
        # Initialize core modules if available
        if CORE_MODULES_AVAILABLE:
            self.config_manager = ConfigManager()
            self.convergence_estimator = ConvergenceEstimatorWrapper()
            self.border_scanner = BorderScanner(gui_context=self) if hasattr(self, '__class__') else None
        else:
            self.config_manager = None
            self.convergence_estimator = None
            self.border_scanner = None

        # --- NEW CACHE AND STATE ---
        self._auto_conv_cache = {"Average": None, "Peak": None}
        self._auto_conv_cached_path = None
        self._is_auto_conv_running = False
        self._preview_debounce_timer = None 
        self.slider_label_updaters = [] 
        self.set_convergence_value_programmatically = None
        self._clip_norm_cache = {} 
        self._gn_warning_shown = False
        
        # Cache: estimated per-clip max Total(D+P) keyed by signature
        self._dp_total_est_cache = {}
        # Cache: measured (render-time) per-clip max Total(D+P) keyed by signature
        self._dp_total_true_cache = {}
        self._dp_total_true_active_sig = None
        self._dp_total_true_active_val = None
        # Cache: AUTO-PASS CSV rows (optional) keyed by depth_map basename
        self._auto_pass_csv_cache = None
        self._auto_pass_csv_path = None

        self._load_config()
        self._load_help_texts()
        
        # --- Variables with defaults ---
        defaults = self.APP_CONFIG_DEFAULTS # Convenience variable

        # Initialize all the parameters as instance variables
        self.input_source_clips = self.app_config.get("input_source_clips", "./input_source_clips")
        self.input_depth_maps = self.app_config.get("input_depth_maps", "./input_depth_maps")
        self.multi_map = False
        self.selected_depth_map = ""
        self.depth_map_subfolders = []  # List of valid subfolders
        self.depth_map_radio_buttons = []         # keep list for UI management
        self.depth_map_radio_dict = {}            # NEW: map text->widget
        self._current_video_sidecar_map = None  # Track sidecar's selected map
        self._suppress_sidecar_map_update = False  # Prevent overwriting manual selections
        self._last_loaded_source_video = None  # Track source video for NEW video detection
        self.output_splatted = self.app_config.get("output_splatted", "./output_splatted")

        self.max_disp = float(self.app_config.get("max_disp", defaults["MAX_DISP"]))
        self.process_length = int(self.app_config.get("process_length", defaults["PROC_LENGTH"]))
        self.process_from = ""
        self.process_to = ""
        self.batch_size = int(self.app_config.get("batch_size", defaults["BATCH_SIZE_FULL"]))
        
        self.dual_output = False
        self.enable_global_norm = False 
        self.enable_full_res = True
        self.enable_low_res = True
        self.pre_res_width = int(self.app_config.get("pre_res_width", "1920"))
        self.pre_res_height = int(self.app_config.get("pre_res_height", "1080"))
        self.low_res_batch_size = int(self.app_config.get("low_res_batch_size", defaults["BATCH_SIZE_LOW"]))
        self.zero_disparity_anchor = float(self.app_config.get("convergence_point", defaults["CONV_POINT"]))
        self.output_crf = int(self.app_config.get("output_crf", defaults["CRF_OUTPUT"]))
        self.output_crf_full = int(self.app_config.get("output_crf_full", defaults["CRF_OUTPUT"]))
        self.output_crf_low = int(self.app_config.get("output_crf_low", defaults["CRF_OUTPUT"]))
        self.move_to_finished = True

        self.auto_convergence_mode = "Off"

        # --- Depth Pre-processing Variables ---
        self.depth_gamma = float(self.app_config.get("depth_gamma", defaults["DEPTH_GAMMA"]))
        self.depth_dilate_size_x = float(self.app_config.get("depth_dilate_size_x", defaults["DEPTH_DILATE_SIZE_X"]))
        self.depth_dilate_size_y = float(self.app_config.get("depth_dilate_size_y", defaults["DEPTH_DILATE_SIZE_Y"]))
        self.depth_blur_size_x = int(float(self.app_config.get("depth_blur_size_x", defaults["DEPTH_BLUR_SIZE_X"])))
        self.depth_blur_size_y = int(float(self.app_config.get("depth_blur_size_y", defaults["DEPTH_BLUR_SIZE_Y"])))
        
        # --- NEW: Left-edge Depth Pre-processing Variables ---
        self.depth_dilate_left = float(self.app_config.get("depth_dilate_left", defaults.get("DEPTH_DILATE_LEFT", "0")))
        self.depth_blur_left = int(float(self.app_config.get("depth_blur_left", defaults.get("DEPTH_BLUR_LEFT", "0"))))
        self.depth_blur_left_mix = float(self.app_config.get("depth_blur_left_mix", defaults.get("DEPTH_BLUR_LEFT_MIX", "0.5")))
        
        # --- NEW: Border Control Variables ---
        self.border_width = float(self.app_config.get("border_width", defaults.get("BORDER_WIDTH", "0.0")))
        self.border_bias = float(self.app_config.get("border_bias", defaults.get("BORDER_BIAS", "0.0")))
        self.border_mode = self.app_config.get("border_mode", defaults.get("BORDER_MODE", "Off"))
        self.auto_border_L = float(self.app_config.get("auto_border_L", defaults.get("AUTO_BORDER_L", "0.0")))
        self.auto_border_R = float(self.app_config.get("auto_border_R", defaults.get("AUTO_BORDER_R", "0.0")))
        
        # --- NEW: Color Tags Mode ---
        self.color_tags_mode = self.app_config.get("color_tags_mode", "Auto")
        
        # --- NEW: Preview Overlay Toggles ---
        self.crosshair_enabled = False
        self.crosshair_white = False
        self.crosshair_multi = False
        self.depth_pop_enabled = False
        
        # --- NEW: Dev Tools Toggles ---
        self.skip_lowres_preproc = False
        self.track_dp_total_true_on_render = False
        self.splat_test = False
        self.map_test = False
        
        # --- NEW: Dark Mode ---
        self.dark_mode = self.app_config.get("dark_mode_enabled", False)
        
        # --- NEW: Sidecar Control Toggle Variables ---
        self.enable_sidecar_gamma = True
        self.enable_sidecar_blur_dilate = True
        self.update_slider_from_sidecar = True
        self.auto_save_sidecar = False

        # --- NEW: Previewer Variables ---
        self.preview_source = "Splat Result"
        self.preview_size = self.app_config.get("preview_size", "75%")

        # --- Variables for "Current Processing Information" display ---
        self.processing_filename = "N/A"
        self.processing_resolution = "N/A"
        self.processing_frames = "N/A"
        self.processing_disparity = "N/A"
        self.processing_convergence = "N/A"
        self.processing_task_name = "N/A"
        self.processing_gamma = "N/A"
        self.processing_map = "N/A"

        self.slider_label_updaters = []
        
        self.widgets_to_disable = [] 

        # --- Processing control variables ---
        self.stop_event = threading.Event()
        self.progress_queue = queue.Queue()
        self.processing_thread = None
        
        # Initialize batch processor if core modules available
        if CORE_MODULES_AVAILABLE:
            self.batch_processor = BatchProcessor(
                progress_queue=self.progress_queue,
                stop_event=self.stop_event,
                sidecar_manager=self.sidecar_manager,
            )
        else:
            self.batch_processor = None

        self._configure_logging() # Ensure this call is still present

        # --- NEW: Add slider release binding for preview updates ---
        # We will add this to the sliders in _create_widgets
        self.slider_widgets = []
        
        # --- Debug logging toggle ---
        self._debug_logging_enabled = False

    def _adjust_window_height_for_content(self):
        """Adjusts the window height to fit the current content, preserving user-set width."""
        # This is for the web UI, so we don't need to adjust window height
        pass

    def _auto_converge_worker(self, depth_map_path, process_length, batch_size, fallback_value, mode):
        """Worker thread for running the Auto-Convergence calculation."""
        
        # Run the existing auto-convergence logic (no mode parameter needed now)
        new_anchor_avg, new_anchor_peak = self._determine_auto_convergence(
            depth_map_path,
            process_length,
            batch_size,
            fallback_value,
        )
        
        # Use self.after to safely update the GUI from the worker thread
        # For Gradio, we need to handle this differently
        self._complete_auto_converge_update(
            new_anchor_avg, 
            new_anchor_peak, 
            fallback_value, 
            mode # Still pass the current mode to know which value to select immediately
        )

    def _auto_save_current_sidecar(self):
        """
        Saves the current GUI values to the sidecar file without user interaction.
        Only runs if self.auto_save_sidecar_var is True.
        """
        if not self.auto_save_sidecar:
            return
            
        self._save_current_sidecar_data(is_auto_save=True)

    def _compute_clip_global_depth_stats(self, depth_map_path: str, chunk_size: int = 100) -> Tuple[float, float]:
        """
        [NEW HELPER] Computes the global min and max depth values from a depth video 
        by reading it in chunks. Used only for the preview's GN cache.
        """
        logger.info(f"==> Starting clip-local depth stats pre-pass for {os.path.basename(depth_map_path)}...")
        global_min, global_max = np.inf, -np.inf

        try:
            temp_reader = VideoReader(depth_map_path, ctx=cpu(0))
            total_frames = len(temp_reader)
            
            if total_frames == 0:
                 logger.error("Depth reader found 0 frames for global stats.")
                 return 0.0, 1.0 # Fallback

            for i in range(0, total_frames, chunk_size):
                if self.stop_event.is_set():
                    logger.warning("Global stats scan stopped by user.")
                    return 0.0, 1.0
                    
                current_indices = list(range(i, min(i + chunk_size, total_frames)))
                chunk_numpy_raw = temp_reader.get_batch(current_indices).asnumpy()
                
                # Handle RGB vs Grayscale depth maps
                if chunk_numpy_raw.ndim == 4:
                    if chunk_numpy_raw.shape[-1] == 3: # RGB
                        chunk_numpy = chunk_numpy_raw.mean(axis=-1)
                    else: # Grayscale with channel dim
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

            logger.info(f"==> Clip-local depth stats computed: min_raw={global_min:.3f}, max_raw={global_max:.3f}")
            
            # Cache the result before returning
            self._clip_norm_cache[depth_map_path] = (float(global_min), float(global_max))
            
            return float(global_min), float(global_max)

        except Exception as e:
            logger.error(f"Error during clip-local depth stats scan for preview: {e}")
            return 0.0, 1.0 # Fallback
        finally:
             gc.collect()

    def _on_multi_map_toggle(self, multi_map_state):
        """Called when Multi-Map checkbox is toggled."""
        self.multi_map = multi_map_state
        if self.multi_map:
            # Multi-Map enabled - scan for subfolders
            self._scan_depth_map_folders()
        else:
            # Multi-Map disabled - clear radio buttons
            self._clear_depth_map_radio_buttons()
            self.selected_depth_map = ""
        return self.depth_map_subfolders if self.multi_map else []

    def _on_depth_map_folder_changed(self, input_depth_maps_path):
        """Called when the Input Depth Maps folder path changes."""
        self.input_depth_maps = input_depth_maps_path
        if self.multi_map:
            # Re-scan if Multi-Map is enabled
            self._scan_depth_map_folders()
        return self.depth_map_subfolders

    def _scan_depth_map_folders(self):
        """Scans the Input Depth Maps folder for subfolders containing *_depth.mp4 files."""
        base_folder = self.input_depth_maps

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
            self.selected_depth_map = self.depth_map_subfolders[0]
        else:
            logger.warning("No valid depth map subfolders found")
            self.selected_depth_map = ""

    def _clear_depth_map_radio_buttons(self):
        """Removes all depth map radio buttons from the GUI."""
        self.depth_map_radio_buttons = []

    def _on_map_selection_changed(self, selected_map, from_sidecar=False):
        """
        Called when the user changes the depth map selection (radio buttons),
        or when a sidecar restores a map (from_sidecar=True).

        In Multi-Map mode this now ONLY updates the CURRENT video's depth map
        path instead of iterating over every video.
        """
        logger.info(f"Depth map selection changed. from_sidecar={from_sidecar}")
        if not from_sidecar:
            # User clicked a radio button – suppress sidecar overwrites
            self._suppress_sidecar_map_update = True

        # Compute the folder for the newly selected map
        new_depth_folder = self._get_effective_depth_map_folder()

        # If there is no previewer / no videos, nothing to do
        # For now, we'll just update the variable
        self.selected_depth_map = selected_map

        # Only log for the current video, and only if it's missing
        logger.info(f"Selected depth map: {selected_map}")

        return selected_map

    def _get_effective_depth_map_folder(self, base_folder=None):
        """Returns the effective depth map folder based on Multi-Map settings.
    
        Args:
            base_folder: Optional override for base folder (used during processing)
    
        Returns:
            str: The folder path to use for depth maps
        """
        if base_folder is None:
            base_folder = self.input_depth_maps
    
        # If the user has selected a single depth MAP FILE, treat its directory as the folder.
        if base_folder and os.path.isfile(base_folder):
            base_folder = os.path.dirname(base_folder)
    
        if self.multi_map and self.selected_depth_map.strip():
            # Multi-Map is enabled and a subfolder is selected
            return os.path.join(base_folder, self.selected_depth_map.strip())
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
        if self.multi_map:
            # Multi-Map mode: store sidecars in 'sidecars' subfolder
            base_folder = self.input_depth_maps
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
            sidecar_path = os.path.join(sidecar_folder, f"{video_name}_depth{sidecar_ext}")

            if not os.path.exists(sidecar_path):
                return None

            sidecar_config = self.sidecar_manager.load_sidecar_data(sidecar_path) or {}
            selected_map_val = sidecar_config.get("selected_depth_map", "")
            if selected_map_val:
                return selected_map_val

        except Exception as e:
            logger.error(f"Error reading sidecar map for {video_path}: {e}")

        return None

    def _complete_auto_converge_update(self, new_anchor_avg: float, new_anchor_peak: float, fallback_value: float, mode: str):
        """
        Safely updates the GUI and preview after Auto-Convergence worker is done.
        
        Now receives both calculated values.
        """
        # Re-enable inputs
        self._is_auto_conv_running = False

        if self.stop_event.is_set():
            logger.info("Auto-Converge pre-pass was stopped.")
            self.stop_event.clear()
            return

        # Check if EITHER calculation yielded a result different from the fallback
        if new_anchor_avg != fallback_value or new_anchor_peak != fallback_value:
            
            # 1. Cache BOTH results
            self._auto_conv_cache["Average"] = new_anchor_avg
            self._auto_conv_cache["Peak"] = new_anchor_peak
            
            # CRITICAL: Store the path of the file that was just scanned
            # For now, we'll just update the variable
            self._auto_conv_cached_path = "current_depth_path"  # This would be updated based on current video
            
            # 2. Determine which value to apply immediately (based on the current 'mode' selection)
            anchor_to_apply = new_anchor_avg if mode == "Average" else new_anchor_peak
            
            # 3. Update the variable and refresh the slider/label
            self.zero_disparity_anchor = anchor_to_apply

            logger.info(f"Auto-Converge: Avg Cached at {new_anchor_avg:.2f}, Peak Cached at {new_anchor_peak:.2f}. Applied: {mode} ({anchor_to_apply:.2f})")
            
        else:
            # Calculation failed (both returned fallback)
            logger.info(f"Auto-Converge: Failed to find a valid anchor. Value remains {fallback_value:.2f}")

    def _configure_logging(self):
        """Sets the logging level based on debug_mode_var."""
        # Use set_logger_level from core.common.cli_utils
        set_logger_level(logger, logging.INFO) # Default to INFO
        logger.info(f"Logging level set.")

    def _scan_for_preview_videos(self):
        """Scans the output folder for splatted videos to preview."""
        output_folder = self.output_splatted
        logger.info(f"Scanning for preview videos in: {output_folder}")
        
        if not os.path.exists(output_folder):
            logger.warning(f"Output folder does not exist: {output_folder}")
            return []
        
        # Find all MP4 files
        all_mp4s = sorted(glob.glob(os.path.join(output_folder, "**", "*.mp4"), recursive=True))
        logger.info(f"Found {len(all_mp4s)} total MP4 files in output folder")
        
        # Filter for splatted videos (both dual and quad outputs)
        splatted_videos = [v for v in all_mp4s if ('_splatted2' in os.path.basename(v) or 
                                                   '_splatted4' in os.path.basename(v))]
        logger.info(f"Found {len(splatted_videos)} splatted videos")
        
        if len(splatted_videos) == 0 and len(all_mp4s) > 0:
            logger.warning(f"Found MP4 files but none match splatted pattern. Example files: {[os.path.basename(v) for v in all_mp4s[:3]]}")
        
        # Return just the basenames for the dropdown
        video_names = [os.path.basename(v) for v in splatted_videos]
        return video_names

    def on_video_select(self, video_name):
        """Handle video selection to update slider range"""
        if not video_name or not self.output_splatted:
            return gr.Slider(value=0, maximum=1), "0", "0", "Ready"
            
        # Search for the video file in output folder and subfolders
        video_path = None
        for root, dirs, files in os.walk(self.output_splatted):
            if video_name in files:
                video_path = os.path.join(root, video_name)
                break
        
        if not video_path or not os.path.exists(video_path):
            return gr.Slider(value=0, maximum=1), "Error", "0", f"File not found: {video_name}"
             
        try:
            reader = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(reader)
            return gr.Slider(value=0, maximum=total_frames - 1, step=1), str(total_frames), "0", f"Loaded {video_name}"
        except Exception as e:
            return gr.Slider(value=0, maximum=1), "Error", "0", f"Error loading video: {e}"

    def _get_preview_frame(self, video_name, preview_mode, frame_index):
        """Generate preview image based on current settings"""
        try:
            if not video_name or not self.output_splatted:
                return None, "No video selected", "0"

            # Search for the video file in output folder and subfolders
            video_path = None
            for root, dirs, files in os.walk(self.output_splatted):
                if video_name in files:
                    video_path = os.path.join(root, video_name)
                    break
            
            if not video_path or not os.path.exists(video_path):
                return None, f"Video not found: {video_name}", "0"
            
            # Load video info
            reader = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(reader)
            frame_index = int(frame_index)
            frame_index = max(0, min(frame_index, total_frames - 1))
            
            # Load frame
            frame = reader.get_batch([frame_index]).asnumpy()[0]  # [H, W, C]
            
            # Determine if dual or quad
            height, width = frame.shape[0], frame.shape[1]
            basename = os.path.basename(video_path)
            is_dual = '_splatted2' in basename
            
            if is_dual:
                # Dual: Left is mask, Right is warped
                half_w = width // 2
                mask_frame = frame[:, :half_w, :]
                warped_frame = frame[:, half_w:, :]
                source_frame = warped_frame  # No separate source in dual output
                depth_frame = mask_frame  # No separate depth in dual output
            else:
                # Quad: TL=source, TR=depth, BL=mask, BR=warped
                half_h, half_w = height // 2, width // 2
                source_frame = frame[:half_h, :half_w, :]
                depth_frame = frame[:half_h, half_w:, :]
                mask_frame = frame[half_h:, :half_w, :]
                warped_frame = frame[half_h:, half_w:, :]

            # Generate preview based on mode
            if preview_mode == 'source':
                preview_np = source_frame
            elif preview_mode == 'warped':
                preview_np = warped_frame
            elif preview_mode == 'depth':
                preview_np = depth_frame
            elif preview_mode == 'mask':
                preview_np = mask_frame
            elif preview_mode == 'wiggle':
                # Create a 2-frame GIF for wiggling
                from PIL import Image as PILImage
                
                # Convert both to PIL
                img_left = PILImage.fromarray(source_frame)
                img_right = PILImage.fromarray(warped_frame)

                # Create temp file for GIF
                temp_dir = os.path.join(self.output_splatted, ".preview_temp")
                os.makedirs(temp_dir, exist_ok=True)
                temp_gif_path = os.path.join(temp_dir, f"preview_wiggle_{basename}_{frame_index}.gif")

                # Save GIF
                img_left.save(
                    temp_gif_path,
                    save_all=True,
                    append_images=[img_right],
                    duration=150,  # 150ms per frame
                    loop=0
                )

                # Return the path to the GIF
                return temp_gif_path, basename, str(frame_index)

            elif preview_mode == 'anaglyph':
                # Standard Red-Cyan Anaglyph
                left = source_frame
                right = warped_frame
                rows, cols, _ = left.shape
                anaglyph = np.zeros((rows, cols, 3), dtype=np.uint8)
                anaglyph[:, :, 0] = left[:, :, 0]  # Red from Left
                anaglyph[:, :, 1] = right[:, :, 1]  # Green from Right
                anaglyph[:, :, 2] = right[:, :, 2]  # Blue from Right
                preview_np = anaglyph
            else:
                # Fallback to source
                preview_np = source_frame

            # Convert to PIL Image
            from PIL import Image as PILImage
            preview_pil = PILImage.fromarray(preview_np)

            return preview_pil, basename, str(frame_index)

        except Exception as e:
            logger.error(f"Preview generation error: {e}")
            return None, f"Error: {str(e)}", "0"


    def depthSplatting(
            self,
            input_video_reader,
            depth_map_reader,
            total_frames_to_process,
            processed_fps,
            output_video_path_base,
            target_output_height,
            target_output_width,
            max_disp,
            process_length,
            batch_size,
            dual_output,
            zero_disparity_anchor_val,
            video_stream_info,
            input_bias,
            assume_raw_input, 
            global_depth_min, 
            global_depth_max,  
            depth_stream_info,
            user_output_crf = None,
            is_low_res_task = False,
            depth_gamma = 1.0,
            depth_dilate_size_x = 0.0,
            depth_dilate_size_y = 0.0,
            depth_blur_size_x = 0.0,
            depth_blur_size_y = 0.0,
        ):
        logger.debug("==> Initializing ForwardWarpStereo module")
        stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

        num_frames = total_frames_to_process
        height, width = target_output_height, target_output_width
        os.makedirs(os.path.dirname(output_video_path_base), exist_ok=True)
        
        # --- Determine output grid dimensions and final path ---
        grid_height, grid_width = (height, width * 2) if dual_output else (height * 2, width * 2)
        suffix = "_splatted2" if dual_output else "_splatted4"
        res_suffix = f"_{width}"
        final_output_video_path = f"{os.path.splitext(output_video_path_base)[0]}{res_suffix}{suffix}.mp4"

        # --- Start FFmpeg pipe process ---
        ffmpeg_process = start_ffmpeg_pipe_process(
            content_width=grid_width,
            content_height=grid_height,
            final_output_mp4_path=final_output_video_path,
            fps=processed_fps,
            video_stream_info=video_stream_info,
            user_output_crf=user_output_crf,
            output_format_str="splatted_grid" # Pass a placeholder for the new argument
        )
        if ffmpeg_process is None:
            logger.error("Failed to start FFmpeg pipe. Aborting splatting task.")
            return False

        # --- Determine max_expected_raw_value for consistent Gamma ---
        max_expected_raw_value = 1.0
        depth_pix_fmt = depth_stream_info.get("pix_fmt") if depth_stream_info else None
        depth_profile = depth_stream_info.get("profile") if depth_stream_info else None
        is_source_10bit = False
        if depth_pix_fmt:
            if "10" in depth_pix_fmt or "gray10" in depth_pix_fmt or "12" in depth_pix_fmt or (depth_profile and "main10" in depth_profile):
                is_source_10bit = True
        if is_source_10bit:
            max_expected_raw_value = 1023.0
        elif depth_pix_fmt and ("8" in depth_pix_fmt or depth_pix_fmt in ["yuv420p", "yuv422p", "yuv444p"]):
             max_expected_raw_value = 255.0
        elif isinstance(depth_pix_fmt, str) and "float" in depth_pix_fmt:
            max_expected_raw_value = 1.0
        logger.debug(f"Determined max_expected_raw_value: {max_expected_raw_value:.1f} (Source: {depth_pix_fmt}/{depth_profile})")

        frame_count = 0
        encoding_successful = True # Assume success unless an error occurs

        try:
            for i in range(0, num_frames, batch_size):
                t_start_batch = time.perf_counter() # <--- TIMER START: Total Batch
                if self.stop_event.is_set() or ffmpeg_process.poll() is not None:
                    if ffmpeg_process.poll() is not None:
                        logger.error("FFmpeg process terminated unexpectedly. Stopping frame processing.")
                    else:
                        logger.warning("Stop event received. Terminating FFmpeg process.")
                    encoding_successful = False
                    break

                # --- TIMER 1: Video/Depth I/O (Disk/Decode/Resize) ---
                t_start_io = time.perf_counter()

                current_frame_indices = list(range(i, min(i + batch_size, num_frames)))
                if not current_frame_indices:
                    break

                batch_frames_numpy = input_video_reader.get_batch(current_frame_indices).asnumpy()
                # This often resolves issues where Decord/FFmpeg loses the internal stream position
                try:
                    # Seek to the first frame of the current batch
                    depth_map_reader.seek(current_frame_indices[0]) 
                    # Then read the full batch from that position
                    batch_depth_numpy_raw = depth_map_reader.get_batch(current_frame_indices).asnumpy()
                except Exception as e:
                    logger.error(f"Error seeking/reading depth map batch starting at index {i}: {e}. Falling back to a potentially blank read.")
                    batch_depth_numpy_raw = depth_map_reader.get_batch(current_frame_indices).asnumpy()
                t_end_io = time.perf_counter()
                
                file_frame_idx = current_frame_indices[0] 
                task_name = "LowRes" if is_low_res_task else "HiRes"
                
                if batch_depth_numpy_raw.min() == batch_depth_numpy_raw.max() == 0:
                    logger.warning(f"Depth map batch starting at index {i} is entirely blank/zero after read. **Seeking failed to resolve.**")
                    
                if batch_depth_numpy_raw.min() == batch_depth_numpy_raw.max():
                    logger.warning(f"Depth map batch starting at index {i} is entirely uniform/flat after read. Min/Max: {batch_depth_numpy_raw.min():.2f}")

                # Use the FIRST frame index for the file name (e.g., 00000.png)
                file_frame_idx = current_frame_indices[0] 
                
                # self._save_debug_numpy(batch_depth_numpy_raw, "01_RAW_INPUT", i, file_frame_idx, task_name) 
                # --- TIMER 2: CPU Pre-processing (Dilate, Blur, Grayscale, Gamma, Min/Max Calc) ---
                t_start_preproc = time.perf_counter()
                
                batch_depth_numpy = self._process_depth_batch(
                    batch_depth_numpy_raw=batch_depth_numpy_raw,
                    depth_stream_info=depth_stream_info,
                    depth_gamma=depth_gamma,
                    depth_dilate_size_x=depth_dilate_size_x,
                    depth_dilate_size_y=depth_dilate_size_y,
                    depth_blur_size_x=depth_blur_size_x,
                    depth_blur_size_y=depth_blur_size_y,
                    is_low_res_task=is_low_res_task,
                    max_raw_value=max_expected_raw_value,
                    global_depth_min=global_depth_min,
                    global_depth_max=global_depth_max,
                    # --- NEW DEBUG ARGS ---
                    debug_batch_index=i,
                    debug_frame_index=file_frame_idx,
                    debug_task_name=task_name,
                    # --- END NEW DEBUG ARGS ---
                )
                # self._save_debug_numpy(batch_depth_numpy, "02_PROCESSED_PRE_NORM", i, file_frame_idx, task_name)

                batch_frames_float = batch_frames_numpy.astype("float32") / 255.0
                batch_depth_normalized = batch_depth_numpy.copy()

                if assume_raw_input:
                    if global_depth_max > 1.0:
                        batch_depth_normalized = batch_depth_numpy / global_depth_max
                else:                    
                    depth_range = global_depth_max - global_depth_min
                    if depth_range > 1e-5: # Use a small epsilon to detect non-zero range
                        batch_depth_normalized = (batch_depth_numpy - global_depth_min) / depth_range
                    else:
                        # If range is zero, fill with a neutral value (e.g., 0.5) to prevent NaN/Inf
                        batch_depth_normalized = np.full_like(batch_depth_numpy, fill_value=zero_disparity_anchor_val, dtype=np.float32)
                        logger.warning(f"Normalization collapsed to zero range ({global_depth_min:.4f} - {global_depth_max:.4f}). Filling with anchor value ({zero_disparity_anchor_val:.2f}).")

                batch_depth_normalized = np.clip(batch_depth_normalized, 0, 1)

                if not assume_raw_input and depth_gamma != 1.0:
                     batch_depth_normalized = np.power(batch_depth_normalized, depth_gamma)
                
                # self._save_debug_numpy(batch_depth_normalized, "03_FINAL_NORMALIZED", i, file_frame_idx, task_name) 
                
                # --- NEW LOGIC: Invert Gamma Effect (Gamma > 1.0 makes near-field brighter) ---
                if not assume_raw_input and round(depth_gamma, 2) != 1.0:
                    logger.debug(f"Applying gamma reversal for intuitive control (Gamma={depth_gamma:.2f}).")
                    # Step 1: Invert normalized depth
                    inverted_depth = 1.0 - batch_depth_normalized
                    # Step 2: Apply gamma to the inverted depth
                    gamma_applied_inverted = np.power(inverted_depth, depth_gamma)
                    # Step 3: Invert back
                    batch_depth_normalized = 1.0 - gamma_applied_inverted
                    # Clamp to ensure no float inaccuracies push values outside [0, 1]
                    batch_depth_normalized = np.clip(batch_depth_normalized, 0.0, 1.0)

                batch_depth_vis_list = []
                for d_frame in batch_depth_normalized:
                    d_frame_vis = d_frame.copy()
                    if d_frame_vis.max() > d_frame_vis.min(): 
                        cv2.normalize(d_frame_vis, d_frame_vis, 0, 1, cv2.NORM_MINMAX)
                    vis_frame_uint8 = (d_frame_vis * 255).astype(np.uint8)
                    vis_frame = cv2.applyColorMap(vis_frame_uint8, cv2.COLORMAP_VIRIDIS)
                    batch_depth_vis_list.append(vis_frame.astype("float32") / 255.0)
                batch_depth_vis = np.stack(batch_depth_vis_list, axis=0) 

                t_end_preproc = time.perf_counter()
                # --- END TIMER 2 ---

                # --- TIMER 3: HtoD Transfer (CPU to GPU) ---
                t_start_transfer_HtoD = time.perf_counter()
                
                left_video_tensor = torch.from_numpy(batch_frames_numpy).permute(0, 3, 1, 2).float().cuda() / 255.0
                disp_map_tensor = torch.from_numpy(batch_depth_normalized).unsqueeze(1).float().cuda()        
                disp_map_tensor = (disp_map_tensor - zero_disparity_anchor_val) * 2.0
                disp_map_tensor = disp_map_tensor * max_disp

                torch.cuda.synchronize() # Force synchronization before compute
                t_end_transfer_HtoD = time.perf_counter()
                # --- END TIMER 3 ---

                # --- TIMER 4: GPU Compute (Core Splatting) ---
                t_start_compute = time.perf_counter()

                with torch.no_grad():
                    right_video_tensor_raw, occlusion_mask_tensor = stereo_projector(left_video_tensor, disp_map_tensor)
                    if is_low_res_task:
                        # 1. Fill Left Edge Occlusions
                        right_video_tensor_left_filled = self._fill_left_edge_occlusions(right_video_tensor_raw, occlusion_mask_tensor, boundary_width_pixels=3)
                        
                        # 2. Fill Right Edge Occlusions (New Call)
                        right_video_tensor = self._fill_right_edge_occlusions(right_video_tensor_left_filled, occlusion_mask_tensor, boundary_width_pixels=3)
                    else:
                        right_video_tensor = right_video_tensor_raw

                torch.cuda.synchronize() # Force synchronization after compute
                t_end_compute = time.perf_counter()
                # --- END TIMER 4 ---

                # --- TIMER 5: DtoH Transfer (GPU to CPU) ---
                t_start_transfer_DtoH = time.perf_counter()
                
                right_video_numpy = right_video_tensor.cpu().permute(0, 2, 3, 1).numpy()
                occlusion_mask_numpy = occlusion_mask_tensor.cpu().permute(0, 2, 3, 1).numpy().repeat(3, axis=-1)

                t_end_transfer_DtoH = time.perf_counter()
                # --- END TIMER 5 ---

                # --- TIMER 6: FFmpeg Write (Blocking I/O) ---
                t_start_write = time.perf_counter()

                for j in range(len(batch_frames_numpy)):
                    if dual_output:
                        video_grid = np.concatenate([occlusion_mask_numpy[j], right_video_numpy[j]], axis=1)
                    else:
                        video_grid_top = np.concatenate([batch_frames_float[j], batch_depth_vis[j]], axis=1)
                        video_grid_bottom = np.concatenate([occlusion_mask_numpy[j], right_video_numpy[j]], axis=1)
                        video_grid = np.concatenate([video_grid_top, video_grid_bottom], axis=0)

                    video_grid_uint16 = (np.clip(video_grid, 0.0, 1.0) * 65535.0).astype(np.uint16)
                    video_grid_bgr = cv2.cvtColor(video_grid_uint16, cv2.COLOR_RGB2BGR)
                    
                    # --- SEND FRAME TO FFMPEG PIPE ---
                    ffmpeg_process.stdin.write(video_grid_bgr.tobytes())
                    frame_count += 1

                t_end_write = time.perf_counter()
                # --- END TIMER 6 ---

                del left_video_tensor, disp_map_tensor, right_video_tensor, occlusion_mask_tensor
                torch.cuda.empty_cache()
                draw_progress_bar(frame_count, num_frames, prefix=f"  Encoding:")
        
                t_end_batch = time.perf_counter() # <--- TIMER END: Total Batch
                
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
                        f"[{task_tag} Batch {i//batch_size_actual + 1}] Frames={batch_size_actual} Total={total_batch_time*1000:.0f}ms | "
                        f"IO={io_time*1000:.0f}ms | CPU_Proc={preproc_time*1000:.0f}ms | HtoD={htod_time*100:.0f}ms | "
                        f"GPU_Comp={compute_time*1000:.0f}ms | DtoH={dtoh_time*1000:.0f}ms | FFmpeg_Write={write_time*1000:.0f}ms"
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
            # Close stdin first to signal end of input, then wait for process
            try:
                if ffmpeg_process.stdin and not ffmpeg_process.stdin.closed:
                    ffmpeg_process.stdin.close()
            except (BrokenPipeError, ValueError):
                pass  # Pipe already closed, ignore
            
            # Wait for the process to finish and get output
            try:
                stdout, stderr = ffmpeg_process.communicate(timeout=120)
            except ValueError:
                # stdin already closed, just wait for process
                ffmpeg_process.wait(timeout=120)
                stdout, stderr = b'', b''
            
            if self.stop_event.is_set():
                ffmpeg_process.terminate()
                logger.warning(f"FFmpeg encoding stopped by user for {os.path.basename(final_output_video_path)}.")
                encoding_successful = False
            elif ffmpeg_process.returncode != 0:
                logger.error(f"FFmpeg encoding failed for {os.path.basename(final_output_video_path)} (return code {ffmpeg_process.returncode}):\n{stderr.decode()}")
                encoding_successful = False
            else:
                logger.info(f"Successfully encoded video to {final_output_video_path}")
                logger.debug(f"FFmpeg stderr log:\n{stderr.decode()}")
        
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
                sidecar_ext = self.APP_CONFIG_DEFAULTS.get('OUTPUT_SIDECAR_EXT', '.spsidecar')
                output_sidecar_path = f"{os.path.splitext(final_output_video_path)[0]}{sidecar_ext}"
                try:
                    with open(output_sidecar_path, 'w', encoding='utf-8') as f:
                        json.dump(output_sidecar_data, f, indent=4)
                    logger.info(f"Created output sidecar file: {output_sidecar_path}")
                except Exception as e:
                    logger.error(f"Error creating output sidecar file '{output_sidecar_path}': {e}")
            else:
                logger.debug("Skipping output sidecar creation: frame_overlap and input_bias are zero.")
        else:
            logger.debug("Skipping output sidecar creation: High-resolution output does not require spsidecar.")
                    
        return True

    def _determine_auto_convergence(self, depth_map_path: str, total_frames_to_process: int, batch_size: int, fallback_value: float) -> Tuple[float, float]:
        """
        Calculates the Auto Convergence points for the entire video (Average and Peak)
        in a single pass.
        
        Args:
            fallback_value (float): The current GUI/Sidecar value to return if auto-convergence fails.
            
        Returns:
            Tuple[float, float]: (new_anchor_avg: 0.0-1.0, new_anchor_peak: 0.0-1.0). 
                                 Returns (fallback_value, fallback_value) if the process fails.
        """
        logger.info("==> Starting Auto-Convergence pre-pass to determine global average and peak depth.")
        
        # --- Constants for Auto-Convergence Logic ---
        BLUR_KERNEL_SIZE = 9
        CENTER_CROP_PERCENT = 0.75
        MIN_VALID_PIXELS = 5
        # The offset is only applied at the end for the 'Average' mode.
        INTERNAL_ANCHOR_OFFSET = 0.1 
        # -------------------------------------------

        all_valid_frame_values = []
        fallback_tuple = (fallback_value, fallback_value) # Value to return on failure

        try:
            # 1. Initialize Decord Reader (No target height/width needed, raw is fine)
            depth_reader = VideoReader(depth_map_path, ctx=cpu(0))
            if len(depth_reader) == 0:
                 logger.error("Depth map reader has no frames. Cannot calculate Auto-Convergence.")
                 return fallback_tuple
        except Exception as e:
            logger.error(f"Error initializing depth map reader for Auto-Convergence: {e}")
            return fallback_tuple

        # 2. Iterate and Collect Data
        
        video_length = len(depth_reader)
        if total_frames_to_process <= 0 or total_frames_to_process > video_length:
             num_frames = video_length
        else:
             num_frames = total_frames_to_process
            
        logger.debug(f"  AutoConv determined actual frames to process: {num_frames} (from input length {total_frames_to_process}).")

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
                batch_depth_numpy_raw = depth_reader.get_batch(current_frame_indices).asnumpy()
            except Exception as e:
                logger.error(f"Error seeking/reading depth map batch starting at index {i}: {e}. Skipping batch.")
                continue

            # Process depth frames (Grayscale, Float conversion)
            if batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 3:
                batch_depth_numpy = batch_depth_numpy_raw.mean(axis=-1)
            elif batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 1:
                batch_depth_numpy = batch_depth_numpy_raw.squeeze(-1)
            else:
                batch_depth_numpy = batch_depth_numpy_raw
            
            batch_depth_float = batch_depth_numpy.astype(np.float32)

            # Get chunk min/max for normalization (using the chunk's range)
            min_val = batch_depth_float.min()
            max_val = batch_depth_float.max()
            
            if max_val - min_val > 1e-5:
                batch_depth_normalized = (batch_depth_float - min_val) / (max_val - min_val)
            else:
                batch_depth_normalized = np.full_like(batch_depth_float, fill_value=0.5, dtype=np.float32)

            # Frame-by-Frame Processing (Blur & Crop)
            for j, frame in enumerate(batch_depth_normalized):
                
                current_frame_idx = current_frame_indices[j]
                H, W = frame.shape
                
                # a) Blur
                frame_blurred = cv2.GaussianBlur(frame, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
                
                # b) Center Crop (75% of H and W)
                margin_h = int(H * (1 - CENTER_CROP_PERCENT) / 2)
                margin_w = int(W * (1 - CENTER_CROP_PERCENT) / 2)
                
                cropped_frame = frame_blurred[margin_h:H-margin_h, margin_w:W-margin_w]
                
                # c) Average (Exclude true black/white pixels (0.0 or 1.0) which may be background/edges)
                valid_pixels = cropped_frame[(cropped_frame > 0.001) & (cropped_frame < 0.999)] 
                
                if valid_pixels.size > MIN_VALID_PIXELS:
                    # Append the mean of the valid pixels for this frame
                    all_valid_frame_values.append(valid_pixels.mean()) 
                else:
                    # FALLBACK FOR THE FRAME: Use the mean of the WHOLE cropped, blurred frame
                    all_valid_frame_values.append(cropped_frame.mean())
                    logger.warning(f"  [AutoConv Frame {current_frame_idx:03d}] SKIPPED: Valid pixel count ({valid_pixels.size}) below threshold ({MIN_VALID_PIXELS}). Forcing mean from full cropped frame.")

            draw_progress_bar(i + len(current_frame_indices), num_frames, prefix="  Auto-Conv Pre-Pass:")
        
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
            
            logger.info(f"\n==> Auto-Convergence Calculated: Avg={raw_anchor_avg:.4f} + Offset ({INTERNAL_ANCHOR_OFFSET:.2f}) = Final Avg {final_anchor_avg:.4f}")
            logger.info(f"==> Auto-Convergence Calculated: Peak={raw_anchor_peak:.4f} = Final Peak {final_anchor_peak:.4f}")
            
            # Return both calculated values
            return float(final_anchor_avg), float(final_anchor_peak)
        else:
            logger.warning("\n==> Auto-Convergence failed: No valid frames found. Using fallback value.")
            return fallback_tuple

    def exit_app(self):
        """Handles application exit, including stopping the processing thread."""
        self._save_config()
        self.stop_event.set()
        if self.processing_thread and self.processing_thread.is_alive():
            logger.info("==> Waiting for processing thread to finish...")
            self.processing_thread.join(timeout=5.0)
            if self.processing_thread.is_alive():
                logger.debug("==> Thread did not terminate gracefully within timeout.")
        release_cuda_memory()

    def _fill_left_edge_occlusions(self, right_video_tensor: torch.Tensor, occlusion_mask_tensor: torch.Tensor, boundary_width_pixels: int = 3) -> torch.Tensor:
        """
        [VECTORIZED] Creates a thin, content-filled boundary at the absolute left edge of the screen
        by replicating the first visible pixels (from the right) into the leftmost columns.
        """
        B, C, H, W = right_video_tensor.shape
        boundary_width_pixels = min(W, boundary_width_pixels)
        if boundary_width_pixels <= 0:
            logger.debug("Boundary width for left-edge occlusions is 0 or less, skipping fill.")
            return right_video_tensor

        modified_right_video_tensor = right_video_tensor.clone()
        
        # 1. Determine the first visible pixel index for every (B, H) slice.
        #    occlusion_mask_tensor is 1.0 for occluded, 0.0 for visible.
        #    visible_mask is True for visible (where mask < 0.5)
        visible_mask = (occlusion_mask_tensor[:, 0, :, :] < 0.5) # Shape [B, H, W]

        # Use argmax to find the index of the FIRST True value along the W dimension.
        # Note: If a row is all False (all occluded), argmax returns index 0.
        # We handle this by clamping/fallback later.
        first_visible_index = torch.argmax(visible_mask.int(), dim=2, keepdim=True) # Shape [B, H, 1]
        
        # Fallback: If a row is entirely occluded (all False), argmax returns 0.
        # The correct fallback is W-1 if W>0, or just leave it at 0 and hope the source pixel is black.
        # Find rows where argmax returned 0, AND the actual first column is occluded (0.0).
        # A visible_mask where all values are False (all occluded) will result in argmax=0 for that row.
        # We need to find if there's *any* visible pixel. Sum(W) > 0.
        fully_occluded = (visible_mask.sum(dim=2, keepdim=True) == 0) # [B, H, 1] True if fully occluded
        
        # Set the index to W-1 for fully occluded rows to pull a border pixel (safer than 0)
        # Note: We must ensure this operation runs on the GPU with the tensors.
        # Clamp to ensure index is always valid (max index is W-1)
        source_column_indices = torch.clamp(first_visible_index, 0, W - 1)
        
        # Override source index for fully occluded rows to a safe boundary (W-1)
        source_column_indices[fully_occluded] = W - 1 

        # 2. Gather the source pixels for filling (Shape [B, C, H])
        #    We need to reshape the right_video_tensor [B, C, H, W] to gather the source_column_indices [B, H, 1].
        #    torch.gather() is the vectorized way to do this.
        #    Gather on dimension W (dim=3), using indices expanded to [B, C, H, 1]
        source_column_indices_expanded = source_column_indices.unsqueeze(1).repeat(1, C, 1, 1) # Shape [B, C, H, 1]
        
        # Gather the color from the source column for all rows
        source_pixel_values_4d = torch.gather(right_video_tensor, dim=3, index=source_column_indices_expanded) # Shape [B, C, H, 1]
        source_pixel_values_3d = source_pixel_values_4d.squeeze(3) # Shape [B, C, H]

        # 3. Create a mask of the leftmost columns that are currently occluded
        #    This mask is True only for pixels (B, C, H, W) that are BOTH in the boundary AND occluded.
        #    Boundary mask [W]: True for x < boundary_width_pixels
        boundary_region_mask = torch.zeros(W, dtype=torch.bool, device=right_video_tensor.device)
        if boundary_width_pixels > 0:
            boundary_region_mask[:boundary_width_pixels] = True
            
        # Occlusion mask (1.0 for occluded)
        is_occluded_4d = (occlusion_mask_tensor > 0.5) # Shape [B, 1, H, W]
        
        # Combine the masks
        # [B, 1, H, W] AND [W] -> [B, 1, H, W]
        fill_target_mask = is_occluded_4d & boundary_region_mask.view(1, 1, 1, W)
        
        # 4. Apply the gathered pixel values to the masked regions
        #    Apply fill mask to the source values to match shape for where the fill should occur.
        #    source_pixel_values_3d is [B, C, H]. Expand it to [B, C, H, W]
        source_to_apply = source_pixel_values_3d.unsqueeze(3).repeat(1, 1, 1, W)

        # Use torch.where to conditionally update the tensor:
        modified_right_video_tensor = torch.where(
            fill_target_mask.repeat(1, C, 1, 1), # Use C-channel mask
            source_to_apply,                     # Value to use if mask is True
            modified_right_video_tensor          # Value to use if mask is False (original pixel)
        )
        
        logger.debug(f"[Vectorized] Created {boundary_width_pixels}-pixel left-edge content boundary.")
        return modified_right_video_tensor

    def _fill_right_edge_occlusions(self, right_video_tensor: torch.Tensor, occlusion_mask_tensor: torch.Tensor, boundary_width_pixels: int = 3) -> torch.Tensor:
        """
        [VECTORIZED] Creates a thin, content-filled boundary at the absolute right edge of the screen
        by replicating the last visible pixels (from the left) into the rightmost columns.
        """
        B, C, H, W = right_video_tensor.shape
        boundary_width_pixels = min(W, boundary_width_pixels)
        if boundary_width_pixels <= 0:
            logger.debug("Boundary width for right-edge occlusions is 0 or less, skipping fill.")
            return right_video_tensor

        modified_right_video_tensor = right_video_tensor.clone()
        
        # 1. Determine the LAST visible pixel index for every (B, H) slice.
        #    occlusion_mask_tensor is 1.0 for occluded, 0.0 for visible.
        #    visible_mask is True for visible (where mask < 0.5)
        visible_mask = (occlusion_mask_tensor[:, 0, :, :] < 0.5) # Shape [B, H, W]

        # Use argmax on the REVERSED tensor to find the index of the first True from the right.
        # The true index is W - 1 - (index in the reversed tensor).
        visible_mask_reversed = torch.flip(visible_mask, dims=[2])
        first_visible_index_reversed = torch.argmax(visible_mask_reversed.int(), dim=2, keepdim=True) # Shape [B, H, 1]

        # Calculate the actual index of the LAST visible pixel (from 0 to W-1)
        last_visible_index = W - 1 - first_visible_index_reversed # Shape [B, H, 1]
        
        # Fallback: Find rows that are fully occluded (no visible pixels)
        fully_occluded = (visible_mask.sum(dim=2, keepdim=True) == 0) # [B, H, 1] True if fully occluded

        # Override source index for fully occluded rows to a safe boundary (0)
        source_column_indices = torch.clamp(last_visible_index, 0, W - 1)
        source_column_indices[fully_occluded] = 0 # If fully occluded, use index 0 as source (safer than W-1)

        # 2. Gather the source pixels for filling (Shape [B, C, H])
        #    Gather on dimension W (dim=3), using indices expanded to [B, C, H, 1]
        source_column_indices_expanded = source_column_indices.unsqueeze(1).repeat(1, C, 1, 1) # Shape [B, C, H, 1]
        
        # Gather the color from the source column for all rows (the last visible pixel's color)
        source_pixel_values_4d = torch.gather(right_video_tensor, dim=3, index=source_column_indices_expanded) # Shape [B, C, H, 1]
        source_pixel_values_3d = source_pixel_values_4d.squeeze(3) # Shape [B, C, H]

        # 3. Create a mask of the rightmost columns that are currently occluded
        #    Boundary mask [W]: True for x >= W - boundary_width_pixels
        boundary_region_mask = torch.zeros(W, dtype=torch.bool, device=right_video_tensor.device)
        if boundary_width_pixels > 0:
            boundary_region_mask[W - boundary_width_pixels:] = True
            
        # Occlusion mask (1.0 for occluded)
        is_occluded_4d = (occlusion_mask_tensor > 0.5) # Shape [B, 1, H, W]
        
        # Combine the masks
        # [B, 1, H, W] AND [W] -> [B, 1, H, W]
        fill_target_mask = is_occluded_4d & boundary_region_mask.view(1, 1, 1, W)

        # 4. Apply the gathered pixel values to the masked regions
        #    Expand source values to [B, C, H, W]
        source_to_apply = source_pixel_values_3d.unsqueeze(3).repeat(1, 1, 1, W)

        # Use torch.where to conditionally update the tensor:
        modified_right_video_tensor = torch.where(
            fill_target_mask.repeat(1, C, 1, 1), # Use C-channel mask
            source_to_apply,                     # Value to use if mask is True
            modified_right_video_tensor          # Value to use if mask is False (original pixel)
        )
        
        logger.debug(f"[Vectorized] Created {boundary_width_pixels}-pixel right-edge content boundary.")
        return modified_right_video_tensor

    def _get_current_config(self):
        """Collects all current GUI variable values into a single dictionary."""
        config = {
            # Folder Configurations
            "input_source_clips": self.input_source_clips,
            "input_depth_maps": self.input_depth_maps,
            "output_splatted": self.output_splatted,

            "enable_full_resolution": self.enable_full_res,
            "batch_size": self.batch_size,
            "enable_low_resolution": self.enable_low_res,
            "pre_res_width": self.pre_res_width,
            "pre_res_height": self.pre_res_height,
            "low_res_batch_size": self.low_res_batch_size,
            
            # Depth Pre-processing
            "depth_dilate_size_x": self.depth_dilate_size_x,
            "depth_dilate_size_y": self.depth_dilate_size_y,
            "depth_blur_size_x": self.depth_blur_size_x,
            "depth_blur_size_y": self.depth_blur_size_y,
            "depth_dilate_left": self.depth_dilate_left,
            "depth_blur_left": self.depth_blur_left,
            "depth_blur_left_mix": self.depth_blur_left_mix,

            # Processing Settings
            "process_length": self.process_length,
            "output_crf": self.output_crf,
            "output_crf_full": self.output_crf_full,
            "output_crf_low": self.output_crf_low,
            "dual_output": self.dual_output,
            "auto_convergence_mode": self.auto_convergence_mode,
            
            # Stereo Projection
            "depth_gamma": self.depth_gamma,
            "max_disp": self.max_disp,
            "convergence_point": self.zero_disparity_anchor,
            "enable_global_norm": self.enable_global_norm,
            "move_to_finished": self.move_to_finished,
            
            # Border Controls
            "border_width": self.border_width,
            "border_bias": self.border_bias,
            "border_mode": self.border_mode,
            "auto_border_L": self.auto_border_L,
            "auto_border_R": self.auto_border_R,
            
            # Color Tags
            "color_tags_mode": self.color_tags_mode,
            
            # Multi-Map
            "multi_map_enabled": self.multi_map,
            "selected_depth_map": self.selected_depth_map,
            
            # UI Settings
            "dark_mode_enabled": self.dark_mode,
            "preview_source": self.preview_source,
            "preview_size": self.preview_size,
            
            # Dev Tools
            "skip_lowres_preproc": self.skip_lowres_preproc,
            "track_dp_total_true_on_render": self.track_dp_total_true_on_render,
        }
        return config

    def _get_current_sidecar_paths_and_data(self):
        """Helper to get current file path, sidecar path, and existing data (merged with defaults)."""
        # For now, we'll return a placeholder
        return None

    def _get_defined_tasks(self, settings):
        """Helper to return a list of processing tasks based on GUI settings."""
        processing_tasks = []
        if settings["enable_full_resolution"]:
            processing_tasks.append({
                "name": "Full-Resolution",
                "output_subdir": "hires",
                "set_pre_res": False,
                "target_width": -1,
                "target_height": -1,
                "batch_size": settings["full_res_batch_size"],
                "is_low_res": False
            })
        if settings["enable_low_resolution"]:
            processing_tasks.append({
                "name": "Low-Resolution",
                "output_subdir": "lowres",
                "set_pre_res": True,
                "target_width": settings["low_res_width"],
                "target_height": settings["low_res_height"],
                "batch_size": settings["low_res_batch_size"],
                "is_low_res": True
            })
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
        json_sidecar_path = os.path.join(sidecar_folder, f"{video_name}_depth{sidecar_ext}")

        merged_config = None
        sidecar_exists = False
        selected_map_for_video = None

        if os.path.exists(json_sidecar_path):
            try:
                merged_config = self.sidecar_manager.load_sidecar_data(json_sidecar_path) or {}
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
            "gamma": float(self.depth_gamma),
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
                logger.info(f"Single-file mode: using depth map file '{actual_depth_map_path}'")
                # Optional: show map as "Direct file" in the info panel
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
            if self.multi_map:
                # 1) First try sidecar's selected map for this video
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
                        logger.info(f"[MM] USING sidecar map '{sidecar_map}' for '{video_name}'")
                        # Show map name PLUS source (Sidecar)
                    else:
                        logger.warning(
                            f"[MM] sidecar map '{sidecar_map}' has no depth file for '{video_name}'"
                        )

                # 2) If sidecar FAILED, fall back to GUI-selected map
                if not actual_depth_map_path:
                    gui_map = self.selected_depth_map
                    if gui_map:
                        candidate_dir = os.path.join(base_folder, gui_map)
                        c_mp4 = os.path.join(candidate_dir, f"{video_name}_depth.mp4")
                        c_npz = os.path.join(candidate_dir, f"{video_name}_depth.npz")

                        if os.path.exists(c_mp4):
                            actual_depth_map_path = c_mp4
                        elif os.path.exists(c_npz):
                            actual_depth_map_path = c_npz

                        if actual_depth_map_path:
                            logger.info(f"[MM] USING GUI map '{gui_map}' for '{video_name}'")
                            # Show map name PLUS source (GUI/Default)

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
                    return {
                        "error": f"No depth for '{video_name}' in '{base_folder}'"
                    }

        actual_depth_map_path = os.path.normpath(actual_depth_map_path)

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
        if self.multi_map:
            map_source = "Sidecar" if sidecar_exists else "GUI/Default"
        else:
            map_source = "N/A"
            
        # --- NEW: Determine Global Normalization Policy ---
        enable_global_norm_policy = self.enable_global_norm
        if sidecar_exists:
            # Policy: If a sidecar exists, GN is DISABLED (manual mode)
            enable_global_norm_policy = False
            logger.debug(f"GN Policy: Sidecar exists for {video_name}. GN forced OFF.")
        
        # Determine the source for GN info
        gn_source = "Sidecar" if sidecar_exists else ("GUI/ON" if enable_global_norm_policy else "GUI/OFF")
        
        settings = {
            "actual_depth_map_path": actual_depth_map_path,
            "convergence_plane": merged_config.get("convergence_plane", gui_config["convergence_plane"]),
            "max_disparity_percentage": merged_config.get("max_disparity", gui_config["max_disparity"]),
            "input_bias": merged_config.get("input_bias"),
            "depth_gamma": merged_config.get("gamma", gui_config["gamma"]),
            # GUI-derived depth pre-processing settings
            "depth_dilate_size_x": float(self.depth_dilate_size_x),
            "depth_dilate_size_y": float(self.depth_dilate_size_y),
            "depth_blur_size_x": int(float(self.depth_blur_size_x)),
            "depth_blur_size_y": int(float(self.depth_blur_size_y)),
            # Tracking / info sources
            "sidecar_found": sidecar_exists,
            "anchor_source": "Sidecar" if sidecar_exists else "GUI/Default",
            "max_disp_source": "Sidecar" if sidecar_exists else "GUI/Default",
            "gamma_source": "Sidecar" if sidecar_exists else "GUI/Default",
            "map_source": map_source,
            "enable_global_norm": enable_global_norm_policy, 
            "gn_source": gn_source,
        }

        # If no sidecar file exists at all, enforce GUI values explicitly
        if not sidecar_exists:
            settings["convergence_plane"] = gui_config["convergence_plane"]
            settings["max_disparity_percentage"] = gui_config["max_disparity"]
            settings["depth_gamma"] = gui_config["gamma"]

        return settings

    def _initialize_video_and_depth_readers(self, video_path, actual_depth_map_path, process_length, task_settings, match_depth_res):
            """
            Initializes VideoReader objects for source video and depth map,
            and returns their metadata.
            Returns: (video_reader, depth_reader, processed_fps, current_processed_height, current_processed_width,
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
            depth_stream_info = None # Initialize to None
    
            try:
                # 1. Initialize input video reader
                video_reader_input, processed_fps, original_vid_h, original_vid_w, \
                current_processed_height, current_processed_width, video_stream_info, \
                total_frames_input = read_video_frames(
                    video_path, process_length,
                    set_pre_res=task_settings["set_pre_res"], pre_res_width=task_settings["target_width"], pre_res_height=task_settings["target_height"]
                )
            except Exception as e:
                logger.error(f"==> Error initializing input video reader for {os.path.basename(video_path)} {task_settings['name']} pass: {e}. Skipping this pass.")
                return None, None, 0.0, 0, 0, None, 0, 0, 0, 0, None # Return None for depth_stream_info
                # Determine map source for Multi-Map
                map_display = "N/A"
                if self.multi_map:
                    if self._current_video_sidecar_map:
                        map_display = f"Sidecar > {self._current_video_sidecar_map}"
                    elif self.selected_depth_map:
                        map_display = f"Default > {self.selected_depth_map}"

            try:
                # 2. Initialize depth maps reader and capture depth_stream_info
                depth_reader_input, total_frames_depth, actual_depth_height, actual_depth_width, depth_stream_info = load_pre_rendered_depth(
                    actual_depth_map_path,
                    process_length=process_length,
                    target_height=current_processed_height,
                    target_width=current_processed_width,
                    match_resolution_to_target=match_depth_res
                )
            except Exception as e:
                logger.error(f"==> Error initializing depth map reader for {os.path.basename(video_path)} {task_settings['name']} pass: {e}. Skipping this pass.")
                if video_reader_input: del video_reader_input
                return None, None, 0.0, 0, 0, None, 0, 0, 0, None # Return None for depth_stream_info
    
            # CRITICAL CHECK: Ensure input video and depth map have consistent frame counts
            if total_frames_input != total_frames_depth:
                logger.error(f"==> Frame count mismatch for {os.path.basename(video_path)} {task_settings['name']} pass: Input video has {total_frames_input} frames, Depth map has {total_frames_depth} frames. Skipping.")
                if video_reader_input: del video_reader_input
                if depth_reader_input: del depth_reader_input
                return None, None, 0.0, 0, 0, None, 0, 0, 0, 0, None # Return None for depth_stream_info
            
            return (video_reader_input, depth_reader_input, processed_fps, current_processed_height, current_processed_width,
                    video_stream_info, total_frames_input, total_frames_depth, actual_depth_height, actual_depth_width, depth_stream_info)

    def _load_config(self):
        """Loads configuration from config_splat.json."""
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
                    old_value = self.app_config.pop("enable_autogain") # Remove old key
                    # New value is the inverse of the old value
                    self.app_config["enable_global_norm"] = not bool(old_value)
                # --- END FIX ---
            except Exception as e:
                logger.error(f"Failed to load config file: {e}. Using defaults.")
                self.app_config = {}

    def _load_help_texts(self):
        """Loads help texts from a JSON file."""
        try:
            with open(os.path.join("dependency", "splatter_help.json"), "r") as f:
                self.help_texts = json.load(f)
        except FileNotFoundError:
            logger.error("Error: splatter_help.json not found. Tooltips will not be available.")
            self.help_texts = {}
        except json.JSONDecodeError:
            logger.error("Error: Could not decode splatter_help.json. Check file format.")
            self.help_texts = {}

    def _move_processed_files(self, video_path, actual_depth_map_path, finished_source_folder, finished_depth_folder):
        """Moves source video, depth map, and its sidecar file to 'finished' folders."""
        max_retries = 5
        retry_delay_sec = 0.5 # Wait half a second between retries

        # Move source video
        if finished_source_folder:
            dest_path_src = os.path.join(finished_source_folder, os.path.basename(video_path))
            for attempt in range(max_retries):
                try:
                    if os.path.exists(dest_path_src):
                        logger.warning(f"File '{os.path.basename(video_path)}' already exists in '{finished_source_folder}'. Overwriting.")
                        os.remove(dest_path_src)
                    shutil.move(video_path, finished_source_folder)
                    logger.debug(f"==> Moved processed video '{os.path.basename(video_path)}' to: {finished_source_folder}")
                    break
                except PermissionError as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries}: PermissionError (file in use) when moving '{os.path.basename(video_path)}'. Retrying in {retry_delay_sec}s...")
                    time.sleep(retry_delay_sec)
                except Exception as e:
                    logger.error(f"==> Failed to move source video '{os.path.basename(video_path)}' to '{finished_source_folder}': {e}", exc_info=True)
                    break
            else:
                logger.error(f"==> Failed to move source video '{os.path.basename(video_path)}' after {max_retries} attempts due to PermissionError.")
        else:
            logger.warning(f"==> Cannot move source video '{os.path.basename(video_path)}': 'finished_source_folder' is not set (not in batch mode).")

        # Move depth map and its sidecar file
        if actual_depth_map_path and finished_depth_folder:
            dest_path_depth = os.path.join(finished_depth_folder, os.path.basename(actual_depth_map_path))
            # --- Retry for Depth Map ---
            for attempt in range(max_retries):
                try:
                    if os.path.exists(dest_path_depth):
                        logger.warning(f"File '{os.path.basename(actual_depth_map_path)}' already exists in '{finished_depth_folder}'. Overwriting.")
                        os.remove(dest_path_depth)
                    shutil.move(actual_depth_map_path, finished_depth_folder)
                    logger.debug(f"==> Moved depth map '{os.path.basename(actual_depth_map_path)}' to: {finished_depth_folder}")
                    break
                except PermissionError as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries}: PermissionError (file in use) when moving depth map '{os.path.basename(actual_depth_map_path)}'. Retrying in {retry_delay_sec}s...")
                    time.sleep(retry_delay_sec)
                except Exception as e:
                    logger.error(f"==> Failed to move depth map '{os.path.basename(actual_depth_map_path)}' to '{finished_depth_folder}': {e}", exc_info=True)
                    break
            else:
                logger.error(f"==> Failed to move depth map '{os.path.basename(actual_depth_map_path)}' after {max_retries} attempts due to PermissionError.")

            # --- Retry for Sidecar file (if it exists) ---
            depth_map_dirname = os.path.dirname(actual_depth_map_path)
            depth_map_basename_without_ext = os.path.splitext(os.path.basename(actual_depth_map_path))[0]
            input_sidecar_ext = self.APP_CONFIG_DEFAULTS.get('SIDECAR_EXT', '.fssidecar') # Fallback to .fssidecar
            
            json_sidecar_path_to_move = os.path.join(depth_map_dirname, f"{depth_map_basename_without_ext}{input_sidecar_ext}")
            dest_path_json = os.path.join(finished_depth_folder, f"{depth_map_basename_without_ext}{input_sidecar_ext}")

            if os.path.exists(json_sidecar_path_to_move):
                for attempt in range(max_retries):
                    try:
                        if os.path.exists(dest_path_json):
                            logger.warning(f"Sidecar file '{os.path.basename(json_sidecar_path_to_move)}' already exists in '{finished_depth_folder}'. Overwriting.")
                            os.remove(dest_path_json)
                        shutil.move(json_sidecar_path_to_move, finished_depth_folder)
                        logger.debug(f"==> Moved sidecar file '{os.path.basename(json_sidecar_path_to_move)}' to: {finished_depth_folder}")
                        break
                    except PermissionError as e:
                        logger.warning(f"Attempt {attempt + 1}/{max_retries}: PermissionError (file in use) when moving file '{os.path.basename(json_sidecar_path_to_move)}'. Retrying in {retry_delay_sec}s...")
                        time.sleep(retry_delay_sec)
                    except Exception as e:
                        logger.error(f"==> Failed to move sidecar file '{os.path.basename(json_sidecar_path_to_move)}' to '{finished_depth_folder}': {e}", exc_info=True)
                        break
                else:
                    logger.error(f"==> Failed to move sidecar file '{os.path.basename(json_sidecar_path_to_move)}' after {max_retries} attempts due to PermissionError.")
            else:
                logger.debug(f"==> No sidecar file '{json_sidecar_path_to_move}' found to move.")
        elif actual_depth_map_path:
            logger.info(f"==> Cannot move depth map '{os.path.basename(actual_depth_map_path)}': 'finished_depth_folder' is not set (not in batch mode).")

    def on_auto_convergence_mode_select(self, mode):
        """
        Handles selection in the Auto-Convergence combo box.
        If a mode is selected, it checks the cache and runs the calculation if needed.
        """
        if mode == "Off":
            # self._auto_conv_cache = {"Average": None, "Peak": None} # Clear cache on Off
            return
        
        if self._is_auto_conv_running:
            logger.warning("Auto-Converge calculation is already running. Please wait.")
            return

        if self._auto_conv_cache[mode] is not None:
            # Value is cached, apply it immediately
            cached_value = self._auto_conv_cache[mode]
            
            # 1. Update the instance variable
            self.zero_disparity_anchor = cached_value
            
            # 3. Update status label
            logger.info(f"Auto-Converge ({mode}): Loaded cached value {cached_value:.2f}")
            return
            
        # Cache miss, run the calculation (using the existing run_preview_auto_converge logic)
        # Note: This function is not currently connected to UI events

    def _process_depth_batch(self, batch_depth_numpy_raw: np.ndarray, depth_stream_info: Optional[dict], depth_gamma: float,
                              depth_dilate_size_x: float, depth_dilate_size_y: float, depth_blur_size_x: float, depth_blur_size_y: float, 
                              is_low_res_task: bool, max_raw_value: float,
                              global_depth_min: float, global_depth_max: float,
                              depth_dilate_left: float = 0.0,
                              depth_blur_left: float = 0.0,
                              debug_batch_index: int = 0, debug_frame_index: int = 0, debug_task_name: str = "PreProcess",
                              ) -> np.ndarray:
        """
        Loads, converts, and pre-processes the raw depth map batch using stable NumPy/OpenCV CPU calls.
        Unified depth processor. Pre-processes filters in float space.
        Gamma is now unified to occur in normalized space.
        """
        # Grayscale conversion
        if batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 3: # RGB
            batch_depth_numpy = batch_depth_numpy_raw.mean(axis=-1)
        elif batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 1:
            batch_depth_numpy = batch_depth_numpy_raw.squeeze(-1)
        else:
            batch_depth_numpy = batch_depth_numpy_raw
        
        # Convert to float32 for processing
        batch_depth_numpy_float = batch_depth_numpy.astype(np.float32)
        
        # Dev Tools: allow skipping ALL low-res preprocessing (gamma/dilate/blur)
        if is_low_res_task and self.skip_lowres_preproc:
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

        needs_processing = (
            abs(render_dilate_left) > 1e-5
            or render_blur_left > 0
            or abs(render_dilate_x) > 1e-5
            or abs(render_dilate_y) > 1e-5
            or render_blur_x > 0
            or render_blur_y > 0
        )
        
        if needs_processing:
            device = torch.device('cpu')
            tensor_4d = torch.from_numpy(batch_depth_numpy_float).unsqueeze(1).to(device)
            
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
                EDGE_STEP_8BIT = 3.0  # raise to blur fewer edges; lower to blur more edges
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
                mix_f = self.depth_blur_left_mix
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
            
            # Normal X/Y Dilate
            if abs(render_dilate_x) > 1e-5 or abs(render_dilate_y) > 1e-5:
                tensor_4d = custom_dilate(
                    tensor_4d,
                    float(render_dilate_x),
                    float(render_dilate_y),
                    False,
                    max_raw_value,
                )
            
            # Normal X/Y Blur
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

    def _process_single_video_tasks(self, video_path, settings, initial_overall_task_counter, is_single_file_mode, finished_source_folder=None, finished_depth_folder=None):
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
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        logger.info(f"==> Processing Video: {video_name}")
        
        # Keep a local counter for tasks processed in this function
        local_task_counter = initial_overall_task_counter

        video_specific_settings = self._get_video_specific_settings(
            video_path,
            settings["input_depth_maps"],
            settings["zero_disparity_anchor"],
            settings["max_disp"],
            is_single_file_mode
        )

        processing_tasks = self._get_defined_tasks(settings)
        expected_task_count = len(processing_tasks)
        processed_tasks_count = 0
        any_task_completed_successfully_for_this_video = False

        if video_specific_settings.get("error"):
            logger.error(f"Error getting video specific settings for {video_name}: {video_specific_settings['error']}. Skipping.")
            # Skip the expected task count in the progress bar
            local_task_counter += expected_task_count
            return expected_task_count, False

        actual_depth_map_path = video_specific_settings["actual_depth_map_path"]
        current_zero_disparity_anchor = video_specific_settings["convergence_plane"]
        current_max_disparity_percentage = video_specific_settings["max_disparity_percentage"]
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
        
        if not processing_tasks:
            logger.debug(f"==> No processing tasks configured for {video_name}. Skipping.")
            return 0, False

        # --- Auto-Convergence Logic (BEFORE initializing readers) ---
        auto_conv_mode = settings["auto_convergence_mode"]

        # --- NEW: Global Normalization Policy variables ---
        enable_global_norm_policy = video_specific_settings["enable_global_norm"]
        gn_source = video_specific_settings["gn_source"]

        if video_specific_settings["sidecar_found"] and self.enable_global_norm:
             # Policy: Sidecar exists AND GUI toggle is ON. Policy forces GN OFF.
             if not self._gn_warning_shown:
                 # For Gradio, we'll just log the warning
                 logger.warning(f"GN Policy: Sidecar found for {video_name}. GN forced OFF (console log only).")
                 self._gn_warning_shown = True # Set flag to log to console only next time
             else:
                  logger.warning(f"GN Policy: Sidecar found for {video_name}. GN forced OFF (console log only).")

        # --- NEW LOGIC: Sidecar overrides Auto-Convergence ---
        if anchor_source == "Sidecar" and auto_conv_mode != "Off":
            logger.info(f"Sidecar found for {video_name}. Convergence Point locked to Sidecar value ({current_zero_disparity_anchor:.4f}). Auto-Convergence SKIPPED.")
            auto_conv_mode = "Off"

        if auto_conv_mode != "Off":
            logger.info(f"Auto-Convergence is ENABLED (Mode: {auto_conv_mode}). Running pre-pass...")

            try:
                anchor_float = float(current_zero_disparity_anchor)
            except (ValueError, TypeError):
                logger.error(f"Invalid convergence anchor value found: {current_zero_disparity_anchor}. Defaulting to 0.5.")
                anchor_float = 0.5

            new_anchor_avg, new_anchor_peak = self._determine_auto_convergence(
                actual_depth_map_path,
                settings["process_length"],
                settings["full_res_batch_size"],
                anchor_float,
            )            
            new_anchor_val = new_anchor_avg if auto_conv_mode == "Average" else new_anchor_peak

            if new_anchor_val != current_zero_disparity_anchor:
                current_zero_disparity_anchor = new_anchor_val
                anchor_source = "Auto"
            
            logger.info(f"Using Convergence Point: {current_zero_disparity_anchor:.4f} (Source: {anchor_source})")

        for task in processing_tasks:
            if self.stop_event.is_set():
                logger.info(f"==> Stopping {task['name']} processing for {video_name} due to user request")
                # Increment the global counter for all remaining, skipped tasks
                remaining_tasks_to_increment = expected_task_count - processed_tasks_count
                local_task_counter += remaining_tasks_to_increment
                return expected_task_count, any_task_completed_successfully_for_this_video

            logger.debug(f"\n==> Starting {task['name']} pass for {video_name}")

            # Decide what to show in the Map field
            if self.multi_map:
                # Multi-Map mode
                if actual_depth_map_path and map_source not in ("", "N/A"):
                    map_folder = os.path.basename(os.path.dirname(actual_depth_map_path)).strip()
                    map_label = f"{map_folder} ({map_source})"
                else:
                    map_label = "N/A"
            else:
                # Normal mode
                map_label = "Direct file" if is_single_file_mode else "Direct folder"

            video_reader_input, depth_reader_input, processed_fps, current_processed_height, current_processed_width, \
            video_stream_info, total_frames_input, total_frames_depth, actual_depth_height, actual_depth_width, \
            depth_stream_info = self._initialize_video_and_depth_readers(
                    video_path, actual_depth_map_path, settings["process_length"],
                    task, settings["match_depth_res"]
                )
            
            # Explicitly check for None for critical components before proceeding
            if video_reader_input is None or depth_reader_input is None or video_stream_info is None:
                logger.error(f"Skipping {task['name']} pass for {video_name} due to reader initialization error, frame count mismatch, or missing stream info.")
                local_task_counter += 1
                processed_tasks_count += 1
                release_cuda_memory()
                continue

            # --- MODIFIED: Use the policy to determine the mode ---
            assume_raw_input_mode = not enable_global_norm_policy # If GN is OFF, assume RAW Input
            global_depth_min = 0.0 
            global_depth_max = 1.0

            # --- UNCONDITIONAL Max Content Value Scan for RAW/Normalization Modes ---
            max_content_value = 1.0 
            raw_depth_reader_temp = None
            try:
                raw_depth_reader_temp = VideoReader(actual_depth_map_path, ctx=cpu(0))
                
                if len(raw_depth_reader_temp) > 0:
                    _, max_content_value = compute_global_depth_stats(
                        depth_map_reader=raw_depth_reader_temp,
                        total_frames=total_frames_depth,
                        chunk_size=task["batch_size"] 
                    )
                    logger.debug(f"Max content depth scanned: {max_content_value:.3f}.")
                else:
                    logger.error("RAW depth reader has no frames for content scan.")
            except Exception as e:
                logger.error(f"Failed to scan max content depth: {e}")
            finally:
                if raw_depth_reader_temp:
                    del raw_depth_reader_temp
                    gc.collect()
            # --- END UNCONDITIONAL SCAN ---

            if not assume_raw_input_mode: 
                logger.info("==> Global Depth Normalization selected. Starting global depth stats pre-pass with RAW reader.")
                
                raw_depth_reader_temp = None
                try:
                    raw_depth_reader_temp = VideoReader(actual_depth_map_path, ctx=cpu(0))
                    
                    if len(raw_depth_reader_temp) > 0:
                        global_depth_min, global_depth_max = compute_global_depth_stats(
                            depth_map_reader=raw_depth_reader_temp,
                            total_frames=total_frames_depth,
                            chunk_size=task["batch_size"] 
                        )
                        logger.debug("Successfully computed global stats from RAW reader.")
                    else:
                        logger.error("RAW depth reader has no frames.")
                except Exception as e:
                    logger.error(f"Failed to initialize/read RAW depth reader for global stats: {e}")
                    global_depth_min = 0.0 
                    global_depth_max = 1.0
                finally:
                    if raw_depth_reader_temp:
                        del raw_depth_reader_temp
                        gc.collect()
            else:
                logger.debug("==> No Normalization (Assume Raw 0-1 Input) selected. Skipping global stats pre-pass.")

                # --- RAW INPUT MODE SCALING ---
                final_scaling_factor = 1.0 

                if max_content_value <= 256.0 and max_content_value > 1.0:
                    final_scaling_factor = 255.0
                    logger.debug(f"Content Max {max_content_value:.2f} <= 8-bit. SCALING BY 255.0.")
                elif max_content_value > 256.0 and max_content_value <= 1024.0:
                    final_scaling_factor = max_content_value
                    logger.debug(f"Content Max {max_content_value:.2f} (9-10bit). SCALING BY CONTENT MAX.")
                else:
                    final_scaling_factor = 1023.0 
                    logger.warning(f"Max content value is too high/low ({max_content_value:.2f}). Using fallback 1023.0.")

                global_depth_max = final_scaling_factor
                global_depth_min = 0.0
                
                logger.debug(f"Raw Input Final Scaling Factor set to: {global_depth_max:.3f}")

            if not (actual_depth_height == current_processed_height and actual_depth_width == current_processed_width):
                logger.warning(f"==> Warning: Depth map reader output resolution ({actual_depth_width}x{actual_depth_height}) does not match processed video resolution ({current_processed_width}x{current_processed_height}) for {task['name']} pass. This indicates an issue with `load_pre_rendered_depth`'s `width`/`height` parameters. Processing may proceed but results might be misaligned.")

            actual_percentage_for_calculation = current_max_disparity_percentage / 20.0
            actual_max_disp_pixels = (actual_percentage_for_calculation / 100.0) * current_processed_width
            logger.debug(f"==> Max Disparity Input: {current_max_disparity_percentage:.1f}% -> Calculated Max Disparity for splatting ({task['name']}): {actual_max_disp_pixels:.2f} pixels")

            current_output_subdir = os.path.join(settings["output_splatted"], task["output_subdir"])
            os.makedirs(current_output_subdir, exist_ok=True)
            output_video_path_base = os.path.join(current_output_subdir, f"{video_name}.mp4")

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
                user_output_crf=settings["output_crf"],
                is_low_res_task=task["is_low_res"],
                depth_gamma=current_depth_gamma,
                depth_dilate_size_x=current_depth_dilate_size_x,
                depth_dilate_size_y=current_depth_dilate_size_y,
                depth_blur_size_x=current_depth_blur_size_x,
                depth_blur_size_y=current_depth_blur_size_y
            )

            if self.stop_event.is_set():
                logger.info(f"==> Stopping {task['name']} pass for {video_name} due to user request")
                break

            if completed_splatting_task:
                logger.debug(f"==> Splatted {task['name']} video saved for {video_name}.")
                any_task_completed_successfully_for_this_video = True 

                if video_reader_input is not None: del video_reader_input
                if depth_reader_input is not None: del depth_reader_input
                torch.cuda.empty_cache()
                gc.collect()
                logger.debug("Explicitly deleted VideoReader objects and forced garbage collection to release file handles.")
            else:
                logger.info(f"==> Splatting task '{task['name']}' for '{video_name}' was skipped or failed. Files will NOT be moved.")
                if video_reader_input: del video_reader_input
                if depth_reader_input: del depth_reader_input
                torch.cuda.empty_cache()
                gc.collect()

            local_task_counter += 1
            processed_tasks_count += 1
            logger.debug(f"==> Completed {task['name']} pass for {video_name}.")

        # Update processing information for this video
        video_filename = os.path.basename(video_path)
        self.processing_filename = video_filename
        self.processing_task_name = "Processing"
        self.processing_resolution = f"{current_processed_width}x{current_processed_height}"
        self.processing_frames = f"{total_frames_input}"
        self.processing_gamma = f"{current_depth_gamma:.2f}"
        self.processing_disparity = f"{current_max_disparity_percentage:.1f}%"
        self.processing_convergence = f"{current_zero_disparity_anchor:.2f}"
        self.processing_map = map_source

        # After all tasks for the current video are processed or stopped
        if self.stop_event.is_set():
            return expected_task_count, any_task_completed_successfully_for_this_video

        # Move to finished logic 
        move_enabled = settings["move_to_finished"] # Use the setting from the dictionary

        if is_single_file_mode:
            # CRITICAL FIX: Get the finished folders directly from settings (set in start_single_processing)
            single_finished_src = settings.get("single_finished_source_folder")
            single_finished_depth = settings.get("single_finished_depth_folder")

            # --- Check move_enabled setting ---
            if single_finished_src and single_finished_depth and move_enabled and any_task_completed_successfully_for_this_video:
                self._move_processed_files(video_path, actual_depth_map_path, single_finished_src, single_finished_depth)
            else:
                logger.debug(f"Single file move skipped. Enabled={move_enabled}, Success={any_task_completed_successfully_for_this_video}, PathsValid={bool(single_finished_src)}")
                
        elif any_task_completed_successfully_for_this_video and finished_source_folder and finished_depth_folder and move_enabled:
            # Batch mode move (uses the arguments passed from _run_batch_process)
            self._move_processed_files(video_path, actual_depth_map_path, finished_source_folder, finished_depth_folder)

        # Return the number of tasks actually processed for the global counter update
        return processed_tasks_count, any_task_completed_successfully_for_this_video

    def _save_current_sidecar_data(self, is_auto_save: bool = False) -> bool:
        """
        Core method to prepare data and save the sidecar file.

        Args:
            is_auto_save (bool): If True, logs are DEBUG/INFO, otherwise ERROR.
        
        Returns:
            bool: True on success, False on failure.
        """
        result = self._get_current_sidecar_paths_and_data()
        if result is None:
            return False

        json_sidecar_path, depth_map_path, current_data = result
        
        # 1. Get current GUI values (the data to override/save)
        try:
            gui_save_data = {
                "convergence_plane": float(self.zero_disparity_anchor),
                "max_disparity": float(self.max_disp),
                "gamma": float(self.depth_gamma),
                "depth_dilate_size_x": float(self.depth_dilate_size_x),
                "depth_dilate_size_y": float(self.depth_dilate_size_y),
                "depth_blur_size_x": float(self.depth_blur_size_x),
                "depth_blur_size_y": float(self.depth_blur_size_y),
                "selected_depth_map": self.selected_depth_map,
            }
        except ValueError:
            logger.error("Sidecar Save: Invalid input value in GUI. Skipping save.")
            return False
        
        # 2. Merge GUI values into current data (preserving overlap/bias)
        current_data.update(gui_save_data)
        
        # 3. Write the updated data back to the file using the manager
        if self.sidecar_manager.save_sidecar_data(json_sidecar_path, current_data):
            action = "Auto-Saved" if is_auto_save else ("Updated" if os.path.exists(json_sidecar_path) else "Created")
            
            logger.info(f"{action} sidecar: {os.path.basename(json_sidecar_path)}")
            
            # Update button text in case a file was just created
            return True
        else:
            logger.error(f"Sidecar Save: Failed to write sidecar file '{os.path.basename(json_sidecar_path)}'.")
            return False

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
                finished_source_folder = os.path.join(input_source_clips_path, "finished")
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
                input_videos.extend(glob.glob(os.path.join(input_source_clips_path, ext)))
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
                self.progress_queue.put(("error", setup_result["error"], 0))
                return

            input_videos = setup_result["input_videos"]
            is_single_file_mode = setup_result["is_single_file_mode"]
            finished_source_folder = setup_result["finished_source_folder"]
            finished_depth_folder = setup_result["finished_depth_folder"]

            if not input_videos:
                logger.error("No input videos found for processing.")
                self.progress_queue.put(("error", "No input videos found for processing.", 0))
                return

            # --- 2. Apply From/To range on the *preview list* when available ---
            # In single-file mode, we always process the one file and ignore From/To.
            if not is_single_file_mode:
                # Multi-file mode: treat From/To as simple 1-based indices over the discovered input_videos list
                total_videos = len(input_videos)
                start_index_0 = 0
                end_index_0 = total_videos

                from_str = settings.get("process_from", "")
                if from_str:
                    try:
                        from_val = int(from_str)
                        if from_val > 0:
                            start_index_0 = max(0, min(total_videos, from_val - 1))
                    except ValueError:
                        logger.warning(f"Invalid 'From' value '{from_str}', ignoring.")

                to_str = settings.get("process_to", "")
                if to_str:
                    try:
                        to_val = int(to_str)
                        if to_val > 0:
                            end_index_0 = max(start_index_0 + 1, min(total_videos, to_val))
                    except ValueError:
                        logger.warning(f"Invalid 'To' value '{to_str}', ignoring.")

                if start_index_0 > 0 or end_index_0 < total_videos:
                    logger.info(
                        f"Processing range: videos {start_index_0 + 1} to {end_index_0} "
                        f"(out of {total_videos} total)"
                    )
                input_videos = input_videos[start_index_0:end_index_0]

            # After applying the range, make sure we still have something to do
            if not input_videos:
                logger.error("No input videos left to process after applying From/To range.")
                self.progress_queue.put(("error", "No input videos left to process after applying From/To range.", 0))
                return

            # --- 3. Determine total tasks for the progress bar ---
            processing_tasks = self._get_defined_tasks(settings)
            if not processing_tasks:
                logger.error("No processing tasks defined. Please enable at least one output resolution.")
                self.progress_queue.put(("error", "No processing tasks defined. Please enable at least one output resolution.", 0))
                return

            tasks_per_video = len(processing_tasks)
            total_tasks = len(input_videos) * tasks_per_video
            logger.info(
                f"Total tasks to process: {total_tasks} "
                f"({len(input_videos)} videos × {tasks_per_video} tasks each)"
            )

            overall_task_counter = 0

            # --- 4. Main processing loop ---
            for idx, video_path in enumerate(input_videos):
                if self.stop_event.is_set():
                    logger.info("==> Stopping processing due to user request")
                    self.progress_queue.put(("stopped", "Processing stopped by user", 0))
                    break

                logger.info(f"Processing video {idx + 1}/{len(input_videos)}: {os.path.basename(video_path)}")

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

            # Signal completion
            if not self.stop_event.is_set():
                logger.info(f"✅ All processing completed! Processed {len(input_videos)} video(s)")
                self.progress_queue.put(("complete", f"✅ Processing complete! Processed {len(input_videos)} video(s)", 100))

        except Exception as e:
            logger.error(f"An unexpected error occurred during batch processing: {e}", exc_info=True)
            self.progress_queue.put(("error", f"Error: {str(e)}", 0))
        finally:
            release_cuda_memory()

    def run_fusion_sidecar_generator(self):
        """Initializes and runs the FusionSidecarGenerator tool."""
        # Use an external thread to prevent the GUI from freezing during the file scan
        def worker():
            logger.info("Starting Fusion Export Sidecar Generation...")
            generator = FusionSidecarGenerator(self, self.sidecar_manager)
            generator.generate_sidecars()
            
        threading.Thread(target=worker, daemon=True).start()

    def run_preview_auto_converge_with_mode(self, mode: str, preview_format: str = "Side-by-Side"):
        """
        Wrapper that updates the mode and preview format before running auto-converge.
        """
        self.auto_convergence_mode = mode
        self.preview_format = preview_format
        return self.run_preview_auto_converge()
    
    def generate_manual_preview(self, selected_video: str, frame_number: int, convergence: float, max_disparity: float, preview_format: str):
        """
        Generate preview with manual frame and settings selection.
        
        Args:
            selected_video: Filename of the selected video
            frame_number: Frame number to preview
            convergence: Convergence value to apply
            max_disparity: Maximum disparity value
            preview_format: Preview format (Side-by-Side, Anaglyph, etc.)
            
        Returns:
            (preview_image, status_message, updated_slider)
        """
        try:
            if not selected_video:
                return None, "⚠️ No video selected", gr.Slider()
            
            # Find the full path of the selected video and corresponding depth map
            video_files = self._scan_video_files(self.input_source_clips)
            depth_files = self._scan_video_files(self.input_depth_maps)
            
            if not video_files or not depth_files:
                return None, "⚠️ No video or depth files found", gr.Slider()
            
            # Find the selected video
            video_path = None
            for vf in video_files:
                if os.path.basename(vf) == selected_video:
                    video_path = vf
                    break
            
            if not video_path:
                return None, f"⚠️ Video not found: {selected_video}", gr.Slider()
            
            # Find corresponding depth map (same base name)
            video_basename = os.path.splitext(selected_video)[0]
            depth_path = None
            
            for df in depth_files:
                depth_basename = os.path.splitext(os.path.basename(df))[0]
                if depth_basename == video_basename:
                    depth_path = df
                    break
            
            if not depth_path:
                # If no exact match, use first depth map as fallback
                depth_path = depth_files[0]
                logger.warning(f"No matching depth map found for {selected_video}, using {os.path.basename(depth_path)}")
            
            # Get total frames first to validate
            import cv2
            temp_cap = cv2.VideoCapture(video_path)
            if not temp_cap.isOpened():
                temp_cap.release()
                logger.error(f"Failed to open video: {video_path}")
                return None, f"❌ Failed to open: {selected_video}", gr.Slider()
            
            total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            temp_cap.release()
            
            if total_frames == 0:
                return None, f"❌ Video has 0 frames: {selected_video}", gr.Slider()
            
            # Clamp frame number to valid range
            frame_number = max(0, min(int(frame_number), total_frames - 1))
            
            # Generate preview with specified frame and settings
            self.preview_format = preview_format
            preview_image, _ = self._generate_preview_frame_at_frame_number(
                video_path, depth_path, convergence, max_disparity, frame_number
            )
            
            if preview_image is None:
                return None, f"❌ Failed to generate preview for {selected_video}", gr.Slider()
            
            status_msg = f"✅ Frame {frame_number}/{total_frames-1} | Conv: {convergence:.2f} | Disp: {max_disparity:.0f}"
            # Return updated slider with correct label
            return preview_image, status_msg, gr.Slider(value=frame_number, label=f"Frame (0-{total_frames-1})")
            
        except Exception as e:
            logger.error(f"Manual preview error: {e}")
            import traceback
            traceback.print_exc()
            return None, f"❌ Error: {str(e)}", gr.Slider()
    
    def refresh_video_list(self):
        """
        Refresh the video list dropdown with files from input folder.
        Returns: (updated_dropdown_choices, status_message)
        """
        try:
            video_files = self._scan_video_files(self.input_source_clips)
            
            if not video_files:
                return gr.Dropdown(choices=[]), "⚠️ No video files found in input folder"
            
            # Extract just the filenames for the dropdown
            video_names = [os.path.basename(f) for f in video_files]
            
            status = f"✅ Found {len(video_names)} video(s)"
            
            # Return dropdown with choices and select first one by default
            return gr.Dropdown(choices=video_names, value=video_names[0] if video_names else None), status
            
        except Exception as e:
            logger.error(f"Video list refresh error: {e}")
            return gr.Dropdown(choices=[]), f"❌ Error: {str(e)}"
    
    def detect_video_frames(self, selected_video: str):
        """
        Detect total frames in the selected video and update the frame slider.
        
        Args:
            selected_video: Filename of the selected video
            
        Returns: (updated_slider, status_message)
        """
        try:
            if not selected_video:
                return gr.Slider(value=0, maximum=1000, step=1), "⚠️ No video selected"
            
            # Find the full path of the selected video
            video_files = self._scan_video_files(self.input_source_clips)
            video_path = None
            
            for vf in video_files:
                if os.path.basename(vf) == selected_video:
                    video_path = vf
                    break
            
            if not video_path:
                return gr.Slider(value=0, maximum=1000, step=1), f"⚠️ Video not found: {selected_video}"
            
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                logger.error(f"Failed to open video: {video_path}")
                return gr.Slider(value=0, maximum=1000, step=1), f"❌ Failed to open: {selected_video}"
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if total_frames == 0:
                return gr.Slider(value=0, maximum=1000, step=1), f"❌ Video has 0 frames: {selected_video}"
            
            status = f"✅ {selected_video}: {total_frames} frames @ {fps:.2f} fps"
            
            logger.info(f"Detected {total_frames} frames in {selected_video}")
            # Return updated slider with correct maximum
            return gr.Slider(value=0, maximum=total_frames - 1, step=1, label=f"Frame (0-{total_frames-1})"), status
            
        except Exception as e:
            logger.error(f"Frame detection error: {e}")
            import traceback
            traceback.print_exc()
            return gr.Slider(value=0, maximum=1000, step=1), f"❌ Error: {str(e)}"
    
    def run_preview_auto_converge(self, force_run=False):
        """
        Analyzes the first video/depth pair to calculate optimal convergence.
        Returns (status_message, updated_convergence_value, preview_image).
        """
        try:
            # Get first video and depth map from input folders
            video_files = self._scan_video_files(self.input_source_clips)
            depth_files = self._scan_video_files(self.input_depth_maps)
            
            if not video_files or not depth_files:
                return "⚠️ No video or depth files found in input folders", self.zero_disparity_anchor, None
            
            video_path = video_files[0]
            depth_path = depth_files[0]
            
            logger.info(f"Running Auto-Converge analysis on: {os.path.basename(video_path)}")
            
            # Analyze depth map to find optimal convergence
            mode = self.auto_convergence_mode
            
            if mode == "Off":
                return "⚠️ Auto-Convergence Mode is set to 'Off'. Please select 'Average' or 'Peak'.", self.zero_disparity_anchor, None
            
            # Calculate convergence value
            convergence_value = self._calculate_convergence(depth_path, mode)
            
            if convergence_value is None:
                return "❌ Failed to calculate convergence", self.zero_disparity_anchor, None
            
            # Generate preview image
            preview_image = self._generate_preview_frame(video_path, depth_path, convergence_value)
            
            # Update the convergence slider value
            self.zero_disparity_anchor = convergence_value
            
            status_msg = f"✅ Auto-Convergence complete!\n{mode} convergence: {convergence_value:.3f}\nZero Disparity Anchor updated."
            return status_msg, convergence_value, preview_image
            
        except Exception as e:
            logger.error(f"Auto-Converge error: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}", self.zero_disparity_anchor, None
    
    def _scan_video_files(self, folder_path: str) -> List[str]:
        """
        Scan folder for video files.
        
        Args:
            folder_path: Path to folder to scan
            
        Returns:
            List of video file paths
        """
        if not folder_path or not os.path.exists(folder_path):
            return []
        
        video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.m4v')
        video_files = []
        
        try:
            for filename in sorted(os.listdir(folder_path)):
                if filename.lower().endswith(video_extensions):
                    full_path = os.path.join(folder_path, filename)
                    if os.path.isfile(full_path):
                        video_files.append(full_path)
        except Exception as e:
            logger.error(f"Error scanning folder {folder_path}: {e}")
        
        return video_files
    
    def _generate_preview_frame_at_frame_number(self, video_path: str, depth_path: str, convergence: float, max_disparity: float, frame_number: int) -> tuple[Optional[np.ndarray], int]:
        """
        Generate preview at a specific frame number using decord for accurate seeking.
        
        Args:
            video_path: Path to source video
            depth_path: Path to depth map
            convergence: Convergence value to apply
            max_disparity: Maximum disparity value
            frame_number: Frame number to preview
            
        Returns:
            (Preview image (numpy array) or None on error, total_frames)
        """
        import torch
        from gui.warp import ForwardWarpStereo
        from decord import VideoReader, cpu
        
        try:
            # Open video and depth with decord (much better seeking than OpenCV)
            video_reader = VideoReader(video_path, ctx=cpu(0))
            depth_reader = VideoReader(depth_path, ctx=cpu(0))
            
            video_total_frames = len(video_reader)
            depth_total_frames = len(depth_reader)
            
            if video_total_frames == 0:
                logger.error(f"Video has 0 frames: {video_path}")
                return None, 0
            
            if depth_total_frames == 0:
                logger.error(f"Depth map has 0 frames: {depth_path}")
                return None, 0
            
            # Clamp frame number to valid range for video
            frame_idx = max(0, min(frame_number, video_total_frames - 1))
            
            # Handle frame count mismatch between video and depth
            if video_total_frames != depth_total_frames:
                # Scale the frame index proportionally for the depth map
                depth_frame_idx = int((frame_idx / (video_total_frames - 1)) * (depth_total_frames - 1))
                depth_frame_idx = max(0, min(depth_frame_idx, depth_total_frames - 1))
                logger.warning(
                    f"Frame count mismatch: video={video_total_frames}, depth={depth_total_frames}. "
                    f"Mapping video frame {frame_idx} to depth frame {depth_frame_idx}"
                )
            else:
                depth_frame_idx = frame_idx
            
            logger.info(f"Reading frame {frame_idx} of {video_total_frames} from {os.path.basename(video_path)}")
            
            # Decord provides accurate frame-level seeking
            video_frame = video_reader[frame_idx].asnumpy()  # Returns RGB numpy array (H, W, 3)
            depth_frame = depth_reader[depth_frame_idx].asnumpy()  # Returns RGB numpy array (H, W, 3)
            
            # video_frame is already in RGB format from decord
            video_frame_rgb = video_frame
            
            # Process depth - convert to grayscale if needed
            if len(depth_frame.shape) == 3:
                # Convert RGB to grayscale
                depth_frame = np.dot(depth_frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            
            # Resize depth to match video dimensions
            h, w = video_frame_rgb.shape[:2]
            if depth_frame.shape[:2] != (h, w):
                import cv2
                depth_frame = cv2.resize(depth_frame, (w, h), interpolation=cv2.INTER_LINEAR)
            
            depth_normalized = depth_frame.astype(np.float32) / 255.0
            
            # Apply gamma
            if self.depth_gamma != 1.0:
                depth_normalized = 1.0 - np.power(1.0 - depth_normalized, self.depth_gamma)
            
            # Apply convergence
            depth_with_convergence = depth_normalized - convergence
            depth_with_convergence = np.clip(depth_with_convergence, -1.0, 1.0)
            
            # Convert to torch tensors
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            video_tensor = torch.from_numpy(video_frame_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            depth_tensor = torch.from_numpy(depth_with_convergence).unsqueeze(0).unsqueeze(0).float().to(device)
            
            # Create forward warper
            forward_warp = ForwardWarpStereo(eps=1e-6, occlu_map=False)
            forward_warp = forward_warp.to(device)
            
            # Warp for right eye (positive disparity) using preview max_disparity
            disparity = depth_tensor * max_disparity
            warped_right = forward_warp(video_tensor, disparity)
            
            # Convert back to numpy
            warped_right_np = (warped_right.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Generate preview based on format
            preview_format = getattr(self, 'preview_format', 'Side-by-Side')
            
            if preview_format == "Anaglyph (Red/Cyan)":
                # Create anaglyph: Red channel from left, Cyan (GB) from right
                anaglyph = np.zeros_like(video_frame_rgb)
                anaglyph[:, :, 0] = video_frame_rgb[:, :, 0]  # Red from left eye
                anaglyph[:, :, 1] = warped_right_np[:, :, 1]   # Green from right eye
                anaglyph[:, :, 2] = warped_right_np[:, :, 2]   # Blue from right eye
                comparison = anaglyph
            elif preview_format == "Depth Map":
                # Show depth map visualization
                import cv2
                depth_vis = (depth_with_convergence * 127.5 + 127.5).astype(np.uint8)
                depth_vis_rgb = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
                depth_vis_rgb = cv2.cvtColor(depth_vis_rgb, cv2.COLOR_BGR2RGB)
                comparison = np.hstack([video_frame_rgb, depth_vis_rgb])
            else:  # Side-by-Side
                comparison = np.hstack([video_frame_rgb, warped_right_np])
            
            # Clean up
            del video_tensor, depth_tensor, warped_right, forward_warp
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            logger.info(f"Preview generated successfully: frame {frame_idx}/{video_total_frames-1}, convergence {convergence:.3f}, disparity {max_disparity:.1f}")
            return comparison, video_total_frames
            
        except Exception as e:
            logger.error(f"Error generating manual preview: {e}")
            import traceback
            traceback.print_exc()
            return None, 0
    
    def _generate_preview_frame_at_position(self, video_path: str, depth_path: str, convergence: float, frame_position: float) -> Optional[np.ndarray]:
        """
        Generate preview at a specific frame position.
        
        Args:
            video_path: Path to source video
            depth_path: Path to depth map
            convergence: Convergence value to apply
            frame_position: Frame position as percentage (0-100)
            
        Returns:
            Preview image (numpy array) or None on error
        """
        import cv2
        import torch
        from gui.warp import ForwardWarpStereo
        
        try:
            # Open video and depth
            video_cap = cv2.VideoCapture(video_path)
            depth_cap = cv2.VideoCapture(depth_path)
            
            if not video_cap.isOpened() or not depth_cap.isOpened():
                logger.error("Failed to open video or depth map")
                return None
            
            # Calculate frame index from position
            total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = int((frame_position / 100.0) * (total_frames - 1))
            frame_idx = max(0, min(frame_idx, total_frames - 1))
            
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            depth_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret_v, video_frame = video_cap.read()
            ret_d, depth_frame = depth_cap.read()
            
            video_cap.release()
            depth_cap.release()
            
            if not ret_v or not ret_d:
                logger.error("Failed to read frames")
                return None
            
            # Convert video frame to RGB
            video_frame_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            
            # Process depth
            if len(depth_frame.shape) == 3:
                depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
            
            # Resize depth to match video dimensions
            h, w = video_frame_rgb.shape[:2]
            if depth_frame.shape[:2] != (h, w):
                depth_frame = cv2.resize(depth_frame, (w, h), interpolation=cv2.INTER_LINEAR)
            
            depth_normalized = depth_frame.astype(np.float32) / 255.0
            
            # Apply gamma
            if self.depth_gamma != 1.0:
                depth_normalized = 1.0 - np.power(1.0 - depth_normalized, self.depth_gamma)
            
            # Apply convergence
            depth_with_convergence = depth_normalized - convergence
            depth_with_convergence = np.clip(depth_with_convergence, -1.0, 1.0)
            
            # Convert to torch tensors
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            video_tensor = torch.from_numpy(video_frame_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            depth_tensor = torch.from_numpy(depth_with_convergence).unsqueeze(0).unsqueeze(0).float().to(device)
            
            # Apply forward warp
            max_disp = self.max_disp
            
            # Create forward warper
            forward_warp = ForwardWarpStereo(eps=1e-6, occlu_map=False)
            forward_warp = forward_warp.to(device)
            
            # Warp for right eye (positive disparity)
            disparity = depth_tensor * max_disp
            warped_right = forward_warp(video_tensor, disparity)
            
            # Convert back to numpy
            warped_right_np = (warped_right.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Generate preview based on format
            preview_format = getattr(self, 'preview_format', 'Side-by-Side')
            
            if preview_format == "Anaglyph (Red/Cyan)":
                # Create anaglyph: Red channel from left, Cyan (GB) from right
                anaglyph = np.zeros_like(video_frame_rgb)
                anaglyph[:, :, 0] = video_frame_rgb[:, :, 0]  # Red from left eye
                anaglyph[:, :, 1] = warped_right_np[:, :, 1]   # Green from right eye
                anaglyph[:, :, 2] = warped_right_np[:, :, 2]   # Blue from right eye
                comparison = anaglyph
            elif preview_format == "Depth Map":
                # Show depth map visualization
                depth_vis = (depth_with_convergence * 127.5 + 127.5).astype(np.uint8)
                depth_vis_rgb = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
                depth_vis_rgb = cv2.cvtColor(depth_vis_rgb, cv2.COLOR_BGR2RGB)
                comparison = np.hstack([video_frame_rgb, depth_vis_rgb])
            else:  # Side-by-Side
                comparison = np.hstack([video_frame_rgb, warped_right_np])
            
            # Clean up
            del video_tensor, depth_tensor, warped_right, forward_warp
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            logger.info(f"Manual preview generated: frame {frame_idx}/{total_frames}, convergence {convergence:.3f}")
            return comparison
            
        except Exception as e:
            logger.error(f"Error generating manual preview: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_preview_frame(self, video_path: str, depth_path: str, convergence: float) -> Optional[np.ndarray]:
        """
        Generate a side-by-side preview showing original frame and splatted result.
        
        Args:
            video_path: Path to source video
            depth_path: Path to depth map
            convergence: Convergence value to apply
            
        Returns:
            Side-by-side comparison image (numpy array) or None on error
        """
        import cv2
        import torch
        from gui.warp import ForwardWarpStereo
        
        try:
            # Open video and depth
            video_cap = cv2.VideoCapture(video_path)
            depth_cap = cv2.VideoCapture(depth_path)
            
            if not video_cap.isOpened() or not depth_cap.isOpened():
                logger.error("Failed to open video or depth map")
                return None
            
            # Get middle frame
            total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame_idx = total_frames // 2
            
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            depth_cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            
            ret_v, video_frame = video_cap.read()
            ret_d, depth_frame = depth_cap.read()
            
            video_cap.release()
            depth_cap.release()
            
            if not ret_v or not ret_d:
                logger.error("Failed to read frames")
                return None
            
            # Convert video frame to RGB
            video_frame_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            
            # Process depth
            if len(depth_frame.shape) == 3:
                depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
            
            # Resize depth to match video dimensions
            h, w = video_frame_rgb.shape[:2]
            if depth_frame.shape[:2] != (h, w):
                depth_frame = cv2.resize(depth_frame, (w, h), interpolation=cv2.INTER_LINEAR)
            
            depth_normalized = depth_frame.astype(np.float32) / 255.0
            
            # Apply gamma
            if self.depth_gamma != 1.0:
                depth_normalized = 1.0 - np.power(1.0 - depth_normalized, self.depth_gamma)
            
            # Apply convergence
            depth_with_convergence = depth_normalized - convergence
            depth_with_convergence = np.clip(depth_with_convergence, -1.0, 1.0)
            
            # Convert to torch tensors
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            video_tensor = torch.from_numpy(video_frame_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            depth_tensor = torch.from_numpy(depth_with_convergence).unsqueeze(0).unsqueeze(0).float().to(device)
            
            # Apply forward warp
            h, w = video_frame_rgb.shape[:2]
            max_disp = self.max_disp
            
            # Create forward warper
            forward_warp = ForwardWarpStereo(eps=1e-6, occlu_map=False)
            forward_warp = forward_warp.to(device)
            
            # Warp for right eye (positive disparity)
            disparity = depth_tensor * max_disp
            warped_right = forward_warp(video_tensor, disparity)
            
            # Convert back to numpy
            warped_right_np = (warped_right.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Generate preview based on format
            preview_format = getattr(self, 'preview_format', 'Side-by-Side')
            
            if preview_format == "Anaglyph (Red/Cyan)":
                # Create anaglyph: Red channel from left, Cyan (GB) from right
                anaglyph = np.zeros_like(video_frame_rgb)
                anaglyph[:, :, 0] = video_frame_rgb[:, :, 0]  # Red from left eye
                anaglyph[:, :, 1] = warped_right_np[:, :, 1]   # Green from right eye
                anaglyph[:, :, 2] = warped_right_np[:, :, 2]   # Blue from right eye
                comparison = anaglyph
            elif preview_format == "Depth Map":
                # Show depth map visualization
                depth_vis = (depth_with_convergence * 127.5 + 127.5).astype(np.uint8)
                depth_vis_rgb = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
                depth_vis_rgb = cv2.cvtColor(depth_vis_rgb, cv2.COLOR_BGR2RGB)
                comparison = np.hstack([video_frame_rgb, depth_vis_rgb])
            else:  # Side-by-Side
                comparison = np.hstack([video_frame_rgb, warped_right_np])
            
            # Clean up
            del video_tensor, depth_tensor, warped_right, forward_warp
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            logger.info("Preview frame generated successfully")
            return comparison
            
        except Exception as e:
            logger.error(f"Error generating preview: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_convergence(self, depth_path: str, mode: str, sample_frames: int = 10) -> Optional[float]:
        """
        Calculate optimal convergence point from depth map.
        
        Args:
            depth_path: Path to depth map video
            mode: "Average" or "Peak"
            sample_frames: Number of frames to sample
            
        Returns:
            Convergence value (0.0 to 1.0) or None on error
        """
        import cv2
        import numpy as np
        
        try:
            cap = cv2.VideoCapture(depth_path)
            if not cap.isOpened():
                logger.error(f"Failed to open depth map: {depth_path}")
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return None
            
            # Sample frames evenly across the video
            sample_indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)
            
            depth_values = []
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Normalize to 0-1 range
                depth_normalized = frame.astype(np.float32) / 255.0
                
                # Apply gamma correction if set
                if self.depth_gamma != 1.0:
                    depth_normalized = 1.0 - np.power(1.0 - depth_normalized, self.depth_gamma)
                
                depth_values.append(depth_normalized)
            
            cap.release()
            
            if not depth_values:
                return None
            
            # Stack all sampled frames
            depth_stack = np.stack(depth_values, axis=0)
            
            # Calculate convergence based on mode
            if mode == "Average":
                # Use mean depth value across all samples
                convergence = float(np.mean(depth_stack))
            elif mode == "Peak":
                # Use the most common depth value (histogram peak)
                hist, bins = np.histogram(depth_stack.flatten(), bins=100, range=(0, 1))
                peak_idx = np.argmax(hist)
                convergence = float((bins[peak_idx] + bins[peak_idx + 1]) / 2)
            else:
                # Default to average
                convergence = float(np.mean(depth_stack))
            
            # Clamp to valid range
            convergence = np.clip(convergence, 0.0, 1.0)
            
            logger.info(f"Calculated {mode} convergence: {convergence:.3f}")
            return convergence
            
        except Exception as e:
            logger.error(f"Error calculating convergence: {e}")
            return None

    def _save_current_settings_and_notify(self):
        """Saves current GUI settings to config_splat.json and notifies the user."""
        config_filename = self.APP_CONFIG_DEFAULTS["DEFAULT_CONFIG_FILENAME"]
        try:
            self._save_config()
            logger.info(f"Settings saved to {config_filename}.")
        except Exception as e:
            logger.error(f"Failed to save settings to {config_filename}:\n{e}")

    def _save_config(self):
        """Saves current GUI settings to the default file."""
        config = self._get_current_config()
        config_filename = self.APP_CONFIG_DEFAULTS["DEFAULT_CONFIG_FILENAME"]
        with open(config_filename, "w") as f:
            json.dump(config, f, indent=4)

    def start_processing(self, 
                         input_source_clips, input_depth_maps, output_splatted,
                         max_disp, process_length, enable_full_res, batch_size,
                         enable_low_res, pre_res_width, pre_res_height, low_res_batch_size,
                         dual_output, zero_disparity_anchor, enable_global_norm,
                         output_crf_full, depth_gamma, depth_dilate_size_x, depth_dilate_size_y,
                         depth_blur_size_x, depth_blur_size_y,
                         auto_convergence_mode,
                         move_to_finished, process_from, process_to,
                         # New parameters
                         output_crf_low, depth_dilate_left, depth_blur_left, depth_blur_left_mix,
                         border_width, border_bias, border_mode, color_tags_mode):
        """Starts the video processing in a separate thread."""
        self.stop_event.clear()
        
        # Clear the progress queue
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except queue.Empty:
                break
        
        # Convert types from Gradio inputs (they may come as strings)
        try:
            max_disp = float(max_disp)
            process_length = int(process_length)
            batch_size = int(batch_size)
            pre_res_width = int(pre_res_width)
            pre_res_height = int(pre_res_height)
            low_res_batch_size = int(low_res_batch_size)
            zero_disparity_anchor = float(zero_disparity_anchor)
            output_crf_full = int(output_crf_full) if output_crf_full else 18
            output_crf_low = int(output_crf_low) if output_crf_low else 18
            depth_gamma = float(depth_gamma)
            depth_dilate_size_x = float(depth_dilate_size_x)
            depth_dilate_size_y = float(depth_dilate_size_y)
            depth_blur_size_x = int(float(depth_blur_size_x))
            depth_blur_size_y = int(float(depth_blur_size_y))
            depth_dilate_left = float(depth_dilate_left) if depth_dilate_left else 0.0
            depth_blur_left = int(float(depth_blur_left)) if depth_blur_left else 0
            depth_blur_left_mix = float(depth_blur_left_mix) if depth_blur_left_mix else 0.5
            border_width = float(border_width) if border_width else 0.0
            border_bias = float(border_bias) if border_bias else 0.0
            process_from = str(process_from).strip() if process_from else ""
            process_to = str(process_to).strip() if process_to else ""
        except (ValueError, TypeError) as e:
            return f"Error: Invalid input type - {str(e)}", 0, gr.Button(interactive=True), gr.Button(interactive=True), gr.Button(interactive=False)
        
        # Update all the parameters from the UI inputs
        self.input_source_clips = input_source_clips
        self.input_depth_maps = input_depth_maps
        self.output_splatted = output_splatted
        self.max_disp = max_disp
        self.process_length = process_length
        self.enable_full_res = enable_full_res
        self.batch_size = batch_size
        self.enable_low_res = enable_low_res
        self.pre_res_width = pre_res_width
        self.pre_res_height = pre_res_height
        self.low_res_batch_size = low_res_batch_size
        self.dual_output = dual_output
        self.zero_disparity_anchor = zero_disparity_anchor
        self.enable_global_norm = enable_global_norm
        self.output_crf = output_crf_full  # Keep for backward compatibility
        self.output_crf_full = output_crf_full
        self.output_crf_low = output_crf_low
        self.depth_gamma = depth_gamma
        self.depth_dilate_size_x = depth_dilate_size_x
        self.depth_dilate_size_y = depth_dilate_size_y
        self.depth_blur_size_x = depth_blur_size_x
        self.depth_blur_size_y = depth_blur_size_y
        self.depth_dilate_left = depth_dilate_left
        self.depth_blur_left = depth_blur_left
        self.depth_blur_left_mix = depth_blur_left_mix
        self.border_width = border_width
        self.border_bias = border_bias
        self.border_mode = border_mode
        self.auto_convergence_mode = auto_convergence_mode
        self.color_tags_mode = color_tags_mode
        self.move_to_finished = move_to_finished
        self.process_from = process_from
        self.process_to = process_to

        # Input validation for all fields
        try:
            if max_disp <= 0:
                raise ValueError("Max Disparity must be positive.")

            if not (0.0 <= zero_disparity_anchor <= 2.0):
                raise ValueError("Zero Disparity Anchor must be between 0.0 and 2.0.")

            if enable_full_res:
                if batch_size <= 0:
                    raise ValueError("Full Resolution Batch Size must be positive.")

            if enable_low_res:
                if pre_res_width <= 0 or pre_res_height <= 0:
                    raise ValueError("Low-Resolution Width and Height must be positive.")
                if low_res_batch_size <= 0:
                    raise ValueError("Low-Resolution Batch Size must be positive.")

            if not (enable_full_res or enable_low_res):
                raise ValueError("At least one resolution (Full or Low) must be enabled to start processing.")
            
            # Depth Pre-processing Validation
            if depth_gamma <= 0:
                raise ValueError("Depth Gamma must be positive.")
            
            # Validate Dilate X/Y
            if depth_dilate_size_x < -10 or depth_dilate_size_y < -10:
                raise ValueError("Depth Dilate Sizes (X/Y) must be >= -10.")
            
            # Validate Blur X/Y
            if depth_blur_size_x < 0 or depth_blur_size_y < 0:
                raise ValueError("Depth Blur Sizes (X/Y) must be non-negative.")
            
            # Validate Left-edge processing
            if depth_dilate_left < 0:
                raise ValueError("Depth Dilate Left must be non-negative.")
            if depth_blur_left < 0:
                raise ValueError("Depth Blur Left must be non-negative.")
            if not (0.0 <= depth_blur_left_mix <= 1.0):
                raise ValueError("Blur Left Mix must be between 0.0 and 1.0.")
            
            # Validate Border controls
            if border_width < 0 or border_width > 5.0:
                raise ValueError("Border Width must be between 0.0 and 5.0.")
            if border_bias < -1.0 or border_bias > 1.0:
                raise ValueError("Border Bias must be between -1.0 and 1.0.")

        except ValueError as e:
            return f"Error: {e}", 0

        settings = {
            "input_source_clips": input_source_clips,
            "input_depth_maps": input_depth_maps,
            "output_splatted": output_splatted,
            "max_disp": max_disp,
            "process_length": process_length,
            "enable_full_resolution": enable_full_res,
            "full_res_batch_size": batch_size,
            "enable_low_resolution": enable_low_res,
            "low_res_width": pre_res_width,
            "low_res_height": pre_res_height,
            "low_res_batch_size": low_res_batch_size,
            "dual_output": dual_output,
            "zero_disparity_anchor": zero_disparity_anchor,
            "enable_global_norm": enable_global_norm,
            "match_depth_res": True,
            "move_to_finished": move_to_finished,
            "output_crf": output_crf_full,  # Use output_crf_full for backward compatibility
            "output_crf_full": output_crf_full,
            "output_crf_low": output_crf_low,
            # Depth Pre-processing Settings
            "depth_gamma": depth_gamma,
            "depth_dilate_size_x": depth_dilate_size_x,
            "depth_dilate_size_y": depth_dilate_size_y,
            "depth_blur_size_x": depth_blur_size_x,
            "depth_blur_size_y": depth_blur_size_y,
            "depth_dilate_left": depth_dilate_left,
            "depth_blur_left": depth_blur_left,
            "depth_blur_left_mix": depth_blur_left_mix,
            # Border Controls
            "border_width": border_width,
            "border_bias": border_bias,
            "border_mode": border_mode,
            # Auto-Convergence Settings
            "auto_convergence_mode": auto_convergence_mode,
            # Color Tags
            "color_tags_mode": color_tags_mode,
            # Sidecar settings
            "enable_sidecar_gamma": self.enable_sidecar_gamma,
            "enable_sidecar_blur_dilate": self.enable_sidecar_blur_dilate,
            # Range controls
            "process_from": process_from,
            "process_to": process_to,
        }
        
        # Run the processing in a separate thread
        self.processing_thread = threading.Thread(target=self._run_batch_process, args=(settings,))
        self.processing_thread.start()
        return "Processing started...", 50, gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=True)

    def check_processing_status(self):
        """Check if processing is complete and return updated UI state."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            # Check for any messages in the queue
            try:
                status_type, message, progress = self.progress_queue.get_nowait()
                if status_type == "complete":
                    return message, 100, gr.Button(interactive=True), gr.Button(interactive=True), gr.Button(interactive=False)
                elif status_type == "error":
                    return message, 0, gr.Button(interactive=True), gr.Button(interactive=True), gr.Button(interactive=False)
                elif status_type == "stopped":
                    return message, 0, gr.Button(interactive=True), gr.Button(interactive=True), gr.Button(interactive=False)
            except queue.Empty:
                # Thread finished but no status message - assume complete
                if self.processing_thread is not None and not self.processing_thread.is_alive():
                    return "✅ Processing complete!", 100, gr.Button(interactive=True), gr.Button(interactive=True), gr.Button(interactive=False)
        
        # Still processing - return no updates (use gr.update() to keep current values)
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    def start_single_processing(self,
                                input_source_clips, input_depth_maps, output_splatted,
                                max_disp, process_length, enable_full_res, batch_size,
                                enable_low_res, pre_res_width, pre_res_height, low_res_batch_size,
                                dual_output, zero_disparity_anchor, enable_global_norm,
                                output_crf_full, depth_gamma, depth_dilate_size_x, depth_dilate_size_y,
                                depth_blur_size_x, depth_blur_size_y,
                                auto_convergence_mode,
                                move_to_finished,
                                # New parameters
                                output_crf_low, depth_dilate_left, depth_blur_left, depth_blur_left_mix,
                                border_width, border_bias, border_mode, color_tags_mode):
        """
        Starts processing for the single video currently loaded in the previewer.
        It runs the batch logic in single-file mode.
        """
        # Convert types from Gradio inputs (they may come as strings)
        try:
            max_disp = float(max_disp)
            process_length = int(process_length)
            batch_size = int(batch_size)
            pre_res_width = int(pre_res_width)
            pre_res_height = int(pre_res_height)
            low_res_batch_size = int(low_res_batch_size)
            zero_disparity_anchor = float(zero_disparity_anchor)
            output_crf_full = int(output_crf_full) if output_crf_full else 18
            output_crf_low = int(output_crf_low) if output_crf_low else 18
            depth_gamma = float(depth_gamma)
            depth_dilate_size_x = float(depth_dilate_size_x)
            depth_dilate_size_y = float(depth_dilate_size_y)
            depth_blur_size_x = int(float(depth_blur_size_x))
            depth_blur_size_y = int(float(depth_blur_size_y))
            depth_dilate_left = float(depth_dilate_left) if depth_dilate_left else 0.0
            depth_blur_left = int(float(depth_blur_left)) if depth_blur_left else 0
            depth_blur_left_mix = float(depth_blur_left_mix) if depth_blur_left_mix else 0.5
            border_width = float(border_width) if border_width else 0.0
            border_bias = float(border_bias) if border_bias else 0.0
        except (ValueError, TypeError) as e:
            return f"Error: Invalid input type - {str(e)}", 0
        
        # Update all the parameters from the UI inputs
        self.input_source_clips = input_source_clips
        self.input_depth_maps = input_depth_maps
        self.output_splatted = output_splatted
        self.max_disp = max_disp
        self.process_length = process_length
        self.enable_full_res = enable_full_res
        self.batch_size = batch_size
        self.enable_low_res = enable_low_res
        self.pre_res_width = pre_res_width
        self.pre_res_height = pre_res_height
        self.low_res_batch_size = low_res_batch_size
        self.dual_output = dual_output
        self.zero_disparity_anchor = zero_disparity_anchor
        self.enable_global_norm = enable_global_norm
        self.output_crf = output_crf
        self.output_crf_full = output_crf_full
        self.output_crf_low = output_crf_low
        self.depth_gamma = depth_gamma
        self.depth_dilate_size_x = depth_dilate_size_x
        self.depth_dilate_size_y = depth_dilate_size_y
        self.depth_blur_size_x = depth_blur_size_x
        self.depth_blur_size_y = depth_blur_size_y
        self.depth_dilate_left = depth_dilate_left
        self.depth_blur_left = depth_blur_left
        self.depth_blur_left_mix = depth_blur_left_mix
        self.border_width = border_width
        self.border_bias = border_bias
        self.border_mode = border_mode
        self.auto_convergence_mode = auto_convergence_mode
        self.color_tags_mode = color_tags_mode
        self.move_to_finished = move_to_finished

        # 1. Get the current single file paths - for now we'll use the provided paths
        single_video_path = input_source_clips
        single_depth_path = input_depth_maps

        if not single_video_path or not single_depth_path:
            return "Error: Could not get both video and depth map paths", 0

        # 2. Perform validation checks
        try:
            # Full Resolution/Low Resolution checks
            if not (enable_full_res or enable_low_res):
                raise ValueError("At least one resolution (Full or Low) must be enabled to start processing.")
            
            # Simplified validation for speed/simplicity
            if max_disp <= 0:
                raise ValueError("Max Disparity must be positive.")
            
        except ValueError as e:
            return f"Error: {e}", 0

        # 3. Compile settings dictionary
        # We explicitly set the input paths to the single files, which forces batch logic 
        # to execute in single-file mode (checking os.path.isfile).
        
        # Determine Finished Folders for Single Process (only if enabled)
        single_finished_source_folder = None
        single_finished_depth_folder = None
        
        if move_to_finished:
            # We assume the finished folder is in the same directory as the original input file/depth map
            single_finished_source_folder = os.path.join(os.path.dirname(single_video_path), "finished")
            single_finished_depth_folder = os.path.join(os.path.dirname(single_depth_path), "finished")
            os.makedirs(single_finished_source_folder, exist_ok=True)
            os.makedirs(single_finished_depth_folder, exist_ok=True)
            logger.debug(f"Single Process: Finished folders set to: {single_finished_source_folder}")

        settings = {
            # OVERRIDDEN INPUTS FOR SINGLE MODE
            "input_source_clips": single_video_path,
            "input_depth_maps": single_depth_path,
            "output_splatted": output_splatted,
            
            "max_disp": max_disp,
            "process_length": process_length,
            "enable_full_resolution": enable_full_res,
            "full_res_batch_size": batch_size,
            "enable_low_resolution": enable_low_res,
            "low_res_width": pre_res_width,
            "low_res_height": pre_res_height,
            "low_res_batch_size": low_res_batch_size,
            "dual_output": dual_output,
            "zero_disparity_anchor": zero_disparity_anchor,
            "enable_global_norm": enable_global_norm,
            "match_depth_res": True,
            "output_crf": output_crf_full,  # Use output_crf_full for backward compatibility
            "output_crf_full": output_crf_full,
            "output_crf_low": output_crf_low,
            
            # Depth Pre-processing Settings
            "depth_gamma": depth_gamma,
            "depth_dilate_size_x": depth_dilate_size_x,
            "depth_dilate_size_y": depth_dilate_size_y,
            "depth_blur_size_x": depth_blur_size_x,
            "depth_blur_size_y": depth_blur_size_y,
            "depth_dilate_left": depth_dilate_left,
            "depth_blur_left": depth_blur_left,
            "depth_blur_left_mix": depth_blur_left_mix,
            
            # Border Controls
            "border_width": border_width,
            "border_bias": border_bias,
            "border_mode": border_mode,
            
            # Auto-Convergence & Color Tags
            "auto_convergence_mode": auto_convergence_mode,
            "color_tags_mode": color_tags_mode,
            
            # Sidecar settings
            "enable_sidecar_gamma": self.enable_sidecar_gamma,
            "enable_sidecar_blur_dilate": self.enable_sidecar_blur_dilate,
            
            # Single mode specific
            "single_finished_source_folder": single_finished_source_folder,
            "single_finished_depth_folder": single_finished_depth_folder,
            "move_to_finished": move_to_finished,
        }

        # 4. Start the processing thread
        self.stop_event.clear()

        self.processing_thread = threading.Thread(target=self._run_batch_process, args=(settings,))
        self.processing_thread.start()
        return "Single processing started...", 50, gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=True)

    def stop_processing(self):
        """Sets the stop event to gracefully halt processing."""
        self.stop_event.set()
        return "Stopping processing...", 0

    def _toggle_debug_logging(self):
        """Toggles debug logging and updates shared logger."""
        # For now, we'll just update the logging level
        set_logger_level(logger, logging.DEBUG if self._debug_logging_enabled else logging.INFO)

    def update_sidecar_file(self):
        """
        Saves the current GUI values to the sidecar file after checking for user confirmation.
        """
        # 1. Get current sidecar path and data (needed for overwrite check)
        result = self._get_current_sidecar_paths_and_data()
        if result is None:
            return "Please load a video in the Previewer first."
        
        json_sidecar_path, _, _ = result
        
        # 2. Call the core saving function
        if self._save_current_sidecar_data(is_auto_save=False):
            return f"Sidecar updated: {os.path.basename(json_sidecar_path)}"
        else:
            return f"Failed to update sidecar: {os.path.basename(json_sidecar_path)}"

    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="StereoCrafter Splatting WebUI") as interface:
            gr.Markdown(f"# Stereocrafter Splatting (Batch) {GUI_VERSION}")
            
            # Input/Output Folders Section (Top)
            with gr.Group():
                gr.Markdown("### Input/Output Folders")
                with gr.Row():
                    self.input_source_clips_comp = gr.Textbox(
                        label="Input Source Clips", 
                        value=self.input_source_clips, 
                        scale=3,
                        info="Path to the folder containing your input video clips (MP4, AVI, MOV, MKV)."
                    )
                with gr.Row():
                    self.input_depth_maps_comp = gr.Textbox(
                        label="Input Depth Maps", 
                        value=self.input_depth_maps, 
                        scale=3,
                        info="Path to the folder containing your pre-rendered depth maps. Depth maps should be named 'videoname_depth.mp4' matching your input videos."
                    )
                with gr.Row():
                    self.output_splatted_comp = gr.Textbox(
                        label="Output Splatted", 
                        value=self.output_splatted, 
                        scale=3,
                        info="Path to the folder where the splatted output videos will be saved."
                    )
                    self.multi_map_comp = gr.Checkbox(
                        label="Multi-Map", 
                        value=self.multi_map, 
                        scale=1,
                        info="Enable Multi-Map mode to use subfolders within the Input Depth Maps folder. Each subfolder containing *_depth.mp4 files will appear as a selectable option. Sidecars are stored in a 'sidecars' subfolder. Shortcut: 7 and 9 switches maps."
                    )
                
                # Conditional visibility for depth map subfolders when multi_map is enabled
                self.depth_map_subfolders_comp = gr.Dropdown(choices=[], label="Select Depth Map Subfolder", visible=False)
            
            # Main Settings Container - Two columns side by side
            with gr.Row():
                # Left Column
                with gr.Column():
                    # Process Resolution
                    with gr.Group():
                        gr.Markdown("### Process Resolution")
                        with gr.Row():
                            self.enable_full_res_comp = gr.Checkbox(
                                label="Enable Full Res", 
                                value=self.enable_full_res,
                                info="When checked, will generate a splatted video at the original resolution of the input video. Use this for blending over the inpaint."
                            )
                            self.batch_size_comp = gr.Number(
                                label="Batch Size", 
                                value=self.batch_size, 
                                precision=0,
                                info="The number of frames to process simultaneously when generating the full-resolution output. A higher value uses more VRAM but can be faster. Adjust based on your GPU's memory. Too high a value will bog down ffmpeg encode."
                            )
                            self.dual_output_comp = gr.Checkbox(
                                label="Dual Output Only", 
                                value=self.dual_output,
                                info="If checked, will generate dual panel for inpaint right eye only. Unchecked will generate quad panel for complete low-res stereo SBS after inpainting. Use this(Dual) to save on resources and manual blending."
                            )
                        
                        with gr.Row():
                            self.enable_low_res_comp = gr.Checkbox(
                                label="Enable Low Res", 
                                value=self.enable_low_res,
                                info="When checked, will generate a splatted video of specified resolution. Use this for inpainting."
                            )
                            self.low_res_batch_size_comp = gr.Number(
                                label="Batch Size", 
                                value=self.low_res_batch_size, 
                                precision=0,
                                info="The number of frames to process simultaneously when generating the low-resolution output. Can often be higher than the full-resolution batch size due to lower memory requirements. Too high a value will bog down ffmpeg encode."
                            )
                        
                        with gr.Row():
                            self.pre_res_width_comp = gr.Number(
                                label="Width", 
                                value=self.pre_res_width, 
                                precision=0,
                                info="The target width for the low-resolution output. The input video and depth maps will be resized to this width for the low-resolution pass."
                            )
                            self.pre_res_height_comp = gr.Number(
                                label="Height", 
                                value=self.pre_res_height, 
                                precision=0,
                                info="The target height for the low-resolution output. The input video and depth maps will be resized to this height for the low-resolution pass."
                            )
                    
                    # Splatting & Output Settings
                    with gr.Group():
                        gr.Markdown("### Splatting & Output Settings")
                        with gr.Row():
                            self.process_length_comp = gr.Number(
                                label="Process Length", 
                                value=self.process_length, 
                                precision=0,
                                info="Set how many frames to process before moving on to the next video, used for testing. Set -1 to process all frames."
                            )
                            self.auto_convergence_comp = gr.Dropdown(
                                choices=["Off", "Average", "Peak", "Hybrid"], 
                                value=self.auto_convergence_mode, 
                                label="Auto-Convergence",
                                info="EXPERIMENTAL: Simple auto convergence derived from 75% center region and averaged or peak throughout the clip."
                            )
                        
                        with gr.Row():
                            self.output_crf_full_comp = gr.Number(
                                label="Output CRF Full", 
                                value=self.output_crf_full, 
                                precision=0,
                                info="Constant Rate Factor (CRF) for H.264/H.265 video encoding. Lower values mean higher quality and larger file sizes. A good starting point is 18 for high quality, 23 for good quality. Range is typically 0-51."
                            )
                            self.output_crf_low_comp = gr.Number(
                                label="Output CRF Low", 
                                value=self.output_crf_low, 
                                precision=0,
                                info="Constant Rate Factor (CRF) for H.264/H.265 video encoding. Lower values mean higher quality and larger file sizes. A good starting point is 18 for high quality, 23 for good quality. Range is typically 0-51."
                            )
                        
                        self.enable_global_norm_comp = gr.Checkbox(
                            label="Enable Global Normalization", 
                            value=self.enable_global_norm,
                            info="Enable if your depthmap needs it and you just want to process the clips without checking."
                        )
                        self.move_to_finished_comp = gr.Checkbox(
                            label="Resume", 
                            value=self.move_to_finished,
                            info="When enabled, will move source input files to finished folder after processing."
                        )
                
                # Right Column
                with gr.Column():
                    # Depth Map Pre-processing
                    with gr.Group():
                        gr.Markdown("### Depth Map Pre-processing")
                        with gr.Row():
                            self.depth_dilate_size_x_comp = gr.Slider(
                                -10, 30, 
                                value=self.depth_dilate_size_x, 
                                step=0.5, 
                                label="Dilate X",
                                info="Horizontal Dilate for the Depthhmask. Bilinear blending for decimals. 0 to disable. Below 0 applies erosion."
                            )
                            self.depth_dilate_size_y_comp = gr.Slider(
                                -10, 30, 
                                value=self.depth_dilate_size_y, 
                                step=0.5, 
                                label="Y",
                                info="Vertical Dilate for the Depthhmask. Bilinear blending for decimals. 0 to disable. Below 0 applies erosion."
                            )
                        
                        with gr.Row():
                            self.depth_blur_size_x_comp = gr.Slider(
                                0, 35, 
                                value=self.depth_blur_size_x, 
                                step=1, 
                                label="Blur X",
                                info="Horizontal Blur for the Depth Mask."
                            )
                            self.depth_blur_size_y_comp = gr.Slider(
                                0, 35, 
                                value=self.depth_blur_size_y, 
                                step=1, 
                                label="Y",
                                info="Vertical Blur for the Depth Mask."
                            )
                        
                        with gr.Row():
                            self.depth_dilate_left_comp = gr.Slider(
                                0, 20, 
                                value=self.depth_dilate_left, 
                                step=0.5, 
                                label="Dilate Left",
                                info="Expands depth values ONLY to the left. Useful for smoothing blocky left edges without effecting right edges or forward-warp holes. Fractional values blend smoothly like other dilates."
                            )
                            self.depth_blur_left_comp = gr.Slider(
                                0, 20, 
                                value=self.depth_blur_left, 
                                step=1, 
                                label="Blur Left",
                                info="Blurs only around strong left-facing depth edges (ignores small gradients), to help smooth jagged left boundaries without blurring the whole depth map. Uses the H/V mix setting."
                            )
                        
                        self.depth_blur_left_mix_comp = gr.Slider(
                            0.0, 1.0, 
                            value=self.depth_blur_left_mix, 
                            step=0.1, 
                            label="Blur Left Mix",
                            info="Controls the balance between horizontal and vertical Blur Left. 0.0 = all horizontal, 0.5 = 50/50, 1.0 = all vertical. Recommend .5"
                        )
                    
                    # Stereo Projection
                    with gr.Group():
                        gr.Markdown("### Stereo Projection")
                        with gr.Row():
                            self.depth_gamma_comp = gr.Slider(
                                0.1, 3.0, 
                                value=self.depth_gamma, 
                                step=0.05, 
                                label="Gamma",
                                info="Applies a non-linear adjustment to the depth map before normalization. above 1.0 moves the midground towards the camera. Below 1.0 move further away. Set to 1.0 to disable."
                            )
                            self.max_disp_comp = gr.Slider(
                                0.0, 100.0, 
                                value=self.max_disp, 
                                step=1.0, 
                                label="Disparity",
                                info="Maximum disparity value as a percentage of the video's width x 20. Higher values result in more extreme parallax effects (depth shift). Example: '30.0' means a maximum of 1.5% shift horizontally at the forground. Shortcut: 4(-) and 6(+)"
                            )
                        
                        self.zero_disparity_anchor_comp = gr.Slider(
                            0.0, 2.0, 
                            value=self.zero_disparity_anchor, 
                            step=0.01, 
                            label="Convergence",
                            info="Set to 1.0 will place depth inside the screen. 0.0 will give 100% pop out. StereoCrafter Default = 0.5, Recommend 0.8. Shortcut: 1(-) and 3(+)"
                        )
                        
                        with gr.Row():
                            self.border_mode_comp = gr.Dropdown(
                                choices=["Manual", "Auto Basic", "Auto Adv.", "Off"], 
                                value=self.border_mode, 
                                label="Border",
                                info="Select border calculation strategy. Manual: uses sliders. Auto Basic: calculates based on Disparity/Convergence. Auto Adv.: uses depth map sampling for independent borders. Shortcut: 2(Cycle)."
                            )
                            self.color_tags_mode_comp = gr.Dropdown(
                                choices=["Off", "Auto", "BT.709", "BT.2020"], 
                                value=self.color_tags_mode, 
                                label="Color Tags",
                                info="Writes color metadata tags into output file headers (primaries/transfer/matrix/range). Auto uses detected tags when available; otherwise falls back to BT.709. This changes metadata only. Increases compatability."
                            )
                        
                        with gr.Row():
                            self.border_width_comp = gr.Slider(
                                0.0, 5.0, 
                                value=self.border_width, 
                                step=0.01, 
                                label="Border Width",
                                info="Total width of black borders at the screen edges as a percentage of width. Used to mask parallax occlusion for pop-out objects."
                            )
                            self.border_bias_comp = gr.Slider(
                                -1.0, 1.0, 
                                value=self.border_bias, 
                                step=0.01, 
                                label="Bias",
                                info="Shift the border distribution between left and right sides. -1.0 puts all border on the left, 1.0 all on the right, 0.0 is balanced."
                            )
            
            # Progress Section
            with gr.Group():
                gr.Markdown("### Progress")
                self.progress_bar = gr.Slider(0, 100, value=0, label="Progress", interactive=False)
                self.status_label = gr.Textbox(label="Status", value="Ready", interactive=False)
            
            # Controls Section (Bottom)
            with gr.Group():
                gr.Markdown("### Controls")
                with gr.Row():
                    self.start_single_button = gr.Button("SINGLE", variant="primary")
                    self.start_button = gr.Button("START", variant="primary")
                    self.process_from_comp = gr.Number(label="From", value=0, precision=0, scale=1)
                    self.process_to_comp = gr.Number(label="To", value=0, precision=0, scale=1)
                    self.stop_button = gr.Button("STOP", variant="stop", interactive=False)
                    self.preview_auto_converge_button = gr.Button("Preview Auto-Converge")
                    self.preview_format_comp = gr.Dropdown(
                        choices=["Side-by-Side", "Anaglyph (Red/Cyan)", "Depth Map"],
                        value="Side-by-Side",
                        label="Preview Format",
                        info="Choose how to display the preview"
                    )
                    # Note: AUTO-PASS and Update Sidecar buttons from GUI not included yet
                    self.update_sidecar_button = gr.Button("Update Sidecar")
                
                # Manual Preview Controls
                with gr.Group():
                    gr.Markdown("### Manual Preview")
                    with gr.Row():
                        self.preview_video_selector = gr.Dropdown(
                            choices=[],
                            label="Select Video",
                            interactive=True,
                            scale=4
                        )
                        self.refresh_video_list_button = gr.Button("🔄", size="sm", scale=0, min_width=40)
                        self.detect_frames_button = gr.Button("🔍 Detect", size="sm", scale=1)
                    with gr.Row():
                        self.preview_frame_number = gr.Slider(
                            minimum=0,
                            maximum=1000,
                            value=50,
                            step=1,
                            label="Frame (0-1000)",
                            info="Drag to scrub through video",
                            scale=6
                        )
                        self.preview_convergence_slider = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.5,
                            step=0.01,
                            label="Convergence",
                            scale=2
                        )
                        self.preview_disparity_slider = gr.Slider(
                            minimum=0.0,
                            maximum=200.0,
                            value=50.0,
                            step=1.0,
                            label="Disparity",
                            scale=2
                        )
                    with gr.Row():
                        self.manual_preview_button = gr.Button("Update Preview", variant="secondary", scale=2)
                        self.apply_preview_settings_button = gr.Button("Apply to Main", size="sm", scale=1)
                
                # Preview image output
                with gr.Row():
                    self.preview_image_output = gr.Image(
                        label="Preview: Left (Original) | Right (With Convergence)",
                        type="numpy",
                        interactive=False
                    )

            # Event handlers
            self.multi_map_comp.change(
                fn=self._on_multi_map_toggle,
                inputs=[self.multi_map_comp],
                outputs=[self.depth_map_subfolders_comp]
            )
            
            self.input_depth_maps_comp.change(
                fn=self._on_depth_map_folder_changed,
                inputs=[self.input_depth_maps_comp],
                outputs=[self.depth_map_subfolders_comp]
            )
            
            # Auto-refresh video list when input folder changes
            self.input_source_clips_comp.change(
                fn=self.refresh_video_list,
                inputs=[],
                outputs=[self.preview_video_selector, self.status_label]
            )
            
            self.depth_map_subfolders_comp.change(
                fn=self._on_map_selection_changed,
                inputs=[self.depth_map_subfolders_comp],
                outputs=[]
            )
            
            self.auto_convergence_comp.change(
                fn=self.on_auto_convergence_mode_select,
                inputs=[self.auto_convergence_comp],
                outputs=[]
            )
            
            self.start_button.click(
                fn=self.start_processing,
                inputs=[
                    self.input_source_clips_comp, self.input_depth_maps_comp, self.output_splatted_comp,
                    self.max_disp_comp, self.process_length_comp, self.enable_full_res_comp, self.batch_size_comp,
                    self.enable_low_res_comp, self.pre_res_width_comp, self.pre_res_height_comp, self.low_res_batch_size_comp,
                    self.dual_output_comp, self.zero_disparity_anchor_comp, self.enable_global_norm_comp,
                    self.output_crf_full_comp, self.depth_gamma_comp, self.depth_dilate_size_x_comp, self.depth_dilate_size_y_comp,
                    self.depth_blur_size_x_comp, self.depth_blur_size_y_comp, self.auto_convergence_comp,
                    self.move_to_finished_comp, self.process_from_comp, self.process_to_comp,
                    # New parameters
                    self.output_crf_low_comp, self.depth_dilate_left_comp, self.depth_blur_left_comp, self.depth_blur_left_mix_comp,
                    self.border_width_comp, self.border_bias_comp, self.border_mode_comp, self.color_tags_mode_comp
                ],
                outputs=[self.status_label, self.progress_bar, self.start_button, self.start_single_button, self.stop_button]
            )
            
            self.start_single_button.click(
                fn=self.start_single_processing,
                inputs=[
                    self.input_source_clips_comp, self.input_depth_maps_comp, self.output_splatted_comp,
                    self.max_disp_comp, self.process_length_comp, self.enable_full_res_comp, self.batch_size_comp,
                    self.enable_low_res_comp, self.pre_res_width_comp, self.pre_res_height_comp, self.low_res_batch_size_comp,
                    self.dual_output_comp, self.zero_disparity_anchor_comp, self.enable_global_norm_comp,
                    self.output_crf_full_comp, self.depth_gamma_comp, self.depth_dilate_size_x_comp, self.depth_dilate_size_y_comp,
                    self.depth_blur_size_x_comp, self.depth_blur_size_y_comp, self.auto_convergence_comp,
                    self.move_to_finished_comp,
                    # New parameters
                    self.output_crf_low_comp, self.depth_dilate_left_comp, self.depth_blur_left_comp, self.depth_blur_left_mix_comp,
                    self.border_width_comp, self.border_bias_comp, self.border_mode_comp, self.color_tags_mode_comp
                ],
                outputs=[self.status_label, self.progress_bar, self.start_button, self.start_single_button, self.stop_button]
            )
            
            self.stop_button.click(
                fn=self.stop_processing,
                inputs=[],
                outputs=[self.status_label, self.progress_bar]
            ).then(
                fn=lambda: (gr.Button(interactive=True), gr.Button(interactive=True), gr.Button(interactive=False)),
                inputs=[],
                outputs=[self.start_button, self.start_single_button, self.stop_button]
            )
            
            self.update_sidecar_button.click(
                fn=self.update_sidecar_file,
                inputs=[],
                outputs=[self.status_label]
            )
            
            self.preview_auto_converge_button.click(
                fn=self.run_preview_auto_converge_with_mode,
                inputs=[self.auto_convergence_comp, self.preview_format_comp],
                outputs=[self.status_label, self.zero_disparity_anchor_comp, self.preview_image_output]
            )
            
            # Manual preview handlers
            self.refresh_video_list_button.click(
                fn=self.refresh_video_list,
                inputs=[],
                outputs=[self.preview_video_selector, self.status_label]
            )
            
            self.detect_frames_button.click(
                fn=self.detect_video_frames,
                inputs=[self.preview_video_selector],
                outputs=[self.preview_frame_number, self.status_label]
            )
            
            # Auto-detect frames and update slider when video is selected
            self.preview_video_selector.change(
                fn=self.detect_video_frames,
                inputs=[self.preview_video_selector],
                outputs=[self.preview_frame_number, self.status_label]
            )
            
            # Auto-refresh preview when frame slider is released
            self.preview_frame_number.release(
                fn=self.generate_manual_preview,
                inputs=[self.preview_video_selector, self.preview_frame_number, self.preview_convergence_slider, self.preview_disparity_slider, self.preview_format_comp],
                outputs=[self.preview_image_output, self.status_label, self.preview_frame_number]
            )
            
            # Auto-refresh preview when convergence slider is released
            self.preview_convergence_slider.release(
                fn=self.generate_manual_preview,
                inputs=[self.preview_video_selector, self.preview_frame_number, self.preview_convergence_slider, self.preview_disparity_slider, self.preview_format_comp],
                outputs=[self.preview_image_output, self.status_label, self.preview_frame_number]
            )
            
            # Auto-refresh preview when disparity slider is released
            self.preview_disparity_slider.release(
                fn=self.generate_manual_preview,
                inputs=[self.preview_video_selector, self.preview_frame_number, self.preview_convergence_slider, self.preview_disparity_slider, self.preview_format_comp],
                outputs=[self.preview_image_output, self.status_label, self.preview_frame_number]
            )
            
            # Auto-refresh preview when format dropdown changes
            self.preview_format_comp.change(
                fn=self.generate_manual_preview,
                inputs=[self.preview_video_selector, self.preview_frame_number, self.preview_convergence_slider, self.preview_disparity_slider, self.preview_format_comp],
                outputs=[self.preview_image_output, self.status_label, self.preview_frame_number]
            )
            
            self.manual_preview_button.click(
                fn=self.generate_manual_preview,
                inputs=[self.preview_video_selector, self.preview_frame_number, self.preview_convergence_slider, self.preview_disparity_slider, self.preview_format_comp],
                outputs=[self.preview_image_output, self.status_label, self.preview_frame_number]
            )
            
            self.apply_preview_settings_button.click(
                fn=lambda conv, disp: (conv, disp, f"✅ Applied convergence {conv:.3f} and disparity {disp:.1f} to main settings"),
                inputs=[self.preview_convergence_slider, self.preview_disparity_slider],
                outputs=[self.zero_disparity_anchor_comp, self.max_disp_comp, self.status_label]
            )
            
            # Add a timer to check processing status every 2 seconds
            status_timer = gr.Timer(value=2.0)
            status_timer.tick(
                fn=self.check_processing_status,
                inputs=[],
                outputs=[self.status_label, self.progress_bar, self.start_button, self.start_single_button, self.stop_button]
            )

        return interface



def compute_global_depth_stats(
        depth_map_reader,
        total_frames,
        chunk_size = 100
    ):
    """
    Computes the global min and max depth values from a depth video by reading it in chunks.
    Assumes raw pixel values that need to be scaled (e.g., from 0-255 or 0-1023 range).
    """
    logger.info(f"==> Starting global depth stats pre-pass for {total_frames} frames...")
    global_min, global_max = np.inf, -np.inf

    for i in range(0, total_frames, chunk_size):
        current_indices = list(range(i, min(i + chunk_size, total_frames)))
        if not current_indices:
            break
        
        chunk_numpy_raw = depth_map_reader.get_batch(current_indices).asnumpy()
        
        # Handle RGB vs Grayscale depth maps
        if chunk_numpy_raw.ndim == 4:
            if chunk_numpy_raw.shape[-1] == 3: # RGB
                chunk_numpy = chunk_numpy_raw.mean(axis=-1)
            else: # Grayscale with channel dim
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

    logger.info(f"==> Global depth stats computed: min_raw={global_min:.3f}, max_raw={global_max:.3f}")
    return float(global_min), float(global_max)


def read_video_frames(
        video_path: str,
        process_length: int,
        set_pre_res: bool,
        pre_res_width: int,
        pre_res_height: int,
        dataset: str = "open"
    ):
    """
    Initializes a VideoReader for chunked reading.
    Returns: (video_reader, fps, original_height, original_width, actual_processed_height, actual_processed_width, video_stream_info, total_frames_to_process)
    """
    if dataset == "open":
        logger.debug(f"==> Initializing VideoReader for: {video_path}")
        vid_info_only = VideoReader(video_path, ctx=cpu(0)) # Use separate reader for info
        original_height, original_width = vid_info_only.get_batch([0]).shape[1:3]
        total_frames_original = len(vid_info_only)
        logger.debug(f"==> Original video shape: {total_frames_original} frames, {original_height}x{original_width} per frame")

        height_for_reader = original_height
        width_for_reader = original_width

        if set_pre_res and pre_res_width > 0 and pre_res_height > 0:
            height_for_reader = pre_res_height
            width_for_reader = pre_res_width
            logger.debug(f"==> Pre-processing resolution set to: {width_for_reader}x{height_for_reader}")
        else:
            logger.debug(f"==> Using original video resolution for reading: {width_for_reader}x{height_for_reader}")

    else:
        raise NotImplementedError(f"Dataset '{dataset}' not supported.")

    # decord automatically resizes if width/height are passed to VideoReader
    video_reader = VideoReader(video_path, ctx=cpu(0), width=width_for_reader, height=height_for_reader)
    
    # Verify the actual shape after Decord processing, using the first frame
    first_frame_shape = video_reader.get_batch([0]).shape
    actual_processed_height, actual_processed_width = first_frame_shape[1:3]
    
    fps = video_reader.get_avg_fps() # Use actual FPS from the reader

    total_frames_available = len(video_reader)
    total_frames_to_process = total_frames_available # Use available frames directly
    if process_length != -1 and process_length < total_frames_available:
        total_frames_to_process = process_length
    
    logger.debug(f"==> VideoReader initialized. Final processing dimensions: {actual_processed_width}x{actual_processed_height}. Total frames for processing: {total_frames_to_process}")

    video_stream_info = get_video_stream_info(video_path) # Get stream info for FFmpeg later

    return video_reader, fps, original_height, original_width, actual_processed_height, actual_processed_width, video_stream_info, total_frames_to_process


def load_pre_rendered_depth(
        depth_map_path: str,
        process_length: int,
        target_height: int,
        target_width: int,
        match_resolution_to_target: bool):
    """
    Initializes a VideoReader for chunked depth map reading.
    No normalization or autogain is applied here.
    Returns: (depth_reader, total_depth_frames_to_process, actual_depth_height, actual_depth_width)
    """
    logger.debug(f"==> Initializing VideoReader for depth maps from: {depth_map_path}")

    # NEW: Get stream info for the depth map video
    depth_stream_info = get_video_stream_info(depth_map_path) 

    if depth_map_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        depth_reader = VideoReader(depth_map_path, ctx=cpu(0), width=target_width, height=target_height)
        
        first_depth_frame_shape = depth_reader.get_batch([0]).shape
        actual_depth_height, actual_depth_width = first_depth_frame_shape[1:3]
        
        total_depth_frames_available = len(depth_reader)
        total_depth_frames_to_process = total_depth_frames_available
        if process_length != -1 and process_length < total_depth_frames_available:
            total_depth_frames_to_process = process_length

        logger.debug(f"==> DepthReader initialized. Final depth dimensions: {actual_depth_width}x{actual_depth_height}. Total frames for processing: {total_depth_frames_to_process}")
        
        return depth_reader, total_depth_frames_to_process, actual_depth_height, actual_depth_width, depth_stream_info
    
    elif depth_map_path.lower().endswith('.npz'):
        logger.error("NPZ support is temporarily disabled with disk chunking refactor. Please convert NPZ to MP4 depth video.")
        raise NotImplementedError("NPZ depth map loading is not yet supported with disk chunking.")
    else:
        raise ValueError(f"Unsupported depth map format: {os.path.basename(depth_map_path)}. Only MP4 are supported with disk chunking.")


if __name__ == "__main__":
    CUDA_AVAILABLE = check_cuda_availability() # Sets the global flag

    app = SplatterWebUI()
    interface = app.create_interface()
    interface.launch(share=True)
