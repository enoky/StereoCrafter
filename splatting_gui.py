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
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from ttkthemes import ThemedTk
import json
import threading
import queue
import subprocess
import time
import logging
from typing import Optional, Tuple, Optional
from PIL import Image
import math # <--- ADD THIS
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    # Fallback/stub for systems without moviepy
    class VideoFileClip:
        def __init__(self, *args, **kwargs):
            logging.warning("moviepy.editor not found. Frame counting disabled.")
        def close(self): pass
        @property
        def fps(self): return None
        @property
        def duration(self): return None

# Import custom modules
CUDA_AVAILABLE = False # start state, will check automaticly later
        
# --- MODIFIED IMPORT ---
from dependency.stereocrafter_util import (
    Tooltip, logger, get_video_stream_info, draw_progress_bar,
    check_cuda_availability, release_cuda_memory, CUDA_AVAILABLE, set_util_logger_level,
    start_ffmpeg_pipe_process, custom_blur, custom_dilate,
    create_single_slider_with_label_updater, create_dual_slider_layout,
    SidecarConfigManager
)
try:
    from Forward_Warp import forward_warp
    logger.info("CUDA Forward Warp is available.")
except:
    from dependency.forward_warp_pytorch import forward_warp
    logger.info("Forward Warp Pytorch is active.")
from dependency.video_previewer import VideoPreviewer

GUI_VERSION = "25.10.21.3"

class FusionSidecarGenerator:
    """Handles parsing Fusion Export files, matching them to depth maps,
    and generating/saving FSSIDECAR files using carry-forward logic."""
    
    FUSION_PARAMETER_CONFIG = {
        # Key: {Label, Type, Default, FusionKey(fsexport), SidecarKey(fssidecar), Decimals}
        "convergence": {
            "label": "Convergence Plane", "type": float, "default": 0.5, 
            "fusion_key": "Convergence", "sidecar_key": "convergence_plane", "decimals": 3
        },
        "max_disparity": {
            "label": "Max Disparity", "type": float, "default": 35.0, 
            "fusion_key": "MaxDisparity", "sidecar_key": "max_disparity", "decimals": 1
        },
        "gamma": {
            "label": "Gamma Correction", "type": float, "default": 1.0,
            "fusion_key": "FrontGamma", "sidecar_key": "gamma", "decimals": 2
        },
        # These keys exist in the sidecar manager but are usually set in the source tool
        # We include them here for completeness if Fusion ever exported them
        "frame_overlap": {
            "label": "Frame Overlap", "type": float, "default": 3,
            "fusion_key": "Overlap", "sidecar_key": "frame_overlap", "decimals": 0
        },
        "input_bias": {
            "label": "Input Bias", "type": float, "default": 0.0, 
            "fusion_key": "Bias", "sidecar_key": "input_bias", "decimals": 2
        }
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
                if duration is None: return 0 
            
            frames = math.ceil(duration * fps)
            clip.close()
            return frames
        except Exception as e:
            self.logger.warning(f"Error getting frame count for {os.path.basename(file_path)}: {e}")
            return 0

    def _load_and_validate_fsexport(self, file_path):
        """Loads, parses, and validates marker data from a Fusion Export file."""
        try:
            with open(file_path, 'r') as f:
                export_data = json.load(f)
        except json.JSONDecodeError as e:
            messagebox.showerror("File Error", f"Failed to parse JSON in {os.path.basename(file_path)}: {e}")
            return None
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to read {os.path.basename(file_path)}: {e}")
            return None

        markers = export_data.get("markers", [])
        if not markers:
            messagebox.showwarning("Data Warning", "No 'markers' found in the export file.")
            return None
        
        # Sort markers by frame number (critical for carry-forward logic)
        markers.sort(key=lambda m: m['frame'])
        self.logger.info(f"Loaded {len(markers)} markers from {os.path.basename(file_path)}.")
        return markers

    def _scan_target_videos(self, folder):
        """Scans the target folder for video files and computes their frame counts."""
        video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
        found_files_paths = []
        for ext in video_extensions:
            found_files_paths.extend(glob.glob(os.path.join(folder, ext)))
        sorted_files_paths = sorted(found_files_paths)
        
        if not sorted_files_paths:
            messagebox.showwarning("No Files", f"No video depth map files found in: {folder}")
            return None

        target_video_data = []
        cumulative_frames = 0
        
        for full_path in sorted_files_paths:
            total_frames = self._get_video_frame_count(full_path)
            
            if total_frames == 0:
                self.logger.warning(f"Skipping {os.path.basename(full_path)} due to zero frame count.")
                continue

            target_video_data.append({
                "full_path": full_path,
                "basename": os.path.basename(full_path),
                "total_frames": total_frames,
                "timeline_start_frame": cumulative_frames,
                "timeline_end_frame": cumulative_frames + total_frames - 1,
            })
            cumulative_frames += total_frames
            
        self.logger.info(f"Scanned {len(target_video_data)} video files. Total timeline frames: {cumulative_frames}.")
        return target_video_data

    def generate_sidecars(self):
        """Main entry point for the Fusion Export to Sidecar generation workflow."""
        
        # 1. Select Fusion Export File
        export_file_path = filedialog.askopenfilename(
            defaultextension=".fsexport",
            filetypes=[("Fusion Export Files", "*.fsexport.txt;*.fsexport"), ("All Files", "*.*")],
            title="Select Fusion Export (.fsexport) File"
        )
        if not export_file_path:
            self.master_gui.status_label.config(text="Fusion export selection cancelled.")
            return

        markers = self._load_and_validate_fsexport(export_file_path)
        if markers is None:
            self.master_gui.status_label.config(text="Fusion export loading failed.")
            return

        # 2. Select Target Depth Map Folder
        target_folder = filedialog.askdirectory(title="Select Target Depth Map Folder")
        if not target_folder:
            self.master_gui.status_label.config(text="Depth map folder selection cancelled.")
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
                if marker['frame'] <= file_start_frame:
                    relevant_marker = marker
                else:
                    break
            
            current_param_vals = last_param_vals.copy()

            if relevant_marker and relevant_marker.get('values'):
                marker_values = relevant_marker['values']
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
                            self.logger.warning(f"Marker value for '{fusion_key}' is invalid ({val}). Using previous/default value.")
                            
                if updated_from_marker:
                    applied_count += 1
            
            # 4. Save Sidecar JSON
            sidecar_data = {}
            for key, config in self.FUSION_PARAMETER_CONFIG.items():
                value = current_param_vals[key]
                # Round to configured decimals for clean sidecar output
                sidecar_data[config["sidecar_key"]] = round(value, config["decimals"])
                
            base_name_without_ext = os.path.splitext(file_data["full_path"])[0]
            json_filename = base_name_without_ext + ".fssidecar" # Target sidecar extension
            
            if not self.sidecar_manager.save_sidecar_data(json_filename, sidecar_data):
                self.logger.error(f"Failed to save sidecar for {file_data['basename']}.")

            # Update last values for carry-forward to the next file
            last_param_vals = current_param_vals.copy()

        # 5. Final Status
        if applied_count == 0:
            self.master_gui.status_label.config(text="Finished: No parameters were applied from the export file.")
        else:
            self.master_gui.status_label.config(text=f"Finished: Applied markers to {applied_count} files, generated {len(target_videos)} FSSIDECARs.")
        messagebox.showinfo("Sidecar Generation Complete", f"Successfully processed {os.path.basename(export_file_path)} and generated {len(target_videos)} FSSIDECAR files.")

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
    # --- GLOBAL CONFIGURATION DICTIONARY ---
    APP_CONFIG_DEFAULTS = {
        # File Extensions
        "SIDECAR_EXT": ".fssidecar",
        "OUTPUT_SIDECAR_EXT": ".spsidecar",
        
        # GUI/Processing Defaults (Used for reset/fallback)
        "MAX_DISP": "30.0",
        "CONV_POINT": "0.5",
        "PROC_LENGTH": "-1",
        "BATCH_SIZE_FULL": "10",
        "BATCH_SIZE_LOW": "50",
        "CRF_OUTPUT": "23",
        
        # Depth Processing Defaults
        "DEPTH_GAMMA": "1.0",
        "DEPTH_DILATE_SIZE_X": "3",
        "DEPTH_DILATE_SIZE_Y": "3",
        "DEPTH_BLUR_SIZE_X": "5",
        "DEPTH_BLUR_SIZE_Y": "5"
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
        "frame_overlap": "FRAME_OVERLAP",
        "input_bias": "INPUT_BIAS"
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
        self.slider_label_updaters = [] 
        self.set_convergence_value_programmatically = None

        self._load_config()
        self._load_help_texts()
        
        self._is_startup = True # NEW: for theme/geometry handling
        self.debug_mode_var = tk.BooleanVar(value=self.app_config.get("debug_mode_enabled", False))
        self._debug_logging_enabled = False # start in INFO mode
        # NEW: Window size and position variables
        self.window_x = self.app_config.get("window_x", None)
        self.window_y = self.app_config.get("window_y", None)
        self.window_width = self.app_config.get("window_width", 620)
        self.window_height = self.app_config.get("window_height", 750)

        # --- Variables with defaults ---
        defaults = self.APP_CONFIG_DEFAULTS # Convenience variable

        self.dark_mode_var = tk.BooleanVar(value=self.app_config.get("dark_mode_enabled", False))
        self.input_source_clips_var = tk.StringVar(value=self.app_config.get("input_source_clips", "./input_source_clips"))
        self.input_depth_maps_var = tk.StringVar(value=self.app_config.get("input_depth_maps", "./input_depth_maps"))
        self.output_splatted_var = tk.StringVar(value=self.app_config.get("output_splatted", "./output_splatted"))

        self.max_disp_var = tk.StringVar(value=self.app_config.get("max_disp", defaults["MAX_DISP"]))
        self.process_length_var = tk.StringVar(value=self.app_config.get("process_length", defaults["PROC_LENGTH"]))
        self.batch_size_var = tk.StringVar(value=self.app_config.get("batch_size", defaults["BATCH_SIZE_FULL"]))
        
        self.dual_output_var = tk.BooleanVar(value=self.app_config.get("dual_output", False))
        self.enable_autogain_var = tk.BooleanVar(value=self.app_config.get("enable_autogain", False)) 
        self.enable_full_res_var = tk.BooleanVar(value=self.app_config.get("enable_full_resolution", True))
        self.enable_low_res_var = tk.BooleanVar(value=self.app_config.get("enable_low_resolution", True))
        self.pre_res_width_var = tk.StringVar(value=self.app_config.get("pre_res_width", "1024"))
        self.pre_res_height_var = tk.StringVar(value=self.app_config.get("pre_res_height", "512"))
        self.low_res_batch_size_var = tk.StringVar(value=self.app_config.get("low_res_batch_size", defaults["BATCH_SIZE_LOW"]))
        self.zero_disparity_anchor_var = tk.StringVar(value=self.app_config.get("convergence_point", defaults["CONV_POINT"]))
        self.output_crf_var = tk.StringVar(value=self.app_config.get("output_crf", defaults["CRF_OUTPUT"]))

        self.auto_convergence_mode_var = tk.StringVar(value=self.app_config.get("auto_convergence_mode", "Off"))

        # --- Depth Pre-processing Variables ---
        self.depth_gamma_var = tk.StringVar(value=self.app_config.get("depth_gamma", defaults["DEPTH_GAMMA"]))
        self.depth_dilate_size_x_var = tk.StringVar(value=self.app_config.get("depth_dilate_size_x", defaults["DEPTH_DILATE_SIZE_X"]))
        self.depth_dilate_size_y_var = tk.StringVar(value=self.app_config.get("depth_dilate_size_y", defaults["DEPTH_DILATE_SIZE_Y"]))
        self.depth_blur_size_x_var = tk.StringVar(value=self.app_config.get("depth_blur_size_x", defaults["DEPTH_BLUR_SIZE_X"]))
        self.depth_blur_size_y_var = tk.StringVar(value=self.app_config.get("depth_blur_size_y", defaults["DEPTH_BLUR_SIZE_Y"]))
        # --- NEW: Sidecar Control Toggle Variables ---
        self.enable_sidecar_gamma_var = tk.BooleanVar(value=self.app_config.get("enable_sidecar_gamma", True))
        self.enable_sidecar_blur_dilate_var = tk.BooleanVar(value=self.app_config.get("enable_sidecar_blur_dilate", True))
        self.override_sidecar_var = tk.BooleanVar(value=self.app_config.get("override_sidecar_preview", False))

        # --- NEW: Previewer Variables ---
        self.preview_source_var = tk.StringVar(value="Splat Result")
        self.preview_size_var = tk.StringVar(value=self.app_config.get("preview_size", "75%"))

        # --- Variables for "Current Processing Information" display ---
        self.processing_filename_var = tk.StringVar(value="N/A")
        self.processing_resolution_var = tk.StringVar(value="N/A")
        self.processing_frames_var = tk.StringVar(value="N/A")
        self.processing_disparity_var = tk.StringVar(value="N/A")
        self.processing_convergence_var = tk.StringVar(value="N/A")
        self.processing_task_name_var = tk.StringVar(value="N/A")
        self.processing_gamma_var = tk.StringVar(value="N/A")

        self.slider_label_updaters = [] 

        # --- Processing control variables ---
        self.stop_event = threading.Event()
        self.progress_queue = queue.Queue()
        self.processing_thread = None

        self._create_widgets()
        self.style = ttk.Style()
        
        self.update_idletasks() # Ensure widgets are rendered for correct reqheight
        self._apply_theme(is_startup=True) # Pass is_startup=True here
        self._set_saved_geometry() # NEW: Call to set initial geometry
        self._is_startup = False # Set to false after initial startup geometry is handled
        self._configure_logging() # Ensure this call is still present

        self.after(10, self.toggle_processing_settings_fields) # Set initial state
        self.after(10, self._toggle_sidecar_update_button_state)
        self.after(100, self.check_queue) # Start checking progress queue

        # Bind closing protocol
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

        # --- NEW: Add slider release binding for preview updates ---
        # We will add this to the sliders in _create_widgets
        self.slider_widgets = []

    def _adjust_window_height_for_content(self):
        """Adjusts the window height to fit the current content, preserving user-set width."""
        if self._is_startup: # Don't adjust during initial setup
            return

        current_actual_width = self.winfo_width()
        if current_actual_width <= 1: # Fallback for very first call
            current_actual_width = self.window_width

        # --- NEW: More accurate height calculation ---
        # --- FIX: Calculate base_height by summing widgets *other* than the previewer ---
        # This is more stable than subtracting a potentially out-of-sync canvas height.
        base_height = 0
        for widget in self.winfo_children():
            if widget is not self.previewer:
                # --- FIX: Correctly handle tuple and int for pady ---
                try:
                    pady_value = widget.pack_info().get('pady', 0)
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
        if hasattr(self.previewer, 'preview_image_tk') and self.previewer.preview_image_tk:
            preview_image_height = self.previewer.preview_image_tk.height()

        # Add a small buffer for padding/borders
        padding = 10 

        # The new total height is the base UI height + the actual image height + padding
        new_height = base_height + preview_image_height + padding
        # --- END NEW ---

        self.geometry(f"{current_actual_width}x{new_height}")
        logger.debug(f"Content resize applied geometry: {current_actual_width}x{new_height}")
        self.window_width = current_actual_width # Update stored width

    def _apply_theme(self, is_startup: bool = False):
        """Applies the selected theme (dark or light) to the GUI."""
        # 1. Define color palettes
        dark_colors = {
            "bg": "#2b2b2b", "fg": "white", "entry_bg": "#3c3c3c",
            "menu_bg": "#3c3c3c", "menu_fg": "white", "active_bg": "#555555", "active_fg": "white",
            "theme": "black"
        }
        light_colors = {
            "bg": "#d9d9d9", "fg": "black", "entry_bg": "#ffffff",
            "menu_bg": "#f0f0f0", "menu_fg": "black", "active_bg": "#dddddd", "active_fg": "black",
            "theme": "default"
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
        self.style.configure("TLabelframe", background=colors["bg"], foreground=colors["fg"])
        self.style.configure("TLabelframe.Label", background=colors["bg"], foreground=colors["fg"])
        self.style.configure("TLabel", background=colors["bg"], foreground=colors["fg"])
        self.style.configure("TCheckbutton", background=colors["bg"], foreground=colors["fg"])
        self.style.map('TCheckbutton', foreground=[('active', colors["fg"])], background=[('active', colors["bg"])])

        # 4. Configure Entry and Combobox widgets using style.map for robust background override
        self.style.map('TEntry', fieldbackground=[('', colors["entry_bg"])], foreground=[('', colors["fg"])])
        self.style.configure("TEntry", insertcolor=colors["fg"])
        self.style.map('TCombobox',
            fieldbackground=[('readonly', colors["entry_bg"])],
            foreground=[('readonly', colors["fg"])],
            selectbackground=[('readonly', colors["entry_bg"])],
            selectforeground=[('readonly', colors["fg"])]
        )

        # Manually set the background for the previewer's canvas widget
        if hasattr(self, 'previewer') and hasattr(self.previewer, 'preview_canvas'):
            self.previewer.preview_canvas.config(bg=colors["bg"], highlightthickness=0)

        # 5. Manually configure non-ttk widgets (Menu, tk.Label)
        if hasattr(self, 'menubar'):
            for menu in [self.menubar, self.file_menu, self.help_menu]:
                menu.config(bg=colors["menu_bg"], fg=colors["menu_fg"], activebackground=colors["active_bg"], activeforeground=colors["active_fg"])
        if hasattr(self, 'info_frame'):
            for label in self.info_labels:
                label.config(bg=colors["bg"], fg=colors["fg"])

        # 6. Handle window geometry adjustment (only after startup)
        self.update_idletasks() # Ensure all theme changes are rendered for accurate reqheight

        # --- Apply geometry only if not during startup (NEW conditional block) ---
        # if not is_startup:
        #     self._adjust_window_height_for_content()

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
        self.after(0, lambda: self._complete_auto_converge_update(
            new_anchor_avg, 
            new_anchor_peak, 
            fallback_value, 
            mode # Still pass the current mode to know which value to select immediately
        ))

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
            initial_dir = os.path.dirname(current_path) if os.path.isfile(current_path) else current_path
        else:
            initial_dir = None

        file_path = filedialog.askopenfilename(initialdir=initial_dir, filetypes=filetypes_list)
        if file_path:
            var.set(file_path)

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
                    self._set_input_state('normal')
                    logger.info(f"==> All process completed.")
                    break
                
                elif message[0] == "total":
                    total_tasks = message[1]
                    self.progress_bar.config(maximum=total_tasks)
                    self.progress_var.set(0)
                    self.status_label.config(text=f"Processing 0 of {total_tasks} tasks")
                elif message[0] == "processed":
                    processed_tasks = message[1]
                    total_tasks = self.progress_bar["maximum"]
                    self.progress_var.set(processed_tasks)
                    self.status_label.config(text=f"Processed tasks: {processed_tasks}/{total_tasks} (overall)")
                elif message[0] == "status":
                    self.status_label.config(text=f"Overall: {self.progress_var.get()}/{self.progress_bar['maximum']} - {message[1].split(':', 1)[-1].strip()}")
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
                    if "gamma" in info_data: # <--- ADD THIS CHECK
                        self.processing_gamma_var.set(info_data["gamma"])
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

    def _complete_auto_converge_update(self, new_anchor_avg: float, new_anchor_peak: float, fallback_value: float, mode: str):
        """
        Safely updates the GUI and preview after Auto-Convergence worker is done.
        
        Now receives both calculated values.
        """
        # Re-enable inputs
        self._is_auto_conv_running = False
        self.btn_auto_converge_preview.config(state="normal")
        self.start_button.config(state="normal")
        self.start_single_button.config(state="normal")
        self.auto_convergence_combo.config(state="readonly") # Re-enable combo

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
                 depth_map_path = self.previewer.video_list[current_index].get('depth_map')
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

            self.status_label.config(text=f"Auto-Converge: Avg Cached at {new_anchor_avg:.2f}, Peak Cached at {new_anchor_peak:.2f}. Applied: {mode} ({anchor_to_apply:.2f})")
            
            # 4. Immediately trigger a preview update to show the change
            self.on_slider_release(None) 
            
        else:
            # Calculation failed (both returned fallback)
            self.status_label.config(text=f"Auto-Converge: Failed to find a valid anchor. Value remains {fallback_value:.2f}")
            messagebox.showwarning("Auto-Converge Preview", f"Failed to find a valid anchor point in any mode. No changes were made.")
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
            if logging.root.level == logging.DEBUG: # Check if this GUI set it
                logging.root.setLevel(logging.INFO) # Reset to a less verbose default

        # Make sure 'set_util_logger_level' is imported and available.
        # It's already in dependency/stereocrafter_util, ensure it's imported at the top.
        # Add 'import logging' at the top of splatting_gui.py if not already present.
        set_util_logger_level(level) # Call the function from stereocrafter_util.py
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

        self.file_menu  = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu )
        
        # Add new commands to the File menu
        self.file_menu.add_command(label="Load Settings...", command=self.load_settings)
        self.file_menu.add_command(label="Save Settings...", command=self.save_settings)
        self.file_menu.add_separator() # Separator for organization

        self.file_menu.add_command(label="Load Fusion Export (.fsexport)...", command=self.run_fusion_sidecar_generator)
        self.file_menu.add_separator()

        self.file_menu .add_checkbutton(label="Dark Mode", variable=self.dark_mode_var, command=self._apply_theme)
        self.file_menu .add_separator()
        self.file_menu .add_command(label="Reset to Default", command=self.reset_to_defaults)
        self.file_menu .add_command(label="Restore Finished", command=self.restore_finished_files)

        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.debug_logging_var = tk.BooleanVar(value=self._debug_logging_enabled)
        self.help_menu.add_checkbutton(label="Debug Logging", variable=self.debug_logging_var, command=self._toggle_debug_logging)
        self.help_menu.add_command(label="User Guide", command=self.show_user_guide)
        self.help_menu.add_separator()

        # Add "About" submenu (after "Debug Logging")
        self.help_menu.add_command(label="About Stereocrafter Splatting", command=self.show_about)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)

        # --- Folder selection frame ---
        self.folder_frame = ttk.LabelFrame(self, text="Input/Output Folders")
        self.folder_frame.pack(pady=10, padx=10, fill="x")
        self.folder_frame.grid_columnconfigure(1, weight=1)

        # Settings Container (NEW)
        self.settings_container_frame = ttk.Frame(self) # <-- ADD self. to settings_container_frame
        self.settings_container_frame.pack(pady=10, padx=10, fill="x")

        # Input Source Clips Row
        self.lbl_source_clips = ttk.Label(self.folder_frame, text="Input Source Clips:")
        self.lbl_source_clips.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        self.entry_source_clips = ttk.Entry(self.folder_frame, textvariable=self.input_source_clips_var)
        self.entry_source_clips.grid(row=current_row, column=1, padx=5, pady=2, sticky="ew")
        self.btn_browse_source_clips_folder = ttk.Button(self.folder_frame, text="Browse Folder", command=lambda: self._browse_folder(self.input_source_clips_var))
        self.btn_browse_source_clips_folder.grid(row=current_row, column=2, padx=2, pady=2)
        self.btn_select_source_clips_file = ttk.Button(self.folder_frame, text="Select File", command=lambda: self._browse_file(self.input_source_clips_var, [("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]))
        self.btn_select_source_clips_file.grid(row=current_row, column=3, padx=2, pady=2)
        self._create_hover_tooltip(self.lbl_source_clips, "input_source_clips")
        self._create_hover_tooltip(self.entry_source_clips, "input_source_clips")
        self._create_hover_tooltip(self.btn_browse_source_clips_folder, "input_source_clips_folder")
        self._create_hover_tooltip(self.btn_select_source_clips_file, "input_source_clips_file")
        current_row += 1

        # Input Depth Maps Row
        self.lbl_input_depth_maps = ttk.Label(self.folder_frame, text="Input Depth Maps:")
        self.lbl_input_depth_maps.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        self.entry_input_depth_maps = ttk.Entry(self.folder_frame, textvariable=self.input_depth_maps_var)
        self.entry_input_depth_maps.grid(row=current_row, column=1, padx=5, pady=2, sticky="ew")
        self.btn_browse_input_depth_maps_folder = ttk.Button(self.folder_frame, text="Browse Folder", command=lambda: self._browse_folder(self.input_depth_maps_var))
        self.btn_browse_input_depth_maps_folder.grid(row=current_row, column=2, padx=2, pady=2)
        self.btn_select_input_depth_maps_file = ttk.Button(self.folder_frame, text="Select File", command=lambda: self._browse_file(self.input_depth_maps_var, [("Depth Files", "*.mp4 *.npz"), ("All files", "*.*")]))
        self.btn_select_input_depth_maps_file.grid(row=current_row, column=3, padx=2, pady=2)
        self._create_hover_tooltip(self.lbl_input_depth_maps, "input_depth_maps")
        self._create_hover_tooltip(self.entry_input_depth_maps, "input_depth_maps")
        self._create_hover_tooltip(self.btn_browse_input_depth_maps_folder, "input_depth_maps_folder")
        self._create_hover_tooltip(self.btn_select_input_depth_maps_file, "input_depth_maps_file")
        current_row += 1

        # Output Splatted Row
        self.lbl_output_splatted = ttk.Label(self.folder_frame, text="Output Splatted:")
        self.lbl_output_splatted.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        self.entry_output_splatted = ttk.Entry(self.folder_frame, textvariable=self.output_splatted_var)
        self.entry_output_splatted.grid(row=current_row, column=1, padx=5, pady=2, sticky="ew")
        self.btn_browse_output_splatted = ttk.Button(self.folder_frame, text="Browse Folder", command=lambda: self._browse_folder(self.output_splatted_var))
        self.btn_browse_output_splatted.grid(row=current_row, column=2, columnspan=2, padx=5, pady=2)
        self._create_hover_tooltip(self.lbl_output_splatted, "output_splatted")
        self._create_hover_tooltip(self.entry_output_splatted, "output_splatted")
        self._create_hover_tooltip(self.btn_browse_output_splatted, "output_splatted")
        # Reset current_row for next frame
        current_row = 0

        # --- NEW: PREVIEW FRAME ---
        self.previewer = VideoPreviewer(
            self,
            processing_callback=self._preview_processing_callback,
            find_sources_callback=self._find_preview_sources_callback,
            get_params_callback=self.get_current_preview_settings,
            preview_size_var=self.preview_size_var, # Pass the preview size variable
            resize_callback=self._adjust_window_height_for_content, # Pass the resize callback
            update_clip_callback=self._update_clip_state_and_text,
            help_data=self.help_texts,
        )
        self.previewer.pack(fill="both", expand=True, padx=10, pady=5)
        self.previewer.preview_source_combo.configure(textvariable=self.preview_source_var)

        # --- NEW: MAIN LAYOUT CONTAINER (Holds Settings Left and Info Right) ---
        self.main_layout_frame = ttk.Frame(self)
        self.main_layout_frame.pack(pady=10, padx=10, fill="x")
        self.main_layout_frame.grid_columnconfigure(0, weight=1) # Left settings column
        self.main_layout_frame.grid_columnconfigure(1, weight=1) # Right info column (fixed width)

        # --- LEFT COLUMN: Settings Stack Frame ---
        self.settings_stack_frame = ttk.Frame(self.main_layout_frame)
        self.settings_stack_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # --- Settings Container Frame (to hold two side-by-side frames) ---
        self.settings_container_frame = ttk.Frame(self.settings_stack_frame)
        self.settings_container_frame.pack(pady=(0, 10), fill="x") # Pack it inside the stack frame
        self.settings_container_frame.grid_columnconfigure(0, weight=1)
        self.settings_container_frame.grid_columnconfigure(1, weight=1)

        # ===================================================================
        # LEFT SIDE: Process Resolution and Settings Frame
        # ===================================================================

        # This container holds both the resolution settings (top) and the splatting/output settings (bottom)
        self.process_settings_container = ttk.Frame(self.settings_container_frame)
        self.process_settings_container.grid(row=0, column=0, padx=(5, 0), sticky="nsew")
        self.process_settings_container.grid_columnconfigure(0, weight=1)

        # --- 1. Process Resolution Frame (Top Left) ---
        self.preprocessing_frame = ttk.LabelFrame(self.process_settings_container, text="Process Resolution")
        self.preprocessing_frame.grid(row=0, column=0, padx=(0, 5), sticky="nsew") # <-- Grid 0,0 in process_settings_container
        self.preprocessing_frame.grid_columnconfigure(1, weight=1) # Allow Entry to expand

        current_row = 0

         # --- Enable Full Resolution Section (ROW 0) ---
        
        # Container for Checkbox and Label/Entry
        self.full_res_control_frame = ttk.Frame(self.preprocessing_frame)
        self.full_res_control_frame.grid(row=current_row, column=0, columnspan=2, sticky="w", padx=5)
        self.full_res_control_frame.grid_columnconfigure(0, weight=1) # Checkbox takes most space
        
        # Checkbox (Left side of the container)
        self.enable_full_res_checkbox = ttk.Checkbutton(self.full_res_control_frame, text="Enable Full Res", variable=self.enable_full_res_var,
                                                        command=self.toggle_processing_settings_fields, width=15)
        self.enable_full_res_checkbox.grid(row=0, column=0, sticky="w")
        self._create_hover_tooltip(self.enable_full_res_checkbox, "enable_full_res")
        
        # Label/Entry (Right side of the container)
        self.lbl_full_res_batch_size = ttk.Label(self.full_res_control_frame, text="Batch Size:")
        self.lbl_full_res_batch_size.grid(row=0, column=1, sticky="w", padx=(10, 2))
        self.entry_full_res_batch_size = ttk.Entry(self.full_res_control_frame, textvariable=self.batch_size_var, width=5)
        self.entry_full_res_batch_size.grid(row=0, column=2, sticky="w", padx=(0, 0))
        self._create_hover_tooltip(self.lbl_full_res_batch_size, "full_res_batch_size")
        self._create_hover_tooltip(self.entry_full_res_batch_size, "full_res_batch_size")
        current_row += 1


        # --- Enable Low Resolution Section (ROW 1) ---
        
        # Container for Checkbox and Label/Entry
        self.low_res_control_frame = ttk.Frame(self.preprocessing_frame)
        self.low_res_control_frame.grid(row=current_row, column=0, columnspan=2, sticky="w", padx=5, pady=(10, 2))
        self.low_res_control_frame.grid_columnconfigure(0, weight=1) # Checkbox takes most space

        # Checkbox (Left side of the container)
        self.enable_low_res_checkbox = ttk.Checkbutton(self.low_res_control_frame, text="Enable Low Res", variable=self.enable_low_res_var,
                                                       command=self.toggle_processing_settings_fields, width=15)
        self.enable_low_res_checkbox.grid(row=0, column=0, sticky="w")
        self._create_hover_tooltip(self.enable_low_res_checkbox, "enable_low_res")
        
        # Label/Entry (Right side of the container)
        self.lbl_low_res_batch_size = ttk.Label(self.low_res_control_frame, text="Batch Size:")
        self.lbl_low_res_batch_size.grid(row=0, column=1, sticky="w", padx=(10, 2))
        self.entry_low_res_batch_size = ttk.Entry(self.low_res_control_frame, textvariable=self.low_res_batch_size_var, width=5)
        self.entry_low_res_batch_size.grid(row=0, column=2, sticky="w", padx=(0, 0))
        self._create_hover_tooltip(self.lbl_low_res_batch_size, "low_res_batch_size")
        self._create_hover_tooltip(self.entry_low_res_batch_size, "low_res_batch_size")
        current_row += 1
        
        # --- Low Res Width/Height (Squeezed onto one row) (ROW 2) ---
        
        # Frame for Width/Height fields (Grid under the Low Res checkbox/batch size row)
        self.low_res_wh_frame = ttk.Frame(self.preprocessing_frame)
        self.low_res_wh_frame.grid(row=current_row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        self.pre_res_width_label = ttk.Label(self.low_res_wh_frame, text="Width:")
        self.pre_res_width_label.pack(side="left", padx=(0, 2))
        self.pre_res_width_entry = ttk.Entry(self.low_res_wh_frame, textvariable=self.pre_res_width_var, width=9)
        self.pre_res_width_entry.pack(side="left", padx=(0, 10))

        self.pre_res_height_label = ttk.Label(self.low_res_wh_frame, text="Height:")
        self.pre_res_height_label.pack(side="left", padx=(0, 2))
        self.pre_res_height_entry = ttk.Entry(self.low_res_wh_frame, textvariable=self.pre_res_height_var, width=9)
        self.pre_res_height_entry.pack(side="left", padx=(0, 0))

        self._create_hover_tooltip(self.pre_res_width_label, "low_res_width")
        self._create_hover_tooltip(self.pre_res_width_entry, "low_res_width")
        self._create_hover_tooltip(self.pre_res_height_label, "low_res_height")
        self._create_hover_tooltip(self.pre_res_height_entry, "low_res_height")
        current_row += 1
        
        # Dual Output Checkbox (Row 3, Column 0/1)
        self.dual_output_checkbox = ttk.Checkbutton(self.preprocessing_frame, text="Dual Output Only", variable=self.dual_output_var)
        self.dual_output_checkbox.grid(row=current_row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.dual_output_checkbox, "dual_output")
        
        # --- 2. Splatting & Output Settings Frame (Bottom Left) ---
        # *** THIS IS THE MOVED FRAME: now attached to self.process_settings_container at row=1 ***
        current_row = 0 # Reset for internal use of output_settings_frame
        self.output_settings_frame = ttk.LabelFrame(self.process_settings_container, text="Splatting & Output Settings")
        self.output_settings_frame.grid(row=1, column=0, padx=(0, 5), sticky="ew", pady=(10, 0)) # <-- Grid 1,0 in process_settings_container
        self.output_settings_frame.grid_columnconfigure(1, weight=1)
                
        # Process Length (Remains Entry)
        self.lbl_process_length = ttk.Label(self.output_settings_frame, text="Process Length:")
        self.lbl_process_length.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        self.entry_process_length = ttk.Entry(self.output_settings_frame, textvariable=self.process_length_var, width=15)
        self.entry_process_length.grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.lbl_process_length, "process_length")
        self._create_hover_tooltip(self.entry_process_length, "process_length")
        current_row += 1
        
        # Output CRF setting (Remains Entry)
        self.lbl_output_crf = ttk.Label(self.output_settings_frame, text="Output CRF:")
        self.lbl_output_crf.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        self.entry_output_crf = ttk.Entry(self.output_settings_frame, textvariable=self.output_crf_var, width=15)
        self.entry_output_crf.grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.lbl_output_crf, "output_crf")
        self._create_hover_tooltip(self.entry_output_crf, "output_crf")
        current_row += 1

        # Auto-Convergence Combo (Row 2, Column 0/1)
        self.lbl_auto_convergence = ttk.Label(self.output_settings_frame, text="Auto-Convergence:")
        self.lbl_auto_convergence.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        self.auto_convergence_combo = ttk.Combobox(self.output_settings_frame, textvariable=self.auto_convergence_mode_var, values=["Off", "Average", "Peak"], state="readonly", width=15)
        self.auto_convergence_combo.grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.lbl_auto_convergence, "auto_convergence_toggle")
        self._create_hover_tooltip(self.auto_convergence_combo, "auto_convergence_toggle")
        self.auto_convergence_combo.bind("<<ComboboxSelected>>", self.on_auto_convergence_mode_select)

        current_row = 0 # Reset for next frame

        # ===================================================================
        # RIGHT SIDE: Depth Map Pre-processing Frame
        # ===================================================================
        self.depth_settings_container = ttk.Frame(self.settings_container_frame)
        self.depth_settings_container.grid(row=0, column=1, padx=(5, 0), sticky="nsew")
        self.depth_settings_container.grid_columnconfigure(0, weight=1)
        
        # --- Hi-Res Depth Pre-processing Frame (Top-Right) ---
        current_depth_row = 0 # Use a new counter for this container
        self.depth_prep_frame = ttk.LabelFrame(self.depth_settings_container, text="Depth Map Pre-processing (Hi-Res Only)")
        self.depth_prep_frame.grid(row=current_depth_row, column=0, sticky="ew") # Use grid here for placement inside container
        self.depth_prep_frame.grid_columnconfigure(1, weight=1)

        # Slider Implementation for dilate and blur
        row_inner = 0
        create_dual_slider_layout(
            self, self.depth_prep_frame, "Dilate X:", "Y:",
            self.depth_dilate_size_x_var, self.depth_dilate_size_y_var, 0, 15,
            row_inner, decimals=0, is_integer=True,
            tooltip_key_x="depth_dilate_size_x",
            tooltip_key_y="depth_dilate_size_y",
            )

        row_inner += 1
        create_dual_slider_layout(
            self, self.depth_prep_frame, "   Blur X:", "Y:",
            self.depth_blur_size_x_var, self.depth_blur_size_y_var, 0, 35,
            row_inner, decimals=0, is_integer=True,
            tooltip_key_x="depth_blur_size_x",
            tooltip_key_y="depth_blur_size_y",
            )

        # --- NEW: Depth Pre-processing (All) Frame (Bottom-Right) ---
        current_depth_row += 1
        self.depth_all_settings_frame = ttk.LabelFrame(self.depth_settings_container, text="Depth Map Settings (All)")
        self.depth_all_settings_frame.grid(row=current_depth_row, column=0, sticky="ew", pady=(10, 0)) # Pack it below Hi-Res frame
        # self.depth_all_settings_frame.grid_columnconfigure(1, weight=1)
        # self.depth_all_settings_frame.grid_columnconfigure(3, weight=1)

        all_settings_row = 0
        
        # Gamma Slider (MOVED FROM OUTPUT FRAME)
        create_single_slider_with_label_updater(
            self, self.depth_all_settings_frame, "Gamma:",
            self.depth_gamma_var, 0.1, 3.0, all_settings_row, decimals=1,
            tooltip_key="depth_gamma",
            )
        all_settings_row += 1

        # Max Disparity Slider (MOVED FROM OUTPUT FRAME)
        create_single_slider_with_label_updater(
            self, self.depth_all_settings_frame, "Disparity:",
            self.max_disp_var, 0.0, 100.0, all_settings_row, decimals=0,
            tooltip_key="max_disp",
            )
        all_settings_row += 1
        
        # Convergence Point Slider (MOVED FROM OUTPUT FRAME)
        setter_func_conv = create_single_slider_with_label_updater(
            self, self.depth_all_settings_frame, "Convergence:",
            self.zero_disparity_anchor_var, 0.0, 1.0, all_settings_row, decimals=2,
            tooltip_key="convergence_point",
            )
        self.set_convergence_value_programmatically = setter_func_conv 
        
        all_settings_row += 1
        
        # Autogain Checkbox (MOVED FROM OUTPUT FRAME, placed in column 2/3)
        self.autogain_checkbox = ttk.Checkbutton(
            self.depth_all_settings_frame, text="Disable Normalization",
            variable=self.enable_autogain_var,
            command=lambda: self.on_slider_release(None),
            width=28
            )
        self.autogain_checkbox.grid(row=all_settings_row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.autogain_checkbox, "no_normalization")   

        all_settings_row += 1
        
        # --- NEW: Override Sidecar Checkbox ---
        self.override_sidecar_checkbox = ttk.Checkbutton(
            self.depth_all_settings_frame, text="Override Sidecar (Preview Only)",
            variable=self.override_sidecar_var,
            command=lambda: [self.on_slider_release(None), self._toggle_sidecar_update_button_state()],
            width=28
            )
        self.override_sidecar_checkbox.grid(row=all_settings_row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.override_sidecar_checkbox, "override_sidecar_preview")

        current_row = 0 # Reset for next frame
        # ===================================================================
        # --- RIGHT COLUMN: Current Processing Information frame ---
        # ===================================================================
        self.info_frame = ttk.LabelFrame(self.main_layout_frame, text="Current Processing Information") # Target main layout frame
        self.info_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 0)) # Stick to North (Top)
        self.info_frame.grid_columnconfigure(1, weight=1) # Allow value column to expand (if frame is stretched)

        self.info_labels = [] # List to hold the tk.Label widgets for easy iteration

        LABEL_VALUE_WIDTH = 25
        info_row = 0


        # Row 0: Filename
        lbl_filename_static = tk.Label(self.info_frame, text="Filename:")
        lbl_filename_static.grid(row=info_row, column=0, sticky="e", padx=5, pady=1)
        lbl_filename_value = tk.Label(self.info_frame, textvariable=self.processing_filename_var, anchor="w", width=LABEL_VALUE_WIDTH)
        lbl_filename_value.grid(row=info_row, column=1, sticky="ew", padx=5, pady=1)
        self.info_labels.extend([lbl_filename_static, lbl_filename_value])
        info_row += 1


        # Row 1: Task Name
        lbl_task_static = tk.Label(self.info_frame, text="Task:")
        lbl_task_static.grid(row=info_row, column=0, sticky="e", padx=5, pady=1)
        lbl_task_value = tk.Label(self.info_frame, textvariable=self.processing_task_name_var, anchor="w", width=LABEL_VALUE_WIDTH)
        lbl_task_value.grid(row=info_row, column=1, sticky="ew", padx=5, pady=1)
        self.info_labels.extend([lbl_task_static, lbl_task_value])
        info_row += 1

        # Row 2: Resolution
        lbl_resolution_static = tk.Label(self.info_frame, text="Resolution:")
        lbl_resolution_static.grid(row=info_row, column=0, sticky="e", padx=5, pady=1)
        lbl_resolution_value = tk.Label(self.info_frame, textvariable=self.processing_resolution_var, anchor="w", width=LABEL_VALUE_WIDTH)
        lbl_resolution_value.grid(row=info_row, column=1, sticky="ew", padx=5, pady=1)
        self.info_labels.extend([lbl_resolution_static, lbl_resolution_value])
        info_row += 1

        # Row 3: Total Frames for current task
        lbl_frames_static = tk.Label(self.info_frame, text="Frames:")
        lbl_frames_static.grid(row=info_row, column=0, sticky="e", padx=5, pady=1)
        lbl_frames_value = tk.Label(self.info_frame, textvariable=self.processing_frames_var, anchor="w", width=LABEL_VALUE_WIDTH)
        lbl_frames_value.grid(row=info_row, column=1, sticky="ew", padx=5, pady=1)
        self.info_labels.extend([lbl_frames_static, lbl_frames_value])
        info_row += 1

        # --- NEW ROW 6: Gamma ---
        lbl_gamma_static = tk.Label(self.info_frame, text="Gamma:")
        lbl_gamma_static.grid(row=info_row, column=0, sticky="e", padx=5, pady=1)
        lbl_gamma_value = tk.Label(self.info_frame, textvariable=self.processing_gamma_var, anchor="w", width=LABEL_VALUE_WIDTH)
        lbl_gamma_value.grid(row=info_row, column=1, sticky="ew", padx=5, pady=1)
        self.info_labels.extend([lbl_gamma_static, lbl_gamma_value])
        info_row += 1
        # ------------------------

        # Row 4: Max Disparity
        lbl_disparity_static = tk.Label(self.info_frame, text="Disparity:")
        lbl_disparity_static.grid(row=info_row, column=0, sticky="e", padx=5, pady=1)
        lbl_disparity_value = tk.Label(self.info_frame, textvariable=self.processing_disparity_var, anchor="w", width=LABEL_VALUE_WIDTH)
        lbl_disparity_value.grid(row=info_row, column=1, sticky="ew", padx=5, pady=1)
        self.info_labels.extend([lbl_disparity_static, lbl_disparity_value])
        info_row += 1

        # Row 5: Convergence Point
        lbl_convergence_static = tk.Label(self.info_frame, text="Converge:")
        lbl_convergence_static.grid(row=info_row, column=0, sticky="e", padx=5, pady=1)
        lbl_convergence_value = tk.Label(self.info_frame, textvariable=self.processing_convergence_var, anchor="w", width=LABEL_VALUE_WIDTH)
        lbl_convergence_value.grid(row=info_row, column=1, sticky="ew", padx=5, pady=1)
        self.info_labels.extend([lbl_convergence_static, lbl_convergence_value])

        # --- Progress frame ---
        progress_frame = ttk.LabelFrame(self, text="Progress")
        progress_frame.pack(pady=10, padx=10, fill="x")
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", expand=True, padx=5, pady=2)
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack(padx=5, pady=2)

        # --- Button frame ---
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)

        
        # --- Single Process Button ---
        self.start_single_button = ttk.Button(button_frame, text="SINGLE", command=self.start_single_processing)
        self.start_single_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.start_single_button, "start_single_button")

        # --- Start Process Button ---
        self.start_button = ttk.Button(button_frame, text="START", command=self.start_processing)
        self.start_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.start_button, "start_button")

        # --- Stop Process Button ---
        self.stop_button = ttk.Button(button_frame, text="STOP", command=self.stop_processing, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.stop_button, "stop_button")
        
        # --- Preview Auto-Converge Button ---
        self.btn_auto_converge_preview = ttk.Button(button_frame, text="Preview Auto-Converge", command=self.run_preview_auto_converge)
        self.btn_auto_converge_preview.pack(side="left", padx=5)
        self._create_hover_tooltip(self.btn_auto_converge_preview, "preview_auto_converge")

        # --- Update Sidecar Button ---
        self.update_sidecar_button = ttk.Button(button_frame, text="Update Sidecar", command=self.update_sidecar_file)
        self.update_sidecar_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.update_sidecar_button, "update_sidecar_button")

        # --- Current Processing Information frame ---
        self.info_frame = ttk.LabelFrame(self, text="Current Processing Information") # Store frame as instance attribute
        self.info_frame.pack(pady=10, padx=10, fill="x")
        self.info_frame.grid_columnconfigure(1, weight=1)
    
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
            frame_overlap: Optional[int],
            input_bias: Optional[float],
            assume_raw_input: bool, 
            global_depth_min: float, 
            global_depth_max: float,  
            depth_stream_info: Optional[dict],
            user_output_crf: Optional[int] = None,
            is_low_res_task: bool = False,
            depth_gamma: float = 1.0,
            depth_dilate_size_x: int = 0,
            depth_dilate_size_y: int = 0,
            depth_blur_size_x: int = 0,
            depth_blur_size_y: int = 0,
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
                if self.stop_event.is_set() or ffmpeg_process.poll() is not None:
                    if ffmpeg_process.poll() is not None:
                        logger.error("FFmpeg process terminated unexpectedly. Stopping frame processing.")
                    else:
                        logger.warning("Stop event received. Terminating FFmpeg process.")
                    encoding_successful = False
                    break

                current_frame_indices = list(range(i, min(i + batch_size, num_frames)))
                if not current_frame_indices:
                    break

                batch_frames_numpy = input_video_reader.get_batch(current_frame_indices).asnumpy()
                # --- CRITICAL FIX: Seek Depth Reader to the first frame of the batch ---
                # This often resolves issues where Decord/FFmpeg loses the internal stream position
                try:
                    # Seek to the first frame of the current batch
                    depth_map_reader.seek(current_frame_indices[0]) 
                    # Then read the full batch from that position
                    batch_depth_numpy_raw = depth_map_reader.get_batch(current_frame_indices).asnumpy()
                except Exception as e:
                    logger.error(f"Error seeking/reading depth map batch starting at index {i}: {e}. Falling back to a potentially blank read.")
                    batch_depth_numpy_raw = depth_map_reader.get_batch(current_frame_indices).asnumpy()
                # --- END CRITICAL FIX ---
                
                # --- NEW: Define debug variables early ---
                file_frame_idx = current_frame_indices[0] 
                task_name = "LowRes" if is_low_res_task else "HiRes"
                # --- END NEW ---
                
                # --- DEBUG CHECK (Keep this to confirm the fix) ---
                if batch_depth_numpy_raw.min() == batch_depth_numpy_raw.max() == 0:
                    logger.warning(f"Depth map batch starting at index {i} is entirely blank/zero after read. **Seeking failed to resolve.**")
                # --- END DEBUG CHECK ---
                    
                if batch_depth_numpy_raw.min() == batch_depth_numpy_raw.max():
                    logger.warning(f"Depth map batch starting at index {i} is entirely uniform/flat after read. Min/Max: {batch_depth_numpy_raw.min():.2f}")
                # --- END NEW DEBUG & CHECK ---

                # Use the FIRST frame index for the file name (e.g., 00000.png)
                file_frame_idx = current_frame_indices[0] 
                
                # self._save_debug_numpy(batch_depth_numpy_raw, "01_RAW_INPUT", i, file_frame_idx, task_name) 
                
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
                
                batch_depth_vis_list = []
                for d_frame in batch_depth_normalized:
                    d_frame_vis = d_frame.copy()
                    if d_frame_vis.max() > d_frame_vis.min(): 
                        cv2.normalize(d_frame_vis, d_frame_vis, 0, 1, cv2.NORM_MINMAX)
                    vis_frame_uint8 = (d_frame_vis * 255).astype(np.uint8)
                    vis_frame = cv2.applyColorMap(vis_frame_uint8, cv2.COLORMAP_VIRIDIS)
                    batch_depth_vis_list.append(vis_frame.astype("float32") / 255.0)
                batch_depth_vis = np.stack(batch_depth_vis_list, axis=0) 

                left_video_tensor = torch.from_numpy(batch_frames_numpy).permute(0, 3, 1, 2).float().cuda() / 255.0
                disp_map_tensor = torch.from_numpy(batch_depth_normalized).unsqueeze(1).float().cuda()        
                disp_map_tensor = (disp_map_tensor - zero_disparity_anchor_val) * 2.0
                disp_map_tensor = disp_map_tensor * max_disp

                with torch.no_grad():
                    right_video_tensor_raw, occlusion_mask_tensor = stereo_projector(left_video_tensor, disp_map_tensor)
                    if is_low_res_task:
                        # 1. Fill Left Edge Occlusions
                        right_video_tensor_left_filled = self._fill_left_edge_occlusions(right_video_tensor_raw, occlusion_mask_tensor, boundary_width_pixels=3)
                        
                        # 2. Fill Right Edge Occlusions (New Call)
                        right_video_tensor = self._fill_right_edge_occlusions(right_video_tensor_left_filled, occlusion_mask_tensor, boundary_width_pixels=3)
                    else:
                        right_video_tensor = right_video_tensor_raw

                right_video_numpy = right_video_tensor.cpu().permute(0, 2, 3, 1).numpy()
                occlusion_mask_numpy = occlusion_mask_tensor.cpu().permute(0, 2, 3, 1).numpy().repeat(3, axis=-1)

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

                del left_video_tensor, disp_map_tensor, right_video_tensor, occlusion_mask_tensor
                torch.cuda.empty_cache()
                draw_progress_bar(frame_count, num_frames, prefix=f"  Encoding:")
        
        except (IOError, BrokenPipeError) as e:
            logger.error(f"FFmpeg pipe error: {e}. Encoding may have failed.")
            encoding_successful = False
        finally:
            del stereo_projector
            torch.cuda.empty_cache()
            gc.collect()

            # --- Finalize FFmpeg process ---
            if ffmpeg_process.stdin:
                ffmpeg_process.stdin.close() # Close the pipe to signal end of input
            
            # Wait for the process to finish and get output
            stdout, stderr = ffmpeg_process.communicate(timeout=120)
            
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

        # --- Write sidecar JSON after successful encoding ---
        output_sidecar_data = {}
        if frame_overlap is not None:
            output_sidecar_data["frame_overlap"] = frame_overlap
        if input_bias is not None:
            output_sidecar_data["input_bias"] = input_bias
        
        if output_sidecar_data:
            sidecar_ext = self.APP_CONFIG_DEFAULTS.get('OUTPUT_SIDECAR_EXT', '.spsidecar')
            output_sidecar_path = f"{os.path.splitext(final_output_video_path)[0]}{sidecar_ext}"
            try:
                with open(output_sidecar_path, 'w', encoding='utf-8') as f:
                    json.dump(output_sidecar_data, f, indent=4)
                logger.info(f"Created output sidecar file: {output_sidecar_path}")
            except Exception as e:
                logger.error(f"Error creating output sidecar file '{output_sidecar_path}': {e}")
        
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
            # --- NEW: Cleanup previewer resources ---
            if hasattr(self, 'previewer'):
                self.previewer.cleanup()
            self.processing_thread.join(timeout=5.0)
            if self.processing_thread.is_alive():
                logger.debug("==> Thread did not terminate gracefully within timeout.")
        release_cuda_memory()
        self.destroy()

    def _fill_left_edge_occlusions(self, right_video_tensor: torch.Tensor, occlusion_mask_tensor: torch.Tensor, boundary_width_pixels: int = 3) -> torch.Tensor:
        """
        Creates a thin, content-filled boundary at the absolute left edge of the screen
        by replicating the first visible pixels (from the right) into the leftmost columns.
        The region between this new boundary and the actual content is left occluded for inpainting.

        Args:
            right_video_tensor (torch.Tensor): The forward-warped right-eye video tensor [B, C, H, W],
                                               values in [0, 1].
            occlusion_mask_tensor (torch.Tensor): The corresponding occlusion mask tensor [B, 1, H, W],
                                                  where 1 indicates occlusion.
            boundary_width_pixels (int): How many columns at the absolute left edge to fill
                                         with replicated content (e.g., 1, 2, or 3 pixels wide).

        Returns:
            torch.Tensor: The modified right-eye video tensor with the left-edge boundary filled.
        """
        B, C, H, W = right_video_tensor.shape

        # Ensure boundary_width_pixels is valid and not too large
        boundary_width_pixels = min(W, boundary_width_pixels)
        if boundary_width_pixels <= 0:
            logger.debug("Boundary width for left-edge occlusions is 0 or less, skipping fill.")
            return right_video_tensor # No filling needed

        modified_right_video_tensor = right_video_tensor.clone()

        # Iterate through each batch item and each row independently
        for b_idx in range(B):
            for h_idx in range(H):
                # Find the first non-occluded column 'X' for this specific row, moving from left to right.
                # If a row is entirely occluded, or only has content on the far right, it defaults to W-1
                # to pick up some content, or just won't apply if col 0 is already visible and we don't overwrite.
                
                # Invert the occlusion mask (0=occluded, 1=visible) to find the first '1' (visible)
                visible_mask_row = (occlusion_mask_tensor[b_idx, 0, h_idx, :] <= 0.5) # True where visible
                
                # Invert the occlusion mask (0=occluded, 1=visible) to find the first '1' (visible)
                # `occlusion_mask_tensor` is 1 for occluded, 0 for visible.
                # We want to find the first column where occlusion is NOT 1 (i.e., visible)
                # Ensure it's a bool tensor for torch.nonzero
                is_visible_col_mask = (occlusion_mask_tensor[b_idx, 0, h_idx, :] < 0.5)
                
                # Find all column indices that are visible
                visible_column_indices = torch.nonzero(is_visible_col_mask, as_tuple=True)[0]

                # Determine the 'source column' for filling
                source_col_for_boundary_fill: int # Explicitly type as int for Pylance
                
                if visible_column_indices.numel() > 0:
                    # If there's any visible content, take the *first* visible column.
                    source_col_for_boundary_fill = int(visible_column_indices[0].item())
                    # Ensure it's not trying to access beyond the tensor boundary if some edge cases exist.
                    source_col_for_boundary_fill = min(source_col_for_boundary_fill, W - 1)
                else:
                    # If the entire row is occluded, or only has very far-right content.
                    # Fallback: Use the last valid column (W-1) to ensure we always get *some* pixels.
                    # This might replicate a black pixel if the whole row is black, but avoids IndexError.
                    source_col_for_boundary_fill = W - 1
                    # logger.debug(f"Row {h_idx} in batch {b_idx} is fully occluded or near-fully. Using last column for boundary fill source.")
                
                # Get the pixel values from this 'source column' to use for the boundary.
                # Pylance should now correctly infer source_col_for_boundary_fill as an int.
                source_pixel_values = right_video_tensor[b_idx, :, h_idx, source_col_for_boundary_fill] # Shape [C]

                # Now, fill the leftmost 'boundary_width_pixels' columns for this row,
                # but ONLY if those columns are currently occluded.
                for x in range(boundary_width_pixels):
                    # Check if the current pixel at (b_idx, :, h_idx, x) is occluded
                    if occlusion_mask_tensor[b_idx, 0, h_idx, x] > 0.5: # If currently occluded
                        modified_right_video_tensor[b_idx, :, h_idx, x] = source_pixel_values

        logger.debug(f"Created {boundary_width_pixels}-pixel left-edge content boundary.")
        return modified_right_video_tensor

    def _fill_right_edge_occlusions(self, right_video_tensor: torch.Tensor, occlusion_mask_tensor: torch.Tensor, boundary_width_pixels: int = 3) -> torch.Tensor:
        """
        Creates a thin, content-filled boundary at the absolute right edge of the screen
        by replicating the last visible pixels (from the left) into the rightmost columns.
        
        Args:
            right_video_tensor (torch.Tensor): The forward-warped right-eye video tensor [B, C, H, W],
                                               values in [0, 1].
            occlusion_mask_tensor (torch.Tensor): The corresponding occlusion mask tensor [B, 1, H, W],
                                                  where 1 indicates occlusion.
            boundary_width_pixels (int): How many columns at the absolute right edge to fill.

        Returns:
            torch.Tensor: The modified right-eye video tensor with the right-edge boundary filled.
        """
        B, C, H, W = right_video_tensor.shape

        boundary_width_pixels = min(W, boundary_width_pixels)
        if boundary_width_pixels <= 0:
            logger.debug("Boundary width for right-edge occlusions is 0 or less, skipping fill.")
            return right_video_tensor

        modified_right_video_tensor = right_video_tensor.clone()

        # Iterate through each batch item and each row independently
        for b_idx in range(B):
            for h_idx in range(H):
                # 1. Find the first non-occluded column 'X' for this specific row, moving from RIGHT to LEFT.
                # Invert the occlusion mask (0=occluded, 1=visible) to find the last '1' (visible)
                is_visible_col_mask = (occlusion_mask_tensor[b_idx, 0, h_idx, :] < 0.5)
                
                # Find all column indices that are visible
                visible_column_indices = torch.nonzero(is_visible_col_mask, as_tuple=True)[0]

                # Determine the 'source column' for filling
                source_col_for_boundary_fill: int
                
                if visible_column_indices.numel() > 0:
                    # If there's any visible content, take the *last* visible column.
                    source_col_for_boundary_fill = int(visible_column_indices[-1].item())
                    # Ensure it's not accessing beyond the tensor boundary
                    source_col_for_boundary_fill = max(source_col_for_boundary_fill, 0)
                else:
                    # Fallback: Use the first valid column (0)
                    source_col_for_boundary_fill = 0
                
                # Get the pixel values from this 'source column' to use for the boundary.
                source_pixel_values = right_video_tensor[b_idx, :, h_idx, source_col_for_boundary_fill] # Shape [C]

                # 2. Fill the rightmost 'boundary_width_pixels' columns for this row
                for x_offset in range(boundary_width_pixels):
                    x = W - 1 - x_offset # Column index (W-1 is the far right)
                    
                    # Only fill if the current pixel is currently occluded
                    if occlusion_mask_tensor[b_idx, 0, h_idx, x] > 0.5: # If currently occluded
                        modified_right_video_tensor[b_idx, :, h_idx, x] = source_pixel_values

        logger.debug(f"Created {boundary_width_pixels}-pixel right-edge content boundary.")
        return modified_right_video_tensor

    def _find_preview_sources_callback(self) -> list:
        """
        Callback for VideoPreviewer. Scans for matching source video and depth map pairs.
        Handles both folder (batch) and file (single) input modes.
        """
        source_path = self.input_source_clips_var.get()
        depth_path = self.input_depth_maps_var.get()
        
        # --- NEW SINGLE-FILE MODE CHECK ---
        is_source_file = os.path.isfile(source_path)
        is_depth_file = os.path.isfile(depth_path)
        
        if is_source_file and is_depth_file:
            # Single file mode activated
            logger.debug(f"Preview Scan: Single file mode detected. Source: {source_path}, Depth: {depth_path}")
            # The previewer expects a list of dictionaries, even for a single file
            return [{
                'source_video': source_path,
                'depth_map': depth_path
            }]
        # --- END NEW SINGLE-FILE MODE CHECK ---

        # Fallback to Batch (Folder) mode check
        if not os.path.isdir(source_path) or not os.path.isdir(depth_path):
            logger.error("Preview Scan Failed: Inputs must either be two files or two valid directories.")
            return []

        # The rest of the original batch/folder logic (using source_path/depth_path as folder variables)
        source_folder = source_path
        depth_folder = depth_path

        video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
        source_videos = []
        for ext in video_extensions:
            source_videos.extend(glob.glob(os.path.join(source_folder, ext)))

        video_source_list = []
        for video_path in sorted(source_videos):
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # --- MODIFIED: More robust search for depth map files ---
            depth_paths_to_check = [
                os.path.join(depth_folder, f"{base_name}_depth.mp4"),
                os.path.join(depth_folder, f"{base_name}_depth.npz"),
                os.path.join(depth_folder, f"{base_name}.mp4"), 
                os.path.join(depth_folder, f"{base_name}.npz"), 
            ]

            matching_depth_path = None
            
            for dp in depth_paths_to_check:
                if os.path.exists(dp):
                    matching_depth_path = dp
                    break
            
            if matching_depth_path:
                logger.debug(f"Preview Scan: Found pair for '{base_name}'.")
                video_source_list.append({
                    'source_video': video_path,
                    'depth_map': matching_depth_path
                })

        return video_source_list

    def get_current_preview_settings(self) -> dict:
        """Gathers settings from the GUI needed for the preview callback."""
        try:
            # Helper function to safely convert StringVar content to int
            def safe_int_conversion(var: tk.StringVar) -> int:
                try:
                    # Convert to float first to handle fractional strings like '35.227...'
                    return int(float(var.get()))
                except ValueError:
                    # If it fails (e.g., empty string), default to 0
                    return 0

            return {
                "max_disp": float(self.max_disp_var.get()),
                "convergence_point": float(self.zero_disparity_anchor_var.get()),
                "depth_gamma": float(self.depth_gamma_var.get()),
                # --- MODIFIED: Use safe conversion for integer kernel sizes (float -> int) ---
                "depth_dilate_size_x": safe_int_conversion(self.depth_dilate_size_x_var),
                "depth_dilate_size_y": safe_int_conversion(self.depth_dilate_size_y_var),
                "depth_blur_size_x": safe_int_conversion(self.depth_blur_size_x_var),
                "depth_blur_size_y": safe_int_conversion(self.depth_blur_size_y_var),
                "preview_size": self.preview_size_var.get(),
                "enable_autogain": self.enable_autogain_var.get(),
            }
        except (ValueError, tk.TclError) as e:
            logger.error(f"Invalid preview setting value: {e}")
            return None

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

            "process_length": self.process_length_var.get(),
            "output_crf": self.output_crf_var.get(),
            "dual_output": self.dual_output_var.get(),
            "auto_convergence_mode": self.auto_convergence_mode_var.get(),
            
            "depth_gamma": self.depth_gamma_var.get(),
            "max_disp": self.max_disp_var.get(),
            "convergence_point": self.zero_disparity_anchor_var.get(),
            "enable_autogain": self.enable_autogain_var.get(),
            "override_sidecar_preview": self.override_sidecar_var.get(),
        }
        return config

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
    
    def _get_video_specific_settings(self, video_path, input_depth_maps_path_setting, default_zero_disparity_anchor, gui_max_disp, is_single_file_mode):
        """
        Determines the actual depth map path and reads video-specific settings from a sidecar JSON.
        Returns a dictionary containing relevant settings or an 'error' key.
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        actual_depth_map_path = None
        
        # Determine actual depth map file path
        if is_single_file_mode:
            actual_depth_map_path = input_depth_maps_path_setting
            if not os.path.exists(actual_depth_map_path):
                return {"error": f"Single depth map file '{actual_depth_map_path}' not found."}
        else:
            depth_map_path_mp4 = os.path.join(input_depth_maps_path_setting, f"{video_name}_depth.mp4")
            depth_map_path_npz = os.path.join(input_depth_maps_path_setting, f"{video_name}_depth.npz")

            if os.path.exists(depth_map_path_mp4):
                actual_depth_map_path = depth_map_path_mp4
            elif os.path.exists(depth_map_path_npz):
                actual_depth_map_path = depth_map_path_npz
            else:
                return {"error": f"No depth map found for {video_name} in {input_depth_maps_path_setting}. Expected '{video_name}_depth.mp4' or '{video_name}_depth.npz'."}
        
        actual_depth_map_path = os.path.normpath(actual_depth_map_path)
        depth_map_basename = os.path.splitext(os.path.basename(actual_depth_map_path))[0]
        
        # --- Get Sidecar Path ---
        sidecar_ext = self.APP_CONFIG_DEFAULTS['SIDECAR_EXT']
        json_sidecar_path = os.path.join(os.path.dirname(actual_depth_map_path), f"{depth_map_basename}{sidecar_ext}")
        # ------------------------

        # [NEW]: Compile GUI Input Fields for the Merge
        try:
            gui_config = {
                "convergence_plane": float(self.zero_disparity_anchor_var.get()),
                "max_disparity": float(self.max_disp_var.get()),
                "gamma": float(self.depth_gamma_var.get()),
                # Dilate/Blur are *not* included as they are local pre-processing steps, not config for the inpaint
            }
        except ValueError:
            gui_config = {}
        
        # The GUI override keys are defined for the batch process here (i.e., NO override)
        # The batch process always prefers the sidecar/default unless manually coded otherwise.
        # This function is used to load the sidecar's values for Auto-Convergence pre-pass calculation.
        
        # Use an empty list of override keys to ensure we get pure sidecar values first.
        # The main loop logic in _run_batch_process handles the final override selection.
        merged_config = self.sidecar_manager.load_sidecar_data(json_sidecar_path)

        # Initialize tracking sources for the main loop info display
        settings = {
            "actual_depth_map_path": actual_depth_map_path,
            "convergence_plane": merged_config["convergence_plane"],
            "max_disparity_percentage": merged_config["max_disparity"],
            "frame_overlap": merged_config.get("frame_overlap"),
            "input_bias": merged_config.get("input_bias"),
            "depth_gamma": merged_config["gamma"],
            
            # --- Placeholders for GUI-derived settings not in sidecar ---
            "depth_dilate_size_x": int(float(self.depth_dilate_size_x_var.get())),
            "depth_dilate_size_y": int(float(self.depth_dilate_size_y_var.get())),
            "depth_blur_size_x": int(float(self.depth_blur_size_x_var.get())),
            "depth_blur_size_y": int(float(self.depth_blur_size_y_var.get())),
            
            # We assume 'Sidecar' as the source for the tracking info if a sidecar was found,
            # otherwise, the main loop will need to infer the source.
            "sidecar_found": os.path.exists(json_sidecar_path),
            
            "anchor_source": "Sidecar" if os.path.exists(json_sidecar_path) else "GUI/Default",
            "max_disp_source": "Sidecar" if os.path.exists(json_sidecar_path) else "GUI/Default",
            "gamma_source": "Sidecar" if os.path.exists(json_sidecar_path) else "GUI/Default",
        }
        
        # Override with GUI values if sidecar not found (i.e., trust GUI over default 0.5/20.0)
        if not os.path.exists(json_sidecar_path):
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

        self.progress_queue.put(("update_info", {
            "resolution": f"{current_processed_width}x{current_processed_height}",
            "frames": total_frames_input
        }))

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
            return None, None, 0.0, 0, 0, None, 0, 0, 0, 0, None # Return None for depth_stream_info

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
        if os.path.exists("config_splat.json"):
            with open("config_splat.json", "r") as f:
                self.app_config = json.load(f)

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

    def load_settings(self):
        """Loads settings from a user-selected JSON file."""
        filename = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Load Settings from File"
        )
        if not filename:
            return

        try:
            with open(filename, "r") as f:
                loaded_config = json.load(f)
            # Apply loaded config values to the variables
            for config_key, config_value in loaded_config.items(): # Iterate over loaded keys
                # --- NEW MAPPING LOGIC ---
                # Construct the expected name of the Tkinter variable
                tk_var_attr_name = config_key + '_var'
                
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
                if key.endswith('_var') and key in loaded_config:
                    # Logic to safely set values:
                    # For tk.StringVar, set()
                    # For tk.BooleanVar, use set() with the bool/int value
                    if isinstance(var, tk.BooleanVar):
                        var.set(bool(loaded_config[key]))
                    elif isinstance(var, tk.StringVar):
                        var.set(str(loaded_config[key]))

            self._apply_theme() # Re-apply theme in case dark mode setting was loaded
            self.toggle_processing_settings_fields() # Update state of dependent fields
            messagebox.showinfo("Settings Loaded", f"Successfully loaded settings from:\n{os.path.basename(filename)}")
            self.status_label.config(text="Settings loaded.")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load settings from {os.path.basename(filename)}:\n{e}")
            self.status_label.config(text="Settings load failed.")
    
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
            self.status_label.config(text=f"Auto-Converge ({mode}): Loaded cached value {cached_value:.2f}")
            
            # 4. Refresh preview
            self.on_slider_release(None)
            
            return
        
        # Cache miss, run the calculation (using the existing run_preview_auto_converge logic)
        self.run_preview_auto_converge(force_run=True)

    def on_slider_release(self, event):
        """Called when a slider is released. Updates the preview."""
        if hasattr(self, 'previewer') and self.previewer.source_readers:
            self.previewer._stop_wigglegram_animation() 
            self.previewer.update_preview()            
            
            # Use the combined function here:
            if hasattr(self, '_update_clip_state_and_text'):
                 self._update_clip_state_and_text()

    def _process_depth_batch(self, batch_depth_numpy_raw: np.ndarray, depth_stream_info: Optional[dict], depth_gamma: float,
                              depth_dilate_size_x: int, depth_dilate_size_y: int, depth_blur_size_x: int, depth_blur_size_y: int, 
                              is_low_res_task: bool, max_raw_value: float,
                              global_depth_min: float, global_depth_max: float,
                              debug_batch_index: int = 0, debug_frame_index: int = 0, debug_task_name: str = "PreProcess",
                              ) -> np.ndarray:
        """
        Loads, converts, and pre-processes the raw depth map batch using stable NumPy/OpenCV CPU calls.
        """
        # 1. Grayscale Conversion (Standard NumPy)
        if batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 3: # RGB
            batch_depth_numpy = batch_depth_numpy_raw.mean(axis=-1)
        elif batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 1:
            batch_depth_numpy = batch_depth_numpy_raw.squeeze(-1)
        else:
            batch_depth_numpy = batch_depth_numpy_raw
        
        # Convert to float32 for processing
        batch_depth_numpy_float = batch_depth_numpy.astype(np.float32)
        # self._save_debug_image(batch_depth_numpy_float, "01_GRAYSCALE", debug_batch_index, debug_frame_index, debug_task_name)


        # 2. Gamma Adjustment (Only in RAW mode, otherwise skipped)
        is_global_norm_active = (global_depth_min != 0.0 or global_depth_max != 1.0) and not (global_depth_min == 0.0 and global_depth_max == 0.0)
        
        if depth_gamma != 1.0:
            if is_global_norm_active:
                logger.debug("Gamma adjustment SKIPPED in helper: Applied post-normalization (Global Norm Mode).")
            else:
                logger.debug(f"Applying depth gamma adjustment on raw range {max_raw_value:.1f}: {depth_gamma:.2f}")
                if max_raw_value > 1.0:
                    normalized_chunk = batch_depth_numpy_float / max_raw_value
                    normalized_chunk_gamma = np.power(normalized_chunk, depth_gamma)
                    batch_depth_numpy_float = normalized_chunk_gamma * max_raw_value
                else:
                    batch_depth_numpy_float = np.power(batch_depth_numpy_float, depth_gamma)

        # self._save_debug_image(batch_depth_numpy_float, "02_POST_GAMMA", debug_batch_index, debug_frame_index, debug_task_name)


        # --- 3. Dilate and Blur (Hi-Res Only, using stable OpenCV CPU calls) ---
        if not is_low_res_task:
            needs_processing = depth_dilate_size_x > 0 or depth_dilate_size_y > 0 or depth_blur_size_x > 0 or depth_blur_size_y > 0
            
            if needs_processing:
                processed_frames = []
                
                # --- CRITICAL FIX: Determine Scale Factor for uint8 Conversion ---
                # This ensures the float data is correctly scaled to 0-255 based on its true max value.
                scale_factor = global_depth_max if global_depth_max > 1.0 else 1.0
                
                for frame_np_float in batch_depth_numpy_float:
                    
                    # Convert float[0, MaxRaw] to uint8[0, 255] for OpenCV operations
                    # This is the stable scaling (frame * (255 / scale_factor))
                    frame_np_uint8 = (frame_np_float * (255.0 / scale_factor)).astype(np.uint8)
                    
                    # --- Dilate ---
                    k_x_dilate, k_y_dilate = depth_dilate_size_x, depth_dilate_size_y
                    if k_x_dilate > 0 or k_y_dilate > 0:
                        # Ensure non-zero kernels for OpenCV
                        k_x_dilate = max(1, k_x_dilate)
                        k_y_dilate = max(1, k_y_dilate)
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_x_dilate, k_y_dilate))
                        frame_np_uint8 = cv2.dilate(frame_np_uint8, kernel, iterations=1)
                    
                    # --- Blur ---
                    k_x_blur, k_y_blur = depth_blur_size_x, depth_blur_size_y
                    if k_x_blur > 0 or k_y_blur > 0:
                        # Ensure kernel size is odd for OpenCV
                        k_x_blur = k_x_blur if k_x_blur % 2 == 1 else k_x_blur + 1
                        k_y_blur = k_y_blur if k_y_blur % 2 == 1 else k_y_blur + 1
                        
                        frame_np_uint8 = cv2.GaussianBlur(frame_np_uint8, (k_x_blur, k_y_blur), 0)

                    processed_frames.append(frame_np_uint8)
                    
                # Convert back to float32 using the scale factor inverse
                batch_depth_numpy_uint8 = np.stack(processed_frames, axis=0)
                batch_depth_numpy_float = batch_depth_numpy_uint8.astype(np.float32) * (scale_factor / 255.0)

        # --- DEBUG SAVE 4: Final Processed Image ---
        # self._save_debug_image(batch_depth_numpy_float, "04_POST_BLUR_FINAL", debug_batch_index, debug_frame_index, debug_task_name)


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
        self.progress_queue.put(("update_info", {"filename": video_name}))
        
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
            self.progress_queue.put(("processed", local_task_counter))
            return expected_task_count, False

        actual_depth_map_path = video_specific_settings["actual_depth_map_path"]
        current_zero_disparity_anchor = video_specific_settings["convergence_plane"]
        current_max_disparity_percentage = video_specific_settings["max_disparity_percentage"]
        current_frame_overlap = video_specific_settings["frame_overlap"]
        current_input_bias = video_specific_settings["input_bias"]
        anchor_source = video_specific_settings["anchor_source"]
        max_disp_source = video_specific_settings["max_disp_source"]
        gamma_source = video_specific_settings["gamma_source"]
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

            # Update variables for current task
            if new_anchor_val != current_zero_disparity_anchor:
                current_zero_disparity_anchor = new_anchor_val
                anchor_source = "Auto"
            
            logger.info(f"Using Convergence Point: {current_zero_disparity_anchor:.4f} (Source: {anchor_source})")
        # --- END Auto-Convergence Logic ---

        for task in processing_tasks:
            if self.stop_event.is_set():
                logger.info(f"==> Stopping {task['name']} processing for {video_name} due to user request")
                # Increment the global counter for all remaining, skipped tasks
                remaining_tasks_to_increment = expected_task_count - processed_tasks_count
                local_task_counter += remaining_tasks_to_increment
                self.progress_queue.put(("processed", local_task_counter))
                return expected_task_count, any_task_completed_successfully_for_this_video

            logger.debug(f"\n==> Starting {task['name']} pass for {video_name}")
            self.progress_queue.put(("status", f"Processing {task['name']} for {video_name}"))
            
            self.progress_queue.put(("update_info", {
                "task_name": task['name'],
                "convergence": f"{current_zero_disparity_anchor:.2f} ({anchor_source})",
                "disparity": f"{current_max_disparity_percentage:.1f}% ({max_disp_source})",
                "gamma": f"{current_depth_gamma:.2f} ({gamma_source})",
            }))

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
                self.progress_queue.put(("processed", local_task_counter))
                release_cuda_memory()
                continue

            assume_raw_input_mode = settings["enable_autogain"] 
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

            self.progress_queue.put(("update_info", {"disparity": f"{current_max_disparity_percentage:.1f}% ({actual_max_disp_pixels:.2f} pixels)"}))

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
                frame_overlap=current_frame_overlap,
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
            self.progress_queue.put(("processed", local_task_counter))
            logger.debug(f"==> Completed {task['name']} pass for {video_name}.")

        # After all tasks for the current video are processed or stopped
        if self.stop_event.is_set():
            return expected_task_count, any_task_completed_successfully_for_this_video

        # Move to finished logic 
        if is_single_file_mode:
            # CRITICAL FIX: Get the finished folders directly from settings (set in start_single_processing)
            single_finished_src = settings.get("single_finished_source_folder")
            single_finished_depth = settings.get("single_finished_depth_folder")
            
            if single_finished_src and single_finished_depth and self.MOVE_TO_FINISHED_ENABLED and any_task_completed_successfully_for_this_video:
                self._move_processed_files(video_path, actual_depth_map_path, single_finished_src, single_finished_depth)
            else:
                logger.debug(f"Single file move skipped. Enabled={self.MOVE_TO_FINISHED_ENABLED}, Success={any_task_completed_successfully_for_this_video}, PathsValid={bool(single_finished_src)}")
        elif any_task_completed_successfully_for_this_video and finished_source_folder and finished_depth_folder:
            # Batch mode move (uses the arguments passed from _run_batch_process)
            self._move_processed_files(video_path, actual_depth_map_path, finished_source_folder, finished_depth_folder)

        # Return the number of tasks actually processed for the global counter update
        return processed_tasks_count, any_task_completed_successfully_for_this_video

    def _preview_processing_callback(self, source_frames: dict, params: dict) -> Optional[Image.Image]:
        """
        Callback for VideoPreviewer. Performs splatting on a single frame for preview.
        """
        self.clear_processing_info() # Clear info at the start of a new preview attempt

        if not globals()['CUDA_AVAILABLE']:
            logger.error("Preview processing requires a CUDA-enabled GPU.")
            return None
        
        logger.debug("--- Starting Preview Processing Callback ---")

        left_eye_tensor = source_frames.get('source_video')
        depth_tensor_raw = source_frames.get('depth_map')

        if left_eye_tensor is None or depth_tensor_raw is None:
            logger.error("Preview failed: Missing source video or depth map tensor.")
            return None

        # --- Get latest settings and Preview Mode ---
        params = self.get_current_preview_settings()
        if not params:
            logger.error("Preview failed: Could not get current preview settings.")
            return None
            
        preview_source = self.preview_source_var.get()
        is_low_res_preview = preview_source in ["Splat Result(Low)", "Occlusion Mask(Low)"]
        
        # Determine the target resolution for the preview tensor
        W_orig = left_eye_tensor.shape[3]
        H_orig = left_eye_tensor.shape[2]
        
        # ----------------------------------------------------------------------
        # NEW SIDECAR LOGIC FOR PREVIEW
        # ----------------------------------------------------------------------
        # --- FIX: Use reliable indexing to get the correct path ---
        depth_map_path = None
        if 0 <= self.previewer.current_video_index < len(self.previewer.video_list):
            current_source_dict = self.previewer.video_list[self.previewer.current_video_index]
            depth_map_path = current_source_dict.get('depth_map')
        # --- END FIX ---
        
        gui_config = {
            "convergence_plane": float(self.zero_disparity_anchor_var.get()),
            "max_disparity": float(self.max_disp_var.get()),
            "gamma": float(self.depth_gamma_var.get()),
        }
        
        # Determine the override keys
        override_keys = []
        if self.override_sidecar_var.get():
            # If override is checked, GUI values override all three
            override_keys = ["convergence_plane", "max_disparity", "gamma"]
        elif not self.enable_sidecar_gamma_var.get():
            # If gamma toggle is off, GUI gamma overrides Sidecar gamma
            override_keys = ["gamma"]

        conv_source = "GUI"
        gamma_source = "GUI"
        max_disp_source = "GUI"
        
        sidecar_path_found = False
        
        # --- MODIFIED BLOCK ---
        if depth_map_path:
            depth_map_basename = os.path.splitext(os.path.basename(depth_map_path))[0]
            sidecar_ext = self.APP_CONFIG_DEFAULTS['SIDECAR_EXT']
            json_sidecar_path = os.path.join(os.path.dirname(depth_map_path), f"{depth_map_basename}{sidecar_ext}")
            
            # Check if sidecar exists
            sidecar_path_found = os.path.exists(json_sidecar_path)

            if sidecar_path_found:
                # Merge: Sidecar is the base, GUI overrides where specified
                merged_config = self.sidecar_manager.get_merged_config(json_sidecar_path, gui_config, override_keys)
                logger.debug("Preview: Sidecar found. Merged config applied.")
            else:
                # Sidecar not found: GUI is the base (uses the initial gui_config)
                merged_config = self.sidecar_manager._get_defaults()
                merged_config.update(gui_config) # Override defaults with current GUI settings
                logger.debug("Preview: Sidecar NOT found. Using GUI values.")
        else:
            merged_config = self.sidecar_manager._get_defaults()
            merged_config.update(gui_config)
            logger.debug("Preview: No depth map path available from previewer's list. Using GUI values.") # <--- MODIFIED LOG
        # --- END MODIFIED BLOCK ---


        # Set final parameters from the merged config
        params['convergence_point'] = merged_config["convergence_plane"]
        params['max_disp'] = merged_config["max_disparity"]
        params['depth_gamma'] = merged_config["gamma"]

        # Determine the final source for display info
        if sidecar_path_found:
            conv_source = "GUI" if "convergence_plane" in override_keys else "Sidecar"
            max_disp_source = "GUI" if "max_disparity" in override_keys else "Sidecar"
            gamma_source = "GUI" if "gamma" in override_keys else "Sidecar"
        else:
            conv_source = "GUI"
            max_disp_source = "GUI"
            gamma_source = "GUI"
        # ----------------------------------------------------------------------
        # END NEW SIDECAR LOGIC FOR PREVIEW
        # ----------------------------------------------------------------------
        
        W_target, H_target = W_orig, H_orig
        
        if is_low_res_preview:
            try:
                W_target_requested = int(self.pre_res_width_var.get())
                
                if W_target_requested <= 0:
                    W_target_requested = W_orig # Fallback
                
                # 1. Calculate aspect-ratio-correct height based on the requested width
                aspect_ratio = W_orig / H_orig
                H_target_calculated = int(round(W_target_requested / aspect_ratio))
                
                # 2. Ensure both W and H are divisible by 2 for codec compatibility
                W_target = W_target_requested if W_target_requested % 2 == 0 else W_target_requested + 1
                H_target = H_target_calculated if H_target_calculated % 2 == 0 else H_target_calculated + 1
                
                # 3. Handle potential extreme fallbacks
                if W_target <= 0 or H_target <= 0:
                    W_target, H_target = W_orig, H_orig
                    logger.warning("Low-Res preview: Calculated dimensions invalid, falling back to original.")
                else:
                    logger.debug(f"Low-Res preview: AR corrected target {W_target}x{H_target}. (Original W: {W_orig}, H: {H_orig})")
                
                # Resize Left Eye to aspect-ratio-correct low-res target for consistency
                left_eye_tensor_resized = F.interpolate(
                    left_eye_tensor.cuda(), 
                    size=(H_target, W_target), 
                    mode='bilinear', 
                    align_corners=False
                )
            except Exception as e:
                logger.error(f"Low-Res preview failed during AR calculation/resize: {e}. Falling back to original res.", exc_info=True)
                W_target, H_target = W_orig, H_orig
                left_eye_tensor_resized = left_eye_tensor.cuda()
        else:
            left_eye_tensor_resized = left_eye_tensor.cuda() # Use original res

        
        logger.debug(f"Preview Params: {params}")
        logger.debug(f"Target Resolution: {W_target}x{H_target} (Low-Res: {is_low_res_preview})")


        # --- Process Depth Frame ---
        depth_numpy_raw = depth_tensor_raw.squeeze(0).permute(1, 2, 0).cpu().numpy()
        logger.debug(f"Raw depth numpy shape: {depth_numpy_raw.shape}, range: [{depth_numpy_raw.min():.2f}, {depth_numpy_raw.max():.2f}]")
        
        # 1. DETERMINE MAX CONTENT VALUE FOR THE FRAME (for AutoGain scaling)
        # We need the max *raw* value of the depth frame content
        max_raw_content_value = depth_numpy_raw.max()
        if max_raw_content_value < 1.0: 
            max_raw_content_value = 1.0 # Fallback for already 0-1 normalized content

        # Determine the scaling factor
        final_scaling_factor = 1.0
        if params['enable_autogain']:
            if max_raw_content_value <= 256.0 and max_raw_content_value > 1.0:
                final_scaling_factor = 255.0
            elif max_raw_content_value > 256.0 and max_raw_content_value <= 1024.0:
                final_scaling_factor = max_raw_content_value
            elif max_raw_content_value > 1024.0:
                final_scaling_factor = 65535.0
            else:
                final_scaling_factor = 1.0 
        else:
            final_scaling_factor = 1.0 # Normalization enabled path doesn't need scaling here

        depth_numpy_processed = self._process_depth_batch(
            batch_depth_numpy_raw=np.expand_dims(depth_numpy_raw, axis=0),
            depth_stream_info=None,
            depth_gamma=params['depth_gamma'],
            depth_dilate_size_x=params['depth_dilate_size_x'],
            depth_dilate_size_y=params['depth_dilate_size_y'],
            depth_blur_size_x=params['depth_blur_size_x'],
            depth_blur_size_y=params['depth_blur_size_y'],
            is_low_res_task=is_low_res_preview, # <-- USE LOW-RES FLAG HERE
            max_raw_value=final_scaling_factor,
            global_depth_min=0.0,
            global_depth_max=1.0 
        )
        logger.debug(f"Processed depth numpy shape: {depth_numpy_processed.shape}, range: [{depth_numpy_processed.min():.2f}, {depth_numpy_processed.max():.2f}]")

        # 2. Normalize based on the 'enable_autogain' (Disable Normalization) setting
        depth_normalized = depth_numpy_processed.squeeze(0)

        if params['enable_autogain']:
            # RAW INPUT MODE: Normalize by the determined scaling factor
            depth_normalized = depth_normalized / final_scaling_factor
            logger.debug(f"Preview: Applied raw scaling by {final_scaling_factor:.2f}")
        else:
            # NORMALIZATION ENABLED: Perform min/max normalization on the processed result
            min_val, max_val = depth_numpy_processed.min(), depth_numpy_processed.max()
            if max_val > min_val:
                depth_normalized = (depth_numpy_processed.squeeze(0) - min_val) / (max_val - min_val)
            else:
                depth_normalized = np.zeros_like(depth_numpy_processed.squeeze(0))
            
            # Apply gamma AFTER normalization for the Global Norm path (as skipped in helper)
            if params['depth_gamma'] != 1.0:
                depth_normalized = np.power(depth_normalized, params['depth_gamma'])
                logger.debug(f"Applied gamma ({params['depth_gamma']}) post-normalization.")

        depth_normalized = np.clip(depth_normalized, 0, 1)
        logger.debug(f"Final normalized depth shape: {depth_normalized.shape}, range: [{depth_normalized.min():.2f}, {depth_normalized.max():.2f}]")

        # --- Perform Splatting ---
        stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()
        # Ensure depth map is resized to the target resolution (low-res or original)
        disp_map_tensor = torch.from_numpy(depth_normalized).unsqueeze(0).unsqueeze(0).float().cuda()
        
        # Resize Disparity Map to match the (potentially resized) Left Eye
        if H_target != disp_map_tensor.shape[2] or W_target != disp_map_tensor.shape[3]:
             logger.debug(f"Resizing depth map to match target {W_target}x{H_target}.")
             disp_map_tensor = F.interpolate(disp_map_tensor, size=(H_target, W_target), mode='bilinear', align_corners=False)

        disp_map_tensor = (disp_map_tensor - params['convergence_point']) * 2.0
        
        # Calculate disparity in pixels based on the TARGET width (W_target)
        actual_max_disp_pixels = (params['max_disp'] / 20.0 / 100.0) * W_target
        disp_map_tensor = disp_map_tensor * actual_max_disp_pixels


        with torch.no_grad():
            # Use the potentially resized Left Eye
            right_eye_tensor_raw, occlusion_mask = stereo_projector(left_eye_tensor_resized, disp_map_tensor)
            
            # Apply low-res specific post-processing
            if is_low_res_preview:
                # 1. Fill Left Edge Occlusions
                right_eye_tensor_left_filled = self._fill_left_edge_occlusions(right_eye_tensor_raw, occlusion_mask, boundary_width_pixels=3)
                
                # 2. Fill Right Edge Occlusions (New Call)
                right_eye_tensor = self._fill_right_edge_occlusions(right_eye_tensor_left_filled, occlusion_mask, boundary_width_pixels=3)
            else:
                right_eye_tensor = right_eye_tensor_raw


        # --- NEW: Update Info Frame for Preview (using Target resolution) ---
        current_source_dict = getattr(self.previewer, 'current_source', {})
        current_video_path = current_source_dict.get('source_video')
            
        video_filename = os.path.basename(current_video_path) if current_video_path else "N/A"
        
        # 2. Frames: Get total frames from metadata (assuming key 'total_frames' or similar)
        preview_metadata = getattr(self.previewer, 'metadata', {})
        total_frames = preview_metadata.get('total_frames')
        frames_display = f"1/{total_frames}" if total_frames else "1 (Preview)"

        self.processing_filename_var.set(video_filename)
        self.processing_task_name_var.set("Preview" + (" (Low-Res)" if is_low_res_preview else ""))
        self.processing_resolution_var.set(f"{W_target}x{H_target}")
        self.processing_frames_var.set(frames_display) 
        self.processing_disparity_var.set(f"{params['max_disp']:.1f}% ({actual_max_disp_pixels:.2f} pixels) ({max_disp_source})")
        self.processing_convergence_var.set(f"{params['convergence_point']:.2f} ({conv_source})")
        self.processing_gamma_var.set(f"{params['depth_gamma']:.2f} ({gamma_source})")
        # --- END NEW: Update Info Frame for Preview ---


        # --- Select Output for Display ---
        self.previewer.set_preview_source_options([
            "Splat Result",
            "Splat Result(Low)",
            "Occlusion Mask",
            "Occlusion Mask(Low)",
            "Original (Left Eye)",
            "Depth Map", 
            "Anaglyph 3D", 
            "Wigglegram",
        ])

        if preview_source == "Splat Result" or preview_source == "Splat Result(Low)":
            final_tensor = right_eye_tensor.cpu()
        elif preview_source == "Occlusion Mask" or preview_source == "Occlusion Mask(Low)":
            final_tensor = occlusion_mask.repeat(1, 3, 1, 1).cpu()
        elif preview_source == "Depth Map":
            depth_vis_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            depth_vis_rgb = cv2.cvtColor(depth_vis_colored, cv2.COLOR_BGR2RGB)
            final_tensor = torch.from_numpy(depth_vis_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        elif preview_source == "Original (Left Eye)":
            # Use the resized or original left eye depending on the low-res flag
            final_tensor = left_eye_tensor_resized.cpu()
        elif preview_source == "Anaglyph 3D":
            left_np_anaglyph = (left_eye_tensor_resized.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            right_np_anaglyph = (right_eye_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            left_gray_np = cv2.cvtColor(left_np_anaglyph, cv2.COLOR_RGB2GRAY)
            anaglyph_np = right_np_anaglyph.copy()
            anaglyph_np[:, :, 0] = left_gray_np
            final_tensor = (torch.from_numpy(anaglyph_np).permute(2, 0, 1).float() / 255.0).unsqueeze(0)
        elif preview_source == "Wigglegram":
            # Pass the resized left eye and the splatted right eye
            self.previewer._start_wigglegram_animation(left_eye_tensor_resized.cpu(), right_eye_tensor.cpu())
            return None
        else:
            final_tensor = right_eye_tensor.cpu()

        pil_img = Image.fromarray((final_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8))

        del stereo_projector, disp_map_tensor, right_eye_tensor_raw, occlusion_mask
        release_cuda_memory()
        logger.debug("--- Finished Preview Processing Callback ---")
        return pil_img

    def reset_to_defaults(self):
        """Resets all GUI parameters to their default hardcoded values."""
        if not messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to their default values?"):
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
        self.enable_autogain_var.set(False) # Default: Global Depth Normalization
        self.zero_disparity_anchor_var.set("0.5")
        self.output_crf_var.set("23")        
        self.auto_convergence_mode_var.set("Off")
        
        self.toggle_processing_settings_fields()
        self._save_config()
        self.clear_processing_info()
        self.status_label.config(text="Settings reset to defaults.")

    def restore_finished_files(self):
        """Moves all files from 'finished' folders back to their original input folders."""
        if not messagebox.askyesno("Restore Finished Files", "Are you sure you want to move all files from 'finished' folders back to their input directories?"):
            return

        source_clip_dir = self.input_source_clips_var.get()
        depth_map_dir = self.input_depth_maps_var.get()

        is_source_dir = os.path.isdir(source_clip_dir)
        is_depth_dir = os.path.isdir(depth_map_dir)

        if not (is_source_dir and is_depth_dir):
            messagebox.showerror("Restore Error", "Restore 'finished' operation is only applicable when Input Source Clips and Input Depth Maps are set to directories (batch mode). Please ensure current settings reflect this.")
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
            logger.info(f"==> Finished source folder not found: {finished_source_folder}")

        if os.path.isdir(finished_depth_folder):
            logger.info(f"==> Restoring depth maps and sidecars from: {finished_depth_folder}")
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
                        logger.error(f"Error moving depth map/sidecar '{filename}': {e}")
        else:
            logger.info(f"==> Finished depth folder not found: {finished_depth_folder}")

        if restored_count > 0 or errors_count > 0:
            self.clear_processing_info()
            self.status_label.config(text=f"Restore complete: {restored_count} files moved, {errors_count} errors.")
            messagebox.showinfo("Restore Complete", f"Finished files restoration attempted.\n{restored_count} files moved.\n{errors_count} errors occurred.")
        else:
            self.clear_processing_info()
            self.status_label.config(text="No files found to restore.")
            messagebox.showinfo("Restore Complete", "No files found in 'finished' folders to restore.")

    def _round_slider_variable_value(self, tk_var: tk.Variable, decimals: int):
        """Rounds the float/string value of a tk.Variable and sets it back."""
        try:
            current_value = float(tk_var.get())
            rounded_value = round(current_value, decimals)
            if current_value != rounded_value:
                tk_var.set(rounded_value)
                logger.debug(f"Rounded {current_value} to {rounded_value} (decimals={decimals})")
        except ValueError:
            pass

    def _run_batch_process(self, settings):
        """
        The main batch processing orchestrator, run in a separate thread.
        It manages the overall flow, progress bar, and calls _process_single_video_tasks.
        """
        self.after(0, self.clear_processing_info) # Clear info display at start

        try:
            # --- 1. Setup and Initial Checks ---
            input_videos, is_single_file_mode, finished_source_folder, finished_depth_folder = self._setup_batch_processing(settings)

            if not input_videos:
                logger.error("No video files found or invalid input setup. Processing stopped.")
                self.progress_queue.put("finished")
                return

            # --- 2. Determine total tasks for progress bar ---
            total_processing_tasks_count = 0
            tasks_per_video = len(self._get_defined_tasks(settings))
            
            for video_path in input_videos:
                video_specific_settings_check = self._get_video_specific_settings(
                    video_path,
                    settings["input_depth_maps"],
                    settings["zero_disparity_anchor"],
                    settings["max_disp"],
                    is_single_file_mode
                )
                if not video_specific_settings_check.get("error"):
                    total_processing_tasks_count += tasks_per_video

            if total_processing_tasks_count == 0:
                logger.error("==> Error: No resolution output enabled or no valid video/depth pairs found. Processing stopped.")
                self.progress_queue.put("finished")
                return

            self.progress_queue.put(("total", total_processing_tasks_count))
            overall_task_counter = 0

            # --- 3. Main Processing Loop ---
            for idx, video_path in enumerate(input_videos):
                if self.stop_event.is_set():
                    logger.info("==> Stopping processing due to user request")
                    break

                # Call the new function for single-video processing
                tasks_processed, _ = self._process_single_video_tasks(
                    video_path=video_path, 
                    settings=settings, 
                    initial_overall_task_counter=overall_task_counter, 
                    is_single_file_mode=is_single_file_mode,
                    # Pass the finished folders (only non-None in batch mode)
                    finished_source_folder=finished_source_folder, 
                    finished_depth_folder=finished_depth_folder
                )
                
                # Update the overall counter based on the tasks completed in the helper
                overall_task_counter += tasks_processed

            # --- 4. Final Cleanup ---
        except Exception as e:
            logger.error(f"An unexpected error occurred during batch processing: {e}", exc_info=True)
            self.progress_queue.put(("status", f"Error: {e}"))
            self.after(0, lambda: messagebox.showerror("Processing Error", f"An unexpected error occurred during batch processing: {e}"))
        finally:
            release_cuda_memory()
            self.progress_queue.put("finished")
            self.after(0, self.clear_processing_info)

    def run_fusion_sidecar_generator(self):
        """Initializes and runs the FusionSidecarGenerator tool."""
        # Use an external thread to prevent the GUI from freezing during the file scan
        def worker():
            self.status_label.config(text="Starting Fusion Export Sidecar Generation...")
            generator = FusionSidecarGenerator(self, self.sidecar_manager)
            generator.generate_sidecars()
            
        threading.Thread(target=worker, daemon=True).start()

    def run_preview_auto_converge(self, force_run=False):
        """
        Starts the Auto-Convergence pre-pass on the current preview clip in a thread,
        and updates the convergence slider/preview upon completion.
        'force_run=True' is used when triggered by the combo box, as validation is needed.
        """
        if not hasattr(self, 'previewer') or not self.previewer.source_readers:
            if force_run:
                messagebox.showwarning("Auto-Converge Preview", "Please load a video in the Previewer first.")
                self.auto_convergence_combo.set("Off") # Reset combo on fail
            return

        current_index = self.previewer.current_video_index
        if current_index == -1:
            if force_run:
                messagebox.showwarning("Auto-Converge Preview", "No video is currently selected for processing.")
                self.auto_convergence_combo.set("Off") # Reset combo on fail
            return
        
        mode = self.auto_convergence_mode_var.get()
        if mode == "Off":
            if force_run: # This should be caught by the cache check, but as a safeguard
                return
            messagebox.showwarning("Auto-Converge Preview", "Auto-Convergence Mode must be set to 'Average' or 'Peak'.")
            return
            
        current_source_dict = self.previewer.video_list[current_index]
        single_video_path = current_source_dict.get('source_video')
        single_depth_path = current_source_dict.get('depth_map')

        # --- NEW: Check if calculation is already done for a different mode/path ---
        is_path_mismatch = (single_depth_path != self._auto_conv_cached_path)
        is_cache_complete = (self._auto_conv_cache["Average"] is not None) or (self._auto_conv_cache["Peak"] is not None)
        
        # If running from the combo box (force_run=True) AND the cache is incomplete 
        # BUT the path has changed, we must clear the cache and run.
        if force_run and is_path_mismatch and is_cache_complete:
            logger.info("New video detected. Clearing Auto-Converge cache.")
            self._auto_conv_cache = {"Average": None, "Peak": None}
            self._auto_conv_cached_path = None

        if not single_video_path or not single_depth_path:
            messagebox.showerror("Auto-Converge Preview Error", "Could not get both video and depth map paths from previewer.")
            if force_run: self.auto_convergence_combo.set("Off")
            return
        
        try:
            current_anchor = float(self.zero_disparity_anchor_var.get())
            process_length = int(self.process_length_var.get())
            batch_size = int(self.batch_size_var.get())
        except ValueError as e:
            messagebox.showerror("Auto-Converge Preview Error", f"Invalid input for slider or process length: {e}")
            if force_run: self.auto_convergence_combo.set("Off")
            return
            
        # Set running flag and disable inputs
        self._is_auto_conv_running = True
        self.btn_auto_converge_preview.config(state="disabled")
        self.start_button.config(state="disabled")
        self.start_single_button.config(state="disabled")
        self.auto_convergence_combo.config(state="disabled") # Disable combo during run

        self.status_label.config(text=f"Auto-Convergence pre-pass started ({mode} mode)...")
        
        # Start the calculation in a new thread
        worker_args = (single_depth_path, process_length, batch_size, current_anchor, mode)
        self.auto_converge_thread = threading.Thread(target=self._auto_converge_worker, args=worker_args)
        self.auto_converge_thread.start()

    def _save_config(self):
        """Saves current GUI settings to config_splat.json."""
        config = self._get_current_config()
        with open("config_splat.json", "w") as f:
            json.dump(config, f, indent=4)
   
    def _save_debug_image(self, data: np.ndarray, filename_tag: str, batch_index: int, frame_index: int, task_name: str):
        """Saves a normalized (0-1) NumPy array as a grayscale PNG to a debug folder."""
        if not self._debug_logging_enabled:
            return

        debug_dir = os.path.join(os.path.dirname(self.input_source_clips_var.get()), "splat_debug", task_name, "images")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create a filename that includes frame index, batch index, and tag
        filename = os.path.join(debug_dir, f"{frame_index:05d}_B{batch_index:02d}_{filename_tag}.png")
        
        try:
            # 1. Normalize data to 0-255 uint8 range for PIL
            # If data is BxHxW, take the first frame (index 0)
            if data.ndim == 3:
                frame_np = data[0]
            elif data.ndim == 4:
                frame_np = data[0].squeeze() # Assuming Bx1xHxW or similar
            else:
                frame_np = data # Assume HxW
                
            # 2. Ensure data is float 0-1 (if not already) and clip
            if frame_np.dtype != np.float32:
                 # Assume raw values (e.g., 0-255) and normalize for visualization
                 frame_np = frame_np.astype(np.float32) / frame_np.max() if frame_np.max() > 0 else frame_np
            
            frame_uint8 = (np.clip(frame_np, 0.0, 1.0) * 255).astype(np.uint8)
            
            # 3. Save as Grayscale PNG
            img = Image.fromarray(frame_uint8, mode='L')
            img.save(filename)
            
            logger.debug(f"Saved debug image {filename_tag} (Shape: {frame_uint8.shape}) to {os.path.basename(debug_dir)}")
        except Exception as e:
            logger.error(f"Failed to save debug image {filename_tag}: {e}")

    def _save_debug_numpy(self, data: np.ndarray, filename_tag: str, batch_index: int, frame_index: int, task_name: str):
        """Saves a NumPy array to a debug folder if debug logging is enabled."""
        if not self._debug_logging_enabled:
            return

        output_path = self.output_splatted_var.get()
        debug_root = os.path.join(os.path.dirname(output_path), "splat_debug")

        # 1. Save NPZ (Existing Logic)
        debug_dir_npz = os.path.join(debug_root, task_name)
        os.makedirs(debug_dir_npz, exist_ok=True)
        filename_npz = os.path.join(debug_dir_npz, f"{frame_index:05d}_B{batch_index:02d}_{filename_tag}.npz")
        logger.debug(f"Save path {filename_tag}")


        try:
            np.savez_compressed(filename_npz, data=data)
            logger.debug(f"Saved debug array {filename_tag} (Shape: {data.shape}) to {os.path.basename(debug_dir_npz)}")
        except Exception as e:
            logger.error(f"Failed to save debug array {filename_tag}: {e}")

        # 2. Save PNG Image (New Logic)
        self._save_debug_image(data, filename_tag, batch_index, frame_index, task_name)

    def save_settings(self):
        """Saves current GUI settings to a user-selected JSON file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Settings to File"
        )
        if not filename:
            return

        try:
            config_to_save = self._get_current_config()
            with open(filename, "w") as f:
                json.dump(config_to_save, f, indent=4)

            messagebox.showinfo("Settings Saved", f"Successfully saved settings to:\n{os.path.basename(filename)}")
            self.status_label.config(text="Settings saved.")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save settings to {os.path.basename(filename)}:\n{e}")
            self.status_label.config(text="Settings save failed.")
        
    def _set_input_state(self, state):
        """Sets the state of all input widgets to 'normal' or 'disabled'."""
        
        # Helper to set the state of all children in a frame
        def set_frame_children_state(frame, state, exclude_frames=False):
            """Recursively sets the state of all configurable widgets within a frame."""
            for child in frame.winfo_children():
                child_type = child.winfo_class()
                
                # Check if the child is a Frame/LabelFrame that we need to recurse into
                if isinstance(child, (ttk.Frame, tk.Frame, ttk.LabelFrame)) and not exclude_frames:
                    set_frame_children_state(child, state, exclude_frames)
                
                # Check for widgets that accept the 'state' configuration
                if child_type in ('TEntry', 'TButton', 'TCheckbutton', 'TCombobox'):
                    try:
                        # Use a keyword argument to pass the state
                        child.config(state=state)
                    except tk.TclError as e:
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

        if hasattr(self, 'update_sidecar_button'):
            if state == 'disabled':
                self.update_sidecar_button.config(state="disabled")
            else: # state == 'normal'
                # When batch is done, re-apply the sidecar override logic immediately
                self._toggle_sidecar_update_button_state()

        # 3. Re-apply the specific field enable/disable logic
        # This is CRITICAL. If we set state='normal' for everything, 
        # toggle_processing_settings_fields will correctly re-disable the Low Res W/H fields
        # if the "Enable Low Resolution" checkbox is unchecked.
        if hasattr(self, 'previewer'):
            self.previewer.set_ui_processing_state(state == 'disabled')

        if state == 'normal':
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
            if calculated_height < 100: calculated_height = 750
            current_height = calculated_height
        else:
            current_height = saved_height
        # --- END MODIFIED ---

        # Fallback if saved width is invalid or too small
        if current_width < 200: # Minimum sensible width
            current_width = 620 # Use default width

        # 2. Construct the geometry string
        geometry_string = f"{current_width}x{current_height}"
        if self.window_x is not None and self.window_y is not None:
            geometry_string += f"+{self.window_x}+{self.window_y}"
        else:
            # If no saved position, let Tkinter center it initially or place it at default
            pass # No position appended, Tkinter will handle default placement

        # 3. Apply the geometry
        self.geometry(geometry_string)
        logger.debug(f"Applied saved geometry: {geometry_string}")

        # Store the actual width that was applied (which is current_width) for save_config
        self.window_width = current_width # Update instance variable for save_config

    def _setup_batch_processing(self, settings):
        """
        Handles input path validation, mode determination (single file vs batch),
        and creates necessary 'finished' folders.
        Returns: input_videos (list), is_single_file_mode (bool),
                 finished_source_folder (str/None), finished_depth_folder (str/None)
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
            is_single_file_mode = True
            logger.debug("==> Running in single file mode. Files will not be moved to 'finished' folders (unless specifically enabled in Single Process mode).")
            input_videos.append(input_source_clips_path)
            os.makedirs(output_splatted, exist_ok=True)

        elif is_source_dir and is_depth_dir:
            logger.debug("==> Running in batch (folder) mode.")

            if self.MOVE_TO_FINISHED_ENABLED:
                finished_source_folder = os.path.join(input_source_clips_path, "finished")
                finished_depth_folder = os.path.join(input_depth_maps_path, "finished")
                os.makedirs(finished_source_folder, exist_ok=True)
                os.makedirs(finished_depth_folder, exist_ok=True)
                logger.debug("Finished folders enabled for batch mode.")
            else:
                logger.debug("Finished folders DISABLED globally. Files will remain in input folders.")

            os.makedirs(output_splatted, exist_ok=True)

            video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
            for ext in video_extensions:
                input_videos.extend(glob.glob(os.path.join(input_source_clips_path, ext)))
            input_videos = sorted(input_videos)
        else:
            logger.error("==> Error: Input Source Clips and Input Depth Maps must both be either files or directories. Skipping processing.")
            return [], False, None, None

        if not input_videos:
            logger.error(f"No video files found in {input_source_clips_path}")
            return [], False, None, None

        return input_videos, is_single_file_mode, finished_source_folder, finished_depth_folder
    
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
        guide_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "merger_gui_guide.md")
        try:
            with open(guide_path, "r", encoding="utf-8") as f:
                guide_content = f.read()
        except FileNotFoundError:
            messagebox.showerror("File Not Found", f"The user guide file could not be found at:\n{guide_path}")
            return
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while reading the user guide:\n{e}")
            return

        # Determine colors based on current theme
        if self.dark_mode_var.get():
            bg_color, fg_color = "#2b2b2b", "white"
        else:
            # Use a standard light bg for text that's slightly different from the main window
            bg_color, fg_color = "#fdfdfd", "black"

        # Create a new Toplevel window
        guide_window = tk.Toplevel(self)
        guide_window.title("SplatterGUI - User Guide") # Corrected title
        guide_window.geometry("600x700")
        guide_window.transient(self) # Keep it on top of the main window
        guide_window.grab_set()      # Modal behavior
        guide_window.configure(bg=bg_color)

        text_frame = ttk.Frame(guide_window, padding="10")
        text_frame.configure(style="TFrame") # Ensure it follows the theme
        text_frame.pack(expand=True, fill="both")

        # Apply theme colors to the Text widget
        text_widget = tk.Text(text_frame, wrap=tk.WORD, relief="flat", borderwidth=0, padx=5, pady=5, font=("Segoe UI", 9),
                              bg=bg_color, fg=fg_color, insertbackground=fg_color)
        text_widget.insert(tk.END, guide_content)
        text_widget.config(state=tk.DISABLED) # Make it read-only

        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget['yscrollcommand'] = scrollbar.set

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(side=tk.LEFT, expand=True, fill="both")

        button_frame = ttk.Frame(guide_window, padding=(0, 0, 0, 10))
        button_frame.pack()
        ok_button = ttk.Button(button_frame, text="Close", command=guide_window.destroy)
        ok_button.pack(pady=10)

    def start_processing(self):
        """Starts the video processing in a separate thread."""
        self.stop_event.clear()
        self.start_button.config(state="disabled")
        self.start_single_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Starting processing...")
        # --- NEW: Disable all inputs at start ---
        self._set_input_state('disabled')
        # --- NEW: Disable previewer widgets ---
        if hasattr(self, 'previewer'):
            self.previewer.set_ui_processing_state(True)
            self.previewer.cleanup() # Release any loaded preview videos

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
                    raise ValueError("Low-Resolution Width and Height must be positive.")
                low_res_batch_size_val = int(self.low_res_batch_size_var.get())
                if low_res_batch_size_val <= 0:
                    raise ValueError("Low-Resolution Batch Size must be positive.")

            if not (self.enable_full_res_var.get() or self.enable_low_res_var.get()):
                raise ValueError("At least one resolution (Full or Low) must be enabled to start processing.")
            
            # --- NEW: Depth Pre-processing Validation ---
            depth_gamma_val = float(self.depth_gamma_var.get())
            if depth_gamma_val <= 0:
                raise ValueError("Depth Gamma must be positive.")
            
            # Validate Dilate X/Y
            depth_dilate_size_x_val = int(float(self.depth_dilate_size_x_var.get()))
            depth_dilate_size_y_val = int(float(self.depth_dilate_size_y_var.get()))
            if depth_dilate_size_x_val < 0 or depth_dilate_size_y_val < 0:
                raise ValueError("Depth Dilate Sizes (X/Y) must be non-negative.")
            
            # Validate Blur X/Y
            depth_blur_size_x_val = int(float(self.depth_blur_size_x_var.get()))
            depth_blur_size_y_val = int(float(self.depth_blur_size_y_var.get()))
            if depth_blur_size_x_val < 0 or depth_blur_size_y_val < 0:
                raise ValueError("Depth Blur Sizes (X/Y) must be non-negative.")


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
            "enable_autogain": self.enable_autogain_var.get(),
            "match_depth_res": True,
            "output_crf": int(self.output_crf_var.get()),
            # --- Depth Pre-processing & Auto-Convergence Settings ---
            "depth_gamma": depth_gamma_val,
            "depth_dilate_size_x": depth_dilate_size_x_val,
            "depth_dilate_size_y": depth_dilate_size_y_val,
            "depth_blur_size_x": depth_blur_size_x_val,
            "depth_blur_size_y": depth_blur_size_y_val,
            "auto_convergence_mode": self.auto_convergence_mode_var.get(),
            "enable_sidecar_gamma": self.enable_sidecar_gamma_var.get(),
            "enable_sidecar_blur_dilate": self.enable_sidecar_blur_dilate_var.get(),
        }
        self.processing_thread = threading.Thread(target=self._run_batch_process, args=(settings,))
        self.processing_thread.start()
        self.check_queue()

    def start_single_processing(self):
        """
        Starts processing for the single video currently loaded in the previewer.
        It runs the batch logic in single-file mode.
        """
        if not hasattr(self, 'previewer') or not self.previewer.source_readers:
            messagebox.showwarning("Process Single Clip", "Please load a video in the Previewer first.")
            return

        current_index = self.previewer.current_video_index
        if current_index == -1:
            messagebox.showwarning("Process Single Clip", "No video is currently selected for processing.")
            return
            
        # 1. Get the current single file paths
        current_source_dict = self.previewer.video_list[current_index]
        single_video_path = current_source_dict.get('source_video')
        single_depth_path = current_source_dict.get('depth_map')

        if not single_video_path or not single_depth_path:
            messagebox.showerror("Process Single Clip Error", "Could not get both video and depth map paths from previewer.")
            return

        # 2. Perform validation checks (copied from start_processing)
        try:
            # Full Resolution/Low Resolution checks
            if not (self.enable_full_res_var.get() or self.enable_low_res_var.get()):
                raise ValueError("At least one resolution (Full or Low) must be enabled to start processing.")
            
            # Simplified validation for speed/simplicity (relying on start_processing for full checks)
            float(self.max_disp_var.get())
            
        except ValueError as e:
            self.status_label.config(text=f"Error: {e}")
            messagebox.showerror("Validation Error", str(e))
            return
        
        if hasattr(self, 'previewer'):
            self.previewer.cleanup()
            
        # 3. Compile settings dictionary
        # We explicitly set the input paths to the single files, which forces batch logic 
        # to execute in single-file mode (checking os.path.isfile).
        
        # --- NEW: Determine Finished Folders for Single Process (only if enabled) ---
        single_finished_source_folder = None
        single_finished_depth_folder = None
        
        if self.MOVE_TO_FINISHED_ENABLED:
            # We assume the finished folder is in the same directory as the original input file/depth map
            single_finished_source_folder = os.path.join(os.path.dirname(single_video_path), "finished")
            single_finished_depth_folder = os.path.join(os.path.dirname(single_depth_path), "finished")
            os.makedirs(single_finished_source_folder, exist_ok=True)
            os.makedirs(single_finished_depth_folder, exist_ok=True)
            logger.debug(f"Single Process: Finished folders set to: {single_finished_source_folder}")

        settings = {
            # --- OVERRIDDEN INPUTS FOR SINGLE MODE ---
            "input_source_clips": single_video_path,
            "input_depth_maps": single_depth_path,
            "output_splatted": self.output_splatted_var.get(), # Use the batch output folder
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
            "enable_autogain": self.enable_autogain_var.get(),
            "match_depth_res": True,
            "output_crf": int(self.output_crf_var.get()),
            
            # --- Depth Pre-processing Settings ---
            "depth_gamma": float(self.depth_gamma_var.get()),
            "depth_dilate_size_x": int(float(self.depth_dilate_size_x_var.get())),
            "depth_dilate_size_y": int(float(self.depth_dilate_size_y_var.get())),
            "depth_blur_size_x": int(float(self.depth_blur_size_x_var.get())),
            "depth_blur_size_y": int(float(self.depth_blur_size_y_var.get())),
            "auto_convergence_mode": self.auto_convergence_mode_var.get(),
            "enable_sidecar_gamma": self.enable_sidecar_gamma_var.get(),
            "enable_sidecar_blur_dilate": self.enable_sidecar_blur_dilate_var.get(),
            "single_finished_source_folder": single_finished_source_folder,
            "single_finished_depth_folder": single_finished_depth_folder,
        }

        # 4. Start the processing thread
        self.stop_event.clear()
        self.start_button.config(state="disabled")
        self.start_single_button.config(state="disabled") # Disable single button too
        self.stop_button.config(state="normal")
        self.status_label.config(text=f"Starting single-clip processing for: {os.path.basename(single_video_path)}")
        self._set_input_state('disabled') # Disable all inputs

        self.processing_thread = threading.Thread(target=self._run_batch_process, args=(settings,))
        self.processing_thread.start()
        self.check_queue()

    def stop_processing(self):
        """Sets the stop event to gracefully halt processing."""
        self.stop_event.set()
        self.status_label.config(text="Stopping...")
        self.stop_button.config(state="disabled")
        self.start_single_button.config(state="normal")
        # --- NEW: Re-enable previewer widgets on stop ---
        if hasattr(self, 'previewer'):
            self.previewer.set_ui_processing_state(False)

    def _toggle_debug_logging(self):
        """Toggles debug logging and updates shared logger."""
        self._debug_logging_enabled = self.debug_logging_var.get() # Get checkbutton state
        
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

    def _toggle_sidecar_update_button_state(self):
        """
        Controls the Update Sidecar button state based on the Override Sidecar checkbox.
        """
        
        # Check if batch processing is currently active (easiest way is to check the stop button's state)
        is_batch_processing_active = (self.stop_button.cget("state") == "normal")
        
        # If batch is active, the button MUST be disabled, regardless of override state.
        if is_batch_processing_active:
            self.update_sidecar_button.config(state="disabled")
            return
            
        # If batch is NOT active, base the state entirely on the Override checkbox.
        if self.override_sidecar_var.get():
            # Override is checked: GUI values are active, so ENABLE saving them.
            self.update_sidecar_button.config(state="normal")
        else:
            # Override is unchecked: Sidecar values are active, so DISABLE saving GUI values.
            self.update_sidecar_button.config(state="disabled")

    def _update_clip_state_and_text(self):
        """Combines state and text updates for the Sidecar button, run after a new video loads."""
        
        # 1. Update the button text (Create vs Update)
        if hasattr(self, '_update_sidecar_button_text'):
            self._update_sidecar_button_text()
            
        # 2. Update the button state (Normal vs Disabled by Override)
        if hasattr(self, '_toggle_sidecar_update_button_state'):
            self._toggle_sidecar_update_button_state()

    def _update_sidecar_button_text(self):
        """Checks if a sidecar exists for the current preview video and updates the button text."""
        is_sidecar_present = False
        
        if 0 <= self.previewer.current_video_index < len(self.previewer.video_list):
            current_source_dict = self.previewer.video_list[self.previewer.current_video_index]
            depth_map_path = current_source_dict.get('depth_map')

            if depth_map_path:
                depth_map_basename = os.path.splitext(os.path.basename(depth_map_path))[0]
                sidecar_ext = self.APP_CONFIG_DEFAULTS['SIDECAR_EXT']
                json_sidecar_path = os.path.join(os.path.dirname(depth_map_path), f"{depth_map_basename}{sidecar_ext}")
                
                is_sidecar_present = os.path.exists(json_sidecar_path)
        
        button_text = "Update Sidecar" if is_sidecar_present else "Create Sidecar"
        self.update_sidecar_button.config(text=button_text)
        
    def update_sidecar_file(self):
        """
        Updates/Creates the current video's sidecar file. 
        Prompts for confirmation only when overwriting an existing file.
        """
        # Ensure a video is loaded in the previewer
        if not hasattr(self, 'previewer') or not self.previewer.video_list or self.previewer.current_video_index == -1:
            messagebox.showwarning("Sidecar Action", "Please load a video in the Previewer first.")
            return

        # 1. Get current sidecar path
        current_source_dict = self.previewer.video_list[self.previewer.current_video_index]
        depth_map_path = current_source_dict.get('depth_map')
        if not depth_map_path:
            messagebox.showerror("Sidecar Error", "Could not determine the depth map path for the current video.")
            return

        depth_map_basename = os.path.splitext(os.path.basename(depth_map_path))[0]
        sidecar_ext = self.APP_CONFIG_DEFAULTS['SIDECAR_EXT']
        json_sidecar_path = os.path.join(os.path.dirname(depth_map_path), f"{depth_map_basename}{sidecar_ext}")
        
        is_sidecar_present = os.path.exists(json_sidecar_path)

        # 2. Conditional Confirmation Dialog
        if is_sidecar_present:
            title = "Overwrite Sidecar File?"
            message = (f"This will overwrite parameters (Convergence, Disparity, Gamma) "
                       f"in the existing sidecar file:\n\n{os.path.basename(json_sidecar_path)}\n\n"
                       f"Do you want to continue?")
            if not messagebox.askyesno(title, message):
                self.status_label.config(text="Sidecar update cancelled.")
                return
        # If the file does NOT exist, we skip the dialog and proceed.

        logger.debug(f"Attempting to {'update' if is_sidecar_present else 'create'} sidecar: {json_sidecar_path}")

        # 3. Get current GUI values (use raw strings for consistency)
        try:
            save_data = {
                "convergence_plane": float(self.zero_disparity_anchor_var.get()),
                "max_disparity": float(self.max_disp_var.get()),
                "gamma": float(self.depth_gamma_var.get()),
            }
        except ValueError as e:
            messagebox.showerror("Sidecar Error", f"Invalid input value in GUI: {e}")
            return
            
        # 4. Read existing sidecar content to preserve frame_overlap and input_bias
        current_data = {}
        if is_sidecar_present:
            try:
                # Use the manager to load current data (which includes overlap/bias)
                current_data = self.sidecar_manager.load_sidecar_data(json_sidecar_path)
            except Exception:
                current_data = {}
        
        # 5. Merge GUI values into current data
        current_data.update(save_data)
        
        # 6. Write the updated data back to the file using the manager
        if self.sidecar_manager.save_sidecar_data(json_sidecar_path, current_data):
            # 7. Success: Log to console and status bar (silent success)
            action = "updated" if is_sidecar_present else "created"
            logger.info(f"Sidecar '{os.path.basename(json_sidecar_path)}' successfully {action}.")
            self.status_label.config(text=f"Sidecar {action}.")
            
            # Update button text in case a file was just created
            self._update_sidecar_button_text()
            
            # Immediately refresh the preview to show the *effect* of the newly saved sidecar 
            self.on_slider_release(None) 
        else:
            # Failure
            messagebox.showerror("Sidecar Error", f"Failed to write sidecar file '{os.path.basename(json_sidecar_path)}'. Check logs.")

def compute_global_depth_stats(
        depth_map_reader: VideoReader,
        total_frames: int,
        chunk_size: int = 100
    ) -> Tuple[float, float]:
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
    ) -> Tuple[VideoReader,float, int, int, int, int, Optional[dict], int]:
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
        match_resolution_to_target: bool) -> Tuple[VideoReader, int, int, int, Optional[dict]]:
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

    app = SplatterGUI()
    app.mainloop()
