import threading
import gc
import os
import sys
import glob
import shutil
import json
import tkinter as tk
from tkinter import Toplevel, Label
from tkinter import filedialog, messagebox, ttk
import queue # Still needed for progress updates
import time
import numpy as np
import torch
import logging # Import standard logging
import random

# Configure a logger for this module
_logger = logging.getLogger(__name__)

# Import the backend logic class
from depthcrafter.depthcrafter_logic import DepthCrafterDemo

from depthcrafter.utils import (
    format_duration,
    get_segment_output_folder_name,
    get_segment_npz_output_filename,
    get_full_video_output_filename,
    get_sidecar_json_filename,
    get_image_sequence_metadata,
    get_single_image_metadata,
    define_video_segments,
    load_json_file,
    save_json_file,
    save_depth_visual_as_mp4_util,
    save_depth_visual_as_png_sequence_util,
    save_depth_visual_as_exr_sequence_util,
    save_depth_visual_as_single_exr_util,
)

try:
    from depthcrafter import merge_depth_segments
except ImportError as e:
    _logger.warning(f"Could not import 'merge_depth_segments'. Merging functionality will not be available. Error: {e}")
    merge_depth_segments = None

try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE_GUI = True
except ImportError:
    OPENEXR_AVAILABLE_GUI = False
    _logger.warning("OpenEXR or Imath module not found. EXR options might be limited.")


from typing import Optional, Tuple, List, Dict

try:
    from ttkthemes import ThemedTk
    THEMEDTK_AVAILABLE = True
except ImportError:
    THEMEDTK_AVAILABLE = False
    _logger.warning("ttkthemes not found. Dark mode functionality will be disabled.")
# --- Imports End ---

GUI_VERSION = "25-10-31.1"
_HELP_TEXTS = {}

DARK_MODE_COLORS = {
    "bg": "#2b2b2b",
    "fg": "white",
    "entry_bg": "#3c3c3c",
    "tooltip_bg": "#4a4a4a",
    "tooltip_fg": "white",
    "theme_name": "black", # A common ttkthemes dark theme
}
LIGHT_MODE_COLORS = {
    "bg": "#d9d9d9",
    "fg": "black",
    "entry_bg": "#ffffff",
    "tooltip_bg": "#ffffe0",
    "tooltip_fg": "black",
    "theme_name": "default", # A solid default theme
}
def _create_hover_tooltip(widget, help_key):
    """Creates a mouse-over tooltip for the given widget using text from _HELP_TEXTS."""
    if help_key in _HELP_TEXTS:
        Tooltip(widget, _HELP_TEXTS[help_key])
    else:
        _logger.warning(f"No help text found for key '{help_key}' to create tooltip for widget {widget}.")

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
        self.widget.bind("<ButtonPress>", self.hide_tooltip) # Hide on click

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return
        # Adjust position slightly for better visibility
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        # Find the main root window (it holds the app_instance attribute)
        root_window = self.widget._root() # A common tkinter internal way to get the root Tk object
        
        # Access the main DepthCrafterGUI instance via the root widget
        gui_instance = getattr(root_window, 'app_instance', None) 
        
        if gui_instance:
            bg_color = gui_instance.current_theme_colors["tooltip_bg"]
            fg_color = gui_instance.current_theme_colors["tooltip_fg"]
        else:
            # Fallback colors if instance couldn't be found
            bg_color = "#ffffe0"
            fg_color = "black"

        self.tooltip_window = Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True) # Remove window decorations
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = Label(self.tooltip_window, text=self.text, background="#ffffe0", relief="solid", borderwidth=1, justify="left", wraplength=250)
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None
        
class DepthCrafterGUI:
    CONFIG_FILENAME = "config_depthcrafter.json"
    HELP_CONTENT_FILENAME = os.path.join("depthcrafter", "help_content.json")
    MOVE_ORIGINAL_TO_FINISHED_FOLDER_ON_COMPLETION = True
    SETTINGS_FILETYPES = [("JSON files", "*.json"), ("All files", "*.*")]
    LAST_SETTINGS_DIR_CONFIG_KEY = "last_settings_dir"
    VIDEO_EXTENSIONS = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm", "*.flv", "*.gif"]
    IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.exr"]

    def __init__(self, root):
        self.root = root
        self.root.title(f"DepthCrafter GUI Seg {GUI_VERSION}")
        self.dark_mode_var = tk.BooleanVar(value=False)
        self.current_theme_colors = LIGHT_MODE_COLORS # Initialize theme colors dictionary
        self.input_dir_or_file_var = tk.StringVar(value=os.path.normpath("./input_clips"))
        self.output_dir = tk.StringVar(value=os.path.normpath("./output_depthmaps"))
        self.guidance_scale = tk.DoubleVar(value=1.0)
        self.inference_steps = tk.IntVar(value=5)
        self.seed = tk.IntVar(value=42)
        self.cpu_offload = tk.StringVar(value="model")
        self.use_cudnn_benchmark = tk.BooleanVar(value=False)
        self.process_length = tk.IntVar(value=-1)
        self.target_fps = tk.DoubleVar(value=-1.0)
        self.window_size = tk.IntVar(value=110)
        self.overlap = tk.IntVar(value=25)
        self.process_as_segments_var = tk.BooleanVar(value=False)
        self.save_final_output_json_var = tk.BooleanVar(value=False)
        self.merge_output_format_var = tk.StringVar(value="mp4")
        self.merge_alignment_method_var = tk.StringVar(value="Shift & Scale")
        self.merge_dither_var = tk.BooleanVar(value=False)
        self.merge_dither_strength_var = tk.DoubleVar(value=0.5)
        self.merge_gamma_correct_var = tk.BooleanVar(value=False)
        self.merge_gamma_value_var = tk.DoubleVar(value=1.5)
        self.merge_percentile_norm_var = tk.BooleanVar(value=False)
        self.merge_norm_low_perc_var = tk.DoubleVar(value=0.1)
        self.merge_norm_high_perc_var = tk.DoubleVar(value=99.9)
        self.keep_intermediate_npz_var = tk.BooleanVar(value=False)
        self.min_frames_to_keep_npz_var = tk.IntVar(value=0)
        self.keep_intermediate_segment_visual_format_var = tk.StringVar(value="mp4")
        self.merge_output_suffix_var = tk.StringVar(value="_depth") # New Variable
        # self.merge_script_gui_silence_level_var = tk.StringVar(value="Normal (Info)") # Removed GUI verbosity control
        self.current_input_mode = "batch_folder" # "batch_folder", "single_video_file", "single_image_file", "image_sequence_folder"
        self.single_file_mode_active = False # True if a single file/sequence folder is explicitly loaded
        self.effective_move_original_on_completion = self.MOVE_ORIGINAL_TO_FINISHED_FOLDER_ON_COMPLETION
        self.use_local_models_only_var = tk.BooleanVar(value=False)
        self.status_message_var = tk.StringVar(value="Ready")
        self.current_filename_var = tk.StringVar(value="N/A")
        self.current_resolution_var = tk.StringVar(value="N/A")
        self.current_frames_var = tk.StringVar(value="N/A")
        self.target_height = tk.IntVar(value=384) # Initial default height
        self.target_width = tk.IntVar(value=640)  # Initial default width
        self.debug_logging_enabled = tk.BooleanVar(value=False) # Default to OFF (INFO level)
        self.enable_dual_output_robust_norm = tk.BooleanVar(value=False) # Default to ON for testing
        self.robust_norm_low_percentile = tk.DoubleVar(value=0.0)      # Example default
        self.robust_norm_high_percentile = tk.DoubleVar(value=75.5)     # Example default
        self.robust_norm_output_min = tk.DoubleVar(value=0.0)
        self.robust_norm_output_max = tk.DoubleVar(value=1.0)
        self.robust_output_suffix = tk.StringVar(value="_clipped_depth")
        self.is_depth_far_black = tk.BooleanVar(value=True)        
        self.disable_xformers_var = tk.BooleanVar(value=True)

        self.all_tk_vars = {
            "input_dir_or_file_var": self.input_dir_or_file_var,
            "output_dir": self.output_dir,
            "guidance_scale": self.guidance_scale,
            "inference_steps": self.inference_steps,
            "seed": self.seed,
            "cpu_offload": self.cpu_offload,
            "use_cudnn_benchmark": self.use_cudnn_benchmark,
            "process_length": self.process_length,
            "target_fps": self.target_fps,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "process_as_segments_var": self.process_as_segments_var,
            "save_final_output_json_var": self.save_final_output_json_var,
            "merge_output_format_var": self.merge_output_format_var,
            "merge_alignment_method_var": self.merge_alignment_method_var,
            "merge_dither_var": self.merge_dither_var,
            "merge_dither_strength_var": self.merge_dither_strength_var,
            "merge_gamma_correct_var": self.merge_gamma_correct_var,
            "merge_gamma_value_var": self.merge_gamma_value_var,
            "merge_percentile_norm_var": self.merge_percentile_norm_var,
            "merge_norm_low_perc_var": self.merge_norm_low_perc_var,
            "merge_norm_high_perc_var": self.merge_norm_high_perc_var,
            "keep_intermediate_npz_var": self.keep_intermediate_npz_var,
            "min_frames_to_keep_npz_var": self.min_frames_to_keep_npz_var,
            "keep_intermediate_segment_visual_format_var": self.keep_intermediate_segment_visual_format_var,
            "merge_output_suffix_var": self.merge_output_suffix_var,
            "use_local_models_only_var": self.use_local_models_only_var,
            "target_height": self.target_height,
            "target_width": self.target_width,
            "enable_dual_output_robust_norm": self.enable_dual_output_robust_norm,
            "robust_norm_low_percentile": self.robust_norm_low_percentile,
            "robust_norm_high_percentile": self.robust_norm_high_percentile,
            "robust_norm_output_min": self.robust_norm_output_min,
            "robust_norm_output_max": self.robust_norm_output_max,
            "robust_output_suffix": self.robust_output_suffix,
            "is_depth_far_black": self.is_depth_far_black,
            "dark_mode_var": self.dark_mode_var,
            "disable_xformers_var": self.disable_xformers_var,
        }
        self.initial_default_settings = self._collect_all_settings()
        self._help_data = None

        self.last_settings_dir = os.getcwd()
        self.message_queue = queue.Queue() # Still needed for progress updates

        # Removed message_catalog setup
        # set_gui_logger_callback(self._queue_message_for_gui_log)
        # set_gui_verbosity(self._get_mapped_gui_verbosity_level()) # Removed
        # mc_configure_timestamps(console=True, gui=False) # Removed

        self.load_config() 
        self.stop_event = threading.Event()
        self.processing_thread = None
        self.secondary_output_widgets_references = []
        self._load_help_content()
        
        self.style = ttk.Style(self.root)
        self._apply_theme(is_startup=True)
        
        # Set initial logging level based on the default value of debug_logging_enabled
        if self.debug_logging_enabled.get():
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
        _logger.info(f"Initial logging level set to {'DEBUG' if self.debug_logging_enabled.get() else 'INFO'}.")
        # --------------------------------------

        self._create_menubar()
        self.create_widgets() 
        self.root.app_instance = self 
        # --------------------------------------
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.toggle_merge_related_options_active_state()
        self.toggle_secondary_output_options_active_state()
                
        _logger.debug("DepthCrafter GUI initialized successfully.")

    def _apply_all_settings(self, settings_data: dict):
        for key, value_from_json in settings_data.items():
            if key == "target_fps": # Specific debug
                _logger.debug(f"_apply_all_settings: Loading target_fps from JSON. Value: {value_from_json}, Type: {type(value_from_json)}")
            if key in self.all_tk_vars:
                try:
                    self.all_tk_vars[key].set(value_from_json)
                    # After setting, get it back to see what DoubleVar stored
                    if key == "target_fps":
                        val_in_doublevar = self.all_tk_vars[key].get()
                        _logger.debug(f"_apply_all_settings: target_fps in DoubleVar after set: {val_in_doublevar}, Type: {type(val_in_doublevar)}")
                except tk.TclError:
                     _logger.warning(f"Warning: Could not set value for setting '{key}' to '{value_from_json}'. Skipping.")
            else:
                _logger.warning(f"Warning: Unknown setting '{key}' found in settings file. Ignoring.")
        if hasattr(self, 'process_as_segments_var'):
            self.toggle_merge_related_options_active_state()
        # Removed update GUI verbosity

    def _apply_theme(self, is_startup: bool = False):
        """Applies the selected theme (dark or light) to the GUI."""
        
        if not THEMEDTK_AVAILABLE:
            # ...
            return
        
        # --- Core Theme Application (Must happen before detailed styling) ---
        if self.dark_mode_var.get():
            colors = DARK_MODE_COLORS
        else:
            colors = LIGHT_MODE_COLORS
        
        theme_name = colors["theme_name"]
        self.current_theme_colors = colors
        
        if THEMEDTK_AVAILABLE:
             # Apply the theme first
             self.root.set_theme(theme_name) 

        # --- Detailed TEntry/TCombobox Styling (Apply to CURRENT Theme) ---
        # NOTE: We use style.map() for backgrounds to override theme defaults
        entry_bg = colors["entry_bg"]
        entry_fg = colors["fg"]
        
        # 1. TEntry Styling
        self.style.configure("TEntry", foreground=entry_fg, insertcolor=entry_fg)
        # Use map to force the fieldbackground for the default state (empty tuple)
        self.style.map('TEntry', 
                       fieldbackground=[('', entry_bg)], # '' is the default state
                       foreground=[('', entry_fg)])
        
        # 2. TCombobox Styling
        self.style.configure("TCombobox", foreground=entry_fg) 
        self.style.map('TCombobox', 
                       fieldbackground=[('readonly', entry_bg)], 
                       foreground=[('readonly', entry_fg)])


        # --- Manual Coloring for raw TK Menu ---
        root_bg_color = colors["bg"]
        root_fg_color = colors["fg"]
        menu_active_bg = "#555555" if self.dark_mode_var.get() else "#dddddd"
        menu_active_fg = "white" if self.dark_mode_var.get() else "black"

        self.root.config(bg=root_bg_color)
        
        # NOTE: Since the widgets are now ttk, this is mainly for the root frame and menu
        if hasattr(self, 'menubar'): 
            # Menubar and Menus are raw tk.Menu and need manual color
            self.menubar.config(bg=root_bg_color, fg=root_fg_color, activebackground=menu_active_bg, activeforeground=menu_active_fg)
            if hasattr(self, 'file_menu'): self.file_menu.config(bg=root_bg_color, fg=root_fg_color)
            if hasattr(self, 'help_menu'): self.help_menu.config(bg=root_bg_color, fg=root_fg_color)
           
        self.root.update_idletasks()
    
    def _cleanup_segment_folder(self, segment_subfolder_path, original_basename, master_meta):
        del_folder = False
        if not self.keep_intermediate_npz_var.get():
            _logger.debug(f"Deleting intermediate segment subfolder for {original_basename} (Keep NPZ unchecked)...")
            del_folder = True
        else:
            min_frames = self.min_frames_to_keep_npz_var.get()
            if min_frames > 0:
                orig_frames = master_meta.get("original_video_details", {}).get("raw_frame_count", 0)
                if orig_frames < min_frames:
                    _logger.info(f"  Video frames ({orig_frames}) < threshold ({min_frames}). Deleting segment folder for {original_basename} despite 'Keep NPZ'.")
                    del_folder = True
                else:
                    _logger.info(f"  Video frames ({orig_frames}) >= threshold ({min_frames}). Segment folder for {original_basename} will be kept.")
            else:
                _logger.debug(f"Keeping intermediate NPZ files for {original_basename} (Keep NPZ checked, no positive frame threshold).")
        if del_folder:
            if os.path.exists(segment_subfolder_path):
                try: 
                    shutil.rmtree(segment_subfolder_path)
                    _logger.debug(f"Successfully deleted segment subfolder for {original_basename}.")
                except Exception as e:
                    _logger.error(f"  Error deleting segment subfolder {segment_subfolder_path}: {e}")
            else:
                _logger.warning(f"  Segment subfolder not found for deletion: {segment_subfolder_path}")
        else:
            _logger.debug(f"Keeping intermediate NPZ files and _master_meta.json in {segment_subfolder_path}")

    def _collect_all_settings(self) -> dict:
        settings_data = {}
        for key, tk_var in self.all_tk_vars.items():
            try:
                value = tk_var.get()
                settings_data[key] = value
                if key == "target_fps": # Specific debug for target_fps
                    _logger.debug(f"_collect_all_settings: target_fps raw value: {value}, type: {type(value)}")
            except tk.TclError:
                _logger.warning(f"Warning: Could not get value for setting '{key}'. Skipping.")
        return settings_data

    def _create_menubar(self):
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Load Settings...", command=self._load_all_settings)
        self.file_menu.add_command(label="Save Settings As...", command=self._save_all_settings_as)
        self.file_menu.add_command(label="Reset Settings to Default", command=self._reset_settings_to_defaults)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Restore Finished Input Files...", command=lambda: self._restore_input_files(folder_type="finished"))
        self.file_menu.add_command(label="Restore Failed Input Files...", command=lambda: self._restore_input_files(folder_type="failed"))
        self.file_menu.add_separator()
        self.file_menu.add_checkbutton(label="Use Local Models Only", variable=self.use_local_models_only_var, onvalue=True, offvalue=False)
        self.file_menu.add_checkbutton(label="Disable xFormers (VRAM Save)", variable=self.disable_xformers_var, onvalue=True, offvalue=False)
        if THEMEDTK_AVAILABLE:
            self.file_menu.add_checkbutton(label="Dark Mode", variable=self.dark_mode_var, command=self._apply_theme)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.on_close)

        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_checkbutton(label="Enable Debug Logging", variable=self.debug_logging_enabled, command=self._toggle_debug_logging)
        self.help_menu.add_separator() # Optional separator for clarity
        self.help_menu.add_command(label="GUI Overview", command=lambda: self._show_help_for("general_gui_overview"))
        # -----------------------------------------

    def _determine_input_mode_from_path(self, path_str: str) -> Tuple[str, bool]:
        """
        Analyzes a path string and determines the input mode and if it's a single source.
        Returns: (input_mode_str, is_single_source_bool)
        """
        if not path_str or not os.path.exists(path_str):
            _logger.warning(f"GUI Input: Path '{path_str}' is invalid or does not exist. Cannot determine input mode accurately.")
            return "batch_folder", False

        is_single_source = False
        mode = "batch_folder"

        if os.path.isfile(path_str):
            is_single_source = True
            ext = os.path.splitext(path_str)[1].lower()
            is_video = any(ext in vid_ext.replace("*", "") for vid_ext in self.VIDEO_EXTENSIONS)
            is_image = any(ext in img_ext.replace("*", "") for img_ext in self.IMAGE_EXTENSIONS)

            if is_video:
                mode = "single_video_file"
            elif is_image:
                mode = "single_image_file"
            else:
                _logger.warning(f"GUI Input: Typed path '{path_str}' is a file of unknown type. Treating as non-single source (batch fallback).")
                mode = "batch_folder"
                is_single_source = False
        elif os.path.isdir(path_str):
            if self._is_image_sequence_folder(path_str):
                mode = "image_sequence_folder"
                is_single_source = True
            else:
                mode = "batch_folder"
                is_single_source = False
        else:
            _logger.warning(f"GUI Input: Path '{path_str}' exists but is not a regular file or directory.")
            mode = "batch_folder"
            is_single_source = False
            
        _logger.debug(f"GUI Input: Determined mode for path '{path_str}' as '{mode}', is_single_source: {is_single_source}.")
        return mode, is_single_source

    def _determine_video_paths_and_processing_mode(self, original_basename, master_meta_for_this_vid):
        main_output_dir_for_video = self.output_dir.get()
        was_processed_as_segments = master_meta_for_this_vid["global_processing_settings"]["processed_as_segments"]
        segment_subfolder_path = None
        if was_processed_as_segments:
            segment_subfolder_name = get_segment_output_folder_name(original_basename)
            segment_subfolder_path = os.path.join(main_output_dir_for_video, segment_subfolder_name)
        return main_output_dir_for_video, segment_subfolder_path, was_processed_as_segments

    def _execute_re_merge_wrapper(self, remerge_args_dict):
        try: self._execute_re_merge(remerge_args_dict)
        finally: self.message_queue.put(("set_ui_state", False))

    def _execute_re_merge(self, remerge_args_dict):
        self.stop_event.clear(); self.progress["value"] = 0; self.progress["maximum"] = 1
        start_time = time.perf_counter()
        primary_output_path = "N/A (Merge Failed)" # Initialize to ensure it's always defined
        try:
            if merge_depth_segments:
                primary_output_path = merge_depth_segments.merge_depth_segments(**remerge_args_dict)
                if primary_output_path:
                    _logger.info(f"Re-Merge completed. Primary output saved to: {primary_output_path}")
                else:
                    _logger.warning("Re-Merge completed, but no primary output path was returned.")
            else: 
                _logger.warning("Segment merging for N/A for re-merge action skipped: merge_depth_segments module not available.")
        except Exception as e:
            _logger.exception(f"ERROR during re-merge execution: {e}")
            self.status_message_var.set(f"Re-Merge Error: {e.__class__.__name__}") # Update GUI status
        finally:
            duration = format_duration(time.perf_counter() - start_time)
            _logger.info(f"--- Re-Merge for: {os.path.basename(remerge_args_dict['master_meta_path'])} finished in {duration}. ---")
            # If a primary output path was generated, show it in status for better feedback
            if primary_output_path and primary_output_path != "N/A (Merge Failed)":
                self.status_message_var.set(f"Re-Merge Finished. Output: {os.path.basename(primary_output_path)}")
            else:
                self.status_message_var.set("Re-Merge Finished (No primary output).")
            self.message_queue.put(("progress", 1))

    def _execute_generate_segment_visuals_wrapper(self, gen_visual_args_dict):
        try: self._execute_generate_segment_visuals(gen_visual_args_dict)
        finally: self.message_queue.put(("set_ui_state", False))

    def _execute_generate_segment_visuals(self, gen_visual_args_dict):
        self.stop_event.clear(); self.progress["value"] = 0
        master_path = gen_visual_args_dict["master_meta_path"]
        vis_fmt = gen_visual_args_dict["visual_format_to_generate"]
        start_time = time.perf_counter()
        
        meta_data = load_json_file(master_path) 
        if not meta_data: return 
        
        jobs = [j for j in meta_data.get("jobs_info", []) if j.get("status") == "success" and j.get("output_segment_filename")]
        if not jobs: 
            _logger.warning(f"No successful segments with output filenames found in {os.path.basename(master_path)} for visual generation.")
            return
        self.progress["maximum"] = len(jobs)
        seg_folder_path = os.path.dirname(master_path)
        updated_visual_paths = {}

        for i, job_meta in enumerate(jobs):
            if self.stop_event.is_set(): 
                _logger.info("Segment visual generation cancelled during processing.")
                break
            seg_id, npz_name = job_meta.get("segment_id"), job_meta.get("output_segment_filename")
            npz_path = os.path.join(seg_folder_path, npz_name)
            _logger.debug(f"  Visual Gen - Processing segment {seg_id + 1 if seg_id is not None else '?'}/{len(jobs)}: {npz_name} for {vis_fmt}") 
            
            if not os.path.exists(npz_path): 
                _logger.error(f"File not found: {npz_path}")
                continue
            try:
                with np.load(npz_path) as data:
                    if 'frames' not in data.files: 
                        _logger.error(f"Key 'frames' not found in NPZ: {npz_name}")
                        continue
                    raw_frames = data['frames']
                if raw_frames.size == 0: 
                    _logger.warning(f"    Visual Gen - WARNING: Segment {npz_name} is empty. Skipping.")
                    continue
                
                norm_frames = (raw_frames - raw_frames.min()) / (raw_frames.max() - raw_frames.min()) if raw_frames.max() != raw_frames.min() else np.zeros_like(raw_frames)
                norm_frames = np.clip(norm_frames, 0, 1)
                base_name_no_ext = os.path.splitext(npz_name)[0]
                save_path, save_err = None, None
                fps = float(job_meta.get("processed_at_fps", meta_data.get("original_video_details", {}).get("original_fps", 30.0)))
                if fps <= 0: fps = 30.0

                if vis_fmt == "mp4" or vis_fmt == "main10_mp4":
                    save_path, save_err = save_depth_visual_as_mp4_util(
                        norm_frames, 
                        os.path.join(seg_folder_path, f"{base_name_no_ext}_visual.mp4"),
                        fps,
                        output_format=vis_fmt
                    )
                elif vis_fmt == "png_sequence":
                    save_path, save_err = save_depth_visual_as_png_sequence_util(norm_frames, seg_folder_path, base_name_no_ext)
                elif vis_fmt == "exr_sequence":
                     if OPENEXR_AVAILABLE_GUI: save_path, save_err = save_depth_visual_as_exr_sequence_util(norm_frames, seg_folder_path, base_name_no_ext)
                     else: save_err = "OpenEXR module not available in GUI environment."
                elif vis_fmt == "exr":
                    if OPENEXR_AVAILABLE_GUI:
                        first_frame = norm_frames[0] if len(norm_frames) > 0 else None
                        if first_frame is None: save_err = "No frame data for single EXR."
                        else: save_path, save_err = save_depth_visual_as_single_exr_util(first_frame, seg_folder_path, base_name_no_ext)
                    else: save_err = "OpenEXR module not available in GUI environment."

                if save_path:
                    _logger.debug(f"    Visual Gen - Successfully saved visual: {save_path}") 
                    if seg_id is not None: updated_visual_paths[seg_id] = {"path": os.path.abspath(save_path), "format": vis_fmt}
                if save_err: 
                    _logger.error(f"    Visual Gen - ERROR saving visual for {npz_name}: {save_err}, format requested: {vis_fmt}") 
            except Exception as e:
                _logger.exception(f"    Visual Gen - ERROR processing segment {npz_name}: {e}") 
            self.message_queue.put(("progress", i + 1))
        
        if updated_visual_paths:
            _logger.info("Visual Gen - Updating master metadata with new visual paths...")
            meta_content_update = load_json_file(master_path)
            if meta_content_update:
                updated_count = 0
                for job_entry in meta_content_update.get("jobs_info", []):
                    s_id = job_entry.get("segment_id")
                    if s_id in updated_visual_paths:
                        job_entry["intermediate_visual_path"] = updated_visual_paths[s_id]["path"]
                        job_entry["intermediate_visual_format_saved"] = updated_visual_paths[s_id]["format"]
                        updated_count +=1
                if updated_count > 0:
                    if save_json_file(meta_content_update, master_path, indent=4):
                         _logger.info(f"Visual Gen - Master metadata updated for {updated_count} segments.")
                else: _logger.info("Visual Gen - No segments in master metadata needed visual path updates.")
        
        duration = format_duration(time.perf_counter() - start_time)
        _logger.info(f"--- Segment Visual Generation for: {os.path.basename(master_path)} (Format: {vis_fmt}) finished in {duration}. ---")
        self.message_queue.put(("progress", len(jobs)))

    def _finalize_video_processing(self, current_video_path, original_basename, master_meta_for_this_vid):
        if master_meta_for_this_vid["completed_failed_jobs"] == 0:
            master_meta_for_this_vid["overall_status"] = "all_success"
        elif master_meta_for_this_vid["completed_successful_jobs"] > 0:
            master_meta_for_this_vid["overall_status"] = "partial_success"
        else:
            master_meta_for_this_vid["overall_status"] = "all_failed"

        _logger.debug(f"Finished processing for {original_basename}. Overall Status: {master_meta_for_this_vid['overall_status']}.")
        
        main_output_dir, segment_subfolder_path, was_segments = self._determine_video_paths_and_processing_mode(original_basename, master_meta_for_this_vid)
        master_meta_filepath, meta_saved = None, False
        # --- FIX: Initialize merge_success and final_merged_path BEFORE conditional assignment ---
        merge_success, final_merged_path = False, "N/A (Merge not applicable or failed)"
        
        try:
            master_meta_filepath, meta_saved = self._save_master_metadata_and_cleanup_segment_json(master_meta_for_this_vid, original_basename, main_output_dir, was_segments, segment_subfolder_path)
            
            if was_segments and meta_saved and master_meta_for_this_vid["overall_status"] in ["all_success", "partial_success"]:
                try: # Nested try-except to catch errors specifically from merging
                    merge_success, final_merged_path = self._handle_segment_merging(master_meta_filepath, original_basename, main_output_dir, master_meta_for_this_vid)
                except Exception as e_merge:
                    _logger.error(f"Error during segment merging for {original_basename}: {e_merge}", exc_info=True)
                    self.status_message_var.set(f"Merge Failed: {e_merge.__class__.__name__}")
                    merge_success, final_merged_path = False, f"N/A (Merge failed due to {e_merge.__class__.__name__})"
            elif was_segments:
                # If segments were processed but not merged (e.g., all_failed status, or no successful segments)
                _logger.debug(f"Skipping merge for {original_basename} (status: {master_meta_for_this_vid['overall_status']}, meta_saved: {meta_saved}). Segments remain in {segment_subfolder_path or 'N/A'}")
                # No change to merge_success/final_merged_path as they were initialized to False/N/A
            
            if self.save_final_output_json_var.get():
                self._save_final_output_sidecar_json(original_basename, final_merged_path, master_meta_filepath, master_meta_for_this_vid, was_segments, merge_success)
            
            if was_segments and segment_subfolder_path:
                self._cleanup_segment_folder(segment_subfolder_path, original_basename, master_meta_for_this_vid)
        except Exception as e:
            _logger.exception(f"Error during finalization for {original_basename}: {e}")
            self.status_message_var.set(f"Finalization Error: {e.__class__.__name__} for {original_basename}")

        final_status = master_meta_for_this_vid.get("overall_status", "all_failed")

        if self.effective_move_original_on_completion:
            target_subfolder_name = ""
            if final_status == "all_success":
                target_subfolder_name = "finished"
            elif final_status in ["partial_success", "all_failed"]:
                target_subfolder_name = "failed"
            else:
                _logger.warning(f"Move Original: Could not determine 'finished' or 'failed' status for '{original_basename}' (status: '{final_status}'). Original file will not be moved.")

            if target_subfolder_name:
                self._move_original_source(current_video_path, original_basename, target_subfolder_name)
        else:
            _logger.info(f"Skipped moving original source '{original_basename}' (single file/sequence mode).")

    def _get_segments_to_resume_or_overwrite(self, vid_path, original_basename, 
                                             segment_subfolder_path, all_potential_segments_from_define,
                                             base_job_info_for_video_ref: dict):
        master_meta_path = os.path.join(segment_subfolder_path, f"{original_basename}_master_meta.json")
        base_job_info_for_video_ref["pre_existing_successful_jobs"] = []

        if os.path.exists(master_meta_path):
            msg_dialog = (f"Master metadata found for '{original_basename}'. This video was previously processed/finalized.\n"
                          f"Path: {master_meta_path}\n\n"
                          f"Do you want to:\n"
                          f"- 'Yes': Re-process only FAILED segments and update master metadata?\n"
                          f"         (Existing successful segments will be preserved in the new master metadata).\n"
                          f"- 'No': Delete ALL existing segments and master metadata and start fresh?\n"
                          f"- 'Cancel': Skip this video entirely?")
            choice = messagebox.askyesnocancel("Resume or Overwrite Finalized Segments?", msg_dialog, parent=self.root)

            if choice is True:
                _logger.info(f"Attempting to re-process failed segments for {original_basename} based on existing master metadata.")
                master_data = load_json_file(master_meta_path)
                if not master_data or "jobs_info" not in master_data:
                    _logger.warning(f"Could not load master metadata or 'jobs_info' missing for {original_basename}. Defaulting to overwrite.")
                    choice = False # Fallthrough
                else:
                    failed_segment_jobs_to_run = []
                    successful_jobs_from_old_master = []
                    potential_segments_dict = {seg_job['segment_id']: seg_job for seg_job in all_potential_segments_from_define}

                    for job_in_meta in master_data.get("jobs_info", []):
                        seg_id = job_in_meta.get("segment_id")
                        if job_in_meta.get("status") == "success":
                            successful_jobs_from_old_master.append(job_in_meta)
                        elif seg_id is not None and seg_id in potential_segments_dict:
                            failed_segment_jobs_to_run.append(potential_segments_dict[seg_id])
                            _logger.debug(f"  Queueing segment ID {seg_id} (status: {job_in_meta.get('status', 'unknown')}) for {original_basename} for re-processing.")
                        else:
                            _logger.warning(f"  Warning: Segment (ID: {seg_id}, Status: {job_in_meta.get('status')}) from master_meta for {original_basename} not re-queueable. It will be ignored.")
                    
                    if not failed_segment_jobs_to_run:
                        _logger.info(f"No re-processable failed segments found in master_meta for {original_basename}. All existing successful segments will be preserved if merging.")
                        base_job_info_for_video_ref["pre_existing_successful_jobs"] = successful_jobs_from_old_master
                        return [], "skipped_no_failed_segments_in_master_for_reprocessing"
                    
                    try:
                        backup_master_meta_path = master_meta_path + f".backup_{time.strftime('%Y%m%d%H%M%S')}"
                        shutil.move(master_meta_path, backup_master_meta_path)
                        _logger.debug(f"Backed up existing file {os.path.basename(master_meta_path)} to: {os.path.basename(backup_master_meta_path)}")
                    except Exception as e:
                        _logger.warning(f"  Warning: Could not back up existing master metadata: {e}. It might be overwritten.")

                    base_job_info_for_video_ref["pre_existing_successful_jobs"] = successful_jobs_from_old_master
                    return failed_segment_jobs_to_run, "reprocessing_failed_from_master"
            
            if choice is False: 
                _logger.info(f"User chose/defaulted to delete existing segment folder and start fresh for {original_basename}: {segment_subfolder_path}")
                try:
                    if os.path.exists(segment_subfolder_path): shutil.rmtree(segment_subfolder_path)
                    _logger.debug(f"  Successfully deleted: {segment_subfolder_path}")
                except Exception as e:
                    _logger.error(f"  Error deleting {segment_subfolder_path}: {e}. Processing may fail or overwrite.")
                return all_potential_segments_from_define, "overwriting_finalized"
            
            else: # Cancel
                _logger.info(f"Skipping {original_basename} (user chose to cancel on finalized segments).")
                return [], "skipped_finalized"

        elif os.path.exists(segment_subfolder_path):
            msg_dialog_incomplete = (f"Incomplete segment data found for '{original_basename}' (no master metadata file).\n"
                                     f"Path: {segment_subfolder_path}\n\n"
                                     f"Do you want to:\n"
                                     f"- 'Yes': Resume by processing only missing/failed segments?\n"
                                     f"         (Existing successful segments will be preserved).\n"
                                     f"- 'No': Delete existing incomplete segments and start fresh?\n"
                                     f"- 'Cancel': Skip this video entirely?")
            choice_incomplete = messagebox.askyesnocancel("Resume Incomplete Segments?", msg_dialog_incomplete, parent=self.root)

            if choice_incomplete is True:
                _logger.debug(f"Attempting to resume incomplete segments for {original_basename}.")
                segments_to_run = []
                num_already_complete = 0
                completed_segment_metadata_from_json = []

                for potential_segment_job in all_potential_segments_from_define:
                    seg_id = potential_segment_job["segment_id"]
                    total_segs = potential_segment_job["total_segments"]
                    expected_npz_filename = get_segment_npz_output_filename(original_basename, seg_id, total_segs)
                    expected_json_filename = get_sidecar_json_filename(expected_npz_filename)
                    npz_path = os.path.join(segment_subfolder_path, expected_npz_filename)
                    json_path = os.path.join(segment_subfolder_path, expected_json_filename)

                    is_complete_and_successful = False
                    if os.path.exists(npz_path) and os.path.exists(json_path):
                        segment_meta = load_json_file(json_path)
                        if segment_meta and segment_meta.get("status") == "success":
                            is_complete_and_successful = True
                            num_already_complete += 1
                            completed_segment_metadata_from_json.append(segment_meta)
                        else:
                            status_msg = segment_meta.get('status', 'unknown') if segment_meta else 'JSON missing/corrupt'
                            _logger.info(f"  Segment {seg_id+1}/{total_segs} for {original_basename} found but not successful (status: {status_msg}). Will re-process.")
                    else:
                        _logger.debug(f"  Segment {seg_id+1}/{total_segs} for {original_basename} (NPZ: {expected_npz_filename}) not found or JSON missing. Will process.")

                    if not is_complete_and_successful:
                        segments_to_run.append(potential_segment_job)
                
                if num_already_complete > 0:
                    _logger.info(f"Found {num_already_complete} successfully completed segments for {original_basename} that will be skipped during processing.")
                
                base_job_info_for_video_ref["pre_existing_successful_jobs"] = completed_segment_metadata_from_json

                if not segments_to_run and num_already_complete == len(all_potential_segments_from_define):
                    _logger.warning(f"  All segments for {original_basename} appear complete from individual files, but master_meta was missing. Consider re-merging. Skipping processing.")
                    return [], "skipped_all_segments_found_complete_no_master"
                elif not segments_to_run and num_already_complete < len(all_potential_segments_from_define):
                     _logger.warning(f"  No segments to run for {original_basename}, but not all were found complete. Total defined: {len(all_potential_segments_from_define)}, Found complete: {num_already_complete}")
                     return [], "skipped_no_segments_to_run_incomplete"
                return segments_to_run, "resuming_incomplete"

            elif choice_incomplete is False:
                _logger.info(f"User chose to delete existing incomplete segment folder and start fresh for {original_basename}: {segment_subfolder_path}")
                try:
                    if os.path.exists(segment_subfolder_path): shutil.rmtree(segment_subfolder_path)
                    _logger.debug(f"  Successfully deleted: {segment_subfolder_path}")
                except Exception as e:
                    _logger.error(f"  Error deleting {segment_subfolder_path}: {e}. Processing may fail or overwrite.")
                return all_potential_segments_from_define, "overwriting_incomplete"
            
            else: # Cancel
                _logger.info(f"Skipping {original_basename} (user chose to cancel on incomplete segments).")
                return [], "skipped_incomplete"
                
        else: # Segment folder does not exist
            return all_potential_segments_from_define, "fresh_processing"

    def _handle_segment_merging(self, master_meta_filepath, original_basename, main_output_dir, master_meta) -> Tuple[bool, str]:
        """
        Handles the merging of segments, potentially generating a second robustly normalized output.
        Returns a tuple: (bool indicating merge success, str path of the primary merged output).
        """
        if not merge_depth_segments:
            _logger.warning(f"Segment merging for {original_basename} skipped: merge_depth_segments module not available.")
            return False, "N/A (Merge module not available - module missing)"
        
        out_fmt = self.merge_output_format_var.get()
        output_suffix = self.merge_output_suffix_var.get()
        merged_base_name = f"{original_basename}{output_suffix}"

        align_method = "linear_blend" if self.merge_alignment_method_var.get() == "Linear Blend" else "shift_scale"
        
        enable_dual_output = self.enable_dual_output_robust_norm.get() 
        robust_low_perc = self.robust_norm_low_percentile.get()
        robust_high_perc = self.robust_norm_high_percentile.get()
        robust_output_min = self.robust_norm_output_min.get()
        robust_output_max = self.robust_norm_output_max.get()
        robust_output_suffix_val = self.robust_output_suffix.get()
        is_depth_far_black_val = self.is_depth_far_black.get()

        try:
            primary_output_path = merge_depth_segments.merge_depth_segments(
                master_meta_path=master_meta_filepath, 
                output_path_arg=main_output_dir,
                do_dithering=self.merge_dither_var.get(), 
                dither_strength_factor=self.merge_dither_strength_var.get(),
                apply_gamma_correction=self.merge_gamma_correct_var.get(), 
                gamma_value=self.merge_gamma_value_var.get(),
                use_percentile_norm=self.merge_percentile_norm_var.get(), 
                norm_low_percentile=self.merge_norm_low_perc_var.get(),
                norm_high_percentile=self.merge_norm_high_perc_var.get(), 
                output_format=out_fmt,
                merge_alignment_method=align_method, 
                output_filename_override_base=merged_base_name,
                enable_dual_output_robust_norm=enable_dual_output,
                robust_norm_low_percentile=robust_low_perc,
                robust_norm_high_percentile=robust_high_perc,
                robust_norm_output_min=robust_output_min,
                robust_norm_output_max=robust_output_max,
                robust_output_suffix=robust_output_suffix_val,
                is_depth_far_black=is_depth_far_black_val
            )
            
            # If primary_output_path is None, the merge failed or didn't produce a path
            if primary_output_path is None:
                _logger.error(f"merge_depth_segments returned None for {original_basename}. Merge considered failed.")
                return False, f"N/A (Merge module returned no path)"
            else:
                _logger.debug(f"Primary merge for {original_basename} successful. Output: {primary_output_path}")
                return True, primary_output_path # Successful merge
                
        except Exception as e: 
            _logger.exception(f"Exception during merge_depth_segments call for {original_basename}: {e}")
            self.status_message_var.set(f"Merge Error: {e.__class__.__name__} for {original_basename}")
            return False, f"N/A (Merge failed due to {e.__class__.__name__})"
        
    def _initialize_master_metadata_entry(self, original_basename, job_info_for_original_details, total_expected_jobs_for_this_video):
        entry = {
            "original_video_basename": original_basename,
            "original_video_details": {
                "raw_frame_count": job_info_for_original_details.get("original_video_raw_frame_count", 0),
                "original_fps": job_info_for_original_details.get("original_video_fps", 30.0)
            },
            "global_processing_settings": {
                "guidance_scale": self.guidance_scale.get(),
                "inference_steps": self.inference_steps.get(),
                "target_height_setting": self.target_height.get(),
                "target_width_setting": self.target_width.get(),
                "seed_setting": self.seed.get(),
                "target_fps_setting": self.target_fps.get(),
                "process_max_frames_setting": self.process_length.get(),
                "gui_window_size_setting": self.window_size.get(),
                "gui_overlap_setting": self.overlap.get(),
                "processed_as_segments": self.process_as_segments_var.get(),
            },
            "jobs_info": [], "overall_status": "pending",
            "total_expected_jobs": total_expected_jobs_for_this_video,
            "completed_successful_jobs": 0, "completed_failed_jobs": 0,
        }
        if self.process_as_segments_var.get():
            entry["global_processing_settings"]["segment_definition_output_window_frames"] = job_info_for_original_details.get("gui_desired_output_window_frames", self.window_size.get())
            entry["global_processing_settings"]["segment_definition_output_overlap_frames"] = job_info_for_original_details.get("gui_desired_output_overlap_frames", self.overlap.get())
        return entry

    def _is_image_sequence_folder(self, folder_path: str) -> bool:
        """Rudimentary check if a folder looks like an image sequence."""
        if not os.path.isdir(folder_path): return False
        
        image_files_count = 0
        video_files_count = 0
        sub_dirs_count = 0

        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                sub_dirs_count += 1
                continue
            
            ext = os.path.splitext(item)[1].lower()
            if any(ext in img_ext.replace("*", "") for img_ext in self.IMAGE_EXTENSIONS):
                image_files_count +=1
            elif any(ext in vid_ext.replace("*", "") for vid_ext in self.VIDEO_EXTENSIONS):
                video_files_count +=1
        
        return image_files_count > 5 and video_files_count == 0 and sub_dirs_count == 0

    def _load_help_content(self):
        if self._help_data is None: # Only load once
            raw_data = load_json_file(DepthCrafterGUI.HELP_CONTENT_FILENAME)
            if raw_data:
                self._help_data = raw_data # Keep raw data for future potential uses
                # Populate the module-level _HELP_TEXTS dictionary for tooltips
                global _HELP_TEXTS
                _HELP_TEXTS.clear() # Clear existing in case of reload (e.g., in future)
                for key, content in raw_data.items():
                    if "text" in content: # Ensure 'text' key exists
                        _HELP_TEXTS[key] = content["text"]
            else:
                self._help_data = {} # Indicate loading failed
                _logger.warning(f"Warning: Could not load help content from {DepthCrafterGUI.HELP_CONTENT_FILENAME}. Tooltips will be limited/show 'not found'.")
        return self._help_data

    def _load_all_settings(self):
        filepath = filedialog.askopenfilename(title="Load Settings File", filetypes=self.SETTINGS_FILETYPES, initialdir=self.last_settings_dir)
        if not filepath:
            _logger.info("Load settings cancelled by user.")
            return
        self.last_settings_dir = os.path.dirname(filepath)
        settings_data = load_json_file(filepath)
        if settings_data:
            self._apply_all_settings(settings_data)
            _logger.debug(f"Successfully loaded settings from: {filepath}")
        else:
            messagebox.showerror("Load Error", f"Could not load settings from:\n{filepath}\nSee console log for details.")

    def _move_original_source(self, current_video_path: str, original_basename: str, target_subfolder: str):
        _logger.debug(f"Moving original source '{original_basename}' to '{target_subfolder}' folder.")
        try:
            path_from_gui_input_field = self.input_dir_or_file_var.get()

            actual_input_root_for_target_folder: str
            if os.path.isdir(path_from_gui_input_field):
                actual_input_root_for_target_folder = path_from_gui_input_field
            elif os.path.isfile(path_from_gui_input_field):
                actual_input_root_for_target_folder = os.path.dirname(path_from_gui_input_field)
            else:
                _logger.warning(f"Move Original: The GUI input path '{path_from_gui_input_field}' is invalid for determining the target folder root. Using dirname of processed item as fallback.")
                actual_input_root_for_target_folder = os.path.dirname(current_video_path)
                if not os.path.isdir(actual_input_root_for_target_folder):
                    _logger.error(f"Move Original: Cannot determine a valid root directory for target folder based on input path '{current_video_path}'.")
                    _logger.error(f"ERROR moving original '{original_basename}': Cannot determine valid root for target folder.")
                    return

            destination_dir = os.path.join(actual_input_root_for_target_folder, target_subfolder)
            os.makedirs(destination_dir, exist_ok=True)
            
            dest_filename = os.path.basename(current_video_path)
            dest_path = os.path.join(destination_dir, dest_filename)

            if os.path.exists(current_video_path):
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(dest_filename) 
                    new_dest_name = f"{base}{time.strftime('_%Y%m%d%H%M%S')}{ext}"
                    dest_path = os.path.join(destination_dir, new_dest_name)
                    _logger.info(f"Move Original: Destination already exists. Renaming '{dest_filename}' to '{new_dest_name}'.")
                
                shutil.move(current_video_path, dest_path)
                _logger.debug(f"Successfully moved original source '{dest_filename}' to '{target_subfolder}' folder.")
            else:
                _logger.warning(f"Move Original: Source path to move does not exist: {current_video_path}")
        except Exception as e:
            _logger.exception(f"ERROR moving original '{original_basename}': {e}")

    def _process_single_job(self, demo, job_info, master_meta_for_this_vid):
        
        returned_job_specific_metadata = {}
        job_successful = False
        is_segment_job = job_info.get("is_segment", False)
        original_basename = job_info["original_basename"]
        
        snapshotted_settings = master_meta_for_this_vid["global_processing_settings"]
        guidance_scale_for_job = snapshotted_settings["guidance_scale"]
        inference_steps_for_job = snapshotted_settings["inference_steps"]
        seed_for_job = snapshotted_settings["seed_setting"]
        process_length_for_run_param = snapshotted_settings["process_max_frames_setting"] if not is_segment_job else -1
        
        window_size_for_pipe_call = snapshotted_settings["gui_window_size_setting"]
        overlap_for_pipe_call = snapshotted_settings["gui_overlap_setting"]

        try:
            keep_npz_for_this_job_run = False
            if is_segment_job:
                if self.keep_intermediate_npz_var.get():
                    min_frames_thresh = self.min_frames_to_keep_npz_var.get()
                    orig_vid_frame_count = job_info.get("original_video_raw_frame_count", 0)
                    if min_frames_thresh <= 0 or orig_vid_frame_count >= min_frames_thresh:
                        keep_npz_for_this_job_run = True
            
            saved_data_filepath, returned_job_specific_metadata = demo.run(
                video_path_or_frames_or_info=job_info,
                num_denoising_steps=inference_steps_for_job, 
                guidance_scale=guidance_scale_for_job,
                base_output_folder=self.output_dir.get(), 
                gui_window_size=window_size_for_pipe_call,
                gui_overlap=overlap_for_pipe_call, 
                process_length_for_read_full_video=process_length_for_run_param, 
                target_height=self.target_height.get(),
                target_width=self.target_width.get(),
                seed=seed_for_job, 
                original_video_basename_override=original_basename,
                segment_job_info_param=job_info if is_segment_job else None,
                keep_intermediate_npz_config=keep_npz_for_this_job_run,
                intermediate_segment_visual_format_config=self.keep_intermediate_segment_visual_format_var.get(),
                save_final_json_for_this_job_config=self.save_final_output_json_var.get()
            )
            if not returned_job_specific_metadata:
                returned_job_specific_metadata = {"status": "failure_no_metadata_from_run"}
                _logger.warning(f"Warning: No job-specific metadata returned from run for {original_basename}.")
            
            if saved_data_filepath and returned_job_specific_metadata.get("status") == "success":
                job_successful = True
            else:
                log_msg_prefix_local = f"Segment {job_info.get('segment_id', -1)+1}/{job_info.get('total_segments', 0)}" if is_segment_job else "Full video"
                _logger.info(f"  Job for {original_basename} ({log_msg_prefix_local}) status: {returned_job_specific_metadata.get('status', 'unknown_status')}")
        
        except Exception as e:
            if not returned_job_specific_metadata: returned_job_specific_metadata = {}
            returned_job_specific_metadata["status"] = "exception_in_gui_process_single_job"
            returned_job_specific_metadata["error_message"] = str(e)
            log_msg_prefix_local = f"Segment {job_info.get('segment_id', -1)+1}/{job_info.get('total_segments', 0)}" if is_segment_job else "Full video"
            _logger.exception(f"  Exception during job for {original_basename} ({log_msg_prefix_local}): {e}")
            self.status_message_var.set(f"Error: {e.__class__.__name__} during {original_basename}")
        return job_successful, returned_job_specific_metadata

    def _recolor_tk_widgets(self, parent, bg_color, fg_color, entry_bg):
        """Recursively recolors raw tk widgets within a parent container."""
        for widget in parent.winfo_children():
            widget_type = widget.winfo_class()
            try:
                # Basic widgets that support bg/fg config
                if widget_type in ('Label', 'Checkbutton'):
                    widget.config(bg=bg_color, fg=fg_color)
                elif widget_type == 'Entry':
                    widget.config(bg=entry_bg, fg=fg_color, insertbackground=fg_color)
                # Buttons usually look better controlled by the theme/style
                # elif widget_type == 'Button':
                #     widget.config(bg=bg_color, fg=fg_color) 
                # Containers
                elif widget_type in ('Frame', 'Toplevel', 'Menubutton'):
                    widget.config(bg=bg_color)
                # LabelFrame title
                elif widget_type == 'Labelframe':
                    widget.config(bg=bg_color, fg=fg_color)
            except tk.TclError:
                # Some widgets (like a tk.Text in a Log window, if you had one)
                # or ttk widgets passed to this function will raise an error. Ignore.
                pass 
            
            # Recurse into children
            self._recolor_tk_widgets(widget, bg_color, fg_color, entry_bg)

    def _reset_settings_to_defaults(self):
        if messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to their default values?"):
            self._apply_all_settings(self.initial_default_settings)
            _logger.info("All settings have been reset to their initial defaults.")
            self.status_message_var.set("Settings reset to defaults.")

    def _restore_input_files(self, folder_type: str): # Added folder_type argument
        """Moves original input files from a specified 'finished' or 'failed' subfolder back to their input directory."""
        display_folder_name = folder_type.capitalize() # "Finished" or "Failed"

        if not messagebox.askyesno(f"Restore {display_folder_name} Input Files", 
                                   f"Are you sure you want to move original input files from the '{display_folder_name}' subfolder "
                                   f"back to their original input directory?"):
            _logger.info(f"Restore {display_folder_name} operation cancelled by user confirmation.")
            return

        source_input_path = self.input_dir_or_file_var.get()

        if not os.path.isdir(source_input_path):
            messagebox.showerror("Restore Error", f"Restore '{display_folder_name}' input files operation is only applicable when 'Input Folder/File' is set to a directory (batch mode).")
            _logger.warning(f"Restore {display_folder_name} operation skipped: Input Folder/File is not a directory: {source_input_path}")
            self.status_message_var.set(f"Restore {display_folder_name} failed: Input is not a directory.")
            return

        restored_count = 0
        errors_count = 0
        
        # Only process the specified folder_type
        finished_source_folder = os.path.join(source_input_path, folder_type) # Use folder_type directly
        
        if os.path.isdir(finished_source_folder):
            _logger.info(f"==> Restoring input files from: {finished_source_folder}")
            for filename in os.listdir(finished_source_folder):
                src_path = os.path.join(finished_source_folder, filename)
                dest_path = os.path.join(source_input_path, filename) 
                
                if os.path.isfile(src_path):
                    try:
                        if os.path.exists(dest_path):
                            base, ext = os.path.splitext(filename)
                            new_filename = f"{base}_restored_{time.strftime('%Y%m%d%H%M%S')}{ext}"
                            dest_path = os.path.join(source_input_path, new_filename)
                            _logger.warning(f"Input file '{filename}' already exists in '{source_input_path}'. Restoring as '{new_filename}'.")
                        
                        shutil.move(src_path, dest_path)
                        restored_count += 1
                        _logger.debug(f"Moved input file '{filename}' to '{source_input_path}'")
                    except Exception as e:
                        errors_count += 1
                        _logger.error(f"Error moving input file '{filename}' from '{finished_source_folder}': {e}", exc_info=True)
            
            # Clean up empty folder after restoring
            try:
                if not os.listdir(finished_source_folder):
                    os.rmdir(finished_source_folder)
                    _logger.info(f"Removed empty folder: {finished_source_folder}")
            except OSError as e:
                _logger.warning(f"Could not remove empty folder '{finished_source_folder}': {e}")
        else:
            _logger.info(f"==> Input '{display_folder_name}' folder not found: {finished_source_folder}")


        # Final status update
        if restored_count > 0 or errors_count > 0:
            self.status_message_var.set(f"Restore {display_folder_name} complete: {restored_count} input files moved, {errors_count} errors.")
            messagebox.showinfo("Restoration Complete", 
                                f"{display_folder_name} input files restoration attempted.\n"
                                f"Successfully restored: {restored_count} file(s)\n"
                                f"Skipped (due to error/conflict): {errors_count} file(s)")
        else:
            self.status_message_var.set(f"No {display_folder_name.lower()} input files found to restore.")
            messagebox.showinfo("Restoration Complete", f"No input files found in the '{display_folder_name}' folder to restore.")

    def _save_all_settings_as(self):
        filepath = filedialog.asksaveasfilename(title="Save Settings As", filetypes=self.SETTINGS_FILETYPES, defaultextension=".json", initialdir=self.last_settings_dir)
        if not filepath:
            _logger.info("Save settings cancelled by user.")
            return
        self.last_settings_dir = os.path.dirname(filepath)
        current_settings = self._collect_all_settings()
        if save_json_file(current_settings, filepath, indent=4):
            _logger.info(f"Successfully saved settings to: {filepath}")
            messagebox.showinfo("Save Successful", f"Settings saved to:\n{filepath}")
        else:
            messagebox.showerror("Save Error", f"Could not save settings to:\n{filepath}\nSee console log for details.")

    def _save_final_output_sidecar_json(self, original_basename, final_merged_path, master_meta_filepath, master_meta, was_segments, merge_successful):
        json_path, json_content = None, {}
        output_suffix_val = self.merge_output_suffix_var.get()

        if was_segments:
            if merge_successful and final_merged_path and not final_merged_path.startswith("N/A"):
                out_fmt_selected = self.merge_output_format_var.get()
                
                json_content = {
                    "source_video_basename": original_basename, "processing_mode": "segmented_then_merged",
                    "final_output_path": os.path.abspath(final_merged_path), 
                    "final_output_format_selected": out_fmt_selected,
                    "master_metadata_path_source": os.path.abspath(master_meta_filepath) if master_meta_filepath else None,
                    "global_processing_settings_summary": master_meta.get("global_processing_settings"),
                    "merge_settings_summary": {
                        "output_format_selected": out_fmt_selected, 
                        "output_suffix": output_suffix_val,
                        "alignment_method": self.merge_alignment_method_var.get(),
                        "dithering": self.merge_dither_var.get(), "dither_strength": self.merge_dither_strength_var.get(),
                        "gamma_correction": self.merge_gamma_correct_var.get(),
                        "gamma_value_if_applied": self.merge_gamma_value_var.get() if self.merge_gamma_correct_var.get() else 1.0,
                        "percentile_norm": self.merge_percentile_norm_var.get(),
                        "norm_low_perc": self.merge_norm_low_perc_var.get(), "norm_high_perc": self.merge_norm_high_perc_var.get(),
                    }, "generation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
                if os.path.isdir(final_merged_path):
                    json_path = os.path.join(os.path.dirname(final_merged_path.rstrip(os.sep)), f"{os.path.basename(final_merged_path.rstrip(os.sep))}.json")
                elif os.path.isfile(final_merged_path):
                    json_path = get_sidecar_json_filename(final_merged_path)
                else: _logger.warning(f"    Cannot determine final JSON path for merged {original_basename} (output path: {final_merged_path}).") 
            else: _logger.info(f"  Skipping final JSON for merged {original_basename} (merge not successful/path invalid).") 
        else:
            if master_meta and master_meta.get("jobs_info"):
                job_info = master_meta["jobs_info"][0]
                relative_output_filename = job_info.get("output_video_filename") 
                if relative_output_filename:
                    out_path = os.path.join(self.output_dir.get(), relative_output_filename)
                    out_fmt_from_ext = os.path.splitext(relative_output_filename)[1].lstrip('.') 

                    json_content = {
                        "source_video_basename": original_basename, "processing_mode": "full_video",
                        "final_output_path": os.path.abspath(out_path), 
                        "final_output_format": out_fmt_from_ext,
                        "global_processing_settings": master_meta.get("global_processing_settings"),
                        "job_specific_details": job_info,
                        "generation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }
                    if os.path.isdir(out_path): 
                         json_path = os.path.join(os.path.dirname(out_path.rstrip(os.sep)), f"{os.path.basename(out_path.rstrip(os.sep))}.json")
                    elif os.path.isfile(out_path):
                        json_path = get_sidecar_json_filename(out_path)
                    else: _logger.warning(f"    Cannot determine final JSON path for full video {original_basename} (output path: {out_path}).") 
                else: _logger.warning(f"  Skipping final JSON for full video {original_basename} (output path/format missing from job_info).") 
            else: _logger.warning(f"  Skipping final JSON for full video {original_basename} (master_meta or job_info missing).") 

        if json_path and json_content:
            _logger.debug(f"    Attempting to save final output JSON to: {json_path}") 
            if save_json_file(json_content, json_path): 
                _logger.debug(f"  Successfully saved sidecar JSON for final output: {json_path}") 
        elif self.save_final_output_json_var.get():
            mode = "merged" if was_segments else "full video"
            _logger.warning(f"  Final output JSON for {mode} '{original_basename}' not created (conditions not met, or save failed).")

    def _save_master_metadata_and_cleanup_segment_json(self, master_meta_to_save, original_basename, main_output_dir, was_segments, segment_subfolder_path):
        master_meta_filepath, meta_saved = None, False
        if was_segments:
            if not segment_subfolder_path:
                segment_subfolder_path = os.path.join(main_output_dir, get_segment_output_folder_name(original_basename))
            os.makedirs(segment_subfolder_path, exist_ok=True)
            master_meta_filepath = os.path.join(segment_subfolder_path, f"{original_basename}_master_meta.json")
        else:
            master_meta_filepath = os.path.join(main_output_dir, f"{original_basename}_master_meta.json")

        should_save_master_meta_here = was_segments
        if should_save_master_meta_here:
            if save_json_file(master_meta_to_save, master_meta_filepath):
                _logger.debug(f"Saved master metadata for {original_basename} to {master_meta_filepath}")
                meta_saved = True
            if was_segments and meta_saved and segment_subfolder_path:
                _logger.debug(f"  Attempting to delete individual segment JSONs for {original_basename} (master created).")
                deleted_count = 0
                for job_data in master_meta_to_save.get("jobs_info", []):
                    npz_file = job_data.get("output_segment_filename")
                    if npz_file:
                        json_to_del = os.path.join(segment_subfolder_path, get_sidecar_json_filename(npz_file))
                        if os.path.exists(json_to_del):
                            try: os.remove(json_to_del); deleted_count += 1
                            except Exception as e: _logger.error(f"ERROR deleting individual segment JSON {json_to_del}: {e}")
                _logger.debug(f"    Deleted {deleted_count} individual segment JSONs.")
            elif was_segments and not meta_saved:
                _logger.warning(f"  Skipping deletion of individual segment JSONs for {original_basename} (master_meta.json not saved).")
        elif not was_segments:
            _logger.debug(f"Skipping save of '{os.path.basename(master_meta_filepath)}' by _save_master_metadata for full video mode for {original_basename}.")
            meta_saved = False
        return master_meta_filepath, meta_saved

    def _set_ui_processing_state(self, is_processing: bool):
        new_state = tk.DISABLED if is_processing else tk.NORMAL
        cancel_state = tk.NORMAL if is_processing else tk.DISABLED
        unique_widgets = list(set(self.widgets_to_disable_during_processing))

        for widget in unique_widgets:
            if widget == self.cancel_button: continue
            if hasattr(widget, 'configure'):
                try:
                    if isinstance(widget, ttk.Combobox): widget.configure(state='disabled' if is_processing else 'readonly')
                    else: widget.configure(state=new_state)
                except tk.TclError: pass 
                
        if hasattr(self, 'help_menu') and self.help_menu:
            try:
                self.help_menu.entryconfig("Enable Debug Logging", state=new_state)
            except tk.TclError: pass
        
        if hasattr(self, 'cancel_button') and self.cancel_button:
             try: self.cancel_button.configure(state=cancel_state)
             except tk.TclError: pass

        if hasattr(self, 'file_menu'):
            try:
                self.file_menu.entryconfig("Use Local Models Only", state=new_state)
                for item_label in ["Load Settings...", "Save Settings As...", "Reset Settings to Default"]:
                    self.file_menu.entryconfig(item_label, state=new_state)
            except tk.TclError: pass

        # --- Phase 2: If processing has *finished*, re-evaluate conditional states ---
        # This prevents conditional toggles from overriding the DISABLED state prematurely.
        if not is_processing:
            self.toggle_merge_related_options_active_state()
            self.toggle_secondary_output_options_active_state()

    def _show_help_for(self, help_key: str):
        """Displays help content for a given key in a Tkinter Toplevel window."""
        # _help_data should already be loaded by __init__
        content = self._help_data.get(help_key)
        
        if not content:
            messagebox.showinfo("Help Not Found", f"No help information available for '{help_key}'.\nEnsure '{DepthCrafterGUI.HELP_CONTENT_FILENAME}' is present and contains this key.")
            _logger.warning(f"No help content found for key: '{help_key}' in {DepthCrafterGUI.HELP_CONTENT_FILENAME}.")
            return

        help_title = content.get("title", "Help Information")
        help_text_str = content.get("text", "No details available.")
        
        # Now, create the Toplevel window as it was before
        help_window = tk.Toplevel(self.root)
        help_window.title(help_title)
        help_window.minsize(400, 200)
        help_window.transient(self.root)
        help_window.grab_set()
        
        text_frame = ttk.Frame(help_window, padding="10")
        text_frame.pack(expand=True, fill="both")
        
        help_text_widget = tk.Text(text_frame, wrap=tk.WORD, relief="flat", borderwidth=0, padx=5, pady=5, font=("Segoe UI", 9))
        help_text_widget.insert(tk.END, help_text_str)
        help_text_widget.config(state=tk.DISABLED)
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=help_text_widget.yview)
        help_text_widget['yscrollcommand'] = scrollbar.set
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        help_text_widget.pack(side=tk.LEFT, expand=True, fill="both")
        
        button_frame = ttk.Frame(help_window, padding=(0, 5, 0, 10))
        button_frame.pack(fill=tk.X)
        ok_button = ttk.Button(button_frame, text="OK", command=help_window.destroy, width=10)
        ok_button.pack()
        
        self.root.update_idletasks()
        help_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (help_window.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (help_window.winfo_height() // 2)
        help_window.geometry(f"+{x}+{y}")
        ok_button.focus_set()
        help_window.wait_window()
        _logger.debug(f"Displayed help overview for '{help_key}'.")
        
    def _start_processing_wrapper(self, source_specs_to_process, effective_seed_for_run):
        try: 
            self.start_processing(source_specs_to_process, effective_seed_for_run)
        finally: 
            self._set_ui_processing_state(False)

    def _toggle_debug_logging(self):
        if self.debug_logging_enabled.get():
            logging.getLogger().setLevel(logging.DEBUG) # Set root logger to DEBUG
            _logger.info("Debug logging ENABLED.")
        else:
            logging.getLogger().setLevel(logging.INFO)  # Set root logger back to INFO
            _logger.info("Debug logging DISABLED (set to INFO level).")

    def _update_gui_info_on_job_start(self, job_info_to_run, original_basename, log_msg_prefix):
        """Updates GUI processing info labels with target/expected values before a job starts."""
        _logger.debug(f"DEBUG GUI UPDATE (Initial): Starting update for {original_basename}")
        
        self.current_filename_var.set(f"{original_basename} ({log_msg_prefix})") 
        
        # Initial Resolution (using target_height/width setting and original dimensions as a hint)
        target_h_setting = self.target_height.get()
        target_w_setting = self.target_width.get()
        is_segment_job = job_info_to_run.get("is_segment", False)

        initial_display_res = "N/A"
        if target_h_setting > 0 and target_w_setting > 0:
            initial_display_res = f"{target_w_setting}x{target_h_setting}"
        else:
            # Fallback to detected original dimensions if target H/W are not set
            original_h = job_info_to_run.get("original_height", "N/A")
            original_w = job_info_to_run.get("original_width", "N/A")
            if original_h != "N/A" and original_w != "N/A":
                initial_display_res = f"{original_w}x{original_h} (Original/Fallback)"
        self.current_resolution_var.set(initial_display_res)

        # Initial Frames (using gui settings for segment/process_length)
        total_frames_orig_vid = job_info_to_run.get('original_video_raw_frame_count', 'N/A')
        
        initial_display_frames_str = "N/A"
        if is_segment_job:
            num_frames_to_load_raw = job_info_to_run.get("num_frames_to_load_raw", "N/A")
            if num_frames_to_load_raw != "N/A":
                initial_display_frames_str = f"{num_frames_to_load_raw}"
            
            if total_frames_orig_vid != "N/A" and str(num_frames_to_load_raw) != str(total_frames_orig_vid):
                initial_display_frames_str += f" of {total_frames_orig_vid}"
        else: # Full video
            process_length_setting = self.process_length.get()
            if process_length_setting != -1:
                initial_display_frames_str = f"{process_length_setting}"
            else:
                initial_display_frames_str = f"{total_frames_orig_vid}"
        self.current_frames_var.set(initial_display_frames_str)

        self.root.update_idletasks() # Force GUI update for initial display
        
    def _update_gui_info_on_job_finish(self, job_info_to_run, current_job_specific_metadata):
        """Updates GUI processing info labels with actual/processed values after a job finishes."""
        # _logger.debug(f"DEBUG GUI UPDATE (Final): Starting update for {job_info_to_run['original_basename']}")

        # RESOLUTION UPDATE (from actual processed values)
        processed_h = current_job_specific_metadata.get("processed_height", "N/A")
        processed_w = current_job_specific_metadata.get("processed_width", "N/A")
        
        final_display_res = "N/A"
        if processed_h != "N/A" and processed_w != "N/A":
            final_display_res = f"{processed_w}x{processed_h}" # This is the actual processed resolution
        else:
            final_display_res = self.current_resolution_var.get() + " (Failed to confirm)" # Append if couldn't get actual
        self.current_resolution_var.set(final_display_res)
        
        # FRAMES UPDATE (from actual processed values)
        processed_frames_for_job = current_job_specific_metadata.get("frames_in_output_video", "N/A")
        total_frames_orig_vid = job_info_to_run.get('original_video_raw_frame_count', 'N/A')
        is_segment_job = job_info_to_run.get("is_segment", False)
        
        final_display_frames_str = "N/A"

        if processed_frames_for_job != "N/A":
            if is_segment_job:
                final_display_frames_str = f"{processed_frames_for_job}"
                if total_frames_orig_vid != "N/A" and str(processed_frames_for_job) != str(total_frames_orig_vid):
                    final_display_frames_str += f" of {total_frames_orig_vid} total"
            else: # Full video processing
                final_display_frames_str = f"{processed_frames_for_job}"
                if total_frames_orig_vid != "N/A" and str(processed_frames_for_job) != str(total_frames_orig_vid):
                    final_display_frames_str += f" (of {total_frames_orig_vid} total)"
        else:
            final_display_frames_str = self.current_frames_var.get() + " (Failed to confirm)" # Append if couldn't get actual
        
        self.current_frames_var.set(final_display_frames_str)
        self.root.update_idletasks() # Force GUI update for final display
    
    def add_param(self, parent, label, var, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry = ttk.Entry(parent, textvariable=var, width=20)
        entry.grid(row=row, column=1, padx=5, pady=2, sticky="w")
        return entry

    def browse_input_folder(self):
        folder = filedialog.askdirectory(initialdir=self.input_dir_or_file_var.get())
        if folder:
            self.input_dir_or_file_var.set(os.path.normpath(folder))
            self.single_file_mode_active = False
            if self._is_image_sequence_folder(folder):
                self.current_input_mode = "image_sequence_folder"
                _logger.info(f"GUI: Input mode set to Image Sequence Folder: {folder}")
            else:
                self.current_input_mode = "batch_folder"
                _logger.info(f"GUI: Input mode set to Batch Folder: {folder}")

    def browse_output(self):
        folder = filedialog.askdirectory(initialdir=self.output_dir.get())
        if folder: self.output_dir.set(os.path.normpath(folder))

    def browse_single_input_file(self):
        filetypes = [("All Supported", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.gif *.png *.jpg *.jpeg *.bmp *.tiff *.exr"),
                     ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.gif"),
                     ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.exr")]
        
        initial_dir_guess = self.input_dir_or_file_var.get()
        if os.path.isfile(initial_dir_guess): initial_dir_guess = os.path.dirname(initial_dir_guess)
        if not os.path.isdir(initial_dir_guess): initial_dir_guess = os.path.expanduser("~")


        filepath = filedialog.askopenfilename(initialdir=initial_dir_guess, filetypes=filetypes)
        if filepath:
            self.input_dir_or_file_var.set(os.path.normpath(filepath))
            self.single_file_mode_active = True
            ext = os.path.splitext(filepath)[1].lower()
            is_video = any(ext in vid_ext.replace("*", "") for vid_ext in self.VIDEO_EXTENSIONS)
            is_image = any(ext in img_ext.replace("*", "") for img_ext in self.IMAGE_EXTENSIONS)

            if is_video:
                self.current_input_mode = "single_video_file"
                _logger.info(f"GUI: Input mode set to Single Video File: {filepath}")
            elif is_image:
                self.current_input_mode = "single_image_file"
                _logger.info(f"GUI: Input mode set to Single Image File: {filepath}")
            else:
                _logger.warning(f"GUI: Could not determine type of single file: {filepath}. Assuming video.")
                self.current_input_mode = "single_video_file" 
                messagebox.showwarning("Unknown File Type", f"Could not determine if '{os.path.basename(filepath)}' is a video or image. Assuming video.")

    def create_widgets(self):
        self.widgets_to_disable_during_processing = []
        
        # --- Input Source Frame ---
        dir_frame = ttk.LabelFrame(self.root, text="Input Source")
        dir_frame.pack(fill="x", padx=10, pady=5, expand=False)        
        
        ttk.Label(dir_frame, text="Input Folder/File:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.entry_input_dir_or_file = ttk.Entry(dir_frame, textvariable=self.input_dir_or_file_var, width=50)
        self.entry_input_dir_or_file.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        _create_hover_tooltip(self.entry_input_dir_or_file, "input_dir_or_file") # Tooltip for entry
        
        browse_buttons_frame = ttk.Frame(dir_frame)
        browse_buttons_frame.grid(row=0, column=2, padx=5, pady=0, sticky="w")
        
        self.browse_input_folder_btn = ttk.Button(browse_buttons_frame, text="Browse Folder", command=self.browse_input_folder)
        self.browse_input_folder_btn.pack(side=tk.LEFT, padx=(0,2))
        _create_hover_tooltip(self.browse_input_folder_btn, "browse_input_folder") # Tooltip for button
        
        self.browse_single_file_btn = ttk.Button(browse_buttons_frame, text="Load Single File", command=self.browse_single_input_file)
        self.browse_single_file_btn.pack(side=tk.LEFT, padx=(2,0))
        _create_hover_tooltip(self.browse_single_file_btn, "browse_single_file") # Tooltip for button
        
        dir_frame.columnconfigure(1, weight=1)
        self.widgets_to_disable_during_processing.extend([
            self.entry_input_dir_or_file, 
            self.browse_input_folder_btn, 
            self.browse_single_file_btn
        ])

        ttk.Label(dir_frame, text="Output Folder:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.entry_output_dir = ttk.Entry(dir_frame, textvariable=self.output_dir, width=50)
        self.entry_output_dir.grid(row=1, column=1, padx=5, pady=2)
        _create_hover_tooltip(self.entry_output_dir, "output_dir") # Tooltip for entry
        
        self.browse_output_btn = ttk.Button(dir_frame, text="Browse", command=self.browse_output)
        self.browse_output_btn.grid(row=1, column=2, padx=5, pady=2)
        _create_hover_tooltip(self.browse_output_btn, "browse_output") # Tooltip for button
        
        self.widgets_to_disable_during_processing.extend([self.entry_output_dir, self.browse_output_btn])

        # --- Settings Container Frame (New) ---
        # This frame will hold the Main Params, Frame & Segment Control, Merged Output, and Secondary Output frames.
        settings_container_frame = ttk.Frame(self.root)
        settings_container_frame.pack(fill="x", padx=10, pady=0, expand=False)
        settings_container_frame.columnconfigure(0, weight=1)
        settings_container_frame.columnconfigure(1, weight=1)

        # --- Main Parameters Frame ---
        main_params_frame = ttk.LabelFrame(settings_container_frame, text="Main Parameters")
        main_params_frame.grid(row=0, column=0, padx=(0,5), pady=5, sticky="nsew") # Placed in new container
        row_idx = 0
        
        # Guidance Scale
        ttk.Label(main_params_frame, text="Guidance Scale:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_guidance_scale = ttk.Entry(main_params_frame, textvariable=self.guidance_scale, width=18)
        entry_guidance_scale.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_guidance_scale, "guidance_scale")
        self.widgets_to_disable_during_processing.append(entry_guidance_scale); row_idx += 1
        
        # Inference Steps
        ttk.Label(main_params_frame, text="Inference Steps:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_inference_steps = ttk.Entry(main_params_frame, textvariable=self.inference_steps, width=18)
        entry_inference_steps.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_inference_steps, "inference_steps")
        self.widgets_to_disable_during_processing.append(entry_inference_steps); row_idx += 1

        # Target Width
        ttk.Label(main_params_frame, text="Target Width:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_target_width = ttk.Entry(main_params_frame, textvariable=self.target_width, width=18)
        entry_target_width.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_target_width, "target_width")
        self.widgets_to_disable_during_processing.append(entry_target_width); row_idx += 1

        # Target Height
        ttk.Label(main_params_frame, text="Target Height:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_target_height = ttk.Entry(main_params_frame, textvariable=self.target_height, width=18)
        entry_target_height.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_target_height, "target_height")
        self.widgets_to_disable_during_processing.append(entry_target_height); row_idx += 1

        # Seed
        ttk.Label(main_params_frame, text="Seed:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_seed = ttk.Entry(main_params_frame, textvariable=self.seed, width=18)
        entry_seed.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_seed, "seed")
        self.widgets_to_disable_during_processing.append(entry_seed); row_idx += 1
        
        # CPU Offload Mode
        ttk.Label(main_params_frame, text="CPU Offload Mode:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        self.combo_cpu_offload = ttk.Combobox(main_params_frame, textvariable=self.cpu_offload, values=["model", "sequential", "none"], width=17, state="readonly")
        self.combo_cpu_offload.grid(row=row_idx, column=1, padx=5, pady=2, sticky="w")
        _create_hover_tooltip(self.combo_cpu_offload, "cpu_offload")
        self.widgets_to_disable_during_processing.append(self.combo_cpu_offload); row_idx += 1

        # --- Frame & Segment Control Frame ---
        fs_frame = ttk.LabelFrame(settings_container_frame, text="Frame & Segment Control")
        fs_frame.grid(row=0, column=1, padx=(5,0), pady=5, sticky="nsew") # Placed in new container

        row_idx = 0 
        
        # Window Size
        ttk.Label(fs_frame, text="Window Size:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_window_size = ttk.Entry(fs_frame, textvariable=self.window_size, width=18)
        entry_window_size.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_window_size, "window_size")
        self.widgets_to_disable_during_processing.append(entry_window_size); row_idx += 1
        
        # Overlap
        ttk.Label(fs_frame, text="Overlap:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_overlap = ttk.Entry(fs_frame, textvariable=self.overlap, width=18)
        entry_overlap.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_overlap, "overlap")
        self.widgets_to_disable_during_processing.append(entry_overlap); row_idx += 1
        
        # Target FPS
        ttk.Label(fs_frame, text="Target FPS (-1 Original):").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_target_fps = ttk.Entry(fs_frame, textvariable=self.target_fps, width=18)
        entry_target_fps.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_target_fps, "target_fps")
        self.widgets_to_disable_during_processing.append(entry_target_fps); row_idx += 1
        
        # Process Max Frames
        ttk.Label(fs_frame, text="Process Max Frames (-1 All):").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_process_length = ttk.Entry(fs_frame, textvariable=self.process_length, width=18)
        entry_process_length.grid(row=row_idx, column=1, padx=(5,0), pady=2, sticky="w")
        _create_hover_tooltip(entry_process_length, "process_length")
        self.widgets_to_disable_during_processing.append(entry_process_length); row_idx += 1
        
        # Save Sidecar JSON for Final Output
        self.save_final_json_cb = ttk.Checkbutton(fs_frame, text="Save Sidecar JSON for Final Output", variable=self.save_final_output_json_var)
        self.save_final_json_cb.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.save_final_json_cb, "save_final_json")
        self.widgets_to_disable_during_processing.append(self.save_final_json_cb); row_idx +=1

        # Process as Segments
        self.process_as_segments_cb = ttk.Checkbutton(fs_frame, text="Process as Segments (Low VRAM Mode)", variable=self.process_as_segments_var, command=self.toggle_merge_related_options_active_state)
        self.process_as_segments_cb.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.process_as_segments_cb, "process_as_segments")
        self.widgets_to_disable_during_processing.append(self.process_as_segments_cb); row_idx += 1

        # --- Merged Output Options Frame ---
        merge_opts_frame = ttk.LabelFrame(settings_container_frame, text="Merged Output Options (if segments processed)")
        merge_opts_frame.grid(row=1, column=0, padx=(0,5), pady=5, sticky="nsew") # Placed in new container
        merge_opts_frame.columnconfigure(0, minsize=120) # Ensure column 0 for labels is wide enough
        self.merge_related_widgets_references = []
        self.keep_npz_dependent_widgets = []
        row_idx = 0

        # Keep intermediate NPZ files
        self.keep_npz_cb = ttk.Checkbutton(merge_opts_frame, text="Keep intermediate NPZ", variable=self.keep_intermediate_npz_var, command=self.toggle_keep_npz_dependent_options_state)
        self.keep_npz_cb.grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.keep_npz_cb, "keep_npz")
        self.merge_related_widgets_references.append(self.keep_npz_cb)
        self.widgets_to_disable_during_processing.append(self.keep_npz_cb); row_idx += 1

        # Min Orig. Vid Frames to Keep NPZ
        self.lbl_min_frames_npz = ttk.Label(merge_opts_frame, text="  ↳ Min thesh. to Keep NPZ:")
        self.lbl_min_frames_npz.grid(row=row_idx, column=0, sticky="e", padx=(20,2), pady=2)
        self.entry_min_frames_npz = ttk.Entry(merge_opts_frame, textvariable=self.min_frames_to_keep_npz_var, width=7)
        self.entry_min_frames_npz.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        _create_hover_tooltip(self.entry_min_frames_npz, "min_frames_npz")
        self.keep_npz_dependent_widgets.extend([self.lbl_min_frames_npz, self.entry_min_frames_npz])
        self.widgets_to_disable_during_processing.extend([self.lbl_min_frames_npz, self.entry_min_frames_npz]); row_idx += 1

        # Segment Visual Format
        self.lbl_intermediate_fmt = ttk.Label(merge_opts_frame, text="  ↳ Segment Format:")
        self.lbl_intermediate_fmt.grid(row=row_idx, column=0, sticky="e", padx=(20,2), pady=2)
        combo_intermediate_fmt_values = ["png_sequence", "mp4", "main10_mp4", "none"]
        if OPENEXR_AVAILABLE_GUI: combo_intermediate_fmt_values.extend(["exr_sequence", "exr"])
        self.combo_intermediate_fmt = ttk.Combobox(merge_opts_frame, textvariable=self.keep_intermediate_segment_visual_format_var, values=combo_intermediate_fmt_values, width=17, state="readonly")
        self.combo_intermediate_fmt.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        _create_hover_tooltip(self.combo_intermediate_fmt, "segment_visual_format")
        self.keep_npz_dependent_widgets.extend([self.lbl_intermediate_fmt, self.combo_intermediate_fmt])
        self.widgets_to_disable_during_processing.extend([self.lbl_intermediate_fmt, self.combo_intermediate_fmt]); row_idx += 1
        self.toggle_keep_npz_dependent_options_state()

        # Dithering (MP4)
        self.merge_dither_cb = ttk.Checkbutton(merge_opts_frame, text="Dithering", variable=self.merge_dither_var, command=self.toggle_dither_options_active_state)
        self.merge_dither_cb.grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.merge_dither_cb, "merge_dither") # Tooltip on checkbox
        
        dither_details_frame = ttk.Frame(merge_opts_frame)
        dither_details_frame.grid(row=row_idx, column=1, sticky="w", padx=(0,0))
        self.lbl_dither_str = ttk.Label(dither_details_frame, text="Strength:")
        self.lbl_dither_str.pack(side=tk.LEFT, padx=(0, 2))
        self.entry_dither_str = ttk.Entry(dither_details_frame, textvariable=self.merge_dither_strength_var, width=7)
        self.entry_dither_str.pack(side=tk.LEFT, padx=(0, 0))
        _create_hover_tooltip(self.entry_dither_str, "merge_dither_strength") # Tooltip on entry
        
        self.merge_related_widgets_references.append((self.merge_dither_cb, dither_details_frame))
        self.widgets_to_disable_during_processing.extend([self.merge_dither_cb, self.lbl_dither_str, self.entry_dither_str]); row_idx += 1
        self.toggle_dither_options_active_state() # Call after creation

        # Gamma Correct (MP4)
        self.merge_gamma_cb = ttk.Checkbutton(merge_opts_frame, text="Gamma Adjust", variable=self.merge_gamma_correct_var, command=self.toggle_gamma_options_active_state)
        self.merge_gamma_cb.grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.merge_gamma_cb, "merge_gamma") # Tooltip on checkbox
        
        gamma_details_frame = ttk.Frame(merge_opts_frame)
        gamma_details_frame.grid(row=row_idx, column=1, sticky="w", padx=(0,0))
        self.lbl_gamma_val = ttk.Label(gamma_details_frame, text="Value:")
        self.lbl_gamma_val.pack(side=tk.LEFT, padx=(0, 2))
        self.entry_gamma_val = ttk.Entry(gamma_details_frame, textvariable=self.merge_gamma_value_var, width=7)
        self.entry_gamma_val.pack(side=tk.LEFT, padx=(0, 0))
        _create_hover_tooltip(self.entry_gamma_val, "merge_gamma_value") # Tooltip on entry
        
        self.merge_related_widgets_references.append((self.merge_gamma_cb, gamma_details_frame))
        self.widgets_to_disable_during_processing.extend([self.merge_gamma_cb, self.lbl_gamma_val, self.entry_gamma_val]); row_idx += 1
        self.toggle_gamma_options_active_state() # Call after creation

        # Percentile Normalization
        self.merge_perc_norm_cb = ttk.Checkbutton(merge_opts_frame, text="Normalization", variable=self.merge_percentile_norm_var, command=self.toggle_percentile_norm_options_active_state)
        self.merge_perc_norm_cb.grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.merge_perc_norm_cb, "merge_percentile_norm") # Tooltip on checkbox
        
        low_high_frame = ttk.Frame(merge_opts_frame)
        low_high_frame.grid(row=row_idx, column=1, sticky="w", padx=(0,0))
        self.lbl_low_perc = ttk.Label(low_high_frame, text="Low:")
        self.lbl_low_perc.pack(side=tk.LEFT, padx=(0,2))
        self.entry_low_perc = ttk.Entry(low_high_frame, textvariable=self.merge_norm_low_perc_var, width=7)
        self.entry_low_perc.pack(side=tk.LEFT, padx=(0,10))
        self.lbl_high_perc = ttk.Label(low_high_frame, text="High:")
        self.lbl_high_perc.pack(side=tk.LEFT, padx=(0,2))
        self.entry_high_perc = ttk.Entry(low_high_frame, textvariable=self.merge_norm_high_perc_var, width=7)
        self.entry_high_perc.pack(side=tk.LEFT, padx=(0,0))
        ttk.Label(merge_opts_frame, text="  ↳").grid(row=row_idx, column=0, sticky="e", padx=(10,2)) # Aligns with the checkbox
        _create_hover_tooltip(self.entry_low_perc, "merge_norm_low_perc") # Tooltip on entry
        _create_hover_tooltip(self.entry_high_perc, "merge_norm_high_perc") # Tooltip on entry
        
        self.merge_related_widgets_references.append(self.merge_perc_norm_cb)
        self.widgets_to_disable_during_processing.extend([self.lbl_low_perc, self.entry_low_perc, self.lbl_high_perc, self.entry_high_perc]); row_idx += 1
        self.toggle_percentile_norm_options_active_state()

        # Alignment Method
        lbl_merge_alignment = ttk.Label(merge_opts_frame, text="Alignment Method:")
        lbl_merge_alignment.grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        self.combo_merge_alignment = ttk.Combobox(merge_opts_frame, textvariable=self.merge_alignment_method_var, values=["Shift & Scale", "Linear Blend"], width=17, state="readonly")
        self.combo_merge_alignment.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        _create_hover_tooltip(self.combo_merge_alignment, "merge_alignment_method")
        self.merge_related_widgets_references.append((lbl_merge_alignment, self.combo_merge_alignment))
        self.widgets_to_disable_during_processing.extend([lbl_merge_alignment, self.combo_merge_alignment]); row_idx += 1
        
        # Output Format
        lbl_merge_fmt = ttk.Label(merge_opts_frame, text="Output Format:")
        lbl_merge_fmt.grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        merge_fmt_values = ["mp4", "main10_mp4", "png_sequence"] + (["exr_sequence", "exr"] if OPENEXR_AVAILABLE_GUI else [])
        self.combo_merge_fmt = ttk.Combobox(merge_opts_frame, textvariable=self.merge_output_format_var, values=merge_fmt_values, width=17, state="readonly")
        self.combo_merge_fmt.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        _create_hover_tooltip(self.combo_merge_fmt, "merge_output_format")
        self.merge_related_widgets_references.append((lbl_merge_fmt, self.combo_merge_fmt))
        self.widgets_to_disable_during_processing.extend([lbl_merge_fmt, self.combo_merge_fmt]); row_idx += 1

        # Output Suffix
        lbl_merge_suffix = ttk.Label(merge_opts_frame, text="Output Suffix:")
        lbl_merge_suffix.grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        self.entry_merge_suffix = ttk.Entry(merge_opts_frame, textvariable=self.merge_output_suffix_var, width=18)
        self.entry_merge_suffix.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        _create_hover_tooltip(self.entry_merge_suffix, "merge_output_suffix")
        self.merge_related_widgets_references.append((lbl_merge_suffix, self.entry_merge_suffix))
        self.widgets_to_disable_during_processing.extend([lbl_merge_suffix, self.entry_merge_suffix]); row_idx += 1

        # --- NEW: Secondary Output Frame ---
        secondary_output_frame = ttk.LabelFrame(settings_container_frame, text="Secondary Output")
        secondary_output_frame.grid(row=1, column=1, padx=(5,0), pady=5, sticky="nsew") # Placed in new container
        secondary_output_frame.columnconfigure(0, minsize=140) # Adjust as needed
        
        row_idx = 0
        # Enable Secondary Output Checkbox
        self.enable_secondary_output_cb = ttk.Checkbutton(secondary_output_frame, text="Enable Secondary Output", variable=self.enable_dual_output_robust_norm, command=self.toggle_secondary_output_options_active_state)
        self.enable_secondary_output_cb.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        _create_hover_tooltip(self.enable_secondary_output_cb, "enable_secondary_output") # Add help_content.json entry
        self.widgets_to_disable_during_processing.append(self.enable_secondary_output_cb); row_idx += 1
        
        # Depth Range (0-1) Low / High
        ttk.Label(secondary_output_frame, text="Depth Output Range (0-1):").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        depth_range_frame = ttk.Frame(secondary_output_frame)
        depth_range_frame.grid(row=row_idx, column=1, sticky="w", padx=0, pady=0)
        
        lbl_out_min = ttk.Label(depth_range_frame, text="Low:")
        lbl_out_min.pack(side=tk.LEFT, padx=(0,2))
        entry_out_min = ttk.Entry(depth_range_frame, textvariable=self.robust_norm_output_min, width=7)
        entry_out_min.pack(side=tk.LEFT, padx=(0,10))
        _create_hover_tooltip(entry_out_min, "robust_norm_output_min") # Add help_content.json entry
        
        lbl_out_max = ttk.Label(depth_range_frame, text="High:")
        lbl_out_max.pack(side=tk.LEFT, padx=(0,2))
        entry_out_max = ttk.Entry(depth_range_frame, textvariable=self.robust_norm_output_max, width=7)
        entry_out_max.pack(side=tk.LEFT, padx=(0,0))
        _create_hover_tooltip(entry_out_max, "robust_norm_output_max") # Add help_content.json entry
        
        self.secondary_output_widgets_references.extend([lbl_out_min, entry_out_min, lbl_out_max, entry_out_max])
        self.widgets_to_disable_during_processing.extend([lbl_out_min, entry_out_min, lbl_out_max, entry_out_max]); row_idx += 1

        # Normalize % Low / High
        ttk.Label(secondary_output_frame, text="Clipped Output % Range:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        norm_perc_frame = ttk.Frame(secondary_output_frame)
        norm_perc_frame.grid(row=row_idx, column=1, sticky="w", padx=0, pady=0)
        
        lbl_norm_low = ttk.Label(norm_perc_frame, text="Low:")
        lbl_norm_low.pack(side=tk.LEFT, padx=(0,2))
        entry_norm_low = ttk.Entry(norm_perc_frame, textvariable=self.robust_norm_low_percentile, width=7)
        entry_norm_low.pack(side=tk.LEFT, padx=(0,10))
        _create_hover_tooltip(entry_norm_low, "robust_norm_low_percentile") # Add help_content.json entry
        
        lbl_norm_high = ttk.Label(norm_perc_frame, text="High:")
        lbl_norm_high.pack(side=tk.LEFT, padx=(0,2))
        entry_norm_high = ttk.Entry(norm_perc_frame, textvariable=self.robust_norm_high_percentile, width=7)
        entry_norm_high.pack(side=tk.LEFT, padx=(0,0))
        _create_hover_tooltip(entry_norm_high, "robust_norm_high_percentile") # Add help_content.json entry
        
        self.secondary_output_widgets_references.extend([lbl_norm_low, entry_norm_low, lbl_norm_high, entry_norm_high])
        self.widgets_to_disable_during_processing.extend([lbl_norm_low, entry_norm_low, lbl_norm_high, entry_norm_high]); row_idx += 1

        # Output Suffix
        lbl_robust_suffix = ttk.Label(secondary_output_frame, text="Output Suffix:")
        lbl_robust_suffix.grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        entry_robust_suffix = ttk.Entry(secondary_output_frame, textvariable=self.robust_output_suffix, width=18)
        entry_robust_suffix.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        _create_hover_tooltip(entry_robust_suffix, "robust_output_suffix") # Add help_content.json entry
        self.secondary_output_widgets_references.extend([lbl_robust_suffix, entry_robust_suffix])
        self.widgets_to_disable_during_processing.extend([lbl_robust_suffix, entry_robust_suffix]); row_idx += 1

        # --- Progress Bar and Status ---
        progress_bar_frame = ttk.Frame(self.root)
        progress_bar_frame.pack(pady=(10, 0), padx=10, fill="x", expand=False)
        
        self.progress = ttk.Progressbar(progress_bar_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(fill=tk.X, expand=True, padx=0, pady=0)

        # Status Label (NEW)
        self.style.configure("Status.TLabel", anchor="center") 
        self.status_label = ttk.Label(progress_bar_frame, text="Ready")
        self.status_label.pack(padx=0, pady=2)

        # --- Control Buttons ---
        ctrl_frame = ttk.Frame(self.root)
        ctrl_frame.pack(pady=(5, 10), padx=10, fill="x", expand=False)

        # --- Container frame for buttons to center them ---
        button_container_frame = ttk.Frame(ctrl_frame)
        button_container_frame.pack(anchor="center") # Centers the button_container_frame within ctrl_frame

        # --- Current Processing Information Frame ---
        processing_info_frame = ttk.LabelFrame(self.root, text="Current Processing Information")
        processing_info_frame.pack(fill="x", padx=10, pady=5, expand=False)
        
        # Grid layout for labels inside this frame
        processing_info_frame.columnconfigure(0, weight=0) # Labels
        processing_info_frame.columnconfigure(1, weight=1) # Values

        row_idx = 0
        # Filename
        ttk.Label(processing_info_frame, text="Filename:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        lbl_filename = ttk.Label(processing_info_frame, textvariable=self.current_filename_var, anchor=tk.W)
        lbl_filename.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=2)
        _create_hover_tooltip(lbl_filename, "current_filename") # Add tooltip
        row_idx += 1

        # Resolution
        ttk.Label(processing_info_frame, text="Resolution:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        lbl_resolution = ttk.Label(processing_info_frame, textvariable=self.current_resolution_var, anchor=tk.W)
        lbl_resolution.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=2)
        _create_hover_tooltip(lbl_resolution, "current_resolution") # Add tooltip
        row_idx += 1

        # Frames
        ttk.Label(processing_info_frame, text="Frames:").grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        lbl_frames = ttk.Label(processing_info_frame, textvariable=self.current_frames_var, anchor=tk.W)
        lbl_frames.grid(row=row_idx, column=1, sticky="ew", padx=5, pady=2)
        _create_hover_tooltip(lbl_frames, "current_frames") # Add tooltip
        row_idx += 1

        start_frame = ttk.Frame(button_container_frame); start_frame.pack(side=tk.LEFT, padx=(0,2))
        self.start_button = ttk.Button(start_frame, text="Start", command=self.start_thread, width=10)
        self.start_button.pack(side=tk.LEFT)
        _create_hover_tooltip(self.start_button, "start_button")        

        cancel_frame = ttk.Frame(button_container_frame); cancel_frame.pack(side=tk.LEFT, padx=(2,2))
        self.cancel_button = ttk.Button(cancel_frame, text="Cancel", command=self.stop_processing, width=10, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT)
        _create_hover_tooltip(self.cancel_button, "cancel_button")

        remerge_frame = ttk.Frame(button_container_frame); remerge_frame.pack(side=tk.LEFT, padx=(2,2))
        self.remerge_button = ttk.Button(remerge_frame, text="Re-Merge Segments", command=self.re_merge_from_gui, width=18)
        self.remerge_button.pack(side=tk.LEFT)
        _create_hover_tooltip(self.remerge_button, "remerge_button")

        genvis_frame = ttk.Frame(button_container_frame); genvis_frame.pack(side=tk.LEFT, padx=(2,2))
        self.generate_visuals_button = ttk.Button(genvis_frame, text="Generate Seg Visuals", command=self.generate_segment_visuals_from_gui, width=20)
        self.generate_visuals_button.pack(side=tk.LEFT)
        _create_hover_tooltip(self.generate_visuals_button, "generate_visuals_button")

        self.widgets_to_disable_during_processing.extend([
            self.start_button, self.remerge_button,
            self.generate_visuals_button
        ])

        # self.toggle_merge_related_options_active_state()

    def generate_segment_visuals_from_gui(self):
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Busy", "Another process is running. Please wait."); return
        meta_file = filedialog.askopenfilename(title="Select Master Metadata JSON for Segment Visual Generation", filetypes=[("JSON files", "*.json"), ("All files", "*.*")], initialdir=self.output_dir.get())
        if not meta_file: 
            _logger.info("Segment visual generation cancelled: No master metadata file selected.")
            return
        vis_fmt = self.keep_intermediate_segment_visual_format_var.get()
        if vis_fmt == "none": 
            messagebox.showinfo("Info", "Segment Visual Format is 'none'. Select a valid format."); return
        if not messagebox.askyesno("Generate/Overwrite Visuals?", f"Generate '{vis_fmt}' visuals for segments in '{os.path.basename(meta_file)}'?\nThis may overwrite existing visuals."):
            _logger.info("Segment visual generation cancelled by user.")
            return
        args = {"master_meta_path": meta_file, "visual_format_to_generate": vis_fmt}
        _logger.info(f"--- Starting Segment Visual Generation for: {os.path.basename(meta_file)} (Format: {vis_fmt}) ---")
        self._set_ui_processing_state(True)
        self.processing_thread = threading.Thread(target=self._execute_generate_segment_visuals_wrapper, args=(args,), daemon=True); self.processing_thread.start()
        self.root.after(100, self.process_queue)

    def load_config(self):
        if os.path.exists(self.CONFIG_FILENAME):
            try:
                with open(self.CONFIG_FILENAME, "r") as f: config = json.load(f)
                loaded_settings_for_tkvars = {k: v for k, v in config.items() if k in self.all_tk_vars}
                for key, value in loaded_settings_for_tkvars.items():
                    if key in self.all_tk_vars:
                        try: self.all_tk_vars[key].set(value)
                        except tk.TclError: 
                            _logger.warning(f"Warning (GUI load_config): Could not set var {key} during early config load.")
                
                self.last_settings_dir = config.get(self.LAST_SETTINGS_DIR_CONFIG_KEY, os.getcwd())
                
                self.current_input_mode = config.get("current_input_mode", "batch_folder")
                self.single_file_mode_active = config.get("single_file_mode_active", False)
                
                _logger.info(f"GUI: Configuration loaded from '{self.CONFIG_FILENAME}'.")
            except Exception as e:
                _logger.warning(f"Warning (GUI load_config): Could not load config '{self.CONFIG_FILENAME}': {e}")
                self.last_settings_dir = os.getcwd()
                self.current_input_mode = "batch_folder"
                self.single_file_mode_active = False
        else: 
            self.last_settings_dir = os.getcwd()
            self.current_input_mode = "batch_folder"
            self.single_file_mode_active = False
            _logger.info(f"GUI: Configuration file '{self.CONFIG_FILENAME}' not found. Using default settings.")

    def on_close(self):
        self.save_config()
        if self.processing_thread and self.processing_thread.is_alive():
            _logger.info("Stopping processing before exit...")
            self.stop_event.set()
            self.processing_thread.join(timeout=10)
            if self.processing_thread.is_alive(): 
                _logger.warning("Processing thread did not terminate gracefully. Forcing exit.")
        
        self.root.destroy()

    def process_queue(self):
        # The message queue is still used for progress bar updates
        while not self.message_queue.empty():
            try:
                msg_type, content = self.message_queue.get_nowait()
                if msg_type == "progress":
                    self.progress["value"] = content
                elif msg_type == "status":
                    self.status_message_var.set(content)
                elif msg_type == "set_ui_state":
                    self._set_ui_processing_state(content)
            except queue.Empty:
                break
            except Exception as e:
                _logger.exception(f"Error processing GUI queue: {e}")
        
        self.root.after(100, self.process_queue)

    def re_merge_from_gui(self):
        if not merge_depth_segments: 
            messagebox.showerror("Error", "Merge module not available."); return
        meta_file = filedialog.askopenfilename(title="Select Master Metadata JSON for Re-Merging", filetypes=[("JSON files", "*.json"), ("All files", "*.*")], initialdir=self.output_dir.get())
        if not meta_file: return
        
        _logger.debug(f"DEBUG (re_merge_from_gui): enable_dual_output_robust_norm.get() is {self.enable_dual_output_robust_norm.get()}")
        
        base_name_from_meta = os.path.splitext(os.path.basename(meta_file))[0].replace("_master_meta", "")
        output_suffix = self.merge_output_suffix_var.get()
        remerge_base_name = f"{base_name_from_meta}{output_suffix}"

        out_fmt = self.merge_output_format_var.get()
        
        def_ext_fmt = out_fmt
        if out_fmt == "main10_mp4":
            def_ext_fmt = "mp4"
        elif out_fmt in ["png_sequence", "exr_sequence"]:
            def_ext_fmt = ""
        elif out_fmt == "exr":
            def_ext_fmt = "exr"

        def_ext = f".{def_ext_fmt}" if def_ext_fmt else ""

        ftypes_map = {
            "mp4": [("MP4 (H.264 8-bit)", "*.mp4")],
            "main10_mp4": [("MP4 (HEVC 10-bit)", "*.mp4")],
            "png_sequence": [("PNG Seq (Select Folder)", "")],
            "exr_sequence": [("EXR Seq (Select Folder)", "")],
            "exr": [("EXR File", "*.exr")]
        }
        curr_ftypes = ftypes_map.get(out_fmt, []) + [("All files", "*.*")]
        out_path = None

        if "sequence" in out_fmt:
            parent_dir = filedialog.askdirectory(title=f"Select Parent Dir for Re-Merged {out_fmt.upper()} Sequence...", initialdir=self.output_dir.get())
            if parent_dir: out_path = parent_dir
        else:
            initial_filename_for_dialog_actual = f"{remerge_base_name}{def_ext}"

            out_path = filedialog.asksaveasfilename(
                title=f"Save Re-Merged {out_fmt.upper()} As...", 
                initialdir=self.output_dir.get(), 
                initialfile=f"{remerge_base_name}{def_ext}",
                defaultextension=def_ext, 
                filetypes=curr_ftypes
            )

        if not out_path: 
            _logger.info("Re-merge cancelled: No output path selected.")
            return
            
        align_method = "linear_blend" if self.merge_alignment_method_var.get() == "Linear Blend" else "shift_scale"
        
        args = {"master_meta_path": meta_file, "output_path_arg": out_path,
                "do_dithering": self.merge_dither_var.get(), "dither_strength_factor": self.merge_dither_strength_var.get(),
                "apply_gamma_correction": self.merge_gamma_correct_var.get(), "gamma_value": self.merge_gamma_value_var.get(),
                "use_percentile_norm": self.merge_percentile_norm_var.get(), "norm_low_percentile": self.merge_norm_low_perc_var.get(),
                "norm_high_percentile": self.merge_norm_high_perc_var.get(), "output_format": out_fmt,
                "merge_alignment_method": align_method,
                "output_filename_override_base": remerge_base_name,
                "enable_dual_output_robust_norm": self.enable_dual_output_robust_norm.get(),
                "robust_norm_low_percentile": self.robust_norm_low_percentile.get(),
                "robust_norm_high_percentile": self.robust_norm_high_percentile.get(),
                "robust_norm_output_min": self.robust_norm_output_min.get(),
                "robust_norm_output_max": self.robust_norm_output_max.get(),
                "robust_output_suffix": self.robust_output_suffix.get(),
                "is_depth_far_black": self.is_depth_far_black.get()
                }

        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Busy", "Another process is running. Please wait."); return
        
        _logger.info(f"--- Starting Re-Merge for: {os.path.basename(meta_file)} ---")
        self._set_ui_processing_state(True)
        self.processing_thread = threading.Thread(target=self._execute_re_merge_wrapper, args=(args,), daemon=True); self.processing_thread.start()
        self.root.after(100, self.process_queue)

    def start_thread(self):
        if self.processing_thread and self.processing_thread.is_alive():
            _logger.warning("Processing is already running.")
            return

        input_path_str = self.input_dir_or_file_var.get()
        if not input_path_str or not os.path.exists(input_path_str):
            _logger.error(f"GUI: Input path field is empty or path does not exist: {input_path_str}")
            messagebox.showerror("Error", f"Input path does not exist: {input_path_str}")
            return
        
        # --- ADD THESE LINES HERE ---
        _logger.info("Scanning input folder: Please wait...")
        self.status_message_var.set("Scanning input folder...")
        self.root.update_idletasks() # Force GUI update to show "Scanning..." immediately
        # ----------------------------

        determined_mode, determined_single_source = self._determine_input_mode_from_path(input_path_str)
        
        self.current_input_mode = determined_mode
        self.single_file_mode_active = determined_single_source

        if not os.path.exists(input_path_str):
            _logger.error(f"GUI: Input path is invalid or does not exist: {input_path_str}")
            messagebox.showerror("Error", f"Input path does not exist: {input_path_str}")
            return

        sources_to_process_specs = []

        if self.single_file_mode_active:
            self.effective_move_original_on_completion = False
            basename = ""
            if self.current_input_mode == "image_sequence_folder":
                basename = os.path.basename(input_path_str)
            else:
                basename = os.path.splitext(os.path.basename(input_path_str))[0]
            
            sources_to_process_specs.append({
                "path": input_path_str,
                "type": self.current_input_mode, 
                "basename": basename
            })
        else:
            self.effective_move_original_on_completion = self.MOVE_ORIGINAL_TO_FINISHED_FOLDER_ON_COMPLETION
            if self.current_input_mode == "batch_folder":
                try:
                    for item_name in os.listdir(input_path_str):
                        item_full_path = os.path.join(input_path_str, item_name)
                        if os.path.isfile(item_full_path):
                            ext = os.path.splitext(item_name)[1].lower()
                            if any(ext in vid_ext.replace("*", "") for vid_ext in self.VIDEO_EXTENSIONS):
                                basename = os.path.splitext(item_name)[0]
                                sources_to_process_specs.append({
                                    "path": item_full_path,
                                    "type": "video_file",
                                    "basename": basename
                                })
                        elif os.path.isdir(item_full_path):
                            if self._is_image_sequence_folder(item_full_path):
                                basename = item_name 
                                sources_to_process_specs.append({
                                    "path": item_full_path,
                                    "type": "image_sequence_folder",
                                    "basename": basename
                                })
                except NotADirectoryError:
                    _logger.error(f"GUI Input: Path '{input_path_str}' is not a directory, but batch processing mode was attempted.")
                    messagebox.showerror("Error", f"Input path is not a directory for batch processing: {input_path_str}")
                    return
                except OSError as e:
                    _logger.error(f"GUI Input: OS error when trying to list directory '{input_path_str}'. Error: {e}")
                    messagebox.showerror("Error", f"Could not read directory contents for '{input_path_str}':\n{e}")
                    return
            else:
                _logger.critical(f"GUI Start Thread: Unexpected mode '{self.current_input_mode}' for path '{input_path_str}' after explicit determination. This indicates a logic error.")
                messagebox.showerror("Internal Error", f"Unexpected input mode '{self.current_input_mode}' for path '{input_path_str}'. Please report this.")
                return


        if not sources_to_process_specs:
            _logger.warning(f"GUI: No valid video files or image sequences found in '{input_path_str}' for mode '{self.current_input_mode}'.")
            return
        
        # --- NEW SEED GENERATION GUARD ---
        gui_seed_setting = self.seed.get()
        effective_seed_for_run = gui_seed_setting
        if effective_seed_for_run < 0:
            effective_seed_for_run = random.randint(0, 2**32 - 1)
            _logger.debug(f"GUI: Seed was set to {gui_seed_setting} (negative). Generating a new random seed for this run: {effective_seed_for_run}")
        else:
            _logger.debug(f"GUI: Using user-specified seed: {effective_seed_for_run}")
        
        # --- PHASE 1: FAST SCAN FOR TOTAL SOURCES (REPLACING HEAVY METADATA/SEGMENT DEFINITION) ---
        final_jobs_to_process_sources = sources_to_process_specs # The list of file/folder specs

        if final_jobs_to_process_sources:
            # --- ADDED THESE LINES TO RESET PREVIOUS JOB INFO ---
            self.current_filename_var.set("N/A")
            self.current_resolution_var.set("N/A")
            self.current_frames_var.set("N/A")
            # --------------------------------------------------
            self.status_message_var.set(f"Starting processing {len(final_jobs_to_process_sources)} files/sequences...")
            self.progress["value"] = 0 # Initialize progress bar value
            self.progress["maximum"] = len(final_jobs_to_process_sources) # Set progress bar max to total files/sources
            self._set_ui_processing_state(True)
            
            # Pass the list of source specs. The ffprobe/segment definition will happen inside start_processing.
            self.processing_thread = threading.Thread(target=self._start_processing_wrapper, 
                                                      args=(final_jobs_to_process_sources, effective_seed_for_run), 
                                                      daemon=True)
            self.processing_thread.start()
            self.root.after(100, self.process_queue) # Start queue processing for progress updates
        else:
            _logger.info("No videos/segments to process after considering existing data and user choices (or all skipped).")

    def start_processing(self, source_specs_to_process, effective_seed_for_run):
        self.stop_event.clear()
        
        # Progress max is already set to len(source_specs_to_process) in start_thread
        _logger.debug(f"Starting lazy batch processing for {len(source_specs_to_process)} sources...")
        self.status_message_var.set("Starting processing...")

        # Initialize a dict to store master metadata for each video/sequence path
        all_videos_master_metadata = {}
        base_job_info_map = {} # Map to store base_job_info for each video path

        try:
            # 1. Initialize DepthCrafterDemo (Model Loading)
            if not self.use_local_models_only_var.get():
                _logger.info("Attempting to check model at Hugging Face Hub against local.")
            else:
                _logger.info("Attempting to load local model.")

            disable_xformers_for_run = self.disable_xformers_var.get()

            
            
            # cpu_offload_value = self.cpu_offload.get()
            # if cpu_offload_value == "none":
            #     cpu_offload_value = None
            # _logger.info(f"cpu_offload_value = {cpu_offload_value}")

            demo = DepthCrafterDemo(
                unet_path="tencent/DepthCrafter",
                pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
                cpu_offload=self.cpu_offload.get(),
                use_cudnn_benchmark=self.use_cudnn_benchmark.get(),
                local_files_only=self.use_local_models_only_var.get(),
                disable_xformers=disable_xformers_for_run,
            )
        except Exception as e:
            _logger.exception(f"CRITICAL: Failed to initialize DepthCrafterDemo: {e}")
            self.status_message_var.set(f"Error: Model initialization failed. See console.")
            self.current_filename_var.set("N/A")
            self.current_resolution_var.set("N/A")
            self.current_frames_var.set("N/A")
            return 
        
        total_sources_processed = 0
        
        # 2. Main Loop: Process one source (file/folder) at a time
        for source_idx, source_spec in enumerate(source_specs_to_process):
            if self.stop_event.is_set():
                _logger.info("Processing cancelled by user.")
                self.status_message_var.set("Cancelled.")
                break

            current_video_path = source_spec["path"] 
            original_basename = source_spec["basename"]
            current_gui_mode = source_spec["type"] 
            
            gui_fps_setting = self.target_fps.get()
            gui_len_setting = self.process_length.get()
            gui_win_setting = self.window_size.get()
            gui_ov_setting = self.overlap.get()

            log_msg_base = f"Source {source_idx+1}/{len(source_specs_to_process)}: {original_basename}"
            _logger.debug(f"--- Defining Jobs for {log_msg_base}...")
            self.status_message_var.set(f"Defining jobs for {source_idx+1} of {len(source_specs_to_process)}")
            self.root.update_idletasks()
            
            # Determine source type for define_video_segments
            source_type_for_define = ""
            if current_gui_mode == "single_video_file" or current_gui_mode == "video_file":
                source_type_for_define = "video_file"
            elif current_gui_mode == "image_sequence_folder":
                source_type_for_define = "image_sequence_folder"
            elif current_gui_mode == "single_image_file":
                source_type_for_define = "single_image_file"
            else:
                _logger.error(f"Lazy Job Definition: Unknown source_spec type '{current_gui_mode}' for basename '{original_basename}'. Skipping.")
                total_sources_processed += 1
                self.message_queue.put(("progress", total_sources_processed))
                continue

            # A. FFPROBE/METADATA EXTRACTION (Heavy lifting happens here - Phase 2 start)
            all_potential_segments_for_video, base_job_info_initial = define_video_segments(
                video_path_or_folder=current_video_path,
                original_basename=original_basename,
                gui_target_fps_setting=gui_fps_setting,
                gui_process_length_overall=gui_len_setting,
                gui_segment_output_window_frames=gui_win_setting,
                gui_segment_output_overlap_frames=gui_ov_setting,
                source_type=source_type_for_define,
                gui_target_height_setting=self.target_height.get(),
                gui_target_width_setting=self.target_width.get(),
            )
            
            if not base_job_info_initial:
                _logger.info(f"Skipping {original_basename}: Issues in metadata extraction/segment definition.")
                total_sources_processed += 1
                self.message_queue.put(("progress", total_sources_processed))
                continue
            
            # Store base info (includes raw frame count, fps, etc. gathered by define_video_segments)
            base_job_info_map[current_video_path] = base_job_info_initial.copy()
            
            jobs_to_process_for_this_source = []
            is_segment_processing = self.process_as_segments_var.get()
            
            # B. Decide on the actual job list (segments or full video)
            if is_segment_processing:
                if not all_potential_segments_for_video:
                    reason_skip = "Too short or invalid overlap/settings" if base_job_info_initial.get("original_video_raw_frame_count", 0) > 0 else "Source issue or zero frames/duration"
                    _logger.info(f"Skipping {original_basename}: No segments defined by settings (Reason: {reason_skip}).")
                    total_sources_processed += 1
                    self.message_queue.put(("progress", total_sources_processed))
                    continue
                    
                segment_subfolder_name = get_segment_output_folder_name(original_basename)
                segment_subfolder_path = os.path.join(self.output_dir.get(), segment_subfolder_name)
                current_video_base_info_ref = base_job_info_map[current_video_path]
                
                # This call handles the resume/overwrite logic
                jobs_to_process_for_this_source, action_taken = self._get_segments_to_resume_or_overwrite(
                    current_video_path, original_basename, segment_subfolder_path, 
                    all_potential_segments_for_video, current_video_base_info_ref
                )
                _logger.debug(f"For source '{original_basename}': Action '{action_taken}', {len(jobs_to_process_for_this_source)} segments will be processed.")
                
                if not jobs_to_process_for_this_source and not current_video_base_info_ref.get("pre_existing_successful_jobs"):
                     _logger.info(f"Skipping {original_basename}: Job definition/resume resulted in no segments to process.")
                     total_sources_processed += 1
                     self.message_queue.put(("progress", total_sources_processed))
                     continue
                     
            else: # Full video processing mode
                full_out_check_path = os.path.join(self.output_dir.get(), get_full_video_output_filename(original_basename, "mp4"))
                proceed_full = True
                if os.path.exists(full_out_check_path):
                    if not messagebox.askyesno("Overwrite?", f"An output file for '{original_basename}' might exist (e.g., MP4):\n{full_out_check_path}\n\nOverwrite if it exists?"):
                        _logger.info(f"Skipping {original_basename} (full video processing, user chose not to overwrite).")
                        proceed_full = False
                
                if proceed_full:
                    full_source_job = {
                        **base_job_info_initial,
                        "is_segment": False,
                        "gui_desired_output_window_frames": gui_win_setting, 
                        "gui_desired_output_overlap_frames": gui_ov_setting 
                    }
                    jobs_to_process_for_this_source.append(full_source_job)
                else:
                    total_sources_processed += 1
                    self.message_queue.put(("progress", total_sources_processed))
                    continue # Skip to next source
            
            # C. Initialize Master Metadata for this video/sequence
            total_expected_jobs_overall = len(all_potential_segments_for_video) if is_segment_processing else 1
            all_videos_master_metadata[current_video_path] = self._initialize_master_metadata_entry(
                original_basename, 
                base_job_info_initial,
                total_expected_jobs_overall
            )
            master_meta_for_this_vid = all_videos_master_metadata[current_video_path]
            
            # --- UPDATE THE SNAPSHOTTED SEED HERE ---
            master_meta_for_this_vid["global_processing_settings"]["seed_setting"] = effective_seed_for_run
            
            # Add pre-existing successful segments (only relevant for resume in segment mode)
            pre_existing_successful_segment_metadatas = base_job_info_map[current_video_path].get("pre_existing_successful_jobs", [])
            if pre_existing_successful_segment_metadatas:
                 _logger.debug(f"Loading {len(pre_existing_successful_segment_metadatas)} pre-existing successful segment metadata entries for {original_basename} into current run's master data.")
                 master_meta_for_this_vid["jobs_info"].extend(pre_existing_successful_segment_metadatas)
                 master_meta_for_this_vid["completed_successful_jobs"] += len(pre_existing_successful_segment_metadatas)

            # D. Process the actual jobs (segments or full video)
            total_jobs_for_source = len(jobs_to_process_for_this_source)
            
            for job_idx, job_info_to_run in enumerate(jobs_to_process_for_this_source):
                if self.stop_event.is_set(): break
                
                is_segment_job = job_info_to_run.get("is_segment", False)
                log_msg_prefix = f"Segment {job_info_to_run.get('segment_id', -1)+1}/{job_info_to_run.get('total_segments', 0)} ({job_idx+1}/{total_jobs_for_source})" if is_segment_job else "Full video (1/1)"
                
                # --- START PART A: INITIAL GUI UPDATE (TARGET/EXPECTED VALUES) ---
                self._update_gui_info_on_job_start(job_info_to_run, original_basename, log_msg_prefix)
                # --- END PART A: INITIAL GUI UPDATE ---

                _logger.info(f"Processing {original_basename} - {log_msg_prefix}")
                self.status_message_var.set(f"Processing {source_idx + 1} of {len(source_specs_to_process)} ({log_msg_prefix})")

                job_successful, current_job_specific_metadata = self._process_single_job(demo, job_info_to_run, master_meta_for_this_vid)
                
                if current_job_specific_metadata is None:
                    _logger.error(f"Error: _process_single_job for {original_basename} returned None metadata. Initializing to empty dict.")
                    current_job_specific_metadata = {}

                # --- START PART B: FINAL GUI UPDATE (ACTUAL/PROCESSED VALUES) ---
                self._update_gui_info_on_job_finish(job_info_to_run, current_job_specific_metadata)
                # --- END PART B: FINAL GUI UPDATE ---

                if is_segment_job and "segment_id" not in current_job_specific_metadata:
                    current_job_specific_metadata["segment_id"] = job_info_to_run.get("segment_id", -1)
                
                if "_individual_metadata_path" in current_job_specific_metadata:
                    del current_job_specific_metadata["_individual_metadata_path"]
                
                master_meta_for_this_vid["jobs_info"].append(current_job_specific_metadata)
                
                if job_successful:
                    master_meta_for_this_vid["completed_successful_jobs"] += 1
                else:
                    master_meta_for_this_vid["completed_failed_jobs"] += 1
            
            # E. Finalize the source (merge/cleanup/move original) if all its jobs are accounted for
            total_accounted_for_vid = master_meta_for_this_vid["completed_successful_jobs"] + master_meta_for_this_vid["completed_failed_jobs"]
            
            if total_accounted_for_vid >= master_meta_for_this_vid["total_expected_jobs"]:
                # Finalize only if not cancelled *within* the segment loop
                if not self.stop_event.is_set():
                    self._finalize_video_processing(current_video_path, original_basename, master_meta_for_this_vid)
                else:
                    _logger.info(f"Skipping finalization of {original_basename} due to user cancellation.")

            # F. Update Main Progress Bar (1 unit per source file/folder)
            total_sources_processed += 1
            self.message_queue.put(("progress", total_sources_processed))

        if not self.stop_event.is_set():
            _logger.info("All processing sources complete!")
            self.status_message_var.set("Processing Finished.")
        else:
            self.status_message_var.set("Processing Cancelled.")
        
        # G. Cleanup
        if 'demo' in locals() and demo is not None:
            try:
                if hasattr(demo, 'pipe') and demo.pipe is not None:
                    if hasattr(demo.pipe, 'vae') and demo.pipe.vae is not None: del demo.pipe.vae
                    if hasattr(demo.pipe, 'unet') and demo.pipe.unet is not None: del demo.pipe.unet
                    del demo.pipe
                del demo
                _logger.debug("DepthCrafter model components released.")
            except Exception as e_cleanup:
                _logger.warning(f"Error during DepthCrafter model cleanup: {e_cleanup}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _logger.info("CUDA cache cleared.")

    def stop_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            _logger.info("Cancel request received. Processing will stop after current item.")
            self.stop_event.set()
        else: 
            _logger.info("No processing is currently active to cancel.")

    def save_config(self):
        config = self._collect_all_settings()
        config[self.LAST_SETTINGS_DIR_CONFIG_KEY] = self.last_settings_dir
        
        config["current_input_mode"] = self.current_input_mode 
        config["single_file_mode_active"] = self.single_file_mode_active
        
        try:
            with open(self.CONFIG_FILENAME, "w") as f: json.dump(config, f, indent=4)
        except Exception as e: 
            _logger.warning(f"Warning (GUI save_config): Could not save config: {e}")

    def toggle_dither_options_active_state(self, *args):
        if not (hasattr(self, 'process_as_segments_var') and hasattr(self, 'merge_dither_var')): return
        active = self.process_as_segments_var.get() and self.merge_dither_var.get()
        state = tk.NORMAL if active else tk.DISABLED
        for attr_name in ['lbl_dither_str', 'entry_dither_str']:
            if hasattr(self, attr_name):
                widget = getattr(self, attr_name)
                if widget and hasattr(widget, 'configure'):
                    try: widget.configure(state=state)
                    except tk.TclError: pass

    def toggle_gamma_options_active_state(self, *args):
        if not (hasattr(self, 'process_as_segments_var') and hasattr(self, 'merge_gamma_correct_var')): return
        active = self.process_as_segments_var.get() and self.merge_gamma_correct_var.get()
        state = tk.NORMAL if active else tk.DISABLED
        for attr_name in ['lbl_gamma_val', 'entry_gamma_val']:
            if hasattr(self, attr_name):
                widget = getattr(self, attr_name)
                if widget and hasattr(widget, 'configure'):
                    try: widget.configure(state=state)
                    except tk.TclError: pass

    def toggle_keep_npz_dependent_options_state(self, *args):
        if not (hasattr(self, 'process_as_segments_var') and hasattr(self, 'keep_intermediate_npz_var') and hasattr(self, 'keep_npz_dependent_widgets')):
            return
        active = self.process_as_segments_var.get() and self.keep_intermediate_npz_var.get()
        state = tk.NORMAL if active else tk.DISABLED
        for widget in self.keep_npz_dependent_widgets:
            if hasattr(widget, 'configure'):
                try:
                    if isinstance(widget, ttk.Combobox): widget.configure(state='readonly' if active else 'disabled')
                    else: widget.configure(state=state)
                except tk.TclError: pass

    def toggle_merge_related_options_active_state(self, *args):
        if not hasattr(self, 'process_as_segments_var'): return
        active = self.process_as_segments_var.get()
        current_processing_state = tk.DISABLED
        if hasattr(self, 'start_button') and self.start_button and hasattr(self, 'cancel_button') and self.cancel_button:
            try:
                if self.start_button.cget('state') == tk.DISABLED and self.cancel_button.cget('state') == tk.NORMAL:
                    current_processing_state = tk.DISABLED
                else: current_processing_state = tk.NORMAL
            except tk.TclError: pass
        effective_state_for_merge_options = tk.DISABLED
        if current_processing_state == tk.NORMAL and active: effective_state_for_merge_options = tk.NORMAL
        if hasattr(self, 'merge_related_widgets_references'):
            for widget_tuple_or_item in self.merge_related_widgets_references:
                items_to_configure = widget_tuple_or_item if isinstance(widget_tuple_or_item, tuple) else (widget_tuple_or_item,)
                for widget_item in items_to_configure:
                    if hasattr(widget_item, 'configure'):
                        try:
                            if isinstance(widget_item, ttk.Combobox): widget_item.configure(state='readonly' if effective_state_for_merge_options == tk.NORMAL else 'disabled')
                            else: widget_item.configure(state=effective_state_for_merge_options)
                        except tk.TclError: pass
        if not active:
            if current_processing_state == tk.NORMAL:
                for var_attr_name in ['keep_intermediate_npz_var', 'merge_dither_var', 'merge_gamma_correct_var', 'merge_percentile_norm_var']:
                    if hasattr(self, var_attr_name):
                        var_to_set = getattr(self, var_attr_name)
                        if var_to_set: var_to_set.set(False)
        self.toggle_keep_npz_dependent_options_state()
        self.toggle_dither_options_active_state()
        self.toggle_gamma_options_active_state()
        self.toggle_percentile_norm_options_active_state()

    def toggle_percentile_norm_options_active_state(self, *args):
        if not (hasattr(self, 'process_as_segments_var') and hasattr(self, 'merge_percentile_norm_var')): return
        active = self.process_as_segments_var.get() and self.merge_percentile_norm_var.get()
        state = tk.NORMAL if active else tk.DISABLED
        for attr_name in ['lbl_low_perc', 'entry_low_perc', 'lbl_high_perc', 'entry_high_perc']:
            if hasattr(self, attr_name):
                widget = getattr(self, attr_name)
                if widget and hasattr(widget, 'configure'):
                    try: widget.configure(state=state)
                    except tk.TclError: pass

    def toggle_secondary_output_options_active_state(self, *args):
        if not hasattr(self, 'enable_dual_output_robust_norm') or not hasattr(self, 'secondary_output_widgets_references'):
            return

        active = self.enable_dual_output_robust_norm.get()
        state = tk.NORMAL if active else tk.DISABLED

        for widget_item in self.secondary_output_widgets_references:
            if isinstance(widget_item, tuple): # Handle cases where we might store (label, entry_frame)
                for item in widget_item:
                    if hasattr(item, 'configure'):
                        try:
                            if isinstance(item, ttk.Combobox): item.configure(state='readonly' if active else 'disabled')
                            else: item.configure(state=state)
                        except tk.TclError: pass
            elif hasattr(widget_item, 'configure'):
                try:
                    if isinstance(widget_item, ttk.Combobox): widget_item.configure(state='readonly' if active else 'disabled')
                    else: widget_item.configure(state=state)
                except tk.TclError: pass

if __name__ == "__main__":
    # Configure basic logging for console output
    logging.basicConfig(level=logging.DEBUG, # Default to INFO level
                        format='%(asctime)s - %(message)s',
                        datefmt='%H:%M:%S')

    
    if THEMEDTK_AVAILABLE:
        root = ThemedTk(theme="default") # Use ThemedTk for theme support
    else:
        root = tk.Tk()
    app = DepthCrafterGUI(root)
    root.mainloop()