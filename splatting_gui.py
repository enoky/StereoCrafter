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

# Import custom modules
from dependency.forward_warp_pytorch import forward_warp
# --- MODIFIED IMPORT ---
from dependency.stereocrafter_util import ( Tooltip, logger, get_video_stream_info, draw_progress_bar,
    check_cuda_availability, release_cuda_memory, CUDA_AVAILABLE, set_util_logger_level,
    start_ffmpeg_pipe_process # <-- IMPORT THE NEW PIPE FUNCTION
)

# Global flag for CUDA availability (set by check_cuda_availability at runtime)
CUDA_AVAILABLE = False
GUI_VERSION = "25.10.12.0"

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
        "DEPTH_DILATE_SIZE": "3",
        "DEPTH_BLUR_SIZE": "7"
    }
    # ---------------------------------------
    # Maps Sidecar JSON Key to the internal variable key (used in APP_CONFIG_DEFAULTS)
    SIDECAR_KEY_MAP = {
        "convergence_plane": "CONV_POINT",
        "max_disparity": "MAX_DISP",
        # Your existing key:
        "gamma": "DEPTH_GAMMA", 
        # Add the two new keys (using internal names for now, we'll confirm sidecar keys later)
        "depth_dilate_size": "DEPTH_DILATE_SIZE",
        "depth_blur_size": "DEPTH_BLUR_SIZE",
        "frame_overlap": "FRAME_OVERLAP", # Add existing sidecar keys for completeness
        "input_bias": "INPUT_BIAS"
    }
    # ---------------------------------------

    def __init__(self):
        super().__init__(theme="default")
        self.title(f"Stereocrafter Splatting (Batch) {GUI_VERSION}")

        self.app_config = {}
        self.help_texts = {}

        self._load_config()
        self._load_help_texts()
        
        self._is_startup = True # NEW: for theme/geometry handling
        self.debug_mode_var = tk.BooleanVar(value=self.app_config.get("debug_mode_enabled", False))
        self._debug_logging_enabled = False # start in INFO mode
        # NEW: Window size and position variables
        self.window_x = self.app_config.get("window_x", None)
        self.window_y = self.app_config.get("window_y", None)
        self.window_width = self.app_config.get("window_width", 620)

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
        self.depth_dilate_size_var = tk.StringVar(value=self.app_config.get("depth_dilate_size", defaults["DEPTH_DILATE_SIZE"]))
        self.depth_blur_size_var = tk.StringVar(value=self.app_config.get("depth_blur_size", defaults["DEPTH_BLUR_SIZE"]))

        # --- NEW: Sidecar Control Toggle Variables ---
        self.enable_sidecar_gamma_var = tk.BooleanVar(value=self.app_config.get("enable_sidecar_gamma", True))
        self.enable_sidecar_blur_dilate_var = tk.BooleanVar(value=self.app_config.get("enable_sidecar_blur_dilate", True))

        # --- Variables for "Current Processing Information" display ---
        self.processing_filename_var = tk.StringVar(value="N/A")
        self.processing_resolution_var = tk.StringVar(value="N/A")
        self.processing_frames_var = tk.StringVar(value="N/A")
        self.processing_disparity_var = tk.StringVar(value="N/A")
        self.processing_convergence_var = tk.StringVar(value="N/A")
        self.processing_task_name_var = tk.StringVar(value="N/A")
        self.processing_gamma_var = tk.StringVar(value="N/A")


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
        self.after(100, self.check_queue) # Start checking progress queue

        # Bind closing protocol
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

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
        if not is_startup:
            current_actual_width = self.winfo_width() # Get current width (including user resize)
            if current_actual_width <= 1: # Fallback for very first call where winfo_width might be 1
                current_actual_width = self.window_width # Use the saved/default width

            new_height = self.winfo_reqheight() # Get the new optimal height based on content and theme

            # Apply the current (potentially user-adjusted) width and the new calculated height
            self.geometry(f"{current_actual_width}x{new_height}")
            logger.debug(f"Theme change applied geometry: {current_actual_width}x{new_height}")
            
            # Update the stored width for next time save_config is called.
            self.window_width = current_actual_width

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

        self.file_menu .add_checkbutton(label="Dark Mode", variable=self.dark_mode_var, command=self._apply_theme)
        self.file_menu .add_separator()
        self.file_menu .add_command(label="Reset to Default", command=self.reset_to_defaults)
        self.file_menu .add_command(label="Restore Finished", command=self.restore_finished_files)

        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.debug_logging_var = tk.BooleanVar(value=self._debug_logging_enabled)
        self.help_menu.add_checkbutton(label="Debug Logging", variable=self.debug_logging_var, command=self._toggle_debug_logging)
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

        # --- Settings Container Frame (to hold two side-by-side frames) ---
        self.settings_container_frame = ttk.Frame(self)
        self.settings_container_frame.pack(pady=10, padx=10, fill="x")
        self.settings_container_frame.grid_columnconfigure(0, weight=1)
        self.settings_container_frame.grid_columnconfigure(1, weight=1)

        # ===================================================================
        # LEFT SIDE: Process Resolution and Settings Frame
        # ===================================================================
        self.preprocessing_frame = ttk.LabelFrame(self.settings_container_frame, text="Process Resolution")
        self.preprocessing_frame.grid(row=0, column=0, padx=(0, 5), sticky="nsew")
        self.preprocessing_frame.grid_columnconfigure(1, weight=1) # Allow Entry to expand

        current_row = 0
        # Enable Full Resolution Section
        self.enable_full_res_checkbox = ttk.Checkbutton(self.preprocessing_frame, text="Enable Full Resolution Output(For Blending)", variable=self.enable_full_res_var, command=self.toggle_processing_settings_fields)
        self.enable_full_res_checkbox.grid(row=current_row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.enable_full_res_checkbox, "enable_full_res")
        current_row += 1

        self.lbl_full_res_batch_size = ttk.Label(self.preprocessing_frame, text="Full Res Batch Size:")
        self.lbl_full_res_batch_size.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        self.entry_full_res_batch_size = ttk.Entry(self.preprocessing_frame, textvariable=self.batch_size_var, width=15)
        self.entry_full_res_batch_size.grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.lbl_full_res_batch_size, "full_res_batch_size")
        self._create_hover_tooltip(self.entry_full_res_batch_size, "full_res_batch_size")
        current_row += 1


        # Enable Low Resolution Section
        self.enable_low_res_checkbox = ttk.Checkbutton(self.preprocessing_frame, text="Enable Low Resolution Output (For Inpainting)", variable=self.enable_low_res_var, command=self.toggle_processing_settings_fields)
        self.enable_low_res_checkbox.grid(row=current_row, column=0, columnspan=2, sticky="w", padx=5, pady=(10, 2))
        self._create_hover_tooltip(self.enable_low_res_checkbox, "enable_low_res")
        current_row += 1

        # Low Res Width/Height (Squeezed onto one row)
        self.low_res_wh_frame = ttk.Frame(self.preprocessing_frame)
        self.low_res_wh_frame.grid(row=current_row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        self.pre_res_width_label = ttk.Label(self.low_res_wh_frame, text="Width:")
        self.pre_res_width_label.pack(side="left", padx=(0, 2))
        self.pre_res_width_entry = ttk.Entry(self.low_res_wh_frame, textvariable=self.pre_res_width_var, width=8)
        self.pre_res_width_entry.pack(side="left", padx=(0, 10))

        self.pre_res_height_label = ttk.Label(self.low_res_wh_frame, text="Heigh:")
        self.pre_res_height_label.pack(side="left", padx=(0, 2))
        self.pre_res_height_entry = ttk.Entry(self.low_res_wh_frame, textvariable=self.pre_res_height_var, width=8)
        self.pre_res_height_entry.pack(side="left", padx=(0, 0))

        self._create_hover_tooltip(self.pre_res_width_label, "low_res_width")
        self._create_hover_tooltip(self.pre_res_width_entry, "low_res_width")
        self._create_hover_tooltip(self.pre_res_height_label, "low_res_height")
        self._create_hover_tooltip(self.pre_res_height_entry, "low_res_height")
        current_row += 1


        self.lbl_low_res_batch_size = ttk.Label(self.preprocessing_frame, text="Low Res Batch Size:")
        self.lbl_low_res_batch_size.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        self.entry_low_res_batch_size = ttk.Entry(self.preprocessing_frame, textvariable=self.low_res_batch_size_var, width=15)
        self.entry_low_res_batch_size.grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.lbl_low_res_batch_size, "low_res_batch_size")
        self._create_hover_tooltip(self.entry_low_res_batch_size, "low_res_batch_size")
        current_row = 0 # Reset for next frame

        # ===================================================================
        # RIGHT SIDE: Depth Map Pre-processing Frame
        # ===================================================================
        self.depth_settings_container = ttk.Frame(self.settings_container_frame)
        self.depth_settings_container.grid(row=0, column=1, padx=(5, 0), sticky="nsew")
        self.depth_settings_container.grid_columnconfigure(0, weight=1)
        
        # # --- Depth Map Pre-processing Frame (Inner Frame) ---
        # sidecar_ext = self.APP_CONFIG_DEFAULTS['SIDECAR_EXT']

        # --- Depth Map Pre-processing Frame (Inner Frame) ---
        self.depth_prep_frame = ttk.LabelFrame(self.depth_settings_container, text="Depth Map Pre-processing (Hi-Res Only)")
        self.depth_prep_frame.grid(row=current_row, column=0, sticky="ew") # Use grid here for placement inside container
        self.depth_prep_frame.grid_columnconfigure(1, weight=1)

        row_inner = 0
        # Dilate Size
        self.lbl_depth_dilate_size = ttk.Label(self.depth_prep_frame, text="Dilate Size (0=Off):")
        self.lbl_depth_dilate_size.grid(row=row_inner, column=0, sticky="e", padx=5, pady=2)
        self.entry_depth_dilate_size = ttk.Entry(self.depth_prep_frame, textvariable=self.depth_dilate_size_var, width=15)
        self.entry_depth_dilate_size.grid(row=row_inner, column=1, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.lbl_depth_dilate_size, "depth_dilate_size")
        self._create_hover_tooltip(self.entry_depth_dilate_size, "depth_dilate_size")
        row_inner += 1
        
        # Blur Size (Sigma = Size/6)
        self.lbl_depth_blur_size = ttk.Label(self.depth_prep_frame, text="Blur Size (0/Odd):")
        self.lbl_depth_blur_size.grid(row=row_inner, column=0, sticky="e", padx=5, pady=2)
        self.entry_depth_blur_size = ttk.Entry(self.depth_prep_frame, textvariable=self.depth_blur_size_var, width=15)
        self.entry_depth_blur_size.grid(row=row_inner, column=1, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.lbl_depth_blur_size, "depth_blur_size")
        self._create_hover_tooltip(self.entry_depth_blur_size, "depth_blur_size")
        row_inner = 0 # Reset for next frame

        current_row += 1

        # ===================================================================
        # --- Sidecar Control Frame (Placed below Depth Prep Frame) ---
        # label_text = "Sidecar (" + sidecar_ext + ") Control"
        # self.sidecar_control_frame = ttk.LabelFrame(self.depth_settings_container, text=label_text) # <-- Already self. and using corrected text
        # self.sidecar_control_frame.grid(row=current_row, column=0, sticky="ew", pady=(10, 0)) # Use grid here for placement inside container
        # self.sidecar_control_frame.grid_columnconfigure(0, weight=1)

        # row_inner = 0
        # # Sidecar Toggles
        # self.sidecar_gamma_checkbox = ttk.Checkbutton(self.sidecar_control_frame, text="Enable Sidecar Depth Gamma Override", variable=self.enable_sidecar_gamma_var)
        # self.sidecar_gamma_checkbox.grid(row=row_inner, column=0, sticky="w", padx=5, pady=2)
        # self._create_hover_tooltip(self.sidecar_gamma_checkbox, "sidecar_gamma_toggle")
        # row_inner += 1

        # self.sidecar_blur_dilate_checkbox = ttk.Checkbutton(self.sidecar_control_frame, text="Enable Sidecar Blur/Dilate Override", variable=self.enable_sidecar_blur_dilate_var)
        # self.sidecar_blur_dilate_checkbox.grid(row=row_inner, column=0, sticky="w", padx=5, pady=2)
        # self._create_hover_tooltip(self.sidecar_blur_dilate_checkbox, "sidecar_blur_dilate_toggle")
        # current_row = 0 # Reset for next frame

        # ===================================================================
        # --- Output Settings Frame (Now placed below the settings_container_frame) ---
        current_row = 0
        self.output_settings_frame = ttk.LabelFrame(self, text="Splatting & Output Settings")
        self.output_settings_frame.pack(pady=10, padx=10, fill="x")
        self.output_settings_frame.grid_columnconfigure(1, weight=1)
        self.output_settings_frame.grid_columnconfigure(3, weight=1)
        
        # Gamma
        self.lbl_depth_gamma = ttk.Label(self.output_settings_frame, text="Gamma (1.0=Off):")
        self.lbl_depth_gamma.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        self.entry_depth_gamma = ttk.Entry(self.output_settings_frame, textvariable=self.depth_gamma_var, width=15)
        self.entry_depth_gamma.grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.lbl_depth_gamma, "depth_gamma")
        self._create_hover_tooltip(self.entry_depth_gamma, "depth_gamma")

        self.lbl_process_length = ttk.Label(self.output_settings_frame, text="Process Length (-1 for all):")
        self.lbl_process_length.grid(row=current_row, column=2, sticky="e", padx=5, pady=2)
        self.entry_process_length = ttk.Entry(self.output_settings_frame, textvariable=self.process_length_var, width=15)
        self.entry_process_length.grid(row=current_row, column=3, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.lbl_process_length, "process_length")
        self._create_hover_tooltip(self.entry_process_length, "process_length")
        current_row += 1
        
        self.lbl_max_disp = ttk.Label(self.output_settings_frame, text="Max Disparity %:")
        self.lbl_max_disp.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        self.entry_max_disp = ttk.Entry(self.output_settings_frame, textvariable=self.max_disp_var, width=15)
        self.entry_max_disp.grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.lbl_max_disp, "max_disp")
        self._create_hover_tooltip(self.entry_max_disp, "max_disp")

        # Output CRF setting, placed on the right side
        self.lbl_output_crf = ttk.Label(self.output_settings_frame, text="Output CRF (0-51):")
        self.lbl_output_crf.grid(row=current_row, column=2, sticky="e", padx=5, pady=2)
        self.entry_output_crf = ttk.Entry(self.output_settings_frame, textvariable=self.output_crf_var, width=15)
        self.entry_output_crf.grid(row=current_row, column=3, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.lbl_output_crf, "output_crf")
        self._create_hover_tooltip(self.entry_output_crf, "output_crf")
        current_row += 1

        self.lbl_zero_disparity_anchor = ttk.Label(self.output_settings_frame, text="Convergence Point (0-1):")
        self.lbl_zero_disparity_anchor.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        self.entry_zero_disparity_anchor = ttk.Entry(self.output_settings_frame, textvariable=self.zero_disparity_anchor_var, width=15)
        self.entry_zero_disparity_anchor.grid(row=current_row, column=1, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.lbl_zero_disparity_anchor, "convergence_point")
        self._create_hover_tooltip(self.entry_zero_disparity_anchor, "convergence_point")

        # --- MODIFIED: Added a label for the Combobox ---
        self.lbl_auto_convergence = ttk.Label(self.output_settings_frame, text="Auto-Convergence:")
        self.lbl_auto_convergence.grid(row=current_row, column=2, sticky="e", padx=5, pady=2)
        self.auto_convergence_combo = ttk.Combobox(self.output_settings_frame, textvariable=self.auto_convergence_mode_var, values=["Off", "Average", "Peak"], state="readonly", width=15)
        self.auto_convergence_combo.grid(row=current_row, column=3, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.lbl_auto_convergence, "auto_convergence_toggle")
        self._create_hover_tooltip(self.auto_convergence_combo, "auto_convergence_toggle")
        current_row += 1

        self.dual_output_checkbox = ttk.Checkbutton(self.output_settings_frame, text="Dual Output Only (Mask & Warped)", variable=self.dual_output_var)
        self.dual_output_checkbox.grid(row=current_row, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.dual_output_checkbox, "dual_output")

        self.autogain_checkbox = ttk.Checkbutton(self.output_settings_frame, text="Disable Normalization (For Seamless Joining)", variable=self.enable_autogain_var)
        self.autogain_checkbox.grid(row=current_row, column=2, columnspan=2, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.autogain_checkbox, "no_normalization")        

        current_row = 0 # Reset for next frame

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
        self.start_button = ttk.Button(button_frame, text="START", command=self.start_processing)
        self.start_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.start_button, "start_button")

        self.stop_button = ttk.Button(button_frame, text="STOP", command=self.stop_processing, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.stop_button, "stop_button")

        exit_button = ttk.Button(button_frame, text="EXIT", command=self.exit_app)
        exit_button.pack(side="left", padx=5)
        self._create_hover_tooltip(exit_button, "exit_button")

        # --- Current Processing Information frame ---
        self.info_frame = ttk.LabelFrame(self, text="Current Processing Information") # Store frame as instance attribute
        self.info_frame.pack(pady=10, padx=10, fill="x")
        self.info_frame.grid_columnconfigure(1, weight=1)

        self.info_labels = [] # List to hold the tk.Label widgets for easy iteration

        # Row 0: Filename
        lbl_filename_static = tk.Label(self.info_frame, text="Filename:")
        lbl_filename_static.grid(row=0, column=0, sticky="e", padx=5, pady=1)
        lbl_filename_value = tk.Label(self.info_frame, textvariable=self.processing_filename_var, anchor="w")
        lbl_filename_value.grid(row=0, column=1, sticky="ew", padx=5, pady=1)
        self.info_labels.extend([lbl_filename_static, lbl_filename_value])

        # Row 1: Task Name
        lbl_task_static = tk.Label(self.info_frame, text="Task:")
        lbl_task_static.grid(row=1, column=0, sticky="e", padx=5, pady=1)
        lbl_task_value = tk.Label(self.info_frame, textvariable=self.processing_task_name_var, anchor="w")
        lbl_task_value.grid(row=1, column=1, sticky="ew", padx=5, pady=1)
        self.info_labels.extend([lbl_task_static, lbl_task_value])

        # Row 2: Resolution
        lbl_resolution_static = tk.Label(self.info_frame, text="Resolution:")
        lbl_resolution_static.grid(row=2, column=0, sticky="e", padx=5, pady=1)
        lbl_resolution_value = tk.Label(self.info_frame, textvariable=self.processing_resolution_var, anchor="w")
        lbl_resolution_value.grid(row=2, column=1, sticky="ew", padx=5, pady=1)
        self.info_labels.extend([lbl_resolution_static, lbl_resolution_value])

        # Row 3: Total Frames for current task
        lbl_frames_static = tk.Label(self.info_frame, text="Frames:")
        lbl_frames_static.grid(row=3, column=0, sticky="e", padx=5, pady=1)
        lbl_frames_value = tk.Label(self.info_frame, textvariable=self.processing_frames_var, anchor="w")
        lbl_frames_value.grid(row=3, column=1, sticky="ew", padx=5, pady=1)
        self.info_labels.extend([lbl_frames_static, lbl_frames_value])

        # Row 4: Max Disparity
        lbl_disparity_static = tk.Label(self.info_frame, text="Max Disparity:")
        lbl_disparity_static.grid(row=4, column=0, sticky="e", padx=5, pady=1)
        lbl_disparity_value = tk.Label(self.info_frame, textvariable=self.processing_disparity_var, anchor="w")
        lbl_disparity_value.grid(row=4, column=1, sticky="ew", padx=5, pady=1)
        self.info_labels.extend([lbl_disparity_static, lbl_disparity_value])

        # Row 5: Convergence Point
        lbl_convergence_static = tk.Label(self.info_frame, text="Convergence:")
        lbl_convergence_static.grid(row=5, column=0, sticky="e", padx=5, pady=1)
        lbl_convergence_value = tk.Label(self.info_frame, textvariable=self.processing_convergence_var, anchor="w")
        lbl_convergence_value.grid(row=5, column=1, sticky="ew", padx=5, pady=1)
        self.info_labels.extend([lbl_convergence_static, lbl_convergence_value])

        # --- NEW ROW 6: Gamma ---
        lbl_gamma_static = tk.Label(self.info_frame, text="Gamma:")
        lbl_gamma_static.grid(row=6, column=0, sticky="e", padx=5, pady=1)
        lbl_gamma_value = tk.Label(self.info_frame, textvariable=self.processing_gamma_var, anchor="w")
        lbl_gamma_value.grid(row=6, column=1, sticky="ew", padx=5, pady=1)
        self.info_labels.extend([lbl_gamma_static, lbl_gamma_value])
        # ------------------------
    
    def _determine_auto_convergence(self, depth_map_path: str, total_frames_to_process: int, batch_size: int, fallback_value: float, mode: str) -> float:
        """
        Calculates the Auto Convergence point for the entire video based on the selected mode.
        Uses a hard blur and crops to the center 75% to eliminate noise/edge artifacts.

        Args:
            mode (str): "Average" or "Peak".
            float: fallback_value: The current GUI/Sidecar value to return if auto-convergence fails.

        Returns:
            float: The new zero_disparity_anchor_val (0.0 to 1.0) or fallback_value.
        """
        logger.info("==> Starting Auto-Convergence pre-pass to determine global average depth.")
        
        # --- Constants for Auto-Convergence Logic ---
        BLUR_KERNEL_SIZE = 9  # Hard blur to capture bulk depth, must be odd
        CENTER_CROP_PERCENT = 0.75 # Use the center 75% of the frame (1 - 0.75 = 0.25 margin total)
        MIN_VALID_PIXELS = 5 # Minimum number of non-zero/non-max pixels to consider a frame valid
        INTERNAL_ANCHOR_OFFSET = 0.1 # <--- ADJUST THIS VALUE FOR TESTING (e.g., 0.02, 0.05, 0.1)
        # -------------------------------------------

        all_valid_frame_values = []

        try:
            # 1. Initialize Decord Reader (No target height/width needed, raw is fine)
            depth_reader = VideoReader(depth_map_path, ctx=cpu(0))
            if len(depth_reader) == 0:
                 logger.error("Depth map reader has no frames. Cannot calculate Auto-Convergence.")
                 return fallback_value # Fallback to user/sidecar default
        except Exception as e:
            logger.error(f"Error initializing depth map reader for Auto-Convergence: {e}")
            return fallback_value # Fallback to user/sidecar default


        # 2. Iterate and Average
        
        # Determine the number of frames to process
        video_length = len(depth_reader)
        if total_frames_to_process <= 0 or total_frames_to_process > video_length:
             num_frames = video_length
        else:
             num_frames = total_frames_to_process
            
        logger.debug(f"  AutoConv determined actual frames to process: {num_frames} (from input length {total_frames_to_process}).")


        # Original line: num_frames = min(total_frames_to_process, len(depth_reader))
        
        # Now use the determined num_frames for the loop:
        for i in range(0, num_frames, batch_size):
            if self.stop_event.is_set():
                logger.warning("Auto-Convergence pre-pass stopped by user.")
                return fallback_value # Fallback

            current_frame_indices = list(range(i, min(i + batch_size, num_frames)))
            if not current_frame_indices:
                break
            
            batch_depth_numpy_raw = depth_reader.get_batch(current_frame_indices).asnumpy()

            # Process depth frames (Grayscale, Float conversion, 0-1 Normalization)
            if batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 3:
                batch_depth_numpy = batch_depth_numpy_raw.mean(axis=-1)
            elif batch_depth_numpy_raw.ndim == 4 and batch_depth_numpy_raw.shape[-1] == 1:
                batch_depth_numpy = batch_depth_numpy_raw.squeeze(-1)
            else:
                batch_depth_numpy = batch_depth_numpy_raw
            
            batch_depth_float = batch_depth_numpy.astype(np.float32)

            # Get chunk min/max for normalization (using the chunk's range, since we don't have global stats yet)
            min_val = batch_depth_float.min()
            max_val = batch_depth_float.max()
            
            if max_val - min_val > 1e-5:
                batch_depth_normalized = (batch_depth_float - min_val) / (max_val - min_val)
            else:
                # If chunk is flat, use a neutral value
                batch_depth_normalized = np.full_like(batch_depth_float, fill_value=0.5, dtype=np.float32)

            # Frame-by-Frame Processing (Blur & Crop)
            for j, frame in enumerate(batch_depth_normalized):
                
                # --- NEW DEBUG LOGGING START ---
                current_frame_idx = current_frame_indices[j]
                H, W = frame.shape
                
                # a) Blur
                frame_blurred = cv2.GaussianBlur(frame, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
                
                # b) Center Crop (75% of H and W)
                margin_h = int(H * (1 - CENTER_CROP_PERCENT) / 2)
                margin_w = int(W * (1 - CENTER_CROP_PERCENT) / 2)
                
                cropped_frame = frame_blurred[margin_h:H-margin_h, margin_w:W-margin_w]
                
                # c) Average (Exclude true black/white pixels (0.0 or 1.0) which may be background/edges)
                # Use a small epsilon to catch near-zero/near-one
                valid_pixels = cropped_frame[(cropped_frame > 0.001) & (cropped_frame < 0.999)] 
                
                logger.debug(f"  [AutoConv Frame {current_frame_idx:03d}] Res: {W}x{H}, Crop Margins: {margin_w}x{margin_h}. Cropped Size: {cropped_frame.size}. Valid Pixels: {valid_pixels.size}")
                # --- NEW DEBUG LOGGING END ---

                if valid_pixels.size > MIN_VALID_PIXELS:
                    if mode == "Average":
                        all_valid_frame_values.append(valid_pixels.mean())
                    elif mode == "Peak":
                        all_valid_frame_values.append(valid_pixels.max())
                else:
                    # FALLBACK FOR THE FRAME: If filter is too strict, just use the mean of the WHOLE cropped, blurred frame
                    all_valid_frame_values.append(cropped_frame.mean())
                    # Note: We still log the failure using a warning, but we count the frame.
                    logger.warning(f"  [AutoConv Frame {current_frame_idx:03d}] SKIPPED: Valid pixel count ({valid_pixels.size}) below threshold ({MIN_VALID_PIXELS}). Forcing mean from full cropped frame.")

            draw_progress_bar(i + len(current_frame_indices), num_frames, prefix="  Auto-Conv Pre-Pass:")
        
        # 3. Final Temporal Average
        if all_valid_frame_values:
            # Use the same aggregation (mean) for both modes for temporal stability
            raw_anchor = np.mean(all_valid_frame_values)
            
            # --- MODIFIED: Apply Offset only for Average mode ---
            offset_to_apply = 0.0
            if mode == "Average":
                offset_to_apply = INTERNAL_ANCHOR_OFFSET
            
            final_anchor_offset = raw_anchor + offset_to_apply
            
            # Clamp to the valid range [0.0, 1.0]
            final_anchor = np.clip(final_anchor_offset, 0.0, 1.0)
            
            logger.info(f"\n==> Auto-Convergence ({mode} mode) Calculated: {raw_anchor:.4f} + Offset ({offset_to_apply:.2f}) = Final Anchor {final_anchor:.4f}")
            return float(final_anchor)
        else:
            logger.warning("\n==> Auto-Convergence failed: No valid frames found. Using GUI/Sidecar value as fallback.")
            return fallback_value # Fallback to user/sidecar default

    def _get_current_config(self):
        """Collects all current GUI variable values into a single dictionary."""
        config = {
            # Folder Configurations
            "input_source_clips": self.input_source_clips_var.get(),
            "input_depth_maps": self.input_depth_maps_var.get(),
            "output_splatted": self.output_splatted_var.get(),
            
            "dark_mode_enabled": self.dark_mode_var.get(),
            "window_width": self.winfo_width(),
            "window_x": self.winfo_x(),
            "window_y": self.winfo_y(),

            "enable_full_resolution": self.enable_full_res_var.get(),
            "batch_size": self.batch_size_var.get(),
            "enable_low_resolution": self.enable_low_res_var.get(),
            "pre_res_width": self.pre_res_width_var.get(),
            "pre_res_height": self.pre_res_height_var.get(),
            "low_res_batch_size": self.low_res_batch_size_var.get(),
            
            "depth_dilate_size": self.depth_dilate_size_var.get(),
            "depth_blur_size": self.depth_blur_size_var.get(),

            "process_length": self.process_length_var.get(),
            "output_crf": self.output_crf_var.get(),
            "dual_output": self.dual_output_var.get(),
            "auto_convergence_mode": self.auto_convergence_mode_var.get(),
            
            "depth_gamma": self.depth_gamma_var.get(),
            "max_disp": self.max_disp_var.get(),
            "convergence_point": self.zero_disparity_anchor_var.get(),
            "enable_autogain": self.enable_autogain_var.get(),
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
        
        # --- DEFINES json_sidecar_path ---
        sidecar_ext = self.APP_CONFIG_DEFAULTS['SIDECAR_EXT']
        json_sidecar_path = os.path.join(os.path.dirname(actual_depth_map_path), f"{depth_map_basename}{sidecar_ext}")
        # ---------------------------------


        # [NEW]: Read GUI Input Fields for defaults
        try:
            default_gamma_gui = float(self.depth_gamma_var.get())
            default_dilate_gui = int(self.depth_dilate_size_var.get())
            default_blur_gui = int(self.depth_blur_size_var.get())
        except ValueError:
            default_gamma_gui, default_dilate_gui, default_blur_gui = 1.0, 0, 0
        
        # Initialize final parameters with GUI defaults (or passed values)
        current_params = {
            "convergence_plane": default_zero_disparity_anchor,
            "max_disparity_percentage": gui_max_disp,
            "frame_overlap": None,
            "input_bias": None,
            "depth_gamma": default_gamma_gui,
            "depth_dilate_size": default_dilate_gui,
            "depth_blur_size": default_blur_gui
        }
        anchor_source = "GUI"
        max_disp_source = "GUI"
        gamma_source = "GUI"

        if os.path.exists(json_sidecar_path):
            try:
                with open(json_sidecar_path, 'r') as f:
                    sidecar_data = json.load(f)

                # --- MAPPING LOGIC: Iterate through the central map ---
                for sidecar_key, internal_key in self.SIDECAR_KEY_MAP.items():
                    if sidecar_key in sidecar_data:
                        value = sidecar_data[sidecar_key]
                        
                        # --- Apply the value based on the internal key ---
                        
                        # Max Disparity and Convergence Plane (Requires special source tracking)
                        if internal_key in ["CONV_POINT", "MAX_DISP"]:
                            if internal_key == "CONV_POINT":
                                current_params["convergence_plane"] = float(value)
                                anchor_source = "Sidecar"
                            if internal_key == "MAX_DISP":
                                current_params["max_disparity_percentage"] = float(value)
                                max_disp_source = "Sidecar"
                            logger.debug(f"==> Using sidecar {sidecar_key}: {value}")

                        # Gamma (Check toggle: self.enable_sidecar_gamma_var)
                        elif internal_key == "DEPTH_GAMMA" and self.enable_sidecar_gamma_var.get():
                            if isinstance(value, (int, float)) and float(value) > 0:
                                current_params["depth_gamma"] = float(value)
                                gamma_source = "Sidecar"
                                logger.debug(f"==> Using sidecar {sidecar_key}: {value} (Override)")
                        
                        # Blur/Dilate (Check toggle: self.enable_sidecar_blur_dilate_var)
                        elif internal_key in ["DEPTH_DILATE_SIZE", "DEPTH_BLUR_SIZE"] and self.enable_sidecar_blur_dilate_var.get():
                            if isinstance(value, (int, float)):
                                processed_val = max(0, int(value))
                                
                                if internal_key == "DEPTH_DILATE_SIZE":
                                    current_params["depth_dilate_size"] = processed_val
                                    logger.debug(f"==> Using sidecar {sidecar_key}: {processed_val} (Override)")
                                    
                                elif internal_key == "DEPTH_BLUR_SIZE":
                                    # Ensure blur size is odd
                                    if processed_val > 0 and processed_val % 2 == 0:
                                        processed_val += 1
                                        logger.warning(f"==> Blur size must be odd. Increased to {processed_val}.")
                                    current_params["depth_blur_size"] = processed_val
                                    logger.debug(f"==> Using sidecar {sidecar_key}: {processed_val} (Override)")

                        # Frame Overlap and Input Bias (No GUI toggle, always use sidecar if present)
                        elif internal_key in ["FRAME_OVERLAP", "INPUT_BIAS"]:
                            if isinstance(value, (int, float)):
                                if internal_key == "FRAME_OVERLAP":
                                    current_params["frame_overlap"] = int(value)
                                elif internal_key == "INPUT_BIAS":
                                    current_params["input_bias"] = float(value)
                                logger.debug(f"==> Using sidecar {sidecar_key} ({internal_key}): {value}")
                            
                # --- END MAPPING LOGIC ---
            
            except json.JSONDecodeError:
                logger.error(f"==> Error: Could not parse sidecar '{json_sidecar_path}'. Using GUI defaults.")
            except Exception as e:
                logger.error(f"==> Unexpected error reading sidecar '{json_sidecar_path}': {e}. Using GUI defaults.")
        else:
            logger.debug(f"==> No sidecar '{json_sidecar_path}' found. Using GUI defaults.")

        return {
            "actual_depth_map_path": actual_depth_map_path,
            "convergence_plane": current_params["convergence_plane"],
            "max_disparity_percentage": current_params["max_disparity_percentage"],
            "frame_overlap": current_params["frame_overlap"],
            "input_bias": current_params["input_bias"],
            "depth_gamma": current_params["depth_gamma"],
            "depth_dilate_size": current_params["depth_dilate_size"],
            "depth_blur_size": current_params["depth_blur_size"],
            "anchor_source": anchor_source,
            "max_disp_source": max_disp_source,
            "gamma_source": gamma_source,
        }
    
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
            json_sidecar_path_to_move = os.path.join(depth_map_dirname, f"{depth_map_basename_without_ext}.fssidecar")
            dest_path_json = os.path.join(finished_depth_folder, f"{depth_map_basename_without_ext}.fssidecar")

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

    def _process_depth_batch(self, batch_depth_numpy_raw: np.ndarray, depth_stream_info: Optional[dict], depth_gamma: float,
                              depth_dilate_size: int, depth_blur_size: int, is_low_res_task: bool, max_raw_value: float,
                              global_depth_min: float, global_depth_max: float) -> np.ndarray:
        """
        Loads, converts, and pre-processes the raw depth map batch (Grayscale, Gamma, Dilate, Blur).
        
        Note: If Global Normalization is active, Gamma is SKIPPED here and applied later
        in depthSplatting (post-normalization) to maintain correct 0-1 range.
        """
        # 1. Grayscale Conversion
        if batch_depth_numpy_raw.ndim == 4:
            if batch_depth_numpy_raw.shape[-1] == 3: # RGB
                batch_depth_numpy = batch_depth_numpy_raw.mean(axis=-1)
            else: # Grayscale with channel dim (e.g., T,H,W,1)
                batch_depth_numpy = batch_depth_numpy_raw.squeeze(-1)
        else: # If depth video is grayscale (ndim 3: T,H,W)
            batch_depth_numpy = batch_depth_numpy_raw
        
        # Convert to float32 for processing
        batch_depth_numpy_float = batch_depth_numpy.astype(np.float32)

        # Determine if we are in GLOBAL NORM mode (if so, we skip pre-scaling Gamma)
        is_global_norm_active = (global_depth_min != 0.0 or global_depth_max != 1.0) and not (global_depth_min == 0.0 and global_depth_max == 0.0)

        # --- 2. Gamma Adjustment ---
        if depth_gamma != 1.0:
            if is_global_norm_active:
                # Skip in helper. Gamma will be applied post-normalization in depthSplatting.
                logger.debug("Gamma adjustment SKIPPED in helper: Applied post-normalization (Global Norm Mode).")
            else:
                # RAW INPUT MODE: Apply Gamma across the MAX RAW VALUE range.
                logger.debug(f"Applying depth gamma adjustment on raw range {max_raw_value:.1f}: {depth_gamma:.2f}")
                
                if max_raw_value > 1.0:
                    # Scale to [0, 1] using max_raw_value, apply gamma, scale back to raw range.
                    normalized_chunk = batch_depth_numpy_float / max_raw_value
                    normalized_chunk_gamma = np.power(normalized_chunk, depth_gamma)
                    batch_depth_numpy_float = normalized_chunk_gamma * max_raw_value
                else:
                    # Fallback for floats/errors: Apply to 0-1 range directly
                    batch_depth_numpy_float = np.power(batch_depth_numpy_float, depth_gamma)

        # --- 3. Dilate and Blur (Conditional on is_low_res_task) ---
        if is_low_res_task:
            if depth_dilate_size > 0 or depth_blur_size > 0:
                 logger.debug(f"Dilate ({depth_dilate_size}) or Blur ({depth_blur_size}) skipped for low-resolution (inpaint) task as per configuration.")
        else:
            # This is the HI-RES path: apply Dilate and Blur based on size setting.

            # 3a. Dilation (if size > 0)
            if depth_dilate_size > 0:
                logger.debug(f"Applying depth dilation with kernel size: {depth_dilate_size}")
                kernel = np.ones((depth_dilate_size, depth_dilate_size), np.uint8)
                for j in range(batch_depth_numpy_float.shape[0]):
                    batch_depth_numpy_float[j] = cv2.dilate(batch_depth_numpy_float[j], kernel, iterations=1)
            
            # 3b. Gaussian Blur (if size > 0)
            if depth_blur_size > 0:
                logger.debug(f"Applying depth Gaussian Blur with kernel size: {depth_blur_size}")
                sigma = depth_blur_size / 6.0 
                for j in range(batch_depth_numpy_float.shape[0]):
                    batch_depth_numpy_float[j] = cv2.GaussianBlur(
                        batch_depth_numpy_float[j], 
                        (depth_blur_size, depth_blur_size), 
                        sigmaX=sigma, 
                        sigmaY=sigma
                    )

        return batch_depth_numpy_float

    def _run_batch_process(self, settings):
        """
        The main processing logic, run in a separate thread.
        This has been refactored to use helper methods.
        """
        self.after(0, self.clear_processing_info) # Clear info display at start

        try:
            input_videos, is_single_file_mode, finished_source_folder, finished_depth_folder = self._setup_batch_processing(settings)

            if not input_videos:
                logger.error("No video files found or invalid input setup. Processing stopped.")
                self.progress_queue.put("finished")
                return

            # Determine total tasks for progress bar
            total_processing_tasks_count = 0
            # Pre-calculate tasks for accurate total_tasks_count, considering only videos with existing depth maps
            for video_path in input_videos:
                video_specific_settings_check = self._get_video_specific_settings(
                    video_path,
                    settings["input_depth_maps"],
                    settings["zero_disparity_anchor"],
                    settings["max_disp"],
                    is_single_file_mode
                )
                if not video_specific_settings_check.get("error"):
                    total_processing_tasks_count += len(self._get_defined_tasks(settings))
            
            if total_processing_tasks_count == 0:
                logger.error("==> Error: No resolution output enabled or no valid video/depth pairs found. Processing stopped.")
                self.progress_queue.put("finished")
                return

            self.progress_queue.put(("total", total_processing_tasks_count))
            overall_task_counter = 0

            for idx, video_path in enumerate(input_videos):
                if self.stop_event.is_set():
                    logger.info("==> Stopping processing due to user request")
                    break

                video_name = os.path.splitext(os.path.basename(video_path))[0]
                logger.info(f"==> Processing Video: {video_name}")
                self.progress_queue.put(("update_info", {"filename": video_name}))

                video_specific_settings = self._get_video_specific_settings(
                    video_path,
                    settings["input_depth_maps"],
                    settings["zero_disparity_anchor"],
                    settings["max_disp"],
                    is_single_file_mode
                )

                if video_specific_settings.get("error"):
                    logger.error(f"Error getting video specific settings for {video_name}: {video_specific_settings['error']}. Skipping.")
                    overall_task_increment = len(self._get_defined_tasks(settings))
                    overall_task_counter += overall_task_increment
                    self.progress_queue.put(("processed", overall_task_counter))
                    continue

                actual_depth_map_path = video_specific_settings["actual_depth_map_path"]
                current_zero_disparity_anchor = video_specific_settings["convergence_plane"]
                current_max_disparity_percentage = video_specific_settings["max_disparity_percentage"]
                current_frame_overlap = video_specific_settings["frame_overlap"]
                current_input_bias = video_specific_settings["input_bias"]
                anchor_source = video_specific_settings["anchor_source"]
                max_disp_source = video_specific_settings["max_disp_source"]
                gamma_source = video_specific_settings["gamma_source"]
                current_depth_gamma = video_specific_settings["depth_gamma"]
                current_depth_dilate_size = video_specific_settings["depth_dilate_size"]
                current_depth_blur_size = video_specific_settings["depth_blur_size"]



                processing_tasks = self._get_defined_tasks(settings)

                # --- NEW: Auto-Convergence Logic (BEFORE initializing readers) ---
                auto_conv_mode = settings["auto_convergence_mode"]
                if auto_conv_mode != "Off":
                    logger.info(f"Auto-Convergence is ENABLED (Mode: {auto_conv_mode}). Running pre-pass...")

                    new_anchor_val = self._determine_auto_convergence(
                        actual_depth_map_path,
                        settings["process_length"],
                        settings["full_res_batch_size"],
                        current_zero_disparity_anchor,
                        mode=auto_conv_mode
                    )
                    
                    # Update variables for current task
                    # If new_anchor_val == current_zero_disparity_anchor, it means the function failed and returned the fallback.
                    if new_anchor_val != current_zero_disparity_anchor:
                        current_zero_disparity_anchor = new_anchor_val
                        anchor_source = "Auto"
                    
                    logger.info(f"Using Convergence Point: {current_zero_disparity_anchor:.4f} (Source: {anchor_source})")
                # --- END Auto-Convergence Logic ---

                if not processing_tasks:
                    logger.debug(f"==> No processing tasks configured for {video_name}. Skipping.")
                    overall_task_counter += 0
                    self.progress_queue.put(("processed", overall_task_counter))
                    continue

                any_task_completed_successfully_for_this_video = False # Flag for current video

                for task in processing_tasks:
                    if self.stop_event.is_set():
                        logger.info(f"==> Stopping {task['name']} processing for {video_name} due to user request")
                        break

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
                        overall_task_counter += 1
                        self.progress_queue.put(("processed", overall_task_counter))
                        release_cuda_memory()
                        continue

                    assume_raw_input_mode = settings["enable_autogain"] 
                    global_depth_min = 0.0 
                    global_depth_max = 1.0 

                    # --- NEW: UNCONDITIONAL Max Content Value Scan ---
                    max_content_value = 1.0 # Default fallback
                    raw_depth_reader_temp = None
                    try:
                        raw_depth_reader_temp = VideoReader(actual_depth_map_path, ctx=cpu(0))
                        
                        if len(raw_depth_reader_temp) > 0:
                            # We only need the max raw value of the content here
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
                        
                        # --- FIX: Use a DEDICATED, RAW reader for consistent global stats ---
                        raw_depth_reader_temp = None
                        try:
                            # 1. Initialize a NON-RESIZING reader
                            raw_depth_reader_temp = VideoReader(actual_depth_map_path, ctx=cpu(0))
                            
                            # 2. Compute stats using the RAW reader
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
                            global_depth_min = 0.0 # Ensure min/max are reset on fatal error
                            global_depth_max = 1.0
                        finally:
                            # 3. Clean up the temporary RAW reader
                            if raw_depth_reader_temp:
                                del raw_depth_reader_temp
                                gc.collect()
                        # -----------------------------------------------------------------
                    else:
                        logger.debug("==> No Normalization (Assume Raw 0-1 Input) selected. Skipping global stats pre-pass.")

                        # Determine the scaling factor:
                        # --- FIX: RAW INPUT MODE SCALING ---
                        final_scaling_factor = 1.0 # Default to 1.0 if scan fails

                        if max_content_value <= 256.0 and max_content_value > 1.0:
                            # 8-bit content saved in 10-bit container. Scale by 255.0 for correct 0-1 range.
                            final_scaling_factor = 255.0
                            logger.debug(f"Content Max {max_content_value:.2f} <= 8-bit. SCALING BY 255.0.")
                        elif max_content_value > 256.0 and max_content_value <= 1024.0:
                             # 9-10bit content. Scale by its actual max value to make it 0-1.
                            final_scaling_factor = max_content_value
                            logger.debug(f"Content Max {max_content_value:.2f} (9-10bit). SCALING BY CONTENT MAX.")
                        else:
                             # Fallback: Use 10-bit theoretical max as the safest upper bound (avoids massive weak shift)
                            final_scaling_factor = 1023.0 
                            logger.warning(f"Max content value is too high/low ({max_content_value:.2f}). Using fallback 1023.0.")

                        # Set global_depth_max to the determined scaling factor.
                        global_depth_max = final_scaling_factor
                        global_depth_min = 0.0 # Raw input assumes 0 min
                        
                        logger.debug(f"Raw Input Final Scaling Factor set to: {global_depth_max:.3f}")

                    if not (actual_depth_height == current_processed_height and actual_depth_width == current_processed_width):
                        logger.warning(f"==> Warning: Depth map reader output resolution ({actual_depth_width}x{actual_depth_height}) does not match processed video resolution ({current_processed_width}x{current_processed_height}) for {task['name']} pass. This indicates an issue with `load_pre_rendered_depth`'s `width`/`height` parameters. Processing may proceed but results might be misaligned.")

                    actual_percentage_for_calculation = current_max_disparity_percentage / 20.0
                    actual_max_disp_pixels = (actual_percentage_for_calculation / 100.0) * current_processed_width
                    logger.debug(f"==> Max Disparity Input: {current_max_disparity_percentage:.1f}% -> Calculated Max Disparity for splatting ({task['name']}): {actual_max_disp_pixels:.2f} pixels")

                    self.progress_queue.put(("update_info", {"disparity": f"{actual_max_disp_pixels:.2f} pixels ({current_max_disparity_percentage:.1f}%)"}))

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
                        user_output_crf=settings["output_crf"], # Pass CRF from settings
                        is_low_res_task=task["is_low_res"],
                        # --- NEW ARGUMENTS ---
                        depth_gamma=current_depth_gamma,
                        depth_dilate_size=current_depth_dilate_size,
                        depth_blur_size=current_depth_blur_size
                    )

                    if self.stop_event.is_set():
                        logger.info(f"==> Stopping {task['name']} pass for {video_name} due to user request")
                        break

                    if completed_splatting_task:
                        logger.debug(f"==> Splatted {task['name']} video saved for {video_name}.")
                        any_task_completed_successfully_for_this_video = True # Set flag if any task succeeds

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

                    overall_task_counter += 1
                    self.progress_queue.put(("processed", overall_task_counter))
                    logger.debug(f"==> Completed {task['name']} pass for {video_name}.")

                # After all tasks for the current video are processed or stopped
                if self.stop_event.is_set():
                    break # Break from video loop if stop requested

                # NEW: Call _move_processed_files here, only if not single file mode
                # and if at least one task for this video completed successfully.
                if not is_single_file_mode and any_task_completed_successfully_for_this_video:
                    self._move_processed_files(video_path, actual_depth_map_path, finished_source_folder, finished_depth_folder)
                elif is_single_file_mode:
                    logger.debug(f"==> Single file mode for {video_name}: Skipping moving files to 'finished' folder.")
                elif not any_task_completed_successfully_for_this_video:
                    logger.info(f"==> No tasks completed successfully for {video_name}. Skipping moving files to 'finished' folder.")


        except Exception as e:
            logger.error(f"An unexpected error occurred during batch processing: {e}", exc_info=True)
            self.progress_queue.put(("status", f"Error: {e}"))
            messagebox.showerror("Processing Error", f"An unexpected error occurred during batch processing: {e}")
        finally:
            release_cuda_memory()
            self.progress_queue.put("finished")
            self.after(0, self.clear_processing_info)

    def _save_config(self):
        """Saves current GUI settings to config_splat.json."""
        config = self._get_current_config()
        with open("config_splat.json", "w") as f:
            json.dump(config, f, indent=4)

    def _save_debug_numpy(self, data: np.ndarray, filename_tag: str, batch_index: int, frame_index: int, task_name: str):
        """Saves a NumPy array to a debug folder if debug logging is enabled."""
        if not self._debug_logging_enabled:
            return

        debug_dir = os.path.join(os.path.dirname(self.input_source_clips_var.get()), "splat_debug", task_name)
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create a filename that includes frame index, batch index, and tag
        filename = os.path.join(debug_dir, f"{frame_index:05d}_B{batch_index:02d}_{filename_tag}.npz")
        
        try:
            np.savez_compressed(filename, data=data)
            logger.debug(f"Saved debug array {filename_tag} (Shape: {data.shape}) to {os.path.basename(debug_dir)}")
        except Exception as e:
            logger.error(f"Failed to save debug array {filename_tag}: {e}")
    
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


        # 3. Re-apply the specific field enable/disable logic
        # This is CRITICAL. If we set state='normal' for everything, 
        # toggle_processing_settings_fields will correctly re-disable the Low Res W/H fields
        # if the "Enable Low Resolution" checkbox is unchecked.
        if state == 'normal':
            self.toggle_processing_settings_fields()
    
    def _set_saved_geometry(self: "SplatterGUI"):
        """Applies the saved window width and position, with dynamic height."""
        # Ensure the window is visible and all widgets are laid out for accurate height calculation
        self.update_idletasks()

        # 1. Get the optimal height for the current content
        calculated_height = self.winfo_reqheight()
        # Fallback in case winfo_reqheight returns a tiny value (shouldn't happen after update_idletasks)
        if calculated_height < 100:
            calculated_height = 750 # A reasonable fallback height if something goes wrong

        # 2. Use the saved/default width
        current_width = self.window_width
        # Fallback if saved width is invalid or too small
        if current_width < 200: # Minimum sensible width
            current_width = 620 # Use default width

        # 3. Construct the geometry string
        geometry_string = f"{current_width}x{calculated_height}"
        if self.window_x is not None and self.window_y is not None:
            geometry_string += f"+{self.window_x}+{self.window_y}"
        else:
            # If no saved position, let Tkinter center it initially or place it at default
            pass # No position appended, Tkinter will handle default placement

        # 4. Apply the geometry
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
            logger.debug("==> Running in single file mode. Files will not be moved to 'finished' folders.")
            input_videos.append(input_source_clips_path)
            os.makedirs(output_splatted, exist_ok=True)
        elif is_source_dir and is_depth_dir:
            logger.debug("==> Running in batch (folder) mode.")
            finished_source_folder = os.path.join(input_source_clips_path, "finished")
            finished_depth_folder = os.path.join(input_depth_maps_path, "finished")
            os.makedirs(finished_source_folder, exist_ok=True)
            os.makedirs(finished_depth_folder, exist_ok=True)
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

    def check_queue(self):
        """Periodically checks the progress queue for updates to the GUI."""
        try:
            while True:
                message = self.progress_queue.get_nowait()
                if message == "finished":
                    self.status_label.config(text="Processing finished")
                    self.start_button.config(state="normal")
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

    # ======================================================================================
    # REFACTORED depthSplatting FUNCTION
    # ======================================================================================
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
            depth_dilate_size: int = 0,
            depth_blur_size: int = 0
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
                batch_depth_numpy_raw = depth_map_reader.get_batch(current_frame_indices).asnumpy()
                self._save_debug_numpy(batch_depth_numpy_raw, "01_RAW_INPUT", i, current_frame_indices[0], task_name="LowRes" if is_low_res_task else "HiRes")
                
                batch_depth_numpy = self._process_depth_batch(
                    batch_depth_numpy_raw=batch_depth_numpy_raw,
                    depth_stream_info=depth_stream_info,
                    depth_gamma=depth_gamma,
                    depth_dilate_size=depth_dilate_size,
                    depth_blur_size=depth_blur_size,
                    is_low_res_task=is_low_res_task,
                    max_raw_value=max_expected_raw_value,
                    global_depth_min=global_depth_min,
                    global_depth_max=global_depth_max
                )
                self._save_debug_numpy(batch_depth_numpy, "02_PROCESSED_PRE_NORM", i, current_frame_indices[0], task_name="LowRes" if is_low_res_task else "HiRes")

                batch_frames_float = batch_frames_numpy.astype("float32") / 255.0
                batch_depth_normalized = batch_depth_numpy.copy()

                if assume_raw_input:
                    if global_depth_max > 1.0:
                        batch_depth_normalized = batch_depth_numpy / global_depth_max
                else:
                    if global_depth_max - global_depth_min > 1e-5:
                        batch_depth_normalized = (batch_depth_numpy - global_depth_min) / (global_depth_max - global_depth_min)
                    else:
                        batch_depth_normalized = np.full_like(batch_depth_numpy, fill_value=zero_disparity_anchor_val, dtype=np.float32)

                batch_depth_normalized = np.clip(batch_depth_normalized, 0, 1)

                if not assume_raw_input and depth_gamma != 1.0:
                     batch_depth_normalized = np.power(batch_depth_normalized, depth_gamma)
                
                self._save_debug_numpy(batch_depth_normalized, "03_FINAL_NORMALIZED", i, current_frame_indices[0], task_name="LowRes" if is_low_res_task else "HiRes")
                
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
                        right_video_tensor = self._fill_left_edge_occlusions(right_video_tensor_raw, occlusion_mask_tensor, boundary_width_pixels=3)
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
        self.destroy()

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
    
    def show_about(self):
        """Displays the 'About' message box."""
        message = (
            f"Stereocrafter Splatting (Batch) - {GUI_VERSION}\n"
            "A tool for generating right-eye stereo views from source video and depth maps.\n"
            "Based on Decord, PyTorch, and OpenCV.\n"
            "\n(C) 2024 Some Rights Reserved"
        )
        tk.messagebox.showinfo("About Stereocrafter Splatting", message)

    def start_processing(self):
        """Starts the video processing in a separate thread."""
        self.stop_event.clear()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Starting processing...")
        # --- NEW: Disable all inputs at start ---
        self._set_input_state('disabled')

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
            
            depth_dilate_size_val = int(self.depth_dilate_size_var.get())
            if depth_dilate_size_val < 0:
                raise ValueError("Depth Dilate Size must be non-negative.")
            
            depth_blur_size_val = int(self.depth_blur_size_var.get())
            if depth_blur_size_val < 0:
                raise ValueError("Depth Blur Size must be non-negative.")
            if depth_blur_size_val > 0 and depth_blur_size_val % 2 == 0:
                raise ValueError("Depth Blur Size must be 0 or an odd number.")
            # --- END NEW Validation ---


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
            "depth_dilate_size": depth_dilate_size_val,
            "depth_blur_size": depth_blur_size_val,
            "auto_convergence_mode": self.auto_convergence_mode_var.get(),
            "enable_sidecar_gamma": self.enable_sidecar_gamma_var.get(),
            "enable_sidecar_blur_dilate": self.enable_sidecar_blur_dilate_var.get(),
        }
        self.processing_thread = threading.Thread(target=self._run_batch_process, args=(settings,))
        self.processing_thread.start()
        self.check_queue()

    def stop_processing(self):
        """Sets the stop event to gracefully halt processing."""
        self.stop_event.set()
        self.status_label.config(text="Stopping...")
        self.stop_button.config(state="disabled")

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
