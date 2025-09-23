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
from typing import Optional, Tuple

# Import custom modules
from dependency.forward_warp_pytorch import forward_warp
from dependency.stereocrafter_util import ( Tooltip, logger, get_video_stream_info, draw_progress_bar,
    check_cuda_availability, release_cuda_memory, CUDA_AVAILABLE
)

# Global flag for CUDA availability (set by check_cuda_availability at runtime)
CUDA_AVAILABLE = False

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
    def __init__(self):
        super().__init__(theme="default")
        self.title("Batch Depth Splatting")
        # self.geometry("620x750")
        self.configure(bg="#2b2b2b") # Set a dark background color for the root window

        self.app_config = {}
        self.help_texts = {}

        self._load_config()
        self._load_help_texts()

        # --- Variables with defaults ---
        self.dark_mode_var = tk.BooleanVar(value=self.app_config.get("dark_mode_enabled", False))
        self.input_source_clips_var = tk.StringVar(value=self.app_config.get("input_source_clips", "./input_source_clips"))
        self.input_depth_maps_var = tk.StringVar(value=self.app_config.get("input_depth_maps", "./input_depth_maps"))
        self.output_splatted_var = tk.StringVar(value=self.app_config.get("output_splatted", "./output_splatted"))
        self.max_disp_var = tk.StringVar(value=self.app_config.get("max_disp", "20.0"))
        self.process_length_var = tk.StringVar(value=self.app_config.get("process_length", "-1"))
        self.batch_size_var = tk.StringVar(value=self.app_config.get("batch_size", "10"))
        self.dual_output_var = tk.BooleanVar(value=self.app_config.get("dual_output", False))
        self.enable_autogain_var = tk.BooleanVar(value=self.app_config.get("enable_autogain", True)) 
        self.enable_full_res_var = tk.BooleanVar(value=self.app_config.get("enable_full_resolution", True))
        self.enable_low_res_var = tk.BooleanVar(value=self.app_config.get("enable_low_resolution", False))
        self.pre_res_width_var = tk.StringVar(value=self.app_config.get("pre_res_width", "1920"))
        self.pre_res_height_var = tk.StringVar(value=self.app_config.get("pre_res_height", "1080"))
        self.low_res_batch_size_var = tk.StringVar(value=self.app_config.get("low_res_batch_size", "25"))
        self.zero_disparity_anchor_var = tk.StringVar(value=self.app_config.get("convergence_point", "0.5"))

        # --- Variables for "Current Processing Information" display ---
        self.processing_filename_var = tk.StringVar(value="N/A")
        self.processing_resolution_var = tk.StringVar(value="N/A")
        self.processing_frames_var = tk.StringVar(value="N/A")
        self.processing_disparity_var = tk.StringVar(value="N/A")
        self.processing_convergence_var = tk.StringVar(value="N/A")
        self.processing_task_name_var = tk.StringVar(value="N/A")

        # --- Processing control variables ---
        self.stop_event = threading.Event()
        self.progress_queue = queue.Queue()
        self.processing_thread = None
        self.fixed_window_width = 620 # <--- NEW: Define your desired fixed width here

        self._create_widgets()
        # Initialize ttk.Style after the main Tk app has been created
        # This ensures the style object is fully functional when _apply_theme is called.
        self.style = ttk.Style() # <--- NEW LINE: Explicitly create ttk.Style instance
        
        self._apply_theme()

        self.after(10, self.toggle_processing_settings_fields) # Set initial state
        self.after(100, self.check_queue) # Start checking progress queue

        # Bind closing protocol
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

    def _apply_theme(self: "SplatterGUI"):
        """Applies the selected theme (dark or light) to the GUI, and adjusts window height."""
        if self.dark_mode_var.get():
            # --- Dark Theme ---
            bg_color = "#2b2b2b"
            fg_color = "white"
            entry_bg = "#3c3c3c"
            
            self.style.theme_use("black")
            self.configure(bg=bg_color)

            if hasattr(self, 'menubar'): 
                menu_bg = "#3c3c3c"
                menu_fg = "white"
                active_bg = "#555555"
                active_fg = "white"
                self.menubar.config(bg=menu_bg, fg=menu_fg, activebackground=active_bg, activeforeground=active_fg)
                self.option_menu.config(bg=menu_bg, fg=menu_fg, activebackground=active_bg, activeforeground=active_fg)
                self.theme_menu.config(bg=menu_bg, fg=menu_fg, activebackground=active_bg, activeforeground=active_fg)
            
            self.style.configure("TEntry", fieldbackground=entry_bg, foreground=fg_color, insertcolor=fg_color)
            
            if hasattr(self, 'info_frame'):
                for label in self.info_labels:
                    label.config(bg=entry_bg, fg=fg_color)

        else:
            # --- Light Theme ---
            bg_color = "#d9d9d9"
            fg_color = "black"
            entry_bg = "#f0f0f0"

            self.style.theme_use("default")
            self.configure(bg=bg_color)

            if hasattr(self, 'menubar'):
                menu_bg = "#f0f0f0"
                menu_fg = "black"
                active_bg = "#dddddd"
                active_fg = "black"
                self.menubar.config(bg=menu_bg, fg=menu_fg, activebackground=active_bg, activeforeground=active_fg)
                self.option_menu.config(bg=menu_bg, fg=menu_fg, activebackground=active_bg, activeforeground=active_fg)
                self.theme_menu.config(bg=menu_bg, fg=menu_fg, activebackground=active_bg, activeforeground=active_fg)

            self.style.configure("TEntry", fieldbackground=entry_bg, foreground=fg_color, insertcolor=fg_color)
            
            if hasattr(self, 'info_frame'):
                for label in self.info_labels:
                    label.config(bg=bg_color, fg=fg_color)

        # --- Dynamic Height Adjustment (always runs after theme change) ---
        # Force Tkinter to calculate the actual required size for the new theme
        self.update_idletasks()
        
        # Get the new optimal height based on the current content and theme padding
        new_height = self.winfo_reqheight() 
        
        # Apply the fixed width and the new calculated height
        self.geometry(f"{self.fixed_window_width}x{new_height}")

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

    def _create_hover_tooltip(self, widget, key):
        """Creates a tooltip for a given widget based on a key from help_texts."""
        if key in self.help_texts:
            Tooltip(widget, self.help_texts[key])

    def _create_widgets(self):
        """Initializes and places all GUI widgets."""
        # --- Menu Bar ---
        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)

        self.option_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Option", menu=self.option_menu)
        self.option_menu.add_command(label="Reset to Default", command=self.reset_to_defaults)
        self.option_menu.add_command(label="Restore Finished", command=self.restore_finished_files)
        self.theme_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Theme", menu=self.theme_menu)
        self.theme_menu.add_checkbutton(label="Dark Mode", variable=self.dark_mode_var, command=self._apply_theme)

        # --- Folder selection frame ---
        folder_frame = ttk.LabelFrame(self, text="Input/Output Folders")
        folder_frame.pack(pady=10, padx=10, fill="x")
        folder_frame.grid_columnconfigure(1, weight=1)

        # Input Source Clips Row
        lbl_source_clips = ttk.Label(folder_frame, text="Input Source Clips:")
        lbl_source_clips.grid(row=0, column=0, sticky="e", padx=5, pady=2)
        entry_source_clips = ttk.Entry(folder_frame, textvariable=self.input_source_clips_var)
        entry_source_clips.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        btn_browse_source_clips_folder = ttk.Button(folder_frame, text="Browse Folder", command=lambda: self._browse_folder(self.input_source_clips_var))
        btn_browse_source_clips_folder.grid(row=0, column=2, padx=2, pady=2)
        btn_select_source_clips_file = ttk.Button(folder_frame, text="Select File", command=lambda: self._browse_file(self.input_source_clips_var, [("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]))
        btn_select_source_clips_file.grid(row=0, column=3, padx=2, pady=2)
        self._create_hover_tooltip(lbl_source_clips, "input_source_clips")
        self._create_hover_tooltip(entry_source_clips, "input_source_clips")
        self._create_hover_tooltip(btn_browse_source_clips_folder, "input_source_clips_folder")
        self._create_hover_tooltip(btn_select_source_clips_file, "input_source_clips_file")

        # Input Depth Maps Row
        lbl_input_depth_maps = ttk.Label(folder_frame, text="Input Depth Maps:")
        lbl_input_depth_maps.grid(row=1, column=0, sticky="e", padx=5, pady=2)
        entry_input_depth_maps = ttk.Entry(folder_frame, textvariable=self.input_depth_maps_var)
        entry_input_depth_maps.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        btn_browse_input_depth_maps_folder = ttk.Button(folder_frame, text="Browse Folder", command=lambda: self._browse_folder(self.input_depth_maps_var))
        btn_browse_input_depth_maps_folder.grid(row=1, column=2, padx=2, pady=2)
        btn_select_input_depth_maps_file = ttk.Button(folder_frame, text="Select File", command=lambda: self._browse_file(self.input_depth_maps_var, [("Depth Files", "*.mp4 *.npz"), ("All files", "*.*")]))
        btn_select_input_depth_maps_file.grid(row=1, column=3, padx=2, pady=2)
        self._create_hover_tooltip(lbl_input_depth_maps, "input_depth_maps")
        self._create_hover_tooltip(entry_input_depth_maps, "input_depth_maps")
        self._create_hover_tooltip(btn_browse_input_depth_maps_folder, "input_depth_maps_folder")
        self._create_hover_tooltip(btn_select_input_depth_maps_file, "input_depth_maps_file")

        # Output Splatted Row
        lbl_output_splatted = ttk.Label(folder_frame, text="Output Splatted:")
        lbl_output_splatted.grid(row=2, column=0, sticky="e", padx=5, pady=2)
        entry_output_splatted = ttk.Entry(folder_frame, textvariable=self.output_splatted_var)
        entry_output_splatted.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        btn_browse_output_splatted = ttk.Button(folder_frame, text="Browse Folder", command=lambda: self._browse_folder(self.output_splatted_var))
        btn_browse_output_splatted.grid(row=2, column=2, columnspan=2, padx=5, pady=2)
        self._create_hover_tooltip(lbl_output_splatted, "output_splatted")
        self._create_hover_tooltip(entry_output_splatted, "output_splatted")
        self._create_hover_tooltip(btn_browse_output_splatted, "output_splatted")

        # --- Process Resolution and Settings Frame ---
        self.preprocessing_frame = ttk.LabelFrame(self, text="Process Resolution and Settings")
        self.preprocessing_frame.pack(pady=10, padx=10, fill="x")
        self.preprocessing_frame.grid_columnconfigure(1, weight=1)
        self.preprocessing_frame.grid_columnconfigure(3, weight=1)

        # Enable Full Resolution Section
        self.enable_full_res_checkbox = ttk.Checkbutton(self.preprocessing_frame, text="Enable Full Resolution Output (Native Video Resolution)", variable=self.enable_full_res_var, command=self.toggle_processing_settings_fields)
        self.enable_full_res_checkbox.grid(row=0, column=0, columnspan=4, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.enable_full_res_checkbox, "enable_full_res")

        self.lbl_full_res_batch_size = ttk.Label(self.preprocessing_frame, text="Full Res Batch Size:")
        self.lbl_full_res_batch_size.grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.entry_full_res_batch_size = ttk.Entry(self.preprocessing_frame, textvariable=self.batch_size_var, width=15)
        self.entry_full_res_batch_size.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.lbl_full_res_batch_size, "full_res_batch_size")
        self._create_hover_tooltip(self.entry_full_res_batch_size, "full_res_batch_size")

        # Enable Low Resolution Section
        self.enable_low_res_checkbox = ttk.Checkbutton(self.preprocessing_frame, text="Enable Low Resolution Output (Pre-defined Below)", variable=self.enable_low_res_var, command=self.toggle_processing_settings_fields)
        self.enable_low_res_checkbox.grid(row=2, column=0, columnspan=4, sticky="w", padx=5, pady=(10, 2))
        self._create_hover_tooltip(self.enable_low_res_checkbox, "enable_low_res")

        self.pre_res_width_label = ttk.Label(self.preprocessing_frame, text="Low Res Width:")
        self.pre_res_width_label.grid(row=3, column=0, sticky="e", padx=5, pady=2)
        self.pre_res_width_entry = ttk.Entry(self.preprocessing_frame, textvariable=self.pre_res_width_var, width=10)
        self.pre_res_width_entry.grid(row=3, column=1, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.pre_res_width_label, "low_res_width")
        self._create_hover_tooltip(self.pre_res_width_entry, "low_res_width")

        self.pre_res_height_label = ttk.Label(self.preprocessing_frame, text="Low Res Height:")
        self.pre_res_height_label.grid(row=3, column=2, sticky="e", padx=5, pady=2)
        self.pre_res_height_entry = ttk.Entry(self.preprocessing_frame, textvariable=self.pre_res_height_var, width=10)
        self.pre_res_height_entry.grid(row=3, column=3, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.pre_res_height_label, "low_res_height")
        self._create_hover_tooltip(self.pre_res_height_entry, "low_res_height")

        self.lbl_low_res_batch_size = ttk.Label(self.preprocessing_frame, text="Low Res Batch Size:")
        self.lbl_low_res_batch_size.grid(row=4, column=0, sticky="e", padx=5, pady=2)
        self.entry_low_res_batch_size = ttk.Entry(self.preprocessing_frame, textvariable=self.low_res_batch_size_var, width=15)
        self.entry_low_res_batch_size.grid(row=4, column=1, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(self.lbl_low_res_batch_size, "low_res_batch_size")
        self._create_hover_tooltip(self.entry_low_res_batch_size, "low_res_batch_size")

        # --- Output Settings Frame ---
        output_settings_frame = ttk.LabelFrame(self, text="Splatting & Output Settings")
        output_settings_frame.pack(pady=10, padx=10, fill="x")

        lbl_max_disp = ttk.Label(output_settings_frame, text="Max Disparity %:")
        lbl_max_disp.grid(row=0, column=0, sticky="e", padx=5, pady=2)
        entry_max_disp = ttk.Entry(output_settings_frame, textvariable=self.max_disp_var, width=15)
        entry_max_disp.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(lbl_max_disp, "max_disp")
        self._create_hover_tooltip(entry_max_disp, "max_disp")

        lbl_zero_disparity_anchor = ttk.Label(output_settings_frame, text="Convergence Point (0-1):")
        lbl_zero_disparity_anchor.grid(row=1, column=0, sticky="e", padx=5, pady=2)
        entry_zero_disparity_anchor = ttk.Entry(output_settings_frame, textvariable=self.zero_disparity_anchor_var, width=15)
        entry_zero_disparity_anchor.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(lbl_zero_disparity_anchor, "convergence_point")
        self._create_hover_tooltip(entry_zero_disparity_anchor, "convergence_point")

        lbl_process_length = ttk.Label(output_settings_frame, text="Process Length (-1 for all):")
        lbl_process_length.grid(row=2, column=0, sticky="e", padx=5, pady=2)
        entry_process_length = ttk.Entry(output_settings_frame, textvariable=self.process_length_var, width=15)
        entry_process_length.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(lbl_process_length, "process_length")
        self._create_hover_tooltip(entry_process_length, "process_length")

        dual_output_checkbox = ttk.Checkbutton(output_settings_frame, text="Dual Output (Mask & Warped)", variable=self.dual_output_var)
        dual_output_checkbox.grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(dual_output_checkbox, "dual_output")

        # Autogain Checkbox (NEW)
        autogain_checkbox = ttk.Checkbutton(output_settings_frame, text="Enable Autogain (Per-Chunk Normalization)", variable=self.enable_autogain_var)
        autogain_checkbox.grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        self._create_hover_tooltip(autogain_checkbox, "enable_autogain")

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

    def _load_config(self):
        """Loads configuration from config_splat.json."""
        if os.path.exists("config_splat.json"):
            with open("config_splat.json", "r") as f:
                self.app_config = json.load(f)

    def _save_config(self):
        """Saves current GUI settings to config_splat.json."""
        config = {
            "input_source_clips": self.input_source_clips_var.get(),
            "input_depth_maps": self.input_depth_maps_var.get(),
            "output_splatted": self.output_splatted_var.get(),
            "max_disp": self.max_disp_var.get(),
            "process_length": self.process_length_var.get(),
            "batch_size": self.batch_size_var.get(),
            "dual_output": self.dual_output_var.get(),
            "enable_autogain": self.enable_autogain_var.get(),
            "enable_full_resolution": self.enable_full_res_var.get(),
            "enable_low_resolution": self.enable_low_res_var.get(),
            "pre_res_width": self.pre_res_width_var.get(),
            "pre_res_height": self.pre_res_height_var.get(),
            "low_res_batch_size": self.low_res_batch_size_var.get(),
            "convergence_point": self.zero_disparity_anchor_var.get(),
            "dark_mode_enabled": self.dark_mode_var.get(),
        }
        with open("config_splat.json", "w") as f:
            json.dump(config, f, indent=4)

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

    def clear_processing_info(self):
        """Resets all 'Current Processing Information' labels to default 'N/A'."""
        self.processing_filename_var.set("N/A")
        self.processing_resolution_var.set("N/A")
        self.processing_frames_var.set("N/A")
        self.processing_disparity_var.set("N/A")
        self.processing_convergence_var.set("N/A")
        self.processing_task_name_var.set("N/A")

    def depthSplatting(
            self: "SplatterGUI", # Added self and type hint for Pylance
            input_video_reader: VideoReader,
            depth_map_reader: VideoReader,
            total_frames_to_process: int,
            processed_fps: float,
            output_video_path_base: str, # This is the full path including name, without final suffix
            target_output_height: int, # The resolution the video readers were configured to output
            target_output_width: int,  # The resolution the video readers were configured to output
            max_disp: float,
            process_length: int, # Global limit from GUI
            batch_size: int, # This is now the chunk size for disk reading
            dual_output: bool,
            zero_disparity_anchor_val: float,
            video_stream_info: Optional[dict],
            frame_overlap: Optional[int],
            input_bias: Optional[float],
            enable_autogain: bool,
            global_depth_min: float,
            global_depth_max: float
        ):
        logger.debug("==> Initializing ForwardWarpStereo module")
        stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

        # `num_frames` now comes from the parameter `total_frames_to_process`
        num_frames = total_frames_to_process
        height, width = target_output_height, target_output_width # Use configured target output dimensions

        os.makedirs(os.path.dirname(output_video_path_base), exist_ok=True)

        # Determine output video suffix
        if dual_output:
            suffix = "_splatted2"
        else:
            suffix = "_splatted4"

        res_suffix = f"_{width}"
        final_output_video_path = f"{os.path.splitext(output_video_path_base)[0]}{res_suffix}{suffix}.mp4"

        temp_png_dir = os.path.join(os.path.dirname(final_output_video_path), "temp_splat_pngs_" + os.path.basename(os.path.splitext(output_video_path_base)[0]))
        os.makedirs(temp_png_dir, exist_ok=True)
        logger.debug(f"==> Writing temporary PNG sequence to: {temp_png_dir}")

        frame_count = 0
        logger.debug(f"==> Generating PNG sequence for {os.path.basename(final_output_video_path)}")
        draw_progress_bar(frame_count, num_frames, prefix=f"  Progress:")


        # global_depth_min and global_depth_max are now passed as parameters and should be used directly.
        # Remove the placeholder redefinitions from here.
        
        if not enable_autogain:
            # The warning message below is slightly misleading now, as global min/max *are* passed.
            # We can update this warning or remove it, as the logic is now in place.
            logger.debug("Autogain is disabled. Using pre-computed global min/max for consistent depth normalization.")
        else:
            logger.info("Autogain is ENABLED. Depth maps will be normalized per-chunk, leading to potential brightness jumps between segments.")


        for i in range(0, num_frames, batch_size): # batch_size is now chunk_size
            if self.stop_event.is_set():
                draw_progress_bar(frame_count, num_frames, suffix='Stopped')
                print()
                del stereo_projector
                torch.cuda.empty_cache()
                gc.collect()
                if os.path.exists(temp_png_dir):
                    shutil.rmtree(temp_png_dir)
                return

            # Read frames directly from VideoReader objects for the current chunk
            current_frame_indices = list(range(i, min(i + batch_size, num_frames)))
            if not current_frame_indices:
                break

            batch_frames_numpy = input_video_reader.get_batch(current_frame_indices).asnumpy()
            batch_depth_numpy_raw = depth_map_reader.get_batch(current_frame_indices).asnumpy()

            # Process depth frames (grayscale conversion if needed)
            if batch_depth_numpy_raw.ndim == 4: # If depth video is RGB
                batch_depth_numpy = batch_depth_numpy_raw.mean(axis=-1)
            else: # If depth video is grayscale (ndim 3: T,H,W or T,H,W,1)
                batch_depth_numpy = batch_depth_numpy_raw.squeeze(-1) if batch_depth_numpy_raw.ndim == 4 else batch_depth_numpy_raw

            # Normalize depth frames
            if enable_autogain:
                # Local per-chunk normalization (will cause brightness jumps)
                local_depth_min = batch_depth_numpy.min()
                local_depth_max = batch_depth_numpy.max()
                if local_depth_max - local_depth_min > 1e-5:
                    batch_depth_normalized = (batch_depth_numpy - local_depth_min) / (local_depth_max - local_depth_min)
                else:
                    batch_depth_normalized = np.zeros_like(batch_depth_numpy)
                logger.debug(f"Chunk {i}-{i+len(current_frame_indices)} depth autogained (min-max scaled) from [{local_depth_min:.3f}, {local_depth_max:.3f}] to [0, 1].")
            else:
                # Global normalization using pre-computed min/max (ensures consistency)
                logger.debug(f"Applying global normalization for chunk {i}-{i+len(current_frame_indices)} using min={global_depth_min:.3f}, max={global_depth_max:.3f}.")
                if global_depth_max - global_depth_min > 1e-5:
                    batch_depth_normalized = (batch_depth_numpy - global_depth_min) / (global_depth_max - global_depth_min)
                else:
                    batch_depth_normalized = np.full_like(batch_depth_numpy, fill_value=zero_disparity_anchor_val)
                    logger.warning(f"Global depth range for normalization is too small ({global_depth_min:.3f}-{global_depth_max:.3f}). Setting depth map to constant {zero_disparity_anchor_val} for this video.")
            
            # Clip to ensure values are strictly within 0-1 range after normalization
            batch_depth_normalized = np.clip(batch_depth_normalized, 0, 1)

            # Convert original batch frames to float 0-1 for display in grid
            batch_frames_float = batch_frames_numpy.astype("float32") / 255.0

            # Generate depth visualization for the current chunk
            batch_depth_vis_list = []
            for d_frame in batch_depth_normalized:
                vis_frame = cv2.applyColorMap((np.clip(d_frame, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                batch_depth_vis_list.append(vis_frame.astype("float32") / 255.0)
            batch_depth_vis = np.stack(batch_depth_vis_list, axis=0) # [T, H, W, 3]

            # Convert to tensors for GPU processing
            left_video_tensor = torch.from_numpy(batch_frames_numpy).permute(0, 3, 1, 2).float().cuda() / 255.0
            disp_map_tensor = torch.from_numpy(batch_depth_normalized).unsqueeze(1).float().cuda()

            disp_map_tensor = (disp_map_tensor - zero_disparity_anchor_val) * 2.0
            disp_map_tensor = disp_map_tensor * max_disp

            with torch.no_grad():
                right_video_tensor, occlusion_mask_tensor = stereo_projector(left_video_tensor, disp_map_tensor)
            
            right_video_numpy = right_video_tensor.cpu().permute(0, 2, 3, 1).numpy()
            occlusion_mask_numpy = occlusion_mask_tensor.cpu().permute(0, 2, 3, 1).numpy().repeat(3, axis=-1)

            for j in range(len(batch_frames_numpy)):
                if dual_output:
                    video_grid = np.concatenate([occlusion_mask_numpy[j], right_video_numpy[j]], axis=1)
                else:
                    video_grid_top = np.concatenate([batch_frames_float[j], batch_depth_vis[j]], axis=1)
                    video_grid_bottom = np.concatenate([occlusion_mask_numpy[j], right_video_numpy[j]], axis=1)
                    video_grid = np.concatenate([video_grid_top, video_grid_bottom], axis=0)

                video_grid_uint16 = np.clip(video_grid, 0.0, 1.0) * 65535.0
                video_grid_uint16 = video_grid_uint16.astype(np.uint16)
                video_grid_bgr = cv2.cvtColor(video_grid_uint16, cv2.COLOR_RGB2BGR)

                png_filename = os.path.join(temp_png_dir, f"{frame_count:05d}.png")
                cv2.imwrite(png_filename, video_grid_bgr)

                frame_count += 1

            del left_video_tensor, disp_map_tensor, right_video_tensor, occlusion_mask_tensor
            torch.cuda.empty_cache()
            gc.collect()
            
            draw_progress_bar(frame_count, num_frames, prefix=f"  Progress:")

        logger.debug(f"==> Temporary PNG sequence generation completed ({frame_count} frames).")
        
        del stereo_projector
        torch.cuda.empty_cache()
        gc.collect()

        # --- FFmpeg encoding (remains largely the same) ---
        # ... (FFmpeg command construction logic, identical to previous version) ...
        # This part will use final_output_video_path, video_stream_info, processed_fps etc.

        # I'm omitting the FFmpeg and cleanup block here for brevity,
        # but it should be kept as it was previously.
        # Ensure final_output_video_path and temp_png_dir are correctly scoped.
        
        # Original FFmpeg block starts here:
        logger.debug(f"==> Encoding final video from PNG sequence using ffmpeg for '{os.path.basename(final_output_video_path)}'.")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-framerate", str(processed_fps), # Input framerate for the PNG sequence
            "-i", os.path.join(temp_png_dir, "%05d.png"), # Input PNG sequence pattern
        ]

        original_codec_name = video_stream_info.get("codec_name") if video_stream_info else None
        original_pix_fmt = video_stream_info.get("pix_fmt") if video_stream_info else None

        is_original_10bit_or_higher = False
        if original_pix_fmt:
            if "10" in original_pix_fmt or "12" in original_pix_fmt or "16" in original_pix_fmt:
                is_original_10bit_or_higher = True
                logger.debug(f"==> Detected original video pixel format: {original_pix_fmt} (>= 10-bit)")
            else:
                logger.debug(f"==> Detected original video pixel format: {original_pix_fmt} (< 10-bit)")
        else:
            logger.debug("==> Could not detect original video pixel format.")

        output_codec = "libx264"
        output_pix_fmt = "yuv420p"
        output_crf = "23" # Default CRF for H.264 (medium quality)
        output_profile = "main"
        x265_params = []

        nvenc_preset = "medium"
        nvenc_cq = "23"

        is_hdr_source = False
        if video_stream_info and video_stream_info.get("color_primaries") == "bt2020" and \
        video_stream_info.get("transfer_characteristics") == "smpte2084":
            is_hdr_source = True
            logger.debug("==> Source detected as HDR.")

        if video_stream_info:
            logger.debug("==> Source video stream info detected. Attempting to match source characteristics and optimize quality.")

            if is_hdr_source:
                logger.debug("==> Detected HDR source. Targeting HEVC (x265) 10-bit HDR output.")
                output_codec = "libx265"
                if CUDA_AVAILABLE:
                    output_codec = "hevc_nvenc"
                    logger.debug("    (Using hevc_nvenc for hardware acceleration)")
                
                output_pix_fmt = "yuv420p10le"
                output_crf = "28"
                output_profile = "main10"
                if video_stream_info.get("mastering_display_metadata"):
                    md_meta = video_stream_info["mastering_display_metadata"]
                    x265_params.append(f"mastering-display={md_meta}")
                    logger.debug(f"==> Adding mastering display metadata: {md_meta}")
                if video_stream_info.get("max_content_light_level"):
                    max_cll_meta = video_stream_info["max_content_light_level"]
                    x265_params.append(f"max-cll={max_cll_meta}")
                    logger.debug(f"==> Adding max content light level: {max_cll_meta}")

            elif original_codec_name == "hevc" and is_original_10bit_or_higher:
                logger.debug("==> Detected 10-bit HEVC (x265) SDR source. Targeting HEVC (x265) 10-bit SDR output.")
                output_codec = "libx265"
                if CUDA_AVAILABLE:
                    output_codec = "hevc_nvenc"
                    logger.debug("    (Using hevc_nvenc for hardware acceleration)")
                
                output_pix_fmt = "yuv420p10le"
                output_crf = "24"
                output_profile = "main10"

            else:
                logger.debug("==> No specific HEVC/HDR source. Targeting H.264 (x264) 8-bit SDR high quality.")
                output_codec = "libx264"
                if CUDA_AVAILABLE:
                    output_codec = "h264_nvenc"
                    logger.debug("    (Using h264_nvenc for hardware acceleration)")
                
                output_pix_fmt = "yuv420p"
                output_crf = "18"
                output_profile = "main"

        else:
            logger.debug("==> No source video stream info detected. Falling back to default H.264 (x264) 8-bit SDR (medium quality).")
            if CUDA_AVAILABLE:
                output_codec = "h264_nvenc"
                logger.debug("    (Using h264_nvenc for hardware acceleration for default output)")


        ffmpeg_cmd.extend(["-c:v", output_codec])
        if "nvenc" in output_codec:
            ffmpeg_cmd.extend(["-preset", nvenc_preset])
            ffmpeg_cmd.extend(["-cq", nvenc_cq])
            if "-crf" in ffmpeg_cmd:
                crf_index = ffmpeg_cmd.index("-crf")
                del ffmpeg_cmd[crf_index:crf_index+2]
        else:
            ffmpeg_cmd.extend(["-crf", output_crf])
        
        ffmpeg_cmd.extend(["-pix_fmt", output_pix_fmt])
        if output_profile:
            ffmpeg_cmd.extend(["-profile:v", output_profile])

        if output_codec == "libx265" and x265_params:
            ffmpeg_cmd.extend(["-x265-params", ":".join(x265_params)])

        if video_stream_info:
            if video_stream_info.get("color_primaries") and video_stream_info["color_primaries"] not in ["N/A", "und", "unknown"]:
                ffmpeg_cmd.extend(["-color_primaries", video_stream_info["color_primaries"]])
            if video_stream_info.get("transfer_characteristics") and video_stream_info["transfer_characteristics"] not in ["N/A", "und", "unknown"]:
                ffmpeg_cmd.extend(["-color_trc", video_stream_info["transfer_characteristics"]])
            if video_stream_info.get("color_space") and video_stream_info["color_space"] not in ["N/A", "und", "unknown"]:
                ffmpeg_cmd.extend(["-colorspace", video_stream_info["color_space"]])

        ffmpeg_cmd.append(final_output_video_path)

        logger.debug(f"==> Executing ffmpeg command: {' '.join(ffmpeg_cmd)}")

        try:
            ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True, encoding='utf-8', timeout=3600*24)
            logger.debug(f"==> FFmpeg stdout: {ffmpeg_result.stdout}")
            logger.debug(f"==> FFmpeg stderr: {ffmpeg_result.stderr}")
            logger.debug(f"==> Final video successfully encoded to '{os.path.basename(final_output_video_path)}'.")
        except FileNotFoundError:
            logger.error("==> Error: ffmpeg not found. Please ensure FFmpeg is installed and in your system's PATH. No final video generated.")
            messagebox.showerror("FFmpeg Error", "ffmpeg not found. Please install FFmpeg and ensure it's in your system's PATH to encode from PNGs.")
        except subprocess.CalledProcessError as e:
            logger.error(f"==> Error running ffmpeg for '{os.path.basename(final_output_video_path)}': {e.returncode}")
            logger.error(f"==> FFmpeg stdout: {e.stdout}")
            logger.error(f"==> FFmpeg stderr: {e.stderr}")
            logger.error("==> Final video encoding failed due to ffmpeg error.")
        except subprocess.TimeoutExpired:
            logger.error(f"==> Error: FFmpeg encoding timed out for '{os.path.basename(final_output_video_path)}'.")
        except Exception as e:
            logger.error(f"==> An unexpected error occurred during ffmpeg execution: {e}")

        logger.debug(f"==> Final output video written to: {final_output_video_path}")

        output_sidecar_data = {}
        output_sidecar_data["convergence_plane"] = zero_disparity_anchor_val
        output_sidecar_data["max_disparity"] = (max_disp / width) * 100.0
        
        if frame_overlap is not None:
            output_sidecar_data["frame_overlap"] = frame_overlap
        if input_bias is not None:
            output_sidecar_data["input_bias"] = input_bias

        if frame_overlap is not None or input_bias is not None:
            output_sidecar_path = f"{os.path.splitext(final_output_video_path)[0]}.json"
            try:
                with open(output_sidecar_path, 'w') as f:
                    json.dump(output_sidecar_data, f, indent=4)
                logger.debug(f"==> Created output sidecar JSON: {output_sidecar_path}")
            except Exception as e:
                logger.error(f"==> Error creating output sidecar JSON '{output_sidecar_path}': {e}")

        if os.path.exists(temp_png_dir):
            try:
                shutil.rmtree(temp_png_dir)
                logger.debug(f"==> Cleaned up temporary PNG directory: {temp_png_dir}")
            except Exception as e:
                logger.error(f"==> Error cleaning up temporary PNG directory {temp_png_dir}: {e}")
                return True

        logger.debug(f"==> Final output video written to: {final_output_video_path}")
        return True

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

    def start_processing(self):
        """Starts the video processing in a separate thread."""
        self.stop_event.clear()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Starting processing...")

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
        }
        self.processing_thread = threading.Thread(target=self._run_batch_process, args=(settings,))
        self.processing_thread.start()
        self.check_queue()

    def stop_processing(self):
        """Sets the stop event to gracefully halt processing."""
        self.stop_event.set()
        self.status_label.config(text="Stopping...")
        self.stop_button.config(state="disabled")

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
                    if "task_name" in info_data:
                        self.processing_task_name_var.set(info_data["task_name"])

        except queue.Empty:
            pass
        self.after(100, self.check_queue)

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
        self.enable_autogain_var.set(True)
        self.zero_disparity_anchor_var.set("0.5")
        
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

    def _run_batch_process(self, settings):
        """
        The main processing logic, run in a separate thread.
        This was the original global 'main' function.
        """
        input_source_clips_path_setting = settings["input_source_clips"]
        input_depth_maps_path_setting = settings["input_depth_maps"]
        output_splatted = settings["output_splatted"]
        gui_max_disp = float(settings["max_disp"])
        process_length = settings["process_length"]
        dual_output = settings["dual_output"]
        default_zero_disparity_anchor = settings["zero_disparity_anchor"]

        enable_full_resolution = settings["enable_full_resolution"]
        full_res_batch_size = settings["full_res_batch_size"]
        enable_low_resolution = settings["enable_low_resolution"]
        low_res_width = settings["low_res_width"]
        low_res_height = settings["low_res_height"]
        low_res_batch_size = settings["low_res_batch_size"]

        is_single_file_mode = False
        input_videos = []
        finished_source_folder = None
        finished_depth_folder = None
        

        is_source_file = os.path.isfile(input_source_clips_path_setting)
        is_source_dir = os.path.isdir(input_source_clips_path_setting)
        is_depth_file = os.path.isfile(input_depth_maps_path_setting)
        is_depth_dir = os.path.isdir(input_depth_maps_path_setting)

        self.after(0, self.clear_processing_info)

        if is_source_file and is_depth_file:
            is_single_file_mode = True
            logger.debug("==> Running in single file mode. Files will not be moved to 'finished' folders.")
            input_videos.append(input_source_clips_path_setting)
            os.makedirs(output_splatted, exist_ok=True)
        elif is_source_dir and is_depth_dir:
            logger.debug("==> Running in batch (folder) mode.")
            finished_source_folder = os.path.join(input_source_clips_path_setting, "finished")
            finished_depth_folder = os.path.join(input_depth_maps_path_setting, "finished")
            os.makedirs(finished_source_folder, exist_ok=True)
            os.makedirs(finished_depth_folder, exist_ok=True)
            os.makedirs(output_splatted, exist_ok=True)

            video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
            for ext in video_extensions:
                input_videos.extend(glob.glob(os.path.join(input_source_clips_path_setting, ext)))
            input_videos = sorted(input_videos)
        else:
            logger.error("==> Error: Input Source Clips and Input Depth Maps must both be either files or directories. Skipping processing.")
            self.progress_queue.put("finished")
            release_cuda_memory()
            return

        if not input_videos:
            logger.error(f"No video files found in {input_source_clips_path_setting}")
            self.progress_queue.put("finished")
            release_cuda_memory()
            return

        total_processing_tasks_count = 0
        if enable_full_resolution:
            total_processing_tasks_count += len(input_videos)
        if enable_low_resolution:
            total_processing_tasks_count += len(input_videos)

        if total_processing_tasks_count == 0:
            logger.error("==> Error: No resolution output enabled. Processing stopped.")
            self.progress_queue.put("finished")
            release_cuda_memory()
            return

        self.progress_queue.put(("total", total_processing_tasks_count))
        overall_task_counter = 0

        for idx, video_path in enumerate(input_videos):
            if self.stop_event.is_set():
                logger.info("==> Stopping processing due to user request")
                release_cuda_memory()
                self.progress_queue.put("finished")
                self.after(0, self.clear_processing_info)
                return

            video_name = os.path.splitext(os.path.basename(video_path))[0]
            logger.info(f"==> Processing Video: {video_name}")

            self.progress_queue.put(("update_info", {"filename": video_name}))

            current_zero_disparity_anchor = default_zero_disparity_anchor
            current_max_disparity_percentage = gui_max_disp

            current_frame_overlap = None
            current_input_bias = None

            actual_depth_map_path = None
            if is_single_file_mode:
                actual_depth_map_path = input_depth_maps_path_setting
                if not os.path.exists(actual_depth_map_path):
                    logger.error(f"==> Error: Single depth map file '{actual_depth_map_path}' not found. Skipping this video.")
                    self.progress_queue.put(("processed", overall_task_counter))
                    continue
            else:
                depth_map_path_mp4 = os.path.join(input_depth_maps_path_setting, f"{video_name}_depth.mp4")
                depth_map_path_npz = os.path.join(input_depth_maps_path_setting, f"{video_name}_depth.npz")

                if os.path.exists(depth_map_path_mp4):
                    actual_depth_map_path = depth_map_path_mp4
                elif os.path.exists(depth_map_path_npz):
                    actual_depth_map_path = depth_map_path_npz

            if actual_depth_map_path:
                actual_depth_map_path = os.path.normpath(actual_depth_map_path)
                depth_map_basename = os.path.splitext(os.path.basename(actual_depth_map_path))[0]
                json_sidecar_path = os.path.join(os.path.dirname(actual_depth_map_path), f"{depth_map_basename}.json")

                anchor_source = "GUI"
                max_disp_source = "GUI"

                if os.path.exists(json_sidecar_path):
                    try:
                        with open(json_sidecar_path, 'r') as f:
                            sidecar_data = json.load(f)

                        if "convergence_plane" in sidecar_data and isinstance(sidecar_data["convergence_plane"], (int, float)):
                            current_zero_disparity_anchor = float(sidecar_data["convergence_plane"])
                            logger.debug(f"==> Using convergence_plane from sidecar JSON '{json_sidecar_path}': {current_zero_disparity_anchor}")
                            anchor_source = "JSON"
                        else:
                            logger.warning(f"==> Warning: Sidecar JSON '{json_sidecar_path}' found but 'convergence_plane' key is missing or invalid. Using GUI anchor: {current_zero_disparity_anchor:.2f}")
                            anchor_source = "GUI (Invalid JSON)"

                        if "max_disparity" in sidecar_data and isinstance(sidecar_data["max_disparity"], (int, float)):
                            current_max_disparity_percentage = float(sidecar_data["max_disparity"])
                            logger.debug(f"==> Using max_disparity from sidecar JSON '{json_sidecar_path}': {current_max_disparity_percentage:.1f}")
                            max_disp_source = "JSON"
                        else:
                            logger.warning(f"==> Warning: Sidecar JSON '{json_sidecar_path}' found but 'max_disparity' key is missing or invalid. Using GUI max_disp: {current_max_disparity_percentage:.1f}")
                            max_disp_source = "GUI (Invalid JSON)"

                        if "frame_overlap" in sidecar_data and isinstance(sidecar_data["frame_overlap"], (int, float)):
                            current_frame_overlap = int(sidecar_data["frame_overlap"])
                            logger.debug(f"==> Using frame_overlap from sidecar JSON '{json_sidecar_path}': {current_frame_overlap}")

                        if "input_bias" in sidecar_data and isinstance(sidecar_data["input_bias"], (int, float)):
                            current_input_bias = float(sidecar_data["input_bias"])
                            logger.debug(f"==> Using input_bias from sidecar JSON '{json_sidecar_path}': {current_input_bias:.2f}")

                    except json.JSONDecodeError:
                        logger.error(f"==> Error: Could not parse sidecar JSON '{json_sidecar_path}'. Using GUI anchor and max_disp. Anchor={current_zero_disparity_anchor:.2f}, MaxDisp={current_max_disparity_percentage:.1f}%")
                    except Exception as e:
                        logger.error(f"==> Unexpected error reading sidecar JSON '{json_sidecar_path}': {e}. Using GUI anchor and max_disp. Anchor={current_zero_disparity_anchor:.2f}, MaxDisp={current_max_disparity_percentage:.1f}%")
                else:
                    logger.debug(f"==> No sidecar JSON '{json_sidecar_path}' found for depth map. Using GUI anchor and max_disp: Anchor={current_zero_disparity_anchor:.2f}, MaxDisp={current_max_disparity_percentage:.1f}%")
            else:
                logger.error(f"==> Error: No depth map found for {video_name} in {input_depth_maps_path_setting}. Expected '{video_name}_depth.mp4' or '{video_name}_depth.npz' in folder mode. Skipping this video.")
                self.progress_queue.put(("processed", overall_task_counter))
                continue

            processing_tasks = []
            if enable_full_resolution:
                processing_tasks.append({
                    "name": "Full-Resolution",
                    "output_subdir": "hires",
                    "set_pre_res": False,
                    "target_width": -1,
                    "target_height": -1,
                    "batch_size": full_res_batch_size
                })
            if enable_low_resolution:
                processing_tasks.append({
                    "name": "Low-Resolution",
                    "output_subdir": "lowres",
                    "set_pre_res": True,
                    "target_width": low_res_width,
                    "target_height": low_res_height,
                    "batch_size": low_res_batch_size
                })

            if not processing_tasks:
                logger.debug(f"==> No processing tasks configured for {video_name}. Skipping.")
                self.progress_queue.put(("processed", overall_task_counter))
                continue

            for task_num, task in enumerate(processing_tasks):
                if self.stop_event.is_set():
                    logger.info(f"==> Stopping {task['name']} processing for {video_name} due to user request")
                    release_cuda_memory()
                    self.progress_queue.put("finished")
                    self.after(0, self.clear_processing_info)
                    return

                logger.debug(f"\n==> Starting {task['name']} pass for {video_name}")

                self.progress_queue.put(("status", f"Processing {task['name']} for {video_name}"))
                
                self.progress_queue.put(("update_info", {
                    "task_name": task['name'],
                    "convergence": f"{current_zero_disparity_anchor:.2f} ({anchor_source})",
                    "disparity": f"{current_max_disparity_percentage:.1f}% ({max_disp_source})"
                }))

                video_reader_input = None
                processed_fps = 0.0
                original_vid_h, original_vid_w = 0, 0
                current_processed_height, current_processed_width = 0, 0
                video_stream_info = None
                total_frames_input = 0

                depth_reader_input = None
                total_frames_depth = 0
                actual_depth_height, actual_depth_width = 0, 0

                try:
                    # 1. Initialize input video reader
                    video_reader_input, processed_fps, original_vid_h, original_vid_w, \
                    current_processed_height, current_processed_width, video_stream_info, \
                    total_frames_input = read_video_frames(
                        video_path, process_length,
                        set_pre_res=task["set_pre_res"], pre_res_width=task["target_width"], pre_res_height=task["target_height"]
                    )
                except Exception as e:
                    logger.error(f"==> Error initializing input video reader for {video_path} {task['name']} pass: {e}. Skipping this pass.")
                    overall_task_counter += 1
                    self.progress_queue.put(("processed", overall_task_counter))
                    continue

                self.progress_queue.put(("update_info", {
                    "resolution": f"{current_processed_width}x{current_processed_height}",
                    "frames": total_frames_input # <--- NOW USES total_frames_input
                }))

                try:
                    # 2. Initialize depth maps reader
                    # Note: target_height/width are the processed dimensions of the input video for matching
                    depth_reader_input, total_frames_depth, actual_depth_height, actual_depth_width = load_pre_rendered_depth(
                        actual_depth_map_path,
                        process_length=process_length,
                        target_height=current_processed_height,
                        target_width=current_processed_width,
                        match_resolution_to_target=settings["match_depth_res"],
                        enable_autogain=settings["enable_autogain"] # This setting will control autogain within depthSplatting
                    )
                except Exception as e:
                    logger.error(f"==> Error initializing depth map reader for {video_name} {task['name']} pass: {e}. Skipping this pass.")
                    overall_task_counter += 1
                    self.progress_queue.put(("processed", overall_task_counter))
                    continue

                # CRITICAL CHECK: Ensure input video and depth map have consistent frame counts
                if total_frames_input != total_frames_depth:
                    logger.error(f"==> Frame count mismatch for {video_name} {task['name']} pass: Input video has {total_frames_input} frames, Depth map has {total_frames_depth} frames. Skipping.")
                    overall_task_counter += 1
                    self.progress_queue.put(("processed", overall_task_counter))
                    continue


                # --- NEW: Global Depth Normalization Pre-pass ---
                global_depth_min = 0.0 # Default/placeholder
                global_depth_max = 1.0 # Default/placeholder

                if not settings["enable_autogain"]:
                    # If autogain is OFF, we need to calculate global min/max for consistent normalization
                    global_depth_min, global_depth_max = compute_global_depth_stats(
                        depth_map_reader=depth_reader_input,
                        total_frames=total_frames_depth,
                        chunk_size=task["batch_size"] # Reuse the task's batch size for chunking
                    )
                else:
                    logger.info("==> Autogain is ENABLED. Skipping global depth stats pre-pass. Normalization will be per-chunk.")

                # Ensure depth map's *actual* output resolution from VideoReader matches input video's processed resolution
                if not (actual_depth_height == current_processed_height and actual_depth_width == current_processed_width):
                    logger.warning(f"==> Warning: Depth map reader output resolution ({actual_depth_width}x{actual_depth_height}) does not match processed video resolution ({current_processed_width}x{current_processed_height}) for {task['name']} pass. This indicates an issue with `load_pre_rendered_depth`'s `width`/`height` parameters. Processing may proceed but results might be misaligned.")
                    # For now, we will proceed assuming the `VideoReader` itself handled the resize.
                    # If further `cv2.resize` is needed, it would happen *within* depthSplatting on the chunk.

                # Use the (potentially overridden) current_max_disparity_percentage
                actual_percentage_for_calculation = current_max_disparity_percentage / 20.0
                actual_max_disp_pixels = (actual_percentage_for_calculation / 100.0) * current_processed_width
                logger.debug(f"==> Max Disparity Input: {current_max_disparity_percentage:.1f}% -> Calculated Max Disparity for splatting ({task['name']}): {actual_max_disp_pixels:.2f} pixels")

                # NEW: Update disparity display with calculated pixel value
                self.progress_queue.put(("update_info", {"disparity": f"{actual_max_disp_pixels:.2f} pixels ({current_max_disparity_percentage:.1f}%)"}))

                # Create output directory and construct video path base (without final suffix)
                current_output_subdir = os.path.join(output_splatted, task["output_subdir"])
                os.makedirs(current_output_subdir, exist_ok=True)
                output_video_path_base = os.path.join(current_output_subdir, f"{video_name}.mp4")

                # 4. Perform Depth Splatting - Pass the readers and relevant metadata
                # Capture the return value (True for completed, False for skipped/stopped internally)
                completed_splatting_task = self.depthSplatting(
                    input_video_reader=video_reader_input,
                    depth_map_reader=depth_reader_input,
                    total_frames_to_process=total_frames_input,
                    processed_fps=processed_fps,
                    output_video_path_base=output_video_path_base,
                    target_output_height=current_processed_height,
                    target_output_width=current_processed_width,
                    max_disp=actual_max_disp_pixels,
                    process_length=process_length,
                    batch_size=task["batch_size"],
                    dual_output=dual_output,
                    zero_disparity_anchor_val=current_zero_disparity_anchor,
                    video_stream_info=video_stream_info,
                    frame_overlap=current_frame_overlap,
                    input_bias=current_input_bias,
                    enable_autogain=settings["enable_autogain"],
                    global_depth_min=global_depth_min,
                    global_depth_max=global_depth_max
                )

                # Check if processing was explicitly stopped by user (self.stop_event)
                # If so, clean up and exit immediately.
                if self.stop_event.is_set():
                    logger.info(f"==> Stopping {task['name']} pass for {video_name} due to user request")
                    release_cuda_memory()
                    self.progress_queue.put("finished")
                    self.after(0, self.clear_processing_info)
                    return # Exit the entire _run_batch_process thread

                # If the specific splatting task was completed successfully
                if completed_splatting_task:
                    logger.debug(f"==> Splatted {task['name']} video saved for {video_name}.")

                    # --- NEW: Explicitly delete VideoReader objects to help release file handles ---
                    # Ensure they are gone from memory before moving files
                    if video_reader_input is not None:
                        del video_reader_input
                    if depth_reader_input is not None:
                        del depth_reader_input
                    torch.cuda.empty_cache() # Clear CUDA memory as well
                    gc.collect() # Force garbage collection
                    logger.debug("Explicitly deleted VideoReader objects and forced garbage collection to release file handles.")
                    # --- END NEW ---

                    # Move files only if NOT in single file mode
                    if not is_single_file_mode:
                        # Move source video
                        if finished_source_folder is not None:
                            dest_path_src = os.path.join(finished_source_folder, os.path.basename(video_path))
                            max_retries = 5
                            retry_delay_sec = 0.5 # Wait half a second between retries
                            
                            for attempt in range(max_retries):
                                try:
                                    if os.path.exists(dest_path_src):
                                        # Use finished_source_folder here
                                        logger.warning(f"File '{os.path.basename(video_path)}' already exists in '{finished_source_folder}'. Overwriting.")
                                        os.remove(dest_path_src) # Remove existing to prevent shutil.move failure
                                    
                                    shutil.move(video_path, finished_source_folder)
                                    logger.debug(f"==> Moved processed video '{os.path.basename(video_path)}' to: {finished_source_folder}")
                                    break # If move succeeds, break out of retry loop
                                except PermissionError as e: # Catch the specific WinError 32
                                    logger.warning(f"Attempt {attempt + 1}/{max_retries}: PermissionError (file in use) when moving '{os.path.basename(video_path)}'. Retrying in {retry_delay_sec}s...")
                                    time.sleep(retry_delay_sec)
                                except Exception as e: # Catch other potential errors during move
                                    logger.error(f"==> Failed to move source video '{os.path.basename(video_path)}' to '{finished_source_folder}': {e}", exc_info=True)
                                    break # If it's another error, don't retry, just log and fail
                            else: # This 'else' block executes if the loop completes without a 'break' (i.e., all retries failed)
                                logger.error(f"==> Failed to move source video '{os.path.basename(video_path)}' after {max_retries} attempts due to PermissionError.")
                        else:
                            logger.warning(f"==> Cannot move source video '{os.path.basename(video_path)}': 'finished_source_folder' is not set.")

                        # Move depth map and its sidecar JSON
                        if actual_depth_map_path and os.path.exists(actual_depth_map_path) and finished_depth_folder is not None:
                            dest_path_depth = os.path.join(finished_depth_folder, os.path.basename(actual_depth_map_path))
                            max_retries = 5
                            retry_delay_sec = 1.0
                            
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


                            # --- Retry for Sidecar JSON (if it exists) ---
                            depth_map_dirname = os.path.dirname(actual_depth_map_path)
                            depth_map_basename_without_ext = os.path.splitext(os.path.basename(actual_depth_map_path))[0]
                            json_sidecar_path_to_move = os.path.join(depth_map_dirname, f"{depth_map_basename_without_ext}.json")
                            dest_path_json = os.path.join(finished_depth_folder, f"{depth_map_basename_without_ext}.json")

                            if os.path.exists(json_sidecar_path_to_move):
                                for attempt in range(max_retries):
                                    try:
                                        if os.path.exists(dest_path_json):
                                            logger.warning(f"Sidecar JSON '{os.path.basename(json_sidecar_path_to_move)}' already exists in '{finished_depth_folder}'. Overwriting.")
                                            os.remove(dest_path_json)
                                        shutil.move(json_sidecar_path_to_move, finished_depth_folder)
                                        logger.debug(f"==> Moved sidecar JSON '{os.path.basename(json_sidecar_path_to_move)}' to: {finished_depth_folder}")
                                        break
                                    except PermissionError as e:
                                        logger.warning(f"Attempt {attempt + 1}/{max_retries}: PermissionError (file in use) when moving JSON '{os.path.basename(json_sidecar_path_to_move)}'. Retrying in {retry_delay_sec}s...")
                                        time.sleep(retry_delay_sec)
                                    except Exception as e:
                                        logger.error(f"==> Failed to move sidecar JSON '{os.path.basename(json_sidecar_path_to_move)}' to '{finished_depth_folder}': {e}", exc_info=True)
                                        break
                                else:
                                    logger.error(f"==> Failed to move sidecar JSON '{os.path.basename(json_sidecar_path_to_move)}' after {max_retries} attempts due to PermissionError.")
                            else:
                                logger.debug(f"==> No sidecar JSON '{json_sidecar_path_to_move}' found to move.")

                        elif actual_depth_map_path and finished_depth_folder is None:
                            logger.info(f"==> Cannot move depth map '{os.path.basename(actual_depth_map_path)}': 'finished_depth_folder' is not set.")
                    else:
                        logger.debug(f"==> Single file mode for {video_name}: Skipping moving files to 'finished' folder.")
                else:
                    logger.info(f"==> Splatting task '{task['name']}' for '{video_name}' was skipped or failed. Files will NOT be moved.")

                release_cuda_memory() # Release resources after EACH pass
                overall_task_counter += 1
                self.progress_queue.put(("processed", overall_task_counter))
                logger.debug(f"==> Completed {task['name']} pass for {video_name}.")

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

    logger.info(f"\n==> Global depth stats computed: min_raw={global_min:.3f}, max_raw={global_max:.3f}")
    return float(global_min), float(global_max)

def round_to_nearest_64(value):
    """Rounds a given value up to the nearest multiple of 64, with a minimum of 64."""
    return max(64, round(value / 64) * 64)

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
        match_resolution_to_target: bool,
        enable_autogain: bool) -> Tuple[VideoReader, int, int, int]:
    """
    Initializes a VideoReader for chunked depth map reading.
    No normalization or autogain is applied here.
    Returns: (depth_reader, total_depth_frames_to_process, actual_depth_height, actual_depth_width)
    """
    logger.debug(f"==> Initializing VideoReader for depth maps from: {depth_map_path}")

    if depth_map_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Decord automatically resizes if width/height are passed
        # We need width and height to potentially resize depth to match video frames
        depth_reader = VideoReader(depth_map_path, ctx=cpu(0), width=target_width, height=target_height)
        
        # Verify the actual shape after Decord processing
        first_depth_frame_shape = depth_reader.get_batch([0]).shape
        actual_depth_height, actual_depth_width = first_depth_frame_shape[1:3]
        
        total_depth_frames_available = len(depth_reader)
        total_depth_frames_to_process = total_depth_frames_available
        if process_length != -1 and process_length < total_depth_frames_available:
            total_depth_frames_to_process = process_length

        logger.debug(f"==> DepthReader initialized. Final depth dimensions: {actual_depth_width}x{actual_depth_height}. Total frames for processing: {total_depth_frames_to_process}")

        # Note: autogain and normalization are explicitly NOT performed here now.
        # This will be handled in the depthSplatting loop if autogain is still desired,
        # or globally if global normalization is implemented later.
        
        return depth_reader, total_depth_frames_to_process, actual_depth_height, actual_depth_width
    
    elif depth_map_path.lower().endswith('.npz'):
        logger.error("NPZ support is temporarily disabled with disk chunking refactor. Please convert NPZ to MP4 depth video.")
        raise NotImplementedError("NPZ depth map loading is not yet supported with disk chunking.")
    else:
        raise ValueError(f"Unsupported depth map format: {os.path.basename(depth_map_path)}. Only MP4 are supported with disk chunking.")

if __name__ == "__main__":
    CUDA_AVAILABLE = check_cuda_availability() # Sets the global flag

    app = SplatterGUI()
    app.mainloop()
