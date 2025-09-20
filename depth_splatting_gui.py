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
from typing import Optional

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
        super().__init__(theme="black")
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
            bg_color = "#ececec"
            fg_color = "black"
            entry_bg = "white"

            self.style.theme_use("plastik")
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

        self._save_config()

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
            "enable_autogain": True,
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

                input_frames_processed = None
                processed_fps = None
                current_processed_height = None
                current_processed_width = None
                video_stream_info = None

                try:
                    input_frames_processed, processed_fps, original_vid_h, original_vid_w, current_processed_height, current_processed_width, video_stream_info = read_video_frames(
                    video_path, process_length, target_fps=-1,
                    set_pre_res=task["set_pre_res"], pre_res_width=task["target_width"], pre_res_height=task["target_height"]
                    )
                except Exception as e:
                    logger.error(f"==> Error reading input video {video_path} for {task['name']} pass: {e}. Skipping this pass.")
                    overall_task_counter += 1
                    self.progress_queue.put(("processed", overall_task_counter))
                    continue

                self.progress_queue.put(("update_info", {
                    "resolution": f"{current_processed_width}x{current_processed_height}",
                    "frames": len(input_frames_processed)
                }))

                video_depth = None
                depth_vis = None
                try:
                    video_depth, depth_vis = load_pre_rendered_depth(
                        actual_depth_map_path,
                        process_length=process_length,
                        target_height=current_processed_height,
                        target_width=current_processed_width,
                        match_resolution_to_target=settings["match_depth_res"],
                        enable_autogain=settings["enable_autogain"]
                    )
                except Exception as e:
                    logger.error(f"==> Error loading depth map for {video_name} {task['name']} pass: {e}. Skipping this pass.")
                    overall_task_counter += 1
                    self.progress_queue.put(("processed", overall_task_counter))
                    continue

                if video_depth is None or depth_vis is None:
                    logger.info(f"==> Skipping {video_name} {task['name']} pass due to depth load failure or stop request.")
                    overall_task_counter += 1
                    self.progress_queue.put(("processed", overall_task_counter))
                    continue

                if not (video_depth.shape[1] == current_processed_height and video_depth.shape[2] == current_processed_width):
                    logger.warning(f"==> Warning: Depth map resolution ({video_depth.shape[2]}x{video_depth.shape[1]}) does not match processed video resolution ({current_processed_width}x{current_processed_height}) for {task['name']} pass. Attempting final resize as safeguard.")
                    resized_video_depth = np.stack([cv2.resize(frame, (current_processed_width, current_processed_height), interpolation=cv2.INTER_LINEAR) for frame in video_depth], axis=0)
                    resized_depth_vis = np.stack([cv2.resize(frame, (current_processed_width, current_processed_height), interpolation=cv2.INTER_LINEAR) for frame in depth_vis], axis=0)
                    video_depth = resized_video_depth
                    depth_vis = resized_depth_vis

                actual_percentage_for_calculation = current_max_disparity_percentage / 20.0
                actual_max_disp_pixels = (actual_percentage_for_calculation / 100.0) * current_processed_width
                logger.debug(f"==> Max Disparity Input: {current_max_disparity_percentage:.1f}% -> Calculated Max Disparity for splatting ({task['name']}): {actual_max_disp_pixels:.2f} pixels")

                self.progress_queue.put(("update_info", {"disparity": f"{actual_max_disp_pixels:.2f} pixels ({current_max_disparity_percentage:.1f}%)"}))

                current_output_subdir = os.path.join(output_splatted, task["output_subdir"])
                os.makedirs(current_output_subdir, exist_ok=True)
                output_video_path_base = os.path.join(current_output_subdir, f"{video_name}.mp4")

                DepthSplatting(
                    input_frames_processed=input_frames_processed,
                    processed_fps=processed_fps,
                    output_video_path=output_video_path_base,
                    video_depth=video_depth,
                    depth_vis=depth_vis,
                    max_disp=actual_max_disp_pixels,
                    process_length=process_length,
                    batch_size=task["batch_size"],
                    dual_output=dual_output,
                    zero_disparity_anchor_val=current_zero_disparity_anchor,
                    video_stream_info=video_stream_info,
                    frame_overlap=current_frame_overlap,
                    input_bias=current_input_bias,
                    progress_queue=self.progress_queue, # Pass the queue
                    stop_event=self.stop_event # Pass the stop event
                )
                if self.stop_event.is_set():
                    logger.info(f"==> Stopping {task['name']} pass for {video_name} due to user request")
                    release_cuda_memory()
                    self.progress_queue.put("finished")
                    self.after(0, self.clear_processing_info)
                    return

                logger.debug(f"==> Splatted {task['name']} video saved for {video_name}.")

                release_cuda_memory()
                overall_task_counter += 1
                self.progress_queue.put(("processed", overall_task_counter))
                logger.debug(f"==> Completed {task['name']} pass for {video_name}.")

            if not self.stop_event.is_set() and not is_single_file_mode:
                if finished_source_folder is not None:
                    try:
                        shutil.move(video_path, finished_source_folder)
                        logger.debug(f"==> Moved processed video to: {finished_source_folder}")
                    except Exception as e:
                        logger.error(f"==> Failed to move video {video_path}: {e}")
                else:
                    logger.warning(f"==> Cannot move source video: 'finished_source_folder' is not set.")

                if actual_depth_map_path and os.path.exists(actual_depth_map_path) and finished_depth_folder is not None:
                    try:
                        shutil.move(actual_depth_map_path, finished_depth_folder)
                        logger.debug(f"==> Moved depth map to: {finished_depth_folder}")

                        depth_map_dirname = os.path.dirname(actual_depth_map_path)
                        depth_map_basename_without_ext = os.path.splitext(os.path.basename(actual_depth_map_path))[0]
                        json_sidecar_path_to_move = os.path.join(depth_map_dirname, f"{depth_map_basename_without_ext}.json")

                        if os.path.exists(json_sidecar_path_to_move):
                            shutil.move(json_sidecar_path_to_move, finished_depth_folder)
                            logger.debug(f"==> Moved sidecar JSON '{os.path.basename(json_sidecar_path_to_move)}' to: {finished_depth_folder}")
                        else:
                            logger.debug(f"==> No sidecar JSON '{json_sidecar_path_to_move}' found to move.")

                    except Exception as e:
                        logger.error(f"==> Failed to move depth map {actual_depth_map_path} or its sidecar: {e}")
                elif actual_depth_map_path and finished_depth_folder is None:
                    logger.info(f"==> Cannot move depth map: 'finished_depth_folder' is not set.")
            elif is_single_file_mode:
                logger.debug(f"==> Single file mode for {video_name}: Skipping moving files to 'finished' folder.")

        release_cuda_memory()
        self.after(0, self.clear_processing_info)
        self.progress_queue.put("finished")
        logger.info("==> Batch Depth Splatting Process Completed Successfully")

def round_to_nearest_64(value):
    """Rounds a given value up to the nearest multiple of 64, with a minimum of 64."""
    return max(64, round(value / 64) * 64)

def read_video_frames(video_path, process_length, target_fps,
                      set_pre_res, pre_res_width, pre_res_height, dataset="open"):
    """
    Reads video frames and determines the processing resolution.
    Resolution is determined by set_pre_res, otherwise original resolution is used.
    """
    if dataset == "open":
        logger.debug(f"==> Processing video: {video_path}")
        vid_info = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = vid_info.get_batch([0]).shape[1:3]
        logger.debug(f"==> Original video shape: {len(vid_info)} frames, {original_height}x{original_width} per frame")

        height_for_processing = original_height
        width_for_processing = original_width

        if set_pre_res and pre_res_width > 0 and pre_res_height > 0:
            # User specified pre-processing resolution
            height_for_processing = pre_res_height
            width_for_processing = pre_res_width
            logger.debug(f"==> Pre-processing video to user-specified resolution: {width_for_processing}x{height_for_processing}")
        else:
            # If set_pre_res is False, use original resolution.
            logger.debug(f"==> Using original video resolution for processing: {width_for_processing}x{height_for_processing}")

    else:
        raise NotImplementedError(f"Dataset '{dataset}' not supported.")

    # decord automatically resizes if width/height are passed to VideoReader
    vid = VideoReader(video_path, ctx=cpu(0), width=width_for_processing, height=height_for_processing)
    fps = vid.get_avg_fps() if target_fps == -1 else target_fps
    stride = max(round(vid.get_avg_fps() / fps), 1)
    frames_idx = list(range(0, len(vid), stride))
    logger.debug(f"==> Downsampled to {len(frames_idx)} frames with stride {stride}")
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]

    # Verify the actual shape after Decord processing
    first_frame_shape = vid.get_batch([0]).shape
    actual_processed_height, actual_processed_width = first_frame_shape[1:3]
    logger.debug(f"==> Final processing shape: {len(frames_idx)} frames, {actual_processed_height}x{actual_processed_width} per frame")

    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0

    video_stream_info = get_video_stream_info(video_path)

    return frames, fps, original_height, original_width, actual_processed_height, actual_processed_width, video_stream_info

def DepthSplatting(input_frames_processed, processed_fps, output_video_path, video_depth, depth_vis, max_disp, process_length, batch_size,
                   dual_output: bool, zero_disparity_anchor_val: float, video_stream_info: Optional[dict],frame_overlap: Optional[int], input_bias: Optional[float],
                   progress_queue: queue.Queue, stop_event: threading.Event): # Added queue and stop_event
    logger.debug("==> Initializing ForwardWarpStereo module")
    stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

    num_frames = len(input_frames_processed)
    height, width, _ = input_frames_processed[0].shape # Get dimensions from already processed frames
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True) # Ensure output directory exists

    # Determine output video suffix and dimensions
    if dual_output:
        suffix = "_splatted2"
    else: # Quad output
        suffix = "_splatted4"

    res_suffix = f"_{width}"

    # Construct the final output path that ffmpeg will write to
    final_output_video_path = f"{os.path.splitext(output_video_path)[0]}{res_suffix}{suffix}.mp4"

    # NEW: Use a temporary directory for PNG sequences
    temp_png_dir = os.path.join(os.path.dirname(final_output_video_path), "temp_splat_pngs_" + os.path.basename(os.path.splitext(output_video_path)[0]))
    os.makedirs(temp_png_dir, exist_ok=True)
    logger.debug(f"==> Writing temporary PNG sequence to: {temp_png_dir}")

    # Process only up to process_length if specified, for consistency
    if process_length != -1 and process_length < num_frames:
        input_frames_processed = input_frames_processed[:process_length]
        video_depth = video_depth[:process_length]
        depth_vis = depth_vis[:process_length]
        num_frames = process_length


    frame_count = 0 # To name PNGs sequentially
    
    
    # Add a single startup message
    logger.debug(f"==> Generating PNG sequence for {os.path.basename(final_output_video_path)}")
    draw_progress_bar(frame_count, num_frames, prefix=f"  Progress:") # Indent slightly
    
    for i in range(0, num_frames, batch_size):
        if stop_event.is_set():
            draw_progress_bar(frame_count, num_frames, suffix='Stopped')
            print() # Ensure a newline after stopping
            logger.info("==> Splatting stopped by user.")
            del stereo_projector
            torch.cuda.empty_cache()
            gc.collect()
            if os.path.exists(temp_png_dir):
                shutil.rmtree(temp_png_dir)
            return

        batch_frames = input_frames_processed[i:i+batch_size]
        batch_depth = video_depth[i:i+batch_size]
        batch_depth_vis = depth_vis[i:i+batch_size]

        left_video = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).float().cuda()
        disp_map = torch.from_numpy(batch_depth).unsqueeze(1).float().cuda()

        disp_map = (disp_map - zero_disparity_anchor_val) * 2.0
        disp_map = disp_map * max_disp
        with torch.no_grad():
            right_video, occlusion_mask = stereo_projector(left_video, disp_map)
        right_video = right_video.cpu().permute(0, 2, 3, 1).numpy()
        occlusion_mask = occlusion_mask.cpu().permute(0, 2, 3, 1).numpy().repeat(3, axis=-1)

        for j in range(len(batch_frames)):
            # Determine the video grid based on dual_output setting
            if dual_output:
                # Dual output: Mask | Warped
                video_grid = np.concatenate([occlusion_mask[j], right_video[j]], axis=1)
            else:
                # Quad output: Original | DepthVis (Top), Mask | Warped (Bottom)
                video_grid_top = np.concatenate([batch_frames[j], batch_depth_vis[j]], axis=1)
                video_grid_bottom = np.concatenate([occlusion_mask[j], right_video[j]], axis=1)
                video_grid = np.concatenate([video_grid_top, video_grid_bottom], axis=0)

            # Convert to 16-bit for lossless saving, scaling float32 (0-1) to uint16 (0-65535)
            # Clip to 0-1 range before scaling to prevent overflow if values slightly exceed 1.
            video_grid_uint16 = np.clip(video_grid, 0.0, 1.0) * 65535.0
            video_grid_uint16 = video_grid_uint16.astype(np.uint16)

            # Convert to BGR for OpenCV
            video_grid_bgr = cv2.cvtColor(video_grid_uint16, cv2.COLOR_RGB2BGR)

            png_filename = os.path.join(temp_png_dir, f"{frame_count:05d}.png")
            cv2.imwrite(png_filename, video_grid_bgr)

            frame_count += 1

        del left_video, disp_map, right_video, occlusion_mask
        torch.cuda.empty_cache()
        gc.collect()

        draw_progress_bar(frame_count, num_frames, prefix=f"  Progress:") # [REQUIRED]
        
    logger.debug(f"==> Temporary PNG sequence generation completed ({frame_count} frames).")
    
    del stereo_projector
    torch.cuda.empty_cache()
    gc.collect()

    # --- FFmpeg encoding from PNG sequence to final MP4 ---
    logger.debug(f"==> Encoding final video from PNG sequence using ffmpeg for '{os.path.basename(final_output_video_path)}'.")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y", # Overwrite output files without asking
        "-framerate", str(processed_fps), # Input framerate for the PNG sequence
        "-i", os.path.join(temp_png_dir, "%05d.png"), # Input PNG sequence pattern
    ]

    # --- Extract original video properties if available ---
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

    # --- Determine Output Codec, Bit-Depth, and Quality ---
    # Set sensible defaults first
    output_codec = "libx264" # Default to H.264
    output_pix_fmt = "yuv420p" # Default to 8-bit
    output_crf = "23" # Default CRF for H.264 (medium quality)
    output_profile = "main" # Default H.264 profile
    x265_params = [] # For specific x265 parameters

    
    # NEW: NVENC specific parameters
    nvenc_preset = "medium" # Default NVENC preset (e.g., fast, medium, slow, quality, etc.)
    # Note: NVENC uses CQP/VBR/CBR not CRF. We'll map CRF to a CQ value or just use a fixed quality.
    nvenc_cq = "23" # Constant Quality value for NVENC (lower is better quality)

    is_hdr_source = False
    if video_stream_info and video_stream_info.get("color_primaries") == "bt2020" and \
       video_stream_info.get("transfer_characteristics") == "smpte2084":
        is_hdr_source = True
        logger.debug("==> Source detected as HDR.")

    # Main logic for choosing output format based on detected info (always attempt smart matching)
    if video_stream_info: # Only proceed with smart matching if info is detected
        logger.debug("==> Source video stream info detected. Attempting to match source characteristics and optimize quality.")

        if is_hdr_source:
            logger.debug("==> Detected HDR source. Targeting HEVC (x265) 10-bit HDR output.")
            output_codec = "libx265" # Default to CPU x265
            if CUDA_AVAILABLE:
                output_codec = "hevc_nvenc" # Use NVENC if available
                logger.debug("    (Using hevc_nvenc for hardware acceleration)")
            
            output_pix_fmt = "yuv420p10le"
            output_crf = "28" # This CRF value is for CPU x265, will use nvenc_cq for NVENC
            output_profile = "main10"
            # Add HDR mastering display and light level metadata if available
            if video_stream_info.get("mastering_display_metadata"):
                md_meta = video_stream_info["mastering_display_metadata"]
                x265_params.append(f"mastering-display={md_meta}") # These are for x265, not nvenc directly.
                logger.debug(f"==> Adding mastering display metadata: {md_meta}")
            if video_stream_info.get("max_content_light_level"):
                max_cll_meta = video_stream_info["max_content_light_level"]
                x265_params.append(f"max-cll={max_cll_meta}") # These are for x265, not nvenc directly.
                logger.debug(f"==> Adding max content light level: {max_cll_meta}")

        elif original_codec_name == "hevc" and is_original_10bit_or_higher:
            logger.debug("==> Detected 10-bit HEVC (x265) SDR source. Targeting HEVC (x265) 10-bit SDR output.")
            output_codec = "libx265" # Default to CPU x265
            if CUDA_AVAILABLE:
                output_codec = "hevc_nvenc" # Use NVENC if available
                logger.debug("    (Using hevc_nvenc for hardware acceleration)")
            
            output_pix_fmt = "yuv420p10le"
            output_crf = "24" # For CPU x265
            output_profile = "main10"

        else: # If not HDR/HEVC 10-bit, default to H.264 high quality
            logger.debug("==> No specific HEVC/HDR source. Targeting H.264 (x264) 8-bit SDR high quality.")
            output_codec = "libx264" # Default to CPU x264
            if CUDA_AVAILABLE:
                output_codec = "h264_nvenc" # Use NVENC if available
                logger.debug("    (Using h264_nvenc for hardware acceleration)")
            
            output_pix_fmt = "yuv420p"
            output_crf = "18" # For CPU x264
            output_profile = "main"

    else: # video_stream_info is None (fallback behavior if no info detected)
        logger.debug("==> No source video stream info detected. Falling back to default H.264 (x264) 8-bit SDR (medium quality).")
        # Defaults already set at the top of this block (libx264, yuv420p, CRF 23, profile main)
        if CUDA_AVAILABLE:
            output_codec = "h264_nvenc" # Use NVENC if CUDA available for default H.264
            logger.debug("    (Using h264_nvenc for hardware acceleration for default output)")


    ffmpeg_cmd.extend(["-c:v", output_codec])
    # NEW: Add NVENC specific parameters
    if "nvenc" in output_codec: # Check if an NVENC codec is chosen
        ffmpeg_cmd.extend(["-preset", nvenc_preset])
        ffmpeg_cmd.extend(["-cq", nvenc_cq]) # Constant Quality for NVENC
        # Remove CRF if NVENC is used, as it's not applicable
        if "-crf" in ffmpeg_cmd:
            crf_index = ffmpeg_cmd.index("-crf")
            del ffmpeg_cmd[crf_index:crf_index+2] # Delete -crf and its value
    else: # Only add CRF if not NVENC
        ffmpeg_cmd.extend(["-crf", output_crf])
    
    ffmpeg_cmd.extend(["-pix_fmt", output_pix_fmt])
    if output_profile:
        # NVENC profiles might differ slightly, but main/main10 generally work.
        ffmpeg_cmd.extend(["-profile:v", output_profile])

    # NVENC doesn't use -x265-params directly for HDR.
    # HDR metadata for NVENC is usually passed via -mastering-display and -max-cll
    # directly as FFmpeg main flags, which we already handle outside -x265-params.
    # The -x265-params block is only relevant if using libx265.
    if output_codec == "libx265" and x265_params:
        ffmpeg_cmd.extend(["-x265-params", ":".join(x265_params)])


    # --- Add general color space flags (primaries, transfer, space) ---
    # This block remains, but now only applies if retain_color_space is true AND info exists
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
        ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True, encoding='utf-8', timeout=3600*24) # 24 hour timeout
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

    # NEW: Create a sidecar JSON for the splatted output
    output_sidecar_data = {}
    
    # Always include convergence_plane and max_disparity if they were used for splatting
    # We'll use the values that were actually passed into this function.
    output_sidecar_data["convergence_plane"] = zero_disparity_anchor_val
    output_sidecar_data["max_disparity"] = (max_disp / width) * 100.0 # Convert back to percentage relative to width
    
    # Add frame_overlap and input_bias if they were provided
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

    # Clean up temporary PNG directory
    if os.path.exists(temp_png_dir):
        try:
            shutil.rmtree(temp_png_dir)
            logger.debug(f"==> Cleaned up temporary PNG directory: {temp_png_dir}")
        except Exception as e:
            logger.error(f"==> Error cleaning up temporary PNG directory {temp_png_dir}: {e}")

    logger.debug(f"==> Final output video written to: {final_output_video_path}")

def load_pre_rendered_depth(depth_map_path, process_length=-1, target_height=-1, target_width=-1, match_resolution_to_target=True, enable_autogain=True):
    """
    Loads pre-rendered depth maps from MP4 or NPZ.
    If match_resolution_to_target is True, it resizes the depth maps to the target_height/width for compatibility.
    Includes an option to enable/disable min-max normalization (autogain).
    If autogain is disabled, MP4 depth maps are scaled to 0-1 based on their detected bit depth (8-bit -> /255, 10-bit -> /1023).
    For NPZ with autogain disabled, it assumes the data is already in the desired absolute range (e.g., 0-1).
    """
    logger.debug(f"==> Loading pre-rendered depth maps from: {depth_map_path}")

    video_depth_working_range = None # This will hold the float32 array before final normalization/scaling

    if depth_map_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        vid = VideoReader(depth_map_path, ctx=cpu(0))
        # Read raw frames as float32, without any initial fixed scaling like /255.0
        # This allows us to inspect the actual raw pixel values for bit depth detection.
        raw_frames_data = vid[:].asnumpy().astype("float32")

        if raw_frames_data.shape[-1] == 3:
            logger.debug("==> Converting RGB depth frames to grayscale")
            raw_frames_data = raw_frames_data.mean(axis=-1)
        else:
            raw_frames_data = raw_frames_data.squeeze(-1)

        if enable_autogain:
            # If autogain is enabled, we work with the raw float data.
            # The min-max normalization below will then scale this observed range to 0-1.
            video_depth_working_range = raw_frames_data
        else:
            # Autogain disabled: Scale to 0-1 absolutely based on detected bit depth.
            # This logic assumes common bit-depth ranges for MP4 depth maps.
            max_raw_value_in_content = np.max(raw_frames_data)
            if max_raw_value_in_content > 255 and max_raw_value_in_content <= 1023: # Heuristic for 10-bit range
                logger.debug(f"==> Autogain disabled. Detected potential 10-bit depth map (max raw value: {max_raw_value_in_content:.0f}). Scaling to absolute 0-1 by dividing by 1023.0.")
                video_depth_working_range = raw_frames_data / 1023.0
            else: # Assume 8-bit or standard 0-255 range
                logger.debug(f"==> Autogain disabled. Detected potential 8-bit depth map (max raw value: {max_raw_value_in_content:.0f}). Scaling to absolute 0-1 by dividing by 255.0.")
                video_depth_working_range = raw_frames_data / 255.0

    elif depth_map_path.lower().endswith('.npz'):
        loaded_data = np.load(depth_map_path)
        if 'depth' in loaded_data:
            video_depth_working_range = loaded_data['depth'].astype("float32")
            if not enable_autogain:
                # For NPZ, if autogain is off, we *assume* it's already 0-1 or has an absolute meaning.
                # If NPZ data is NOT already 0-1 and needs a fixed scaling (e.g., from 0-1000 to 0-1),
                # the user would need to preprocess the NPZ or a new GUI setting for NPZ absolute scaling factor would be required.
                logger.debug("==> Autogain disabled for NPZ. Assuming depth data is already in desired absolute range (e.g., 0-1).")
        else:
            raise ValueError("NPZ file does not contain a 'depth' array.")
    else:
        raise ValueError(f"Unsupported depth map format: {os.path.basename(depth_map_path)}. Only MP4/NPZ are supported.")

    if process_length != -1 and process_length < len(video_depth_working_range):
        video_depth_working_range = video_depth_working_range[:process_length]

    if enable_autogain:
        # Perform Autogain (Min-Max Scaling) on the raw working range
        video_depth_min = video_depth_working_range.min()
        video_depth_max = video_depth_working_range.max()
        if video_depth_max - video_depth_min > 1e-5:
            video_depth_normalized = (video_depth_working_range - video_depth_min) / (video_depth_max - video_depth_min)
            logger.debug(f"==> Depth maps autogained (min-max scaled) from observed range [{video_depth_min:.3f}, {video_depth_max:.3f}] to [0, 1].")
        else:
            video_depth_normalized = np.zeros_like(video_depth_working_range)
            logger.debug("==> Depth map range too small, setting to zeros after autogain.")
    else:
        # Autogain is disabled; video_depth_working_range already contains absolute 0-1 for MP4, or assumed for NPZ.
        video_depth_normalized = video_depth_working_range
        logger.debug("==> Autogain (min-max scaling) disabled. Depth maps are used with their absolute scaling.")


    # Resize logic remains the same as before
    if match_resolution_to_target and target_height > 0 and target_width > 0:
        logger.debug(f"==> Resizing loaded depth maps to target resolution: {target_width}x{target_height}")
        resized_depths = []
        resized_viss = []
        for i in range(video_depth_normalized.shape[0]):
            depth_frame = video_depth_normalized[i]
            # OpenCV expects (width, height)
            resized_depth_frame = cv2.resize(depth_frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized_depths.append(resized_depth_frame)

            # Apply colormap to the resized depth for visualization.
            # Clip to 0-1 range for reliable visualization, as the colormap expects this.
            vis_frame = cv2.applyColorMap((np.clip(resized_depth_frame, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            resized_viss.append(vis_frame.astype("float32") / 255.0)

        video_depth = np.stack(resized_depths, axis=0)
        depth_vis = np.stack(resized_viss, axis=0)
    else:
        logger.debug(f"==> Not resizing loaded depth maps (match_resolution_to_target is False or target dimensions invalid). Current resolution: {video_depth_normalized.shape[2]}x{video_depth_normalized.shape[1]}")
        video_depth = video_depth_normalized
        # Visualization: Ensure 0-1 range for colormap if autogain is off and values might naturally exceed 1.
        depth_vis = np.stack([cv2.applyColorMap((np.clip(frame, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO).astype("float32") / 255.0 for frame in video_depth_normalized], axis=0)

    logger.debug("==> Depth maps and visualizations loaded successfully")
    return video_depth, depth_vis

if __name__ == "__main__":
    CUDA_AVAILABLE = check_cuda_availability() # Sets the global flag

    app = SplatterGUI()
    app.mainloop()
