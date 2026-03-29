import os
import glob
import json
import shutil
import threading
import tkinter as tk  # Used for PanedWindow
from tkinter import filedialog, messagebox, ttk
from ttkthemes import ThemedTk
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from decord import VideoReader, cpu
import logging
import time
import queue
import re
from core.common.video_io import start_ffmpeg_pipe_process
from core.common.cli_utils import set_logger_level, draw_progress_bar

logger = logging.getLogger(__name__)

from core.common.gpu_utils import release_cuda_memory
from core.common.image_processing import apply_dubois_anaglyph, apply_optimized_anaglyph
from core.common.video_io import get_video_stream_info
from core.common.sidecar_manager import (
    SidecarConfigManager,
    find_video_by_core_name,
    find_sidecar_file,
    read_clip_sidecar,
)
from core.common.image_processing import (
    apply_mask_dilation,
    apply_gaussian_blur,
    apply_shadow_blur,
    apply_dubois_anaglyph_torch,
    apply_optimized_anaglyph_torch,
    apply_color_transfer,
    apply_dubois_anaglyph,
    apply_optimized_anaglyph,
    apply_borders_to_frames,
)
from core.ui.widgets import Tooltip, create_single_slider_with_label_updater
from core.ui.video_previewer import VideoPreviewer
from core.ui.encoding_settings import EncodingSettingsDialog
from core.common.file_organizer import move_files_to_finished, restore_finished_files as _restore_finished_files
from core.ui.theme_manager import ThemeManager
from core.ui.dnd_support import init_dnd, register_dnd_entries, configure_dnd_styles

GUI_VERSION = "26-03-08.1"


class MergingGUI(ThemedTk):
    # --- Centralized Default Settings ---
    APP_DEFAULTS = {
        "inpainted_folder": "./completed_output",
        "original_folder": "./input_source_clips",
        "mask_folder": "./output_splatted/hires",
        "output_folder": "./final_videos",
        "mask_binarize_threshold": 0.3,
        "mask_dilate_kernel_size": 3.0,
        "mask_blur_kernel_size": 5.0,
        "shadow_shift": 5.0,
        "shadow_decay_gamma": 1.3,
        "shadow_start_opacity": 0.87,
        "shadow_opacity_decay": 0.08,
        "shadow_min_opacity": 0.14,
        "use_gpu": False,
        "output_format": "Full SBS (Left-Right)",
        "pad_to_16_9": False,
        "enable_color_transfer": True,
        "batch_chunk_size": "20",
        "preview_size": "100%",
        "MESH_WARP_DISPARITY": "25.0",
        "MESH_WARP_CONVERGENCE": "0.5",
        "MESH_WARP_VIEW_BIAS": "0.0",
        "MESH_WARP_DOLLY_ZOOM": "0.0",
        "MESH_WARP_EXTRUSION_SCALE": "0.1",
        "MESH_WARP_DENSITY_X": "512",
        "MESH_WARP_DENSITY_Y": "512",
    }

    def __init__(self):
        super().__init__(theme="clam")
        self.title(f"Stereocrafter Merging GUI {GUI_VERSION}")
        self.app_config = self._load_config()
        self.help_data = self._load_help_texts()

        # --- Drag-and-drop support (requires tkinterdnd2) ---
        self._dnd_enabled = init_dnd(self)

        # --- Sidecar Config Manager ---
        self.sidecar_manager = SidecarConfigManager()

        # --- Window Geometry ---
        self.window_x = self.app_config.get("window_x", None)
        self.window_y = self.app_config.get("window_y", None)
        self.window_width = self.app_config.get("window_width", 700)  # A reasonable default
        self.window_height = self.app_config.get("window_height", 800)  # A reasonable default

        # --- Core App State ---
        self.stop_event = threading.Event()
        self.is_processing = False
        self.cleanup_queue = queue.Queue()

        self.current_filename_var = tk.StringVar(value="No video loaded")
        self.current_resolution_var = tk.StringVar(value="N/A")
        self.current_flip_status_var = tk.StringVar(value="N/A")
        self.border_info_var = tk.StringVar(value="Borders: N/A")
        self.status_label_var = tk.StringVar(value="Ready")
        self._is_startup = True  # Flag to prevent resizing during initialization
        self.preview_original_left_tensor = None
        self.preview_blended_right_tensor = None
        # --- GUI Variables ---
        self.pil_image_for_preview = None
        self.inpainted_folder_var = tk.StringVar(
            value=self.app_config.get("inpainted_folder", self.APP_DEFAULTS["inpainted_folder"])
        )
        self.inpainted_folder_var.trace_add("write", self._on_folder_changed)
        self.original_folder_var = tk.StringVar(
            value=self.app_config.get("original_folder", self.APP_DEFAULTS["original_folder"])
        )
        self.original_folder_var.trace_add("write", self._on_folder_changed)
        self.mask_folder_var = tk.StringVar(value=self.app_config.get("mask_folder", self.APP_DEFAULTS["mask_folder"]))
        self.mask_folder_var.trace_add("write", self._on_folder_changed)
        self.output_folder_var = tk.StringVar(
            value=self.app_config.get("output_folder", self.APP_DEFAULTS["output_folder"])
        )

        # --- Mask Processing Parameters ---
        self.mask_binarize_threshold_var = tk.DoubleVar(
            value=float(self.app_config.get("mask_binarize_threshold", self.APP_DEFAULTS["mask_binarize_threshold"]))
        )
        self.mask_dilate_kernel_size_var = tk.DoubleVar(
            value=float(self.app_config.get("mask_dilate_kernel_size", self.APP_DEFAULTS["mask_dilate_kernel_size"]))
        )
        self.mask_blur_kernel_size_var = tk.DoubleVar(
            value=float(self.app_config.get("mask_blur_kernel_size", self.APP_DEFAULTS["mask_blur_kernel_size"]))
        )
        self.shadow_shift_var = tk.DoubleVar(
            value=float(self.app_config.get("shadow_shift", self.APP_DEFAULTS["shadow_shift"]))
        )
        self.shadow_decay_gamma_var = tk.DoubleVar(
            value=float(self.app_config.get("shadow_decay_gamma", self.APP_DEFAULTS["shadow_decay_gamma"]))
        )
        self.shadow_start_opacity_var = tk.DoubleVar(
            value=float(self.app_config.get("shadow_start_opacity", self.APP_DEFAULTS["shadow_start_opacity"]))
        )
        self.shadow_opacity_decay_var = tk.DoubleVar(
            value=float(self.app_config.get("shadow_opacity_decay", self.APP_DEFAULTS["shadow_opacity_decay"]))
        )
        self.shadow_min_opacity_var = tk.DoubleVar(
            value=float(self.app_config.get("shadow_min_opacity", self.APP_DEFAULTS["shadow_min_opacity"]))
        )

        self.use_gpu_var = tk.BooleanVar(value=self.app_config.get("use_gpu", self.APP_DEFAULTS["use_gpu"]))
        self.output_format_var = tk.StringVar(
            value=self.app_config.get("output_format", self.APP_DEFAULTS["output_format"])
        )
        self.pad_to_16_9_var = tk.BooleanVar(value=self.app_config.get("pad_to_16_9", self.APP_DEFAULTS["pad_to_16_9"]))
        self.enable_color_transfer_var = tk.BooleanVar(
            value=self.app_config.get("enable_color_transfer", self.APP_DEFAULTS["enable_color_transfer"])
        )
        self.debug_logging_var = tk.BooleanVar(value=self.app_config.get("debug_logging_enabled", False))
        self.dark_mode_var = tk.BooleanVar(value=self.app_config.get("dark_mode_enabled", False))
        self.theme_manager = ThemeManager(dark_mode_var=self.dark_mode_var, config=self.app_config)
        self.batch_chunk_size_var = tk.StringVar(
            value=str(self.app_config.get("batch_chunk_size", self.APP_DEFAULTS["batch_chunk_size"]))
        )
        self.preview_source_var = tk.StringVar(value=self.app_config.get("preview_source", "Blended Image"))
        self.preview_size_var = tk.StringVar(value=str(self.app_config.get("preview_size", "100%")))

        # --- Encoding Settings ---
        self.codec_var = tk.StringVar(value=self.app_config.get("codec", "H.265"))
        self.encoder_var = tk.StringVar(value=self.app_config.get("encoding_encoder", "Auto"))
        self.encoding_quality_var = tk.StringVar(value=self.app_config.get("encoding_quality", "Medium"))
        self.encoding_tune_var = tk.StringVar(value=self.app_config.get("encoding_tune", "None"))
        self.output_crf_var = tk.StringVar(value=str(self.app_config.get("output_crf", 23)))
        self.nvenc_lookahead_enabled_var = tk.BooleanVar(value=self.app_config.get("nvenc_lookahead_enabled", False))
        self.nvenc_lookahead_var = tk.IntVar(value=self.app_config.get("nvenc_lookahead", 16))
        self.nvenc_spatial_aq_var = tk.BooleanVar(value=self.app_config.get("nvenc_spatial_aq", False))
        self.nvenc_temporal_aq_var = tk.BooleanVar(value=self.app_config.get("nvenc_temporal_aq", False))
        self.nvenc_aq_strength_var = tk.IntVar(value=self.app_config.get("nvenc_aq_strength", 8))
        self.color_tags_var = tk.StringVar(value=self.app_config.get("color_tags", "Auto"))
        self.dnxhr_fullres_split_var = tk.BooleanVar(value=self.app_config.get("dnxhr_fullres_split", False))
        self.dnxhr_profile_var = tk.StringVar(value=self.app_config.get("dnxhr_profile", "HQX (10-bit 4:2:2)"))

        # --- GUI Status Variables ---
        self.slider_label_updaters = []
        # --- END FIX ---
        self.progress_var = tk.DoubleVar(value=0)
        self.widgets_to_disable = []

        self.create_widgets()

        # Define a custom style for the loading button
        self.style = ttk.Style(self)
        self.style.configure("Loading.TButton", foreground="red")

        self._apply_theme()
        self._configure_logging()  # Set initial logging level
        self.after(0, lambda: setattr(self, "_is_startup", False))  # Set startup flag to false after GUI is built
        self.after(0, self._set_saved_geometry)  # Restore window position
        self.protocol("WM_DELETE_WINDOW", self.exit_application)

        # Call all the label updaters to set the initial text from the loaded config
        for updater in self.slider_label_updaters:
            updater()
        self.update_status_label("Ready.")

        # --- FIX: Initialize the previewer AFTER the main GUI is fully built ---
        # This ensures the previewer gets the correct initial slider values.
        # No longer needed, previewer will call get_current_settings() itself.
        pass

    def _set_saved_geometry(self):
        """
        Applies the saved window width, height, and position.
        """
        logger.debug("--- Setting Saved Geometry (Startup) ---")
        self.update_idletasks()

        # 1. Use the saved/default width and height, with fallbacks
        current_width = self.window_width
        current_height = self.window_height
        logger.debug(f"  - Using saved/default width: {current_width}, height: {current_height}")

        if current_width < 500:  # Minimum sensible width
            current_width = 700
            logger.debug(f"  - Width was < 500, using fallback: {current_width}")
        if current_height < 400:  # Minimum sensible height
            current_height = 800
            logger.debug(f"  - Height was < 400, using fallback: {current_height}")

        # 2. Construct the geometry string
        geometry_string = f"{current_width}x{current_height}"
        if self.window_x is not None and self.window_y is not None:
            geometry_string += f"+{self.window_x}+{self.window_y}"
            logger.debug(f"  - Using saved position: +{self.window_x}+{self.window_y}")

        # 3. Apply the geometry
        self.geometry(geometry_string)
        logger.debug(f"  - Applied geometry string: '{geometry_string}'")
        logger.debug("--- End Setting Saved Geometry ---")

    def create_menubar(self):
        """Creates the main menu bar for the application."""
        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)

        # --- File Menu ---
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Load Settings...", command=self.load_settings_dialog)
        self.file_menu.add_command(label="Save Settings...", command=self.save_settings_dialog)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Save Preview Frame...", command=lambda: self.previewer.save_preview_frame())
        self.file_menu.add_command(
            label="Save Preview as SBS...", command=self._save_preview_sbs_frame
        )  # Keep this one here as it needs access to both eyes
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Reset to Default", command=self.reset_to_defaults)
        self.file_menu.add_command(label="Restore Finished Files", command=self.restore_finished_files)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.exit_application)

        # --- Options Menu ---
        self.options_menu = tk.Menu(self.menubar, tearoff=0)
        self.options_menu.add_command(label="Encoding Settings...", command=self._show_encoding_settings)
        self.options_menu.add_separator()
        self.options_menu.add_checkbutton(label="Dark Mode", variable=self.dark_mode_var, command=self._apply_theme)
        self.menubar.add_cascade(label="Options", menu=self.options_menu)

        # --- Help Menu ---
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_checkbutton(
            label="Enable Debug Logging", variable=self.debug_logging_var, command=self._toggle_debug_logging
        )
        self.help_menu.add_separator()
        self.help_menu.add_command(label="User Guide", command=self.show_user_guide)
        self.help_menu.add_command(label="About", command=self.show_about_dialog)

    def _create_hover_tooltip(self, widget, help_key):
        """Creates a mouse-over tooltip for the given widget."""
        if help_key in self.help_data:
            Tooltip(widget, self.help_data[help_key])

    def _apply_theme(self):
        """Applies the selected theme (dark or light) to the GUI using ThemeManager."""
        if not hasattr(self, "theme_manager"):
            return

        # 1. Apply styles to ttk widgets and root window
        self.theme_manager.apply_theme_to_style(self.style, root_window=self)

        # 2. Apply theme to non-ttk widgets (Menu, Canvas, Labels)
        colors = self.theme_manager.get_colors()

        # Menus
        if hasattr(self, "menubar"):
            menus = []
            if hasattr(self, "file_menu"):
                menus.append(self.file_menu)
            if hasattr(self, "help_menu"):
                menus.append(self.help_menu)
            if hasattr(self, "options_menu"):
                menus.append(self.options_menu)
            self.theme_manager.apply_theme_to_menus(menus=menus, menubar=self.menubar)

        # Previewer Canvas
        if hasattr(self, "previewer") and hasattr(self.previewer, "canvas_window"):
            self.theme_manager.apply_theme_to_canvas(self.previewer.canvas_window.preview_canvas)

        # 3. Apply compact/custom widget styles via ThemeManager
        self.theme_manager.configure_compact_styles(self.style)
        self.theme_manager.configure_progressbar_style(self.style)

        # DnD drop-target highlight style (theme-aware)
        configure_dnd_styles(self.style, self.dark_mode_var.get(), self._dnd_enabled)

        # --- FIX: Re-apply the custom loading button style after the theme changes ---
        # This ensures the red text color is not overridden by the theme's default button style.
        self.style.configure("Loading.TButton", foreground="red")

        # Adjust window height for new theme if not starting up
        if not self._is_startup:
            self._adjust_window_height_for_content()

    def show_about_dialog(self):
        """Displays an 'About' dialog for the application."""
        about_text = (
            f"Stereocrafter Merging GUI\n"
            f"Version: {GUI_VERSION}\n\n"
            "This tool blends inpainted right-eye videos with their corresponding "
            "high-resolution source files to create final stereoscopic videos.\n\n"
            "It provides interactive controls for mask processing and color matching."
        )
        messagebox.showinfo("About Merging GUI", about_text)

    def show_user_guide(self):
        """Reads and displays the user guide from a markdown file in a new window."""
        guide_path = os.path.join("assets", "merger_gui_guide.md")
        try:
            with open(guide_path, "r", encoding="utf-8") as f:
                guide_content = f.read()
        except FileNotFoundError:
            messagebox.showerror(
                "File Not Found", f"The user guide file could not be found at:\n{os.path.abspath(guide_path)}"
            )
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
        guide_window.title("Merging GUI - User Guide")
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
            pady=5,
            font=("Segoe UI", 9),
            bg=bg_color,
            fg=fg_color,
            insertbackground=fg_color,
        )
        text_widget.insert(tk.END, guide_content)
        text_widget.config(state=tk.DISABLED)  # Make it read-only

        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget["yscrollcommand"] = scrollbar.set

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(side=tk.LEFT, expand=True, fill="both")

        button_frame = ttk.Frame(guide_window, padding=(0, 0, 0, 10))
        button_frame.pack()
        ok_button = ttk.Button(button_frame, text="Close", command=guide_window.destroy)
        ok_button.pack(pady=10)

    def reset_to_defaults(self):
        """Resets all GUI parameters to their default values using the _apply_settings method."""
        if not messagebox.askyesno(
            "Reset Settings", "Are you sure you want to reset all settings to their default values?"
        ):
            return  # User cancelled

        self._apply_settings(self.APP_DEFAULTS)
        self.save_config()
        # messagebox.showinfo("Settings Reset", "All settings have been reset to their default values.")
        logger.info("GUI settings reset to defaults.")

    def _apply_settings(self, settings_dict: dict):
        """
        A centralized function to apply a dictionary of settings to the GUI's tk.Variables.
        This is used by both Load Settings and Reset to Defaults.
        """
        logger.debug(f"Applying settings dictionary:\n{json.dumps(settings_dict, indent=2)}")
        for key, value in settings_dict.items():
            var_name = key + "_var"
            if hasattr(self, var_name):
                tk_var = getattr(self, var_name)
                try:
                    tk_var.set(value)
                except (ValueError, tk.TclError) as e:
                    logger.error(f"Could not apply setting for '{key}' with value '{value}': {e}")

        # After setting all variables, manually update the slider labels to match.
        for updater in self.slider_label_updaters:
            updater()
        logger.info("Applied settings to GUI and updated labels.")

    def _configure_logging(self):
        """Sets the logging level based on the debug_logging_var."""
        if self.debug_logging_var.get():
            level = logging.DEBUG
        else:
            level = logging.INFO

        set_logger_level(logger, level)
        logging.getLogger().setLevel(level)
        logger.info(f"Logging level set to {logging.getLevelName(level)}.")

    def _adjust_window_height_for_content(self):
        """Adjusts the window height to fit the current content, preserving user-set width."""
        if self._is_startup:  # Don't adjust during initial setup
            return

        current_actual_width = self.winfo_width()
        if current_actual_width <= 1:  # Fallback for very first call
            current_actual_width = self.window_width

        # --- NEW: More accurate height calculation ---
        # Sum heights of all packed children (the previewer frame now only contains controls).
        base_height = 0
        for widget in self.winfo_children():
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

        # Add a small buffer for padding/borders
        padding = 10

        # The new total height is the base UI height + padding.
        # The preview image is now in a separate Toplevel (canvas_window).
        new_height = base_height + padding
        # --- END NEW ---

        self.geometry(f"{current_actual_width}x{new_height}")
        logger.debug(f"Content resize applied geometry: {current_actual_width}x{new_height}")

        # Update stored width and height for the next time save_config is called.
        self.window_width = current_actual_width

    def _toggle_debug_logging(self):
        """Callback for the debug logging checkbox."""
        self._configure_logging()
        self.save_config()

    def _load_help_texts(self):
        """Loads help texts from the dedicated JSON file."""
        try:
            with open(os.path.join("dependency", "merge_help.json"), "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def create_widgets(self):
        self.create_menubar()
        # The main window will now be a simple vertical layout.
        # We will pack frames directly into `self`.

        # --- FOLDER FRAME ---
        folder_frame = ttk.LabelFrame(self, text="Folders", padding=10)
        folder_frame.pack(fill="x", padx=10, pady=5)
        folder_frame.grid_columnconfigure(1, weight=1)

        # Inpainted Video Folder
        ttk.Label(folder_frame, text="Inpainted Video Folder:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        entry_inpaint = ttk.Entry(folder_frame, textvariable=self.inpainted_folder_var)
        entry_inpaint.grid(row=0, column=1, padx=5, sticky="ew")
        self._create_hover_tooltip(entry_inpaint, "inpainted_folder")
        btn_inpaint = ttk.Button(
            folder_frame, text="Browse", command=lambda: self._browse_folder(self.inpainted_folder_var)
        )
        btn_inpaint.grid(row=0, column=2, padx=5)
        self.widgets_to_disable.append(entry_inpaint)
        self.widgets_to_disable.append(btn_inpaint)

        # Original Video Folder (for Left Eye)
        ttk.Label(folder_frame, text="Original Video Folder:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        entry_orig = ttk.Entry(folder_frame, textvariable=self.original_folder_var)
        entry_orig.grid(row=1, column=1, padx=5, sticky="ew")
        self._create_hover_tooltip(entry_orig, "original_folder")
        btn_orig = ttk.Button(
            folder_frame, text="Browse", command=lambda: self._browse_folder(self.original_folder_var)
        )
        btn_orig.grid(row=1, column=2, padx=5)
        self.widgets_to_disable.append(entry_orig)
        self.widgets_to_disable.append(btn_orig)

        # Mask Folder
        ttk.Label(folder_frame, text="Mask Folder:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        entry_mask = ttk.Entry(folder_frame, textvariable=self.mask_folder_var)
        entry_mask.grid(row=2, column=1, padx=5, sticky="ew")
        self._create_hover_tooltip(entry_mask, "mask_folder")
        btn_mask = ttk.Button(folder_frame, text="Browse", command=lambda: self._browse_folder(self.mask_folder_var))
        btn_mask.grid(row=2, column=2, padx=5)
        self.widgets_to_disable.append(entry_mask)
        self.widgets_to_disable.append(btn_mask)

        # Output Folder
        ttk.Label(folder_frame, text="Output Folder:").grid(row=3, column=0, sticky="e", padx=5, pady=2)
        entry_out = ttk.Entry(folder_frame, textvariable=self.output_folder_var)
        entry_out.grid(row=3, column=1, padx=5, sticky="ew")
        self._create_hover_tooltip(entry_out, "output_folder")
        btn_out = ttk.Button(folder_frame, text="Browse", command=lambda: self._browse_folder(self.output_folder_var))
        btn_out.grid(row=3, column=2, padx=5)
        self.widgets_to_disable.append(entry_out)
        self.widgets_to_disable.append(btn_out)

        # Register Drag & Drop for folder entries
        register_dnd_entries(
            [
                (entry_inpaint, self.inpainted_folder_var, True, None),
                (entry_orig, self.original_folder_var, True, None),
                (entry_mask, self.mask_folder_var, True, None),
                (entry_out, self.output_folder_var, True, None),
            ],
            dnd_enabled=self._dnd_enabled,
        )

        # --- PREVIEW FRAME (using the new module) ---
        # Moved back to its original position after the folder frame.
        self.previewer = VideoPreviewer(
            self,
            processing_callback=self._preview_processing_callback,
            find_sources_callback=self._find_preview_sources_callback,
            get_params_callback=self.get_current_settings,  # Pass the settings getter
            preview_size_var=self.preview_size_var,  # Pass the preview size variable
            resize_callback=self._adjust_window_height_for_content,  # Pass the resize callback
            update_clip_callback=self._update_info_display,  # Update info panel
            help_data=self.help_data,
        )
        self.previewer.preview_source_combo.configure(textvariable=self.preview_source_var)
        self.previewer.preview_source_combo["values"] = [
            "Blended Image",
            "Original (Left Eye)",
            "Warped (Right BG)",
            "Inpainted Right Eye",
            "Processed Mask",
            "Anaglyph 3D",
            "Dubois Anaglyph",
            "Optimized Anaglyph",
            "Side-by-Side",
            "Wigglegram",
            "Depth Map",
            "Mesh Warp",
        ]

        # --- FIX: Add previewer's buttons to the list of widgets to disable ---
        self.widgets_to_disable.append(self.previewer.load_preview_button)
        self.widgets_to_disable.append(self.previewer.prev_video_button)
        self.widgets_to_disable.append(self.previewer.next_video_button)
        self.widgets_to_disable.append(self.previewer.video_jump_entry)
        # Pack the previewer right after the folder frame
        self.previewer.pack(fill="both", expand=True, padx=10, pady=5)

        # --- SLIDERS & INFO CONTAINER ---
        middle_container = ttk.Frame(self)
        middle_container.pack(fill="x", padx=10, pady=5)
        middle_container.columnconfigure(0, weight=3)  # Sliders get more space
        middle_container.columnconfigure(1, weight=1)  # Info panel

        # --- MASK PROCESSING PARAMETERS ---
        param_frame = ttk.LabelFrame(middle_container, text="Mask Processing Parameters", padding=10)
        param_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        param_frame.grid_columnconfigure(1, weight=1)

        create_single_slider_with_label_updater(
            self,
            param_frame,
            "Binarize Thresh (<0=Off):",
            self.mask_binarize_threshold_var,
            -0.01,
            1.0,
            0,
            decimals=2,
            step_size=0.01,
        )
        create_single_slider_with_label_updater(
            self, param_frame, "Dilate Kernel:", self.mask_dilate_kernel_size_var, 0, 101, 1
        )
        create_single_slider_with_label_updater(
            self, param_frame, "Blur Kernel:", self.mask_blur_kernel_size_var, 0, 101, 2
        )
        create_single_slider_with_label_updater(self, param_frame, "Shadow Shift:", self.shadow_shift_var, 0, 50, 3)
        create_single_slider_with_label_updater(
            self, param_frame, "Shadow Gamma:", self.shadow_decay_gamma_var, 0.1, 5.0, 4, decimals=2, step_size=0.01
        )
        create_single_slider_with_label_updater(
            self,
            param_frame,
            "Shadow Opacity Start:",
            self.shadow_start_opacity_var,
            0.0,
            1.0,
            5,
            decimals=2,
            step_size=0.01,
        )
        create_single_slider_with_label_updater(
            self,
            param_frame,
            "Shadow Opacity Decay:",
            self.shadow_opacity_decay_var,
            0.0,
            1.0,
            6,
            decimals=2,
            step_size=0.01,
        )
        create_single_slider_with_label_updater(
            self,
            param_frame,
            "Shadow Opacity Min:",
            self.shadow_min_opacity_var,
            0.0,
            1.0,
            7,
            decimals=2,
            step_size=0.01,
        )

        # --- INFO FRAME ---
        info_frame = ttk.LabelFrame(middle_container, text="Current Clip Info", padding=10)
        info_frame.grid(row=0, column=1, sticky="nsew")

        # Row 0: Filename
        ttk.Label(info_frame, text="File:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        ttk.Label(info_frame, textvariable=self.current_filename_var, font=("Segoe UI", 9, "bold")).grid(
            row=0, column=1, sticky="w"
        )

        # Row 1: Resolution
        ttk.Label(info_frame, text="Res:").grid(row=1, column=0, sticky="w", padx=(0, 5))
        ttk.Label(info_frame, textvariable=self.current_resolution_var).grid(row=1, column=1, sticky="w")

        # Row 2: Borders (Moved from Progress frame)
        ttk.Label(info_frame, text="Borders:").grid(row=2, column=0, sticky="w", padx=(0, 5))
        self.border_info_label = ttk.Label(info_frame, textvariable=self.border_info_var)
        self.border_info_label.grid(row=2, column=1, sticky="w")

        # Row 3: Flip Status
        ttk.Label(info_frame, text="Flip:").grid(row=3, column=0, sticky="w", padx=(0, 5))
        ttk.Label(info_frame, textvariable=self.current_flip_status_var).grid(row=3, column=1, sticky="w")

        # --- OPTIONS FRAME ---
        options_frame = ttk.LabelFrame(self, text="Options", padding=10)
        options_frame.pack(fill="x", padx=10, pady=5)

        gpu_check = ttk.Checkbutton(options_frame, text="Use GPU for Mask Processing", variable=self.use_gpu_var)
        gpu_check.pack(side="left", padx=5)
        self._create_hover_tooltip(gpu_check, "use_gpu")
        self.widgets_to_disable.append(gpu_check)

        # --- NEW: Output Format Dropdown ---
        ttk.Label(options_frame, text="Output Format:").pack(side="left", padx=(15, 5))
        output_formats = [
            "Full SBS (Left-Right)",
            "Double SBS",
            "Half SBS (Left-Right)",
            "Full SBS Cross-eye (Right-Left)",
            "Anaglyph (Red/Cyan)",
            "Anaglyph Half-Color",
            "Right-Eye Only",
        ]
        output_format_combo = ttk.Combobox(
            options_frame, textvariable=self.output_format_var, values=output_formats, state="readonly", width=28
        )
        output_format_combo.pack(side="left", padx=5)
        self._create_hover_tooltip(output_format_combo, "output_format")
        self.widgets_to_disable.append(output_format_combo)
        # --- END NEW ---

        color_check = ttk.Checkbutton(
            options_frame, text="Enable Color Transfer", variable=self.enable_color_transfer_var
        )
        color_check.pack(side="left", padx=5)
        self._create_hover_tooltip(color_check, "enable_color_transfer")
        self.widgets_to_disable.append(color_check)

        # --- NEW: Pad to 16:9 Checkbox ---
        pad_check = ttk.Checkbutton(options_frame, text="Pad to 16:9", variable=self.pad_to_16_9_var)
        pad_check.pack(side="left", padx=(15, 5))
        self._create_hover_tooltip(pad_check, "pad_to_16_9")
        self.widgets_to_disable.append(pad_check)

        # --- NEW: Add Borders checkbox ---
        self.add_borders_var = tk.BooleanVar(value=True)
        self.add_borders_var.trace_add("write", self._on_add_borders_changed)
        borders_check = ttk.Checkbutton(options_frame, text="Add Borders", variable=self.add_borders_var)
        borders_check.pack(side="left", padx=(15, 5))
        self._create_hover_tooltip(borders_check, "add_borders")
        self.widgets_to_disable.append(borders_check)
        # --- END NEW ---

        # --- NEW: Resume checkbox ---
        self.resume_var = tk.BooleanVar(value=self.app_config.get("resume", False))
        self.resume_var.trace_add("write", self._on_resume_changed)
        resume_check = ttk.Checkbutton(options_frame, text="Resume", variable=self.resume_var)
        resume_check.pack(side="left", padx=(15, 5))
        self._create_hover_tooltip(resume_check, "resume")
        self.widgets_to_disable.append(resume_check)
        # --- END NEW ---

        # Add Batch Chunk Size option
        ttk.Label(options_frame, text="Batch Chunk Size:").pack(side="left", padx=(20, 5))
        entry_chunk = ttk.Entry(options_frame, textvariable=self.batch_chunk_size_var, width=7)
        entry_chunk.pack(side="left")
        self._create_hover_tooltip(entry_chunk, "batch_chunk_size")
        self.widgets_to_disable.append(entry_chunk)

        # --- PROGRESS & BUTTONS ---
        progress_frame = ttk.LabelFrame(self, text="Progress", padding=10)
        progress_frame.pack(fill="x", padx=10, pady=5)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            length=400,
            mode="determinate",
            style="Custom.Horizontal.TProgressbar",
        )
        self.progress_bar.pack(fill="x")

        # Current Filename
        # self.filename_label = ttk.Label(
        #     progress_frame, textvariable=self.current_filename_var, font=("Segoe UI", 9, "bold")
        # )
        # self.filename_label.pack(pady=(5, 0))

        self.status_label_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_label_var)
        self.status_label.pack(pady=1)

        buttons_frame = ttk.Frame(self, padding=1)
        buttons_frame.pack(fill="x")
        self.start_button = ttk.Button(buttons_frame, text="Start Blending", command=self.start_processing)
        self.start_button.pack(side="left", padx=5, expand=True)
        self._create_hover_tooltip(self.start_button, "start_blending")
        self.stop_button = ttk.Button(buttons_frame, text="Stop", command=self.stop_processing, state="disabled")
        self.widgets_to_disable.append(self.start_button)  # Add to disable list
        # Stop button is handled separately in _set_ui_processing_state
        self.stop_button.pack(side="left", padx=5, expand=True)
        self._create_hover_tooltip(self.stop_button, "stop_blending")

        # --- NEW: Process Current Clip button ---
        self.process_current_button = ttk.Button(
            buttons_frame, text="Process Current Clip", command=self.process_current_clip
        )
        self.process_current_button.pack(side="left", padx=5, expand=True)
        self._create_hover_tooltip(self.process_current_button, "process_current_clip")
        self.widgets_to_disable.append(self.process_current_button)
        # --- END NEW ---

        # --- NEW: Move to Finish button ---
        self.move_to_finish_button = ttk.Button(
            buttons_frame, text="Move to Finish", command=self._move_current_to_finish
        )
        self.move_to_finish_button.pack(side="left", padx=5, expand=True)
        self._create_hover_tooltip(self.move_to_finish_button, "move_to_finish")
        self.widgets_to_disable.append(self.move_to_finish_button)
        # --- END NEW ---

    def _browse_folder(self, var: tk.StringVar):
        folder = filedialog.askdirectory(initialdir=var.get())
        if folder:
            var.set(folder)

    def _show_encoding_settings(self):
        """Show the encoding settings dialog."""
        config = {
            "codec": self.codec_var.get(),
            "encoding_quality": self.encoding_quality_var.get(),
            "encoding_tune": self.encoding_tune_var.get(),
            "output_crf": self.output_crf_var.get(),
            "nvenc_lookahead_enabled": self.nvenc_lookahead_enabled_var.get(),
            "nvenc_lookahead": self.nvenc_lookahead_var.get(),
            "nvenc_spatial_aq": self.nvenc_spatial_aq_var.get(),
            "nvenc_temporal_aq": self.nvenc_temporal_aq_var.get(),
            "nvenc_aq_strength": self.nvenc_aq_strength_var.get(),
            "color_tags": self.color_tags_var.get(),
            "dnxhr_fullres_split": self.dnxhr_fullres_split_var.get(),
            "dnxhr_profile": self.dnxhr_profile_var.get(),
        }

        dialog = EncodingSettingsDialog(
            self,
            app_config=config,
            help_data=self.help_data,
            title="Merging GUI - Encoding Settings",
            show_extra_options=True,
            show_color_tags=True,
        )
        self.wait_window(dialog.dialog)

        if dialog.result:
            self.codec_var.set(dialog.result.get("codec", "H.265"))
            self.encoder_var.set(dialog.result.get("encoding_encoder", "Auto"))
            self.encoding_quality_var.set(dialog.result.get("encoding_quality", "Medium"))
            self.encoding_tune_var.set(dialog.result.get("encoding_tune", "None"))
            self.output_crf_var.set(str(dialog.result.get("output_crf", 23)))
            self.nvenc_lookahead_enabled_var.set(dialog.result.get("nvenc_lookahead_enabled", False))
            self.nvenc_lookahead_var.set(dialog.result.get("nvenc_lookahead", 16))
            self.nvenc_spatial_aq_var.set(dialog.result.get("nvenc_spatial_aq", False))
            self.nvenc_temporal_aq_var.set(dialog.result.get("nvenc_temporal_aq", False))
            self.nvenc_aq_strength_var.set(dialog.result.get("nvenc_aq_strength", 8))
            self.color_tags_var.set(dialog.result.get("color_tags", "Auto"))
            self.dnxhr_fullres_split_var.set(dialog.result.get("dnxhr_fullres_split", False))
            self.dnxhr_profile_var.set(dialog.result.get("dnxhr_profile", "HQX (10-bit 4:2:2)"))

    def _find_video_by_core_name(self, folder: str, core_name: str) -> Optional[str]:
        """Scans a folder for a file matching the core_name with any common video extension."""
        return find_video_by_core_name(folder, core_name)

    def _find_sidecar_file(self, base_path: str) -> Optional[str]:
        """Looks for a sidecar JSON file next to the video file."""
        return find_sidecar_file(base_path)

    def _read_clip_sidecar(self, video_path: str, core_name: str) -> dict:
        """
        Reads the sidecar file for a clip if it exists.
        Returns a dictionary of sidecar data merged with defaults.
        """
        search_folders = []
        if self.inpainted_folder_var.get():
            search_folders.append(self.inpainted_folder_var.get())
        if self.original_folder_var.get():
            search_folders.append(self.original_folder_var.get())
        return read_clip_sidecar(self.sidecar_manager, video_path, core_name, search_folders)

    def _update_border_info(self, left_border: float, right_border: float):
        """Updates the border info display in the GUI."""
        if left_border > 0 or right_border > 0:
            self.border_info_var.set(f"Borders: L={left_border:.3f}%, R={right_border:.3f}%")
        else:
            self.border_info_var.set("Borders: None")

    def _clear_border_info(self):
        """Clears the border info display."""
        self.border_info_var.set("Borders: N/A")

    def on_slider_release(self, event):
        """Called when a slider is released. Updates the preview."""
        # This now just collects parameters and sends them to the previewer module.
        params = self.get_current_settings()
        if params:
            self.previewer.set_parameters(params)

    def _on_add_borders_changed(self, *args):
        """Called when the Add Borders checkbox is toggled. Updates the preview."""
        if hasattr(self, "previewer") and self.previewer.video_list:
            self.previewer.update_preview()

    def _on_folder_changed(self, *args):
        """Called when a folder path changes. Resets the video list scan flag."""
        if hasattr(self, "previewer"):
            self.previewer.reset_video_list_scan()

    def _on_resume_changed(self, *args):
        """Called when the Resume checkbox is changed. Clears preview to apply new setting."""
        if hasattr(self, "previewer") and self.previewer.video_list:
            # Update preview to reflect the new setting
            self.previewer.update_preview()

    def _set_ui_processing_state(self, is_processing: bool):
        """Disables or enables all interactive widgets during processing."""
        # --- FIX: Explicitly handle start/stop button states ---
        try:
            self.start_button.config(state="disabled" if is_processing else "normal")
            self.stop_button.config(state="normal" if is_processing else "disabled")
        except tk.TclError:
            pass  # Ignore if widgets don't exist yet
        # --- END FIX ---
        state = "disabled" if is_processing else "normal"
        for widget in self.widgets_to_disable:
            try:
                # Special handling for combobox which uses 'readonly' instead of 'normal'
                if isinstance(widget, ttk.Combobox):
                    widget.config(state="disabled" if is_processing else "readonly")
                else:
                    widget.config(state=state)
            except tk.TclError:
                # Widget might have been destroyed, ignore
                pass

    def _update_info_display(self):
        """Updates the current filename display, resolution, flip status, and window title."""
        if hasattr(self, "previewer") and self.previewer.video_list and self.previewer.current_video_index != -1:
            source_dict = self.previewer.video_list[self.previewer.current_video_index]
            inpainted_path = source_dict.get("inpainted", "")
            filename = os.path.basename(inpainted_path)

            # Update Filename
            self.current_filename_var.set(filename)
            self.title(f"Stereocrafter Merging GUI {GUI_VERSION} - {filename}")

            # Update Resolution (from inpainted video)
            try:
                if inpainted_path and os.path.exists(inpainted_path):
                    info = get_video_stream_info(inpainted_path)
                    if info:
                        w, h = info.get("width", "N/A"), info.get("height", "N/A")
                        self.current_resolution_var.set(f"{w}x{h}")
                    else:
                        self.current_resolution_var.set("N/A")
                else:
                    self.current_resolution_var.set("N/A")
            except Exception:
                self.current_resolution_var.set("Error")

            # Update Flip Status (from sidecar or filename)
            sidecar_data = source_dict.get("sidecar", {})
            is_flipped = sidecar_data.get("flip_horizontal", False)
            if not is_flipped and os.path.splitext(filename)[0].endswith("F"):
                is_flipped = True
            self.current_flip_status_var.set("Yes" if is_flipped else "No")
        else:
            self.current_filename_var.set("No video loaded")
            self.current_resolution_var.set("N/A")
            self.current_flip_status_var.set("N/A")
            self.title(f"Stereocrafter Merging GUI {GUI_VERSION}")

    def update_status_label(self, message):
        self.status_label_var.set(message)
        self.update_idletasks()

    def _clear_preview_resources(self):
        """Closes all preview-related video readers and clears the preview display."""
        self.previewer.cleanup()

    def _cleanup_worker(self):
        """
        A worker thread that processes a queue of files to be moved.
        It will retry moving a file until it succeeds.
        """
        stop_signal_received = False
        while not stop_signal_received or not self.cleanup_queue.empty():
            try:
                # Wait for an item, but with a timeout so the loop can check the stop condition
                item = self.cleanup_queue.get(timeout=1)

                if item is None:
                    logger.debug("Cleanup worker received stop signal. Will exit when queue is empty.")
                    stop_signal_received = True
                    continue  # Continue loop to check if queue is empty

                src_path, dest_folder = item

                try:
                    if not os.path.exists(src_path):
                        logger.debug(
                            f"Cleanup: Source file '{os.path.basename(src_path)}' no longer exists. Skipping move."
                        )
                        continue

                    finished_dir = os.path.join(dest_folder, "finished")
                    os.makedirs(finished_dir, exist_ok=True)
                    dest_path = os.path.join(finished_dir, os.path.basename(src_path))

                    if os.path.exists(dest_path):
                        logger.debug(f"Cleanup: Destination '{os.path.basename(dest_path)}' exists. Deleting source.")
                        os.remove(src_path)
                    else:
                        shutil.move(src_path, finished_dir)
                    logger.debug(f"Cleanup: Successfully moved '{os.path.basename(src_path)}'.")
                except (PermissionError, OSError):
                    logger.debug(f"Cleanup: File '{os.path.basename(src_path)}' is locked. Retrying in 3 seconds...")
                    time.sleep(3)
                    self.cleanup_queue.put(item)  # Put it back on the queue to retry
                except Exception as e:
                    logger.error(
                        f"Cleanup worker encountered an unexpected error for {os.path.basename(src_path)}: {e}",
                        exc_info=True,
                    )

            except queue.Empty:
                # This is expected when waiting for items. The loop condition will handle exit.
                continue
        logger.debug("Cleanup worker has finished its queue and is now exiting.")

    def _retry_failed_moves(self):
        """Attempts to move any files that previously failed to move."""
        if not self.failed_moves:
            return

        logger.info(f"Retrying {len(self.failed_moves)} previously failed file moves...")

        # Use a copy of the list to iterate over, so we can safely remove from the original
        remaining_failures = []
        for src_path, dest_folder in self.failed_moves:
            try:
                # --- FIX: Check for source existence FIRST ---
                if not os.path.exists(src_path):
                    logger.debug(
                        f"Retry: Source file '{os.path.basename(src_path)}' no longer exists. Assuming it was moved successfully."
                    )
                    continue  # This item is resolved, do not add to remaining_failures

                finished_dir = os.path.join(dest_folder, "finished")
                dest_path = os.path.join(finished_dir, os.path.basename(src_path))

                if os.path.exists(dest_path):
                    # Destination exists, so the move likely succeeded. We just need to delete the source.
                    logger.info(
                        f"Retry: Destination '{os.path.basename(dest_path)}' exists. Deleting source '{os.path.basename(src_path)}'."
                    )
                    try:
                        os.remove(src_path)
                    except Exception as e_del:
                        logger.error(
                            f"Retry: Failed to delete source '{os.path.basename(src_path)}' even though destination exists: {e_del}"
                        )
                        remaining_failures.append((src_path, dest_folder))  # Keep it for the next final retry
                else:
                    # Destination does not exist, but we know the source does. This is a true move retry.
                    shutil.move(src_path, finished_dir)
                    logger.debug(f"Successfully moved previously failed file: {os.path.basename(src_path)}")

            except (PermissionError, OSError) as e:
                logger.warning(f"Retry failed for {os.path.basename(src_path)}: {e}. Will try again later.")
                remaining_failures.append((src_path, dest_folder))  # Add back to the list for the next attempt
            except Exception as e:
                logger.error(f"Unexpected error during retry for {os.path.basename(src_path)}: {e}", exc_info=True)

        self.failed_moves = remaining_failures

    def start_processing(self):
        if self.is_processing:
            messagebox.showwarning("Busy", "Processing is already in progress.")
            return

        self.is_processing = True
        self.stop_event.clear()
        self._set_ui_processing_state(True)  # Disable UI

        # --- NEW: Start the cleanup worker thread ---
        self.cleanup_queue = queue.Queue()  # Clear the queue from any previous run
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        logger.info("File cleanup worker thread started.")
        # --- END NEW ---

        # --- NEW: Clear preview resources before starting batch processing ---
        self._clear_preview_resources()

        self.update_status_label("Starting...")

        # Collect settings
        settings = self.get_current_settings()

        # Run in a separate thread
        self.processing_thread = threading.Thread(target=self.run_batch_process, args=(settings, None), daemon=True)
        self.processing_thread.start()

    def stop_processing(self):
        if self.is_processing:
            self.stop_event.set()
            self.update_status_label("Stopping...")

    def process_current_clip(self):
        """Process the currently selected clip only."""
        if self.is_processing:
            messagebox.showwarning("Busy", "Processing is already in progress.")
            return

        # Get current video from previewer
        if not hasattr(self, "previewer") or not self.previewer.video_list:
            messagebox.showwarning("No Video", "No video loaded in previewer.")
            return

        current_index = getattr(self.previewer, "current_video_index", 0)
        if current_index < 0 or current_index >= len(self.previewer.video_list):
            messagebox.showwarning("Invalid Index", "No video selected.")
            return

        source_dict = self.previewer.video_list[current_index]
        inpainted_path = source_dict.get("inpainted")

        if not inpainted_path or not os.path.exists(inpainted_path):
            messagebox.showwarning("Invalid Path", "Inpainted video path not found.")
            return

        # Get current settings
        settings = self.get_current_settings()
        if not settings:
            return

        # Temporarily set inpainted_folder to just this file's directory
        settings["inpainted_folder"] = os.path.dirname(inpainted_path)

        self.is_processing = True
        self.stop_event.clear()
        self._set_ui_processing_state(True)
        self._clear_preview_resources()
        base_name = os.path.basename(inpainted_path)
        self.update_status_label(f"Processing single clip: {base_name}")

        # Run in a separate thread using the existing batch processor
        # Pass the specific video path to process only this one
        self.processing_thread = threading.Thread(
            target=self.run_batch_process, args=(settings, inpainted_path), daemon=True
        )
        self.processing_thread.start()

    def _move_current_to_finish(self):
        """Moves the currently selected clip's source files to the finished folder without processing."""
        if self.is_processing:
            messagebox.showwarning("Busy", "Processing is already in progress.")
            return

        # Get current video from previewer
        if not hasattr(self, "previewer") or not self.previewer.video_list:
            messagebox.showwarning("No Video", "No video loaded in previewer.")
            return

        current_index = getattr(self.previewer, "current_video_index", 0)
        if current_index < 0 or current_index >= len(self.previewer.video_list):
            messagebox.showwarning("Invalid Index", "No video selected.")
            return

        source_dict = self.previewer.video_list[current_index]
        inpainted_path = source_dict.get("inpainted")
        splatted_path = source_dict.get("splatted")
        original_path = source_dict.get("original")

        if not inpainted_path or not os.path.exists(inpainted_path):
            messagebox.showwarning("Invalid Path", "Inpainted video path not found.")
            return

        # Close preview resources to release file handles before moving
        self.previewer._clear_preview_resources()
        time.sleep(0.5)  # Give OS time to release file handles

        base_name = os.path.basename(inpainted_path)

        # Get folders from settings
        inpainted_folder = self.inpainted_folder_var.get()
        mask_folder = self.mask_folder_var.get()
        original_folder = self.original_folder_var.get()

        # Get core name for sidecar
        inpaint_suffix_reg = r"_inpainted_right_eyeF?\.mp4$"
        sbs_suffix_reg = r"_inpainted_sbsF?\.mp4$"

        is_sbs_input = bool(re.search(sbs_suffix_reg, base_name))
        suffix_to_remove = re.search(sbs_suffix_reg if is_sbs_input else inpaint_suffix_reg, base_name).group(0)
        core_name_with_width = base_name[: -len(suffix_to_remove)]
        last_underscore_idx = core_name_with_width.rfind("_")
        core_name = core_name_with_width[:last_underscore_idx] if last_underscore_idx != -1 else core_name_with_width

        # Queue files for moving
        files_to_move = []

        # Add inpainted file
        if os.path.exists(inpainted_path):
            files_to_move.append((inpainted_path, inpainted_folder))

        # Add splatted file
        if splatted_path and os.path.exists(splatted_path):
            files_to_move.append((splatted_path, mask_folder))

        # Add original file
        if original_path and os.path.exists(original_path):
            files_to_move.append((original_path, original_folder))

        # Add sidecar files for inpainted
        inpainted_base = os.path.splitext(inpainted_path)[0]
        for ext in [".fssidecar", ".json"]:
            sidecar_path = inpainted_base + ext
            if os.path.exists(sidecar_path):
                files_to_move.append((sidecar_path, inpainted_folder))

        # Add sidecar files for original
        if original_path:
            original_base = os.path.splitext(original_path)[0]
            for ext in [".fssidecar", ".json"]:
                sidecar_path = original_base + ext
                if os.path.exists(sidecar_path):
                    files_to_move.append((sidecar_path, original_folder))

        if not files_to_move:
            messagebox.showwarning("No Files", "No source files found to move.")
            return

        # Confirm with user
        file_count = len(files_to_move)
        if not messagebox.askyesno(
            "Confirm Move", f"Move {file_count} file(s) for '{core_name}' to finished folder?\n\nThis cannot be undone."
        ):
            return

        # Move files using the common utility
        moved_count, failed_count, failed_files = move_files_to_finished(
            files_to_move=files_to_move,
            logger=logger,
            wait_before_move=0.5,
            close_handles_callback=lambda: self.previewer._clear_preview_resources(),
        )

        # Remove the clip from video_list without rescanning
        self.previewer.video_list.pop(current_index)

        # Adjust index if needed
        total_clips = len(self.previewer.video_list)
        if current_index >= total_clips:
            # Was at the last clip, go to previous
            new_index = max(0, total_clips - 1)
        else:
            new_index = current_index

        # Update preview
        if total_clips > 0:
            self.previewer.current_video_index = new_index
            self.previewer._load_preview_by_index(new_index)
            # Update video counter
            if hasattr(self.previewer, "video_status_label_var"):
                self.previewer.video_status_label_var.set(f"Video: {new_index + 1} / {total_clips}")
        else:
            # No more clips
            self.previewer.current_video_index = -1
            self.previewer._clear_preview_resources()
            if hasattr(self.previewer, "video_status_label_var"):
                self.previewer.video_status_label_var.set("Video: 0 / 0")

        # Show result
        if failed_files:
            error_msg = "\n".join([f"{f}: {e}" for f, e in failed_files])
            messagebox.showwarning(
                "Move Complete with Errors", f"Moved {moved_count}/{file_count} files.\n\nErrors:\n{error_msg}"
            )
        else:
            messagebox.showinfo("Move Complete", f"Successfully moved {moved_count} file(s) to finished folder.")

    def processing_done(self, stopped=False):
        self.is_processing = False
        self._set_ui_processing_state(False)  # Re-enable UI
        message = "Processing stopped." if stopped else "Processing completed."
        self.update_status_label(message)
        self.progress_var.set(0)
        self._clear_border_info()
        if hasattr(self, "previewer"):
            self.previewer.reset_video_list_scan()

        # --- NEW: Schedule VRAM release after a short delay to ensure stability ---
        delay_ms = 2000  # 2 seconds
        logger.info(f"Scheduling VRAM release in {delay_ms / 1000} seconds...")
        self.after(delay_ms, release_cuda_memory)
        # --- END NEW ---

    def get_current_settings(self):
        """Collects all GUI settings into a dictionary, performing type conversion."""
        try:
            settings = {
                "inpainted_folder": self.inpainted_folder_var.get(),
                "original_folder": self.original_folder_var.get(),
                "mask_folder": self.mask_folder_var.get(),
                "output_folder": self.output_folder_var.get(),
                "use_gpu": self.use_gpu_var.get(),
                "pad_to_16_9": self.pad_to_16_9_var.get(),
                "add_borders": self.add_borders_var.get(),
                "resume": self.resume_var.get(),
                "output_format": self.output_format_var.get(),
                "batch_chunk_size": int(self.batch_chunk_size_var.get()),
                "enable_color_transfer": self.enable_color_transfer_var.get(),
                "preview_size": self.preview_size_var.get(),
                "preview_source": self.preview_source_var.get(),
                # Encoding params
                "codec": self.codec_var.get(),
                "encoding_quality": self.encoding_quality_var.get(),
                "encoding_tune": self.encoding_tune_var.get(),
                "output_crf": int(self.output_crf_var.get()),
                "nvenc_lookahead_enabled": self.nvenc_lookahead_enabled_var.get(),
                "nvenc_lookahead": self.nvenc_lookahead_var.get(),
                "nvenc_spatial_aq": self.nvenc_spatial_aq_var.get(),
                "nvenc_temporal_aq": self.nvenc_temporal_aq_var.get(),
                "nvenc_aq_strength": self.nvenc_aq_strength_var.get(),
                "color_tags": self.color_tags_var.get(),
                "dnxhr_fullres_split": self.dnxhr_fullres_split_var.get(),
                "dnxhr_profile": self.dnxhr_profile_var.get(),
                # Mask params
                "mask_binarize_threshold": float(self.mask_binarize_threshold_var.get()),
                "mask_dilate_kernel_size": int(self.mask_dilate_kernel_size_var.get()),
                "mask_blur_kernel_size": int(self.mask_blur_kernel_size_var.get()),
                "shadow_shift": int(self.shadow_shift_var.get()),
                "shadow_start_opacity": float(self.shadow_start_opacity_var.get()),
                "shadow_opacity_decay": float(self.shadow_opacity_decay_var.get()),
                "shadow_min_opacity": float(self.shadow_min_opacity_var.get()),
                "shadow_decay_gamma": float(self.shadow_decay_gamma_var.get()),
            }
            return settings
        except (ValueError, TypeError) as e:
            messagebox.showerror(
                "Invalid Settings", f"Please check your parameter values. They must be valid numbers.\n\nError: {e}"
            )
            return None

    def _read_ffmpeg_output(self, pipe, log_level):
        """Helper method to read FFmpeg's output without blocking."""
        try:
            # Use iter to read line by line
            for line in iter(pipe.readline, b""):  # Read bytes until an empty byte string
                if line:
                    # Decode bytes to string for logging, ignoring potential decoding errors
                    logger.log(log_level, f"FFmpeg: {line.decode('utf-8', errors='ignore').strip()}")
        except Exception as e:
            logger.error(f"Error reading FFmpeg pipe: {e}")
        finally:
            if pipe:
                pipe.close()

    def run_batch_process(self, settings, single_video_path=None):
        """
        This is the main logic that will run in a background thread.
        If single_video_path is provided, only process that one video.
        """
        if settings is None:
            self.after(0, self.processing_done, True)
            return

        # Single video mode
        if single_video_path and os.path.exists(single_video_path):
            inpainted_videos = [single_video_path]
            single_mode = True
        else:
            inpainted_videos = sorted(glob.glob(os.path.join(settings["inpainted_folder"], "*.mp4")))
            single_mode = False

        if not inpainted_videos:
            self.after(0, lambda: messagebox.showinfo("Info", "No .mp4 files found in the inpainted video folder."))
            self.after(0, self.processing_done)
            return

        # --- NEW: Skip already finished files when Resume is enabled ---
        resume_enabled = settings.get("resume", False)
        if resume_enabled:
            finished_dir = os.path.join(settings["inpainted_folder"], "finished")
            if os.path.isdir(finished_dir):
                finished_files = set(os.listdir(finished_dir))
                original_count = len(inpainted_videos)
                inpainted_videos = [v for v in inpainted_videos if os.path.basename(v) not in finished_files]
                skipped_count = original_count - len(inpainted_videos)
                if skipped_count > 0:
                    logger.info(f"Resume: Skipped {skipped_count} already processed files.")
            else:
                logger.info("Resume: No 'finished' folder found. Processing all files.")

        if not inpainted_videos:
            self.after(0, lambda: messagebox.showinfo("Info", "All videos have already been processed (Resume mode)."))
            self.after(0, self.processing_done)
            return
        # --- END NEW ---

        # --- NEW: Clear any failed moves from a previous run ---
        self.failed_moves = []

        total_videos = len(inpainted_videos)
        self.progress_bar.config(maximum=total_videos)

        for i, inpainted_video_path in enumerate(inpainted_videos):
            if self.stop_event.is_set():
                logger.info("Processing stopped by user.")
                break

            # In single mode, stop after processing the first video
            if single_mode and i > 0:
                break

            base_name = os.path.basename(inpainted_video_path)
            self.after(0, self.update_status_label, f"Processing {i + 1}/{total_videos}: {base_name}")

            # Initialize readers to None for robust cleanup
            inpainted_reader, splatted_reader, original_reader = None, None, None
            original_video_path_to_move = None  # To track which original file to move
            try:
                # --- 1. Find corresponding files (same logic as preview) ---
                inpaint_suffix_reg = r"_inpainted_right_eyeF?\.mp4$"
                sbs_suffix_reg = r"_inpainted_sbsF?\.mp4$"

                is_sbs_input = bool(re.search(sbs_suffix_reg, base_name))
                match = re.search(sbs_suffix_reg if is_sbs_input else inpaint_suffix_reg, base_name)
                if not match:
                    logger.error(f"Could not identify suffix for '{base_name}'. Skipping.")
                    continue
                suffix_to_remove = match.group(0)
                core_name_with_width = base_name[: -len(suffix_to_remove)]

                # --- FIX: Gracefully handle cases where the filename format is unexpected ---
                last_underscore_idx = core_name_with_width.rfind("_")
                if last_underscore_idx == -1:
                    logger.error(
                        f"Could not parse core name from '{core_name_with_width}'. Skipping video '{base_name}'."
                    )
                    self.after(0, self.progress_var.set, i + 1)  # Still advance progress bar
                    continue
                core_name = core_name_with_width[:last_underscore_idx]
                # --- END FIX ---

                # --- NEW: Read sidecar file for this clip ---
                clip_sidecar_data = self._read_clip_sidecar(inpainted_video_path, core_name)
                flip_horizontal = clip_sidecar_data.get("flip_horizontal", False)
                if not flip_horizontal and os.path.splitext(base_name)[0].endswith("F"):
                    flip_horizontal = True
                logger.info(
                    f"Sidecar/Filename for '{core_name}': left_border={clip_sidecar_data.get('left_border')}, right_border={clip_sidecar_data.get('right_border')}, flip_horizontal={flip_horizontal}"
                )

                left_border = clip_sidecar_data.get("left_border", 0.0)
                right_border = clip_sidecar_data.get("right_border", 0.0)
                self._update_border_info(left_border, right_border)
                # --- END NEW ---

                mask_folder = settings["mask_folder"]
                splatted4_pattern = os.path.join(mask_folder, f"{core_name}_*_splatted4*.mp4")
                splatted2_pattern = os.path.join(mask_folder, f"{core_name}_*_splatted2*.mp4")
                splatted4_matches = glob.glob(splatted4_pattern)
                splatted2_matches = glob.glob(splatted2_pattern)

                splatted_file_path = None
                if splatted4_matches:
                    splatted_file_path = splatted4_matches[0]
                    is_dual_input = False
                elif splatted2_matches:
                    splatted_file_path = splatted2_matches[0]
                    is_dual_input = True

                # 2. Open readers, don't load all frames
                # --- FIX: Validate all file paths before attempting to open them ---
                if not splatted_file_path or not os.path.exists(splatted_file_path):
                    logger.error(
                        f"Missing required splatted file for '{core_name}'. Searched for '{splatted4_pattern}' and '{splatted2_pattern}'. Skipping video."
                    )
                    self.after(0, self.progress_var.set, i + 1)
                    continue

                inpainted_reader = VideoReader(inpainted_video_path, ctx=cpu(0))
                splatted_reader = VideoReader(splatted_file_path, ctx=cpu(0))

                # --- FIX: Determine original_reader based on input type ---
                original_reader = None  # Assume None initially
                if is_dual_input:  # splatted2
                    # --- MODIFIED: Use helper to find original video with any extension ---
                    original_video_path = self._find_video_by_core_name(settings["original_folder"], core_name)
                    original_video_path_to_move = original_video_path  # Track for moving later

                    if original_video_path and os.path.exists(original_video_path):
                        logger.info(
                            f"Found matching original video for dual-input: {os.path.basename(original_video_path)}"
                        )
                        original_reader = VideoReader(original_video_path, ctx=cpu(0))
                    else:
                        logger.warning(f"Original video not found for dual-input mode: '{core_name}.*'.")
                        logger.warning(
                            "Will proceed, but only 'Right-Eye Only' output will be possible for this video."
                        )
                else:  # splatted4 (quad)
                    # For quad-splatted files, the splatted file itself is the source for the left eye.
                    # We can use the splatted_reader as a placeholder to indicate a valid left-eye source exists.
                    original_reader = splatted_reader
                # --- END FIX ---

                # 3. Setup encoder pipe
                num_frames = len(inpainted_reader)
                fps = inpainted_reader.get_avg_fps()
                video_stream_info = get_video_stream_info(inpainted_video_path)

                # Determine output dimensions from a sample frame
                sample_splatted_np = splatted_reader.get_batch([0]).asnumpy()
                _, H_splat, W_splat, _ = sample_splatted_np.shape
                if is_dual_input:
                    hires_H, hires_W = H_splat, W_splat // 2
                else:
                    hires_H, hires_W = H_splat // 2, W_splat // 2

                # --- NEW: Check if SBS/3D output is possible ---
                output_format = settings["output_format"]
                if original_reader is None and output_format != "Right-Eye Only":
                    logger.warning(
                        f"Original video is missing for '{base_name}'. Forcing output format to 'Right-Eye Only'."
                    )
                    output_format = "Right-Eye Only"
                # --- END NEW ---

                # --- NEW: Determine output dimensions, perceived width for filename, and suffix ---
                perceived_width_for_filename = hires_W  # Default to single-eye width

                if output_format == "Full SBS Cross-eye (Right-Left)":
                    output_width = hires_W * 2
                    output_suffix = "_merged_full_sbsx.mp4"
                    # Perceived width is single eye
                elif output_format == "Full SBS (Left-Right)":
                    output_width = hires_W * 2
                    output_suffix = "_merged_full_sbs.mp4"
                    # Perceived width is single eye
                elif output_format == "Double SBS":
                    output_width = hires_W * 2
                    output_height = hires_H * 2
                    output_suffix = "_merged_half_sbs.mp4"
                    perceived_width_for_filename = hires_W * 2  # Use the full file width for the filename
                elif output_format == "Half SBS (Left-Right)":
                    output_width = hires_W
                    output_suffix = "_merged_half_sbs.mp4"
                    # Perceived width is single eye, as player will stretch it.
                elif output_format in ["Anaglyph (Red/Cyan)", "Anaglyph Half-Color"]:
                    output_width = hires_W
                    output_suffix = "_merged_anaglyph.mp4"
                    # Perceived width is the full output width
                else:  # Right-Eye Only
                    output_width = hires_W
                    output_suffix = "_merged_right_eye.mp4"
                    # Perceived width is the full output width

                if "output_height" not in locals():  # Set default height if not already set by a special format
                    output_height = hires_H

                # Construct the final filename using the core name and the new perceived width
                output_filename = f"{core_name}_{perceived_width_for_filename}{output_suffix}"
                output_path = os.path.join(settings["output_folder"], output_filename)
                # --- END NEW ---

                # --- NEW: Pass encoding settings to FFmpeg ---
                ffmpeg_process = start_ffmpeg_pipe_process(
                    content_width=output_width,
                    content_height=output_height,
                    final_output_mp4_path=output_path,
                    fps=fps,
                    video_stream_info=video_stream_info,
                    pad_to_16_9=settings["pad_to_16_9"],
                    output_format_str=output_format,
                    encoding_options={
                        "codec": settings.get("codec", "Auto"),
                        "encoding_quality": settings.get("encoding_quality", "Medium"),
                        "encoding_tune": settings.get("encoding_tune", "None"),
                        "output_crf": settings.get("output_crf", 23),
                        "color_tags": settings.get("color_tags", "Auto"),
                        "nvenc_lookahead_enabled": settings.get("nvenc_lookahead_enabled", False),
                        "nvenc_lookahead": settings.get("nvenc_lookahead", 16),
                        "nvenc_spatial_aq": settings.get("nvenc_spatial_aq", False),
                        "nvenc_temporal_aq": settings.get("nvenc_temporal_aq", False),
                        "nvenc_aq_strength": settings.get("nvenc_aq_strength", 8),
                    },
                )  # Pass the encoding options

                if ffmpeg_process is None:
                    raise RuntimeError("Failed to start FFmpeg pipe process.")

                # --- NEW: Start threads to read stdout and stderr to prevent deadlock ---
                stdout_thread = threading.Thread(
                    target=self._read_ffmpeg_output, args=(ffmpeg_process.stdout, logging.DEBUG), daemon=True
                )
                stderr_thread = threading.Thread(
                    target=self._read_ffmpeg_output, args=(ffmpeg_process.stderr, logging.DEBUG), daemon=True
                )
                stdout_thread.start()
                stderr_thread.start()

                # 4. Loop through chunks
                chunk_size = settings.get("batch_chunk_size", 32)
                for frame_start in range(0, num_frames, chunk_size):
                    if self.stop_event.is_set():
                        break

                    frame_end = min(frame_start + chunk_size, num_frames)
                    frame_indices = list(range(frame_start, frame_end))
                    if not frame_indices:
                        break

                    self.after(
                        0, self.update_status_label, f"Processing frames {frame_start + 1}-{frame_end}/{num_frames}..."
                    )

                    # Load current chunk
                    inpainted_np = inpainted_reader.get_batch(frame_indices).asnumpy()
                    splatted_np = splatted_reader.get_batch(frame_indices).asnumpy()

                    # Convert to tensors and extract parts (same logic as preview)
                    # ... (this logic is identical to update_preview's frame loading part)
                    inpainted_tensor_full = torch.from_numpy(inpainted_np).permute(0, 3, 1, 2).float() / 255.0
                    splatted_tensor = torch.from_numpy(splatted_np).permute(0, 3, 1, 2).float() / 255.0
                    inpainted = (
                        inpainted_tensor_full[:, :, :, inpainted_tensor_full.shape[3] // 2 :]
                        if is_sbs_input
                        else inpainted_tensor_full
                    )
                    _, _, H, W = splatted_tensor.shape

                    if is_dual_input:
                        # --- NEW: Handle missing original_reader for dual input ---
                        if original_reader is None:
                            # Create a black tensor as a placeholder for the left eye
                            original_left = torch.zeros_like(inpainted)  # Match inpainted shape
                        else:
                            original_np = original_reader.get_batch(frame_indices).asnumpy()
                            original_left = torch.from_numpy(original_np).permute(0, 3, 1, 2).float() / 255.0
                        # --- END NEW ---
                        mask_raw = splatted_tensor[:, :, :, : W // 2]
                        warped_original = splatted_tensor[:, :, :, W // 2 :]
                    else:
                        original_left = splatted_tensor[:, :, : H // 2, : W // 2]
                        mask_raw = splatted_tensor[:, :, H // 2 :, : W // 2]
                        warped_original = splatted_tensor[:, :, H // 2 :, W // 2 :]

                    if flip_horizontal and is_dual_input:
                        # In Dual mode, original_left is the raw video (not flipped).
                        # We flip it early to align with the already-flipped inpainted/splatted files.
                        # We then process everything in 'Flipped Space'.
                        original_left = torch.flip(original_left, dims=[3])
                    # (In Quad mode, original_left comes from the splatted file and is already flipped)

                    mask_np = mask_raw.permute(0, 2, 3, 1).cpu().numpy()
                    mask_gray_np = np.mean(mask_np, axis=3)
                    mask = torch.from_numpy(mask_gray_np).float().unsqueeze(1)

                    # Process chunk
                    use_gpu = settings["use_gpu"] and torch.cuda.is_available()
                    device = "cuda" if use_gpu else "cpu"
                    mask, inpainted, original_left, warped_original = (
                        mask.to(device),
                        inpainted.to(device),
                        original_left.to(device),
                        warped_original.to(device),
                    )

                    if inpainted.shape[2] != hires_H or inpainted.shape[3] != hires_W:
                        # Calculate aspect-ratio-correct dimensions for inpaint
                        # The inpaint should match the source aspect ratio, not be stretched uniformly
                        target_aspect = hires_W / hires_H
                        inpaint_aspect = inpainted.shape[3] / inpainted.shape[2]

                        if abs(inpaint_aspect - target_aspect) > 0.01:
                            # Aspect ratios differ - resize to fit within target while preserving aspect ratio
                            logger.debug(
                                f"Inpaint aspect ratio ({inpaint_aspect:.4f}) differs from target ({target_aspect:.4f}). "
                                f"Resizing from {inpainted.shape[3]}x{inpainted.shape[2]} to fit {hires_W}x{hires_H}."
                            )
                            # Calculate new dimensions that fit within target while preserving aspect ratio
                            if inpaint_aspect > target_aspect:
                                # Inpaint is wider - fit to width
                                new_w = hires_W
                                new_h = int(round(hires_W / inpaint_aspect))
                            else:
                                # Inpaint is taller - fit to height
                                new_h = hires_H
                                new_w = int(round(hires_H * inpaint_aspect))
                            # First resize to aspect-ratio-correct dimensions
                            inpainted = F.interpolate(
                                inpainted, size=(new_h, new_w), mode="bicubic", align_corners=False
                            )
                            mask = F.interpolate(mask, size=(new_h, new_w), mode="bilinear", align_corners=False)
                            # Then do final resize to match target dimensions
                            inpainted = F.interpolate(
                                inpainted, size=(hires_H, hires_W), mode="bicubic", align_corners=False
                            )
                            mask = F.interpolate(mask, size=(hires_H, hires_W), mode="bilinear", align_corners=False)
                        else:
                            # Aspect ratios match - direct resize
                            inpainted = F.interpolate(
                                inpainted, size=(hires_H, hires_W), mode="bicubic", align_corners=False
                            )
                            mask = F.interpolate(mask, size=(hires_H, hires_W), mode="bilinear", align_corners=False)

                    if settings["enable_color_transfer"]:
                        adjusted_frames = []
                        for frame_idx in range(inpainted.shape[0]):
                            adjusted_frame = apply_color_transfer(
                                original_left[frame_idx].cpu(), inpainted[frame_idx].cpu()
                            )
                            adjusted_frames.append(adjusted_frame.to(device))
                        inpainted = torch.stack(adjusted_frames)

                    processed_mask = mask.clone()
                    # --- NEW: Binarization as the first step ---
                    if settings["mask_binarize_threshold"] >= 0.0:
                        processed_mask = (mask > settings["mask_binarize_threshold"]).float()

                    if settings["mask_dilate_kernel_size"] > 0:
                        processed_mask = apply_mask_dilation(
                            processed_mask, settings["mask_dilate_kernel_size"], use_gpu
                        )
                    if settings["mask_blur_kernel_size"] > 0:
                        processed_mask = apply_gaussian_blur(processed_mask, settings["mask_blur_kernel_size"], use_gpu)

                    if settings["shadow_shift"] > 0:
                        processed_mask = apply_shadow_blur(
                            processed_mask,
                            settings["shadow_shift"],
                            settings["shadow_start_opacity"],
                            settings["shadow_opacity_decay"],
                            settings["shadow_min_opacity"],
                            settings["shadow_decay_gamma"],
                            use_gpu,
                        )

                    blended_right_eye = warped_original * (1 - processed_mask) + inpainted * processed_mask

                    # --- NEW: Apply borders from sidecar ---
                    left_border = clip_sidecar_data.get("left_border", 0.0)
                    right_border = clip_sidecar_data.get("right_border", 0.0)
                    logger.debug(f"Borders: left={left_border}%, right={right_border}%")
                    if settings.get("add_borders", True) and (left_border > 0 or right_border > 0):
                        logger.debug(
                            f"Before border: original_left shape={original_left.shape}, blended_right_eye shape={blended_right_eye.shape}"
                        )
                        original_left, blended_right_eye = apply_borders_to_frames(
                            left_border, right_border, original_left, blended_right_eye
                        )
                        logger.debug(
                            f"After border: original_left shape={original_left.shape}, blended_right_eye shape={blended_right_eye.shape}"
                        )
                    # --- END NEW ---

                    # --- NEW: Assemble final frame based on output format ---
                    if output_format == "Full SBS (Left-Right)":
                        final_chunk = torch.cat([original_left, blended_right_eye], dim=3)
                    elif output_format == "Full SBS Cross-eye (Right-Left)":
                        final_chunk = torch.cat([blended_right_eye, original_left], dim=3)
                    elif output_format == "Half SBS (Left-Right)":
                        resized_left = F.interpolate(
                            original_left, size=(hires_H, hires_W // 2), mode="bilinear", align_corners=False
                        )
                        resized_right = F.interpolate(
                            blended_right_eye, size=(hires_H, hires_W // 2), mode="bilinear", align_corners=False
                        )
                        final_chunk = torch.cat([resized_left, resized_right], dim=3)
                    elif output_format == "Double SBS":
                        sbs_chunk = torch.cat([original_left, blended_right_eye], dim=3)
                        final_chunk = F.interpolate(
                            sbs_chunk, size=(hires_H * 2, hires_W * 2), mode="bilinear", align_corners=False
                        )
                    elif output_format == "Anaglyph (Red/Cyan)":
                        # Red from Left, Green/Blue from Right
                        final_chunk = torch.cat(
                            [
                                original_left[:, 0:1, :, :],  # R channel from left
                                blended_right_eye[:, 1:3, :, :],  # G, B channels from right
                            ],
                            dim=1,
                        )
                    elif output_format == "Anaglyph Half-Color":
                        # Convert left to grayscale for the red channel
                        left_gray = (
                            original_left[:, 0, :, :] * 0.299
                            + original_left[:, 1, :, :] * 0.587
                            + original_left[:, 2, :, :] * 0.114
                        )
                        left_gray = left_gray.unsqueeze(1)  # Add channel dimension back
                        final_chunk = torch.cat(
                            [
                                left_gray,  # R channel from grayscale left
                                blended_right_eye[:, 1:3, :, :],  # G, B channels from right
                            ],
                            dim=1,
                        )
                    else:
                        # Default to Right-Eye Only
                        final_chunk = blended_right_eye
                    # --- END NEW ---

                    if flip_horizontal:
                        final_chunk = torch.flip(final_chunk, dims=[3])

                    cpu_chunk = final_chunk.cpu()

                    for frame_tensor in cpu_chunk:
                        frame_np = frame_tensor.permute(1, 2, 0).numpy()
                        frame_uint16 = (np.clip(frame_np, 0.0, 1.0) * 65535.0).astype(np.uint16)
                        frame_bgr = cv2.cvtColor(frame_uint16, cv2.COLOR_RGB2BGR)
                        ffmpeg_process.stdin.write(frame_bgr.tobytes())

                    # --- NEW: Draw console progress bar for the current video's chunks ---
                    draw_progress_bar(frame_end, num_frames, prefix=f"  Encoding {base_name}:")

                # 5. Finalize FFmpeg process
                if ffmpeg_process.stdin:
                    ffmpeg_process.stdin.close()

                # --- FIX: Wait for the process to finish FIRST, then join threads ---
                ffmpeg_process.wait(timeout=120)  # Wait for ffmpeg to exit
                stdout_thread.join(timeout=5)  # Wait for stdout reader to finish
                stderr_thread.join(timeout=5)  # Wait for stderr reader to finish
                # --- END FIX ---

                if ffmpeg_process.returncode != 0:
                    logger.error(f"FFmpeg encoding failed for {base_name}. Check console for details.")
                elif self.stop_event.is_set():
                    logger.warning(f"Processing was stopped for {base_name}. Source files will not be moved.")
                    # Do not queue files for moving if the job was stopped.
                else:
                    logger.debug("FFmpeg process and threads terminated, proceeding to move files.")
                    logger.info(f"Successfully encoded video to {output_path}")

                    # Explicitly close video readers BEFORE attempting to move their files
                    del ffmpeg_process
                    if inpainted_reader:
                        del inpainted_reader
                    if splatted_reader:
                        del splatted_reader
                    if original_reader:
                        del original_reader
                    inpainted_reader, splatted_reader, original_reader = (None, None, None)
                    time.sleep(0.1)  # Give OS a moment to release file handles
                    logger.debug("Source video file handles released.")

                    # --- NEW: Move files to finished folder if Resume is enabled ---
                    if settings.get("resume", False):
                        self.cleanup_queue.put((inpainted_video_path, settings["inpainted_folder"]))
                        self.cleanup_queue.put((splatted_file_path, settings["mask_folder"]))
                        if original_video_path_to_move:
                            self.cleanup_queue.put((original_video_path_to_move, settings["original_folder"]))
                            # Also move sidecar for original video
                            original_base = os.path.splitext(original_video_path_to_move)[0]
                            for ext in [".fssidecar", ".json"]:
                                sidecar_path = original_base + ext
                                if os.path.exists(sidecar_path):
                                    self.cleanup_queue.put((sidecar_path, settings["original_folder"]))
                        # Also move sidecar if it exists
                        inpainted_base = os.path.splitext(inpainted_video_path)[0]
                        for ext in [".fssidecar", ".json"]:
                            sidecar_path = inpainted_base + ext
                            if os.path.exists(sidecar_path):
                                self.cleanup_queue.put((sidecar_path, settings["inpainted_folder"]))

                        # In single mode, reset video list scan after moving files
                        if single_mode:
                            self.after(
                                0, lambda: getattr(self, "previewer", None) and self.previewer.reset_video_list_scan()
                            )
                    # --- END NEW ---
            except Exception as e:
                # --- FIX: Ensure readers are closed on exception before the finally block ---
                if splatted_reader:
                    del splatted_reader
                if original_reader:
                    del original_reader
                inpainted_reader, splatted_reader, original_reader = None, None, None
                # --- END FIX ---
                logger.error(f"Failed to process {base_name}: {e}", exc_info=True)
                self.after(
                    0,
                    lambda base_name=base_name, e=e: messagebox.showerror(
                        "Processing Error", f"An error occurred while processing {base_name}:\n\n{e}"
                    ),
                )
                # --- NEW: Stop the entire batch if one video fails critically ---
                self.stop_event.set()
                # --- END NEW ---
            finally:
                # Ensure readers are always cleaned up, even on error
                # This is now a secondary safety net; the primary cleanup happens before file moves.
                if inpainted_reader:
                    del inpainted_reader
                if splatted_reader:
                    del splatted_reader
                if original_reader:
                    del original_reader
                # --- END: CHUNK-BASED PROCESSING ---

            self.after(0, self.progress_var.set, i + 1)

        # --- NEW: Signal the cleanup worker to stop after it finishes its queue ---
        self.cleanup_queue.put(None)
        logger.info("Main processing loop finished. Stop signal sent to cleanup worker.")

        self.after(0, self.processing_done, self.stop_event.is_set())

    def restore_finished_files(self):
        """Moves all files from 'finished' subfolders back to their parent directories."""
        if not messagebox.askyesno(
            "Restore Finished Files",
            "Are you sure you want to move all processed videos from the 'finished' folders back to their respective input directories?",
        ):
            return

        restore_dirs = [
            ("Inpainted", self.inpainted_folder_var.get()),
            ("Original", self.original_folder_var.get()),
            ("Mask", self.mask_folder_var.get()),
        ]

        # Filter valid directories
        valid_restore_dirs = []
        for folder_name, base_folder in restore_dirs:
            if base_folder and os.path.isdir(base_folder):
                valid_restore_dirs.append((base_folder, "finished"))
            else:
                logger.warning(
                    f"Skipping restore for '{folder_name}' folder: Path is not a valid directory ('{base_folder}')."
                )

        if not valid_restore_dirs:
            messagebox.showinfo("Restore Complete", "No valid folders to restore.")
            return

        restored_count, error_count, failed_files = _restore_finished_files(
            restore_dirs=valid_restore_dirs, logger=logger
        )

        messagebox.showinfo(
            "Restore Complete",
            f"Restore operation finished.\n\nFiles Restored: {restored_count}\nErrors: {error_count}",
        )
        self.update_status_label(f"Restore complete. Moved {restored_count} files with {error_count} errors.")

        # --- NEW: Reset video list scan flag and refresh preview ---
        if hasattr(self, "previewer"):
            self.previewer.reset_video_list_scan()
            # Trigger a full refresh scan
            if self.previewer.find_sources_callback:
                self.previewer.load_video_list(find_sources_callback=self.previewer.find_sources_callback)
        # --- END NEW ---

    def _find_preview_sources_callback(self) -> list:
        """
        A callback function for the VideoPreviewer.
        It scans the folders and returns a list of dictionaries,
        where each dictionary contains the paths to the source files for one video.
        """
        inpainted_folder = self.inpainted_folder_var.get()
        if not os.path.isdir(inpainted_folder):
            messagebox.showerror("Error", "Inpainted Video Folder is not a valid directory.")
            return []

        all_mp4s = sorted(glob.glob(os.path.join(inpainted_folder, "*.mp4")))
        inpaint_pattern = re.compile(r"_inpainted_(right_eye|sbs)F?\.mp4$")
        valid_inpainted_videos = [f for f in all_mp4s if inpaint_pattern.search(f)]

        video_source_list = []
        self._clear_border_info()  # Clear border info before scanning

        for inpainted_path in valid_inpainted_videos:
            base_name = os.path.basename(inpainted_path)
            inpaint_suffix_reg = r"_inpainted_right_eyeF?\.mp4$"
            sbs_suffix_reg = r"_inpainted_sbsF?\.mp4$"

            is_sbs_input = bool(re.search(sbs_suffix_reg, base_name))
            match = re.search(sbs_suffix_reg if is_sbs_input else inpaint_suffix_reg, base_name)
            if not match:
                continue
            suffix_to_remove = match.group(0)
            core_name_with_width = base_name[: -len(suffix_to_remove)]

            last_underscore_idx = core_name_with_width.rfind("_")
            if last_underscore_idx == -1:
                logger.warning(
                    f"Preview Scan: Skipping '{base_name}'. Could not determine core name (expected format '..._width_suffix.mp4')."
                )
                continue
            core_name = core_name_with_width[:last_underscore_idx]

            # --- NEW: Read sidecar file for this clip ---
            clip_sidecar_data = self._read_clip_sidecar(inpainted_path, core_name)
            logger.debug(
                f"Preview Scan: Sidecar for '{core_name}': convergence_plane={clip_sidecar_data.get('convergence_plane')}, max_disparity={clip_sidecar_data.get('max_disparity')}, left_border={clip_sidecar_data.get('left_border')}, right_border={clip_sidecar_data.get('right_border')}"
            )
            left_border = clip_sidecar_data.get("left_border", 0.0)
            right_border = clip_sidecar_data.get("right_border", 0.0)
            self._update_border_info(left_border, right_border)
            # --- END NEW ---

            mask_folder = self.mask_folder_var.get()
            splatted4_pattern = os.path.join(mask_folder, f"{core_name}_*_splatted4*.mp4")
            splatted2_pattern = os.path.join(mask_folder, f"{core_name}_*_splatted2*.mp4")
            logger.debug(
                f"  - Searching for splatted file with patterns: '{splatted4_pattern}' and '{splatted2_pattern}'"
            )
            splatted4_matches = glob.glob(splatted4_pattern)
            splatted2_matches = glob.glob(splatted2_pattern)

            source_dict = {
                "source_video": inpainted_path,  # Primary key for VideoPreviewer
                "inpainted": inpainted_path,
                "splatted": None,
                "original": None,
                "is_sbs_input": is_sbs_input,
                "is_quad_input": False,
                "sidecar": clip_sidecar_data,  # Store sidecar data for borders
            }

            if splatted4_matches:
                splatted_path = splatted4_matches[0]
                logger.debug(f"  - Found quad-splatted match: {os.path.basename(splatted_path)}")
                source_dict["splatted"] = splatted_path
                source_dict["is_quad_input"] = True  # Set flag for quad-splatted input
                # 'original' remains None, which is the necessary structural fix for the crash
            elif splatted2_matches:
                splatted_path = splatted2_matches[0]
                logger.debug(f"  - Found dual-splatted match: {os.path.basename(splatted_path)}")
                source_dict["splatted"] = splatted_path
                original_path = self._find_video_by_core_name(self.original_folder_var.get(), core_name)

                if original_path:
                    logger.debug(f"  - Found matching original video: {os.path.basename(original_path)}")
                    source_dict["original"] = original_path
                else:
                    logger.warning(
                        f"  - For dual-splatted input '{base_name}', the original video '{core_name}.*' was not found. It will be treated as optional."
                    )
            else:
                logger.warning(
                    f"Preview Scan: Skipping '{base_name}'. No matching splatted file found in '{mask_folder}'."
                )
                continue  # Skip to the next video if no splatted file is found

            video_source_list.append(source_dict)
        return video_source_list

    def _preview_processing_callback(self, source_frames: dict, params: dict) -> Optional[Image.Image]:
        """
        This function contains the actual blending logic for the preview.
        It's called by the VideoPreviewer module.
        """
        try:
            # --- FIX: Always get the latest parameters when the preview is updated ---
            # This ensures that changing the preview source uses the current slider values.
            params = self.get_current_settings()
            if not params:
                return None  # Exit if settings are invalid
            # --- END FIX ---

            # 1. Extract raw tensors from the source_frames dict
            inpainted_tensor_full = source_frames.get("inpainted")
            splatted_tensor_raw = source_frames.get("splatted")
            original_tensor = source_frames.get("original")  # Will be None for quad input

            if inpainted_tensor_full is None or splatted_tensor_raw is None:
                raise ValueError("Missing 'inpainted' or 'splatted' source for preview.")

            # 2. Determine input type and metadata
            current_source_metadata = self.previewer.video_list[self.previewer.current_video_index]
            is_sbs_input = current_source_metadata.get("is_sbs_input", False)
            is_quad_input = current_source_metadata.get("is_quad_input", False)

            # Get flip flag from sidecar or filename
            sidecar_data = current_source_metadata.get("sidecar", {})
            flip_horizontal = sidecar_data.get("flip_horizontal", False)
            if not flip_horizontal:
                # Check for "F" suffix in inpainted filename as fallback
                inpainted_path = current_source_metadata.get("inpainted", "")
                if inpainted_path and os.path.splitext(os.path.basename(inpainted_path))[0].endswith("F"):
                    flip_horizontal = True

            # Define the processing device based on the 'use_gpu' parameter
            use_gpu = params.get("use_gpu", False) and torch.cuda.is_available()
            device = "cuda" if use_gpu else "cpu"

            # 3. Extract frame parts and move to device
            # Extract Inpainted Right Eye
            inpainted = (
                inpainted_tensor_full[:, :, :, inpainted_tensor_full.shape[3] // 2 :]
                if is_sbs_input
                else inpainted_tensor_full
            ).to(device)

            splatted_tensor = splatted_tensor_raw.to(device)
            _, _, H, W = splatted_tensor.shape

            if is_quad_input:
                half_h, half_w = H // 2, W // 2
                original_left = splatted_tensor[:, :, :half_h, :half_w]
                depth_map_vis = splatted_tensor[:, :, :half_h, half_w:]
                mask_raw = splatted_tensor[:, :, half_h:, :half_w]
                right_eye_original = splatted_tensor[:, :, half_h:, half_w:]
                is_dual_input = False
            else:
                half_w = W // 2
                mask_raw = splatted_tensor[:, :, :, :half_w]
                right_eye_original = splatted_tensor[:, :, :, half_w:]
                original_left = original_tensor.to(device) if original_tensor is not None else None
                depth_map_vis = None
                is_dual_input = True

            if flip_horizontal and is_dual_input:
                # In Dual mode, original_left is the raw video (not flipped).
                # We flip it to align with the flipped inpainted/splatted tensors.
                original_left = torch.flip(original_left, dims=[3])
            # (In Quad mode, original_left is sliced from the splatted tensor and is already flipped)

            # Configure preview source dropdown based on input type
            preview_options = [
                "Blended Image",
                "Original (Left Eye)",
                "Warped (Right BG)",
                "Inpainted Right Eye",
                "Processed Mask",
                "Anaglyph 3D",
                "Dubois Anaglyph",
                "Optimized Anaglyph",
                "Side-by-Side",
                "Wigglegram",
            ]
            preview_options.extend(["Depth Map", "Mesh Warp"])
            self.previewer.set_preview_source_options(preview_options)

            # Convert mask to grayscale ON DEVICE
            mask = torch.mean(mask_raw, dim=1, keepdim=True)

            hires_H, hires_W = right_eye_original.shape[2], right_eye_original.shape[3]

            # --- Optimized Resizing ---
            if inpainted.shape[2] != hires_H or inpainted.shape[3] != hires_W:
                target_aspect = hires_W / hires_H
                inpaint_aspect = inpainted.shape[3] / inpainted.shape[2]

                if abs(inpaint_aspect - target_aspect) > 0.01:
                    if inpaint_aspect > target_aspect:
                        new_w = hires_W
                        new_h = int(round(hires_W / inpaint_aspect))
                    else:
                        new_h = hires_H
                        new_w = int(round(hires_H * inpaint_aspect))
                    inpainted = F.interpolate(inpainted, size=(new_h, new_w), mode="bicubic", align_corners=False)
                    mask = F.interpolate(mask, size=(new_h, new_w), mode="bilinear", align_corners=False)

                # Final resize to target dimensions
                inpainted = F.interpolate(inpainted, size=(hires_H, hires_W), mode="bicubic", align_corners=False)
                mask = F.interpolate(mask, size=(hires_H, hires_W), mode="bilinear", align_corners=False)

            # ... (Mask processing) ...
            processed_mask = mask.clone()
            if params.get("mask_binarize_threshold", -1.0) >= 0.0:
                processed_mask = (processed_mask > params["mask_binarize_threshold"]).float()

            if params.get("mask_dilate_kernel_size", 0) > 0:
                processed_mask = apply_mask_dilation(processed_mask, int(params["mask_dilate_kernel_size"]), use_gpu)

            if params.get("mask_blur_kernel_size", 0) > 0:
                processed_mask = apply_gaussian_blur(processed_mask, int(params["mask_blur_kernel_size"]), use_gpu)

            if params.get("shadow_shift", 0) > 0:
                processed_mask = apply_shadow_blur(
                    processed_mask,
                    int(params["shadow_shift"]),
                    params["shadow_start_opacity"],
                    params["shadow_opacity_decay"],
                    params["shadow_min_opacity"],
                    params["shadow_decay_gamma"],
                    use_gpu,
                )

            # Color Transfer
            if params.get("enable_color_transfer", False):
                if original_left is not None:
                    # Color transfer returns 3D, we must unsqueeze to keep it 4D
                    inpainted = apply_color_transfer(original_left.cpu(), inpainted.cpu()).unsqueeze(0).to(device)

            blended_frame = right_eye_original * (1 - processed_mask) + inpainted * processed_mask

            # Borders
            left_border = sidecar_data.get("left_border", 0.0)
            right_border = sidecar_data.get("right_border", 0.0)

            if self.add_borders_var.get() and (left_border > 0 or right_border > 0):
                if original_left is not None:
                    original_left, blended_frame = apply_borders_to_frames(
                        left_border, right_border, original_left, blended_frame
                    )
                else:
                    _, blended_frame = apply_borders_to_frames(left_border, right_border, blended_frame, blended_frame)

            # ... (Preview selection logic) ...
            preview_source = self.preview_source_var.get()
            final_frame_4d = None

            if preview_source == "Blended Image":
                final_frame_4d = blended_frame
            elif preview_source == "Inpainted Right Eye":
                final_frame_4d = inpainted
            elif preview_source == "Original (Left Eye)":
                final_frame_4d = original_left if original_left is not None else torch.zeros_like(blended_frame)
            elif preview_source == "Warped (Right BG)":
                final_frame_4d = right_eye_original
            elif preview_source == "Processed Mask":
                final_frame_4d = processed_mask.repeat(1, 3, 1, 1)
            elif preview_source == "Anaglyph 3D":
                if original_left is not None:
                    # Red channel from grayscale left, Green/Blue from right
                    left_gray = (
                        original_left[:, 0:1, :, :] * 0.299
                        + original_left[:, 1:2, :, :] * 0.587
                        + original_left[:, 2:3, :, :] * 0.114
                    )
                    final_frame_4d = torch.cat(
                        [left_gray, blended_frame[:, 1:2, :, :], blended_frame[:, 2:3, :, :]], dim=1
                    )
                else:
                    final_frame_4d = blended_frame
            elif preview_source == "Dubois Anaglyph":
                if original_left is not None:
                    final_frame_4d = apply_dubois_anaglyph_torch(original_left, blended_frame)
                else:
                    final_frame_4d = blended_frame
            elif preview_source == "Optimized Anaglyph":
                if original_left is not None:
                    final_frame_4d = apply_optimized_anaglyph_torch(original_left, blended_frame)
                else:
                    final_frame_4d = blended_frame
            elif preview_source == "Side-by-Side":
                if original_left is not None:
                    left_np = (original_left[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    right_np = (blended_frame[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    if getattr(self.previewer, "sbs_cross_eye_var", None) and self.previewer.sbs_cross_eye_var.get():
                        sbs_np = np.concatenate([right_np, left_np], axis=1)
                    else:
                        sbs_np = np.concatenate([left_np, right_np], axis=1)
                    return Image.fromarray(sbs_np)
                else:
                    final_frame_4d = blended_frame
            elif preview_source == "Wigglegram":
                if original_left is not None:
                    self.previewer._start_wigglegram_animation(original_left, blended_frame)
                return None
            elif preview_source == "Depth Map" and depth_map_vis is not None:
                final_frame_4d = depth_map_vis
            elif preview_source == "Mesh Warp" and depth_map_vis is not None:
                from core.common.mesh_warp import run_fusion_stereo

                left_np = (original_left.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                left_bgr = cv2.cvtColor(left_np, cv2.COLOR_RGB2BGR)
                depth_np = depth_map_vis.squeeze(0).mean(dim=0).cpu().numpy()

                l_img, r_img = run_fusion_stereo(
                    image=left_bgr,
                    depth=depth_np,
                    disparity=float(sidecar_data.get("max_disparity", 25.0)),
                    convergence=1.0 - float(sidecar_data.get("convergence_plane", 0.5)),
                    view_bias=float(self.APP_DEFAULTS.get("MESH_WARP_VIEW_BIAS", -1.0)),
                    dolly_zoom=float(self.APP_DEFAULTS.get("MESH_WARP_DOLLY_ZOOM", 0.0)),
                    extrusion_scale=float(self.APP_DEFAULTS.get("MESH_WARP_EXTRUSION_SCALE", 1.0)),
                    density_x=int(left_bgr.shape[1]),
                    density_y=int(left_bgr.shape[0]),
                )

                if getattr(self.previewer, "sbs_cross_eye_var", None) and self.previewer.sbs_cross_eye_var.get():
                    sbs_np = np.concatenate([r_img, l_img], axis=1)
                else:
                    sbs_np = np.concatenate([l_img, r_img], axis=1)

                sbs_rgb = cv2.cvtColor(sbs_np, cv2.COLOR_BGR2RGB)
                return Image.fromarray(sbs_rgb)
            else:
                final_frame_4d = blended_frame

            if final_frame_4d is None:
                final_frame_4d = blended_frame

            if flip_horizontal:
                final_frame_4d = torch.flip(final_frame_4d, dims=[3])

            # Store for saving SBS (CPU side)
            if flip_horizontal:
                # When unflipping an SBS [L, R], the result is [flip(R), flip(L)]
                self.preview_original_left_tensor = torch.flip(blended_frame, dims=[3]).squeeze(0).cpu()
                if original_left is not None:
                    self.preview_blended_right_tensor = torch.flip(original_left, dims=[3]).squeeze(0).cpu()
                else:
                    self.preview_blended_right_tensor = torch.zeros_like(blended_frame).squeeze(0).cpu()
            else:
                if original_left is not None:
                    self.preview_original_left_tensor = original_left.squeeze(0).cpu()
                else:
                    self.preview_original_left_tensor = torch.zeros_like(blended_frame).squeeze(0).cpu()
                self.preview_blended_right_tensor = blended_frame.squeeze(0).cpu()

            # 5. Convert to PIL Image
            final_uint8 = (final_frame_4d[0].permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8)
            return Image.fromarray(final_uint8.cpu().numpy())

        except Exception as e:
            logger.error(f"Error in preview processing callback: {e}", exc_info=True)
            return None

    def save_config(self):
        """Gathers current settings and saves them to the config file."""
        config = self.get_current_settings()
        if config:
            # Add window geometry and other non-processing settings to the config dictionary
            config["window_x"] = self.winfo_x()
            config["window_y"] = self.winfo_y()
            config["window_width"] = self.winfo_width()
            config["window_height"] = self.winfo_height()
            config["debug_logging_enabled"] = self.debug_logging_var.get()
            config["dark_mode_enabled"] = self.dark_mode_var.get()

            try:
                with open("config_merging.mergecfg", "w") as f:
                    json.dump(config, f, indent=4)
                logger.info("Merging GUI configuration saved.")
            except Exception as e:
                logger.error(f"Failed to save merging GUI config: {e}")

    def _load_config(self):
        """Loads configuration from a JSON file."""
        try:
            with open("config_merging.mergecfg", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.error(f"Failed to load merging GUI config: {e}")
            return {}

    def load_settings_dialog(self):
        """Loads settings from a user-selected JSON file."""
        filepath = filedialog.askopenfilename(
            defaultextension=".mergecfg",
            filetypes=[("Merge Config Files", "*.mergecfg"), ("All files", "*.*")],
            title="Load Settings from File",
        )
        if not filepath:
            return
        try:
            with open(filepath, "r") as f:
                settings_to_load = json.load(f)

            self._apply_settings(settings_to_load)
            self._apply_theme()
            logger.info(f"Settings loaded from {filepath}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load settings from {filepath}:\n{e}")

    def save_settings_dialog(self):
        """Saves current GUI settings to a user-selected JSON file."""
        config_to_save = self.get_current_settings()
        if not config_to_save:
            return  # get_current_settings failed validation

        filepath = filedialog.asksaveasfilename(
            defaultextension=".mergecfg",
            filetypes=[("Merge Config Files", "*.mergecfg"), ("All files", "*.*")],
            title="Save Settings to File",
        )
        if not filepath:
            return
        try:
            with open(filepath, "w") as f:
                json.dump(config_to_save, f, indent=4)
            logger.info(f"Settings saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save settings to {filepath}:\n{e}")

    def _save_preview_sbs_frame(self):
        """Saves the current preview as a full side-by-side image."""
        if self.preview_original_left_tensor is None or self.preview_blended_right_tensor is None:
            messagebox.showwarning(
                "No Preview Data", "There is no preview data to save. Please load and preview a video first."
            )
            return

        try:
            # Convert tensors to PIL Images
            left_np = (self.preview_original_left_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            right_np = (self.preview_blended_right_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            left_pil = Image.fromarray(left_np)
            right_pil = Image.fromarray(right_np)

            # Check if dimensions match
            if left_pil.size != right_pil.size:
                messagebox.showerror(
                    "Dimension Mismatch",
                    "The left and right eye images have different dimensions. Cannot create SBS image.",
                )
                return

            # Create SBS image
            width, height = left_pil.size
            sbs_image = Image.new("RGB", (width * 2, height))
            sbs_image.paste(left_pil, (0, 0))
            sbs_image.paste(right_pil, (width, 0))

            # Suggest a default filename
            default_filename = "preview_sbs_frame.png"
            if self.previewer.current_video_index != -1:
                source_paths = self.previewer.video_list[self.previewer.current_video_index]
                base_name = os.path.splitext(os.path.basename(next(iter(source_paths.values()))))[0]
                frame_num = int(self.previewer.frame_scrubber_var.get())
                default_filename = f"{base_name}_frame_{frame_num:05d}_SBS.png"

            filepath = filedialog.asksaveasfilename(
                title="Save SBS Preview Frame As...",
                initialfile=default_filename,
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")],
            )

            if filepath:
                sbs_image.save(filepath)
                logger.info(f"SBS preview frame saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save SBS preview frame: {e}", exc_info=True)
            messagebox.showerror("Save Error", f"An error occurred while creating or saving the SBS image:\n{e}")

    def exit_application(self):
        """Handles application exit gracefully."""
        if self.is_processing:
            if messagebox.askyesno(
                "Confirm Exit", "Processing is in progress. Are you sure you want to stop and exit?"
            ):
                self.stop_processing()
                self.previewer.cleanup()
                self.save_config()
                self.destroy()
        else:
            self.save_config()
            self.previewer.cleanup()
            self.destroy()


if __name__ == "__main__":
    # Basic logging setup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    app = MergingGUI()
    app.mainloop()
