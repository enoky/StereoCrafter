import os
import glob
import json
import shutil
import threading
import gc
import tkinter as tk # Used for PanedWindow
from tkinter import filedialog, messagebox, ttk
from ttkthemes import ThemedTk
from typing import Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageTk
from decord import VideoReader, cpu
import logging
import time
from dependency.stereocrafter_util import Tooltip, logger, get_video_stream_info, draw_progress_bar, release_cuda_memory, set_util_logger_level, encode_frames_to_mp4, read_video_frames_decord, start_ffmpeg_pipe_process

GUI_VERSION = "25-10-11"

# --- MASK PROCESSING FUNCTIONS (from test.py) ---
def apply_mask_dilation(mask: torch.Tensor, kernel_size: int, use_gpu: bool = True) -> torch.Tensor:
    if kernel_size <= 0: return mask
    kernel_val = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    if use_gpu:
        padding = kernel_val // 2
        return F.max_pool2d(mask, kernel_size=kernel_val, stride=1, padding=padding)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_val, kernel_val))
        processed_frames = []
        for t in range(mask.shape[0]):
            frame_np = (mask[t].squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            dilated_np = cv2.dilate(frame_np, kernel, iterations=1)
            dilated_tensor = torch.from_numpy(dilated_np).float() / 255.0
            processed_frames.append(dilated_tensor.unsqueeze(0))
        return torch.stack(processed_frames).to(mask.device)

def apply_gaussian_blur(mask: torch.Tensor, kernel_size: int, use_gpu: bool = True) -> torch.Tensor:
    if kernel_size <= 0: return mask
    kernel_val = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    if use_gpu:
        sigma = kernel_val / 6.0
        ax = torch.arange(-kernel_val // 2 + 1., kernel_val // 2 + 1., device=mask.device)
        gauss = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        kernel_1d = (gauss / gauss.sum()).view(1, 1, 1, kernel_val)
        blurred_mask = F.conv2d(mask, kernel_1d, padding=(0, kernel_val // 2), groups=mask.shape[1])
        blurred_mask = F.conv2d(blurred_mask, kernel_1d.permute(0, 1, 3, 2), padding=(kernel_val // 2, 0), groups=mask.shape[1])
        return torch.clamp(blurred_mask, 0.0, 1.0)
    else:
        processed_frames = []
        for t in range(mask.shape[0]):
            frame_np = (mask[t].squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            blurred_np = cv2.GaussianBlur(frame_np, (kernel_val, kernel_val), 0)
            blurred_tensor = torch.from_numpy(blurred_np).float() / 255.0
            processed_frames.append(blurred_tensor.unsqueeze(0))
        return torch.stack(processed_frames).to(mask.device)

def apply_shadow_blur(mask: torch.Tensor, shift_per_step: int, start_opacity: float, opacity_decay_per_step: float, min_opacity: float, decay_gamma: float = 1.0, use_gpu: bool = True) -> torch.Tensor:
    if shift_per_step <= 0: return mask
    num_steps = int((start_opacity - min_opacity) / opacity_decay_per_step) + 1
    if num_steps <= 0: return mask

    if use_gpu:
        canvas_mask = mask.clone()
        stamp_source = mask.clone()
        for i in range(num_steps):
            t = 1.0 - (i / (num_steps - 1)) if num_steps > 1 else 1.0
            curved_t = t ** decay_gamma
            current_opacity = min_opacity + (start_opacity - min_opacity) * curved_t
            total_shift = (i + 1) * shift_per_step
            padded_stamp = F.pad(stamp_source, (total_shift, 0), "constant", 0)
            shifted_stamp = padded_stamp[:, :, :, :-total_shift]
            canvas_mask = torch.max(canvas_mask, shifted_stamp * current_opacity)
        return canvas_mask
    else:
        processed_frames = []
        for t in range(mask.shape[0]):
            canvas_np = mask[t].squeeze(0).cpu().numpy() # Process one frame at a time
            stamp_source_np = canvas_np.copy()
            for i in range(num_steps):
                time_step = 1.0 - (i / (num_steps - 1)) if num_steps > 1 else 1.0
                curved_t = time_step ** decay_gamma
                current_opacity = min_opacity + (start_opacity - min_opacity) * curved_t
                total_shift = (i + 1) * shift_per_step
                shifted_stamp = np.roll(stamp_source_np, total_shift, axis=1) # axis=1 for HxW
                canvas_np = np.maximum(canvas_np, shifted_stamp * current_opacity)
            processed_frames.append(torch.from_numpy(canvas_np).unsqueeze(0))
        return torch.stack(processed_frames).to(mask.device)

def apply_color_transfer(source_frame: torch.Tensor, target_frame: torch.Tensor) -> torch.Tensor:
    """
    Transfers the color statistics from the source_frame to the target_frame using LAB color space.
    Expects source_frame and target_frame in [C, H, W] float [0, 1] format on CPU.
    Returns the color-adjusted target_frame in [C, H, W] float [0, 1] format.
    """
    try:
        # Ensure tensors are on CPU and convert to numpy arrays in HWC format
        source_np = source_frame.permute(1, 2, 0).numpy()  # [H, W, C]
        target_np = target_frame.permute(1, 2, 0).numpy()  # [H, W, C]

        # Scale from [0, 1] to [0, 255] and convert to uint8
        source_np_uint8 = (np.clip(source_np, 0.0, 1.0) * 255).astype(np.uint8)
        target_np_uint8 = (np.clip(target_np, 0.0, 1.0) * 255).astype(np.uint8)

        # Convert to LAB color space
        source_lab = cv2.cvtColor(source_np_uint8, cv2.COLOR_RGB2LAB)
        target_lab = cv2.cvtColor(target_np_uint8, cv2.COLOR_RGB2LAB)

        src_mean, src_std = cv2.meanStdDev(source_lab)
        tgt_mean, tgt_std = cv2.meanStdDev(target_lab)

        src_mean = src_mean.flatten()
        src_std = src_std.flatten()
        tgt_mean = tgt_mean.flatten()
        tgt_std = tgt_std.flatten()

        src_std = np.clip(src_std, 1e-6, None)
        tgt_std = np.clip(tgt_std, 1e-6, None)

        target_lab_float = target_lab.astype(np.float32)
        for i in range(3): # For L, A, B channels
            target_lab_float[:, :, i] = (target_lab_float[:, :, i] - tgt_mean[i]) / tgt_std[i] * src_std[i] + src_mean[i]

        adjusted_lab_uint8 = np.clip(target_lab_float, 0, 255).astype(np.uint8)
        adjusted_rgb = cv2.cvtColor(adjusted_lab_uint8, cv2.COLOR_LAB2RGB)
        return torch.from_numpy(adjusted_rgb).permute(2, 0, 1).float() / 255.0
    except Exception as e:
        logger.error(f"Error during color transfer: {e}. Returning original target frame.", exc_info=True)
        return target_frame

class MergingGUI(ThemedTk):
    def __init__(self):
        super().__init__(theme="clam")
        self.title(f"Stereocrafter Merging GUI {GUI_VERSION}")
        self.app_config = self._load_config()
        self.help_data = self._load_help_texts()

        # --- Window Geometry ---
        self.window_x = self.app_config.get("window_x", None)
        self.window_y = self.app_config.get("window_y", None)
        self.window_width = self.app_config.get("window_width", 700) # A reasonable default

        # --- Core App State ---
        self.stop_event = threading.Event()
        self.is_processing = False
        
        # --- Preview State ---
        # Store VideoReader objects instead of full numpy arrays to save memory
        self.preview_inpainted_reader = None
        self.preview_original_reader = None
        self.preview_splatted_reader = None
        self.preview_is_splatted_dual = False
        self.wiggle_after_id = None # For managing the wigglegram animation

        self.preview_video_list = []
        self.current_preview_video_index = -1
        self.is_sbs_preview = False # NEW: To track if the preview source is an SBS file
        self.preview_mask_frame = None
        self.preview_image_tk = None # To prevent garbage collection
        self.preview_loaded_file = None

        self.preview_original_left_tensor = None
        self.preview_blended_right_tensor = None
        # --- GUI Variables ---
        self.pil_image_for_preview = None # Store the PIL image for resizing
        self.inpainted_folder_var = tk.StringVar(value=self.app_config.get("inpainted_folder", "./completed_output"))
        self.original_folder_var = tk.StringVar(value=self.app_config.get("original_folder", "./input_source_clips"))
        self.mask_folder_var = tk.StringVar(value=self.app_config.get("mask_folder", "./output_splatted/hires")) # Assuming masks are in splatted folder
        self.output_folder_var = tk.StringVar(value=self.app_config.get("output_folder", "./final_videos"))

        # --- Mask Processing Parameters ---
        self.mask_binarize_threshold_var = tk.DoubleVar(value=float(self.app_config.get("mask_binarize_threshold", 0.30)))
        self.mask_dilate_kernel_size_var = tk.DoubleVar(value=float(self.app_config.get("mask_dilate_kernel_size", 3)))
        self.mask_blur_kernel_size_var = tk.DoubleVar(value=float(self.app_config.get("mask_blur_kernel_size", 5)))
        self.shadow_shift_var = tk.DoubleVar(value=float(self.app_config.get("shadow_shift", 5)))
        self.shadow_decay_gamma_var = tk.DoubleVar(value=float(self.app_config.get("shadow_decay_gamma", 1.3)))
        self.shadow_start_opacity_var = tk.DoubleVar(value=float(self.app_config.get("shadow_start_opacity", 0.87)))
        self.shadow_opacity_decay_var = tk.DoubleVar(value=float(self.app_config.get("shadow_opacity_decay", 0.08)))
        self.shadow_min_opacity_var = tk.DoubleVar(value=float(self.app_config.get("shadow_min_opacity", 0.14)))

        self.use_gpu_var = tk.BooleanVar(value=self.app_config.get("use_gpu", False))
        self.output_format_var = tk.StringVar(value=self.app_config.get("output_format", "Full SBS (Left-Right)"))
        self.pad_to_16_9_var = tk.BooleanVar(value=self.app_config.get("pad_to_16_9", False))
        self.enable_color_transfer_var = tk.BooleanVar(value=self.app_config.get("enable_color_transfer", True))
        self.debug_logging_var = tk.BooleanVar(value=self.app_config.get("debug_logging_enabled", False))
        self.dark_mode_var = tk.BooleanVar(value=self.app_config.get("dark_mode_enabled", False))
        self.batch_chunk_size_var = tk.StringVar(value=str(self.app_config.get("batch_chunk_size", 20)))
        self.frame_scrubber_var = tk.DoubleVar(value=0)
        self.video_jump_to_var = tk.StringVar(value="1")
        self.video_status_label_var = tk.StringVar(value="Video: 0 / 0")
        self.preview_source_var = tk.StringVar(value="Blended Image")
        self.frame_label_var = tk.StringVar(value="Frame: 0 / 0")
        self.preview_size_var = tk.StringVar(value=str(self.app_config.get("preview_size", 1024)))

        # --- GUI Status Variables ---
        self.status_label_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0)

        self.create_widgets()

        # Define a custom style for the loading button
        self.style = ttk.Style(self)
        self.style.configure('Loading.TButton', foreground='red', font=('Helvetica', '9', 'bold'))

        self._apply_theme()
        self._configure_logging() # Set initial logging level
        self.after(0, self._set_saved_geometry) # Restore window position
        self.protocol("WM_DELETE_WINDOW", self.exit_application)
        self.update_status_label("Ready.")

    def _set_saved_geometry(self):
        """
        Applies the saved window width and position, with dynamic height.
        This is more robust than the simpler _refresh_window_geometry.
        """
        logger.debug("--- Setting Saved Geometry (Startup) ---")
        self.update_idletasks()

        # 1. Get the optimal height for the current content
        calculated_height = self.winfo_reqheight()
        logger.debug(f"  - Initial required height: {calculated_height}")
        if calculated_height < 200: # Fallback for initial state
            calculated_height = 800
            logger.debug(f"  - Height was < 200, using fallback: {calculated_height}")

        # 2. Use the saved/default width, with a fallback
        current_width = self.window_width
        logger.debug(f"  - Using saved/default width: {current_width}")
        if current_width < 500: # Minimum sensible width
            current_width = 700
            logger.debug(f"  - Width was < 500, using fallback: {current_width}")

        # 3. Construct the geometry string
        geometry_string = f"{current_width}x{calculated_height}"
        if self.window_x is not None and self.window_y is not None:
            geometry_string += f"+{self.window_x}+{self.window_y}"
            logger.debug(f"  - Using saved position: +{self.window_x}+{self.window_y}")

        # 4. Apply the geometry
        self.geometry(geometry_string)
        logger.debug(f"  - Applied geometry string: '{geometry_string}'")

        # Store the width that was actually applied for the next save operation
        self.window_width = current_width
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
        self.file_menu.add_command(label="Save Preview Frame...", command=self._save_preview_frame)
        self.file_menu.add_command(label="Save Preview as SBS...", command=self._save_preview_sbs_frame)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Reset to Default", command=self.reset_to_defaults)
        self.file_menu.add_command(label="Restore Finished Files", command=self.restore_finished_files)
        self.file_menu.add_separator()
        self.file_menu.add_checkbutton(label="Dark Mode", variable=self.dark_mode_var, command=self._apply_theme)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.exit_application)

        # --- Help Menu ---
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_checkbutton(label="Enable Debug Logging", variable=self.debug_logging_var, command=self._toggle_debug_logging)
        self.help_menu.add_separator()
        self.help_menu.add_command(label="User Guide", command=self.show_user_guide)
        self.help_menu.add_command(label="About", command=self.show_about_dialog)

    def _create_hover_tooltip(self, widget, help_key):
        """Creates a mouse-over tooltip for the given widget."""
        if help_key in self.help_data:
            Tooltip(widget, self.help_data[help_key])

    def _apply_theme(self):
        """Applies the selected theme (dark or light) to the GUI."""
        if self.dark_mode_var.get():
            bg_color, fg_color, entry_bg = "#2b2b2b", "white", "#3c3c3c"
            self.style.theme_use("black")
        else:
            bg_color, fg_color, entry_bg = "#d9d9d9", "black", "#ffffff"
            self.style.theme_use("clam")

        self.configure(bg=bg_color)
        self.style.configure("TFrame", background=bg_color)
        self.style.configure("TLabel", background=bg_color, foreground=fg_color)
        self.style.configure("TLabelframe", background=bg_color, foreground=fg_color)
        self.style.configure("TLabelframe.Label", background=bg_color, foreground=fg_color)
        self.style.configure("TCheckbutton", background=bg_color, foreground=fg_color)
        self.style.map('TCheckbutton',
            foreground=[('active', fg_color)],
            background=[('active', bg_color)]
        )
        self.style.configure("TEntry", fieldbackground=entry_bg, foreground=fg_color, insertcolor=fg_color)
        # --- NEW: Add Combobox styling ---
        self.style.map('TCombobox',
            fieldbackground=[('readonly', entry_bg)],
            foreground=[('readonly', fg_color)],
            selectbackground=[('readonly', entry_bg)],
            selectforeground=[('readonly', fg_color)]
        )
        # --- FIX: Manually set the background for the tk.Canvas widget ---
        if hasattr(self, 'preview_canvas'):
            self.preview_canvas.config(bg=bg_color, highlightthickness=0)
        # -----------------------------------------------------------------

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
            messagebox.showerror("File Not Found", f"The user guide file could not be found at:\n{os.path.abspath(guide_path)}")
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

    def reset_to_defaults(self):
        """Resets all GUI parameters to their default hardcoded values."""
        if not messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to their default values?"):
            return

        # Set default values for all your configuration variables
        self.inpainted_folder_var.set("./completed_output")
        self.original_folder_var.set("./input_source_clips")
        self.mask_folder_var.set("./output_splatted/hires")
        self.output_folder_var.set("./final_videos")
        self.mask_dilate_kernel_size_var.set(15.0)
        self.mask_blur_kernel_size_var.set(25.0)
        self.shadow_shift_var.set(2.0)
        self.shadow_start_opacity_var.set(0.8)
        self.mask_binarize_threshold_var.set(-0.01)
        self.shadow_opacity_decay_var.set(0.1)
        self.shadow_min_opacity_var.set(0.2)
        self.shadow_decay_gamma_var.set(2.0)
        self.use_gpu_var.set(True)
        self.pad_to_16_9_var.set(False)
        self.output_format_var.set("Full SBS (Left-Right)")
        self.enable_color_transfer_var.set(True)
        self.batch_chunk_size_var.set("32")
        self.preview_size_var.set("512")

        self.save_config()
        messagebox.showinfo("Settings Reset", "All settings have been reset to their default values.")
        logger.info("GUI settings reset to defaults.")

    def _configure_logging(self):
        """Sets the logging level based on the debug_logging_var."""
        if self.debug_logging_var.get():
            level = logging.DEBUG
        else:
            level = logging.INFO
        
        set_util_logger_level(level)
        logging.getLogger().setLevel(level)
        logger.info(f"Logging level set to {logging.getLevelName(level)}.")

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
        ttk.Button(folder_frame, text="Browse", command=lambda: self._browse_folder(self.inpainted_folder_var)).grid(row=0, column=2, padx=5)

        # Original Video Folder (for Left Eye)
        ttk.Label(folder_frame, text="Original Video Folder:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        entry_orig = ttk.Entry(folder_frame, textvariable=self.original_folder_var)
        entry_orig.grid(row=1, column=1, padx=5, sticky="ew")
        self._create_hover_tooltip(entry_orig, "original_folder")
        ttk.Button(folder_frame, text="Browse", command=lambda: self._browse_folder(self.original_folder_var)).grid(row=1, column=2, padx=5)

        # Mask Folder
        ttk.Label(folder_frame, text="Mask Folder:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        entry_mask = ttk.Entry(folder_frame, textvariable=self.mask_folder_var)
        entry_mask.grid(row=2, column=1, padx=5, sticky="ew")
        self._create_hover_tooltip(entry_mask, "mask_folder")
        ttk.Button(folder_frame, text="Browse", command=lambda: self._browse_folder(self.mask_folder_var)).grid(row=2, column=2, padx=5)

        # Output Folder
        ttk.Label(folder_frame, text="Output Folder:").grid(row=3, column=0, sticky="e", padx=5, pady=2)
        entry_out = ttk.Entry(folder_frame, textvariable=self.output_folder_var)
        entry_out.grid(row=3, column=1, padx=5, sticky="ew")
        self._create_hover_tooltip(entry_out, "output_folder")
        ttk.Button(folder_frame, text="Browse", command=lambda: self._browse_folder(self.output_folder_var)).grid(row=3, column=2, padx=5)

        # --- PREVIEW FRAME (Moved here) ---
        preview_container_frame = ttk.LabelFrame(self, text="Live Preview", padding=10)
        preview_container_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Create a canvas with scrollbars
        self.preview_canvas = tk.Canvas(preview_container_frame)
        v_scrollbar = ttk.Scrollbar(preview_container_frame, orient="vertical", command=self.preview_canvas.yview)
        h_scrollbar = ttk.Scrollbar(preview_container_frame, orient="horizontal", command=self.preview_canvas.xview)
        self.preview_canvas.bind("<Configure>", lambda e: self._update_preview_layout()) # Re-center on resize
        self.preview_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Grid layout for canvas and scrollbars
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        preview_container_frame.grid_rowconfigure(0, weight=1)
        preview_container_frame.grid_columnconfigure(0, weight=1)

        # Create a frame inside the canvas to hold the image label
        self.preview_inner_frame = ttk.Frame(self.preview_canvas)
        self.preview_canvas_window_id = self.preview_canvas.create_window((0, 0), window=self.preview_inner_frame, anchor="nw")
        self.preview_label = ttk.Label(self.preview_inner_frame, text="Load a video to see preview", anchor="center")
        # Store references to scrollbars to hide/show them
        self.v_scrollbar = v_scrollbar
        self.h_scrollbar = h_scrollbar

        self.preview_label.pack(fill="both", expand=True)

        # Scrubber Frame
        scrubber_frame = ttk.Frame(preview_container_frame) # Parent is preview_container_frame
        scrubber_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5) # Use grid instead of pack
        scrubber_frame.grid_columnconfigure(1, weight=1)

        self.frame_label = ttk.Label(scrubber_frame, textvariable=self.frame_label_var, width=15)
        self.frame_label.grid(row=0, column=0, padx=5)
        self.frame_scrubber = ttk.Scale(scrubber_frame, from_=0, to=0, variable=self.frame_scrubber_var, orient="horizontal")
        self.frame_scrubber.grid(row=0, column=1, sticky="ew")
        self.frame_scrubber.bind("<ButtonRelease-1>", self.on_slider_release)
        self.frame_scrubber.configure(command=self.on_scrubber_move)

        # --- Video Navigation Frame ---
        preview_button_frame = ttk.Frame(preview_container_frame) # Parent is preview_container_frame
        preview_button_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5) # Use grid instead of pack

        # Add Preview Source dropdown
        ttk.Label(preview_button_frame, text="Preview Source:").pack(side="left", padx=(0, 5))
        self.preview_source_combo = ttk.Combobox(preview_button_frame, textvariable=self.preview_source_var, state="readonly", width=18)
        self.preview_source_combo.pack(side="left", padx=5)
        self.preview_source_combo.bind("<<ComboboxSelected>>", lambda event: self.on_slider_release(event))
        self._create_hover_tooltip(self.preview_source_combo, "preview_source")
        
        self.load_preview_button = ttk.Button(preview_button_frame, text="Load/Refresh List", command=self._refresh_preview_video_list, width=20)
        self.load_preview_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.load_preview_button, "load_refresh_list")

        self.prev_video_button = ttk.Button(preview_button_frame, text="< Prev", command=lambda: self._nav_preview_video(-1))
        self.prev_video_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.prev_video_button, "prev_video")

        self.next_video_button = ttk.Button(preview_button_frame, text="Next >", command=lambda: self._nav_preview_video(1))
        self.next_video_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.next_video_button, "next_video")

        ttk.Label(preview_button_frame, text="Jump to:").pack(side="left", padx=(15, 2))
        self.video_jump_entry = ttk.Entry(preview_button_frame, textvariable=self.video_jump_to_var, width=5)
        self.video_jump_entry.pack(side="left")
        self.video_jump_entry.bind("<Return>", self._jump_to_video)
        self._create_hover_tooltip(self.video_jump_entry, "jump_to_video")
        ttk.Label(preview_button_frame, textvariable=self.video_status_label_var).pack(side="left", padx=5)

        # --- MASK PROCESSING PARAMETERS ---
        param_frame = ttk.LabelFrame(self, text="Mask Processing Parameters", padding=10)
        param_frame.pack(fill="x", padx=10, pady=5)
        param_frame.grid_columnconfigure(1, weight=1)

        # Helper to create a slider with a label
        def create_slider(parent, text, var, from_, to, row, decimals=0):
            ttk.Label(parent, text=text).grid(row=row, column=0, sticky="e", padx=5, pady=2)
            slider = ttk.Scale(parent, from_=from_, to=to, variable=var, orient="horizontal")
            slider.grid(row=row, column=1, sticky="ew", padx=5)
            value_label = ttk.Label(parent, text=f"{var.get():.{decimals}f}", width=5)
            value_label.grid(row=row, column=2, sticky="w", padx=5)

            # Update label continuously, update preview on release
            slider.configure(command=lambda v, label=value_label: label.config(text=f"{float(v):.{decimals}f}"))
            slider.bind("<ButtonRelease-1>", self.on_slider_release)
            self._create_hover_tooltip(slider, text.lower().replace(":", "").replace(" ", "_").replace(".", ""))
            return slider

        create_slider(param_frame, "Binarize Thresh (<0=Off):", self.mask_binarize_threshold_var, -0.01, 1.0, 0, decimals=2)
        create_slider(param_frame, "Dilate Kernel:", self.mask_dilate_kernel_size_var, 0, 101, 1)
        create_slider(param_frame, "Blur Kernel:", self.mask_blur_kernel_size_var, 0, 101, 2)
        create_slider(param_frame, "Shadow Shift:", self.shadow_shift_var, 0, 50, 3)
        create_slider(param_frame, "Shadow Gamma:", self.shadow_decay_gamma_var, 0.1, 5.0, 4, decimals=2)
        create_slider(param_frame, "Shadow Opacity Start:", self.shadow_start_opacity_var, 0.0, 1.0, 5, decimals=2)
        create_slider(param_frame, "Shadow Opacity Decay:", self.shadow_opacity_decay_var, 0.0, 1.0, 6, decimals=2)
        create_slider(param_frame, "Shadow Opacity Min:", self.shadow_min_opacity_var, 0.0, 1.0, 7, decimals=2)

        # --- OPTIONS FRAME ---
        options_frame = ttk.LabelFrame(self, text="Options", padding=10)
        options_frame.pack(fill="x", padx=10, pady=5)

        gpu_check = ttk.Checkbutton(options_frame, text="Use GPU for Mask Processing", variable=self.use_gpu_var)
        gpu_check.pack(side="left", padx=5)
        self._create_hover_tooltip(gpu_check, "use_gpu")

        # --- NEW: Output Format Dropdown ---
        ttk.Label(options_frame, text="Output Format:").pack(side="left", padx=(15, 5))
        output_formats = ["Full SBS (Left-Right)", "Double SBS", "Half SBS (Left-Right)", "Full SBS Cross-eye (Right-Left)", "Anaglyph (Red/Cyan)", "Anaglyph Half-Color", "Right-Eye Only"]
        output_format_combo = ttk.Combobox(options_frame, textvariable=self.output_format_var, values=output_formats, state="readonly", width=32)
        output_format_combo.pack(side="left", padx=5)
        self._create_hover_tooltip(output_format_combo, "output_format")
        # --- END NEW ---
        
        color_check = ttk.Checkbutton(options_frame, text="Enable Color Transfer", variable=self.enable_color_transfer_var)
        color_check.pack(side="left", padx=5)
        self._create_hover_tooltip(color_check, "enable_color_transfer")
        
        # --- NEW: Pad to 16:9 Checkbox ---
        pad_check = ttk.Checkbutton(options_frame, text="Pad to 16:9", variable=self.pad_to_16_9_var)
        pad_check.pack(side="left", padx=(15, 5))
        self._create_hover_tooltip(pad_check, "pad_to_16_9")
        
        # Add Preview Size option
        ttk.Label(options_frame, text="Preview Size:").pack(side="left", padx=(20, 5))
        entry_preview = ttk.Entry(options_frame, textvariable=self.preview_size_var, width=7)
        entry_preview.pack(side="left")
        self._create_hover_tooltip(entry_preview, "preview_size")
        # Add Batch Chunk Size option
        ttk.Label(options_frame, text="Batch Chunk Size:").pack(side="left", padx=(20, 5))
        entry_chunk = ttk.Entry(options_frame, textvariable=self.batch_chunk_size_var, width=7)
        entry_chunk.pack(side="left")
        self._create_hover_tooltip(entry_chunk, "batch_chunk_size")


        # --- PROGRESS & BUTTONS ---
        progress_frame = ttk.LabelFrame(self, text="Progress", padding=10)
        progress_frame.pack(fill="x", padx=10, pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, length=400, mode='determinate')
        self.progress_bar.pack(fill="x")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_label_var)
        self.status_label.pack(pady=5)

        buttons_frame = ttk.Frame(self, padding=10)
        buttons_frame.pack(fill="x")
        self.start_button = ttk.Button(buttons_frame, text="Start Blending", command=self.start_processing)
        self.start_button.pack(side="left", padx=5, expand=True)
        self._create_hover_tooltip(self.start_button, "start_blending")
        self.stop_button = ttk.Button(buttons_frame, text="Stop", command=self.stop_processing, state="disabled")
        self.stop_button.pack(side="left", padx=5, expand=True)
        self._create_hover_tooltip(self.stop_button, "stop_blending")

    def _browse_folder(self, var: tk.StringVar):
        folder = filedialog.askdirectory(initialdir=var.get())
        if folder:
            var.set(folder)

    def on_slider_release(self, event):
        """Called when a slider is released. Updates the preview."""
        self._stop_wigglegram_animation()
        if self.preview_inpainted_reader is not None:
            self.update_preview()

    def _stop_wigglegram_animation(self):
        if self.wiggle_after_id:
            self.after_cancel(self.wiggle_after_id)
            self.wiggle_after_id = None
        # --- NEW: Explicitly clear the stored PhotoImage objects to release memory ---
        if hasattr(self, 'wiggle_left_tk'):
            del self.wiggle_left_tk
        if hasattr(self, 'wiggle_right_tk'):
            del self.wiggle_right_tk

    def on_scrubber_move(self, value):
        """Called continuously as the frame scrubber moves to update the label."""
        frame_idx = int(float(value))
        total_frames = int(self.frame_scrubber.cget("to")) + 1
        self.frame_label_var.set(f"Frame: {frame_idx + 1} / {total_frames}")

    def update_status_label(self, message):
        self.status_label_var.set(message)
        self.update_idletasks()

    def _clear_preview_resources(self):
        """Closes all preview-related video readers and clears the preview display."""
        self._stop_wigglegram_animation()

        if self.preview_inpainted_reader:
            del self.preview_inpainted_reader
            self.preview_inpainted_reader = None
        if self.preview_original_reader:
            del self.preview_original_reader
            self.preview_original_reader = None
        if self.preview_splatted_reader:
            del self.preview_splatted_reader
            self.preview_splatted_reader = None

        self.preview_label.config(image=None, text="Load a video to see preview")
        self.preview_image_tk = None
        self.pil_image_for_preview = None
        self.preview_original_left_tensor = None
        self.preview_blended_right_tensor = None
        gc.collect()
        logger.info("Preview resources and file handles have been released.")

    def start_processing(self):
        if self.is_processing:
            messagebox.showwarning("Busy", "Processing is already in progress.")
            return

        self.is_processing = True
        self.stop_event.clear()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        # --- NEW: Clear preview resources before starting batch processing ---
        self._clear_preview_resources()

        self.update_status_label("Starting...")

        # Collect settings
        settings = self.get_current_settings()

        # Run in a separate thread
        self.processing_thread = threading.Thread(
            target=self.run_batch_process,
            args=(settings,),
            daemon=True
        )
        self.processing_thread.start()

    def stop_processing(self):
        if self.is_processing:
            self.stop_event.set()
            release_cuda_memory() # Explicitly release VRAM on stop
            self.update_status_label("Stopping...")

    def processing_done(self, stopped=False):
        self.is_processing = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        message = "Processing stopped." if stopped else "Processing completed."
        self.update_status_label(message)
        release_cuda_memory() # Explicitly release VRAM
        self.progress_var.set(0)

    def get_current_settings(self):
        self.save_config() # Save settings on every run
        """Collects all GUI settings into a dictionary."""
        try:
            settings = {
                "inpainted_folder": self.inpainted_folder_var.get(),
                "original_folder": self.original_folder_var.get(),
                "mask_folder": self.mask_folder_var.get(),
                "output_folder": self.output_folder_var.get(),
                "use_gpu": self.use_gpu_var.get(),
                "pad_to_16_9": self.pad_to_16_9_var.get(),
                "output_format": self.output_format_var.get(),
                "batch_chunk_size": int(self.batch_chunk_size_var.get()),
                "enable_color_transfer": self.enable_color_transfer_var.get(),
                "preview_size": int(self.preview_size_var.get()),
                # Mask params
                "binarize_threshold": float(self.mask_binarize_threshold_var.get()),
                "dilate_kernel": int(self.mask_dilate_kernel_size_var.get()),
                "blur_kernel": int(self.mask_blur_kernel_size_var.get()),
                "shadow_shift": int(self.shadow_shift_var.get()),
                "shadow_start_opacity": float(self.shadow_start_opacity_var.get()),
                "shadow_opacity_decay": float(self.shadow_opacity_decay_var.get()),
                "shadow_min_opacity": float(self.shadow_min_opacity_var.get()),
                "shadow_decay_gamma": float(self.shadow_decay_gamma_var.get()),
            }
            return settings
        except (ValueError, TypeError) as e:
            messagebox.showerror("Invalid Settings", f"Please check your parameter values. They must be valid numbers.\n\nError: {e}")
            return None

    def _read_ffmpeg_output(self, pipe, log_level):
        """Helper method to read FFmpeg's output without blocking."""
        try:
            # Use iter to read line by line
            for line in iter(pipe.readline, ''):
                if line:
                    logger.log(log_level, f"FFmpeg: {line.strip()}")
        except Exception as e:
            logger.error(f"Error reading FFmpeg pipe: {e}")
        finally:
            if pipe:
                pipe.close()

    def run_batch_process(self, settings):
        """
        This is the main logic that will run in a background thread.
        """
        if settings is None:
            self.after(0, self.processing_done, True)
            return

        inpainted_videos = sorted(glob.glob(os.path.join(settings["inpainted_folder"], "*.mp4")))
        if not inpainted_videos:
            self.after(0, lambda: messagebox.showinfo("Info", "No .mp4 files found in the inpainted video folder."))
            self.after(0, self.processing_done)
            return

        total_videos = len(inpainted_videos)
        self.progress_bar.config(maximum=total_videos)

        for i, inpainted_video_path in enumerate(inpainted_videos):
            if self.stop_event.is_set():
                logger.info("Processing stopped by user.")
                break

            base_name = os.path.basename(inpainted_video_path)
            self.after(0, self.update_status_label, f"Processing {i+1}/{total_videos}: {base_name}")

            # Initialize readers to None for robust cleanup
            inpainted_reader, splatted_reader, original_reader = None, None, None
            original_video_path_to_move = None # To track which original file to move
            try:
                # --- 1. Find corresponding files (same logic as preview) ---
                inpaint_suffix = "_inpainted_right_eye.mp4"
                sbs_suffix = "_inpainted_sbs.mp4"
                is_sbs_input = base_name.endswith(sbs_suffix)
                core_name_with_width = base_name[:-len(sbs_suffix)] if is_sbs_input else base_name[:-len(inpaint_suffix)]
                core_name = core_name_with_width[:core_name_with_width.rfind('_')]

                mask_folder = settings["mask_folder"]
                splatted4_pattern = os.path.join(mask_folder, f"{core_name}_*_splatted4.mp4")
                splatted2_pattern = os.path.join(mask_folder, f"{core_name}_*_splatted2.mp4")
                splatted4_matches = glob.glob(splatted4_pattern)
                splatted2_matches = glob.glob(splatted2_pattern)

                if splatted4_matches:
                    splatted_file_path = splatted4_matches[0]
                    is_dual_input = False
                elif splatted2_matches:
                    splatted_file_path = splatted2_matches[0]
                    is_dual_input = True
                else:
                    raise FileNotFoundError(f"Could not find a matching splatted file for '{core_name}'")

                # 2. Open readers, don't load all frames
                inpainted_reader = VideoReader(inpainted_video_path, ctx=cpu(0))
                splatted_reader = VideoReader(splatted_file_path, ctx=cpu(0))
                original_reader = None

                if is_dual_input:
                    original_video_path = os.path.join(settings["original_folder"], f"{core_name}.mp4")
                    original_video_path_to_move = original_video_path # Track for moving later
                    original_reader = VideoReader(original_video_path, ctx=cpu(0))

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
                
                # --- NEW: Determine output dimensions and suffix based on format ---
                output_format = settings["output_format"]
                if output_format == "Full SBS Cross-eye (Right-Left)":
                    output_width = hires_W * 2
                    output_suffix = "_merged_full_sbsx.mp4"
                elif output_format == "Full SBS (Left-Right)":
                    output_width = hires_W * 2
                    output_suffix = "_merged_full_sbs.mp4"
                elif output_format == "Double SBS":
                    output_width = hires_W * 2
                    output_height = hires_H * 2
                    output_suffix = "_merged_half_sbs.mp4"
                elif output_format == "Half SBS (Left-Right)":
                    output_width = hires_W
                    output_suffix = "_merged_half_sbs.mp4"
                elif output_format in ["Anaglyph (Red/Cyan)", "Anaglyph Half-Color"]:
                    output_width = hires_W
                    output_suffix = "_merged_anaglyph.mp4"
                else: # Right-Eye Only
                    output_width = hires_W
                    output_suffix = "_merged_right_eye.mp4"
                
                if 'output_height' not in locals(): # Set default height if not already set by a special format
                    output_height = hires_H
                output_filename = f"{core_name_with_width}{output_suffix}"
                output_path = os.path.join(settings["output_folder"], output_filename)
                # --- END NEW ---

                # --- NEW: Pass padding setting to FFmpeg ---
                ffmpeg_process = start_ffmpeg_pipe_process(
                    content_width=output_width,
                    content_height=output_height,
                    final_output_mp4_path=output_path,
                    fps=fps, video_stream_info=video_stream_info,
                    pad_to_16_9=settings["pad_to_16_9"],
                    output_format_str=output_format) # Pass the format string

                if ffmpeg_process is None:
                    raise RuntimeError("Failed to start FFmpeg pipe process.")

                # --- NEW: Start threads to read stdout and stderr to prevent deadlock ---
                stdout_thread = threading.Thread(
                    target=self._read_ffmpeg_output,
                    args=(ffmpeg_process.stdout, logging.DEBUG),
                    daemon=True
                )
                stderr_thread = threading.Thread(
                    target=self._read_ffmpeg_output,
                    args=(ffmpeg_process.stderr, logging.DEBUG),
                    daemon=True
                )
                stdout_thread.start()
                stderr_thread.start()

                # 4. Loop through chunks
                chunk_size = settings.get("batch_chunk_size", 32)
                for frame_start in range(0, num_frames, chunk_size):
                    if self.stop_event.is_set(): break
                    
                    frame_end = min(frame_start + chunk_size, num_frames)
                    frame_indices = list(range(frame_start, frame_end))
                    if not frame_indices: break

                    self.after(0, self.update_status_label, f"Processing frames {frame_start+1}-{frame_end}/{num_frames}...")

                    # Load current chunk
                    inpainted_np = inpainted_reader.get_batch(frame_indices).asnumpy()
                    splatted_np = splatted_reader.get_batch(frame_indices).asnumpy()
                    
                    # Convert to tensors and extract parts (same logic as preview)
                    # ... (this logic is identical to update_preview's frame loading part)
                    inpainted_tensor_full = torch.from_numpy(inpainted_np).permute(0, 3, 1, 2).float() / 255.0
                    splatted_tensor = torch.from_numpy(splatted_np).permute(0, 3, 1, 2).float() / 255.0
                    inpainted = inpainted_tensor_full[:, :, :, inpainted_tensor_full.shape[3]//2:] if is_sbs_input else inpainted_tensor_full
                    _, _, H, W = splatted_tensor.shape
                    if is_dual_input:
                        original_np = original_reader.get_batch(frame_indices).asnumpy()
                        original_left = torch.from_numpy(original_np).permute(0, 3, 1, 2).float() / 255.0
                        mask_raw = splatted_tensor[:, :, :, :W//2]
                        warped_original = splatted_tensor[:, :, :, W//2:]
                    else:
                        original_left = splatted_tensor[:, :, :H//2, :W//2]
                        mask_raw = splatted_tensor[:, :, H//2:, :W//2]
                        warped_original = splatted_tensor[:, :, H//2:, W//2:]
                    mask_np = mask_raw.permute(0, 2, 3, 1).cpu().numpy()
                    mask_gray_np = np.mean(mask_np, axis=3)
                    mask = torch.from_numpy(mask_gray_np).float().unsqueeze(1)

                    # Process chunk
                    use_gpu = settings["use_gpu"] and torch.cuda.is_available()
                    device = "cuda" if use_gpu else "cpu"
                    mask, inpainted, original_left, warped_original = mask.to(device), inpainted.to(device), original_left.to(device), warped_original.to(device)
                    
                    if inpainted.shape[2] != hires_H or inpainted.shape[3] != hires_W:
                        inpainted = F.interpolate(inpainted, size=(hires_H, hires_W), mode='bicubic', align_corners=False)
                        mask = F.interpolate(mask, size=(hires_H, hires_W), mode='bilinear', align_corners=False)

                    if settings["enable_color_transfer"]:
                        adjusted_frames = []
                        for frame_idx in range(inpainted.shape[0]):
                            adjusted_frame = apply_color_transfer(original_left[frame_idx].cpu(), inpainted[frame_idx].cpu())
                            adjusted_frames.append(adjusted_frame.to(device))
                        inpainted = torch.stack(adjusted_frames)

                    processed_mask = mask.clone()
                    if settings["dilate_kernel"] > 0: processed_mask = apply_mask_dilation(processed_mask, settings["dilate_kernel"], use_gpu)
                    if settings["blur_kernel"] > 0: processed_mask = apply_gaussian_blur(processed_mask, settings["blur_kernel"], use_gpu)
                    
                    # --- NEW: Binarization as the first step ---
                    if settings["binarize_threshold"] >= 0.0:
                        processed_mask = (mask > settings["binarize_threshold"]).float()

                    if settings["shadow_shift"] > 0: processed_mask = apply_shadow_blur(processed_mask, settings["shadow_shift"], settings["shadow_start_opacity"], settings["shadow_opacity_decay"], settings["shadow_min_opacity"], settings["shadow_decay_gamma"], use_gpu)

                    blended_right_eye = warped_original * (1 - processed_mask) + inpainted * processed_mask

                    # --- NEW: Assemble final frame based on output format ---
                    if output_format == "Full SBS (Left-Right)":
                        final_chunk = torch.cat([original_left, blended_right_eye], dim=3)
                    elif output_format == "Full SBS Cross-eye (Right-Left)":
                        final_chunk = torch.cat([blended_right_eye, original_left], dim=3)
                    elif output_format == "Half SBS (Left-Right)":
                        resized_left = F.interpolate(original_left, size=(hires_H, hires_W // 2), mode='bilinear', align_corners=False)
                        resized_right = F.interpolate(blended_right_eye, size=(hires_H, hires_W // 2), mode='bilinear', align_corners=False)
                        final_chunk = torch.cat([resized_left, resized_right], dim=3)
                    elif output_format == "Double SBS":
                        sbs_chunk = torch.cat([original_left, blended_right_eye], dim=3)
                        final_chunk = F.interpolate(sbs_chunk, size=(hires_H * 2, hires_W * 2), mode='bilinear', align_corners=False)
                    elif output_format == "Anaglyph (Red/Cyan)":
                        # Red from Left, Green/Blue from Right
                        final_chunk = torch.cat([
                            original_left[:, 0:1, :, :],      # R channel from left
                            blended_right_eye[:, 1:3, :, :]   # G, B channels from right
                        ], dim=1)
                    elif output_format == "Anaglyph Half-Color":
                        # Convert left to grayscale for the red channel
                        left_gray = original_left[:, 0, :, :] * 0.299 + original_left[:, 1, :, :] * 0.587 + original_left[:, 2, :, :] * 0.114
                        left_gray = left_gray.unsqueeze(1) # Add channel dimension back
                        final_chunk = torch.cat([
                            left_gray,                        # R channel from grayscale left
                            blended_right_eye[:, 1:3, :, :]   # G, B channels from right
                        ], dim=1)
                    else:
                        # Default to Right-Eye Only
                        final_chunk = blended_right_eye
                    # --- END NEW ---

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
                stdout_thread.join(timeout=5) # Wait for reader threads to finish
                stderr_thread.join(timeout=5)
                ffmpeg_process.wait()

                if ffmpeg_process.returncode != 0:
                    logger.error(f"FFmpeg encoding failed for {base_name}. Check console for details.")
                else:
                    logger.info(f"Successfully encoded video to {output_path}")
                    # --- NEW: Move files on success ---
                    self._move_processed_file(inpainted_video_path, settings["inpainted_folder"])
                    self._move_processed_file(splatted_file_path, settings["mask_folder"])
                    if original_video_path_to_move:
                        self._move_processed_file(original_video_path_to_move, settings["original_folder"])
            except Exception as e:
                logger.error(f"Failed to process {base_name}: {e}", exc_info=True)
                self.after(0, lambda base_name=base_name, e=e: messagebox.showerror("Processing Error", f"An error occurred while processing {base_name}:\n\n{e}"))
            finally:
                # Ensure readers are always cleaned up, even on error
                if inpainted_reader: del inpainted_reader
                if splatted_reader: del splatted_reader
                if original_reader: del original_reader
                gc.collect()
                # --- END: CHUNK-BASED PROCESSING ---

            self.after(0, self.progress_var.set, i + 1)

        self.after(0, self.processing_done, self.stop_event.is_set())

    def _move_processed_file(self, file_path: str, base_folder: str, max_retries: int = 5, retry_delay_sec: float = 0.5):
        """Moves a single file to a 'finished' subfolder within its base folder."""
        if not file_path or not os.path.exists(file_path):
            return
        
        finished_dir = os.path.join(base_folder, "finished")
        os.makedirs(finished_dir, exist_ok=True)
        dest_path = os.path.join(finished_dir, os.path.basename(file_path))

        for attempt in range(max_retries):
            try:
                # --- NEW: Check if destination exists before moving ---
                if os.path.exists(dest_path):
                    logger.warning(f"Destination '{os.path.basename(dest_path)}' already exists. Assuming previous move was incomplete. Deleting source file.")
                    try:
                        os.remove(file_path)
                        logger.info(f"Successfully removed source file: {os.path.basename(file_path)}")
                    except Exception as e_del:
                        logger.error(f"Failed to remove source file '{os.path.basename(file_path)}' after finding existing destination: {e_del}")
                    return # End the operation for this file

                # If destination does not exist, attempt the move
                shutil.move(file_path, dest_path)
                logger.info(f"Moved processed file to finished folder: {os.path.basename(file_path)}")
                return # Success
            except PermissionError:
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: Could not move '{os.path.basename(file_path)}' (file in use). Retrying...")
                time.sleep(retry_delay_sec)
            except Exception as e:
                logger.error(f"Failed to move file {os.path.basename(file_path)} to finished folder: {e}")
                return # Don't retry on other errors

        logger.error(f"Failed to move file '{os.path.basename(file_path)}' after {max_retries} attempts. It may still be locked.")

    def restore_finished_files(self):
        """Moves all files from 'finished' subfolders back to their parent directories."""
        if not messagebox.askyesno("Restore Finished Files", "Are you sure you want to move all processed videos from the 'finished' folders back to their respective input directories?"):
            return

        folders_to_check = [
            self.inpainted_folder_var.get(),
            self.original_folder_var.get(),
            self.mask_folder_var.get()
        ]
        restored_count = 0
        for folder in set(folders_to_check): # Use set to avoid duplicates
            finished_dir = os.path.join(folder, "finished")
            if os.path.isdir(finished_dir):
                for filename in os.listdir(finished_dir):
                    src_path = os.path.join(finished_dir, filename)
                    dest_path = os.path.join(folder, filename)
                    try:
                        shutil.move(src_path, dest_path)
                        restored_count += 1
                        logger.info(f"Restored '{filename}' to '{folder}'")
                    except Exception as e:
                        logger.error(f"Error restoring file '{filename}': {e}")
        messagebox.showinfo("Restore Complete", f"Restored {restored_count} files.")

    def _refresh_preview_video_list(self):
        """Scans the inpainted folder for valid videos and loads the first one."""
        self.preview_source_combo.set("Blended Image") # Reset preview mode when loading a new list
        inpainted_folder = self.inpainted_folder_var.get()
        if not os.path.isdir(inpainted_folder):
            messagebox.showerror("Error", "Inpainted Video Folder is not a valid directory.")
            return

        # Find all valid inpainted videos
        all_mp4s = sorted(glob.glob(os.path.join(inpainted_folder, "*.mp4")))
        self.preview_video_list = [
            f for f in all_mp4s 
            if f.endswith("_inpainted_right_eye.mp4") or f.endswith("_inpainted_sbs.mp4")
        ]

        if not self.preview_video_list:
            messagebox.showwarning("Not Found", "No validly named inpainted videos found (*_inpainted_right_eye.mp4 or *_inpainted_sbs.mp4).")
            self.current_preview_video_index = -1
            self._update_nav_controls()
            return

        self.current_preview_video_index = 0
        self._load_preview_by_index(self.current_preview_video_index)

    def _nav_preview_video(self, direction: int):
        """Navigate to the previous or next video in the preview list."""
        if not self.preview_video_list:
            return
        
        new_index = self.current_preview_video_index + direction
        if 0 <= new_index < len(self.preview_video_list):
            self._load_preview_by_index(new_index)

    def _jump_to_video(self, event=None):
        """Jump to a specific video number in the preview list."""
        if not self.preview_video_list:
            return
        try:
            target_index = int(self.video_jump_to_var.get()) - 1
            if 0 <= target_index < len(self.preview_video_list):
                self._load_preview_by_index(target_index)
            else:
                messagebox.showwarning("Out of Range", f"Please enter a number between 1 and {len(self.preview_video_list)}.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number.")

    def _load_preview_by_index(self, index: int):
        """Loads a specific video from the preview list by its index."""
        self._stop_wigglegram_animation()
        if not (0 <= index < len(self.preview_video_list)):
            return

        self.current_preview_video_index = index
        self._update_nav_controls()

        inpainted_video_path = self.preview_video_list[index]
        base_name = os.path.basename(inpainted_video_path)
        self.preview_loaded_file = base_name
        self.update_status_label(f"Loading preview for: {base_name}")

        self.load_preview_button.config(text="LOADING...", style="Loading.TButton")
        self.update_idletasks()

        # --- NEW: Close any previously opened video readers to release file handles ---
        if self.preview_inpainted_reader: del self.preview_inpainted_reader
        if self.preview_original_reader: del self.preview_original_reader
        if self.preview_splatted_reader: del self.preview_splatted_reader
        self.preview_inpainted_reader = None
        self.preview_original_reader = None
        self.preview_splatted_reader = None
        # -----------------------------------------------------------------------------

        try:
            # 1. Parse the inpainted video name to get the base and width
            inpaint_suffix = "_inpainted_right_eye.mp4"
            sbs_suffix = "_inpainted_sbs.mp4"

            if base_name.endswith(inpaint_suffix):
                core_name_with_width = base_name[:-len(inpaint_suffix)]
                self.is_sbs_preview = False
            elif base_name.endswith(sbs_suffix):
                core_name_with_width = base_name[:-len(sbs_suffix)]
                self.is_sbs_preview = True
                logger.info("Detected pre-blended SBS input for preview.")
            else:
                raise ValueError(f"Inpainted file '{base_name}' does not have an expected suffix ('{inpaint_suffix}' or '{sbs_suffix}').")

            last_underscore_idx = core_name_with_width.rfind('_')
            if last_underscore_idx == -1:
                raise ValueError(f"Could not determine width from '{core_name_with_width}'. Expected format '..._WIDTH'.")
            
            core_name = core_name_with_width[:last_underscore_idx]
            
            # 2. Find the corresponding splatted file (_splatted4 or _splatted2)
            # Use glob to find the file with a wildcard for the resolution part.
            mask_folder = self.mask_folder_var.get()
            splatted4_pattern = os.path.join(mask_folder, f"{core_name}_*_splatted4.mp4")
            splatted2_pattern = os.path.join(mask_folder, f"{core_name}_*_splatted2.mp4")

            splatted4_matches = glob.glob(splatted4_pattern)
            splatted2_matches = glob.glob(splatted2_pattern)

            splatted_file_path = None
            is_dual_input = False
            if splatted4_matches:
                splatted_file_path = splatted4_matches[0] # Use the first match
                is_dual_input = False
                logger.info(f"Found quad-splatted file: {os.path.basename(splatted_file_path)}")
            elif splatted2_matches:
                splatted_file_path = splatted2_matches[0] # Use the first match
                is_dual_input = True
                logger.info(f"Found dual-splatted file: {os.path.basename(splatted_file_path)}")
            else:
                raise FileNotFoundError(f"Could not find a matching _splatted4 or _splatted2 file for '{core_name}' in {mask_folder}")

            # 3. Initialize VideoReader objects for all sources
            self.preview_inpainted_reader = VideoReader(inpainted_video_path, ctx=cpu(0))
            self.preview_splatted_reader = VideoReader(splatted_file_path, ctx=cpu(0))
            self.preview_is_splatted_dual = is_dual_input

            if is_dual_input:
                original_video_path = os.path.join(self.original_folder_var.get(), f"{core_name}.mp4") # Assumes original is named without width
                if not os.path.exists(original_video_path):
                    raise FileNotFoundError(f"Dual input requires original video, but not found: {original_video_path}")
                self.preview_original_reader = VideoReader(original_video_path, ctx=cpu(0))
            # For quad input, the original_reader will be None, as the left-eye comes from the splatted file itself.

            # Configure the scrubber
            num_frames = len(self.preview_inpainted_reader)
            self.frame_scrubber.config(to=num_frames - 1)
            self.frame_scrubber_var.set(0)
            self.on_scrubber_move(0) # Update label

            # Configure preview source dropdown
            preview_options = ["Blended Image", "Original (Left Eye)", "Warped (Right BG)", "Processed Mask", "Anaglyph 3D", "Wigglegram"]
            if not is_dual_input: # Depth map is only in quad-splatted files
                preview_options.append("Depth Map")
            self.preview_source_combo['values'] = preview_options

            self.update_preview()
            self.update_status_label(f"Preview loaded for: {base_name}")

        except Exception as e:
            messagebox.showerror("Preview Load Error", f"Failed to load files for preview:\n\n{e}")
            self.update_status_label("Preview load failed.")
            logger.error("Preview load failed", exc_info=True)
        finally:
            # Revert button to normal state
            self.load_preview_button.config(text="Load/Refresh List", style="TButton")

    def _update_nav_controls(self):
        """Updates the state and labels of the video navigation controls."""
        total_videos = len(self.preview_video_list)
        current_index = self.current_preview_video_index

        self.video_status_label_var.set(f"Video: {current_index + 1} / {total_videos}" if total_videos > 0 else "Video: 0 / 0")
        self.video_jump_to_var.set(str(current_index + 1) if total_videos > 0 else "1")

        self.prev_video_button.config(state="normal" if current_index > 0 else "disabled")
        self.next_video_button.config(state="normal" if 0 <= current_index < total_videos - 1 else "disabled")
        self.video_jump_entry.config(state="normal" if total_videos > 0 else "disabled")

    def update_preview(self):
        """Processes and displays the preview frame based on current slider values."""
        self._stop_wigglegram_animation()
        self.load_preview_button.config(text="LOADING...", style="Loading.TButton")
        self.update_idletasks()
        # Add checks to ensure all required preview frames are loaded.
        if self.preview_inpainted_reader is None or self.preview_splatted_reader is None:
            return

        try:
            # Get settings
            settings = self.get_current_settings()
            if settings is None: return

            use_gpu = settings["use_gpu"] and torch.cuda.is_available()
            device = "cuda" if use_gpu else "cpu"

            # Get the current frame index from the scrubber
            frame_idx = int(self.frame_scrubber_var.get())

            # --- NEW: Load a single frame from each reader ---
            inpainted_np = self.preview_inpainted_reader.get_batch([frame_idx]).asnumpy()
            splatted_np = self.preview_splatted_reader.get_batch([frame_idx]).asnumpy()

            inpainted_tensor_full = torch.from_numpy(inpainted_np).permute(0, 3, 1, 2).float() / 255.0
            splatted_tensor = torch.from_numpy(splatted_np).permute(0, 3, 1, 2).float() / 255.0

            # Extract inpainted right-eye if source is SBS
            if self.is_sbs_preview:
                _, _, _, sbs_W = inpainted_tensor_full.shape
                inpainted = inpainted_tensor_full[:, :, :, sbs_W//2:]
            else:
                inpainted = inpainted_tensor_full

            # Extract parts from the splatted frame
            _, _, H, W = splatted_tensor.shape
            if self.preview_is_splatted_dual:
                half_w = W // 2
                mask_raw = splatted_tensor[:, :, :, :half_w]
                right_eye_original = splatted_tensor[:, :, :, half_w:]
                
                # Load left eye from its separate reader
                original_np = self.preview_original_reader.get_batch([frame_idx]).asnumpy()
                original_left = torch.from_numpy(original_np).permute(0, 3, 1, 2).float() / 255.0
                depth_map_vis = None
            else: # Quad
                half_h, half_w = H // 2, W // 2
                original_left = splatted_tensor[:, :, :half_h, :half_w]
                depth_map_vis = splatted_tensor[:, :, :half_h, half_w:]
                mask_raw = splatted_tensor[:, :, half_h:, :half_w]
                right_eye_original = splatted_tensor[:, :, half_h:, half_w:]

            # Convert mask to grayscale
            mask_frame_np = mask_raw.squeeze(0).permute(1, 2, 0).cpu().numpy()
            mask_gray_np = np.mean(mask_frame_np, axis=2)
            mask = torch.from_numpy(mask_gray_np).float().unsqueeze(0).unsqueeze(0)
            # ----------------------------------------------------

            # Move tensors to the processing device
            mask = mask.to(device)
            inpainted = inpainted.to(device)
            original_left = original_left.to(device)
            right_eye_original = right_eye_original.to(device)

            # --- FIX: Upscale low-res inpainted frame and mask to match hi-res warped frame ---
            hires_H, hires_W = right_eye_original.shape[2], right_eye_original.shape[3]
            if inpainted.shape[2] != hires_H or inpainted.shape[3] != hires_W:
                logger.debug(f"Upscaling preview frames from {inpainted.shape[3]}x{inpainted.shape[2]} to {hires_W}x{hires_H}")
                inpainted = F.interpolate(inpainted, size=(hires_H, hires_W), mode='bicubic', align_corners=False)
                mask = F.interpolate(mask, size=(hires_H, hires_W), mode='bilinear', align_corners=False)

            # --- Process the mask (using a simplified chain from test.py) ---
            processed_mask = mask.clone()

            # --- NEW: Binarization as the first step ---
            if settings["binarize_threshold"] >= 0.0:
                processed_mask = (processed_mask > settings["binarize_threshold"]).float()

            if settings["dilate_kernel"] > 0:
                processed_mask = apply_mask_dilation(processed_mask, settings["dilate_kernel"], use_gpu)
            if settings["blur_kernel"] > 0:
                processed_mask = apply_gaussian_blur(processed_mask, settings["blur_kernel"], use_gpu)
            if settings["shadow_shift"] > 0:
                processed_mask = apply_shadow_blur(processed_mask, settings["shadow_shift"], settings["shadow_start_opacity"], settings["shadow_opacity_decay"], settings["shadow_min_opacity"], settings["shadow_decay_gamma"], use_gpu)

            # --- Apply Color Transfer if enabled ---
            if settings["enable_color_transfer"]:
                # Ensure frames are on CPU for the numpy conversion in the function
                # The function expects [C, H, W] so we squeeze the batch dim
                if original_left is not None:
                    logger.debug("Applying color transfer to preview frame...")
                    source_cpu = original_left.squeeze(0).cpu()
                else: # Should not happen with current logic, but as a fallback
                    source_cpu = right_eye_original.squeeze(0).cpu()
                target_cpu = inpainted.squeeze(0).cpu()
                
                adjusted_target_cpu = apply_color_transfer(source_cpu, target_cpu)
                # Move back to device and add batch dim back for blending
                inpainted = adjusted_target_cpu.unsqueeze(0).to(device)

            blended_frame = right_eye_original * (1 - processed_mask) + inpainted * processed_mask

            # --- NEW: Store tensors for SBS saving ---
            self.preview_original_left_tensor = original_left.cpu()
            self.preview_blended_right_tensor = blended_frame.cpu()

            # --- Select the final frame to display based on the dropdown ---
            preview_source = self.preview_source_var.get()
            if preview_source == "Blended Image":
                final_frame = blended_frame
            elif preview_source == "Original (Left Eye)":
                final_frame = original_left
            elif preview_source == "Warped (Right BG)":
                final_frame = right_eye_original
            elif preview_source == "Processed Mask":
                final_frame = processed_mask.repeat(1, 3, 1, 1) # Convert grayscale mask to 3-channel for display
            elif preview_source == "Anaglyph 3D":
                # Convert left eye to grayscale for a cleaner half-color anaglyph
                left_gray_np = cv2.cvtColor((original_left.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                right_np = (blended_frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                anaglyph_np = right_np.copy()
                # Use the grayscale left eye for the red channel to reduce retinal rivalry
                anaglyph_np[:, :, 0] = left_gray_np # Red channel from grayscale left eye
                final_frame = torch.from_numpy(anaglyph_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            elif preview_source == "Wigglegram":
                # For wigglegram, we start the animation loop instead of setting a static frame
                self._start_wigglegram_animation(original_left, blended_frame)
                # We can return here as the animation will handle updates
                self.load_preview_button.config(text="Load/Refresh List", style="TButton")
                return
            elif preview_source == "Depth Map" and depth_map_vis is not None:
                final_frame = depth_map_vis.to(device)
            else: # Fallback to blended
                final_frame = blended_frame

            # --- Convert to displayable image ---
            # Ensure final_frame is on CPU before numpy conversion
            final_frame_cpu = final_frame.cpu()
            blended_np = (final_frame_cpu.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # Resize for display if too large, maintaining aspect ratio
            max_size = settings.get("preview_size", 512)
            if blended_np.shape[0] > max_size or blended_np.shape[1] > max_size:
                pil_img = Image.fromarray(blended_np)
                pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                self.preview_image_tk = ImageTk.PhotoImage(pil_img)
                self.pil_image_for_preview = pil_img
            else:
                pil_img = Image.fromarray(blended_np)
                self.preview_image_tk = ImageTk.PhotoImage(pil_img)
                self.pil_image_for_preview = pil_img

            self.preview_label.config(image=self.preview_image_tk, text="")
            self._update_preview_layout()

        except Exception as e:
            logger.error(f"Error updating preview: {e}", exc_info=True)
            self.status_label_var.set("Error updating preview.")
        finally:
            # --- NEW: Explicitly release VRAM used by the preview render ---
            release_cuda_memory()
            self.load_preview_button.config(text="Load/Refresh List", style="TButton")

    def _update_preview_layout(self):
        """Centers the image if it's smaller than the canvas, and hides/shows scrollbars."""
        if not hasattr(self, 'preview_canvas'):
            return

        if not self.pil_image_for_preview:
            # No image is loaded, so ensure scrollbars are hidden.
            self.v_scrollbar.grid_remove()
            self.h_scrollbar.grid_remove()
            return

        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()
        img_w = self.pil_image_for_preview.width
        img_h = self.pil_image_for_preview.height

        # Determine if scrollbars are needed
        v_scroll_needed = img_h > canvas_h
        h_scroll_needed = img_w > canvas_w

        # Show or hide scrollbars
        if v_scroll_needed: self.v_scrollbar.grid()
        else: self.v_scrollbar.grid_remove()
        if h_scroll_needed: self.h_scrollbar.grid()
        else: self.h_scrollbar.grid_remove()

        # Center the image if it's smaller than the canvas
        x = max(0, (canvas_w - img_w) // 2)
        y = max(0, (canvas_h - img_h) // 2)
        
        # Update the position of the inner frame on the canvas
        self.preview_canvas.coords(self.preview_canvas_window_id, x, y)

        # Update the scrollable region
        self.preview_inner_frame.update_idletasks()
        self.preview_canvas.config(scrollregion=self.preview_canvas.bbox("all"))

    def _start_wigglegram_animation(self, left_frame, right_frame):
        """Starts the wigglegram animation loop."""
        self._stop_wigglegram_animation() # Ensure any previous loop is stopped

        # Pre-convert frames to PhotoImage to make the loop faster
        max_size = int(self.preview_size_var.get())
        
        left_np = (left_frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        left_pil = Image.fromarray(left_np)
        left_pil.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        self.wiggle_left_tk = ImageTk.PhotoImage(left_pil)

        right_np = (right_frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        right_pil = Image.fromarray(right_np)
        right_pil.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        self.wiggle_right_tk = ImageTk.PhotoImage(right_pil)

        self._wiggle_step(True) # Start with the first frame

    def _wiggle_step(self, show_left):
        """A single step in the wigglegram animation."""
        self.preview_label.config(image=self.wiggle_left_tk if show_left else self.wiggle_right_tk)
        # Schedule the next flip after ~125ms (for 8fps)
        self.wiggle_after_id = self.after(60, self._wiggle_step, not show_left)

    def save_config(self):
        """Saves the current GUI configuration to a JSON file."""
        config = {
            "inpainted_folder": self.inpainted_folder_var.get(),
            "original_folder": self.original_folder_var.get(),
            "mask_folder": self.mask_folder_var.get(),
            "output_folder": self.output_folder_var.get(),
            "use_gpu": self.use_gpu_var.get(),            
            "output_format": self.output_format_var.get(),            
            "pad_to_16_9": self.pad_to_16_9_var.get(),
            "batch_chunk_size": self.batch_chunk_size_var.get(),
            "enable_color_transfer": self.enable_color_transfer_var.get(),
            "debug_logging_enabled": self.debug_logging_var.get(),
            "dark_mode_enabled": self.dark_mode_var.get(),
            "preview_size": self.preview_size_var.get(),
            "mask_dilate_kernel_size": str(self.mask_dilate_kernel_size_var.get()),
            "mask_binarize_threshold": str(self.mask_binarize_threshold_var.get()),
            "window_x": self.winfo_x(),
            "window_y": self.winfo_y(),
            "window_width": self.winfo_width(),
            "mask_blur_kernel_size": str(self.mask_blur_kernel_size_var.get()),
            "shadow_shift": str(self.shadow_shift_var.get()),
            "shadow_start_opacity": str(self.shadow_start_opacity_var.get()),
            "shadow_opacity_decay": str(self.shadow_opacity_decay_var.get()),
            "shadow_min_opacity": str(self.shadow_min_opacity_var.get()),
            "shadow_decay_gamma": str(self.shadow_decay_gamma_var.get()),
        }
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
            title="Load Settings from File"
        )
        if not filepath: return
        try:
            with open(filepath, "r") as f:
                config = json.load(f)
            # Apply settings
            for key, value in config.items():
                var_name = key + "_var"
                if hasattr(self, var_name):
                    getattr(self, var_name).set(value)
            self._apply_theme()
            logger.info(f"Settings loaded from {filepath}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load settings from {filepath}:\n{e}")

    def save_settings_dialog(self):
        """Saves current GUI settings to a user-selected JSON file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".mergecfg",
            filetypes=[("Merge Config Files", "*.mergecfg"), ("All files", "*.*")],
            title="Save Settings to File"
        )
        if not filepath: return
        self.save_config() # First save to internal config to gather all current values
        # Now copy the saved config to the new location
        try:
            shutil.copyfile("config_merging.mergecfg", filepath)
            logger.info(f"Settings saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save settings to {filepath}:\n{e}")

    def _save_preview_frame(self):
        """Saves the current preview image to a file."""
        if self.pil_image_for_preview is None:
            messagebox.showwarning("No Preview", "There is no preview image to save. Please load a video first.")
            return

        # Suggest a default filename
        default_filename = "preview_frame.png"
        if self.preview_loaded_file:
            base_name = os.path.splitext(self.preview_loaded_file)[0]
            frame_num = int(self.frame_scrubber_var.get())
            default_filename = f"{base_name}_frame_{frame_num:05d}.png"

        filepath = filedialog.asksaveasfilename(
            title="Save Preview Frame As...",
            initialfile=default_filename,
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")]
        )

        if filepath:
            try:
                self.pil_image_for_preview.save(filepath)
                logger.info(f"Preview frame saved to: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save preview frame: {e}", exc_info=True)
                messagebox.showerror("Save Error", f"An error occurred while saving the image:\n{e}")

    def _save_preview_sbs_frame(self):
        """Saves the current preview as a full side-by-side image."""
        if self.preview_original_left_tensor is None or self.preview_blended_right_tensor is None:
            messagebox.showwarning("No Preview Data", "There is no preview data to save. Please load and preview a video first.")
            return

        try:
            # Convert tensors to PIL Images
            left_np = (self.preview_original_left_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            right_np = (self.preview_blended_right_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            left_pil = Image.fromarray(left_np)
            right_pil = Image.fromarray(right_np)

            # Check if dimensions match
            if left_pil.size != right_pil.size:
                messagebox.showerror("Dimension Mismatch", "The left and right eye images have different dimensions. Cannot create SBS image.")
                return

            # Create SBS image
            width, height = left_pil.size
            sbs_image = Image.new('RGB', (width * 2, height))
            sbs_image.paste(left_pil, (0, 0))
            sbs_image.paste(right_pil, (width, 0))

            # Suggest a default filename
            default_filename = "preview_sbs_frame.png"
            if self.preview_loaded_file:
                base_name = os.path.splitext(self.preview_loaded_file)[0]
                frame_num = int(self.frame_scrubber_var.get())
                default_filename = f"{base_name}_frame_{frame_num:05d}_SBS.png"

            filepath = filedialog.asksaveasfilename(
                title="Save SBS Preview Frame As...",
                initialfile=default_filename,
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")]
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
            if messagebox.askyesno("Confirm Exit", "Processing is in progress. Are you sure you want to stop and exit?"):
                self.stop_processing()
                self.save_config()
                self.destroy()
        else:
            self.save_config()
            self.destroy()

if __name__ == "__main__":
    # Basic logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    app = MergingGUI()
    app.mainloop()
