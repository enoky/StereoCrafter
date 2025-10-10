import os
import glob
import json
import shutil
import threading
import tkinter as tk # Used for PanedWindow
from tkinter import filedialog, messagebox, ttk
from ttkthemes import ThemedTk
from typing import Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageTk
import logging
import time
from dependency.stereocrafter_util import Tooltip, logger, get_video_stream_info, draw_progress_bar, release_cuda_memory, set_util_logger_level, encode_frames_to_mp4, read_video_frames_decord

GUI_VERSION = "1.0.0"

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

def apply_stamp_blur(mask: torch.Tensor, shift_per_step: int, start_opacity: float, opacity_decay_per_step: float, min_opacity: float, decay_gamma: float = 1.0, use_gpu: bool = True) -> torch.Tensor:
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
        canvas_np = mask.squeeze(0).cpu().numpy()
        stamp_source_np = canvas_np.copy()
        for i in range(num_steps):
            t = 1.0 - (i / (num_steps - 1)) if num_steps > 1 else 1.0
            curved_t = t ** decay_gamma
            current_opacity = min_opacity + (start_opacity - min_opacity) * curved_t
            total_shift = (i + 1) * shift_per_step
            shifted_stamp = np.roll(stamp_source_np, total_shift, axis=2)
            canvas_np = np.maximum(canvas_np, shifted_stamp * current_opacity)
        return torch.from_numpy(canvas_np).unsqueeze(0).to(mask.device)

class MergingGUI(ThemedTk):
    def __init__(self):
        super().__init__(theme="clam")
        self.title(f"Stereocrafter Merging GUI {GUI_VERSION}")
        self.app_config = self.load_config()
        self.help_data = {} # Simplified for now

        # --- Core App State ---
        self.stop_event = threading.Event()
        self.is_processing = False
        
        # --- Preview State ---
        self.preview_inpainted_frame = None
        self.preview_original_frame = None
        self.preview_warped_frame = None # NEW: To store the warped frame for blending
        self.preview_mask_frame = None
        self.preview_image_tk = None # To prevent garbage collection
        self.preview_loaded_file = None

        # --- GUI Variables ---
        self.inpainted_folder_var = tk.StringVar(value=self.app_config.get("inpainted_folder", "./completed_output"))
        self.original_folder_var = tk.StringVar(value=self.app_config.get("original_folder", "./input_source_clips"))
        self.mask_folder_var = tk.StringVar(value=self.app_config.get("mask_folder", "./output_splatted/hires")) # Assuming masks are in splatted folder
        self.output_folder_var = tk.StringVar(value=self.app_config.get("output_folder", "./final_videos"))

        # --- Mask Processing Parameters ---
        self.mask_dilate_kernel_size_var = tk.DoubleVar(value=float(self.app_config.get("mask_dilate_kernel_size", 15)))
        self.mask_blur_kernel_size_var = tk.DoubleVar(value=float(self.app_config.get("mask_blur_kernel_size", 25)))
        self.stamp_shift_var = tk.DoubleVar(value=float(self.app_config.get("stamp_shift", 2)))
        self.stamp_start_opacity_var = tk.DoubleVar(value=float(self.app_config.get("stamp_start_opacity", 0.8)))
        self.stamp_opacity_decay_var = tk.DoubleVar(value=float(self.app_config.get("stamp_opacity_decay", 0.1)))
        self.stamp_min_opacity_var = tk.DoubleVar(value=float(self.app_config.get("stamp_min_opacity", 0.2)))
        self.stamp_decay_gamma_var = tk.DoubleVar(value=float(self.app_config.get("stamp_decay_gamma", 2.0)))

        self.use_gpu_var = tk.BooleanVar(value=self.app_config.get("use_gpu", True))
        self.use_sbs_output_var = tk.BooleanVar(value=self.app_config.get("use_sbs_output", True))

        # --- GUI Status Variables ---
        self.status_label_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0)

        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.exit_application)
        self.update_status_label("Ready.")

    def create_widgets(self):
        # --- Main Paned Window for resizable layout ---
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # --- Left Pane for Controls ---
        left_pane = ttk.Frame(main_pane, padding=5)
        main_pane.add(left_pane, weight=1)

        # --- Right Pane for Preview ---
        right_pane = ttk.LabelFrame(main_pane, text="Live Preview", padding=10)
        main_pane.add(right_pane, weight=2)

        # --- FOLDER FRAME ---
        folder_frame = ttk.LabelFrame(left_pane, text="Folders", padding=10)
        folder_frame.pack(fill="x", padx=10, pady=5, anchor="n")
        folder_frame.grid_columnconfigure(1, weight=1)

        # Inpainted Video Folder
        ttk.Label(folder_frame, text="Inpainted Video Folder:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(folder_frame, textvariable=self.inpainted_folder_var).grid(row=0, column=1, padx=5, sticky="ew")
        ttk.Button(folder_frame, text="Browse", command=lambda: self._browse_folder(self.inpainted_folder_var)).grid(row=0, column=2, padx=5)

        # Original Video Folder (for Left Eye)
        ttk.Label(folder_frame, text="Original Video Folder:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(folder_frame, textvariable=self.original_folder_var).grid(row=1, column=1, padx=5, sticky="ew")
        ttk.Button(folder_frame, text="Browse", command=lambda: self._browse_folder(self.original_folder_var)).grid(row=1, column=2, padx=5)

        # Mask Folder
        ttk.Label(folder_frame, text="Mask Folder:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(folder_frame, textvariable=self.mask_folder_var).grid(row=2, column=1, padx=5, sticky="ew")
        ttk.Button(folder_frame, text="Browse", command=lambda: self._browse_folder(self.mask_folder_var)).grid(row=2, column=2, padx=5)

        # Output Folder
        ttk.Label(folder_frame, text="Output Folder:").grid(row=3, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(folder_frame, textvariable=self.output_folder_var).grid(row=3, column=1, padx=5, sticky="ew")
        ttk.Button(folder_frame, text="Browse", command=lambda: self._browse_folder(self.output_folder_var)).grid(row=3, column=2, padx=5)

        # --- MASK PROCESSING PARAMETERS ---
        param_frame = ttk.LabelFrame(left_pane, text="Mask Processing Parameters", padding=10)
        param_frame.pack(fill="x", padx=10, pady=5, anchor="n")
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
            return slider

        create_slider(param_frame, "Dilate Kernel:", self.mask_dilate_kernel_size_var, 0, 101, 0)
        create_slider(param_frame, "Blur Kernel:", self.mask_blur_kernel_size_var, 0, 101, 1)
        create_slider(param_frame, "Stamp Shift:", self.stamp_shift_var, 0, 50, 2)
        create_slider(param_frame, "Stamp Gamma:", self.stamp_decay_gamma_var, 0.1, 5.0, 3, decimals=2)
        create_slider(param_frame, "Stamp Opacity Start:", self.stamp_start_opacity_var, 0.0, 1.0, 4, decimals=2)
        create_slider(param_frame, "Stamp Opacity Decay:", self.stamp_opacity_decay_var, 0.0, 1.0, 5, decimals=2)
        create_slider(param_frame, "Stamp Opacity Min:", self.stamp_min_opacity_var, 0.0, 1.0, 6, decimals=2)

        # --- OPTIONS FRAME ---
        options_frame = ttk.LabelFrame(left_pane, text="Options", padding=10)
        options_frame.pack(fill="x", padx=10, pady=5, anchor="n")

        ttk.Checkbutton(options_frame, text="Use GPU for Mask Processing", variable=self.use_gpu_var).pack(side="left", padx=5)
        ttk.Checkbutton(options_frame, text="Create Side-by-Side (SBS) Output", variable=self.use_sbs_output_var).pack(side="left", padx=5)

        # --- PROGRESS & BUTTONS ---
        progress_frame = ttk.LabelFrame(left_pane, text="Progress", padding=10)
        progress_frame.pack(fill="x", padx=10, pady=5, anchor="n")
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, length=400, mode='determinate')
        self.progress_bar.pack(fill="x")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_label_var)
        self.status_label.pack(pady=5)

        buttons_frame = ttk.Frame(left_pane, padding=10)
        buttons_frame.pack(fill="x", anchor="n")
        self.start_button = ttk.Button(buttons_frame, text="Start Blending", command=self.start_processing)
        self.start_button.pack(side="left", padx=5, expand=True)
        self.stop_button = ttk.Button(buttons_frame, text="Stop", command=self.stop_processing, state="disabled")
        self.stop_button.pack(side="left", padx=5, expand=True)

        # --- PREVIEW WIDGETS (in right_pane) ---
        self.preview_label = ttk.Label(right_pane, text="Load a video to see preview", anchor="center")
        self.preview_label.pack(fill="both", expand=True, padx=5, pady=5)

        preview_button_frame = ttk.Frame(right_pane)
        preview_button_frame.pack(fill="x", pady=5)
        self.load_preview_button = ttk.Button(preview_button_frame, text="Load First Video for Preview", command=self.load_for_preview)
        self.load_preview_button.pack()

    def _browse_folder(self, var: tk.StringVar):
        folder = filedialog.askdirectory(initialdir=var.get())
        if folder:
            var.set(folder)

    def on_slider_release(self, event):
        """Called when a slider is released. Updates the preview."""
        if self.preview_inpainted_frame is not None:
            self.update_preview()

    def update_status_label(self, message):
        self.status_label_var.set(message)
        self.update_idletasks()

    def start_processing(self):
        if self.is_processing:
            messagebox.showwarning("Busy", "Processing is already in progress.")
            return

        self.is_processing = True
        self.stop_event.clear()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
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
            self.update_status_label("Stopping...")

    def processing_done(self, stopped=False):
        self.is_processing = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        message = "Processing stopped." if stopped else "Processing completed."
        self.update_status_label(message)
        self.progress_var.set(0)

    def get_current_settings(self):
        """Collects all GUI settings into a dictionary."""
        try:
            settings = {
                "inpainted_folder": self.inpainted_folder_var.get(),
                "original_folder": self.original_folder_var.get(),
                "mask_folder": self.mask_folder_var.get(),
                "output_folder": self.output_folder_var.get(),
                "use_gpu": self.use_gpu_var.get(),
                "use_sbs": self.use_sbs_output_var.get(),
                # Mask params
                "dilate_kernel": int(self.mask_dilate_kernel_size_var.get()),
                "blur_kernel": int(self.mask_blur_kernel_size_var.get()),
                "stamp_shift": int(self.stamp_shift_var.get()),
                "stamp_start_opacity": float(self.stamp_start_opacity_var.get()),
                "stamp_opacity_decay": float(self.stamp_opacity_decay_var.get()),
                "stamp_min_opacity": float(self.stamp_min_opacity_var.get()),
                "stamp_decay_gamma": float(self.stamp_decay_gamma_var.get()),
            }
            return settings
        except (ValueError, TypeError) as e:
            messagebox.showerror("Invalid Settings", f"Please check your parameter values. They must be valid numbers.\n\nError: {e}")
            return None

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

            try:
                # =================================================================
                # TODO: CORE LOGIC WILL GO HERE
                # 1. Find matching original video and mask video
                # 2. Load all three videos as tensors
                # 3. Process the mask tensor using the functions from test.py and GUI settings
                # 4. Blend the inpainted video and original video using the processed mask
                # 5. If SBS is enabled, find and concatenate the left eye
                # 6. Encode the final result to the output folder
                # =================================================================
                logger.info(f"Simulating processing for {base_name}")
                time.sleep(2) # Placeholder for actual work

            except Exception as e:
                logger.error(f"Failed to process {base_name}: {e}", exc_info=True)
                self.after(0, lambda: messagebox.showerror("Processing Error", f"An error occurred while processing {base_name}:\n\n{e}"))

            self.after(0, self.progress_var.set, i + 1)

        self.after(0, self.processing_done, self.stop_event.is_set())

    def load_for_preview(self):
        """Loads the first frame of the first found video for live previewing."""
        inpainted_folder = self.inpainted_folder_var.get()
        inpainted_videos = sorted(glob.glob(os.path.join(inpainted_folder, "*.mp4")))
        if not inpainted_videos:
            messagebox.showwarning("Not Found", "No .mp4 files found in the Inpainted Video Folder to load for preview.")
            return

        inpainted_video_path = inpainted_videos[0]
        base_name = os.path.basename(inpainted_video_path)
        self.preview_loaded_file = base_name
        self.update_status_label(f"Loading preview for: {base_name}")

        try: # --- START: NEW FILE FINDING & EXTRACTION LOGIC ---
            # 1. Parse the inpainted video name to get the base and width
            inpaint_suffix = "_inpainted_right_eye.mp4"
            if not base_name.endswith(inpaint_suffix):
                raise ValueError(f"Inpainted file '{base_name}' does not have the expected '{inpaint_suffix}' suffix.")
            
            core_name_with_width = base_name[:-len(inpaint_suffix)]
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

            # 3. Load the necessary frames
            inpainted_frames_np, _, _, _, _, _, _ = read_video_frames_decord(inpainted_video_path, process_length=1)
            splatted_frames_np, _, _, _, _, _, _ = read_video_frames_decord(splatted_file_path, process_length=1)

            # Convert to tensors [T, C, H, W]
            inpainted_tensor = torch.from_numpy(inpainted_frames_np).permute(0, 3, 1, 2)
            splatted_tensor = torch.from_numpy(splatted_frames_np).permute(0, 3, 1, 2)

            # 4. Extract the correct parts based on splat type
            _, _, H, W = splatted_tensor.shape
            
            if is_dual_input:
                # Dual input: _splatted2
                half_w = W // 2
                mask_raw = splatted_tensor[:, :, :, :half_w]
                warped_original = splatted_tensor[:, :, :, half_w:]
                
                # For dual, the "original" left eye comes from a separate file
                original_video_path = os.path.join(self.original_folder_var.get(), f"{core_name}.mp4") # Assumes original is named without width
                if not os.path.exists(original_video_path):
                    raise FileNotFoundError(f"Dual input requires original video, but not found: {original_video_path}")
                original_frames_np, _, _, _, _, _, _ = read_video_frames_decord(original_video_path, process_length=1)
                original_tensor = torch.from_numpy(original_frames_np).permute(0, 3, 1, 2)
                left_eye_original = original_tensor
            else:
                # Quad input: _splatted4
                half_h, half_w = H // 2, W // 2
                left_eye_original = splatted_tensor[:, :, :half_h, :half_w]
                mask_raw = splatted_tensor[:, :, half_h:, :half_w]
                warped_original = splatted_tensor[:, :, half_h:, half_w:]

            # 5. Store the frames for the preview function
            self.preview_inpainted_frame = inpainted_tensor
            self.preview_original_frame = left_eye_original # This is the left eye for SBS
            self.preview_warped_frame = warped_original # Store the warped frame
            
            # Convert mask to grayscale and store
            mask_frame_np = mask_raw.squeeze(0).permute(1, 2, 0).cpu().numpy()
            mask_gray_np = np.mean(mask_frame_np, axis=2)
            self.preview_mask_frame = torch.from_numpy(mask_gray_np).float().unsqueeze(0).unsqueeze(0)
            # --- END: NEW LOGIC ---

            self.update_preview()
            self.update_status_label(f"Preview loaded for: {base_name}")

        except Exception as e:
            messagebox.showerror("Preview Load Error", f"Failed to load files for preview:\n\n{e}")
            self.update_status_label("Preview load failed.")

    def update_preview(self):
        """Processes and displays the preview frame based on current slider values."""
        # Add checks to ensure all required preview frames are loaded.
        if self.preview_inpainted_frame is None or self.preview_warped_frame is None or self.preview_mask_frame is None:
            return

        try:
            # Get settings
            settings = self.get_current_settings()
            if settings is None: return

            use_gpu = settings["use_gpu"] and torch.cuda.is_available()
            device = "cuda" if use_gpu else "cpu"

            # Move tensors to device
            mask = self.preview_mask_frame.to(device)
            inpainted = self.preview_inpainted_frame.to(device)
            # The warped frame is the correct background for blending
            right_eye_original = self.preview_warped_frame.to(device)

            # --- FIX: Upscale low-res inpainted frame and mask to match hi-res warped frame ---
            hires_H, hires_W = right_eye_original.shape[2], right_eye_original.shape[3]
            if inpainted.shape[2] != hires_H or inpainted.shape[3] != hires_W:
                logger.debug(f"Upscaling preview frames from {inpainted.shape[3]}x{inpainted.shape[2]} to {hires_W}x{hires_H}")
                inpainted = F.interpolate(inpainted, size=(hires_H, hires_W), mode='bicubic', align_corners=False)
                mask = F.interpolate(mask, size=(hires_H, hires_W), mode='bilinear', align_corners=False)

            # --- Process the mask (using a simplified chain from test.py) ---
            processed_mask = mask.clone()
            if settings["dilate_kernel"] > 0:
                processed_mask = apply_mask_dilation(processed_mask, settings["dilate_kernel"], use_gpu)
            if settings["blur_kernel"] > 0:
                processed_mask = apply_gaussian_blur(processed_mask, settings["blur_kernel"], use_gpu)
            if settings["stamp_shift"] > 0:
                processed_mask = apply_stamp_blur(processed_mask, settings["stamp_shift"], settings["stamp_start_opacity"], settings["stamp_opacity_decay"], settings["stamp_min_opacity"], settings["stamp_decay_gamma"], use_gpu)

            # --- Blend the frames ---
            blended_frame = right_eye_original * (1 - processed_mask) + inpainted * processed_mask

            # --- Convert to displayable image ---
            blended_np = (blended_frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Resize for display if too large, maintaining aspect ratio
            max_h, max_w = 512, 512
            if blended_np.shape[0] > max_h or blended_np.shape[1] > max_w:
                pil_img = Image.fromarray(blended_np)
                pil_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
                self.preview_image_tk = ImageTk.PhotoImage(pil_img)
            else:
                self.preview_image_tk = ImageTk.PhotoImage(Image.fromarray(blended_np))

            self.preview_label.config(image=self.preview_image_tk)

        except Exception as e:
            logger.error(f"Error updating preview: {e}", exc_info=True)
            self.status_label_var.set("Error updating preview.")

    def save_config(self):
        """Saves the current GUI configuration to a JSON file."""
        config = {
            "inpainted_folder": self.inpainted_folder_var.get(),
            "original_folder": self.original_folder_var.get(),
            "mask_folder": self.mask_folder_var.get(),
            "output_folder": self.output_folder_var.get(),
            "use_gpu": self.use_gpu_var.get(),
            "use_sbs_output": self.use_sbs_output_var.get(),
            "mask_dilate_kernel_size": str(self.mask_dilate_kernel_size_var.get()),
            "mask_blur_kernel_size": str(self.mask_blur_kernel_size_var.get()),
            "stamp_shift": str(self.stamp_shift_var.get()),
            "stamp_start_opacity": str(self.stamp_start_opacity_var.get()),
            "stamp_opacity_decay": str(self.stamp_opacity_decay_var.get()),
            "stamp_min_opacity": str(self.stamp_min_opacity_var.get()),
            "stamp_decay_gamma": str(self.stamp_decay_gamma_var.get()),
        }
        try:
            with open("config_merging.json", "w") as f:
                json.dump(config, f, indent=4)
            logger.info("Merging GUI configuration saved.")
        except Exception as e:
            logger.error(f"Failed to save merging GUI config: {e}")

    def load_config(self):
        """Loads configuration from a JSON file."""
        try:
            with open("config_merging.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.error(f"Failed to load merging GUI config: {e}")
            return {}

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
