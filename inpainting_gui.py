import os
import glob
import json
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Toplevel, Label
from ttkthemes import ThemedTk
from typing import Optional, Tuple, Callable 

import numpy as np
import torch
from decord import VideoReader, cpu
# FlashAttention requires optional dependency; attempt safe imports

import torch.nn.functional as F
import time
import subprocess # NEW: For running ffprobe and ffmpeg
import cv2 # NEW: For saving 16-bit PNGs
import logging

from dependency.stereocrafter_util import Tooltip, logger, get_video_stream_info, draw_progress_bar, release_cuda_memory, set_util_logger_level, encode_frames_to_mp4
from pipelines.stereo_video_inpainting import (
    StableVideoDiffusionInpaintingPipeline,
    tensor2vid,
    load_inpainting_pipeline
)

GUI_VERSION = "25.10.09"

# torch.backends.cudnn.benchmark = True

class InpaintingGUI(ThemedTk):    
    def __init__(self):
        super().__init__(theme="clam")
        self.title(f"Stereocrafter Inpainting (Batch) {GUI_VERSION}")   
        self.app_config = self.load_config()
        self.help_data = self.load_help_data()

        self.dark_mode_var = tk.BooleanVar(value=self.app_config.get("dark_mode_enabled", False))
        # Window size and position variables
        # Load from config or use defaults
        self.window_x = self.app_config.get("window_x", None)
        self.window_y = self.app_config.get("window_y", None)
        self.window_width = self.app_config.get("window_width", 550)
        
        self._is_startup = True
        self.debug_mode_var = tk.BooleanVar(value=self.app_config.get("debug_mode_enabled", False))

        self.input_folder_var = tk.StringVar(value=self.app_config.get("input_folder", "./output_splatted"))
        self.output_folder_var = tk.StringVar(value=self.app_config.get("output_folder", "./completed_output"))
        self.num_inference_steps_var = tk.StringVar(value=str(self.app_config.get("num_inference_steps", 5)))
        self.tile_num_var = tk.StringVar(value=str(self.app_config.get("tile_num", 2)))
        self.frames_chunk_var = tk.StringVar(value=str(self.app_config.get("frames_chunk", 23)))
        self.overlap_var = tk.StringVar(value=str(self.app_config.get("frame_overlap", 3)))
        self.original_input_blend_strength_var = tk.StringVar(value=str(self.app_config.get("original_input_blend_strength", 0.0)))
        self.output_crf_var = tk.StringVar(value=str(self.app_config.get("output_crf", 23)))
        self.process_length_var = tk.StringVar(value=str(self.app_config.get("process_length", -1)))
        self.offload_type_var = tk.StringVar(value=self.app_config.get("offload_type", "model"))
        self.hires_blend_folder_var = tk.StringVar(value=self.app_config.get("hires_blend_folder", "./output_splatted_hires"))
        
        # --- NEW: Granular Mask Processing Toggles & Parameters (Full Pipeline) ---
        self.mask_initial_threshold_var = tk.StringVar(value=str(self.app_config.get("mask_initial_threshold", 0.3)))
        self.mask_morph_kernel_size_var = tk.StringVar(value=str(self.app_config.get("mask_morph_kernel_size", 0.0)))
        self.mask_dilate_kernel_size_var = tk.StringVar(value=str(self.app_config.get("mask_dilate_kernel_size", 5)))        
        self.mask_blur_kernel_size_var = tk.StringVar(value=str(self.app_config.get("mask_blur_kernel_size", 10)))

        self.enable_post_inpainting_blend = tk.BooleanVar(value=self.app_config.get("enable_post_inpainting_blend", False))
        self.enable_color_transfer = tk.BooleanVar(value=self.app_config.get("enable_color_transfer", True))
        
        self.processed_count = tk.IntVar(value=0)
        self.total_videos = tk.IntVar(value=0)
        self.stop_event = threading.Event()
        self.pipeline = None
        self.video_name_var = tk.StringVar(value="N/A")
        self.video_res_var = tk.StringVar(value="N/A")
        self.video_frames_var = tk.StringVar(value="N/A")
        self.video_overlap_var = tk.StringVar(value="N/A")
        self.video_bias_var = tk.StringVar(value="N/A")

        self.mask_param_widgets = [] 

        self.create_widgets()
        self.style = ttk.Style()
        
        self.update_idletasks() 
        self._apply_theme(is_startup=True) 
        self._set_saved_geometry()
        self._is_startup = False 
        self._configure_logging() 

        self.update_progress()
        self.update_status_label("Ready")
        self.protocol("WM_DELETE_WINDOW", self.exit_application)
        self.after(0, self._set_saved_geometry)

    def _apply_color_transfer(self, source_frame: torch.Tensor, target_frame: torch.Tensor) -> torch.Tensor:
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

            # Compute mean and standard deviation of each channel for source and target
            # cv2.meanStdDev returns 2D arrays, reshape to 1D for easier handling
            src_mean, src_std = cv2.meanStdDev(source_lab)
            tgt_mean, tgt_std = cv2.meanStdDev(target_lab)

            src_mean = src_mean.flatten()
            src_std = src_std.flatten()
            tgt_mean = tgt_mean.flatten()
            tgt_std = tgt_std.flatten()

            # Ensure no division by zero by replacing zero std with a small value
            src_std = np.clip(src_std, 1e-6, None)
            tgt_std = np.clip(tgt_std, 1e-6, None)

            # Normalize target LAB channels based on source statistics
            target_lab_float = target_lab.astype(np.float32)
            for i in range(3): # For L, A, B channels
                target_lab_float[:, :, i] = (target_lab_float[:, :, i] - tgt_mean[i]) / tgt_std[i] * src_std[i] + src_mean[i]

            # Clip values to valid LAB range [0, 255] and convert back to uint8
            target_lab_float = np.clip(target_lab_float, 0, 255)
            adjusted_lab_uint8 = target_lab_float.astype(np.uint8)

            # Convert back to RGB
            adjusted_rgb = cv2.cvtColor(adjusted_lab_uint8, cv2.COLOR_LAB2RGB)

            # Convert back to tensor [C, H, W] in [0, 1]
            adjusted_tensor = torch.from_numpy(adjusted_rgb).permute(2, 0, 1).float() / 255.0

            return adjusted_tensor
        except Exception as e:
            logger.error(f"Error during color transfer: {e}. Returning original target frame.", exc_info=True)
            return target_frame
    
    def _apply_directional_dilation(self, frame_chunk: torch.Tensor, mask_chunk: torch.Tensor) -> torch.Tensor:
        """
        Fills occluded areas in a warped frame chunk (float [0,1], [T, C, H, W]) 
        by dilating/growing valid pixels from the right (background side) using OpenCV.
        The result is a frame chunk with clean color statistics for transfer.
        
        Args:
            frame_chunk: The warped frames with black occlusion holes.
            mask_chunk: The mask (float [0,1], [T, 1, H, W]) indicating occluded areas.
            
        Returns:
            The occlusion-filled frame chunk.
        """
        try:
            # We want to fill the black holes in the frame_chunk (warped image)
            # using the mask_chunk (processed mask).

            if frame_chunk.shape[0] != mask_chunk.shape[0]:
                logger.error("Frame and mask chunks must have the same temporal dimension.")
                return frame_chunk
                
            # Define the directional kernel for right-to-left growth
            # A (1, 5) or (1, 7) kernel biases the fill horizontally from the right
            # We use a non-symmetric kernel that is wider on the right side of the center.
            # Example: [0, 0, 1, 1, 1, 1, 1] means 2 pixels left, 4 pixels right (or similar)
            # For pure right-to-left growth, a simple horizontal line kernel is sufficient for dilation
            kernel_width = 7 # Adjust as needed (e.g., 5, 7, 9)
            kernel = np.zeros((1, kernel_width), dtype=np.uint8)
            # Set the pixel on the right-most side (or center) to 1 to promote growth from that direction
            kernel[0, kernel_width - 1] = 1 # Simple 1xN kernel, just for basic right-to-left bias

            # Use a slightly more complex kernel for stronger right bias.
            # Example: a 1x7 kernel where the right side has a higher influence
            # kernel = np.array([[0, 0, 0, 0, 1, 1, 1]], dtype=np.uint8) 
            # Dilation with a horizontal kernel:
            # We will use the simplest, most effective one: a 1xN horizontal kernel on the valid (non-masked) pixels.
            
            # Convert tensors to numpy, scale to 0-255, and convert to BGR (OpenCV)
            # Since the tensor is already float 0-1, we need to get the RGB image and the mask.
            
            filled_frames_list = []
            device = frame_chunk.device
            
            for t in range(frame_chunk.shape[0]):
                frame_np = (frame_chunk[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) # [H, W, C] RGB
                mask_np = (mask_chunk[t].squeeze(0).cpu().numpy() * 255).astype(np.uint8) # [H, W] grayscale mask
                
                # Invert the mask: The mask defines the hole (1) but we want to dilate the *source* (0)
                # No, wait, we want to fill the hole. We need to create a kernel/operation that expands the
                # non-zero (valid) areas into the zero (black) areas.
                
                # Method: Simple Image Dilation on the entire image
                # Dilation will grow bright pixels into dark ones.
                # Since the "hole" is black (0) and the "valid area" is colored (>0),
                # dilating the *image* will fill the black hole with surrounding color.
                
                # The kernel is now a simple horizontal line (e.g., 1x7)
                dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))

                # Apply dilation to the BGR frame (OpenCV uses BGR by default)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                
                # Apply dilation on the entire image. This will expand the texture from the right/left/etc.
                # but with a strong horizontal bias from the 1xN kernel.
                dilated_frame_bgr = cv2.dilate(frame_bgr, dilation_kernel, iterations=5) # 5 iterations for more fill depth

                # The key is to only replace the masked (black) areas with the dilated result.
                
                # 1. Prepare mask for blending: HWC, 3 channels, float [0,1]
                mask_float = mask_np.astype(np.float32) / 255.0
                mask_3ch = np.stack([mask_float] * 3, axis=2) # [H, W, 3]
                
                # 2. Convert back to RGB
                dilated_frame_rgb = cv2.cvtColor(dilated_frame_bgr, cv2.COLOR_BGR2RGB)
                
                # 3. Blend: (Original Frame * (1 - Mask)) + (Dilated Frame * Mask)
                # This keeps the original valid pixels and only uses the dilated content for the hole.
                
                # Both np arrays are HWC, uint8
                original_frame_np = frame_np 
                dilated_frame_np = dilated_frame_rgb 
                
                # Use a slightly softer mask for blending by applying a small blur to the mask
                mask_for_blend = cv2.GaussianBlur(mask_float, (21, 21), 0) # Apply blur to the 1-channel mask
                mask_3ch_blend = np.stack([mask_for_blend] * 3, axis=2) # [H, W, 3]

                # Blend (still in uint8 or convert to float 0-1)
                # It's cleaner to convert to float 0-1 for blending
                original_frame_float = original_frame_np.astype(np.float32) / 255.0
                dilated_frame_float = dilated_frame_np.astype(np.float32) / 255.0
                
                blended_float = (original_frame_float * (1 - mask_3ch_blend)) + (dilated_frame_float * mask_3ch_blend)
                
                # Convert back to tensor [C, H, W] float [0, 1]
                blended_tensor = torch.from_numpy(blended_float).permute(2, 0, 1).float().to(device)
                filled_frames_list.append(blended_tensor)

            logger.debug(f"Applied directional dilation with {kernel_width}x1 kernel for color reference.")
            return torch.stack(filled_frames_list)
            
        except Exception as e:
            logger.error(f"Error during directional dilation for color transfer reference: {e}. Returning original frames.", exc_info=True)
            return frame_chunk
    
    def _apply_gaussian_blur(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Applies Gaussian blur to the mask using separate 1D convolutions for X and Y.
        Expects mask in [T, C, H, W] format, where C=1.
        """
        try:

            # sets kernel_size, sigma is derived ---
            if kernel_size <= 0:
                logger.warning(f"Invalid blur kernel size ({kernel_size}). Skipping blur.")
                return mask
            
            # Ensure kernel size is odd
            kernel_val = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            if kernel_val < 3: # Enforce minimum odd kernel size
                kernel_val = 3

            sigma = kernel_val / 6.0 # Derive sigma from kernel_size
            if sigma < 0.1: # Minimum sensible sigma
                sigma = 0.1

            kernel_x = self._create_1d_gaussian_kernel(kernel_val, sigma).to(mask.device) # Use derived kernel_val and sigma
            kernel_y = self._create_1d_gaussian_kernel(kernel_val, sigma).to(mask.device) # Use derived kernel_val and sigma

            kernel_x = kernel_x.view(1, 1, 1, kernel_val)
            kernel_y = kernel_y.view(1, 1, kernel_val, 1)

            padding_x = kernel_val // 2
            blurred_mask = F.conv2d(mask, kernel_x, padding=(0, padding_x), groups=mask.shape[1])
            
            padding_y = kernel_val // 2
            blurred_mask = F.conv2d(blurred_mask, kernel_y, padding=(padding_y, 0), groups=mask.shape[1])
            
            logger.debug(f"Applied Gaussian blur with kernel {kernel_val}x{kernel_val} (derived sigma {sigma:.2f}).") # Updated log message
            return torch.clamp(blurred_mask, 0.0, 1.0)
        except ValueError:
            logger.error("Invalid input for mask blur parameters. Skipping blur.", exc_info=True)
            return mask
        except Exception as e:
            logger.error(f"Error during mask blurring: {e}. Skipping blur.", exc_info=True)
            return mask
    
    def _apply_mask_dilation(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Applies dilation to the mask using max pooling.
        Expects mask in [T, C, H, W] format, where C=1.
        # """
        # if not self.enable_mask_processing.get():
        #     return mask

        try:
            
            # Ensure kernel size is positive and odd for symmetry
            if kernel_size <= 0:
                logger.warning(f"Invalid dilation kernel size ({kernel_size}). Skipping dilation.")
                return mask
            
            kernel_val = kernel_size if kernel_size % 2 == 1 else kernel_size + 1 # Ensure odd
            
            dilated_mask = F.max_pool2d(
                mask,
                kernel_size=(kernel_val, kernel_val), # Use single kernel_val for both dimensions
                stride=1,
                padding=(kernel_val // 2, kernel_val // 2)
            )
            logger.debug(f"Applied mask dilation with kernel ({kernel_val}x{kernel_val}).") # Updated log message
            return dilated_mask
        except ValueError:
            logger.error("Invalid input for mask dilation kernel size. Skipping dilation.", exc_info=True) # Updated log message
            return mask
        except Exception as e:
            logger.error(f"Error during mask dilation: {e}. Skipping dilation.", exc_info=True)
            return mask
    
    def _apply_morphological_closing(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Applies morphological closing to fill small holes and smooth boundaries.
        Expects mask in [T, C, H, W] float [0, 1] format on the GPU.
        Returns processed mask in the same format.
        """
        try:
            if kernel_size <= 0:
                logger.warning(f"Invalid morphological closing kernel size ({kernel_size}). Skipping closing.")
                return mask
            
            # Ensure kernel_size is odd for symmetry
            kernel_val = kernel_size if kernel_size % 2 == 1 else kernel_size + 1 
            padding = kernel_val // 2

            # Dilation step of closing
            dilated_mask = F.max_pool2d(
                mask, kernel_size=kernel_val, stride=1, padding=padding
            )
            
            # Erosion step of closing (using min_pool, which is equivalent to erosion on a binary mask)
            # Note: min_pool is not a standard PyTorch function, but erosion is F.max_pool on the inverted mask.
            eroded_mask = 1.0 - F.max_pool2d(
                1.0 - dilated_mask, kernel_size=kernel_val, stride=1, padding=padding
            )
            
            logger.debug(f"Applied morphological closing with kernel ({kernel_val}x{kernel_val}).") # Updated log message
            return eroded_mask
        except ValueError:
            logger.error("Invalid input for morphological closing kernel size. Skipping closing.", exc_info=True) # Updated log message
            return mask
        except Exception as e:
            logger.error(f"Error during morphological closing: {e}. Skipping closing.", exc_info=True)
            return mask
        
    def _apply_post_inpainting_blend(
        self,
        inpainted_frames: torch.Tensor,       # Generated frames from pipeline
        original_warped_frames: torch.Tensor, # Original warped frames (bottom-right)
        mask: torch.Tensor,                    # Processed mask (dilated, blurred)        
        base_video_name: str
    ) -> torch.Tensor:
        """
        Blends the inpainted frames with the original warped frames using the mask.
        Ensures all input tensors are on CPU and have matching shapes before blending.
        Expected format: [T, C, H, W] float [0, 1].
        """
        if not self.enable_post_inpainting_blend.get():
            return inpainted_frames

        # Check if temporal (T) and spatial (H, W) dimensions match
        if (inpainted_frames.shape[0] != original_warped_frames.shape[0] or
            inpainted_frames.shape[2] != original_warped_frames.shape[2] or
            inpainted_frames.shape[3] != original_warped_frames.shape[3]):
            logger.error(f"Temporal or Spatial shape mismatch for post-inpainting blend: Inpainted {inpainted_frames.shape} vs Original Warped {original_warped_frames.shape}. Skipping blend.")
            return inpainted_frames

        if (inpainted_frames.shape[0] != mask.shape[0] or
            inpainted_frames.shape[2] != mask.shape[2] or
            inpainted_frames.shape[3] != mask.shape[3] or
            mask.shape[1] != 1): # Explicitly check mask has 1 channel
            logger.error(f"Mask shape mismatch for post-inpainting blend: Inpainted {inpainted_frames.shape} vs Mask {mask.shape} (Mask must be 1-channel). Skipping blend.")
            return inpainted_frames

        try:
            # Ensure tensors are on CPU for blending if not already (they should be after previous steps)
            inpainted_frames_cpu = inpainted_frames.cpu()
            original_warped_frames_cpu = original_warped_frames.cpu()
            mask_cpu = mask.cpu()

            # Ensure mask is single channel for broadcasting if needed (though it should be [T, 1, H, W])
            if mask_cpu.shape[1] != 1:
                logger.warning(f"Mask has {mask_cpu.shape[1]} channels for blending, expecting 1. Using mean for blending if necessary.")
                mask_blend = mask_cpu.mean(dim=1, keepdim=True)
            else:
                mask_blend = mask_cpu
            
            # Blend: original content where mask is 0, inpainted content where mask is 1, smooth blend in between
            blended_frames = original_warped_frames_cpu * (1 - mask_blend) + inpainted_frames_cpu * mask_blend
            
            logger.debug("Applied post-inpainting blending.")

            # --- MODIFIED: TEMPORARY DEBUG CODE START (now conditional) ---
            if self.debug_mode_var.get():
                debug_output_dir = os.path.join(self.output_folder_var.get(), "debug_blend")
                os.makedirs(debug_output_dir, exist_ok=True)
                # MODIFIED: Use base_video_name directly
                video_basename_for_debug_blend = os.path.splitext(base_video_name)[0] 

                for t in range(min(5, inpainted_frames_cpu.shape[0])):
                    original_warped_img = (original_warped_frames_cpu[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    inpainted_img = (inpainted_frames_cpu[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    mask_img = (mask_blend[t].squeeze(0).numpy() * 255).astype(np.uint8)
                    blended_img = (blended_frames[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                    cv2.imwrite(os.path.join(debug_output_dir, f"{video_basename_for_debug_blend}_frame_{t:04d}_original_warped.png"), cv2.cvtColor(original_warped_img, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(debug_output_dir, f"{video_basename_for_debug_blend}_frame_{t:04d}_inpainted.png"), cv2.cvtColor(inpainted_img, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(debug_output_dir, f"{video_basename_for_debug_blend}_frame_{t:04d}_mask.png"), mask_img)
                    cv2.imwrite(os.path.join(debug_output_dir, f"{video_basename_for_debug_blend}_frame_{t:04d}_blended.png"), cv2.cvtColor(blended_img, cv2.COLOR_RGB2BGR))
                logger.debug(f"Saved debug blend frames to {debug_output_dir}")
            # --- END MODIFIED TEMPORARY DEBUG CODE ---

            return blended_frames
        except Exception as e:
            logger.error(f"Error during post-inpainting blending: {e}. Returning original inpainted frames.", exc_info=True)
            return inpainted_frames
    
    def _apply_theme(self: "InpaintingGUI", is_startup: bool = False):
        """Applies the selected theme (dark or light) to the GUI, and adjusts window height."""
        if self.dark_mode_var.get():
            # --- Dark Theme ---
            bg_color = "#2b2b2b" # Background for root and tk.Label
            fg_color = "white"   # Foreground for tk.Label text
            entry_field_bg = "#3c3c3c" # Background for ttk.Entry field
            
            self.style.theme_use("black")
            self.configure(bg=bg_color)

            # Menu bar styling (tk.Menu widgets)
            if hasattr(self, 'menubar'): # Check if menu widgets exist yet
                menu_bg = "#3c3c3c"
                menu_fg = "white"
                active_bg = "#555555"
                active_fg = "white"

                self.menubar.config(bg=menu_bg, fg=menu_fg, activebackground=active_bg, activeforeground=active_fg)
                self.file_menu.config(bg=menu_bg, fg=menu_fg, activebackground=active_bg, activeforeground=active_fg)
                self.help_menu.config(bg=menu_bg, fg=menu_fg, activebackground=active_bg, activeforeground=active_fg)
            
            # ttk.Entry widget styling
            self.style.configure("TEntry", fieldbackground=entry_field_bg, foreground=fg_color, insertcolor=fg_color)
            self.style.configure("TFrame", background=bg_color, foreground=fg_color)
            self.style.configure("TLabelframe", background=bg_color, foreground=fg_color)
            self.style.configure("TLabelframe.Label", background=bg_color, foreground=fg_color) # For the title text

            # ttk.Label styling (for all ttk.Label widgets including the info frame ones)
            self.style.configure("TLabel", background=bg_color, foreground=fg_color)

        else:
            # --- Light Theme ---
            bg_color = "#d9d9d9"
            fg_color = "black"
            entry_field_bg = "#f0f0f0"

            self.style.theme_use("default")
            self.configure(bg=bg_color)

            # Menu bar styling (tk.Menu widgets)
            if hasattr(self, 'menubar'):
                menu_bg = "#f0f0f0"
                menu_fg = "black"
                active_bg = "#dddddd"
                active_fg = "black"
                self.menubar.config(bg=menu_bg, fg=menu_fg, activebackground=active_bg, activeforeground=active_fg)
                self.file_menu.config(bg=menu_bg, fg=menu_fg, activebackground=active_bg, activeforeground=active_fg)
                self.help_menu.config(bg=menu_bg, fg=menu_fg, activebackground=active_bg, activeforeground=active_fg)

            self.style.configure("TEntry", fieldbackground=entry_field_bg, foreground=fg_color, insertcolor=fg_color)
            self.style.configure("TFrame", background=bg_color, foreground=fg_color)
            self.style.configure("TLabelframe", background=bg_color, foreground=fg_color)
            self.style.configure("TLabelframe.Label", background=bg_color, foreground=fg_color)
            self.style.configure("TLabel", background=bg_color, foreground=fg_color)

        self.update_idletasks() # Ensure all theme changes are rendered for accurate reqheight

        # --- Apply geometry only if not during startup ---
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

    def _browse_hires_folder(self):
        folder = filedialog.askdirectory(initialdir=self.hires_blend_folder_var.get())
        if folder:
            self.hires_blend_folder_var.set(folder)

    def _browse_input(self):
        folder = filedialog.askdirectory(initialdir=self.input_folder_var.get())
        if folder:
            self.input_folder_var.set(folder)

    def _browse_output(self):
        folder = filedialog.askdirectory(initialdir=self.output_folder_var.get())
        if folder:
            self.output_folder_var.set(folder)

    def _create_1d_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        Creates a 1D Gaussian kernel.
        """
        if kernel_size <= 0 or sigma <= 0:
            logger.warning(f"Invalid kernel_size ({kernel_size}) or sigma ({sigma}) for Gaussian kernel. Returning identity.")
            # Return a kernel that effectively does nothing
            identity_kernel = torch.zeros(kernel_size)
            if kernel_size > 0:
                identity_kernel[kernel_size // 2] = 1.0 # Central pixel is 1
            return identity_kernel.unsqueeze(0).unsqueeze(0) # Shape (1, 1, kernel_size) for conv1d

        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        gauss = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        kernel = gauss / gauss.sum()
        return kernel

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

        set_util_logger_level(level) # Call the function from stereocrafter_util.py
        logger.info(f"Logging level set to {logging.getLevelName(level)}.")
    
    def _finalize_output_frames(
        self,
        inpainted_frames: torch.Tensor,
        mask_frames: torch.Tensor,
        original_warped_frames: torch.Tensor,
        original_left_frames: Optional[torch.Tensor],
        hires_data: dict,
        base_video_name: str,
        is_dual_input: bool,
    ) -> Optional[torch.Tensor]:
        """
        Applies Hi-Res upscaling/blending (if enabled), Color Transfer, and final SBS concatenation.
        Returns the final tensor for encoding, or None on error.
        """
        frames_output_final = inpainted_frames
        frames_mask_processed = mask_frames
        frames_warpped_original_unpadded_normalized = original_warped_frames
        frames_left_original_cropped = original_left_frames
        
        # --- START: HI-RES BLENDING (If Enabled) ---
        if hires_data["is_hires_blend_enabled"]:
            hires_H, hires_W = hires_data["hires_H"], hires_data["hires_W"]
            num_frames_original = frames_output_final.shape[0]

            logger.info(f"Starting Hi-Res Blending at {hires_W}x{hires_H}...")
            
            # 1. Upscale Low-Res Inpainted Output
            frames_output_final_hires = F.interpolate(
                frames_output_final, size=(hires_H, hires_W), mode='bicubic', align_corners=False
            ).to(frames_output_final.device)

            # 2. Upscale Low-Res Mask (Bilinear for smoother blend)
            frames_mask_processed_unpadded_original_length_hires = F.interpolate(
                frames_mask_processed, size=(hires_H, hires_W), mode='bilinear', align_corners=False
            ).to(frames_output_final.device)
            
            # 3. Load High-Res Original/Warped Frames (The risk is here, but necessary with current read_video_frames)
            # hires_video_path is guaranteed to be str if is_hires_blend_enabled is True here
            hires_full_frames_torch, _, _ = read_video_frames(hires_data["hires_video_path"]) 
            hires_full_frames_torch = hires_full_frames_torch[:num_frames_original] # Crop HR video

            if is_dual_input:
                half_w_hires = hires_full_frames_torch.shape[3] // 2
                hires_warped_for_blend = hires_full_frames_torch[:, :, :, half_w_hires:] / 255.0
                hires_left_for_concat = None
            else:
                half_h_hires = hires_full_frames_torch.shape[2] // 2
                half_w_hires = hires_full_frames_torch.shape[3] // 2
                hires_left_for_concat = hires_full_frames_torch[:, :, :half_h_hires, :half_w_hires] / 255.0
                hires_warped_for_blend = hires_full_frames_torch[:, :, half_h_hires:, half_w_hires:] / 255.0
            
            # 4. Overwrite Low-Res Variables with Hi-Res Upscaled/Loaded Variables
            frames_output_final = frames_output_final_hires
            frames_mask_processed = frames_mask_processed_unpadded_original_length_hires
            frames_warpped_original_unpadded_normalized = hires_warped_for_blend.to(frames_output_final.device)
            frames_left_original_cropped = hires_left_for_concat.to(frames_output_final.device) if hires_left_for_concat is not None else None
            
            logger.info("Hi-Res Upscaling and Data Overwrite complete.")
            
        # --- END: HI-RES BLENDING ---

        # The rest of the logic remains largely the same, but uses the now-guaranteed-to-be-set frames_output_final
        
        # --- Apply Color Transfer (if enabled) ---
        if self.enable_color_transfer.get():
            # ... (Color Transfer logic using frames_output_final, frames_mask_processed, etc.) ...
            # ... (Replace the large Color Transfer block in your code with its body using the simplified variable names) ...
            reference_frames_for_transfer: Optional[torch.Tensor] = None

            if is_dual_input:
                # DUAL Input: Create an occlusion-free reference from the warped frames (bottom-right)
                logger.debug("Dual input detected. Creating occlusion-free reference via directional dilation for color transfer...")
                
                warped_frames_base = frames_warpped_original_unpadded_normalized.cpu() 
                processed_mask = frames_mask_processed.cpu() 
                
                reference_frames_for_transfer = self._apply_directional_dilation(
                    frame_chunk=warped_frames_base, mask_chunk=processed_mask
                ).to(frames_output_final.device)

                if self.debug_mode_var.get():
                    debug_output_dir = os.path.join(self.output_folder_var.get(), "debug_color_ref")
                    os.makedirs(debug_output_dir, exist_ok=True)
                    video_basename_for_debug = base_video_name.rsplit('.', 1)[0]
                    for t in range(min(5, reference_frames_for_transfer.shape[0])):
                        ref_img = (reference_frames_for_transfer[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        cv2.imwrite(
                            os.path.join(debug_output_dir, f"{video_basename_for_debug}_frame_{t:04d}_color_ref_dilated.png"), 
                            cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR)
                        )
                    logger.debug(f"Saved debug color reference frames to {debug_output_dir}")
                
            else: 
                reference_frames_for_transfer = frames_left_original_cropped
                
            # --- Perform the Color Transfer ---
            if reference_frames_for_transfer is None or reference_frames_for_transfer.numel() == 0:
                logger.warning("Color transfer skipped: No valid reference frames available.")
            else:
                logger.debug("Applying color transfer from reference view to inpainted right view...")
                target_H, target_W = frames_output_final.shape[2], frames_output_final.shape[3]
                adjusted_frames_output = []
                for t in range(frames_output_final.shape[0]):
                    ref_frame_resized = F.interpolate(
                        reference_frames_for_transfer[t].unsqueeze(0),
                        size=(target_H, target_W),
                        mode='bilinear', align_corners=False
                    ).squeeze(0).cpu()
                    target_frame_cpu = frames_output_final[t].cpu()
                    adjusted_frame = self._apply_color_transfer(ref_frame_resized, target_frame_cpu)
                    adjusted_frames_output.append(adjusted_frame.to(frames_output_final.device))
                
                frames_output_final = torch.stack(adjusted_frames_output)
                logger.debug("Color transfer complete.")
        # --- END Apply Color Transfer ---


        # --- Apply Post-Inpainting Blending (if enabled) ---
        if self.enable_post_inpainting_blend.get():
            logger.debug("Applying post-inpainting blend...")
            frames_output_final = self._apply_post_inpainting_blend(
                inpainted_frames=frames_output_final,
                original_warped_frames=frames_warpped_original_unpadded_normalized,
                mask=frames_mask_processed, # Note: using the simplified variable name
                base_video_name=base_video_name 
            )
            logger.debug("Post-inpainting blend complete.")

        # --- Final Concatenation ---
        final_output_frames_for_encoding: Optional[torch.Tensor] = None

        if is_dual_input:
            final_output_frames_for_encoding = frames_output_final
        else:
            if frames_left_original_cropped is None or frames_left_original_cropped.numel() == 0:
                logger.error(f"Original left frames are missing or empty for non-dual input {base_video_name}. Cannot create SBS output.")
                return None
                    
            if frames_left_original_cropped.shape[0] != frames_output_final.shape[0] or \
            frames_left_original_cropped.shape[1] != frames_output_final.shape[1] or \
            frames_left_original_cropped.shape[2] != frames_output_final.shape[2]:
                logger.error(f"Dimension mismatch for SBS concatenation: Left {frames_left_original_cropped.shape}, Inpainted {frames_output_final.shape} for {base_video_name}.")
                return None

            sbs_frames = torch.cat([frames_left_original_cropped, frames_output_final], dim=3)
            final_output_frames_for_encoding = sbs_frames

        # Final check: ensure the tensor to be encoded is actually populated
        if final_output_frames_for_encoding is None or final_output_frames_for_encoding.numel() == 0:
            logger.error(f"Final output frames for encoding are empty or None after preparation for {base_video_name}.")
            return None

        return final_output_frames_for_encoding
    
    def _find_high_res_match(self, low_res_video_path: str) -> Optional[str]:
        """
        Attempts to find a matching high-resolution splatted file in the hi-res folder.
        Applies safety checks.
        Returns the full path to the hi-res video or None.
        """
        low_res_input_folder = self.input_folder_var.get()
        hires_blend_folder = self.hires_blend_folder_var.get()

        logger.debug(f"Hires Check: Low-Res Path: {low_res_video_path}")
        logger.debug(f"Hires Check: Low-Res Folder: {low_res_input_folder}")
        logger.debug(f"Hires Check: Hi-Res Folder: {hires_blend_folder}")

        # Safety Check 1: Hires folder is the same as the low-res input folder
        if os.path.normpath(low_res_input_folder) == os.path.normpath(hires_blend_folder):
            logger.warning("Hi-Res Blend Folder is the same as Input Folder. Disabling Hi-Res blending.")
            return None
        
        # 1. Extract Base Name and Splatting Suffix
        low_res_filename = os.path.basename(low_res_video_path)
        low_res_name_without_ext = os.path.splitext(low_res_filename)[0]
        
        splatted_suffix = None
        if low_res_name_without_ext.endswith('_splatted2'):
            splatted_suffix = '_splatted2.mp4'
            splatted_core = '_splatted2'
        elif low_res_name_without_ext.endswith('_splatted4'):
            splatted_suffix = '_splatted4.mp4'
            splatted_core = '_splatted4'
        else:
            logger.warning(f"Could not parse splatting suffix from {low_res_filename}. Skipping Hi-Res match.")
            return None
        
        # --- NEW ULTRA-SIMPLIFIED NAME STRIPPING ---
        # The key is to strip the resolution number AND the splatting suffix.
        
        # Find the index of the splatted core (e.g., '_splatted2')
        splat_index = low_res_name_without_ext.rfind(splatted_core)
        if splat_index == -1:
             logger.warning(f"Failed to find splatted core in {low_res_name_without_ext}. Skipping Hi-Res match.")
             return None

        # Take everything before the splatted core, e.g., 'FSC-clips_crp_cropped-0006_640'
        name_core_with_dim = low_res_name_without_ext[:splat_index]
        
        # Find the last underscore, which precedes the dimension
        last_underscore_index = name_core_with_dim.rfind('_')
        
        if last_underscore_index == -1:
            # If no underscore is found (unlikely for your file names)
            base_pattern_no_dim = name_core_with_dim
        else:
            # Take everything up to the last underscore (removes the resolution number)
            # Result: 'FSC-clips_crp_cropped-0006'
            base_pattern_no_dim = name_core_with_dim[:last_underscore_index]
            
        if not base_pattern_no_dim:
            logger.warning(f"Failed to find true base name for {low_res_filename} after stripping resolution. Skipping Hi-Res match.")
            return None
        # --- END NEW ULTRA-SIMPLIFIED NAME STRIPPING ---

        # 2. Search Hi-Res Folder for Match
        search_pattern = os.path.join(hires_blend_folder, f"{base_pattern_no_dim}_*{splatted_suffix}")
        logger.debug(f"Hi-Res Search Pattern: {search_pattern}")
        matches = glob.glob(search_pattern)

        logger.debug(f"Hi-Res Glob Matches Found: {[os.path.basename(m) for m in matches]}")
        
        if not matches:
            logger.debug(f"No Hi-Res match found for {low_res_filename} in {hires_blend_folder}.")
            return None

        # Filter out the current low-res video if it somehow ended up in the search list
        matches = [m for m in matches if os.path.normpath(m) != os.path.normpath(low_res_video_path)]

        if len(matches) > 1:
            logger.warning(f"Multiple Hi-Res matches found for {low_res_filename}. Using the first match: {os.path.basename(matches[0])}")
            
        # 3. Final Path
        hires_path = matches[0] if matches else None
        
        # Safety Check 2: Check resolution equality (requires loading a frame)
        if hires_path:
            try:
                # 1. Get low-res width
                low_res_reader = VideoReader(low_res_video_path, ctx=cpu(0))
                low_res_w_raw = low_res_reader.get_batch([0]).shape[2] 
                del low_res_reader
                
                # 2. Get hi-res width
                hires_reader = VideoReader(hires_path, ctx=cpu(0))
                hires_w_raw = hires_reader.get_batch([0]).shape[2]
                del hires_reader
            except Exception as e:
                logger.error(f"Failed to read raw video width for resolution check: {e}")
                return None

            # --- NEW DEBUG LINE HERE ---
            logger.debug(f"Hires Check: Low-Res Raw Width: {low_res_w_raw} | Hi-Res Raw Width: {hires_w_raw}")
            # --- END NEW DEBUG LINE HERE ---
            
            if hires_w_raw <= low_res_w_raw: # Check if Hi-Res is NOT strictly higher resolution
                logger.warning(f"Hi-Res candidate {os.path.basename(hires_path)} ({hires_w_raw}px) is not higher resolution than Low-Res ({low_res_w_raw}px). Disabling Hi-Res blending.")
                return None
            
            logger.info(f"Found Hi-Res match: {os.path.basename(hires_path)} ({hires_w_raw}px).")
            return hires_path

        return None
    
    def _get_current_config(self):
        """Collects all current GUI variable values into a single dictionary."""
        config = {
            # Folder Configurations
            "input_folder": self.input_folder_var.get(),
            "output_folder": self.output_folder_var.get(),
            "hires_blend_folder": self.hires_blend_folder_var.get(),

            # GUI State Configurations
            "dark_mode_enabled": self.dark_mode_var.get(),
            "window_width": self.winfo_width(),
            "window_x": self.winfo_x(),
            "window_y": self.winfo_y(),
            
            # Parameter Configurations
            "num_inference_steps": self.num_inference_steps_var.get(),
            "tile_num": self.tile_num_var.get(),
            "process_length": self.process_length_var.get(),
            "frames_chunk": self.frames_chunk_var.get(),
            "frame_overlap": self.overlap_var.get(),
            "original_input_blend_strength": self.original_input_blend_strength_var.get(),            
            "output_crf": self.output_crf_var.get(),
            "offload_type": self.offload_type_var.get(),

            # --- Granular Mask Processing Toggles & Parameters (Full Pipeline) ---
            "mask_initial_threshold": self.mask_initial_threshold_var.get(),
            "mask_morph_kernel_size": self.mask_morph_kernel_size_var.get(),
            "mask_dilate_kernel_size": self.mask_dilate_kernel_size_var.get(),
            "mask_blur_kernel_size": self.mask_blur_kernel_size_var.get(),
            
            "enable_post_inpainting_blend": self.enable_post_inpainting_blend.get(),
            "enable_color_transfer": self.enable_color_transfer.get(),
        }
        return config
    
    def _prepare_video_inputs(
        self,
        input_video_path: str,
        base_video_name: str,
        is_dual_input: bool,
        frames_chunk: int,
        tile_num: int,
        update_info_callback: Optional[Callable],
        overlap: int, # Needed for display, not logic here
        original_input_blend_strength: float,
        process_length: int = -1
    ) -> Optional[Tuple[
        torch.Tensor,                  # frames_warpped_padded
        torch.Tensor,                  # frames_mask_padded
        Optional[torch.Tensor],        # frames_left_original_cropped
        int,                           # num_frames_original
        int,                           # padded_H
        int,                           # padded_W
        Optional[dict],                # video_stream_info
        float,                         # fps
        torch.Tensor,                  # frames_warpped_original_unpadded_normalized
        torch.Tensor                   # frames_mask_processed_unpadded_original_length
    ]]:
        """
        Helper method to prepare video inputs: loads frames, applies padding,
        validates dimensions, splits views, normalizes, and prepares for tiling.

        Returns: (frames_warpped_padded, frames_mask_padded, frames_left_original_cropped,
                  num_frames_original, padded_H, padded_W, video_stream_info)
                 or None if an error occurs.
        """
        frames, fps, video_stream_info = read_video_frames(input_video_path)

        # --- Process Length Logic ---
        total_frames_in_video = frames.shape[0]
        actual_frames_to_process_count = total_frames_in_video

        if process_length != -1 and process_length > 0:
            actual_frames_to_process_count = min(total_frames_in_video, process_length)
            logger.info(f"Limiting processing to first {actual_frames_to_process_count} frames (out of {total_frames_in_video}).")
        
        if actual_frames_to_process_count == 0:
            logger.warning(f"No frames to process in {input_video_path} (after applying process_length), skipping.")
            if update_info_callback:
                self.after(0, lambda: update_info_callback(base_video_name, "N/A", f"0 (out of {total_frames_in_video})", overlap, original_input_blend_strength))
            return None

        frames = frames[:actual_frames_to_process_count]
        num_frames_original = frames.shape[0]
        
        if num_frames_original == 0:
            logger.warning(f"No frames found in {input_video_path}, skipping.")
            if update_info_callback:
                self.after(0, lambda: update_info_callback(base_video_name, "N/A", "0 (skipped)", overlap, original_input_blend_strength))
            return None

        # --- Dimension Divisibility Check and Resizing (if needed) ---
        _, _, total_h_raw_input_before_resize, total_w_raw_input_before_resize = frames.shape
        required_divisor = 8

        new_h = total_h_raw_input_before_resize
        new_w = total_w_raw_input_before_resize

        if new_h % required_divisor != 0:
            new_h = (new_h // required_divisor + 1) * required_divisor
            logger.warning(f"Video height {total_h_raw_input_before_resize} is not divisible by {required_divisor}. Resizing to {new_h}.")

        if new_w % required_divisor != 0:
            new_w = (new_w // required_divisor + 1) * required_divisor
            logger.warning(f"Video width {total_w_raw_input_before_resize} is not divisible by {required_divisor}. Resizing to {new_w}.")

        if new_h != total_h_raw_input_before_resize or new_w != total_w_raw_input_before_resize:
            if frames.shape[0] > 0:
                frames = F.interpolate(frames, size=(new_h, new_w), mode='bicubic', align_corners=False)
                logger.info(f"Frames resized from {total_h_raw_input_before_resize}x{total_w_raw_input_before_resize} to {new_h}x{new_w}.")
            else:
                logger.warning("Attempted to resize empty frames tensor. Skipping resize.")
        
        # --- Update current dimensions after potential resize ---
        total_h_current, total_w_current = frames.shape[2], frames.shape[3]

        if total_h_current < required_divisor or total_w_current < required_divisor:
            error_msg = f"Video {base_video_name} is too small after resize ({total_w_current}x{total_h_current}), skipping."
            logger.error(error_msg)
            if update_info_callback:
                self.after(0, lambda: update_info_callback(base_video_name, f"{total_w_current}x{total_h_current} (INVALID)", num_frames_original, overlap, original_input_blend_strength))
            self.after(0, lambda: messagebox.showerror("Input Error", error_msg))
            return None


        # --- Input Splitting based on Dual/Quad ---
        frames_left_original_cropped: Optional[torch.Tensor] = None # For SBS output, cropped to original length

        if is_dual_input:
            half_w = total_w_current // 2
            frames_mask_raw = frames[:, :, :, :half_w]  # Left half is mask
            frames_warpped_raw = frames[:, :, :, half_w:] # Right half is warped
            
            # The target_output_h/w for GUI info will be the dimensions of the inpainted part
            output_display_h = total_h_current
            output_display_w = half_w

        else: # Quad input
            half_h = total_h_current // 2
            half_w = total_w_current // 2

            # frames_left_original_full_padded is the top-left quadrant, padded temporally
            frames_left_original_full_padded = frames[:, :, :half_h, :half_w]
            # Now, crop it to the original video length for eventual SBS concatenation
            frames_left_original_cropped = frames_left_original_full_padded[:num_frames_original].float() / 255.0 # Normalize for concat

            frames_mask_raw = frames[:, :, half_h:, :half_w]  # Bottom-Left is mask
            frames_warpped_raw = frames[:, :, half_h:, half_w:] # Bottom-Right is warped

            output_display_h = half_h
            output_display_w = half_w

        # --- Normalization and Grayscale Conversion (Using OpenCV) ---
        frames_warpped_normalized = frames_warpped_raw / 255.0

        processed_masks_grayscale = []
        for t in range(frames_mask_raw.shape[0]):
            frame_np_rgb = frames_mask_raw[t].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            frame_np_gray = cv2.cvtColor(frame_np_rgb, cv2.COLOR_RGB2GRAY)
            frame_tensor_gray = torch.from_numpy(frame_np_gray).float() / 255.0
            processed_masks_grayscale.append(frame_tensor_gray.unsqueeze(0))
        current_processed_mask = torch.stack(processed_masks_grayscale).to(frames_mask_raw.device)
        logger.debug(f"Mask: Initial grayscale (OpenCV, min={current_processed_mask.min().item():.2f}, max={current_processed_mask.max().item():.2f})")

        # --- Granular Mask Processing Steps (Direct Binarization Pipeline) ---

       # 1. Binarization (Direct Thresholding)
        try:
            binarize_threshold = float(self.mask_initial_threshold_var.get())
            if binarize_threshold != 0.0: # Step enabled if threshold is not 0
                if not (0.0 <= binarize_threshold <= 1.0):
                    logger.warning(f"Invalid binarize threshold ({binarize_threshold}). Using default 0.1.")
                    binarize_threshold = 0.1
                current_processed_mask = (current_processed_mask > binarize_threshold).float()
                logger.debug(f"Mask: Binarized (threshold > {binarize_threshold}, min={current_processed_mask.min().item():.2f}, max={current_processed_mask.max().item():.2f})")
            else:
                logger.debug("Mask: Binarization step skipped (threshold is 0). Using grayscale (might be unsuitable for subsequent steps).")
        except ValueError:
            logger.error(f"Invalid value for binarize threshold: {self.mask_initial_threshold_var.get()}. Falling back to 0.1.", exc_info=True)
            current_processed_mask = (current_processed_mask > 0.1).float() # Fallback to default behavior if error

        # 2. Morphological Closing
        try:
            morph_kernel_size = int(float(self.mask_morph_kernel_size_var.get()))
            if morph_kernel_size != 0: # Step enabled if kernel size is not 0
                current_processed_mask = self._apply_morphological_closing(current_processed_mask, morph_kernel_size)
                logger.debug(f"Mask: After morphological closing (min={current_processed_mask.min().item():.2f}, max={current_processed_mask.max().item():.2f})")
            else:
                logger.debug("Mask: Morphological closing step skipped (kernel size is 0).")
        except ValueError:
            logger.error(f"Invalid value for mask_morph_kernel_size: {self.mask_morph_kernel_size_var.get()}. Skipping morphological closing.", exc_info=True)
        except Exception as e:
            logger.error(f"Error during morphological closing step: {e}. Skipping.", exc_info=True)

        # 3. Mask Dilation
        try:
            dilate_kernel_size = int(self.mask_dilate_kernel_size_var.get())
            if dilate_kernel_size != 0: # Step enabled if kernel size is not 0
                current_processed_mask = self._apply_mask_dilation(current_processed_mask, dilate_kernel_size)
                logger.debug(f"Mask: After dilation (min={current_processed_mask.min().item():.2f}, max={current_processed_mask.max().item():.2f})")
            else:
                logger.debug("Mask: Dilation step skipped (kernel size is 0).")
        except ValueError:
            logger.error(f"Invalid value for mask_dilate_kernel_size: {self.mask_dilate_kernel_size_var.get()}. Skipping dilation.", exc_info=True)
        except Exception as e:
            logger.error(f"Error during mask dilation step: {e}. Skipping.", exc_info=True)

        # 4. Mask Gaussian Blur
        try:
            blur_kernel_size = int(self.mask_blur_kernel_size_var.get()) # NEW: Parse blur kernel size
            if blur_kernel_size != 0: # Step enabled if kernel size is not 0
                current_processed_mask = self._apply_gaussian_blur(current_processed_mask, blur_kernel_size) # NEW: Pass kernel size
                logger.debug(f"Mask: After blur (min={current_processed_mask.min().item():.2f}, max={current_processed_mask.max().item():.2f})")
            else:
                logger.debug("Mask: Gaussian blur step skipped (kernel size is 0).")
        except ValueError:
            logger.error(f"Invalid value for mask_blur_kernel_size: {self.mask_blur_kernel_size_var.get()}. Skipping blur.", exc_info=True)
        except Exception as e:
            logger.error(f"Error during mask blur step: {e}. Skipping.", exc_info=True)
        # --- END NEW Granular Mask Processing Steps ---

        # --- Store original-length, unpadded versions for post-blending ---
        frames_warpped_original_unpadded_normalized = frames_warpped_normalized[:num_frames_original].clone()
        frames_mask_processed_unpadded_original_length = current_processed_mask[:num_frames_original].clone()

        # --- Pad for Tiling (for pipeline input) ---
        frames_warpped_padded = pad_for_tiling(frames_warpped_normalized, tile_num, tile_overlap=(128, 128))
        frames_mask_padded = pad_for_tiling(current_processed_mask, tile_num, tile_overlap=(128, 128))
        
        padded_H, padded_W = frames_warpped_padded.shape[2], frames_warpped_padded.shape[3]

        # Update GUI with video info after processing initial dimensions
        if update_info_callback:
            display_frames_info = f"{actual_frames_to_process_count} (out of {total_frames_in_video})" if process_length != -1 else str(total_frames_in_video)
            self.after(0, lambda: update_info_callback(base_video_name, f"{output_display_w}x{output_display_h}", display_frames_info, overlap, original_input_blend_strength))

        return (frames_warpped_padded, frames_mask_padded, frames_left_original_cropped,
                num_frames_original, padded_H, padded_W, video_stream_info, fps,
                frames_warpped_original_unpadded_normalized, frames_mask_processed_unpadded_original_length)

    def _set_saved_geometry(self: "InpaintingGUI"):
        """Applies the saved window width and position, with dynamic height."""
        # Ensure the window is visible and all widgets are laid out for accurate height calculation
        self.update_idletasks() 

        # 1. Get the optimal height for the current content
        calculated_height = self.winfo_reqheight()
        # Fallback in case winfo_reqheight returns a tiny value (shouldn't happen after update_idletasks)
        if calculated_height < 100:
            calculated_height = 500 # A reasonable fallback height if something goes wrong

        # 2. Use the saved/default width
        current_width = self.window_width
        # Fallback if saved width is invalid or too small
        if current_width < 200: # Minimum sensible width
            current_width = 550 # Use default width

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
    
    def _setup_video_info_and_hires(
        self,
        input_video_path: str,
        save_dir: str,
        is_dual_input: bool,
    ) -> Tuple[Optional[str], dict]:
        """
        Initializes Hi-Res variables, finds a Hi-Res match, determines the final output path,
        and initializes variables for process flow.
        Returns (output_video_path, hires_data).
        """
        base_video_name = os.path.basename(input_video_path)
        video_name_without_ext = os.path.splitext(base_video_name)[0]
        output_suffix = "_inpainted_right_eye" if is_dual_input else "_inpainted_sbs"

        # --- INITIALIZE HI-RES VARIABLES & FIND MATCH (STEP 1) ---
        hires_video_path: Optional[str] = self._find_high_res_match(input_video_path)
        is_hires_blend_enabled = False
        hires_H, hires_W = 0, 0
        
        if hires_video_path:
            is_hires_blend_enabled = True
            try:
                # Load first frame of Hi-Res video to get its dimensions
                temp_reader = VideoReader(hires_video_path, ctx=cpu(0))
                full_h_hires, full_w_hires = temp_reader.get_batch([0]).shape[1:3]
                del temp_reader

                if is_dual_input:
                    hires_H, hires_W = full_h_hires, full_w_hires // 2
                else:
                    hires_H, hires_W = full_h_hires // 2, full_w_hires // 2
                
                logger.info(f"Hi-Res blending enabled. Target resolution: {hires_W}x{hires_H}")
            except Exception as e:
                logger.error(f"Failed to read Hi-Res video dimensions from {hires_video_path}: {e}")
                is_hires_blend_enabled = False # Disable blending if dimensions can't be read
                hires_video_path = None # Ensure it's None on failure

        # --- CALCULATE FINAL OUTPUT FILENAME (STEP 2) ---
        if is_hires_blend_enabled and hires_video_path:
            hires_base_name = os.path.basename(hires_video_path)
            hires_name_without_ext = os.path.splitext(hires_base_name)[0]
            video_name_for_output = hires_name_without_ext.replace("_splatted4", "").replace("_splatted2", "")
            logger.debug(f"Output filename base set to Hi-Res: {video_name_for_output}")
        else:
            video_name_for_output = video_name_without_ext.replace("_splatted4", "").replace("_splatted2", "")
        
        output_video_filename = f"{video_name_for_output}{output_suffix}.mp4"
        output_video_path = os.path.join(save_dir, output_video_filename)

        hires_data = {
            "hires_video_path": hires_video_path,
            "is_hires_blend_enabled": is_hires_blend_enabled,
            "hires_H": hires_H,
            "hires_W": hires_W,
            "base_video_name": base_video_name, # Keep this for update_info_callback
            "video_name_for_output": video_name_for_output, # For temp PNG dir
        }
        return output_video_path, hires_data
    
    def _toggle_color_transfer_state(self):
        """Callback for the Enable Color Transfer checkbox. Saves config."""
        self.save_config() # Simply save the config to persist the checkbox state
        logger.debug(f"Color Transfer state changed to: {self.enable_color_transfer.get()}")
    
    def _toggle_debug_mode(self):
        """Toggles debug mode on/off and updates logging."""
        self._configure_logging()
        # Save the current debug mode state to config immediately
        self.save_config() 
        # messagebox.showinfo("Debug Mode", f"Debug mode is now {'ON' if self.debug_mode_var.get() else 'OFF'}.\nLog level set to {logging.getLevelName(logger.level)}.\n(Restart may be needed for some changes to take full effect).")
    
    def _toggle_blend_parameters_state(self):
        """Enables or disables mask processing parameter entry widgets based on the blend toggle."""
        state = tk.NORMAL if self.enable_post_inpainting_blend.get() else tk.DISABLED
        for widget in self.mask_param_widgets:
            widget.config(state=state)
        # We might also want to disable the blending execution if the toggle is off,
        # but the `if not self.enable_post_inpainting_blend.get(): return inpainted_frames`
        # check in `_apply_post_inpainting_blend` already handles this.
        # This function primarily affects the GUI state.
        logger.debug(f"Blend parameters state set to: {state}")
    
    def create_widgets(self):
        
        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)

        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)        
        self.file_menu.add_command(label="Load Settings...", command=self.load_settings)
        self.file_menu.add_command(label="Save Settings...", command=self.save_settings)
        self.file_menu.add_separator() # Separator for organization
        self.file_menu.add_checkbutton(label="Dark Mode", variable=self.dark_mode_var, command=self._apply_theme)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Reset to Default", command=self.reset_to_defaults)
        self.file_menu.add_command(label="Restore Finished", command=self.restore_finished_files)

        # --- Help Menu ---
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_checkbutton(label="Enable Debugging", variable=self.debug_mode_var, command=self._toggle_debug_mode)
        self.help_menu.add_separator()
        self.help_menu.add_command(label="About", command=self.show_about_dialog)

        # --- FOLDER FRAME ---
        folder_frame = ttk.LabelFrame(self, text="Folders", padding=10)
        folder_frame.pack(fill="x", padx=10, pady=5)
        folder_frame.grid_columnconfigure(1, weight=1)
        
        current_row = 0 # Initialize row counter for folder_frame

        # Input Folder
        input_label = ttk.Label(folder_frame, text="Input Folder:")
        input_label.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        Tooltip(input_label, self.help_data.get("input_folder", ""))
        ttk.Entry(folder_frame, textvariable=self.input_folder_var, width=40).grid(row=current_row, column=1, padx=5, sticky="ew")
        ttk.Button(folder_frame, text="Browse", command=self._browse_input).grid(row=current_row, column=2, padx=5)
        current_row += 1

        # --- NEW: Hi-Res Blend Folder ---
        hires_label = ttk.Label(folder_frame, text="Hi-Res Blend Folder:")
        hires_label.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        Tooltip(hires_label, "Folder containing matching high-resolution splatted files for final blending.")
        ttk.Entry(folder_frame, textvariable=self.hires_blend_folder_var, width=40).grid(row=current_row, column=1, padx=5, sticky="ew")
        ttk.Button(folder_frame, text="Browse", command=self._browse_hires_folder).grid(row=current_row, column=2, padx=5)
        current_row += 1
        
        # Output Folder
        output_label = ttk.Label(folder_frame, text="Output Folder:")
        output_label.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        Tooltip(output_label, self.help_data.get("output_folder", ""))
        ttk.Entry(folder_frame, textvariable=self.output_folder_var, width=40).grid(row=current_row, column=1, padx=5, sticky="ew")
        ttk.Button(folder_frame, text="Browse", command=self._browse_output).grid(row=current_row, column=2, padx=5)


        # --- MAIN PARAMETERS FRAME ---
        param_frame = ttk.LabelFrame(self, text="Parameters", padding=10)
        param_frame.pack(fill="x", padx=10, pady=5)
        
        # Configure 4 columns for param_frame: Label | Entry | Label | Entry
        param_frame.grid_columnconfigure(0, weight=1) # Left Label
        param_frame.grid_columnconfigure(1, weight=1) # Left Entry
        param_frame.grid_columnconfigure(2, weight=1) # Right Label
        param_frame.grid_columnconfigure(3, weight=1) # Right Entry

        current_row = 0 # Reset row counter for param_frame

        # Row 0: Inference Steps (Left) & Output CRF (Right)
        inference_steps_label = ttk.Label(param_frame, text="Inference Steps:")
        inference_steps_label.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        Tooltip(inference_steps_label, self.help_data.get("num_inference_steps", ""))
        ttk.Entry(param_frame, textvariable=self.num_inference_steps_var, width=10).grid(row=current_row, column=1, sticky="w", padx=5)

        original_blend_label = ttk.Label(param_frame, text="Original Input Bias:")
        original_blend_label.grid(row=current_row, column=2, sticky="e", padx=5, pady=2)
        Tooltip(original_blend_label, self.help_data.get("original_input_blend_strength", ""))
        ttk.Entry(param_frame, textvariable=self.original_input_blend_strength_var, width=10).grid(row=current_row, column=3, sticky="w", padx=5)
        current_row += 1

        # Row 1: Tile Number (Left) & Process Length (Right)
        tile_num_label = ttk.Label(param_frame, text="Tile Number:")
        tile_num_label.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        Tooltip(tile_num_label, self.help_data.get("tile_num", ""))
        ttk.Entry(param_frame, textvariable=self.tile_num_var, width=10).grid(row=current_row, column=1, sticky="w", padx=5)
        
        frame_overlap_label = ttk.Label(param_frame, text="Frame Overlap:")
        frame_overlap_label.grid(row=current_row, column=2, sticky="e", padx=5, pady=2)
        Tooltip(frame_overlap_label, self.help_data.get("frame_overlap", "")) 
        ttk.Entry(param_frame, textvariable=self.overlap_var, width=10).grid(row=current_row, column=3, sticky="w", padx=5)
        current_row += 1

        # Row 2: Frames Chunk (Left) & Frame Overlap (Right)
        frames_chunk_label = ttk.Label(param_frame, text="Frames Chunk:")
        frames_chunk_label.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        Tooltip(frames_chunk_label, self.help_data.get("frames_chunk", ""))
        ttk.Entry(param_frame, textvariable=self.frames_chunk_var, width=10).grid(row=current_row, column=1, sticky="w", padx=5)
        
        output_crf_label = ttk.Label(param_frame, text="Output CRF:")
        output_crf_label.grid(row=current_row, column=2, sticky="e", padx=5, pady=2)
        Tooltip(output_crf_label, self.help_data.get("output_crf", ""))
        ttk.Entry(param_frame, textvariable=self.output_crf_var, width=10).grid(row=current_row, column=3, sticky="w", padx=5)
        current_row += 1

        # Row 3: Original Input Bias (Left) & CPU Offload (Right)
        process_length_label = ttk.Label(param_frame, text="Process Length:")
        process_length_label.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        Tooltip(process_length_label, self.help_data.get("process_length", "Number of frames to process. Use -1 for all frames."))
        ttk.Entry(param_frame, textvariable=self.process_length_var, width=10).grid(row=current_row, column=1, sticky="w", padx=5)

        offload_label = ttk.Label(param_frame, text="CPU Offload:")
        offload_label.grid(row=current_row, column=2, sticky="e", padx=5, pady=2)
        Tooltip(offload_label, self.help_data.get("offload_type", ""))
        offload_options = ["model", "sequential", "none"]
        ttk.OptionMenu(param_frame, self.offload_type_var, self.offload_type_var.get(), *offload_options).grid(row=current_row, column=3, sticky="w", padx=5)
        # current_row += 1 # No need to increment here, param_frame is done


        # --- POST-PROCESSING FRAME ---
        post_process_frame = ttk.LabelFrame(self, text="Post-Processing", padding=10)
        post_process_frame.pack(fill="x", padx=10, pady=5)
        
        # Configure 4 columns for post_process_frame: Label | Entry | Label | Entry
        post_process_frame.grid_columnconfigure(0, weight=1) # Left Label
        post_process_frame.grid_columnconfigure(1, weight=1) # Left Entry
        post_process_frame.grid_columnconfigure(2, weight=1) # Right Label
        post_process_frame.grid_columnconfigure(3, weight=1) # Right Entry

        current_row = 0 # Reset row counter for post_process_frame

        # Row 0: Enable Post-Inpainting Blend Checkbox
        blend_enable_check = ttk.Checkbutton(post_process_frame, text="Enable Post-Inpainting Blend", 
                                             variable=self.enable_post_inpainting_blend, 
                                             command=self._toggle_blend_parameters_state)
        blend_enable_check.grid(row=current_row, column=0, columnspan=4, sticky="w", padx=5, pady=2) # Spans all 4 columns
        Tooltip(blend_enable_check, self.help_data.get("enable_post_inpainting_blend", ""))

        color_transfer_check = ttk.Checkbutton(post_process_frame, text="Enable Color Transfer", 
                                               variable=self.enable_color_transfer,
                                               command=self._toggle_color_transfer_state) # Will create this command
        color_transfer_check.grid(row=current_row, column=2, columnspan=2, sticky="w", padx=5, pady=2) # Occupies right side
        Tooltip(color_transfer_check, self.help_data.get("enable_color_transfer", ""))
        current_row += 1

        # Row 1: Mask Binarization Threshold (Left) & Morphological Closing Kernel Size (Right)
        bin_thresh_label = ttk.Label(post_process_frame, text="Mask Binarize Thresh:")
        bin_thresh_label.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        Tooltip(bin_thresh_label, self.help_data.get("mask_initial_threshold", ""))
        bin_thresh_entry = ttk.Entry(post_process_frame, textvariable=self.mask_initial_threshold_var, width=10)
        bin_thresh_entry.grid(row=current_row, column=1, sticky="w", padx=5)
        self.mask_param_widgets.append(bin_thresh_entry) # Store reference
        
        dilate_kernel_label = ttk.Label(post_process_frame, text="Mask Dilate Kernel:")
        dilate_kernel_label.grid(row=current_row, column=2, sticky="e", padx=5, pady=2)
        Tooltip(dilate_kernel_label, self.help_data.get("mask_dilate_kernel_size", ""))
        dilate_kernel_entry = ttk.Entry(post_process_frame, textvariable=self.mask_dilate_kernel_size_var, width=10)
        dilate_kernel_entry.grid(row=current_row, column=3, sticky="w", padx=5)
        self.mask_param_widgets.append(dilate_kernel_entry) # Store reference
        current_row += 1

        # Row 2: Mask Dilation Kernel (Left) & Mask Blur Kernel Size (Right)
        morph_kernel_label = ttk.Label(post_process_frame, text="Morph Close Kernel:")
        morph_kernel_label.grid(row=current_row, column=0, sticky="e", padx=5, pady=2)
        Tooltip(morph_kernel_label, self.help_data.get("mask_morph_kernel_size", ""))
        morph_kernel_entry = ttk.Entry(post_process_frame, textvariable=self.mask_morph_kernel_size_var, width=10)
        morph_kernel_entry.grid(row=current_row, column=1, sticky="w", padx=5)
        self.mask_param_widgets.append(morph_kernel_entry) # Store reference

        blur_kernel_label = ttk.Label(post_process_frame, text="Mask Blur Kernel:")
        blur_kernel_label.grid(row=current_row, column=2, sticky="e", padx=5, pady=2)
        Tooltip(blur_kernel_label, self.help_data.get("mask_blur_kernel_size", ""))
        blur_kernel_entry = ttk.Entry(post_process_frame, textvariable=self.mask_blur_kernel_size_var, width=10)
        blur_kernel_entry.grid(row=current_row, column=3, sticky="w", padx=5)
        self.mask_param_widgets.append(blur_kernel_entry) # Store reference
        # current_row += 1 # No need to increment here, post_process_frame is done
        
        # Initialize the state of blend parameters immediately after creation
        self._toggle_blend_parameters_state()


        # --- PROGRESS FRAME (no change) ---
        progress_frame = ttk.LabelFrame(self, text="Progress", padding=10)
        progress_frame.pack(fill="x", padx=10, pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress_bar.pack(fill="x")
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack(pady=5)

        # --- BUTTONS FRAME (no change) ---
        buttons_frame = ttk.Frame(self, padding=10)
        buttons_frame.pack(fill="x", pady=10)
        
        inner_buttons_frame = ttk.Frame(buttons_frame)
        inner_buttons_frame.pack(anchor="center")

        self.start_button = ttk.Button(inner_buttons_frame, text="Start", command=self.start_processing)
        self.start_button.pack(side="left", padx=5)
        self.stop_button = ttk.Button(inner_buttons_frame, text="Stop", command=self.stop_processing, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        # ttk.Button(inner_buttons_frame, text="Help", command=self.show_general_help).pack(side="left", padx=5)
        ttk.Button(inner_buttons_frame, text="Exit", command=self.exit_application).pack(side="left", padx=5)

        # --- INFO FRAME (no change) ---
        self.info_frame = ttk.LabelFrame(self, text="Current Video Information", padding=10)
        self.info_frame.pack(fill="x", padx=10, pady=5)
        
        self.info_frame.grid_columnconfigure(0, weight=0)
        self.info_frame.grid_columnconfigure(1, weight=1)

        current_row = 0

        ttk.Label(self.info_frame, text="Name:").grid(row=current_row, column=0, sticky="e", padx=(5, 2), pady=1)
        self.video_name_label = ttk.Label(self.info_frame, textvariable=self.video_name_var, anchor="w")
        self.video_name_label.grid(row=current_row, column=1, sticky="ew", padx=(2, 5), pady=1)
        current_row += 1
        
        ttk.Label(self.info_frame, text="Resolution:").grid(row=current_row, column=0, sticky="e", padx=(5, 2), pady=1)
        self.video_res_label = ttk.Label(self.info_frame, textvariable=self.video_res_var, anchor="w")
        self.video_res_label.grid(row=current_row, column=1, sticky="ew", padx=(2, 5), pady=1)
        current_row += 1
        
        ttk.Label(self.info_frame, text="Frames:").grid(row=current_row, column=0, sticky="e", padx=(5, 2), pady=1)
        self.video_frames_label = ttk.Label(self.info_frame, textvariable=self.video_frames_var, anchor="w")
        self.video_frames_label.grid(row=current_row, column=1, sticky="ew", padx=(2, 5), pady=1)
        current_row += 1

        ttk.Label(self.info_frame, text="Overlap:").grid(row=current_row, column=0, sticky="e", padx=(5, 2), pady=1)
        self.video_overlap_label = ttk.Label(self.info_frame, textvariable=self.video_overlap_var, anchor="w")
        self.video_overlap_label.grid(row=current_row, column=1, sticky="ew", padx=(2, 5), pady=1)
        current_row += 1

        ttk.Label(self.info_frame, text="Input Bias:").grid(row=current_row, column=0, sticky="e", padx=(5, 2), pady=1)
        self.video_bias_label = ttk.Label(self.info_frame, textvariable=self.video_bias_var, anchor="w")
        self.video_bias_label.grid(row=current_row, column=1, sticky="ew", padx=(2, 5), pady=1)

    def process_single_video(
        self,
        pipeline: StableVideoDiffusionInpaintingPipeline,
        input_video_path: str,
        save_dir: str,
        frames_chunk: int = 23,
        overlap: int = 3,
        tile_num: int = 1,
        vf: Optional[str] = None,
        num_inference_steps: int = 5,
        stop_event: Optional[threading.Event] = None,
        update_info_callback=None,
        original_input_blend_strength: float = 0.8,
        output_crf: int = 23,
        process_length: int = -1,
    ) -> Tuple[bool, Optional[str]]:
        """
        Orchestrates the processing of a single video: Setup, Inpainting, Finalization, Encoding.
        Returns (completion_status, hi_res_input_path).
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Determine splat type early
        base_video_name = os.path.basename(input_video_path)
        video_name_without_ext = os.path.splitext(base_video_name)[0]
        is_dual_input = video_name_without_ext.endswith("_splatted2")

        # 1. SETUP & HI-RES DETECTION
        # output_video_path is str (guaranteed), hires_data is dict (guaranteed)
        output_video_path, hires_data = self._setup_video_info_and_hires(
            input_video_path, save_dir, is_dual_input
        )
        base_video_name = hires_data["base_video_name"]
        video_name_for_output = hires_data["video_name_for_output"]
        hires_video_path = hires_data["hires_video_path"] # Optional[str]
        
        # 2. INPUT PREPARATION (Low-Res)
        prepared_inputs = self._prepare_video_inputs(
            input_video_path=input_video_path,
            base_video_name=base_video_name,
            is_dual_input=is_dual_input,
            frames_chunk=frames_chunk,
            tile_num=tile_num,
            update_info_callback=update_info_callback,
            overlap=overlap,
            original_input_blend_strength=original_input_blend_strength,
            process_length=process_length
        )

        if prepared_inputs is None:
            return False, None # Preparation failed
        
        # Unpack, ensuring all torch.Tensor return values are not None
        (frames_warpped_padded, frames_mask_padded, frames_left_original_cropped,
        num_frames_original, padded_H, padded_W, video_stream_info, fps,
        frames_warpped_original_unpadded_normalized, frames_mask_processed_unpadded_original_length) = prepared_inputs

        # 3. INPAINTING CHUNKS (The main loop)
        # This part of the loop remains the same, but the logic inside is simplified
        total_frames_to_process_actual = num_frames_original        
        stride = max(1, frames_chunk - overlap)
        results = [] 
        previous_chunk_output_frames: Optional[torch.Tensor] = None

        for i in range(0, total_frames_to_process_actual, stride):
            if stop_event and stop_event.is_set():
                logger.info(f"Stopping processing of {input_video_path}")
                return False, None
            
            # --- CHUNK SLICING AND PADDING LOGIC (Remains from your last correct version) ---
            end_idx_for_slicing = min(i + frames_chunk, total_frames_to_process_actual)
            original_input_frames_slice = frames_warpped_padded[i:end_idx_for_slicing].clone()
            mask_frames_slice = frames_mask_padded[i:end_idx_for_slicing].clone()
            actual_sliced_length = original_input_frames_slice.shape[0]

            padding_needed_for_pipeline_input = 0
            if actual_sliced_length <= 4:
                target_length = 6
                padding_needed_for_pipeline_input = target_length - actual_sliced_length
                logger.debug(f"End-of-video optimization: Short chunk of {actual_sliced_length} frames padded to minimum {target_length}.")
            
            if padding_needed_for_pipeline_input > 0:
                logger.debug(f"Dynamically padding input for chunk starting at frame {i}: {actual_sliced_length} frames sliced, {padding_needed_for_pipeline_input} frames needed.")
                last_original_frame_warpped = frames_warpped_padded[total_frames_to_process_actual - 1].unsqueeze(0).clone()
                last_original_frame_mask = frames_mask_padded[total_frames_to_process_actual - 1].unsqueeze(0).clone()
                repeated_warpped = last_original_frame_warpped.repeat(padding_needed_for_pipeline_input, 1, 1, 1)
                repeated_mask = last_original_frame_mask.repeat(padding_needed_for_pipeline_input, 1, 1, 1)
                input_frames_to_pipeline = torch.cat([original_input_frames_slice, repeated_warpped], dim=0)
                mask_frames_i = torch.cat([mask_frames_slice, repeated_mask], dim=0)
            else:
                input_frames_to_pipeline = original_input_frames_slice
                mask_frames_i = mask_frames_slice
            # --- END CHUNK SLICING AND PADDING LOGIC ---

            # --- INPUT-LEVEL BLENDING (Remains from your last correct version) ---
            if previous_chunk_output_frames is not None and overlap > 0:
                # ... (Input-level blending logic) ...
                overlap_actual = min(overlap, input_frames_to_pipeline.shape[0]) 
                if overlap_actual > 0:
                    prev_gen_overlap_frames = previous_chunk_output_frames[-overlap_actual:]
                    if original_input_blend_strength > 0:
                        orig_input_overlap_frames = input_frames_to_pipeline[:overlap_actual]
                        original_weights_scaled = torch.linspace(0.0, 1.0, overlap_actual, device=prev_gen_overlap_frames.device).view(-1, 1, 1, 1) * original_input_blend_strength
                        blended_input_overlap_frames = (1 - original_weights_scaled) * prev_gen_overlap_frames + original_weights_scaled * orig_input_overlap_frames
                        input_frames_to_pipeline[:overlap_actual] = blended_input_overlap_frames
                        del orig_input_overlap_frames, original_weights_scaled, blended_input_overlap_frames
                    else:
                        input_frames_to_pipeline[:overlap_actual] = prev_gen_overlap_frames
                    del prev_gen_overlap_frames
            # --- END INPUT-LEVEL BLENDING ---

            # --- INFERENCE ---
            logger.info(f"Starting inference for chunk {i}-{i+input_frames_to_pipeline.shape[0]} (Temporal length: {input_frames_to_pipeline.shape[0]})...")
            start_time = time.time()

            with torch.no_grad():
                video_latents = spatial_tiled_process(
                    # ... (spatial_tiled_process arguments) ...
                    cond_frames=input_frames_to_pipeline, mask_frames=mask_frames_i, process_func=pipeline, tile_num=tile_num,
                    spatial_n_compress=8, min_guidance_scale=1.01, max_guidance_scale=1.01, decode_chunk_size=2,
                    fps=7, motion_bucket_id=127, noise_aug_strength=0.0, num_inference_steps=num_inference_steps,
                )
                video_latents = video_latents.unsqueeze(0)
                pipeline.vae.to(dtype=torch.float16)
                decoded_frames = pipeline.decode_latents(video_latents, num_frames=video_latents.shape[1], decode_chunk_size=2)

            # --- DECODING & CHUNK COLLECT ---
            inference_duration = time.time() - start_time
            logger.debug(f"Inference for chunk {i}-{i+input_frames_to_pipeline.shape[0]} completed in {inference_duration:.2f} seconds.")
            
            video_frames = tensor2vid(decoded_frames, pipeline.image_processor, output_type="pil")[0]
            current_chunk_generated = torch.stack([
                torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in video_frames
            ])

            # Append only the "new" frames
            if i == 0:
                results.append(current_chunk_generated[:actual_sliced_length])
            else:
                results.append(current_chunk_generated[overlap:actual_sliced_length])
            
            previous_chunk_output_frames = current_chunk_generated
        # --- END INPAINTING CHUNKS ---

        # 4. PREPARE FRAMES FOR FINALIZATION (Temporal/Spatial Cropping)
        if not results:
            logger.warning(f"No frames generated for {input_video_path}.")
            if update_info_callback:
                self.after(0, lambda: update_info_callback(base_video_name, "N/A", "0 (No Output)", overlap, original_input_blend_strength))
            return False, None

        frames_output = torch.cat(results, dim=0).cpu()
        if frames_output.numel() == 0 or frames_output.shape[2] < padded_H or frames_output.shape[3] < padded_W:
            logger.error(f"Generated frames_output has invalid dimensions (actual {frames_output.shape[2]}x{frames_output.shape[3]} vs target {padded_H}x{padded_W}).")
            return False, None

        frames_output_final = frames_output[:, :, :padded_H, :padded_W][:num_frames_original]
        
        # 5. FINALIZATION (Hi-Res Upscale, Color Transfer, Blend, Concat)
        final_output_frames_for_encoding = self._finalize_output_frames(
            inpainted_frames=frames_output_final,
            mask_frames=frames_mask_processed_unpadded_original_length,
            original_warped_frames=frames_warpped_original_unpadded_normalized,
            original_left_frames=frames_left_original_cropped,
            hires_data=hires_data,
            base_video_name=base_video_name,
            is_dual_input=is_dual_input,
        )

        if final_output_frames_for_encoding is None or final_output_frames_for_encoding.numel() == 0:
            logger.error(f"Final output frames are empty after finalization for {base_video_name}.")
            if update_info_callback:
                self.after(0, lambda: update_info_callback(base_video_name, "N/A", "0 (Empty Final)", overlap, original_input_blend_strength))
            return False, None
            
        # 6. ENCODING
        temp_png_dir = os.path.join(save_dir, f"temp_inpainted_pngs_{video_name_for_output}_{os.getpid()}")
        os.makedirs(temp_png_dir, exist_ok=True)
        logger.debug(f"Saving intermediate 16-bit PNG sequence to {temp_png_dir}")

        total_output_frames = final_output_frames_for_encoding.shape[0]
        stop_event_non_optional = stop_event if stop_event is not None else threading.Event()
        
        try:
            # 6a. Save PNG Sequence
            for frame_idx in range(total_output_frames):
                if stop_event_non_optional.is_set():
                    logger.debug(f"Stopping PNG sequence saving for {input_video_path}")
                    shutil.rmtree(temp_png_dir, ignore_errors=True)
                    return False, None

                frame_tensor = final_output_frames_for_encoding[frame_idx] 
                frame_np = frame_tensor.permute(1, 2, 0).numpy()
                frame_uint16 = (np.clip(frame_np, 0.0, 1.0) * 65535.0).astype(np.uint16)
                frame_bgr = cv2.cvtColor(frame_uint16, cv2.COLOR_RGB2BGR)
                png_path = os.path.join(temp_png_dir, f"{frame_idx:05d}.png")
                cv2.imwrite(png_path, frame_bgr)
                draw_progress_bar(frame_idx + 1, total_output_frames)
            logger.debug(f"\nFinished saving {total_output_frames} PNG frames.")
            
            # 6b. Encode to MP4
            if update_info_callback:
                self.after(0, lambda: update_info_callback(base_video_name, f"Encoding video...", total_output_frames, overlap, original_input_blend_strength))
            
            encoding_success = encode_frames_to_mp4(
                temp_png_dir=temp_png_dir, final_output_mp4_path=output_video_path, fps=fps,
                total_output_frames=total_output_frames, video_stream_info=video_stream_info,
                stop_event=stop_event_non_optional, sidecar_json_data=None, user_output_crf=output_crf,
                output_sidecar_ext=".spsidecar",
            )
            
            if not encoding_success:
                logger.info(f"Encoding stopped or failed for {input_video_path}.")
                return False, None

        except Exception as e:
            logger.error(f"Error during PNG saving or Encoding for {base_video_name}: {e}", exc_info=True)
            messagebox.showerror("Error", f"Error during PNG saving or Encoding for {base_video_name}: {str(e)}")
            shutil.rmtree(temp_png_dir, ignore_errors=True)
            return False, None

        logger.info(f"Done processing {input_video_path} -> {output_video_path}")
        return True, hires_video_path

    def processing_done(self, stopped=False):
        if self.pipeline:
            # Ensure pipeline is properly released and cache cleared
            try:
                del self.pipeline
                release_cuda_memory()
            except RuntimeError as e:
                logger.warning(f"Failed to clear CUDA cache during cleanup: {e}")
            self.pipeline = None

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        if stopped:
            self.update_status_label("Processing stopped.")
        else:
            self.update_status_label("Processing completed.")
            
        self.update_video_info_display("N/A", "N/A", "N/A", "N/A", "N/A")

    def reset_to_defaults(self):
        if not messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to their default values?"):
            return

        # Set default values for all your configuration variables
        self.input_folder_var.set("./output_splatted")
        self.output_folder_var.set("./completed_output")
        self.num_inference_steps_var.set("5")
        self.tile_num_var.set("2")
        self.frames_chunk_var.set("23")
        self.overlap_var.set("3")
        self.original_input_blend_strength_var.set("0.5")
        self.offload_type_var.set("model")

        self.mask_initial_threshold_var.set("0.3")
        self.mask_morph_kernel_size_var.set("0.0")
        self.mask_dilate_kernel_size_var.set("5")
        self.mask_blur_kernel_size_var.set("7")

        self.enable_post_inpainting_blend.set(False) # Default state is OFF
        self.enable_color_transfer.set(True) # Default state is ON
        
        # Crucially, call the function to disable the entry fields if the blend toggle is now False
        self._toggle_blend_parameters_state() 

        self.save_config() # Save these new default settings
        messagebox.showinfo("Settings Reset", "All settings have been reset to their default values.")
        logger.info("GUI settings reset to defaults.")

    def restore_finished_files(self):
        if not messagebox.askyesno("Restore Finished Files", "Are you sure you want to move all processed videos from the 'finished' folders back to their respective input directories?"):
            return

        input_folder = self.input_folder_var.get()
        hires_input_folder = self.hires_blend_folder_var.get()

        restore_dirs = [
            (input_folder, os.path.join(input_folder, "finished"))
        ]
        
        # Only check the hires folder if it's different from the low-res folder
        if os.path.normpath(input_folder) != os.path.normpath(hires_input_folder):
            restore_dirs.append((hires_input_folder, os.path.join(hires_input_folder, "finished")))


        restored_count = 0
        errors_count = 0
        
        for input_dir, finished_dir in restore_dirs:
            if not os.path.isdir(finished_dir):
                logger.info(f"Restore skipped: 'finished' folder not found at {finished_dir}")
                continue

            # Collect files to move first
            files_to_move = [f for f in os.listdir(finished_dir) if os.path.isfile(os.path.join(finished_dir, f))]

            if not files_to_move:
                logger.info(f"Restore skipped: No files found in {finished_dir}")
                continue

            for filename in files_to_move:
                src_path = os.path.join(finished_dir, filename)
                dest_path = os.path.join(input_dir, filename)
                try:
                    shutil.move(src_path, dest_path)
                    restored_count += 1
                    logger.info(f"Moved '{filename}' from '{finished_dir}' back to '{input_dir}'")
                except Exception as e:
                    errors_count += 1
                    logger.error(f"Error moving file '{filename}' during restore: {e}")

        if restored_count > 0 or errors_count > 0:
            messagebox.showinfo("Restore Complete", f"Finished files restoration attempted.\n{restored_count} files moved.\n{errors_count} errors occurred.")
            logger.info(f"Restore complete: {restored_count} files moved, {errors_count} errors.")
        else:
            messagebox.showinfo("Restore Complete", "No files found to restore.")
            logger.info("Restore complete: No files found to restore.")

    def run_batch_process(
            self,
            input_folder,
            output_folder,
            num_inference_steps,
            tile_num, offload_type,
            frames_chunk, gui_overlap,
            gui_original_input_blend_strength,
            gui_output_crf,
            process_length
        ):
        """
        Orchestrates the batch processing of videos, handling sidecar JSON,
        thread-safe GUI updates, and error management.
        """
        try:
            self.pipeline = load_inpainting_pipeline(
                pre_trained_path="./weights/stable-video-diffusion-img2vid-xt-1-1",
                unet_path="./weights/StereoCrafter",
                device="cuda",
                dtype=torch.float16,
                offload_type=offload_type
            )
            input_videos = sorted(glob.glob(os.path.join(input_folder, "*.mp4")))
            if not input_videos:
                self.after(0, lambda: messagebox.showinfo("Info", "No .mp4 files found in input folder"))
                self.after(0, self.processing_done)
                return

            self.total_videos.set(len(input_videos))
            # finished_folder = os.path.join(input_folder, "finished")
            # os.makedirs(finished_folder, exist_ok=True)
            os.makedirs(output_folder, exist_ok=True)

            # Define a thread-safe wrapper for GUI updates
            # This ensures that calls from the processing thread are marshaled back to the main Tkinter thread.
            def _threaded_update_info_callback(name, resolution, frames, overlap_val, bias_val):
                self.after(0, self.update_video_info_display, name, resolution, frames, overlap_val, bias_val)

            for idx, video_path in enumerate(input_videos):
                if self.stop_event.is_set():
                    logger.info("Processing stopped by user.")
                    break
                
                # Initialize current video's parameters with GUI fallbacks
                current_overlap = gui_overlap
                current_original_input_blend_strength = gui_original_input_blend_strength
                current_output_crf = gui_output_crf # NEW: Initialize current_output_crf
                current_process_length = process_length # NEW: Current process_length (from GUI initially)

                json_path = os.path.splitext(video_path)[0] + ".spsidecar"
                if os.path.exists(json_path):
                    logger.info(f"Found sidecar fssidecar for {os.path.basename(video_path)} at {json_path}")
                    try:
                        with open(json_path, 'r') as f:
                            sidecar_data = json.load(f)
                        
                        if "frame_overlap" in sidecar_data:
                            sidecar_overlap = int(sidecar_data["frame_overlap"])
                            if sidecar_overlap >= 0:
                                current_overlap = sidecar_overlap
                                logger.debug(f"Using frame_overlap from sidecar: {current_overlap}")
                            else:
                                logger.warning(f"Invalid 'frame_overlap' in sidecar file for {os.path.basename(video_path)}. Using GUI value ({gui_overlap}).")

                        if "input_bias" in sidecar_data:
                            sidecar_input_bias = float(sidecar_data["input_bias"])
                            if 0.0 <= sidecar_input_bias <= 1.0:
                                current_original_input_blend_strength = sidecar_input_bias
                                logger.debug(f"Using input_bias from sidecar: {current_original_input_blend_strength}")
                            else:
                                logger.warning(f"Invalid 'input_bias' in sidecar file for {os.path.basename(video_path)}. Using GUI value ({gui_original_input_blend_strength}).")
                        
                        # NEW: Load CRF from sidecar
                        if "output_crf" in sidecar_data:
                            sidecar_crf = int(sidecar_data["output_crf"])
                            if sidecar_crf >= 0:
                                current_output_crf = sidecar_crf
                                logger.debug(f"Using output_crf from sidecar: {current_output_crf}")
                            else:
                                logger.warning(f"Invalid 'output_crf' in sidecar file for {os.path.basename(video_path)}. Using GUI value ({gui_output_crf}).")

                         # --- NEW: Load Process Length from sidecar ---
                        if "process_length" in sidecar_data:
                            sidecar_process_length = int(sidecar_data["process_length"])
                            if sidecar_process_length == -1 or sidecar_process_length > 0:
                                current_process_length = sidecar_process_length
                                logger.debug(f"Using process_length from sidecar: {current_process_length}")
                            else:
                                logger.warning(f"Invalid 'process_length' in sidecar file for {os.path.basename(video_path)}. Using GUI value ({process_length}).")

                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Error reading or parsing sidecar file {json_path}: {e}. Falling back to GUI parameters for this video.")
                else:
                    logger.debug(f"No sidecar file found for {os.path.basename(video_path)}. Using GUI parameters.")

                # Update status label to indicate which video is starting processing
                self.after(0, self.update_status_label, f"Processing video {idx + 1} of {self.total_videos.get()}")

                logger.info(f"Starting processing of {video_path}")
                completed, hi_res_input_path = self.process_single_video(
                    pipeline=self.pipeline,
                    input_video_path=video_path,
                    save_dir=output_folder,
                    frames_chunk=frames_chunk,
                    overlap=current_overlap,
                    tile_num=tile_num,
                    vf=None, 
                    num_inference_steps=num_inference_steps,
                    stop_event=self.stop_event,
                    update_info_callback=_threaded_update_info_callback, 
                    original_input_blend_strength=current_original_input_blend_strength,
                    output_crf=current_output_crf,
                    process_length=current_process_length
                )
                
                if completed:
                    # Define finished folder paths dynamically
                    low_res_input_folder = input_folder
                    hires_input_folder = self.hires_blend_folder_var.get()

                    low_res_finished_folder = os.path.join(low_res_input_folder, "finished")
                    
                    # 1. Move LOW-RES input file
                    try:
                        os.makedirs(low_res_finished_folder, exist_ok=True) # Ensure low-res finished exists
                        shutil.move(video_path, low_res_finished_folder)
                        logger.debug(f"Moved {video_path} to {low_res_finished_folder}")
                    except Exception as e:
                        logger.error(f"Failed to move {video_path} to {low_res_finished_folder}: {e}")
                        
                    # 2. Move HI-RES input file if it was used
                    if hi_res_input_path:
                        # Ensure the high-res folder is different before trying to move
                        if os.path.normpath(low_res_input_folder) != os.path.normpath(hires_input_folder):
                            hires_finished_folder = os.path.join(hires_input_folder, "finished")
                            try:
                                os.makedirs(hires_finished_folder, exist_ok=True) # Ensure hi-res finished exists
                                shutil.move(hi_res_input_path, hires_finished_folder)
                                logger.debug(f"Moved Hi-Res input {hi_res_input_path} to {hires_finished_folder}")
                            except Exception as e:
                                logger.error(f"Failed to move Hi-Res input {hi_res_input_path} to {hires_finished_folder}: {e}")
                        else:
                            logger.warning(f"Skipping Hi-Res move: Folder {hires_input_folder} is same as Low-Res folder.")
                else:
                    logger.info(f"Processing of {video_path} was stopped or skipped due to issues.")
                
                self.processed_count.set(idx + 1)
                
            stopped = self.stop_event.is_set()
            self.after(0, lambda: self.processing_done(stopped))

        except Exception as e:
            logger.exception("An unhandled error occurred during batch processing.") # Log full traceback
            self.after(0, lambda: messagebox.showerror("Error", f"An error occurred during batch processing: {str(e)}"))
            self.after(0, self.processing_done)
    
    def show_about_dialog(self):
        """Displays an 'About' dialog for the application."""
        about_text = (
            "Batch Video Inpainting Application\n"
            "Version: 1.0\n"
            "This tool processes 'splatted' videos to fill occlusions using a Stable Video Diffusion inpainting pipeline.\n"
            "It supports custom mask processing, color transfer, and post-inpainting blending for high-quality outputs.\n\n"
            "Developed by [Your Name/Alias] for StereoCrafter projects." # Customize this!
        )
        messagebox.showinfo("About Batch Video Inpainting", about_text)
    
    def start_processing(self):
        input_folder = self.input_folder_var.get()
        output_folder = self.output_folder_var.get()
        try:
            num_inference_steps = int(self.num_inference_steps_var.get())
            tile_num = int(self.tile_num_var.get())
            frames_chunk = int(self.frames_chunk_var.get())
            gui_overlap = int(self.overlap_var.get())
            gui_original_input_blend_strength = float(self.original_input_blend_strength_var.get())
            gui_output_crf = int(self.output_crf_var.get()) # NEW: Get CRF
            # Get Process Length and Validate
            process_length = int(self.process_length_var.get())
            if process_length != -1 and process_length <= 0:
                raise ValueError("Process Length must be -1 or a positive integer.")
            
            if num_inference_steps < 1 or tile_num < 1 or frames_chunk < 1 or gui_overlap  < 0 or \
               not (0.0 <= gui_original_input_blend_strength  <= 1.0) or gui_output_crf < 0: # NEW VALIDATION for CRF
                raise ValueError("Invalid parameter values")
        except ValueError:
            # UPDATED ERROR MESSAGE
            messagebox.showerror("Error", "Please enter valid values: Inference Steps >=1, Tile Number >=1, Frames Chunk >=1, Frame Overlap >=0, Original Input Bias between 0.0 and 1.0, Output CRF >=0.")
            return
        offload_type = self.offload_type_var.get()

        if not os.path.isdir(input_folder) or not os.path.isdir(output_folder):
            messagebox.showerror("Error", "Invalid input or output folder")
            return

        self.processed_count.set(0)
        self.total_videos.set(0)
        self.stop_event.clear()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.update_status_label("Starting processing...")
        self.update_video_info_display("N/A", "N/A", "N/A", "N/A", "N/A")

        threading.Thread(target=self.run_batch_process,
                         args=(input_folder, output_folder, num_inference_steps, tile_num, offload_type, frames_chunk, gui_overlap, gui_original_input_blend_strength, gui_output_crf, process_length),
                         daemon=True).start()

    def stop_processing(self):
        self.stop_event.set()
        if self.pipeline:
            # Attempt to clear CUDA cache if pipeline exists
            try:
                release_cuda_memory()
            except RuntimeError as e:
                logger.warning(f"Failed to clear CUDA cache: {e}")
        self.update_status_label("Stopping...")

    def update_progress(self):
        total = self.total_videos.get()
        processed = self.processed_count.get()
        if total > 0:
            progress = (processed / total) * 100
            self.progress_bar['value'] = progress
        else:
            self.progress_bar['value'] = 0
            # Status label is updated directly by start/run_batch_process/processing_done
        self.after(100, self.update_progress) # Schedule next update

    def update_status_label(self, message):
        self.status_label.config(text=message)

    def update_video_info_display(self, name, resolution, frames, overlap_val="N/A", bias_val="N/A"):
        self.video_name_var.set(name)
        self.video_res_var.set(resolution)
        self.video_frames_var.set(frames)
        self.video_overlap_var.set(overlap_val)
        self.video_bias_var.set(bias_val)

    def load_config(self):
        try:
            with open("config_inpaint.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def load_help_data(self):
        try:
            with open(os.path.join("dependency", "inpaint_help.json"), "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("dependency/inpaint_help.json not found. No help tips will be available.")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding inpaint_help.json: {e}")
            return {}

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
            
            # Iterate through the loaded config and apply values to the correct instance attributes
            for key, value in loaded_config.items():
                
                # 1. Try to find a corresponding tk.Variable (e.g., 'input_folder' -> 'input_folder_var')
                var_attr_name = key + "_var"

                if hasattr(self, var_attr_name):
                    var_instance = getattr(self, var_attr_name)
                    
                    if isinstance(var_instance, tk.BooleanVar):
                        var_instance.set(bool(value))
                    elif isinstance(var_instance, tk.StringVar):
                        var_instance.set(str(value))
                    else:
                        logger.warning(f"Skipping config key {key}: unknown tk.Variable type.")
                
                # 2. Handle direct instance attributes (e.g., window position/size)
                elif hasattr(self, key) and key in ["window_x", "window_y", "window_width"]:
                    setattr(self, key, value)
                
                else:
                    logger.debug(f"Skipping config key {key}: No matching tk.Variable or direct attribute found.")

            self._apply_theme() # Re-apply theme in case dark mode setting was loaded
            # --- FIX: Correct function name for updating blend fields state ---
            self._toggle_blend_parameters_state() # Update state of dependent fields
            # --- END FIX ---
            
            messagebox.showinfo("Settings Loaded", f"Successfully loaded settings from:\n{os.path.basename(filename)}")
            self.status_label.config(text="Settings loaded.")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load settings from {os.path.basename(filename)}:\n{e}")
            self.status_label.config(text="Settings load failed.")

    def save_config(self):
        config = self._get_current_config()
        try:
            with open("config_inpaint.json", "w", encoding='utf-8') as f: # Added encoding for robustness
                json.dump(config, f, indent=4)
            logger.info("Configuration saved successfully.")
        except Exception as e:
            logger.warning(f"Failed to save config: {e}", exc_info=True)

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
            with open(filename, "w", encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4)

            messagebox.showinfo("Settings Saved", f"Successfully saved settings to:\n{os.path.basename(filename)}")
            self.status_label.config(text="Settings saved.")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save settings to {os.path.basename(filename)}:\n{e}")
            self.status_label.config(text="Settings save failed.")
    
    def show_general_help(self):
        help_text = self.help_data.get("general_help", "No general help information available.")
        messagebox.showinfo("Help", help_text)

    def exit_application(self):
        self.save_config() 
        self.destroy()

def read_video_frames(video_path: str, decord_ctx=cpu(0)) -> Tuple[torch.Tensor, float, Optional[dict]]:
    """
    Reads a video using decord and returns frames as a 4D float tensor [T, C, H, W], the FPS,
    and video stream metadata.
    """
    video_stream_info = get_video_stream_info(video_path)

    video_reader = VideoReader(video_path, ctx=decord_ctx)
    num_frames = len(video_reader)

    if num_frames == 0:
        return torch.empty(0), 0.0, video_stream_info

    # Use ffprobe's detected FPS if available and reliable, otherwise decord's
    fps = 0.0
    if video_stream_info and "r_frame_rate" in video_stream_info:
        try:
            r_frame_rate_str = video_stream_info["r_frame_rate"].split('/')
            if len(r_frame_rate_str) == 2:
                fps = float(r_frame_rate_str[0]) / float(r_frame_rate_str[1])
            else:
                fps = float(r_frame_rate_str[0])
            logger.debug(f"Using ffprobe FPS: {fps:.2f} for {os.path.basename(video_path)}")
        except (ValueError, ZeroDivisionError):
            fps = video_reader.get_avg_fps()
            logger.warning(f"Failed to parse ffprobe FPS. Falling back to Decord FPS: {fps:.2f}")
    else:
        fps = video_reader.get_avg_fps()
        logger.debug(f"Using Decord FPS: {fps:.2f} for {os.path.basename(video_path)}")

    frames = video_reader.get_batch(range(num_frames))
    frames = torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2).float()

    return frames, fps, video_stream_info

def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """
    Blend two tensors horizontally along the right edge of `a` and left edge of `b`.
    """
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1) / overlap_size).to(b.device)
    b[:, :, :, :overlap_size] = (
        (1 - weight_b) * a[:, :, :, -overlap_size:] + weight_b * b[:, :, :, :overlap_size]
    )
    return b

def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """
    Blend two tensors vertically along the bottom edge of `a` and top edge of `b`.
    """
    weight_b = (torch.arange(overlap_size).view(1, 1, -1, 1) / overlap_size).to(b.device)
    b[:, :, :overlap_size, :] = (
        (1 - weight_b) * a[:, :, -overlap_size:, :] + weight_b * b[:, :, :overlap_size, :]
    )
    return b

def pad_for_tiling(frames: torch.Tensor, tile_num: int, tile_overlap=(128, 128)) -> torch.Tensor:
    """
    Zero-pads a batch of frames (shape [T, C, H, W]) so that (H, W) fits perfectly into 'tile_num' splits plus overlap.
    """
    if tile_num <= 1:
        return frames

    T, C, H, W = frames.shape
    overlap_y, overlap_x = tile_overlap

    # Calculate ideal tile dimensions and strides
    # Ensure stride is at least 1 to avoid infinite loops or zero-sized tiles with small inputs
    stride_y = max(1, (H + overlap_y * (tile_num - 1)) // tile_num - overlap_y)
    stride_x = max(1, (W + overlap_x * (tile_num - 1)) // tile_num - overlap_x)
    
    # Recalculate size_y and size_x based on minimum stride
    size_y = stride_y + overlap_y
    size_x = stride_x + overlap_x

    ideal_H = stride_y * tile_num + overlap_y
    ideal_W = stride_x * tile_num + overlap_x

    pad_bottom = max(0, ideal_H - H)
    pad_right = max(0, ideal_W - W)

    if pad_bottom > 0 or pad_right > 0:
        logger.debug(f"Padding frames from ({H}x{W}) to ({H+pad_bottom}x{W+pad_right}) for tiling.")
        frames = F.pad(frames, (0, pad_right, 0, pad_bottom), mode="constant", value=0.0)
    return frames

def spatial_tiled_process(
    cond_frames: torch.Tensor,
    mask_frames: torch.Tensor,
    process_func,
    tile_num: int,
    spatial_n_compress: int = 8,
    num_inference_steps: int = 5,
    **kwargs,
) -> torch.Tensor:
    """
    Splits frames into tiles, processes them with `process_func`, then blends the results back together.
    """
    height = cond_frames.shape[2]
    width = cond_frames.shape[3]

    tile_overlap = (128, 128)
    overlap_y, overlap_x = tile_overlap

    # Calculate tile sizes and strides, ensuring minimum stride
    size_y = (height + overlap_y * (tile_num - 1)) // tile_num
    size_x = (width + overlap_x * (tile_num - 1)) // tile_num
    tile_size = (size_y, size_x)

    tile_stride = (max(1, size_y - overlap_y), max(1, size_x - overlap_x)) # Ensure stride is at least 1

    cols = []
    for i in range(tile_num):
        row_tiles = []
        for j in range(tile_num):
            y_start = i * tile_stride[0]
            x_start = j * tile_stride[1]
            y_end = y_start + tile_size[0]
            x_end = x_start + tile_size[1]

            # Ensure bounds do not exceed original image dimensions if padding was used
            y_end = min(y_end, height)
            x_end = min(x_end, width)

            cond_tile = cond_frames[:, :, y_start:y_end, x_start:x_end]
            mask_tile = mask_frames[:, :, y_start:y_end, x_start:x_end]

            if cond_tile.numel() == 0 or mask_tile.numel() == 0:
                logger.warning(f"Skipping empty tile: y_start={y_start}, y_end={y_end}, x_start={x_start}, x_end={x_end}")
                # Append a zero tensor of expected latent output size to keep structure consistent
                # This needs careful consideration if `tile_output` becomes empty, it could break blending.
                # A better approach for empty tiles might be to just skip and fill later, or ensure valid tiles.
                # For simplicity, assuming pipeline handles small/empty inputs gracefully or valid tiles are always generated.
                # Here, we'll try to let the pipeline handle it, or it will error out if it can't.
                pass # Let the process_func handle if it gets an empty tile.

            with torch.no_grad():
                tile_output = process_func(
                    frames=cond_tile,
                    frames_mask=mask_tile,
                    height=cond_tile.shape[2],
                    width=cond_tile.shape[3],
                    num_frames=len(cond_tile),
                    output_type="latent",
                    num_inference_steps=num_inference_steps,
                    **kwargs,
                ).frames[0]

            row_tiles.append(tile_output)
        cols.append(row_tiles)

    latent_stride = (
        tile_stride[0] // spatial_n_compress,
        tile_stride[1] // spatial_n_compress
    )
    latent_overlap = (
        overlap_y // spatial_n_compress,
        overlap_x // spatial_n_compress
    )

    blended_rows = []
    for i, row_tiles in enumerate(cols):
        row_result = []
        for j, tile in enumerate(row_tiles):
            if i > 0:
                # Ensure the previous tile exists for blending
                if len(cols[i - 1]) > j and cols[i - 1][j] is not None:
                    tile = blend_v(cols[i - 1][j], tile, latent_overlap[0])
            if j > 0:
                # Ensure the previous tile in the row exists for blending
                if len(row_result) > j - 1 and row_result[j - 1] is not None:
                    tile = blend_h(row_result[j - 1], tile, latent_overlap[1])
            row_result.append(tile)
        blended_rows.append(row_result)

    final_rows = []
    for i, row_tiles in enumerate(blended_rows):
        for j, tile in enumerate(row_tiles):
            if tile is None:
                logger.warning(f"Skipping None tile during final row concatenation at ({i}, {j})")
                continue # Skip None tiles, this might cause dimension mismatch later if not handled

            # Ensure the slice is valid and does not result in empty tensor
            if i < len(blended_rows) - 1:
                if latent_stride[0] > 0:
                    tile = tile[:, :, :latent_stride[0], :]
                else:
                    logger.warning(f"latent_stride[0] is zero, skipping vertical crop for tile ({i}, {j}).")
            if j < len(row_tiles) - 1:
                if latent_stride[1] > 0:
                    tile = tile[:, :, :, :latent_stride[1]]
                else:
                    logger.warning(f"latent_stride[1] is zero, skipping horizontal crop for tile ({i}, {j}).")
            row_tiles[j] = tile
        
        # Filter out None tiles before concatenation
        valid_row_tiles = [t for t in row_tiles if t is not None]
        if valid_row_tiles:
            final_rows.append(torch.cat(valid_row_tiles, dim=3))
        else:
            logger.warning(f"Row {i} ended up empty after filtering None tiles.")

    if not final_rows:
        logger.error("No final rows to concatenate after spatial tiling. This indicates a major issue with tile processing or blending.")
        raise ValueError("Spatial tiling failed to produce any valid output rows.")

    x = torch.cat(final_rows, dim=2)

    return x

if __name__ == "__main__":
    app = InpaintingGUI()
    app.mainloop()