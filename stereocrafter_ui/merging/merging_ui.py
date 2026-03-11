"""
Merging WebUI Component
"""


import os
import threading
import queue
import glob
import time
import json
import shutil
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
import gradio as gr
from decord import VideoReader, cpu

from ..base.base_ui import BaseWebUI
from core.common.cli_utils import draw_progress_bar
from core.common.gpu_utils import release_cuda_memory
from core.common.video_io import get_video_stream_info, encode_frames_to_mp4, start_ffmpeg_pipe_process
from core.common.image_processing import apply_color_transfer, apply_dubois_anaglyph, apply_optimized_anaglyph
import logging

logger = logging.getLogger(__name__)

# --- MASK PROCESSING FUNCTIONS (Ported from merging_gui.py) ---
def apply_mask_dilation(mask: torch.Tensor, kernel_size: int, use_gpu: bool = True) -> torch.Tensor:
    if kernel_size <= 0: return mask
    kernel_val = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    if use_gpu and mask.is_cuda:
        padding = kernel_val // 2
        return F.max_pool2d(mask, kernel_size=kernel_val, stride=1, padding=padding)
    else:
        # CPU fallback using OpenCV
        processed_frames = []
        # Ensure mask is on CPU for numpy conversion
        mask_cpu = mask.cpu()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_val, kernel_val))
        for t in range(mask.shape[0]):
            # Squeeze channel dim if present (C=1)
            frame_np = (mask_cpu[t].squeeze(0).numpy() * 255).astype(np.uint8)
            dilated_np = cv2.dilate(frame_np, kernel, iterations=1)
            dilated_tensor = torch.from_numpy(dilated_np).float() / 255.0
            processed_frames.append(dilated_tensor.unsqueeze(0)) # Add back channel dim
        return torch.stack(processed_frames).to(mask.device)

def apply_gaussian_blur(mask: torch.Tensor, kernel_size: int, use_gpu: bool = True) -> torch.Tensor:
    if kernel_size <= 0: return mask
    kernel_val = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    if use_gpu and mask.is_cuda:
        sigma = kernel_val / 6.0
        ax = torch.arange(-kernel_val // 2 + 1., kernel_val // 2 + 1., device=mask.device)
        gauss = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        kernel_1d = (gauss / gauss.sum()).view(1, 1, 1, kernel_val)
        blurred_mask = F.conv2d(mask, kernel_1d, padding=(0, kernel_val // 2), groups=mask.shape[1])
        blurred_mask = F.conv2d(blurred_mask, kernel_1d.permute(0, 1, 3, 2), padding=(kernel_val // 2, 0), groups=mask.shape[1])
        return torch.clamp(blurred_mask, 0.0, 1.0)
    else:
        processed_frames = []
        mask_cpu = mask.cpu()
        for t in range(mask.shape[0]):
            frame_np = (mask_cpu[t].squeeze(0).numpy() * 255).astype(np.uint8)
            blurred_np = cv2.GaussianBlur(frame_np, (kernel_val, kernel_val), 0)
            blurred_tensor = torch.from_numpy(blurred_np).float() / 255.0
            processed_frames.append(blurred_tensor.unsqueeze(0))
        return torch.stack(processed_frames).to(mask.device)

def apply_shadow_blur(mask: torch.Tensor, shift_per_step: int, start_opacity: float, opacity_decay_per_step: float, min_opacity: float, decay_gamma: float = 1.0, use_gpu: bool = True) -> torch.Tensor:
    if shift_per_step <= 0: return mask
    if opacity_decay_per_step <= 1e-6: return mask
    
    num_steps = int((start_opacity - min_opacity) / opacity_decay_per_step) + 1
    if num_steps <= 0: return mask

    if use_gpu and mask.is_cuda:
        canvas_mask = mask.clone()
        stamp_source = mask.clone()
        for i in range(num_steps):
            t = 1.0 - (i / (num_steps - 1)) if num_steps > 1 else 1.0
            curved_t = t ** decay_gamma
            current_opacity = min_opacity + (start_opacity - min_opacity) * curved_t
            total_shift = (i + 1) * shift_per_step
            # Shift horizontally (assuming NCHW layout)
            padded_stamp = F.pad(stamp_source, (total_shift, 0), "constant", 0)
            shifted_stamp = padded_stamp[:, :, :, :-total_shift]
            canvas_mask = torch.max(canvas_mask, shifted_stamp * current_opacity)
        return canvas_mask
    else:
        processed_frames = []
        mask_cpu = mask.cpu()
        for t in range(mask.shape[0]):
            canvas_np = mask_cpu[t].squeeze(0).numpy()
            stamp_source_np = canvas_np.copy()
            for i in range(num_steps):
                time_step = 1.0 - (i / (num_steps - 1)) if num_steps > 1 else 1.0
                curved_t = time_step ** decay_gamma
                current_opacity = min_opacity + (start_opacity - min_opacity) * curved_t
                total_shift = (i + 1) * shift_per_step
                shifted_stamp = np.roll(stamp_source_np, total_shift, axis=1) # axis=1 for HxW (since squeezed)
                # Handle edge case where roll wraps around - we want 0 filling
                shifted_stamp[:, :total_shift] = 0 
                canvas_np = np.maximum(canvas_np, shifted_stamp * current_opacity)
            processed_frames.append(torch.from_numpy(canvas_np).unsqueeze(0))
        return torch.stack(processed_frames).to(mask.device)

def _find_video_by_core_name(folder: str, core_name: str) -> str:
    """Scans a folder for a file matching the core_name with any common video extension."""
    if not os.path.exists(folder): return None
    video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm')
    for ext in video_extensions:
        # Simple glob check - might miss if strict core_name is needed
        full_path = os.path.join(folder, f"{core_name}{ext[1:]}")
        if os.path.exists(full_path):
            return full_path
    
    # Fallback search if exact match fails
    files = glob.glob(os.path.join(folder, f"{core_name}.*"))
    for f in files:
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            return f
    return None

class MergingWebUI:
    def __init__(self):
        # Initialize all the variables that were in the original GUI
        self.stop_event = threading.Event()
        self.progress_queue = queue.Queue()
        
        # Folder paths
        self.inpainted_folder = "./completed_output"
        self.original_folder = "./input_source_clips"
        self.mask_folder = "./output_splatted/hires"
        self.output_folder = "./final_videos"

        # Mask processing parameters (with proper defaults from GUI)
        self.mask_binarize_threshold = 0.3
        self.mask_dilate_kernel_size = 3
        self.mask_blur_kernel_size = 5
        self.shadow_shift = 5
        self.shadow_decay_gamma = 1.3
        self.shadow_start_opacity = 0.87
        self.shadow_opacity_decay = 0.08
        self.shadow_min_opacity = 0.14

        # Options
        self.use_gpu = False
        self.output_format = "Full SBS (Left-Right)"
        self.pad_to_16_9 = False
        self.enable_color_transfer = True
        self.batch_chunk_size = 20
        
        # Preview cache for faster updates
        self._frame_cache = {}  # Cache loaded frames
        self._last_video_path = None
        self._last_frame_index = None

    def _scan_for_preview_videos(self, inpainted_folder):
        """Scans folders to find valid videos for preview."""
        # For merging preview, we should look at SPLATTED files (which have masks)
        # not inpainted files (which are final outputs without masks)
        mask_folder = self.mask_folder
        
        if not os.path.exists(mask_folder):
            return []
        
        # Find splatted videos (these have the mask data)
        all_mp4s = sorted(glob.glob(os.path.join(mask_folder, "**", "*.mp4"), recursive=True))
        splatted_videos = [v for v in all_mp4s if ('_splatted2' in os.path.basename(v) or 
                                                   '_splatted4' in os.path.basename(v))]
        
        video_names = [os.path.basename(v) for v in splatted_videos]
        return gr.Dropdown(choices=video_names, value=video_names[0] if video_names else None)

    def generate_preview(self, input_folder, video_name, frame_index, preview_mode,
                        use_gpu, mask_threshold, mask_dilate_kernel, mask_blur_kernel,
                        shadow_shift, shadow_start_opacity, shadow_opacity_decay,
                        shadow_min_opacity, shadow_decay_gamma):
        """Wrapper for _get_preview_frame with simplified signature"""
        # For merging, input_folder should be the mask folder (where splatted files are)
        return self._get_preview_frame(
            input_folder, video_name, preview_mode, frame_index,
            mask_threshold, mask_dilate_kernel, mask_blur_kernel,
            shadow_shift, shadow_decay_gamma, shadow_start_opacity,
            shadow_opacity_decay, shadow_min_opacity,
            use_gpu, self.enable_color_transfer
        )

    def on_video_select(self, folder, video_name):
        """Handle video selection to update slider range"""
        if not video_name or not folder:
            return gr.Slider(value=0, maximum=1), "0", "Ready"
        
        # Search for video in folder and subfolders
        video_path = None
        for root, dirs, files in os.walk(folder):
            if video_name in files:
                video_path = os.path.join(root, video_name)
                break
                
        if not video_path or not os.path.exists(video_path):
             return gr.Slider(value=0, maximum=1), "Error", f"File not found: {video_name}"
             
        try:
            reader = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(reader)
            return gr.Slider(value=0, maximum=total_frames - 1, step=1), str(total_frames), f"Loaded {video_name}"
        except Exception as e:
            return gr.Slider(value=0, maximum=1), "Error", f"Error loading video: {e}"


    def _get_preview_frame(self, input_folder, video_name, preview_mode, frame_index, 
                          mask_threshold, mask_dilate_kernel, mask_blur_kernel,
                          shadow_shift, shadow_decay_gamma, shadow_start_opacity,
                          shadow_opacity_decay, shadow_min_opacity,
                          use_gpu, enable_color_transfer):
        """Generate preview image based on current settings"""
        try:
            if not video_name or not input_folder:
                 return None, "No video selected", "0"

            # Search for video in folder and subfolders (for mask folder structure)
            video_path = None
            for root, dirs, files in os.walk(input_folder):
                if video_name in files:
                    video_path = os.path.join(root, video_name)
                    break
            
            if not video_path or not os.path.exists(video_path):
                return None, f"Video not found: {video_name}", "0"
            
            frame_index = int(frame_index)
            
            # Check cache for faster preview
            cache_key = f"{video_path}_{frame_index}"
            if cache_key in self._frame_cache:
                frame = self._frame_cache[cache_key]
            else:
                # Load video info
                reader = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(reader)
                frame_index = max(0, min(frame_index, total_frames - 1))
                
                # Load frame
                frame = reader.get_batch([frame_index]).asnumpy()[0]  # [H, W, C]
                
                # Cache the frame (limit cache size to 10 frames)
                if len(self._frame_cache) > 10:
                    self._frame_cache.clear()
                self._frame_cache[cache_key] = frame
            
            # Determine if dual or quad
            height, width = frame.shape[0], frame.shape[1]
            basename = os.path.basename(video_path)
            is_dual = '_splatted2' in basename
            is_left_right = False 

            if is_dual:
                # Dual: Left is mask, Right is warped
                half_w = width // 2
                occlu_frame = frame[:, :half_w, :]
                warped_frame = frame[:, half_w:, :]
                source_frame = warped_frame  # No source available in dual splat usually
                depth_frame = occlu_frame  # No depth in dual
            elif '_splatted4' in basename or (height == width): 
                # Quad: TL=source, TR=depth, BL=mask, BR=warped
                half_h, half_w = height // 2, width // 2
                source_frame = frame[:half_h, :half_w, :]
                depth_frame = frame[:half_h, half_w:, :]  # TOP RIGHT
                occlu_frame = frame[half_h:, :half_w, :]
                warped_frame = frame[half_h:, half_w:, :]
            elif '_inpainted' in basename:
                # If previewing an inpainted file
                # Assuming SBS result for preview
                half_w = width // 2
                source_frame = frame[:, :half_w, :] 
                warped_frame = frame[:, half_w:, :] # Right eye (inpainted)
                occlu_frame = np.zeros_like(source_frame) # No mask in final output
                depth_frame = source_frame  # No depth
                is_left_right = True
            else:
                 # Fallback
                half_w = width // 2
                occlu_frame = frame[:, :half_w, :]
                warped_frame = frame[:, half_w:, :]
                source_frame = warped_frame
                depth_frame = occlu_frame

            # Generate preview based on mode
            if preview_mode == 'Blended Image':
                # Show actual blended result with mask processing
                occlu_tensor = torch.from_numpy(occlu_frame).permute(2, 0, 1).float() / 255.0
                if occlu_tensor.shape[0] == 3:
                    mask_tensor = occlu_tensor.mean(dim=0, keepdim=True)
                else:
                    mask_tensor = occlu_tensor
                mask_tensor = mask_tensor.unsqueeze(0)  # [1, 1, H, W]
                
                dev = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
                mask_tensor = mask_tensor.to(dev)
                
                # Apply mask processing
                if mask_threshold >= 0:
                    mask_tensor = (mask_tensor > mask_threshold).float()
                if mask_dilate_kernel > 0:
                    mask_tensor = apply_mask_dilation(mask_tensor, int(mask_dilate_kernel), use_gpu=(dev=="cuda"))
                if mask_blur_kernel > 0:
                    mask_tensor = apply_gaussian_blur(mask_tensor, int(mask_blur_kernel), use_gpu=(dev=="cuda"))
                if shadow_shift > 0:
                    mask_tensor = apply_shadow_blur(
                        mask_tensor, int(shadow_shift), shadow_start_opacity, 
                        shadow_opacity_decay, shadow_min_opacity, shadow_decay_gamma, 
                        use_gpu=(dev=="cuda")
                    )
                
                # Blend source and warped using processed mask
                source_tensor = torch.from_numpy(source_frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                warped_tensor = torch.from_numpy(warped_frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                
                # Resize if needed
                if source_tensor.shape[2:] != mask_tensor.shape[2:]:
                    source_tensor = F.interpolate(source_tensor, size=mask_tensor.shape[2:], mode='bilinear')
                if warped_tensor.shape[2:] != mask_tensor.shape[2:]:
                    warped_tensor = F.interpolate(warped_tensor, size=mask_tensor.shape[2:], mode='bilinear')
                
                source_tensor = source_tensor.to(dev)
                warped_tensor = warped_tensor.to(dev)
                
                # Blend: warped * (1-mask) + source * mask (assuming inpainted replaces masked areas)
                blended = warped_tensor * (1 - mask_tensor) + source_tensor * mask_tensor
                blended_np = (blended.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                preview_np = blended_np
                
            elif preview_mode == 'Original (Left Eye)':
                preview_np = source_frame
            elif preview_mode == 'Warped (Right BG)':
                preview_np = warped_frame
            elif preview_mode == 'Inpainted Right Eye':
                # If viewing inpainted file, show right eye
                if '_inpainted' in basename:
                    preview_np = warped_frame
                else:
                    preview_np = source_frame
            elif preview_mode == 'Depth Map':
                print(f"[DEBUG] Depth Map mode - basename: {basename}, is_dual: {is_dual}")
                print(f"[DEBUG] Frame shape: {frame.shape}, depth_frame shape: {depth_frame.shape}")
                print(f"[DEBUG] Depth frame min: {depth_frame.min()}, max: {depth_frame.max()}, mean: {depth_frame.mean()}")
                preview_np = depth_frame
            elif preview_mode == 'Anaglyph 3D':
                # Standard Red-Cyan Anaglyph
                anaglyph = np.zeros_like(source_frame)
                anaglyph[:, :, 0] = source_frame[:, :, 0]  # Red from left
                anaglyph[:, :, 1] = warped_frame[:, :, 1]  # Green from right
                anaglyph[:, :, 2] = warped_frame[:, :, 2]  # Blue from right
                preview_np = anaglyph
            elif preview_mode == 'Dubois Anaglyph':
                # Use Dubois anaglyph algorithm
                preview_np = apply_dubois_anaglyph(source_frame, warped_frame)
            elif preview_mode == 'Optimized Anaglyph':
                # Use optimized anaglyph algorithm
                preview_np = apply_optimized_anaglyph(source_frame, warped_frame)
            elif preview_mode == 'Wigglegram':
                # Create a 2-frame GIF for wiggling
                img_left = PILImage.fromarray(source_frame)
                img_right = PILImage.fromarray(warped_frame)
                temp_gif_path = os.path.join(input_folder, f"preview_wiggle_{basename}_{frame_index}.gif")
                img_left.save(
                    temp_gif_path,
                    save_all=True,
                    append_images=[img_right],
                    duration=150,
                    loop=0
                )
                return temp_gif_path, basename, str(frame_index)
            elif preview_mode == 'Processed Mask':
                # Process mask with all parameters - FIXED VERSION
                print(f"[DEBUG] Processed Mask mode - occlu_frame shape: {occlu_frame.shape}")
                print(f"[DEBUG] Occlu frame min: {occlu_frame.min()}, max: {occlu_frame.max()}, mean: {occlu_frame.mean()}")
                
                occlu_tensor = torch.from_numpy(occlu_frame).permute(2, 0, 1).float() / 255.0
                
                # Convert to grayscale if RGB
                if occlu_tensor.shape[0] == 3:
                    mask_tensor = occlu_tensor.mean(dim=0, keepdim=True)
                else:
                    mask_tensor = occlu_tensor[0:1]  # Take first channel

                print(f"[DEBUG] Mask tensor shape after grayscale: {mask_tensor.shape}")
                print(f"[DEBUG] Mask tensor min: {mask_tensor.min()}, max: {mask_tensor.max()}, mean: {mask_tensor.mean()}")

                mask_tensor = mask_tensor.unsqueeze(0)  # [1, 1, H, W]
                dev = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
                mask_tensor = mask_tensor.to(dev)

                # Apply threshold
                if mask_threshold >= 0:
                    mask_tensor = (mask_tensor > mask_threshold).float()
                    print(f"[DEBUG] After threshold {mask_threshold}: min={mask_tensor.min()}, max={mask_tensor.max()}, mean={mask_tensor.mean()}")

                # Apply dilation
                if mask_dilate_kernel > 0:
                    mask_tensor = apply_mask_dilation(mask_tensor, int(mask_dilate_kernel), use_gpu=(dev=="cuda"))
                    print(f"[DEBUG] After dilation {mask_dilate_kernel}: min={mask_tensor.min()}, max={mask_tensor.max()}, mean={mask_tensor.mean()}")

                # Apply blur
                if mask_blur_kernel > 0:
                    mask_tensor = apply_gaussian_blur(mask_tensor, int(mask_blur_kernel), use_gpu=(dev=="cuda"))
                    print(f"[DEBUG] After blur {mask_blur_kernel}: min={mask_tensor.min()}, max={mask_tensor.max()}, mean={mask_tensor.mean()}")
                
                # Apply shadows
                if shadow_shift > 0:
                    mask_tensor = apply_shadow_blur(
                        mask_tensor, int(shadow_shift), shadow_start_opacity, 
                        shadow_opacity_decay, shadow_min_opacity, shadow_decay_gamma, 
                        use_gpu=(dev=="cuda")
                    )
                    print(f"[DEBUG] After shadow {shadow_shift}: min={mask_tensor.min()}, max={mask_tensor.max()}, mean={mask_tensor.mean()}")

                # Convert to displayable image - FIXED
                mask_processed = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel dims
                print(f"[DEBUG] Mask processed shape: {mask_processed.shape}, min: {mask_processed.min()}, max: {mask_processed.max()}")
                mask_uint8 = (np.clip(mask_processed, 0, 1) * 255).astype(np.uint8)
                print(f"[DEBUG] Mask uint8 min: {mask_uint8.min()}, max: {mask_uint8.max()}")
                preview_np = np.stack([mask_uint8]*3, axis=2)  # Convert to RGB
                print(f"[DEBUG] Final preview_np shape: {preview_np.shape}, min: {preview_np.min()}, max: {preview_np.max()}")
            else:
                # Fallback to source frame
                preview_np = source_frame

            # Convert to PIL Image
            preview_pil = PILImage.fromarray(preview_np)

            return preview_pil, basename, str(frame_index)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Preview generation error: {e}")
            return None, f"Error: {str(e)}", "0"


    def create_interface(self):
        """Creates the Gradio interface matching the GUI layout"""
        
        with gr.Blocks(title="Merging - StereoCrafter") as interface:
            gr.Markdown("## 🎬 Stereocrafter Merging")
            
            # Folders Section at TOP (full width)
            with gr.Group():
                gr.Markdown("### Folders")
                with gr.Row():
                    inpainted_folder_input = gr.Textbox(
                        label="Inpainted Video Folder",
                        value=self.inpainted_folder,
                        scale=2
                    )
                    original_folder_input = gr.Textbox(
                        label="Original Video Folder",
                        value=self.original_folder,
                        scale=2
                    )
                with gr.Row():
                    mask_folder_input = gr.Textbox(
                        label="Mask Folder",
                        value=self.mask_folder,
                        scale=2
                    )
                    output_folder_input = gr.Textbox(
                        label="Output Folder",
                        value=self.output_folder,
                        scale=2
                    )
            
            # Main layout: Side-by-side (Preview LEFT, Parameters RIGHT)
            with gr.Row():
                # LEFT COLUMN: Preview Section
                with gr.Column(scale=3):
                    gr.Markdown("### Preview")
                    preview_image = gr.Image(label="", interactive=False, height=600)
                    
                    # Frame slider
                    frame_slider = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0,
                        step=1,
                        label="Frame: 0 / 0",
                        interactive=True
                    )
                    
                    # Preview controls
                    with gr.Row():
                        prev_btn = gr.Button("< Prev", scale=1)
                        next_btn = gr.Button("Next >", scale=1)
                        jump_to_input = gr.Number(label="Jump to", value=1, precision=0, scale=1)
                    
                    with gr.Row():
                        preview_source = gr.Dropdown(
                            choices=["Blended Image", "Original (Left Eye)", "Warped (Right BG)", 
                                    "Inpainted Right Eye", "Processed Mask", "Depth Map",
                                    "Anaglyph 3D", "Dubois Anaglyph", "Optimized Anaglyph", "Wigglegram"],
                            value="Blended Image",
                            label="Preview Source",
                            scale=2
                        )
                        load_preview_btn = gr.Button("Load/Refresh List", scale=1)
                    
                    preview_video_dropdown = gr.Dropdown(
                        label="Video: 0 / 0",
                        choices=[],
                        interactive=True
                    )
                
                # RIGHT COLUMN: Parameters & Controls
                with gr.Column(scale=2):
                    # Mask Processing Parameters Section
                    with gr.Group():
                        gr.Markdown("### Mask Processing Parameters")
                        gr.Markdown("💡 **Tip:** Adjust parameters - preview updates on slider release")
                        
                        mask_binarize_threshold_input = gr.Slider(
                            minimum=-0.01,
                            maximum=1.0,
                            value=self.mask_binarize_threshold,
                            step=0.01,
                            label="Binarize Thresh (<0=Off)",
                            interactive=True
                        )
                        mask_dilate_kernel_input = gr.Slider(
                            minimum=0,
                            maximum=101,
                            value=self.mask_dilate_kernel_size,
                            step=1,
                            label="Dilate Kernel",
                            interactive=True
                        )
                        mask_blur_kernel_input = gr.Slider(
                            minimum=0,
                            maximum=101,
                            value=self.mask_blur_kernel_size,
                            step=1,
                            label="Blur Kernel",
                            interactive=True
                        )
                        shadow_shift_input = gr.Slider(
                            minimum=0,
                            maximum=50,
                            value=self.shadow_shift,
                            step=1,
                            label="Shadow Shift",
                            interactive=True
                        )
                        
                        with gr.Accordion("Advanced Shadow Settings", open=False):
                            shadow_gamma_input = gr.Slider(
                                minimum=0.1,
                                maximum=5.0,
                                value=self.shadow_decay_gamma,
                                step=0.01,
                                label="Shadow Gamma",
                                interactive=True
                            )
                            shadow_start_opacity_input = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=self.shadow_start_opacity,
                                step=0.01,
                                label="Shadow Opacity Start",
                                interactive=True
                            )
                            shadow_opacity_decay_input = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=self.shadow_opacity_decay,
                                step=0.01,
                                label="Shadow Opacity Decay",
                                interactive=True
                            )
                            shadow_min_opacity_input = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=self.shadow_min_opacity,
                                step=0.01,
                                label="Shadow Opacity Min",
                                interactive=True
                            )
                    
                    # Options Section
                    with gr.Group():
                        gr.Markdown("### Options")
                        with gr.Row():
                            use_gpu_input = gr.Checkbox(
                                label="Use GPU",
                                value=self.use_gpu
                            )
                            enable_color_transfer_input = gr.Checkbox(
                                label="Color Transfer",
                                value=self.enable_color_transfer
                            )
                            pad_to_16_9_input = gr.Checkbox(
                                label="Pad 16:9",
                                value=self.pad_to_16_9
                            )
                        output_format_input = gr.Dropdown(
                            choices=["Full SBS (Left-Right)", "Double SBS", "Half SBS (Left-Right)", 
                                    "Full SBS Cross-eye (Right-Left)", "Anaglyph (Red/Cyan)", 
                                    "Anaglyph Half-Color", "Right-Eye Only"],
                            value=self.output_format,
                            label="Output Format"
                        )
                        batch_chunk_size_input = gr.Number(
                            label="Batch Chunk Size",
                            value=self.batch_chunk_size,
                            precision=0
                        )
                    
                    # Progress Section
                    with gr.Group():
                        gr.Markdown("### Progress")
                        progress_bar = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            label="Progress",
                            interactive=False
                        )
                        status_label = gr.Textbox(
                            label="Status",
                            value="Ready",
                            interactive=False
                        )
                    
                    # Control Buttons
                    with gr.Row():
                        start_button = gr.Button("Start Blending", variant="primary")
                        stop_button = gr.Button("Stop", variant="secondary")
            
            # Event Handlers
            start_button.click(
                fn=self.start_processing,
                inputs=[
                    inpainted_folder_input, original_folder_input,
                    mask_folder_input, output_folder_input,
                    use_gpu_input, pad_to_16_9_input,
                    output_format_input, batch_chunk_size_input,
                    enable_color_transfer_input,
                    mask_binarize_threshold_input, mask_dilate_kernel_input,
                    mask_blur_kernel_input, shadow_shift_input,
                    shadow_gamma_input, shadow_start_opacity_input,
                    shadow_opacity_decay_input, shadow_min_opacity_input
                ],
                outputs=[status_label, progress_bar]
            )
            
            stop_button.click(
                fn=self.stop_processing,
                inputs=[],
                outputs=[status_label, progress_bar]
            )
            
            # Preview Event Handlers
            def load_videos_for_preview(mask_folder):
                """Load video list from mask folder (splatted files)"""
                try:
                    if not os.path.exists(mask_folder):
                        return gr.update(choices=[], value=None, label="Video: 0 / 0"), None, "No folder", gr.update(value=0, maximum=1, label="Frame: 0 / 0")
                    
                    # Find splatted videos recursively
                    videos = sorted(glob.glob(os.path.join(mask_folder, "**", "*.mp4"), recursive=True))
                    splatted_videos = [v for v in videos if ('_splatted2' in os.path.basename(v) or 
                                                             '_splatted4' in os.path.basename(v))]
                    
                    if not splatted_videos:
                        return gr.update(choices=[], value=None, label="Video: 0 / 0"), None, "No splatted videos found", gr.update(value=0, maximum=1, label="Frame: 0 / 0")
                    
                    video_names = [os.path.basename(v) for v in splatted_videos]
                    label = f"Video: 1 / {len(video_names)}"
                    
                    # Generate preview for first video, first frame
                    # Need to find the full path
                    first_video_path = splatted_videos[0]
                    first_video_folder = os.path.dirname(first_video_path)
                    
                    preview_img, basename, frame_str = self.generate_preview(
                        first_video_folder, video_names[0], 0, "Blended Image",
                        self.use_gpu, self.mask_binarize_threshold,
                        self.mask_dilate_kernel_size, self.mask_blur_kernel_size,
                        self.shadow_shift, self.shadow_start_opacity,
                        self.shadow_opacity_decay, self.shadow_min_opacity,
                        self.shadow_decay_gamma
                    )
                    
                    # Get frame count for slider
                    reader = VideoReader(first_video_path, ctx=cpu(0))
                    total_frames = len(reader)
                    
                    return (
                        gr.update(choices=video_names, value=video_names[0], label=label),
                        preview_img,
                        basename,
                        gr.update(value=0, maximum=total_frames-1, label=f"Frame: 0 / {total_frames}")
                    )
                except Exception as e:
                    print(f"Error loading videos: {e}")
                    import traceback
                    traceback.print_exc()
                    return gr.update(choices=[], value=None, label="Video: 0 / 0"), None, f"Error: {str(e)}", gr.update(value=0, maximum=1, label="Frame: 0 / 0")
            
            def update_preview_from_slider(mask_folder, video_name, frame_idx, preview_source,
                                          use_gpu, threshold, dilate, blur, shadow_shift,
                                          shadow_start, shadow_decay, shadow_min, shadow_gamma):
                """Update preview when slider changes"""
                if not video_name:
                    return None, "No video selected", gr.update()
                
                # Find video path in mask folder
                video_path = None
                for root, dirs, files in os.walk(mask_folder):
                    if video_name in files:
                        video_path = os.path.join(root, video_name)
                        break
                
                if not video_path:
                    return None, f"Video not found: {video_name}", gr.update()
                
                video_folder = os.path.dirname(video_path)
                
                # Pass preview_source directly as the mode
                preview_img, basename, frame_str = self.generate_preview(
                    video_folder, video_name, int(frame_idx), preview_source,
                    use_gpu, threshold, dilate, blur, shadow_shift,
                    shadow_start, shadow_decay, shadow_min, shadow_gamma
                )
                
                # Update frame label
                reader = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(reader)
                
                return preview_img, basename, gr.update(value=int(frame_idx), label=f"Frame: {int(frame_idx)} / {total_frames}")
            
            # Connect preview events
            load_preview_btn.click(
                fn=load_videos_for_preview,
                inputs=[mask_folder_input],
                outputs=[preview_video_dropdown, preview_image, status_label, frame_slider]
            )
            
            frame_slider.change(
                fn=update_preview_from_slider,
                inputs=[
                    mask_folder_input, preview_video_dropdown, frame_slider, preview_source,
                    use_gpu_input, mask_binarize_threshold_input, mask_dilate_kernel_input,
                    mask_blur_kernel_input, shadow_shift_input, shadow_start_opacity_input,
                    shadow_opacity_decay_input, shadow_min_opacity_input, shadow_gamma_input
                ],
                outputs=[preview_image, status_label, frame_slider]
            )
            
            preview_source.change(
                fn=update_preview_from_slider,
                inputs=[
                    mask_folder_input, preview_video_dropdown, frame_slider, preview_source,
                    use_gpu_input, mask_binarize_threshold_input, mask_dilate_kernel_input,
                    mask_blur_kernel_input, shadow_shift_input, shadow_start_opacity_input,
                    shadow_opacity_decay_input, shadow_min_opacity_input, shadow_gamma_input
                ],
                outputs=[preview_image, status_label, frame_slider]
            )
            
            # Auto-refresh when mask parameters change
            for param_slider in [mask_binarize_threshold_input, mask_dilate_kernel_input, 
                                mask_blur_kernel_input, shadow_shift_input, shadow_gamma_input,
                                shadow_start_opacity_input, shadow_opacity_decay_input, 
                                shadow_min_opacity_input]:
                param_slider.release(
                    fn=update_preview_from_slider,
                    inputs=[
                        mask_folder_input, preview_video_dropdown, frame_slider, preview_source,
                        use_gpu_input, mask_binarize_threshold_input, mask_dilate_kernel_input,
                        mask_blur_kernel_input, shadow_shift_input, shadow_start_opacity_input,
                        shadow_opacity_decay_input, shadow_min_opacity_input, shadow_gamma_input
                    ],
                    outputs=[preview_image, status_label, frame_slider]
                )
            
            preview_video_dropdown.change(
                fn=lambda folder, video: load_videos_for_preview(folder) if video else (gr.update(), None, "No video", gr.update(value=0, maximum=1)),
                inputs=[mask_folder_input, preview_video_dropdown],
                outputs=[preview_video_dropdown, preview_image, status_label, frame_slider]
            )
            
            # Navigation button handlers
            def prev_frame(folder, video, current_frame, preview_src, use_gpu, threshold, dilate, blur,
                          shadow_shift, shadow_start, shadow_decay, shadow_min, shadow_gamma):
                """Go to previous frame"""
                new_frame = max(0, int(current_frame) - 1)
                return update_preview_from_slider(folder, video, new_frame, preview_src, use_gpu,
                                                  threshold, dilate, blur, shadow_shift, shadow_start,
                                                  shadow_decay, shadow_min, shadow_gamma)
            
            def next_frame(folder, video, current_frame, preview_src, use_gpu, threshold, dilate, blur,
                          shadow_shift, shadow_start, shadow_decay, shadow_min, shadow_gamma):
                """Go to next frame"""
                # Find video path
                video_path = None
                for root, dirs, files in os.walk(folder):
                    if video in files:
                        video_path = os.path.join(root, video)
                        break
                
                if not video_path or not os.path.exists(video_path):
                    return None, "No video", gr.update()
                reader = VideoReader(video_path, ctx=cpu(0))
                max_frame = len(reader) - 1
                new_frame = min(max_frame, int(current_frame) + 1)
                return update_preview_from_slider(folder, video, new_frame, preview_src, use_gpu,
                                                  threshold, dilate, blur, shadow_shift, shadow_start,
                                                  shadow_decay, shadow_min, shadow_gamma)
            
            def jump_to_frame(folder, video, jump_frame, preview_src, use_gpu, threshold, dilate, blur,
                             shadow_shift, shadow_start, shadow_decay, shadow_min, shadow_gamma):
                """Jump to specific frame"""
                return update_preview_from_slider(folder, video, int(jump_frame)-1, preview_src, use_gpu,
                                                  threshold, dilate, blur, shadow_shift, shadow_start,
                                                  shadow_decay, shadow_min, shadow_gamma)
            
            prev_btn.click(
                fn=prev_frame,
                inputs=[mask_folder_input, preview_video_dropdown, frame_slider, preview_source,
                       use_gpu_input, mask_binarize_threshold_input, mask_dilate_kernel_input,
                       mask_blur_kernel_input, shadow_shift_input, shadow_start_opacity_input,
                       shadow_opacity_decay_input, shadow_min_opacity_input, shadow_gamma_input],
                outputs=[preview_image, status_label, frame_slider]
            )
            
            next_btn.click(
                fn=next_frame,
                inputs=[mask_folder_input, preview_video_dropdown, frame_slider, preview_source,
                       use_gpu_input, mask_binarize_threshold_input, mask_dilate_kernel_input,
                       mask_blur_kernel_input, shadow_shift_input, shadow_start_opacity_input,
                       shadow_opacity_decay_input, shadow_min_opacity_input, shadow_gamma_input],
                outputs=[preview_image, status_label, frame_slider]
            )
            
            jump_to_input.submit(
                fn=jump_to_frame,
                inputs=[mask_folder_input, preview_video_dropdown, jump_to_input, preview_source,
                       use_gpu_input, mask_binarize_threshold_input, mask_dilate_kernel_input,
                       mask_blur_kernel_input, shadow_shift_input, shadow_start_opacity_input,
                       shadow_opacity_decay_input, shadow_min_opacity_input, shadow_gamma_input],
                outputs=[preview_image, status_label, frame_slider]
            )
        
        return interface

    def start_processing(self, *args, progress=gr.Progress()):
        # Extract parameters from args
        (inpainted_folder, original_folder, mask_folder, output_folder,
         use_gpu, pad_to_16_9, output_format, batch_chunk_size,
         enable_color_transfer, mask_binarize_threshold, mask_dilate_kernel_size,
         mask_blur_kernel_size, shadow_shift, shadow_decay_gamma,
         shadow_start_opacity, shadow_opacity_decay, shadow_min_opacity) = args

        self.stop_event.clear()

        # Validate parameters
        try:
            batch_chunk_size = int(batch_chunk_size)
            mask_binarize_threshold = float(mask_binarize_threshold)
            mask_dilate_kernel_size = int(float(mask_dilate_kernel_size))  # Convert to int for kernel size
            mask_blur_kernel_size = int(float(mask_blur_kernel_size))  # Convert to int for kernel size
            shadow_shift = int(float(shadow_shift))  # Convert to int
            shadow_decay_gamma = float(shadow_decay_gamma)
            shadow_start_opacity = float(shadow_start_opacity)
            shadow_opacity_decay = float(shadow_opacity_decay)
            shadow_min_opacity = float(shadow_min_opacity)
        except ValueError:
            yield "Error: Please enter valid values", 0
            return

        if not os.path.exists(inpainted_folder):
             yield f"Error: Inpainted folder not found: {inpainted_folder}", 0
             return
        if not os.path.exists(original_folder):
             # Original folder might be optional for quad input but good to warn
             pass
        os.makedirs(output_folder, exist_ok=True)

        import traceback
        try:
            yield "Starting Batch Process...", 0
            
            inpainted_videos = sorted(glob.glob(os.path.join(inpainted_folder, "*.mp4")))
            print(f"[DEBUG] Found {len(inpainted_videos)} videos in {inpainted_folder}")
            if not inpainted_videos:
                print(f"[DEBUG] No videos found.")
                yield "No .mp4 files found in inpainted video folder", 0
                return

            total_videos = len(inpainted_videos)
            
            for i, inpainted_video_path in enumerate(inpainted_videos):
                if self.stop_event.is_set():
                    print(f"[DEBUG] Stop event set.")
                    yield "Processing stopped by user", (i / total_videos * 100)
                    break
                    
                base_name = os.path.basename(inpainted_video_path)
                print(f"[DEBUG] Processing video: {base_name}")
                current_percent = (i / total_videos * 100)
                yield f"Processing {i+1}/{total_videos}: {base_name}", current_percent
                
                # --- 1. Find corresponding files ---
                inpaint_suffix = "_inpainted_right_eye.mp4"
                sbs_suffix = "_inpainted_sbs.mp4"
                webui_suffix = "_inpainted.mp4"
                is_sbs_input = False
                
                if base_name.endswith(inpaint_suffix):
                    core_name_with_width = base_name[:-len(inpaint_suffix)]
                elif base_name.endswith(sbs_suffix):
                        core_name_with_width = base_name[:-len(sbs_suffix)]
                        is_sbs_input = True
                elif base_name.endswith(webui_suffix):
                        core_name_with_width = base_name[:-len(webui_suffix)]
                        is_sbs_input = True
                else:
                    print(f"[DEBUG] Skipping {base_name} - Unknown suffix")
                    continue # Skip invalid files
                    
                last_underscore_idx = core_name_with_width.rfind('_')
                if last_underscore_idx == -1: 
                    print(f"[DEBUG] Skipping {base_name} - Could not parse width")
                    continue
                
                # Original core extraction
                core_name = core_name_with_width[:last_underscore_idx]
                print(f"[DEBUG] Core name identified: {core_name}")

                # Helper to find matches
                def find_matches(c_name):
                    # Try exact pattern first
                    p4 = glob.glob(os.path.join(mask_folder, f"{c_name}_*_splatted4.mp4"))
                    p2 = glob.glob(os.path.join(mask_folder, f"{c_name}_*_splatted2.mp4"))
                    
                    # Try looser pattern if greedy failed (e.g. no intermediate tag)
                    if not p4 and not p2:
                        p4 = glob.glob(os.path.join(mask_folder, f"{c_name}*splatted4*.mp4"))
                        p2 = glob.glob(os.path.join(mask_folder, f"{c_name}*splatted2*.mp4"))
                    return p4, p2

                splatted4_matches, splatted2_matches = find_matches(core_name)

                # Fallback: exact name failed, try swapping low -> high?
                if not splatted4_matches and not splatted2_matches and "_low" in core_name:
                    core_high = core_name.replace("_low", "_high")
                    print(f"[DEBUG] match failed, trying core_high: {core_high}")
                    splatted4_matches, splatted2_matches = find_matches(core_high)
                    # Identify it as finding original from correct place
                    if splatted4_matches or splatted2_matches:
                        core_name = core_high # Update core name for original reader lookup too

                splatted_file_path = None
                is_dual_input = False
                
                if splatted4_matches:
                    splatted_file_path = splatted4_matches[0]
                elif splatted2_matches:
                    splatted_file_path = splatted2_matches[0]
                    is_dual_input = True
                
                if not splatted_file_path:
                    print(f"[DEBUG] Skipping {base_name} - No splatted file found in {mask_folder}")
                    yield f"Skipping {base_name} - No splatted file found", current_percent
                    continue

                # --- 2. Setup Readers and Encoder ---
                print(f"[DEBUG] Using splatted file: {os.path.basename(splatted_file_path)}")
                inpainted_reader = VideoReader(inpainted_video_path, ctx=cpu(0))
                splatted_reader = VideoReader(splatted_file_path, ctx=cpu(0))
                
                original_reader = None
                if is_dual_input:
                    original_path = _find_video_by_core_name(original_folder, core_name)
                    if original_path:
                        original_reader = VideoReader(original_path, ctx=cpu(0))
                else:
                    original_reader = splatted_reader # Placeholder

                num_frames = len(inpainted_reader)
                fps = inpainted_reader.get_avg_fps()
                video_stream_info = get_video_stream_info(inpainted_video_path)
                
                # Determine Output Dimensions
                sample_splatted = splatted_reader[0].asnumpy()
                H_splat, W_splat, _ = sample_splatted.shape
                if is_dual_input:
                    hires_H, hires_W = H_splat, W_splat // 2
                else:
                    hires_H, hires_W = H_splat // 2, W_splat // 2

                if original_reader is None and output_format != "Right-Eye Only":
                    # Fallback if original is missing
                    output_format_current = "Right-Eye Only"
                else:
                    output_format_current = output_format

                # Determine Output Width/Height for FFmpeg
                perceived_width = hires_W
                output_width = hires_W
                output_height = hires_H
                suffix = "_merged.mp4"

                if output_format_current == "Full SBS (Left-Right)":
                    output_width = hires_W * 2
                    suffix = "_merged_full_sbs.mp4"
                elif output_format_current == "Full SBS Cross-eye (Right-Left)":
                    output_width = hires_W * 2
                    suffix = "_merged_full_sbsx.mp4"
                elif output_format_current == "Double SBS":
                    output_width = hires_W * 2
                    output_height = hires_H * 2
                    suffix = "_merged_half_sbs.mp4"
                    perceived_width = hires_W * 2
                elif output_format_current == "Half SBS (Left-Right)":
                    output_width = hires_W
                    suffix = "_merged_half_sbs.mp4"
                elif output_format_current.startswith("Anaglyph"):
                    output_width = hires_W
                    suffix = "_merged_anaglyph.mp4"
                else:
                    output_width = hires_W
                    suffix = "_merged_right_eye.mp4"

                output_filename = f"{core_name}_{perceived_width}{suffix}"
                output_path = os.path.join(output_folder, output_filename)

                print(f"[DEBUG] Starting FFmpeg process for: {output_path}")
                print(f"[DEBUG] Params: {output_width}x{output_height}, fps={fps}")
                print(f"[DEBUG] CUDA_AVAILABLE in merging_ui BEFORE: {sc_util.CUDA_AVAILABLE}")
                
                # Temporarily disable CUDA to force CPU encoding (libx264 CRF=18) to match GUI
                original_cuda = sc_util.CUDA_AVAILABLE
                sc_util.CUDA_AVAILABLE = False
                
                print(f"[DEBUG] CUDA_AVAILABLE in merging_ui AFTER: {sc_util.CUDA_AVAILABLE}")
                print(f"[DEBUG] video_stream_info: {video_stream_info}")
                
                ffmpeg_process = start_ffmpeg_pipe_process(
                    content_width=output_width,
                    content_height=output_height,
                    final_output_mp4_path=output_path,
                    fps=fps, 
                    video_stream_info=video_stream_info,
                    pad_to_16_9=pad_to_16_9,
                    output_format_str=output_format_current,
                    debug_label="WEBUI_MERGE"
                )
                
                # Restore CUDA flag
                sc_util.CUDA_AVAILABLE = original_cuda
                
                print(f"[DEBUG] FFmpeg started: {ffmpeg_process}")
                
                # Print encoder info if available
                if ffmpeg_process and hasattr(ffmpeg_process, 'sc_encode_flags'):
                    flags = ffmpeg_process.sc_encode_flags
                    print(f"[DEBUG] Encoder flags: codec={flags.get('enc_codec')}, {flags.get('quality_mode')}={flags.get('quality_value')}")
                    print(f"[DEBUG] Full encoder info: {flags}")

                if not ffmpeg_process:
                    yield f"Error starting FFmpeg for {base_name}", current_percent
                    continue

                # --- 3. Process Chunks ---
                print(f"[DEBUG] Starting processing chunks...")
                chunk_size = batch_chunk_size
                for frame_start in range(0, num_frames, chunk_size):
                    if self.stop_event.is_set(): break
                    
                    # Update progress every chunk
                    video_progress = frame_start / num_frames
                    overall_progress = (i + video_progress) / total_videos * 100
                    yield f"Processing {base_name}: {int(video_progress*100)}% ({frame_start}/{num_frames})", overall_progress

                    frame_end = min(frame_start + chunk_size, num_frames)
                    indices = list(range(frame_start, frame_end))
                    
                    inpainted_np = inpainted_reader.get_batch(indices).asnumpy()
                    splatted_np = splatted_reader.get_batch(indices).asnumpy()
                    
                    inpainted_tensor = torch.from_numpy(inpainted_np).permute(0, 3, 1, 2).float() / 255.0
                    splatted_tensor = torch.from_numpy(splatted_np).permute(0, 3, 1, 2).float() / 255.0
                    
                    inpainted_chunk = inpainted_tensor[:, :, :, inpainted_tensor.shape[3]//2:] if is_sbs_input else inpainted_tensor
                    _, _, H_chunk, W_chunk = splatted_tensor.shape

                    if is_dual_input:
                        if original_reader is not None:
                            # For dual input, original_reader is separate file
                            original_np = original_reader.get_batch(indices).asnumpy()
                            original_left = torch.from_numpy(original_np).permute(0, 3, 1, 2).float() / 255.0
                        else:
                            original_left = torch.zeros_like(inpainted_chunk)
                        
                        mask_raw = splatted_tensor[:, :, :, :W_chunk//2]
                        warped_original = splatted_tensor[:, :, :, W_chunk//2:]
                    else:
                        # Quad input
                        half_h, half_w = H_chunk // 2, W_chunk // 2
                        original_left = splatted_tensor[:, :, :half_h, :half_w]
                        mask_raw = splatted_tensor[:, :, half_h:, :half_w]
                        warped_original = splatted_tensor[:, :, half_h:, half_w:]

                    mask_np = mask_raw.permute(0, 2, 3, 1).numpy() # Move to CPU/numpy for mean
                    mask_gray_np = np.mean(mask_np, axis=3)
                    mask = torch.from_numpy(mask_gray_np).float().unsqueeze(1) # [B, 1, H, W]

                    # Move to GPU if requested
                    dev = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
                    mask = mask.to(dev)
                    inpainted_chunk = inpainted_chunk.to(dev)
                    original_left = original_left.to(dev)
                    warped_original = warped_original.to(dev)

                    # Resize
                    if inpainted_chunk.shape[2] != hires_H or inpainted_chunk.shape[3] != hires_W:
                        inpainted_chunk = F.interpolate(inpainted_chunk, size=(hires_H, hires_W), mode='bicubic', align_corners=False)
                        mask = F.interpolate(mask, size=(hires_H, hires_W), mode='bilinear', align_corners=False)

                    # Color Transfer
                    if enable_color_transfer:
                        adjusted = []
                        for idx in range(len(inpainted_chunk)):
                            adj = apply_color_transfer(original_left[idx].cpu(), inpainted_chunk[idx].cpu())
                            adjusted.append(adj.to(dev))
                        inpainted_chunk = torch.stack(adjusted)

                    # Mask Processing
                    processed_mask = mask.clone()
                    print(f"[DEBUG] Mask processing params: threshold={mask_binarize_threshold}, dilate={mask_dilate_kernel_size}, blur={mask_blur_kernel_size}, shadow_shift={shadow_shift}")
                    print(f"[DEBUG] Mask shape before processing: {processed_mask.shape}, min={processed_mask.min():.4f}, max={processed_mask.max():.4f}, mean={processed_mask.mean():.4f}")
                    
                    if mask_binarize_threshold >= 0:
                        processed_mask = (processed_mask > mask_binarize_threshold).float()
                        print(f"[DEBUG] After binarization: min={processed_mask.min():.4f}, max={processed_mask.max():.4f}, mean={processed_mask.mean():.4f}")
                    
                    if mask_dilate_kernel_size > 0:
                        processed_mask = apply_mask_dilation(processed_mask, int(mask_dilate_kernel_size), use_gpu=(dev=="cuda"))
                        print(f"[DEBUG] After dilation: min={processed_mask.min():.4f}, max={processed_mask.max():.4f}, mean={processed_mask.mean():.4f}")
                    
                    if mask_blur_kernel_size > 0:
                        processed_mask = apply_gaussian_blur(processed_mask, int(mask_blur_kernel_size), use_gpu=(dev=="cuda"))
                        print(f"[DEBUG] After blur: min={processed_mask.min():.4f}, max={processed_mask.max():.4f}, mean={processed_mask.mean():.4f}")
                    
                    if shadow_shift > 0:
                        processed_mask = apply_shadow_blur(
                            processed_mask, int(shadow_shift), shadow_start_opacity, 
                            shadow_opacity_decay, shadow_min_opacity, shadow_decay_gamma, 
                            use_gpu=(dev=="cuda")
                        )
                        print(f"[DEBUG] After shadow: min={processed_mask.min():.4f}, max={processed_mask.max():.4f}, mean={processed_mask.mean():.4f}")

                    # Blending
                    blended_right = warped_original * (1 - processed_mask) + inpainted_chunk * processed_mask

                    # Assemble Final Output
                    final_chunk = None
                    if output_format_current == "Full SBS (Left-Right)":
                        final_chunk = torch.cat([original_left, blended_right], dim=3)
                    elif output_format_current == "Full SBS Cross-eye (Right-Left)":
                        final_chunk = torch.cat([blended_right, original_left], dim=3)
                    elif output_format_current == "Half SBS (Left-Right)":
                        res_l = F.interpolate(original_left, size=(hires_H, hires_W // 2), mode='bilinear')
                        res_r = F.interpolate(blended_right, size=(hires_H, hires_W // 2), mode='bilinear')
                        final_chunk = torch.cat([res_l, res_r], dim=3)
                    elif output_format_current == "Double SBS":
                        sbs = torch.cat([original_left, blended_right], dim=3)
                        final_chunk = F.interpolate(sbs, size=(hires_H*2, hires_W*2), mode='bilinear')
                    elif output_format_current == "Anaglyph (Red/Cyan)":
                        final_chunk = torch.cat([original_left[:, 0:1], blended_right[:, 1:3]], dim=1)
                    elif output_format_current == "Anaglyph Half-Color":
                        left_gray = original_left[:, 0] * 0.299 + original_left[:, 1] * 0.587 + original_left[:, 2] * 0.114
                        left_gray = left_gray.unsqueeze(1)
                        final_chunk = torch.cat([left_gray, blended_right[:, 1:3]], dim=1)
                    else:
                        final_chunk = blended_right

                    # Write to pipe
                    cpu_chunk = final_chunk.cpu()
                    for chunk_idx, frame_tensor in enumerate(cpu_chunk):
                        if ffmpeg_process.poll() is not None:
                            print(f"[DEBUG] FFmpeg process finish/died unexpectedly with code {ffmpeg_process.returncode}")
                            raise RuntimeError("FFmpeg process finished early")

                        frame_np = frame_tensor.permute(1, 2, 0).numpy()
                        frame_uint16 = (np.clip(frame_np, 0.0, 1.0) * 65535.0).astype(np.uint16)
                        frame_bgr = cv2.cvtColor(frame_uint16, cv2.COLOR_RGB2BGR)
                        try:
                            ffmpeg_process.stdin.write(frame_bgr.tobytes())
                        except BrokenPipeError:
                            print("[DEBUG] Broken Pipe Error writing to FFmpeg")
                            raise
                        except Exception as e:
                            print(f"[DEBUG] Error writing frame {chunk_idx} of chunk {frame_start}: {e}")
                            raise

                # Close FFmpeg
                if ffmpeg_process.stdin:
                    ffmpeg_process.stdin.close()
                ffmpeg_process.wait()

                # Move files to finished (simplified for WebUI)
                yield f"Completed: {base_name}", ((i+1)/total_videos*100)

            yield "Processing completed", 100
            
        except Exception as e:
            traceback.print_exc()
            yield f"Error: {str(e)}", 0

    def stop_processing(self):
        self.stop_event.set()
        return "Stopping processing...", 0
