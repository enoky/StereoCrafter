import os
import glob
import json
import logging
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Toplevel, Label
from typing import Optional, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
import imageio_ffmpeg
import torch.nn.functional as F
import time

from pipelines.stereo_video_inpainting import (
    StableVideoDiffusionInpaintingPipeline,
    tensor2vid
)

# torch.backends.cudnn.benchmark = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Tooltip Class ---
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

        self.tooltip_window = Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True) # Remove window decorations
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = Label(self.tooltip_window, text=self.text, background="#ffffe0", relief="solid", borderwidth=1,
                      font=("tahoma", "8", "normal"), justify="left", wraplength=250)
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None
# --- END Tooltip Class ---


def load_inpainting_pipeline(
    pre_trained_path: str,
    unet_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    offload_type: str = "model"
) -> StableVideoDiffusionInpaintingPipeline:
    """
    Loads the stable video diffusion inpainting pipeline components and returns the pipeline object.
    """
    logger.info("Loading pipeline components...")

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pre_trained_path,
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=dtype,
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pre_trained_path,
        subfolder="vae",
        variant="fp16",
        torch_dtype=dtype,
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        subfolder="unet_diffusers",
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )

    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(
        pre_trained_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=dtype,
    ).to(device)

    if offload_type == "model":
        pipeline.enable_model_cpu_offload()
    elif offload_type == "sequential":
        pipeline.enable_sequential_cpu_offload()
    elif offload_type == "none":
        pass  # No offloading
    else:
        raise ValueError("Invalid offload_type")

    return pipeline

def read_video_frames(video_path: str, decord_ctx=cpu(0)) -> Tuple[torch.Tensor, float]:
    """
    Reads a video using decord and returns frames as a 4D float tensor [T, C, H, W] and the FPS.
    """
    video_reader = VideoReader(video_path, ctx=decord_ctx)
    num_frames = len(video_reader)

    if num_frames == 0:
        return torch.empty(0), 0.0

    fps = video_reader.get_avg_fps()
    frames = video_reader.get_batch(range(num_frames))
    frames = torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2).float()

    return frames, fps

def write_video_ffmpeg(
    frames: np.ndarray,
    fps: float,
    output_path: str,
    vf: Optional[str] = None,
    codec: str = "libx264",
    crf: int = 16,
    preset: str = "ultrafast"
) -> None:
    """
    Writes a sequence of frames (uint8) to a video file using imageio-ffmpeg.
    """
    if frames.dtype != np.uint8:
        frames = (frames * 255).astype(np.uint8)

    frames = np.ascontiguousarray(frames)
    height, width = frames.shape[1:3]

    output_params = ["-crf", str(crf), "-preset", preset]
    if vf:
        output_params += ["-vf", vf]

    writer = imageio_ffmpeg.write_frames(
        output_path,
        (width, height),
        fps=fps,
        codec=codec,
        quality=None,
        bitrate=None,
        macro_block_size=1,  # Set to 1 to avoid resizing
        output_params=output_params
    )
    writer.send(None)
    for frame in frames:
        writer.send(frame)
    writer.close()

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

    size_y = (H + overlap_y * (tile_num - 1)) // tile_num
    size_x = (W + overlap_x * (tile_num - 1)) // tile_num

    stride_y = size_y - overlap_y
    stride_x = size_x - overlap_x

    ideal_H = stride_y * tile_num + overlap_y * (tile_num - 1)
    ideal_W = stride_x * tile_num + overlap_x * (tile_num - 1)

    pad_bottom = max(0, ideal_H - H)
    pad_right = max(0, ideal_W - W)

    if pad_bottom > 0 or pad_right > 0:
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

    size_y = (height + overlap_y * (tile_num - 1)) // tile_num
    size_x = (width + overlap_x * (tile_num - 1)) // tile_num
    tile_size = (size_y, size_x)

    tile_stride = (size_y - overlap_y, size_x - overlap_x)

    cols = []
    for i in range(tile_num):
        row_tiles = []
        for j in range(tile_num):
            y_start = i * tile_stride[0]
            x_start = j * tile_stride[1]
            y_end = y_start + tile_size[0]
            x_end = x_start + tile_size[1]

            y_end = min(y_end, height)
            x_end = min(x_end, width)

            cond_tile = cond_frames[:, :, y_start:y_end, x_start:x_end]
            mask_tile = mask_frames[:, :, y_start:y_end, x_start:x_end]

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
                tile = blend_v(cols[i - 1][j], tile, latent_overlap[0])
            if j > 0:
                tile = blend_h(row_result[j - 1], tile, latent_overlap[1])
            row_result.append(tile)
        blended_rows.append(row_result)

    final_rows = []
    for i, row_tiles in enumerate(blended_rows):
        for j, tile in enumerate(row_tiles):
            if i < len(blended_rows) - 1:
                tile = tile[:, :, :latent_stride[0], :]
            if j < len(row_tiles) - 1:
                tile = tile[:, :, :, :latent_stride[1]]
            row_tiles[j] = tile
        final_rows.append(torch.cat(row_tiles, dim=3))
    x = torch.cat(final_rows, dim=2)

    return x

def process_single_video(
    pipeline: StableVideoDiffusionInpaintingPipeline,
    input_video_path: str,
    save_dir: str,
    frames_chunk: int = 23,
    overlap: int = 3,
    tile_num: int = 1,
    vf: Optional[str] = None,
    num_inference_steps: int = 5,
    stop_event: threading.Event = None,
    update_info_callback=None, # Callback to update GUI info
    original_input_blend_strength: float = 0.8, # NEW PARAMETER WITH DEFAULT
) -> bool:
    """
    Processes a single input video.
    Determines input format (quad or dual) based on filename suffix.
    Outputs SBS (left view | inpainted output) for quad input,
    or only inpainted output for dual input.
    Returns True if processing completed successfully, False if stopped.
    """
    os.makedirs(save_dir, exist_ok=True)

    base_video_name = os.path.basename(input_video_path)
    video_name_without_ext = os.path.splitext(base_video_name)[0]

    # Determine if input is dual splatted based on suffix
    is_dual_input = video_name_without_ext.endswith("_splatted2")
    if is_dual_input:
        logger.info(f"Dual Splat detected for '{base_video_name}'. Processing and outputting Inpainted Right Eye only.")
    else:
        logger.info(f"Quad Splat (or default) detected for '{base_video_name}'. Processing and outputting Inpainted SBS video.")

    output_suffix = "_inpainted_right_eye" if is_dual_input else "_inpainted_sbs"

    video_name_for_output = video_name_without_ext.replace("_splatted4", "").replace("_splatted2", "")
    output_video_filename = f"{video_name_for_output}{output_suffix}.mp4"
    output_video_path = os.path.join(save_dir, output_video_filename)


    frames, fps = read_video_frames(input_video_path)
    num_frames = frames.shape[0]

    if num_frames == 0:
        logger.warning(f"No frames found in {input_video_path}, skipping.")
        if update_info_callback:
            update_info_callback(base_video_name, "N/A", "0 (skipped)")
        return False

    _, _, num_h, num_w = frames.shape
    if num_h < 2 or num_w < 2:
        logger.warning(f"Video {input_video_path} is too small ({num_h}x{num_w}), skipping.")
        if update_info_callback:
            update_info_callback(base_video_name, f"{num_w}x{num_h}", num_frames)
        return False

    # Update GUI with video info before processing
    if update_info_callback:
        update_info_callback(base_video_name, f"{num_w}x{num_h}", num_frames)

    # Frame chunking and mask extraction based on input type
    if is_dual_input:
        # Input is (Mask | Warped) from _splatted2
        half_w_input = num_w // 2 # Half width of the _splatted2 video
        frames_mask_raw = frames[:, :, :, :half_w_input]  # Left half is mask
        frames_warpped_raw = frames[:, :, :, half_w_input:] # Right half is warped
        frames_left_original = None # No original left eye in dual input
        
        # Ensure the warped frame for inpainting has correct dimensions (height, half_w_input)
        frames_warpped = frames_warpped_raw
        frames_mask = frames_mask_raw.mean(dim=1, keepdim=True) # Convert mask to grayscale
        
        # The target dimensions for the *single* inpainted output frame
        target_output_h = frames_warpped.shape[2]
        target_output_w = frames_warpped.shape[3]

    else:
        # Input is (Original | DepthVis (Top), Mask | Warped (Bottom)) from _splatted4 or other
        half_h_input = num_h // 2 # Half height of the quad video
        half_w_input = num_w // 2 # Half width of the quad video

        frames_left_original = frames[:, :, :half_h_input, :half_w_input] # Top-Left is original view
        frames_mask_raw = frames[:, :, half_h_input:, :half_w_input]  # Bottom-Left is mask
        frames_warpped_raw = frames[:, :, half_h_input:, half_w_input:] # Bottom-Right is warped

        # The warped frames are the input for inpainting
        frames_warpped = frames_warpped_raw
        frames_mask = frames_mask_raw.mean(dim=1, keepdim=True) # Convert mask to grayscale

        # The target dimensions for the *single* inpainted output frame (which is half of the full output)
        target_output_h = frames_warpped.shape[2]
        target_output_w = frames_warpped.shape[3]

    # Pre-process frames for inpainting pipeline
    frames_warpped = frames_warpped / 255.0 # Normalize to 0-1
    # frames_mask is already normalized and grayscale after mean(dim=1) and is 0-1 from read_video_frames

    frames_warpped = pad_for_tiling(frames_warpped, tile_num, tile_overlap=(128, 128))
    frames_mask = pad_for_tiling(frames_mask, tile_num, tile_overlap=(128, 128))

    stride = max(1, frames_chunk - overlap)
    results = [] # Stores chunks to be concatenated in final video
    previous_chunk_output_frames = None # Will hold the *full* generated output of the previous chunk

    for i in range(0, num_frames, stride):
        if stop_event and stop_event.is_set():
            logger.info(f"Stopping processing of {input_video_path}")
            return False
        end_idx = min(i + frames_chunk, num_frames)
        chunk_size = end_idx - i
        if chunk_size <= 0:
            break

        # Get the original input for the current chunk
        original_input_frames_for_chunk = frames_warpped[i:end_idx].clone()
        mask_frames_i = frames_mask[i:end_idx].clone()

        input_frames_to_pipeline = original_input_frames_for_chunk.clone() # Start with original, modify if blending

        # --- NEW: Input-level blending for overlapping frames ---
        if previous_chunk_output_frames is not None and overlap > 0:
            # Determine actual number of frames to overlap, limited by available frames
            # Ensure we don't try to blend more frames than are available in either the previous output or current input
            overlap_actual = min(overlap, len(previous_chunk_output_frames), len(original_input_frames_for_chunk))

            if overlap_actual > 0:
                # 1. Get the generated overlap frames from the end of the previous chunk's output
                prev_gen_overlap_frames = previous_chunk_output_frames[-overlap_actual:]
                
                # 2. Get the corresponding original input frames for the start of the current chunk
                orig_input_overlap_frames = original_input_frames_for_chunk[:overlap_actual]

                # Calculate weights for blending based on original_input_blend_strength
                # This scales the influence of the original input across the overlap period.
                # If original_input_blend_strength is 0, original_weights_scaled will be 0, meaning no original influence.
                # If original_input_blend_strength is 1, original_weights_scaled will be linspace(0,1), meaning full influence.
                original_weights_scaled = torch.linspace(0.0, 1.0, overlap_actual, device=prev_gen_overlap_frames.device).view(-1, 1, 1, 1) * original_input_blend_strength

                # The weight for the previous generated output will be (1 - original_weights_scaled)
                # The weight for the original input will be original_weights_scaled
                blended_input_overlap_frames = (1 - original_weights_scaled) * prev_gen_overlap_frames + \
                                                original_weights_scaled * orig_input_overlap_frames
                
                # Replace the overlapping part of the current chunk's input with the blended frames
                input_frames_to_pipeline[:overlap_actual] = blended_input_overlap_frames
            # else: overlap_actual is 0, no blending, input_frames_to_pipeline remains original_input_frames_for_chunk
        # --- END NEW Input-level blending ---

        logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting inference for chunk {i}-{end_idx}...")
        start_time = time.time()

        with torch.no_grad():
            video_latents = spatial_tiled_process(
                input_frames_to_pipeline, # Use the potentially blended input frames
                mask_frames_i,
                pipeline,
                tile_num,
                spatial_n_compress=8,
                min_guidance_scale=1.01,
                max_guidance_scale=1.01,
                decode_chunk_size=8,
                fps=7, # Fixed FPS for pipeline, might not match video's original FPS
                motion_bucket_id=127,
                noise_aug_strength=0.0,
                num_inference_steps=num_inference_steps,
            )
            video_latents = video_latents.unsqueeze(0)

            pipeline.vae.to(dtype=torch.float16)
            decoded_frames = pipeline.decode_latents(
                video_latents,
                num_frames=video_latents.shape[1],
                decode_chunk_size=2,
            )

        end_time = time.time()
        inference_duration = end_time - start_time
        logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Inference for chunk {i}-{end_idx} completed in {inference_duration:.2f} seconds.")
        
        video_frames = tensor2vid(decoded_frames, pipeline.image_processor, output_type="pil")[0]
        current_chunk_generated_frames = []
        for j in range(len(video_frames)):
            img = video_frames[j]
            current_chunk_generated_frames.append(torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0)

        current_chunk_generated = torch.stack(current_chunk_generated_frames)

        # --- CRITICAL FIX: Append only the non-overlapping part for subsequent chunks ---
        if i == 0:
            # For the first chunk, append the full generated output.
            results.append(current_chunk_generated)
        else:
            # For subsequent chunks, append only the portion beyond the overlap.
            # The overlap frames from this chunk's output are already accounted for (or blended into)
            # by the previous chunk's output.
            results.append(current_chunk_generated[overlap:])
        
        # Always store the full generated chunk for the *next* iteration's input blend
        previous_chunk_output_frames = current_chunk_generated 

        if end_idx == num_frames:
            break

    frames_output = torch.cat(results, dim=0).cpu()

    # Crop the output to the original dimensions before tiling
    # This assumes pad_for_tiling added padding at the bottom/right.
    # The dimensions here should match `frames_warpped_raw` or `frames_left_original` for the height/width.
    frames_output_final = frames_output[:, :, :target_output_h, :target_output_w]

    if is_dual_input:
        # Output only the inpainted right eye
        final_video_frames_np = (frames_output_final * 255).permute(0, 2, 3, 1).byte().numpy()
    else:
        # Output SBS: Original Left | Inpainted Right
        # Ensure frames_left_original is in the correct format (0-1 float)
        frames_left_original_normalized = frames_left_original.float() / 255.0

        sbs_frames = torch.cat([frames_left_original_normalized, frames_output_final], dim=3)
        final_video_frames_np = (sbs_frames * 255).permute(0, 2, 3, 1).byte().numpy()

    write_video_ffmpeg(
        frames=final_video_frames_np,
        fps=fps,
        output_path=output_video_path,
        vf=vf if vf else None,
        codec="libx264",
        crf=16,
        preset="ultrafast",
    )

    logger.info(f"Done processing {input_video_path} -> {output_video_path}")
    return True

class InpaintingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Batch Video Inpainting")
        # Adjusted geometry to accommodate new widgets
        self.geometry("600x550") # Increased height from 400 to 550
        self.config = self.load_config()
        # NEW: Load help data from renamed file
        self.help_data = self.load_help_data()

        self.input_folder_var = tk.StringVar(value=self.config.get("input_folder", "./output_splatted"))
        self.output_folder_var = tk.StringVar(value=self.config.get("output_folder", "./completed_output"))
        self.num_inference_steps_var = tk.StringVar(value=str(self.config.get("num_inference_steps", 5)))
        self.tile_num_var = tk.StringVar(value=str(self.config.get("tile_num", 2)))
        self.frames_chunk_var = tk.StringVar(value=str(self.config.get("frames_chunk", 23)))
        # Renamed variable key for consistency
        self.overlap_var = tk.StringVar(value=str(self.config.get("frame_overlap", 3)))
        self.original_input_blend_strength_var = tk.StringVar(value=str(self.config.get("original_input_blend_strength", 0.5))) # Default to 0.5
        self.offload_type_var = tk.StringVar(value=self.config.get("offload_type", "model"))
        self.processed_count = tk.IntVar(value=0)
        self.total_videos = tk.IntVar(value=0)
        self.stop_event = threading.Event()
        self.pipeline = None

        self.create_widgets()
        self.update_progress() # Start the progress bar and status updates
        self.update_status_label("Ready")

    def create_widgets(self):
        folder_frame = ttk.LabelFrame(self, text="Folders", padding=10)
        folder_frame.pack(fill="x", padx=10, pady=5)
        
        # Input Folder
        input_label = ttk.Label(folder_frame, text="Input Folder:")
        input_label.grid(row=0, column=0, sticky="e", padx=5, pady=2)
        Tooltip(input_label, self.help_data.get("input_folder", ""))
        ttk.Entry(folder_frame, textvariable=self.input_folder_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(folder_frame, text="Browse", command=self.browse_input).grid(row=0, column=2, padx=5)
        
        # Output Folder
        output_label = ttk.Label(folder_frame, text="Output Folder:")
        output_label.grid(row=1, column=0, sticky="e", padx=5, pady=2)
        Tooltip(output_label, self.help_data.get("output_folder", ""))
        ttk.Entry(folder_frame, textvariable=self.output_folder_var, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(folder_frame, text="Browse", command=self.browse_output).grid(row=1, column=2, padx=5)

        param_frame = ttk.LabelFrame(self, text="Parameters", padding=10)
        param_frame.pack(fill="x", padx=10, pady=5)
        
        # Inference Steps
        inference_steps_label = ttk.Label(param_frame, text="Inference Steps:")
        inference_steps_label.grid(row=0, column=0, sticky="e", padx=5, pady=2)
        Tooltip(inference_steps_label, self.help_data.get("num_inference_steps", ""))
        ttk.Entry(param_frame, textvariable=self.num_inference_steps_var, width=10).grid(row=0, column=1, sticky="w")
        
        # Tile Number
        tile_num_label = ttk.Label(param_frame, text="Tile Number:")
        tile_num_label.grid(row=1, column=0, sticky="e", padx=5, pady=2)
        Tooltip(tile_num_label, self.help_data.get("tile_num", ""))
        ttk.Entry(param_frame, textvariable=self.tile_num_var, width=10).grid(row=1, column=1, sticky="w")
        
        # Frames Chunk
        frames_chunk_label = ttk.Label(param_frame, text="Frames Chunk:")
        frames_chunk_label.grid(row=2, column=0, sticky="e", padx=5, pady=2)
        Tooltip(frames_chunk_label, self.help_data.get("frames_chunk", ""))
        ttk.Entry(param_frame, textvariable=self.frames_chunk_var, width=10).grid(row=2, column=1, sticky="w")
        
        # Frame Overlap (Renamed from Overlap)
        # Updated label text and tooltip key
        frame_overlap_label = ttk.Label(param_frame, text="Frame Overlap:")
        frame_overlap_label.grid(row=3, column=0, sticky="e", padx=5, pady=2)
        Tooltip(frame_overlap_label, self.help_data.get("frame_overlap", "")) 
        ttk.Entry(param_frame, textvariable=self.overlap_var, width=10).grid(row=3, column=1, sticky="w")
        
        # Original Input Bias (NEW PARAMETER)
        original_blend_label = ttk.Label(param_frame, text="Original Input Bias:") # Concise name for GUI
        original_blend_label.grid(row=4, column=0, sticky="e", padx=5, pady=2)
        Tooltip(original_blend_label, self.help_data.get("original_input_blend_strength", ""))
        ttk.Entry(param_frame, textvariable=self.original_input_blend_strength_var, width=10).grid(row=4, column=1, sticky="w")

        # CPU Offload
        offload_label = ttk.Label(param_frame, text="CPU Offload:")
        offload_label.grid(row=5, column=0, sticky="e", padx=5, pady=2)
        Tooltip(offload_label, self.help_data.get("offload_type", ""))
        offload_options = ["model", "sequential", "none"]
        ttk.OptionMenu(param_frame, self.offload_type_var, self.offload_type_var.get(), *offload_options).grid(row=5, column=1, sticky="w")

        progress_frame = ttk.LabelFrame(self, text="Progress", padding=10)
        progress_frame.pack(fill="x", padx=10, pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress_bar.pack(fill="x")
        # New: Progress count and status label
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack(pady=5)

        buttons_frame = ttk.Frame(self, padding=10)
        buttons_frame.pack(fill="x")
        self.start_button = ttk.Button(buttons_frame, text="Start", command=self.start_processing)
        self.start_button.pack(side="left", padx=5)
        self.stop_button = ttk.Button(buttons_frame, text="Stop", command=self.stop_processing, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        # NEW: Help button
        ttk.Button(buttons_frame, text="Help", command=self.show_general_help).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Exit", command=self.exit_application).pack(side="left", padx=5)

        # New: Information window for current video
        info_frame = ttk.LabelFrame(self, text="Current Video Information", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        self.video_name_label = ttk.Label(info_frame, text="Name: N/A")
        self.video_name_label.pack(anchor="w", padx=5, pady=1)
        
        self.video_res_label = ttk.Label(info_frame, text="Resolution: N/A")
        self.video_res_label.pack(anchor="w", padx=5, pady=1)
        
        self.video_frames_label = ttk.Label(info_frame, text="Frames: N/A")
        self.video_frames_label.pack(anchor="w", padx=5, pady=1)

    def browse_input(self):
        folder = filedialog.askdirectory(initialdir=self.input_folder_var.get())
        if folder:
            self.input_folder_var.set(folder)

    def browse_output(self):
        folder = filedialog.askdirectory(initialdir=self.output_folder_var.get())
        if folder:
            self.output_folder_var.set(folder)

    def update_status_label(self, message):
        self.status_label.config(text=message)

    def update_video_info_display(self, name, resolution, frames):
        self.video_name_label.config(text=f"Name: {name}")
        self.video_res_label.config(text=f"Resolution: {resolution}")
        self.video_frames_label.config(text=f"Frames: {frames}")

    def start_processing(self):
        input_folder = self.input_folder_var.get()
        output_folder = self.output_folder_var.get()
        try:
            num_inference_steps = int(self.num_inference_steps_var.get())
            tile_num = int(self.tile_num_var.get())
            frames_chunk = int(self.frames_chunk_var.get())
            overlap = int(self.overlap_var.get()) # Use self.overlap_var
            original_input_blend_strength = float(self.original_input_blend_strength_var.get()) # NEW LINE
            if num_inference_steps < 1 or tile_num < 1 or frames_chunk < 1 or overlap < 0 or \
               not (0.0 <= original_input_blend_strength <= 1.0): # NEW VALIDATION
                raise ValueError("Invalid parameter values")
        except ValueError:
            # UPDATED ERROR MESSAGE
            messagebox.showerror("Error", "Please enter valid values: Inference Steps >=1, Tile Number >=1, Frames Chunk >=1, Frame Overlap >=0, Original Input Bias between 0.0 and 1.0")
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
        self.update_video_info_display("N/A", "N/A", "N/A") # Clear info on start

        threading.Thread(target=self.run_batch_process,
                         args=(input_folder, output_folder, num_inference_steps, tile_num, offload_type, frames_chunk, overlap, original_input_blend_strength),
                         daemon=True).start()

    def run_batch_process(self, input_folder, output_folder, num_inference_steps, tile_num, offload_type, frames_chunk, overlap, original_input_blend_strength):
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
            finished_folder = os.path.join(input_folder, "finished")
            os.makedirs(finished_folder, exist_ok=True)
            os.makedirs(output_folder, exist_ok=True)

            for idx, video_path in enumerate(input_videos):
                if self.stop_event.is_set():
                    break
                
                # Get video info for display before processing
                temp_frames, _ = read_video_frames(video_path)
                current_video_name = os.path.basename(video_path)
                current_num_frames = temp_frames.shape[0] if temp_frames.numel() > 0 else 0
                current_display_res = "N/A"
                if current_num_frames > 0:
                    _, _, total_h, total_w = temp_frames.shape # Get total dimensions of the raw input video
                    video_name_without_ext = os.path.splitext(current_video_name)[0]
                    is_dual_input = video_name_without_ext.endswith("_splatted2")

                    if is_dual_input:
                        # For dual input (Mask | Warped), the output is the size of the warped half
                        display_h = total_h
                        display_w = total_w // 2
                    else:
                        # For quad input (Original, Depth, Mask, Warped), the output is the size of one of the quarters (e.g., warped)
                        display_h = total_h // 2
                        display_w = total_w // 2
                    current_display_res = f"{display_w}x{display_h}"

                self.after(0, self.update_video_info_display, current_video_name, current_display_res, current_num_frames)
                self.after(0, self.update_status_label, f"Processing video {idx + 1} of {self.total_videos.get()}")

                logger.info(f"Processing {video_path}")
                completed = process_single_video(
                    pipeline=self.pipeline,
                    input_video_path=video_path,
                    save_dir=output_folder,
                    frames_chunk=frames_chunk,
                    overlap=overlap,
                    tile_num=tile_num,
                    vf=None,
                    num_inference_steps=num_inference_steps,
                    stop_event=self.stop_event,
                    update_info_callback=None, # Info updated by run_batch_process directly
                    original_input_blend_strength=original_input_blend_strength # NEW LINE: Pass the 
                )
                if completed:
                    try:
                        shutil.move(video_path, finished_folder)
                        logger.info(f"Moved {video_path} to {finished_folder}")
                    except Exception as e:
                        logger.error(f"Failed to move {video_path} to {finished_folder}: {e}")
                else:
                    logger.info(f"Processing of {video_path} was stopped")
                self.processed_count.set(idx + 1)
                
            stopped = self.stop_event.is_set()
            self.after(0, lambda: self.processing_done(stopped))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {str(e)}"))
            self.after(0, self.processing_done)

    def stop_processing(self):
        self.stop_event.set()
        if self.pipeline:
            # Attempt to clear CUDA cache if pipeline exists
            try:
                torch.cuda.empty_cache()
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

    def processing_done(self, stopped=False):
        if self.pipeline:
            # Ensure pipeline is properly released and cache cleared
            try:
                del self.pipeline
                torch.cuda.empty_cache()
            except RuntimeError as e:
                logger.warning(f"Failed to clear CUDA cache during cleanup: {e}")
            self.pipeline = None

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        if stopped:
            self.update_status_label("Processing stopped.")
        else:
            self.update_status_label("Processing completed.")
            
        self.update_video_info_display("N/A", "N/A", "N/A") # Clear info after completion

    def exit_application(self):
        config = {
            "input_folder": self.input_folder_var.get(),
            "output_folder": self.output_folder_var.get(),
            "num_inference_steps": self.num_inference_steps_var.get(),
            "tile_num": self.tile_num_var.get(),
            "offload_type": self.offload_type_var.get(),
            "frames_chunk": self.frames_chunk_var.get(),
            # Renamed config key to match GUI and help file
            "frame_overlap": self.overlap_var.get(),
            "original_input_blend_strength": self.original_input_blend_strength_var.get()             
        }
        try:
            with open("config_inpaint.json", "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            messagebox.showwarning("Warning", f"Failed to save config: {str(e)}")
        self.destroy()

    def load_config(self):
        try:
            with open("config_inpaint.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    # NEW: Method to load help data from renamed file
    def load_help_data(self):
        try:
            with open("inpaint_help.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("inpaint_help.json not found. No help tips will be available.")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding inpaint_help.json: {e}")
            return {}

    # NEW: Method to show general help
    def show_general_help(self):
        help_text = self.help_data.get("general_help", "No general help information available.")
        messagebox.showinfo("Help", help_text)

if __name__ == "__main__":
    app = InpaintingGUI()
    app.mainloop()