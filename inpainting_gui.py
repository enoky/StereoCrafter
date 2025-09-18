import os
import glob
import json
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Toplevel, Label
from typing import Optional, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
# FlashAttention requires optional dependency; attempt safe imports
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
try:
    from diffusers.models.attention_processor import (
        AttnProcessor2_0,
        XFormersAttnProcessor,
    )
except Exception:  # diffusers may not provide these processors
    AttnProcessor2_0 = None  # type: ignore
    XFormersAttnProcessor = None  # type: ignore
import torch.nn.functional as F
import time
import subprocess # NEW: For running ffprobe and ffmpeg
import cv2 # NEW: For saving 16-bit PNGs

from dependency.stereocrafter_util import Tooltip, logger, get_video_stream_info, draw_progress_bar, release_cuda_memory
from pipelines.stereo_video_inpainting import (
    StableVideoDiffusionInpaintingPipeline,
    tensor2vid
)

# torch.backends.cudnn.benchmark = True

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

    # Try enabling efficient attention (AttnProcessor2_0) for faster and more memory-efficient inference.
    # This leverages PyTorch's scaled_dot_product_attention, which uses Flash Attention if available.
    # Fallback to xFormers if AttnProcessor2_0 isn't available.
    attention_set = False
    if AttnProcessor2_0 is not None:
        try:
            pipeline.unet.set_attn_processor(AttnProcessor2_0())
            logger.info("Efficient attention (AttnProcessor2_0) enabled for UNet")
            attention_set = True
        except Exception as e:
            logger.warning(f"Failed to enable AttnProcessor2_0: {e}")
    if not attention_set and XFormersAttnProcessor is not None:
        try:
            pipeline.unet.set_attn_processor(XFormersAttnProcessor())
            logger.info("xFormers attention enabled for UNet")
            attention_set = True
        except Exception as e:
            logger.warning(f"Failed to enable xFormers attention: {e}")
    if not attention_set:
        logger.info("Using default attention processor")
    if offload_type == "model":
        pipeline.enable_model_cpu_offload()
    elif offload_type == "sequential":
        pipeline.enable_sequential_cpu_offload()
    elif offload_type == "none":
        pass  # No offloading
    else:
        raise ValueError("Invalid offload_type")

    return pipeline

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

def process_single_video(
    pipeline: StableVideoDiffusionInpaintingPipeline,
    input_video_path: str,
    save_dir: str,
    frames_chunk: int = 23,
    overlap: int = 3,
    tile_num: int = 1,
    vf: Optional[str] = None, # This vf parameter is no longer directly used by new FFmpeg logic
    num_inference_steps: int = 5,
    stop_event: threading.Event = None,
    update_info_callback=None, # Callback to update GUI info (now wrapped for threading)
    original_input_blend_strength: float = 0.8,
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

    is_dual_input = video_name_without_ext.endswith("_splatted2")
    if is_dual_input:
        logger.info(f"Dual Splat detected for '{base_video_name}'. Processing and outputting Inpainted Right Eye only.")
    else:
        logger.info(f"Quad Splat (or default) detected for '{base_video_name}'. Processing and outputting Inpainted SBS video.")

    output_suffix = "_inpainted_right_eye" if is_dual_input else "_inpainted_sbs"
    video_name_for_output = video_name_without_ext.replace("_splatted4", "").replace("_splatted2", "")
    output_video_filename = f"{video_name_for_output}{output_suffix}.mp4"
    output_video_path = os.path.join(save_dir, output_video_filename)

    # Read video frames and stream info
    frames, fps, video_stream_info = read_video_frames(input_video_path)
    num_frames_original  = frames.shape[0]

    if num_frames_original  == 0:
        logger.warning(f"No frames found in {input_video_path}, skipping.")
        if update_info_callback:
            update_info_callback(base_video_name, "N/A", "0 (skipped)", overlap, original_input_blend_strength)
        return False
    
    # NEW: Determine padding frames count
    # A reasonable amount, e.g., frames_chunk // 2 or more, to give temporal context
    padding_frames_count = frames_chunk # Example: pad with a full chunk's worth of the last frame

    # NEW: Create padding frames by repeating the last frame
    # Ensure there's at least one frame to repeat
    if num_frames_original > 0:
        last_frame_to_repeat = frames[-1:].clone() # Shape [1, C, H, W]
        repeated_frames = last_frame_to_repeat.repeat(padding_frames_count, 1, 1, 1)
        frames = torch.cat([frames, repeated_frames], dim=0) # NEW: Concatenate padding to main frames tensor
        logger.debug(f"Padded video frames from {num_frames_original} to {frames.shape[0]} by repeating the last frame.")
    else:
        # If num_frames_original is 0, this block won't be reached due to the earlier check.
        # But for safety, handle the case where it might slip through.
        logger.warning("Attempted to pad an empty video; padding skipped.")
    
    num_frames = frames.shape[0] # NEW: Update num_frames to include padding

    _, _, total_h_raw_input, total_w_raw_input = frames.shape
    if total_h_raw_input < 2 or total_w_raw_input < 2:
        logger.warning(f"Video {input_video_path} is too small ({total_h_raw_input}x{total_w_raw_input}), skipping.")
        if update_info_callback:
            update_info_callback(base_video_name, f"{total_w_raw_input}x{total_h_raw_input}", num_frames, overlap, original_input_blend_strength)
        return False

    # Determine output resolution for GUI display (after splitting/inpainting)
    output_display_w = total_w_raw_input // 2
    output_display_h = total_h_raw_input // 2 if not is_dual_input else total_h_raw_input
    current_display_res_str = f"{output_display_w}x{output_display_h}"
    
    # Update GUI with video info before processing
    if update_info_callback:
        update_info_callback(base_video_name, current_display_res_str, num_frames, overlap, original_input_blend_strength)

    # Frame chunking and mask extraction based on input type
    frames_left_original: Optional[torch.Tensor] = None # Initialize outside if/else
    if is_dual_input:
        half_w_input = total_w_raw_input // 2
        frames_mask_raw = frames[:, :, :, :half_w_input]  # Left half is mask
        frames_warpped_raw = frames[:, :, :, half_w_input:] # Right half is warped
        
        frames_warpped = frames_warpped_raw
        frames_mask = frames_mask_raw.mean(dim=1, keepdim=True) # Convert mask to grayscale
        
        target_output_h = frames_warpped.shape[2]
        target_output_w = frames_warpped.shape[3]

    else:
        half_h_input = total_h_raw_input // 2
        half_w_input = total_w_raw_input // 2

        frames_left_original_before_padding = frames[:, :, :half_h_input, :half_w_input] # Top-Left is original view

        # NEW: Apply padding to frames_left_original as well
        if num_frames_original > 0: # Ensure original video had frames
            last_frame_left = frames_left_original_before_padding[-1:].clone()
            repeated_frames_left = last_frame_left.repeat(padding_frames_count, 1, 1, 1)
            frames_left_original = torch.cat([frames_left_original_before_padding, repeated_frames_left], dim=0)
            logger.debug(f"Padded frames_left_original from {num_frames_original} to {frames_left_original.shape[0]}.")
        else:
            frames_left_original = frames_left_original_before_padding # Should be empty anyway

        frames_mask_raw = frames[:, :, half_h_input:, :half_w_input]  # Bottom-Left is mask
        frames_warpped_raw = frames[:, :, half_h_input:, half_w_input:] # Bottom-Right is warped

        frames_warpped = frames_warpped_raw
        frames_mask = frames_mask_raw.mean(dim=1, keepdim=True) # Convert mask to grayscale

        target_output_h = frames_warpped.shape[2]
        target_output_w = frames_warpped.shape[3]

    # Pre-process frames for inpainting pipeline
    frames_warpped = frames_warpped / 255.0 # Normalize to 0-1

    # Pad for tiling before processing
    frames_warpped_padded = pad_for_tiling(frames_warpped, tile_num, tile_overlap=(128, 128))
    frames_mask_padded = pad_for_tiling(frames_mask, tile_num, tile_overlap=(128, 128))
    
    padded_H, padded_W = frames_warpped_padded.shape[2], frames_warpped_padded.shape[3]


    stride = max(1, frames_chunk - overlap)
    results = [] # Stores chunks to be concatenated in final video
    previous_chunk_output_frames = None

    for i in range(0, num_frames, stride): # Loop over original num_frames
        if stop_event and stop_event.is_set():
            logger.info(f"Stopping processing of {input_video_path}")
            return False
        
        # Adjust end_idx to use the original number of frames, not padded
        end_idx = min(i + frames_chunk, num_frames)
        chunk_size = end_idx - i
        if chunk_size <= 0:
            break

        # Get the original input for the current chunk (from padded frames)
        # We need to slice the padded frames for inference, but the logic for
        # `previous_chunk_output_frames` and `input_frames_to_pipeline` handles
        # the blending of *actual* generated content.
        # It's important that `frames_warpped_padded` and `frames_mask_padded`
        # are used here.
        original_input_frames_for_chunk = frames_warpped_padded[i:end_idx].clone()
        mask_frames_i = frames_mask_padded[i:end_idx].clone()

        input_frames_to_pipeline = original_input_frames_for_chunk.clone()

        # Input-level blending for overlapping frames
        if previous_chunk_output_frames is not None and overlap > 0:
            overlap_actual = min(overlap, len(previous_chunk_output_frames), len(original_input_frames_for_chunk))

            if overlap_actual > 0:
                prev_gen_overlap_frames = previous_chunk_output_frames[-overlap_actual:]
                
                if original_input_blend_strength > 0:
                    orig_input_overlap_frames = original_input_frames_for_chunk[:overlap_actual]
                    original_weights_scaled = torch.linspace(0.0, 1.0, overlap_actual, device=prev_gen_overlap_frames.device).view(-1, 1, 1, 1) * original_input_blend_strength
                    
                    blended_input_overlap_frames = (1 - original_weights_scaled) * prev_gen_overlap_frames + \
                                                    original_weights_scaled * orig_input_overlap_frames
                    
                    input_frames_to_pipeline[:overlap_actual] = blended_input_overlap_frames
                    del orig_input_overlap_frames
                    del original_weights_scaled
                    del blended_input_overlap_frames
                else:
                    input_frames_to_pipeline[:overlap_actual] = prev_gen_overlap_frames
                
                del prev_gen_overlap_frames

        logger.info(f"Starting inference for chunk {i}-{end_idx} (padded size {input_frames_to_pipeline.shape[2]}x{input_frames_to_pipeline.shape[3]})...")
        start_time = time.time()

        with torch.no_grad():
            video_latents = spatial_tiled_process(
                input_frames_to_pipeline,
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
        logger.debug(f"Inference for chunk {i}-{end_idx} completed in {inference_duration:.2f} seconds.")
        
        video_frames = tensor2vid(decoded_frames, pipeline.image_processor, output_type="pil")[0]
        current_chunk_generated_frames = []
        for j in range(len(video_frames)):
            img = video_frames[j]
            current_chunk_generated_frames.append(torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0)

        current_chunk_generated = torch.stack(current_chunk_generated_frames)

        if i == 0:
            results.append(current_chunk_generated)
        else:
            results.append(current_chunk_generated[overlap:])
        
        previous_chunk_output_frames = current_chunk_generated 

        # The loop condition `end_idx == num_frames` is good for breaking.

    # --- START CRITICAL FIXES FOR NAMEERROR AND ROBUSTNESS ---

    # Check if any frames were successfully generated and collected in `results`.
    if not results:
        logger.warning(f"No frames were successfully generated for {input_video_path} after inference. Skipping video output.")
        if update_info_callback:
            update_info_callback(base_video_name, "N/A", "0 (No Output)", overlap, original_input_blend_strength)
        return False

    frames_output = torch.cat(results, dim=0).cpu()

    # Initialize frames_output_final defensively.
    frames_output_final: Optional[torch.Tensor] = None

    # Check for valid dimensions before cropping
    # frames_output.shape[0] will be the number of frames from `torch.cat(results)`
    if frames_output.numel() > 0 and frames_output.shape[2] >= target_output_h and frames_output.shape[3] >= target_output_w:
        # Crop the output to the original, unpadded dimensions for this section
        frames_output_final = frames_output[:, :, :target_output_h, :target_output_w]
        # NEW: Temporally crop the output back to the original video length
        frames_output_final = frames_output_final[:num_frames_original]
        logger.debug(f"Temporally cropped generated frames from {frames_output.shape[0]} to {frames_output_final.shape[0]} (original length).")
    else:
        logger.error(f"Generated frames_output has invalid dimensions for final cropping (actual {frames_output.shape[2]}x{frames_output.shape[3]} vs target {target_output_h}x{target_output_w}) for {input_video_path}. Skipping video output.")
        if update_info_callback:
            update_info_callback(base_video_name, "N/A", "0 (Crop Error)", overlap, original_input_blend_strength)
        return False

    # Initialize final_output_frames_for_encoding defensively
    final_output_frames_for_encoding: Optional[torch.Tensor] = None

    if is_dual_input:
        final_output_frames_for_encoding = frames_output_final # (T, C, H, W) float 0-1
    else:
        # Ensure frames_left_original is valid for SBS concatenation
        if frames_left_original is None or frames_left_original.numel() == 0:
            logger.error(f"Original left frames are missing or empty for non-dual input {input_video_path}. Cannot create SBS output. Skipping video output.")
            if update_info_callback:
                update_info_callback(base_video_name, "N/A", "0 (SBS Error)", overlap, original_input_blend_strength)
            return False
        
        # NEW: frames_left_original was already padded, now crop it back
        frames_left_original_cropped = frames_left_original[:num_frames_original].float() / 255.0
        
        # Ensure dimensions match for concatenation (time, channel, height should match)
        if frames_left_original_cropped.shape[0] != frames_output_final.shape[0] or \
           frames_left_original_cropped.shape[1] != frames_output_final.shape[1] or \
           frames_left_original_cropped.shape[2] != frames_output_final.shape[2]:
            logger.error(f"Dimension mismatch for SBS concatenation: Left {frames_left_original_cropped.shape}, Inpainted {frames_output_final.shape} for {input_video_path}. Skipping video output.")
            if update_info_callback:
                update_info_callback(base_video_name, "N/A", "0 (Dim Mismatch)", overlap, original_input_blend_strength)
            return False

        sbs_frames = torch.cat([frames_left_original_cropped, frames_output_final], dim=3)
        final_output_frames_for_encoding = sbs_frames

    # Final check: ensure the tensor to be encoded is actually populated
    if final_output_frames_for_encoding is None or final_output_frames_for_encoding.numel() == 0:
        logger.error(f"Final output frames for encoding are empty or None after preparation for {input_video_path}. Skipping video output.")
        if update_info_callback:
            update_info_callback(base_video_name, "N/A", "0 (Empty Final)", overlap, original_input_blend_strength)
        return False

    # --- END CRITICAL FIXES ---

    # --- START NEW: Intermediate PNG Sequence Saving and Final FFmpeg Encoding ---
    temp_png_dir = os.path.join(save_dir, f"temp_inpainted_pngs_{video_name_for_output}_{os.getpid()}")
    os.makedirs(temp_png_dir, exist_ok=True)
    logger.debug(f"Saving intermediate 16-bit PNG sequence to {temp_png_dir}")

    total_output_frames = final_output_frames_for_encoding.shape[0]
    try:
        for frame_idx in range(total_output_frames):
            if stop_event and stop_event.is_set():
                logger.debug(f"Stopping PNG sequence saving for {input_video_path}")
                return False

            frame_tensor = final_output_frames_for_encoding[frame_idx] # (C, H, W) float 0-1
            frame_np = frame_tensor.permute(1, 2, 0).numpy() # (H, W, C) float 0-1

            # Convert to 16-bit
            frame_uint16 = (np.clip(frame_np, 0.0, 1.0) * 65535.0).astype(np.uint16)
            
            # Convert to BGR for OpenCV (OpenCV uses BGR by default for imwrite)
            # Assuming the pipeline output is RGB (common for image models)
            frame_bgr = cv2.cvtColor(frame_uint16, cv2.COLOR_RGB2BGR)

            png_path = os.path.join(temp_png_dir, f"{frame_idx:05d}.png")
            cv2.imwrite(png_path, frame_bgr)
            draw_progress_bar(frame_idx + 1, total_output_frames)
        logger.debug(f"\nFinished saving {total_output_frames} PNG frames.")

        # --- Construct FFmpeg Command ---
        ffmpeg_cmd = [
            "ffmpeg",
            "-y", # Overwrite output files without asking
            "-framerate", str(fps), # Use the detected FPS
            "-i", os.path.join(temp_png_dir, "%05d.png"), # Input PNG sequence
        ]

        # Default output parameters
        output_codec = "libx264"
        output_pix_fmt = "yuv420p"
        output_crf = "18" # Default CRF for SDR x264 (lower is better quality)
        output_profile = "main"
        x265_params = [] # For HDR metadata

        if video_stream_info:
            logger.debug(f"Applying color metadata from source: {video_stream_info}")
            input_pix_fmt = video_stream_info.get("pix_fmt", "")
            color_primaries = video_stream_info.get("color_primaries")
            transfer_characteristics = video_stream_info.get("transfer_characteristics")
            color_space = video_stream_info.get("color_space")

            # Determine HDR status
            is_hdr_source = (color_primaries == "bt2020" and transfer_characteristics == "smpte2084")

            # Determine if original source was 10-bit or higher
            is_original_10bit_or_higher = "10" in input_pix_fmt or "12" in input_pix_fmt or "16" in input_pix_fmt

            if is_hdr_source:
                logger.debug("Detected HDR source. Encoding with H.265 10-bit and HDR metadata.")
                output_codec = "libx265"
                output_pix_fmt = "yuv420p10le"
                output_crf = "20" # Higher CRF for HEVC 10-bit (higher is lower quality, but 28 is common for HDR)
                output_profile = "main10"
                
                # Add HDR mastering display and CLL metadata
                mastering_display = video_stream_info.get("mastering_display_metadata")
                max_cll = video_stream_info.get("max_content_light_level")
                if mastering_display:
                    x265_params.append(f"master-display={mastering_display}")
                if max_cll:
                    x265_params.append(f"max-cll={max_cll}")
            elif is_original_10bit_or_higher and video_stream_info.get("codec_name") == "hevc":
                logger.debug("Detected SDR 10-bit HEVC source. Encoding with H.265 10-bit.")
                output_codec = "libx265"
                output_pix_fmt = "yuv420p10le"
                output_crf = "20" # Slightly lower CRF for SDR 10-bit HEVC (better quality than HDR)
                output_profile = "main10"
            else: # SDR 8-bit, or other source codecs
                logger.debug("Detected SDR (8-bit H.264 or other) source. Encoding with H.264 8-bit.")
                output_codec = "libx264"
                output_pix_fmt = "yuv420p"
                output_crf = "18" # Higher quality CRF for SDR H.264 (lower is better quality)
                output_profile = "main"

            # Apply general color flags if present
            if color_primaries:
                ffmpeg_cmd.extend(["-color_primaries", color_primaries])
            if transfer_characteristics:
                ffmpeg_cmd.extend(["-color_trc", transfer_characteristics])
            if color_space:
                ffmpeg_cmd.extend(["-colorspace", color_space])

        # Add codec, profile, pix_fmt, and CRF
        ffmpeg_cmd.extend([
            "-c:v", output_codec,
            "-profile:v", output_profile,
            "-pix_fmt", output_pix_fmt,
            "-crf", output_crf,
        ])
        
        if x265_params:
            ffmpeg_cmd.extend(["-x265-params", ":".join(x265_params)])

        # Final output path
        ffmpeg_cmd.append(output_video_path)

        logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        # Update GUI status for encoding phase
        if update_info_callback:
            update_info_callback(base_video_name, f"Encoding {output_codec}...", total_output_frames, overlap, original_input_blend_strength)

        subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True, encoding='utf-8', timeout=3600*24) # 24h timeout
        logger.debug(f"Successfully encoded video to {output_video_path}")

    except FileNotFoundError:
        messagebox.showerror("Error", "FFmpeg not found. Please ensure FFmpeg is installed and in your system PATH.")
        logger.error("FFmpeg not found in PATH.")
        return False
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"FFmpeg encoding failed for {base_video_name}:\n{e.stderr}\n{e.stdout}")
        logger.error(f"FFmpeg encoding failed for {base_video_name}: {e.stderr}\n{e.stdout}")
        return False
    except subprocess.TimeoutExpired as e:
        messagebox.showerror("Error", f"FFmpeg encoding timed out for {base_video_name}:\n{e.stderr}")
        logger.error(f"FFmpeg encoding timed out for {base_video_name}: {e.stderr}")
        return False
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred during encoding for {base_video_name}: {str(e)}")
        logger.error(f"Unexpected error during encoding for {base_video_name}: {e}", exc_info=True)
        return False
    finally:
        # Cleanup temporary PNGs
        if os.path.exists(temp_png_dir):
            try:
                shutil.rmtree(temp_png_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_png_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary PNG directory {temp_png_dir}: {e}")
    # --- END NEW ---

    logger.info(f"Done processing {input_video_path} -> {output_video_path}")
    return True

class InpaintingGUI(tk.Tk):
    
    def __init__(self):
        super().__init__()
        self.title("Batch Video Inpainting")   
        self.geometry("600x590")
        self.app_config = self.load_config()
        self.help_data = self.load_help_data()

        self.input_folder_var = tk.StringVar(value=self.app_config.get("input_folder", "./output_splatted"))
        self.output_folder_var = tk.StringVar(value=self.app_config.get("output_folder", "./completed_output"))
        self.num_inference_steps_var = tk.StringVar(value=str(self.app_config.get("num_inference_steps", 5)))
        self.tile_num_var = tk.StringVar(value=str(self.app_config.get("tile_num", 2)))
        self.frames_chunk_var = tk.StringVar(value=str(self.app_config.get("frames_chunk", 23)))
        self.overlap_var = tk.StringVar(value=str(self.app_config.get("frame_overlap", 3)))
        self.original_input_blend_strength_var = tk.StringVar(value=str(self.app_config.get("original_input_blend_strength", 0.5))) # Default to 0.5
        self.offload_type_var = tk.StringVar(value=self.app_config.get("offload_type", "model"))
        self.processed_count = tk.IntVar(value=0)
        self.total_videos = tk.IntVar(value=0)
        self.stop_event = threading.Event()
        self.pipeline = None

        self.create_widgets()
        self.update_progress()
        self.update_status_label("Ready")
        self.protocol("WM_DELETE_WINDOW", self.exit_application)

    def create_widgets(self):
        
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        option_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Option", menu=option_menu)
        option_menu.add_command(label="Reset to Default", command=self.reset_to_defaults)
        option_menu.add_command(label="Restore Finished", command=self.restore_finished_files)

        folder_frame = ttk.LabelFrame(self, text="Folders", padding=10)
        folder_frame.pack(fill="x", padx=10, pady=5)
        folder_frame.grid_columnconfigure(1, weight=1)
        
        # Input Folder
        input_label = ttk.Label(folder_frame, text="Input Folder:")
        input_label.grid(row=0, column=0, sticky="e", padx=5, pady=2)
        Tooltip(input_label, self.help_data.get("input_folder", ""))
        ttk.Entry(folder_frame, textvariable=self.input_folder_var, width=40).grid(row=0, column=1, padx=5, sticky="ew")
        ttk.Button(folder_frame, text="Browse", command=self.browse_input).grid(row=0, column=2, padx=5)
        
        # Output Folder
        output_label = ttk.Label(folder_frame, text="Output Folder:")
        output_label.grid(row=1, column=0, sticky="e", padx=5, pady=2)
        Tooltip(output_label, self.help_data.get("output_folder", ""))
        ttk.Entry(folder_frame, textvariable=self.output_folder_var, width=40).grid(row=1, column=1, padx=5, sticky="ew")
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

        self.video_overlap_label = ttk.Label(info_frame, text="Overlap: N/A")
        self.video_overlap_label.pack(anchor="w", padx=5, pady=1)

        self.video_bias_label = ttk.Label(info_frame, text="Input Bias: N/A")
        self.video_bias_label.pack(anchor="w", padx=5, pady=1)

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

        self.save_config() # Save these new default settings
        messagebox.showinfo("Settings Reset", "All settings have been reset to their default values.")
        logger.info("GUI settings reset to defaults.")

    def restore_finished_files(self):
        if not messagebox.askyesno("Restore Finished Files", "Are you sure you want to move all processed videos from the 'finished' folder back to the input directory?"):
            return

        input_folder = self.input_folder_var.get()
        finished_folder = os.path.join(input_folder, "finished")

        if not os.path.isdir(finished_folder):
            messagebox.showinfo("Restore Info", f"The 'finished' folder does not exist at '{finished_folder}'. No files to restore.")
            logger.info(f"Restore finished: 'finished' folder not found at {finished_folder}")
            return

        restored_count = 0
        errors_count = 0
        
        # Collect files to move first, to avoid issues if the directory changes during iteration
        files_to_move = [f for f in os.listdir(finished_folder) if os.path.isfile(os.path.join(finished_folder, f))]

        if not files_to_move:
            messagebox.showinfo("Restore Info", "No files found in the 'finished' folder to restore.")
            logger.info(f"Restore finished: No files found in {finished_folder}")
            return

        for filename in files_to_move:
            src_path = os.path.join(finished_folder, filename)
            dest_path = os.path.join(input_folder, filename)
            try:
                shutil.move(src_path, dest_path)
                restored_count += 1
                logger.info(f"Moved '{filename}' from '{finished_folder}' to '{input_folder}'")
            except Exception as e:
                errors_count += 1
                logger.error(f"Error moving file '{filename}' during restore: {e}")

        if restored_count > 0 or errors_count > 0:
            messagebox.showinfo("Restore Complete", f"Finished files restoration attempted.\n{restored_count} files moved.\n{errors_count} errors occurred.")
            logger.info(f"Restore complete: {restored_count} files moved, {errors_count} errors.")
        else:
            messagebox.showinfo("Restore Complete", "No files found to restore.")
            logger.info("Restore complete: No files found to restore.")

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

    def update_video_info_display(self, name, resolution, frames, overlap_val="N/A", bias_val="N/A"):
        self.video_name_label.config(text=f"Name: {name}")
        self.video_res_label.config(text=f"Resolution: {resolution}")
        self.video_frames_label.config(text=f"Frames: {frames}")
        self.video_overlap_label.config(text=f"Overlap: {overlap_val}")
        self.video_bias_label.config(text=f"Input Bias: {bias_val}")

    def start_processing(self):
        input_folder = self.input_folder_var.get()
        output_folder = self.output_folder_var.get()
        try:
            num_inference_steps = int(self.num_inference_steps_var.get())
            tile_num = int(self.tile_num_var.get())
            frames_chunk = int(self.frames_chunk_var.get())
            gui_overlap = int(self.overlap_var.get()) # Renamed for clarity
            gui_original_input_blend_strength = float(self.original_input_blend_strength_var.get())
            if num_inference_steps < 1 or tile_num < 1 or frames_chunk < 1 or gui_overlap  < 0 or \
               not (0.0 <= gui_original_input_blend_strength  <= 1.0): # NEW VALIDATION
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
        self.update_video_info_display("N/A", "N/A", "N/A", "N/A", "N/A")

        threading.Thread(target=self.run_batch_process,
                         args=(input_folder, output_folder, num_inference_steps, tile_num, offload_type, frames_chunk, gui_overlap, gui_original_input_blend_strength),
                         daemon=True).start()

    def run_batch_process(self, input_folder, output_folder, num_inference_steps, tile_num, offload_type, frames_chunk, gui_overlap, gui_original_input_blend_strength):
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
            finished_folder = os.path.join(input_folder, "finished")
            os.makedirs(finished_folder, exist_ok=True)
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

                json_path = os.path.splitext(video_path)[0] + ".json"
                if os.path.exists(json_path):
                    logger.info(f"Found sidecar JSON for {os.path.basename(video_path)} at {json_path}")
                    try:
                        with open(json_path, 'r') as f:
                            sidecar_data = json.load(f)
                        
                        if "frame_overlap" in sidecar_data:
                            sidecar_overlap = int(sidecar_data["frame_overlap"])
                            if sidecar_overlap >= 0:
                                current_overlap = sidecar_overlap
                                logger.debug(f"Using frame_overlap from sidecar: {current_overlap}")
                            else:
                                logger.warning(f"Invalid 'frame_overlap' in sidecar JSON for {os.path.basename(video_path)}. Using GUI value ({gui_overlap}).")

                        if "input_bias" in sidecar_data:
                            sidecar_input_bias = float(sidecar_data["input_bias"])
                            if 0.0 <= sidecar_input_bias <= 1.0:
                                current_original_input_blend_strength = sidecar_input_bias
                                logger.debug(f"Using input_bias from sidecar: {current_original_input_blend_strength}")
                            else:
                                logger.warning(f"Invalid 'input_bias' in sidecar JSON for {os.path.basename(video_path)}. Using GUI value ({gui_original_input_blend_strength}).")

                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Error reading or parsing sidecar JSON {json_path}: {e}. Falling back to GUI parameters for this video.")
                else:
                    logger.debug(f"No sidecar JSON found for {os.path.basename(video_path)}. Using GUI parameters.")

                # Update status label to indicate which video is starting processing
                self.after(0, self.update_status_label, f"Processing video {idx + 1} of {self.total_videos.get()}")

                logger.info(f"Starting processing of {video_path}")
                completed = process_single_video(
                    pipeline=self.pipeline,
                    input_video_path=video_path,
                    save_dir=output_folder,
                    frames_chunk=frames_chunk,
                    overlap=current_overlap, # Pass the (potentially overridden) overlap
                    tile_num=tile_num,
                    vf=None, # Keep as None, not actively used with new FFmpeg logic
                    num_inference_steps=num_inference_steps,
                    stop_event=self.stop_event,
                    update_info_callback=_threaded_update_info_callback, # Pass the wrapped callback
                    original_input_blend_strength=current_original_input_blend_strength # Pass the (potentially overridden) bias
                )
                
                if completed:
                    try:
                        shutil.move(video_path, finished_folder)
                        logger.debug(f"Moved {video_path} to {finished_folder}")
                    except Exception as e:
                        logger.error(f"Failed to move {video_path} to {finished_folder}: {e}")
                else:
                    logger.info(f"Processing of {video_path} was stopped or skipped due to issues.")
                
                self.processed_count.set(idx + 1)
                
            stopped = self.stop_event.is_set()
            self.after(0, lambda: self.processing_done(stopped))

        except Exception as e:
            logger.exception("An unhandled error occurred during batch processing.") # Log full traceback
            self.after(0, lambda: messagebox.showerror("Error", f"An error occurred during batch processing: {str(e)}"))
            self.after(0, self.processing_done)

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

    def exit_application(self):
        self.save_config() 
        self.destroy()

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

    def save_config(self):
        config = {
            "input_folder": self.input_folder_var.get(),
            "output_folder": self.output_folder_var.get(),
            "num_inference_steps": self.num_inference_steps_var.get(),
            "tile_num": self.tile_num_var.get(),
            "offload_type": self.offload_type_var.get(),
            "frames_chunk": self.frames_chunk_var.get(),
            "frame_overlap": self.overlap_var.get(),
            "original_input_blend_strength": self.original_input_blend_strength_var.get()             
        }
        try:
            with open("config_inpaint.json", "w", encoding='utf-8') as f: # Added encoding for robustness
                json.dump(config, f, indent=4)
            logger.info("Configuration saved successfully.")
        except Exception as e:
            logger.warning(f"Failed to save config: {e}", exc_info=True)

    def show_general_help(self):
        help_text = self.help_data.get("general_help", "No general help information available.")
        messagebox.showinfo("Help", help_text)

if __name__ == "__main__":
    app = InpaintingGUI()
    app.mainloop()