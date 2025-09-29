import argparse
import json
import os
import numpy as np
import shutil 
import sys
import time 
import logging # Import standard logging

# Configure a logger for this module
_logger = logging.getLogger(__name__)

from depthcrafter import dav_util as util
from depthcrafter import utils as dc_utils
from depthcrafter.utils import (
    normalize_video_data,
    apply_gamma_correction_to_video,
    apply_dithering_to_video,
    load_json_file,
)
import imageio
from typing import Optional

_HAS_OPENEXR = False
try:
    import OpenEXR
    import Imath
    _HAS_OPENEXR = True
except ImportError:
    _logger.warning("OpenEXR/Imath libraries not found. EXR features will be limited/unavailable. Context: merge_depth_segments.py")


def _apply_robust_global_normalization(
    raw_stitched_depth_frames: np.ndarray,
    low_perc_for_range: float,
    high_perc_for_range: float,
    target_output_min: float = 0.0,
    target_output_max: float = 1.0,
    is_far_black: bool = True
) -> np.ndarray:
    """
    Applies a robust global normalization to a raw stitched depth map
    based on percentiles, mapping to a specified output range.

    Args:
        raw_stitched_depth_frames (np.ndarray): The 3D NumPy array of raw depth frames (H, W, Frames).
        low_perc_for_range (float): The percentile (e.g., 0.1) that defines the numerically smallest
                                    stable raw depth value.
        high_perc_for_range (float): The percentile (e.g., 99.9) that defines the numerically largest
                                     stable raw depth value.
        target_output_min (float, optional): The desired output minimum value (e.g., 0.0 for black). Defaults to 0.0.
        target_output_max (float, optional): The desired output maximum value (e.g., 1.0 for white). Defaults to 1.0.
        is_far_black (bool, optional): True if far objects are black (0) and close objects are white (1)
                                       in the *final visual output*. Defaults to True (common for DepthCrafter visual output).

    Returns:
        np.ndarray: The normalized and clipped depth frames.
    """
    if raw_stitched_depth_frames.size == 0:
        _logger.warning("Attempted robust global normalization on empty frames. Returning empty array.")
        return np.array([])

    # Flatten and filter out non-finite values for robust percentile calculation
    filtered_data = raw_stitched_depth_frames[np.isfinite(raw_stitched_depth_frames)].flatten()

    if filtered_data.size == 0:
        _logger.warning("No finite depth values found for percentile calculation. Returning zeros.")
        return np.zeros_like(raw_stitched_depth_frames, dtype=np.float32)

    # Determine raw bounds based on percentiles.
    # val_at_low_perc will be the numerically smallest raw depth (e.g., the 0.1th percentile).
    # val_at_high_perc will be the numerically largest raw depth (e.g., the 99.9th percentile).
    val_at_low_perc = np.percentile(filtered_data, low_perc_for_range)
    val_at_high_perc = np.percentile(filtered_data, high_perc_for_range)

    # Assume DepthCrafter raw output is "Inverse Depth": smaller raw value = farther, larger raw value = closer.
    # Based on this, we define the effective raw bounds for mapping:
    effective_raw_farthest_val = val_at_low_perc
    effective_raw_closest_val = val_at_high_perc

    # Handle edge case where the range is extremely small or zero
    if effective_raw_closest_val <= effective_raw_farthest_val + 1e-6: # Add small epsilon to handle near-zero range
        _logger.warning(f"Robust normalization: Effective closest raw val ({effective_raw_closest_val:.4f}) is effectively <= effective farthest raw val ({effective_raw_farthest_val:.4f}). Returning uniform output.")
        # If range is zero, all values map to the mid-point of the target output range
        return np.full_like(raw_stitched_depth_frames, (target_output_min + target_output_max) / 2.0, dtype=np.float32)

    s_custom, t_custom = 1.0, 0.0

    # Calculate custom scale and shift for the desired final visual output (Far=Black, Close=White)
    if is_far_black: # If final visual output wants Far=Black (0) and Close=White (1)
        # Map the effective_raw_farthest_val to target_output_min (black side)
        # Map the effective_raw_closest_val to target_output_max (white side)
        s_custom = (target_output_max - target_output_min) / (effective_raw_closest_val - effective_raw_farthest_val)
        t_custom = target_output_min - (s_custom * effective_raw_farthest_val)
    else: # If final visual output wants Close=Black (0) and Far=White (1)
        # Map the effective_raw_closest_val to target_output_min (black side)
        # Map the effective_raw_farthest_val to target_output_max (white side)
        s_custom = (target_output_max - target_output_min) / (effective_raw_farthest_val - effective_raw_closest_val)
        t_custom = target_output_min - (s_custom * effective_raw_closest_val)


    # Apply transformation
    normalized_frames = (raw_stitched_depth_frames * s_custom) + t_custom

    # Clip to the target output range
    clipped_frames = np.clip(normalized_frames, target_output_min, target_output_max)

    _logger.debug(f"Robust norm: Raw percentiles low={val_at_low_perc:.4f}, high={val_at_high_perc:.4f}")
    _logger.debug(f"Robust norm: Effective raw farthest={effective_raw_farthest_val:.4f}, closest={effective_raw_closest_val:.4f}")
    _logger.debug(f"Robust norm: Calculated s_custom={s_custom:.4f}, t_custom={t_custom:.4f}")
    _logger.debug(f"Robust norm output range: [{target_output_min:.4f}, {target_output_max:.4f}]")
    _logger.debug(f"Robust norm final min/max (after clip): {clipped_frames.min():.4f}/{clipped_frames.max():.4f}")

    return clipped_frames

def save_single_frame_exr(frame_data: np.ndarray, output_path: str):
    if not _HAS_OPENEXR:
        _logger.error("OpenEXR/Imath libraries not found by save_single_frame_exr. Cannot save EXR.")
        raise RuntimeError("OpenEXR/Imath libraries not found. Cannot save EXR.")
    if frame_data.ndim != 2:
        _logger.error(f"Frame data for EXR must be 2D (H, W). Got shape: {frame_data.shape}")
        raise ValueError(f"Frame data must be 2D (H, W) for EXR saving. Got shape: {frame_data.shape}")
    
    height, width = frame_data.shape
    frame_data_float32 = frame_data.astype(np.float32)
    header = OpenEXR.Header(width, height)
    header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    exr_file = OpenEXR.OutputFile(output_path, header)
    try:
        exr_file.writePixels({"Z": frame_data_float32.tobytes()})
    finally:
        exr_file.close()

def _load_and_validate_metadata(master_meta_path: str):
    _logger.debug(f"Loading merge metadata from: {master_meta_path}")
    meta_data = load_json_file(master_meta_path)
    if not meta_data:
        raise FileNotFoundError(f"Failed to load or parse master metadata file: {master_meta_path}")

    if not meta_data.get("global_processing_settings", {}).get("processed_as_segments"):
        _logger.critical(f"'processed_as_segments' is not true in metadata: {master_meta_path}. Aborting merge.")
        raise ValueError("'processed_as_segments' is not true in metadata.")

    global_settings = meta_data.get("global_processing_settings", {})
    N_overlap_from_meta = global_settings.get("segment_definition_output_overlap_frames")
    if N_overlap_from_meta is None:
        _logger.critical(f"'segment_definition_output_overlap_frames' not found in metadata: {master_meta_path}. Aborting merge.")
        raise ValueError("'segment_definition_output_overlap_frames' not found.")
    _logger.debug(f"Defined overlap frames (N_overlap) from metadata: {N_overlap_from_meta}")

    jobs_info = meta_data.get("jobs_info", [])
    if not jobs_info:
        _logger.warning(f"Warning: No job segments found in metadata: {master_meta_path}.")
    
    successful_jobs_info = [job for job in jobs_info if job.get("status") == "success" and job.get("output_segment_filename")]
    if not successful_jobs_info:
        _logger.critical(f"No successful segments found in metadata to merge: {master_meta_path}.")
        raise ValueError("No successful segments found in metadata to merge.")
        
    sorted_jobs_info = sorted(successful_jobs_info, key=lambda x: x.get("segment_id", -1))
    _logger.debug(f"Found {len(sorted_jobs_info)} successful segments to process from metadata.")
    base_dir = os.path.dirname(master_meta_path) if master_meta_path and os.path.dirname(master_meta_path) else "."
    return meta_data, N_overlap_from_meta, sorted_jobs_info, base_dir

def _load_single_segment_frames(job_meta: dict, base_dir: str):
    segment_filename = job_meta.get("output_segment_filename")
    input_segment_format = job_meta.get("output_segment_format", "npz").lower()
    processed_fps_from_meta = job_meta.get("processed_at_fps")
    segment_path = os.path.join(base_dir, segment_filename)

    if not os.path.exists(segment_path):
        _logger.error(f"File not found: {segment_path}")
        raise FileNotFoundError(f"Segment file not found: {segment_path}.")

    if input_segment_format != "npz":
        _logger.critical(f"Unsupported segment format '{input_segment_format}' for {segment_filename}. Expecting NPZ.")
        raise ValueError(f"Unsupported segment format '{input_segment_format}'. Expecting NPZ.")
    
    try:
        with np.load(segment_path) as data:
            if 'frames' not in data.files:
                _logger.error(f"Key 'frames' not found in NPZ: {segment_path}")
                raise KeyError(f"Key 'frames' not found in NPZ: {segment_path}.")
            frames = data['frames']
    except Exception as e:
        _logger.critical(f"Could not load frames from NPZ {segment_path}: {e}")
        raise

    if frames is None or frames.size == 0:
        _logger.critical(f"Segment {segment_filename} is empty.")
        raise ValueError(f"Segment {segment_filename} is empty.")

    fps = float(processed_fps_from_meta) if processed_fps_from_meta else 30.0
    _logger.debug(f"Single segment {segment_filename} loaded. Shape: {frames.shape}, FPS: {fps:.2f}")
    return frames.astype(np.float32), fps

def _load_multiple_segments_data(sorted_jobs_info: list, base_dir: str):
    _logger.debug("\n--- Pass 1: Loading Segments ---")
    all_loaded_segments_frames = []
    segment_job_meta_map = []
    determined_fps = None

    for idx, job_meta in enumerate(sorted_jobs_info):
        segment_id = job_meta.get("segment_id", f"unknown_id_{idx}")
        segment_filename = job_meta.get("output_segment_filename")
        input_segment_format = job_meta.get("output_segment_format", "npz").lower()
        segment_path = os.path.join(base_dir, segment_filename)
        processed_fps_from_meta = job_meta.get("processed_at_fps")

        if processed_fps_from_meta is None:
            _logger.critical(f"'processed_at_fps' missing for segment ID {segment_id}.")
            raise ValueError(f"'processed_at_fps' missing for segment ID {segment_id}.")
        
        current_fps = float(processed_fps_from_meta)
        if determined_fps is None:
            determined_fps = current_fps
        elif abs(determined_fps - current_fps) > 1e-3:
            _logger.warning(f"Warning: Inconsistent FPS. Using {determined_fps:.2f}. Segment {segment_id} has {current_fps:.2f}.")

        _logger.debug(f"Loading segment {segment_id} ({idx+1}/{len(sorted_jobs_info)}): {segment_filename}")
        if input_segment_format != "npz":
            _logger.critical(f"Unsupported segment format '{input_segment_format}' for {segment_filename}. Expecting NPZ.")
            raise ValueError(f"Unsupported segment format '{input_segment_format}' for {segment_filename}.")
        
        try:
            with np.load(segment_path) as data:
                if 'frames' not in data.files:
                    _logger.error(f"Key 'frames' not found in NPZ: {segment_path}")
                    raise KeyError(f"Key 'frames' not found in NPZ: {segment_path}.")
                frames = data['frames']
        except Exception as e:
            _logger.critical(f"Could not load frames from NPZ {segment_path}: {e}")
            raise

        if frames is None or frames.size == 0:
            _logger.warning(f"Warning: Segment {segment_id} ({segment_filename}) data is empty. Skipping.")
            continue
        
        all_loaded_segments_frames.append(frames.astype(np.float32).copy())
        segment_job_meta_map.append(job_meta)
        _logger.debug(f"  Loaded {frames.shape[0]} frames. Shape: {frames.shape}")

    if not all_loaded_segments_frames:
        _logger.critical("No valid segments loaded after filtering/loading.")
        raise ValueError("No valid segments loaded after filtering/loading.")
    
    return all_loaded_segments_frames, segment_job_meta_map, determined_fps

def _align_segments_data(all_loaded_segments_frames: list, segment_job_meta_map: list, N_overlap: int, merge_alignment_method: str):
    if not all_loaded_segments_frames: return [] 
    if len(all_loaded_segments_frames) == 1: return all_loaded_segments_frames 

    _logger.debug("\n--- Pass 1.5: Aligning Segments ---")
    all_aligned_segments_frames = [all_loaded_segments_frames[0].astype(np.float32)] 
    _logger.debug(f"Segment 0 (ID {segment_job_meta_map[0]['segment_id']}) is baseline for alignment.")

    for idx in range(1, len(all_loaded_segments_frames)):
        current_raw = all_loaded_segments_frames[idx].astype(np.float32)
        prev_aligned = all_aligned_segments_frames[-1]
        current_id = segment_job_meta_map[idx]['segment_id']
        prev_id = segment_job_meta_map[idx-1]['segment_id']
        
        _logger.debug(f"Aligning segment (ID {current_id}) to previous (ID {prev_id}). Method: {merge_alignment_method}")
        aligned_current = current_raw.copy()

        if N_overlap > 0:
            target_raw_for_align = prev_aligned[-N_overlap:]
            pred_raw_for_align = current_raw[:N_overlap]
            eff_overlap = min(len(target_raw_for_align), len(pred_raw_for_align))

            if eff_overlap > 0:
                if merge_alignment_method.lower() == "shift_scale":
                    target_align_frames = target_raw_for_align[-eff_overlap:]
                    pred_align_frames = pred_raw_for_align[:eff_overlap]
                    mask = np.ones_like(pred_align_frames.reshape(-1), dtype=np.float32)
                    s, t = util.compute_scale_and_shift_full(
                        pred_align_frames.reshape(-1), 
                        target_align_frames.reshape(-1), 
                        mask
                    )
                    _logger.debug(f"Aligning segment {current_id} to {prev_id}. Method: shift_scale. Scale: {s:.4f}, Shift: {t:.4f}")
                    aligned_current = s * current_raw + t
                elif merge_alignment_method.lower() == "linear_blend":
                    _logger.debug(f"  Linear Blend: No explicit S&S alignment for segment ID {current_id}. Blending will occur in stitching.")
                else:
                    _logger.warning(f"CRITICAL WARNING: Unknown alignment method '{merge_alignment_method}'. No alignment performed on segment ID {current_id}.")
            else:
                _logger.debug(f"Warning: No actual overlap for alignment between {prev_id} and {current_id}. No S&S alignment performed.")
        else:
            _logger.debug(f"  N_overlap is 0. No explicit alignment for segment ID {current_id}.")
        all_aligned_segments_frames.append(aligned_current)
    return all_aligned_segments_frames

def _stitch_and_blend_segments_data(all_aligned_segments: list, segment_job_meta_map: list, N_overlap: int):
    if not all_aligned_segments: 
        _logger.critical("No aligned segments for stitching.")
        raise ValueError("No aligned segments for stitching.")

    _logger.debug("\n--- Pass 2: Stitching and Blending Segments ---")
    final_frames_list = []

    if N_overlap == 0:
        _logger.debug("  N_overlap is 0. Concatenating segments.")
        for i, segment_frames in enumerate(all_aligned_segments):
            seg_id = segment_job_meta_map[i]['segment_id']
            _logger.debug(f"Stitching segment {i} (ID: {seg_id}). Adding {len(segment_frames)} frames.")
            if len(segment_frames) > 0: final_frames_list.extend(list(segment_frames))
    else:
        for idx, current_segment_aligned in enumerate(all_aligned_segments):
            seg_id = segment_job_meta_map[idx]['segment_id']
            _logger.debug(f"Stitching segment {idx} (ID {seg_id}; {idx + 1}/{len(all_aligned_segments)})")
            
            if len(current_segment_aligned) == 0:
                _logger.warning(f"  Segment {idx} (ID {seg_id}) is empty after alignment. Skipping.")
                continue
            
            if idx == 0:
                frames_to_add_count = len(current_segment_aligned) - N_overlap if len(all_aligned_segments) > 1 else len(current_segment_aligned)
                frames_to_add_count = max(0, frames_to_add_count)
                _logger.debug(f"  First segment: adding {frames_to_add_count} non-overlapping frames.")
                if frames_to_add_count > 0:
                    final_frames_list.extend(list(current_segment_aligned[:frames_to_add_count]))
            else:
                prev_segment_aligned = all_aligned_segments[idx-1]
                prev_seg_id = segment_job_meta_map[idx-1]['segment_id']
                blend_pre_raw = prev_segment_aligned[-N_overlap:]
                blend_post_raw = current_segment_aligned[:N_overlap]
                eff_blend_len = min(len(blend_pre_raw), len(blend_post_raw))

                if eff_blend_len <= 0:
                    _logger.warning(f"  Warning: No frames for blending between {prev_seg_id} and {seg_id}. Hard cut implies adding all of current.")
                    if len(current_segment_aligned) > 0:
                        final_frames_list.extend(list(current_segment_aligned))
                else:
                    blend_pre_frames = list(blend_pre_raw[-eff_blend_len:])
                    blend_post_frames = list(blend_post_raw[:eff_blend_len])
                    _logger.debug(f"  Blending {eff_blend_len} frames with previous (ID {prev_seg_id}).")
                    
                    blended_frames = util.get_interpolate_frames(blend_pre_frames, blend_post_frames) if eff_blend_len > 1 else \
                                     [(0.5 * blend_pre_frames[0] + 0.5 * blend_post_frames[0])] if eff_blend_len == 1 else []
                    
                    if blended_frames:
                        final_frames_list.extend(blended_frames)

                    start_idx_for_remainder = eff_blend_len 
                    frames_after_blend_desc = ""
                    if idx == len(all_aligned_segments) - 1:
                        frames_after_blend = current_segment_aligned[start_idx_for_remainder:] if start_idx_for_remainder < len(current_segment_aligned) else []
                        frames_after_blend_desc = "last segment remaining"
                    else:
                        end_idx_for_remainder = max(start_idx_for_remainder, len(current_segment_aligned) - N_overlap)
                        frames_after_blend = current_segment_aligned[start_idx_for_remainder:end_idx_for_remainder] if start_idx_for_remainder < end_idx_for_remainder else []
                        frames_after_blend_desc = "intermediate segment middle"
                    
                    _logger.debug(f"    Adding {len(frames_after_blend)} {frames_after_blend_desc} frames after blend.")
                    if len(frames_after_blend) > 0:
                        final_frames_list.extend(list(frames_after_blend))
            
    if not final_frames_list:
        _logger.critical("No frames in final list after stitching.")
        raise ValueError("No frames in final list after stitching.")
    return np.array(final_frames_list, dtype=np.float32)

def _apply_mp4_postprocessing_refactored(
    video_normalized: np.ndarray, 
    apply_gamma: bool, 
    gamma_val: float, 
    do_dithering: bool, 
    dither_strength: float
):
    video_processed = video_normalized.copy()
    if apply_gamma:
        video_processed = apply_gamma_correction_to_video(video_processed, gamma_val)
    else:
        _logger.debug("Gamma correction disabled for MP4 output.")

    if do_dithering:
        video_processed = apply_dithering_to_video(video_processed, dither_strength)
    return video_processed

def _determine_output_path(
    out_path_arg: str,
    master_meta_p: str,
    original_basename_from_meta: str,
    out_format: str,
    final_filename_base_override: Optional[str] = None # This is the complete desired base filename (e.g., "myvideo_depth" or "myvideo_clipped_depth")
) -> str:
    """
    Determines the full output path, handling sequence folders, single files, and filename overrides.
    Ensures that final_filename_base_override takes precedence for the filename portion.

    Args:
        out_path_arg (str): User-provided output path. Can be a directory path or a specific file path.
        master_meta_p (str): Path to the master metadata file (used for default output directory).
        original_basename_from_meta (str): The original base name of the video.
        out_format (str): Desired output format (e.g., "mp4", "png_sequence").
        final_filename_base_override (Optional[str]): The complete desired base filename
                                                      (e.g., "myvideo_depth" or "myvideo_clipped_depth").
                                                      If None, `original_basename_from_meta` is used.

    Returns:
        str: The full, resolved output path.
    """
    output_path_final = None

    # Determine the effective base name to use for the output file/folder.
    # This `final_filename_base_override` already includes any suffixes (e.g., "_depth", "_clipped_depth").
    effective_base_filename = final_filename_base_override if final_filename_base_override else original_basename_from_meta

    # Determine the target directory for the output
    target_output_dir = None
    if out_path_arg:
        if os.path.isdir(out_path_arg): # If out_path_arg is an existing directory
            target_output_dir = out_path_arg
        elif not os.path.exists(out_path_arg) and "." not in os.path.basename(out_path_arg): # If it's a non-existing path without extension, assume directory
             target_output_dir = out_path_arg
        else: # out_path_arg is an existing file, or a new file path (contains an extension)
            target_output_dir = os.path.dirname(out_path_arg)
            if not target_output_dir: target_output_dir = "." # Fallback if only a filename was provided
    else: # No out_path_arg, use default based on master_meta_p
        target_output_dir = os.path.dirname(master_meta_p) if master_meta_p and os.path.dirname(master_meta_p) else "."
    
    os.makedirs(target_output_dir, exist_ok=True) # Ensure the target directory exists


    if out_format in ["png_sequence", "exr_sequence"]:
        # For sequences, the output path is a new subfolder within target_output_dir
        seq_type_suffix = "png_seq" if out_format == "png_sequence" else "exr_seq"
        subfolder_name = f"{effective_base_filename}_{seq_type_suffix}"
        
        # Auto-generate unique name if folder exists and final_filename_base_override was not explicitly set
        # This prevents overwriting previous auto-generated sequence folders unless explicitly named.
        constructed_seq_path = os.path.join(target_output_dir, subfolder_name)
        if not final_filename_base_override and os.path.exists(constructed_seq_path):
            ts = time.strftime("_%Y%m%d-%H%M%S")
            subfolder_name = f"{effective_base_filename}_{seq_type_suffix}{ts}"
            output_path_final = os.path.join(target_output_dir, subfolder_name)
        else:
            output_path_final = constructed_seq_path
        
        _logger.debug(f"  Sequence output resolved to: {output_path_final}")
        os.makedirs(output_path_final, exist_ok=True) # Ensure the sequence directory itself exists
        
    else: # Single file formats (mp4, exr)
        # Construct the filename using the effective_base_filename and the format's extension.
        extension = out_format.replace('main10_','') # Remove main10 for filename extension
        output_path_final = os.path.join(target_output_dir, f"{effective_base_filename}.{extension}")
        _logger.debug(f"  Single file output resolved to: {output_path_final}")

    if output_path_final is None:
        _logger.critical("Could not determine a valid output path.")
        raise ValueError("Could not determine a valid output path.")
    return output_path_final

def _save_output_to_disk(video_data: np.ndarray, save_path: str, out_format: str, fps_val: float):
    _logger.info(f"Saving merged output to: {save_path} (Format: {out_format}) FPS: {fps_val:.2f}")
    if video_data is None or video_data.size == 0:
        _logger.critical("Video data for saving is empty.")
        raise ValueError("Video data for saving is empty.")

    try:
        if out_format == "png_sequence":
            for i, frame_f in enumerate(video_data):
                frame_u16 = (np.clip(frame_f, 0, 1) * 65535.0).astype(np.uint16)
                imageio.imwrite(os.path.join(save_path, f"frame_{i:05d}.png"), frame_u16)
            _logger.debug(f"Successfully saved {len(video_data)} PNGs in {save_path}")
        elif out_format == "exr_sequence":
            if not _HAS_OPENEXR:
                _logger.warning("OpenEXR/Imath libraries not found. EXR sequence save skipped.")
                raise ImportError("OpenEXR/Imath missing for EXR sequence.")
            saved_count, failed_count = 0,0
            for i, frame_f in enumerate(video_data):
                try:
                    save_single_frame_exr(frame_f.astype(np.float32), os.path.join(save_path, f"frame_{i:05d}.exr"))
                    saved_count +=1
                except Exception as e_exr_frame:
                    _logger.error(f"  ERROR saving EXR frame {i}: {e_exr_frame}. Skipping.")
                    failed_count +=1
            _logger.info(f"Saved {saved_count}/{len(video_data)} EXRs in {save_path}")
            if failed_count > 0:
                _logger.warning(f"Warning: {failed_count} EXR frames failed to save.")
        elif out_format == "exr":
            if not _HAS_OPENEXR:
                _logger.warning("OpenEXR/Imath libraries not found. Single EXR save skipped.")
                raise ImportError("OpenEXR/Imath missing for single EXR.")
            if len(video_data) > 0:
                save_single_frame_exr(video_data[0].astype(np.float32), save_path)
                _logger.info(f"Saved first frame as single EXR: {save_path}")
            else:
                _logger.warning("No frames available to save as single EXR.")
        elif out_format == "mp4":
            dc_utils.save_video(video_data, save_path, fps=fps_val, output_format="mp4")
            _logger.debug(f"Successfully saved MP4: {save_path}")
        elif out_format == "main10_mp4":
            dc_utils.save_video(video_data, save_path, fps=fps_val, output_format="main10_mp4")
            _logger.debug(f"Successfully saved HEVC Main10 MP4: {save_path}")
        else:
            _logger.error(f"Unknown output format for saving: {out_format}")
            raise ValueError(f"Unknown output format for saving: {out_format}")
    except Exception as e_save_disk:
        _logger.critical(f"CRITICAL ERROR during final disk save: {e_save_disk}", exc_info=True)
        raise

def merge_depth_segments(
    master_meta_path: str,
    output_path_arg: str = None,
    do_dithering: bool = False,
    dither_strength_factor: float = 0.5,    
    apply_gamma_correction: bool = False,
    gamma_value: float = 1.5,
    use_percentile_norm: bool = False, # Standard normalization setting
    norm_low_percentile: float = 0.1,  # Standard normalization setting
    norm_high_percentile: float = 99.9, # Standard normalization setting
    output_format: str = "mp4",
    merge_alignment_method: str = "shift_scale",
    output_filename_override_base: Optional[str] = None,
    # --- NEW PARAMETERS FOR DUAL OUTPUT ROBUST NORMALIZATION ---
    enable_dual_output_robust_norm: bool = False,
    robust_norm_low_percentile: float = 0.1,
    robust_norm_high_percentile: float = 99.9,
    robust_norm_output_min: float = 0.0,
    robust_norm_output_max: float = 1.0,
    robust_output_suffix: str = "_robust_norm_depth",
    is_depth_far_black: bool = True # Assuming 0=far/black, 1=close/white for DepthCrafter visual outputs
) -> Optional[str]:
    
    _logger.info(f"Starting depth segment merging process... Format: {output_format}, Alignment: {merge_alignment_method}")
    _logger.debug(f"Merge Settings - Dithering: {do_dithering}, Strength: {dither_strength_factor}")
    _logger.debug(f"Merge Settings - Gamma: {apply_gamma_correction}, Value: {gamma_value}")
    _logger.debug(f"Merge Settings - Standard Percentile Norm: {use_percentile_norm}, Low: {norm_low_percentile}%, High: {norm_high_percentile}%")
    if enable_dual_output_robust_norm:
        _logger.debug(f"Merge Settings - Robust Dual Output Enabled. Low: {robust_norm_low_percentile}%, High: {robust_norm_high_percentile}%, Out Min: {robust_norm_output_min}, Out Max: {robust_norm_output_max}, Suffix: '{robust_output_suffix}'")


    final_video_unclipped = None
    final_fps = 30.0
    actual_saved_output_path = None # This will store the path of the *first* (standard) output

    try:
        meta_data, N_overlap, sorted_jobs, base_dir = _load_and_validate_metadata(master_meta_path)

        if len(sorted_jobs) == 1:
            _logger.debug("Processing as single segment (only one valid segment found).")
            final_video_unclipped, final_fps = _load_single_segment_frames(sorted_jobs[0], base_dir)
        else:
            _logger.debug(f"Processing {len(sorted_jobs)} segments.")
            loaded_frames_list, job_meta_map, initial_fps = _load_multiple_segments_data(sorted_jobs, base_dir)
            final_fps = initial_fps 

            if len(loaded_frames_list) == 1:
                 _logger.debug("Only one valid segment remained after loading. Using its frames directly.")
                 final_video_unclipped = loaded_frames_list[0]
            elif len(loaded_frames_list) > 1:
                aligned_segments = _align_segments_data(loaded_frames_list, job_meta_map, N_overlap, merge_alignment_method)
                final_video_unclipped = _stitch_and_blend_segments_data(aligned_segments, job_meta_map, N_overlap)
            else: 
                _logger.critical("No valid segments to process after loading stage.")
                raise ValueError("No valid segments to process after loading stage.")

        if final_video_unclipped is None or final_video_unclipped.size == 0:
            _logger.critical("Resulting video array is empty before normalization.")
            raise ValueError("Resulting video array is empty before normalization.")
        if final_fps is None or final_fps <= 0: 
            _logger.warning(f"Warning: Invalid FPS {final_fps}. Defaulting to 30.0.")
            final_fps = 30.0

        original_basename_from_meta = meta_data.get("original_video_basename", "merged_video")
        file_extension_for_path = "mp4" # Default for _determine_output_path suffix logic
        if output_format == "png_sequence":
            file_extension_for_path = "png_sequence"
        elif output_format == "exr_sequence":
            file_extension_for_path = "exr_sequence"
        elif output_format == "exr":
            file_extension_for_path = "exr"
        
        # --- FIRST OUTPUT: Standard Normalization & Save ---
        _logger.info("Generating standard output...")
        standard_normalized_video = normalize_video_data(
            final_video_unclipped, # Use original unclipped data
            use_percentile_norm,
            norm_low_percentile,
            norm_high_percentile
        )
        
        standard_video_to_save = standard_normalized_video
        if "mp4" in output_format.lower(): # Covers "mp4", "main10_mp4"
            standard_video_to_save = _apply_mp4_postprocessing_refactored(
                standard_normalized_video,
                apply_gamma_correction,
                gamma_value,
                do_dithering,
                dither_strength_factor
            )
        
        standard_output_path = _determine_output_path(
            output_path_arg,
            master_meta_path,
            original_basename_from_meta,
            file_extension_for_path,
            output_filename_override_base # Use the potentially overridden base for first output
        )
        
        _save_output_to_disk(standard_video_to_save, standard_output_path, output_format, final_fps)
        actual_saved_output_path = standard_output_path # This is the primary output path

        # --- SECOND OUTPUT (if enabled): Robust Global Normalization & Save ---
        if enable_dual_output_robust_norm:
            _logger.info("Generating robustly normalized (secondary) output...")
            robust_normalized_video = _apply_robust_global_normalization(
                final_video_unclipped, # Use original unclipped data again
                robust_norm_low_percentile,
                robust_norm_high_percentile,
                robust_norm_output_min,
                robust_norm_output_max,
                is_depth_far_black # Pass the convention explicitly
            )

            robust_video_to_save = robust_normalized_video
            if "mp4" in output_format.lower(): # Apply same post-processing if target is MP4
                robust_video_to_save = _apply_mp4_postprocessing_refactored(
                    robust_normalized_video,
                    apply_gamma_correction, # Use same gamma/dither settings as first output
                    gamma_value,
                    do_dithering,
                    dither_strength_factor
                )

            # Determine path for the second output with the unique suffix
            # Ensure the robust suffix is applied correctly even if override_base was used
            robust_output_base_name = f"{original_basename_from_meta}{robust_output_suffix}"
            if output_filename_override_base:
                # If output_filename_override_base was already given, append robust_output_suffix to it.
                # Example: If original was "video.mp4", override_base "_fancy_depth", suffix "_clipped_depth"
                # Result: "video_fancy_depth_clipped_depth.mp4"
                robust_output_base_name = f"{output_filename_override_base}{robust_output_suffix}"
            
            robust_output_path = _determine_output_path(
                output_path_arg,
                master_meta_path,
                original_basename_from_meta, # This is for determining the base directory, actual filename comes from robust_output_base_name
                file_extension_for_path,
                final_filename_base_override=robust_output_base_name # Pass the new base name for this second output
            )
            _save_output_to_disk(robust_video_to_save, robust_output_path, output_format, final_fps)

    except Exception as e:
        _logger.critical(f"CRITICAL ERROR during merge process: {e}", exc_info=True)
        # Re-raise to ensure GUI gets the error for its message queue and UI state
        raise
    
    _logger.info("Depth segment merging process finished successfully.")
    return actual_saved_output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge segmented depth map videos.")
    parser.add_argument("master_meta_path", type=str, help="Path to the _master_meta.json file.")
    parser.add_argument("--output_filename_override_base", type=str, default=None, help="Override the base for the output filename (e.g., 'myvideo_remerged').")
    parser.add_argument("--output_path", "-o", type=str, default=None, help="Output path for video or sequence directory.")
    parser.add_argument("--dither", action="store_true", help="Enable dithering for MP4.")
    parser.add_argument("--dither_strength", type=float, default=0.5, help="Dithering strength.")
    parser.add_argument("--percentile_norm", action="store_true", help="Use percentile clipping for normalization.")
    parser.add_argument("--norm_low_perc", type=float, default=0.1, help="Low percentile.")
    parser.add_argument("--norm_high_perc", type=float, default=99.9, help="High percentile.")
    output_format_choices = ["mp4", "main10_mp4", "png_sequence", "exr_sequence", "exr"]
    parser.add_argument("--output_format", type=str, default="mp4", choices=output_format_choices, help="Output format.")
    parser.add_argument("--apply_gamma", action="store_true", help="Enable gamma correction for MP4.")
    parser.add_argument("--gamma_value", type=float, default=1.5, help="Gamma value for MP4.")
    parser.add_argument("--merge_alignment_method", type=str, default="shift_scale", choices=["shift_scale", "linear_blend"], help="Segment alignment method.")
    
    # Map verbosity levels to logging module levels
    verbosity_choices = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
        "silent": logging.CRITICAL + 1 # Higher than critical to suppress all
    }
    parser.add_argument("--verbosity", type=str, default="info", choices=list(verbosity_choices.keys()), 
                        help=f"Console verbosity level. Default: info. Choices: {', '.join(verbosity_choices.keys())}")

    args = parser.parse_args()
    
    # Set console verbosity using logging.basicConfig
    logging.basicConfig(level=verbosity_choices[args.verbosity],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%H:%M:%S')

    try:
        merge_depth_segments(
            args.master_meta_path,
            output_path_arg=args.output_path,
            do_dithering=args.dither,
            dither_strength_factor=args.dither_strength,
            apply_gamma_correction=args.apply_gamma,
            gamma_value=args.gamma_value, 
            use_percentile_norm=args.percentile_norm,
            norm_low_percentile=args.norm_low_perc,
            norm_high_percentile=args.norm_high_perc,
            output_format=args.output_format,
            merge_alignment_method=args.merge_alignment_method,
            output_filename_override_base=args.output_filename_override_base
        )
    except Exception as e_main_call:
        _logger.critical(f"Unhandled script-level error: {e_main_call}", exc_info=True)
        sys.exit(1)