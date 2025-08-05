import argparse
import json # Keep for direct use if utils.load_json_file is not used everywhere (it is, mostly)
import os
import numpy as np
import shutil 
import sys
import time 
import message_catalog
from typing import Optional

# Import from the new message catalog
from message_catalog import (
    log_message,
    set_console_verbosity, # For __main__
    INFO, DEBUG, WARNING, ERROR, CRITICAL, # Severity levels
    VERBOSITY_LEVEL_INFO, VERBOSITY_LEVEL_DEBUG, VERBOSITY_LEVEL_SILENT # Verbosity levels for __main__
)

from depthcrafter import dav_util as util
from depthcrafter import utils as dc_utils
from depthcrafter.utils import (
    normalize_video_data,
    apply_gamma_correction_to_video,
    apply_dithering_to_video,
    load_json_file, # Uses global log_message
)
import imageio

_HAS_OPENEXR = False
try:
    import OpenEXR
    import Imath
    _HAS_OPENEXR = True
except ImportError:
    log_message("OPENEXR_UNAVAILABLE", context="merge_depth_segments.py")


def save_single_frame_exr(frame_data: np.ndarray, output_path: str):
    if not _HAS_OPENEXR:
        log_message("MERGE_SAVE_EXR_NO_LIB_ERROR") # New ID
        raise RuntimeError("OpenEXR/Imath libraries not found. Cannot save EXR.")
    if frame_data.ndim != 2:
        log_message("MERGE_SAVE_EXR_INVALID_DIMS_ERROR", shape=frame_data.shape) # New ID
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
    log_message("MERGE_METADATA_LOAD_ATTEMPT", path=master_meta_path) # New ID
    meta_data = load_json_file(master_meta_path) # Uses global log_message
    if not meta_data:
        # Error logged by load_json_file or a generic error if it raised something else
        raise FileNotFoundError(f"Failed to load or parse master metadata file: {master_meta_path}")

    if not meta_data.get("global_processing_settings", {}).get("processed_as_segments"):
        log_message("MERGE_METADATA_NOT_SEGMENTED_ERROR", path=master_meta_path) # New ID
        raise ValueError("'processed_as_segments' is not true in metadata.")

    global_settings = meta_data.get("global_processing_settings", {})
    N_overlap_from_meta = global_settings.get("segment_definition_output_overlap_frames")
    if N_overlap_from_meta is None:
        log_message("MERGE_METADATA_NO_OVERLAP_ERROR", path=master_meta_path) # New ID
        raise ValueError("'segment_definition_output_overlap_frames' not found.")
    log_message("MERGE_METADATA_OVERLAP_INFO", overlap=N_overlap_from_meta) # New ID

    jobs_info = meta_data.get("jobs_info", [])
    if not jobs_info:
        log_message("MERGE_METADATA_NO_JOBS_WARN", path=master_meta_path) # New ID
    
    successful_jobs_info = [job for job in jobs_info if job.get("status") == "success" and job.get("output_segment_filename")]
    if not successful_jobs_info:
        log_message("MERGE_METADATA_NO_SUCCESSFUL_JOBS_ERROR", path=master_meta_path) # New ID
        raise ValueError("No successful segments found in metadata to merge.")
        
    sorted_jobs_info = sorted(successful_jobs_info, key=lambda x: x.get("segment_id", -1))
    log_message("MERGE_METADATA_SUCCESSFUL_JOBS_COUNT", count=len(sorted_jobs_info)) # New ID
    base_dir = os.path.dirname(master_meta_path) if master_meta_path and os.path.dirname(master_meta_path) else "."
    return meta_data, N_overlap_from_meta, sorted_jobs_info, base_dir

def _load_single_segment_frames(job_meta: dict, base_dir: str):
    segment_filename = job_meta.get("output_segment_filename")
    input_segment_format = job_meta.get("output_segment_format", "npz").lower()
    processed_fps_from_meta = job_meta.get("processed_at_fps")
    segment_path = os.path.join(base_dir, segment_filename)

    if not os.path.exists(segment_path):
        log_message("FILE_NOT_FOUND", filepath=segment_path)
        raise FileNotFoundError(f"Segment file not found: {segment_path}.")

    if input_segment_format != "npz":
        log_message("MERGE_SEGMENT_UNSUPPORTED_FORMAT_ERROR", format=input_segment_format, filename=segment_filename) # New ID
        raise ValueError(f"Unsupported segment format '{input_segment_format}'. Expecting NPZ.")
    
    try:
        with np.load(segment_path) as data:
            if 'frames' not in data.files:
                log_message("NPZ_LOAD_KEY_ERROR", key='frames', filepath=segment_path)
                raise KeyError(f"Key 'frames' not found in NPZ: {segment_path}.")
            frames = data['frames']
    except Exception as e:
        log_message("MERGE_SEGMENT_LOAD_NPZ_ERROR", filepath=segment_path, error=str(e)) # New ID
        raise

    if frames is None or frames.size == 0:
        log_message("MERGE_SEGMENT_EMPTY_ERROR", filename=segment_filename) # New ID
        raise ValueError(f"Segment {segment_filename} is empty.")

    fps = float(processed_fps_from_meta) if processed_fps_from_meta else 30.0
    log_message("MERGE_SINGLE_SEGMENT_LOAD_INFO", filename=segment_filename, shape=frames.shape, fps=fps) # New ID
    return frames.astype(np.float32), fps

def _load_multiple_segments_data(sorted_jobs_info: list, base_dir: str):
    log_message("MERGE_PASS_LOAD_SEGMENTS_START") # New ID
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
            log_message("MERGE_SEGMENT_MISSING_FPS_ERROR", segment_id=segment_id) # New ID
            raise ValueError(f"'processed_at_fps' missing for segment ID {segment_id}.")
        
        current_fps = float(processed_fps_from_meta)
        if determined_fps is None:
            determined_fps = current_fps
        elif abs(determined_fps - current_fps) > 1e-3:
            log_message("MERGE_SEGMENT_INCONSISTENT_FPS_WARN", 
                        expected_fps=determined_fps, segment_id=segment_id, actual_fps=current_fps) # New ID

        log_message("MERGE_SEGMENT_LOADING_PROGRESS", segment_id=segment_id, current=idx+1, 
                    total=len(sorted_jobs_info), filename=segment_filename) # New ID
        if input_segment_format != "npz":
            log_message("MERGE_SEGMENT_UNSUPPORTED_FORMAT_ERROR", format=input_segment_format, filename=segment_filename)
            raise ValueError(f"Unsupported segment format '{input_segment_format}' for {segment_filename}.")
        
        try:
            with np.load(segment_path) as data:
                if 'frames' not in data.files:
                    log_message("NPZ_LOAD_KEY_ERROR", key='frames', filepath=segment_path)
                    raise KeyError(f"Key 'frames' not found in NPZ: {segment_path}.")
                frames = data['frames']
        except Exception as e:
            log_message("MERGE_SEGMENT_LOAD_NPZ_ERROR", filepath=segment_path, error=str(e))
            raise

        if frames is None or frames.size == 0:
            log_message("MERGE_SEGMENT_DATA_EMPTY_WARN", segment_id=segment_id, filename=segment_filename) # New ID
            continue
        
        all_loaded_segments_frames.append(frames.astype(np.float32).copy())
        segment_job_meta_map.append(job_meta)
        log_message("MERGE_SEGMENT_LOADED_FRAMES_INFO", num_frames=frames.shape[0], shape=frames.shape) # New ID

    if not all_loaded_segments_frames:
        log_message("MERGE_NO_VALID_SEGMENTS_LOADED_ERROR") # New ID
        raise ValueError("No valid segments loaded after filtering/loading.")
    
    return all_loaded_segments_frames, segment_job_meta_map, determined_fps


def _align_segments_data(all_loaded_segments_frames: list, segment_job_meta_map: list, N_overlap: int, merge_alignment_method: str):
    if not all_loaded_segments_frames: return [] 
    if len(all_loaded_segments_frames) == 1: return all_loaded_segments_frames 

    log_message("MERGE_PASS_ALIGN_SEGMENTS_START") # New ID
    all_aligned_segments_frames = [all_loaded_segments_frames[0].astype(np.float32)] 
    log_message("MERGE_ALIGN_BASELINE_INFO", segment_id=segment_job_meta_map[0]['segment_id']) # New ID

    for idx in range(1, len(all_loaded_segments_frames)):
        current_raw = all_loaded_segments_frames[idx].astype(np.float32)
        prev_aligned = all_aligned_segments_frames[-1]
        current_id = segment_job_meta_map[idx]['segment_id']
        prev_id = segment_job_meta_map[idx-1]['segment_id']
        
        log_message("MERGE_ALIGN_PROGRESS", current_id=current_id, prev_id=prev_id, method=merge_alignment_method) # New ID
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
                    log_message("MERGE_ALIGN_INFO", current_id=current_id, prev_id=prev_id, method="shift_scale", scale=s, t=t) # Using existing ID
                    aligned_current = s * current_raw + t
                elif merge_alignment_method.lower() == "linear_blend":
                    log_message("MERGE_ALIGN_LINEAR_BLEND_SKIP_S_S", segment_id=current_id) # New ID
                else:
                    log_message("MERGE_ALIGN_UNKNOWN_METHOD_WARN", method=merge_alignment_method, segment_id=current_id) # New ID
            else:
                log_message("MERGE_NO_OVERLAP_ALIGN", prev_id=prev_id, current_id=current_id) # Using existing ID
        else:
            log_message("MERGE_ALIGN_NO_OVERLAP_SPECIFIED", segment_id=current_id) # New ID
        all_aligned_segments_frames.append(aligned_current)
    return all_aligned_segments_frames


def _stitch_and_blend_segments_data(all_aligned_segments: list, segment_job_meta_map: list, N_overlap: int):
    if not all_aligned_segments: 
        log_message("MERGE_STITCH_NO_ALIGNED_SEGMENTS_ERROR") # New ID
        raise ValueError("No aligned segments for stitching.")

    log_message("MERGE_PASS_STITCH_BLEND_START") # New ID
    final_frames_list = []

    if N_overlap == 0:
        log_message("MERGE_STITCH_CONCATENATING_INFO") # New ID
        for i, segment_frames in enumerate(all_aligned_segments):
            seg_id = segment_job_meta_map[i]['segment_id']
            log_message("MERGE_STITCH_INFO", idx=i, seg_id=seg_id, count=len(segment_frames)) # Using existing ID
            if len(segment_frames) > 0: final_frames_list.extend(list(segment_frames))
    else:
        for idx, current_segment_aligned in enumerate(all_aligned_segments):
            seg_id = segment_job_meta_map[idx]['segment_id']
            log_message("MERGE_STITCH_PROGRESS", 
                        current_segment_idx=idx, 
                        current_segment_idx_plus_1=idx + 1, # <<< ADD THIS LINE
                        total_segments=len(all_aligned_segments), 
                        segment_id=seg_id)
            
            if len(current_segment_aligned) == 0:
                log_message("MERGE_STITCH_SEGMENT_EMPTY_WARN", segment_idx=idx, segment_id=seg_id) # New ID
                continue
            
            if idx == 0:
                frames_to_add_count = len(current_segment_aligned) - N_overlap if len(all_aligned_segments) > 1 else len(current_segment_aligned)
                frames_to_add_count = max(0, frames_to_add_count)
                log_message("MERGE_STITCH_FIRST_SEGMENT_INFO", count=frames_to_add_count) # New ID
                if frames_to_add_count > 0:
                    final_frames_list.extend(list(current_segment_aligned[:frames_to_add_count]))
            else:
                prev_segment_aligned = all_aligned_segments[idx-1]
                prev_seg_id = segment_job_meta_map[idx-1]['segment_id']
                blend_pre_raw = prev_segment_aligned[-N_overlap:]
                blend_post_raw = current_segment_aligned[:N_overlap]
                eff_blend_len = min(len(blend_pre_raw), len(blend_post_raw))

                if eff_blend_len <= 0:
                    log_message("MERGE_STITCH_NO_BLEND_FRAMES_WARN", prev_id=prev_seg_id, current_id=seg_id) # New ID
                    if len(current_segment_aligned) > 0:
                        final_frames_list.extend(list(current_segment_aligned))
                else:
                    blend_pre_frames = list(blend_pre_raw[-eff_blend_len:])
                    blend_post_frames = list(blend_post_raw[:eff_blend_len])
                    log_message("MERGE_STITCH_BLENDING_INFO", count=eff_blend_len, prev_id=prev_seg_id) # New ID
                    
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
                    
                    log_message("MERGE_STITCH_ADDING_REMAINDER", description=frames_after_blend_desc, count=len(frames_after_blend)) # New ID
                    if len(frames_after_blend) > 0:
                        final_frames_list.extend(list(frames_after_blend))
            
    if not final_frames_list:
        log_message("MERGE_STITCH_NO_FINAL_FRAMES_ERROR") # New ID
        raise ValueError("No frames in final list after stitching.")
    return np.array(final_frames_list, dtype=np.float32)


def _apply_mp4_postprocessing_refactored(
    video_normalized: np.ndarray, 
    apply_gamma: bool, 
    gamma_val: float, 
    do_dither: bool, 
    dither_strength: float
):
    video_processed = video_normalized.copy()
    if apply_gamma:
        video_processed = apply_gamma_correction_to_video(video_processed, gamma_val) # log_func removed from util
    else:
        log_message("MERGE_POSTPROC_GAMMA_DISABLED") # New ID

    if do_dither:
        video_processed = apply_dithering_to_video(video_processed, dither_strength) # log_func removed from util
    # else not needed as util handles logging for application
    return video_processed


def _determine_output_path(
    out_path_arg: str, 
    master_meta_p: str, 
    original_basename_from_meta: str,
    out_format: str,
    output_filename_override_base: Optional[str] = None
) -> str:
    output_path_final = None
    current_base_for_naming = output_filename_override_base if output_filename_override_base else original_basename_from_meta

    if out_path_arg:
        path_is_dir = os.path.isdir(out_path_arg)
        if out_format in ["png_sequence", "exr_sequence"]:
            base_dir_for_sequence = out_path_arg
            seq_suffix = "png_seq" if out_format == "png_sequence" else "exr_seq"
            if not path_is_dir and (out_path_arg.endswith(seq_suffix) or not os.path.splitext(out_path_arg)[1]):
                 output_path_final = out_path_arg
            else:
                base_dir_for_sequence = out_path_arg if path_is_dir else os.path.dirname(out_path_arg)
                if not base_dir_for_sequence : base_dir_for_sequence = "."
                subfolder_name = f"{current_base_for_naming}_{seq_suffix}"
                if not output_filename_override_base and os.path.exists(os.path.join(base_dir_for_sequence, subfolder_name)):
                    ts = time.strftime("_%Y%m%d-%H%M%S")
                    subfolder_name = f"{current_base_for_naming}_{seq_suffix}{ts}"
                output_path_final = os.path.join(base_dir_for_sequence, subfolder_name)
            log_message("MERGE_OUTPUT_PATH_SEQ_RESOLVED", path=output_path_final) # New ID
            os.makedirs(output_path_final, exist_ok=True)
        else: # Single file formats
            if path_is_dir:
                fname = f"{current_base_for_naming}.{out_format}"
                output_path_final = os.path.join(out_path_arg, fname)
                os.makedirs(out_path_arg, exist_ok=True)
            else:
                output_path_final = out_path_arg
                parent_dir = os.path.dirname(output_path_final)
                if parent_dir: os.makedirs(parent_dir, exist_ok=True)
            log_message("MERGE_OUTPUT_PATH_FILE_RESOLVED", path=output_path_final) # New ID
            if os.path.exists(output_path_final) and not path_is_dir:
                 log_message("MERGE_OUTPUT_PATH_FILE_OVERWRITE_WARN", path=output_path_final) # New ID
    else: # Auto-generate path
        meta_dir = os.path.dirname(master_meta_p) if master_meta_p and os.path.dirname(master_meta_p) else "."
        os.makedirs(meta_dir, exist_ok=True)
        if out_format in ["png_sequence", "exr_sequence"]:
            seq_type_name = 'png' if 'png' in out_format else 'exr'
            seq_folder_base_name = f"{current_base_for_naming}_{seq_type_name}_seq"
            output_path_final = os.path.join(meta_dir, seq_folder_base_name)
            if os.path.exists(output_path_final):
                unique_suffix = time.strftime("_%Y%m%d%H%M%S")
                output_path_final = os.path.join(meta_dir, f"{seq_folder_base_name}{unique_suffix}")
            os.makedirs(output_path_final, exist_ok=True)
        else: # Single file
            output_path_final = os.path.join(meta_dir, f"{current_base_for_naming}.{out_format}")
            if os.path.exists(output_path_final):
                log_message("MERGE_OUTPUT_PATH_AUTOGEN_OVERWRITE_WARN", path=output_path_final) # New ID
        log_message("MERGE_OUTPUT_PATH_AUTOGEN_INFO", path=output_path_final) # New ID
    
    if output_path_final is None:
        log_message("MERGE_OUTPUT_PATH_DETERMINE_ERROR") # New ID
        raise ValueError("Could not determine a valid output path.")
    return output_path_final

def _save_output_to_disk(video_data: np.ndarray, save_path: str, out_format: str, fps_val: float):
    log_message("MERGE_SAVING_TO_DISK_START", path=save_path, format=out_format, fps=fps_val)
    if video_data is None or video_data.size == 0:
        log_message("MERGE_SAVE_EMPTY_DATA_ERROR")
        raise ValueError("Video data for saving is empty.")

    try:
        if out_format == "png_sequence":
            for i, frame_f in enumerate(video_data):
                frame_u16 = (np.clip(frame_f, 0, 1) * 65535.0).astype(np.uint16)
                imageio.imwrite(os.path.join(save_path, f"frame_{i:05d}.png"), frame_u16)
            log_message("MERGE_SAVE_PNG_SEQ_SUCCESS", count=len(video_data), path=save_path)
        elif out_format == "exr_sequence":
            if not _HAS_OPENEXR:
                log_message("OPENEXR_UNAVAILABLE", context="_save_output_to_disk (exr_sequence)")
                raise ImportError("OpenEXR/Imath missing for EXR sequence.")
            saved_count, failed_count = 0,0
            for i, frame_f in enumerate(video_data):
                try:
                    save_single_frame_exr(frame_f.astype(np.float32), os.path.join(save_path, f"frame_{i:05d}.exr"))
                    saved_count +=1
                except Exception as e_exr_frame:
                    log_message("MERGE_SAVE_EXR_FRAME_ERROR", frame_num=i, error=str(e_exr_frame))
                    failed_count +=1
            log_message("MERGE_SAVE_EXR_SEQ_SUMMARY", saved=saved_count, total=len(video_data), path=save_path)
            if failed_count > 0:
                log_message("MERGE_SAVE_EXR_SEQ_FAILURES_WARN", failed_count=failed_count)
        elif out_format == "exr":
            if not _HAS_OPENEXR:
                log_message("OPENEXR_UNAVAILABLE", context="_save_output_to_disk (single exr)")
                raise ImportError("OpenEXR/Imath missing for single EXR.")
            if len(video_data) > 0:
                save_single_frame_exr(video_data[0].astype(np.float32), save_path)
                log_message("MERGE_SAVE_SINGLE_EXR_SUCCESS", path=save_path)
            else:
                log_message("MERGE_SAVE_SINGLE_EXR_NO_FRAMES_WARN")
        # START MODIFICATION for _save_output_to_disk
        elif out_format == "mp4": # Standard H.264 8-bit MP4
            dc_utils.save_video(video_data, save_path, fps=fps_val, output_format="mp4") # Pass explicit format
            log_message("MERGE_SAVE_MP4_SUCCESS", path=save_path)
        elif out_format == "main10_mp4": # HEVC (H.265) 10-bit MP4
            # Ensure the save_path has .mp4 extension, _determine_output_path should handle this.
            dc_utils.save_video(video_data, save_path, fps=fps_val, output_format="main10_mp4")
            log_message("MERGE_SAVE_HEVC_MAIN10_MP4_SUCCESS", path=save_path) # New specific log ID
        # END MODIFICATION
        else:
            log_message("MERGE_SAVE_UNKNOWN_FORMAT_ERROR", format=out_format)
            raise ValueError(f"Unknown output format for saving: {out_format}") # This was your error point
    except Exception as e_save_disk:
        log_message("MERGE_SAVE_DISK_ERROR_CRITICAL", error=str(e_save_disk), traceback_info=sys.exc_info()) # Use traceback_info for new catalog
        raise

def merge_depth_segments(
    master_meta_path: str,
    output_path_arg: str = None,
    do_dithering: bool = False,
    dither_strength_factor: float = 0.5,    
    apply_gamma_correction: bool = False,
    gamma_value: float = 1.5,
    use_percentile_norm: bool = False,
    norm_low_percentile: float = 0.1,
    norm_high_percentile: float = 99.9,
    output_format: str = "mp4",
    merge_alignment_method: str = "shift_scale",
    # script_caller_silence_level: int = None, # Removed
    output_filename_override_base: Optional[str] = None
) -> Optional[str]:
    
    # Verbosity is now set globally via message_catalog.set_console_verbosity (e.g. in __main__)
    # or message_catalog.set_gui_verbosity (e.g. from GUI code)

    log_message("MERGE_PROCESS_START", format=output_format, alignment=merge_alignment_method) # New ID
    log_message("MERGE_PROCESS_SETTINGS_DITHER", enabled=do_dithering, strength=dither_strength_factor) # New ID
    log_message("MERGE_PROCESS_SETTINGS_GAMMA", enabled=apply_gamma_correction, value=gamma_value) # New ID
    log_message("MERGE_PROCESS_SETTINGS_NORM", enabled=use_percentile_norm, low=norm_low_percentile, high=norm_high_percentile) # New ID

    final_video_unclipped = None
    final_fps = 30.0
    actual_saved_output_path = None

    try:
        meta_data, N_overlap, sorted_jobs, base_dir = _load_and_validate_metadata(master_meta_path)

        if len(sorted_jobs) == 1:
            log_message("MERGE_PROCESSING_SINGLE_SEGMENT_INFO") # New ID
            final_video_unclipped, final_fps = _load_single_segment_frames(sorted_jobs[0], base_dir)
        else:
            log_message("MERGE_PROCESSING_MULTIPLE_SEGMENTS_INFO", count=len(sorted_jobs)) # New ID
            loaded_frames_list, job_meta_map, initial_fps = _load_multiple_segments_data(sorted_jobs, base_dir)
            final_fps = initial_fps 

            if len(loaded_frames_list) == 1:
                 log_message("MERGE_ONE_VALID_SEGMENT_REMAINED_INFO") # New ID
                 final_video_unclipped = loaded_frames_list[0]
            elif len(loaded_frames_list) > 1:
                aligned_segments = _align_segments_data(loaded_frames_list, job_meta_map, N_overlap, merge_alignment_method)
                final_video_unclipped = _stitch_and_blend_segments_data(aligned_segments, job_meta_map, N_overlap)
            else: 
                log_message("MERGE_NO_VALID_SEGMENTS_FOR_STITCH_ERROR") # New ID
                raise ValueError("No valid segments to process after loading stage.")

        if final_video_unclipped is None or final_video_unclipped.size == 0:
            log_message("MERGE_FINAL_VIDEO_EMPTY_PRE_NORM_ERROR") # New ID
            raise ValueError("Resulting video array is empty before normalization.")
        if final_fps is None or final_fps <= 0: 
            log_message("MERGE_INVALID_FPS_WARN", fps=final_fps) # New ID
            final_fps = 30.0

        normalized_video = normalize_video_data(
            final_video_unclipped, 
            use_percentile_norm, 
            norm_low_percentile, 
            norm_high_percentile
            # log_func removed from util
        )
        
        video_to_save = normalized_video
        if "mp4" in output_format.lower(): # Covers "mp4", "main10_mp4"
            video_to_save = _apply_mp4_postprocessing_refactored(
                normalized_video,
                apply_gamma_correction,
                gamma_value,
                do_dithering,
                dither_strength_factor
            )
        else:
            log_message("MERGE_POSTPROC_SKIPPED_NON_MP4", format=output_format)
        
        log_message("MERGE_FINAL_VIDEO_STATS_INFO", shape=video_to_save.shape, dtype=str(video_to_save.dtype), 
                    min_val=video_to_save.min(), max_val=video_to_save.max()) # New ID

        original_basename_from_meta = meta_data.get("original_video_basename", "merged_video")
        file_extension_for_path = "mp4" # Default for our video formats
        if output_format == "png_sequence":
            file_extension_for_path = "png_sequence" # Special case for _determine_output_path
        elif output_format == "exr_sequence":
            file_extension_for_path = "exr_sequence" # Special case for _determine_output_path
        elif output_format == "exr":
            file_extension_for_path = "exr"
        # For "mp4", "mp4_main10", "main10_mp4", the file_extension_for_path remains "mp4"

        actual_saved_output_path = _determine_output_path(
            output_path_arg,
            master_meta_path,
            original_basename_from_meta,
            file_extension_for_path, # Pass the simple, standard extension here
            output_filename_override_base
        )
        
        _save_output_to_disk(video_to_save, actual_saved_output_path, output_format, final_fps)

    except Exception as e:
        # Log with CRITICAL and ensure the exception is re-raised for __main__ or GUI to handle
        log_message("MERGE_PROCESS_CRITICAL_ERROR", error=str(e), traceback=sys.exc_info()) # New ID
        raise
    
    log_message("MERGE_PROCESS_FINISHED_SUCCESS") # New ID
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
    
    # Map verbosity levels from message_catalog to choices
    verbosity_choices_map = {
        "debug": message_catalog.VERBOSITY_LEVEL_DEBUG,      # 10
        "detail": message_catalog.VERBOSITY_LEVEL_DETAIL,    # 15
        "info": message_catalog.VERBOSITY_LEVEL_INFO,        # 20 (Normal)
        "warning": message_catalog.VERBOSITY_LEVEL_WARNING,  # 30
        "error": message_catalog.VERBOSITY_LEVEL_ERROR,      # 40
        "critical": message_catalog.VERBOSITY_LEVEL_CRITICAL,# 50
        "silent": message_catalog.VERBOSITY_LEVEL_SILENT     # 100
    }
    parser.add_argument("--verbosity", type=str, default="info", choices=list(verbosity_choices.keys()), 
                        help=f"Console verbosity level. Default: info.")

    args = parser.parse_args()
    
    # Set console verbosity using the new system
    message_catalog.set_console_verbosity(verbosity_choices[args.verbosity])
    
    # No GUI logger callback when run as a script
    # message_catalog.set_gui_logger_callback(None) # Already default

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
            # script_caller_silence_level removed
            output_filename_override_base=args.output_filename_override_base
        )
    except Exception as e_main_call:
        # The critical error should have been logged by merge_depth_segments itself.
        # This is a fallback print if it wasn't or for truly unhandled cases.
        print(f"Unhandled script-level error: {e_main_call}", file=sys.stderr)
        # import traceback # Already imported in merge_depth_segments exception handler
        # traceback.print_exc()
        sys.exit(1)