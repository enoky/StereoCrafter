from typing import Union, List, Optional, Callable, Tuple
import tempfile
import numpy as np
import PIL.Image
# import matplotlib.cm as cm # No longer directly used here, ColorMapper will import it
import mediapy # Ensure mediapy is installed: pip install mediapy
import torch
from decord import VideoReader, cpu # Ensure decord is installed: pip install decord
import os
import shutil
import imageio # Added, as it's used for PNG/EXR saving
import time # Added for get_formatted_timestamp (though message_catalog has its own)
import json # Added for JSON utilities
import gc # Added for define_video_segments
import glob # For read_image_sequence_as_frames

# Import from the new message catalog
from message_catalog import (
    log_message,
    INFO, DEBUG, WARNING, ERROR, CRITICAL # For direct level checks if ever needed
)

dataset_res_dict = {
    "sintel": [448, 1024],
    "scannet": [640, 832],
    "KITTI": [384, 1280],
    "bonn": [512, 640],
    "NYUv2": [448, 640],
}


try:
    import OpenEXR
    import Imath
    _OPENEXR_AVAILABLE_IN_UTILS = True
except ImportError:
    _OPENEXR_AVAILABLE_IN_UTILS = False
    log_message("OPENEXR_UNAVAILABLE", context="utils.py") # Log using the new system

DEFAULT_SINGLE_IMAGE_CLIP_FRAMES = 5
if DEFAULT_SINGLE_IMAGE_CLIP_FRAMES <= 0:
    DEFAULT_SINGLE_IMAGE_CLIP_FRAMES = 1 # Safety fallback
    log_message("UTIL_WARN_INVALID_SINGLE_IMG_CONST", requested=DEFAULT_SINGLE_IMAGE_CLIP_FRAMES, fallback=1)

# --- NEW UTILITY FUNCTIONS ---

def format_duration(seconds: float) -> str:
    """Converts seconds to H:MM:SS.s format."""
    if seconds < 0:
        return "0:00:00.0"
    
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    return f"{hours}:{minutes:02}:{seconds:04.1f}"

def get_formatted_timestamp_utils() -> str: # Renamed to avoid clash with message_catalog's internal one
    """Generates a timestamp string in HH:MM:SS.s format for use within utils if needed directly."""
    current_time_val = time.time()
    time_struct = time.localtime(current_time_val)
    milliseconds_tenths = int((current_time_val - int(current_time_val)) * 10)
    return f"{time_struct.tm_hour:02d}:{time_struct.tm_min:02d}:{time_struct.tm_sec:02d}.{milliseconds_tenths}"

def get_segment_output_folder_name(original_video_basename: str) -> str:
    """Returns the standard name for a segment subfolder."""
    return f"{original_video_basename}_seg"

def get_segment_npz_output_filename(original_video_basename: str, segment_id: int, total_segments: int) -> str:
    """Returns the standard NPZ filename for a segment."""
    return f"{original_video_basename}_depth_{segment_id + 1}of{total_segments}.npz"

def get_full_video_output_filename(original_video_basename: str, extension: str = "mp4") -> str:
    """Returns the standard filename for a full video output."""
    return f"{original_video_basename}_depth.{extension}"

def get_image_sequence_metadata(folder_path: str, target_fps_from_gui: int) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int]]:
    """Gets metadata (frame count, fps, H, W) for an image sequence."""
    supported_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".exr")
    # Use glob for better pattern matching and case-insensitivity if needed, or stick to listdir for simplicity
    frames_found = []
    for ext in supported_exts:
        frames_found.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        frames_found.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}"))) # For case-insensitivity

    frames = sorted(list(set(frames_found))) # Remove duplicates and sort

    if not frames:
        log_message("IMAGE_SEQUENCE_NO_FRAMES_FOUND", folder_path=folder_path, extensions=str(supported_exts))
        return None, None, None, None

    try:
        first_frame_img = imageio.v2.imread(frames[0])
        h, w = first_frame_img.shape[:2]
    except Exception as e:
        log_message("IMAGE_READ_ERROR", filepath=frames[0], error=str(e))
        return None, None, None, None

    total_frames = len(frames)
    effective_fps = float(target_fps_from_gui) if target_fps_from_gui != -1 else 24.0
    log_message("IMAGE_SEQUENCE_METADATA_SUCCESS", folder_path=folder_path, count=total_frames, fps=effective_fps, height=h, width=w)
    return total_frames, effective_fps, h, w

def get_single_image_metadata(
    image_path: str, 
    target_fps_from_gui: int
    # Removed num_frames_to_generate_for_clip parameter
) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int]]:
    """Gets metadata for a single image. Frame count uses DEFAULT_SINGLE_IMAGE_CLIP_FRAMES."""
    try:
        img = imageio.v2.imread(image_path) # Or your preferred image loading
        h, w = img.shape[:2]
    except Exception as e:
        log_message("IMAGE_READ_ERROR", filepath=image_path, error=str(e))
        return None, None, None, None

    effective_fps = float(target_fps_from_gui) if target_fps_from_gui != -1 else 24.0
    
    # Use the globally defined constant from this utils.py file
    num_generated_frames = DEFAULT_SINGLE_IMAGE_CLIP_FRAMES

    log_message("SINGLE_IMAGE_METADATA_SUCCESS", image_path=image_path, 
                frames_for_clip=num_generated_frames,
                fps=effective_fps, height=h, width=w,
                source_setting_is_constant=True) # Indicate it's from constant
    return num_generated_frames, effective_fps, h, w
def get_sidecar_json_filename(base_filepath_with_ext: str) -> str:
    """Returns the corresponding .json sidecar filename for a given base file."""
    return os.path.splitext(base_filepath_with_ext)[0] + ".json"


def define_video_segments(
    video_path_or_folder: str,
    original_basename: str,
    gui_target_fps_setting: int,
    gui_process_length_overall: int,
    gui_segment_output_window_frames: int,
    gui_segment_output_overlap_frames: int,
    source_type: str # This parameter is still needed
    # Removed num_frames_for_single_image_clip parameter
) -> Tuple[List[dict], Optional[dict]]:
    """
    Defines video segments based on input parameters.
    Handles different source types: video files, image sequence folders, and single image files.
    For single image files, uses DEFAULT_SINGLE_IMAGE_CLIP_FRAMES from this module.
    """
    segment_jobs = []
    base_job_info_for_video = {} 

    total_raw_frames_in_original_video = 0
    original_video_fps = 30.0

    if source_type == "video_file":
        try:
            vr = VideoReader(video_path_or_folder, ctx=cpu(0))
            total_raw_frames_in_original_video = len(vr)
            original_video_fps = vr.get_avg_fps()
            del vr
            gc.collect()
            if original_video_fps <= 0:
                log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Invalid original FPS ({original_video_fps}) for video. Assuming 30 FPS.")
                original_video_fps = 30.0
        except Exception as e:
            log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Error getting metadata of video file {video_path_or_folder}: {e}")
            return [], None
    elif source_type == "image_sequence_folder":
        count, fps, h, w = get_image_sequence_metadata(video_path_or_folder, gui_target_fps_setting)
        if count is None:
            # ... (error handling) ...
            return [], None
        total_raw_frames_in_original_video = count
        original_video_fps = fps
    elif source_type == "single_image_file":
        # get_single_image_metadata now uses the internal DEFAULT_SINGLE_IMAGE_CLIP_FRAMES
        count, fps, h, w = get_single_image_metadata( 
            video_path_or_folder, 
            gui_target_fps_setting
        )
        if count is None:
            # ... (error handling) ...
            return [], None
        total_raw_frames_in_original_video = count
        original_video_fps = fps
    else:
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Unknown source_type: {source_type}")
        return [], None

    base_job_info_for_video = {
        "video_path": video_path_or_folder,
        "source_type": source_type, 
        "gui_fps_setting_at_definition": gui_target_fps_setting,
        "original_basename": original_basename,
        "original_video_raw_frame_count": total_raw_frames_in_original_video,
        "original_video_fps": original_video_fps,
        # No longer need to store num_frames_generated_for_single_image_clip_at_definition here,
        # as it's now a fixed constant within utils.py (though get_single_image_metadata logs it)
    }

    fps_for_stride_calc = original_video_fps
    if source_type == "video_file": # Only for video files might the GUI target FPS differ for stride
        fps_for_stride_calc = original_video_fps if gui_target_fps_setting == -1 else gui_target_fps_setting
    
    if fps_for_stride_calc <= 0:
        fps_for_stride_calc = original_video_fps if original_video_fps > 0 else 24.0
    
    stride_for_fps_adjustment = 1 # Default for image sequences / single images
    if source_type == "video_file" and original_video_fps > 0 and fps_for_stride_calc > 0 :
        stride_for_fps_adjustment = max(round(original_video_fps / fps_for_stride_calc), 1)
    elif source_type != "video_file":
        # For image sequences or single images, the frames are already at the "target" rate
        # or generated as such. Stride is effectively 1.
        pass # Stride remains 1


    max_possible_output_frames_after_fps = (total_raw_frames_in_original_video + stride_for_fps_adjustment - 1) // stride_for_fps_adjustment
    
    effective_total_output_frames_to_target_for_video = max_possible_output_frames_after_fps
    if gui_process_length_overall != -1 and gui_process_length_overall < effective_total_output_frames_to_target_for_video:
        effective_total_output_frames_to_target_for_video = gui_process_length_overall
    
    if effective_total_output_frames_to_target_for_video <= 0:
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason="Effective output frames is zero or less.")
        return [], base_job_info_for_video # Return base_job_info even on failure here for context

    # Log actual frames that will be processed considering stride & process_length
    actual_raw_frames_to_be_processed_count = min(effective_total_output_frames_to_target_for_video * stride_for_fps_adjustment, total_raw_frames_in_original_video)
    log_message("SEGMENT_DEFINE_PROGRESS", video_name=original_basename, 
                output_frames=effective_total_output_frames_to_target_for_video, 
                raw_frames=actual_raw_frames_to_be_processed_count)


    if gui_segment_output_window_frames <= 0:
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Segment output frame count ({gui_segment_output_window_frames}) must be positive.")
        return [], base_job_info_for_video
    if gui_segment_output_overlap_frames < 0 or gui_segment_output_overlap_frames >= gui_segment_output_window_frames:
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Segment output frame overlap ({gui_segment_output_overlap_frames}) invalid for window {gui_segment_output_window_frames}.")
        return [], base_job_info_for_video

    # Raw frames for window/overlap based on stride
    segment_def_window_raw = gui_segment_output_window_frames * stride_for_fps_adjustment
    segment_def_overlap_raw = gui_segment_output_overlap_frames * stride_for_fps_adjustment
    advance_per_segment_raw = segment_def_window_raw - segment_def_overlap_raw
    
    # This is the length of the source (raw video frames, or generated clip frames) we'll iterate over.
    effective_raw_video_length_to_consider = actual_raw_frames_to_be_processed_count
    
    if advance_per_segment_raw <= 0 and effective_raw_video_length_to_consider > segment_def_window_raw :
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Segment raw advance ({advance_per_segment_raw}) is not positive when multiple segments expected. Check window/overlap/FPS.")
        return [], base_job_info_for_video

    current_raw_frame_idx = 0
    segment_id_counter = 0
    temp_segment_jobs = []

    while current_raw_frame_idx < effective_raw_video_length_to_consider:
        num_raw_frames_for_this_segment_def = min(
            segment_def_window_raw,
            effective_raw_video_length_to_consider - current_raw_frame_idx
        )
        if num_raw_frames_for_this_segment_def <= 0:
            break 
        
        segment_job = {
            **base_job_info_for_video,
            "start_frame_raw_index": current_raw_frame_idx,
            "num_frames_to_load_raw": num_raw_frames_for_this_segment_def,
            "segment_id": segment_id_counter,
            "is_segment": True,
            "gui_desired_output_window_frames": gui_segment_output_window_frames,
            "gui_desired_output_overlap_frames": gui_segment_output_overlap_frames,
        }
        temp_segment_jobs.append(segment_job)
        segment_id_counter += 1

        if current_raw_frame_idx + num_raw_frames_for_this_segment_def >= effective_raw_video_length_to_consider:
            break 
        
        if advance_per_segment_raw <= 0 : # Should have been caught if multiple segments were expected
             # If we are here, it implies only one segment was expected, or the check above missed something.
             log_message("SEGMENT_DEFINE_ZERO_ADVANCE_LOOP_BREAK", video_name=original_basename, advance=advance_per_segment_raw)
             break 

        current_raw_frame_idx += advance_per_segment_raw
        if current_raw_frame_idx >= effective_raw_video_length_to_consider:
            break
            
    total_segments_for_this_vid = len(temp_segment_jobs)
    if total_segments_for_this_vid == 0 and effective_total_output_frames_to_target_for_video > 0 :
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason="No segments defined after loop, but frames were expected.")
    elif total_segments_for_this_vid > 0:
        for i_job in range(total_segments_for_this_vid):
            temp_segment_jobs[i_job]["total_segments"] = total_segments_for_this_vid
        segment_jobs.extend(temp_segment_jobs)
        log_message("SEGMENT_DEFINE_SUCCESS", num_segments=total_segments_for_this_vid, video_name=original_basename)
        
    return segment_jobs, base_job_info_for_video


def normalize_video_data(
    video_data: np.ndarray,
    use_percentile_norm: bool,
    low_perc: float,
    high_perc: float
    # log_func: Optional[Callable[[str], None]] = None # Removed
) -> np.ndarray:
    """Normalizes video data to the 0-1 range."""
    if video_data is None or video_data.size == 0:
        log_message("UTIL_NORMALIZE_EMPTY_VIDEO_ERROR") # New ID
        raise ValueError("Cannot normalize empty video array.")

    log_message("UTIL_NORMALIZE_VIDEO_START", shape=video_data.shape) # New ID
    
    normalized_video = video_data.copy().astype(np.float32)
    min_val_for_norm, max_val_for_norm = np.min(normalized_video), np.max(normalized_video)
    method_str = "percentile"

    if use_percentile_norm:
        if normalized_video.ndim > 0 and normalized_video.shape[0] > 2 and normalized_video.flatten().size > 20:
            min_val_for_norm = np.percentile(normalized_video.flatten(), low_perc)
            max_val_for_norm = np.percentile(normalized_video.flatten(), high_perc)
        else:
            log_message("UTIL_NORMALIZE_PERCENTILE_FALLBACK", low=low_perc, high=high_perc) # New ID
            method_str = "absolute (percentile fallback)"
    else:
        method_str = "absolute"

    log_message("UTIL_NORMALIZE_VIDEO", shape=video_data.shape, method=method_str, 
                min_val=min_val_for_norm, max_val=max_val_for_norm)

    if abs(max_val_for_norm - min_val_for_norm) < 1e-6:
        log_message("UTIL_NORMALIZE_FLAT_VIDEO_WARN") # New ID
        flat_value = 0.5
        if (0.0 <= min_val_for_norm <= 1.0 and 0.0 <= max_val_for_norm <= 1.0 and abs(max_val_for_norm - min_val_for_norm) < 1e-7):
            flat_value = np.clip(min_val_for_norm, 0.0, 1.0)
        
        normalized_video = np.full_like(normalized_video, flat_value, dtype=np.float32)
        log_message("UTIL_NORMALIZE_FLAT_VIDEO_RESULT", value=flat_value) # New ID
    else:
        normalized_video = (normalized_video - min_val_for_norm) / (max_val_for_norm - min_val_for_norm)
    
    normalized_video = np.clip(normalized_video, 0.0, 1.0)
    log_message("UTIL_NORMALIZE_FINAL_RANGE", min_val=np.min(normalized_video), max_val=np.max(normalized_video)) # New ID
    return normalized_video


def apply_gamma_correction_to_video(
    video_data: np.ndarray,
    gamma_value: float
    # log_func: Optional[Callable[[str], None]] = None # Removed
) -> np.ndarray:
    """Applies gamma correction to video data."""
    processed_video = video_data.copy()
    actual_gamma = max(0.1, gamma_value)

    if abs(actual_gamma - 1.0) > 1e-3:
        log_message("UTIL_GAMMA_CORRECTION", gamma_val=actual_gamma)
        processed_video = np.power(np.clip(processed_video, 0, 1), 1.0 / actual_gamma)
        processed_video = np.clip(processed_video, 0, 1)
    else:
        log_message("UTIL_GAMMA_CORRECTION_SKIPPED", gamma_val=actual_gamma) # New ID
    return processed_video


def apply_dithering_to_video(
    video_data: np.ndarray,
    dither_strength_factor: float
    # log_func: Optional[Callable[[str], None]] = None # Removed
) -> np.ndarray:
    """Applies dithering to video data."""
    processed_video = video_data.copy()
    log_message("UTIL_DITHERING_START") # New ID
    
    dither_range = (1.0 / 255.0) * dither_strength_factor
    noise = np.random.uniform(-dither_range, dither_range, processed_video.shape).astype(np.float32)
    processed_video = np.clip(processed_video + noise, 0, 1)
    
    log_message("UTIL_DITHERING", strength_factor=dither_strength_factor, dither_range=dither_range)
    return processed_video


def load_json_file(filepath: str) -> Optional[dict]: # log_func removed
    """Loads data from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        log_message("FILE_LOAD_SUCCESS", filepath=filepath)
        return data
    except FileNotFoundError:
        log_message("FILE_NOT_FOUND", filepath=filepath)
    except json.JSONDecodeError as e:
        log_message("JSON_DECODE_ERROR", filepath=filepath, reason=str(e))
    except Exception as e:
        log_message("GENERAL_ERROR", message=f"loading JSON from {filepath}: {e}")
    return None

def save_json_file(data: dict, filepath: str, indent: int = 4) -> bool: # log_func removed
    """Saves data to a JSON file."""
    try:
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        log_message("FILE_SAVE_SUCCESS", filepath=os.path.basename(filepath)) # Log only basename for brevity in repeated calls
        return True
    except TypeError as e:
        log_message("FILE_SAVE_FAILURE", filepath=filepath, reason=f"Data not JSON serializable: {e}")
    except (IOError, OSError) as e:
        log_message("FILE_SAVE_FAILURE", filepath=filepath, reason=str(e))
    except Exception as e:
        log_message("FILE_SAVE_FAILURE", filepath=filepath, reason=f"Unexpected error: {e}")
    return False

def read_video_frames(video_path, process_length, target_fps, max_res, dataset="open",
                      start_frame_index=0, num_frames_to_load=-1) -> Tuple[np.ndarray, float, int, int]:
    original_height, original_width = 0, 0
    original_video_fps = 30.0 # Default
    total_frames_in_video = 0 # Default

    try:
        temp_vid_for_meta = VideoReader(video_path, ctx=cpu(0))
        if len(temp_vid_for_meta) > 0:
            first_frame_data = temp_vid_for_meta.get_batch([0])
            if first_frame_data is not None and first_frame_data.shape[0] > 0:
                 original_height, original_width = first_frame_data.shape[1:3]
            else:
                log_message("VIDEO_READ_METADATA_EMPTY_FRAME_WARN", video_path=video_path)
                return np.array([]), (target_fps if target_fps != -1 else 30), 0, 0
        else:
            log_message("VIDEO_READ_METADATA_ZERO_LENGTH_WARN", video_path=video_path)
            return np.array([]), (target_fps if target_fps != -1 else 30), 0, 0

        original_video_fps = temp_vid_for_meta.get_avg_fps()
        if original_video_fps <= 0: # Handle invalid FPS from metadata
            log_message("VIDEO_INVALID_FPS_METADATA_WARN", video_path=video_path, fps_read=original_video_fps)
            original_video_fps = 30.0 # Fallback
        total_frames_in_video = len(temp_vid_for_meta)
        del temp_vid_for_meta
        gc.collect() # Added gc.collect after del
    except Exception as e:
        log_message("VIDEO_READ_METADATA_ERROR", video_path=video_path, error=str(e))
        return np.array([]), (target_fps if target_fps != -1 else 30), 0, 0

    # Resolution calculation logic (moved here, before VideoReader init with target dims)
    # These 'height' and 'width' are the TARGET dimensions for VideoReader
    if original_height == 0 or original_width == 0: # Check if metadata read failed to get dimensions
        log_message("VIDEO_READ_NO_ORIGINAL_DIMS_ERROR", video_path=video_path)
        return np.array([]), (target_fps if target_fps != -1 else original_video_fps), 0, 0


    if dataset == "open":
        height = round(original_height / 64) * 64
        width = round(original_width / 64) * 64
        if max_res > 0 and max(height, width) > max_res : # Check if max_res is positive AND if calculated H/W exceeds it
            # Check if original_height or original_width is zero before division
            if original_height == 0 or original_width == 0:
                log_message("VIDEO_READ_ZERO_DIM_SCALE_ERROR", video_path=video_path)
                # Fallback to something, or return error
                height = 64 
                width = 64
            else:
                scale = max_res / max(original_height, original_width) # Use original_height/width for scaling basis
                height = round(original_height * scale / 64) * 64
                width = round(original_width * scale / 64) * 64
    else:
        if dataset in dataset_res_dict:
            height = dataset_res_dict[dataset][0]
            width = dataset_res_dict[dataset][1]
        else:
            log_message("VIDEO_UNKNOWN_DATASET_WARN", dataset_name=dataset)
            # Fallback to 'open' style calculation if dataset is unknown
            height = round(original_height / 64) * 64
            width = round(original_width / 64) * 64
            if max_res > 0 and max(height, width) > max_res:
                if original_height == 0 or original_width == 0:
                    log_message("VIDEO_READ_ZERO_DIM_SCALE_ERROR", video_path=video_path)
                    height = 64
                    width = 64
                else:
                    scale = max_res / max(original_height, original_width)
                    height = round(original_height * scale / 64) * 64
                    width = round(original_width * scale / 64) * 64
    
    # Ensure height and width are at least 64 (or some minimum)
    height = max(64, int(height))
    width = max(64, int(width))

    try:
        # Initialize VideoReader with the calculated target height and width
        vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
    except Exception as e:
        log_message("VIDEO_READER_INIT_ERROR", video_path=video_path, error=f"{e} (target H:{height}, W:{width})") # Add target dims to log
        return np.array([]), (target_fps if target_fps != -1 else original_video_fps), original_height, original_width

    # FPS and Stride Calculation
    actual_fps_for_save = original_video_fps if target_fps == -1 else target_fps
    if actual_fps_for_save <= 0: 
        actual_fps_for_save = original_video_fps if original_video_fps > 0 else 23.976

    stride = 1
    if original_video_fps > 0 and actual_fps_for_save > 0:
        stride = round(original_video_fps / actual_fps_for_save)
    stride = max(stride, 1) # Stride must be at least 1
    
    # Frame Index Calculation
    effective_num_frames_in_source_segment = num_frames_to_load
    if num_frames_to_load == -1: # Read until end of video from start_frame_index
        effective_num_frames_in_source_segment = total_frames_in_video - start_frame_index
    
    # Ensure effective_num_frames_in_source_segment is not negative
    effective_num_frames_in_source_segment = max(0, effective_num_frames_in_source_segment)

    segment_end_frame_exclusive = min(start_frame_index + effective_num_frames_in_source_segment, total_frames_in_video)
    
    if start_frame_index >= segment_end_frame_exclusive :
        log_message("VIDEO_SEGMENT_EMPTY_WARN", video_path=video_path, start_index=start_frame_index, num_frames=num_frames_to_load)
        return np.array([]), actual_fps_for_save, original_height, original_width

    source_indices_for_segment = list(range(start_frame_index, segment_end_frame_exclusive))

    if not source_indices_for_segment:
        log_message("VIDEO_NO_SOURCE_INDICES_WARN", video_path=video_path)
        return np.array([]), actual_fps_for_save, original_height, original_width

    final_indices_to_read = [source_indices_for_segment[i] for i in range(0, len(source_indices_for_segment), stride)]

    # Apply overall process_length limit if it's not for a specific segment (num_frames_to_load was -1)
    # AND if process_length is not -1 (meaning unlimited)
    if num_frames_to_load == -1 and process_length != -1 and process_length < len(final_indices_to_read):
        final_indices_to_read = final_indices_to_read[:process_length]
    
    if not final_indices_to_read:
        log_message("VIDEO_NO_FRAMES_TO_READ_WARN", video_path=video_path)
        return np.array([]), actual_fps_for_save, original_height, original_width

    try:
        frames = vid.get_batch(final_indices_to_read).asnumpy().astype("float32") / 255.0
    except Exception as e:
        log_message("VIDEO_GET_BATCH_ERROR", video_path=video_path, error=str(e))
        return np.array([]), actual_fps_for_save, original_height, original_width
        
    del vid
    gc.collect() # Added gc.collect
    return frames, actual_fps_for_save, original_height, original_width

def save_video(video_frames: Union[List[np.ndarray], List[PIL.Image.Image], np.ndarray], output_video_path: str = None,
               fps: Union[int, float] = 10.0, crf: int = 18, output_format: Optional[str] = None ) -> str:
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name
    elif output_format == "main10_mp4" and not output_video_path.lower().endswith(".mp4"):
        # If HEVC is chosen but extension isn't mp4, log warning or adjust.
        # For now, we'll assume output_video_path is correctly set by the caller.
        log_message("VIDEO_SAVE_HEVC_MP4_EXTENSION_WARN", filepath=output_video_path, level=WARNING)

    # Frame conversion logic (remains the same)
    if isinstance(video_frames, np.ndarray):
        if video_frames.ndim == 3:
            if video_frames.dtype == np.float32 or video_frames.dtype == np.float64:
                 video_frames = (video_frames * 255).astype(np.uint8)
        elif video_frames.ndim == 4:
            if video_frames.dtype == np.float32 or video_frames.dtype == np.float64:
                video_frames = (video_frames * 255).astype(np.uint8)
    elif isinstance(video_frames, list) and len(video_frames) > 0 and isinstance(video_frames[0], np.ndarray):
        processed_frames = []
        for frame in video_frames:
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                processed_frames.append((frame * 255).astype(np.uint8))
            elif frame.dtype == np.uint8:
                processed_frames.append(frame)
            else:
                log_message("VIDEO_SAVE_UNSUPPORTED_DTYPE_ERROR", dtype=str(frame.dtype))
                raise ValueError(f"Unsupported numpy array dtype in list: {frame.dtype}")
        video_frames = processed_frames
    elif isinstance(video_frames, list) and len(video_frames) > 0 and isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    elif isinstance(video_frames, list) and len(video_frames) == 0:
        log_message("VIDEO_SAVE_EMPTY_FRAMES_WARN")
        return output_video_path
    else:
        log_message("VIDEO_SAVE_INVALID_FRAMES_TYPE_ERROR")
        raise ValueError("video_frames must be a list/array of np.ndarray or a list of PIL.Image.Image")

    mediapy_kwargs = {'fps': fps}
    ffmpeg_custom_args = [] # List to hold specific ffmpeg arguments

    if output_format == "main10_mp4" and output_video_path.lower().endswith(".mp4"):
        mediapy_kwargs['codec'] = 'libx265' # Tell mediapy to use libx265
        ffmpeg_custom_args.extend(['-pix_fmt', 'yuv420p10le']) # Pass pix_fmt via ffmpeg_args
        # Add x265 specific params if needed, e.g., tag for HEVC in MP4
        ffmpeg_custom_args.extend(['-tag:v', 'hvc1']) # Common tag for HEVC in MP4
        # CRF for libx265 is typically set with -crf for the libx265 encoder
        # mediapy's 'crf' argument should ideally pass this to the specified codec.
        mediapy_kwargs['crf'] = crf # Assuming mediapy passes this correctly to libx265.
                                     # If not, you might need: ffmpeg_custom_args.extend(['-crf', str(crf)])
        log_message("VIDEO_SAVE_HEVC_MAIN10_MP4_SELECTED", filepath=output_video_path, kwargs=str(mediapy_kwargs), ffmpeg_args=str(ffmpeg_custom_args))
    else: # Default to H.264 8-bit for .mp4
        if output_video_path.lower().endswith(".mp4"):
            if output_format == "mp4" or output_format is None: # Explicitly "mp4" or default for .mp4
                mediapy_kwargs['codec'] = 'libx264'
                ffmpeg_custom_args.extend(['-pix_fmt', 'yuv420p']) # Standard 8-bit
        mediapy_kwargs['crf'] = crf

    if ffmpeg_custom_args:
        mediapy_kwargs['ffmpeg_args'] = ffmpeg_custom_args

    try:
        if video_frames is not None and len(video_frames) > 0:
            first_frame_shape_len = len(video_frames[0].shape)
            if first_frame_shape_len == 2:
                 pass
            elif first_frame_shape_len == 3 and video_frames[0].shape[-1] != 3 and video_frames[0].shape[-1] != 4 :
                 pass
        mediapy.write_video(output_video_path, video_frames, **mediapy_kwargs)
    except Exception as e:
        log_message("VIDEO_SAVE_MEDIAPY_ERROR", filepath=output_video_path, error=str(e), format_details=f"Requested format: {output_format}, mediapy_kwargs: {mediapy_kwargs}")
        raise
    return output_video_path


class ColorMapper:
    def __init__(self, colormap: str = "inferno"):
        self.colormap_name = colormap
        self._cmap_data = None

    def _get_cmap_data(self):
        if self._cmap_data is None:
            try:
                import matplotlib.cm as cm_mpl
                self._cmap_data = torch.tensor(cm_mpl.get_cmap(self.colormap_name).colors)
            except ImportError:
                log_message("COLORMAP_MPL_IMPORT_ERROR") # New ID
                # Fallback to a very simple grayscale if matplotlib is not available
                # This is a basic fallback, not a full replacement.
                ramp = torch.linspace(0, 1, 256)
                self._cmap_data = torch.stack([ramp, ramp, ramp], dim=1) # (N, 3)
        return self._cmap_data


    def apply(self, image: torch.Tensor, v_min=None, v_max=None):
        if image.ndim not in [2,3]:
            log_message("COLORMAP_INVALID_INPUT_DIMS_ERROR", ndim=image.ndim) # New ID
            raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")

        cmap_data = self._get_cmap_data().to(image.device)
        
        if v_min is None: v_min = image.min()
        if v_max is None: v_max = image.max()
        
        if v_max == v_min:
            image_normalized = torch.zeros_like(image)
        else:
            image_normalized = (image - v_min) / (v_max - v_min)
        
        image_long = (image_normalized * (len(cmap_data) -1) ).long()
        image_long = torch.clamp(image_long, 0, len(cmap_data) - 1)
        colored_image = cmap_data[image_long]
        return colored_image


def vis_sequence_depth(depths: np.ndarray, v_min=None, v_max=None, colormap: str = "inferno"):
    if not isinstance(depths, np.ndarray):
        depths = np.array(depths)
    if depths.ndim != 3:
        log_message("VIS_SEQ_INVALID_INPUT_DIMS_ERROR", ndim=depths.ndim) # New ID
        raise ValueError(f"Input depths must be a 3D array (T, H, W), got {depths.ndim}D")

    visualizer = ColorMapper(colormap=colormap)
    if v_min is None: v_min = depths.min()
    if v_max is None: v_max = depths.max()
    
    depths_tensor = torch.from_numpy(depths.astype(np.float32))
    colored_sequence_tensor = visualizer.apply(depths_tensor, v_min=v_min, v_max=v_max)
    colored_sequence_np = colored_sequence_tensor.cpu().numpy()
    
    if colored_sequence_np.shape[-1] == 4:
        colored_sequence_np = colored_sequence_np[..., :3]
    return colored_sequence_np

def save_depth_visual_as_mp4_util(depth_frames_normalized: np.ndarray, output_filepath: str, fps: Union[int, float],
                                  output_format: str = "mp4" ) -> Tuple[Optional[str], Optional[str]]:
    try:
        # output_format string is passed directly to save_video
        save_video(depth_frames_normalized, output_filepath, fps=fps, output_format=output_format)
        return output_filepath, None
    except Exception as e:
        log_message("VIDEO_SAVE_MP4_UTIL_ERROR", filepath=output_filepath, error=str(e), format_requested=output_format)
        return None, str(e)

def save_depth_visual_as_png_sequence_util(depth_frames_normalized: np.ndarray,  output_dir_base: str,
                                           base_filename_no_ext: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        visual_dirname = f"{base_filename_no_ext}_visual_png_seq"
        png_dir_path = os.path.join(output_dir_base, visual_dirname)
        if os.path.exists(png_dir_path): 
            shutil.rmtree(png_dir_path)
        os.makedirs(png_dir_path, exist_ok=True)
        for i, frame_float in enumerate(depth_frames_normalized):
            frame_uint16 = (np.clip(frame_float, 0, 1) * 65535.0).astype(np.uint16)
            frame_filename = os.path.join(png_dir_path, f"frame_{i:05d}.png")
            imageio.imwrite(frame_filename, frame_uint16)
        # log_message("FILE_SAVE_SUCCESS", filepath=png_dir_path) # Potentially too verbose
        return png_dir_path, None
    except Exception as e:
        log_message("IMAGE_SAVE_PNG_SEQ_UTIL_ERROR", dir_path=png_dir_path, error=str(e)) # New ID
        return None, str(e)

def save_depth_visual_as_exr_sequence_util(
    depth_frames_normalized: np.ndarray, 
    output_dir_base: str, 
    base_filename_no_ext: str
) -> Tuple[Optional[str], Optional[str]]:
    if not _OPENEXR_AVAILABLE_IN_UTILS:
        log_message("OPENEXR_UNAVAILABLE", context="save_depth_visual_as_exr_sequence_util")
        return None, "OpenEXR libraries not available in utils.py for EXR sequence saving."

    exr_sequence_output_dir = "unknown_path_exr_seq" # Default for logging if error before assignment
    try:
        if depth_frames_normalized.ndim != 3: # T, H, W
             err_msg = f"EXR sequence expects 3D array (T,H,W), got {depth_frames_normalized.ndim}D"
             log_message("IMAGE_SAVE_EXR_SEQ_UTIL_ERROR", dir_path=exr_sequence_output_dir, error=err_msg)
             return None, err_msg
        
        num_frames, height, width = depth_frames_normalized.shape
        if num_frames == 0:
            err_msg = "No frames to save in EXR sequence."
            log_message("IMAGE_SAVE_EXR_SEQ_UTIL_ERROR", dir_path=exr_sequence_output_dir, error=err_msg)
            return None, err_msg

        sequence_subfolder_name = f"{base_filename_no_ext}_visual_exr_seq"
        exr_sequence_output_dir = os.path.join(output_dir_base, sequence_subfolder_name)
        
        if os.path.exists(exr_sequence_output_dir): 
            shutil.rmtree(exr_sequence_output_dir)
        os.makedirs(exr_sequence_output_dir, exist_ok=True)

        for i in range(num_frames):
            frame_data_float32 = depth_frames_normalized[i].astype(np.float32)
            output_exr_filepath = os.path.join(exr_sequence_output_dir, f"frame_{i:05d}.exr")

            try:
                header = OpenEXR.Header(width, height)
                # Save as single channel (e.g., 'Z' for depth) or RGB depending on input
                # Assuming depth_frames_normalized is (T, H, W) - so single channel depth
                if frame_data_float32.ndim == 2: # Grayscale (H, W)
                    header['channels'] = {'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
                    pixel_data = {'Z': frame_data_float32.tobytes()}
                elif frame_data_float32.ndim == 3 and frame_data_float32.shape[-1] == 1: # (H, W, 1)
                    header['channels'] = {'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
                    pixel_data = {'Z': frame_data_float32.squeeze().tobytes()}
                elif frame_data_float32.ndim == 3 and frame_data_float32.shape[-1] == 3: # RGB (H, W, 3)
                    header['channels'] = {
                        'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                        'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                        'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                    }
                    pixel_data = {
                        'R': frame_data_float32[:, :, 0].tobytes(),
                        'G': frame_data_float32[:, :, 1].tobytes(),
                        'B': frame_data_float32[:, :, 2].tobytes()
                    }
                else:
                    err_msg_frame = f"Unsupported frame shape for EXR: {frame_data_float32.shape}"
                    log_message("IMAGE_SAVE_EXR_FRAME_ERROR", filepath=output_exr_filepath, error=err_msg_frame)
                    # Decide to skip frame or error out; for now, skip and continue
                    continue 
                
                exr_file = OpenEXR.OutputFile(output_exr_filepath, header)
                exr_file.writePixels(pixel_data)
                exr_file.close()
            except Exception as frame_ex:
                log_message("IMAGE_SAVE_EXR_FRAME_ERROR", filepath=output_exr_filepath, error=str(frame_ex))
                # Optionally, if one frame fails, maybe the whole sequence should fail?
                # For now, it continues.
                
        # log_message("FILE_SAVE_SUCCESS", filepath=exr_sequence_output_dir) # Potentially too verbose
        return exr_sequence_output_dir, None 
    except Exception as e:
        # Ensure path_for_log is defined even if exr_sequence_output_dir wasn't set due to early error
        path_for_log = exr_sequence_output_dir if 'exr_sequence_output_dir' in locals() and exr_sequence_output_dir else "unknown_path_exr_seq"
        log_message("IMAGE_SAVE_EXR_SEQ_UTIL_ERROR", dir_path=path_for_log, error=str(e))
        return None, str(e)

def save_depth_visual_as_single_exr_util(
    first_depth_frame_normalized: np.ndarray, 
    output_dir_base: str, 
    base_filename_no_ext: str
) -> Tuple[Optional[str], Optional[str]]:
    if not _OPENEXR_AVAILABLE_IN_UTILS:
        log_message("OPENEXR_UNAVAILABLE", context="save_depth_visual_as_single_exr_util")
        return None, "OpenEXR libraries not available in utils.py for single EXR saving."

    output_exr_filepath = "unknown_path_single_exr.exr" # Default for logging
    try:
        if first_depth_frame_normalized is None or first_depth_frame_normalized.size == 0:
            err_msg = "No frame data to save for single EXR"
            log_message("IMAGE_SAVE_SINGLE_EXR_UTIL_ERROR", filepath=output_exr_filepath, error=err_msg)
            return None, err_msg
        
        frame_float32 = first_depth_frame_normalized.astype(np.float32)
        
        if frame_float32.ndim == 2: # Grayscale (H, W)
            height, width = frame_float32.shape
        elif frame_float32.ndim == 3 and (frame_float32.shape[-1] == 1 or frame_float32.shape[-1] == 3) : # (H,W,1) or (H,W,3)
            height, width = frame_float32.shape[:2]
        else:
            err_msg = f"Unsupported frame shape for single EXR: {frame_float32.shape}"
            log_message("IMAGE_SAVE_SINGLE_EXR_UTIL_ERROR", filepath=output_exr_filepath, error=err_msg)
            return None, err_msg

        os.makedirs(output_dir_base, exist_ok=True)
        output_exr_filepath = os.path.join(output_dir_base, f"{base_filename_no_ext}_visual.exr")
        
        header = OpenEXR.Header(width, height)
        pixel_data = {}

        if frame_float32.ndim == 2: # Grayscale (H, W)
            header['channels'] = {'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
            pixel_data = {'Z': frame_float32.tobytes()}
        elif frame_float32.ndim == 3 and frame_float32.shape[-1] == 1: # (H, W, 1)
            header['channels'] = {'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
            pixel_data = {'Z': frame_float32.squeeze().tobytes()}
        elif frame_float32.ndim == 3 and frame_float32.shape[-1] == 3: # RGB (H, W, 3)
            header['channels'] = {
                'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            pixel_data = {
                'R': frame_float32[:, :, 0].tobytes(),
                'G': frame_float32[:, :, 1].tobytes(),
                'B': frame_float32[:, :, 2].tobytes()
            }
        
        exr_file = OpenEXR.OutputFile(output_exr_filepath, header)
        exr_file.writePixels(pixel_data)
        exr_file.close()
        
        # log_message("FILE_SAVE_SUCCESS", filepath=output_exr_filepath) # Potentially too verbose
        return output_exr_filepath, None
    except Exception as e:
        path_for_log = output_exr_filepath if 'output_exr_filepath' in locals() and output_exr_filepath else "unknown_path_single_exr.exr"
        log_message("IMAGE_SAVE_SINGLE_EXR_UTIL_ERROR", filepath=path_for_log, error=str(e))
        return None, str(e)

def read_image_sequence_as_frames(
    folder_path: str,
    num_frames_to_load: int, # Renamed from process_length for clarity (this is for the specific segment/call)
    max_res: int,
    start_index: int = 0    # New parameter: the starting frame index for this segment
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[int]]:
    """Reads a segment of an image sequence from a folder into a NumPy array."""
    supported_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".exr")
    frames_found = []
    for ext in supported_exts:
        frames_found.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        frames_found.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))

    all_image_paths_sorted = sorted(list(set(frames_found)))

    if not all_image_paths_sorted:
        log_message("IMAGE_SEQUENCE_LOAD_NO_IMAGES", folder_path=folder_path)
        return None, None, None

    # Determine the slice of image paths for the current segment
    # Ensure start_index is within bounds
    if start_index < 0 or start_index >= len(all_image_paths_sorted):
        log_message("IMAGE_SEQUENCE_LOAD_INVALID_START_INDEX", 
                    folder_path=folder_path, start_index=start_index, 
                    total_images=len(all_image_paths_sorted))
        return None, None, None # Or return empty array if preferred

    # num_frames_to_load: if -1, load all from start_index. Otherwise, load specified number.
    end_index: Optional[int]
    if num_frames_to_load == -1: # Load all remaining frames
        end_index = len(all_image_paths_sorted)
    else:
        end_index = start_index + num_frames_to_load
    
    # Slice the list of paths for the current segment
    image_paths_for_segment = all_image_paths_sorted[start_index:end_index]

    if not image_paths_for_segment:
        log_message("IMAGE_SEQUENCE_LOAD_NO_FRAMES_FOR_SEGMENT", 
                    folder_path=folder_path, start_index=start_index, 
                    num_to_load=num_frames_to_load, end_index_calc=end_index,
                    total_images_in_folder=len(all_image_paths_sorted))
        return None, None, None


    loaded_frames_list = []
    original_h, original_w = 0, 0 # For the first frame of the *sequence* (not necessarily segment)
    target_h, target_w = 0, 0   # Target dimensions after resizing

    # Get original dimensions from the very first image of the sequence for consistent resizing
    # This is important if segments are processed independently and need same target res.
    try:
        # Use the first image of the *entire sequence* to determine original H, W for scaling
        first_image_of_sequence = imageio.v2.imread(all_image_paths_sorted[0]) 
        ref_h, ref_w = first_image_of_sequence.shape[:2]

        # Calculate target_h, target_w based on max_res and this reference frame
        current_h_ref, current_w_ref = ref_h, ref_w
        if max_res > 0 and max(current_h_ref, current_w_ref) > max_res:
            scale = max_res / max(current_h_ref, current_w_ref)
            target_h = round(current_h_ref * scale / 64) * 64
            target_w = round(current_w_ref * scale / 64) * 64
        else:
            target_h = round(current_h_ref / 64) * 64
            target_w = round(current_w_ref / 64) * 64
        target_h = max(64, target_h)
        target_w = max(64, target_w)
        
        # Store the originally detected H,W of the sequence (not the target)
        original_h, original_w = ref_h, ref_w

    except Exception as e_ref:
        log_message("IMAGE_SEQUENCE_REF_FRAME_READ_ERROR", 
                    filepath=all_image_paths_sorted[0] if all_image_paths_sorted else "N/A", 
                    error=str(e_ref))
        return None, None, None # Cannot proceed without reference dimensions


    for i, frame_path in enumerate(image_paths_for_segment):
        try:
            img = imageio.v2.imread(frame_path)
            
            # Resize if necessary to target_h, target_w (calculated once above)
            if img.shape[0] != target_h or img.shape[1] != target_w:
                pil_img = PIL.Image.fromarray(img)
                resized_pil_img = pil_img.resize((target_w, target_h), PIL.Image.LANCZOS)
                img = np.array(resized_pil_img)

            # ... (channel handling: grayscale, RGBA) ...
            if img.ndim == 2: 
                img = np.stack([img]*3, axis=-1) 
            if img.shape[2] == 4: 
                img = img[..., :3]

            loaded_frames_list.append((img.astype(np.float32) / 255.0))
        except Exception as e:
            # ... (error logging for individual frame read) ...
            continue
    
    if not loaded_frames_list:
        # ... (error logging if no frames loaded for segment) ...
        return None, None, None

    frames_array = np.stack(loaded_frames_list, axis=0)
    log_message("IMAGE_SEQUENCE_LOAD_SUCCESS", folder_path=folder_path, 
                num_frames=frames_array.shape[0], 
                height=frames_array.shape[1], width=frames_array.shape[2],
                segment_start_index=start_index) # Log segment info
    # Return the original H, W of the *sequence* (from first frame), not necessarily of the loaded frames if resized
    return frames_array, original_h, original_w


def create_frames_from_single_image(
    image_path: str, 
    num_frames_to_generate: int, 
    max_res: int
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[int]]: # frames, original_h, original_w
    """Creates a sequence of identical frames from a single image."""
    try:
        img_arr = imageio.v2.imread(image_path)
        original_h, original_w = img_arr.shape[:2]

        current_h, current_w = original_h, original_w
        if max_res > 0 and max(current_h, current_w) > max_res:
            scale = max_res / max(current_h, current_w)
            target_h = round(current_h * scale / 64) * 64
            target_w = round(current_w * scale / 64) * 64
        else:
            target_h = round(current_h / 64) * 64
            target_w = round(current_w / 64) * 64
        target_h = max(64, target_h)
        target_w = max(64, target_w)

        if img_arr.shape[0] != target_h or img_arr.shape[1] != target_w:
            pil_img = PIL.Image.fromarray(img_arr)
            resized_pil_img = pil_img.resize((target_w, target_h), PIL.Image.LANCZOS)
            img_arr = np.array(resized_pil_img)
            
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr]*3, axis=-1)
        if img_arr.shape[2] == 4:
            img_arr = img_arr[..., :3]

        frame_float = img_arr.astype(np.float32) / 255.0
        frames_array = np.stack([frame_float] * num_frames_to_generate, axis=0)
        log_message("SINGLE_IMAGE_FRAMES_GENERATED", image_path=image_path, num_frames=frames_array.shape[0],
                    height=frames_array.shape[1], width=frames_array.shape[2])
        return frames_array, original_h, original_w
    except Exception as e:
        log_message("SINGLE_IMAGE_FRAMES_GENERATION_ERROR", image_path=image_path, error=str(e))
        return None, None, None

# Modify define_video_segments
def define_video_segments(
    video_path_or_folder: str, # Path to video, image sequence folder, or single image
    original_basename: str,
    gui_target_fps_setting: int,
    gui_process_length_overall: int,
    gui_segment_output_window_frames: int,
    gui_segment_output_overlap_frames: int,
    source_type: str, # "video_file", "image_sequence_folder", "single_image_file"
    # source_dimensions: Optional[Tuple[int, int]] = None # H, W - no longer needed here, get from metadata funcs
) -> Tuple[List[dict], Optional[dict]]:
    """
    Defines video segments based on input parameters.
    Returns:
        A tuple containing:
        - A list of segment job dictionaries.
        - A base job info dictionary (common details for the video).
          Returns None for base_job_info if metadata read fails.
    """
    segment_jobs = []
    base_job_info_for_video = {}

    total_raw_frames_in_original_video = 0
    original_video_fps = 30.0
    # original_h, original_w = 0, 0 # Not strictly needed by this function anymore for segment definition

    if source_type == "video_file":
        try:
            vr = VideoReader(video_path_or_folder, ctx=cpu(0))
            total_raw_frames_in_original_video = len(vr)
            original_video_fps = vr.get_avg_fps()
            # temp_frame_shape = vr.get_batch([0]).shape # To get H, W if needed, but not directly by this func
            # original_h, original_w = temp_frame_shape[1], temp_frame_shape[2]
            del vr
            gc.collect()
            if original_video_fps <= 0:
                log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Invalid original FPS ({original_video_fps}) from video. Assuming 30 FPS.")
                original_video_fps = 30.0
        except Exception as e:
            log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Error getting metadata of video {video_path_or_folder}: {e}")
            return [], None
    elif source_type == "image_sequence_folder":
        count, fps, h, w = get_image_sequence_metadata(video_path_or_folder, gui_target_fps_setting)
        if count is None:
            log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Error getting metadata for image sequence folder {video_path_or_folder}")
            return [], None
        total_raw_frames_in_original_video = count
        original_video_fps = fps
        # original_h, original_w = h, w
    elif source_type == "single_image_file":
        count, fps, h, w = get_single_image_metadata(video_path_or_folder, gui_target_fps_setting)
        if count is None:
            log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Error getting metadata for single image {video_path_or_folder}")
            return [], None
        total_raw_frames_in_original_video = count
        original_video_fps = fps
        # original_h, original_w = h, w
    else:
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Unknown source_type: {source_type}")
        return [], None

    base_job_info_for_video = {
        "video_path": video_path_or_folder, # This is the source path
        "source_type": source_type, # Store the type
        "gui_fps_setting_at_definition": gui_target_fps_setting, # Store the GUI FPS setting used
        "original_basename": original_basename,
        "original_video_raw_frame_count": total_raw_frames_in_original_video,
        "original_video_fps": original_video_fps, # This is the effective FPS
        # "original_height": original_h, # Store if needed for other parts, though _load_frames will get it again
        # "original_width": original_w,
    }
    # ... (rest of the define_video_segments logic remains the same)
    # It uses total_raw_frames_in_original_video and original_video_fps for its calculations.

    fps_for_stride_calc = original_video_fps if gui_target_fps_setting == -1 else gui_target_fps_setting
    # For image sequences/single images, original_video_fps already reflects gui_target_fps_setting (or default 24)
    # So, if source_type is not "video_file", fps_for_stride_calc should ideally be original_video_fps.
    if source_type != "video_file":
        fps_for_stride_calc = original_video_fps
    
    if fps_for_stride_calc <= 0: # Should be caught by earlier checks, but as a safeguard
        fps_for_stride_calc = original_video_fps if original_video_fps > 0 else 24.0 # Fallback for safety
    
    # For image sequences or single images processed as 1s clips, stride should ideally be 1,
    # as we are reading/generating discrete frames matching the target FPS already.
    # However, the existing logic calculates stride based on original_video_fps vs fps_for_stride_calc.
    # If original_video_fps IS fps_for_stride_calc (as it would be for image seq/single image), stride becomes 1.
    # This seems okay.
    stride_for_fps_adjustment = max(round(original_video_fps / fps_for_stride_calc), 1)
    
    max_possible_output_frames_after_fps = (total_raw_frames_in_original_video + stride_for_fps_adjustment - 1) // stride_for_fps_adjustment
    
    effective_total_output_frames_to_target_for_video = max_possible_output_frames_after_fps
    if gui_process_length_overall != -1 and gui_process_length_overall < effective_total_output_frames_to_target_for_video:
        effective_total_output_frames_to_target_for_video = gui_process_length_overall
    
    if effective_total_output_frames_to_target_for_video <= 0:
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason="Effective output frames is zero or less.")
        return [], base_job_info_for_video

    log_message("SEGMENT_DEFINE_PROGRESS", video_name=original_basename, 
                output_frames=effective_total_output_frames_to_target_for_video, 
                raw_frames=min(effective_total_output_frames_to_target_for_video * stride_for_fps_adjustment, total_raw_frames_in_original_video))


    if gui_segment_output_window_frames <= 0:
        # For single image made into 1s clip, if this window is larger than the clip, it might be an issue.
        # However, the logic below adjusts `num_raw_frames_for_this_segment_def`
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Segment output frame count ({gui_segment_output_window_frames}) must be positive.")
        return [], base_job_info_for_video
    if gui_segment_output_overlap_frames < 0 or gui_segment_output_overlap_frames >= gui_segment_output_window_frames:
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Segment output frame overlap ({gui_segment_output_overlap_frames}) invalid for window {gui_segment_output_window_frames}.")
        return [], base_job_info_for_video

    segment_def_window_raw = gui_segment_output_window_frames * stride_for_fps_adjustment
    segment_def_overlap_raw = gui_segment_output_overlap_frames * stride_for_fps_adjustment
    advance_per_segment_raw = segment_def_window_raw - segment_def_overlap_raw
    
    effective_raw_video_length_to_consider = min(
        effective_total_output_frames_to_target_for_video * stride_for_fps_adjustment,
        total_raw_frames_in_original_video
    )

    if advance_per_segment_raw <= 0 and effective_raw_video_length_to_consider > segment_def_window_raw :
        # This condition might be problematic if effective_raw_video_length_to_consider IS segment_def_window_raw (e.g. short clip, one segment)
        # The original check was fine, but let's be careful.
        # If total length is just one window, advance_per_segment_raw being non-positive is fine.
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Segment raw advance ({advance_per_segment_raw}) is not positive. Check window/overlap/FPS. This may be an issue if video requires multiple segments.")
        # This might not be a fatal error if only one segment is produced.
        # If it results in no segments, that's handled later.

    current_raw_frame_idx = 0
    segment_id_counter = 0
    temp_segment_jobs = []

    while current_raw_frame_idx < effective_raw_video_length_to_consider:
        num_raw_frames_for_this_segment_def = min(
            segment_def_window_raw,
            effective_raw_video_length_to_consider - current_raw_frame_idx
        )
        if num_raw_frames_for_this_segment_def <= 0: # Should not happen if effective_total_output_frames_to_target_for_video > 0
            break 
        
        segment_job = {
            **base_job_info_for_video, # This now includes video_path, source_type, original_basename, etc.
            "start_frame_raw_index": current_raw_frame_idx, # This is an index into the "original" sequence (raw video, image sequence, or 1s generated clip)
            "num_frames_to_load_raw": num_raw_frames_for_this_segment_def, # How many "original" frames this segment definition covers
            "segment_id": segment_id_counter,
            "is_segment": True, # This job is a segment
            "gui_desired_output_window_frames": gui_segment_output_window_frames, # For reference
            "gui_desired_output_overlap_frames": gui_segment_output_overlap_frames, # For reference
        }
        temp_segment_jobs.append(segment_job)
        segment_id_counter += 1

        if current_raw_frame_idx + num_raw_frames_for_this_segment_def >= effective_raw_video_length_to_consider:
            break 
        
        # If advance_per_segment_raw is <=0, this loop will be infinite if it doesn't break above.
        # This should only happen if the video is shorter than one segment window minus overlap.
        if advance_per_segment_raw <= 0:
            log_message("SEGMENT_DEFINE_ZERO_ADVANCE", video_name=original_basename, advance=advance_per_segment_raw, window_raw=segment_def_window_raw, effective_length=effective_raw_video_length_to_consider)
            # If we are here, it means the video is long enough that it *should* advance, but can't. This is an error state.
            if effective_raw_video_length_to_consider > segment_def_window_raw: # Only an error if more than one segment was expected.
                 log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason=f"Segment raw advance ({advance_per_segment_raw}) is not positive when multiple segments expected. Check window/overlap/FPS.")
                 return [], base_job_info_for_video # Or temp_segment_jobs if we allow one segment in this case. For safety, fail.
            # If effective_raw_video_length_to_consider <= segment_def_window_raw, then only one segment is expected,
            # and the loop will break above. advance_per_segment_raw doesn't matter.
            # The earlier check for advance_per_segment_raw should be sufficient.

        current_raw_frame_idx += advance_per_segment_raw
        if current_raw_frame_idx >= effective_raw_video_length_to_consider: # Ensure we don't start a segment beyond the considered length
            break
            
    total_segments_for_this_vid = len(temp_segment_jobs)
    if total_segments_for_this_vid == 0 and effective_total_output_frames_to_target_for_video > 0 :
        log_message("SEGMENT_DEFINE_FAILURE", video_name=original_basename, reason="No segments defined after loop, but_frames_were_expected.") # Modified message
    elif total_segments_for_this_vid > 0: # Only proceed if segments were actually defined
        for i_job in range(total_segments_for_this_vid):
            temp_segment_jobs[i_job]["total_segments"] = total_segments_for_this_vid
        segment_jobs.extend(temp_segment_jobs)
        log_message("SEGMENT_DEFINE_SUCCESS", num_segments=total_segments_for_this_vid, video_name=original_basename)
    # If no segments and no frames expected, this is also fine (e.g. process_length_overall = 0)
        
    return segment_jobs, base_job_info_for_video