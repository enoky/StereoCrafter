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
import logging # Import standard logging
import subprocess

# Configure a logger for this module
_logger = logging.getLogger(__name__)

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
    _logger.warning("OpenEXR/Imath libraries not found. EXR features will be limited/unavailable. Context: utils.py")

DEFAULT_SINGLE_IMAGE_CLIP_FRAMES = 5
if DEFAULT_SINGLE_IMAGE_CLIP_FRAMES <= 0:
    DEFAULT_SINGLE_IMAGE_CLIP_FRAMES = 1 # Safety fallback
    _logger.warning(f"Warning: Invalid single image clip frames requested ({DEFAULT_SINGLE_IMAGE_CLIP_FRAMES}), falling back to {1}.")

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

def get_segment_output_folder_name(original_video_basename: str) -> str:
    """Returns the standard name for a segment subfolder."""
    return f"{original_video_basename}_seg"

def get_segment_npz_output_filename(original_video_basename: str, segment_id: int, total_segments: int) -> str:
    """Returns the standard NPZ filename for a segment."""
    return f"{original_video_basename}_depth_{segment_id + 1}of{total_segments}.npz"

def get_full_video_output_filename(original_video_basename: str, extension: str = "mp4") -> str:
    """Returns the standard filename for a full video output."""
    return f"{original_video_basename}_depth.{extension}"

def get_image_sequence_metadata(folder_path: str, target_fps_from_gui: int) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int], Optional[int]]:
    """Gets metadata (frame count, fps, H, W) for an image sequence."""
    supported_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".exr")
    # Use glob for better pattern matching and case-insensitivity if needed, or stick to listdir for simplicity
    frames_found = []
    for ext in supported_exts:
        frames_found.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        frames_found.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}"))) # For case-insensitivity

    frames = sorted(list(set(frames_found))) # Remove duplicates and sort

    if not frames:
        _logger.warning(f"Image Sequence: No compatible image files found in folder '{folder_path}'. Supported extensions: {str(supported_exts)}.")
        return None, None, None, None, None

    try:
        first_frame_img = imageio.v2.imread(frames[0])
        h, w = first_frame_img.shape[:2]
    except Exception as e:
        _logger.error(f"Image Read Error: Could not read image file '{frames[0]}'. Error: {e}.")
        return None, None, None, None, None

    total_frames = len(frames)
    effective_fps = float(target_fps_from_gui) if target_fps_from_gui != -1 else 24.0
    _logger.debug(f"Image Sequence Metadata: Folder '{folder_path}', Frames: {total_frames}, Effective FPS: {effective_fps:.2f}, H: {h}, W: {w}.")
    return total_frames, effective_fps, h, w, None

def get_single_image_metadata(
    image_path: str, 
    gui_target_fps_setting: int
) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int], Optional[dict]]:
    """Gets metadata for a single image. Frame count uses DEFAULT_SINGLE_IMAGE_CLIP_FRAMES."""
    try:
        img = imageio.v2.imread(image_path)
        h, w = img.shape[:2]
    except Exception as e:
        _logger.error(f"Image Read Error: Could not read image file '{image_path}'. Error: {e}.")
        return None, None, None, None, None

    effective_fps = float(gui_target_fps_setting) if gui_target_fps_setting != -1 else 24.0
    num_generated_frames = DEFAULT_SINGLE_IMAGE_CLIP_FRAMES

    _logger.debug(f"Single Image Metadata: File '{image_path}', Frames for 1s clip: {num_generated_frames}, Effective FPS: {effective_fps:.2f}, H: {h}, W: {w}.")
    return num_generated_frames, effective_fps, h, w, None # Return None for stream_info

def get_sidecar_json_filename(base_filepath_with_ext: str) -> str:
    """Returns the corresponding .json sidecar filename for a given base file."""
    return os.path.splitext(base_filepath_with_ext)[0] + ".json"

def get_video_stream_info(video_path: str) -> Optional[dict]:
    """
    Extracts comprehensive video stream metadata using ffprobe.
    Returns a dict with relevant color properties, codec, pixel format, and HDR mastering metadata
    or None if ffprobe fails/info not found.
    Requires ffprobe to be installed and in PATH.
    This function *does not* show messageboxes; the caller should handle errors.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0", # Select the first video stream
        "-show_entries", "stream=codec_name,profile,pix_fmt,color_primaries,transfer_characteristics,color_space,"
        "r_frame_rate,width,height,nb_frames,duration,side_data_list", # Consolidated all relevant stream entries
        "-of", "json",
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', timeout=60)
        raw_stdout = result.stdout # <--- Capture stdout
        data = json.loads(raw_stdout) # <--- Use captured stdout
        # No debug print of raw stdout here
        
        stream_info = {}
        
        # --- Extract Stream Information ---
        # The -select_streams v:0 and consolidated -show_entries should give us one video stream here
        if "streams" in data and len(data["streams"]) > 0:
            s = data["streams"][0] # Focus on the first video stream (which should be v:0)
            
            # Directly extract the fields we need. Use .get() for safety.
            stream_info["codec_name"] = s.get("codec_name")
            stream_info["profile"] = s.get("profile")
            stream_info["pix_fmt"] = s.get("pix_fmt")
            stream_info["color_primaries"] = s.get("color_primaries")
            stream_info["transfer_characteristics"] = s.get("transfer_characteristics")
            stream_info["color_space"] = s.get("color_space")
            stream_info["r_frame_rate"] = s.get("r_frame_rate")
            stream_info["width"] = s.get("width")
            stream_info["height"] = s.get("height")
            stream_info["nb_frames"] = s.get("nb_frames")
            stream_info["duration"] = s.get("duration") # Duration might be here too with this syntax
            stream_info["nb_read_frames"] = s.get("nb_read_frames")

        # If nb_frames is missing or zero, re-run ffprobe with -count_frames
        if not stream_info.get("nb_frames") or int(stream_info.get("nb_frames", "0")) == 0:
            _logger.info(f"Rerunning ffprobe with -count_frames for {video_path} as nb_frames is zero or missing.")
            cmd_count_frames = [
                "ffprobe",
                "-count_frames",  # Force full decode to count frames
                "-v", "error",
                "-select_streams", "v:0",  # Select the first video stream
                "-show_entries", "stream=nb_read_frames",  # Only need frame count this time
                "-of", "json",
                video_path
            ]
            try:
                result_count_frames = subprocess.run(cmd_count_frames, capture_output=True, text=True, check=True,
                                                      encoding='utf-8', timeout=60)
                raw_stdout_count_frames = result_count_frames.stdout
                data_count_frames = json.loads(raw_stdout_count_frames)

                if "streams" in data_count_frames and len(data_count_frames["streams"]) > 0:
                    s = data_count_frames["streams"][0]
                    # Update nb_frames with the counted value, if available
                    if "nb_read_frames" in s:
                        stream_info["nb_frames"] = s["nb_read_frames"]
                        stream_info["nb_read_frames"] = s["nb_read_frames"]
                        _logger.info(
                            f"Successfully updated nb_frames to {stream_info['nb_frames']} using -count_frames for {video_path}."
                        )
                else:
                    _logger.warning(
                        f"No stream info found in -count_frames output for {video_path}."
                    )

            except subprocess.CalledProcessError as e:
                _logger.error(
                    f"ffprobe -count_frames failed for {video_path} (return code {e.returncode}):\n{e.stderr}"
                )
            except json.JSONDecodeError as e:
                _logger.error(
                    f"Failed to parse ffprobe -count_frames output for {video_path}: {e}. Raw output might be malformed JSON."
                )
                _logger.debug(
                    f"Raw ffprobe -count_frames stdout (if available): {result_count_frames.stdout if 'result_count_frames' in locals() else 'N/A'}"
                )
            except Exception as e:
                _logger.error(
                    f"An unexpected error occurred with ffprobe -count_frames for {video_path}: {e}",
                    exc_info=True
                )

        # --- After potentially using count_frames, prioritize nb_read_frames if available ---
        if stream_info.get("nb_read_frames"):
            # if stream_info.get("nb_read_frames")
            stream_info["nb_frames"] = stream_info["nb_read_frames"]

        # stream_info["nb_frames"] = s.get("nb_frames")



            stream_info["nb_read_frames"] = s.get("nb_read_frames") # <--- ADD THIS. Get the frame count with -count_frames
            
            # HDR mastering display and CLL metadata (from side_data_list within the stream)
            if s.get("side_data_list"):
                for sd in s["side_data_list"]:
                    if sd.get("side_data_type") == "Mastering display metadata":
                        stream_info["mastering_display_metadata"] = sd.get("mastering_display_metadata")
                    if sd.get("side_data_type") == "Content light level metadata":
                        stream_info["max_content_light_level"] = sd.get("max_content_light_level")

        # --- Guess nb_frames from duration * r_frame_rate if nb_frames is still missing/zero ---
        if not stream_info.get("nb_frames") or int(stream_info.get("nb_frames", "0")) == 0:
            if stream_info.get("duration") and stream_info.get("r_frame_rate"):
                _logger.debug(f"DEBUG: Attempting to guess frame count from duration and r_frame_rate for {video_path}")
                _logger.debug(f"DEBUG:   Duration = {stream_info.get('duration')}")
                _logger.debug(f"DEBUG:   r_frame_rate = {stream_info.get('r_frame_rate')}")

                if not stream_info.get("duration") or not stream_info.get("r_frame_rate"):
                    _logger.warning(f"ffprobe is missing duration OR frame rate; cannot reliably determine frame count for {video_path}")
                    return None, raw_stdout

                # [Validation] if duration is 0 and there is no nb_frames detected return None
                if stream_info.get("duration") == '0.0':
                    _logger.error(f"ffprobe reports zero duration and zero frames for {video_path}")
                    return None, raw_stdout
                try:
                    # [Validation] Validate that stream_info values are not null to allow the division calculation

                    duration_f = float(stream_info["duration"])
                    r_frame_rate_str = stream_info["r_frame_rate"].split('/')
                    if len(r_frame_rate_str) == 2 and float(r_frame_rate_str[1]) != 0:
                        fps_val = float(r_frame_rate_str[0]) / float(r_frame_rate_str[1])
                    else:
                        fps_val = float(r_frame_rate_str[0])
                    # Only guess if duration is reasonable and fps is not zero
                    if duration_f > 0 and fps_val > 0:
                        stream_info["nb_frames"] = str(round(duration_f * fps_val))
                        _logger.debug(f"Guessed nb_frames from duration*fps: {stream_info['nb_frames']}")
                except (ValueError, TypeError, ZeroDivisionError):
                    pass # Couldn't guess

        # Filter out empty strings/None values and values '0' or '0.0'
        filtered_info = {k: v for k, v in stream_info.items() if v is not None and str(v).strip() not in ["N/A", "und", "unknown", "0", "0.0"]}
        
        return filtered_info, raw_stdout if filtered_info else (None, raw_stdout)

    except subprocess.CalledProcessError as e:
        _logger.error(f"ffprobe failed for {video_path} (return code {e.returncode}):\n{e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        _logger.error(f"ffprobe timed out for {video_path}.")
        return None
    except json.JSONDecodeError as e:
        _logger.error(f"Failed to parse ffprobe output for {video_path}: {e}. Raw output might be malformed JSON.")
        _logger.debug(f"Raw ffprobe stdout (if available): {result.stdout if 'result' in locals() else 'N/A'}") # [Bug] There might be the possibility of result does not get defined and a NameError occurs
        return None
    except Exception as e:
        _logger.error(f"An unexpected error occurred with ffprobe for {video_path}: {e}", exc_info=True)
        return None

def define_video_segments(
    video_path_or_folder: str,
    original_basename: str,
    gui_target_fps_setting: int,
    gui_process_length_overall: int,
    gui_segment_output_window_frames: int,
    gui_segment_output_overlap_frames: int,
    source_type: str,
    gui_target_height_setting: int, # New: Pass GUI H/W settings
    gui_target_width_setting: int   # New: Pass GUI H/W settings
) -> Tuple[List[dict], Optional[dict]]:
    """
    Defines video segments based on input parameters.
    """
    segment_jobs = []
    base_job_info_for_video = {}

    total_raw_frames_in_original_video = 0
    original_video_fps = 30.0
    original_h_detected, original_w_detected = 0, 0 # From source (ffprobe/imageio)

    if source_type == "video_file":
        # Call the new read_video_frames just to get metadata without loading all frames
        # Use dummy target_height/width for this metadata read, as we only need original dimensions and FPS.
        # This will call ffprobe/decord internally.
        frames_numpy_dummy, fps_detected, h_orig, w_orig, _, _, video_stream_info_from_ffprobe, _ = read_video_frames(
            video_path_or_folder,
            process_length=-1, # Don't limit frames here
            target_fps=-1.0,   # Get the original FPS
            target_height=128, target_width=128, # Dummy values for metadata extraction
            start_frame_index=0,
            num_frames_to_load=-1 # Get total frames
        )
        total_raw_frames_in_original_video = frames_numpy_dummy.shape[0] if frames_numpy_dummy is not None else 0
        original_video_fps = fps_detected if fps_detected is not None and fps_detected > 0 else 24.0
        original_h_detected, original_w_detected = h_orig, w_orig

        if original_video_fps <= 0:
            _logger.warning(f"Segment Definition for {original_basename}: Invalid original FPS ({original_video_fps}) from video. Assuming 30 FPS.")
            original_video_fps = 30.0
    elif source_type == "image_sequence_folder":
        count, fps, h_orig, w_orig, _ = get_image_sequence_metadata(video_path_or_folder, gui_target_fps_setting)
        if count is None:
            _logger.error(f"Segment Definition for {original_basename}: Error getting metadata for image sequence folder {video_path_or_folder}")
            return [], None
        total_raw_frames_in_original_video = count
        original_video_fps = fps if fps is not None and fps > 0 else 24.0
        original_h_detected, original_w_detected = h_orig, w_orig
    elif source_type == "single_image_file":
        count, fps, h_orig, w_orig, _ = get_single_image_metadata( 
            video_path_or_folder, 
            gui_target_fps_setting
        )
        if count is None:
            _logger.error(f"Segment Definition for {original_basename}: Error getting metadata for single image {video_path_or_folder}")
            return [], None
        total_raw_frames_in_original_video = count
        original_video_fps = fps if fps is not None and fps > 0 else 24.0
        original_h_detected, original_w_detected = h_orig, w_orig
    else:
        _logger.error(f"Segment Definition for {original_basename}: Unknown source_type: {source_type}")
        return [], None

    base_job_info_for_video = {
        "video_path": video_path_or_folder,
        "source_type": source_type, 
        "gui_fps_setting_at_definition": gui_target_fps_setting,
        "original_basename": original_basename,
        "original_video_raw_frame_count": total_raw_frames_in_original_video,
        "original_video_fps": original_video_fps,
        "original_height": original_h_detected, # Use detected original height
        "original_width": original_w_detected,  # Use detected original width
        "video_stream_ffprobe_info": video_stream_info_from_ffprobe if source_type == "video_file" else None,
        "gui_target_height_setting": gui_target_height_setting, # Store GUI's target H/W
        "gui_target_width_setting": gui_target_width_setting,
    }

    fps_for_stride_calc = original_video_fps
    if source_type == "video_file":
        fps_for_stride_calc = original_video_fps if gui_target_fps_setting == -1 else gui_target_fps_setting
    
    if fps_for_stride_calc <= 0:
        fps_for_stride_calc = original_video_fps if original_video_fps > 0 else 24.0
    
    stride_for_fps_adjustment = max(round(original_video_fps / fps_for_stride_calc), 1)
    
    max_possible_output_frames_after_fps = (total_raw_frames_in_original_video + stride_for_fps_adjustment - 1) // stride_for_fps_adjustment
    
    effective_total_output_frames_to_target_for_video = max_possible_output_frames_after_fps
    if gui_process_length_overall != -1 and gui_process_length_overall < effective_total_output_frames_to_target_for_video:
        effective_total_output_frames_to_target_for_video = gui_process_length_overall
    
    if effective_total_output_frames_to_target_for_video <= 0:
        _logger.warning(f"Segment Definition for {original_basename}: Effective output frames is zero or less. Skipping segment definition.")
        return [], base_job_info_for_video

    _logger.debug(f"Defining segments for {original_basename} (up to {effective_total_output_frames_to_target_for_video} output frames from {min(effective_total_output_frames_to_target_for_video * stride_for_fps_adjustment, total_raw_frames_in_original_video)} raw frames).")


    if gui_segment_output_window_frames <= 0:
        _logger.error(f"Segment Definition for {original_basename}: Segment output frame count ({gui_segment_output_window_frames}) must be positive.")
        return [], base_job_info_for_video
    if gui_segment_output_overlap_frames < 0 or gui_segment_output_overlap_frames >= gui_segment_output_window_frames:
        _logger.error(f"Segment Definition for {original_basename}: Segment output frame overlap ({gui_segment_output_overlap_frames}) invalid for window {gui_segment_output_window_frames}.")
        return [], base_job_info_for_video

    segment_def_window_raw = gui_segment_output_window_frames * stride_for_fps_adjustment
    segment_def_overlap_raw = gui_segment_output_overlap_frames * stride_for_fps_adjustment
    advance_per_segment_raw = segment_def_window_raw - segment_def_overlap_raw
    
    effective_raw_video_length_to_consider = min(
        effective_total_output_frames_to_target_for_video * stride_for_fps_adjustment,
        total_raw_frames_in_original_video
    )

    if advance_per_segment_raw <= 0 and effective_raw_video_length_to_consider > segment_def_window_raw :
        _logger.warning(f"Segment Definition for '{original_basename}': Raw advance per segment is {advance_per_segment_raw} (window_raw: {segment_def_window_raw}). This is okay if total length ({effective_raw_video_length_to_consider}) is less than one window, otherwise it's an issue.")

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
        
        if advance_per_segment_raw <= 0:
            _logger.warning(f"Segment Definition for '{original_basename}': Segment raw advance ({advance_per_segment_raw}) is zero or negative. Breaking segment definition loop after first segment.")
            break 

        current_raw_frame_idx += advance_per_segment_raw
        if current_raw_frame_idx >= effective_raw_video_length_to_consider:
            break
            
    total_segments_for_this_vid = len(temp_segment_jobs)
    if total_segments_for_this_vid == 0 and effective_total_output_frames_to_target_for_video > 0 :
        _logger.warning(f"Segment Definition for {original_basename}: No segments defined after loop, but frames were expected. This might indicate an issue with window/overlap settings or video length.")
    elif total_segments_for_this_vid > 0:
        for i_job in range(total_segments_for_this_vid):
            temp_segment_jobs[i_job]["total_segments"] = total_segments_for_this_vid
        segment_jobs.extend(temp_segment_jobs)
        _logger.debug(f"Defined {total_segments_for_this_vid} segments for {original_basename}.")
        
    return segment_jobs, base_job_info_for_video

def normalize_video_data(
    video_data: np.ndarray,
    use_percentile_norm: bool,
    low_perc: float,
    high_perc: float
) -> np.ndarray:
    """Normalizes video data to the 0-1 range."""
    if video_data is None or video_data.size == 0:
        _logger.critical("CRITICAL: Cannot normalize empty video array.")
        raise ValueError("Cannot normalize empty video array.")

    _logger.debug(f"Normalizing video data. Shape: {video_data.shape}")
    
    normalized_video = video_data.copy().astype(np.float32)
    min_val_for_norm, max_val_for_norm = np.min(normalized_video), np.max(normalized_video)
    method_str = "percentile"

    if use_percentile_norm:
        if normalized_video.ndim > 0 and normalized_video.shape[0] > 2 and normalized_video.flatten().size > 20:
            min_val_for_norm = np.percentile(normalized_video.flatten(), low_perc)
            max_val_for_norm = np.percentile(normalized_video.flatten(), high_perc)
        else:
            _logger.debug(f"Normalization: Array too small for robust percentile ({low_perc}%/{high_perc}%), using absolute min/max.")
            method_str = "absolute (percentile fallback)"
    else:
        method_str = "absolute"

    _logger.debug(f"Normalizing video data. Shape: {video_data.shape}. Method: {method_str}. Range: {min_val_for_norm:.4f}-{max_val_for_norm:.4f}")

    if abs(max_val_for_norm - min_val_for_norm) < 1e-6:
        _logger.warning("Normalization: Range very small. Video appears flat.")
        flat_value = 0.5
        if (0.0 <= min_val_for_norm <= 1.0 and 0.0 <= max_val_for_norm <= 1.0 and abs(max_val_for_norm - min_val_for_norm) < 1e-7):
            flat_value = np.clip(min_val_for_norm, 0.0, 1.0)
        
        normalized_video = np.full_like(normalized_video, flat_value, dtype=np.float32)
        _logger.debug(f"Normalization: Video normalized to constant value: {flat_value:.2f}")
    else:
        normalized_video = (normalized_video - min_val_for_norm) / (max_val_for_norm - min_val_for_norm)
    
    normalized_video = np.clip(normalized_video, 0.0, 1.0)
    _logger.debug(f"Normalization: Global min/max after final clip: {np.min(normalized_video):.4f} / {np.max(normalized_video):.4f}")
    return normalized_video

def apply_gamma_correction_to_video(
    video_data: np.ndarray,
    gamma_value: float
) -> np.ndarray:
    """Applies gamma correction to video data."""
    processed_video = video_data.copy()
    actual_gamma = max(0.1, gamma_value)

    if abs(actual_gamma - 1.0) > 1e-3:
        _logger.debug(f"Applying Gamma ({actual_gamma:.2f}) to video.")
        processed_video = np.power(np.clip(processed_video, 0, 1), 1.0 / actual_gamma)
        processed_video = np.clip(processed_video, 0, 1)
    else:
        _logger.debug(f"Gamma value {actual_gamma:.2f} (effectively 1.0), no gamma transform applied.")
    return processed_video

def apply_dithering_to_video(
    video_data: np.ndarray,
    dither_strength_factor: float
) -> np.ndarray:
    """Applies dithering to video data."""
    processed_video = video_data.copy()
    _logger.debug("Applying dithering...")
    
    dither_range = (1.0 / 255.0) * dither_strength_factor
    noise = np.random.uniform(-dither_range, dither_range, processed_video.shape).astype(np.float32)
    processed_video = np.clip(processed_video + noise, 0, 1)
    
    _logger.debug(f"Applying dithering to video. Strength factor: {dither_strength_factor:.2f}, Dither range: {dither_range:.4f}")
    return processed_video

def load_json_file(filepath: str) -> Optional[dict]:
    """Loads data from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _logger.debug(f"Successfully loaded: {filepath}")
        return data
    except FileNotFoundError:
        _logger.error(f"File not found: {filepath}")
    except json.JSONDecodeError as e:
        _logger.error(f"Could not decode JSON from: {filepath}. Reason: {e}")
    except Exception as e:
        _logger.error(f"ERROR loading JSON from {filepath}: {e}")
    return None

def save_json_file(data: dict, filepath: str, indent: int = 4) -> bool:
    """Saves data to a JSON file."""
    try:
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        _logger.debug(f"Successfully saved: {os.path.basename(filepath)}") # Log only basename for brevity in repeated calls
        return True
    except TypeError as e:
        _logger.error(f"Failed to save: {filepath}. Reason: Data not JSON serializable: {e}")
    except (IOError, OSError) as e:
        _logger.error(f"Failed to save: {filepath}. Reason: {e}")
    except Exception as e:
        _logger.error(f"Failed to save: {filepath}. Reason: Unexpected error: {e}")
    return False

def read_video_frames(
    video_path: str,
    process_length: int = -1,
    target_fps: float = -1.0,
    target_height: int = 512,
    target_width: int = 768,
    start_frame_index: int = 0,
    num_frames_to_load: int = -1,
    cached_ffprobe_info: Optional[dict] = None
) -> Tuple[np.ndarray, float, int, int, int, int, Optional[dict], Optional[str]]:
    """
    Reads video frames using decord, optionally resizing and downsampling frame rate.
    Returns frames as a 4D float32 numpy array [T, H, W, C] normalized to 0-1,
    the actual output FPS, original video height/width, actual processed height/width,
    and video stream metadata.
    """
    _logger.debug(f"Reading video: {os.path.basename(video_path)}")

    # --- REPLACEMENT BLOCK START ---
    # Get video stream info: first try cached, then call ffprobe if not cached.
    video_stream_info_actual = None
    ffprobe_raw_stdout_safe = "" # Initialize to empty string for safety

    if cached_ffprobe_info:
        video_stream_info_actual = cached_ffprobe_info 
        _logger.debug(f"Reusing cached ffprobe info for {os.path.basename(video_path)} (skipping ffprobe call).")
    else:
        ffprobe_result_tuple = get_video_stream_info(video_path)
        if ffprobe_result_tuple:
            video_stream_info_actual, raw_stdout_from_func = ffprobe_result_tuple
            ffprobe_raw_stdout_safe = raw_stdout_from_func if raw_stdout_from_func is not None else "" # Ensure it's a string
            _logger.debug(f"FFprobe called for {os.path.basename(video_path)} (no cached info available).")
            _logger.debug(f"DEBUG: Raw ffprobe stdout for {os.path.basename(video_path)}:\n{ffprobe_raw_stdout_safe}")
        else:
            _logger.warning(f"Failed to get video stream info for {os.path.basename(video_path)}.")
    
    video_stream_info = video_stream_info_actual
    ffprobe_raw_stdout = ffprobe_raw_stdout_safe # Assign to the variable returned by the function

    original_height_detected = 0
    original_width_detected = 0
    original_fps_detected = 0.0
    num_total_frames_detected = 0

    if video_stream_info:
        try:
            original_width_detected = int(video_stream_info.get("width", 0))
            original_height_detected = int(video_stream_info.get("height", 0))
            num_total_frames_detected = int(video_stream_info.get("nb_frames", 0))
            
            r_frame_rate_str = video_stream_info.get("r_frame_rate", "0/1").split('/')
            if len(r_frame_rate_str) == 2 and float(r_frame_rate_str[1]) != 0:
                original_fps_detected = float(r_frame_rate_str[0]) / float(r_frame_rate_str[1])
            elif len(r_frame_rate_str) == 1:
                original_fps_detected = float(r_frame_rate_str[0])
            _logger.debug(f"FFprobe detected: {original_width_detected}x{original_height_detected} @ {original_fps_detected:.2f} FPS, {num_total_frames_detected} frames.")
        except (ValueError, TypeError, ZeroDivisionError) as e:
            _logger.warning(f"Failed to parse ffprobe stream info for {os.path.basename(video_path)}: {e}. Falling back to Decord for metadata.")
            video_stream_info = None # Invalidate ffprobe info to force Decord fallback
    
    # If ffprobe failed or parsing failed, use Decord as a fallback for initial metadata
    if not video_stream_info:
        try:
            temp_reader = VideoReader(video_path, ctx=cpu(0))
            num_total_frames_detected = len(temp_reader)
            if num_total_frames_detected > 0:
                first_frame_shape = temp_reader.get_batch([0]).shape
                original_height_detected = first_frame_shape[1]
                original_width_detected = first_frame_shape[2]
            original_fps_detected = temp_reader.get_avg_fps()
            del temp_reader
            gc.collect()
            _logger.debug(f"Decord fallback detected: {original_width_detected}x{original_height_detected} @ {original_fps_detected:.2f} FPS, {num_total_frames_detected} frames.")
        except Exception as e:
            _logger.error(f"Failed to get initial metadata from Decord for {os.path.basename(video_path)}: {e}", exc_info=True)
            return np.empty((0, 0, 0, 0), dtype=np.float32), 0.0, 0, 0, 0, 0, None, None

    if num_total_frames_detected == 0:
        _logger.warning(f"No frames detected in {os.path.basename(video_path)}.")
        return np.empty((0, 0, 0, 0), dtype=np.float32), 0.0, original_height_detected, original_width_detected, 0, 0, video_stream_info, None

    if original_height_detected == 0 or original_width_detected == 0:
        _logger.warning(f"Original dimensions could not be detected for {os.path.basename(video_path)}. Setting to default 128x128.")
        original_height_detected = 128
        original_width_detected = 128


    # Determine height/width for Decord based on GUI's target_height/width
    # Ensure they are multiples of 64 and at least 64
    final_height_for_decord = max(64, round(target_height / 64) * 64)
    final_width_for_decord = max(64, round(target_width / 64) * 64)
    _logger.debug(f"Targeting final processing resolution (rounded to mult of 64): {final_width_for_decord}x{final_height_for_decord}")

    try:
        vid_reader = VideoReader(video_path, ctx=cpu(0), width=final_width_for_decord, height=final_height_for_decord)
    except Exception as e:
        _logger.error(f"Decord init EXCEPTION: {e}")
        _logger.error(f"Failed to initialize Decord VideoReader for {os.path.basename(video_path)} with target resolution {final_width_for_decord}x{final_height_for_decord}: {e}", exc_info=True)
        return np.empty((0, 0, 0, 0), dtype=np.float32), 0.0, original_height_detected, original_width_detected, 0, 0, video_stream_info, None

    # Determine output FPS: prioritize GUI target, then ffprobe, then decord
    actual_output_fps = original_fps_detected if target_fps == -1.0 else target_fps
    if actual_output_fps <= 0:
        actual_output_fps = 23.976 # Hardcoded fallback if all else fails
        _logger.warning(f"Could not determine valid FPS for {os.path.basename(video_path)}. Falling back to {actual_output_fps:.2f} FPS.")

    # No specific FPS snapping here; ffprobe's original resolution is kept or user target is used.
    # mediapy typically handles standard fractional FPS values well.

    # Stride calculation based on detected original FPS and final output FPS
    stride = 1
    if original_fps_detected > 0 and actual_output_fps > 0:
        stride = max(round(original_fps_detected / actual_output_fps), 1)
    
    _logger.debug(f"FPS: Original {original_fps_detected:.2f}, Target Output {actual_output_fps:.2f}, Stride {stride}.")


    # Determine the end frame for this segment/full video
    end_frame_exclusive = num_total_frames_detected
    if num_frames_to_load != -1: # If segment or process_length limit from GUI
        end_frame_exclusive = min(start_frame_index + num_frames_to_load, num_total_frames_detected)

    # Generate indices respecting start_frame_index, num_frames_to_load, and stride
    frames_idx = list(range(start_frame_index, end_frame_exclusive, stride))

    if process_length != -1 and process_length < len(frames_idx): # Overall process_length limit from GUI
        frames_idx = frames_idx[:process_length]
        _logger.info(f"Limiting to {len(frames_idx)} frames based on process_length parameter.")
    
    if not frames_idx:
        _logger.warning(f"No frames selected for processing after stride and process_length filters for {os.path.basename(video_path)}.")
        return np.empty((0, 0, 0, 0), dtype=np.float32), 0.0, original_height_detected, original_width_detected, final_height_for_decord, final_width_for_decord, video_stream_info, None

    _logger.debug(f"Loading {len(frames_idx)} frames using Decord for {os.path.basename(video_path)}.")
    frames_batch = vid_reader.get_batch(frames_idx)
    frames_numpy = frames_batch.asnumpy().astype("float32") / 255.0 # Normalize to 0-1 float32
    _logger.debug(f"Successfully Loaded batch, frames_numpy {frames_numpy.shape}")

    del vid_reader
    gc.collect()

    # The actual processed height/width are what Decord delivered, which should be final_height/width
    actual_processed_height = frames_numpy.shape[1]
    actual_processed_width = frames_numpy.shape[2]
    _logger.debug(f"Read {len(frames_idx)} frames. Original: {original_width_detected}x{original_height_detected}, Decord Processing: {actual_processed_width}x{actual_processed_height} (Final).")

    return frames_numpy, actual_output_fps, original_height_detected, original_width_detected, actual_processed_height, actual_processed_width, video_stream_info, ffprobe_raw_stdout

def save_video(video_frames: Union[List[np.ndarray], List[PIL.Image.Image], np.ndarray], output_video_path: str = None,
               fps: Union[int, float] = 10.0, crf: int = 18, output_format: Optional[str] = None) -> str:
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name
    elif output_format == "main10_mp4" and not output_video_path.lower().endswith(".mp4"):
        _logger.warning(f"Saving HEVC to an .mp4 container, but the output path '{output_video_path}' does not end with .mp4. This might lead to issues if an explicit format isn't forced.")

    # Determine scaling factor and dtype based on output format
    target_max_val = 255.0  # Default for 8-bit (H.264)
    target_dtype = np.uint8
    
    # Use 16-bit scale for 10-bit encoding
    if output_format == "main10_mp4":
        target_max_val = 65535.0 
        target_dtype = np.uint16
    # elif output_format in ["exr", "exr_sequence", "png_sequence"]:
    #    # These formats are often handled later/separately in a post-processing step
    #    # and should be left as 0-1 float or converted specifically.
    #    # Since save_video is for mp4 by default, we'll focus on that.
    #    pass

    # Frame conversion logic (Modified to use target_max_val and target_dtype)
    if isinstance(video_frames, np.ndarray):
        # We assume video_frames is already clipped to 0-1 float if it came from the pipe/merge
        if video_frames.dtype == np.float32 or video_frames.dtype == np.float64:
             video_frames = (video_frames * target_max_val).astype(target_dtype)

    elif isinstance(video_frames, list) and len(video_frames) > 0 and isinstance(video_frames[0], np.ndarray):
        processed_frames = []
        for frame in video_frames:
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                processed_frames.append((frame * target_max_val).astype(target_dtype))
            elif frame.dtype == np.uint8 or frame.dtype == np.uint16:
                processed_frames.append(frame)
            else:
                _logger.error(f"Unsupported numpy array dtype in list for video saving: {frame.dtype}")
                raise ValueError(f"Unsupported numpy array dtype in list: {frame.dtype}")
        video_frames = processed_frames
        
    elif isinstance(video_frames, list) and len(video_frames) > 0 and isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    elif isinstance(video_frames, list) and len(video_frames) == 0:
        _logger.warning("Empty list of frames provided. Cannot save video.")
        return output_video_path
    else:
        _logger.error("video_frames must be a list/array of np.ndarray or a list of PIL.Image.Image for saving.")
        raise ValueError("video_frames must be a list/array of np.ndarray or a list of PIL.Image.Image")

    mediapy_kwargs = {'fps': fps}
    ffmpeg_custom_args = [] # List to hold specific ffmpeg arguments

    if output_format == "main10_mp4" and output_video_path.lower().endswith(".mp4"):
        mediapy_kwargs['codec'] = 'libx265' # Tell mediapy to use libx265
        ffmpeg_custom_args.extend(['-pix_fmt', 'yuv420p10le']) # Pass pix_fmt via ffmpeg_args
        ffmpeg_custom_args.extend(['-tag:v', 'hvc1']) # Common tag for HEVC in MP4
        mediapy_kwargs['crf'] = crf
        _logger.debug(f"Attempting to save HEVC Main10 (libx265) to: {output_video_path} with mediapy_kwargs: {mediapy_kwargs}, ffmpeg_args: {ffmpeg_custom_args}")
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
        _logger.error(f"Error writing video to {output_video_path} using mediapy: {e}. Format details: Requested format: {output_format}, mediapy_kwargs: {mediapy_kwargs}")
        raise
    return output_video_path

class ColorMapper:
    def __init__(self, colormap: str = "inferno"):
        # other colors = viridis, plasma, inferno, magma
        self.colormap_name = colormap
        self._cmap_data = None

    def _get_cmap_data(self):
        if self._cmap_data is None:
            try:
                import matplotlib.cm as cm_mpl
                self._cmap_data = torch.tensor(cm_mpl.get_cmap(self.colormap_name).colors)
            except ImportError:
                _logger.warning("Matplotlib.cm not found. ColorMapper will use a basic grayscale fallback.")
                # Fallback to a very simple grayscale if matplotlib is not available
                # This is a basic fallback, not a full replacement.
                ramp = torch.linspace(0, 1, 256)
                self._cmap_data = torch.stack([ramp, ramp, ramp], dim=1) # (N, 3)
        return self._cmap_data

    def apply(self, image: torch.Tensor, v_min=None, v_max=None):
        if image.ndim not in [2,3]:
            _logger.error(f"ColorMapper.apply: Image must be 2D or 3D, got {image.ndim}D")
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
        _logger.error(f"vis_sequence_depth: Input depths must be a 3D array (T, H, W), got {depths.ndim}D")
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
                                  output_format: str = "mp4") -> Tuple[Optional[str], Optional[str]]:
    try:
        save_video(depth_frames_normalized, output_filepath, fps=fps, output_format=output_format)
        return output_filepath, None
    except Exception as e:
        _logger.error(f"Error saving MP4 visual to {output_filepath}: {e}, requested format: {output_format}")
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
        _logger.debug(f"Successfully saved PNG sequence to {png_dir_path}")
        return png_dir_path, None
    except Exception as e:
        _logger.error(f"Error saving PNG sequence to {png_dir_path}: {e}")
        return None, str(e)

def save_depth_visual_as_exr_sequence_util(
    depth_frames_normalized: np.ndarray, 
    output_dir_base: str, 
    base_filename_no_ext: str
) -> Tuple[Optional[str], Optional[str]]:
    if not _OPENEXR_AVAILABLE_IN_UTILS:
        _logger.warning("OpenEXR libraries not available in utils.py for EXR sequence saving.")
        return None, "OpenEXR libraries not available in utils.py for EXR sequence saving."

    exr_sequence_output_dir = "unknown_path_exr_seq" # Default for logging if error before assignment
    try:
        if depth_frames_normalized.ndim != 3: # T, H, W
             err_msg = f"EXR sequence expects 3D array (T,H,W), got {depth_frames_normalized.ndim}D"
             _logger.error(f"Error saving EXR sequence (path: {exr_sequence_output_dir}): {err_msg}")
             return None, err_msg
        
        num_frames, height, width = depth_frames_normalized.shape
        if num_frames == 0:
            err_msg = "No frames to save in EXR sequence."
            _logger.warning(f"Error saving EXR sequence (path: {exr_sequence_output_dir}): {err_msg}")
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
                    _logger.error(f"Error saving frame to EXR file '{output_exr_filepath}': {err_msg_frame}")
                    continue 
                
                exr_file = OpenEXR.OutputFile(output_exr_filepath, header)
                exr_file.writePixels(pixel_data)
                exr_file.close()
            except Exception as frame_ex:
                _logger.error(f"Error saving frame to EXR file '{output_exr_filepath}': {frame_ex}")
                
        _logger.debug(f"Successfully saved EXR sequence to {exr_sequence_output_dir}")
        return exr_sequence_output_dir, None 
    except Exception as e:
        path_for_log = exr_sequence_output_dir if 'exr_sequence_output_dir' in locals() and exr_sequence_output_dir else "unknown_path_exr_seq"
        _logger.error(f"Error saving EXR sequence (path: {path_for_log}): {e}")
        return None, str(e)

def save_depth_visual_as_single_exr_util(
    first_depth_frame_normalized: np.ndarray, 
    output_dir_base: str, 
    base_filename_no_ext: str
) -> Tuple[Optional[str], Optional[str]]:
    if not _OPENEXR_AVAILABLE_IN_UTILS:
        _logger.warning("OpenEXR libraries not available in utils.py for single EXR saving.")
        return None, "OpenEXR libraries not available in utils.py for single EXR saving."

    output_exr_filepath = "unknown_path_single_exr.exr" # Default for logging
    try:
        if first_depth_frame_normalized is None or first_depth_frame_normalized.size == 0:
            err_msg = "No frame data to save for single EXR"
            _logger.error(f"Error saving single EXR (filepath: {output_exr_filepath}): {err_msg}")
            return None, err_msg
        
        frame_float32 = first_depth_frame_normalized.astype(np.float32)
        
        if frame_float32.ndim == 2: # Grayscale (H, W)
            height, width = frame_float32.shape
        elif frame_float32.ndim == 3 and (frame_float32.shape[-1] == 1 or frame_float32.shape[-1] == 3) : # (H,W,1) or (H,W,3)
            height, width = frame_float32.shape[:2]
        else:
            err_msg = f"Unsupported frame shape for single EXR: {frame_float32.shape}"
            _logger.error(f"Error saving single EXR (filepath: {output_exr_filepath}): {err_msg}")
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
        
        _logger.debug(f"Successfully saved single EXR to {output_exr_filepath}")
        return output_exr_filepath, None
    except Exception as e:
        path_for_log = output_exr_filepath if 'output_exr_filepath' in locals() and output_exr_filepath else "unknown_path_single_exr.exr"
        _logger.error(f"Error saving single EXR (filepath: {path_for_log}): {e}")
        return None, str(e)

def read_image_sequence_as_frames(
    folder_path: str,
    num_frames_to_load: int, # Renamed from process_length for clarity (this is for the specific segment/call)
    target_height: int, target_width: int,
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
        _logger.warning(f"Image Sequence Load: No compatible image files found in folder '{folder_path}'. Supported extensions: {supported_exts}.")
        return None, None, None

    # Determine the slice of image paths for the current segment
    # Ensure start_index is within bounds
    if start_index < 0 or start_index >= len(all_image_paths_sorted):
        _logger.error(f"Image Sequence Load: Invalid start_index {start_index} for folder '{folder_path}'. Total images: {len(all_image_paths_sorted)}.")
        return None, None, None

    # num_frames_to_load: if -1, load all from start_index. Otherwise, load specified number.
    end_index: Optional[int]
    if num_frames_to_load == -1: # Load all remaining frames
        end_index = len(all_image_paths_sorted)
    else:
        end_index = start_index + num_frames_to_load
    
    # Slice the list of paths for the current segment
    image_paths_for_segment = all_image_paths_sorted[start_index:end_index]

    if not image_paths_for_segment:
        _logger.warning(f"Image Sequence Load: No image frames found for segment in '{folder_path}' (start: {start_index}, num_to_load: {num_frames_to_load}, end_calc: {end_index}, total_in_folder: {len(all_image_paths_sorted)}).")
        return None, None, None

    loaded_frames_list = []
    original_h, original_w = 0, 0 # For the first frame of the *sequence* (not necessarily segment)
    target_h, target_w = 0, 0   # Target dimensions after resizing

    # Get original dimensions from the very first image of the sequence for consistent resizing
    # This is important if segments are processed independently and need same target res.
    try:
        # Calculate target_h, target_w based on direct inputs, ensuring divisibility by 64 and min size
        target_h = max(64, round(target_height / 64) * 64)
        target_w = max(64, round(target_width / 64) * 64)
        
        # Store the originally detected H,W of the sequence (not the target)
        original_h, original_w = ref_h, ref_w

    except Exception as e_ref:
        _logger.error(f"Image Sequence Load: Error reading reference frame '{all_image_paths_sorted[0] if all_image_paths_sorted else 'N/A'}' for consistent resizing. Error: {e_ref}")
        return None, None, None


    for i, frame_path in enumerate(image_paths_for_segment):
        try:
            img = imageio.v2.imread(frame_path)
            
            # Resize if necessary to target_h, target_w (calculated once above)
            if img.shape[0] != target_h or img.shape[1] != target_w:
                pil_img = PIL.Image.fromarray(img)
                resized_pil_img = pil_img.resize((target_w, target_h), PIL.Image.LANCZOS)
                img = np.array(resized_pil_img)

            if img.ndim == 2: 
                img = np.stack([img]*3, axis=-1) 
            if img.shape[2] == 4: 
                img = img[..., :3]

            loaded_frames_list.append((img.astype(np.float32) / 255.0))
        except Exception as e:
            _logger.warning(f"Image Sequence Load: Error reading frame '{frame_path}'. Error: {e}. Skipping frame.")
            continue
    
    if not loaded_frames_list:
        _logger.error(f"Image Sequence Load: Failed to load any frames from '{folder_path}' for segment (start: {start_index}, num: {num_frames_to_load}).")
        return None, None, None

    frames_array = np.stack(loaded_frames_list, axis=0)
    _logger.debug(f"Image Sequence Load: Successfully loaded {frames_array.shape[0]} frames from '{folder_path}' (H:{frames_array.shape[1]}, W:{frames_array.shape[2]}, Segment Start Idx: {start_index}).")
    return frames_array, original_h, original_w

def create_frames_from_single_image(
    image_path: str, 
    num_frames_to_generate: int, 
    target_height: int, target_width: int
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[int]]: # frames, original_h, original_w
    """Creates a sequence of identical frames from a single image."""
    try:
        img_arr = imageio.v2.imread(image_path)
        original_h, original_w = img_arr.shape[:2]

        target_h = max(64, round(target_height / 64) * 64)
        target_w = max(64, round(target_width / 64) * 64)
        # target_h = max(64, target_h)
        # target_w = max(64, target_w)

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
        _logger.info(f"Single Image: Generated {frames_array.shape[0]} frames for clip from '{image_path}' (H:{frames_array.shape[1]}, W:{frames_array.shape[2]}).")
        return frames_array, original_h, original_w
    except Exception as e:
        _logger.error(f"Single Image: Error generating frames from '{image_path}'. Error: {e}.")
        return None, None, None