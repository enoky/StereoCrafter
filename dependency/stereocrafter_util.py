import os
import glob
import json
import logging
import shutil
import threading
import tkinter as tk # Required for Tooltip class
from tkinter import Toplevel, Label # Required for Tooltip class
from typing import Optional, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
import subprocess
import cv2
import gc
import time

# --- Configure Logging ---
# Only configure basic logging if no handlers are already set up.
# This prevents duplicate log messages if a calling script configures logging independently.
if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def set_util_logger_level(level):
    """Sets the logging level for the 'stereocrafter_util' logger."""
    logger.setLevel(level)
    # If basicConfig was already called, its handlers might not update automatically.
    # Ensure handlers also reflect the new level.
    for handler in logger.handlers:
        handler.setLevel(level)
        
# --- Global Flags ---
CUDA_AVAILABLE = False

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

def check_cuda_availability():
    """
    Checks if CUDA is available via PyTorch and if nvidia-smi can run.
    Sets the global CUDA_AVAILABLE flag.
    """
    global CUDA_AVAILABLE
    if torch.cuda.is_available():
        logger.info("PyTorch reports CUDA is available.")
        try:
            # Further check with nvidia-smi for robustness
            subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=5, encoding='utf-8')
            logger.info("CUDA detected (nvidia-smi also ran successfully). NVENC can be used.")
            CUDA_AVAILABLE = True
        except FileNotFoundError:
            logger.warning("nvidia-smi not found. CUDA is reported by PyTorch but NVENC availability cannot be fully confirmed. Proceeding with PyTorch's report.")
            CUDA_AVAILABLE = True # Rely on PyTorch if nvidia-smi not found
        except subprocess.CalledProcessError:
            logger.warning("nvidia-smi failed. CUDA is reported by PyTorch but NVENC availability cannot be fully confirmed. Proceeding with PyTorch's report.")
            CUDA_AVAILABLE = True # Rely on PyTorch if nvidia-smi fails
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi check timed out. CUDA is reported by PyTorch but NVENC availability cannot be fully confirmed. Proceeding with PyTorch's report.")
            CUDA_AVAILABLE = True # Rely on PyTorch if nvidia-smi times out
        except Exception as e:
            logger.error(f"Unexpected error during nvidia-smi check: {e}. Relying on PyTorch's report for CUDA.")
            CUDA_AVAILABLE = True # Rely on PyTorch as a fallback
    else:
        logger.info("PyTorch reports CUDA is NOT available. NVENC will not be used.")
        CUDA_AVAILABLE = False

def release_cuda_memory():
    """Releases GPU memory and performs garbage collection."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared.")
        gc.collect()
        logger.debug("Python garbage collector invoked.")
    except Exception as e:
        logger.error(f"Error releasing VRAM or during garbage collection: {e}", exc_info=True)

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
        "-show_entries", "stream=codec_name,profile,pix_fmt,color_primaries,transfer_characteristics,color_space,r_frame_rate",
        "-show_entries", "side_data=mastering_display_metadata,max_content_light_level", # ADDED entries
        "-of", "json",
        video_path
    ]
    
    try:
        # Check if ffprobe is available without showing a messagebox
        subprocess.run(["ffprobe", "-version"], check=True, capture_output=True, text=True, encoding='utf-8', timeout=10)
    except FileNotFoundError:
        logger.error("ffprobe not found. Please ensure FFmpeg is installed and in your system PATH.")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running ffprobe check: {e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        logger.error("ffprobe check timed out.")
        return None


    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', timeout=60)
        data = json.loads(result.stdout)
        
        stream_info = {}
        if "streams" in data and len(data["streams"]) > 0:
            s = data["streams"][0]
            # Common video stream properties
            for key in ["codec_name", "profile", "pix_fmt", "color_primaries", "transfer_characteristics", "color_space", "r_frame_rate"]:
                if key in s:
                    stream_info[key] = s[key]
            
            # HDR mastering display and CLL metadata (often in side_data_list, but sometimes also directly in stream)
            # Prioritize stream-level if available, otherwise check side_data_list
            if "mastering_display_metadata" in s:
                stream_info["mastering_display_metadata"] = s["mastering_display_metadata"]
            if "max_content_light_level" in s:
                stream_info["max_content_light_level"] = s["max_content_light_level"]

        # Check side_data_list if stream-level properties weren't found or for additional data
        if "side_data_list" in data:
            for sd in data["side_data_list"]:
                if "mastering_display_metadata" in sd and "mastering_display_metadata" not in stream_info:
                    stream_info["mastering_display_metadata"] = sd["mastering_display_metadata"]
                if "max_content_light_level" in sd and "max_content_light_level" not in stream_info:
                    stream_info["max_content_light_level"] = sd["max_content_light_level"]

        # Filter out empty strings/None/N/A values
        filtered_info = {k: v for k, v in stream_info.items() if v and v not in ["N/A", "und", "unknown"]}
        return filtered_info if filtered_info else None

    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed for {video_path} (return code {e.returncode}):\n{e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe timed out for {video_path}.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse ffprobe output for {video_path}: {e}")
        logger.debug(f"Raw ffprobe stdout: {result.stdout}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred with ffprobe for {video_path}: {e}", exc_info=True)
        return None

def draw_progress_bar(current, total, bar_length=50, prefix='Progress:', suffix=''):
    """
    Draws an ASCII progress bar in the console, overwriting the same line.
    Adds a newline only when 100% complete. This uses `print` for direct console output.
    """
    if total == 0:
        print(f"\r{prefix} [Skipped (Total 0)] {suffix}", end='')
        return

    percent = 100 * (current / float(total))
    filled_length = int(round(bar_length * current / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    # Format the suffix for completion
    actual_suffix = suffix
    if current == total:
        actual_suffix = "Complete"

    print(f'\r{prefix} |{bar}| {percent:.1f}% {actual_suffix}', end='', flush=True)

    if current == total:
        print() # Add a final newline when done

def read_video_frames_decord(
    video_path: str,
    process_length: int = -1,
    target_fps: float = -1.0,
    set_res_width: Optional[int] = None,
    set_res_height: Optional[int] = None,
    decord_ctx=cpu(0)
) -> Tuple[np.ndarray, float, int, int, int, int, Optional[dict]]:
    """
    Reads video frames using decord, optionally resizing and downsampling frame rate.
    Returns frames as a 4D float32 numpy array [T, H, W, C] normalized to 0-1,
    the actual output FPS, original video height/width, actual processed height/width,
    and video stream metadata.
    """
    logger.info(f"Reading video: {os.path.basename(video_path)}")

    # Get video stream info first for FPS detection
    video_stream_info = get_video_stream_info(video_path)

    # Use a dummy VideoReader to get original dimensions without loading all frames
    temp_reader = VideoReader(video_path, ctx=cpu(0))
    original_height, original_width = temp_reader.get_batch([0]).shape[1:3]
    del temp_reader # Release immediately

    height_for_decord = original_height
    width_for_decord = original_width

    if set_res_width is not None and set_res_width > 0 and \
       set_res_height is not None and set_res_height > 0:
        height_for_decord = set_res_height
        width_for_decord = set_res_width
        logger.info(f"Targeting specific resolution for decord: {width_for_decord}x{height_for_decord}")
    else:
        logger.info(f"Using original video resolution for decord: {original_width}x{original_height}")

    # Initialize VideoReader with potential target resolution
    vid_reader = VideoReader(video_path, ctx=decord_ctx, width=width_for_decord, height=height_for_decord)
    num_total_frames = len(vid_reader)

    if num_total_frames == 0:
        logger.warning(f"No frames found in {video_path}.")
        return np.empty((0, 0, 0, 0), dtype=np.float32), 0.0, original_height, original_width, 0, 0, video_stream_info

    # Determine FPS: Use ffprobe's r_frame_rate if reliable, otherwise decord's avg_fps, or target_fps
    actual_output_fps = 0.0
    if target_fps != -1.0 and target_fps > 0:
        actual_output_fps = target_fps
        logger.info(f"Using user-specified target FPS: {actual_output_fps:.2f}")
    elif video_stream_info and "r_frame_rate" in video_stream_info:
        try:
            r_frame_rate_str = video_stream_info["r_frame_rate"].split('/')
            if len(r_frame_rate_str) == 2:
                actual_output_fps = float(r_frame_rate_str[0]) / float(r_frame_rate_str[1])
            else:
                actual_output_fps = float(r_frame_rate_str[0])
            logger.info(f"Using ffprobe FPS: {actual_output_fps:.2f} for {os.path.basename(video_path)}")
        except (ValueError, ZeroDivisionError):
            actual_output_fps = vid_reader.get_avg_fps()
            logger.warning(f"Failed to parse ffprobe FPS. Falling back to Decord avg_fps: {actual_output_fps:.2f}")
    else:
        actual_output_fps = vid_reader.get_avg_fps()
        logger.info(f"Using Decord avg_fps: {actual_output_fps:.2f} for {os.path.basename(video_path)}")

    stride = max(round(vid_reader.get_avg_fps() / actual_output_fps), 1)
    frames_idx = list(range(0, num_total_frames, stride))

    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
        logger.info(f"Limiting to {len(frames_idx)} frames based on process_length parameter.")
    
    if not frames_idx:
        logger.warning(f"No frames selected for processing after stride and process_length filters.")
        return np.empty((0, 0, 0, 0), dtype=np.float32), 0.0, original_height, original_width, 0, 0, video_stream_info

    frames_batch = vid_reader.get_batch(frames_idx)
    frames_numpy = frames_batch.asnumpy().astype("float32") / 255.0 # Normalize to 0-1 float32

    # Get actual processed height/width after Decord (might differ from target if source is smaller)
    actual_processed_height, actual_processed_width = frames_numpy.shape[1:3]
    logger.info(f"Read {len(frames_idx)} frames. Original: {original_width}x{original_height}, Processed: {actual_processed_width}x{actual_processed_height}")

    return frames_numpy, actual_output_fps, original_height, original_width, actual_processed_height, actual_processed_width, video_stream_info

def encode_frames_to_mp4(
    temp_png_dir: str,
    final_output_mp4_path: str,
    fps: float,
    total_output_frames: int,
    video_stream_info: Optional[dict],
    stop_event: threading.Event, # Added stop_event to allow early exit during encoding
    sidecar_json_data: Optional[dict] = None,
    user_output_crf: Optional[int] = None # NEW: Add this parameter
) -> bool:
    """
    Encodes a sequence of 16-bit PNG frames from a temporary directory into an MP4 video
    using FFmpeg, attempting to preserve color metadata and using NVENC if available.
    Also creates a sidecar JSON file if sidecar_json_data is provided.
    Returns True on success, False on failure or stop.
    """
    if total_output_frames == 0:
        logger.warning(f"No frames to encode for {os.path.basename(final_output_mp4_path)}. Skipping encoding.")
        if os.path.exists(temp_png_dir):
            shutil.rmtree(temp_png_dir)
        return False

    logger.info(f"Starting FFmpeg encoding from PNG sequence to {os.path.basename(final_output_mp4_path)}")
    logger.info(f"Input PNG directory: {temp_png_dir}")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y", # Overwrite output files without asking
        "-framerate", str(fps), # Input framerate for the PNG sequence
        "-i", os.path.join(temp_png_dir, "%05d.png"), # Input PNG sequence pattern
    ]

    # --- Determine Output Codec, Bit-Depth, and Quality ---
    output_codec = "libx264" # Default to H.264 CPU encoder
    output_pix_fmt = "yuv420p" # Default to 8-bit
    default_cpu_crf = "23" # Default CRF for H.264 (lower is better quality)
    output_profile = "main"
    x265_params = [] # For specific x265 parameters

    nvenc_preset = "medium" # Default NVENC preset (e.g., fast, medium, slow, quality)
    default_nvenc_cq = "23" # Constant Quality value for NVENC (lower is better quality)

    # NEW: Apply user-specified CRF if provided
    if user_output_crf is not None and user_output_crf >= 0:
        logger.info(f"Using user-specified output CRF: {user_output_crf}")
        default_cpu_crf = str(user_output_crf)
        default_nvenc_cq = str(user_output_crf) # Assume user CRF applies to NVENC CQ as well for simplicity
    else:
        logger.info("Using auto-determined output CRF.")

    is_hdr_source = False
    original_codec_name = video_stream_info.get("codec_name") if video_stream_info else None
    original_pix_fmt = video_stream_info.get("pix_fmt") if video_stream_info else None

    if video_stream_info:
        if video_stream_info.get("color_primaries") == "bt2020" and \
           video_stream_info.get("transfer_characteristics") == "smpte2084":
            is_hdr_source = True
            logger.info("Detected HDR source. Targeting HEVC 10-bit HDR output.")

    is_original_10bit_or_higher = False
    if original_pix_fmt:
        if "10" in original_pix_fmt or "12" in original_pix_fmt or "16" in original_pix_fmt:
            is_original_10bit_or_higher = True

    if is_hdr_source:
        output_codec = "libx265"
        if CUDA_AVAILABLE:
            output_codec = "hevc_nvenc"
            logger.info("    (Using hevc_nvenc for hardware acceleration)")
        output_pix_fmt = "yuv420p10le"
        if user_output_crf is None:
            default_cpu_crf = "28" # For CPU x265 (HDR often needs higher CRF to look "good")
        output_profile = "main10"
        if video_stream_info.get("mastering_display_metadata"):
            x265_params.append(f"master-display={video_stream_info['mastering_display_metadata']}")
        if video_stream_info.get("max_content_light_level"):
            x265_params.append(f"max-cll={video_stream_info['max_content_light_level']}")
    elif original_codec_name == "hevc" and is_original_10bit_or_higher:
        logger.info("Detected SDR 10-bit HEVC source. Targeting HEVC 10-bit SDR output.")
        output_codec = "libx265"
        if CUDA_AVAILABLE:
            output_codec = "hevc_nvenc"
            logger.info("    (Using hevc_nvenc for hardware acceleration)")
        output_pix_fmt = "yuv420p10le"
        if user_output_crf is None:
            default_cpu_crf = "24" # For CPU x265 (SDR 10-bit)
        output_profile = "main10"
    else: # Default to H.264 8-bit, or if no info
        logger.info("Detected SDR (8-bit H.264 or other) source or no specific info. Targeting H.264 8-bit.")
        output_codec = "libx264"
        if CUDA_AVAILABLE:
            output_codec = "h264_nvenc"
            logger.info("    (Using h264_nvenc for hardware acceleration)")
        output_pix_fmt = "yuv420p"
        if user_output_crf is None:
            default_cpu_crf = "18" # For CPU x264 (SDR 8-bit, higher quality)
        output_profile = "main"

    logger.debug("default_cpu_crf = {default_cpu_crf}")
    # Add codec, profile, pix_fmt
    ffmpeg_cmd.extend(["-c:v", output_codec])
    if "nvenc" in output_codec:
        ffmpeg_cmd.extend(["-preset", nvenc_preset])
        ffmpeg_cmd.extend(["-cq", default_nvenc_cq]) # NVENC uses CQ, not CRF
    else:
        ffmpeg_cmd.extend(["-crf", default_cpu_crf])
    
    ffmpeg_cmd.extend(["-pix_fmt", output_pix_fmt])
    if output_profile:
        ffmpeg_cmd.extend(["-profile:v", output_profile])

    # Add x265-params if using libx265 and params are available
    if output_codec == "libx265" and x265_params:
        ffmpeg_cmd.extend(["-x265-params", ":".join(x265_params)])

    # Add general color flags if present in source info
    if video_stream_info:
        if video_stream_info.get("color_primaries"):
            ffmpeg_cmd.extend(["-color_primaries", video_stream_info["color_primaries"]])
        if video_stream_info.get("transfer_characteristics"):
            ffmpeg_cmd.extend(["-color_trc", video_stream_info["transfer_characteristics"]])
        if video_stream_info.get("color_space"):
            ffmpeg_cmd.extend(["-colorspace", video_stream_info["color_space"]])

    # Final output path
    ffmpeg_cmd.append(final_output_mp4_path)

    logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")

    try:
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        
        while process.poll() is None: # While process is still running
            if stop_event.is_set():
                logger.warning(f"FFmpeg encoding stopped by user for {os.path.basename(final_output_mp4_path)}.")
                process.terminate() # or process.kill()
                process.wait(timeout=5)
                return False
            time.sleep(0.1) # Check stop_event frequently

        stdout, stderr = process.communicate(timeout=60) # A final communicate in case something was buffered
        
        if process.returncode != 0:
            logger.error(f"FFmpeg encoding failed for {os.path.basename(final_output_mp4_path)} (return code {process.returncode}):\n{stderr}\n{stdout}")
            return False
        else:
            logger.info(f"Successfully encoded video to {final_output_mp4_path}")
            logger.debug(f"FFmpeg stdout:\n{stdout}")
            logger.debug(f"FFmpeg stderr:\n{stderr}")

    except FileNotFoundError:
        logger.error("FFmpeg not found. Please ensure FFmpeg is installed and in your system PATH.")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg encoding failed for {os.path.basename(final_output_mp4_path)}: {e.stderr}\n{e.stdout}")
        return False
    except subprocess.TimeoutExpired as e:
        logger.error(f"FFmpeg encoding timed out for {os.path.basename(final_output_mp4_path)}: {e.stderr}")
        process.kill()
        process.wait() # Ensure the process is cleaned up
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during encoding for {os.path.basename(final_output_mp4_path)}: {str(e)}", exc_info=True)
        return False
    finally:
        # Cleanup temporary PNGs
        if os.path.exists(temp_png_dir):
            try:
                shutil.rmtree(temp_png_dir)
                logger.info(f"Cleaned up temporary directory: {temp_png_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary PNG directory {temp_png_dir}: {e}")

    # Write sidecar JSON if data is provided
    if sidecar_json_data:
        output_sidecar_path = f"{os.path.splitext(final_output_mp4_path)[0]}.json"
        try:
            with open(output_sidecar_path, 'w', encoding='utf-8') as f:
                json.dump(sidecar_json_data, f, indent=4)
            logger.info(f"Created output sidecar JSON: {output_sidecar_path}")
        except Exception as e:
            logger.error(f"Error creating output sidecar JSON '{output_sidecar_path}': {e}")
            # This is not a critical error for video encoding, so don't return False here.

    logger.info(f"Done processing {os.path.basename(final_output_mp4_path)}")
    return True

# --- END OF FILE dependency/stereocrafter_util.py ---