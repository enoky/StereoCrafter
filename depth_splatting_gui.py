import gc
import os
import cv2
import glob
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import write_video
from decord import VideoReader, cpu
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import json
import threading
import queue
import subprocess
import time
from typing import Optional

# Import custom modules
from dependency.forward_warp_pytorch import forward_warp

# torch.backends.cudnn.benchmark = True

# Global variables for GUI control
stop_event = threading.Event()
progress_queue = queue.Queue()
processing_thread = None
help_texts = {} # Dictionary to store help texts
CUDA_AVAILABLE = False # NEW: Global flag for CUDA availability

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + self.widget.winfo_height() + 1
        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tip_window, text=self.text, background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"), wraplength=300)
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
        self.tip_window = None

def create_hover_tooltip(widget, key):
    if key in help_texts:
        Tooltip(widget, help_texts[key])

def round_to_nearest_64(value):
    return max(64, round(value / 64) * 64)

def draw_progress_bar(current_progress, total_progress, bar_length=50, prefix='Progress:', suffix=''): # Changed default suffix
    """
    Draws an ASCII progress bar in the console, overwriting the same line.
    Adds a newline only when 100% complete.
    """
    if total_progress == 0:
        percent = 100
    else:
        percent = 100 * (current_progress / float(total_progress))
    filled_length = int(round(bar_length * current_progress / float(total_progress)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    # Format the suffix for completion
    actual_suffix = suffix
    if current_progress == total_progress:
        actual_suffix = "Complete"

    print(f'\r{prefix} |{bar}| {percent:.1f}% {actual_suffix}', end='\r')

    # Add a final newline only when the progress is 100% to ensure subsequent prints are on a new line
    if current_progress == total_progress:
        print()

def clear_processing_info():
    """Resets all 'Current Processing Information' labels to default 'N/A'."""
    processing_filename_var.set("N/A")
    processing_resolution_var.set("N/A")
    processing_frames_var.set("N/A")
    processing_disparity_var.set("N/A")
    processing_convergence_var.set("N/A")
    processing_task_name_var.set("N/A")

def check_cuda_availability():
    """
    Checks if CUDA is available via PyTorch and if nvidia-smi can run.
    Sets the global CUDA_AVAILABLE flag.
    """
    global CUDA_AVAILABLE
    if torch.cuda.is_available():
        print("==> PyTorch reports CUDA is available.")
        # Further check with nvidia-smi for robustness, though torch.cuda.is_available is often enough.
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=5, encoding='utf-8')
            print("==> CUDA detected (nvidia-smi also ran successfully). NVENC can be used.")
            CUDA_AVAILABLE = True
        except FileNotFoundError:
            print("==> Warning: nvidia-smi not found. CUDA is reported by PyTorch but NVENC availability cannot be fully confirmed. Proceeding with PyTorch's report.")
            CUDA_AVAILABLE = True # Rely on PyTorch if nvidia-smi not found
        except subprocess.CalledProcessError:
            print("==> Warning: nvidia-smi failed. CUDA is reported by PyTorch but NVENC availability cannot be fully confirmed. Proceeding with PyTorch's report.")
            CUDA_AVAILABLE = True # Rely on PyTorch if nvidia-smi fails
        except subprocess.TimeoutExpired:
            print("==> Warning: nvidia-smi timed out. CUDA is reported by PyTorch but NVENC availability cannot be fully confirmed. Proceeding with PyTorch's report.")
            CUDA_AVAILABLE = True # Rely on PyTorch if nvidia-smi times out
        except Exception as e:
            print(f"==> Unexpected error during nvidia-smi check: {e}. Relying on PyTorch's report for CUDA.")
            CUDA_AVAILABLE = True # Rely on PyTorch as a fallback
    else:
        print("==> PyTorch reports CUDA is NOT available. NVENC will not be used.")
        CUDA_AVAILABLE = False

def get_video_stream_info(video_path): # RENAMED FUNCTION
    """
    Uses ffprobe to extract comprehensive video stream information, including
    color space, codec, pixel format, and HDR mastering metadata.
    Returns a dictionary or None if ffprobe fails/info not found.
    Requires ffprobe to be installed and in PATH.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0", # Select the first video stream
        "-show_entries", "stream=codec_name,profile,pix_fmt,color_primaries,transfer_characteristics,color_space,mastering_display_metadata,max_content_light_level", # ADDED entries
        "-of", "json",
        video_path
    ]
    # print(f"==> Running ffprobe command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        ffprobe_stdout = result.stdout.strip()
        ffprobe_stderr = result.stderr.strip()

        if ffprobe_stdout:
            # print(f"==> ffprobe stdout: {ffprobe_stdout}")
            output = json.loads(ffprobe_stdout)
            if "streams" in output and len(output["streams"]) > 0:
                # Store all stream info, then filter out 'N/A'/'und'
                stream_info = output["streams"][0]
                filtered_stream_info = {k: v for k, v in stream_info.items() if v not in ["N/A", "und", ""] and v is not None}
                if filtered_stream_info:
                    print(f"==> Detected video stream info for '{os.path.basename(video_path)}': {filtered_stream_info}")
                    return filtered_stream_info
                else:
                    # print(f"==> ffprobe found stream but no specific video metadata for '{os.path.basename(video_path)}'.")
                    return None
            else:
                print(f"==> ffprobe output had no 'streams' or no video stream detected for '{os.path.basename(video_path)}'.")
                return None
        else:
            print(f"==> ffprobe returned empty stdout for '{os.path.basename(video_path)}'.")
            if ffprobe_stderr:
                print(f"==> ffprobe stderr: {ffprobe_stderr}")
            return None
    except FileNotFoundError:
        print("==> Error: ffprobe not found. Please ensure FFmpeg is installed and in your system's PATH.")
        messagebox.showerror("FFprobe Error", "ffprobe not found. Please install FFmpeg and ensure it's in your system's PATH to detect video stream information.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"==> Error running ffprobe for '{os.path.basename(video_path)}': {e.returncode}")
        # print(f"==> ffprobe stderr: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"==> Error decoding ffprobe JSON output for '{os.path.basename(video_path)}': {e}")
        # print(f"==> Raw ffprobe stdout that failed JSON decoding: {ffprobe_stdout}")
        return None
    except Exception as e:
        print(f"==> An unexpected error occurred during ffprobe execution: {e}")
        return None
    
def read_video_frames(video_path, process_length, target_fps,
                      set_pre_res, pre_res_width, pre_res_height, dataset="open"):
    """
    Reads video frames and determines the processing resolution.
    Resolution is determined by set_pre_res, otherwise original resolution is used.
    """
    if dataset == "open":
        print(f"==> Processing video: {video_path}")
        vid_info = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = vid_info.get_batch([0]).shape[1:3]
        print(f"==> Original video shape: {len(vid_info)} frames, {original_height}x{original_width} per frame")

        height_for_processing = original_height
        width_for_processing = original_width

        if set_pre_res and pre_res_width > 0 and pre_res_height > 0:
            # User specified pre-processing resolution
            height_for_processing = pre_res_height
            width_for_processing = pre_res_width
            print(f"==> Pre-processing video to user-specified resolution: {width_for_processing}x{height_for_processing}")
        else:
            # If set_pre_res is False, use original resolution.
            print(f"==> Using original video resolution for processing: {width_for_processing}x{height_for_processing}")

    else:
        raise NotImplementedError(f"Dataset '{dataset}' not supported.")

    # decord automatically resizes if width/height are passed to VideoReader
    vid = VideoReader(video_path, ctx=cpu(0), width=width_for_processing, height=height_for_processing)
    fps = vid.get_avg_fps() if target_fps == -1 else target_fps
    stride = max(round(vid.get_avg_fps() / fps), 1)
    frames_idx = list(range(0, len(vid), stride))
    print(f"==> Downsampled to {len(frames_idx)} frames with stride {stride}")
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]

    # Verify the actual shape after Decord processing
    first_frame_shape = vid.get_batch([0]).shape
    actual_processed_height, actual_processed_width = first_frame_shape[1:3]
    print(f"==> Final processing shape: {len(frames_idx)} frames, {actual_processed_height}x{actual_processed_width} per frame")

    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0

    video_stream_info = get_video_stream_info(video_path)

    return frames, fps, original_height, original_width, actual_processed_height, actual_processed_width, video_stream_info

class ForwardWarpStereo(nn.Module):
    def __init__(self, eps=1e-6, occlu_map=False):
        super(ForwardWarpStereo, self).__init__()
        self.eps = eps
        self.occlu_map = occlu_map
        self.fw = forward_warp()

    def forward(self, im, disp):
        im = im.contiguous()
        disp = disp.contiguous()
        weights_map = disp - disp.min()
        weights_map = (1.414) ** weights_map
        flow = -disp.squeeze(1)
        dummy_flow = torch.zeros_like(flow, requires_grad=False)
        flow = torch.stack((flow, dummy_flow), dim=-1)
        res_accum = self.fw(im * weights_map, flow)
        mask = self.fw(weights_map, flow)
        mask.clamp_(min=self.eps)
        res = res_accum / mask
        if not self.occlu_map:
            return res
        else:
            ones = torch.ones_like(disp, requires_grad=False)
            occlu_map = self.fw(ones, flow)
            occlu_map.clamp_(0.0, 1.0)
            occlu_map = 1.0 - occlu_map
            return res, occlu_map

def DepthSplatting(input_frames_processed, processed_fps, output_video_path, video_depth, depth_vis, max_disp, process_length, batch_size,
                   dual_output: bool, zero_disparity_anchor_val: float, video_stream_info: Optional[dict]):
    print("==> Initializing ForwardWarpStereo module")
    stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

    num_frames = len(input_frames_processed)
    height, width, _ = input_frames_processed[0].shape # Get dimensions from already processed frames
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True) # Ensure output directory exists

    # Determine output video suffix and dimensions
    if dual_output:
        suffix = "_splatted2"
        # The actual output width/height for the grid will be determined by concatenation,
        # but these are the base frame dimensions.
        # output_video_width = width * 2 # Mask + Warped
        # output_video_height = height
    else: # Quad output
        suffix = "_splatted4"
        # output_video_width = width * 2
        # output_video_height = height * 2

    res_suffix = f"_{width}"

    # Construct the final output path that ffmpeg will write to
    final_output_video_path = f"{os.path.splitext(output_video_path)[0]}{res_suffix}{suffix}.mp4"

    # NEW: Use a temporary directory for PNG sequences
    temp_png_dir = os.path.join(os.path.dirname(final_output_video_path), "temp_splat_pngs_" + os.path.basename(os.path.splitext(output_video_path)[0]))
    os.makedirs(temp_png_dir, exist_ok=True)
    print(f"==> Writing temporary PNG sequence to: {temp_png_dir}")

    # Process only up to process_length if specified, for consistency
    if process_length != -1 and process_length < num_frames:
        input_frames_processed = input_frames_processed[:process_length]
        video_depth = video_depth[:process_length]
        depth_vis = depth_vis[:process_length]
        num_frames = process_length


    frame_count = 0 # To name PNGs sequentially
    
    # Add a single startup message
    print(f"==> Generating PNG sequence for {os.path.basename(final_output_video_path)}")
    # Initial draw of the progress bar
    draw_progress_bar(frame_count, num_frames, prefix=f"  Progress:") # Indent slightly
    
    for i in range(0, num_frames, batch_size):
        if stop_event.is_set():
            draw_progress_bar(frame_count, num_frames, suffix='Stopped')
            print() # Ensure a newline after stopping
            del stereo_projector # Added for early exit cleanup
            torch.cuda.empty_cache()
            gc.collect()
            if os.path.exists(temp_png_dir):
                shutil.rmtree(temp_png_dir)
            return

        batch_frames = input_frames_processed[i:i+batch_size]
        batch_depth = video_depth[i:i+batch_size]
        batch_depth_vis = depth_vis[i:i+batch_size]

        left_video = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).float().cuda()
        disp_map = torch.from_numpy(batch_depth).unsqueeze(1).float().cuda()

        disp_map = (disp_map - zero_disparity_anchor_val) * 2.0
        disp_map = disp_map * max_disp
        with torch.no_grad():
            right_video, occlusion_mask = stereo_projector(left_video, disp_map)
        right_video = right_video.cpu().permute(0, 2, 3, 1).numpy()
        occlusion_mask = occlusion_mask.cpu().permute(0, 2, 3, 1).numpy().repeat(3, axis=-1)

        for j in range(len(batch_frames)):
            # Determine the video grid based on dual_output setting
            if dual_output:
                # Dual output: Mask | Warped
                video_grid = np.concatenate([occlusion_mask[j], right_video[j]], axis=1)
            else:
                # Quad output: Original | DepthVis (Top), Mask | Warped (Bottom)
                video_grid_top = np.concatenate([batch_frames[j], batch_depth_vis[j]], axis=1)
                video_grid_bottom = np.concatenate([occlusion_mask[j], right_video[j]], axis=1)
                video_grid = np.concatenate([video_grid_top, video_grid_bottom], axis=0)

            # Convert to 16-bit for lossless saving, scaling float32 (0-1) to uint16 (0-65535)
            # Clip to 0-1 range before scaling to prevent overflow if values slightly exceed 1.
            video_grid_uint16 = np.clip(video_grid, 0.0, 1.0) * 65535.0
            video_grid_uint16 = video_grid_uint16.astype(np.uint16)

            # Convert to BGR for OpenCV
            video_grid_bgr = cv2.cvtColor(video_grid_uint16, cv2.COLOR_RGB2BGR)

            png_filename = os.path.join(temp_png_dir, f"{frame_count:05d}.png")
            cv2.imwrite(png_filename, video_grid_bgr)

            frame_count += 1

        del left_video, disp_map, right_video, occlusion_mask
        torch.cuda.empty_cache()
        gc.collect()
        
        draw_progress_bar(frame_count, num_frames, prefix=f"  Progress:")

    # <--- THESE TWO LINES MUST BE OUTSIDE THE 'for i' loop
    print(f"==> Temporary PNG sequence generation completed ({frame_count} frames).")
    
    del stereo_projector
    torch.cuda.empty_cache()
    gc.collect()

    # --- FFmpeg encoding from PNG sequence to final MP4 ---
    print(f"==> Encoding final video from PNG sequence using ffmpeg for '{os.path.basename(final_output_video_path)}'.")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y", # Overwrite output files without asking
        "-framerate", str(processed_fps), # Input framerate for the PNG sequence
        "-i", os.path.join(temp_png_dir, "%05d.png"), # Input PNG sequence pattern
    ]

    # --- Extract original video properties if available ---
    original_codec_name = video_stream_info.get("codec_name") if video_stream_info else None
    original_pix_fmt = video_stream_info.get("pix_fmt") if video_stream_info else None

    is_original_10bit_or_higher = False
    if original_pix_fmt:
        if "10" in original_pix_fmt or "12" in original_pix_fmt or "16" in original_pix_fmt:
            is_original_10bit_or_higher = True
            print(f"==> Detected original video pixel format: {original_pix_fmt} (>= 10-bit)")
        else:
            print(f"==> Detected original video pixel format: {original_pix_fmt} (< 10-bit)")
    else:
        print("==> Could not detect original video pixel format.")

    # --- Determine Output Codec, Bit-Depth, and Quality ---
    # Set sensible defaults first
    output_codec = "libx264" # Default to H.264
    output_pix_fmt = "yuv420p" # Default to 8-bit
    output_crf = "23" # Default CRF for H.264 (medium quality)
    output_profile = "main" # Default H.264 profile
    x265_params = [] # For specific x265 parameters

    
    # NEW: NVENC specific parameters
    nvenc_preset = "medium" # Default NVENC preset (e.g., fast, medium, slow, quality, etc.)
    # Note: NVENC uses CQP/VBR/CBR not CRF. We'll map CRF to a CQ value or just use a fixed quality.
    nvenc_cq = "23" # Constant Quality value for NVENC (lower is better quality)

    is_hdr_source = False
    if video_stream_info and video_stream_info.get("color_primaries") == "bt2020" and \
       video_stream_info.get("transfer_characteristics") == "smpte2084":
        is_hdr_source = True
        print("==> Source detected as HDR.")
    else:
        print("==> Source detected as SDR.")


    # Main logic for choosing output format based on detected info (always attempt smart matching)
    if video_stream_info: # Only proceed with smart matching if info is detected
        print("==> Source video stream info detected. Attempting to match source characteristics and optimize quality.")

        if is_hdr_source:
            print("==> Detected HDR source. Targeting HEVC (x265) 10-bit HDR output.")
            output_codec = "libx265" # Default to CPU x265
            if CUDA_AVAILABLE:
                output_codec = "hevc_nvenc" # Use NVENC if available
                print("    (Using hevc_nvenc for hardware acceleration)")
            
            output_pix_fmt = "yuv420p10le"
            output_crf = "28" # This CRF value is for CPU x265, will use nvenc_cq for NVENC
            output_profile = "main10"
            # Add HDR mastering display and light level metadata if available
            if video_stream_info.get("mastering_display_metadata"):
                md_meta = video_stream_info["mastering_display_metadata"]
                x265_params.append(f"mastering-display={md_meta}") # These are for x265, not nvenc directly.
                print(f"==> Adding mastering display metadata: {md_meta}")
            if video_stream_info.get("max_content_light_level"):
                max_cll_meta = video_stream_info["max_content_light_level"]
                x265_params.append(f"max-cll={max_cll_meta}") # These are for x265, not nvenc directly.
                print(f"==> Adding max content light level: {max_cll_meta}")

        elif original_codec_name == "hevc" and is_original_10bit_or_higher:
            print("==> Detected 10-bit HEVC (x265) SDR source. Targeting HEVC (x265) 10-bit SDR output.")
            output_codec = "libx265" # Default to CPU x265
            if CUDA_AVAILABLE:
                output_codec = "hevc_nvenc" # Use NVENC if available
                print("    (Using hevc_nvenc for hardware acceleration)")
            
            output_pix_fmt = "yuv420p10le"
            output_crf = "24" # For CPU x265
            output_profile = "main10"

        else: # If not HDR/HEVC 10-bit, default to H.264 high quality
            print("==> No specific HEVC/HDR source. Targeting H.264 (x264) 8-bit SDR high quality.")
            output_codec = "libx264" # Default to CPU x264
            if CUDA_AVAILABLE:
                output_codec = "h264_nvenc" # Use NVENC if available
                print("    (Using h264_nvenc for hardware acceleration)")
            
            output_pix_fmt = "yuv420p"
            output_crf = "18" # For CPU x264
            output_profile = "main"

    else: # video_stream_info is None (fallback behavior if no info detected)
        print("==> No source video stream info detected. Falling back to default H.264 (x264) 8-bit SDR (medium quality).")
        # Defaults already set at the top of this block (libx264, yuv420p, CRF 23, profile main)
        if CUDA_AVAILABLE:
            output_codec = "h264_nvenc" # Use NVENC if CUDA available for default H.264
            print("    (Using h264_nvenc for hardware acceleration for default output)")


    ffmpeg_cmd.extend(["-c:v", output_codec])
    # NEW: Add NVENC specific parameters
    if "nvenc" in output_codec: # Check if an NVENC codec is chosen
        ffmpeg_cmd.extend(["-preset", nvenc_preset])
        ffmpeg_cmd.extend(["-cq", nvenc_cq]) # Constant Quality for NVENC
        # Remove CRF if NVENC is used, as it's not applicable
        if "-crf" in ffmpeg_cmd:
            crf_index = ffmpeg_cmd.index("-crf")
            del ffmpeg_cmd[crf_index:crf_index+2] # Delete -crf and its value
    else: # Only add CRF if not NVENC
        ffmpeg_cmd.extend(["-crf", output_crf])
    
    ffmpeg_cmd.extend(["-pix_fmt", output_pix_fmt])
    if output_profile:
        # NVENC profiles might differ slightly, but main/main10 generally work.
        ffmpeg_cmd.extend(["-profile:v", output_profile])

    # NVENC doesn't use -x265-params directly for HDR.
    # HDR metadata for NVENC is usually passed via -mastering-display and -max-cll
    # directly as FFmpeg main flags, which we already handle outside -x265-params.
    # The -x265-params block is only relevant if using libx265.
    if output_codec == "libx265" and x265_params:
        ffmpeg_cmd.extend(["-x265-params", ":".join(x265_params)])


    # --- Add general color space flags (primaries, transfer, space) ---
    # This block remains, but now only applies if retain_color_space is true AND info exists
    if video_stream_info:
        if video_stream_info.get("color_primaries") and video_stream_info["color_primaries"] not in ["N/A", "und", "unknown"]:
            ffmpeg_cmd.extend(["-color_primaries", video_stream_info["color_primaries"]])
        if video_stream_info.get("transfer_characteristics") and video_stream_info["transfer_characteristics"] not in ["N/A", "und", "unknown"]:
            ffmpeg_cmd.extend(["-color_trc", video_stream_info["transfer_characteristics"]])
        if video_stream_info.get("color_space") and video_stream_info["color_space"] not in ["N/A", "und", "unknown"]:
            ffmpeg_cmd.extend(["-colorspace", video_stream_info["color_space"]])

    ffmpeg_cmd.append(final_output_video_path)

    print(f"==> Executing ffmpeg command: {' '.join(ffmpeg_cmd)}")

    try:
        ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True, encoding='utf-8', timeout=3600*24) # 24 hour timeout
        # print(f"==> FFmpeg stdout: {ffmpeg_result.stdout}")
        # print(f"==> FFmpeg stderr: {ffmpeg_result.stderr}")
        print(f"==> Final video successfully encoded to '{os.path.basename(final_output_video_path)}'.")
    except FileNotFoundError:
        print("==> Error: ffmpeg not found. Please ensure FFmpeg is installed and in your system's PATH. No final video generated.")
        messagebox.showerror("FFmpeg Error", "ffmpeg not found. Please install FFmpeg and ensure it's in your system's PATH to encode from PNGs.")
    except subprocess.CalledProcessError as e:
        print(f"==> Error running ffmpeg for '{os.path.basename(final_output_video_path)}': {e.returncode}")
        # print(f"==> FFmpeg stdout: {e.stdout}")
        # print(f"==> FFmpeg stderr: {e.stderr}")
        print("==> Final video encoding failed due to ffmpeg error.")
    except subprocess.TimeoutExpired:
        print(f"==> Error: FFmpeg encoding timed out for '{os.path.basename(final_output_video_path)}'.")
    except Exception as e:
        print(f"==> An unexpected error occurred during ffmpeg execution: {e}")

    # Clean up temporary PNG directory
    if os.path.exists(temp_png_dir):
        try:
            shutil.rmtree(temp_png_dir)
            print(f"==> Cleaned up temporary PNG directory: {temp_png_dir}")
        except Exception as e:
            print(f"==> Error cleaning up temporary PNG directory {temp_png_dir}: {e}")

    print(f"==> Final output video written to: {final_output_video_path}")

def load_pre_rendered_depth(depth_map_path, process_length=-1, target_height=-1, target_width=-1, match_resolution_to_target=True, enable_autogain=True):
    """
    Loads pre-rendered depth maps from MP4 or NPZ.
    If match_resolution_to_target is True, it resizes the depth maps to the target_height/width for compatibility.
    Includes an option to enable/disable min-max normalization (autogain).
    If autogain is disabled, MP4 depth maps are scaled to 0-1 based on their detected bit depth (8-bit -> /255, 10-bit -> /1023).
    For NPZ with autogain disabled, it assumes the data is already in the desired absolute range (e.g., 0-1).
    """
    print(f"==> Loading pre-rendered depth maps from: {depth_map_path}")

    video_depth_working_range = None # This will hold the float32 array before final normalization/scaling

    if depth_map_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        vid = VideoReader(depth_map_path, ctx=cpu(0))
        # Read raw frames as float32, without any initial fixed scaling like /255.0
        # This allows us to inspect the actual raw pixel values for bit depth detection.
        raw_frames_data = vid[:].asnumpy().astype("float32")

        if raw_frames_data.shape[-1] == 3:
            print("==> Converting RGB depth frames to grayscale")
            raw_frames_data = raw_frames_data.mean(axis=-1)
        else:
            raw_frames_data = raw_frames_data.squeeze(-1)

        if enable_autogain:
            # If autogain is enabled, we work with the raw float data.
            # The min-max normalization below will then scale this observed range to 0-1.
            video_depth_working_range = raw_frames_data
        else:
            # Autogain disabled: Scale to 0-1 absolutely based on detected bit depth.
            # This logic assumes common bit-depth ranges for MP4 depth maps.
            max_raw_value_in_content = np.max(raw_frames_data)
            if max_raw_value_in_content > 255 and max_raw_value_in_content <= 1023: # Heuristic for 10-bit range
                print(f"==> Autogain disabled. Detected potential 10-bit depth map (max raw value: {max_raw_value_in_content:.0f}). Scaling to absolute 0-1 by dividing by 1023.0.")
                video_depth_working_range = raw_frames_data / 1023.0
            else: # Assume 8-bit or standard 0-255 range
                print(f"==> Autogain disabled. Detected potential 8-bit depth map (max raw value: {max_raw_value_in_content:.0f}). Scaling to absolute 0-1 by dividing by 255.0.")
                video_depth_working_range = raw_frames_data / 255.0

    elif depth_map_path.lower().endswith('.npz'):
        loaded_data = np.load(depth_map_path)
        if 'depth' in loaded_data:
            video_depth_working_range = loaded_data['depth'].astype("float32")
            if not enable_autogain:
                # For NPZ, if autogain is off, we *assume* it's already 0-1 or has an absolute meaning.
                # If NPZ data is NOT already 0-1 and needs a fixed scaling (e.g., from 0-1000 to 0-1),
                # the user would need to preprocess the NPZ or a new GUI setting for NPZ absolute scaling factor would be required.
                print("==> Autogain disabled for NPZ. Assuming depth data is already in desired absolute range (e.g., 0-1).")
        else:
            raise ValueError("NPZ file does not contain a 'depth' array.")
    else:
        raise ValueError(f"Unsupported depth map format: {os.path.basename(depth_map_path)}. Only MP4/NPZ are supported.")

    if process_length != -1 and process_length < len(video_depth_working_range):
        video_depth_working_range = video_depth_working_range[:process_length]

    if enable_autogain:
        # Perform Autogain (Min-Max Scaling) on the raw working range
        video_depth_min = video_depth_working_range.min()
        video_depth_max = video_depth_working_range.max()
        if video_depth_max - video_depth_min > 1e-5:
            video_depth_normalized = (video_depth_working_range - video_depth_min) / (video_depth_max - video_depth_min)
            print(f"==> Depth maps autogained (min-max scaled) from observed range [{video_depth_min:.3f}, {video_depth_max:.3f}] to [0, 1].")
        else:
            video_depth_normalized = np.zeros_like(video_depth_working_range)
            print("==> Depth map range too small, setting to zeros after autogain.")
    else:
        # Autogain is disabled; video_depth_working_range already contains absolute 0-1 for MP4, or assumed for NPZ.
        video_depth_normalized = video_depth_working_range
        print("==> Autogain (min-max scaling) disabled. Depth maps are used with their absolute scaling.")


    # Resize logic remains the same as before
    if match_resolution_to_target and target_height > 0 and target_width > 0:
        print(f"==> Resizing loaded depth maps to target resolution: {target_width}x{target_height}")
        resized_depths = []
        resized_viss = []
        for i in range(video_depth_normalized.shape[0]):
            depth_frame = video_depth_normalized[i]
            # OpenCV expects (width, height)
            resized_depth_frame = cv2.resize(depth_frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized_depths.append(resized_depth_frame)

            # Apply colormap to the resized depth for visualization.
            # Clip to 0-1 range for reliable visualization, as the colormap expects this.
            vis_frame = cv2.applyColorMap((np.clip(resized_depth_frame, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            resized_viss.append(vis_frame.astype("float32") / 255.0)

        video_depth = np.stack(resized_depths, axis=0)
        depth_vis = np.stack(resized_viss, axis=0)
    else:
        print(f"==> Not resizing loaded depth maps (match_resolution_to_target is False or target dimensions invalid). Current resolution: {video_depth_normalized.shape[2]}x{video_depth_normalized.shape[1]}")
        video_depth = video_depth_normalized
        # Visualization: Ensure 0-1 range for colormap if autogain is off and values might naturally exceed 1.
        depth_vis = np.stack([cv2.applyColorMap((np.clip(frame, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO).astype("float32") / 255.0 for frame in video_depth_normalized], axis=0)

    print("==> Depth maps and visualizations loaded successfully")
    return video_depth, depth_vis

def release_resources():
    try:
        torch.cuda.empty_cache()
        gc.collect()
        print("==> VRAM and resources released")
    except Exception as e:
        print(f"==> Error releasing VRAM: {e}")

def main(settings):
    input_source_clips_path_setting = settings["input_source_clips"]
    input_depth_maps_path_setting = settings["input_depth_maps"]
    output_splatted = settings["output_splatted"]
    gui_max_disp = float(settings["max_disp"])
    process_length = settings["process_length"]
    dual_output = settings["dual_output"]
    default_zero_disparity_anchor = settings["zero_disparity_anchor"]

    # NEW: Retrieve resolution-specific settings
    enable_full_resolution = settings["enable_full_resolution"]
    full_res_batch_size = settings["full_res_batch_size"]
    enable_low_resolution = settings["enable_low_resolution"]
    low_res_width = settings["low_res_width"]
    low_res_height = settings["low_res_height"]
    low_res_batch_size = settings["low_res_batch_size"]


    is_single_file_mode = False
    input_videos = []
    finished_source_folder = None
    finished_depth_folder = None
    

    is_source_file = os.path.isfile(input_source_clips_path_setting)
    is_source_dir = os.path.isdir(input_source_clips_path_setting)
    is_depth_file = os.path.isfile(input_depth_maps_path_setting)
    is_depth_dir = os.path.isdir(input_depth_maps_path_setting)

    root.after(0, clear_processing_info)

    if is_source_file and is_depth_file:
        is_single_file_mode = True
        print("==> Running in single file mode. Files will not be moved to 'finished' folders.")
        input_videos.append(input_source_clips_path_setting)
        os.makedirs(output_splatted, exist_ok=True)
    elif is_source_dir and is_depth_dir:
        print("==> Running in batch (folder) mode.")
        finished_source_folder = os.path.join(input_source_clips_path_setting, "finished")
        finished_depth_folder = os.path.join(input_depth_maps_path_setting, "finished")
        os.makedirs(finished_source_folder, exist_ok=True)
        os.makedirs(finished_depth_folder, exist_ok=True)
        os.makedirs(output_splatted, exist_ok=True)

        video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
        for ext in video_extensions:
            input_videos.extend(glob.glob(os.path.join(input_source_clips_path_setting, ext)))
        input_videos = sorted(input_videos)
    else:
        print("==> Error: Input Source Clips and Input Depth Maps must both be either files or directories. Skipping processing.")
        progress_queue.put("finished")
        release_resources()
        return

    if not input_videos:
        print(f"No video files found in {input_source_clips_path_setting}")
        progress_queue.put("finished")
        release_resources()
        return

    # Determine total tasks for the progress bar
    total_processing_tasks_count = 0
    if enable_full_resolution:
        total_processing_tasks_count += len(input_videos)
    if enable_low_resolution:
        total_processing_tasks_count += len(input_videos)

    if total_processing_tasks_count == 0:
        print("==> Error: No resolution output enabled. Processing stopped.")
        progress_queue.put("finished")
        release_resources()
        return

    progress_queue.put(("total", total_processing_tasks_count))
    overall_task_counter = 0

    for idx, video_path in enumerate(input_videos):
        if stop_event.is_set():
            print("==> Stopping processing due to user request")
            release_resources()
            progress_queue.put("finished")
            root.after(0, clear_processing_info) # Clear on stop
            return

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n==> Processing Video: {video_name}")

        
        # NEW: Update filename immediately
        progress_queue.put(("update_info", {"filename": video_name}))

        current_zero_disparity_anchor = default_zero_disparity_anchor
        current_max_disparity_percentage = gui_max_disp

        actual_depth_map_path = None
        if is_single_file_mode:
            actual_depth_map_path = input_depth_maps_path_setting
            if not os.path.exists(actual_depth_map_path):
                 print(f"==> Error: Single depth map file '{actual_depth_map_path}' not found. Skipping this video.")
                 progress_queue.put(("processed", overall_task_counter)) # Still update to indicate skipping
                 continue
        else: # Batch mode
            depth_map_path_mp4 = os.path.join(input_depth_maps_path_setting, f"{video_name}_depth.mp4")
            depth_map_path_npz = os.path.join(input_depth_maps_path_setting, f"{video_name}_depth.npz")

            if os.path.exists(depth_map_path_mp4):
                actual_depth_map_path = depth_map_path_mp4
            elif os.path.exists(depth_map_path_npz):
                actual_depth_map_path = depth_map_path_npz

        if actual_depth_map_path:
            actual_depth_map_path = os.path.normpath(actual_depth_map_path)
            depth_map_basename = os.path.splitext(os.path.basename(actual_depth_map_path))[0]
            json_sidecar_path = os.path.join(os.path.dirname(actual_depth_map_path), f"{depth_map_basename}.json")

            anchor_source = "GUI"
            max_disp_source = "GUI"

            if os.path.exists(json_sidecar_path):
                try:
                    with open(json_sidecar_path, 'r') as f:
                        sidecar_data = json.load(f)

                    if "convergence_plane" in sidecar_data and isinstance(sidecar_data["convergence_plane"], (int, float)):
                        current_zero_disparity_anchor = float(sidecar_data["convergence_plane"])
                        print(f"==> Using convergence_plane from sidecar JSON '{json_sidecar_path}': {current_zero_disparity_anchor}")
                        anchor_source = "JSON"
                    else:
                        print(f"==> Warning: Sidecar JSON '{json_sidecar_path}' found but 'convergence_plane' key is missing or invalid. Using GUI anchor: {current_zero_disparity_anchor:.2f}")
                        anchor_source = "GUI (Invalid JSON)"

                    if "max_disparity" in sidecar_data and isinstance(sidecar_data["max_disparity"], (int, float)):
                        current_max_disparity_percentage = float(sidecar_data["max_disparity"])
                        print(f"==> Using max_disparity from sidecar JSON '{json_sidecar_path}': {current_max_disparity_percentage:.1f}")
                        max_disp_source = "JSON"
                    else:
                        print(f"==> Warning: Sidecar JSON '{json_sidecar_path}' found but 'max_disparity' key is missing or invalid. Using GUI max_disp: {current_max_disparity_percentage:.1f}")
                        max_disp_source = "GUI (Invalid JSON)"

                    # Do NOT put detailed info into "status", put into "update_info" later

                except json.JSONDecodeError:
                    print(f"==> Error: Could not parse sidecar JSON '{json_sidecar_path}'. Using GUI anchor and max_disp. Anchor={current_zero_disparity_anchor:.2f}, MaxDisp={current_max_disparity_percentage:.1f}%")
                    # Do NOT put detailed info into "status"
                except Exception as e:
                    print(f"==> Unexpected error reading sidecar JSON '{json_sidecar_path}': {e}. Using GUI anchor and max_disp. Anchor={current_zero_disparity_anchor:.2f}, MaxDisp={current_max_disparity_percentage:.1f}%")
                    progress_queue.put(("status", f"Video {idx+1}/{len(input_videos)}: (GUI Anchor:{current_zero_disparity_anchor:.2f}, GUI MaxDisp:{current_max_disparity_percentage:.1f}%)"))
            else:
                print(f"==> No sidecar JSON '{json_sidecar_path}' found for depth map. Using GUI anchor and max_disp: Anchor={current_zero_disparity_anchor:.2f}, MaxDisp={current_max_disparity_percentage:.1f}%")
                # Do NOT put detailed info into "status"
        else:
            print(f"==> Error: No depth map found for {video_name} in {input_depth_maps_path_setting}. Expected '{video_name}_depth.mp4' or '{video_name}_depth.npz' in folder mode. Skipping this video.")
            progress_queue.put(("processed", overall_task_counter))
            continue

        # --- Dynamic Processing Task Generation ---
        processing_tasks = []
        if enable_full_resolution:
            processing_tasks.append({
                "name": "Full-Resolution",
                "output_subdir": "hires",
                "set_pre_res": False,
                "target_width": -1, # Signals to read_video_frames to use original
                "target_height": -1, # Signals to read_video_frames to use original
                "batch_size": full_res_batch_size
            })
        if enable_low_resolution:
            processing_tasks.append({
                "name": "Low-Resolution",
                "output_subdir": "lowres",
                "set_pre_res": True,
                "target_width": low_res_width,
                "target_height": low_res_height,
                "batch_size": low_res_batch_size
            })

        if not processing_tasks:
            print(f"==> No processing tasks configured for {video_name}. Skipping.")
            progress_queue.put(("processed", overall_task_counter))
            continue

        for task_num, task in enumerate(processing_tasks):
            if stop_event.is_set():
                print(f"==> Stopping {task['name']} processing for {video_name} due to user request")
                release_resources()
                progress_queue.put("finished")
                root.after(0, clear_processing_info) # Clear on stop
                return

            print(f"\n==> Starting {task['name']} pass for {video_name}")

            # NEW: Set a general status message for the GUI progress bar
            progress_queue.put(("status", f"Processing {task['name']} for {video_name}"))
            
            # NEW: Update task name and static convergence/disparity
            progress_queue.put(("update_info", {
                "task_name": task['name'],
                "convergence": f"{current_zero_disparity_anchor:.2f} ({anchor_source})",
                "disparity": f"{current_max_disparity_percentage:.1f}% ({max_disp_source})"
            }))

            input_frames_processed = None
            processed_fps = None
            current_processed_height = None
            current_processed_width = None
            video_stream_info = None

            try:
                # 1. Read input video frames at the determined resolution for this pass
                input_frames_processed, processed_fps, original_vid_h, original_vid_w, current_processed_height, current_processed_width, video_stream_info = read_video_frames(
                video_path, process_length, target_fps=-1,
                set_pre_res=task["set_pre_res"], pre_res_width=task["target_width"], pre_res_height=task["target_height"]
                )
            except Exception as e:
                print(f"==> Error reading input video {video_path} for {task['name']} pass: {e}. Skipping this pass.")
                overall_task_counter += 1
                progress_queue.put(("processed", overall_task_counter))
                continue # Skip to next task

            # NEW: Update resolution and frames after reading video
            progress_queue.put(("update_info", {
                "resolution": f"{current_processed_width}x{current_processed_height}",
                "frames": len(input_frames_processed)
            }))

            video_depth = None
            depth_vis = None
            try:
                # 2. Load and resize depth maps for this pass's resolution
                video_depth, depth_vis = load_pre_rendered_depth(
                    actual_depth_map_path,
                    process_length=process_length,
                    target_height=current_processed_height,
                    target_width=current_processed_width,
                    match_resolution_to_target=settings["match_depth_res"], # This should always be True now
                    enable_autogain=settings["enable_autogain"]
                )
            except Exception as e:
                print(f"==> Error loading depth map for {video_name} {task['name']} pass: {e}. Skipping this pass.")
                overall_task_counter += 1
                progress_queue.put(("processed", overall_task_counter))
                continue

            if video_depth is None or depth_vis is None:
                print(f"==> Skipping {video_name} {task['name']} pass due to depth load failure or stop request.")
                overall_task_counter += 1
                progress_queue.put(("processed", overall_task_counter))
                continue

            # Ensure depth map matches current processed video frames (safeguard)
            if not (video_depth.shape[1] == current_processed_height and video_depth.shape[2] == current_processed_width):
                print(f"==> Warning: Depth map resolution ({video_depth.shape[2]}x{video_depth.shape[1]}) does not match processed video resolution ({current_processed_width}x{current_processed_height}) for {task['name']} pass. Attempting final resize as safeguard.")
                resized_video_depth = np.stack([cv2.resize(frame, (current_processed_width, current_processed_height), interpolation=cv2.INTER_LINEAR) for frame in video_depth], axis=0)
                resized_depth_vis = np.stack([cv2.resize(frame, (current_processed_width, current_processed_height), interpolation=cv2.INTER_LINEAR) for frame in depth_vis], axis=0)
                video_depth = resized_video_depth
                depth_vis = resized_depth_vis

            # Use the (potentially overridden) current_max_disparity_percentage
            actual_percentage_for_calculation = current_max_disparity_percentage / 20.0
            actual_max_disp_pixels = (actual_percentage_for_calculation / 100.0) * current_processed_width
            print(f"==> Max Disparity Input: {current_max_disparity_percentage:.1f}% -> Calculated Max Disparity for splatting ({task['name']}): {actual_max_disp_pixels:.2f} pixels")

            
            # NEW: Update disparity display with calculated pixel value
            progress_queue.put(("update_info", {"disparity": f"{actual_max_disp_pixels:.2f} pixels ({current_max_disparity_percentage:.1f}%)"}))

            # 3. Create output directory and construct video path
            current_output_subdir = os.path.join(output_splatted, task["output_subdir"])
            os.makedirs(current_output_subdir, exist_ok=True)
            output_video_path_base = os.path.join(current_output_subdir, f"{video_name}.mp4")

            # 4. Perform Depth Splatting
            DepthSplatting(
                input_frames_processed=input_frames_processed,
                processed_fps=processed_fps,
                output_video_path=output_video_path_base,
                video_depth=video_depth,
                depth_vis=depth_vis,
                max_disp=actual_max_disp_pixels,
                process_length=process_length,
                batch_size=task["batch_size"],
                dual_output=dual_output,
                zero_disparity_anchor_val=current_zero_disparity_anchor,
                video_stream_info=video_stream_info
            )
            if stop_event.is_set():
                print(f"==> Stopping {task['name']} pass for {video_name} due to user request")
                release_resources()
                progress_queue.put("finished")
                return

            print(f"==> Splatted {task['name']} video saved for {video_name}.")

            # Release resources after EACH pass to manage VRAM
            release_resources()
            overall_task_counter += 1
            progress_queue.put(("processed", overall_task_counter))
            print(f"==> Completed {task['name']} pass for {video_name}.")

        # --- End of all tasks for a single video. Now move files if not single file mode. ---
        if not stop_event.is_set() and not is_single_file_mode:
            # Move source video
            if finished_source_folder is not None:
                try:
                    shutil.move(video_path, finished_source_folder)
                    print(f"==> Moved processed video to: {finished_source_folder}")
                except Exception as e:
                    print(f"==> Failed to move video {video_path}: {e}")
            else:
                print(f"==> Cannot move source video: 'finished_source_folder' is not set.")

            # Move depth map and its sidecar JSON
            if actual_depth_map_path and os.path.exists(actual_depth_map_path) and finished_depth_folder is not None:
                try:
                    shutil.move(actual_depth_map_path, finished_depth_folder)
                    print(f"==> Moved depth map to: {finished_depth_folder}")

                    # Logic to move the sidecar JSON file
                    # Reconstruct the sidecar path based on the depth map's original location
                    # Note: actual_depth_map_path is already normalized, so its dirname is safe
                    depth_map_dirname = os.path.dirname(actual_depth_map_path)
                    depth_map_basename_without_ext = os.path.splitext(os.path.basename(actual_depth_map_path))[0]
                    json_sidecar_path_to_move = os.path.join(depth_map_dirname, f"{depth_map_basename_without_ext}.json")

                    if os.path.exists(json_sidecar_path_to_move):
                        shutil.move(json_sidecar_path_to_move, finished_depth_folder)
                        print(f"==> Moved sidecar JSON '{os.path.basename(json_sidecar_path_to_move)}' to: {finished_depth_folder}")
                    else:
                        print(f"==> No sidecar JSON '{json_sidecar_path_to_move}' found to move.")

                except Exception as e:
                    print(f"==> Failed to move depth map {actual_depth_map_path} or its sidecar: {e}")
            elif actual_depth_map_path and finished_depth_folder is None:
                print(f"==> Cannot move depth map: 'finished_depth_folder' is not set.")
        elif is_single_file_mode:
            print(f"==> Single file mode for {video_name}: Skipping moving files to 'finished' folder.")

    release_resources()
    root.after(0, clear_processing_info) # NEW: Clear info labels when all processing finished
    progress_queue.put("finished")
    print("\n==> Batch Depth Splatting Process Completed Successfully")

def browse_folder(var):
    current_path = var.get()
    # Ensure the path exists and is a directory, otherwise get its parent directory if it's a file
    # If path is invalid or empty, initialdir will be None, letting filedialog choose default.
    if os.path.isdir(current_path):
        initial_dir = current_path
    elif os.path.exists(current_path): # It's a file, so open in its directory
        initial_dir = os.path.dirname(current_path)
    else: # Path doesn't exist
        initial_dir = None

    folder = filedialog.askdirectory(initialdir=initial_dir)
    if folder:
        var.set(folder)

def browse_file(var, filetypes_list):
    current_path = var.get()
    # Ensure the path exists, otherwise get its parent directory if it's a file
    # If path is invalid or empty, initialdir will be None, letting filedialog choose default.
    if os.path.exists(current_path): # It's a file or directory
        initial_dir = os.path.dirname(current_path) if os.path.isfile(current_path) else current_path
    else: # Path doesn't exist
        initial_dir = None

    file_path = filedialog.askopenfilename(initialdir=initial_dir, filetypes=filetypes_list)
    if file_path:
        var.set(file_path)

def start_processing():
    global processing_thread
    stop_event.clear()
    start_button.config(state="disabled") # Disable immediately until all checks pass
    stop_button.config(state="normal")
    status_label.config(text="Starting processing...")

    # Input validation for all fields
    try:
        # Validate Max Disparity
        max_disp_val = float(max_disp_var.get())
        if max_disp_val <= 0:
            raise ValueError("Max Disparity must be positive.")

        # Validate Zero Disparity Anchor
        anchor_val = float(zero_disparity_anchor_var.get())
        if not (0.0 <= anchor_val <= 1.0):
            raise ValueError("Zero Disparity Anchor must be between 0.0 and 1.0.")

        # Validate Full Resolution Batch Size if enabled
        if enable_full_res_var.get():
            full_res_batch_size_val = int(batch_size_var.get())
            if full_res_batch_size_val <= 0:
                raise ValueError("Full Resolution Batch Size must be positive.")

        # Validate Low Resolution settings if enabled
        if enable_low_res_var.get():
            pre_res_w = int(pre_res_width_var.get())
            pre_res_h = int(pre_res_height_var.get())
            if pre_res_w <= 0 or pre_res_h <= 0:
                raise ValueError("Low-Resolution Width and Height must be positive.")
            low_res_batch_size_val = int(low_res_batch_size_var.get())
            if low_res_batch_size_val <= 0:
                raise ValueError("Low-Resolution Batch Size must be positive.")

        # Crucial check: At least one resolution must be enabled
        if not (enable_full_res_var.get() or enable_low_res_var.get()):
            raise ValueError("At least one resolution (Full or Low) must be enabled to start processing.")

    except ValueError as e:
        status_label.config(text=f"Error: {e}")
        start_button.config(state="normal") # Re-enable start button on error
        stop_button.config(state="disabled")
        return

    settings = {
        "input_source_clips": input_source_clips_var.get(),
        "input_depth_maps": input_depth_maps_var.get(),
        "output_splatted": output_splatted_var.get(),
        "max_disp": float(max_disp_var.get()),
        "process_length": int(process_length_var.get()),

        # NEW & RENAMED settings
        "enable_full_resolution": enable_full_res_var.get(),
        "full_res_batch_size": int(batch_size_var.get()), # Renamed internally for clarity
        "enable_low_resolution": enable_low_res_var.get(),
        "low_res_width": int(pre_res_width_var.get()),
        "low_res_height": int(pre_res_height_var.get()),
        "low_res_batch_size": int(low_res_batch_size_var.get()), # NEW

        "dual_output": dual_output_var.get(),
        "zero_disparity_anchor": float(zero_disparity_anchor_var.get()),
        "enable_autogain": True,
        "match_depth_res": True,
    }
    processing_thread = threading.Thread(target=main, args=(settings,))
    processing_thread.start()
    check_queue()

def stop_processing():
    global processing_thread
    stop_event.set()
    status_label.config(text="Stopping...")
    stop_button.config(state="disabled")

def exit_app():
    global processing_thread
    save_config()
    stop_event.set()
    if processing_thread and processing_thread.is_alive():
        print("==> Waiting for processing thread to finish...")
        processing_thread.join(timeout=5.0)  # Give thread a chance to terminate
        if processing_thread.is_alive():
            print("==> Thread did not terminate gracefully within timeout.")
    release_resources()
    root.destroy()

def check_queue():
    try:
        while True:
            message = progress_queue.get_nowait()
            if message == "finished":
                status_label.config(text="Processing finished")
                start_button.config(state="normal")
                stop_button.config(state="disabled")
                progress_var.set(0) # Reset progress bar
                break
            elif message[0] == "total":
                total_tasks = message[1] # <--- total_tasks is defined here
                progress_bar.config(maximum=total_tasks)
                progress_var.set(0) # Start from 0
                status_label.config(text=f"Processing 0 of {total_tasks} tasks") # Initial status
            elif message[0] == "processed":
                processed_tasks = message[1] # <--- processed_tasks is defined here
                total_tasks = progress_bar["maximum"] # Get total from configured bar

                progress_var.set(processed_tasks) # Update GUI progress bar

                # Update the GUI status_label.
                # The detailed message is set by "status", this just updates the count.
                status_label.config(text=f"Processed tasks: {processed_tasks}/{total_tasks} (overall)")

            elif message[0] == "status": # This is for detailed text updates (e.g., sidecar info, current video/task)
                # NEW: Make this very minimal, as detailed info is in the new info_frame
                status_label.config(text=f"Overall: {progress_var.get()}/{progress_bar['maximum']} - {message[1].split(':', 1)[-1].strip()}") # Only show the action part

            elif message[0] == "update_info": # NEW: Handle updates for the dedicated info frame
                info_data = message[1]
                if "filename" in info_data:
                    processing_filename_var.set(info_data["filename"])
                if "resolution" in info_data:
                    processing_resolution_var.set(info_data["resolution"])
                if "frames" in info_data:
                    processing_frames_var.set(str(info_data["frames"]))
                if "disparity" in info_data:
                    processing_disparity_var.set(info_data["disparity"])
                if "convergence" in info_data:
                    processing_convergence_var.set(info_data["convergence"])
                if "task_name" in info_data:
                    processing_task_name_var.set(info_data["task_name"])

    except queue.Empty:
        pass
    root.after(100, check_queue)

def reset_to_defaults():
    """Resets all GUI parameters to their default hardcoded values."""
    if not messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to their default values?"):
        return

    input_source_clips_var.set("./input_source_clips")
    input_depth_maps_var.set("./input_depth_maps")
    output_splatted_var.set("./output_splatted")
    max_disp_var.set("20.0")
    process_length_var.set("-1")
    enable_full_res_var.set(True)
    batch_size_var.set("10") # Full Res Batch Size
    enable_low_res_var.set(False)
    pre_res_width_var.set("1920")
    pre_res_height_var.set("1080")
    low_res_batch_size_var.set("50") # Low Res Batch Size
    dual_output_var.set(False)
    zero_disparity_anchor_var.set("0.5")
    
    toggle_processing_settings_fields()
    save_config() # Save the reset defaults
    clear_processing_info() # NEW: Clear info display on reset
    status_label.config(text="Settings reset to defaults.")

def restore_finished_files():
    """Moves all files from 'finished' folders back to their original input folders."""
    if not messagebox.askyesno("Restore Finished Files", "Are you sure you want to move all files from 'finished' folders back to their input directories?"):
        return

    source_clip_dir = input_source_clips_var.get()
    depth_map_dir = input_depth_maps_var.get()

    # Determine if we are in folder mode based on current settings
    is_source_dir = os.path.isdir(source_clip_dir)
    is_depth_dir = os.path.isdir(depth_map_dir)

    if not (is_source_dir and is_depth_dir):
        messagebox.showerror("Restore Error", "Restore 'finished' operation is only applicable when Input Source Clips and Input Depth Maps are set to directories (batch mode). Please ensure current settings reflect this.")
        status_label.config(text="Restore finished: Not in batch mode.")
        return

    finished_source_folder = os.path.join(source_clip_dir, "finished")
    finished_depth_folder = os.path.join(depth_map_dir, "finished")

    restored_count = 0
    errors_count = 0
    
    # Restore Source Clips
    if os.path.isdir(finished_source_folder):
        print(f"==> Restoring source clips from: {finished_source_folder}")
        for filename in os.listdir(finished_source_folder):
            src_path = os.path.join(finished_source_folder, filename)
            dest_path = os.path.join(source_clip_dir, filename)
            if os.path.isfile(src_path):
                try:
                    shutil.move(src_path, dest_path)
                    restored_count += 1
                    # print(f"Moved '{filename}' to '{source_clip_dir}'")
                except Exception as e:
                    errors_count += 1
                    print(f"Error moving source clip '{filename}': {e}")
    else:
        print(f"==> Finished source folder not found: {finished_source_folder}")

    # Restore Depth Maps and Sidecar JSONs
    if os.path.isdir(finished_depth_folder):
        print(f"==> Restoring depth maps and sidecars from: {finished_depth_folder}")
        for filename in os.listdir(finished_depth_folder):
            src_path = os.path.join(finished_depth_folder, filename)
            dest_path = os.path.join(depth_map_dir, filename)
            if os.path.isfile(src_path):
                try:
                    shutil.move(src_path, dest_path)
                    restored_count += 1
                    # print(f"Moved '{filename}' to '{depth_map_dir}'")
                except Exception as e:
                    errors_count += 1
                    print(f"Error moving depth map/sidecar '{filename}': {e}")
    else:
        print(f"==> Finished depth folder not found: {finished_depth_folder}")

    if restored_count > 0 or errors_count > 0:
        clear_processing_info() # NEW: Clear info display on restore completion
        status_label.config(text=f"Restore complete: {restored_count} files moved, {errors_count} errors.")
        messagebox.showinfo("Restore Complete", f"Finished files restoration attempted.\n{restored_count} files moved.\n{errors_count} errors occurred.")
    else:
        clear_processing_info() # NEW: Clear info display even if nothing to restore
        status_label.config(text="No files found to restore.")
        messagebox.showinfo("Restore Complete", "No files found in 'finished' folders to restore.")

def load_help_texts():
    global help_texts
    try:
        with open("splatter_help.json", "r") as f:
            help_texts = json.load(f)
    except FileNotFoundError:
        print("Error: splatter_help.json not found. Tooltips will not be available.")
        help_texts = {}
    except json.JSONDecodeError:
        print("Error: Could not decode splatter_help.json. Check file format.")
        help_texts = {}

def save_config():
    config = {
        "input_source_clips": input_source_clips_var.get(),
        "input_depth_maps": input_depth_maps_var.get(),
        "output_splatted": output_splatted_var.get(),
        "max_disp": max_disp_var.get(),
        "process_length": process_length_var.get(),
        "batch_size": batch_size_var.get(),
        "dual_output": dual_output_var.get(),
        "enable_full_resolution": enable_full_res_var.get(), # NEW
        "enable_low_resolution": enable_low_res_var.get(),   # RENAMED from set_pre_res
        "pre_res_width": pre_res_width_var.get(),
        "pre_res_height": pre_res_height_var.get(),
        "low_res_batch_size": low_res_batch_size_var.get(),
        "convergence_point": zero_disparity_anchor_var.get(),
    }
    with open("config_splat.json", "w") as f:
        json.dump(config, f, indent=4)

def load_config():
    if os.path.exists("config_splat.json"):
        with open("config_splat.json", "r") as f:
            config = json.load(f)
            input_source_clips_var.set(config.get("input_source_clips", "./input_source_clips"))
            input_depth_maps_var.set(config.get("input_depth_maps", "./input_depth_maps"))
            output_splatted_var.set(config.get("output_splatted", "./output_splatted"))
            max_disp_var.set(config.get("max_disp", "20.0"))
            process_length_var.set(config.get("process_length", "-1"))
            batch_size_var.set(config.get("batch_size", "10"))
            dual_output_var.set(config.get("dual_output", False))
            enable_full_res_var.set(config.get("enable_full_resolution", True))
            enable_low_res_var.set(config.get("enable_low_resolution", False))
            pre_res_width_var.set(config.get("pre_res_width", "1920"))
            pre_res_height_var.set(config.get("pre_res_height", "1080"))
            low_res_batch_size_var.set(config.get("low_res_batch_size", "25")) # NEW
            zero_disparity_anchor_var.set(config.get("convergence_point", "0.5"))

# Load help texts at the start
load_help_texts()

# GUI Setup
root = tk.Tk()
root.title("Batch Depth Splatting")
root.geometry("620x770")

# Create a menu bar
menubar = tk.Menu(root)
root.config(menu=menubar)

# Create "Option" menu
option_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Option", menu=option_menu)

# Add commands to the "Option" menu
option_menu.add_command(label="Reset to Default", command=reset_to_defaults)
option_menu.add_command(label="Restore Finished", command=restore_finished_files)

# Variables with defaults
input_source_clips_var = tk.StringVar(value="./input_source_clips")
input_depth_maps_var = tk.StringVar(value="./input_depth_maps")
output_splatted_var = tk.StringVar(value="./output_splatted")
max_disp_var = tk.StringVar(value="20.0")
process_length_var = tk.StringVar(value="-1")
batch_size_var = tk.StringVar(value="10")
dual_output_var = tk.BooleanVar(value=False) # Default to Quad
enable_full_res_var = tk.BooleanVar(value=True) # NEW: Default to True
enable_low_res_var = tk.BooleanVar(value=False) # RENAMED from set_pre_res_var
pre_res_width_var = tk.StringVar(value="1920")
pre_res_height_var = tk.StringVar(value="1080")
low_res_batch_size_var = tk.StringVar(value="25")
zero_disparity_anchor_var = tk.StringVar(value="0.5") # NEW: Default to 0.5 (mid-ground anchor)
# NEW: Variables for "Current Processing Information" display
processing_filename_var = tk.StringVar(value="N/A")
processing_resolution_var = tk.StringVar(value="N/A")
processing_frames_var = tk.StringVar(value="N/A")
processing_disparity_var = tk.StringVar(value="N/A")
processing_convergence_var = tk.StringVar(value="N/A")
processing_task_name_var = tk.StringVar(value="N/A") # To show Full-Res/Low-Res

# Load configuration
load_config()

# Folder selection frame
folder_frame = tk.LabelFrame(root, text="Input/Output Folders")
folder_frame.pack(pady=10, padx=10, fill="x")
folder_frame.grid_columnconfigure(1, weight=1) # This makes column 1 expand horizontally

# Input Source Clips Row
lbl_source_clips = tk.Label(folder_frame, text="Input Source Clips:")
lbl_source_clips.grid(row=0, column=0, sticky="e", padx=5, pady=2)
entry_source_clips = tk.Entry(folder_frame, textvariable=input_source_clips_var) # Reduced width for more buttons
entry_source_clips.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
btn_browse_source_clips_folder = tk.Button(folder_frame, text="Browse Folder", command=lambda: browse_folder(input_source_clips_var))
btn_browse_source_clips_folder.grid(row=0, column=2, padx=2, pady=2)
btn_select_source_clips_file = tk.Button(folder_frame, text="Select File", command=lambda: browse_file(input_source_clips_var, [("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]))
btn_select_source_clips_file.grid(row=0, column=3, padx=2, pady=2)
create_hover_tooltip(lbl_source_clips, "input_source_clips") # Existing
create_hover_tooltip(entry_source_clips, "input_source_clips") # Existing
create_hover_tooltip(btn_browse_source_clips_folder, "input_source_clips_folder") # NEW TOOLTIP
create_hover_tooltip(btn_select_source_clips_file, "input_source_clips_file") # NEW TOOLTIP


# Input Depth Maps Row
lbl_input_depth_maps = tk.Label(folder_frame, text="Input Depth Maps:")
lbl_input_depth_maps.grid(row=1, column=0, sticky="e", padx=5, pady=2)
entry_input_depth_maps = tk.Entry(folder_frame, textvariable=input_depth_maps_var) # Reduced width
entry_input_depth_maps.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
btn_browse_input_depth_maps_folder = tk.Button(folder_frame, text="Browse Folder", command=lambda: browse_folder(input_depth_maps_var))
btn_browse_input_depth_maps_folder.grid(row=1, column=2, padx=2, pady=2)
btn_select_input_depth_maps_file = tk.Button(folder_frame, text="Select File", command=lambda: browse_file(input_depth_maps_var, [("Depth Files", "*.mp4 *.npz"), ("All files", "*.*")]))
btn_select_input_depth_maps_file.grid(row=1, column=3, padx=2, pady=2)
create_hover_tooltip(lbl_input_depth_maps, "input_depth_maps") # Existing
create_hover_tooltip(entry_input_depth_maps, "input_depth_maps") # Existing
create_hover_tooltip(btn_browse_input_depth_maps_folder, "input_depth_maps_folder") # NEW TOOLTIP
create_hover_tooltip(btn_select_input_depth_maps_file, "input_depth_maps_file") # NEW TOOLTIP


# Output Splatted Row
lbl_output_splatted = tk.Label(folder_frame, text="Output Splatted:")
lbl_output_splatted.grid(row=2, column=0, sticky="e", padx=5, pady=2)
entry_output_splatted = tk.Entry(folder_frame, textvariable=output_splatted_var) # Reduced width
entry_output_splatted.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
btn_browse_output_splatted = tk.Button(folder_frame, text="Browse Folder", command=lambda: browse_folder(output_splatted_var))
btn_browse_output_splatted.grid(row=2, column=2, columnspan=2, padx=5, pady=2) # Spanning two columns
create_hover_tooltip(lbl_output_splatted, "output_splatted")
create_hover_tooltip(entry_output_splatted, "output_splatted")
create_hover_tooltip(btn_browse_output_splatted, "output_splatted")


# Process Resolution and Settings Frame (RENAMED)
preprocessing_frame = tk.LabelFrame(root, text="Process Resolution and Settings")
preprocessing_frame.pack(pady=10, padx=10, fill="x")
preprocessing_frame.grid_columnconfigure(1, weight=1) # Makes column 1 expand
preprocessing_frame.grid_columnconfigure(3, weight=1) # Makes column 3 expand (for horizontal spacing)


# --- Enable Full Resolution Section ---
enable_full_res_checkbox = tk.Checkbutton(preprocessing_frame, text="Enable Full Resolution Output (Native Video Resolution)", variable=enable_full_res_var)
enable_full_res_checkbox.grid(row=0, column=0, columnspan=4, sticky="w", padx=5, pady=2)
create_hover_tooltip(enable_full_res_checkbox, "enable_full_res")

lbl_full_res_batch_size = tk.Label(preprocessing_frame, text="Full Res Batch Size:") # RENAMED LABEL
lbl_full_res_batch_size.grid(row=1, column=0, sticky="e", padx=5, pady=2)
entry_full_res_batch_size = tk.Entry(preprocessing_frame, textvariable=batch_size_var, width=15) # Uses existing batch_size_var
entry_full_res_batch_size.grid(row=1, column=1, sticky="w", padx=5, pady=2)
create_hover_tooltip(lbl_full_res_batch_size, "full_res_batch_size")
create_hover_tooltip(entry_full_res_batch_size, "full_res_batch_size")


# --- Enable Low Resolution Section ---
enable_low_res_checkbox = tk.Checkbutton(preprocessing_frame, text="Enable Low Resolution Output (Pre-defined Below)", variable=enable_low_res_var) # RENAMED
enable_low_res_checkbox.grid(row=2, column=0, columnspan=4, sticky="w", padx=5, pady=(10, 2)) # Use pady=(top, bottom) for desired separation
create_hover_tooltip(enable_low_res_checkbox, "enable_low_res")

pre_res_width_label = tk.Label(preprocessing_frame, text="Low Res Width:") # RENAMED LABEL
pre_res_width_label.grid(row=3, column=0, sticky="e", padx=5, pady=2)
pre_res_width_entry = tk.Entry(preprocessing_frame, textvariable=pre_res_width_var, width=10)
pre_res_width_entry.grid(row=3, column=1, sticky="w", padx=5, pady=2)
create_hover_tooltip(pre_res_width_label, "low_res_width")
create_hover_tooltip(pre_res_width_entry, "low_res_width")

pre_res_height_label = tk.Label(preprocessing_frame, text="Low Res Height:") # RENAMED LABEL
pre_res_height_label.grid(row=3, column=2, sticky="e", padx=5, pady=2)
pre_res_height_entry = tk.Entry(preprocessing_frame, textvariable=pre_res_height_var, width=10)
pre_res_height_entry.grid(row=3, column=3, sticky="w", padx=5, pady=2)
create_hover_tooltip(pre_res_height_label, "low_res_height")
create_hover_tooltip(pre_res_height_entry, "low_res_height")

lbl_low_res_batch_size = tk.Label(preprocessing_frame, text="Low Res Batch Size:") # NEW
lbl_low_res_batch_size.grid(row=4, column=0, sticky="e", padx=5, pady=2)
entry_low_res_batch_size = tk.Entry(preprocessing_frame, textvariable=low_res_batch_size_var, width=15) # NEW
entry_low_res_batch_size.grid(row=4, column=1, sticky="w", padx=5, pady=2)
create_hover_tooltip(lbl_low_res_batch_size, "low_res_batch_size")
create_hover_tooltip(entry_low_res_batch_size, "low_res_batch_size")


# Function to enable/disable required resolution.
def toggle_processing_settings_fields():
    """Enables/disables resolution input fields and the START button based on checkbox states."""
    # Full Resolution controls
    if enable_full_res_var.get():
        entry_full_res_batch_size.config(state="normal")
        lbl_full_res_batch_size.config(state="normal")
    else:
        entry_full_res_batch_size.config(state="disabled")
        lbl_full_res_batch_size.config(state="disabled")

    # Low Resolution controls
    if enable_low_res_var.get():
        pre_res_width_label.config(state="normal")
        pre_res_width_entry.config(state="normal")
        pre_res_height_label.config(state="normal")
        pre_res_height_entry.config(state="normal")
        lbl_low_res_batch_size.config(state="normal")
        entry_low_res_batch_size.config(state="normal")
    else:
        pre_res_width_label.config(state="disabled")
        pre_res_width_entry.config(state="disabled")
        pre_res_height_label.config(state="disabled")
        pre_res_height_entry.config(state="disabled")
        lbl_low_res_batch_size.config(state="disabled")
        entry_low_res_batch_size.config(state="disabled")

    # START button enable/disable logic: Must have at least one resolution enabled
    if enable_full_res_var.get() or enable_low_res_var.get():
        start_button.config(state="normal")
    else:
        start_button.config(state="disabled")

# Link the toggle function to the checkboxes' state changes
enable_full_res_var.trace_add("write", lambda *args: toggle_processing_settings_fields())
enable_low_res_var.trace_add("write", lambda *args: toggle_processing_settings_fields())

# Output Settings Frame
output_settings_frame = tk.LabelFrame(root, text="Splatting & Output Settings")
output_settings_frame.pack(pady=10, padx=10, fill="x")

# Max Disparity and Batch Size within Output Settings
lbl_max_disp = tk.Label(output_settings_frame, text="Max Disparity %:")
lbl_max_disp.grid(row=0, column=0, sticky="e", padx=5, pady=2)
entry_max_disp = tk.Entry(output_settings_frame, textvariable=max_disp_var, width=15)
entry_max_disp.grid(row=0, column=1, sticky="w", padx=5, pady=2)
create_hover_tooltip(lbl_max_disp, "max_disp")
create_hover_tooltip(entry_max_disp, "max_disp")

# NEW: Zero Disparity Anchor Point input (Moved to row 1)
lbl_zero_disparity_anchor = tk.Label(output_settings_frame, text="Convergence Point (0-1):")
lbl_zero_disparity_anchor.grid(row=1, column=0, sticky="e", padx=5, pady=2) # CHANGED FROM row=2 to row=1
entry_zero_disparity_anchor = tk.Entry(output_settings_frame, textvariable=zero_disparity_anchor_var, width=15)
entry_zero_disparity_anchor.grid(row=1, column=1, sticky="w", padx=5, pady=2) # CHANGED FROM row=2 to row=1
create_hover_tooltip(lbl_zero_disparity_anchor, "convergence_point")
create_hover_tooltip(entry_zero_disparity_anchor, "convergence_point")

# Process Length (Moved to row 2)
lbl_process_length = tk.Label(output_settings_frame, text="Process Length (-1 for all):")
lbl_process_length.grid(row=2, column=0, sticky="e", padx=5, pady=2) # CHANGED FROM row=3 to row=2
entry_process_length = tk.Entry(output_settings_frame, textvariable=process_length_var, width=15)
entry_process_length.grid(row=2, column=1, sticky="w", padx=5, pady=2) # CHANGED FROM row=3 to row=2
create_hover_tooltip(lbl_process_length, "process_length")
create_hover_tooltip(entry_process_length, "process_length")

# Dual Output Checkbox (Moved to row 3, now at the bottom of these settings)
dual_output_checkbox = tk.Checkbutton(output_settings_frame, text="Dual Output (Mask & Warped)", variable=dual_output_var)
dual_output_checkbox.grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=2) # CHANGED FROM row=1 to row=3
create_hover_tooltip(dual_output_checkbox, "dual_output")

# Progress frame
progress_frame = tk.LabelFrame(root, text="Progress")
progress_frame.pack(pady=10, padx=10, fill="x")
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100)
progress_bar.pack(fill="x", expand=True, padx=5, pady=2)
status_label = tk.Label(progress_frame, text="Ready")
status_label.pack(padx=5, pady=2)

# Button frame
button_frame = tk.Frame(root)
button_frame.pack(pady=10)
start_button = tk.Button(button_frame, text="START", command=start_processing)
start_button.pack(side="left", padx=5)
create_hover_tooltip(start_button, "start_button")

stop_button = tk.Button(button_frame, text="STOP", command=stop_processing, state="disabled")
stop_button.pack(side="left", padx=5)
create_hover_tooltip(stop_button, "stop_button")

exit_button = tk.Button(button_frame, text="EXIT", command=exit_app)
exit_button.pack(side="left", padx=5)
create_hover_tooltip(exit_button, "exit_button")

# NEW: Current Processing Information frame
info_frame = tk.LabelFrame(root, text="Current Processing Information")
info_frame.pack(pady=10, padx=10, fill="x")
info_frame.grid_columnconfigure(1, weight=1) # Makes value columns expand

# Row 0: Filename
tk.Label(info_frame, text="Filename:").grid(row=0, column=0, sticky="e", padx=5, pady=1)
tk.Label(info_frame, textvariable=processing_filename_var, anchor="w").grid(row=0, column=1, sticky="ew", padx=5, pady=1)

# Row 1: Task Name (e.g., Full-Resolution, Low-Resolution)
tk.Label(info_frame, text="Task:").grid(row=1, column=0, sticky="e", padx=5, pady=1)
tk.Label(info_frame, textvariable=processing_task_name_var, anchor="w").grid(row=1, column=1, sticky="ew", padx=5, pady=1)

# Row 2: Resolution
tk.Label(info_frame, text="Resolution:").grid(row=2, column=0, sticky="e", padx=5, pady=1)
tk.Label(info_frame, textvariable=processing_resolution_var, anchor="w").grid(row=2, column=1, sticky="ew", padx=5, pady=1)

# Row 3: Total Frames for current task
tk.Label(info_frame, text="Frames:").grid(row=3, column=0, sticky="e", padx=5, pady=1)
tk.Label(info_frame, textvariable=processing_frames_var, anchor="w").grid(row=3, column=1, sticky="ew", padx=5, pady=1)

# Row 4: Max Disparity
tk.Label(info_frame, text="Max Disparity:").grid(row=4, column=0, sticky="e", padx=5, pady=1)
tk.Label(info_frame, textvariable=processing_disparity_var, anchor="w").grid(row=4, column=1, sticky="ew", padx=5, pady=1)

# Row 5: Convergence Point
tk.Label(info_frame, text="Convergence:").grid(row=5, column=0, sticky="e", padx=5, pady=1)
tk.Label(info_frame, textvariable=processing_convergence_var, anchor="w").grid(row=5, column=1, sticky="ew", padx=5, pady=1)

# Initial calls to set the correct state based on loaded config
root.after(10, toggle_processing_settings_fields) # UPDATED CALL

# NEW: Check CUDA availability once at script start
check_cuda_availability()

# Run the GUI
root.mainloop()