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
from tkinter import filedialog, ttk
import json
import threading
import queue
import subprocess
import time

# Import custom modules
from dependency.forward_warp_pytorch import forward_warp

# torch.backends.cudnn.benchmark = True

# Global variables for GUI control
stop_event = threading.Event()
progress_queue = queue.Queue()
processing_thread = None
help_texts = {} # Dictionary to store help texts

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

    return frames, fps, original_height, original_width, actual_processed_height, actual_processed_width

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
        # Reverted to original flow calculation with negative sign for standard right-eye view.
        # If inverting is desired, this line is the one to change from -disp to disp.
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
                   dual_output: bool, zero_disparity_anchor_val: float): # Added zero_disparity_anchor_val

    print("==> Initializing ForwardWarpStereo module")
    stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

    num_frames = len(input_frames_processed)
    height, width, _ = input_frames_processed[0].shape # Get dimensions from already processed frames
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True) # Ensure output directory exists

    # Determine output video suffix and dimensions
    if dual_output:
        suffix = "_splatted2"
        output_video_width = width * 2 # Mask + Warped
        output_video_height = height
    else: # Quad output
        suffix = "_splatted4"
        output_video_width = width * 2
        output_video_height = height * 2

    res_suffix = f"_{width}"

    # Update the main output path
    base_output_name = os.path.splitext(output_video_path)[0]
    main_output_video_path = f"{base_output_name}{res_suffix}{suffix}.mp4"

    # Initialize the main video writer
    main_out = cv2.VideoWriter(
        main_output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        processed_fps, # Use the fps of the processed video
        (output_video_width, output_video_height),
        True
    )
    print(f"==> Writing main output video to: {main_output_video_path}")

    # Process only up to process_length if specified, for consistency
    if process_length != -1 and process_length < num_frames:
        input_frames_processed = input_frames_processed[:process_length]
        video_depth = video_depth[:process_length]
        depth_vis = depth_vis[:process_length]
        num_frames = process_length


    for i in range(0, num_frames, batch_size):
        if stop_event.is_set():
            print("==> Stopping DepthSplatting due to user request")
            del stereo_projector
            torch.cuda.empty_cache()
            gc.collect()
            main_out.release()
            return

        batch_frames = input_frames_processed[i:i+batch_size]
        batch_depth = video_depth[i:i+batch_size]
        batch_depth_vis = depth_vis[i:i+batch_size]

        left_video = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).float().cuda()
        disp_map = torch.from_numpy(batch_depth).unsqueeze(1).float().cuda()

        # REMOVED: disp_map = disp_map * 2.0 - 1.0
        
        # MODIFIED: Use the passed anchor value
        disp_map = (disp_map - zero_disparity_anchor_val) * 2.0 # Use the anchor value passed to the function

        disp_map = disp_map * max_disp # This line scales the adjusted disparity and remains.
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

            video_grid_uint8 = np.clip(video_grid * 255.0, 0, 255).astype(np.uint8)
            video_grid_bgr = cv2.cvtColor(video_grid_uint8, cv2.COLOR_RGB2BGR)

            # Write to main output video
            main_out.write(video_grid_bgr)

        del left_video, disp_map, right_video, occlusion_mask
        torch.cuda.empty_cache()
        gc.collect()
        print(f"==> Processed frames {i+1} to {min(i+batch_size, num_frames)}")
    main_out.release()
    print("==> Output video writing completed.")

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
    # Original max_disp from GUI
    gui_max_disp = float(settings["max_disp"]) 
    process_length = settings["process_length"]
    batch_size = settings["batch_size"]
    dual_output = settings["dual_output"]
    set_pre_res = settings["set_pre_res"]
    pre_res_width = int(settings["pre_res_width"])
    pre_res_height = int(settings["pre_res_height"])
    
    # NEW: Retrieve default anchor value from GUI settings
    default_zero_disparity_anchor = settings["zero_disparity_anchor"]

    is_single_file_mode = False
    input_videos = []
    finished_source_folder = None
    finished_depth_folder = None

    is_source_file = os.path.isfile(input_source_clips_path_setting)
    is_source_dir = os.path.isdir(input_source_clips_path_setting)
    is_depth_file = os.path.isfile(input_depth_maps_path_setting)
    is_depth_dir = os.path.isdir(input_depth_maps_path_setting)

    if is_source_file and is_depth_file:
        is_single_file_mode = True
        print("==> Running in single file mode. Files will not be moved to 'finished' folders.")
        input_videos.append(input_source_clips_path_setting)
        # Ensure output directory exists for single file output
        os.makedirs(output_splatted, exist_ok=True)
    elif is_source_dir and is_depth_dir:
        print("==> Running in batch (folder) mode.")
        finished_source_folder = os.path.join(input_source_clips_path_setting, "finished")
        finished_depth_folder = os.path.join(input_depth_maps_path_setting, "finished")
        os.makedirs(finished_source_folder, exist_ok=True)
        os.makedirs(finished_depth_folder, exist_ok=True)
        os.makedirs(output_splatted, exist_ok=True) # Ensure main output folder exists

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

    progress_queue.put(("total", len(input_videos)))

    for idx, video_path in enumerate(input_videos):
        if stop_event.is_set():
            print("==> Stopping processing due to user request")
            release_resources()
            progress_queue.put("finished")
            return

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n==> Processing Video: {video_name}")

        # Determine the initial anchor value from the GUI setting (this can be overridden by JSON)
        current_zero_disparity_anchor = default_zero_disparity_anchor
        # NEW: Determine the initial max_disparity percentage from the GUI setting (this can be overridden by JSON)
        current_max_disparity_percentage = gui_max_disp # <--- THIS LINE IS CRUCIAL AND MUST BE HERE

        # 1. Read input video frames at the determined pre-processing resolution
        # This resolution will be the target for depth maps and splatting output
        try:
            input_frames_processed, processed_fps, original_vid_h, original_vid_w, current_video_processed_height, current_video_processed_width = read_video_frames(
                video_path, process_length, target_fps=-1, # target_fps unused here, -1 default
                set_pre_res=set_pre_res, pre_res_width=pre_res_width, pre_res_height=pre_res_height
            )
        except Exception as e:
            print(f"==> Error reading input video {video_path}: {e}. Skipping this video.")
            progress_queue.put(("processed", idx + 1)) # Still count it as processed to move progress bar
            continue # Skip to next video

        actual_depth_map_path = None
        if is_single_file_mode:
            actual_depth_map_path = input_depth_maps_path_setting
            # If in single file mode, and the provided depth map doesn't exist, this is an error.
            if not os.path.exists(actual_depth_map_path):
                 print(f"==> Error: Single depth map file '{actual_depth_map_path}' not found. Skipping this video.")
                 progress_queue.put(("processed", idx + 1))
                 continue
        else: # Batch mode
            depth_map_path_mp4 = os.path.join(input_depth_maps_path_setting, f"{video_name}_depth.mp4")
            depth_map_path_npz = os.path.join(input_depth_maps_path_setting, f"{video_name}_depth.npz")

            if os.path.exists(depth_map_path_mp4):
                actual_depth_map_path = depth_map_path_mp4
            elif os.path.exists(depth_map_path_npz):
                actual_depth_map_path = depth_map_path_npz

        video_depth = None
        depth_vis = None

        if actual_depth_map_path:
            print(f"==> Found pre-rendered depth map: {actual_depth_map_path}")
            try:
                video_depth, depth_vis = load_pre_rendered_depth(
                    actual_depth_map_path,
                    process_length=process_length,
                    target_height=current_video_processed_height, # Pass the height of the processed video
                    target_width=current_video_processed_width,   # Pass the width of the processed video
                    match_resolution_to_target=True,
                    enable_autogain=settings["enable_autogain"] # NEW
                )
            except Exception as e:
                print(f"==> Error loading depth map from {actual_depth_map_path}: {e}. Skipping this video.")
                progress_queue.put(("processed", idx + 1))
                continue
        else:
            print(f"==> Error: No depth map found for {video_name} in {input_depth_maps_path_setting}. Expected '{video_name}_depth.mp4' or '{video_name}_depth.npz' in folder mode. Skipping this video.")
            progress_queue.put(("processed", idx + 1))
            continue

        if video_depth is None or depth_vis is None:
            print(f"==> Skipping video {video_name} due to depth load failure or stop request.")
            progress_queue.put(("processed", idx + 1))
            continue

        # Ensure the depth map resolution matches the input frames for splatting
        if not (video_depth.shape[1] == current_video_processed_height and video_depth.shape[2] == current_video_processed_width):
            print(f"==> Warning: Depth map resolution ({video_depth.shape[2]}x{video_depth.shape[1]}) does not match processed video resolution ({current_video_processed_width}x{current_video_processed_height}). This should have been handled by 'match_depth_res' being True. Attempting final resize as safeguard.")
            # As a safeguard, perform a final resize if mismatch persists (should be rare with match_depth_res=True)
            resized_video_depth = np.stack([cv2.resize(frame, (current_video_processed_width, current_video_processed_height), interpolation=cv2.INTER_LINEAR) for frame in video_depth], axis=0)
            resized_depth_vis = np.stack([cv2.resize(frame, (current_video_processed_width, current_video_processed_height), interpolation=cv2.INTER_LINEAR) for frame in depth_vis], axis=0)
            video_depth = resized_video_depth
            depth_vis = resized_depth_vis

        if stop_event.is_set():
            print("==> Stopping processing due to user request")
            release_resources()
            progress_queue.put("finished")
            return

        percentage_max_disp_input = float(settings["max_disp"]) # Get the percentage value from settings
        actual_percentage_for_calculation = percentage_max_disp_input / 20.0 
        actual_max_disp_pixels = (actual_percentage_for_calculation / 100.0) * current_video_processed_width
        print(f"==> Max Disparity Input: {current_max_disparity_percentage:.1f}% -> Calculated Max Disparity for splatting: {actual_max_disp_pixels:.2f} pixels")

        output_video_path_base = os.path.join(output_splatted, f"{video_name}.mp4") # This is the base path before _res_suffix_suffix is added
        DepthSplatting(
            input_frames_processed=input_frames_processed, # Pass already loaded frames
            processed_fps=processed_fps,                   # Pass calculated FPS
            output_video_path=output_video_path_base,
            video_depth=video_depth,
            depth_vis=depth_vis,
            max_disp=actual_max_disp_pixels,
            process_length=process_length,
            batch_size=batch_size,
            dual_output=dual_output,
            zero_disparity_anchor_val=current_zero_disparity_anchor # Pass the (potentially overridden) anchor value
        )
        if stop_event.is_set():
            print("==> Stopping after DepthSplatting due to user request")
            release_resources()
            progress_queue.put("finished")
            return
        # The DepthSplatting function returns the actual output path, but for simplicity we rely on its print statement
        print(f"==> Splatted video saved for {video_name}.") # Actual path printed by DepthSplatting

        if not stop_event.is_set() and not is_single_file_mode: # Only move if not single file mode
            if finished_source_folder is not None: # ADD THIS CHECK
                try:
                    shutil.move(video_path, finished_source_folder)
                    print(f"==> Moved processed video to: {finished_source_folder}")
                except Exception as e:
                    print(f"==> Failed to move video {video_path}: {e}")
            else:
                print(f"==> Cannot move source video: 'finished_source_folder' is not set.")

            if actual_depth_map_path and os.path.exists(actual_depth_map_path) and finished_depth_folder is not None: # ADD THIS CHECK
                try:
                    shutil.move(actual_depth_map_path, finished_depth_folder)
                    print(f"==> Moved depth map to: {finished_depth_folder}")
                except Exception as e:
                    print(f"==> Failed to move depth map {actual_depth_map_path}: {e}")
            elif actual_depth_map_path and finished_depth_folder is None:
                print(f"==> Cannot move depth map: 'finished_depth_folder' is not set.")
        elif is_single_file_mode:
            print(f"==> Single file mode for {video_name}: Skipping moving files to 'finished' folder.")

        progress_queue.put(("processed", idx + 1))
    release_resources()
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
    start_button.config(state="disabled")
    stop_button.config(state="normal")
    status_label.config(text="Starting processing...")

    # Input validation for new fields
    try:
        if set_pre_res_var.get():
            pre_res_w = int(pre_res_width_var.get())
            pre_res_h = int(pre_res_height_var.get())
            if pre_res_w <= 0 or pre_res_h <= 0:
                raise ValueError("Pre-processing Width and Height must be positive.")

        max_disp_val = float(max_disp_var.get())
        if max_disp_val <= 0:
            raise ValueError("Max Disparity must be positive.")
            
        # NEW: Validate Zero Disparity Anchor
        anchor_val = float(zero_disparity_anchor_var.get())
        if not (0.0 <= anchor_val <= 1.0):
            raise ValueError("Zero Disparity Anchor must be between 0.0 and 1.0.")

    except ValueError as e:
        status_label.config(text=f"Error: {e}")
        start_button.config(state="normal")
        stop_button.config(state="disabled")
        return

    settings = {
        "input_source_clips": input_source_clips_var.get(),
        "input_depth_maps": input_depth_maps_var.get(),
        "output_splatted": output_splatted_var.get(),
        "max_disp": float(max_disp_var.get()),
        "process_length": int(process_length_var.get()),
        "batch_size": int(batch_size_var.get()),
        "dual_output": dual_output_var.get(),
        "enable_low_res_output": False, # Permanently disabled
        "low_res_width": 0,             # Placeholder value
        "low_res_height": 0,            # Placeholder value
        "set_pre_res": set_pre_res_var.get(),
        "pre_res_width": pre_res_width_var.get(),
        "pre_res_height": pre_res_height_var.get(),
        "match_depth_res": True,         # Permanently set to True
        "zero_disparity_anchor": float(zero_disparity_anchor_var.get()),
        "enable_autogain": enable_autogain_var.get(), # NEW
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
                total_videos = message[1]
                progress_bar.config(maximum=total_videos)
                progress_var.set(0) # Start from 0
                status_label.config(text=f"Processing 0 of {total_videos}")
            elif message[0] == "processed":
                processed = message[1]
                total = progress_bar["maximum"]
                progress_var.set(processed)
                status_label.config(text=f"Processing {processed} of {total}")
    except queue.Empty:
        pass
    root.after(100, check_queue)

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
        "set_pre_res": set_pre_res_var.get(),
        "pre_res_width": pre_res_width_var.get(),
        "pre_res_height": pre_res_height_var.get(),
        "convergence_point": zero_disparity_anchor_var.get(),
        "enable_autogain": enable_autogain_var.get(),
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
            set_pre_res_var.set(config.get("set_pre_res", False))
            pre_res_width_var.set(config.get("pre_res_width", "1920"))
            pre_res_height_var.set(config.get("pre_res_height", "1080"))
            zero_disparity_anchor_var.set(config.get("convergence_point", "0.5"))
            enable_autogain_var.set(config.get("enable_autogain", True)) # NEW, default to True

# Load help texts at the start
load_help_texts()

# GUI Setup
root = tk.Tk()
root.title("Batch Depth Splatting")

# Variables with defaults
input_source_clips_var = tk.StringVar(value="./input_source_clips")
input_depth_maps_var = tk.StringVar(value="./input_depth_maps")
output_splatted_var = tk.StringVar(value="./output_splatted")
max_disp_var = tk.StringVar(value="20.0")
process_length_var = tk.StringVar(value="-1")
batch_size_var = tk.StringVar(value="10")
dual_output_var = tk.BooleanVar(value=False) # Default to Quad
set_pre_res_var = tk.BooleanVar(value=False)
pre_res_width_var = tk.StringVar(value="1920")
pre_res_height_var = tk.StringVar(value="1080")
zero_disparity_anchor_var = tk.StringVar(value="0.5") # NEW: Default to 0.5 (mid-ground anchor)
enable_autogain_var = tk.BooleanVar(value=True) # Default to True (current behavior)

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


# Pre-processing Settings Frame
preprocessing_frame = tk.LabelFrame(root, text="Pre-processing & Resolution Settings")
preprocessing_frame.pack(pady=10, padx=10, fill="x")

# Set Pre-processing Resolution
set_pre_res_checkbox = tk.Checkbutton(preprocessing_frame, text="Set Pre-processing Resolution", variable=set_pre_res_var)
set_pre_res_checkbox.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=2)
create_hover_tooltip(set_pre_res_checkbox, "set_pre_res")

pre_res_width_label = tk.Label(preprocessing_frame, text="Width:")
pre_res_width_label.grid(row=1, column=0, sticky="e", padx=5, pady=2)
pre_res_width_entry = tk.Entry(preprocessing_frame, textvariable=pre_res_width_var, width=10)
pre_res_width_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
create_hover_tooltip(pre_res_width_label, "pre_res_width")
create_hover_tooltip(pre_res_width_entry, "pre_res_width")

pre_res_height_label = tk.Label(preprocessing_frame, text="Height:")
pre_res_height_label.grid(row=1, column=2, sticky="e", padx=5, pady=2)
pre_res_height_entry = tk.Entry(preprocessing_frame, textvariable=pre_res_height_var, width=10)
pre_res_height_entry.grid(row=1, column=3, sticky="w", padx=5, pady=2)
create_hover_tooltip(pre_res_height_label, "pre_res_height")
create_hover_tooltip(pre_res_height_entry, "pre_res_height")


# Function to enable/disable pre-res input fields
def toggle_pre_res_fields():
    state = "normal" if set_pre_res_var.get() else "disabled"
    pre_res_width_label.config(state=state)
    pre_res_width_entry.config(state=state)
    pre_res_height_label.config(state=state)
    pre_res_height_entry.config(state=state)

# Trace the enable_low_res_output_var to call the toggle function
set_pre_res_var.trace_add("write", lambda *args: toggle_pre_res_fields())


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


lbl_batch_size = tk.Label(output_settings_frame, text="Batch Size:")
lbl_batch_size.grid(row=0, column=2, sticky="e", padx=5, pady=2)
entry_batch_size = tk.Entry(output_settings_frame, textvariable=batch_size_var, width=15)
entry_batch_size.grid(row=0, column=3, sticky="w", padx=5, pady=2)
create_hover_tooltip(lbl_batch_size, "batch_size")
create_hover_tooltip(entry_batch_size, "batch_size")


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

# New: Autogain/Normalization Checkbox
enable_autogain_checkbox = tk.Checkbutton(output_settings_frame, text="Enable Autogain (Min-Max Normalize Depth)", variable=enable_autogain_var)
enable_autogain_checkbox.grid(row=4, column=0, columnspan=4, sticky="w", padx=5, pady=2) # Spanning more columns
create_hover_tooltip(enable_autogain_checkbox, "enable_autogain") # Need to add this to splatter_help.json

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


# Initial calls to set the correct state based on loaded config
root.after(10, toggle_pre_res_fields)

# Run the GUI
root.mainloop()