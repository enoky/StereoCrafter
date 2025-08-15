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
from diffusers.training_utils import set_seed
from decord import VideoReader, cpu
import tkinter as tk
from tkinter import filedialog, ttk
import json
import threading
import queue
import subprocess

# Import custom modules
from dependency.DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from dependency.DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from dependency.DepthCrafter.depthcrafter.utils import vis_sequence_depth
# from Forward_Warp import forward_warp
from dependency.forward_warp_pytorch import forward_warp

# torch.backends.cudnn.benchmark = True

# Global variables for GUI control
stop_event = threading.Event()
progress_queue = queue.Queue()
depthcrafter_instance = None
processing_thread = None

def round_to_nearest_64(value):
    return max(64, round(value / 64) * 64)

def read_video_frames(video_path, process_length, target_fps,
                      set_pre_res, pre_res_width, pre_res_height, dataset="open"):
    """
    Reads video frames and determines the processing resolution.
    Resolution is determined by set_pre_res, otherwise original resolution is used (rounded to 64).
    max_res is NOT used here; it's handled during DepthCrafter inference.
    """
    if dataset == "open":
        print(f"==> Processing video: {video_path}")
        vid_info = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = vid_info.get_batch([0]).shape[1:3]
        print(f"==> Original video shape: {len(vid_info)} frames, {original_height}x{original_width} per frame")
        
        height_for_processing = original_height
        width_for_processing = original_width

        if set_pre_res and pre_res_width > 0 and pre_res_height > 0:
            # User specified pre-processing resolution, overrides max_res
            height_for_processing = round_to_nearest_64(pre_res_height)
            width_for_processing = round_to_nearest_64(pre_res_width)
            print(f"==> Pre-processing video to user-specified resolution: {width_for_processing}x{height_for_processing}")
        else:
            # If set_pre_res is False, use original resolution rounded to 64.
            height_for_processing = round_to_nearest_64(original_height)
            width_for_processing = round_to_nearest_64(original_width)
            print(f"==> Using original video resolution ({original_width}x{original_height}) rounded to nearest 64 for processing: {width_for_processing}x{height_for_processing}")

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

class DepthCrafterDemo:
    def __init__(self, unet_path: str, pre_trained_path: str, cpu_offload: str = "sequential"):
        print(f"==> Loading UNet model from {unet_path}")
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path, low_cpu_mem_usage=True, torch_dtype=torch.float16
        )
        print(f"==> Loading DepthCrafter pipeline from {pre_trained_path}")
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_trained_path, unet=unet, torch_dtype=torch.float16, variant="fp16"
        )
        if cpu_offload == "sequential":
            print("==> Enabling sequential CPU offload")
            self.pipe.enable_sequential_cpu_offload()
        elif cpu_offload == "model":
            print("==> Enabling model CPU offload")
            self.pipe.enable_model_cpu_offload()
        elif cpu_offload == "none":
            print("==> Moving pipeline to CUDA")
            self.pipe.to("cuda")
        else:
            raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("==> Enabled Xformers memory efficient attention")
        except Exception as e:
            print("Xformers is not enabled:", e)
        self.pipe.enable_attention_slicing()
        print("==> Enabled attention slicing")

    def infer(
        self, frames_to_process: np.ndarray, output_video_path: str, processed_fps: float,
        num_denoising_steps: int = 5, guidance_scale: float = 1.0, window_size: int = 70,
        overlap: int = 25, max_inference_res: int = 960, seed: int = 42, save_depth: bool = False
    ):
        """
        Performs depth inference on pre-loaded frames.
        max_inference_res caps the resolution for the UNet model's processing,
        but depth output is scaled back to the original frames_to_process resolution.
        """
        set_seed(seed)
        if stop_event.is_set():
            print("==> Depth inference aborted due to stop request")
            return None, None
        
        current_frames_height, current_frames_width = frames_to_process.shape[1:3]
        
        inference_height = current_frames_height
        inference_width = current_frames_width

        # Apply max_inference_res as a cap for the model's processing dimensions
        if max_inference_res > 0 and max(current_frames_height, current_frames_width) > max_inference_res:
            scale = max_inference_res / max(current_frames_height, current_frames_width)
            inference_height = round_to_nearest_64(current_frames_height * scale)
            inference_width = round_to_nearest_64(current_frames_width * scale)
            print(f"==> Input frames exceed max_inference_res ({max_inference_res}). Downscaling for inference to: {inference_width}x{inference_height}")
        else:
             print(f"==> Inferring at input resolution: {current_frames_width}x{current_frames_height}")

        print("==> Performing depth inference using DepthCrafter pipeline")
        with torch.inference_mode():
            res = self.pipe(
                frames_to_process, height=inference_height, width=inference_width, output_type="np",
                guidance_scale=guidance_scale, num_inference_steps=num_denoising_steps,
                window_size=window_size, overlap=overlap
            ).frames[0]

        res = res.sum(-1) / res.shape[-1]
        tensor_res = torch.tensor(res).unsqueeze(1).float().contiguous().cuda()
        
        # Interpolate depth back to the *input resolution* of frames_to_process
        # This ensures depth map matches the video frames it will be splatted with.
        res = F.interpolate(tensor_res, size=(current_frames_height, current_frames_width), mode='bilinear', align_corners=False)
        res = res.cpu().numpy()[:, 0, :, :]
        res = (res - res.min()) / (res.max() - res.min())
        print("==> Visualizing depth maps")
        vis = vis_sequence_depth(res) # vis will also be at current_frames_height/width

        save_path = os.path.splitext(output_video_path)[0]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_depth:
            print(f"==> Saving depth maps to {save_path}.npz at processed resolution.")
            np.savez_compressed(save_path + ".npz", depth=res)
            print(f"==> Saving depth visualization video to {save_path}_depth_vis.mp4 at processed resolution.")
            write_video(save_path + "_depth_vis.mp4", vis * 255.0, fps=processed_fps, video_codec="h264", options={"crf": "16"})
        
        return res, vis # res and vis are now at current_frames_height/width

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
                   dual_output: bool, enable_low_res_output: bool, low_res_width: int, low_res_height: int):
    
    print("==> Initializing ForwardWarpStereo module")
    stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()
    
    num_frames = len(input_frames_processed)
    height, width, _ = input_frames_processed[0].shape # Get dimensions from already processed frames
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Determine output video suffix and dimensions
    if dual_output:
        suffix = "_splatted2"
        output_video_width = width * 2 # Mask + Warped
        output_video_height = height
    else: # Quad output
        suffix = "_splatted4"
        output_video_width = width * 2
        output_video_height = height * 2

    # Update the main output path
    base_output_name = os.path.splitext(output_video_path)[0]
    main_output_video_path = f"{base_output_name}{suffix}.mp4"

    # Initialize the main video writer
    main_out = cv2.VideoWriter(
        main_output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        processed_fps, # Use the fps of the processed video
        (output_video_width, output_video_height),
        True
    )
    print(f"==> Writing main output video to: {main_output_video_path}")

    low_res_out = None
    if enable_low_res_output:
        # Determine low-res output path
        low_res_output_video_path = f"{base_output_name}_low{suffix}.mp4"

        # Validate low_res_width/height (ensure they are positive)
        if low_res_width <= 0 or low_res_height <= 0:
            print("==> Warning: Low resolution dimensions must be positive. Disabling low-res output.")
            enable_low_res_output = False # Disable it if dimensions are invalid
        else:
            # Calculate the overall grid dimensions for the low-resolution output
            if dual_output:
                low_res_output_grid_width = low_res_width * 2
                low_res_output_grid_height = low_res_height
            else: # Quad output
                low_res_output_grid_width = low_res_width * 2
                low_res_output_grid_height = low_res_height * 2

            low_res_out = cv2.VideoWriter(
                low_res_output_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                processed_fps,
                (low_res_output_grid_width, low_res_output_grid_height),
                True
            )
            print(f"==> Writing low-resolution output video to: {low_res_output_video_path}")
            
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
            if low_res_out: low_res_out.release()
            return
        
        batch_frames = input_frames_processed[i:i+batch_size]
        batch_depth = video_depth[i:i+batch_size]
        batch_depth_vis = depth_vis[i:i+batch_size]
        
        left_video = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).float().cuda()
        disp_map = torch.from_numpy(batch_depth).unsqueeze(1).float().cuda()
        disp_map = disp_map * 2.0 - 1.0
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

            video_grid_uint8 = np.clip(video_grid * 255.0, 0, 255).astype(np.uint8)
            video_grid_bgr = cv2.cvtColor(video_grid_uint8, cv2.COLOR_RGB2BGR)

            # Write to main output video
            main_out.write(video_grid_bgr)

            # Write to low-resolution output video if enabled
            if enable_low_res_output and low_res_out is not None:
                resized_frame = cv2.resize(video_grid_bgr, (low_res_output_grid_width, low_res_output_grid_height), interpolation=cv2.INTER_AREA)
                low_res_out.write(resized_frame)
        del left_video, disp_map, right_video, occlusion_mask
        torch.cuda.empty_cache()
        gc.collect()
        print(f"==> Processed frames {i+1} to {min(i+batch_size, num_frames)}")
    main_out.release()
    if low_res_out is not None:
        low_res_out.release()
    print("==> Output video writing completed.")

def load_pre_rendered_depth(depth_video_path, process_length=-1, target_height=-1, target_width=-1, match_resolution_to_target=False):
    """
    Loads pre-rendered depth maps. If match_resolution_to_target is True, it resizes
    the depth maps to the target_height/width for compatibility.
    """
    print(f"==> Loading pre-rendered depth maps from: {depth_video_path}")
    vid = VideoReader(depth_video_path, ctx=cpu(0))
    frames = vid[:].asnumpy().astype("float32") / 255.0
    if process_length != -1 and process_length < len(frames):
        frames = frames[:process_length]
    
    video_depth_raw = None
    if frames.shape[-1] == 3:
        print("==> Converting RGB depth frames to grayscale")
        video_depth_raw = frames.mean(axis=-1)
    else:
        video_depth_raw = frames.squeeze(-1)
    
    # Normalize depth
    video_depth_min = video_depth_raw.min()
    video_depth_max = video_depth_raw.max()
    if video_depth_max - video_depth_min > 1e-5:
        video_depth_normalized = (video_depth_raw - video_depth_min) / (video_depth_max - video_depth_min)
    else:
        video_depth_normalized = np.zeros_like(video_depth_raw)

    # Resize if matching resolution to target video
    if match_resolution_to_target and target_height > 0 and target_width > 0:
        print(f"==> Resizing loaded depth maps to target resolution: {target_width}x{target_height}")
        resized_depths = []
        resized_viss = []
        for i in range(video_depth_normalized.shape[0]):
            depth_frame = video_depth_normalized[i]
            # OpenCV expects (width, height)
            resized_depth_frame = cv2.resize(depth_frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized_depths.append(resized_depth_frame)
            
            # Apply colormap to the resized depth for visualization
            vis_frame = cv2.applyColorMap((resized_depth_frame * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            resized_viss.append(vis_frame.astype("float32") / 255.0)

        video_depth = np.stack(resized_depths, axis=0)
        depth_vis = np.stack(resized_viss, axis=0)
    else:
        print(f"==> Not resizing loaded depth maps. Current resolution: {video_depth_normalized.shape[2]}x{video_depth_normalized.shape[1]}")
        video_depth = video_depth_normalized
        depth_vis = np.stack([cv2.applyColorMap((frame * 255).astype(np.uint8), cv2.COLORMAP_INFERNO).astype("float32") / 255.0 for frame in video_depth_normalized], axis=0)
    
    print("==> Depth maps and visualizations loaded successfully")
    return video_depth, depth_vis

def release_resources():
    global depthcrafter_instance
    if depthcrafter_instance is not None:
        print("==> Releasing DepthCrafter resources")
        try:
            del depthcrafter_instance
            depthcrafter_instance = None
        except Exception as e:
            print(f"==> Error releasing DepthCrafter: {e}")
    try:
        torch.cuda.empty_cache()
        gc.collect()
        print("==> VRAM and resources released")
    except Exception as e:
        print(f"==> Error releasing VRAM: {e}")

def main(settings):
    global depthcrafter_instance
    input_source_clips = settings["input_source_clips"]
    input_depth_maps = settings["input_depth_maps"]
    output_splatted = settings["output_splatted"]
    unet_path = settings["unet_path"]
    pre_trained_path = settings["pre_trained_path"]
    cpu_offload = settings["cpu_offload"]
    max_disp = settings["max_disp"]
    process_length = settings["process_length"]
    batch_size = settings["batch_size"]
    num_denoising_steps = settings["num_denoising_steps"]
    guidance_scale = settings["guidance_scale"]
    window_size = settings["window_size"]
    overlap = settings["overlap"]
    max_inference_res = settings["max_inference_res"] # Renamed from max_res
    seed = settings["seed"]
    save_depth = settings["save_depth"]
    dual_output = settings["dual_output"]
    enable_low_res_output = settings["enable_low_res_output"]
    low_res_width = settings["low_res_width"]
    low_res_height = settings["low_res_height"]
    set_pre_res = settings["set_pre_res"]
    pre_res_width = int(settings["pre_res_width"])
    pre_res_height = int(settings["pre_res_height"])
    match_depth_res = settings["match_depth_res"]

    os.makedirs(output_splatted, exist_ok=True)
    finished_source_folder = os.path.join(input_source_clips, "finished")
    finished_depth_folder = os.path.join(input_depth_maps, "finished")
    os.makedirs(finished_source_folder, exist_ok=True)
    os.makedirs(finished_depth_folder, exist_ok=True)

    video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
    input_videos = []
    for ext in video_extensions:
        input_videos.extend(glob.glob(os.path.join(input_source_clips, ext)))
    input_videos = sorted(input_videos)

    if not input_videos:
        print(f"No video files found in {input_source_clips}")
        progress_queue.put("finished")
        release_resources()
        return

    progress_queue.put(("total", len(input_videos)))
    depthcrafter_instance = DepthCrafterDemo(unet_path=unet_path, pre_trained_path=pre_trained_path, cpu_offload=cpu_offload)

    for idx, video_path in enumerate(input_videos):
        if stop_event.is_set():
            print("==> Stopping processing due to user request")
            release_resources()
            progress_queue.put("finished")
            return
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n==> Processing Video: {video_name}")

        # 1. Read input video frames at the determined pre-processing resolution
        # This resolution will be the target for depth maps and splatting output
        try:
            input_frames_processed, processed_fps, original_vid_h, original_vid_w, current_video_processed_height, current_video_processed_width = read_video_frames(
                video_path, process_length, target_fps=-1, # target_fps unused here, -1 default
                set_pre_res=set_pre_res, pre_res_width=pre_res_width, pre_res_height=pre_res_height
            )
        except Exception as e:
            print(f"==> Error reading input video {video_path}: {e}. Skipping this video.")
            continue # Skip to next video

        depth_map_path = os.path.join(input_depth_maps, f"{video_name}_depth.mp4")
        used_pre_rendered_depth = False
        video_depth = None
        depth_vis = None

        if os.path.exists(depth_map_path):
            print(f"==> Found pre-rendered depth map: {depth_map_path}")
            try:
                video_depth, depth_vis = load_pre_rendered_depth(
                    depth_map_path,
                    process_length=process_length,
                    target_height=current_video_processed_height, # Pass the height of the processed video
                    target_width=current_video_processed_width,   # Pass the width of the processed video
                    match_resolution_to_target=match_depth_res    # Use new setting
                )
                used_pre_rendered_depth = True
            except Exception as e:
                print(f"==> Error loading pre-rendered depth: {e}")
                video_depth, depth_vis = None, None
        else:
            print(f"==> Pre-rendered depth map not found. Performing depth inference.")
            temp_depth_vis_path = os.path.join(output_splatted, f"{video_name}_depth_vis_temp.mp4")
            video_depth, depth_vis = depthcrafter_instance.infer(
                frames_to_process=input_frames_processed, # Pass already loaded and sized frames
                output_video_path=temp_depth_vis_path,
                processed_fps=processed_fps,
                num_denoising_steps=num_denoising_steps,
                guidance_scale=guidance_scale,
                window_size=window_size,
                overlap=overlap,
                max_inference_res=max_inference_res, # Use this for internal inference cap
                seed=seed,
                save_depth=save_depth
            )
        
        if video_depth is None or depth_vis is None:
            print("==> Skipping video due to depth inference/load failure or stop request")
            continue
        
        # Ensure the depth map resolution matches the input frames for splatting
        if not (video_depth.shape[1] == current_video_processed_height and video_depth.shape[2] == current_video_processed_width):
            print(f"==> Warning: Depth map resolution ({video_depth.shape[2]}x{video_depth.shape[1]}) does not match processed video resolution ({current_video_processed_width}x{current_video_processed_height}). Attempting resize for splatting.")
            # This case should ideally not happen with the new logic, but as a safeguard:
            resized_video_depth = np.stack([cv2.resize(frame, (current_video_processed_width, current_video_processed_height), interpolation=cv2.INTER_LINEAR) for frame in video_depth], axis=0)
            resized_depth_vis = np.stack([cv2.resize(frame, (current_video_processed_width, current_video_processed_height), interpolation=cv2.INTER_LINEAR) for frame in depth_vis], axis=0)
            video_depth = resized_video_depth
            depth_vis = resized_depth_vis

        if stop_event.is_set():
            print("==> Stopping processing due to user request")
            release_resources()
            progress_queue.put("finished")
            return

        output_video_path = os.path.join(output_splatted, f"{video_name}.mp4")
        DepthSplatting(
            input_frames_processed=input_frames_processed, # Pass already loaded frames
            processed_fps=processed_fps,                   # Pass calculated FPS
            output_video_path=output_video_path,
            video_depth=video_depth,
            depth_vis=depth_vis,
            max_disp=max_disp,
            process_length=process_length,
            batch_size=batch_size,
            dual_output=dual_output,
            enable_low_res_output=enable_low_res_output,
            low_res_width=low_res_width,
            low_res_height=low_res_height
        )
        if stop_event.is_set():
            print("==> Stopping after DepthSplatting due to user request")
            release_resources()
            progress_queue.put("finished")
            return
        print(f"==> Splatted video saved to: {output_video_path}")
        if not stop_event.is_set():
            try:
                shutil.move(video_path, finished_source_folder)
                print(f"==> Moved processed video to: {finished_source_folder}")
            except Exception as e:
                print(f"==> Failed to move video {video_path}: {e}")
            if used_pre_rendered_depth and os.path.exists(depth_map_path):
                try:
                    shutil.move(depth_map_path, finished_depth_folder)
                    print(f"==> Moved depth map to: {finished_depth_folder}")
                except Exception as e:
                    print(f"==> Failed to move depth map {depth_map_path}: {e}")
        progress_queue.put(("processed", idx + 1))
    release_resources()
    progress_queue.put("finished")
    print("\n==> Batch Depth Splatting Process Completed Successfully")

def browse_folder(var):
    folder = filedialog.askdirectory()
    if folder:
        var.set(folder)

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

        # Validate max_inference_res_var
        max_inf_res = int(max_res_var.get())
        # No explicit check for positive, as 0 is now a valid "disable" value
        
    except ValueError as e:
        status_label.config(text=f"Error: {e}")
        start_button.config(state="normal")
        stop_button.config(state="disabled")
        return
    
    settings = {
        "input_source_clips": input_source_clips_var.get(),
        "input_depth_maps": input_depth_maps_var.get(),
        "output_splatted": output_splatted_var.get(),
        "unet_path": unet_path_var.get(),
        "pre_trained_path": pre_trained_path_var.get(),
        "cpu_offload": cpu_offload_var.get(),
        "max_disp": float(max_disp_var.get()),
        "process_length": int(process_length_var.get()),
        "batch_size": int(batch_size_var.get()),
        "num_denoising_steps": int(num_denoising_steps_var.get()),
        "guidance_scale": float(guidance_scale_var.get()),
        "window_size": int(window_size_var.get()),
        "overlap": int(overlap_var.get()),
        "max_inference_res": int(max_res_var.get()), # Pass this renamed variable
        "seed": int(seed_var.get()),
        "save_depth": save_depth_var.get(),
        "dual_output": dual_output_var.get(),
        "enable_low_res_output": enable_low_res_output_var.get(),
        "low_res_width": int(low_res_width_var.get()),
        "low_res_height": int(low_res_height_var.get()),
        "set_pre_res": set_pre_res_var.get(),
        "pre_res_width": pre_res_width_var.get(),
        "pre_res_height": pre_res_height_var.get(),
        "match_depth_res": match_depth_res_var.get()
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
        processing_thread.join(timeout=5.0)  # Ensure thread is terminated
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
                progress_var.set(processed)
                total = progress_bar["maximum"]
                status_label.config(text=f"Processing {processed} of {total}")
    except queue.Empty:
        pass
    root.after(100, check_queue)

def save_config():
    config = {
        "input_source_clips": input_source_clips_var.get(),
        "input_depth_maps": input_depth_maps_var.get(),
        "output_splatted": output_splatted_var.get(),
        "unet_path": unet_path_var.get(),
        "pre_trained_path": pre_trained_path_var.get(),
        "cpu_offload": cpu_offload_var.get(),
        "max_disp": max_disp_var.get(),
        "process_length": process_length_var.get(),
        "batch_size": batch_size_var.get(),
        "num_denoising_steps": num_denoising_steps_var.get(),
        "guidance_scale": guidance_scale_var.get(),
        "window_size": window_size_var.get(),
        "overlap": overlap_var.get(),
        "max_inference_res": max_res_var.get(), # Save under new name
        "seed": seed_var.get(),
        "save_depth": save_depth_var.get(),
        "dual_output": dual_output_var.get(),
        "enable_low_res_output": enable_low_res_output_var.get(),
        "low_res_width": low_res_width_var.get(),
        "low_res_height": low_res_height_var.get(),
        "set_pre_res": set_pre_res_var.get(),
        "pre_res_width": pre_res_width_var.get(),
        "pre_res_height": pre_res_height_var.get(),
        "match_depth_res": match_depth_res_var.get()
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
            unet_path_var.set(config.get("unet_path", "./weights/DepthCrafter"))
            pre_trained_path_var.set(config.get("pre_trained_path", "./weights/stable-video-diffusion-img2vid-xt-1-1"))
            cpu_offload_var.set(config.get("cpu_offload", "model"))
            max_disp_var.set(config.get("max_disp", "20.0"))
            process_length_var.set(config.get("process_length", "-1"))
            batch_size_var.set(config.get("batch_size", "10"))
            num_denoising_steps_var.set(config.get("num_denoising_steps", "5"))
            guidance_scale_var.set(config.get("guidance_scale", "1.0"))
            window_size_var.set(config.get("window_size", "70"))
            overlap_var.set(config.get("overlap", "25"))
            max_res_var.set(config.get("max_inference_res", "960")) # Load from new name
            seed_var.set(config.get("seed", "42"))
            save_depth_var.set(config.get("save_depth", False))
            dual_output_var.set(config.get("dual_output", False))
            enable_low_res_output_var.set(config.get("enable_low_res_output", False))
            low_res_width_var.set(config.get("low_res_width", "1024"))
            low_res_height_var.set(config.get("low_res_height", "576"))
            set_pre_res_var.set(config.get("set_pre_res", False))
            pre_res_width_var.set(config.get("pre_res_width", "1920"))
            pre_res_height_var.set(config.get("pre_res_height", "1080"))
            match_depth_res_var.set(config.get("match_depth_res", False))

# GUI Setup
root = tk.Tk()
root.title("Batch Depth Splatting")

# Variables with defaults
input_source_clips_var = tk.StringVar(value="./input_source_clips")
input_depth_maps_var = tk.StringVar(value="./input_depth_maps")
output_splatted_var = tk.StringVar(value="./output_splatted")
unet_path_var = tk.StringVar(value="./weights/DepthCrafter")
pre_trained_path_var = tk.StringVar(value="./weights/stable-video-diffusion-img2vid-xt-1-1")
cpu_offload_var = tk.StringVar(value="sequential")
max_disp_var = tk.StringVar(value="20.0")
process_length_var = tk.StringVar(value="-1")
batch_size_var = tk.StringVar(value="10")
num_denoising_steps_var = tk.StringVar(value="5")
guidance_scale_var = tk.StringVar(value="1.0")
window_size_var = tk.StringVar(value="70")
overlap_var = tk.StringVar(value="25")
max_res_var = tk.StringVar(value="960") # This now represents max_inference_res
seed_var = tk.StringVar(value="42")
save_depth_var = tk.BooleanVar(value=False)
dual_output_var = tk.BooleanVar(value=False) # Default to Quad
enable_low_res_output_var = tk.BooleanVar(value=False)
low_res_width_var = tk.StringVar(value="1024")
low_res_height_var = tk.StringVar(value="576")
set_pre_res_var = tk.BooleanVar(value=False)
pre_res_width_var = tk.StringVar(value="1920")
pre_res_height_var = tk.StringVar(value="1080")
match_depth_res_var = tk.BooleanVar(value=False)

# Load configuration
load_config()

# Folder selection frame
folder_frame = tk.Frame(root)
folder_frame.pack(pady=10)
tk.Label(folder_frame, text="Input Source Clips:").grid(row=0, column=0, sticky="e")
tk.Entry(folder_frame, textvariable=input_source_clips_var, width=50).grid(row=0, column=1)
tk.Button(folder_frame, text="Browse", command=lambda: browse_folder(input_source_clips_var)).grid(row=0, column=2)
tk.Label(folder_frame, text="Input Depth Maps (optional):").grid(row=1, column=0, sticky="e")
tk.Entry(folder_frame, textvariable=input_depth_maps_var, width=50).grid(row=1, column=1)
tk.Button(folder_frame, text="Browse", command=lambda: browse_folder(input_depth_maps_var)).grid(row=1, column=2)
tk.Label(folder_frame, text="Output Splatted:").grid(row=2, column=0, sticky="e")
tk.Entry(folder_frame, textvariable=output_splatted_var, width=50).grid(row=2, column=1)
tk.Button(folder_frame, text="Browse", command=lambda: browse_folder(output_splatted_var)).grid(row=2, column=2)

# Pre-processing Settings Frame
preprocessing_frame = tk.LabelFrame(root, text="Pre-processing & Resolution Settings")
preprocessing_frame.pack(pady=10, padx=10, fill="x")

# Set Pre-processing Resolution
set_pre_res_checkbox = tk.Checkbutton(preprocessing_frame, text="Set Pre-processing Resolution", variable=set_pre_res_var)
set_pre_res_checkbox.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=2)

pre_res_width_label = tk.Label(preprocessing_frame, text="Width:")
pre_res_width_label.grid(row=1, column=0, sticky="e", padx=5, pady=2)
pre_res_width_entry = tk.Entry(preprocessing_frame, textvariable=pre_res_width_var, width=10)
pre_res_width_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)

pre_res_height_label = tk.Label(preprocessing_frame, text="Height:")
pre_res_height_label.grid(row=1, column=2, sticky="e", padx=5, pady=2)
pre_res_height_entry = tk.Entry(preprocessing_frame, textvariable=pre_res_height_var, width=10)
pre_res_height_entry.grid(row=1, column=3, sticky="w", padx=5, pady=2)

# Match Depthmap Resolution
match_depth_res_checkbox = tk.Checkbutton(preprocessing_frame, text="Match Depthmap Resolution to Processed Source", variable=match_depth_res_var)
match_depth_res_checkbox.grid(row=2, column=0, columnspan=4, sticky="w", padx=5, pady=2)

# Function to enable/disable pre-res input fields
def toggle_pre_res_fields():
    state = "normal" if set_pre_res_var.get() else "disabled"
    pre_res_width_label.config(state=state)
    pre_res_width_entry.config(state=state)
    pre_res_height_label.config(state=state)
    pre_res_height_entry.config(state=state)

# Trace the enable_low_res_output_var to call the toggle function
set_pre_res_var.trace_add("write", lambda *args: toggle_pre_res_fields())


# DepthCrafter settings frame
depthcrafter_frame = tk.LabelFrame(root, text="DepthCrafter Inference Settings")
depthcrafter_frame.pack(pady=10, padx=10, fill="x")
tk.Label(depthcrafter_frame, text="CPU Offload:").grid(row=0, column=0, sticky="e")
cpu_offload_menu = ttk.Combobox(depthcrafter_frame, textvariable=cpu_offload_var, values=["model", "sequential", "none"], state="readonly")
cpu_offload_menu.grid(row=0, column=1, sticky="w")
tk.Label(depthcrafter_frame, text="Num Denoising Steps:").grid(row=1, column=0, sticky="e")
tk.Entry(depthcrafter_frame, textvariable=num_denoising_steps_var).grid(row=1, column=1)
tk.Label(depthcrafter_frame, text="Guidance Scale:").grid(row=2, column=0, sticky="e")
tk.Entry(depthcrafter_frame, textvariable=guidance_scale_var).grid(row=2, column=1)
tk.Label(depthcrafter_frame, text="Window Size:").grid(row=3, column=0, sticky="e")
tk.Entry(depthcrafter_frame, textvariable=window_size_var).grid(row=3, column=1)
tk.Label(depthcrafter_frame, text="Overlap:").grid(row=4, column=0, sticky="e")
tk.Entry(depthcrafter_frame, textvariable=overlap_var).grid(row=4, column=1)
tk.Label(depthcrafter_frame, text="Max Inference Resolution (0 to disable cap):").grid(row=5, column=0, sticky="e")
tk.Entry(depthcrafter_frame, textvariable=max_res_var).grid(row=5, column=1)
tk.Label(depthcrafter_frame, text="Seed:").grid(row=6, column=0, sticky="e")
tk.Entry(depthcrafter_frame, textvariable=seed_var).grid(row=6, column=1)
tk.Checkbutton(depthcrafter_frame, text="Save Inferred Depth Maps", variable=save_depth_var).grid(row=7, column=0, columnspan=2, sticky="w")


# Output Settings Frame
output_settings_frame = tk.LabelFrame(root, text="Splatting Output Settings")
output_settings_frame.pack(pady=10, padx=10, fill="x")

# Max Disparity and Batch Size within Output Settings (or keep them where they are if preferred, but for this instruction, they are here)
tk.Label(output_settings_frame, text="Max Disparity:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
tk.Entry(output_settings_frame, textvariable=max_disp_var, width=15).grid(row=0, column=1, sticky="w", padx=5, pady=2)

tk.Label(output_settings_frame, text="Batch Size:").grid(row=0, column=2, sticky="e", padx=5, pady=2)
tk.Entry(output_settings_frame, textvariable=batch_size_var, width=15).grid(row=0, column=3, sticky="w", padx=5, pady=2)

# Dual Output Checkbox
dual_output_checkbox = tk.Checkbutton(output_settings_frame, text="Dual Output (Mask & Warped)", variable=dual_output_var)
dual_output_checkbox.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=2)

# Enable 2nd downscaled output Checkbox
enable_low_res_output_checkbox = tk.Checkbutton(output_settings_frame, text="Enable 2nd Downscaled Output", variable=enable_low_res_output_var)
enable_low_res_output_checkbox.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=2)

# Low Resolution Width and Height fields
low_res_width_label = tk.Label(output_settings_frame, text="Low Res Width:")
low_res_width_label.grid(row=3, column=0, sticky="e", padx=5, pady=2)
low_res_width_entry = tk.Entry(output_settings_frame, textvariable=low_res_width_var, width=10)
low_res_width_entry.grid(row=3, column=1, sticky="w", padx=5, pady=2)

low_res_height_label = tk.Label(output_settings_frame, text="Low Res Height:")
low_res_height_label.grid(row=3, column=2, sticky="e", padx=5, pady=2)
low_res_height_entry = tk.Entry(output_settings_frame, textvariable=low_res_height_var, width=10)
low_res_height_entry.grid(row=3, column=3, sticky="w", padx=5, pady=2)

# Function to enable/disable low-res input fields
def toggle_low_res_fields():
    state = "normal" if enable_low_res_output_var.get() else "disabled"
    low_res_width_label.config(state=state)
    low_res_width_entry.config(state=state)
    low_res_height_label.config(state=state)
    low_res_height_entry.config(state=state)

# Trace the enable_low_res_output_var to call the toggle function
enable_low_res_output_var.trace_add("write", lambda *args: toggle_low_res_fields())

# Initial calls to set the correct state based on loaded config
root.after(10, toggle_low_res_fields)
root.after(10, toggle_pre_res_fields)


# Progress frame
progress_frame = tk.Frame(root)
progress_frame.pack(pady=10)
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100)
progress_bar.pack(fill="x", expand=True)
status_label = tk.Label(progress_frame, text="Ready")
status_label.pack()

# Button frame
button_frame = tk.Frame(root)
button_frame.pack(pady=10)
start_button = tk.Button(button_frame, text="START", command=start_processing)
start_button.pack(side="left", padx=5)
stop_button = tk.Button(button_frame, text="STOP", command=stop_processing, state="disabled")
stop_button.pack(side="left", padx=5)
exit_button = tk.Button(button_frame, text="EXIT", command=exit_app)
exit_button.pack(side="left", padx=5)

# Run the GUI
root.mainloop()