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

# Import custom modules
from dependency.DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from dependency.DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from dependency.DepthCrafter.depthcrafter.utils import vis_sequence_depth
from Forward_Warp import forward_warp

# Global variables for GUI control
stop_event = threading.Event()
progress_queue = queue.Queue()
depthcrafter_instance = None
processing_thread = None

def read_video_frames(video_path, process_length, target_fps, max_res, dataset="open"):
    if dataset == "open":
        print(f"==> Processing video: {video_path}")
        vid = VideoReader(video_path, ctx=cpu(0))
        print(f"==> Original video shape: {len(vid)} frames, {vid.get_batch([0]).shape[1:]} per frame")
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        height = round(original_height / 64) * 64
        width = round(original_width / 64) * 64
        if max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale / 64) * 64
            width = round(original_width * scale / 64) * 64
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not supported.")

    vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
    fps = vid.get_avg_fps() if target_fps == -1 else target_fps
    stride = max(round(vid.get_avg_fps() / fps), 1)
    frames_idx = list(range(0, len(vid), stride))
    print(f"==> Downsampled to {len(frames_idx)} frames with stride {stride}")
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    print(f"==> Final processing shape: {len(frames_idx)} frames, {vid.get_batch([0]).shape[1:]} per frame")
    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0
    return frames, fps, original_height, original_width

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
        self, input_video_path: str, output_video_path: str, process_length: int = -1,
        num_denoising_steps: int = 5, guidance_scale: float = 1.0, window_size: int = 70,
        overlap: int = 25, max_res: int = 960, seed: int = 42, save_depth: bool = False
    ):
        set_seed(seed)
        print(f"==> Reading video frames from {input_video_path}")
        frames, target_fps, original_height, original_width = read_video_frames(
            input_video_path, process_length, target_fps=-1, max_res=max_res
        )
        if stop_event.is_set():
            print("==> Depth inference aborted due to stop request")
            return None, None
        print("==> Performing depth inference using DepthCrafter pipeline")
        with torch.inference_mode():
            res = self.pipe(
                frames, height=frames.shape[1], width=frames.shape[2], output_type="np",
                guidance_scale=guidance_scale, num_inference_steps=num_denoising_steps,
                window_size=window_size, overlap=overlap
            ).frames[0]
        res = res.sum(-1) / res.shape[-1]
        tensor_res = torch.tensor(res).unsqueeze(1).float().contiguous().cuda()
        res = F.interpolate(tensor_res, size=(original_height, original_width), mode='bilinear', align_corners=False)
        res = res.cpu().numpy()[:, 0, :, :]
        res = (res - res.min()) / (res.max() - res.min())
        print("==> Visualizing depth maps")
        vis = vis_sequence_depth(res)
        save_path = os.path.splitext(output_video_path)[0]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_depth:
            print(f"==> Saving depth maps to {save_path}.npz")
            np.savez_compressed(save_path + ".npz", depth=res)
            print(f"==> Saving depth visualization video to {save_path}_depth_vis.mp4")
            write_video(save_path + "_depth_vis.mp4", vis * 255.0, fps=target_fps, video_codec="h264", options={"crf": "16"})
        return res, vis

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

def DepthSplatting(input_video_path, output_video_path, video_depth, depth_vis, max_disp, process_length, batch_size):
    print(f"==> Reading input video for splatting: {input_video_path}")
    vid_reader = VideoReader(input_video_path, ctx=cpu(0))
    original_fps = vid_reader.get_avg_fps()
    input_frames = vid_reader[:].asnumpy() / 255.0
    if process_length != -1 and process_length < len(input_frames):
        input_frames = input_frames[:process_length]
        video_depth = video_depth[:process_length]
        depth_vis = depth_vis[:process_length]
    print("==> Initializing ForwardWarpStereo module")
    stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()
    num_frames = len(input_frames)
    height, width, _ = input_frames[0].shape
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), original_fps, (width * 2, height * 2))
    print(f"==> Writing output video to: {output_video_path}")
    for i in range(0, num_frames, batch_size):
        if stop_event.is_set():
            print("==> Stopping DepthSplatting due to user request")
            del stereo_projector
            torch.cuda.empty_cache()
            gc.collect()
            out.release()
            return
        batch_frames = input_frames[i:i+batch_size]
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
            video_grid_top = np.concatenate([batch_frames[j], batch_depth_vis[j]], axis=1)
            video_grid_bottom = np.concatenate([occlusion_mask[j], right_video[j]], axis=1)
            video_grid = np.concatenate([video_grid_top, video_grid_bottom], axis=0)
            video_grid_uint8 = np.clip(video_grid * 255.0, 0, 255).astype(np.uint8)
            video_grid_bgr = cv2.cvtColor(video_grid_uint8, cv2.COLOR_RGB2BGR)
            out.write(video_grid_bgr)
        del left_video, disp_map, right_video, occlusion_mask
        torch.cuda.empty_cache()
        gc.collect()
        print(f"==> Processed frames {i+1} to {min(i+batch_size, num_frames)}")
    out.release()
    print("==> Output video writing completed")

def load_pre_rendered_depth(depth_video_path, process_length=-1, max_res=1024):
    print(f"==> Loading pre-rendered depth maps from: {depth_video_path}")
    vid = VideoReader(depth_video_path, ctx=cpu(0))
    frames = vid[:].asnumpy().astype("float32") / 255.0
    if process_length != -1 and process_length < len(frames):
        frames = frames[:process_length]
    if frames.shape[-1] == 3:
        print("==> Converting RGB depth frames to grayscale")
        video_depth = frames.mean(axis=-1)
    else:
        video_depth = frames.squeeze(-1)
    video_depth_min = video_depth.min()
    video_depth_max = video_depth.max()
    if video_depth_max - video_depth_min > 1e-5:
        video_depth = (video_depth - video_depth_min) / (video_depth_max - video_depth_min)
    else:
        video_depth = np.zeros_like(video_depth)
    print("==> Applying colormap to depth maps")
    depth_vis = [cv2.applyColorMap((frame * 255).astype(np.uint8), cv2.COLORMAP_INFERNO).astype("float32") / 255.0 for frame in video_depth]
    depth_vis = np.stack(depth_vis, axis=0)
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
    max_res = settings["max_res"]
    seed = settings["seed"]
    save_depth = settings["save_depth"]

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
        depth_map_path = os.path.join(input_depth_maps, f"{video_name}_depth.mp4")
        used_pre_rendered_depth = False
        if os.path.exists(depth_map_path):
            print(f"==> Found pre-rendered depth map: {depth_map_path}")
            try:
                video_depth, depth_vis = load_pre_rendered_depth(depth_map_path, process_length=process_length, max_res=max_res)
                used_pre_rendered_depth = True
            except Exception as e:
                print(f"==> Error loading pre-rendered depth: {e}")
                video_depth, depth_vis = None, None
        else:
            print(f"==> Pre-rendered depth map not found. Performing depth inference.")
            temp_depth_vis_path = os.path.join(output_splatted, f"{video_name}_depth_vis_temp.mp4")
            video_depth, depth_vis = depthcrafter_instance.infer(
                input_video_path=video_path, output_video_path=temp_depth_vis_path,
                process_length=process_length, num_denoising_steps=num_denoising_steps,
                guidance_scale=guidance_scale, window_size=window_size, overlap=overlap,
                max_res=max_res, seed=seed, save_depth=save_depth
            )
        if video_depth is None or depth_vis is None:
            print("==> Skipping video due to depth inference failure or stop request")
            continue
        if stop_event.is_set():
            print("==> Stopping processing due to user request")
            release_resources()
            progress_queue.put("finished")
            return
        output_video_path = os.path.join(output_splatted, f"{video_name}_splatted.mp4")
        DepthSplatting(
            input_video_path=video_path, output_video_path=output_video_path,
            video_depth=video_depth, depth_vis=depth_vis, max_disp=max_disp,
            process_length=process_length, batch_size=batch_size
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
        "max_res": int(max_res_var.get()),
        "seed": int(seed_var.get()),
        "save_depth": save_depth_var.get()
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
                break
            elif message[0] == "total":
                total_videos = message[1]
                progress_bar.config(maximum=total_videos)
                status_label.config(text=f"Processing 1 of {total_videos}")
            elif message[0] == "processed":
                processed = message[1]
                progress_var.set(processed)
                total = progress_bar["maximum"]
                status_label.config(text=f"Processing {processed + 1} of {total}")
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
        "max_res": max_res_var.get(),
        "seed": seed_var.get(),
        "save_depth": save_depth_var.get()
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
            max_res_var.set(config.get("max_res", "1024"))
            seed_var.set(config.get("seed", "42"))
            save_depth_var.set(config.get("save_depth", False))

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
max_res_var = tk.StringVar(value="960")
seed_var = tk.StringVar(value="42")
save_depth_var = tk.BooleanVar(value=False)

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

# DepthCrafter settings frame
depthcrafter_frame = tk.Frame(root)
depthcrafter_frame.pack(pady=10)
tk.Label(depthcrafter_frame, text="DepthCrafter Settings:").grid(row=0, column=0, columnspan=2)
tk.Label(depthcrafter_frame, text="CPU Offload:").grid(row=3, column=0, sticky="e")
cpu_offload_menu = ttk.Combobox(depthcrafter_frame, textvariable=cpu_offload_var, values=["model", "sequential", "none"], state="readonly")
cpu_offload_menu.grid(row=3, column=1, sticky="w")
tk.Label(depthcrafter_frame, text="Num Denoising Steps:").grid(row=4, column=0, sticky="e")
tk.Entry(depthcrafter_frame, textvariable=num_denoising_steps_var).grid(row=4, column=1)
tk.Label(depthcrafter_frame, text="Guidance Scale:").grid(row=5, column=0, sticky="e")
tk.Entry(depthcrafter_frame, textvariable=guidance_scale_var).grid(row=5, column=1)
tk.Label(depthcrafter_frame, text="Window Size:").grid(row=6, column=0, sticky="e")
tk.Entry(depthcrafter_frame, textvariable=window_size_var).grid(row=6, column=1)
tk.Label(depthcrafter_frame, text="Overlap:").grid(row=7, column=0, sticky="e")
tk.Entry(depthcrafter_frame, textvariable=overlap_var).grid(row=7, column=1)
tk.Label(depthcrafter_frame, text="Max Resolution:").grid(row=8, column=0, sticky="e")
tk.Entry(depthcrafter_frame, textvariable=max_res_var).grid(row=8, column=1)
tk.Label(depthcrafter_frame, text="Seed:").grid(row=9, column=0, sticky="e")
tk.Entry(depthcrafter_frame, textvariable=seed_var).grid(row=9, column=1)
tk.Checkbutton(depthcrafter_frame, text="Save Depth Maps", variable=save_depth_var).grid(row=10, column=0, columnspan=2)

# DepthSplatting settings frame
splatting_frame = tk.Frame(root)
splatting_frame.pack(pady=10)
tk.Label(splatting_frame, text="DepthSplatting Settings:").grid(row=0, column=0, columnspan=2)
tk.Label(splatting_frame, text="Max Disparity:").grid(row=1, column=0, sticky="e")
tk.Entry(splatting_frame, textvariable=max_disp_var).grid(row=1, column=1)
tk.Label(splatting_frame, text="Batch Size:").grid(row=2, column=0, sticky="e")
tk.Entry(splatting_frame, textvariable=batch_size_var).grid(row=2, column=1)

# Progress frame
progress_frame = tk.Frame(root)
progress_frame.pack(pady=10)
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100)
progress_bar.pack()
status_label = tk.Label(progress_frame, text="")
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