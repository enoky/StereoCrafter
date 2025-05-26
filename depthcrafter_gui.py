import threading
import gc
import os
import glob
import shutil
import json
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import queue

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers.models.transformers.transformer_2d")

from diffusers.training_utils import set_seed
from dependency.DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from dependency.DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from dependency.DepthCrafter.depthcrafter.utils import save_video, read_video_frames
import gc

class DepthCrafterDemo:
    """
    Class to handle DepthCrafter inference with optimized VRAM usage.
    """
    def __init__(self, unet_path: str, pre_train_path: str, cpu_offload: str = "model", use_cudnn_benchmark: bool = True):
        """
        Initializes the DepthCrafter pipeline.

        Args:
            unet_path (str): Path to the UNet model.
            pre_train_path (str): Path to the pre-trained model.
            cpu_offload (str, optional): CPU offload strategy ('model', 'sequential'). Defaults to "model".
            use_cudnn_benchmark (bool, optional): Enable cuDNN benchmarking. Defaults to True.
        """
        torch.backends.cudnn.benchmark = use_cudnn_benchmark
        print(f"==> Loading UNet model from {unet_path}")
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path, low_cpu_mem_usage=True, torch_dtype=torch.float16,
        )
        print(f"==> Loading DepthCrafter pipeline from {pre_train_path}")
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path, unet=unet, torch_dtype=torch.float16, variant="fp16",
        )
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("==> Enabled xformers memory-efficient attention")
        except Exception as e:
            print(f"Warning: xformers not enabled: {e}")
        if cpu_offload == "sequential":
            print("==> Enabling sequential CPU offload")
            self.pipe.enable_sequential_cpu_offload()
        elif cpu_offload == "model":
            print("==> Enabling model CPU offload")
            self.pipe.enable_model_cpu_offload()
        else:
            raise ValueError(f"Unknown CPU offload option: {cpu_offload}")
        self.pipe.enable_attention_slicing()
        print("==> Enabled attention slicing")

    def infer(self, video, num_denoising_steps, guidance_scale, save_folder, window_size, process_length, overlap, max_res, seed):
        """
        Performs depth inference on a video with optimized frame processing.

        Args:
            video (str): Path to the input video.
            num_denoising_steps (int): Number of denoising steps.
            guidance_scale (float): Guidance scale for inference.
            save_folder (str): Folder to save the depth map video.
            window_size (int): Window size for temporal processing.
            process_length (int): Number of frames to process.
            overlap (int): Overlap between windows.
            max_res (int): Maximum resolution of the video.
            seed (int): Random seed for reproducibility.

        Returns:
            str: Path to the saved depth map video.
        """
        set_seed(seed)
        frames, target_fps = read_video_frames(video, process_length, -1, max_res, "open")
        with torch.inference_mode():
            res = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
            ).frames[0]
        res = res.sum(-1) / res.shape[-1]
        res_min, res_max = res.min(), res.max()
        if res_max != res_min:
            res = (res - res_min) / (res_max - res_min)
        else:
            res = np.zeros_like(res)
        save_path = os.path.join(save_folder, os.path.splitext(os.path.basename(video))[0])
        os.makedirs(save_folder, exist_ok=True)
        save_video(res, save_path + "_depth.mp4", fps=target_fps)
        return save_path + "_depth.mp4"

    def run(self, video, **kwargs):
        """
        Runs depth inference with cleanup.

        Args:
            video (str): Path to the input video.
            **kwargs: Additional parameters for inference.
        """
        save_path = self.infer(video, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()
        return save_path

class DepthCrafterGUI:
    """
    GUI class for the DepthCrafter application.
    """
    CONFIG_FILENAME = "config_dc.json"

    def __init__(self, root):
        """
        Initializes the GUI with default settings.

        Args:
            root (tk.Tk): The main Tkinter window.
        """
        self.root = root
        self.root.title("DepthCrafter GUI")
        
        self.input_dir = tk.StringVar(value="./input_clips")
        self.output_dir = tk.StringVar(value="./output_depthmaps")
        self.guidance_scale = tk.DoubleVar(value=1.0)
        self.inference_steps = tk.IntVar(value=5)
        self.window_size = tk.IntVar(value=70)
        self.max_res = tk.IntVar(value=960)
        self.overlap = tk.IntVar(value=25)
        self.seed = tk.IntVar(value=42)
        self.cpu_offload = tk.StringVar(value="model")
        self.use_cudnn_benchmark = tk.BooleanVar(value=True)

        self.load_config()
        self.message_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.processing_thread = None
        self.create_widgets()
        self.root.after(100, self.process_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        """Creates and arranges GUI widgets."""
        frame = tk.LabelFrame(self.root, text="Directories")
        frame.pack(fill="x", padx=10, pady=5)
        tk.Label(frame, text="Input Folder:").grid(row=0, column=0, sticky="e")
        tk.Entry(frame, textvariable=self.input_dir, width=50).grid(row=0, column=1)
        tk.Button(frame, text="Browse", command=self.browse_input).grid(row=0, column=2)
        tk.Label(frame, text="Output Folder:").grid(row=1, column=0, sticky="e")
        tk.Entry(frame, textvariable=self.output_dir, width=50).grid(row=1, column=1)
        tk.Button(frame, text="Browse", command=self.browse_output).grid(row=1, column=2)

        param_frame = tk.LabelFrame(self.root, text="Parameters")
        param_frame.pack(fill="x", padx=10, pady=5)
        self.add_param(param_frame, "Guidance Scale", self.guidance_scale, 0)
        self.add_param(param_frame, "Inference Steps", self.inference_steps, 1)
        self.add_dropdown(param_frame, "Window Size", self.window_size, ["70", "90", "110"], 2)
        self.add_dropdown(param_frame, "Max Resolution", self.max_res, ["576", "640", "704", "768", "832", "896", "960", "1024", "1088"], 3)
        self.add_param(param_frame, "Overlap", self.overlap, 4)
        self.add_param(param_frame, "Seed", self.seed, 5)
        tk.Label(param_frame, text="CPU Offload Mode:").grid(row=6, column=0, sticky="e")
        ttk.Combobox(param_frame, textvariable=self.cpu_offload, values=["model", "sequential"]).grid(row=6, column=1, padx=5)
        tk.Label(param_frame, text="Enable cuDNN Benchmark:").grid(row=7, column=0, sticky="e")
        tk.Checkbutton(param_frame, variable=self.use_cudnn_benchmark).grid(row=7, column=1, padx=5, sticky="w")

        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(pady=10)
        self.progress = ttk.Progressbar(ctrl_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(side="left", padx=5)
        tk.Button(ctrl_frame, text="Start", command=self.start_thread).pack(side="left", padx=5)
        tk.Button(ctrl_frame, text="Cancel", command=self.stop_processing).pack(side="left", padx=5)
        tk.Button(ctrl_frame, text="Exit", command=self.on_close).pack(side="right", padx=5)

        log_frame = tk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.log = tk.Text(log_frame, state="disabled", height=10)
        self.log.pack(fill="both", expand=True)

    def add_param(self, parent, label, var, row):
        """Helper to add a parameter entry field."""
        tk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="e")
        tk.Entry(parent, textvariable=var).grid(row=row, column=1, padx=5, pady=2)

    def add_dropdown(self, parent, label, var, values, row):
        """Helper to add a parameter dropdown menu."""
        tk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="e")
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly")
        combo.grid(row=row, column=1, padx=5, pady=2)
        try:
            combo.current(values.index(str(var.get())))
        except ValueError:
            combo.current(0)

    def browse_input(self):
        """Selects input folder."""
        folder = filedialog.askdirectory(initialdir=self.input_dir.get())
        if folder:
            self.input_dir.set(os.path.normpath(folder))

    def browse_output(self):
        """Selects output folder."""
        folder = filedialog.askdirectory(initialdir=self.output_dir.get())
        if folder:
            self.output_dir.set(os.path.normpath(folder))

    def log_message(self, message):
        """Queues a log message for GUI update."""
        self.message_queue.put(("log", message))

    def process_queue(self):
        """Updates GUI from the message queue."""
        while not self.message_queue.empty():
            message = self.message_queue.get()
            if message[0] == "log":
                self.log.config(state="normal")
                self.log.insert("end", f"{message[1]}\n")
                self.log.config(state="disabled")
                self.log.see("end")
            elif message[0] == "progress":
                self.progress["value"] = message[1]
        self.root.after(100, self.process_queue)

    def start_thread(self):
        """Initiates processing after handling file overwrite prompts."""
        videos = []
        for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
            videos.extend(glob.glob(os.path.join(self.input_dir.get(), ext)))
        
        to_process = []
        for video in videos:
            save_path = os.path.join(self.output_dir.get(), os.path.splitext(os.path.basename(video))[0] + "_depth.mp4")
            if not os.path.exists(save_path):
                to_process.append(video)
            elif messagebox.askyesno("Overwrite?", f"{save_path} already exists. Overwrite?"):
                to_process.append(video)
            else:
                self.log_message(f"Skipping {video}")

        if to_process:
            if self.processing_thread is None or not self.processing_thread.is_alive():
                self.processing_thread = threading.Thread(target=self.start_processing, args=(to_process,), daemon=True)
                self.processing_thread.start()
        else:
            self.log_message("No videos to process.")

    def start_processing(self, videos):
        """
        Processes videos with lazy model loading and progress updates.

        Args:
            videos (list): List of video paths to process.
        """
        self.stop_event.clear()
        self.message_queue.put(("log", "Starting processing..."))
        self.message_queue.put(("progress", 0))
        self.progress["maximum"] = len(videos)

        demo = DepthCrafterDemo(
            unet_path="tencent/DepthCrafter",
            pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
            cpu_offload=self.cpu_offload.get(),
            use_cudnn_benchmark=self.use_cudnn_benchmark.get(),
        )
        finished_folder = os.path.join(self.input_dir.get(), "finished")
        os.makedirs(finished_folder, exist_ok=True)

        for i, video in enumerate(videos):
            if self.stop_event.is_set():
                self.message_queue.put(("log", "Processing cancelled."))
                break
            self.message_queue.put(("log", f"Processing {video}"))
            try:
                demo.run(
                    video,
                    num_denoising_steps=self.inference_steps.get(),
                    guidance_scale=self.guidance_scale.get(),
                    save_folder=self.output_dir.get(),
                    window_size=self.window_size.get(),
                    process_length=-1,
                    overlap=self.overlap.get(),
                    max_res=self.max_res.get(),
                    seed=self.seed.get(),
                )
                shutil.move(video, finished_folder)
                self.message_queue.put(("progress", i + 1))
            except Exception as e:
                self.message_queue.put(("log", f"Error processing {video}: {e}"))
        self.message_queue.put(("log", "Processing complete!"))

    def stop_processing(self):
        """Cancels the ongoing processing."""
        self.stop_event.set()
        self.log_message("Cancelling processing...")

    def on_close(self):
        """Saves config and closes the application."""
        self.save_config()
        self.root.destroy()

    def save_config(self):
        """Saves current settings to a config file."""
        config = {
            "input_dir": self.input_dir.get(),
            "output_dir": self.output_dir.get(),
            "guidance_scale": self.guidance_scale.get(),
            "inference_steps": self.inference_steps.get(),
            "window_size": self.window_size.get(),
            "max_res": self.max_res.get(),
            "overlap": self.overlap.get(),
            "seed": self.seed.get(),
            "cpu_offload": self.cpu_offload.get(),
            "use_cudnn_benchmark": self.use_cudnn_benchmark.get(),
        }
        with open(self.CONFIG_FILENAME, "w") as f:
            json.dump(config, f, indent=4)

    def load_config(self):
        """Loads settings from a config file if it exists."""
        if os.path.exists(self.CONFIG_FILENAME):
            try:
                with open(self.CONFIG_FILENAME, "r") as f:
                    config = json.load(f)
                self.input_dir.set(os.path.normpath(config.get("input_dir", "./input_clips")))
                self.output_dir.set(os.path.normpath(config.get("output_dir", "./output_depthmaps")))
                self.guidance_scale.set(config.get("guidance_scale", 1.0))
                self.inference_steps.set(config.get("inference_steps", 5))
                self.window_size.set(config.get("window_size", 70))
                self.max_res.set(config.get("max_res", 960))
                self.overlap.set(config.get("overlap", 25))
                self.seed.set(config.get("seed", 42))
                self.cpu_offload.set(config.get("cpu_offload", "model"))
                self.use_cudnn_benchmark.set(config.get("use_cudnn_benchmark", True))
            except Exception as e:
                print(f"Warning: Could not load config: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DepthCrafterGUI(root)
    root.mainloop()
