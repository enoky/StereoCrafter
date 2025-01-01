import gc
import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import write_video

from diffusers.training_utils import set_seed
from fire import Fire
from decord import VideoReader, cpu

# Import your custom modules
from dependency.DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from dependency.DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from dependency.DepthCrafter.depthcrafter.utils import vis_sequence_depth

from Forward_Warp import forward_warp


def read_video_frames(video_path, process_length, target_fps, max_res, dataset="open"):
    """
    Reads and preprocesses video frames from the given video path.

    Args:
        video_path (str): Path to the input video.
        process_length (int): Number of frames to process. -1 for all.
        target_fps (int): Target frames per second. -1 to use original FPS.
        max_res (int): Maximum resolution for height or width.
        dataset (str): Dataset identifier (default: "open").

    Returns:
        frames (np.ndarray): Normalized video frames as a NumPy array.
        fps (float): Frames per second of the processed video.
        original_height (int): Original video height.
        original_width (int): Original video width.
    """
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
        # Placeholder for other datasets; extend as needed
        raise NotImplementedError(f"Dataset '{dataset}' not supported.")

    vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

    fps = vid.get_avg_fps() if target_fps == -1 else target_fps
    stride = round(vid.get_avg_fps() / fps)
    stride = max(stride, 1)
    frames_idx = list(range(0, len(vid), stride))
    print(f"==> Downsampled to {len(frames_idx)} frames with stride {stride}")
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    print(f"==> Final processing shape: {len(frames_idx)} frames, {vid.get_batch([0]).shape[1:]} per frame")
    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0

    return frames, fps, original_height, original_width


class DepthCrafterDemo:
    """
    A demo class for the DepthCrafter pipeline to infer depth maps from video frames.
    """

    def __init__(
        self,
        unet_path: str,
        pre_trained_path: str,
        cpu_offload: str = "model",
    ):
        """
        Initializes the DepthCrafterPipeline with the specified models.

        Args:
            unet_path (str): Path to the pre-trained UNet model.
            pre_trained_path (str): Path to the pre-trained DepthCrafter model.
            cpu_offload (str, optional): CPU offloading strategy ('model', 'sequential', or None). Default is 'model'.
        """
        print(f"==> Loading UNet model from {unet_path}")
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        print(f"==> Loading DepthCrafter pipeline from {pre_trained_path}")
        # Load weights of other components from the provided checkpoint
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_trained_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # For saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will slow down processing but save more memory
                print("==> Enabling sequential CPU offload")
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                print("==> Enabling model CPU offload")
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            print("==> Moving pipeline to CUDA")
            self.pipe.to("cuda")
        # Enable attention slicing and xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("==> Enabled Xformers memory efficient attention")
        except Exception as e:
            print("Xformers is not enabled:", e)
        self.pipe.enable_attention_slicing()
        print("==> Enabled attention slicing")

    def infer(
        self,
        input_video_path: str,
        output_video_path: str,
        process_length: int = -1,
        num_denoising_steps: int = 8,
        guidance_scale: float = 1.2,
        window_size: int = 70,
        overlap: int = 25,
        max_res: int = 1024,
        dataset: str = "open",
        target_fps: int = -1,
        seed: int = 42,
        track_time: bool = False,
        save_depth: bool = False,
    ):
        """
        Infers depth maps from the input video using the DepthCrafter pipeline.

        Args:
            input_video_path (str): Path to the input video.
            output_video_path (str): Path to save output visualizations (unused if save_depth is False).
            process_length (int, optional): Number of frames to process. -1 for all. Default is -1.
            num_denoising_steps (int, optional): Number of denoising steps for the pipeline. Default is 8.
            guidance_scale (float, optional): Guidance scale for the pipeline. Default is 1.2.
            window_size (int, optional): Window size parameter for the pipeline. Default is 70.
            overlap (int, optional): Overlap parameter for the pipeline. Default is 25.
            max_res (int, optional): Maximum resolution for processing. Default is 1024.
            dataset (str, optional): Dataset identifier. Default is "open".
            target_fps (int, optional): Target frames per second. -1 to use original FPS. Default is -1.
            seed (int, optional): Random seed for reproducibility. Default is 42.
            track_time (bool, optional): Whether to track processing time. Default is False.
            save_depth (bool, optional): Whether to save depth maps. Default is False.

        Returns:
            video_depth (np.ndarray): Depth maps with shape [T, H, W], normalized to [0, 1].
            depth_vis (np.ndarray): Visualized depth maps with shape [T, H, W, 3].
        """
        set_seed(seed)
        print(f"==> Reading video frames from {input_video_path}")
        frames, target_fps, original_height, original_width = read_video_frames(
            input_video_path,
            process_length,
            target_fps,
            max_res,
            dataset,
        )

        print("==> Performing depth inference using DepthCrafter pipeline")
        # Inference the depth map using the DepthCrafter pipeline
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
                track_time=track_time,
            ).frames[0]

        # Convert the three-channel output to a single channel depth map
        res = res.sum(-1) / res.shape[-1]

        # Resize the depth to the original size
        tensor_res = torch.tensor(res).unsqueeze(1).float().contiguous().cuda()
        res = F.interpolate(tensor_res, size=(original_height, original_width), mode='bilinear', align_corners=False)
        res = res.cpu().numpy()[:, 0, :, :]

        # Normalize the depth map to [0, 1] across the whole video
        res = (res - res.min()) / (res.max() - res.min())

        # Visualize the depth map and save the results
        print("==> Visualizing depth maps")
        vis = vis_sequence_depth(res)

        # Save the depth map and visualization with the target FPS
        save_path = os.path.join(
            os.path.dirname(output_video_path),
            os.path.splitext(os.path.basename(output_video_path))[0]
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_depth:
            print(f"==> Saving depth maps to {save_path}.npz")
            np.savez_compressed(save_path + ".npz", depth=res)
            print(f"==> Saving depth visualization video to {save_path}_depth_vis.mp4")
            write_video(save_path + "_depth_vis.mp4", vis * 255.0, fps=target_fps, video_codec="h264", options={"crf": "16"})

        return res, vis


class ForwardWarpStereo(nn.Module):
    """
    A module for performing forward warping for stereo projection based on disparity maps.
    """

    def __init__(self, eps=1e-6, occlu_map=False):
        """
        Initializes the ForwardWarpStereo module.

        Args:
            eps (float, optional): Epsilon value to prevent division by zero. Default is 1e-6.
            occlu_map (bool, optional): Whether to return an occlusion map. Default is False.
        """
        super(ForwardWarpStereo, self).__init__()
        self.eps = eps
        self.occlu_map = occlu_map
        self.fw = forward_warp()

    def forward(self, im, disp):
        """
        Performs forward warping on the input image using the disparity map.

        Args:
            im (torch.Tensor): Input image tensor with shape [B, C, H, W].
            disp (torch.Tensor): Disparity map tensor with shape [B, 1, H, W].

        Returns:
            res (torch.Tensor): Warped image tensor with shape [B, C, H, W].
            occlu_map (torch.Tensor, optional): Occlusion map tensor with shape [B, 1, H, W] if occlu_map is True.
        """
        im = im.contiguous()
        disp = disp.contiguous()
        # weights_map = torch.abs(disp)
        weights_map = disp - disp.min()
        weights_map = (
            1.414
        ) ** weights_map  # Using 1.414 instead of EXP to avoid numerical overflow.
        flow = -disp.squeeze(1)
        dummy_flow = torch.zeros_like(flow, requires_grad=False)
        flow = torch.stack((flow, dummy_flow), dim=-1)
        res_accum = self.fw(im * weights_map, flow)
        # mask = self.fw(weights_map, flow.detach())
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


def DepthSplatting(
        input_video_path, 
        output_video_path, 
        video_depth, 
        depth_vis, 
        max_disp, 
        process_length, 
        batch_size):
    """
    Performs depth-based video splatting using the provided depth maps.

    Args:
        input_video_path (str): Path to the input video.
        output_video_path (str): Path to the output video.
        video_depth (np.ndarray): Depth maps with shape [T, H, W] in [0, 1].
        depth_vis (np.ndarray): Visualized depth maps with shape [T, H, W, 3] in [0, 1].
        max_disp (float): Maximum disparity value for warping.
        process_length (int): Number of frames to process. -1 for all.
        batch_size (int): Batch size for processing to manage GPU memory.
    """
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

    # Initialize OpenCV VideoWriter
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(
        output_video_path, 
        cv2.VideoWriter_fourcc(*"mp4v"),
        original_fps, 
        (width * 2, height * 2)
    )
    print(f"==> Writing output video to: {output_video_path}")

    for i in range(0, num_frames, batch_size):
        batch_frames = input_frames[i:i+batch_size]
        batch_depth = video_depth[i:i+batch_size]
        batch_depth_vis = depth_vis[i:i+batch_size]

        # Convert to tensors and move to GPU
        left_video = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).float().cuda()
        disp_map = torch.from_numpy(batch_depth).unsqueeze(1).float().cuda()

        # Adjust disparity map
        disp_map = disp_map * 2.0 - 1.0
        disp_map = disp_map * max_disp

        with torch.no_grad():
            right_video, occlusion_mask = stereo_projector(left_video, disp_map)

        # Move tensors back to CPU and convert to NumPy
        right_video = right_video.cpu().permute(0, 2, 3, 1).numpy()
        occlusion_mask = occlusion_mask.cpu().permute(0, 2, 3, 1).numpy().repeat(3, axis=-1)

        for j in range(len(batch_frames)):
            # Create a grid: [Original | Depth Visualization]
            video_grid_top = np.concatenate([batch_frames[j], batch_depth_vis[j]], axis=1)
            # Create a grid: [Occlusion Mask | Warped Right View]
            video_grid_bottom = np.concatenate([occlusion_mask[j], right_video[j]], axis=1)
            # Combine top and bottom grids
            video_grid = np.concatenate([video_grid_top, video_grid_bottom], axis=0)

            # Convert to uint8 and BGR for OpenCV
            video_grid_uint8 = np.clip(video_grid * 255.0, 0, 255).astype(np.uint8)
            video_grid_bgr = cv2.cvtColor(video_grid_uint8, cv2.COLOR_RGB2BGR)
            out.write(video_grid_bgr)

        # Free up GPU memory
        del left_video, disp_map, right_video, occlusion_mask
        torch.cuda.empty_cache()
        gc.collect()

        print(f"==> Processed frames {i+1} to {min(i+batch_size, num_frames)}")

    out.release()
    print("==> Output video writing completed")


def load_pre_rendered_depth(depth_video_path, process_length=-1, max_res=1024):
    """
    Loads pre-rendered depth maps from an MP4 file.

    Args:
        depth_video_path (str): Path to the depth maps video file.
        process_length (int): Number of frames to process. -1 for all.
        max_res (int): Maximum resolution to resize depth maps.

    Returns:
        video_depth (np.ndarray): Depth maps with shape [T, H, W], normalized to [0, 1].
        depth_vis (np.ndarray): Visualized depth maps with shape [T, H, W, 3].
    """
    print(f"==> Loading pre-rendered depth maps from: {depth_video_path}")
    vid = VideoReader(depth_video_path, ctx=cpu(0))
    original_fps = vid.get_avg_fps()
    frames = vid[:].asnumpy().astype("float32") / 255.0  # Normalize to [0,1]

    if process_length != -1 and process_length < len(frames):
        frames = frames[:process_length]

    # Assuming depth maps are grayscale; convert to single channel
    if frames.shape[-1] == 3:
        # Convert RGB to grayscale if necessary
        print("==> Converting RGB depth frames to grayscale")
        video_depth = frames.mean(axis=-1)
    else:
        video_depth = frames.squeeze(-1)

    # Normalize depth maps to [0,1]
    video_depth_min = video_depth.min()
    video_depth_max = video_depth.max()
    if video_depth_max - video_depth_min > 1e-5:
        video_depth = (video_depth - video_depth_min) / (video_depth_max - video_depth_min)
    else:
        video_depth = np.zeros_like(video_depth)

    # Create visualized depth maps (apply colormap to each frame individually)
    print("==> Applying colormap to depth maps")
    depth_vis = []
    for idx, frame in enumerate(video_depth):
        # Ensure the frame is 2D
        if frame.ndim == 2:
            colored_frame = cv2.applyColorMap((frame * 255).astype(np.uint8), cv2.COLORMAP_JET)
            colored_frame = colored_frame.astype("float32") / 255.0
            depth_vis.append(colored_frame)
        else:
            raise ValueError(f"Expected 2D frame for applyColorMap, but got shape {frame.shape} at index {idx}")
    depth_vis = np.stack(depth_vis, axis=0)

    print("==> Depth maps and visualizations loaded successfully")
    return video_depth, depth_vis


def main(
    input_source_clips: str = "./input_source_clips",
    input_depth_maps: str = "./input_depth_maps",
    output_splatted: str = "./output_splatted",
    unet_path: str = "./weights/DepthCrafter",
    pre_trained_path: str = "./weights/stable-video-diffusion-img2vid-xt-1-1",
    max_disp: float = 20.0,
    process_length: int = -1,
    batch_size: int = 10
):
    """
    Main function to perform batch Depth Splatting using either inferred or pre-rendered depth maps.

    Args:
        input_source_clips (str, optional): Path to the folder containing input source videos. Default is "./input_source_clips".
        input_depth_maps (str, optional): Path to the folder containing pre-rendered depth maps. Default is "./input_depth_maps".
        output_splatted (str, optional): Path to the folder where output splatted videos will be saved. Default is "./output_splatted".
        unet_path (str, optional): Path to the pre-trained UNet model. Required if not using pre-rendered depth.
        pre_trained_path (str, optional): Path to the pre-trained DepthCrafter model. Required if not using pre-rendered depth.
        max_disp (float, optional): Maximum disparity for splatting. Default is 20.0.
        process_length (int, optional): Number of frames to process per video. -1 for all. Default is -1.
        batch_size (int, optional): Batch size for processing to manage GPU memory. Default is 10.
    """
    print("==> Starting Batch Depth Splatting Process")
    print(f"Input Source Clips Folder: {input_source_clips}")
    print(f"Input Depth Maps Folder: {input_depth_maps}")
    print(f"Output Splatted Folder: {output_splatted}")
    print(f"Max Disparity: {max_disp}")
    print(f"Process Length per Video: {process_length}")
    print(f"Batch Size: {batch_size}")

    # Ensure output directory exists
    os.makedirs(output_splatted, exist_ok=True)

    # Gather all video files in the input_source_clips folder
    video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
    input_videos = []
    for ext in video_extensions:
        input_videos.extend(glob.glob(os.path.join(input_source_clips, ext)))
    input_videos = sorted(input_videos)

    if not input_videos:
        print(f"No video files found in {input_source_clips} with extensions {video_extensions}")
        return

    # Initialize DepthCrafterDemo if depth inference is needed
    if unet_path and pre_trained_path:
        depthcrafter_demo = DepthCrafterDemo(
            unet_path=unet_path,
            pre_trained_path=pre_trained_path
        )
    else:
        depthcrafter_demo = None
        print("==> Depth inference will be skipped. Ensure pre-rendered depth maps are available for all videos.")

    for video_path in input_videos:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n==> Processing Video: {video_name}")

        # Corresponding depth map path
        depth_map_path = os.path.join(input_depth_maps, f"{video_name}.mp4")
        if os.path.exists(depth_map_path):
            print(f"==> Found pre-rendered depth map: {depth_map_path}")
            video_depth, depth_vis = load_pre_rendered_depth(
                depth_map_path,
                process_length=process_length
            )
        elif depthcrafter_demo:
            print(f"==> Pre-rendered depth map not found. Performing depth inference.")
            # Output path for depth visualization (can be ignored if not saving depth)
            temp_depth_vis_path = os.path.join(output_splatted, f"{video_name}_depth_vis_temp.mp4")
            video_depth, depth_vis = depthcrafter_demo.infer(
                input_video_path=video_path,
                output_video_path=temp_depth_vis_path,  # Temporary path for visualization
                process_length=process_length
            )
        else:
            print(f"==> No pre-rendered depth map found for {video_name} and DepthCrafterDemo is not initialized. Skipping.")
            continue

        # Output splatted video path
        output_video_path = os.path.join(output_splatted, f"{video_name}_splatted.mp4")

        # Perform Depth Splatting
        DepthSplatting(
            input_video_path=video_path,
            output_video_path=output_video_path,
            video_depth=video_depth,
            depth_vis=depth_vis,
            max_disp=max_disp,
            process_length=process_length,
            batch_size=batch_size
        )
        print(f"==> Splatted video saved to: {output_video_path}")

    print("\n==> Batch Depth Splatting Process Completed Successfully")


if __name__ == "__main__":
    Fire(main)
