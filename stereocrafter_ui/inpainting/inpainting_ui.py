
"""
Inpainting WebUI Component - Complete Gradio Implementation API compatible
Combines features from tkinter GUI, Flask reference, and existing Gradio implementation
"""

import os
import threading
import queue
import glob
import json
import shutil
import numpy as np
import torch
import gradio as gr
from typing import Optional, Tuple, List
import torch.nn.functional as F
import time
import cv2
from decord import VideoReader, cpu
import base64
from io import BytesIO
from PIL import Image as PILImage

from pipelines.stereo_video_inpainting import (
    load_inpainting_pipeline, 
    StableVideoDiffusionInpaintingPipeline, 
    tensor2vid
)

# Import additional components for local loading
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor


def load_inpainting_pipeline_local(
    svd_path: str,
    unet_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    offload_type: str = "model"
) -> StableVideoDiffusionInpaintingPipeline:
    """
    Load inpainting pipeline from local weights folder.
    Loads StereoCrafter UNet directly without subfolder (HF repo has no unet_diffusers folder).
    """
    logger.info("Loading pipeline from local weights...")
    
    # Load components from local paths
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        os.path.join(svd_path, "image_encoder"),
        variant="fp16",
        torch_dtype=dtype,
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        os.path.join(svd_path, "vae"),
        variant="fp16",
        torch_dtype=dtype,
    )
    # Load UNet directly from StereoCrafter root (no unet_diffusers subfolder in HF repo)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )
    
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    # Load feature extractor and scheduler from SVD path
    feature_extractor = CLIPImageProcessor.from_pretrained(
        os.path.join(svd_path, "feature_extractor")
    )
    scheduler = EulerDiscreteScheduler.from_pretrained(
        os.path.join(svd_path, "scheduler")
    )
    
    # Create pipeline
    pipeline = StableVideoDiffusionInpaintingPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
    )
    pipeline = pipeline.to(device, dtype=dtype)
    
    # Configure attention processors
    attention_set = False
    if AttnProcessor2_0 is not None:
        try:
            pipeline.unet.set_attn_processor(AttnProcessor2_0())
            logger.info("Efficient attention (AttnProcessor2_0) enabled for UNet")
            attention_set = True
        except Exception as e:
            logger.warning(f"Failed to enable AttnProcessor2_0: {e}")
    if not attention_set and XFormersAttnProcessor is not None:
        try:
            pipeline.unet.set_attn_processor(XFormersAttnProcessor())
            logger.info("xFormers attention enabled for UNet")
            attention_set = True
        except Exception as e:
            logger.warning(f"Failed to enable xFormers attention: {e}")
    if not attention_set:
        logger.info("Using default attention processor")
    
    # Apply offloading
    if offload_type == "model":
        pipeline.enable_model_cpu_offload()
    elif offload_type == "sequential":
        pipeline.enable_sequential_cpu_offload()
    elif offload_type == "none":
        pass
    else:
        raise ValueError("Invalid offload_type")
    
    return pipeline

from core.common.cli_utils import draw_progress_bar, set_logger_level
from core.common.gpu_utils import release_cuda_memory
from core.common.video_io import get_video_stream_info, encode_frames_to_mp4, read_video_frames_decord
import logging

logger = logging.getLogger(__name__)

# ==================== HELPER FUNCTIONS (moved to top for availability) ====================

def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """Blend two tensors horizontally"""
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1) / overlap_size).to(b.device)
    b[:, :, :, :overlap_size] = (
        (1 - weight_b) * a[:, :, :, -overlap_size:] + weight_b * b[:, :, :, :overlap_size]
    )
    return b

def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """Blend two tensors vertically"""
    weight_b = (torch.arange(overlap_size).view(1, 1, -1, 1) / overlap_size).to(b.device)
    b[:, :, :overlap_size, :] = (
        (1 - weight_b) * a[:, :, -overlap_size:, :] + weight_b * b[:, :, :overlap_size, :]
    )
    return b

def pad_for_tiling(frames: torch.Tensor, tile_num: int, tile_overlap=(128, 128)) -> torch.Tensor:
    """
    Zero-pads a batch of frames (shape [T, C, H, W]) so that (H, W) fits perfectly into 'tile_num' splits plus overlap.
    """
    if tile_num <= 1:
        return frames

    T, C, H, W = frames.shape
    overlap_y, overlap_x = tile_overlap

    # Calculate ideal tile dimensions and strides
    stride_y = max(1, (H + overlap_y * (tile_num - 1)) // tile_num - overlap_y)
    stride_x = max(1, (W + overlap_x * (tile_num - 1)) // tile_num - overlap_x)

    # Recalculate size based on stride
    ideal_H = stride_y * tile_num + overlap_y
    ideal_W = stride_x * tile_num + overlap_x

    pad_bottom = max(0, ideal_H - H)
    pad_right = max(0, ideal_W - W)

    if pad_bottom > 0 or pad_right > 0:
        frames = F.pad(frames, (0, pad_right, 0, pad_bottom), mode="constant", value=0.0)
    return frames

def spatial_tiled_process(
    cond_frames: torch.Tensor,
    mask_frames: torch.Tensor,
    process_func,
    tile_num: int,
    spatial_n_compress: int = 8,
    num_inference_steps: int = 5,
    **kwargs,
) -> torch.Tensor:
    """
    Splits frames into tiles, processes them with process_func, then blends result.
    """
    height = cond_frames.shape[2]
    width = cond_frames.shape[3]

    tile_overlap = (128, 128)
    overlap_y, overlap_x = tile_overlap

    # Calculate tile sizes and strides
    size_y = (height + overlap_y * (tile_num - 1)) // tile_num
    size_x = (width + overlap_x * (tile_num - 1)) // tile_num
    tile_size = (size_y, size_x)
    tile_stride = (max(1, size_y - overlap_y), max(1, size_x - overlap_x))

    cols = []
    for i in range(tile_num):
        row_tiles = []
        for j in range(tile_num):
            y_start = i * tile_stride[0]
            x_start = j * tile_stride[1]
            y_end = min(y_start + tile_size[0], height)
            x_end = min(x_start + tile_size[1], width)

            cond_tile = cond_frames[:, :, y_start:y_end, x_start:x_end]
            mask_tile = mask_frames[:, :, y_start:y_end, x_start:x_end]

            # Pad tile to multiple of 8
            h_tile, w_tile = cond_tile.shape[2], cond_tile.shape[3]
            pad_h = (8 - h_tile % 8) % 8
            pad_w = (8 - w_tile % 8) % 8

            if pad_h > 0 or pad_w > 0:
                cond_tile_proc = F.pad(cond_tile, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
                mask_tile_proc = F.pad(mask_tile, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            else:
                cond_tile_proc = cond_tile
                mask_tile_proc = mask_tile

            with torch.no_grad():
                tile_output_padded = process_func(
                    frames=cond_tile_proc,
                    frames_mask=mask_tile_proc,
                    height=cond_tile_proc.shape[2],
                    width=cond_tile_proc.shape[3],
                    num_frames=len(cond_tile_proc),
                    output_type="latent",
                    num_inference_steps=num_inference_steps,
                    **kwargs,
                ).frames[0]

                # Crop back
                h_latent = h_tile // 8
                w_latent = w_tile // 8
                tile_output = tile_output_padded[:, :, :h_latent, :w_latent]

            row_tiles.append(tile_output)
        cols.append(row_tiles)

    # Blend tiles
    latent_stride = (tile_stride[0] // spatial_n_compress, tile_stride[1] // spatial_n_compress)
    latent_overlap = (overlap_y // spatial_n_compress, overlap_x // spatial_n_compress)

    blended_rows = []
    for i, row_tiles in enumerate(cols):
        row_result = []
        for j, tile in enumerate(row_tiles):
            if i > 0:
                tile = blend_v(cols[i - 1][j], tile, latent_overlap[0])
            if j > 0:
                tile = blend_h(row_result[j - 1], tile, latent_overlap[1])
            row_result.append(tile)
        blended_rows.append(row_result)

    final_rows = []
    for i, row_tiles in enumerate(blended_rows):
        for j, tile in enumerate(row_tiles):
            if i < len(blended_rows) - 1:
                tile = tile[:, :, :latent_stride[0], :]
            if j < len(row_tiles) - 1:
                tile = tile[:, :, :, :latent_stride[1]]
            row_tiles[j] = tile
        final_rows.append(torch.cat(row_tiles, dim=3))

    x = torch.cat(final_rows, dim=2)
    return x

# ==================== END HELPER FUNCTIONS ====================



class InpaintingWebUI:
    """Complete Gradio-based UI for stereo video inpainting with all features"""
    
    def __init__(self):
        # Configuration
        self.app_config = self.load_config()
        
        # Processing control
        self.stop_event = threading.Event()
        self.progress_queue = queue.Queue()
        self.processing_thread = None
        self.pipeline = None
        


    def load_config(self):
        """Load configuration from config_inpaint.json"""
        try:
            with open("config_inpaint.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Default structure based on user feedback
            return {
                'input_folder': './output_splatted/lowres',
                'output_folder': './completed_output',
                'hires_blend_folder': './output_splatted/hires',
                'num_inference_steps': 5,
                'tile_num': 2,
                'frames_chunk': 23,
                'frame_overlap': 3,
                'original_input_blend_strength': 0.0,
                'output_crf': 23,
                'process_length': -1,
                'offload_type': 'model',
                'mask_initial_threshold': 0.3,
                'mask_morph_kernel_size': 0.0,
                'mask_dilate_kernel_size': 5,
                'mask_blur_kernel_size': 10,
                'enable_post_inpainting_blend': False,
                'enable_color_transfer': True
            }
    
    def save_config(self, config_dict):
        """Save configuration to config_inpaint.json"""
        try:
            with open("config_inpaint.json", "w", encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4)
            return "✓ Configuration saved successfully"
        except Exception as e:
            return f"✗ Failed to save config: {e}"
    
    def load_config_to_ui(self):
        """Load configuration and return values for all UI components"""
        try:
            config = self.load_config()
            return (
                config.get('input_folder', './output_splatted/lowres'),
                config.get('output_folder', './completed_output'),
                config.get('hires_blend_folder', './output_splatted/hires'),
                config.get('num_inference_steps', 5),
                config.get('decode_chunk_size', 8),
                config.get('tile_num', 2),
                config.get('frames_chunk', 23),
                config.get('frame_overlap', 3),
                config.get('original_input_blend_strength', 0.0),
                config.get('output_crf', 23),
                config.get('process_length', -1),
                config.get('offload_type', 'model'),
                config.get('mask_initial_threshold', 0.3),
                config.get('mask_morph_kernel_size', 0.0),
                config.get('mask_dilate_kernel_size', 5),
                config.get('mask_blur_kernel_size', 10),
                config.get('enable_post_inpainting_blend', False),
                config.get('enable_color_transfer', True),
                "✓ Configuration loaded successfully"
            )
        except Exception as e:
            # Return current values on error
            return tuple([None] * 18 + [f"✗ Failed to load config: {e}"])
    
    def reset_to_defaults(self):
        """Reset all parameters to default values"""
        return (
            './output_splatted/lowres',  # input_folder
            './completed_output',  # output_folder
            './output_splatted/hires',  # hires_blend_folder
            5,  # num_inference_steps
            8,  # decode_chunk_size (optimized for high-end GPUs)
            2,  # tile_num
            23,  # frames_chunk
            3,  # frame_overlap
            0.0,  # original_input_blend_strength
            23,  # output_crf
            -1,  # process_length
            'model',  # offload_type
            0.3,  # mask_initial_threshold
            0.0,  # mask_morph_kernel_size
            5,  # mask_dilate_kernel_size
            10,  # mask_blur_kernel_size
            False,  # enable_post_inpainting_blend
            True,  # enable_color_transfer
            "✓ Reset to default values"
        )

    def create_interface(self):
        """Create the complete Gradio interface"""
        
        with gr.Blocks(title="Inpainting - StereoCrafter") as interface:
            gr.Markdown("## 🎨 Stereocrafter Inpainting (Batch)")
            
            # Folders Section (Top)
            with gr.Group():
                gr.Markdown("### Folders")
                input_folder = gr.Textbox(
                    label="Input Folder",
                    value=self.app_config.get("input_folder", "./output_splatted/lowres"),
                    info="Select the directory containing your input MP4 videos. The script expects '_splatted4' for quad inputs (Original, Depth, Mask, Warped) or '_splatted2' for dual inputs (Mask, Warped)."
                )
                hires_blend_folder = gr.Textbox(
                    label="Hi-Res Blend Folder",
                    value=self.app_config.get("hires_blend_folder", "./output_splatted/hires"),
                    info="Path to hi-res splatted videos for final blending (optional)."
                )
                output_folder = gr.Textbox(
                    label="Output Folder",
                    value=self.app_config.get("output_folder", "./completed_output"),
                    info="Choose the directory where the processed (inpainted) videos will be saved. Output will be Side-by-Side (original | inpainted) for quad inputs, or only the inpainted right eye for dual inputs."
                )
            
            # Parameters Section
            with gr.Group():
                gr.Markdown("### Parameters")
                with gr.Row():
                    with gr.Column():
                        num_inference_steps = gr.Slider(
                            minimum=1, maximum=50, value=float(self.app_config.get("num_inference_steps", 6)),
                            step=1, label="Inference Steps",
                            info="Number of denoising steps. Higher = better quality but slower. Default: 6"
                        )
                        
                        with gr.Accordion("⚡ Performance Settings (Advanced)", open=False):
                            gr.Markdown("""
                            **Decode Chunk Size**: Higher values = faster but more VRAM
                            - RTX 4090/6000 Ada (24-48GB): Use 8-14 (safe) or 14-20 (aggressive)
                            - RTX 3090/4080 (12-24GB): Use 4-8  
                            - RTX 3060/4060 (8-12GB): Use 2-4
                            
                            ⚠️ **Warning**: Values above 14 may cause OOM (Out of Memory) errors on GPUs with less than 24GB VRAM!
                            """)
                            decode_chunk_size = gr.Slider(
                                minimum=1, maximum=23, value=int(self.app_config.get("decode_chunk_size", 8)),
                                step=1, label="Decode Chunk Size",
                                info="Frames decoded at once. Higher = faster + more VRAM. Safe: 8-12, Max: 23. Default: 8"
                            )
                        tile_num = gr.Slider(
                            minimum=1, maximum=10, value=float(self.app_config.get("tile_num", 1)),
                            step=1, label="Tile Number",
                            info="Number of spatial tiles (e.g., 2 means 2x2 grid) to split each video frame into. Useful for processing high-resolution videos with limited GPU memory. Set to 1 to disable tiling. Default: 1"
                        )
                        frames_chunk = gr.Slider(
                            minimum=1, maximum=50, value=float(self.app_config.get("frames_chunk", 23)),
                            step=1, label="Frames Chunk",
                            info="The number of frames processed together in a single temporal batch. Adjust based on your GPU memory. Larger chunks can be faster but require more VRAM. Default: 23"
                        )
                        process_length = gr.Number(
                            label="Process Length (-1 for all)",
                            value=int(self.app_config.get("process_length", -1)),
                            precision=0,
                            info="Number of frames to process for each video. Enter -1 to process all frames, or a positive integer to limit the processing to the first N frames. Useful for quick testing."
                        )
                    
                    with gr.Column():
                        original_input_blend_strength = gr.Slider(
                            minimum=0.0, maximum=1.0, value=float(self.app_config.get("original_input_blend_strength", 0.0)),
                            step=0.01, label="Original Input Bias",
                            info="Controls how much the original warped input (1.0) versus the previous generated inpainted frame (0.0) influences the blend during temporal overlap. Higher for less hallucination but less consistency. Default: 0 = Off"
                        )
                        frame_overlap = gr.Slider(
                            minimum=0, maximum=20, value=float(self.app_config.get("frame_overlap", 3)),
                            step=1, label="Frame Overlap",
                            info="The number of frames that temporally overlap between consecutive processing chunks. These overlapping frames from the previous generated output and original input are smoothly blended to condition the start of the current chunk, reducing visual glitches. Default: 3"
                        )
                        output_crf = gr.Slider(
                            minimum=0, maximum=51, value=float(self.app_config.get("output_crf", 18)),
                            step=1, label="Output CRF",
                            info="Constant Rate Factor for video encoding (lower is higher quality). Adjust based on codec (H.264/H.265)."
                        )
                        offload_type = gr.Dropdown(
                            choices=["model", "sequential", "none"],
                            value=self.app_config.get("offload_type", "model"),
                            label="CPU Offload Type",
                            info="Determines how parts of the model are moved to CPU memory to reduce GPU VRAM usage. 'model' offloads entire components, 'sequential' offloads layers one by one, 'none' keeps everything on GPU."
                        )
            
            # Mask Processing Section
            with gr.Group():
                gr.Markdown("### Mask Processing")
                with gr.Row():
                    mask_initial_threshold = gr.Slider(
                        minimum=0.0, maximum=1.0, value=float(self.app_config.get("mask_initial_threshold", 0.3)),
                        step=0.01, label="Mask Binarize Threshold",
                        info="Sets a mid value to turn a grayscale image into black and white, separating light and dark areas (0 to 1). Set to 0 to disable."
                    )
                    mask_morph_kernel_size = gr.Slider(
                        minimum=0, maximum=50, value=float(self.app_config.get("mask_morph_kernel_size", 0.0)),
                        step=0.5, label="Morph Close Kernel",
                        info="Kernel size for morphological closing: Defines the size of the shape used to fill small holes and smooth edges in an image during the closing process (e.g., 3, 5). Set to 0 to disable."
                    )
                
                with gr.Row():
                    mask_dilate_kernel_size = gr.Slider(
                        minimum=0, maximum=50, value=float(self.app_config.get("mask_dilate_kernel_size", 5.0)),
                        step=0.5, label="Mask Dilate Kernel",
                        info="Sets the size of the shape used to expand white areas in a mask, making objects larger and filling small gaps during dilation (e.g., 7, 15). Set to 0 to disable."
                    )
                    mask_blur_kernel_size = gr.Slider(
                        minimum=0, maximum=50, value=float(self.app_config.get("mask_blur_kernel_size", 10.0)),
                        step=0.5, label="Mask Blur Kernel",
                        info="Kernel size for Gaussian blur (e.g., 15, 25). Sigma is derived automatically. Set to 0 to disable."
                    )
            
            # Post-Processing Section
            with gr.Group():
                gr.Markdown("### Post-Processing")
                with gr.Row():
                    enable_post_inpainting_blend = gr.Checkbox(
                        label="Enable Post-Inpainting Blend",
                        value=self.app_config.get("enable_post_inpainting_blend", False),
                        info="Toggle the final post-inpainting blending step and enable/disable its parameters."
                    )
                    enable_color_transfer = gr.Checkbox(
                        label="Enable Color Transfer",
                        value=self.app_config.get("enable_color_transfer", True),
                        info="If enabled, the color palette and statistics of the inpainted output will be adjusted to match the original left eye view, improving visual consistency. Now also works in dual-splatted inputs."
                    )
            
            # Progress Section
            with gr.Group():
                gr.Markdown("### Progress")
                status_label = gr.Textbox(label="Status", value="Ready", interactive=False)
                progress_bar = gr.Slider(
                    minimum=0, maximum=100, value=0,
                    label="Progress (%)", interactive=False
                )
                batch_progress = gr.Textbox(label="Batch Progress", value="0/0", interactive=False)
            
            # Control Buttons
            with gr.Row():
                start_button = gr.Button("Start", variant="primary")
                stop_button = gr.Button("Stop", variant="secondary", interactive=False)
            
            with gr.Row():
                save_config_button = gr.Button("Save Config", variant="secondary")
                load_config_button = gr.Button("Load Config", variant="secondary")
                reset_config_button = gr.Button("Reset to Defaults", variant="secondary")
            
            # Current Video Information (Bottom)
            with gr.Group():
                gr.Markdown("### Current Video Information")
                with gr.Row():
                    video_name = gr.Textbox(label="Name", value="N/A", interactive=False)
                    video_res = gr.Textbox(label="Resolution", value="N/A", interactive=False)
                with gr.Row():
                    video_frames = gr.Textbox(label="Frames", value="N/A", interactive=False)
                    video_overlap = gr.Textbox(label="Overlap", value="N/A", interactive=False)
                with gr.Row():
                    video_bias = gr.Textbox(label="Input Bias", value="N/A", interactive=False)



            # ==================== EVENT HANDLERS ====================

            # Toggle mask sliders based on Post-Inpainting Blend checkbox (matches GUI)
            def toggle_mask_sliders(enabled):
                return [
                    gr.update(interactive=enabled),  # mask_initial_threshold
                    gr.update(interactive=enabled),  # mask_morph_kernel_size
                    gr.update(interactive=enabled),  # mask_dilate_kernel_size
                    gr.update(interactive=enabled)   # mask_blur_kernel_size
                ]
            
            enable_post_inpainting_blend.change(
                fn=toggle_mask_sliders,
                inputs=[enable_post_inpainting_blend],
                outputs=[mask_initial_threshold, mask_morph_kernel_size, mask_dilate_kernel_size, mask_blur_kernel_size]
            )
            
            # Collect all parameters
            all_params = [
                input_folder, output_folder, hires_blend_folder,
                num_inference_steps, decode_chunk_size, tile_num, frames_chunk, frame_overlap,
                original_input_blend_strength, output_crf, process_length, offload_type,
                mask_initial_threshold, mask_morph_kernel_size,
                mask_dilate_kernel_size, mask_blur_kernel_size,
                enable_post_inpainting_blend, enable_color_transfer
            ]
            
            # All output components
            all_outputs = [status_label, progress_bar, batch_progress, video_name, video_res, 
                          video_frames, video_overlap, video_bias, start_button, stop_button]

            # Start processing
            start_button.click(
                fn=self.start_processing,
                inputs=all_params,
                outputs=all_outputs
            )

            # Stop processing
            stop_button.click(
                fn=self.stop_processing,
                inputs=[],
                outputs=[status_label, start_button, stop_button]
            )

            # Save config
            save_config_button.click(
                fn=lambda *args: self.save_config({
                    "input_folder": args[0], "output_folder": args[1], "hires_blend_folder": args[2],
                    "num_inference_steps": int(args[3]), "decode_chunk_size": int(args[4]),
                    "tile_num": int(args[5]), "frames_chunk": int(args[6]),
                    "frame_overlap": int(args[7]), "original_input_blend_strength": float(args[8]),
                    "output_crf": int(args[9]), "process_length": int(args[10]), "offload_type": args[11],
                    "mask_initial_threshold": float(args[12]), "mask_morph_kernel_size": float(args[13]),
                    "mask_dilate_kernel_size": float(args[14]), "mask_blur_kernel_size": float(args[15]),
                    "enable_post_inpainting_blend": args[16], "enable_color_transfer": args[17]
                }),
                inputs=all_params,
                outputs=[status_label]
            )
            
            # Load config
            load_config_button.click(
                fn=self.load_config_to_ui,
                outputs=[input_folder, output_folder, hires_blend_folder,
                        num_inference_steps, decode_chunk_size, tile_num, frames_chunk, frame_overlap,
                        original_input_blend_strength, output_crf, process_length, offload_type,
                        mask_initial_threshold, mask_morph_kernel_size,
                        mask_dilate_kernel_size, mask_blur_kernel_size,
                        enable_post_inpainting_blend, enable_color_transfer, status_label]
            )
            
            # Reset to defaults
            reset_config_button.click(
                fn=self.reset_to_defaults,
                outputs=[input_folder, output_folder, hires_blend_folder,
                        num_inference_steps, decode_chunk_size, tile_num, frames_chunk, frame_overlap,
                        original_input_blend_strength, output_crf, process_length, offload_type,
                        mask_initial_threshold, mask_morph_kernel_size,
                        mask_dilate_kernel_size, mask_blur_kernel_size,
                        enable_post_inpainting_blend, enable_color_transfer, status_label]
            )

        return interface

    def scan_for_videos(self, input_folder):

        """Scan for splatted videos"""
        logger.info(f"Scanning for videos in: {input_folder}")
        if not os.path.exists(input_folder):
            logger.warning(f"Input folder does not exist: {input_folder}")
            return []

        logger.info("Starting glob search for *.mp4 files...")
        videos = sorted(glob.glob(os.path.join(input_folder, "*.mp4")))
        logger.info(f"Found {len(videos)} MP4 files")
        
        # Filter for splatted videos
        splatted_videos = [v for v in videos if ('_splatted2' in os.path.basename(v) or 
                                                  '_splatted4' in os.path.basename(v))]
        logger.info(f"Found {len(splatted_videos)} splatted videos")
        return splatted_videos

    # ==================== PROCESSING METHODS ====================

    def start_processing(self, *args, progress=gr.Progress()):
        """Start batch processing"""
        logger.info("=== Start button clicked ===")
        
        # Extract parameters
        (input_folder, output_folder, hires_blend_folder,
         num_inference_steps, decode_chunk_size, tile_num, frames_chunk, frame_overlap,
         original_input_blend_strength, output_crf, process_length, offload_type,
         mask_initial_threshold, mask_morph_kernel_size,
         mask_dilate_kernel_size, mask_blur_kernel_size,
         enable_post_inpainting_blend, enable_color_transfer) = args

        logger.info(f"Input folder: {input_folder}")
        logger.info(f"Output folder: {output_folder}")

        # Validate
        try:
            num_inference_steps = int(num_inference_steps)
            decode_chunk_size = int(decode_chunk_size)
            tile_num = int(tile_num)
            frames_chunk = int(frames_chunk)
            frame_overlap = int(frame_overlap)
            original_input_blend_strength = float(original_input_blend_strength)
            output_crf = int(output_crf)
            process_length = int(process_length)

            if num_inference_steps < 1 or tile_num < 1 or frames_chunk < 1:
                logger.error("Invalid parameter values")
                return ("❌ Error: Invalid parameter values", 0, "0/0", "N/A", "N/A", "N/A", "N/A", "N/A",
                       gr.update(interactive=True), gr.update(interactive=False))
        except ValueError as e:
            logger.error(f"Invalid numeric values: {e}")
            return ("❌ Error: Please enter valid numeric values", 0, "0/0", "N/A", "N/A", "N/A", "N/A", "N/A",
                   gr.update(interactive=True), gr.update(interactive=False))

        if not os.path.isdir(input_folder):
            logger.error(f"Input folder does not exist: {input_folder}")
            return (f"❌ Error: Input folder '{input_folder}' does not exist", 0, "0/0", "N/A", "N/A", "N/A", "N/A", "N/A",
                   gr.update(interactive=True), gr.update(interactive=False))

        os.makedirs(output_folder, exist_ok=True)
        logger.info("Validation passed, starting processing thread...")

        # Store parameters
        params = {
            'input_folder': input_folder,
            'output_folder': output_folder,
            'hires_blend_folder': hires_blend_folder,
            'num_inference_steps': num_inference_steps,
            'decode_chunk_size': decode_chunk_size,
            'tile_num': tile_num,
            'frames_chunk': frames_chunk,
            'frame_overlap': frame_overlap,
            'original_input_blend_strength': original_input_blend_strength,
            'output_crf': output_crf,
            'process_length': process_length,
            'offload_type': offload_type,
            'mask_initial_threshold': mask_initial_threshold,
            'mask_morph_kernel_size': mask_morph_kernel_size,
            'mask_dilate_kernel_size': mask_dilate_kernel_size,
            'mask_blur_kernel_size': mask_blur_kernel_size,
            'enable_post_inpainting_blend': enable_post_inpainting_blend,
            'enable_color_transfer': enable_color_transfer
        }

        # Start processing thread
        self.stop_event.clear()
        # Clear the queue
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except:
                break
                
        self.processing_thread = threading.Thread(
            target=self.process_batch,
            args=(params,),
            daemon=True
        )
        self.processing_thread.start()

        # Poll for progress updates
        import time
        last_status = "🚀 Processing started..."
        last_progress = 0
        last_batch = "0/0"
        last_video_name = "N/A"
        last_video_res = "N/A"
        last_video_frames = "N/A"
        last_video_overlap = "N/A"
        last_video_bias = "N/A"
        
        max_wait_time = 300  # Maximum 5 minutes wait
        start_time = time.time()
        
        while self.processing_thread.is_alive():
            # Check if we've been waiting too long or stop was requested
            if time.time() - start_time > max_wait_time or self.stop_event.is_set():
                if self.stop_event.is_set():
                    # Give thread a moment to clean up
                    self.processing_thread.join(timeout=2.0)
                    last_status = "⏹️ Processing stopped by user"
                break
            
            # Check queue for updates
            updated = False
            while not self.progress_queue.empty():
                try:
                    msg_type, msg_value = self.progress_queue.get_nowait()
                    updated = True
                    
                    if msg_type == "status":
                        last_status = msg_value
                    elif msg_type == "progress":
                        last_progress = msg_value
                    elif msg_type == "batch_progress":
                        last_batch = msg_value
                    elif msg_type == "video_name":
                        last_video_name = msg_value
                    elif msg_type == "video_res":
                        last_video_res = msg_value
                    elif msg_type == "video_frames":
                        last_video_frames = msg_value
                    elif msg_type == "video_overlap":
                        last_video_overlap = msg_value
                    elif msg_type == "video_bias":
                        last_video_bias = msg_value
                except:
                    break
            
            if updated:
                # Update progress bar
                progress(last_progress / 100.0, desc=last_status)
                
                # Yield current state
                yield (last_status, last_progress, last_batch, last_video_name, last_video_res,
                      last_video_frames, last_video_overlap, last_video_bias,
                      gr.update(interactive=False), gr.update(interactive=True))
            
            time.sleep(0.1)  # Poll every 100ms
        
        # Final update after thread completes
        while not self.progress_queue.empty():
            try:
                msg_type, msg_value = self.progress_queue.get_nowait()
                if msg_type == "status":
                    last_status = msg_value
                elif msg_type == "progress":
                    last_progress = msg_value
                elif msg_type == "batch_progress":
                    last_batch = msg_value
                elif msg_type == "video_name":
                    last_video_name = msg_value
                elif msg_type == "video_res":
                    last_video_res = msg_value
                elif msg_type == "video_frames":
                    last_video_frames = msg_value
                elif msg_type == "video_overlap":
                    last_video_overlap = msg_value
                elif msg_type == "video_bias":
                    last_video_bias = msg_value
            except:
                break
        
        return (last_status, last_progress, last_batch, last_video_name, last_video_res,
               last_video_frames, last_video_overlap, last_video_bias,
               gr.update(interactive=True), gr.update(interactive=False))

    def stop_processing(self):
        """Stop processing"""
        self.stop_event.set()
        if self.pipeline:
            try:
                release_cuda_memory()
            except RuntimeError as e:
                logger.warning(f"Failed to clear CUDA cache: {e}")
        return "⏹️ Stopping processing...", gr.update(interactive=True), gr.update(interactive=False)

    def process_batch(self, params):
        """Main batch processing function"""
        try:
            logger.info("=== Starting batch processing ===")
            logger.info(f"Input folder: {params['input_folder']}")
            logger.info(f"Output folder: {params['output_folder']}")
            
            # Load pipeline
            self.progress_queue.put(("status", "Loading inpainting pipeline..."))
            logger.info("Loading inpainting pipeline...")
            
            # Use local weights loader (loads StereoCrafter UNet without subfolder)
            svd_path = os.path.abspath("./weights/stable-video-diffusion-img2vid-xt-1-1")
            unet_path = os.path.abspath("./weights/StereoCrafter")
            
            logger.info(f"SVD path: {svd_path}")
            logger.info(f"UNet path: {unet_path}")
            
            self.pipeline = load_inpainting_pipeline_local(
                svd_path=svd_path,
                unet_path=unet_path,
                device="cuda",
                dtype=torch.float16,
                offload_type=params['offload_type']
            )
            
            logger.info("Pipeline loaded successfully")

            # Find videos
            logger.info(f"Scanning for videos in: {params['input_folder']}")
            input_videos = self.scan_for_videos(params['input_folder'])
            if not input_videos:
                logger.warning("No splatted videos found")
                self.progress_queue.put(("status", "No splatted videos found"))
                self.progress_queue.put(("batch_progress", "0/0"))
                return

            total_videos = len(input_videos)
            logger.info(f"Found {total_videos} videos to process")
            self.progress_queue.put(("status", f"Processing {total_videos} videos..."))
            self.progress_queue.put(("batch_progress", f"0/{total_videos}"))

            processed_count = 0

            # Process each video
            for idx, video_path in enumerate(input_videos):
                if self.stop_event.is_set():
                    self.progress_queue.put(("status", "Processing stopped by user"))
                    break

                basename = os.path.basename(video_path)
                self.progress_queue.put(("status", f"Processing {idx+1}/{total_videos}: {basename}"))
                self.progress_queue.put(("batch_progress", f"{idx}/{total_videos}"))
                
                # Update video info
                self.progress_queue.put(("video_name", basename))
                self.progress_queue.put(("video_res", "Loading..."))
                self.progress_queue.put(("video_frames", "Loading..."))
                self.progress_queue.put(("video_overlap", str(params['frame_overlap'])))
                self.progress_queue.put(("video_bias", str(params['original_input_blend_strength'])))

                success, hires_path = self.process_single_video(
                    pipeline=self.pipeline,
                    input_video_path=video_path,
                    params=params
                )

                if success:
                    processed_count += 1
                    msg = f"Completed {basename}"
                    if hires_path:
                        msg += " (+HiRes)"
                    self.progress_queue.put(("status", msg))
                else:
                    self.progress_queue.put(("status", f"Failed {basename}"))

                progress = int(((idx + 1) / total_videos) * 100)
                self.progress_queue.put(("progress", progress))
                self.progress_queue.put(("batch_progress", f"{idx+1}/{total_videos}"))

            if self.stop_event.is_set():
                self.progress_queue.put(("status", "❌ Processing stopped."))
            else:
                self.progress_queue.put(("status", f"✅ Batch completed! ({processed_count}/{total_videos} successful)"))
            self.progress_queue.put(("progress", 100))
            self.progress_queue.put(("batch_progress", f"{total_videos}/{total_videos}"))
            
            # Reset video info
            self.progress_queue.put(("video_name", "N/A"))
            self.progress_queue.put(("video_res", "N/A"))
            self.progress_queue.put(("video_frames", "N/A"))
            self.progress_queue.put(("video_overlap", "N/A"))
            self.progress_queue.put(("video_bias", "N/A"))

        except Exception as e:
            logger.exception(f"Error during batch processing: {e}")
            self.progress_queue.put(("status", f"❌ Error: {str(e)}"))
        finally:
            if self.pipeline:
                try:
                    del self.pipeline
                    release_cuda_memory()
                except:
                    pass
                self.pipeline = None

    def process_single_video(self, pipeline, input_video_path, params):
        """Process a single video through the complete pipeline"""
        try:
            save_dir = params['output_folder']
            hires_blend_folder = params.get('hires_blend_folder', '')

            # 1. Setup & Hi-Res Detection
            base_video_name = os.path.basename(input_video_path)
            video_name_without_ext = os.path.splitext(base_video_name)[0]
            is_dual_input = video_name_without_ext.endswith("_splatted2")

            output_video_path, hires_data = self._setup_video_info_and_hires(
                input_video_path, save_dir, is_dual_input, hires_blend_folder
            )

            # Read sidecar if available (update params for this video specific)
            sidecar_params = self._read_sidecar_json(input_video_path)

            # Merge sidecar params with global params
            current_overlap = sidecar_params.get('frame_overlap', params['frame_overlap'])
            current_blend = sidecar_params.get('original_input_blend_strength', params['original_input_blend_strength'])
            current_crf = sidecar_params.get('output_crf', params['output_crf'])

            # 2. Input Preparation
            prepared_inputs = self._prepare_video_inputs(
                input_video_path=input_video_path,
                base_video_name=base_video_name,
                is_dual_input=is_dual_input,
                frames_chunk=params['frames_chunk'],
                tile_num=params['tile_num'],
                overlap=current_overlap,
                original_input_blend_strength=current_blend,
                process_length=params['process_length'],
                mask_params=params
            )

            if prepared_inputs is None:
                return False, None

            (frames_warpped_padded, frames_mask_padded, frames_left_original_cropped,
             num_frames_original, padded_H, padded_W, video_stream_info, fps,
             frames_warpped_original_unpadded, frames_mask_original_unpadded_processed) = prepared_inputs

            # 3. Inpainting Loop
            total_frames = num_frames_original
            frames_chunk = params['frames_chunk']
            overlap = current_overlap
            stride = max(1, frames_chunk - overlap)
            results = []
            previous_chunk_output = None

            for i in range(0, total_frames, stride):
                if self.stop_event.is_set():
                    return False, None

                # Slice logic with dynamic padding
                end_idx = min(i + frames_chunk, total_frames)
                input_slice = frames_warpped_padded[i:end_idx].clone()
                mask_slice = frames_mask_padded[i:end_idx].clone()
                actual_len = input_slice.shape[0]

                padding_needed = 0
                if actual_len <= 4:
                    padding_needed = 6 - actual_len
                    # Pad with last frame repetition
                    last_frame = input_slice[-1:].clone()
                    last_mask = mask_slice[-1:].clone()
                    input_slice = torch.cat([input_slice] + [last_frame]*padding_needed, dim=0)
                    mask_slice = torch.cat([mask_slice] + [last_mask]*padding_needed, dim=0)

                # Input blending
                if previous_chunk_output is not None and overlap > 0:
                    overlap_actual = min(overlap, input_slice.shape[0])
                    prev_overlap = previous_chunk_output[-overlap_actual:]

                    if current_blend > 0:
                        weights = torch.linspace(0.0, 1.0, overlap_actual, device=prev_overlap.device).view(-1, 1, 1, 1) * current_blend
                        input_slice[:overlap_actual] = (1 - weights) * prev_overlap + weights * input_slice[:overlap_actual]
                    else:
                        input_slice[:overlap_actual] = prev_overlap

                # 4. Inference - OPTIMIZED FOR HIGH-END GPU
                with torch.no_grad():
                    video_latents = spatial_tiled_process(
                        cond_frames=input_slice,
                        mask_frames=mask_slice,
                        process_func=pipeline,
                        tile_num=params['tile_num'],
                        spatial_n_compress=8,
                        min_guidance_scale=1.01,
                        max_guidance_scale=1.01,
                        decode_chunk_size=params['decode_chunk_size'],  # Use user setting
                        fps=7,
                        motion_bucket_id=127,
                        noise_aug_strength=0.0,
                        num_inference_steps=params['num_inference_steps']
                    )
                    
                    video_latents = video_latents.unsqueeze(0)
                    pipeline.vae.to(dtype=torch.float16)
                    # Decode - Use user setting for chunk size
                    decoded_frames = pipeline.decode_latents(video_latents, num_frames=video_latents.shape[1], decode_chunk_size=params['decode_chunk_size'])

                # Convert to tensor [T, C, H, W]
                video_frames = tensor2vid(decoded_frames, pipeline.image_processor, output_type="pil")[0]
                chunk_generated = torch.stack([
                    torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in video_frames
                ]).cpu()

                # Handle output collection
                if i == 0:
                    results.append(chunk_generated[:actual_len])
                else:
                    results.append(chunk_generated[overlap:actual_len])

                previous_chunk_output = chunk_generated

                # Update progress
                current_percent = int((i / total_frames) * 100)
                self.progress_queue.put(("progress", current_percent))

            # Concatenate results
            if not results:
                return False, None

            frames_output = torch.cat(results, dim=0).cpu()

            # Crop to original size
            frames_output_final = frames_output[:, :, :padded_H, :padded_W][:num_frames_original]

            # 5. Finalization (Color Transfer, Blending, Hi-Res)
            final_output = self._finalize_output_frames(
                inpainted_frames=frames_output_final,
                mask_frames=frames_mask_original_unpadded_processed,
                original_warped_frames=frames_warpped_original_unpadded,
                original_left_frames=frames_left_original_cropped,
                hires_data=hires_data,
                base_video_name=base_video_name,
                is_dual_input=is_dual_input,
                params=params
            )

            if final_output is None:
                return False, None

            # 6. Encoding
            self.progress_queue.put(("status", f"Encoding {base_video_name}..."))
            temp_png_dir = os.path.join(save_dir, f"temp_{base_video_name}_{int(time.time())}")
            os.makedirs(temp_png_dir, exist_ok=True)

            try:
                # Save PNGs
                for idx, frame in enumerate(final_output):
                    if self.stop_event.is_set(): return False, None
                    frame_np = (frame.permute(1, 2, 0).numpy() * 65535.0).astype(np.uint16)
                    cv2.imwrite(
                        os.path.join(temp_png_dir, f"{idx:05d}.png"),
                        cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    )

                # Encode MP4
                # GUI ALIGNMENT: FORCE CPU ENCODING
                # The Windows GUI appears to use libx264 (CPU) resulting in ~2700kbps bitrate.
                # WebUI defaults to NVENC if available (~5500kbps).
                # To match 1:1, we temporarily disable CUDA for the util module during encoding.
                original_cuda_flag = sc_util.CUDA_AVAILABLE
                sc_util.CUDA_AVAILABLE = False
                try:
                    encode_frames_to_mp4(
                        temp_png_dir=temp_png_dir,
                        final_output_mp4_path=output_video_path,
                        fps=fps,
                        total_output_frames=len(final_output),
                        video_stream_info=video_stream_info,
                        user_output_crf=current_crf,
                        output_sidecar_ext=".spsidecar"
                    )
                finally:
                    sc_util.CUDA_AVAILABLE = original_cuda_flag
            finally:
                shutil.rmtree(temp_png_dir, ignore_errors=True)

            return True, hires_data.get('hires_video_path')

        except Exception as e:
            logger.exception(f"Error processing single video {input_video_path}: {e}")
            return False, None

    # ==================== HELPER METHODS ====================

    def _prepare_video_inputs(self, input_video_path, base_video_name, is_dual_input, 
                             frames_chunk, tile_num, overlap, original_input_blend_strength, 
                             process_length, mask_params):
        """Prepare video frames, masks and padding for processing"""
        try:
            # Load video
            vr = VideoReader(input_video_path, ctx=cpu(0))
            total_frames = len(vr)
            fps = vr.get_avg_fps()

            if process_length > 0:
                total_frames = min(process_length, total_frames)

            # Read all frames (returns numpy array normalized to 0-1 from read_video_frames_decord)
            frames, fps, orig_h, orig_w, proc_h, proc_w, video_stream_info = read_video_frames_decord(input_video_path, total_frames)
            # frames is [T, H, W, C] float32 0-1
            
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2) # Convert to tensor [T,C,H,W]

            # --- GUI ALIGNMENT: Quantize to uint8 (0-255) immediately ---
            # The GUI performs this step: frames = (frames * 255).to(torch.uint8)
            # We must replicate this to ensure identical "artifacts" and processing behavior.
            frames = (frames * 255.0).clamp(0, 255).to(torch.uint8)
            
            # --- Dimension Divisibility Check and Resizing (if needed) ---
            _, _, total_h_raw_input_before_resize, total_w_raw_input_before_resize = frames.shape
            required_divisor = 8

            new_h = total_h_raw_input_before_resize
            new_w = total_w_raw_input_before_resize

            if new_h % required_divisor != 0:
                new_h = (new_h // required_divisor + 1) * required_divisor
                logger.warning(f"Video height {total_h_raw_input_before_resize} is not divisible by {required_divisor}. Resizing to {new_h}.")

            if new_w % required_divisor != 0:
                new_w = (new_w // required_divisor + 1) * required_divisor
                logger.warning(f"Video width {total_w_raw_input_before_resize} is not divisible by {required_divisor}. Resizing to {new_w}.")

            if new_h != total_h_raw_input_before_resize or new_w != total_w_raw_input_before_resize:
                if frames.shape[0] > 0:
                    # F.interpolate handles uint8 inputs (often by casting to float internally or returning float)
                    # To be essentially 1:1 with GUI, we call it on the current 'frames' tensor.
                    # Note: If frames is uint8, F.interpolate might return float depending on PyTorch version/backend.
                    frames = F.interpolate(frames.float(), size=(new_h, new_w), mode='bicubic', align_corners=False)
                    
                    # If interpolated to float, we should ideally cast back to uint8 if we strictly follow "keep it uint8" philosophy,
                    # BUT the GUI code (line 915) does: frames = F.interpolate(...) and does NOT explicitly cast back.
                    # However, subsequent lines in GUI treat it as if it might be uint8 or float-0-255.
                    # For safety and strict alignment with "quantized" look, we should enforce uint8.
                    frames = frames.clamp(0, 255).to(torch.uint8)
                    
                    logger.info(f"Frames resized from {total_h_raw_input_before_resize}x{total_w_raw_input_before_resize} to {new_h}x{new_w}.")
                else:
                    logger.warning("Attempted to resize empty frames tensor. Skipping resize.")
            
            # --- Update current dimensions after potential resize ---
            T, C, H, W = frames.shape

            # Split logic
            if is_dual_input:
                # Left=Mask, Right=Warped
                # Mask needs processing (it's grayscale usually)
                half_w = W // 2
                mask_frames = frames[:, :, :, :half_w]
                warped_frames = frames[:, :, :, half_w:]
                left_frames = warped_frames # No true left view in dual
            else:
                # Quad: TL=Source, TR=Depth, BL=Mask, BR=Warped
                half_h = H // 2
                half_w = W // 2
                left_frames = frames[:, :, :half_h, :half_w]
                mask_frames = frames[:, :, half_h:, :half_w]
                warped_frames = frames[:, :, half_h:, half_w:]

                H, W = half_h, half_w

            # --- GUI ALIGNMENT: Normalize left_frames immediately ---
            # The GUI normalizes this crop here. We must do the same to ensure valid float 0-1 for SBS output.
            if left_frames is not None:
                left_frames = left_frames.float() / 255.0
            
            # Mask Processing (Binarize, Dilate, Blur)
            # Convert to grayscale using OpenCV (matches GUI method exactly)
            processed_masks_grayscale = []
            for t in range(mask_frames.shape[0]):
                # frames is uint8 (0-255), so mask_frames is also uint8 [C, H, W]
                # Permute to [H, W, C] for OpenCV
                frame_nhwc = mask_frames[t].permute(1, 2, 0).cpu().numpy()
                
                # Ensure it is uint8 (should be already, but safety first if resize made it float coverage)
                if frame_nhwc.dtype != np.uint8:
                    frame_nhwc = frame_nhwc.astype(np.uint8)
                
                # Use OpenCV's proper RGB to grayscale conversion (luminance weights)
                frame_gray = cv2.cvtColor(frame_nhwc, cv2.COLOR_RGB2GRAY)
                
                # Convert back to float [0-1] tensor and add channel dimension
                frame_tensor = torch.from_numpy(frame_gray).float() / 255.0
                frame_tensor = frame_tensor.unsqueeze(0)  # Add channel dim [1, H, W]
                processed_masks_grayscale.append(frame_tensor)
            
            mask_frames = torch.stack(processed_masks_grayscale).to(frames.device)
            
            # Only apply mask processing if enable_post_inpainting_blend is True
            if mask_params.get('enable_post_inpainting_blend', False):
                # Binarize
                mask_frames = (mask_frames > mask_params['mask_initial_threshold']).float()

                # Apply Dilate
                k_d = int(mask_params['mask_dilate_kernel_size'])
                if k_d > 0:
                    k_d = k_d if k_d % 2 == 1 else k_d + 1
                    mask_frames = F.max_pool2d(mask_frames, kernel_size=k_d, stride=1, padding=k_d//2)

                # Apply Blur
                k_b = int(mask_params['mask_blur_kernel_size'])
                if k_b > 0:
                    mask_frames = self._apply_gaussian_blur(mask_frames, k_b)

            frames_mask_processed = mask_frames

            # Calculate Padding
            pad_h = (16 - H % 16) % 16
            pad_w = (16 - W % 16) % 16

            if pad_h > 0 or pad_w > 0:
                # Convert to float for padding and model input
                warped_padded = F.pad(warped_frames.float() / 255.0, (0, pad_w, 0, pad_h), mode='constant', value=0)
                mask_padded = F.pad(frames_mask_processed, (0, pad_w, 0, pad_h), mode='constant', value=0)
            else:
                # Convert to float for model input
                warped_padded = warped_frames.float() / 255.0
                mask_padded = frames_mask_processed

            # Info extraction
            video_stream_info = get_video_stream_info(input_video_path)
            
            # Send video info update
            self.progress_queue.put(("video_res", f"{W}x{H}"))
            self.progress_queue.put(("video_frames", str(T)))

            return (warped_padded, mask_padded, left_frames,
                    T, H, W, video_stream_info, fps,
                    warped_frames, frames_mask_processed)

        except Exception as e:
            logger.error(f"Error preparing video input: {e}")
            return None

    def _finalize_output_frames(self, inpainted_frames, mask_frames, original_warped_frames, 
                               original_left_frames, hires_data, base_video_name, is_dual_input, params):
        """Finalize frames: blending, color transfer, hi-res application"""
        try:
            frames_output_final = inpainted_frames
            frames_mask_processed = mask_frames
            frames_warpped_original_unpadded = original_warped_frames # This is uint8 from _prepare_video_inputs
            frames_left_original_cropped = original_left_frames # This is uint8 from _prepare_video_inputs

            # --- Hi-Res Blending Logic ---
            if hires_data.get("is_hires_blend_enabled"):
                hires_H, hires_W = hires_data["hires_H"], hires_data["hires_W"]
                num_frames_original = frames_output_final.shape[0]
                hires_video_path = hires_data["hires_video_path"]

                logger.info(f"Starting Hi-Res Blending at {hires_W}x{hires_H}...")

                hires_reader = VideoReader(hires_video_path, ctx=cpu(0))
                chunk_size = params['frames_chunk']

                final_hires_output_chunks = []
                final_hires_left_chunks = []

                for i in range(0, num_frames_original, chunk_size):
                    start_idx, end_idx = i, min(i + chunk_size, num_frames_original)
                    frame_indices = list(range(start_idx, end_idx))
                    if not frame_indices: break

                    # Get chunks
                    inpainted_chunk = frames_output_final[start_idx:end_idx]
                    mask_chunk = frames_mask_processed[start_idx:end_idx]

                    # Upscale
                    inpainted_chunk_hires = F.interpolate(inpainted_chunk, size=(hires_H, hires_W), mode='bicubic', align_corners=False)
                    mask_chunk_hires = F.interpolate(mask_chunk, size=(hires_H, hires_W), mode='bilinear', align_corners=False)

                    # Load Hi-Res
                    hires_frames_np = hires_reader.get_batch(frame_indices).asnumpy()
                    hires_frames_torch = torch.from_numpy(hires_frames_np).permute(0, 3, 1, 2).float()

                    # Split Hi-Res
                    if is_dual_input:
                        half_w_hires = hires_frames_torch.shape[3] // 2
                        hires_warped_chunk = hires_frames_torch[:, :, :, half_w_hires:].float() / 255.0
                        hires_left_chunk = None
                    else:
                        half_h_hires, half_w_hires = hires_frames_torch.shape[2] // 2, hires_frames_torch.shape[3] // 2
                        hires_left_chunk = hires_frames_torch[:, :, :half_h_hires, :half_w_hires].float() / 255.0
                        hires_warped_chunk = hires_frames_torch[:, :, half_h_hires:, half_w_hires:].float() / 255.0
                        final_hires_left_chunks.append(hires_left_chunk)

                    final_hires_output_chunks.append({
                        "inpainted": inpainted_chunk_hires,
                        "mask": mask_chunk_hires,
                        "warped": hires_warped_chunk
                    })

                # Concatenate back
                frames_output_final = torch.cat([d["inpainted"] for d in final_hires_output_chunks], dim=0)
                frames_mask_processed = torch.cat([d["mask"] for d in final_hires_output_chunks], dim=0)
                frames_warpped_original_unpadded = torch.cat([d["warped"] for d in final_hires_output_chunks], dim=0) # This is now float 0-1
                if not is_dual_input:
                    frames_left_original_cropped = torch.cat(final_hires_left_chunks, dim=0) # This is now float 0-1

                del hires_reader, final_hires_output_chunks
                release_cuda_memory()

            # --- Color Transfer ---
            if params['enable_color_transfer']:
                reference_frames = None
                # frames_warpped_original_unpadded is uint8 if no hires, float 0-1 if hires
                # frames_left_original_cropped is uint8 if no hires, float 0-1 if hires
                # Need to ensure they are float 0-1 for color transfer
                warped_for_ct = frames_warpped_original_unpadded.float() / 255.0 if frames_warpped_original_unpadded.dtype == torch.uint8 else frames_warpped_original_unpadded
                left_for_ct = frames_left_original_cropped.float() / 255.0 if frames_left_original_cropped is not None and frames_left_original_cropped.dtype == torch.uint8 else frames_left_original_cropped

                if is_dual_input:
                    # Create occlusion-free reference from warped frames
                    warped_frames_base = warped_for_ct.cpu()
                    processed_mask = frames_mask_processed.cpu()
                    reference_frames = self._apply_directional_dilation(
                        frame_chunk=warped_frames_base, mask_chunk=processed_mask
                    ).to(frames_output_final.device)
                else:
                    reference_frames = left_for_ct

                if reference_frames is not None:
                    # Apply to each frame (expensive loop, but accurate)
                    target_H, target_W = frames_output_final.shape[2], frames_output_final.shape[3]
                    adjusted_frames_output = []
                    for t in range(frames_output_final.shape[0]):
                        ref_frame_resized = F.interpolate(
                            reference_frames[t].unsqueeze(0),
                            size=(target_H, target_W),
                            mode='bilinear', align_corners=False
                        ).squeeze(0).cpu()
                        target_frame_cpu = frames_output_final[t].cpu()
                        adjusted_frame = self._apply_color_transfer(ref_frame_resized, target_frame_cpu)
                        adjusted_frames_output.append(adjusted_frame.to(frames_output_final.device))

                    frames_output_final = torch.stack(adjusted_frames_output)

            # --- Normalization using same logic as GUI ---
            # frames_warpped_original_unpadded (from _prepare_video_inputs) is uint8 0-255 if no hires.
            # If hires was enabled, it's already float 0-1.
            # Ensure it's float 0-1 for blending.
            frames_warpped_original_unpadded_normalized = frames_warpped_original_unpadded.float() / 255.0 if frames_warpped_original_unpadded.dtype == torch.uint8 else frames_warpped_original_unpadded
            
            # frames_left_original_cropped also needs to be normalized if it was taken from frames
            if frames_left_original_cropped is not None:
                frames_left_original_cropped_normalized = frames_left_original_cropped.float() / 255.0 if frames_left_original_cropped.dtype == torch.uint8 else frames_left_original_cropped
            else:
                frames_left_original_cropped_normalized = None

            # --- Note: frames_mask_padded is derived from processed_masks_grayscale which is already float 0-1 ---
            
            # --- Post-Inpainting Blend ---
            if params['enable_post_inpainting_blend']:
                # Blend: original * (1 - mask) + inpainted * mask
                mask_cpu = frames_mask_processed.cpu()
                if mask_cpu.shape[1] != 1: mask_cpu = mask_cpu.mean(dim=1, keepdim=True)

                inpainted_cpu = frames_output_final.cpu()
                original_cpu = frames_warpped_original_unpadded_normalized.cpu()

                # Check shapes match
                if inpainted_cpu.shape == original_cpu.shape:
                    blended = original_cpu * (1 - mask_cpu) + inpainted_cpu * mask_cpu
                    frames_output_final = blended.to(frames_output_final.device)

            # --- Final Concatenation ---
            if is_dual_input:
                return frames_output_final
            else:
                if frames_left_original_cropped_normalized is not None:
                    # Resize left to match output if needed
                    # Note: We use the NORMALIZED version here
                    if frames_left_original_cropped_normalized.shape[-2:] != frames_output_final.shape[-2:]:
                        frames_left_original_cropped_normalized = F.interpolate(
                            frames_left_original_cropped_normalized, 
                            size=frames_output_final.shape[-2:], 
                            mode='bilinear',
                            align_corners=False # Explicitly set align_corners for safety
                        )
                    return torch.cat([frames_left_original_cropped_normalized, frames_output_final], dim=3)

            return None

        except Exception as e:
            logger.error(f"Error finalization: {e}")
            logger.exception("Full traceback:")
            return None

    def _setup_video_info_and_hires(self, input_video_path, save_dir, is_dual_input, hires_blend_folder):
        """Determine output path and find hi-res match"""
        base = os.path.basename(input_video_path)
        name_no_ext = os.path.splitext(base)[0]

        # Determine output filename
        suffix = "_inpainted"
        out_name = name_no_ext.replace("_splatted2", "").replace("_splatted4", "")
        out_path = os.path.join(save_dir, f"{out_name}{suffix}.mp4")

        # Determine input folder from path for the check
        input_folder = os.path.dirname(input_video_path)

        hires_info = {"is_hires_blend_enabled": False}
        hires_path = self._find_high_res_match(input_video_path, input_folder, hires_blend_folder)

        if hires_path:
            try:
                # Read Hi-Res dimensions
                vr_hi = VideoReader(hires_path, ctx=cpu(0))
                # H, W, since decord format is usually HWC in get_batch, but check shape
                # get_batch returns [Batch, H, W, C]
                shape = vr_hi.get_batch([0]).shape
                hires_H, hires_W = shape[1], shape[2]

                # Handling Quad/Dual for actual frame size vs canvas size
                if is_dual_input:
                    # In Dual, width is 2x view width
                    hires_W_view = hires_W // 2
                    hires_H_view = hires_H
                    # We render view size usually.
                    hires_W = hires_W_view 
                else:
                     # Quad
                    hires_W = hires_W // 2
                    hires_H = hires_H // 2

                hires_info = {
                    "is_hires_blend_enabled": True,
                    "hires_video_path": hires_path,
                    "hires_H": hires_H,
                    "hires_W": hires_W
                }
            except Exception as e:
                logger.error(f"Failed to load Hi-Res info: {e}")
                hires_info = {"is_hires_blend_enabled": False}

        return out_path, hires_info

    def _find_high_res_match(self, low_res_video_path, input_folder, hires_blend_folder) -> Optional[str]:
        """Find matching hi-res video"""
        if not hires_blend_folder or not os.path.exists(hires_blend_folder):
            return None

        if os.path.normpath(input_folder) == os.path.normpath(hires_blend_folder):
            return None

        low_res_filename = os.path.basename(low_res_video_path)
        name_no_ext = os.path.splitext(low_res_filename)[0]

        splatted_suffix = None
        if name_no_ext.endswith('_splatted2'):
            splatted_suffix = '_splatted2.mp4'
            splatted_core = '_splatted2'
        elif name_no_ext.endswith('_splatted4'):
            splatted_suffix = '_splatted4.mp4'
            splatted_core = '_splatted4'
        else:
            return None

        # Strip resolution and suffix
        splat_index = name_no_ext.rfind(splatted_core)
        if splat_index == -1: return None

        name_core = name_no_ext[:splat_index]
        last_underscore = name_core.rfind('_')
        if last_underscore != -1:
            base_pattern = name_core[:last_underscore]
        else:
            base_pattern = name_core

        if not base_pattern: return None

        search_pattern = os.path.join(hires_blend_folder, f"{base_pattern}_*{splatted_suffix}")
        matches = glob.glob(search_pattern)

        matches = [m for m in matches if os.path.normpath(m) != os.path.normpath(low_res_video_path)]

        if not matches: return None

        # Check resolution
        hires_path = matches[0]
        try:
            vr_lo = VideoReader(low_res_video_path, ctx=cpu(0))
            lo_w = vr_lo.get_batch([0]).shape[2]

            vr_hi = VideoReader(hires_path, ctx=cpu(0))
            hi_w = vr_hi.get_batch([0]).shape[2]

            if hi_w <= lo_w:
                return None
            return hires_path
        except:
            return None

    def _read_sidecar_json(self, video_path):
        """Read .spsidecar file if exists"""
        json_path = os.path.splitext(video_path)[0] + ".spsidecar"
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _apply_color_transfer(self, source_frame: torch.Tensor, target_frame: torch.Tensor) -> torch.Tensor:
        """Apply color transfer using LAB"""
        try:
            # Ensure tensors are on CPU and convert to numpy arrays in HWC format
            source_np = source_frame.permute(1, 2, 0).numpy()  # [H, W, C]
            target_np = target_frame.permute(1, 2, 0).numpy()  # [H, W, C]

            # Scale from [0, 1] to [0, 255] and convert to uint8
            source_np_uint8 = (np.clip(source_np, 0.0, 1.0) * 255).astype(np.uint8)
            target_np_uint8 = (np.clip(target_np, 0.0, 1.0) * 255).astype(np.uint8)

            # Convert to LAB color space
            source_lab = cv2.cvtColor(source_np_uint8, cv2.COLOR_RGB2LAB)
            target_lab = cv2.cvtColor(target_np_uint8, cv2.COLOR_RGB2LAB)

            # Compute mean and standard deviation
            src_mean, src_std = cv2.meanStdDev(source_lab)
            tgt_mean, tgt_std = cv2.meanStdDev(target_lab)

            src_mean, src_std = src_mean.flatten(), src_std.flatten()
            tgt_mean, tgt_std = tgt_mean.flatten(), tgt_std.flatten()

            src_std = np.clip(src_std, 1e-6, None)
            tgt_std = np.clip(tgt_std, 1e-6, None)

            # Normalize target LAB channels
            target_lab_float = target_lab.astype(np.float32)
            for i in range(3):
                target_lab_float[:, :, i] = (target_lab_float[:, :, i] - tgt_mean[i]) / tgt_std[i] * src_std[i] + src_mean[i]

            # Clip and convert back
            target_lab_float = np.clip(target_lab_float, 0, 255)
            adjusted_lab_uint8 = target_lab_float.astype(np.uint8)
            adjusted_rgb = cv2.cvtColor(adjusted_lab_uint8, cv2.COLOR_LAB2RGB)

            return torch.from_numpy(adjusted_rgb).permute(2, 0, 1).float() / 255.0
        except Exception as e:
            logger.error(f"Error during color transfer: {e}")
            return target_frame

    def _apply_directional_dilation(self, frame_chunk: torch.Tensor, mask_chunk: torch.Tensor) -> torch.Tensor:
        """Fills occluded areas by dilating from right to create clean reference"""
        try:
            if frame_chunk.shape[0] != mask_chunk.shape[0]:
                return frame_chunk

            filled_frames_list = []
            device = frame_chunk.device

            for t in range(frame_chunk.shape[0]):
                frame_np = (frame_chunk[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                mask_np = (mask_chunk[t].squeeze(0).cpu().numpy() * 255).astype(np.uint8)

                # Inpaint using Telea
                inpainted = cv2.inpaint(frame_np, mask_np, 3, cv2.INPAINT_TELEA)
                filled_tensor = torch.from_numpy(inpainted).permute(2, 0, 1).float() / 255.0
                filled_frames_list.append(filled_tensor.to(device))

            return torch.stack(filled_frames_list)
        except Exception as e:
            logger.error(f"Error during directional dilation: {e}")
            return frame_chunk

    def _apply_gaussian_blur(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply Gaussian blur to mask"""
        if kernel_size <= 0:
            return mask

        kernel_val = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        if kernel_val < 3:
            kernel_val = 3

        sigma = kernel_val / 6.0
        if sigma < 0.1:
            sigma = 0.1

        kernel = self._create_1d_gaussian_kernel(kernel_val, sigma).to(mask.device)
        kernel_x = kernel.view(1, 1, 1, kernel_val)
        kernel_y = kernel.view(1, 1, kernel_val, 1)

        padding_x = kernel_val // 2
        blurred_mask = F.conv2d(mask, kernel_x, padding=(0, padding_x), groups=mask.shape[1])

        padding_y = kernel_val // 2
        blurred_mask = F.conv2d(blurred_mask, kernel_y, padding=(padding_y, 0), groups=mask.shape[1])

        return torch.clamp(blurred_mask, 0.0, 1.0)

    def _create_1d_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel"""
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        gauss = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        kernel = gauss / gauss.sum()
        return kernel


