import os
import glob
import logging
import shutil  # Step 1: Import shutil for moving files
from typing import Optional, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from fire import Fire
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
import imageio_ffmpeg
import torch.nn.functional as F

from pipelines.stereo_video_inpainting import (
    StableVideoDiffusionInpaintingPipeline,
    tensor2vid
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_inpainting_pipeline(
    pre_trained_path: str,
    unet_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> StableVideoDiffusionInpaintingPipeline:
    """
    Loads the stable video diffusion inpainting pipeline components (image encoder, VAE, UNet)
    and returns the pipeline object with CPU offloading enabled.
    """
    logger.info("Loading pipeline components...")

    # Load components
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pre_trained_path,
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=dtype,
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pre_trained_path,
        subfolder="vae",
        variant="fp16",
        torch_dtype=dtype,
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        subfolder="unet_diffusers",
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )

    # Freeze parameters
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(
        pre_trained_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=dtype,
    ).to(device)

    # Enable CPU offloading to reduce GPU memory usage
    pipeline.enable_model_cpu_offload()

    return pipeline

def read_video_frames(video_path: str, decord_ctx=cpu(0)) -> Tuple[torch.Tensor, float]:
    """
    Reads a video using decord and returns frames as a 4D float tensor [T, C, H, W] and the FPS.
    """
    video_reader = VideoReader(video_path, ctx=decord_ctx)
    num_frames = len(video_reader)

    if num_frames == 0:
        return torch.empty(0), 0.0

    fps = video_reader.get_avg_fps()
    frames = video_reader.get_batch(range(num_frames))  # [T, H, W, C]
    frames = torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2).float()  # [T, C, H, W]

    return frames, fps

def write_video_ffmpeg(
    frames: np.ndarray,
    fps: float,
    output_path: str,
    vf: Optional[str] = None,
    codec: str = "libx264",
    crf: int = 16,
    preset: str = "ultrafast"
) -> None:
    """
    Writes a sequence of frames (uint8) to a video file using imageio-ffmpeg.
    """
    if frames.dtype != np.uint8:
        frames = (frames * 255).astype(np.uint8)

    frames = np.ascontiguousarray(frames)
    height, width = frames.shape[1:3]

    # Build FFmpeg output parameters
    output_params = ["-crf", str(crf), "-preset", preset]
    if vf:
        output_params += ["-vf", vf]

    writer = imageio_ffmpeg.write_frames(
        output_path,
        (width, height),
        fps=fps,
        codec=codec,
        quality=None,
        bitrate=None,
        output_params=output_params
    )
    writer.send(None)  # Initialize the writer
    for frame in frames:
        writer.send(frame)
    writer.close()

def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """
    Blend two tensors horizontally along the right edge of `a` and left edge of `b`.
    """
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1) / overlap_size).to(b.device)
    b[:, :, :, :overlap_size] = (
        (1 - weight_b) * a[:, :, :, -overlap_size:] + weight_b * b[:, :, :, :overlap_size]
    )
    return b

def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """
    Blend two tensors vertically along the bottom edge of `a` and top edge of `b`.
    """
    weight_b = (torch.arange(overlap_size).view(1, 1, -1, 1) / overlap_size).to(b.device)
    b[:, :, :overlap_size, :] = (
        (1 - weight_b) * a[:, :, -overlap_size:, :] + weight_b * b[:, :, :overlap_size, :]
    )
    return b

def pad_for_tiling(frames: torch.Tensor, tile_num: int, tile_overlap=(128, 128)) -> torch.Tensor:
    """
    Zero-pads a batch of frames (shape [T, C, H, W]) so that
    (H, W) fits perfectly into 'tile_num' splits plus overlap.

    If tile_num=1, this typically adds no padding unless
    your height/width is slightly smaller than required.
    """
    if tile_num <= 1:
        # No need to do complex tiling; no significant padding
        return frames

    T, C, H, W = frames.shape
    overlap_y, overlap_x = tile_overlap

    # Compute tile size (same as spatial_tiled_process)
    size_y = (H + overlap_y * (tile_num - 1)) // tile_num
    size_x = (W + overlap_x * (tile_num - 1)) // tile_num

    stride_y = size_y - overlap_y
    stride_x = size_x - overlap_x

    # Ideal dimension that lines up with tile_num + overlap
    ideal_H = stride_y * tile_num + overlap_y * (tile_num - 1)
    ideal_W = stride_x * tile_num + overlap_x * (tile_num - 1)

    pad_bottom = max(0, ideal_H - H)
    pad_right = max(0, ideal_W - W)

    if pad_bottom > 0 or pad_right > 0:
        # (pad_left, pad_right, pad_top, pad_bottom)
        frames = F.pad(frames, (0, pad_right, 0, pad_bottom), mode="constant", value=0.0)
    return frames

def spatial_tiled_process(
    cond_frames: torch.Tensor,
    mask_frames: torch.Tensor,
    process_func,
    tile_num: int,
    spatial_n_compress: int = 8,
    **kargs,
) -> torch.Tensor:
    """
    Splits frames into tiles, processes them with `process_func`, then blends the results back together.
    """
    height = cond_frames.shape[2]
    width = cond_frames.shape[3]

    tile_overlap = (128, 128)
    overlap_y, overlap_x = tile_overlap

    # Compute tile size
    size_y = (height + overlap_y * (tile_num - 1)) // tile_num
    size_x = (width + overlap_x * (tile_num - 1)) // tile_num
    tile_size = (size_y, size_x)

    # Stride
    tile_stride = (size_y - overlap_y, size_x - overlap_x)

    # Process each tile
    cols = []
    for i in range(tile_num):
        row_tiles = []
        for j in range(tile_num):
            y_start = i * tile_stride[0]
            x_start = j * tile_stride[1]
            y_end = y_start + tile_size[0]
            x_end = x_start + tile_size[1]

            # Clip edges to avoid out-of-bounds
            y_end = min(y_end, height)
            x_end = min(x_end, width)

            cond_tile = cond_frames[:, :, y_start:y_end, x_start:x_end]
            mask_tile = mask_frames[:, :, y_start:y_end, x_start:x_end]

            with torch.no_grad():
                tile_output = process_func(
                    frames=cond_tile,
                    frames_mask=mask_tile,
                    height=cond_tile.shape[2],
                    width=cond_tile.shape[3],
                    num_frames=len(cond_tile),
                    output_type="latent",
                    **kargs,
                ).frames[0]

            row_tiles.append(tile_output)
        cols.append(row_tiles)

    # Blending in latent space
    latent_stride = (
        tile_stride[0] // spatial_n_compress,
        tile_stride[1] // spatial_n_compress
    )
    latent_overlap = (
        overlap_y // spatial_n_compress,
        overlap_x // spatial_n_compress
    )

    # Blend tiles: vertical, then horizontal
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

    # Stitch final
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

def process_single_video(
    pipeline: StableVideoDiffusionInpaintingPipeline,
    input_video_path: str,
    save_dir: str,
    frames_chunk: int = 63,
    overlap: int = 3,
    tile_num: int = 2,
    vf: Optional[str] = None,
) -> None:
    """
    Processes a single input video, writes the right side view output to `save_dir`.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Output video name
    video_name = (
        os.path.basename(input_video_path)
        .replace(".mp4", "")
        .replace("_splatting_results", "")
        + "_inpainting_results"
    )

    # Read video
    frames, fps = read_video_frames(input_video_path)
    num_frames = frames.shape[0]

    if num_frames == 0:
        logger.warning(f"No frames found in {input_video_path}, skipping.")
        return

    # Guard against extremely small frames
    _, _, num_h, num_w = frames.shape
    if num_h < 2 or num_w < 2:
        logger.warning(f"Video {input_video_path} is too small ({num_h}x{num_w}), skipping.")
        return

    # Split the frames:
    #   top-left = left
    #   bottom-left = mask
    #   bottom-right = warpped
    half_h = num_h // 2
    half_w = num_w // 2

    frames_left = frames[:, :, :half_h, :half_w]
    frames_mask = frames[:, :, half_h:, :half_w]
    frames_warpped = frames[:, :, half_h:, half_w:]

    # Combine warpped, left, mask for consistent shape transformations
    frames_combined = torch.cat([frames_warpped, frames_left, frames_mask], dim=0)
    frames_combined = frames_combined / 255.0  # scale to [0,1]

    frames_warpped, frames_left, frames_mask = torch.chunk(frames_combined, 3, dim=0)

    # Convert mask to grayscale
    frames_mask = frames_mask.mean(dim=1, keepdim=True)

    # -- Add padding if tile_num>1 --
    # This ensures warpped and mask have heights/widths that fit exactly
    # into the tile_num+overlap math.
    frames_warpped = pad_for_tiling(frames_warpped, tile_num, tile_overlap=(128, 128))
    frames_mask = pad_for_tiling(frames_mask, tile_num, tile_overlap=(128, 128))

    # Adjust overlap/chunk if video is too short
    if num_frames < overlap:
        overlap = 0
    if num_frames < frames_chunk:
        frames_chunk = num_frames

    stride = max(1, frames_chunk - overlap)
    results = []
    generated = None

    # Process in temporal chunks
    for i in range(0, num_frames, stride):
        end_idx = min(i + frames_chunk, num_frames)
        chunk_size = end_idx - i
        if chunk_size <= 0:
            break

        input_frames_i = frames_warpped[i:end_idx].clone()
        mask_frames_i = frames_mask[i:end_idx].clone()

        # Overlap from previous chunk
        if generated is not None and overlap > 0 and i != 0:
            overlap_actual = min(overlap, len(generated))
            input_frames_i[:overlap_actual] = generated[-overlap_actual:]

        with torch.no_grad():
            # Tiled inference in latent space
            video_latents = spatial_tiled_process(
                input_frames_i,
                mask_frames_i,
                pipeline,
                tile_num,
                spatial_n_compress=8,
                min_guidance_scale=1.01,
                max_guidance_scale=1.01,
                decode_chunk_size=8,
                fps=7,
                motion_bucket_id=127,
                noise_aug_strength=0.0,
                num_inference_steps=8,
            )
            video_latents = video_latents.unsqueeze(0)  # [1, T, C, H, W]

            # Decode latents
            pipeline.vae.to(dtype=torch.float16)
            decoded_frames = pipeline.decode_latents(
                video_latents,
                num_frames=video_latents.shape[1],
                decode_chunk_size=2,
            )

        # Convert to PIL then back to torch
        video_frames = tensor2vid(decoded_frames, pipeline.image_processor, output_type="pil")[0]
        for j in range(len(video_frames)):
            img = video_frames[j]
            video_frames[j] = (
                torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            )

        generated = torch.stack(video_frames)
        if i != 0 and overlap > 0:
            generated = generated[overlap:]

        results.append(generated)
        if end_idx == num_frames:
            break

    # Concatenate all chunks
    frames_output = torch.cat(results, dim=0).cpu()  # [T, C, H, W]

    # Write the "right side view"
    frames_output_np = (frames_output * 255).permute(0, 2, 3, 1).byte().numpy()
    frames_output_path = os.path.join(save_dir, f"{video_name}_right.mp4")

    write_video_ffmpeg(
        frames=frames_output_np,
        fps=fps,
        output_path=frames_output_path,
        vf=vf if vf else None,
        codec="libx264",
        crf=16,
        preset="ultrafast",
    )

    logger.info(f"Done processing {input_video_path} -> {frames_output_path}")

def batch_process(
    pre_trained_path: str = "./weights/stable-video-diffusion-img2vid-xt-1-1",
    unet_path: str = "./weights/StereoCrafter",
    input_folder: str = "./output_splatted",
    output_folder: str = "./completed_output",
    frames_chunk: int = 63,
    overlap: int = 3,
    tile_num: int = 2,
    vf: Optional[str] = None,
) -> None:
    """
    Batch-process all .mp4 files in `input_folder` using the inpainting pipeline
    and save results into `output_folder`.
    """
    # Load pipeline once
    pipeline = load_inpainting_pipeline(
        pre_trained_path=pre_trained_path,
        unet_path=unet_path,
        device="cuda",
        dtype=torch.float16,
    )

    # Gather input videos
    input_videos = sorted(glob.glob(os.path.join(input_folder, "*.mp4")))
    if not input_videos:
        logger.info(f"No .mp4 files found in {input_folder}. Exiting.")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Step 2: Create 'finished' directory inside input_folder
    finished_folder = os.path.join(input_folder, "finished")
    os.makedirs(finished_folder, exist_ok=True)

    # Process each video
    for video_path in input_videos:
        logger.info(f"Processing {video_path}")
        process_single_video(
            pipeline=pipeline,
            input_video_path=video_path,
            save_dir=output_folder,
            frames_chunk=frames_chunk,
            overlap=overlap,
            tile_num=tile_num,
            vf=vf
        )
        torch.cuda.empty_cache()

        # Step 3: Move the processed video to 'finished' folder
        try:
            shutil.move(video_path, finished_folder)
            logger.info(f"Moved {video_path} to {finished_folder}")
        except Exception as e:
            logger.error(f"Failed to move {video_path} to {finished_folder}: {e}")

if __name__ == "__main__":
    Fire(batch_process)
