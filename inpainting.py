import os
import glob
import cv2
import numpy as np
from fire import Fire
from decord import VideoReader, cpu

import torch

from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder
from diffusers import UNetSpatioTemporalConditionModel

from pipelines.stereo_video_inpainting import (
    StableVideoDiffusionInpaintingPipeline,
    tensor2vid
)


def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1) / overlap_size).to(
        b.device
    )
    b[:, :, :, :overlap_size] = (1 - weight_b) * a[:, :, :, -overlap_size:] + weight_b * b[:, :, :, :overlap_size]
    return b


def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, -1, 1) / overlap_size).to(
        b.device
    )
    b[:, :, :overlap_size, :] = (1 - weight_b) * a[:, :, -overlap_size:, :] + weight_b * b[:, :, :overlap_size, :]
    return b


def spatial_tiled_process(
    cond_frames,
    mask_frames,
    process_func,
    tile_num,
    spatial_n_compress=8,
    **kargs,
):
    """
    Splits frames into tiles, processes them, then blends the results back together.
    """
    height = cond_frames.shape[2]
    width = cond_frames.shape[3]

    tile_overlap = (128, 128)
    tile_size = (
        int((height + tile_overlap[0] * (tile_num - 1)) / tile_num),
        int((width + tile_overlap[1] * (tile_num - 1)) / tile_num),
    )
    tile_stride = (
        (tile_size[0] - tile_overlap[0]),
        (tile_size[1] - tile_overlap[1]),
    )

    # Process each tile
    cols = []
    for i in range(tile_num):
        rows = []
        for j in range(tile_num):
            cond_tile = cond_frames[
                :,
                :,
                i * tile_stride[0] : i * tile_stride[0] + tile_size[0],
                j * tile_stride[1] : j * tile_stride[1] + tile_size[1],
            ]
            mask_tile = mask_frames[
                :,
                :,
                i * tile_stride[0] : i * tile_stride[0] + tile_size[0],
                j * tile_stride[1] : j * tile_stride[1] + tile_size[1],
            ]

            tile = process_func(
                frames=cond_tile,
                frames_mask=mask_tile,
                height=cond_tile.shape[2],
                width=cond_tile.shape[3],
                num_frames=len(cond_tile),
                output_type="latent",
                **kargs,
            ).frames[0]

            rows.append(tile)
        cols.append(rows)

    latent_stride = (
        tile_stride[0] // spatial_n_compress,
        tile_stride[1] // spatial_n_compress,
    )
    latent_overlap = (
        tile_overlap[0] // spatial_n_compress,
        tile_overlap[1] // spatial_n_compress,
    )

    # Blend the tiles vertically, then horizontally
    results_cols = []
    for i, rows in enumerate(cols):
        results_rows = []
        for j, tile in enumerate(rows):
            if i > 0:
                tile = blend_v(cols[i - 1][j], tile, latent_overlap[0])
            if j > 0:
                tile = blend_h(rows[j - 1], tile, latent_overlap[1])
            results_rows.append(tile)
        results_cols.append(results_rows)

    # Stitch the final result
    pixels = []
    for i, rows in enumerate(results_cols):
        for j, tile in enumerate(rows):
            if i < len(results_cols) - 1:
                tile = tile[:, :, : latent_stride[0], :]
            if j < len(rows) - 1:
                tile = tile[:, :, :, : latent_stride[1]]
            rows[j] = tile
        pixels.append(torch.cat(rows, dim=3))
    x = torch.cat(pixels, dim=2)
    return x


def write_video_opencv(input_frames, fps, output_video_path):
    """
    Writes a sequence of RGB frames [T, H, W, C] to an mp4 file.
    """
    num_frames = len(input_frames)
    height, width, _ = input_frames[0].shape

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for i in range(num_frames):
        # Convert from RGB -> BGR for OpenCV
        out.write(input_frames[i, :, :, ::-1])

    out.release()


def process_single_video(
    pipeline,
    input_video_path,
    save_dir,
    frames_chunk=23,
    overlap=3,
    tile_num=2
):
    """
    Processes a single input video, writes the side-by-side (SBS) output to `save_dir`.
    Handles corner cases for very short videos (including single-frame).
    """
    # Create output folder
    os.makedirs(save_dir, exist_ok=True)
    # Generate output video name
    video_name = (
        os.path.basename(input_video_path)
        .replace(".mp4", "")
        .replace("_splatting_results", "")
        + "_inpainting_results"
    )

    # Read input video
    video_reader = VideoReader(input_video_path, ctx=cpu(0))
    num_frames = len(video_reader)

    # If there's literally no frame, just return
    if num_frames == 0:
        print(f"[WARN] No frames found in {input_video_path}, skipping.")
        return

    fps = video_reader.get_avg_fps()
    frame_indices = list(range(num_frames))
    frames = video_reader.get_batch(frame_indices)  # [T, H, W, C]

    # Convert to Torch tensor: [T, H, W, C] -> [T, C, H, W]
    frames = torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2).float()

    num_h = frames.shape[2]
    num_w = frames.shape[3]

    # Guard against extremely small frames
    if num_h < 2 or num_w < 2:
        print(f"[WARN] Video {input_video_path} is too small ({num_h}x{num_w}), skipping.")
        return

    half_h = num_h // 2
    half_w = num_w // 2

    # The original script assumes:
    #   top-left      = left
    #   bottom-left   = mask
    #   bottom-right  = warpped
    frames_left = frames[:, :, :half_h, :half_w]
    frames_mask = frames[:, :, half_h:, :half_w]
    frames_warpped = frames[:, :, half_h:, half_w:]

    # Now re-stack them as [3T, C, H, W]
    frames = torch.cat([frames_warpped, frames_left, frames_mask], dim=0)

    # Round down to multiples of 128 for H/W
    h_aligned = (half_h // 128) * 128
    w_aligned = (half_w // 128) * 128

    frames = frames[:, :, :h_aligned, :w_aligned]
    frames = frames / 255.0

    frames_warpped, frames_left, frames_mask = torch.chunk(frames, chunks=3, dim=0)
    # Convert frames_mask to grayscale
    frames_mask = frames_mask.mean(dim=1, keepdim=True)

    # Adjust overlap/frames_chunk if video is very short
    if num_frames < overlap:
        overlap = 0
    if num_frames < frames_chunk:
        frames_chunk = num_frames

    # Process video frames in chunks
    results = []
    generated = None
    stride = max(1, frames_chunk - overlap)

    for i in range(0, num_frames, stride):
        end_idx = min(i + frames_chunk, num_frames)
        chunk_size = end_idx - i
        if chunk_size <= 0:
            break

        input_frames_i = frames_warpped[i:end_idx].clone()
        mask_frames_i = frames_mask[i:end_idx].clone()

        # If we have a previously generated tail, apply overlap
        if generated is not None and overlap > 0 and i != 0:
            overlap_actual = min(overlap, len(generated))
            input_frames_i[:overlap_actual] = generated[-overlap_actual:]

        # Tiled process to get latents
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
        pipeline.vae.to(dtype=torch.float16)

        # Decode latents to frames
        video_frames = pipeline.decode_latents(
            video_latents,
            num_frames=video_latents.shape[1],
            decode_chunk_size=2
        )
        # Convert decoded frames to PIL -> then to torch float
        video_frames = tensor2vid(video_frames, pipeline.image_processor, output_type="pil")[0]

        for j in range(len(video_frames)):
            img = video_frames[j]
            video_frames[j] = (
                torch.tensor(np.array(img))
                .permute(2, 0, 1)
                .to(dtype=torch.float32)
                / 255.0
            )

        generated = torch.stack(video_frames)
        # If this is not the very first chunk, discard the overlapped frames
        if i != 0 and overlap > 0:
            generated = generated[overlap:]

        results.append(generated)

        if end_idx == num_frames:
            break

    # Concatenate all chunks
    frames_output = torch.cat(results, dim=0).cpu()  # [T, C, H, W]

    # Write side-by-side video (left | output)
    frames_sbs = torch.cat([frames_left[: len(frames_output)], frames_output], dim=3)
    frames_sbs = (frames_sbs * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
    frames_sbs_path = os.path.join(save_dir, f"{video_name}_sbs.mp4")
    write_video_opencv(frames_sbs, fps, frames_sbs_path)

    print(f"[INFO] Done processing {input_video_path} -> {frames_sbs_path}")


def batch_process(
    pre_trained_path="./weights/stable-video-diffusion-img2vid-xt-1-1",
    unet_path="./weights/StereoCrafter",
    input_folder="./output_splatted",
    output_folder="./completed_output",
    frames_chunk=23,
    overlap=3,
    tile_num=2
):
    """
    Batch-process all .mp4 files in `input_folder` using the inpainting pipeline
    and save results into `output_folder`.
    """
    # 1. Load pipeline components ONCE
    print("[INFO] Loading pipeline components...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pre_trained_path,
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=torch.float16
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pre_trained_path,
        subfolder="vae",
        variant="fp16",
        torch_dtype=torch.float16
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        subfolder="unet_diffusers",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    # Freeze params
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(
        pre_trained_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=torch.float16,
    ).to("cuda")

    # 2. Now loop over videos
    input_videos = sorted(glob.glob(os.path.join(input_folder, "*.mp4")))
    if not input_videos:
        print(f"[INFO] No .mp4 files found in {input_folder}. Exiting.")
        return

    os.makedirs(output_folder, exist_ok=True)

    for video_path in input_videos:
        print(f"[INFO] Processing {video_path}")
        process_single_video(
            pipeline=pipeline,
            input_video_path=video_path,
            save_dir=output_folder,
            frames_chunk=frames_chunk,
            overlap=overlap,
            tile_num=tile_num
        )
        # Clear GPU cache if memory is tight or you're experiencing OOM issues
        torch.cuda.empty_cache()


if __name__ == "__main__":
    Fire(batch_process)
