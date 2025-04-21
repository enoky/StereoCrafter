import os
import cv2
import numpy as np
from fire import Fire

import torch
from decord import VideoReader, cpu

from transformers import CLIPVisionModelWithProjection
from diffusers import (
    AutoencoderKLTemporalDecoder,
)
from diffusers import UNetSpatioTemporalConditionModel

from pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline, tensor2vid


def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1) / overlap_size).to(
        b.device
    )
    b[:, :, :, :overlap_size] = (1 - weight_b) * a[
        :, :, :, -overlap_size:
    ] + weight_b * b[:, :, :, :overlap_size]
    return b


def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, -1, 1) / overlap_size).to(
        b.device
    )
    b[:, :, :overlap_size, :] = (1 - weight_b) * a[
        :, :, -overlap_size:, :
    ] + weight_b * b[:, :, :overlap_size, :]
    return b


def spatial_tiled_process(
    cond_frames,
    mask_frames,
    process_func,
    tile_num,
    spatial_n_compress=8,
    **kargs,
):
    height = cond_frames.shape[2]
    width = cond_frames.shape[3]

    tile_overlap = (128, 128)
    tile_size = (
        int((height + tile_overlap[0] *  (tile_num - 1)) / tile_num), 
        int((width  + tile_overlap[1] * (tile_num - 1)) / tile_num)
    )
    tile_stride = (
        (tile_size[0] - tile_overlap[0]), 
        (tile_size[1] - tile_overlap[1])
        )
    
    cols = []
    for i in range(0, tile_num):
        rows = []
        for j in range(0, tile_num):

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

    num_frames = len(input_frames)
    height, width, _ = input_frames[0].shape

    out = cv2.VideoWriter(
        output_video_path, 
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, 
        (width, height)
    )

    for i in range(num_frames):
        out.write(input_frames[i, :, :, ::-1])

    out.release()



def main(
    pre_trained_path,
    unet_path,
    input_video_path,
    save_dir,
    frames_chunk=23,
    overlap=3,
    tile_num=1
):
    
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
        # variant="fp16",
        torch_dtype=torch.float16
    )

    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(
        pre_trained_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=torch.float16,
    )
    pipeline = pipeline.to("cuda")

    os.makedirs(save_dir, exist_ok=True)
    video_name = input_video_path.split("/")[-1].replace(".mp4", "").replace("_splatting_results", "") + "_inpainting_results"

    video_reader = VideoReader(input_video_path, ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    frame_indices = list(range(len(video_reader)))
    frames = video_reader.get_batch(frame_indices)
    num_frames = len(video_reader)

    # [t,h,w,c] -> [t,c,h,w]
    frames = (
        torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2).float()
    )  

    height, width = frames.shape[2] // 2, frames.shape[3] // 2
    frames_left = frames[:, :, :height, :width]
    frames_mask = frames[:, :, height:, :width]
    frames_warpped = frames[:, :, height:, width:]
    frames = torch.cat([frames_warpped, frames_left, frames_mask], dim=0)

    height = height // 128 * 128
    width = width // 128 * 128
    frames = frames[:, :, 0:height, 0:width]

    frames = frames / 255.0
    frames_warpped, frames_left, frames_mask = torch.chunk(frames, chunks=3, dim=0)
    frames_mask = frames_mask.mean(dim=1, keepdim=True)

    results = []
    generated = None
    for i in range(0, num_frames, frames_chunk - overlap):

        if i + overlap >= frames_warpped.shape[0]:
            break

        if generated is not None and i + frames_chunk > frames_warpped.shape[0]:
            cur_i = max(frames_warpped.shape[0] + overlap - frames_chunk, 0)
            cur_overlap = i - cur_i + overlap
        else:
            cur_i = i
            cur_overlap = overlap

        input_frames_i = frames_warpped[cur_i : cur_i + frames_chunk].clone()
        mask_frames_i = frames_mask[cur_i : cur_i + frames_chunk]

        if generated is not None:

            try:
                input_frames_i[:cur_overlap] = generated[-cur_overlap:]
            except Exception as e:
                print(e)
                print(
                    f"i: {i}, cur_i: {cur_i}, cur_overlap: {cur_overlap}, input_frames_i: {input_frames_i.shape}, generated: {generated.shape}"
                )

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

        video_latents = video_latents.unsqueeze(0)
        if video_latents == torch.float16:
            pipeline.vae.to(dtype=torch.float16)

        video_frames = pipeline.decode_latents(video_latents, num_frames=video_latents.shape[1], decode_chunk_size=2)
        video_frames = tensor2vid(video_frames, pipeline.image_processor, output_type="pil")[0]

        for j in range(len(video_frames)):
            img = video_frames[j]
            video_frames[j] = (
                torch.tensor(np.array(img)).permute(2, 0, 1).to(dtype=torch.float32)
                / 255.0
            )
        generated = torch.stack(video_frames)
        if i != 0:
            generated = generated[cur_overlap:]
        results.append(generated)

    frames_output = torch.cat(results, dim=0).cpu()


    frames_sbs = torch.cat([frames_left, frames_output], dim=3)
    frames_sbs_path = os.path.join(save_dir, f"{video_name}_sbs.mp4")
    frames_sbs = (frames_sbs * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
    write_video_opencv(frames_sbs, fps, frames_sbs_path)


    vid_left = (frames_left * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
    vid_right = (frames_output * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()

    vid_left[:, :, :, 1] = 0
    vid_left[:, :, :, 2] = 0
    vid_right[:, :, :, 0] = 0

    vid_anaglyph = vid_left + vid_right
    vid_anaglyph_path = os.path.join(save_dir, f"{video_name}_anaglyph.mp4")
    write_video_opencv(vid_anaglyph, fps, vid_anaglyph_path)


if __name__ == "__main__":
    Fire(main)
