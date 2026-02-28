"""Main splatting render processor.

Implements the core video splatting algorithm, handling the processing loop,
GPU computation (forward warping), and FFmpeg encoding.
"""

import math
import os
import time
import logging
import queue
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader

from dependency.stereocrafter_util import (
    start_ffmpeg_pipe_process,
    start_ffmpeg_pipe_process_dnxhr,
    release_cuda_memory,
    draw_progress_bar,
)
from .forward_warp import ForwardWarpStereo
from .depth_processing import process_depth_batch, normalize_and_gamma_depth

logger = logging.getLogger(__name__)


class RenderProcessor:
    """Handles the core splatting render loop for a single video task."""

    def __init__(
        self,
        stop_event: threading.Event,
        progress_queue: queue.Queue,
    ):
        """Initialize render processor.

        Args:
            stop_event: Event to signal stop/cancellation
            progress_queue: Queue for sending progress updates to GUI
        """
        self.stop_event = stop_event
        self.progress_queue = progress_queue
        self._color_encode_flags = {}

    def render_video(
        self,
        input_video_reader: VideoReader,
        depth_map_reader: VideoReader,
        total_frames_to_process: int,
        processed_fps: float,
        output_video_path_base: str,
        target_output_height: int,
        target_output_width: int,
        max_disp: float,
        batch_size: int,
        dual_output: bool,
        zero_disparity_anchor_val: float,
        video_stream_info: Optional[dict],
        input_bias: float,
        assume_raw_input: bool,
        global_depth_min: float,
        global_depth_max: float,
        depth_stream_info: Optional[dict],
        user_output_crf: Optional[int] = None,
        is_low_res_task: bool = False,
        depth_gamma: float = 1.0,
        depth_dilate_size_x: float = 0.0,
        depth_dilate_size_y: float = 0.0,
        depth_blur_size_x: float = 0.0,
        depth_blur_size_y: float = 0.0,
        depth_dilate_left: float = 0.0,
        depth_blur_left: float = 0.0,
        depth_blur_left_mix: float = 0.5,
        skip_lowres_preproc: bool = False,
        color_tags_mode: str = "Auto",
        encoding_options: Optional[dict] = None,
        dnxhr_fullres_split: bool = False,
        dnxhr_profile: str = "HQX",
        is_test_mode: bool = False,
        test_target_frame_idx: Optional[int] = None,
    ) -> bool:
        """Core splatting render loop.

        Args:
            input_video_reader: Reader for source video
            depth_map_reader: Reader for depth map
            total_frames_to_process: Number of frames to process
            processed_fps: Output video FPS
            output_video_path_base: Base path for output video
            target_output_height: Target height
            target_output_width: Target width
            max_disp: Max disparity percentage
            batch_size: Frames per batch
            dual_output: Whether to output left/right eyes
            zero_disparity_anchor_val: Convergence anchor (0-1)
            video_stream_info: Metadata for source video
            input_bias: Depth input bias
            assume_raw_input: Whether to skip normalization
            global_depth_min: Global min depth used for normalization
            global_max_depth: Global max depth used for normalization
            depth_stream_info: Metadata for depth map
            user_output_crf: FFmpeg CRF value
            is_low_res_task: Whether this is a low-res pass
            depth_gamma: Gamma correction for depth
            depth_dilate_size_x: X dilation for depth
            depth_dilate_size_y: Y dilation for depth
            depth_blur_size_x: X blur for depth
            depth_blur_size_y: Y blur for depth
            depth_dilate_left: Left-eye dilation
            depth_blur_left: Left-eye blur
            depth_blur_left_mix: Mix factor for left-eye blur
            skip_lowres_preproc: Whether to skip preprocessing for low-res
            color_tags_mode: FFmpeg color tagging mode
            is_test_mode: Whether in diagnostic test mode
            test_target_frame_idx: Specific frame for diagnostic test

        Returns:
            True if completed successfully, False otherwise
        """
        logger.debug("==> Initializing ForwardWarpStereo module")
        stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

        height, width = target_output_height, target_output_width
        if not is_test_mode:
            os.makedirs(os.path.dirname(output_video_path_base), exist_ok=True)

        # Determine output grid dimensions and final path
        grid_height, grid_width = (height, width * 2) if dual_output else (height * 2, width * 2)
        suffix = "_splatted2" if dual_output else "_splatted4"
        res_suffix = f"_{width}"
        final_output_video_path = f"{os.path.splitext(output_video_path_base)[0]}{res_suffix}{suffix}.mp4"

        task_name = "LowRes" if is_low_res_task else "HiRes"
        self._log_color_metadata(video_stream_info, task_name)

        ffmpeg_process = None
        mask_process = None
        splat_process = None
        use_dnxhr_split = (
            bool(dnxhr_fullres_split) and bool(dual_output) and (not is_low_res_task) and (not is_test_mode)
        )

        if not is_test_mode:
            encode_stream_info = self._get_encode_stream_info(video_stream_info, color_tags_mode)

            if use_dnxhr_split:
                base_dir = os.path.dirname(output_video_path_base)
                stem = os.path.splitext(os.path.basename(output_video_path_base))[0]
                prefix = f"{stem}{res_suffix}"

                mask_dir = os.path.join(base_dir, "mask")
                splat_dir = os.path.join(base_dir, "splat")
                os.makedirs(mask_dir, exist_ok=True)
                os.makedirs(splat_dir, exist_ok=True)

                mask_output_path = os.path.join(mask_dir, f"{prefix}_mask.mp4")
                splat_output_path = os.path.join(splat_dir, f"{prefix}_splat.mov")

                mask_process = start_ffmpeg_pipe_process(
                    content_width=width,
                    content_height=height,
                    final_output_mp4_path=mask_output_path,
                    fps=processed_fps,
                    video_stream_info=encode_stream_info,
                    user_output_crf=user_output_crf,
                    output_format_str="mask_only",
                    debug_label=task_name,
                    encoding_options=encoding_options,
                )
                if mask_process is None:
                    logger.error("Failed to start FFmpeg pipe for mask output. Aborting splatting task.")
                    return False

                splat_process = start_ffmpeg_pipe_process_dnxhr(
                    content_width=width,
                    content_height=height,
                    final_output_mov_path=splat_output_path,
                    fps=processed_fps,
                    dnxhr_profile=dnxhr_profile,
                )
                if splat_process is None:
                    logger.error("Failed to start DNxHR pipe for splat output. Aborting splatting task.")
                    try:
                        mask_process.stdin.close()
                        mask_process.wait(timeout=10)
                    except Exception:
                        pass
                    return False

                ffmpeg_process = mask_process
            else:
                ffmpeg_process = start_ffmpeg_pipe_process(
                    content_width=grid_width,
                    content_height=grid_height,
                    final_output_mp4_path=final_output_video_path,
                    fps=processed_fps,
                    video_stream_info=encode_stream_info,
                    user_output_crf=user_output_crf,
                    output_format_str="splatted_grid",
                    debug_label=task_name,
                    encoding_options=encoding_options,
                )
                if ffmpeg_process is None:
                    logger.error("Failed to start FFmpeg pipe. Aborting splatting task.")
                    return False
            
            self._compare_encoding_flags(ffmpeg_process, task_name)

        max_expected_raw_value = self._get_max_expected_raw_depth(depth_stream_info)
        logger.debug(f"[DEPTH] Max expected raw value: {max_expected_raw_value}, assume_raw_input: {assume_raw_input}, global_depth_min: {global_depth_min:.2f}, global_depth_max: {global_depth_max:.2f}")
        
        tv_disp_comp = 1.0
        if assume_raw_input and depth_stream_info and max_expected_raw_value > 256.0:
            if str(depth_stream_info.get("color_range", "")).lower() == "tv":
                from core.splatting.depth_processing import DEPTH_VIS_TV10_WHITE_NORM, DEPTH_VIS_TV10_BLACK_NORM
                tv_disp_comp = 1.0 / (DEPTH_VIS_TV10_WHITE_NORM - DEPTH_VIS_TV10_BLACK_NORM)
                logger.debug(f"[DEPTH] TV range compensation enabled: {tv_disp_comp:.3f}")
        
        frame_count = 0
        encoding_successful = True

        try:
            frame_index_iter = (
                [test_target_frame_idx]
                if test_target_frame_idx is not None
                else range(0, total_frames_to_process, batch_size)
            )

            for i in frame_index_iter:
                if self.stop_event.is_set() or (ffmpeg_process is not None and ffmpeg_process.poll() is not None) or (use_dnxhr_split and splat_process is not None and splat_process.poll() is not None):
                    break

                batch_indices = list(range(i, min(i + batch_size, total_frames_to_process)))
                if not batch_indices:
                    break

                # 1. Fetch frames
                batch_video_numpy = input_video_reader.get_batch(batch_indices).asnumpy()
                batch_depth_numpy_raw = depth_map_reader.get_batch(batch_indices).asnumpy()

                # --- NEW: Aspect Ratio Parity ---
                # Immediate resize to ensure all following steps (normalization, dilation, blur)
                # occur at the correct target aspect ratio, matching the GUI previewer.
                video_h, video_w = batch_video_numpy.shape[1], batch_video_numpy.shape[2]
                depth_h, depth_w = batch_depth_numpy_raw.shape[1], batch_depth_numpy_raw.shape[2]
                
                if depth_h != video_h or depth_w != video_w:
                    logger.debug(f"Resizing depth from {depth_w}x{depth_h} to match video {video_w}x{video_h} for aspect-ratio parity.")
                    interp = cv2.INTER_AREA if (video_w < depth_w and video_h < depth_h) else cv2.INTER_LINEAR
                    resized_depth = np.empty((batch_depth_numpy_raw.shape[0], video_h, video_w), dtype=batch_depth_numpy_raw.dtype)
                    for idx in range(batch_depth_numpy_raw.shape[0]):
                        resized_depth[idx] = cv2.resize(
                            batch_depth_numpy_raw[idx],
                            (video_w, video_h),
                            interpolation=interp
                        )
                    batch_depth_numpy_raw = resized_depth

                # 2. Normalize and apply gamma (BEFORE dilation/blur)
                batch_depth_normalized = normalize_and_gamma_depth(
                    batch_depth_numpy_raw=batch_depth_numpy_raw,
                    assume_raw_input=assume_raw_input,
                    global_depth_max=global_depth_max,
                    global_depth_min=global_depth_min,
                    max_expected_raw_value=max_expected_raw_value,
                    zero_disparity_anchor_val=zero_disparity_anchor_val,
                    depth_gamma=depth_gamma,
                    debug_task_name="Render",
                )
                
                logger.debug(f"[DEPTH] After normalization and gamma: min={batch_depth_normalized.min():.4f}, max={batch_depth_normalized.max():.4f}")
                
                # Convert back to "raw" format for process_depth_batch (which expects raw-like values)
                # Scale back to max_raw_value range so dilation/blur work correctly
                batch_depth_for_processing = batch_depth_normalized * max_expected_raw_value
                
                # Add channel dimension for process_depth_batch
                if batch_depth_for_processing.ndim == 3:
                    batch_depth_for_processing = batch_depth_for_processing[..., None]

                # 3. Process depth batch (dilation/blur)
                batch_depth_processed = process_depth_batch(
                    batch_depth_numpy_raw=batch_depth_for_processing,
                    depth_gamma=1.0,  # Already applied above
                    depth_dilate_size_x=depth_dilate_size_x,
                    depth_dilate_size_y=depth_dilate_size_y,
                    depth_blur_size_x=depth_blur_size_x,
                    depth_blur_size_y=depth_blur_size_y,
                    max_raw_value=max_expected_raw_value,
                    depth_dilate_left=depth_dilate_left,
                    depth_blur_left=depth_blur_left,
                    depth_blur_left_mix=depth_blur_left_mix,
                    skip_preprocessing=skip_lowres_preproc and is_low_res_task,
                )
                
                # Normalize back to 0-1 after processing
                batch_depth_numpy_float = batch_depth_processed / max(max_expected_raw_value, 1.0)
                batch_depth_numpy_float = np.clip(batch_depth_numpy_float, 0.0, 1.0)

                # 4. GPU Splatting
                batch_processed_frames = self._process_gpu_splatting(
                    stereo_projector=stereo_projector,
                    batch_video_numpy=batch_video_numpy,
                    batch_depth_numpy_float=batch_depth_numpy_float,
                    target_width=width,
                    target_height=height,
                    max_disp=max_disp,
                    zero_disparity_anchor_val=zero_disparity_anchor_val,
                    input_bias=input_bias,
                    tv_disp_comp=tv_disp_comp,
                )

                # 5. Handle results (diag tests or FFmpeg write)
                if is_test_mode and test_target_frame_idx is not None:
                    self._handle_diagnostic_capture(batch_processed_frames, dual_output, task_name)
                elif ffmpeg_process:
                    if use_dnxhr_split and mask_process and splat_process:
                        self._write_split_to_ffmpeg(mask_process, splat_process, batch_processed_frames)
                    else:
                        self._write_to_ffmpeg(ffmpeg_process, batch_processed_frames, dual_output)

                frame_count += len(batch_indices)
                self.progress_queue.put(("processed", frame_count))
                if not is_test_mode:
                    draw_progress_bar(frame_count, total_frames_to_process, suffix=f"{task_name} Batch {i//batch_size}")
                
                # Cleanup batch
                del batch_video_numpy, batch_depth_numpy_raw, batch_depth_numpy_float, batch_processed_frames
                release_cuda_memory()

        except Exception as e:
            logger.error(f"Render error: {e}", exc_info=True)
            encoding_successful = False
        finally:
            if ffmpeg_process:
                try:
                    ffmpeg_process.stdin.close()
                    ffmpeg_process.wait(timeout=30)
                except Exception as e:
                    logger.warning(f"Error closing FFmpeg: {e}")
            
            if use_dnxhr_split and splat_process:
                try:
                    splat_process.stdin.close()
                    splat_process.wait(timeout=30)
                except Exception as e:
                    logger.warning(f"Error closing DNxHR pipe: {e}")
            
            del stereo_projector
            release_cuda_memory()

        return encoding_successful

    def _log_color_metadata(self, info: Optional[dict], task_name: str):
        if not info: return
        try:
            logger.debug(
                f"[COLOR_META][{task_name}] input ffprobe: "
                f"pix_fmt={info.get('pix_fmt')}, range={info.get('color_range')}, "
                f"primaries={info.get('color_primaries')}, trc={info.get('transfer_characteristics')}, "
                f"matrix={info.get('color_space')}"
            )
        except Exception: pass

    def _get_encode_stream_info(self, source_info: Optional[dict], mode: str) -> dict:
        info = dict(source_info) if source_info else {}
        defaults = {
            "color_primaries": "bt709",
            "transfer_characteristics": "bt709",
            "color_space": "bt709",
            "color_range": "tv",
        }
        
        if mode == "Auto":
            for k, v in defaults.items(): info.setdefault(k, v)
        elif mode in ("BT.709", "BT.709 L"):
            info.update(defaults)
        elif mode == "BT.709 F":
            info.update(defaults)
            info["color_range"] = "pc"
        elif mode in ("BT.2020", "BT.2020 PQ"):
            info.update({"color_primaries": "bt2020", "transfer_characteristics": "smpte2084", "color_space": "bt2020nc", "color_range": "tv"})
        elif mode == "BT.2020 HLG":
            info.update({"color_primaries": "bt2020", "transfer_characteristics": "arib-std-b67", "color_space": "bt2020nc", "color_range": "tv"})
        else:
            for k, v in defaults.items(): info.setdefault(k, v)
        return info

    def _compare_encoding_flags(self, process: Any, task_name: str):
        try:
            flags = getattr(process, "sc_encode_flags", None)
            if not flags: return
            subset_keys = ["enc_codec", "enc_pix_fmt", "enc_profile", "enc_color_primaries", "enc_color_trc", "enc_colorspace", "quality_mode", "quality_value"]
            subset = {k: flags.get(k) for k in subset_keys}
            self._color_encode_flags[task_name] = subset
            
            other_name = "HiRes" if task_name == "LowRes" else "LowRes"
            if other_name in self._color_encode_flags:
                other = self._color_encode_flags[other_name]
                diffs = {k: (other.get(k), subset.get(k)) for k in subset_keys if other.get(k) != subset.get(k)}
                if diffs:
                    logger.warning(f"[COLOR_META] Encoding flags differ ({other_name} vs {task_name}): {diffs}")
                else:
                    logger.debug(f"[COLOR_META] Encoding flags match between {other_name} and {task_name}.")
        except Exception: pass

    def _get_max_expected_raw_depth(self, info: Optional[dict]) -> float:
        pix_fmt = str(info.get("pix_fmt", "")).lower() if info else ""
        profile = str(info.get("profile", "")).lower() if info else ""
        
        logger.debug(f"[DEPTH] Detecting bit depth: pix_fmt='{pix_fmt}', profile='{profile}'")
        
        # 1. Check for High-Bit Formats
        if "16" in pix_fmt or "gray16" in pix_fmt:
            return 65535.0
        if "12" in pix_fmt:
            return 4095.0
        if "10" in pix_fmt or "main10" in profile or "gray10" in pix_fmt:
            return 1023.0
            
        # 2. Check for Float
        if "float" in pix_fmt or "f32" in pix_fmt:
            return 1.0
            
        # 3. Default to 8-bit Range (255.0) for everything else (yuv420p, nv12, gray, etc.)
        # This is safer than 1.0 as it prevents "washed out/white" depth maps if detection is slightly off.
        if pix_fmt:
            logger.debug(f"[DEPTH] Defaulting to 8-bit (255.0) for pix_fmt='{pix_fmt}'")
            return 255.0
            
        return 255.0

    def _process_gpu_splatting(
        self,
        stereo_projector: ForwardWarpStereo,
        batch_video_numpy: np.ndarray,
        batch_depth_numpy_float: np.ndarray,
        target_width: int,
        target_height: int,
        max_disp: float,
        zero_disparity_anchor_val: float,
        input_bias: float,
        tv_disp_comp: float = 1.0,
    ) -> List[np.ndarray]:
        """Process GPU splatting on normalized depth maps.
        
        Args:
            batch_depth_numpy_float: Pre-normalized depth in range [0, 1]
        """
        # CRITICAL: Ensure depth matches video resolution before GPU processing
        # batch_video_numpy: [B, H, W, 3]
        # batch_depth_numpy_float: [B, H', W'] - already normalized to [0, 1]
        
        video_h, video_w = batch_video_numpy.shape[1], batch_video_numpy.shape[2]
        
        # Handle depth shape - ensure it's [B, H, W]
        if batch_depth_numpy_float.ndim == 4:
            if batch_depth_numpy_float.shape[-1] == 1:
                batch_depth_numpy_float = batch_depth_numpy_float.squeeze(-1)
            elif batch_depth_numpy_float.shape[-1] == 3:
                batch_depth_numpy_float = batch_depth_numpy_float[..., 0]
        
        # Depth is already resized to video dimensions at the start of the render loop.
        
        # Move to GPU
        source_tensor = torch.from_numpy(batch_video_numpy).permute(0, 3, 1, 2).float().cuda() / 255.0
        depth_tensor = torch.from_numpy(batch_depth_numpy_float).unsqueeze(1).float().cuda()

        from core.splatting.forward_warp import execute_forward_warp
        
        right_eye_raw, occlusion_mask = execute_forward_warp(
            stereo_projector=stereo_projector,
            source_tensor=source_tensor,
            depth_tensor=depth_tensor,
            target_width=target_width,
            max_disp=max_disp,
            zero_disparity_anchor_val=zero_disparity_anchor_val,
            input_bias=input_bias,
            tv_disp_comp=tv_disp_comp,
            debug_task_name="Render",
        )
        
        
        # CPU conversion
        left_cpu = source_tensor.cpu().numpy()
        right_cpu = right_eye_raw.cpu().numpy()
        occl_cpu = occlusion_mask.cpu().numpy()
        depth_cpu = depth_tensor.cpu().numpy()

        results = []
        for j in range(len(batch_video_numpy)):
            results.append({
                "left": np.clip(left_cpu[j].transpose(1, 2, 0), 0.0, 1.0),
                "right": np.clip(right_cpu[j].transpose(1, 2, 0), 0.0, 1.0),
                "occlusion": np.clip(occl_cpu[j].transpose(1, 2, 0), 0.0, 1.0),
                "depth": np.clip(depth_cpu[j].transpose(1, 2, 0), 0.0, 1.0),
            })
        return results


    def _handle_diagnostic_capture(self, batch_results: List[dict], dual_output: bool, task_name: str):
        # In test mode, we usually only have one frame
        if not batch_results:
            return
        res = batch_results[0]
        # For diagnostic captures, we always use the 4-panel grid so the depth map is available
        grid = self._construct_grid(res, dual_output=False)
        self.progress_queue.put(("diagnostic_capture", {"grid": grid, "task_name": task_name}))

    def _write_to_ffmpeg(self, process: Any, batch_results: List[dict], dual_output: bool):
        for res in batch_results:
            grid = self._construct_grid(res, dual_output)
            # Convert to 16-bit and BGR for FFmpeg
            grid_uint16 = (np.clip(grid, 0.0, 1.0) * 65535.0).astype(np.uint16)
            grid_bgr = cv2.cvtColor(grid_uint16, cv2.COLOR_RGB2BGR)
            process.stdin.write(grid_bgr.tobytes())

    def _write_split_to_ffmpeg(self, mask_process: Any, splat_process: Any, batch_results: List[dict]):
        """Write mask + splat as separate full-res files (used by DNxHR split mode)."""
        for res in batch_results:
            occlusion = res["occlusion"]
            right = res["right"]

            if occlusion.ndim == 2 or (occlusion.ndim == 3 and occlusion.shape[-1] == 1):
                occlusion = np.stack([occlusion.squeeze()] * 3, axis=-1)

            occl_u16 = (np.clip(occlusion, 0.0, 1.0) * 65535.0).astype(np.uint16)
            right_u16 = (np.clip(right, 0.0, 1.0) * 65535.0).astype(np.uint16)

            occl_bgr = cv2.cvtColor(occl_u16, cv2.COLOR_RGB2BGR)
            right_bgr = cv2.cvtColor(right_u16, cv2.COLOR_RGB2BGR)

            mask_process.stdin.write(occl_bgr.tobytes())
            splat_process.stdin.write(right_bgr.tobytes())

    def _construct_grid(self, res: dict, dual_output: bool) -> np.ndarray:
        """Construct output grid for encoding.
        
        dual_output=True: [occlusion_mask | right_eye] (2-panel)
        dual_output=False: [left_eye | depth_vis]
                           [occlusion_mask | right_eye] (4-panel)
        
        Returns float32 array in range [0, 1]
        """
        # Convert uint8 back to float for grid assembly
        # Input frames are already float32 in range [0, 1]
        left = res["left"]
        right = res["right"]
        occlusion = res["occlusion"]
        depth = res["depth"]
        
        # Ensure all are 3-channel
        if occlusion.ndim == 2 or (occlusion.ndim == 3 and occlusion.shape[-1] == 1):
            occlusion = np.stack([occlusion.squeeze()] * 3, axis=-1)
        if depth.ndim == 2 or (depth.ndim == 3 and depth.shape[-1] == 1):
            depth = np.stack([depth.squeeze()] * 3, axis=-1)
        
        if dual_output:
            # 2-panel: occlusion on left, warped right eye on right
            return np.concatenate([occlusion, right], axis=1)
        else:
            # 4-panel: top row (left, depth), bottom row (occlusion, right)
            top_row = np.concatenate([left, depth], axis=1)
            bot_row = np.concatenate([occlusion, right], axis=1)
            return np.concatenate([top_row, bot_row], axis=0)

