import os
import gc
import numpy as np
import torch
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers.models.transformers.transformer_2d")
import logging # Import standard logging

# Configure a logger for this module
_logger = logging.getLogger(__name__)

from diffusers.training_utils import set_seed
from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

# --- MODIFIED IMPORTS from depthcrafter.utils ---
from depthcrafter.utils import (
    save_video, read_video_frames,
    save_depth_visual_as_mp4_util,
    save_depth_visual_as_png_sequence_util,
    save_depth_visual_as_exr_sequence_util,
    save_depth_visual_as_single_exr_util,
    read_image_sequence_as_frames,
    create_frames_from_single_image,
    format_duration,
    get_segment_output_folder_name,
    get_segment_npz_output_filename,
    get_full_video_output_filename,
    get_sidecar_json_filename,
    save_json_file
)
# --- END MODIFIED IMPORTS ---

try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE_LOGIC = True
except ImportError:
    OPENEXR_AVAILABLE_LOGIC = False
    _logger.warning("OpenEXR/Imath libraries not found. EXR features will be limited/unavailable. Context: depth_crafter_logic.py")


warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers.models.transformers.transformer_2d")

from typing import Optional, Tuple, List, Dict, Union

# --- Global Configuration Flags ---
_ENABLE_XFORMERS_ATTENTION = True # Set to True or False to enable/disable xFormers.

class DepthCrafterDemo:
    def __init__(self, unet_path: str, pre_train_path: str, cpu_offload: str = "model", use_cudnn_benchmark: bool = False, local_files_only: bool = False, disable_xformers=False):
        torch.backends.cudnn.benchmark = use_cudnn_benchmark
        try:
            unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
                unet_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                local_files_only=local_files_only
            )
            self.pipe = DepthCrafterPipeline.from_pretrained(
                pre_train_path,
                unet=unet,
                torch_dtype=torch.float16,
                variant="fp16",
                local_files_only=local_files_only
            )
            if cpu_offload == "sequential":
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                _logger.warning(f"Unknown CPU offload option '{cpu_offload}'. Defaulting to 'model'.")
                self.pipe.enable_model_cpu_offload() # Defaulting
            self.pipe.enable_attention_slicing()
            _logger.debug("DepthCrafterPipeline initialized successfully.")

            # Decide if xFormers should be enabled:
            # It's only enabled if the global flag is True AND the GUI flag (disable_xformers) is False.
            should_enable_xformers = _ENABLE_XFORMERS_ATTENTION and not disable_xformers
            
            if should_enable_xformers:
                try:
                    from diffusers.utils.logging import set_verbosity_info
                    set_verbosity_info() 

                    self.pipe.enable_xformers_memory_efficient_attention()
                    _logger.info("xFormers memory-efficient attention ENABLED (Globally ON, GUI OFF).")
                except ImportError:
                    _logger.warning("xFormers library not found, cannot enable.")
                except Exception as e:
                    _logger.warning(f"Failed to enable xFormers: {e}. Falling back to standard attention.")
                finally:
                    pass 
            else:
                # Even if the global flag was True, if disable_xformers is True, we explicitly disable it
                if disable_xformers:
                    try:
                        # Explicitly call disable_xformers_memory_efficient_attention to ensure standard kernels are used.
                        self.pipe.disable_xformers_memory_efficient_attention()
                        _logger.info("xFormers memory-efficient attention DISABLED by GUI setting (VRAM Save Mode).")
                    except Exception:
                        _logger.debug("Attempt to disable xformers failed, likely already disabled or not present.")
                        pass
                else: # This block handles the case where _ENABLE_XFORMERS_ATTENTION was False
                    _logger.info("xFormers memory-efficient attention disabled by global setting.")

            _logger.info("DepthCrafterPipeline initialized successfully.") # This was already there, ensure it remains.
        except Exception as e:
            _logger.critical(f"CRITICAL: Failed to initialize DepthCrafterPipeline: {e}", exc_info=True)
            raise # Re-raise after logging

    def _setup_paths(self, base_output_folder: str, original_video_basename: str,
                     segment_job_info: Optional[dict]) -> Tuple[str, str, str]:
        actual_save_folder_for_output = base_output_folder
        output_filename_for_meta = ""

        if segment_job_info:
            segment_subfolder_name = get_segment_output_folder_name(original_video_basename)
            actual_save_folder_for_output = os.path.join(base_output_folder, segment_subfolder_name)
            output_filename_for_meta = get_segment_npz_output_filename(
                original_video_basename,
                segment_job_info['segment_id'],
                segment_job_info['total_segments']
            )
        else:
            output_filename_for_meta = get_full_video_output_filename(original_video_basename)

        full_save_path = os.path.join(actual_save_folder_for_output, output_filename_for_meta)
        os.makedirs(actual_save_folder_for_output, exist_ok=True)
        return actual_save_folder_for_output, output_filename_for_meta, full_save_path

    def _initialize_job_metadata(self, guidance_scale: float, num_denoising_steps: int,
                                    user_target_height: int, user_target_width: int, seed_val: int,
                                    target_fps_for_read: float, segment_job_info: Optional[dict],
                                    output_filename_for_meta: str, pipe_call_window_size: int,
                                    pipe_call_overlap: int,
                                    original_video_basename: str) -> dict:
        job_specific_metadata = {
            "original_video_basename": original_video_basename, 
            "guidance_scale": float(guidance_scale),
            "inference_steps": int(num_denoising_steps),
            "target_height_during_process": int(user_target_height),
            "target_width_during_process": int(user_target_width),
            "seed": int(seed_val),
            "target_fps_setting": float(target_fps_for_read),
            "status": "pending",
            "_individual_metadata_path": None
        }

        if segment_job_info:
            job_specific_metadata.update({
                "segment_id": int(segment_job_info["segment_id"]),
                "source_start_frame_raw_index": int(segment_job_info["start_frame_raw_index"]),
                "source_num_frames_raw_for_segment": int(segment_job_info["num_frames_to_load_raw"]),
                "output_segment_filename": output_filename_for_meta,
                "output_segment_format": "npz",
                "segment_definition_window_setting": int(pipe_call_window_size),
                "segment_definition_overlap_setting": int(pipe_call_overlap)
            })
        else:
            job_specific_metadata.update({
                "output_video_filename": output_filename_for_meta,
                "pipeline_window_size_used_for_full_video_pass": int(pipe_call_window_size),
                "pipeline_overlap_used_for_full_video_pass": int(pipe_call_overlap)
            })
        return job_specific_metadata

    def _load_frames(self,
                     video_path_or_job_info: Union[str, dict],
                     frames_array_if_provided: Optional[np.ndarray],
                     process_length_for_read: int,
                     user_target_height: int,
                     user_target_width: int,
                     segment_job_info: Optional[dict],
                     job_specific_metadata: dict
                     ) -> Tuple[Optional[np.ndarray], float, int, int]:
        actual_frames_to_process = None
        actual_fps_for_save = 30.0
        original_h_loaded, original_w_loaded = None, None

        if frames_array_if_provided is not None:
            actual_frames_to_process = frames_array_if_provided
            if segment_job_info and "original_video_fps" in segment_job_info:
                 actual_fps_for_save = segment_job_info["original_video_fps"]
            elif "target_fps_setting" in job_specific_metadata:
                 actual_fps_for_save = job_specific_metadata["target_fps_setting"] if job_specific_metadata["target_fps_setting"] != -1 else 24.0
            else:
                 actual_fps_for_save = 24.0
            _logger.debug(f"Loaded {len(actual_frames_to_process)} frames from numpy array. Using FPS: {actual_fps_for_save:.2f}")
            if actual_frames_to_process.ndim > 0 and len(actual_frames_to_process) > 0:
                 original_h_loaded, original_w_loaded = actual_frames_to_process.shape[1:3]

        elif isinstance(video_path_or_job_info, str):
            video_path_for_read = video_path_or_job_info
            start_frame_idx = 0
            num_frames_to_load_for_seg = -1
            
            if segment_job_info:
                start_frame_idx = segment_job_info["start_frame_raw_index"]
                num_frames_to_load_for_seg = segment_job_info["num_frames_to_load_raw"]
                target_fps_for_video_read = segment_job_info.get("gui_fps_setting_at_definition", -1)
            else:
                target_fps_for_video_read = job_specific_metadata.get("target_fps_setting", -1)

            # --- ADDED: Check for cached ffprobe info and pass it to read_video_frames ---
            cached_ffprobe_info = None
            if segment_job_info and "video_stream_ffprobe_info" in segment_job_info:
                cached_ffprobe_info = segment_job_info["video_stream_ffprobe_info"]
                _logger.debug(f"Reusing cached ffprobe info for {os.path.basename(video_path_for_read)}.")
            # -----------------------------------------------------------------------------

            loaded_frames, fps_from_read, original_h, original_w, processed_h, processed_w, video_stream_info, ffprobe_raw_stdout = read_video_frames(
                video_path_for_read, 
                process_length=process_length_for_read if not segment_job_info else -1,
                target_fps=target_fps_for_video_read,
                target_height=user_target_height, target_width=user_target_width,
                start_frame_index=start_frame_idx, 
                num_frames_to_load=num_frames_to_load_for_seg,
                cached_ffprobe_info=cached_ffprobe_info
            )
            actual_frames_to_process = loaded_frames
            actual_fps_for_save = fps_from_read
            job_specific_metadata["original_height_detected"] = original_h
            job_specific_metadata["original_width_detected"] = original_w
            job_specific_metadata["processed_height"] = processed_h # This is the actual height Decord delivered
            job_specific_metadata["processed_width"] = processed_w  # This is the actual width Decord delivered
            job_specific_metadata["video_stream_metadata"] = video_stream_info # Keep ffprobe info
            job_specific_metadata["ffprobe_raw_stdout"] = ffprobe_raw_stdout
            _logger.debug(f"Loaded {len(actual_frames_to_process) if actual_frames_to_process is not None else 0} frames from video '{video_path_for_read}'. Original FPS for save: {actual_fps_for_save:.2f}")
        
        elif isinstance(video_path_or_job_info, dict):
            source_info = video_path_or_job_info
            source_type = source_info.get("type")
            source_path = source_info.get("path")
            
            if "target_fps_setting" in job_specific_metadata and job_specific_metadata["target_fps_setting"] != -1.0:
                effective_output_fps = job_specific_metadata["target_fps_setting"]
            else: 
                effective_output_fps = 24.0
            
            actual_fps_for_save = effective_output_fps

            if source_type == "image_sequence_folder":
                start_idx_for_segment = 0
                num_img_to_load_for_segment = process_length_for_read

                if segment_job_info:
                    start_idx_for_segment = segment_job_info.get("start_frame_raw_index", 0)
                    num_img_to_load_for_segment = segment_job_info.get("num_frames_to_load_raw", -1)

                frames_this_segment, h, w = read_image_sequence_as_frames(
                    folder_path=source_path,
                    num_frames_to_load=num_img_to_load_for_segment, 
                    target_height=user_target_height,
                    target_width=user_target_width,
                    start_index=start_idx_for_segment
                )
                actual_frames_to_process = frames_this_segment
                original_h_loaded, original_w_loaded = h, w
                _logger.debug(f"Frames Load: Attempting to load image sequence from '{source_path}'. Target FPS: {actual_fps_for_save:.3f}. Segment Start Idx: {start_idx_for_segment}, Num to Load: {num_img_to_load_for_segment}. Loaded: {len(actual_frames_to_process) if actual_frames_to_process is not None else 0} frames.")

            elif source_type == "single_image_file":
                num_frames_gen = segment_job_info["num_frames_to_load_raw"] if segment_job_info else int(round(effective_output_fps))
                
                frames_this_segment, h, w = create_frames_from_single_image(
                    image_path=source_path,
                    num_frames_to_generate=num_frames_gen,
                    target_height=user_target_height,
                    target_width=user_target_width
                )
                actual_frames_to_process = frames_this_segment
                original_h_loaded, original_w_loaded = h, w
                _logger.debug(f"Frames Load: Loaded {len(actual_frames_to_process) if actual_frames_to_process is not None else 0} frames from single image '{source_path}' at {actual_fps_for_save:.1f} FPS (for 1s clip).")
            else:
                job_specific_metadata["status"] = "failure_unknown_source_type_in_dict"
                _logger.error(f"Frames Load: Unknown source type '{source_type}' in input dictionary.")
                return None, 0.0

        else:
            job_specific_metadata["status"] = "failure_no_input_source"
            _logger.error("Cannot load frames: No video path or numpy array provided.")
            return None, 0.0
        
        if original_h_loaded is not None:
            job_specific_metadata["original_height_loaded"] = original_h_loaded
            job_specific_metadata["original_width_loaded"] = original_w_loaded

        return actual_frames_to_process, actual_fps_for_save, job_specific_metadata["processed_height"], job_specific_metadata["processed_width"]

    def _handle_no_frames_failure(self, job_specific_metadata: dict, full_save_path: str,
                                  infer_start_time: float, actual_fps_for_save: float,
                                  segment_job_info: Optional[dict],
                                  save_final_output_json_config_passed_in: bool) -> Tuple[None, dict]:
        video_basename_for_log = job_specific_metadata.get("original_video_basename", "unknown_video")
        _logger.warning(f"No frames to process for {video_basename_for_log}. Skipping.")

        job_specific_metadata["status"] = "failure_no_frames"
        job_specific_metadata["frames_in_output_video"] = 0
        job_specific_metadata["processed_at_fps"] = float(actual_fps_for_save if actual_fps_for_save is not None and actual_fps_for_save > 0 else 0)
        
        infer_duration_sec_noframes = time.perf_counter() - infer_start_time
        job_specific_metadata["internal_processing_duration_seconds"] = round(infer_duration_sec_noframes, 2)
        job_specific_metadata["internal_processing_duration_formatted"] = format_duration(infer_duration_sec_noframes)
        job_specific_metadata["processing_timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        should_save_failure_json = (segment_job_info is not None) or \
                                   (not segment_job_info and save_final_output_json_config_passed_in)
        
        if should_save_failure_json and full_save_path:
            individual_metadata_json_path_noframes = get_sidecar_json_filename(full_save_path)
            if save_json_file(job_specific_metadata, individual_metadata_json_path_noframes):
                job_specific_metadata["_individual_metadata_path"] = os.path.abspath(individual_metadata_json_path_noframes)
                _logger.info(f"Saved failure JSON (no frames): {individual_metadata_json_path_noframes}")
            else:
                job_specific_metadata["_individual_metadata_path"] = None
        else:
            job_specific_metadata["_individual_metadata_path"] = None
        return None, job_specific_metadata

    def _perform_inference(self, actual_frames_to_process: np.ndarray,
                           guidance_scale: float, num_denoising_steps: int,
                           pipe_call_window_size: int, pipe_call_overlap: int,
                           segment_job_info: Optional[dict],
                           actual_processed_height: int, actual_processed_width: int # <--- RENAMED
                           ) -> np.ndarray:
        current_pipe_window_for_call = pipe_call_window_size
        current_pipe_overlap_for_call = pipe_call_overlap
        if segment_job_info: 
            current_pipe_window_for_call = actual_frames_to_process.shape[0]
            current_pipe_overlap_for_call = 0

        _logger.debug(f"Starting inference: Frames: {actual_frames_to_process.shape[0]}, Res: {actual_frames_to_process.shape[1]}x{actual_frames_to_process.shape[2]}, Scale: {guidance_scale}, Steps: {num_denoising_steps}, Win: {current_pipe_window_for_call}, Ovlp: {current_pipe_overlap_for_call}")
        with torch.inference_mode():
            res = self.pipe(
                actual_frames_to_process,
                height=actual_processed_height, # <--- USE RENAMED PARAM
                width=actual_processed_width,   # <--- USE RENAMED PARAM
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=current_pipe_window_for_call,
                overlap=current_pipe_overlap_for_call,
            ).frames[0]
        _logger.debug(f"Inference completed. Result shape: {res.shape}")

        if res.ndim == 4 and res.shape[-1] > 1: 
            res = res.sum(-1) / res.shape[-1]
            _logger.debug(f"Inference result (RGB/RGBA) averaged to grayscale. Final shape: {res.shape}")
        return res

    def _save_segment_npz(self, res: np.ndarray, full_save_path: str, job_specific_metadata: dict) -> bool:
        try:
            np.savez_compressed(full_save_path, frames=res)
            job_specific_metadata["npz_segment_path"] = os.path.abspath(full_save_path)
            _logger.debug(f"Successfully saved: {full_save_path}")
            return True
        except Exception as e_save_npz:
            _logger.error(f"Failed to save: {full_save_path}. Reason: NPZ segment save error: {e_save_npz}")
            job_specific_metadata["status"] = "failure_npz_save"
            return False

    def _save_intermediate_visual_for_segment(self, res_normalized_for_visual: np.ndarray,
                                               actual_save_folder_for_output: str,
                                               output_filename_for_meta: str,
                                               intermediate_visual_format_to_save: str,
                                               actual_fps_for_save: float,
                                               job_specific_metadata: dict):
        base_filename_no_ext_for_visual = os.path.splitext(os.path.basename(output_filename_for_meta))[0]
        
        visual_save_path_or_dir = None
        visual_save_error = None 
        target_fps_for_visual_float = actual_fps_for_save if actual_fps_for_save > 0 else 23.976

        save_func = None
        save_args = []
        save_kwargs = {} 
        
        if intermediate_visual_format_to_save == "mp4":
            mp4_path = os.path.join(actual_save_folder_for_output, f"{base_filename_no_ext_for_visual}_visual.mp4")
            save_func = save_depth_visual_as_mp4_util
            save_args = [res_normalized_for_visual, mp4_path, target_fps_for_visual_float]
            save_kwargs = {"output_format": "mp4"}
        elif intermediate_visual_format_to_save == "main10_mp4":
            mp4_path = os.path.join(actual_save_folder_for_output, f"{base_filename_no_ext_for_visual}_visual.mp4")
            save_func = save_depth_visual_as_mp4_util
            save_args = [res_normalized_for_visual, mp4_path, target_fps_for_visual_float]
            save_kwargs = {"output_format": "main10_mp4"}
        elif intermediate_visual_format_to_save == "png_sequence":
            save_func = save_depth_visual_as_png_sequence_util
            save_args = [res_normalized_for_visual, actual_save_folder_for_output, base_filename_no_ext_for_visual]
        elif intermediate_visual_format_to_save == "exr_sequence":
            if OPENEXR_AVAILABLE_LOGIC:
                save_func = save_depth_visual_as_exr_sequence_util
                save_args = [res_normalized_for_visual, actual_save_folder_for_output, base_filename_no_ext_for_visual]
            else:
                visual_save_error = "OpenEXR libraries not available in logic module."
        elif intermediate_visual_format_to_save == "exr":
            if OPENEXR_AVAILABLE_LOGIC:
                first_frame_to_save = res_normalized_for_visual[0] if len(res_normalized_for_visual) > 0 else None
                if first_frame_to_save is None: 
                    visual_save_error = "No frame data for single EXR."
                else: 
                    save_func = save_depth_visual_as_single_exr_util
                    save_args = [first_frame_to_save, actual_save_folder_for_output, base_filename_no_ext_for_visual]
            else:
                visual_save_error = "OpenEXR libraries not available in logic module."
        elif intermediate_visual_format_to_save == "none":
            pass 
        else:
            visual_save_error = f"Unknown intermediate visual format: {intermediate_visual_format_to_save}"

        if save_func and not visual_save_error:
            visual_save_path_or_dir, visual_save_error = save_func(*save_args, **save_kwargs)

        if visual_save_path_or_dir:
            job_specific_metadata["intermediate_visual_path"] = os.path.abspath(visual_save_path_or_dir)
            job_specific_metadata["intermediate_visual_format_saved"] = intermediate_visual_format_to_save
            _logger.info(f"Saved intermediate segment visual in {intermediate_visual_format_to_save}")
        
        if visual_save_error: 
            job_specific_metadata["intermediate_visual_save_error"] = visual_save_error 
            _logger.error(f"Error saving intermediate segment visual ({intermediate_visual_format_to_save}): {visual_save_error}")

    def _save_full_video_output(self, res: np.ndarray, full_save_path: str,
                                actual_fps_for_save: float, job_specific_metadata: dict) -> bool:
        res_min_full, res_max_full = res.min(), res.max()
        if res_max_full != res_min_full:
            res_normalized_for_mp4 = (res - res_min_full) / (res_max_full - res_min_full)
        else:
            res_normalized_for_mp4 = np.zeros_like(res)
        res_normalized_for_mp4 = np.clip(res_normalized_for_mp4, 0, 1)

        try:
            save_video_fps_full = actual_fps_for_save
            if save_video_fps_full == -1.0:
                _logger.warning(f"Logic Save Video: FPS value is still -1.0 at save point, falling back to 30.0. FPS: {save_video_fps_full}")
                save_video_fps_full = 30.0 
            elif save_video_fps_full <= 0:
                _logger.warning(f"Logic Save Video: FPS value is zero or negative at save point, falling back to 30.0. FPS: {save_video_fps_full}")
                save_video_fps_full = 30.0
            
            output_format_for_full_video = job_specific_metadata.get("preferred_output_format", "mp4")

            save_video(res_normalized_for_mp4, full_save_path, fps=save_video_fps_full, output_format=output_format_for_full_video)
            _logger.debug(f"Successfully saved: {full_save_path}")
            return True
        except Exception as e_save_mp4:
            _logger.error(f"Failed to save: {full_save_path}. Reason: Full video MP4 save error: {e_save_mp4}")
            job_specific_metadata["status"] = "failure_mp4_save"
            return False

    def _finalize_job_metadata_and_save_json(self, job_specific_metadata: dict, infer_start_time: float,
                                           actual_fps_for_save: float, frames_processed_count: int,
                                           saved_output_successfully: bool, full_save_path: Optional[str],
                                           segment_job_info: Optional[dict],
                                           save_final_output_json_config_passed_in: bool):
        if "internal_processing_duration_seconds" not in job_specific_metadata: 
            infer_duration_sec = time.perf_counter() - infer_start_time
            job_specific_metadata["internal_processing_duration_seconds"] = round(infer_duration_sec, 2)
            job_specific_metadata["internal_processing_duration_formatted"] = format_duration(infer_duration_sec)

        job_specific_metadata["processed_at_fps"] = float(actual_fps_for_save)
        job_specific_metadata["frames_in_output_video"] = frames_processed_count
        
        if saved_output_successfully and job_specific_metadata["status"] == "pending":
            job_specific_metadata["status"] = "success"
        elif job_specific_metadata["status"] == "pending": 
            job_specific_metadata["status"] = "failure_at_finalize" 
            
        if "processing_timestamp_utc" not in job_specific_metadata: 
            job_specific_metadata["processing_timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        should_save_this_job_json = (segment_job_info is not None) or \
                                    (not segment_job_info and save_final_output_json_config_passed_in)
        
        if should_save_this_job_json and full_save_path:
            individual_metadata_json_path = get_sidecar_json_filename(full_save_path)
            if save_json_file(job_specific_metadata, individual_metadata_json_path):
                job_specific_metadata["_individual_metadata_path"] = os.path.abspath(individual_metadata_json_path)
            else:
                if job_specific_metadata["status"] == "success": 
                    job_specific_metadata["status"] = "failure_metadata_save" 
                job_specific_metadata["_individual_metadata_path"] = None
        elif job_specific_metadata.get("_individual_metadata_path") is None : 
            job_specific_metadata["_individual_metadata_path"] = None

    def _internal_infer(self,
                        video_path_or_job_info_dict: Union[str, dict],
                        frames_array_if_provided: Optional[np.ndarray],
                        num_denoising_steps: int, guidance_scale: float,
                        base_output_folder: str,
                        user_target_height: int, user_target_width: int,
                        seed_val: int, original_video_basename: str,
                        process_length_for_read: int, gui_target_fps_for_job: float,
                        pipe_call_window_size: int, pipe_call_overlap: int,
                        segment_job_info: Optional[dict] = None,
                        should_save_intermediate_visuals: bool = False,
                        intermediate_visual_format_to_save: str = "none",
                        save_final_output_json_config_passed_in: bool = False
                        ) -> Tuple[Optional[str], dict]:

        infer_start_time = time.perf_counter()
        set_seed(seed_val)
        _logger.debug(f"Starting inference job for: {original_video_basename} (Seed: {seed_val}, Segment: {bool(segment_job_info)}, ID: {segment_job_info.get('segment_id', -1) if segment_job_info else -1})")

        actual_save_folder_for_output, output_filename_for_meta, full_save_path = \
            self._setup_paths(base_output_folder, original_video_basename, segment_job_info)

        job_specific_metadata = self._initialize_job_metadata(
            guidance_scale, num_denoising_steps, user_target_height, user_target_width, seed_val,
            gui_target_fps_for_job,
            segment_job_info, output_filename_for_meta,
            pipe_call_window_size, pipe_call_overlap, original_video_basename
        )

        actual_frames_to_process, actual_fps_for_save, actual_processed_h, actual_processed_w = self._load_frames(
            video_path_or_job_info=video_path_or_job_info_dict,
            frames_array_if_provided=frames_array_if_provided,
            process_length_for_read=process_length_for_read,
            user_target_height=user_target_height,
            user_target_width=user_target_width,
            segment_job_info=segment_job_info,
            job_specific_metadata=job_specific_metadata
        )
        # Update job_specific_metadata with the *actual* processed H/W if not already set (e.g. from np.array input)
        job_specific_metadata["processed_height"] = actual_processed_h
        job_specific_metadata["processed_width"] = actual_processed_w

        if job_specific_metadata["status"] == "failure_no_input_source":
            self._finalize_job_metadata_and_save_json(
                job_specific_metadata, infer_start_time,
                0.0, 0, False, 
                full_save_path, segment_job_info, save_final_output_json_config_passed_in
            )
            return None, job_specific_metadata

        if actual_frames_to_process is None or actual_frames_to_process.shape[0] == 0:
            return self._handle_no_frames_failure(
                job_specific_metadata, full_save_path, infer_start_time,
                actual_fps_for_save if actual_fps_for_save is not None else 0.0,
                segment_job_info, save_final_output_json_config_passed_in
            )

        inference_result = self._perform_inference(
            actual_frames_to_process, guidance_scale, num_denoising_steps,
            pipe_call_window_size, pipe_call_overlap, segment_job_info,
            actual_processed_h, actual_processed_w
            )

        if inference_result is not None and inference_result.ndim >= 3: # Should be (T, H, W)
            job_specific_metadata["processed_height"] = inference_result.shape[1]
            job_specific_metadata["processed_width"] = inference_result.shape[2]
        else:
            _logger.warning(f"Inference result was not valid for dimension extraction. Inference_result shape: {inference_result.shape if inference_result is not None else 'None'}")
            job_specific_metadata["processed_height"] = "N/A"
            job_specific_metadata["processed_width"] = "N/A"

        saved_output_successfully = False
        if segment_job_info:
            saved_output_successfully = self._save_segment_npz(
                inference_result, full_save_path, job_specific_metadata 
            )
            if saved_output_successfully and should_save_intermediate_visuals and \
               intermediate_visual_format_to_save != "none" and inference_result.size > 0:
                
                res_min_seg, res_max_seg = inference_result.min(), inference_result.max()
                if res_max_seg != res_min_seg:
                    res_normalized_for_visual = (inference_result - res_min_seg) / (res_max_seg - res_min_seg)
                else:
                    res_normalized_for_visual = np.zeros_like(inference_result)
                res_normalized_for_visual = np.clip(res_normalized_for_visual, 0, 1)

                self._save_intermediate_visual_for_segment(
                    res_normalized_for_visual, actual_save_folder_for_output, 
                    output_filename_for_meta, 
                    intermediate_visual_format_to_save,
                    actual_fps_for_save, job_specific_metadata
                )
        else: 
            saved_output_successfully = self._save_full_video_output(
                inference_result, full_save_path, actual_fps_for_save, job_specific_metadata 
            )

        self._finalize_job_metadata_and_save_json(
            job_specific_metadata, infer_start_time,
            actual_fps_for_save, actual_frames_to_process.shape[0],
            saved_output_successfully, full_save_path, 
            segment_job_info, save_final_output_json_config_passed_in
        )
        
        _logger.debug(f"Inference job for {original_video_basename} finished. Status: {job_specific_metadata['status']}. Duration: {job_specific_metadata['internal_processing_duration_formatted']}. Output: {full_save_path if saved_output_successfully else 'N/A'}")
        return full_save_path if saved_output_successfully else None, job_specific_metadata
    
    def run(self,
            video_path_or_frames_or_info: Union[str, np.ndarray, dict],
            num_denoising_steps: int, guidance_scale: float,
            base_output_folder: str, gui_window_size: int, gui_overlap: int,
            process_length_for_read_full_video: int, target_height: int, target_width: int, 
            seed: int, original_video_basename_override: Optional[str] = None,
            segment_job_info_param: Optional[dict] = None,
            keep_intermediate_npz_config: bool = False,
            intermediate_segment_visual_format_config: str = "none",
            save_final_json_for_this_job_config: bool = False
            ):
        
        video_path_or_info_for_infer_load: Union[str, dict]
        frames_array_input = None
        original_basename_for_job: str

        current_job_spec: dict
        if segment_job_info_param:
            current_job_spec = segment_job_info_param
            video_path_or_info_for_infer_load = current_job_spec["video_path"]
            if current_job_spec["source_type"] != "video_file":
                 video_path_or_info_for_infer_load = {
                     "type": current_job_spec["source_type"],
                     "path": current_job_spec["video_path"],
                     "gui_fps": current_job_spec["gui_fps_setting_at_definition"]
                 }
            original_basename_for_job = current_job_spec["original_basename"]
        elif isinstance(video_path_or_frames_or_info, dict):
            current_job_spec = video_path_or_frames_or_info
            video_path_or_info_for_infer_load = current_job_spec["video_path"]
            if current_job_spec["source_type"] != "video_file":
                 video_path_or_info_for_infer_load = {
                     "type": current_job_spec["source_type"],
                     "path": current_job_spec["video_path"],
                     "gui_fps": current_job_spec["gui_fps_setting_at_definition"]
                 }
            original_basename_for_job = current_job_spec["original_basename"]
        elif isinstance(video_path_or_frames_or_info, str):
            current_job_spec = {}
            video_path_or_info_for_infer_load = video_path_or_frames_or_info
            original_basename_for_job = original_video_basename_override if original_video_basename_override else os.path.splitext(os.path.basename(video_path_or_frames_or_info))[0]
            _logger.warning(f"DepthCrafterDemo.run received a direct video path '{video_path_or_frames_or_info}' without a full job specification dictionary. Assuming default FPS of -1 (original). It is recommended to pass a job dictionary for full metadata tracking.")
        elif isinstance(video_path_or_frames_or_info, np.ndarray):
            current_job_spec = {}
            frames_array_input = video_path_or_frames_or_info
            video_path_or_info_for_infer_load = None
            if not original_video_basename_override:
                _logger.error("DepthCrafterDemo.run: original_video_basename_override is required for np.ndarray input.")
                raise ValueError("original_video_basename_override needed for np.ndarray input.")
            original_basename_for_job = original_video_basename_override
        else:
            _logger.error(f"DepthCrafterDemo.run: video_path_or_frames must be str or np.ndarray or dict, got {type(video_path_or_frames_or_info).__name__}.")
            raise ValueError("video_path_or_frames_or_info invalid.")


        gui_fps_setting_for_job = current_job_spec.get("gui_fps_setting_at_definition", -1.0)
        if gui_fps_setting_for_job == -1.0 and isinstance(video_path_or_info_for_infer_load, dict):
            gui_fps_setting_for_job = video_path_or_info_for_infer_load.get("gui_fps", -1.0)
        if gui_fps_setting_for_job == -1.0 and not frames_array_input and video_path_or_info_for_infer_load is None:
             _logger.warning(f"Run Logic: Target FPS setting missing in job spec for {original_basename_for_job} and frames/video path not directly available to infer. Defaulting to -1 (original).")


        should_save_visuals_for_infer = False
        intermediate_visual_fmt_for_infer = "none"
        if segment_job_info_param and keep_intermediate_npz_config:
            should_save_visuals_for_infer = True
            intermediate_visual_fmt_for_infer = intermediate_segment_visual_format_config
        
        effective_process_length_for_infer = current_job_spec.get("num_frames_to_load_raw") \
                                             if segment_job_info_param else process_length_for_read_full_video


        save_path, job_metadata_dict = self._internal_infer(
            video_path_or_job_info_dict=video_path_or_info_for_infer_load,
            frames_array_if_provided=frames_array_input,
            num_denoising_steps=num_denoising_steps, guidance_scale=guidance_scale,
            base_output_folder=base_output_folder,
            user_target_width=target_width, user_target_height=target_height, 
            seed_val=seed,
            original_video_basename=original_basename_for_job,
            process_length_for_read=effective_process_length_for_infer,
            gui_target_fps_for_job=gui_fps_setting_for_job,
            pipe_call_window_size=gui_window_size, pipe_call_overlap=gui_overlap,
            segment_job_info=segment_job_info_param,
            should_save_intermediate_visuals=should_save_visuals_for_infer,
            intermediate_visual_format_to_save=intermediate_visual_fmt_for_infer,
            save_final_output_json_config_passed_in=save_final_json_for_this_job_config
        )
        gc.collect(); torch.cuda.empty_cache()
        return save_path, job_metadata_dict