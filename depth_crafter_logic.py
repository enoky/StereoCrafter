import os
import gc
import numpy as np
import torch
import time # For perf_counter, strftime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers.models.transformers.transformer_2d")

from diffusers.training_utils import set_seed
from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

# Import from the new message catalog
from message_catalog import (
    log_message,
    INFO, DEBUG, WARNING, ERROR, CRITICAL
)

# --- MODIFIED IMPORTS from depthcrafter.utils ---
from depthcrafter.utils import (
    save_video, read_video_frames,
    save_depth_visual_as_mp4_util,
    save_depth_visual_as_png_sequence_util,
    save_depth_visual_as_exr_sequence_util,
    save_depth_visual_as_single_exr_util,
    read_image_sequence_as_frames, # New
    create_frames_from_single_image, # New
    format_duration,
    get_segment_output_folder_name,
    get_segment_npz_output_filename,
    get_full_video_output_filename,
    get_sidecar_json_filename,
    save_json_file # Now uses global log_message
)
# --- END MODIFIED IMPORTS ---

try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE_LOGIC = True
except ImportError:
    OPENEXR_AVAILABLE_LOGIC = False
    log_message("OPENEXR_UNAVAILABLE", context="depth_crafter_logic.py")


warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers.models.transformers.transformer_2d")

from typing import Optional, Tuple, List, Dict, Union

class DepthCrafterDemo:
    def __init__(self, unet_path: str, pre_train_path: str, cpu_offload: str = "model", use_cudnn_benchmark: bool = True):
        torch.backends.cudnn.benchmark = use_cudnn_benchmark
        try:
            unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
                unet_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
            self.pipe = DepthCrafterPipeline.from_pretrained(
                pre_train_path,
                unet=unet,
                torch_dtype=torch.float16,
                variant="fp16",
            )
            if cpu_offload == "sequential":
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                log_message("MODEL_CPU_OFFLOAD_UNKNOWN_WARN", option=cpu_offload) # New ID
                self.pipe.enable_model_cpu_offload() # Defaulting
            self.pipe.enable_attention_slicing()
            log_message("MODEL_INIT_SUCCESS", component="DepthCrafterPipeline") # Updated existing ID
        except Exception as e:
            log_message("MODEL_INIT_FAILURE", component="DepthCrafterPipeline", reason=str(e)) # Updated existing ID
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
                                    user_max_res_for_read: int, seed_val: int,
                                    target_fps_for_read: float, segment_job_info: Optional[dict],
                                    output_filename_for_meta: str, pipe_call_window_size: int,
                                    pipe_call_overlap: int,
                                    original_video_basename: str) -> dict:
        job_specific_metadata = {
            "original_video_basename": original_video_basename, 
            "guidance_scale": float(guidance_scale),
            "inference_steps": int(num_denoising_steps),
            "max_res_during_process": int(user_max_res_for_read),
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
                     video_path_or_job_info: Union[str, dict], # Can be video path str or dict for image/seq
                     frames_array_if_provided: Optional[np.ndarray],
                     process_length_for_read: int, # Max frames overall (for full video or sequence)
                     # target_fps_for_read: float, # This is the GUI FPS setting, now part of job_info
                     user_max_res_for_read: int,
                     segment_job_info: Optional[dict], # If processing a segment of any source type
                     job_specific_metadata: dict # To store H, W if loaded
                     ) -> Tuple[Optional[np.ndarray], float]: # frames, actual_fps_for_save
        actual_frames_to_process = None
        actual_fps_for_save = 30.0 # Default
        original_h_loaded, original_w_loaded = None, None

        if frames_array_if_provided is not None:
            actual_frames_to_process = frames_array_if_provided
            # FPS for pre-loaded array: use segment_job_info's original_video_fps if available,
            # or fall back to a sensible default (e.g., 30, or from job_specific_metadata if set there)
            if segment_job_info and "original_video_fps" in segment_job_info:
                 actual_fps_for_save = segment_job_info["original_video_fps"]
            elif "target_fps_setting" in job_specific_metadata: # This should be the GUI setting
                 actual_fps_for_save = job_specific_metadata["target_fps_setting"] if job_specific_metadata["target_fps_setting"] != -1 else 24.0
            else:
                 actual_fps_for_save = 24.0 # Default if nothing else
            log_message("FRAMES_LOAD_FROM_ARRAY_INFO", num_frames=len(actual_frames_to_process), fps=actual_fps_for_save)
            if actual_frames_to_process.ndim > 0 and len(actual_frames_to_process) > 0:
                 original_h_loaded, original_w_loaded = actual_frames_to_process.shape[1:3]

        elif isinstance(video_path_or_job_info, str): # Assumed to be a video file path
            video_path_for_read = video_path_or_job_info
            start_frame_idx = 0
            num_frames_to_load_for_seg = -1 # Means load up to 'process_length_for_read' or end of video
            
            # If it's a segment of a video file
            if segment_job_info:
                start_frame_idx = segment_job_info["start_frame_raw_index"]
                num_frames_to_load_for_seg = segment_job_info["num_frames_to_load_raw"]
                # Use the FPS from the segment_job_info, which came from define_video_segments
                # This 'target_fps_for_read' for read_video_frames should be the GUI setting
                # to achieve desired stride.
                target_fps_for_video_read = segment_job_info.get("gui_fps_setting_at_definition", -1)
            else: # Full video processing
                # target_fps_for_read is the GUI setting from job_specific_metadata
                target_fps_for_video_read = job_specific_metadata.get("target_fps_setting", -1)


            loaded_frames, fps_from_read, h, w = read_video_frames( # read_video_frames now returns H, W
                video_path_for_read, 
                process_length=process_length_for_read if not segment_job_info else -1, # For full video, apply process_length. For segment, num_frames_to_load_for_seg controls it.
                target_fps=target_fps_for_video_read, # This is the GUI setting
                max_res=user_max_res_for_read, 
                dataset="open", # Hardcoded in original
                start_frame_index=start_frame_idx, 
                num_frames_to_load=num_frames_to_load_for_seg
            )
            actual_frames_to_process = loaded_frames
            actual_fps_for_save = fps_from_read # read_video_frames returns the actual FPS it will be saved at
            original_h_loaded, original_w_loaded = h, w
            log_message("FRAMES_LOAD_FROM_VIDEO_INFO", video_path=video_path_for_read, num_frames=len(actual_frames_to_process) if actual_frames_to_process is not None else 0, fps=actual_fps_for_save)
        
        elif isinstance(video_path_or_job_info, dict): # Image sequence or single image
            source_info = video_path_or_job_info
            source_type = source_info.get("type")
            source_path = source_info.get("path") # This is the folder path for sequence
            # ... (effective_output_fps, actual_fps_for_save logic) ...
            
            # # === DEBUG PRINT ===
            # print(f"DEBUG _load_frames (dict input): source_type='{source_type}'")
            # print(f"DEBUG _load_frames: job_specific_metadata['target_fps_setting'] = {job_specific_metadata.get('target_fps_setting')}")
            # === END DEBUG PRINT ===

            if "target_fps_setting" in job_specific_metadata and job_specific_metadata["target_fps_setting"] != -1.0:
                effective_output_fps = job_specific_metadata["target_fps_setting"]
                # print(f"DEBUG _load_frames: Using target_fps_setting: {effective_output_fps}") # DEBUG
            else: 
                effective_output_fps = 24.0 # Default for image sequences/single images if not specified
                # print(f"DEBUG _load_frames: Using default FPS 24.0 (target_fps_setting was {job_specific_metadata.get('target_fps_setting')})") # DEBUG
            
            actual_fps_for_save = effective_output_fps 
            # print(f"DEBUG _load_frames: actual_fps_for_save set to: {actual_fps_for_save}") # DEBUG

            if source_type == "image_sequence_folder":
                # Get start_index and num_frames_to_load_raw from segment_job_info
                start_idx_for_segment = 0 # Default for full sequence
                num_img_to_load_for_segment = process_length_for_read # Default for full sequence (process_length_for_read from GUI)

                if segment_job_info: # If this is a segment of an image sequence
                    start_idx_for_segment = segment_job_info.get("start_frame_raw_index", 0)
                    num_img_to_load_for_segment = segment_job_info.get("num_frames_to_load_raw", -1) # -1 means all from start

                frames_this_segment, h, w = read_image_sequence_as_frames(
                    folder_path=source_path,
                    num_frames_to_load=num_img_to_load_for_segment, 
                    max_res=user_max_res_for_read,
                    start_index=start_idx_for_segment # PASS THE START INDEX
                )
                actual_frames_to_process = frames_this_segment
                original_h_loaded, original_w_loaded = h, w
                log_message("FRAMES_LOAD_FROM_IMG_SEQ_INFO", folder_path=source_path, 
                            num_frames=len(actual_frames_to_process) if actual_frames_to_process is not None else 0, 
                            fps=actual_fps_for_save, 
                            start_index=start_idx_for_segment,
                            num_loaded=num_img_to_load_for_segment)

            elif source_type == "single_image_file":
                num_frames_for_1s_clip = int(round(effective_output_fps))
                # If segment_job_info exists for a single_image, it means it's a segment *of the 1s clip*.
                # num_frames_to_generate should be what define_video_segments calculated for this segment.
                num_frames_gen = segment_job_info["num_frames_to_load_raw"] if segment_job_info else num_frames_for_1s_clip
                
                frames_this_segment, h, w = create_frames_from_single_image(
                    image_path=source_path,
                    num_frames_to_generate=num_frames_gen,
                    max_res=user_max_res_for_read
                )
                actual_frames_to_process = frames_this_segment
                original_h_loaded, original_w_loaded = h, w
                log_message("FRAMES_LOAD_FROM_SINGLE_IMG_INFO", image_path=source_path, num_frames=len(actual_frames_to_process) if actual_frames_to_process is not None else 0, fps=actual_fps_for_save)
            else:
                job_specific_metadata["status"] = "failure_unknown_source_type_in_dict"
                log_message("FRAMES_LOAD_UNKNOWN_SOURCE_DICT_ERROR", type=source_type)
                return None, 0.0

        else:
            job_specific_metadata["status"] = "failure_no_input_source"
            log_message("FRAMES_LOAD_NO_SOURCE_ERROR")
            return None, 0.0
        
        # Store original dimensions in job_specific_metadata if available
        if original_h_loaded is not None:
            job_specific_metadata["original_height_loaded"] = original_h_loaded
            job_specific_metadata["original_width_loaded"] = original_w_loaded

        return actual_frames_to_process, actual_fps_for_save

    def _handle_no_frames_failure(self, job_specific_metadata: dict, full_save_path: str,
                                  infer_start_time: float, actual_fps_for_save: float,
                                  segment_job_info: Optional[dict],
                                  save_final_output_json_config_passed_in: bool) -> Tuple[None, dict]:
        video_basename_for_log = job_specific_metadata.get("original_video_basename", "unknown_video")
        log_message("PROCESSING_NO_FRAMES", item_name=video_basename_for_log) # Using existing ID

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
            if save_json_file(job_specific_metadata, individual_metadata_json_path_noframes): # log_func removed
                job_specific_metadata["_individual_metadata_path"] = os.path.abspath(individual_metadata_json_path_noframes)
                log_message("METADATA_SAVE_NOFRAMES_JSON_SUCCESS", filepath=individual_metadata_json_path_noframes) # New ID
            else:
                job_specific_metadata["_individual_metadata_path"] = None
                # save_json_file now logs its own errors via log_message
        else:
            job_specific_metadata["_individual_metadata_path"] = None
        return None, job_specific_metadata

    def _perform_inference(self, actual_frames_to_process: np.ndarray,
                           guidance_scale: float, num_denoising_steps: int,
                           pipe_call_window_size: int, pipe_call_overlap: int,
                           segment_job_info: Optional[dict]) -> np.ndarray:
        current_pipe_window_for_call = pipe_call_window_size
        current_pipe_overlap_for_call = pipe_call_overlap
        if segment_job_info: 
            current_pipe_window_for_call = actual_frames_to_process.shape[0]
            current_pipe_overlap_for_call = 0

        log_message("INFERENCE_START", num_frames=actual_frames_to_process.shape[0], 
                    height=actual_frames_to_process.shape[1], width=actual_frames_to_process.shape[2],
                    guidance=guidance_scale, steps=num_denoising_steps, 
                    window=current_pipe_window_for_call, overlap=current_pipe_overlap_for_call) # New ID
        with torch.inference_mode():
            res = self.pipe(
                actual_frames_to_process,
                height=actual_frames_to_process.shape[1],
                width=actual_frames_to_process.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=current_pipe_window_for_call,
                overlap=current_pipe_overlap_for_call,
            ).frames[0]
        log_message("INFERENCE_COMPLETE", result_shape=res.shape) # New ID

        if res.ndim == 4 and res.shape[-1] > 1: 
            res = res.sum(-1) / res.shape[-1]
            log_message("INFERENCE_CHANNEL_AVERAGED", final_shape=res.shape) # New ID
        return res

    def _save_segment_npz(self, res: np.ndarray, full_save_path: str, job_specific_metadata: dict) -> bool:
        try:
            np.savez_compressed(full_save_path, frames=res)
            job_specific_metadata["npz_segment_path"] = os.path.abspath(full_save_path)
            log_message("FILE_SAVE_SUCCESS", filepath=full_save_path) # Using existing ID
            return True
        except Exception as e_save_npz:
            log_message("FILE_SAVE_FAILURE", filepath=full_save_path, reason=f"NPZ segment save error: {e_save_npz}") # Using existing ID
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
            save_kwargs = {"output_format": "mp4"} # Explicitly standard mp4
        elif intermediate_visual_format_to_save == "main10_mp4": # HEVC 10-bit in MP4
            mp4_path = os.path.join(actual_save_folder_for_output, f"{base_filename_no_ext_for_visual}_visual.mp4") # Still .mp4 extension
            save_func = save_depth_visual_as_mp4_util
            save_args = [res_normalized_for_visual, mp4_path, target_fps_for_visual_float]
            save_kwargs = {"output_format": "main10_mp4"}
        elif intermediate_visual_format_to_save == "png_sequence":
            # ... (same as before)
            save_func = save_depth_visual_as_png_sequence_util
            save_args = [res_normalized_for_visual, actual_save_folder_for_output, base_filename_no_ext_for_visual]
        elif intermediate_visual_format_to_save == "exr_sequence":
            # ... (same as before, with OPENEXR_AVAILABLE_LOGIC check)
            if OPENEXR_AVAILABLE_LOGIC:
                save_func = save_depth_visual_as_exr_sequence_util
                save_args = [res_normalized_for_visual, actual_save_folder_for_output, base_filename_no_ext_for_visual]
            else:
                visual_save_error = "OpenEXR libraries not available in logic module."
        elif intermediate_visual_format_to_save == "exr":
            # ... (same as before, with OPENEXR_AVAILABLE_LOGIC check)
            if OPENEXR_AVAILABLE_LOGIC:
                first_frame_to_save = res_normalized_for_visual[0] if len(res_normalized_for_visual) > 0 else None
                if first_frame_to_save is None: 
                    visual_save_error = "Cannot save single EXR from empty or invalid visual data."
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
            log_filename_or_dirname = os.path.basename(visual_save_path_or_dir)
            log_message("VISUAL_SAVE_SEGMENT_SUCCESS", format=intermediate_visual_format_to_save, name=log_filename_or_dirname) # New ID
        
        if visual_save_error: 
            job_specific_metadata["intermediate_visual_save_error"] = visual_save_error 
            log_message("VISUAL_SAVE_SEGMENT_ERROR", format=intermediate_visual_format_to_save, error=visual_save_error) # New ID

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
            if save_video_fps_full == -1.0: # Should be resolved to actual FPS by _load_frames
                # This case means original FPS was requested and should be in actual_fps_for_save
                # If it's still -1.0 here, it's an issue. For safety:
                log_message("LOGIC_SAVE_FPS_STILL_NEGATIVE_WARN", fps_val=save_video_fps_full)
                save_video_fps_full = 30.0 
            elif save_video_fps_full <= 0:
                log_message("LOGIC_SAVE_FPS_ZERO_OR_NEGATIVE_WARN", fps_val=save_video_fps_full)
                save_video_fps_full = 30.0 # Fallback to float
            
            output_format_for_full_video = job_specific_metadata.get("preferred_output_format", "mp4")

            save_video(res_normalized_for_mp4, full_save_path, fps=save_video_fps_full, output_format=output_format_for_full_video)
            log_message("FILE_SAVE_SUCCESS", filepath=full_save_path)
            return True
        except Exception as e_save_mp4:
            log_message("FILE_SAVE_FAILURE", filepath=full_save_path, reason=f"Full video MP4 save error: {e_save_mp4}") # Using existing ID
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
            if save_json_file(job_specific_metadata, individual_metadata_json_path): # log_func removed
                job_specific_metadata["_individual_metadata_path"] = os.path.abspath(individual_metadata_json_path)
                # log_message itself will log success from save_json_file
            else:
                if job_specific_metadata["status"] == "success": 
                    job_specific_metadata["status"] = "failure_metadata_save" 
                job_specific_metadata["_individual_metadata_path"] = None
                # save_json_file logs its own errors
        elif job_specific_metadata.get("_individual_metadata_path") is None : 
            job_specific_metadata["_individual_metadata_path"] = None

    def _internal_infer(self,
                        video_path_or_job_info_dict: Union[str, dict], # video path string, or dict for img/seq
                        frames_array_if_provided: Optional[np.ndarray],
                        num_denoising_steps: int, guidance_scale: float,
                        base_output_folder: str, user_max_res_for_read: int,
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
        log_message("INFERENCE_JOB_START", basename=original_video_basename, seed=seed_val, 
                    is_segment=bool(segment_job_info), segment_id=segment_job_info.get('segment_id', -1) if segment_job_info else -1) # New ID

        actual_save_folder_for_output, output_filename_for_meta, full_save_path = \
            self._setup_paths(base_output_folder, original_video_basename, segment_job_info)

        job_specific_metadata = self._initialize_job_metadata(
            guidance_scale, num_denoising_steps, user_max_res_for_read, seed_val,
            gui_target_fps_for_job, # Pass the explicit GUI FPS setting for this job
            segment_job_info, output_filename_for_meta,
            pipe_call_window_size, pipe_call_overlap, original_video_basename
        )

        actual_frames_to_process, actual_fps_for_save = self._load_frames(
            video_path_or_job_info=video_path_or_job_info_dict,
            frames_array_if_provided=frames_array_if_provided,
            process_length_for_read=process_length_for_read,
            user_max_res_for_read=user_max_res_for_read,
            segment_job_info=segment_job_info,
            job_specific_metadata=job_specific_metadata # Contains target_fps_setting from init
        )

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
            pipe_call_window_size, pipe_call_overlap, segment_job_info
        )

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
        
        log_message("INFERENCE_JOB_COMPLETE", basename=original_video_basename, 
                    status=job_specific_metadata["status"],
                    duration_fmt=job_specific_metadata["internal_processing_duration_formatted"],
                    output_path=full_save_path if saved_output_successfully else "N/A") # New ID
        return full_save_path if saved_output_successfully else None, job_specific_metadata
    
    def run(self,
            video_path_or_frames_or_info: Union[str, np.ndarray, dict], # This is the job dict from GUI or segment_job
            num_denoising_steps: int, guidance_scale: float,
            base_output_folder: str, gui_window_size: int, gui_overlap: int,
            process_length_for_read_full_video: int, max_res: int, seed: int,
            original_video_basename_override: Optional[str] = None,
            segment_job_info_param: Optional[dict] = None, # This IS the full job info if it's a segment
            keep_intermediate_npz_config: bool = False,
            intermediate_segment_visual_format_config: str = "none",
            save_final_json_for_this_job_config: bool = False
            ):
        
        video_path_or_info_for_infer_load: Union[str, dict]
        frames_array_input = None
        original_basename_for_job: str

        current_job_spec: dict
        if segment_job_info_param: # This is a segment processing job
            current_job_spec = segment_job_info_param
            video_path_or_info_for_infer_load = current_job_spec["video_path"] # Path string
            if current_job_spec["source_type"] != "video_file":
                 video_path_or_info_for_infer_load = { # Make dict for image/seq for _load_frames
                     "type": current_job_spec["source_type"],
                     "path": current_job_spec["video_path"],
                     "gui_fps": current_job_spec["gui_fps_setting_at_definition"]
                 }
            original_basename_for_job = current_job_spec["original_basename"]
        elif isinstance(video_path_or_frames_or_info, dict): # Full processing job (not a segment)
            current_job_spec = video_path_or_frames_or_info
            video_path_or_info_for_infer_load = current_job_spec["video_path"]
            if current_job_spec["source_type"] != "video_file":
                 video_path_or_info_for_infer_load = {
                     "type": current_job_spec["source_type"],
                     "path": current_job_spec["video_path"],
                     "gui_fps": current_job_spec["gui_fps_setting_at_definition"]
                 }
            original_basename_for_job = current_job_spec["original_basename"]
        elif isinstance(video_path_or_frames_or_info, str): # Legacy: direct video path for full processing
            # This path should ideally not be hit if GUI always wraps full jobs in dicts.
            current_job_spec = {} # Minimal spec, rely on other params
            video_path_or_info_for_infer_load = video_path_or_frames_or_info
            original_basename_for_job = original_video_basename_override if original_video_basename_override else os.path.splitext(os.path.basename(video_path_or_frames_or_info))[0]
            # Need to get gui_fps_setting for this case if it's a video file.
            # This indicates a design flaw if this path is common. Assume GUI sets up dict.
            log_message("LOGIC_RUN_LEGACY_PATH_WARN", path=video_path_or_frames_or_info)
        elif isinstance(video_path_or_frames_or_info, np.ndarray):
            current_job_spec = {} # Minimal spec
            frames_array_input = video_path_or_frames_or_info
            video_path_or_info_for_infer_load = None # Frames are provided directly
            if not original_video_basename_override:
                log_message("RUN_MISSING_BASENAME_ERROR")
                raise ValueError("original_video_basename_override needed for np.ndarray input.")
            original_basename_for_job = original_video_basename_override
        else:
            log_message("RUN_INVALID_INPUT_TYPE_ERROR", type=type(video_path_or_frames_or_info).__name__)
            raise ValueError("video_path_or_frames_or_info invalid.")


        # Determine the GUI FPS setting for this job
        gui_fps_setting_for_job = current_job_spec.get("gui_fps_setting_at_definition", -1.0)
        if gui_fps_setting_for_job == -1.0 and isinstance(video_path_or_info_for_infer_load, dict): # Check dict for image/seq
            gui_fps_setting_for_job = video_path_or_info_for_infer_load.get("gui_fps", -1.0)
        if gui_fps_setting_for_job == -1.0 and not frames_array_input and video_path_or_info_for_infer_load is None: # np.array or direct video path without info
             # This is tricky for direct np.array or un-meta-ed video path.
             # Assume GUI sets gui_fps_setting_at_definition in current_job_spec for all scenarios.
             log_message("LOGIC_RUN_TARGET_FPS_MISSING_WARN", job_basename=original_basename_for_job)


        should_save_visuals_for_infer = False
        intermediate_visual_fmt_for_infer = "none"
        if segment_job_info_param and keep_intermediate_npz_config: # Use segment_job_info_param here
            should_save_visuals_for_infer = True
            intermediate_visual_fmt_for_infer = intermediate_segment_visual_format_config
        
        effective_process_length_for_infer = current_job_spec.get("num_frames_to_load_raw") \
                                             if segment_job_info_param else process_length_for_read_full_video


        save_path, job_metadata_dict = self._internal_infer(
            video_path_or_job_info_dict=video_path_or_info_for_infer_load,
            frames_array_if_provided=frames_array_input,
            num_denoising_steps=num_denoising_steps, guidance_scale=guidance_scale,
            base_output_folder=base_output_folder,
            user_max_res_for_read=max_res, seed_val=seed,
            original_video_basename=original_basename_for_job,
            process_length_for_read=effective_process_length_for_infer,
            gui_target_fps_for_job=gui_fps_setting_for_job, # PASS THE DETERMINED GUI FPS SETTING
            pipe_call_window_size=gui_window_size, pipe_call_overlap=gui_overlap,
            segment_job_info=segment_job_info_param, # This is the actual segment job info dict from GUI
            should_save_intermediate_visuals=should_save_visuals_for_infer,
            intermediate_visual_format_to_save=intermediate_visual_fmt_for_infer,
            save_final_output_json_config_passed_in=save_final_json_for_this_job_config
        )
        gc.collect(); torch.cuda.empty_cache()
        return save_path, job_metadata_dict