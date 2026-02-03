"""Batch video processing module.

Handles batch video splatting workflow including multi-resolution output,
sidecar integration, auto-convergence, and move-to-finished functionality.
"""

import gc
import glob
import logging
import os
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from decord import VideoReader, cpu

from dependency.stereocrafter_util import (
    get_video_stream_info,
    release_cuda_memory,
)
from .depth_processing import (
    DEPTH_VIS_TV10_BLACK_NORM,
    DEPTH_VIS_TV10_WHITE_NORM,
    FFmpegDepthPipeReader,
    _infer_depth_bit_depth,
    compute_global_depth_stats,
    load_pre_rendered_depth,
)
from .render_processor import RenderProcessor
from .convergence import ConvergenceEstimatorWrapper
from core.common.video_io import read_video_frames

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Represents a single video processing task configuration.

    Attributes:
        name: Task name (e.g., "Full-Resolution", "Low-Resolution")
        output_subdir: Output subdirectory name
        set_pre_res: Whether to set preprocessing resolution
        target_width: Target output width (-1 for original)
        target_height: Target output height (-1 for original)
        batch_size: Frames per batch
        is_low_res: Whether this is a low-resolution task
    """

    name: str
    output_subdir: str
    set_pre_res: bool
    target_width: int
    target_height: int
    batch_size: int
    is_low_res: bool


@dataclass
class ProcessingSettings:
    """Processing settings for batch operations.

    Attributes:
        input_source_clips: Path to source clips folder or file
        input_depth_maps: Path to depth maps folder or file
        output_splatted: Output directory for splatted videos
        max_disp: Maximum disparity percentage
        process_length: Number of frames to process (-1 for all)
        enable_full_resolution: Whether to output full resolution
        full_res_batch_size: Batch size for full resolution
        enable_low_resolution: Whether to output low resolution
        low_res_width: Width for low resolution output
        low_res_height: Height for low resolution output
        low_res_batch_size: Batch size for low resolution
        dual_output: Whether to output both left and right eyes
        zero_disparity_anchor: Convergence plane value (0.0-1.0)
        enable_global_norm: Whether to enable global normalization
        match_depth_res: Whether to match depth resolution to video
        move_to_finished: Whether to move processed files to finished folder
        output_crf: Quality CRF value (legacy)
        output_crf_full: CRF for full resolution
        output_crf_low: CRF for low resolution
        depth_gamma: Gamma correction for depth
        depth_dilate_size_x: Horizontal dilation size
        depth_dilate_size_y: Vertical dilation size
        depth_blur_size_x: Horizontal blur size
        depth_blur_size_y: Vertical blur size
        depth_dilate_left: Left eye dilation
        depth_blur_left: Left eye blur
        auto_convergence_mode: Auto-convergence mode ("Off", "Average", "Peak", "Hybrid")
        enable_sidecar_gamma: Whether sidecar controls gamma
        enable_sidecar_blur_dilate: Whether sidecar controls blur/dilate
        single_finished_source_folder: Optional finished folder for single-file mode
        single_finished_depth_folder: Optional finished folder for single-file mode
    """

    input_source_clips: str
    input_depth_maps: str
    output_splatted: str
    max_disp: float = 20.0
    process_length: int = -1
    enable_full_resolution: bool = True
    full_res_batch_size: int = 10
    enable_low_resolution: bool = False
    low_res_width: int = 1920
    low_res_height: int = 1080
    low_res_batch_size: int = 50
    dual_output: bool = False
    zero_disparity_anchor: float = 0.5
    enable_global_norm: bool = False
    match_depth_res: bool = True
    move_to_finished: bool = True
    output_crf: int = 23
    output_crf_full: int = 23
    output_crf_low: int = 23
    depth_gamma: float = 1.0
    depth_dilate_size_x: float = 0.0
    depth_dilate_size_y: float = 0.0
    depth_blur_size_x: float = 0.0
    depth_blur_size_y: float = 0.0
    depth_dilate_left: float = 0.0
    depth_blur_left: float = 0.0
    depth_blur_left_mix: float = 0.5
    auto_convergence_mode: str = "Off"
    enable_sidecar_gamma: bool = True
    enable_sidecar_blur_dilate: bool = True
    single_finished_source_folder: Optional[str] = None
    single_finished_depth_folder: Optional[str] = None
    # NEW FIELDS for deeper orchestration
    multi_map: bool = False
    selected_depth_map: str = ""
    color_tags_mode: str = "Auto"
    is_test_mode: bool = False
    test_target_frame_idx: Optional[int] = None
    skip_lowres_preproc: bool = False
    sidecar_ext: str = ".fssidecar"
    sidecar_folder: str = ""
    track_dp_total_true_on_render: bool = False


@dataclass
class BatchSetupResult:
    """Result of batch processing setup.

    Attributes:
        input_videos: List of input video paths
        is_single_file_mode: Whether running in single-file mode
        finished_source_folder: Optional finished folder for source
        finished_depth_folder: Optional finished folder for depth
        error: Optional error message
    """

    input_videos: List[str] = field(default_factory=list)
    is_single_file_mode: bool = False
    finished_source_folder: Optional[str] = None
    finished_depth_folder: Optional[str] = None
    error: Optional[str] = None


class BatchProcessor:
    """Handles batch video splatting workflow.

    Manages the processing queue, worker threads, and progress
    reporting for batch video conversion. Coordinates multiple
    processing tasks (Full/Low resolution) per video.
    """

    def __init__(
        self,
        progress_queue: queue.Queue,
        stop_event: threading.Event,

        sidecar_manager: Optional[Any] = None,
    ):
        """Initialize batch processor.

        Args:
            progress_queue: Queue for progress updates to GUI
            stop_event: Event for cancellation

            sidecar_manager: Optional sidecar manager for loading/saving settings
        """
        self.progress_queue = progress_queue
        self.stop_event = stop_event

        self.sidecar_manager = sidecar_manager
        self.logger = logging.getLogger(__name__)

    def setup_batch_processing(
        self, settings: ProcessingSettings
    ) -> BatchSetupResult:
        """Setup batch processing, validate inputs, and determine mode.

        Handles input path validation, mode determination (single file vs batch),
        and creates necessary 'finished' folders.

        Args:
            settings: Processing settings

        Returns:
            BatchSetupResult with setup information or error
        """
        input_source = settings.input_source_clips
        input_depth = settings.input_depth_maps
        output_dir = settings.output_splatted

        is_source_file = os.path.isfile(input_source)
        is_source_dir = os.path.isdir(input_source)
        is_depth_file = os.path.isfile(input_depth)
        is_depth_dir = os.path.isdir(input_depth)

        result = BatchSetupResult()

        if is_source_file and is_depth_file:
            # Single-file mode
            result.is_single_file_mode = True
            self.logger.debug(
                "==> Running in single file mode. Files will not be moved to 'finished' folders."
            )
            result.input_videos.append(input_source)
            os.makedirs(output_dir, exist_ok=True)

        elif is_source_dir and is_depth_dir:
            # Batch (folder) mode
            self.logger.debug("==> Running in batch (folder) mode.")

            if settings.move_to_finished:
                result.finished_source_folder = os.path.join(input_source, "finished")
                result.finished_depth_folder = os.path.join(input_depth, "finished")
                os.makedirs(result.finished_source_folder, exist_ok=True)
                os.makedirs(result.finished_depth_folder, exist_ok=True)
                self.logger.debug("Finished folders enabled for batch mode.")
            else:
                self.logger.debug(
                    "Finished folders DISABLED by user setting. Files will remain in input folders."
                )

            os.makedirs(output_dir, exist_ok=True)

            # Collect video files
            video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
            for ext in video_extensions:
                result.input_videos.extend(glob.glob(os.path.join(input_source, ext)))
            result.input_videos = sorted(result.input_videos)

        else:
            result.error = (
                "==> Error: Input Source Clips and Input Depth Maps must both be "
                "either files or directories. Skipping processing."
            )
            self.logger.error(result.error)
            return result

        if not result.input_videos:
            result.error = f"No video files found in {input_source}"
            self.logger.error(result.error)

        return result

    def get_defined_tasks(self, settings: ProcessingSettings) -> List[ProcessingTask]:
        """Get list of processing tasks based on settings.

        Args:
            settings: Processing settings

        Returns:
            List of ProcessingTask objects
        """
        tasks = []

        if settings.enable_full_resolution:
            tasks.append(
                ProcessingTask(
                    name="Full-Resolution",
                    output_subdir="hires",
                    set_pre_res=False,
                    target_width=-1,
                    target_height=-1,
                    batch_size=settings.full_res_batch_size,
                    is_low_res=False,
                )
            )

        if settings.enable_low_resolution:
            tasks.append(
                ProcessingTask(
                    name="Low-Resolution",
                    output_subdir="lowres",
                    set_pre_res=True,
                    target_width=settings.low_res_width,
                    target_height=settings.low_res_height,
                    batch_size=settings.low_res_batch_size,
                    is_low_res=True,
                )
            )

        return tasks

    def run_batch_process(
        self,
        settings: ProcessingSettings,
        from_index: int = 0,
        to_index: Optional[int] = None,
        video_list: Optional[List[Dict]] = None,
    ) -> None:
        """Run batch processing loop."""
        try:
            # Setup
            setup_result = self.setup_batch_processing(settings)
            if setup_result.error:
                self.logger.error(setup_result.error)
                return

            input_videos = setup_result.input_videos
            is_single_file = setup_result.is_single_file_mode

            if not input_videos:
                self.logger.error("No input videos found for processing.")
                return

            # Range selection
            if not is_single_file and video_list:
                total = len(video_list)
                start = max(0, min(total, from_index))
                end = max(start + 1, min(total, to_index or total))
                selected = video_list[start:end]
                input_videos = [e.get("source_video") for e in selected if e.get("source_video")]
            elif not is_single_file:
                total = len(input_videos)
                start = max(0, min(total, from_index))
                end = max(start + 1, min(total, to_index or total))
                input_videos = input_videos[start:end]

            if not input_videos:
                self.logger.error("No input videos left to process.")
                return

            tasks = self.get_defined_tasks(settings)
            if not tasks:
                self.logger.error("No processing tasks defined.")
                return

            total_tasks = len(input_videos) * len(tasks)
            self.progress_queue.put(("total", total_tasks))

            # Initialize RenderProcessor
            renderer = RenderProcessor(
                stop_event=self.stop_event,
                progress_queue=self.progress_queue,
            )

            # Process each video
            task_counter = 0
            for vid_path in input_videos:
                if self.stop_event.is_set(): break

                tasks_processed = self._process_single_video_orchestration(
                    video_path=vid_path,
                    settings=settings,
                    renderer=renderer,
                    initial_task_counter=task_counter,
                    is_single_file_mode=is_single_file,
                )
                task_counter += tasks_processed

        except Exception as e:
            self.logger.error(f"Batch processing error: {e}", exc_info=True)
            self.progress_queue.put(("status", f"Error: {e}"))
        finally:
            release_cuda_memory()
            self.progress_queue.put("finished")

    def _process_single_video_orchestration(
        self,
        video_path: str,
        settings: ProcessingSettings,
        renderer: RenderProcessor,
        initial_task_counter: int,
        is_single_file_mode: bool,
    ) -> int:
        """Handles the full processing lifecycle for a single video."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.logger.info(f"==> Processing Video: {video_name}")
        self.progress_queue.put(("update_info", {"filename": video_name}))

        # 1. Resolve Settings (Sidecar + GUI)
        vid_settings = self._get_video_specific_settings(video_path, settings, is_single_file_mode)
        if vid_settings.get("error"):
            self.logger.error(f"Settings error for {video_name}: {vid_settings['error']}")
            return len(self.get_defined_tasks(settings))

        # 2. Auto-Convergence
        conv_val = vid_settings["convergence_plane"]
        if settings.auto_convergence_mode != "Off" and vid_settings["anchor_source"] != "Sidecar":
            conv_val = self._handle_auto_convergence(video_path, vid_settings["actual_depth_map_path"], settings)

        # 3. Tasks Loop
        tasks = self.get_defined_tasks(settings)
        processed_count = 0
        for task in tasks:
            if self.stop_event.is_set(): break
            
            self.progress_queue.put(("status", f"Processing {task.name} for {video_name}"))
            
            # Initialize Readers for this task/resolution
            readers = self._initialize_readers(video_path, vid_settings["actual_depth_map_path"], settings, task)
            if not readers:
                processed_count += 1
                self.progress_queue.put(("processed", initial_task_counter + processed_count))
                continue

            # Run Rendering
            success = renderer.render_video(
                input_video_reader=readers["source"],
                depth_map_reader=readers["depth"],
                total_frames_to_process=readers["total_frames"],
                processed_fps=readers["fps"],
                output_video_path_base=os.path.join(settings.output_splatted, task.output_subdir, f"{video_name}.mp4"),
                target_output_height=readers["target_h"],
                target_output_width=readers["target_w"],
                max_disp=vid_settings["max_disparity_percentage"],
                batch_size=task.batch_size,
                dual_output=settings.dual_output,
                zero_disparity_anchor_val=conv_val,
                video_stream_info=readers["source_info"],
                input_bias=vid_settings["input_bias"],
                assume_raw_input=not vid_settings["enable_global_norm"],
                global_depth_min=vid_settings.get("global_min", 0.0), # Will handle normalization inside renderer if needed
                global_depth_max=vid_settings.get("global_max", 1.0),
                depth_stream_info=readers["depth_info"],
                user_output_crf=settings.output_crf_low if task.is_low_res else settings.output_crf_full,
                is_low_res_task=task.is_low_res,
                depth_gamma=vid_settings["depth_gamma"],
                depth_dilate_size_x=vid_settings["depth_dilate_size_x"],
                depth_dilate_size_y=vid_settings["depth_dilate_size_y"],
                depth_blur_size_x=vid_settings["depth_blur_size_x"],
                depth_blur_size_y=vid_settings["depth_blur_size_y"],
                depth_dilate_left=vid_settings["depth_dilate_left"],
                depth_blur_left=vid_settings["depth_blur_left"],
                depth_blur_left_mix=vid_settings["depth_blur_left_mix"],
                skip_lowres_preproc=settings.skip_lowres_preproc,
                color_tags_mode=settings.color_tags_mode,
                is_test_mode=settings.is_test_mode,
                test_target_frame_idx=settings.test_target_frame_idx,
            )
            processed_count += 1
            # Note: RenderProcessor already puts 'processed' events, but maybe we should synchronize here?
            # renderer.render_video might need to be told the start index.
            # For now, let's assume Rendering handles it and we just return count.

        return len(tasks)

    def _get_video_specific_settings(self, video_path: str, settings: ProcessingSettings, is_single_file_mode: bool) -> dict:
        """Resolve settings for a specific video, merging sidecar and GUI defaults."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        sidecar_path = os.path.join(settings.sidecar_folder, f"{video_name}_depth{settings.sidecar_ext}")
        
        sidecar_data = {}
        if self.sidecar_manager and os.path.exists(sidecar_path):
            sidecar_data = self.sidecar_manager.load_sidecar_data(sidecar_path) or {}

        # Resolve Depth Map Path
        actual_depth_path = self._resolve_depth_path(video_path, settings, sidecar_data, is_single_file_mode)
        if not actual_depth_path:
            return {"error": "Depth map not found"}

        # Merging Logic
        res = {
            "actual_depth_map_path": actual_depth_path,
            "convergence_plane": sidecar_data.get("convergence_plane", settings.zero_disparity_anchor),
            "max_disparity_percentage": sidecar_data.get("max_disparity", settings.max_disp),
            "input_bias": sidecar_data.get("input_bias", 0.0),
            "depth_gamma": sidecar_data.get("gamma", settings.depth_gamma) if settings.enable_sidecar_gamma else settings.depth_gamma,
            "depth_dilate_size_x": sidecar_data.get("depth_dilate_size_x", settings.depth_dilate_size_x) if settings.enable_sidecar_blur_dilate else settings.depth_dilate_size_x,
            "depth_dilate_size_y": sidecar_data.get("depth_dilate_size_y", settings.depth_dilate_size_y) if settings.enable_sidecar_blur_dilate else settings.depth_dilate_size_y,
            "depth_blur_size_x": sidecar_data.get("depth_blur_size_x", settings.depth_blur_size_x) if settings.enable_sidecar_blur_dilate else settings.depth_blur_size_x,
            "depth_blur_size_y": sidecar_data.get("depth_blur_size_y", settings.depth_blur_size_y) if settings.enable_sidecar_blur_dilate else settings.depth_blur_size_y,
            "depth_dilate_left": sidecar_data.get("depth_dilate_left", settings.depth_dilate_left),
            "depth_blur_left": sidecar_data.get("depth_blur_left", settings.depth_blur_left),
            "depth_blur_left_mix": sidecar_data.get("depth_blur_left_mix", settings.depth_blur_left_mix),
            "anchor_source": "Sidecar" if "convergence_plane" in sidecar_data else "GUI",
            "enable_global_norm": settings.enable_global_norm and ("convergence_plane" not in sidecar_data), # Policy
        }
        return res

    def _resolve_depth_path(self, video_path: str, settings: ProcessingSettings, sidecar_data: dict, is_single_file: bool) -> Optional[str]:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        if is_single_file:
            return settings.input_depth_maps if os.path.isfile(settings.input_depth_maps) else None
        
        base_folder = settings.input_depth_maps
        if settings.multi_map:
            # Sidecar check
            selected = sidecar_data.get("selected_depth_map") or settings.selected_depth_map
            if selected:
                candidate = os.path.join(base_folder, selected, f"{video_name}_depth.mp4")
                if os.path.exists(candidate): return candidate
                candidate = os.path.join(base_folder, selected, f"{video_name}_depth.npz")
                if os.path.exists(candidate): return candidate
        
        # Default check
        c_mp4 = os.path.join(base_folder, f"{video_name}_depth.mp4")
        if os.path.exists(c_mp4): return c_mp4
        c_npz = os.path.join(base_folder, f"{video_name}_depth.npz")
        if os.path.exists(c_npz): return c_npz
        return None

    def _handle_auto_convergence(self, video_path: str, depth_path: str, settings: ProcessingSettings) -> float:
        # Placeholder for auto-convergence integration
        # In actual implementation, initialize ConvergenceEstimatorWrapper and call estimate_convergence
        return settings.zero_disparity_anchor

    def _initialize_readers(self, video_path: str, depth_path: str, settings: ProcessingSettings, task: ProcessingTask) -> Optional[dict]:
        try:
            source, fps, orig_h, orig_w, target_h, target_w, info, total = read_video_frames(
                video_path, settings.process_length, set_pre_res=task.set_pre_res,
                pre_res_width=task.target_width, pre_res_height=task.target_height
            )
            
            # Depth reader setup
            depth_target_h, depth_target_w = (orig_h, orig_w) if (task.is_low_res and not settings.skip_lowres_preproc) else (target_h, target_w)
            depth_match = True if task.is_low_res else settings.match_depth_res
            
            d_reader, d_total, d_h, d_w, d_info = load_pre_rendered_depth(
                depth_path, process_length=settings.process_length,
                target_height=depth_target_h, target_width=depth_target_w,
                match_resolution_to_target=depth_match
            )
            
            if total != d_total:
                self.logger.error("Frame count mismatch")
                return None
                
            return {
                "source": source, "depth": d_reader, "fps": fps, "target_h": target_h, "target_w": target_w,
                "source_info": info, "depth_info": d_info, "total_frames": total,
                "orig_h": orig_h, "orig_w": orig_w
            }
        except Exception as e:
            self.logger.error(f"Reader init error: {e}")
            return None

    def validate_settings(self, settings: ProcessingSettings) -> Tuple[bool, str]:
        """Validate processing settings.

        Args:
            settings: Processing settings to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check max disparity
        if settings.max_disp <= 0:
            return False, "Max Disparity must be positive."

        # Check convergence anchor
        if not (0.0 <= settings.zero_disparity_anchor <= 1.0):
            return False, "Zero Disparity Anchor must be between 0.0 and 1.0."

        # Check batch sizes
        if settings.enable_full_resolution and settings.full_res_batch_size <= 0:
            return False, "Full Resolution Batch Size must be positive."

        if settings.enable_low_resolution:
            if settings.low_res_width <= 0 or settings.low_res_height <= 0:
                return False, "Low-Resolution Width and Height must be positive."
            if settings.low_res_batch_size <= 0:
                return False, "Low-Resolution Batch Size must be positive."

        # Check at least one resolution enabled
        if not (settings.enable_full_resolution or settings.enable_low_resolution):
            return False, "At least one resolution (Full or Low) must be enabled."

        # Check gamma
        if settings.depth_gamma <= 0:
            return False, "Depth Gamma must be positive."

        # Check dilate/blur values
        if (
            settings.depth_dilate_size_x < -10.0
            or settings.depth_dilate_size_x > 30.0
            or settings.depth_dilate_size_y < -10.0
            or settings.depth_dilate_size_y > 30.0
        ):
            return False, "Depth Dilate Sizes (X/Y) must be between -10 and 30."

        if settings.depth_blur_size_x < 0 or settings.depth_blur_size_y < 0:
            return False, "Depth Blur Sizes (X/Y) must be non-negative."

        if settings.depth_dilate_left < 0.0 or settings.depth_dilate_left > 20.0:
            return False, "Dilate Left must be between 0 and 20."

        if settings.depth_blur_left < 0 or settings.depth_blur_left > 20:
            return False, "Blur Left must be between 0 and 20."

        return True, ""

    def compute_depth_normalization(
        self,
        depth_map_path: str,
        enable_global_norm: bool,
        total_frames: int,
        batch_size: int,
        actual_width: int,
        actual_height: int,
    ) -> Tuple[bool, float, float, float]:
        """Compute depth normalization parameters.

        Performs global depth stats scan when global normalization is enabled,
        or determines scaling factor for raw input mode.

        Args:
            depth_map_path: Path to depth map video
            enable_global_norm: Whether global normalization is enabled
            total_frames: Total frames in depth video
            batch_size: Processing batch size
            actual_width: Actual depth width
            actual_height: Actual depth height

        Returns:
            Tuple of (assume_raw_mode, global_min, global_max, max_content_value)
        """
        assume_raw_mode = not enable_global_norm
        global_min, global_max = 0.0, 1.0
        max_content_value = 1.0

        # First scan: Get max content value (unconditional)
        raw_reader = None
        try:
            depth_info = get_video_stream_info(depth_map_path)
            bit_depth = _infer_depth_bit_depth(depth_info)
            pix_fmt = str((depth_info or {}).get("pix_fmt", ""))

            if bit_depth > 8:
                raw_reader = FFmpegDepthPipeReader(
                    depth_map_path,
                    out_w=actual_width,
                    out_h=actual_height,
                    bit_depth=bit_depth,
                    num_frames=total_frames,
                    pix_fmt=pix_fmt,
                )
            else:
                raw_reader = VideoReader(
                    depth_map_path, ctx=cpu(0), width=actual_width, height=actual_height
                )

            if len(raw_reader) > 0:
                _, max_content_value = compute_global_depth_stats(
                    depth_map_reader=raw_reader,
                    total_frames=total_frames,
                    chunk_size=batch_size,
                )
                self.logger.debug(f"Max content depth scanned: {max_content_value:.3f}.")
            else:
                self.logger.error("RAW depth reader has no frames for content scan.")
        except Exception as e:
            self.logger.error(f"Failed to scan max content depth: {e}")
        finally:
            if raw_reader:
                if hasattr(raw_reader, "close"):
                    try:
                        raw_reader.close()
                    except Exception:
                        pass
                del raw_reader
                gc.collect()

        # Second scan: Global normalization (if enabled)
        if not assume_raw_mode:
            self.logger.info(
                "==> Global Depth Normalization selected. Starting global depth stats pre-pass."
            )

            raw_reader = None
            try:
                depth_info = get_video_stream_info(depth_map_path)
                bit_depth = _infer_depth_bit_depth(depth_info)
                pix_fmt = str((depth_info or {}).get("pix_fmt", ""))

                if bit_depth > 8:
                    raw_reader = FFmpegDepthPipeReader(
                        depth_map_path,
                        out_w=actual_width,
                        out_h=actual_height,
                        bit_depth=bit_depth,
                        num_frames=total_frames,
                        pix_fmt=pix_fmt,
                    )
                else:
                    raw_reader = VideoReader(
                        depth_map_path,
                        ctx=cpu(0),
                        width=actual_width,
                        height=actual_height,
                    )

                if len(raw_reader) > 0:
                    global_min, global_max = compute_global_depth_stats(
                        depth_map_reader=raw_reader,
                        total_frames=total_frames,
                        chunk_size=batch_size,
                    )
                    self.logger.debug(
                        "Successfully computed global stats from RAW reader."
                    )
                else:
                    self.logger.error("RAW depth reader has no frames.")
            except Exception as e:
                self.logger.error(f"Failed to compute global stats: {e}")
                global_min, global_max = 0.0, 1.0
            finally:
                if raw_reader:
                    if hasattr(raw_reader, "close"):
                        try:
                            raw_reader.close()
                        except Exception:
                            pass
                    del raw_reader
                    gc.collect()
        else:
            self.logger.debug(
                "==> No Normalization (Assume Raw 0-1 Input) selected. Skipping global stats pre-pass."
            )

            # Determine scaling factor for raw input mode
            if max_content_value <= 256.0 and max_content_value > 1.0:
                global_max = 255.0
            elif max_content_value > 256.0 and max_content_value <= 1024.0:
                global_max = 1023.0
            else:
                global_max = 1023.0
                self.logger.warning(
                    f"Max content value is unusual ({max_content_value:.2f}). Using fallback 1023.0."
                )
            global_min = 0.0

        return assume_raw_mode, global_min, global_max, max_content_value

    def check_stop_requested(self) -> bool:
        """Check if processing stop has been requested.

        Returns:
            True if stop event is set
        """
        return self.stop_event.is_set()
