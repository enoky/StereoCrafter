import sys
import time # For default timestamping if needed

# --- Severity Levels ---
CRITICAL = 50
ERROR = 40
WARNING = 30
INFO = 20
DETAIL = 15
DEBUG = 10
NOTSET = 0

# --- Verbosity Levels (map to minimum severity to log) ---
# Higher verbosity number means more restrictive (less output)
VERBOSITY_LEVEL_DEBUG = DEBUG         # Log everything: DEBUG, INFO, WARNING, ERROR, CRITICAL
VERBOSITY_LEVEL_DETAIL = DETAIL 
VERBOSITY_LEVEL_INFO = INFO           # Log: INFO, WARNING, ERROR, CRITICAL (Normal operation)
VERBOSITY_LEVEL_WARNING = WARNING     # Log: WARNING, ERROR, CRITICAL (Quiet)
VERBOSITY_LEVEL_ERROR = ERROR         # Log: ERROR, CRITICAL
VERBOSITY_LEVEL_CRITICAL = CRITICAL   # Log: Only CRITICAL
VERBOSITY_LEVEL_SILENT = 100          # Log nothing (practically)

# --- Global State ---
_GUI_LOGGER_CALLBACK = None
_CONSOLE_VERBOSITY_LEVEL = VERBOSITY_LEVEL_INFO  # Default console verbosity
_GUI_VERBOSITY_LEVEL = VERBOSITY_LEVEL_INFO      # Default GUI verbosity
_USE_TIMESTAMPS_CONSOLE = True
_USE_TIMESTAMPS_GUI = True # GUI might add its own, but this can prefix the message string itself

# --- Configuration Functions ---
def set_gui_logger_callback(logger_func: callable):
    """Sets the callback function for sending messages to the GUI."""
    global _GUI_LOGGER_CALLBACK
    _GUI_LOGGER_CALLBACK = logger_func

def set_console_verbosity(level: int):
    """Sets the verbosity level for console output."""
    global _CONSOLE_VERBOSITY_LEVEL
    _CONSOLE_VERBOSITY_LEVEL = level

def set_gui_verbosity(level: int):
    """Sets the verbosity level for GUI output."""
    global _GUI_VERBOSITY_LEVEL
    _GUI_VERBOSITY_LEVEL = level

def configure_timestamps(console: bool, gui: bool):
    """Configure whether timestamps are prepended to messages for console/GUI."""
    global _USE_TIMESTAMPS_CONSOLE, _USE_TIMESTAMPS_GUI
    _USE_TIMESTAMPS_CONSOLE = console
    _USE_TIMESTAMPS_GUI = gui

def get_formatted_timestamp_mc() -> str: # mc for message_catalog to avoid name clash
    """Generates a timestamp string in HH:MM:SS.s format."""
    current_time_val = time.time()
    time_struct = time.localtime(current_time_val)
    milliseconds_tenths = int((current_time_val - int(current_time_val)) * 10)
    return f"{time_struct.tm_hour:02d}:{time_struct.tm_min:02d}:{time_struct.tm_sec:02d}.{milliseconds_tenths}"

# --- Message Catalog ---
MESSAGES = {
    "OPENEXR_UNAVAILABLE": { # Already existed, ensure context can be passed
        "template": "OpenEXR/Imath libraries not found. EXR features will be limited/unavailable. Context: {context}",
        "level": WARNING,
    },
    "SEGMENT_DEFINE_PROGRESS": {
        "template": "Defining segments for {video_name} (up to {output_frames} output frames from {raw_frames} raw frames).",
        "level": DEBUG, # Or INFO if you want this by default
    },
    "UTIL_NORMALIZE_EMPTY_VIDEO_ERROR": {
        "template": "CRITICAL: Cannot normalize empty video array.",
        "level": CRITICAL,
    },
    "UTIL_NORMALIZE_VIDEO_START": {
        "template": "Normalizing video data. Shape: {shape}",
        "level": DEBUG,
    },
    "UTIL_NORMALIZE_PERCENTILE_FALLBACK": {
        "template": "Normalization: Array too small for robust percentile ({low}%/{high}%), using absolute min/max.",
        "level": DEBUG, # Or WARNING if this is a concern
    },
    # "UTIL_NORMALIZE_VIDEO" already exists, check template arguments
    "UTIL_NORMALIZE_FLAT_VIDEO_WARN": {
        "template": "Normalization: Range very small. Video appears flat.",
        "level": WARNING,
    },
    "UTIL_NORMALIZE_FLAT_VIDEO_RESULT": {
        "template": "Normalization: Video normalized to constant value: {value:.2f}",
        "level": DEBUG,
    },
    "UTIL_NORMALIZE_FINAL_RANGE": {
        "template": "Normalization: Global min/max after final clip: {min_val:.4f} / {max_val:.4f}",
        "level": DEBUG,
    },
    "UTIL_GAMMA_CORRECTION_SKIPPED": {
        "template": "Gamma value {gamma_val:.2f} (effectively 1.0), no gamma transform applied.",
        "level": DEBUG,
    },
    "UTIL_DITHERING_START": {
        "template": "Applying dithering...",
        "level": DEBUG,
    },
    # "UTIL_DITHERING" already exists, check template arguments

    "VIDEO_READ_METADATA_ERROR": {
        "template": "Error reading video metadata for {video_path}: {error}",
        "level": ERROR,
    },
    "VIDEO_UNKNOWN_DATASET_WARN": {
        "template": "Unknown dataset '{dataset_name}'. Using 'open' logic for resolution.",
        "level": WARNING,
    },
    "VIDEO_READER_INIT_ERROR": {
        "template": "Error initializing VideoReader with target dimensions for {video_path}: {error}",
        "level": ERROR,
    },
    "VIDEO_SEGMENT_EMPTY_WARN": {
        "template": "Segment for {video_path} from frame {start_index} for {num_frames} frames is empty or invalid.",
        "level": WARNING,
    },
    "VIDEO_NO_SOURCE_INDICES_WARN": {
        "template": "No source indices for segment of {video_path}.",
        "level": WARNING,
    },
    # Optional debug for frame indices:
    # "VIDEO_FRAME_INDICES_DEBUG": {
    # "template": "Video: {video_path}, OrigIdx: {source_start}-{source_end}, Stride: {stride}, TargetFPS: {target_fps}, Reading: {num_to_read} frames (FinalIdx: {final_start}...{final_end})",
    # "level": DEBUG
    # },
    "VIDEO_NO_FRAMES_TO_READ_WARN": {
        "template": "No frames to read for segment of {video_path} after all filters.",
        "level": WARNING,
    },
    "VIDEO_GET_BATCH_ERROR": {
        "template": "Error in get_batch for {video_path} with indices: {error}", # Kwargs changed
        "level": ERROR,
    },
    "VIDEO_SAVE_UNSUPPORTED_DTYPE_ERROR": {
        "template": "Unsupported numpy array dtype in list for video saving: {dtype}",
        "level": ERROR,
    },
    "VIDEO_SAVE_EMPTY_FRAMES_WARN": {
        "template": "Empty list of frames provided. Cannot save video.",
        "level": WARNING,
    },
    "VIDEO_SAVE_INVALID_FRAMES_TYPE_ERROR": {
        "template": "video_frames must be a list/array of np.ndarray or a list of PIL.Image.Image for saving.",
        "level": ERROR,
    },
    "VIDEO_SAVE_MEDIAPY_ERROR": {
        "template": "Error writing video to {filepath} using mediapy: {error}",
        "level": ERROR,
    },
    "VIDEO_SAVE_HEVC_MAIN10_MP4_SELECTED": {
        "template": "Attempting to save HEVC Main10 (libx265) to: {filepath} with parameters: {kwargs}",
        "level": DETAIL,
    },
    "VIDEO_SAVE_HEVC_MAIN10_MP4_SELECTED": {
        "template": "Attempting to save HEVC Main10 (libx265) to: {filepath} with mediapy_kwargs: {kwargs}, ffmpeg_args: {ffmpeg_args}",
        "level": DETAIL,
    },
    "VIDEO_SAVE_HEVC_MP4_EXTENSION_WARN": {
        "template": "Saving HEVC to an .mp4 container, but the output path '{filepath}' does not end with .mp4. This might lead to issues if an explicit format isn't forced.",
        "level": WARNING, # As per your log
    },
    "COLORMAP_MPL_IMPORT_ERROR": {
        "template": "Matplotlib.cm not found. ColorMapper will use a basic grayscale fallback.",
        "level": WARNING,
    },
    "COLORMAP_INVALID_INPUT_DIMS_ERROR": {
        "template": "ColorMapper.apply: Image must be 2D or 3D, got {ndim}D",
        "level": ERROR,
    },
    "VIS_SEQ_INVALID_INPUT_DIMS_ERROR": {
        "template": "vis_sequence_depth: Input depths must be a 3D array (T, H, W), got {ndim}D",
        "level": ERROR,
    },
    "VIDEO_SAVE_MP4_UTIL_ERROR": {
        "template": "Error saving MP4 visual to {filepath}: {error}",
        "level": ERROR,
    },
    "IMAGE_SAVE_PNG_SEQ_UTIL_ERROR": {
        "template": "Error saving PNG sequence to {dir_path}: {error}",
        "level": ERROR,
    },
    "IMAGE_SAVE_EXR_SEQ_UTIL_ERROR": { # Keep generic or make more specific if needed
        "template": "Error saving EXR sequence (path: {dir_path if dir_path else 'unknown'}): {error}",
        "level": ERROR,
    },
    "IMAGE_SAVE_SINGLE_EXR_UTIL_ERROR": { # Keep generic or make more specific if needed
        "template": "Error saving single EXR (path: {filepath if filepath else 'unknown'}): {error}",
        "level": ERROR,
    },
    "MODEL_CPU_OFFLOAD_UNKNOWN_WARN": {
        "template": "Unknown CPU offload option '{option}'. Defaulting to 'model'.",
        "level": WARNING,
    },
    # "MODEL_INIT_SUCCESS" & "MODEL_INIT_FAILURE" likely exist, ensure they can take a 'component' kwarg
    "MODEL_INIT_SUCCESS": { # Example update
        "template": "{component} initialized successfully.",
        "level": INFO,
    },
    "MODEL_INIT_FAILURE": { # Example update
        "template": "CRITICAL: Failed to initialize {component}: {reason}",
        "level": CRITICAL,
    },
    "FRAMES_LOAD_FROM_ARRAY_INFO": {
        "template": "Loaded {num_frames} frames from numpy array. Using FPS: {fps:.2f}",
        "level": DEBUG, # Or INFO
    },
    "FRAMES_LOAD_FROM_VIDEO_INFO": {
        "template": "Loaded {num_frames} frames from video '{video_path}'. Original FPS for save: {fps:.2f}",
        "level": DEBUG, # Or INFO
    },
    "FRAMES_LOAD_NO_SOURCE_ERROR": {
        "template": "Cannot load frames: No video path or numpy array provided.",
        "level": ERROR,
    },
    "METADATA_SAVE_NOFRAMES_JSON_SUCCESS": { # More specific than generic FILE_SAVE_SUCCESS
        "template": "Saved failure JSON (no frames): {filepath}",
        "level": INFO,
    },
    "INFERENCE_START": {
        "template": "Starting inference: Frames: {num_frames}, Res: {height}x{width}, Scale: {guidance}, Steps: {steps}, Win: {window}, Ovlp: {overlap}",
        "level": DETAIL, # Or DEBUG if too verbose
    },
    "INFERENCE_COMPLETE": {
        "template": "Inference completed. Result shape: {result_shape}",
        "level": DEBUG,
    },
    "INFERENCE_CHANNEL_AVERAGED": {
        "template": "Inference result (RGB/RGBA) averaged to grayscale. Final shape: {final_shape}",
        "level": DEBUG,
    },
    "VISUAL_SAVE_SEGMENT_SUCCESS": {
        "template": "Saved intermediate segment visual in {format}",
        "level": INFO,
    },
    "VISUAL_SAVE_SEGMENT_ERROR": {
        "template": "Error saving intermediate segment visual ({format}): {error}",
        "level": ERROR,
    },
    "INFERENCE_JOB_START": {
        "template": "Starting inference job for: {basename} (Seed: {seed}, Segment: {is_segment}, ID: {segment_id})",
        "level": DETAIL,
    },
    "INFERENCE_JOB_COMPLETE": {
        "template": "Inference job for {basename} finished. Status: {status}. Duration: {duration_fmt}. Output: {output_path}",
        "level": DETAIL,
    },
    "RUN_MISSING_BASENAME_ERROR": {
        "template": "DepthCrafterDemo.run: original_video_basename_override is required for np.ndarray input.",
        "level": ERROR,
    },
    "RUN_INVALID_INPUT_TYPE_ERROR": {
        "template": "DepthCrafterDemo.run: video_path_or_frames must be str or np.ndarray, got {type}.",
        "level": ERROR,
    },
    # OPENEXR_UNAVAILABLE: (ensure context can be passed) "OpenEXR/Imath libraries not found. EXR features will be limited/unavailable. Context: {context}"
    "MERGE_SAVE_EXR_NO_LIB_ERROR": {
        "template": "OpenEXR/Imath libraries not found by save_single_frame_exr. Cannot save EXR.",
        "level": ERROR, # Critical for the operation
    },
    "MERGE_SAVE_EXR_INVALID_DIMS_ERROR": {
        "template": "Frame data for EXR must be 2D (H, W). Got shape: {shape}",
        "level": ERROR,
    },
    "MERGE_SAVE_HEVC_MAIN10_MP4_SUCCESS": {
        "template": "Succesfully merged HEVC10 MP4",
        "level": INFO,
    },
    "MERGE_METADATA_LOAD_ATTEMPT": {
        "template": "Loading merge metadata from: {path}",
        "level": DETAIL,
    },
    "MERGE_METADATA_NOT_SEGMENTED_ERROR": {
        "template": "CRITICAL: 'processed_as_segments' is not true in metadata: {path}. Aborting merge.",
        "level": CRITICAL,
    },
    "MERGE_METADATA_NO_OVERLAP_ERROR": {
        "template": "CRITICAL: 'segment_definition_output_overlap_frames' not found in metadata: {path}. Aborting merge.",
        "level": CRITICAL,
    },
    "MERGE_METADATA_OVERLAP_INFO": {
        "template": "Defined overlap frames (N_overlap) from metadata: {overlap}",
        "level": DEBUG, # Was INFO
    },
    "MERGE_METADATA_NO_JOBS_WARN": {
        "template": "Warning: No job segments found in metadata: {path}.",
        "level": WARNING,
    },
    "MERGE_METADATA_NO_SUCCESSFUL_JOBS_ERROR": {
        "template": "CRITICAL: No successful segments found in metadata to merge: {path}.",
        "level": CRITICAL,
    },
    "MERGE_METADATA_SUCCESSFUL_JOBS_COUNT": {
        "template": "Found {count} successful segments to process from metadata.",
        "level": INFO,
    },
    "MERGE_SEGMENT_UNSUPPORTED_FORMAT_ERROR": {
        "template": "CRITICAL: Unsupported segment format '{format}' for {filename}. Expecting NPZ.",
        "level": CRITICAL,
    },
    "MERGE_SEGMENT_LOAD_NPZ_ERROR": {
        "template": "CRITICAL: Could not load frames from NPZ {filepath}: {error}",
        "level": CRITICAL,
    },
    "MERGE_SEGMENT_EMPTY_ERROR": {
        "template": "CRITICAL: Segment {filename} is empty.",
        "level": CRITICAL,
    },
    "MERGE_SINGLE_SEGMENT_LOAD_INFO": { # This is for when only one segment is processed
        "template": "Single segment {filename} loaded. Shape: {shape}, FPS: {fps:.2f}",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_PASS_LOAD_SEGMENTS_START": {
        "template": "\n--- Pass 1: Loading Segments ---",
        "level": DETAIL, # Was SILENCE_PASSES
    },
    "MERGE_SEGMENT_MISSING_FPS_ERROR": {
        "template": "CRITICAL: 'processed_at_fps' missing for segment ID {segment_id}.",
        "level": CRITICAL,
    },
    "MERGE_SEGMENT_INCONSISTENT_FPS_WARN": {
        "template": "Warning: Inconsistent FPS. Using {expected_fps:.2f}. Segment {segment_id} has {actual_fps:.2f}.",
        "level": WARNING, # Was SILENCE_PASSES
    },
    "MERGE_SEGMENT_LOADING_PROGRESS": {
        "template": "Loading segment {segment_id} ({current}/{total}): {filename}",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_SEGMENT_DATA_EMPTY_WARN": {
        "template": "Warning: Segment {segment_id} ({filename}) data is empty. Skipping.",
        "level": WARNING, # Was SILENCE_PASSES
    },
    "MERGE_SEGMENT_LOADED_FRAMES_INFO": {
        "template": "  Loaded {num_frames} frames. Shape: {shape}",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_NO_VALID_SEGMENTS_LOADED_ERROR": {
        "template": "CRITICAL: No valid segments loaded after filtering/loading.",
        "level": CRITICAL,
    },
    "MERGE_PASS_ALIGN_SEGMENTS_START": {
        "template": "\n--- Pass 1.5: Aligning Segments ---",
        "level": DETAIL, # Was SILENCE_PASSES
    },
    "MERGE_ALIGN_BASELINE_INFO": {
        "template": "Segment 0 (ID {segment_id}) is baseline for alignment.",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_ALIGN_PROGRESS": {
        "template": "Aligning segment (ID {current_id}) to previous (ID {prev_id}). Method: {method}",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    # MERGE_ALIGN_INFO (exists, ensure template matches)
    "MERGE_ALIGN_LINEAR_BLEND_SKIP_S_S": {
        "template": "  Linear Blend: No explicit S&S alignment for segment ID {segment_id}. Blending will occur in stitching.",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_ALIGN_UNKNOWN_METHOD_WARN": {
        "template": "CRITICAL WARNING: Unknown alignment method '{method}'. No alignment performed on segment ID {segment_id}.",
        "level": WARNING, # Was CRITICAL + SILENCE_PASSES
    },
    # MERGE_NO_OVERLAP_ALIGN (exists, ensure template matches)
    "MERGE_ALIGN_NO_OVERLAP_SPECIFIED": {
        "template": "  N_overlap is 0. No explicit alignment for segment ID {segment_id}.",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_STITCH_NO_ALIGNED_SEGMENTS_ERROR": {
        "template": "CRITICAL: No aligned segments for stitching.",
        "level": CRITICAL,
    },
    "MERGE_PASS_STITCH_BLEND_START": {
        "template": "\n--- Pass 2: Stitching and Blending Segments ---",
        "level": DETAIL, # Was SILENCE_PASSES
    },
    "MERGE_STITCH_CONCATENATING_INFO": {
        "template": "  N_overlap is 0. Concatenating segments.",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    # MERGE_STITCH_INFO (exists, ensure template matches)
    "MERGE_STITCH_PROGRESS": {
        "template": "Stitching segment {current_segment_idx} (ID {segment_id}; {current_segment_idx_plus_1}/{total_segments})", # Add current_segment_idx_plus_1
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_STITCH_SEGMENT_EMPTY_WARN": {
        "template": "  Segment {segment_idx} (ID {segment_id}) is empty after alignment. Skipping.",
        "level": WARNING, # Was SILENCE_PASSES
    },
    "MERGE_STITCH_FIRST_SEGMENT_INFO": {
        "template": "  First segment: adding {count} non-overlapping frames.",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_STITCH_NO_BLEND_FRAMES_WARN": {
        "template": "  Warning: No frames for blending between {prev_id} and {current_id}. Hard cut implies adding all of current.",
        "level": WARNING, # Was SILENCE_PASSES
    },
    "MERGE_STITCH_BLENDING_INFO": {
        "template": "  Blending {count} frames with previous (ID {prev_id}).",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_STITCH_ADDING_REMAINDER": {
        "template": "    Adding {count} {description} frames after blend.",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_STITCH_NO_FINAL_FRAMES_ERROR": {
        "template": "CRITICAL: No frames in final list after stitching.",
        "level": CRITICAL,
    },
    "MERGE_POSTPROC_GAMMA_DISABLED": {
        "template": "Gamma correction disabled for MP4 output.",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_OUTPUT_PATH_SEQ_RESOLVED": {
        "template": "  Sequence output resolved to: {path}",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_OUTPUT_PATH_FILE_RESOLVED": {
        "template": "  Single file output resolved to: {path}",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_OUTPUT_PATH_FILE_OVERWRITE_WARN": {
        "template": "  Output file {path} exists and will be overwritten.",
        "level": WARNING, # Was SILENCE_PASSES
    },
    "MERGE_OUTPUT_PATH_AUTOGEN_OVERWRITE_WARN": {
        "template": "  Auto-generated output file {path} exists, will overwrite.",
        "level": WARNING, # Was SILENCE_PASSES
    },
    "MERGE_OUTPUT_PATH_AUTOGEN_INFO": {
        "template": "  Auto-generated output path: {path}",
        "level": DEBUG,
    },
    "MERGE_OUTPUT_PATH_DETERMINE_ERROR": {
        "template": "Could not determine a valid output path.",
        "level": CRITICAL,
    },
    "MERGE_SAVING_TO_DISK_START": {
        "template": "Saving merged output to: {path} (Format: {format}) FPS: {fps:.2f}",
        "level": DETAIL, # Was is_final_output
    },
    "MERGE_SAVE_EMPTY_DATA_ERROR": {
        "template": "CRITICAL: Video data for saving is empty.",
        "level": CRITICAL,
    },
    "MERGE_SAVE_PNG_SEQ_SUCCESS": {
        "template": "Successfully saved {count} PNGs in {path}",
        "level": INFO, # Was is_final_output
    },
    "MERGE_SAVE_EXR_FRAME_ERROR": {
        "template": "  ERROR saving EXR frame {frame_num}: {error}. Skipping.",
        "level": ERROR, # Was CRITICAL
    },
    "MERGE_SAVE_EXR_SEQ_SUMMARY": {
        "template": "Saved {saved}/{total} EXRs in {path}",
        "level": INFO, # Was is_final_output
    },
    "MERGE_SAVE_EXR_SEQ_FAILURES_WARN": {
        "template": "Warning: {failed_count} EXR frames failed to save.",
        "level": WARNING, # Was CRITICAL + is_final_output
    },
    "MERGE_SAVE_SINGLE_EXR_SUCCESS": {
        "template": "Saved first frame as single EXR: {path}",
        "level": INFO, # Was is_final_output
    },
    "MERGE_SAVE_SINGLE_EXR_NO_FRAMES_WARN": {
        "template": "No frames available to save as single EXR.",
        "level": WARNING, # Was CRITICAL
    },
    "MERGE_SAVE_MP4_SUCCESS": {
        "template": "Successfully saved MP4: {path}",
        "level": INFO, # Was is_final_output
    },
    "MERGE_SAVE_UNKNOWN_FORMAT_ERROR": {
        "template": "Unknown output format for saving: {format}",
        "level": ERROR,
    },
    "MERGE_SAVE_DISK_ERROR_CRITICAL": {
        "template": "CRITICAL ERROR during final disk save: {error}", # Optionally log traceback
        "level": CRITICAL,
    },
    "MERGE_PROCESS_START": {
        "template": "Starting depth segment merging process... Format: {format}, Alignment: {alignment}",
        "level": INFO,
    },
    "MERGE_PROCESS_SETTINGS_DITHER": {
        "template": "Merge Settings - Dithering: {enabled}, Strength: {strength}",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_PROCESS_SETTINGS_GAMMA": {
        "template": "Merge Settings - Gamma: {enabled}, Value: {value}",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_PROCESS_SETTINGS_NORM": {
        "template": "Merge Settings - Percentile Norm: {enabled}, Low: {low}%, High: {high}%",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_PROCESSING_SINGLE_SEGMENT_INFO": {
        "template": "Processing as single segment (only one valid segment found).",
        "level": INFO,
    },
    "MERGE_PROCESSING_MULTIPLE_SEGMENTS_INFO": {
        "template": "Processing {count} segments.",
        "level": INFO,
    },
    "MERGE_ONE_VALID_SEGMENT_REMAINED_INFO": {
        "template": "Only one valid segment remained after loading. Using its frames directly.",
        "level": DEBUG, # Was SILENCE_PASSES
    },
    "MERGE_NO_VALID_SEGMENTS_FOR_STITCH_ERROR": {
        "template": "CRITICAL: No valid segments to process after loading stage (should have been caught earlier).", # Already exists, ensure consistent
        "level": CRITICAL,
    },
    "MERGE_FINAL_VIDEO_EMPTY_PRE_NORM_ERROR": {
        "template": "CRITICAL: Resulting video array is empty before normalization.",
        "level": CRITICAL,
    },
    "MERGE_INVALID_FPS_WARN": {
        "template": "Warning: Invalid FPS {fps}. Defaulting to 30.0.",
        "level": WARNING, # Was CRITICAL
    },
    "MERGE_FINAL_VIDEO_STATS_INFO": {
        "template": "Final video array for saving: Shape {shape}, Dtype {dtype}, Min {min_val:.4f}, Max {max_val:.4f}",
        "level": DEBUG,
    },
    "MERGE_PROCESS_CRITICAL_ERROR": { # This is for the main try-except in merge_depth_segments function
        "template": "CRITICAL ERROR during merge process: {error}. Traceback: {traceback}",
        "level": CRITICAL,
    },
    "MERGE_PROCESS_FINISHED_SUCCESS": {
        "template": "Depth segment merging process finished.",
        "level": INFO, # Was is_final_output
    },"GUI_INIT_COMPLETE": {
        "template": "DepthCrafter GUI initialized successfully.",
        "level": INFO,
    },
    "GUI_HELP_LOAD_FAIL_WARN": {
        "template": "Warning: Could not load help content from {filename}. Help icons will be disabled or show 'not found'.",
        "level": WARNING,
    },
    "GUI_SETTINGS_GET_VAL_WARN": {
        "template": "Warning: Could not get value for setting '{setting_key}'. Skipping.",
        "level": WARNING,
    },
    "GUI_SETTINGS_SET_VAL_WARN": {
        "template": "Warning: Could not set value for setting '{setting_key}' to '{value}'. Skipping.",
        "level": WARNING,
    },
    "GUI_SETTINGS_UNKNOWN_KEY_WARN": {
        "template": "Warning: Unknown setting '{setting_key}' found in settings file. Ignoring.",
        "level": WARNING,
    },
    "GUI_ACTION_CANCELLED_BY_USER": {
        "template": "{action} cancelled by user.",
        "level": INFO,
    },
    "GUI_PROCESSING_BATCH_START": {
        "template": "Starting batch processing for {num_jobs} items...",
        "level": INFO,
    },
    "GUI_CANCEL_REQUEST_HONORED": {
        "template": "Processing cancelled by user after current item.",
        "level": INFO,
    },
    "GUI_PROCESSING_MISSING_TOTAL_SEGS_WARN": {
        "template": "Warning: 'total_segments' missing for segment job of {basename}. Defaulting to 1 for master_meta init.",
        "level": WARNING,
    },
    "GUI_PROCESSING_LOADING_PREEXISTING_SEGS": {
        "template": "Loading {count} pre-existing successful segment metadata entries for {basename} into current run's master data.",
        "level": DETAIL,
    },
    "MODEL_RELEASE_SUCCESS": {
        "template": "{component} model components released.",
        "level": DETAIL,
    },
    "MODEL_RELEASE_ERROR": {
        "template": "Error during {component} model cleanup: {error}",
        "level": WARNING,
    },
    "GUI_PROCESSING_BATCH_COMPLETE": {
        "template": "All processing jobs complete!",
        "level": INFO,
    },
    "GUI_JOB_NO_METADATA_WARN": {
        "template": "Warning: No job-specific metadata returned from run for {basename}.",
        "level": WARNING,
    },
    "GUI_JOB_STATUS_REPORT": {
        "template": "  Job for {basename} ({job_prefix}) status: {status}",
        "level": DETAIL, # Or WARNING if status is a failure
    },
    "GUI_JOB_EXCEPTION": {
        "template": "  Exception during job for {basename} ({job_prefix}): {error}. Traceback: {traceback_info}",
        "level": ERROR,
    },
    "PROCESSING_VIDEO_FINAL_STATUS": { # More specific than generic PROCESSING_COMPLETE
        "template": "Finished processing for {basename}. Overall Status: {status}.",
        "level": INFO,
    },
    "GUI_MERGE_SKIPPED_DUE_TO_STATUS": {
        "template": "Skipping merge for {basename} (status: {status}, meta_saved: {meta_saved}). Segments remain in {segment_folder}",
        "level": DETAIL,
    },
    "GUI_ORIGINAL_VIDEO_MOVE_SKIPPED": {
        "template": "Skipped moving original '{basename}' (MOVE_ORIGINAL_TO_FINISHED_FOLDER_ON_COMPLETION is False).",
        "level": DETAIL,
    },
    "GUI_ORIGINAL_VIDEO_MOVE_ATTEMPT": {
        "template": "Attempting to move original '{basename}' to 'finished' subfolder.",
        "level": DETAIL,
    },
    "GUI_ORIGINAL_VIDEO_MOVE_RENAMED": {
        "template": "  Original video destination '{old_name}' existed. Moving as '{new_name}'.",
        "level": DETAIL,
    },
    "GUI_ORIGINAL_VIDEO_MOVE_SUCCESS": {
        "template": "  Successfully moved original '{filename}' to finished folder.",
        "level": DETAIL,
    },
    "GUI_ORIGINAL_VIDEO_MOVE_SOURCE_NOT_FOUND": {
        "template": "  Source video not found for move (moved/deleted?): {path}",
        "level": WARNING,
    },
    "GUI_ORIGINAL_VIDEO_MOVE_ERROR": {
        "template": "  ERROR moving original '{basename}': {error}. Traceback: {traceback_info}",
        "level": ERROR,
    },
    "GUI_MASTER_META_SAVED": { # More specific than generic FILE_SAVE_SUCCESS
        "template": "Saved master metadata for {basename} to {path}",
        "level": DETAIL,
    },
    "GUI_CLEANUP_INDIVIDUAL_SEG_JSON_START": {
        "template": "  Attempting to delete individual segment JSONs for {basename} (master created).",
        "level": DEBUG,
    },
    "GUI_CLEANUP_INDIVIDUAL_SEG_JSON_ERROR": {
        "template": "ERROR deleting individual segment JSON {path}: {error}",
        "level": ERROR,
    },
    "GUI_CLEANUP_INDIVIDUAL_SEG_JSON_SUMMARY": {
        "template": "    Deleted {count} individual segment JSONs.",
        "level": DEBUG,
    },
    "GUI_CLEANUP_INDIVIDUAL_SEG_JSON_SKIPPED_NO_MASTER": {
        "template": "  Skipping deletion of individual segment JSONs for {basename} (master_meta.json not saved).",
        "level": WARNING,
    },
    "GUI_MASTER_META_SAVE_SKIPPED_FULL_VIDEO": {
        "template": "Skipping save of '{path}' by _save_master_metadata for full video mode for {basename}.",
        "level": DEBUG,
    },
    "GUI_MERGE_CALL_EXCEPTION": {
        "template": "Exception calling merge_depth_segments for {basename}: {error}. Traceback: {traceback_info}",
        "level": ERROR,
    },
    "GUI_MERGE_PATH_RETURN_ISSUE_WARN": {
        "template": "Warning: merge_depth_segments completed for {basename} but did not return a valid output path. Final JSON might use a guessed path.",
        "level": WARNING,
    },
    "GUI_FINAL_JSON_PATH_DETERMINE_ERROR_MERGED": {
        "template": "    Cannot determine final JSON path for merged {basename} (output path: {path}).",
        "level": WARNING,
    },
    "GUI_FINAL_JSON_SKIPPED_MERGE_FAIL": {
        "template": "  Skipping final JSON for merged {basename} (merge not successful/path invalid).",
        "level": INFO,
    },
    "GUI_FINAL_JSON_PATH_DETERMINE_ERROR_FULL": {
        "template": "    Cannot determine final JSON path for full video {basename} (output path: {path}).",
        "level": WARNING,
    },
    "GUI_FINAL_JSON_SKIPPED_FULL_NO_PATH_INFO": {
        "template": "  Skipping final JSON for full video {basename} (output path/format missing from job_info).",
        "level": WARNING,
    },
    "GUI_FINAL_JSON_SKIPPED_FULL_NO_MASTER_META": {
        "template": "  Skipping final JSON for full video {basename} (master_meta or job_info missing).",
        "level": WARNING,
    },
    "GUI_FINAL_JSON_SAVE_ATTEMPT": {
        "template": "    Attempting to save final output JSON to: {path}",
        "level": DEBUG,
    },
    "GUI_FINAL_JSON_SAVE_SUCCESS": { # More specific than generic FILE_SAVE_SUCCESS
        "template": "  Successfully saved sidecar JSON for final output: {path}",
        "level": DETAIL,
    },
    "GUI_FINAL_JSON_NOT_CREATED_WARN": {
        "template": "  Final output JSON for {mode} '{basename}' not created (conditions not met, or save failed).",
        "level": WARNING,
    },
    "GUI_CLEANUP_SEG_FOLDER_NO_KEEP_NPZ": {
        "template": "Deleting intermediate segment subfolder for {basename} (Keep NPZ unchecked)...",
        "level": INFO,
    },
    "GUI_CLEANUP_SEG_FOLDER_THRESHOLD_NOT_MET": {
        "template": "  Video frames ({frames}) < threshold ({threshold}). Deleting segment folder for {basename} despite 'Keep NPZ'.",
        "level": INFO,
    },
    "GUI_CLEANUP_SEG_FOLDER_THRESHOLD_MET_KEPT": {
        "template": "  Video frames ({frames}) >= threshold ({threshold}). Segment folder for {basename} will be kept.",
        "level": INFO,
    },
    "GUI_CLEANUP_SEG_FOLDER_KEPT_NO_THRESHOLD": {
        "template": "Keeping intermediate NPZ files for {basename} (Keep NPZ checked, no positive frame threshold).",
        "level": INFO,
    },
    "GUI_CLEANUP_SEG_FOLDER_DELETE_SUCCESS": {
        "template": "  Successfully deleted segment subfolder for {basename}.",
        "level": INFO,
    },
    "GUI_CLEANUP_SEG_FOLDER_DELETE_ERROR": {
        "template": "  Error deleting segment subfolder {path}: {error}",
        "level": ERROR,
    },
    "GUI_CLEANUP_SEG_FOLDER_NOT_FOUND_WARN": {
        "template": "  Segment subfolder not found for deletion: {path}",
        "level": WARNING,
    },
    "GUI_CLEANUP_SEG_FOLDER_KEPT_INFO": {
        "template": "Keeping intermediate NPZ files and _master_meta.json",
        "level": INFO,
    },
    "GUI_NO_PROCESSING_TO_CANCEL_INFO": {
        "template": "No processing is currently active to cancel.",
        "level": INFO,
    },
    "GUI_PROCESSING_ALREADY_RUNNING_WARN": {
        "template": "Processing is already running.",
        "level": WARNING,
    },
    "GUI_NO_SEGMENTS_DEFINED_SKIP": {
        "template": "No segments defined by settings for {basename} (Reason: {reason}). Skipping processing for this video.",
        "level": INFO,
    },
    "GUI_FULL_VIDEO_SKIP_NO_OVERWRITE": {
        "template": "Skipping {basename} (full video processing, user chose not to overwrite).",
        "level": INFO,
    },
    "GUI_NO_JOBS_TO_PROCESS_FINAL_INFO": {
        "template": "No videos/segments to process after considering existing data and user choices (or all skipped).",
        "level": INFO,
    },
    "GUI_STOPPING_ON_CLOSE_INFO": {
        "template": "Stopping processing before exit...",
        "level": INFO,
    },
    "GUI_THREAD_FORCE_EXIT_WARN": {
        "template": "Processing thread did not terminate gracefully. Forcing exit.",
        "level": WARNING,
    },
    "GUI_REMERGE_CANCELLED_NO_PATH": {
        "template": "Re-merge cancelled: No output path selected.",
        "level": INFO,
    },
    "GUI_REMERGE_EXEC_ERROR": {
        "template": "ERROR during re-merge execution: {error}. Traceback: {traceback_info}",
        "level": ERROR,
    },
    "GUI_GEN_VISUALS_CANCELLED_NO_META": {
        "template": "Segment visual generation cancelled: No master metadata file selected.",
        "level": INFO,
    },
    "GUI_GEN_VISUALS_CANCELLED_BY_USER": {
        "template": "Segment visual generation cancelled by user.",
        "level": INFO,
    },
    "GUI_GEN_VISUALS_NO_SUCCESSFUL_SEGS": {
        "template": "No successful segments with output filenames found in {master_file} for visual generation.",
        "level": WARNING,
    },
    "GUI_GEN_VISUALS_CANCELLED_MID_PROCESS": {
        "template": "Segment visual generation cancelled during processing.",
        "level": INFO,
    },
    "GUI_GEN_VISUALS_PROCESSING_SEGMENT": {
        "template": "  Visual Gen - Processing segment {segment_id}/{total_segments}: {npz_name} for {format}",
        "level": DEBUG, # Or INFO
    },
    "GUI_GEN_VISUALS_SEGMENT_EMPTY_WARN": {
        "template": "    Visual Gen - WARNING: Segment {npz_name} is empty. Skipping.",
        "level": WARNING,
    },
    "GUI_GEN_VISUALS_SAVE_SUCCESS": {
        "template": "    Visual Gen - Successfully saved visual: {path}",
        "level": DETAIL,
    },
    "GUI_GEN_VISUALS_SAVE_ERROR": {
        "template": "    Visual Gen - ERROR saving visual for {npz_name}: {error}",
        "level": ERROR,
    },
    "GUI_GEN_VISUALS_SEGMENT_PROCESSING_ERROR": {
        "template": "    Visual Gen - ERROR processing segment {npz_name}: {error}. Traceback: {traceback_info}",
        "level": ERROR,
    },
    "GUI_GEN_VISUALS_UPDATING_MASTER_META": {
        "template": "Visual Gen - Updating master metadata with new visual paths...",
        "level": INFO,
    },
    "GUI_GEN_VISUALS_MASTER_META_UPDATE_SUCCESS": {
        "template": "Visual Gen - Master metadata updated for {count} segments.",
        "level": INFO,
    },
    "GUI_GEN_VISUALS_MASTER_META_NO_UPDATES_NEEDED": {
        "template": "Visual Gen - No segments in master metadata needed visual path updates.",
        "level": INFO,
    },
    "GUI_VERBOSITY_CHANGED": {
        "template": "GUI log verbosity changed to: {level_name} (Numeric: {numeric_level})",
        "level": INFO,
    },
    "IMAGE_SEQUENCE_NO_FRAMES_FOUND": {
        "level": WARNING,
        "template": "Image Sequence: No compatible image files found in folder '{folder_path}'. Supported extensions: {extensions}."
    },
    "IMAGE_READ_ERROR": {
        "level": ERROR,
        "template": "Image Read Error: Could not read image file '{filepath}'. Error: {error}."
    },
    "IMAGE_SEQUENCE_METADATA_SUCCESS": {
        "level": DEBUG,
        "template": "Image Sequence Metadata: Folder '{folder_path}', Frames: {count}, Effective FPS: {fps:.2f}, H: {height}, W: {width}."
    },
    "SINGLE_IMAGE_METADATA_SUCCESS": {
        "level": DEBUG,
        "template": "Single Image Metadata: File '{image_path}', Frames for 1s clip: {frames_for_clip}, Effective FPS: {fps:.2f}, H: {height}, W: {width}."
    },
    "IMAGE_SEQUENCE_LOAD_NO_IMAGES": {
        "level": WARNING,
        "template": "Image Sequence Load: No images found or readable in '{folder_path}'."
    },
    "IMAGE_SEQUENCE_FRAME_READ_ERROR": {
        "level": WARNING,
        "template": "Image Sequence Load: Error reading frame '{filepath}'. Error: {error}. Skipping frame."
    },
    "IMAGE_SEQUENCE_LOAD_FAILED_ALL_FRAMES": {
        "level": ERROR,
        "template": "Image Sequence Load: Failed to load any frames from '{folder_path}'."
    },
    "IMAGE_SEQUENCE_LOAD_SUCCESS": {
        "level": INFO,
        "template": "Image Sequence Load: Successfully loaded {num_frames} frames from '{folder_path}' (H:{height}, W:{width})."
    },
    "SINGLE_IMAGE_FRAMES_GENERATED": {
        "level": INFO,
        "template": "Single Image: Generated {num_frames} frames for 1s clip from '{image_path}' (H:{height}, W:{width})."
    },
    "SINGLE_IMAGE_FRAMES_GENERATION_ERROR": {
        "level": ERROR,
        "template": "Single Image: Error generating frames from '{image_path}'. Error: {error}."
    },
    "SEGMENT_DEFINE_ZERO_ADVANCE": {
        "level": WARNING,
        "template": "Segment Definition for '{video_name}': Raw advance per segment is {advance} (window_raw: {window_raw}). This is okay if total length ({effective_length}) is less than one window, otherwise it's an issue."
    },
    "FRAMES_LOAD_UNKNOWN_SOURCE_DICT_ERROR": {
        "level": ERROR,
        "template": "Frames Load: Unknown source type '{type}' in input dictionary."
    },
    "RUN_MISSING_BASENAME_FOR_DICT_ERROR": {
        "level": ERROR,
        "template": "Run Logic: original_video_basename_override is required for dictionary input. Dict keys: {info_dict_keys}"
    },
    "GUI_INPUT_MODE_SET_IMG_SEQ_FOLDER": {
        "level": INFO,
        "template": "GUI: Input mode set to Image Sequence Folder: {path}"
    },
    "GUI_INPUT_MODE_SET_BATCH_FOLDER": {
        "level": INFO,
        "template": "GUI: Input mode set to Batch Folder: {path}"
    },
    "GUI_INPUT_MODE_SET_SINGLE_VIDEO": {
        "level": INFO,
        "template": "GUI: Input mode set to Single Video File: {path}"
    },
    "GUI_INPUT_MODE_SET_SINGLE_IMAGE": {
        "level": INFO,
        "template": "GUI: Input mode set to Single Image File: {path}"
    },
    "GUI_INPUT_MODE_UNKNOWN_SINGLE_FILE_WARN": {
        "level": WARNING,
        "template": "GUI: Could not determine type of single file: {path}. Assuming video."
    },
    "GUI_INPUT_PATH_INVALID_ERROR": {
        "level": ERROR,
        "template": "GUI: Input path is invalid or does not exist: {path}."
    },
    "GUI_START_THREAD_INVALID_MODE_ERROR": {
        "level": ERROR,
        "template": "GUI Start Thread: Invalid internal input mode '{mode}'."
    },
    "GUI_NO_VALID_SOURCES_FOUND": {
        "level": WARNING,
        "template": "GUI: No valid video files or image sequences found in '{path_scanned}' for mode '{mode}'."
    },
    "GUI_ORIGINAL_VIDEO_MOVE_SKIPPED_SINGLE_MODE": {
        "level": INFO,
        "template": "GUI: Skipped moving original source '{basename}' (single file/sequence mode)."
    },
    "GUI_JOB_UNKNOWN_SOURCE_TYPE_ERROR":{
        "level": ERROR,
        "template": "GUI Process Job: Unknown source type '{type}' for '{basename}'."
    },
    "VIDEO_UNKNOWN_DATASET_NO_DIMS_ERROR": {
        "level": ERROR,
        "template": "Video Read: Unknown dataset '{dataset_name}' for '{video_path}' and could not determine original dimensions for fallback."
    },
    "VIDEO_READ_METADATA_EMPTY_FRAME_WARN": {
        "level": WARNING,
        "template": "Video Read Metadata: Could not get frame data from '{video_path}' (possibly corrupt or empty first frame)."
    },
    "VIDEO_READ_METADATA_ZERO_LENGTH_WARN": {
        "level": WARNING,
        "template": "Video Read Metadata: Video '{video_path}' has zero length."
    },
    "VIDEO_INVALID_FPS_METADATA_WARN": {
        "level": WARNING,
        "template": "Video Read: Invalid FPS ({fps_read}) from metadata for '{video_path}'. Falling back."
    },
    "VIDEO_READ_NO_ORIGINAL_DIMS_ERROR": {
        "level": ERROR,
        "template": "Video Read: Could not obtain original dimensions for '{video_path}' during metadata scan. Cannot proceed with dimension calculation."
    },
    "VIDEO_READ_ZERO_DIM_SCALE_ERROR": {
        "level": ERROR,
        "template": "Video Read: Original dimension is zero for '{video_path}', cannot calculate scale for resizing. Falling back."
    },
    "GUI_CONFIG_LOADED_INFO": {
        "level": INFO,
        "template": "GUI: Configuration loaded from '{filename}'."
    },
    "GUI_CONFIG_NOT_FOUND_INFO": {
        "level": INFO,
        "template": "GUI: Configuration file '{filename}' not found. Using default settings."
    },
    "GUI_INPUT_PATH_INVALID_FOR_MODE_DETECT": {
        "level": WARNING,
        "template": "GUI Input: Path '{path}' is invalid or does not exist. Cannot determine input mode accurately."
    },
    "GUI_INPUT_MODE_UNKNOWN_TYPED_FILE_WARN": {
        "level": WARNING,
        "template": "GUI Input: Typed path '{path}' is a file of unknown type. Treating as non-single source."
    },
    "GUI_INPUT_PATH_NOT_FILE_OR_DIR_WARN": {
        "level": WARNING,
        "template": "GUI Input: Path '{path}' exists but is not a regular file or directory."
    },
    "GUI_INPUT_MODE_DETERMINED": {
        "level": DEBUG, # Or INFO if you want to see it always
        "template": "GUI Input: Determined mode for path '{path}' as '{mode}', is_single_source: {is_single}."
    },
    "GUI_INPUT_PATH_EMPTY_ERROR": {
        "level": ERROR,
        "template": "GUI Input: Input path field is empty. Please provide a source."
    },
    "GUI_INPUT_PATH_NOT_DIR_FOR_BATCH_ERROR": {
        "level": ERROR,
        "template": "GUI Input: Path '{path}' is not a directory, but batch processing mode was attempted."
    },
    "GUI_LISTDIR_OS_ERROR": {
        "level": ERROR,
        "template": "GUI Input: OS error when trying to list directory '{path}'. Error: {error}"
    },
    "GUI_START_THREAD_UNEXPECTED_MODE_AFTER_DETERMINATION": {
        "level": CRITICAL,
        "template": "GUI Start Thread: Unexpected mode '{mode}' for path '{path}' after explicit determination. This indicates a logic error."
    },
    "GUI_UNKNOWN_SOURCE_SPEC_TYPE_ERROR": {
         "level": ERROR,
         "template": "GUI Start Thread: Unknown source_spec type '{type}' for basename '{basename}'. Cannot map to define_video_segments source type."
    },
    "FRAMES_LOAD_FROM_SINGLE_IMG_INFO": {
        "level": DETAIL,
        "template": "Frames Load: Loaded {num_frames} frames from single image '{image_path}' at {fps:.1f} FPS (for 1s clip)."
    },
    "IMAGE_SAVE_EXR_SEQ_UTIL_ERROR": {
        "level": ERROR,
        "template": "Error saving EXR sequence (path: {dir_path}): {error}" # Simplified template
    },
    "IMAGE_SAVE_SINGLE_EXR_UTIL_ERROR": {
        "level": ERROR,
        "template": "Error saving single EXR (filepath: {filepath}): {error}" # Simplified template
    },
    "IMAGE_SAVE_EXR_FRAME_ERROR": {
        "level": ERROR,
        "template": "Error saving frame to EXR file '{filepath}': {error}"
    },
    "IMAGE_SEQUENCE_LOAD_INVALID_START_INDEX": {
        "level": ERROR,
        "template": "Image Sequence Load: Invalid start_index {start_index} for folder '{folder_path}'. Total images: {total_images}."
    },
    "IMAGE_SEQUENCE_LOAD_NO_FRAMES_FOR_SEGMENT": {
        "level": WARNING,
        "template": "Image Sequence Load: No image frames found for segment in '{folder_path}' (start: {start_index}, num_to_load: {num_to_load}, end_calc: {end_index_calc}, total_in_folder: {total_images_in_folder})."
    },
    "IMAGE_SEQUENCE_REF_FRAME_READ_ERROR": {
        "level": ERROR,
        "template": "Image Sequence Load: Error reading reference frame '{filepath}' for consistent resizing. Error: {error}"
    },
    # Modify existing IMAGE_SEQUENCE_LOAD_SUCCESS to include segment_start_index if you want
    "IMAGE_SEQUENCE_LOAD_SUCCESS": {
        "level": DETAIL,
        "template": "Image Sequence Load: Successfully loaded {num_frames} frames from '{folder_path}' (H:{height}, W:{width}, Segment Start Idx: {segment_start_index})."
    },
    # Modify FRAMES_LOAD_FROM_IMG_SEQ_INFO to include start_index and num_loaded
    "FRAMES_LOAD_FROM_IMG_SEQ_INFO": {
        "level": DETAIL,
        "template": "Frames Load: Attempting to load image sequence from '{folder_path}'. Target FPS: {fps:.3f}. Segment Start Idx: {start_index}, Num to Load: {num_loaded}. Loaded: {num_frames} frames."
    },
    "LOGIC_SAVE_FPS_STILL_NEGATIVE_WARN": {
        "level": WARNING,
        "template": "Logic Save Video: FPS value is still -1.0 at save point, falling back. FPS: {fps_val}"
    },
    "LOGIC_SAVE_FPS_ZERO_OR_NEGATIVE_WARN": {
        "level": WARNING,
        "template": "Logic Save Video: FPS value is zero or negative at save point, falling back. FPS: {fps_val}"
    },
    "GUI_ORIGINAL_VIDEO_MOVE_CANNOT_DETERMINE_ROOT": {
        "level": ERROR,
        "template": "Move Original: Cannot determine a valid root directory for 'finished' folder based on input path '{path}'."
    },
    # Update GUI_ORIGINAL_VIDEO_MOVE_SUCCESS if you changed the keys
    "GUI_ORIGINAL_VIDEO_MOVE_SUCCESS": {
        "level": DETAIL,
        "template": "Successfully moved original source '{filename}' to '{destination_folder}' folder." # Example update
    },
    "GUI_MOVE_UNKNOWN_STATUS_WARN": {
        "level": WARNING,
        "template": "Move Original: Could not determine 'finished' or 'failed' status for '{basename}' (status: '{status}'). Original file will not be moved."
    },
    "GUI_ORIGINAL_SOURCE_MOVE_ATTEMPT": {
        "level": INFO, # Renamed from GUI_ORIGINAL_VIDEO_MOVE_ATTEMPT
        "template": "Attempting to move original source '{basename}' to '{target_folder}' folder."
    },
    "GUI_ORIGINAL_SOURCE_MOVE_SUCCESS": {
        "level": INFO, # Renamed from GUI_ORIGINAL_VIDEO_MOVE_SUCCESS
        "template": "Successfully moved original source '{filename}' to '{destination_folder}' folder."
    },

    # Resume logic messages
    "GUI_RESUME_REPROCESS_FAILED_MASTER_START": { "template": "Attempting to re-process failed segments for {basename} based on existing master metadata.", "level": INFO },
    "GUI_RESUME_MASTER_LOAD_FAIL_WARN": { "template": "Could not load master metadata or 'jobs_info' missing for {basename}. Defaulting to overwrite.", "level": WARNING },
    "GUI_RESUME_QUEUEING_FAILED_SEGMENT": { "template": "  Queueing segment ID {segment_id} (status: {status}) for {basename} for re-processing.", "level": DETAIL },
    "GUI_RESUME_UNQUEUEABLE_SEGMENT_WARN": { "template": "  Warning: Segment (ID: {segment_id}, Status: {status}) from master_meta for {basename} not re-queueable. It will be ignored.", "level": WARNING },
    "GUI_RESUME_NO_FAILED_SEGMENTS_IN_MASTER": { "template": "No re-processable failed segments found in master_meta for {basename}. All existing successful segments will be preserved if merging.", "level": DETAIL },
    "GUI_RESUME_BACKUP_MASTER_FAIL_WARN": { "template": "  Warning: Could not back up existing master metadata: {error}. It might be overwritten.", "level": WARNING },
    "GUI_RESUME_DELETING_FINALIZED_START": { "template": "User chose/defaulted to delete existing segment folder and start fresh for {basename}: {path}", "level": DETAIL },
    "GUI_RESUME_DELETING_FOLDER_SUCCESS": { "template": "  Successfully deleted: {path}", "level": DETAIL },
    "GUI_RESUME_DELETING_FOLDER_ERROR": { "template": "  Error deleting {path}: {error}. Processing may fail or overwrite.", "level": ERROR },
    "GUI_RESUME_SKIPPING_FINALIZED": { "template": "Skipping {basename} (user chose to cancel on finalized segments).", "level": INFO },
    "GUI_RESUME_INCOMPLETE_START": { "template": "Attempting to resume incomplete segments for {basename}.", "level": INFO },
    "GUI_RESUME_INCOMPLETE_SEGMENT_REPROCESS": { "template": "  Segment {segment_id}/{total_segments} for {basename} found but not successful (status: {status}). Will re-process.", "level": INFO },
    "GUI_RESUME_INCOMPLETE_SEGMENT_MISSING": { "template": "  Segment {segment_id}/{total_segments} for {basename} (NPZ: {npz_filename}) not found or JSON missing. Will process.", "level": DETAIL },
    "GUI_RESUME_ALL_SEGS_COMPLETE_NO_MASTER_WARN": { "template": "  All segments for {basename} appear complete from individual files, but master_meta was missing. Consider re-merging. Skipping processing.", "level": WARNING },
    "GUI_RESUME_NO_SEGS_TO_RUN_INCOMPLETE_WARN": { "template": "  No segments to run for {basename}, but not all were found complete. Total defined: {total_defined}, Found complete: {found_complete}", "level": WARNING },
    "GUI_RESUME_DELETING_INCOMPLETE_START": { "template": "User chose to delete existing incomplete segment folder and start fresh for {basename}: {path}", "level": DETAIL },
    "GUI_RESUME_SKIPPING_INCOMPLETE": { "template": "Skipping {basename} (user chose to cancel on incomplete segments).", "level": INFO },

    # General
    "GENERAL_INFO": {"template": "{message}", "level": INFO},
    "GENERAL_WARNING": {"template": "Warning: {message}", "level": WARNING},
    "GENERAL_ERROR": {"template": "ERROR: {message}", "level": ERROR},
    "GENERAL_CRITICAL": {"template": "CRITICAL ERROR: {message}", "level": CRITICAL},
    "GENERAL_DEBUG": {"template": "DEBUG: {message}", "level": DEBUG},

    # File Operations
    "FILE_NOT_FOUND": {"template": "File not found: {filepath}", "level": ERROR},
    "FILE_LOAD_SUCCESS": {"template": "Successfully loaded: {filepath}", "level": DETAIL},
    "FILE_SAVE_SUCCESS": {"template": "Successfully saved: {filepath}", "level": DETAIL},
    "FILE_SAVE_FAILURE": {"template": "Failed to save: {filepath}. Reason: {reason}", "level": ERROR},
    "JSON_DECODE_ERROR": {"template": "Could not decode JSON from: {filepath}. Reason: {reason}", "level": ERROR},
    "NPZ_LOAD_KEY_ERROR": {"template": "Key '{key}' not found in NPZ: {filepath}", "level": ERROR},

    # Processing Status
    "PROCESSING_STARTED": {"template": "Starting processing for: {item_name}", "level": INFO},
    "PROCESSING_COMPLETE": {"template": "Finished processing for: {item_name}. Status: {status}", "level": INFO},
    "PROCESSING_SKIPPED": {"template": "Skipping {item_name}: {reason}", "level": INFO},
    "PROCESSING_JOB_PROGRESS": {"template": "Processing {item_name} - {job_type}", "level": INFO},
    "PROCESSING_ERROR": {"template": "Error processing {item_name}: {error_message}", "level": ERROR},
    "PROCESSING_NO_FRAMES": {"template": "No frames to process for {item_name}. Skipping.", "level": WARNING},
    "PROCESSING_FINALIZE_ERROR": {"template": "Error during finalization for {item_name}: {error_message}", "level": ERROR},


    # Model / System
    "MODEL_INIT_SUCCESS": {"template": "DepthCrafter model initialized successfully.", "level": INFO},
    "MODEL_INIT_FAILURE": {"template": "CRITICAL: Failed to initialize DepthCrafter model: {reason}", "level": CRITICAL},
    "CUDA_AVAILABLE": {"template": "CUDA available. Device: {device_name}", "level": INFO},
    "CUDA_UNAVAILABLE": {"template": "CUDA not available. Processing will use CPU (if supported).", "level": WARNING},
    "CUDA_CACHE_CLEARED": {"template": "CUDA cache cleared.", "level": INFO},
    "OPENEXR_UNAVAILABLE": {"template": "OpenEXR/Imath libraries not found. EXR features will be limited/unavailable.", "level": WARNING},

    # Segments & Merging
    "SEGMENT_DEFINE_SUCCESS": {"template": "Defined {num_segments} segments for {video_name}.", "level": DEBUG},
    "SEGMENT_DEFINE_FAILURE": {"template": "Failed to define segments for {video_name}: {reason}", "level": WARNING},
    "SEGMENT_LOAD_SUCCESS": {"template": "Segment {segment_id} ({filename}) loaded. Shape: {shape}, FPS: {fps}", "level": DEBUG},
    "SEGMENTS_ALL_SUCCESS": {"template": "All segments for {video_name} processed successfully.", "level": INFO},
    "MERGE_STARTED": {"template": "--- Starting segment merging for {video_name} ---", "level": INFO},
    "MERGE_COMPLETE": {"template": "--- Segment merging for {video_name} completed. Output: {output_path} ---", "level": DETAIL},
    "MERGE_FAILURE": {"template": "ERROR during segment merging for {video_name}: {reason}", "level": ERROR},
    "MERGE_MODULE_UNAVAILABLE": {"template": "Segment merging for {video_name} skipped: merge_depth_segments module not available.", "level": WARNING},
    "MERGE_ALIGN_INFO": {"template": "Aligning segment {current_id} to {prev_id}. Method: {method}. Scale: {scale:.4f}, Shift: {t:.4f}", "level": DEBUG},
    "MERGE_NO_OVERLAP_ALIGN": {"template": "Warning: No actual overlap for alignment between {prev_id} and {current_id}. No S&S alignment performed.", "level": DEBUG}, # Was level=SILENCE_PASSES
    "MERGE_STITCH_INFO": {"template": "Stitching segment {idx} (ID: {seg_id}). Adding {count} frames.", "level": DEBUG},

    # GUI Specific
    "GUI_LOG_CLEARED": {"template": "Log cleared by user.", "level": INFO},
    "GUI_SETTINGS_LOADED": {"template": "Successfully loaded settings from: {filepath}", "level": DETAIL},
    "GUI_SETTINGS_SAVED": {"template": "Successfully saved settings to: {filepath}", "level": DETAIL},
    "GUI_SETTINGS_RESET": {"template": "All settings have been reset to their initial defaults.", "level": INFO},
    "GUI_NO_VIDEOS_FOUND": {"template": "No videos found in input folder. Check extensions (mp4, avi, mov, mkv, webm, flv, gif).", "level": WARNING},
    "GUI_CANCEL_REQUEST": {"template": "Cancel request received. Processing will stop after current item.", "level": INFO},
    "GUI_INVALID_INPUT_ERROR": {"template": "Invalid input: {field_name} - {issue}", "level": ERROR},
    "GUI_ACTION_STARTED": {"template": "--- Starting {action_name} for: {target_name} ---", "level": DETAIL},
    "GUI_ACTION_COMPLETE": {"template": "--- {action_name} for: {target_name} finished in {duration}. ---", "level": DETAIL},
    "GUI_RESUME_ACTION": {"template": "For video '{video_name}': Action '{action_taken}', {num_segments} segments will be processed.", "level": DETAIL},
    "GUI_PRESERVING_SUCCESSFUL_SEGMENTS": {"template": "Found {num_complete} successfully completed segments for {video_name} that will be skipped during processing.", "level": INFO},
    "GUI_BACKED_UP_FILE": {"template": "Backed up existing file {original_path} to: {backup_path}", "level": DETAIL},
    "GUI_ORIGINAL_SOURCE_MOVE_INVALID_INPUT_ROOT_WARN": { "level": WARNING, "template": "Move Original: The GUI input path '{gui_input_path}' is invalid for determining the target folder root. Using dirname of processed item as fallback." },
    "GUI_ORIGINAL_SOURCE_MOVE_CANNOT_DETERMINE_ROOT": { "level": ERROR, "template": "Move Original: Cannot determine a valid root directory for target folder based on input path '{path}'." },
    "GUI_ORIGINAL_SOURCE_MOVE_SOURCE_NOT_FOUND": { "level": WARNING, "template": "Move Original: Source path to move does not exist: {path}" },
    "GUI_ORIGINAL_SOURCE_MOVE_RENAMED": { "level": INFO, "template": "Move Original: Destination already exists. Renaming '{old_name}' to '{new_name}'." },
    "GUI_ORIGINAL_SOURCE_MOVE_ERROR": { "level": ERROR, "template": "ERROR moving original source '{basename}': {error}", "traceback": True },


    # Utilities
    "UTIL_NORMALIZE_VIDEO": {"template": "Normalizing video data. Shape: {shape}. Method: {method}. Range: {min_val:.4f}-{max_val:.4f}", "level": DEBUG},
    "UTIL_GAMMA_CORRECTION": {"template": "Applying Gamma ({gamma_val:.2f}) to video.", "level": DEBUG},
    "UTIL_DITHERING": {"template": "Applying dithering to video. Strength factor: {strength_factor:.2f}", "level": DEBUG},

}

# --- Logging Function ---
def log_message(msg_id: str, **kwargs):
    """
    Logs a message based on its ID from the catalog.
    Uses global verbosity levels and GUI callback.
    """
    if msg_id not in MESSAGES:
        # Fallback for unknown message ID
        formatted_message = f"Unknown message ID: {msg_id} - Args: {kwargs}"
        if _CONSOLE_VERBOSITY_LEVEL <= ERROR:
            print(f"{get_formatted_timestamp_mc()} [ERROR] {formatted_message}", file=sys.stderr)
        if _GUI_LOGGER_CALLBACK and _GUI_VERBOSITY_LEVEL <= ERROR:
            _GUI_LOGGER_CALLBACK(f"[ERROR] {formatted_message}") # GUI callback expects only the message string
        return

    msg_info = MESSAGES[msg_id]
    msg_template = msg_info["template"]
    msg_level = msg_info["level"]

    try:
        formatted_message = msg_template.format(**kwargs)
    except KeyError as e:
        formatted_message = f"KeyError formatting msg_id '{msg_id}': {e}. Template: '{msg_template}', Args: {kwargs}"
        msg_level = ERROR # Escalate to error if formatting fails

    # Console Output
    if msg_level >= _CONSOLE_VERBOSITY_LEVEL:
        prefix = ""
        if _USE_TIMESTAMPS_CONSOLE:
            prefix += f"{get_formatted_timestamp_mc()} "
        
        level_name = {DEBUG: "[DEBUG]", INFO: "[INFO]", WARNING: "[WARNING]", ERROR: "[ERROR]", CRITICAL: "[CRITICAL]"}.get(msg_level, "[MSG]")
        prefix += f"{level_name} "

        output_stream = sys.stderr if msg_level >= ERROR else sys.stdout
        print(f"{prefix}{formatted_message}", file=output_stream)

    # GUI Output
    if _GUI_LOGGER_CALLBACK and msg_level >= _GUI_VERBOSITY_LEVEL:
        gui_message = formatted_message
        if _USE_TIMESTAMPS_GUI: # GUI might add its own, this is if we want it in the string itself
            gui_message = f"{get_formatted_timestamp_mc()} {gui_message}"
        
        # Optionally, you could add level prefixes for GUI too, if the GUI doesn't do it.
        # level_name_gui = {DEBUG: "DEBUG: ", INFO: "", WARNING: "Warning: ", ERROR: "ERROR: ", CRITICAL: "CRITICAL: "}.get(msg_level, "")
        # _GUI_LOGGER_CALLBACK(f"{level_name_gui}{gui_message}")
        _GUI_LOGGER_CALLBACK(gui_message) # Assuming GUI callback handles any further formatting/prefixing