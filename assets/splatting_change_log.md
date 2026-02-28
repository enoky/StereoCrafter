# Splatting GUI Changelog

All notable changes to the splatting GUI and related components.

## Version 26-02-27.9

### Added (v26-02-27.9)

- **Preview Frame Buffer**: New frame caching system to enable faster playback in the preview window.
  - Processed frames are cached in memory to avoid re-processing on loop playback
  - Supports both main preview and SBS (Side-by-Side) window playback
  - Buffer automatically clears when processing parameters change (convergence, disparity, gamma, depth settings, border settings, etc.)
  - Caches pre-scaled display images to avoid repeated resizing operations
  - LRU eviction when max frames (500) is reached
  - Implemented in `core/common/preview_buffer.py` and integrated into `dependency/video_previewer.py`
  - Callback system notifies parent GUI when frames are displayed for SBS window updates

## Version 26-02-27.8

### Refactored (v26-02-27.8)

- **ProcessingSettings.from_config()**: New classmethod on `ProcessingSettings` that builds an instance from a flat config dict with automatic type coercion and key aliasing. `_get_processing_settings` in the GUI reduced from 60 → 25 lines.
- **Delegated Validation**: The 55-line inline validation block in `start_processing()` now delegates to `BatchProcessor.validate_settings()` (single source of truth).
- **ConvergenceCache**: New `core/splatting/convergence_cache.py` module centralises all per-clip cache state (`_auto_conv_cache`, `_dp_total_est_cache`, `_dp_total_true_cache`, `_clip_norm_cache`) into a single `ConvergenceCache` object. The GUI uses backward-compatible aliases during the transition.

## Version 26-02-27.7

### Added (v26-02-27.7)

- **Cross-Eye SBS Toggle**: Press **'X'** while the SBS Preview window is in focus to swap left/right views for cross-eye viewing.
- **SBS State Persistence**: The SBS preview toggle state is now saved to the configuration and restored automatically upon application startup.
- **Unified Keyboard Navigation**: Arrows and spacebar events are now relayed from the SBS window to the main GUI, allowing seamless playback control while inspecting the 3D view.
- **Click-to-Focus Management**: Improved focus behavior—clicking the background or the preview canvas now successfully "defocuses" text input fields, restoring hotkey functionality immediately.

### Refactored

- **Core Sidecar Manager**: Centralized all sidecar reading, writing, and path resolution into `core/common/sidecar_manager.py`. This logic is now shared across both Splatting and Merging GUIs.
- **Automated Parameter Syncing**: Replaced ~200 lines of manual mapping in the GUI with an automated sync system that maps JSON keys directly to Tkinter variables via a central schema.
- **Geometric Isolation**: Moved the math for converting between UI Width/Bias and storage Left/Right border values into the core library, isolating the "math" from the "clicks."

- **AnalysisService**: Pure analysis pipelines migrated to `core/splatting/analysis_service.py` (`_auto_converge_worker`, `_estimate_dp_total_max_for_depth_video`, etc.)

## Version 26-02-27.4

### Added (v26-02-27.4)

- **10-bit & Color Tagging**: Added support for 10-bit HEVC (libx265/hevc_nvenc) and DNxHR HQX encoding with accurate color space metadata (BT.2020 PQ/HLG).
- **DNxHR Split Mode**: New output mode for high-resolution dual-stream delivery ([Occlusion Mask] and [Right Eye] as separate files).
- **SBS Preview Window**: Integrated real-time Side-By-Side (SBS) preview toggle for immediate stereo alignment checks.
- **Strict FFmpeg Decode**: New toggle to force bit-accurate depth map reading via FFmpeg pipes, bypassing 8-bit truncation issues in some decoders.
- **Diagnostic Capture Suite**: Integrated PNG export for "Map Test" and "Splat Test" frames, featuring auto-switching preview sources and metadata labeling.

### Fixed

- **Aspect Ratio Parity**: Fixed horizontal stretching mismatch between preview and render; depth maps are now resized early to match source video aspect ratios.
- **Diagnostic Quad-Cropping**: Fixed quadrant cropping logic to correctly handle both 2-panel (Dual) and 4-panel grid layouts in PNG exports.
- **Depth Unification**: Disabled automatic TV-range expansion in preview to ensure visual parity with "raw" render outputs.
- **Bit-Depth Detection**: Unified bit-depth inference between preview and render paths to prevent contrast mismatches during normalization.

### Changed

- **Renderer Optimization**: Diagnostic captures now force a 4-panel grid internally to ensure all data (Depth, Occlusion, etc.) is available regardless of output mode.

## Version 26-02-26.2

### Bug Fixes (v26-02-26.2)

- **Depth Preprocessing**: Fixed issue where preview depth pre-processing was using local and not the core library functions.
- **Depth Normalization**: unified and refactored depth normalization logic between preview and batch processing.

## Version 26-02-21.0

### Bug Fixes (v26-02-21.0)

- **10-bit Precision Loss**: Restored `float32` rendering pipeline to prevent bit-depth truncation to 8-bit before encoding.
- **HDR/HLG Output**: Fixed issue where BT.2020 PQ and HLG selections incorrectly produced BT.709 output.
- **Missing GUI Options**: Restored "BT.709 L/F" and "BT.2020 PQ/HLG" modes to the Color Tags dropdown.

### Improvements (v26-02-21.0)

- **HDR Detection**: Enhanced `stereocrafter_util.py` to correctly detect HLG transfer characteristics for 10-bit HEVC encoding.
- **Legacy Compatibility**: Added internal mappings for old "BT.709" and "BT.2020" config strings to ensure high-quality profile selection.

---

## Version 26-02-07.1

### Added (v26-02-07.1)

- **Manual Mode for Auto-Convergence:** New "Manual" mode writes current slider values to sidecars during AUTO-PASS without calculating auto-convergence.
- **Sidecar Migration Menu Items:** Two new File menu options to move sidecars between folders:
  - "Sidecars: Depth → Source (remove _depth)" - moves sidecars from depth folder to source folder
  - "Sidecars: Source → Depth (add _depth)" - moves sidecars from source folder to depth folder

### Changed (v26-02-07.1)

- **Auto-Convergence "Off" Behavior:** AUTO-PASS no longer overwrites sidecar values when set to "Off" - existing sidecar values are now preserved.
- **AUTO-PASS Border Mode:** When GUI Border Mode is "Auto Basic" or "Auto Adv.", AUTO-PASS now stores values in `auto_border_L`/`auto_border_R` fields (for UI caching) and keeps `border_mode` as "Auto Basic"/"Auto Adv." instead of switching to "Manual".

### Bug Fixes (v26-02-07.1)

- **Auto-Convergence Cache Clearing:** Fixed clip navigation not clearing cached Average/Peak values, which caused incorrect values to be applied when switching between clips.

---

## Version 2026-02-04

### Refactored (v2026-02-04)

- **Code Consolidation**: Removed duplicate depth processing code from `splatting_gui.py` (lines 7018-7234)
  - `compute_global_depth_stats()` function now imported from `core.splatting.depth_processing`
  - `load_pre_rendered_depth()` function now imported from `core.splatting.depth_processing`
  - Removed local redefinitions of `_NumpyBatch` and `_ResizingDepthReader` classes
  - Eliminated ~216 lines of duplicated code

### Changed (v2026-02-04)

- **Slider Behavior Enhancement** (`dependency/stereocrafter_util.py`):
  - **Middle-click**: Slider now jumps directly to mouse pointer position (matches frame scrubber behavior)
  - **Right-click**: Resets slider to default value
  - **Left-click**: Maintains original stepped increment behavior
- **Browes Folder Buttons:** Right-Click opens explorer window at paths location.
- **Combined Scanning:** `Auto Convergence` and `Boarder Depth` into a single pass.

### Bug Fixes (v2026-02-04)

- Slider middle-click functionality now provides precise positioning like the frame scrubber
