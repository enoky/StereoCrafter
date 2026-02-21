# Splatting GUI Changelog

All notable changes to the splatting GUI and related components.

## Version 26-02-21.0

### Fixed

- **10-bit Precision Loss**: Restored `float32` rendering pipeline to prevent bit-depth truncation to 8-bit before encoding.
- **HDR/HLG Output**: Fixed issue where BT.2020 PQ and HLG selections incorrectly produced BT.709 output.
- **Missing GUI Options**: Restored "BT.709 L/F" and "BT.2020 PQ/HLG" modes to the Color Tags dropdown.

### Changed

- **HDR Detection**: Enhanced `stereocrafter_util.py` to correctly detect HLG transfer characteristics for 10-bit HEVC encoding.
- **Legacy Compatibility**: Added internal mappings for old "BT.709" and "BT.2020" config strings to ensure high-quality profile selection.

---

## Version 26-02-07.1

### Added

- **Manual Mode for Auto-Convergence:** New "Manual" mode writes current slider values to sidecars during AUTO-PASS without calculating auto-convergence.
- **Sidecar Migration Menu Items:** Two new File menu options to move sidecars between folders:
  - "Sidecars: Depth → Source (remove _depth)" - moves sidecars from depth folder to source folder
  - "Sidecars: Source → Depth (add _depth)" - moves sidecars from source folder to depth folder

### Changed

- **Auto-Convergence "Off" Behavior:** AUTO-PASS no longer overwrites sidecar values when set to "Off" - existing sidecar values are now preserved.
- **AUTO-PASS Border Mode:** When GUI Border Mode is "Auto Basic" or "Auto Adv.", AUTO-PASS now stores values in `auto_border_L`/`auto_border_R` fields (for UI caching) and keeps `border_mode` as "Auto Basic"/"Auto Adv." instead of switching to "Manual".

### Fixed

- **Auto-Convergence Cache Clearing:** Fixed clip navigation not clearing cached Average/Peak values, which caused incorrect values to be applied when switching between clips.

---

## Version 2026-02-04

### Refactored

- **Code Consolidation**: Removed duplicate depth processing code from `splatting_gui.py` (lines 7018-7234)
  - `compute_global_depth_stats()` function now imported from `core.splatting.depth_processing`
  - `load_pre_rendered_depth()` function now imported from `core.splatting.depth_processing`
  - Removed local redefinitions of `_NumpyBatch` and `_ResizingDepthReader` classes
  - Eliminated ~216 lines of duplicated code

### Changed

- **Slider Behavior Enhancement** (`dependency/stereocrafter_util.py`):
  - **Middle-click**: Slider now jumps directly to mouse pointer position (matches frame scrubber behavior)
  - **Right-click**: Resets slider to default value
  - **Left-click**: Maintains original stepped increment behavior
- **Browes Folder Buttons:** Right-Click opens explorer window at paths location.
- **Combined Scanning:** `Auto Convergence` and `Boarder Depth` into a single pass.

### Fixed

- Slider middle-click functionality now provides precise positioning like the frame scrubber

---
