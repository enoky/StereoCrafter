# Splatting GUI Changelog
All notable changes to the splatting GUI and related components.
## [Released] - 2026-02-04
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
