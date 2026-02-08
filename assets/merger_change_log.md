# Merging GUI Change Log

## Version 26-02-07.2 (Today's Changes)

### New Features

#### Load/Refresh List Improvements
- First press scans all video files
- Subsequent presses only refresh preview (faster)
- Automatically rescans if folder paths change

#### Resume Mode
- Added "Resume" checkbox in Options section
- When enabled, skips files already in "finished" subfolder
- Automatically moves completed files to "finished" subfolder
- Moves: inpainted videos, splatted videos, original videos, and sidecar files
- Useful for resuming interrupted batch processing
- "Restore Finished Files" in File menu can undo the moves

#### Preview Source Persistence
- Preview Source dropdown selection is now saved to config
- Restored on next session launch

#### Add Borders Live Update
- Toggling "Add Borders" checkbox now updates preview instantly

### UI Improvements

- Updated "Load/Refresh List" button behavior documentation
- Added "Resume" documentation to Options section
- Added Preview Source persistence to documentation

### Files Modified

- `merging_gui.py`: Added resume mode, preview source persistence, live update for borders
- `dependency/video_previewer.py`: Modified _handle_load_refresh for smarter scanning
- `dependency/merge_help.json`: Added help text for resume option
- `assets/merger_gui_guide.md`: Updated documentation
- `assets/merger_change_log.md`: Updated with latest changes

---

## Version 26-02-07.1

### New Features

#### Sidecar File Support

- Added ability to read `.fssidecar` files for each clip
- Sidecars are searched in: inpainted folder (first), original video folder, next to video file
- Supports both `.fssidecar` and `.json` extensions for backwards compatibility
- Logs loaded sidecar files at INFO level

#### Border Application from Sidecars

- Added "Add Borders" checkbox in Options section
- Reads `left_border` and `right_border` values from sidecar files
- Borders are applied by zeroing out edge pixels (black bars)
- Border values can be percentages (<=100) or pixels (>100)
- Added "Borders: L=X%, R=Y%" info display in Progress section

#### Process Current Clip Button

- Added "Process Current Clip" button for testing settings
- Only processes the currently selected preview video
- Useful for testing without processing the entire batch

### Bug Fixes

- Fixed border application to zero out pixels instead of cropping frames
- Fixed variable naming issues in border processing code
- Fixed preview code to correctly apply borders

### UI Improvements

- Changed border info label color for better visibility in dark mode
- Updated tooltip help text for new features
- Added debug logging for border values and pixel calculations

### Code Refactoring

- Moved sidecar and border helper functions to `dependency/stereocrafter_util.py`
- Functions moved: `find_video_by_core_name`, `find_sidecar_file`, `read_clip_sidecar`, `apply_borders_to_frames`
- Prepared codebase for migrating non-GUI related code

### Files Modified

- `merging_gui.py`: Added sidecar reading, border application, UI elements
- `dependency/stereocrafter_util.py`: Added sidecar and border helper functions
- `dependency/merge_help.json`: Added help text for new options
- `assets/merger_gui_guide.md`: Added documentation for new features and sidecar files
