# Splatting GUI User Guide

The Splatting GUI is used to generate splatted videos from source video and depth maps. These splatted videos are a crucial intermediate step in creating stereoscopic 3D videos.

## Main Interface

The interface is divided into key sections:

### 1. Input/Output Folders

Specify the locations of your source and output files here.

- **Input Source Clips:** The folder containing your input video clips (MP4, AVI, MOV, MKV). You can also select a single file to enable **Single File Mode**.
- **Input Depth Maps:** The folder containing your pre-rendered depth maps. Depth maps should be named `videoname_depth.mp4` or `videoname_depth.npz` matching your input videos.
- **Output Splatted:** The destination folder where the splatted output videos will be saved.
- **Multi-Map:** If enabled, the GUI scans for subfolders within the Depth Maps folder. Each subfolder is assumed to contain a full set of depth maps. Radio buttons will appear in the Preview area to easily switch between different depth map versions (e.g., from different models or settings).

### 2. Process Resolution

These settings determine the resolution at which the splatting process will occur, and which resolution outputs are enabled.

- **Enable Full Res:** Generates a splatted video at the original resolution of the input video. This is typically used for the final blending pass.
  - **Batch Size:** The number of frames to process simultaneously. A higher value uses more VRAM but can be faster.
- **Enable Low Res:** Generates a splatted video at a specified lower resolution. This output is primarily used for the inpainting process.
  - **Width / Height:** The target width and height for the low-resolution output.
  - **Batch Size:** The number of frames to process simultaneously for the low-resolution output.
- **Dual Output Only:** If checked, will generate dual panel (Right Eye Inpaint + Occlusion Mask) for inpainting. Unchecked will generate quad panel (Left, Depth, Occlusion, Right) for debugging or manual blending.

### 3. Splatting & Output Settings

These parameters configure the core splatting and output encoding process.

- **Process Length:** Sets how many frames to process before moving on to the next video. Use `-1` to process all frames.
- **Auto-Convergence:**
  - **Off:** Disable auto-convergence and preserve existing sidecar values.
  - **Manual:** Write current slider values to sidecar (useful for batch-updating settings).
  - **Average:** Simple auto-convergence derived from the temporal average of the center 75% depth region.
  - **Peak:** Simple auto-convergence derived from the temporal maximum of the center 75% depth region.
  - **Hybrid:** Combines Average and Peak (50/50 blend).
- **Output CRF (Full / Low):** Constant Rate Factor for video encoding. Lower values mean higher quality. You can now set different quality levels for the Full Resolution and Low Resolution outputs independently.
- **Color Tags:** Metadata-only tags written into the output file headers (e.g., BT.709, BT.2020). This does not affect the splatting math but helps players/editors interpret the color space correctly.

### 4. Depth Map Pre-processing (Hi-Res Only)

These settings allow adjustments to the depth map before splatting, applied primarily to the Hi-Res output.

- **Dilate X/Y:** Horizontal/Vertical Dilation for the depth map. Positive values expand bright areas (foreground); negative values (down to -10) perform erosion.
- **Blur X/Y:** Horizontal/Vertical Gaussian Blur for the depth map.
- **Dilate Left:** Specific dilation applied to the "left" side or edge handling (useful for fixing seam artifacts).
- **Blur Left / Mix:** Specific blur applied to the left edge, with a **Mix** selector to balance between horizontal and vertical blur components.

### 5. Depth Map Settings (All)

Settings here apply to both Full and Low-resolution outputs.

- **Gamma:** Non-linear adjustment to the depth map. Above 1.0 moves midground towards the camera; below 1.0 moves it further away.
- **Disparity:** Maximum disparity value as a percentage of the video width.
- **Convergence:** Set the **Zero Disparity Plane**. 1.0 places depth inside the screen; 0.0 gives maximum pop-out.
- **Enable Global Normalization:** If checked, the depth map is normalized based on the global min/max values of the entire clip (requires a pre-pass). If unchecked, normalization is local to each frame.
- **Resume:** If enabled, the processor will skip files that already exist in the output folder.

### 6. Preview Controls

- **Load/Refresh List:** Scans the input folders for matching video/depth map pairs.
- **Preview Auto-Converge (Button):** Runs a scan of the current clip to calculate **Average** and **Peak** depth.
- **< Prev / Next >:** Navigate between loaded video clips.
- **Update Sidecar:** Saves current GUI slider values to the `.fssidecar` file.
- **Preview Source:** Selects the display mode (e.g., Splat Result, Occlusion Mask, Anaglyph, Wigglegram, Depth Map Color).
- **Preview Scale:** Selects the scale factor for the preview image.
- **Crosshair:** (Checkbox) Enables a centering crosshair on the preview to help with alignment and convergence checking. **White** and **Multi** options adjust the crosshair style.
- **D/P:** (Checkbox) Depth/Pop readout. Shows depth information at the mouse cursor position in the preview.

## Keyboard Shortcuts (Global)

| Shortcut                       | Action      | Jump Size       |
| :----------------------------- | :---------- | :-------------- |
| **Left / Right Arrow**         | Jump frames | 10 frames       |
| **Shift + Left / Right Arrow** | Jump frames | 100 frames      |
| **Ctrl + Left / Right Arrow**  | Jump clips  | Previous / Next |

## Basic Workflow

1.  **Set Input/Output Folders:** Fill in paths for Source Clips, Depth Maps, and Output Splatted.
2.  **Load Preview:** Click `Load/Refresh List`.
3.  **Adjust Settings**: Adjust `Dilate`, `Blur`, `Gamma`, `Disparity`, and `Convergence`. Use **Preview Auto-Converge** to find a starting point.
4.  **Save Sidecar:** Click **Update Sidecar** (or check **Auto Save Sidecar on Next** in the File menu).
5.  **Start Processing:** Click the `START` button to begin batch processing.

## Menu Bar

- **File Menu:**
  - **Load Fusion Export (.fsexport)...:** Imports markers from a Fusion Export file to generate `.fssidecar` files matching existing depth map videos.
  - **FSExport to custom sidecar...:** Generates sidecar files with a custom name and destination from a Fusion Export, without requiring existing video files.
  - **Auto Save Sidecar on Next:** Automatically saves current slider values to the sidecar when navigating clips.
  - **Update Slider from Sidecar:** Automatically updates sliders to match the sidecar values when a new clip is loaded.
  - **Reset to Default / Restore Finished:** Reset GUI settings or move files back from a "finished" folder.
  - **Sidecars: Depth → Source (remove _depth):** Moves sidecar files from the Depth Maps folder to the Source Clips folder, removing the `_depth` suffix from filenames. Useful for migrating sidecars to the new location.
  - **Sidecars: Source → Depth (add _depth):** Moves sidecar files from the Source Clips folder to the Depth Maps folder, adding the `_depth` suffix to filenames. Useful for moving sidecars back to the depth folder.
- **Help Menu:**
  - **User Guide:** Opens this document.
  - **Debug Logging:** Enables verbose console output for troubleshooting.

## Output Sidecar (`.spsidecar`) Notes

The splatting process generates a secondary sidecar (`.spsidecar`) attached to the **low-resolution** output video. This file contains metadata like `frame_overlap` and `input_bias` required for the subsequent inpainting/merging step.

## Current Processing Information

This section displays real-time metadata about the active task:

- **Filename, Task, Resolution, Frames:** Metadata about the video and current pass.
- **Disparity, Convergence, Gamma, Map:** The exact values being used for the process.
