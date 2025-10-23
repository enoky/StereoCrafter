# Splatting GUI User Guide

The Splatting GUI is used to generate splatted videos from source video and depth maps. These splatted videos are a crucial intermediate step in creating stereoscopic 3D videos.

## Main Interface

The interface is divided into key sections:

### 1. Input/Output Folders

Specify the locations of your source and output files here.

-   **Input Source Clips:** The folder containing your input video clips (MP4, AVI, MOV, MKV). You can also select a single file to enable **Single File Mode** for both input clips and depth maps.
-   **Input Depth Maps:** The folder containing your pre-rendered depth maps. Depth maps should be named `videoname_depth.mp4` or `videoname_depth.npz` matching your input videos.
-   **Output Splatted:** The destination folder where the splatted output videos will be saved.

### 2. Process Resolution

These settings determine the resolution at which the splatting process will occur, and which resolution outputs are enabled.

-   **Enable Full Res:** Generates a splatted video at the original resolution of the input video. This is typically used for the final blending pass.
    -   **Batch Size:** The number of frames to process simultaneously when generating the full-resolution output. A higher value uses more VRAM but can be faster.
-   **Enable Low Res:** Generates a splatted video at a specified lower resolution. This output is primarily used for the inpainting process.
    -   **Width / Height:** The target width and height for the low-resolution output.
    -   **Batch Size:** The number of frames to process simultaneously when generating the low-resolution output.

### 3. Splatting & Output Settings

These parameters configure the core splatting and output encoding process.

-   **Process Length:** Sets how many frames to process before moving on to the next video. Use `-1` to process all frames. Primarily for testing.
-   **Output CRF:** Constant Rate Factor (CRF) for H.264/H.265 video encoding. Lower values mean higher quality and larger file sizes. A good starting point is 18 for high quality, 23 for good quality. Range is typically 0-51.
-   **Dual Output Only:** If checked, will generate dual panel (Right Eye Inpaint + Occlusion Mask) for inpainting. Unchecked will generate quad panel (Left, Depth, Occlusion, Right) for debugging or manual blending.
-   **Auto-Convergence:**
    -   **Off:** Disable auto-convergence.
    -   **Average:** EXPERIMENTAL: Simple auto-convergence derived from the temporal average of the center 75% depth region throughout the clip.
    -   **Peak:** EXPERIMENTAL: Simple auto-convergence derived from the temporal maximum of the center 75% depth region throughout the clip.
    -   **Note:** If a sidecar file exists, its convergence value will take precedence over Auto-Convergence.

### 4. Depth Map Pre-processing (Hi-Res Only)

These settings allow adjustments to the depth map before splatting, but only apply to the Hi-Res output.

-   **Dilate X/Y:** Horizontal/Vertical Dilate for the Depthmask. Must be an odd number (3, 5, 7, etc.) or 0 to disable.
-   **Blur X/Y:** Horizontal/Vertical Blur for the Depthmask. Must be an odd number (3, 5, 7, etc.) or 0 to disable. Sigma is automatically calculated as Size/6.

### 5. Depth Map Settings (All)

Settings here apply to both Full and Low-resolution outputs.

-   **Gamma:** Applies a non-linear adjustment to the depth map before normalization. Above 1.0 moves the midground towards the camera. Below 1.0 moves further away. Set to 1.0 to disable.
-   **Disparity:** Maximum disparity value as a percentage of the video's width x 20. Higher values result in more extreme parallax effects (depth shift). Example: '30.0' means a maximum of 1.5% shift horizontally at the foreground.
-   **Convergence:** Set to 1.0 will place depth inside the screen. 0.0 will give 100% pop out. This is the **Zero Disparity Plane** value.
-   **Disable Normalization**: If checked, the depth map is normalized based on its raw content values (0-255 or 0-1023), assuming raw input. This is typically used for split videos that will be rejoined later. Otherwise, a global min/max normalization is applied.

### 6. Preview Controls

-   **Load/Refresh List:** Scans the input folders for matching video/depth map pairs and loads the first one for preview.
-   **Preview Auto-Converge:** **(Button)** Runs a one-time scan of the current clip to calculate the **Average** and **Peak** depth. The calculated values are stored internally and applied to the Convergence slider based on the selected **Auto-Convergence** mode.
-   **< Prev / Next >:** Navigate between the loaded video clips.
-   **Update Sidecar:** Updates or creates the sidecar file (`.fssidecar`) for the current depth map using the values from the GUI sliders.
-   **Preview Source:** Selects the display mode (e.g., Splat Result, Occlusion Mask, Anaglyph, Wigglegram).
-   **Preview Scale:** Selects the scale factor for the preview image.

## Keyboard Shortcuts (Global)

| Shortcut | Action | Jump Size |
| :--- | :--- | :--- |
| **Left / Right Arrow** | Jump frames | 10 frames |
| **Shift + Left / Right Arrow** | Jump frames | 100 frames |
| **Ctrl + Left / Right Arrow** | Jump clips | Previous / Next |

## Basic Workflow

1.  **Set Input/Output Folders:** Fill in the paths for the `Input Source Clips`, `Input Depth Maps`, and `Output Splatted` folders.
2.  **Load Preview:** Click `Load/Refresh List` to load the clip list.
3.  **Adjust Settings**: Adjust `Depth Dilate`, `Blur`, `Gamma`, `Disparity`, and `Convergence`. Use **Preview Auto-Converge** to find a starting point for Convergence.
4.  **Save Sidecar:** Click **Update Sidecar** (or use the **Auto Save** toggle) to save your settings to the clip's sidecar file.
5.  **Start Processing:** Click the `START` button to begin batch processing.

## Menu Bar

-   **File Menu:**
    -   **Load Fusion Export (.fsexport)...:** Imports markers from a Fusion Export file and uses them to generate `.fssidecar` files for all videos in a target folder.
    -   **Auto Save Sidecar on Next:** If checked, automatically saves the current GUI slider values to the clip's sidecar file when navigating to the next/previous clip.
    -   **Update Slider from Sidecar:** If checked, the GUI sliders are automatically updated to the values saved in the sidecar file when a new clip is loaded.
    -   Allows you to load and save GUI settings, reset to defaults, or restore finished files.
-   **Help Menu:**
    -   Contains "About" and "User Guide" dialogs.
    -   Provides a checkbox to **Enable Debug Logging**.

## Output Sidecar (`.spsidecar`) Notes

The splatting process generates a secondary sidecar (`.spsidecar`) attached to the **low-resolution** output video only. This file is only created if the input video's sidecar contained a non-zero `frame_overlap` or `input_bias` value, which are essential for the subsequent inpainting/merging step.

## Current Processing Information

This section dynamically displays information about the currently processed video or image sequence, including:

-   **Filename, Task, Resolution, Frames:** Metadata about the video and current processing pass.
-   **Gamma, Disparity, Convergence:** The final parameter values being used for the current processing task (which can come from the GUI, Sidecar, or Auto-Convergence).