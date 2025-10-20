# Splatting GUI User Guide

The Splatting GUI is used to generate splatted videos from source video and depth maps. These splatted videos are a crucial intermediate step in creating stereoscopic 3D videos.

## Main Interface

The interface is divided into key sections:

### 1. Input/Output Folders

Specify the locations of your source and output files here.

-   **Input Source Clips:**  The folder containing your input video clips (MP4, AVI, MOV, MKV).
-   **Input Depth Maps:** The folder containing your pre-rendered depth maps.  Depth maps should be named `videoname_depth.mp4` or `videoname_depth.npz` matching your input videos.
-   **Output Splatted:** The destination folder where the splatted output videos will be saved.

### 2. Process Resolution

These settings determine the resolution at which the splatting process will occur, and which resolution outputs are enabled.

-   **Enable Full Res:** Generates a splatted video at the original resolution of the input video.  This is typically used for the final blending pass.
    -   **Batch Size:** The number of frames to process simultaneously when generating the full-resolution output.  A higher value uses more VRAM but can be faster.
-   **Enable Low Res:** Generates a splatted video at a specified lower resolution.  This output is primarily used for the inpainting process.
    -   **Width:** The target width for the low-resolution output.
    -   **Height:** The target height for the low-resolution output.
    -   **Batch Size:**  The number of frames to process simultaneously when generating the low-resolution output.  Can often be higher than the full-resolution batch size due to lower memory requirements.

### 3. Splatting & Output Settings

These parameters configure the core splatting and output encoding process.

-   **Process Length:**  Sets how many frames to process before moving on to the next video.  Use `-1` to process all frames. Primarily for testing.
-   **Output CRF:** Constant Rate Factor (CRF) for H.264/H.265 video encoding.  Lower values mean higher quality and larger file sizes. A good starting point is 18 for high quality, 23 for good quality. Range is typically 0-51.
-   **Dual Output Only:** If checked, will generate dual panel for inpaint right eye only. Unchecked will generate quad panel for complete low-res stereo SBS after inpainting. Use this(Dual) to save on resources and manual blending.
-   **Auto-Convergence:**
    - Off: Disable auto-convergence
    - Average: EXPERIMENTAL: Simple auto convergence derived from 75% center region and averaged throughout the clip.
    - Peak: EXPERIMENTAL: Simple auto convergence derived from 75% center region and peaked throughout the clip.


### 4. Depth Map Pre-processing (Hi-Res Only)

These settings allow adjustments to the depth map before splatting, but only apply to the Hi-Res output

-   **Dilate X/Y:** Horizontal/Vertical Dilate for the Depthmask. Must be an odd number (3, 5, 7, etc.) or 0 to disable.
-   **Blur X/Y:** Horizontal/Vertical Blur for the Depthmask. Must be an odd number (3, 5, 7, etc.) or 0 to disable. Sigma is automatically calculated as Size/6.

### 5. Depth Map Settings (All)

Settings here apply to both Full and Low-resolution outputs.

-   **Gamma:** Applies a non-linear adjustment to the depth map before normalization.  Above 1.0 moves the midground towards the camera. Below 1.0 move further away. Set to 1.0 to disable.
-   **Disparity:**  Maximum disparity value as a percentage of the video's width x 20. Higher values result in more extreme parallax effects (depth shift). Example: '30.0' means a maximum of 1.5% shift horizontally at the foreground.
-   **Convergence:**  Set to 1.0 will place depth inside the screen. 0.0 will give 100% pop out. StereoCrafter Default = 0.5, Recommend 0.8.
-   **Disable Normalization**: Use this only if your scene is too long and you have split your video/depthmap into shorter lengths to be rejoined after processing. Otherwise, your cut scenes depth will not align

### 6. Additional Options

Various options to control the GUI and processing.

-   **Dual Output Only**: If checked, will generate dual panel for inpaint right eye only. Unchecked will generate quad panel for complete low-res stereo SBS after inpainting. Use this(Dual) to save on resources and manual blending.

### 7. Progress and Controls

This section displays the progress of batch processing and provides controls to start and stop the process.

-   **Progress Bar:** Shows the progress of the overall batch operation.
-   **Status Label:** Displays the current status, such as which video is being processed or if an error has occurred.
-   **SINGLE:** Start processing for current preview clip.
-   **START:** Starts the batch processing: loads videos, depth maps, and performs the depth splatting.
-   **STOP:** Halts the batch processing after the current video chunk is finished.
-   **Update Sidecar:** Updates the current sidecar file with the sidecar values

## Basic Workflow

1.  **Set Input/Output Folders:** Fill in the paths for the `Input Source Clips`, `Input Depth Maps`, and `Output Splatted` folders.
2.  **Configure Processing:** Select the desired processing options, like enabling Full Res or Low Res output and setting batch sizes and the CRF output
3.  **Adjust Settings**: Adjust the `Depth Dilate Size`, `Depth Blur Size`, `Gamma` and others to your liking, check the previewer for a sample frame
4.  **(Optional)Load Preview**: Select single `*.mp4` from `Input Source Clips` or `Select Depth Map`
5.  **Start Processing:** Click the `START` button to begin batch processing all videos in the input folder.

## Menu Bar

-   **File Menu:**
    -   Allows you to load and save settings to a JSON file.
    -   Provides a checkbox to switch to **Dark Mode**.
    -   Allows you to reset all settings to their defaults or restore finished files from subfolders.
-   **Help Menu:**
    -   Contains an "About" dialog.
    -   Provides a checkbox to **Enable Debug Logging**, which prints much more detailed information to the console for troubleshooting.

> **Tip:** Hover your mouse over any control to see a detailed tooltip explaining its function.

## Notes
- Requires the  "forward_warp_pytorch"  folder is in the  "dependency"  folder or the forward_warp cuda complied in your environment.

## Current Processing Information

This section dynamically displays information about the currently processed video or image sequence, including:

-   **Filename:** The name of the currently processing source video or image sequence.
-   **Resolution:** The resolution of the currently processed source video or image sequence.
-   **Frames:** The number of frames in the current source video or image sequence.
-   **Gamma:** Current Gamma.
-   **Disparity:** Current Disparity
-   **Convergence:** Current Convergence

This section is updated automatically as the batch processing progresses.