#!/usr/bin/env python

# --- DaVinci Resolve Script: Export Timeline Markers to SRT ---
# Action: Creates a .srt subtitle file from Timeline Markers.
# Update: Uses a 'Select Folder' dialog instead of 'Save As'.

import sys
import os
import tkinter as tk
from tkinter import filedialog

# --- HELPER: Timecode Calculation ---
def frames_to_srt_time(frame, fps):
    """Converts frame count to SRT format: HH:MM:SS,ms"""
    total_seconds = frame / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def main():
    # --- API Setup ---
    try:
        global resolve
        resolve = resolve
    except NameError:
        import DaVinciResolveScript as dvr_script
        resolve = dvr_script.scriptapp("Resolve")

    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    timeline = project.GetCurrentTimeline()

    if not (project and timeline):
        print("Error: No project/timeline loaded.")
        sys.exit()

    print("--- Export Markers to SRT ---")

    # 1. GET FRAME RATE
    try:
        setting_fps = project.GetSetting('timelineFrameRate')
        if "fps" in str(setting_fps):
            fps = float(str(setting_fps).split(' ')[0])
        else:
            fps = float(setting_fps)
    except Exception as e:
        print(f"Warning: Could not determine FPS ({e}). Defaulting to 24.0")
        fps = 24.0
    
    print(f"Project FPS: {fps}")

    # 2. GET TIMELINE OFFSET
    timeline_start_frame = int(timeline.GetStartFrame())

    # 3. GET MARKERS
    markers_dict = timeline.GetMarkers()
    if not markers_dict:
        print("No markers found on Timeline Ruler.")
        sys.exit()

    sorted_frame_ids = sorted([int(k) for k in markers_dict.keys()])
    
    # 4. GENERATE SRT CONTENT
    srt_lines = []
    counter = 1

    for i in range(len(sorted_frame_ids)):
        current_frame_abs = sorted_frame_ids[i]
        
        marker_data = markers_dict.get(float(current_frame_abs)) 
        if not marker_data: marker_data = markers_dict.get(int(current_frame_abs))
        
        text_content = marker_data.get("name", f"Marker {counter}")
        
        start_frame_rel = current_frame_abs - timeline_start_frame
        
        if i < len(sorted_frame_ids) - 1:
            next_frame_abs = sorted_frame_ids[i+1]
            end_frame_rel = next_frame_abs - timeline_start_frame
        else:
            end_frame_rel = start_frame_rel + (fps * 2) 

        if end_frame_rel <= start_frame_rel:
            end_frame_rel = start_frame_rel + 1

        start_tc = frames_to_srt_time(start_frame_rel, fps)
        end_tc = frames_to_srt_time(end_frame_rel, fps)
        
        srt_lines.append(f"{counter}")
        srt_lines.append(f"{start_tc} --> {end_tc}")
        srt_lines.append(f"{text_content}\n")
        
        counter += 1

    # 5. PROMPT FOR FOLDER (Using tkinter)
    # We create a hidden root window to handle the dialog
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window
    root.attributes('-topmost', True) # Force the dialog to appear on top of Resolve
    
    print("Waiting for folder selection...")
    target_dir = filedialog.askdirectory(title="Select Folder to Save SRT")
    
    root.destroy() # Clean up UI resources

    if not target_dir:
        print("Save operation cancelled by user.")
        sys.exit()

    # Construct final filename automatically
    filename = f"{timeline.GetName()}_Markers.srt"
    full_path = os.path.join(target_dir, filename)

    # 6. WRITE FILE
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))
        print(f"\nSUCCESS: Saved SRT file to:\n{full_path}")
    except Exception as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    main()