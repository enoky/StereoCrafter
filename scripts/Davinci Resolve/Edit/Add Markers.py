#!/usr/bin/env python

# --- COMPLETE SCRIPT V16 (Final Working Script: All Requirements Met) ---


# Avalible Marker Colors (Blue, Cyan, Green, Yellow, Red, Pink, Purple, Fuchsia,
#                         Rose, Lavender, Sky, Mint, Lemon, Sand, Cocoa, Cream)

import sys

try:
    resolve = resolve
except NameError:
    # Fallback for external execution
    import DaVinciResolveScript as dvr_script
    resolve = dvr_script.scriptapp("Resolve")

project_manager = resolve.GetProjectManager()
project = project_manager.GetCurrentProject()
timeline = project.GetCurrentTimeline()

if not (project and timeline):
    print("Error: No project or timeline is currently loaded.")
    sys.exit()

resolve.OpenPage("edit")

# --- PARAMETERS (Using only PROVEN, Valid Values) ---
TRACK_TYPE = "video"
TRACK_INDEX = 1 
MARKER_COLOR_CUT = "Cyan"
MARKER_COLOR_EDGE = "Red" # Proven valid color for Start/End
MARKER_NAME_TEMPLATE = "Marker "
MARKER_NOTE = ""
MARKER_DURATION = 1

timeline_name = timeline.GetName()
print(f"--- FINAL WORKING SCRIPT on Timeline: '{timeline_name}' (Track {TRACK_INDEX}) ---")

timeline_items = timeline.GetItemListInTrack(TRACK_TYPE, TRACK_INDEX)
if not timeline_items:
    print(f"Info: No clips found on {TRACK_TYPE} track {TRACK_INDEX}. Exiting.")
    sys.exit()

# Initialize counters
marker_count = 0
marker_index = 1 
successful_frames = []

# --- Helper function for adding a marker (Uses strictly POSITIONAL args and increments index) ---
def add_timeline_marker(frame_id, color, name_template, note, duration, custom_data):
    global marker_count
    global marker_index

    # Generate the unique name for this marker (e.g., "Marker 1")
    unique_name = f"{name_template}{marker_index}"
    
    # The actual call using strictly POSITIONAL ARGUMENTS
    success = timeline.AddMarker(
        int(frame_id),          # 1. frameId (MUST be Integer)
        color,                  # 2. color (MUST be a valid color string, e.g., 'Blue', 'Red')
        unique_name,            # 3. name
        note,                   # 4. note
        int(duration),          # 5. duration (MUST be Integer)
        custom_data             # 6. customData
    )
    
    if success:
        marker_count += 1
        marker_index += 1
        successful_frames.append(int(frame_id))
        return True
    else:
        # This should now never run unless a new bug is triggered
        print(f"CRITICAL FAILURE: Could not place Marker '{unique_name}' at frame: {frame_id}")
        return False

# --------------------------------------------------------------------------------
# 1. MARKER AT BEGINNING (Frame 0 is safe)
# --------------------------------------------------------------------------------
timeline_start_frame = int(timeline.GetStartFrame()) 

if add_timeline_marker(
    timeline_start_frame, 
    MARKER_COLOR_EDGE, 
    MARKER_NAME_TEMPLATE,
    MARKER_NOTE,
    MARKER_DURATION, 
    "Timeline_Start"
):
    print(f"SUCCESS: Placed Start Marker at frame {timeline_start_frame}")


# --------------------------------------------------------------------------------
# 2. MARKERS AT CUTS
# --------------------------------------------------------------------------------
# Loop from the SECOND clip (the first cut) to the end
for i in range(1, len(timeline_items)):
    clip = timeline_items[i]

    # The cut frame is the start of the current clip
    cut_frame_position_int = int(clip.GetStart(False))
    
    add_timeline_marker(
        cut_frame_position_int, 
        MARKER_COLOR_CUT, 
        MARKER_NAME_TEMPLATE, 
        MARKER_NOTE,
        MARKER_DURATION, 
        f"Cut_{i}"
    )


# --------------------------------------------------------------------------------
# 3. MARKER AT END (Frame after the last frame, which V14 proved safe)
# --------------------------------------------------------------------------------
# V14 worked with GetEndFrame() + 1, proving this boundary is markable.
timeline_end_frame = int(timeline.GetEndFrame()) + 1 

if add_timeline_marker(
    timeline_end_frame, 
    MARKER_COLOR_EDGE, 
    MARKER_NAME_TEMPLATE,
    MARKER_NOTE,
    MARKER_DURATION, 
    "Timeline_End"
):
    print(f"SUCCESS: Placed End Marker at frame {timeline_end_frame}")


# --------------------------------------------------------------------------------
# 4. FINAL OUTPUT
# --------------------------------------------------------------------------------

print(f"\n--- SCRIPT COMPLETE ---")
print(f"Total TIMELINE markers added: {marker_count}")
print(f"Successful frames: {successful_frames}")

# --- END OF FILE ---