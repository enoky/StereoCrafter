# ResolveMarkersToFusionKeyframes_Robust.py

import sys
from pprint import pprint
FRAME_OFFSET = 1 

# --- RESOLVE API SETUP (To get markers) ---

controls_to_keyframe = {
    "MaxDisparity": {"mode": "STATIC_VALUE",},    
    "Convergence": {"mode": "STATIC_VALUE",},
    "FrontGain": {"mode": "STATIC_VALUE",},    
    }

# --- FUSION/RESOLVE API SETUP ---
try:
    resolve = resolve 
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    timeline = project.GetCurrentTimeline()
    
    if not (project and timeline):
        raise Exception("Project or Timeline not loaded in Resolve.")
        
except Exception as e:
    print(f"Error accessing Resolve API or Timeline: {e}", file=sys.stderr)
    sys.exit()

# Get Marker Frame IDs
markers_dict = timeline.GetMarkers()
if not markers_dict:
    print("No Timeline Markers found. Aborting Fusion Keyframe creation.")
    sys.exit()
marker_frame_ids = sorted([int(frame_id) for frame_id in markers_dict.keys()])

# ... (get comp and control_panel is the same) ...
comp = fusion.GetCurrentComp() if 'fusion' in locals() else None
control_panel = comp.Control_Panel if comp else None
if comp is None or control_panel is None:
    print("Error: Could not access the current composition or 'Control_Panel' node. Aborting.")
    sys.exit()

# Ensure all controls exist before running the dialogue
missing_controls = [c for c in controls_to_keyframe if not hasattr(control_panel, c)]

if missing_controls:
    print(f"Error: The following controls were not found on 'Control_Panel': {missing_controls}. Aborting.")
    sys.exit()

# Move playhead to start time
comp_start_time = comp.GetAttrs()["COMPN_GlobalStart"]
comp.CurrentTime = comp_start_time

# ----------------------------------------------------
# --- NEW: CONFIRMATION DIALOG ---
# ----------------------------------------------------

# Construct the message detail
controls_list = "\n - ".join(controls_to_keyframe.keys())
marker_frames_str = ", ".join(map(str, marker_frame_ids))

dialog_message = (
    "Are you sure you want to **DELETE ALL EXISTING KEYFRAMES** "
    "for the following controls and replace them with new keyframes "
    "at the specified marker positions?"
)

# Composition:AskUser() takes a title and a list of controls to display.
# We only need a read-only LabelControl to display the warning message.
dialog_controls = {
    # Key 1 is for the Title/Warning Label
    1: {
        1: "WarningLabel", # Internal ID
        "Name": dialog_message, 
        2: "Text",           # Input Type: Text
        "ReadOnly": True,    # Essential: make it non-editable
        "Lines": 5,
        "Wrap": True,
        "INP_External": False,
        "LINKID_DataType": "Number",
    },
    # Key 2 is for the list of controls being affected
    2: {
        1: "ControlsAffected",
        "Name": "Controls to be Cleared:",
        2: "Text",
        "ReadOnly": True,
        "Lines": len(controls_to_keyframe) + 1,
        "Wrap": False,
        "INP_External": False,
        "Default": "\n - " + controls_list, # Display the list of controls
        "LINKID_DataType": "Number",
    },
    # Key 3 is for the list of frames being inserted
    3: {
        1: "FramesInserted",
        "Name": f"Frames to be Inserted ({len(marker_frame_ids)} total):",
        2: "Text",
        "ReadOnly": True,
        "Lines": 3,
        "Wrap": True,
        "INP_External": False,
        "Default": marker_frames_str,
        "LINKID_DataType": "Number",
    }
}

# Ask the user for confirmation
dialog_result = comp.AskUser(
    "!! WARNING: Keyframe Data Will Be Deleted !!", 
    dialog_controls
)

# Check the result: dialog_result will be None if the user clicked Cancel
if dialog_result is None:
    print("\n--- OPERATION CANCELED BY USER. NO KEYFRAMES WERE MODIFIED. ---")
    sys.exit()

print("\n--- Confirmation received. Proceeding with keyframe replacement. ---")

# ----------------------------------------------------
# --- UTILITY FUNCTIONS ---
# ----------------------------------------------------

def clean_and_add_spline(tool, input_name):
    """Clears old modifiers, gets static value, and adds a fresh spline at current time (Frame 0)."""
    input_obj = getattr(tool, input_name)
    initial_value = input_obj[comp_start_time] # Read value at comp start time (now playhead)
    
    if input_obj.GetConnectedOutput() is not None:
        input_obj.ConnectTo(None)
        initial_value = input_obj[comp_start_time] # Re-read static value
        print(f"  - Cleared existing modifier. Static Value: {initial_value:.4f}")
    
    success = tool.AddModifier(input_name, "BezierSpline")
    if not success:
         print(f"  - WARNING: Failed to add BezierSpline to '{input_name}'. Skipping.", file=sys.stderr)
         return None, None
         
    # Clean up the artifact keyframe created by AddModifier
    input_obj[comp_start_time] = None 
    
    return initial_value, getattr(tool, input_name) 


# --- MAIN EXECUTION LOOP ---

try:
    for control_id, config in controls_to_keyframe.items():
        print(f"\nProcessing control: {control_id}")

        # 1. READ STATIC VALUE, CLEAN INPUT, AND ADD SPLINE
        initial_value, input_obj = clean_and_add_spline(control_panel, control_id)

        if input_obj is None:
            continue
            
        # Initialize keyframe value setting variables (moved from previous version for safety)
        value_toggle = True # Only used for ALTERNATE_VALUES mode

        # 2. SET KEYFRAMES
        print(f"  - Setting {len(marker_frame_ids)} keyframes with value {initial_value:.4f}...")
        
        for frame_id in marker_frame_ids:
            
            # Use the offset ONLY IF the frame_id is > 0.
            corrected_id = frame_id
            if frame_id > 0:
                corrected_id = frame_id - FRAME_OFFSET
            
            # --- CALCULATE KEYFRAME VALUE BASED ON CONFIG ---
            current_keyframe_value = initial_value # <--- FIX: Initialize it to the initial_value first

            if config["mode"] == "ALTERNATE_VALUES":
                # Use ALTERNATE logic: alternates between low and high value
                val_low = config.get("value_low", 0.0)
                val_high = config.get("value_high", 1.0)
                
                current_keyframe_value = val_high if value_toggle else val_low
                value_toggle = not value_toggle # Flip the state for the next marker
                
            elif config["mode"] == "STATIC_VALUE":
                # Use STATIC_VALUE logic: it should just use the current_keyframe_value which 
                # was already set to initial_value above. No further assignment needed here.
                pass
            
            else:
                 print(f"  - WARNING: Invalid mode '{config['mode']}' for {control_id}. Skipping.", file=sys.stderr)
                 # Break the inner for loop (over markers)
                 break 
            
            # Set the keyframe
            input_obj[corrected_id] = current_keyframe_value
            
            print(f"  - Frame {corrected_id} (R{frame_id}): Value {current_keyframe_value:.4f}")
            
        # Optional: Print the final keyframes for verification
        keyframes = input_obj.GetKeyFrames()
        final_frames = sorted([float(t) for t in keyframes.values()])
        print(f"  - Final Keyframes: {final_frames}")
        
finally:
    comp.Unlock()