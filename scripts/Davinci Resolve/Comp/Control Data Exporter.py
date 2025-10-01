# FusionKeyframeExportToJSON_Modular.py

import sys
import json
import os
from datetime import datetime

# Define the list of controls to export
CONTROLS_TO_EXPORT = [
    "MaxDisparity", 
    "Convergence",
]
FRAME_OFFSET = 1 
LAST_PATH_KEY = "MyStudioInc.MarkerExport.LastPath" 
DEFAULT_BASE_DIR = os.environ.get('USERPROFILE', os.path.expanduser('~')) 

# --- 1. DATA GATHERING FUNCTION ---
def get_export_data(comp, timeline, controls_list):
    """Reads markers from Resolve and control values from Fusion."""
    
    control_panel = comp.Control_Panel
    markers_dict = timeline.GetMarkers() # Keys are frame IDs (strings), Values are marker data (dict)

    if not markers_dict:
        print("No Timeline Markers found to export. Returning empty data.")
        return None

    # Collect all input objects and validate them
    input_objects = {}
    for control_id in controls_list:
        input_obj = getattr(control_panel, control_id, None)
        if input_obj is None:
            print(f"WARNING: Control '{control_id}' not found. Skipping.", file=sys.stderr)
        else:
            input_objects[control_id] = input_obj

    if not input_objects:
        print("ERROR: No valid controls found to export. Aborting.")
        return None

    # Build the Export Dictionary
    export_data = {
        "export_date": datetime.now().isoformat(),
        "timeline_name": timeline.GetName(),
        "project_name": resolve.GetProjectManager().GetCurrentProject().GetName(),
        "data_units": "frames",
        "markers": []
    }

    # ITERATE DIRECTLY OVER THE VALID MARKER ITEMS (keys are FRAME_ID_STR, values are marker_data dict)
    for frame_id_str, marker_data in markers_dict.items():
        
        try:
            frame_id = int(frame_id_str)
        except ValueError:
            print(f"WARNING: Skipping non-numeric marker frame ID: {frame_id_str}", file=sys.stderr)
            continue # Skip this loop iteration if the frame ID is invalid
            
        # Calculate Fusion frame ID (correcting for the offset)
        corrected_frame_id = frame_id
        if frame_id > 0:
            corrected_frame_id = frame_id - FRAME_OFFSET
            
        marker_entry = {
            "frame": corrected_frame_id, 
            "name": marker_data.get("name", "N/A"), # Marker data is guaranteed to be a dict here
            "color": marker_data.get("color", "N/A"),
            "note": marker_data.get("note", ""),
            "values": {}
        }
        
        # Read values from Fusion controls
        for control_id, input_obj in input_objects.items():
            try:
                # Read the value at the corrected time: input_obj[time]
                value = input_obj[corrected_frame_id]
                marker_entry["values"][control_id] = float(value) 
            except Exception as e:
                print(f"WARNING: Could not read value for '{control_id}' at frame {corrected_frame_id}. Error: {e}", file=sys.stderr)
                marker_entry["values"][control_id] = None

        export_data["markers"].append(marker_entry)

    # Sort the markers list by frame number before returning
    export_data["markers"].sort(key=lambda x: x["frame"])
    
    print(f"SUCCESS: Extracted data for {len(export_data['markers'])} markers.")
    return export_data

# --- 2. DIALOG AND SAVE PATH FUNCTION ---
def get_save_path(comp, fusion, timeline_name):
    """Displays a dialog, gets the save path, and saves the last used directory."""
    
    # 1. Determine Default Path from Persistence/Fallback
    default_path_dir = fusion.GetData(LAST_PATH_KEY)

    if default_path_dir is None:
        default_path_dir = DEFAULT_BASE_DIR 
        
    # --- ROBUST PATH CONSTRUCTION ---
    # Python's os.path.join doesn't always handle Windows drive letters well.
    # We ensure the base directory ends with a slash and manually join.
    if not default_path_dir.endswith((':', '/', '\\')):
        # For a Windows drive like 'Z:', ensure it becomes 'Z:/'
        if len(default_path_dir) == 2 and default_path_dir[1] == ':':
            default_path_dir += '/'
        # For a folder path, ensure it has a separator
        else:
            default_path_dir += os.sep

    filename = f"Fusion_Export_{timeline_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.fsexport"
    default_save_path = default_path_dir + filename # Manual, safer join

    # 2. Define the Save Dialog Control
    dialog_controls = {
        1: {
            1: "SaveFilePath",
            "Name": "Select Output fsexport File Path",
            2: "FileBrowse",             
            "Default": default_save_path, 
            "Save": True,                
            "Filter": "fsexport Files (*.fsexport)|*", 
            "LINKID_DataType": "Text",   
        }
    }

    # 3. Ask the user for the save path
    dialog_result = comp.AskUser(
        "Select Destination for Keyframe fsexport Export", 
        dialog_controls
    )

    if dialog_result is None:
        print("\n--- OPERATION CANCELED BY USER. ---")
        return None

    # --- CRITICAL FIX: Check the path value ---
    
    output_path = dialog_result.get(1)
    if output_path is None:
        output_path = dialog_result.get(1.0) # Try float 1.0
    if output_path is None:
        output_path = dialog_result.get("SaveFilePath") # Try the internal ID string
        
    # Final validation check
    if not output_path or not isinstance(output_path, str) or len(output_path) < 3:
        # Diagnostic print to see what the dialog actually returned
        print(f"\n!!! DIAGNOSTIC: Dialog returned: {dialog_result} !!!", file=sys.stderr)
        print("\n--- ERROR: Invalid path returned. Path may be empty or corrupted. ---")
        return None

    # 4. Save the path for future use (Persistence)
    output_path_dir = os.path.dirname(output_path)
    fusion.SetData(LAST_PATH_KEY, output_path_dir)
    print(f"INFO: Saved last-used directory for next run: {output_path_dir}")
    
    return output_path
    
# --- 3. MAIN COORDINATION FUNCTION ---
def main():
    """Coordinates the entire export process."""
    
    # --- API Setup (Local Scope) ---
    try:
        global resolve, fusion # Ensure global access for utility functions
        resolve = resolve
        fusion = fusion
        project = resolve.GetProjectManager().GetCurrentProject()
        timeline = project.GetCurrentTimeline()
        comp = fusion.GetCurrentComp()
        
        if not (project and timeline and comp and comp.Control_Panel):
            raise Exception("Project, Timeline, or 'Control_Panel' node not loaded/found.")
            
    except Exception as e:
        print(f"Error during API setup: {e}", file=sys.stderr)
        sys.exit()

    print("--- Fusion/Resolve Keyframe Data Export (Modular) ---")
    
    # 1. GATHER DATA
    export_data = get_export_data(comp, timeline, CONTROLS_TO_EXPORT)
    
    if export_data is None:
        sys.exit()

    # 2. GET SAVE PATH
    output_path = get_save_path(comp, fusion, timeline.GetName())

    if output_path is None:
        sys.exit()

    # 3. PERFORM SAVE
    try:
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"\nExport complete! Data saved to:\n{output_path}")

    except Exception as e:
        print(f"\nFATAL ERROR: Could not write fsexport file to {output_path}. Error: {e}", file=sys.stderr)
        sys.exit()

    print("\n--- DATA EXPORT COMPLETE ---")


if __name__ == "__main__":
    main()