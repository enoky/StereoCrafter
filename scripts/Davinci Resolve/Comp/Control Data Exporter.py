# FusionKeyframeExportToJSON_Modular.py

import sys
import json
import os
from datetime import datetime

# Define the list of controls to export
CONTROLS_TO_EXPORT = [
    "MaxDisparity", 
    "Convergence",
    "FrontGamma",
    "Bias",
    "Overlap",
]
FRAME_OFFSET = 1 
LAST_PATH_KEY = "MyStudioInc.MarkerExport.LastPath" 
DEFAULT_BASE_DIR = os.environ.get('USERPROFILE', os.path.expanduser('~')) 

# --- 1. DATA GATHERING FUNCTION ---
def get_export_data(comp, timeline, controls_list, control_panel): 
    """Reads markers from Resolve and control values from Fusion."""
    
    # --- CRITICAL FIX: Ensure markers_dict is defined in this function's scope ---
    markers_dict = timeline.GetMarkers() 
    # --------------------------------------------------------------------------

    if not markers_dict:
        print("No Timeline Markers found to export. Returning empty data.")
        return None

    # Collect all input objects and validate them
    input_objects = {}
    
    # ... (input_objects retrieval logic remains the same) ...
    for control_id in controls_list:
        
        # --- ROBUST INPUT LOOKUP SEQUENCE ---
        input_obj = None
        
        try:
            # 1. Try Pythonic Item Access (Most Reliable for Macros/Groups: tool["ID"])
            input_obj = control_panel[control_id]
        except (AttributeError, KeyError):
            # 2. Fallback to GetAttr (Standard Pythonic access: tool.ID)
            input_obj = getattr(control_panel, control_id, None)
            
        if input_obj is None:
            print(f"WARNING: Control '{control_id}' not found on node '{control_panel.Name}'. Skipping.", file=sys.stderr)
        else:
            # Validate that the object found has the expected properties (e.g., 'Name')
            if not hasattr(input_obj, 'Name'):
                print(f"WARNING: Found object for '{control_id}' but it is not a valid input object. Skipping.", file=sys.stderr)
            else:
                input_objects[control_id] = input_obj

    if not input_objects:
        print(f"ERROR: No valid controls found on '{control_panel.Name}' to export. Aborting.")
        return None

    # --- BUILD THE EXPORT DICTIONARY ---
    export_data = {
        "export_date": datetime.now().isoformat(),
        "timeline_name": timeline.GetName(),
        "project_name": resolve.GetProjectManager().GetCurrentProject().GetName(),
        "data_units": "frames",
        "markers": [] 
    }

    # --- DATA POPULATION LOOP WITH ROBUST ERROR HANDLING (Type Mismatch Fix) ---

    # Get the keys (Frame IDs) from the Resolve dictionary, then convert to integers for sorting
    marker_keys = list(markers_dict.keys())
    
    try:
        # Create list of tuples: (original_key_type, int_frame_id) for sorting and safe lookup
        sorted_keys_and_frames = sorted(
            [(k, int(k)) for k in marker_keys], 
            key=lambda x: x[1]
        )
    except ValueError:
        print("WARNING: Marker keys are non-integer. Sorting by string name.", file=sys.stderr)
        sorted_keys_and_frames = [(k, k) for k in marker_keys]
        sorted_keys_and_frames.sort(key=lambda x: x[0])


    for original_key, frame_id in sorted_keys_and_frames:
        
        # Wrap the entire frame processing in a try/except
        try:
            # Look up the marker data using its original key
            marker_data = markers_dict.get(original_key)
            
            if marker_data is None:
                # Should not happen if original_key is from markers_dict.keys(), but safest to check.
                print(f"ERROR: Marker data for key {original_key} is genuinely missing. Skipping.", file=sys.stderr)
                continue 
            
            # Use the integer frame_id for all calculations
            
            # Calculate Fusion frame ID
            corrected_frame_id = frame_id
            if frame_id > 0:
                corrected_frame_id = frame_id - FRAME_OFFSET
                
            marker_entry = {
                "frame": corrected_frame_id, 
                "name": marker_data.get("name", "N/A"),
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
                    # Inner except: If one control fails, log and set to None, but continue
                    print(f"WARNING: Control '{control_id}' failed read at frame {corrected_frame_id}. Error: {e}", file=sys.stderr)
                    marker_entry["values"][control_id] = None

            export_data["markers"].append(marker_entry)
            
        except Exception as e:
            # Outer except: If an entire frame fails, log and continue to the next frame
            print(f"FATAL LOOP ERROR: Frame {frame_id} failed to process entirely. Skipping. Error: {e}", file=sys.stderr)
            continue 

    # Sort the markers list by frame number before returning
    if export_data["markers"]:
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

def get_target_tool(comp):
    """
    Tries to get the active tool. If not valid, asks the user.
    """
    
    # 1. Check for active tool
    active_tool = comp.ActiveTool
    if active_tool is not None:
        print(f"INFO: Using currently selected node: '{active_tool.Name}'")
        return active_tool
    
    # 2. If no tool is active, prompt the user
    dialog_controls = {
        1: {
            1: "NodeName",
            "Name": "Enter Control Node Name (e.g., 'Control_Panel')",
            2: "Text",
            "Default": "Control_Panel",
            "LINKID_DataType": "Text",
        }
    }

    dialog_result = comp.AskUser(
        "Select Node for Export",
        dialog_controls
    )

    if dialog_result is None:
        print("\n--- NODE SELECTION CANCELED BY USER. ---")
        return None
    
    node_name = dialog_result.get(1)
    
    if not node_name:
        print("\n--- ERROR: Node name cannot be empty. ---")
        return None
        
    # 3. Search for the tool by the entered name
    target_tool = comp.FindTool(node_name)
    
    if target_tool is None:
        print(f"\n--- ERROR: Node named '{node_name}' not found. ---")
        return None
        
    print(f"INFO: Found node: '{target_tool.Name}'")
    return target_tool

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
        
        if not (project and timeline and comp):
            raise Exception("Project, Timeline, or Composition not loaded/found.")
            
    except Exception as e:
        print(f"Error during API setup: {e}", file=sys.stderr)
        sys.exit()

    print("--- Fusion/Resolve Keyframe Data Export (Modular) ---")
    
    # --- GET THE TARGET NODE ---
    target_tool = get_target_tool(comp) # Returns the actual Tool object
    if target_tool is None:
        sys.exit()

    # 1. GATHER DATA
    # PASS THE TOOL OBJECT DIRECTLY, not just the name.
    export_data = get_export_data(comp, timeline, CONTROLS_TO_EXPORT, target_tool)
    
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